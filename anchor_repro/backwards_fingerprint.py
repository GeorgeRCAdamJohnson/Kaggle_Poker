"""Build the EMPIRICAL planting fingerprint discovered by backwards analysis, then pair-risk.

Backwards analysis (§127) found planted hands are NOT subtle soft-play — they are BIG AGGRESSIVE
HIGH-COMMITMENT pots with a directional transfer. Top within-pair discriminators (planted vs the
same pairs' other hands):
  pair_agg 0.83, contrib_bb 0.80, opposite_flow 0.80, abs_flow 0.79, n_act 0.78, winner_pot_share 0.73.

Forward models used these as SEPARATE features and capped at ~0.71 because ordinary pairs also have
occasional big-aggressive hands. The untried move: build the CONJUNCTION as a single hand-level
"planting fingerprint" (all conditions at once, thresholds learned from the planted hands), then
score each PAIR by its RATE of fingerprint-matching hands vs the ordinary-pair rate. A colluding
pair has MANY fingerprint hands; an ordinary pair has few.

We LEARN the fingerprint from the planted-vs-nonplanted contrast (a per-hand model), score EVERY
eval-pair hand, and aggregate the fingerprint rate + count per pair -> risk. Judge eval-honest.

Run:  python -m anchor_repro.backwards_fingerprint
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, CACHE, D

OUT = CACHE / "backwards"
CONTRAST = OUT / "planted_contrast.parquet"
SEED = 42

# the empirically-strong per-hand dims (from §127 separation ranking)
FP_FEATS = ["pair_agg", "a_contrib_bb", "b_contrib_bb", "opposite_flow", "abs_flow", "n_act",
            "n_act_b", "n_folded", "pair_won", "winner_pot_share", "pot_bb", "n_check", "n_check_b",
            "max_strength", "min_strength", "strength_gap", "n_board"]


def _build_eval_perhand_dims() -> pl.DataFrame:
    """Reconstruct the SAME per-hand dims for ALL eval-pair shared hands (to score the fingerprint)."""
    from anchor_repro.hand_eval_gpu import strength_for_seats
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
    uni = pl.scan_parquet(V30_EVAL).select(["pair_id", "hand_id"]).unique()
    shared = uni.join(eval_pairs.lazy(), on="pair_id", how="inner").collect()

    seats = pl.read_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "total_contribution", "net_chips", "folded", "went_to_showdown",
         "won_share", "hole_card_1", "hole_card_2"])
    hands = pl.read_parquet(D / "hands.parquet").select(["hand_id", "board_cards", "final_pot", "big_blind"])
    acts = pl.scan_parquet(D / "actions.parquet").group_by(["hand_id", "player_id"]).agg(
        (pl.col("action").is_in(["bet", "raise", "all_in"])).sum().alias("n_agg"),
        (pl.col("action") == "check").sum().alias("n_check"),
        pl.len().alias("n_act"),
    ).collect()

    def side(col):
        return shared.join(eval_pairs.select(["pair_id", col]).rename({col: "player_id"}), on="pair_id").join(
            seats, on=["hand_id", "player_id"], how="left").join(acts, on=["hand_id", "player_id"], how="left").join(
            hands, on="hand_id", how="left")
    a = side("player_1"); b = side("player_2")

    def strength(df):
        h1 = df["hole_card_1"].to_list(); h2 = df["hole_card_2"].to_list(); bd = df["board_cards"].to_list()
        cl = [[x, y] + (z.split() if z else []) for x, y, z in zip(h1, h2, bd)]
        return strength_for_seats(cl)
    a = a.with_columns(pl.Series("strength", strength(a)))
    b = b.with_columns(pl.Series("strength", strength(b)))
    A = a.select(["pair_id", "hand_id", "total_contribution", "net_chips", "folded", "went_to_showdown",
                  "won_share", "n_agg", "n_check", "n_act", "strength", "board_cards", "final_pot", "big_blind"])
    B = b.select(["pair_id", "hand_id", "total_contribution", "net_chips", "folded", "went_to_showdown",
                  "won_share", "n_agg", "n_check", "n_act", "strength"])
    df = A.join(B, on=["pair_id", "hand_id"], suffix="_b")
    bb = pl.col("big_blind")
    return df.with_columns(
        (pl.col("n_agg") + pl.col("n_agg_b")).alias("pair_agg"),
        (pl.col("total_contribution") / bb).alias("a_contrib_bb"),
        (pl.col("total_contribution_b") / bb).alias("b_contrib_bb"),
        ((pl.col("net_chips") * pl.col("net_chips_b")) < 0).cast(pl.Int8).alias("opposite_flow"),
        (pl.col("net_chips").abs() + pl.col("net_chips_b").abs()).alias("abs_flow"),
        (pl.col("folded").cast(pl.Int8) + pl.col("folded_b").cast(pl.Int8)).alias("n_folded"),
        (pl.col("won_share") + pl.col("won_share_b")).alias("pair_won"),
        (pl.max_horizontal("net_chips", "net_chips_b") / (pl.col("final_pot") + 1)).alias("winner_pot_share"),
        (pl.col("final_pot") / bb).alias("pot_bb"),
        pl.max_horizontal("strength", "strength_b").alias("max_strength"),
        pl.min_horizontal("strength", "strength_b").alias("min_strength"),
        (pl.col("strength") - pl.col("strength_b")).abs().alias("strength_gap"),
        (pl.col("board_cards").str.split(" ").list.len()).alias("n_board"),
    )


def main() -> int:
    H = Harness()
    con = pl.read_parquet(CONTRAST)
    y_hand = con["planted"].to_numpy()
    Xh = con.select(FP_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    groups_h = con["pair_id"].to_numpy()
    print(f"per-hand fingerprint model: {con.height:,} hands, planted={int(y_hand.sum())}")

    # --- learn the per-hand planting fingerprint (planted vs same-pairs' other hands), pair-grouped OOF ---
    fp_params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.05,
                           subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=4.0)
    oof_h = np.zeros(len(y_hand), np.float32)
    gkf = GroupKFold(5)
    for tr, va in gkf.split(Xh, y_hand, groups_h):
        spw = (y_hand[tr] == 0).sum() / max((y_hand[tr] == 1).sum(), 1)
        m = xgb.train({**fp_params, "scale_pos_weight": float(spw)}, xgb.DMatrix(Xh.iloc[tr], label=y_hand[tr]), num_boost_round=300)
        oof_h[va] = m.predict(xgb.DMatrix(Xh.iloc[va]))
    print(f"per-hand fingerprint: planted-vs-nonplanted AUC={roc_auc_score(y_hand, oof_h):.4f} AP={average_precision_score(y_hand, oof_h):.4f}")
    # full fingerprint model
    spw = (y_hand == 0).sum() / max((y_hand == 1).sum(), 1)
    fp_full = xgb.train({**fp_params, "scale_pos_weight": float(spw)}, xgb.DMatrix(Xh, label=y_hand), num_boost_round=300)

    # --- score EVERY eval-pair hand + dev-pair hand with the fingerprint, aggregate per pair ---
    t = time.time()
    ev = _build_eval_perhand_dims()
    Xe = ev.select(FP_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    ev = ev.select(["pair_id"]).with_columns(pl.Series("fp", fp_full.predict(xgb.DMatrix(Xe))))
    print(f"scored eval-pair hands: {ev.height:,} ({time.time()-t:.0f}s)")

    def pair_agg(df):
        return df.group_by("pair_id").agg(
            (pl.col("fp") > 0.5).mean().alias("fp_rate_hi"),
            (pl.col("fp") > 0.3).mean().alias("fp_rate_mid"),
            pl.col("fp").mean().alias("fp_mean"),
            pl.col("fp").top_k(5).mean().alias("fp_top5"),
            pl.col("fp").max().alias("fp_max"),
            (pl.col("fp") > 0.5).sum().alias("fp_count_hi"),
            pl.len().alias("n_hands"),
        )
    eval_pair = pair_agg(ev)

    # dev-pair fingerprint from the OOF (colluding pairs) — need ALL dev pairs (incl negatives) scored.
    # Score dev negatives' hands too: reuse the contrast build path but for ALL dev pairs. For speed,
    # score dev via the same per-hand dims we already have for colluding pairs (positives) + build negs.
    # Simpler + honest: score the FULL dev shared-hand set with fp_full (need dims). Build them:
    dev_dims = _build_dev_perhand_dims()
    Xd = dev_dims.select(FP_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    dev_scored = dev_dims.select(["pair_id"]).with_columns(pl.Series("fp", fp_full.predict(xgb.DMatrix(Xd))))
    dev_pair = pair_agg(dev_scored)

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    ptbl = (pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
            .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
            .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
            .select(["pair_id", "table_id"]).collect())
    dev = labels.join(dev_pair, on="pair_id", how="left").join(ptbl, on="pair_id", how="left")
    pfeats = [c for c in dev_pair.columns if c != "pair_id"]
    Xdp = dev.select(pfeats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].fill_null("?").to_numpy()

    print("\n=== pair-level fingerprint-rate features (univariate AUC vs labels) ===")
    for c in pfeats:
        v = Xdp[c].to_numpy()
        try:
            a = roc_auc_score(y, v); print(f"  {max(a,1-a):.4f}  {c}")
        except Exception: pass

    oof = np.zeros(len(y), np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    for tr, va in sgkf.split(Xdp, y, g):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(Xdp.iloc[tr], label=y[tr]), num_boost_round=300)
        oof[va] = m.predict(xgb.DMatrix(Xdp.iloc[va]))
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(Xdp, label=y), num_boost_round=300)
    evf = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id"]).join(eval_pair, on="pair_id", how="left")
    Xef = evf.select(pfeats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    escore = full.predict(xgb.DMatrix(Xef))
    r = H.judge("backwards_fingerprint", oof, dev["pair_id"].to_list(), escore, evf["pair_id"].to_list())
    print(f"\n=== BACKWARDS FINGERPRINT pair risk (eval-honest) ===")
    print(f"  {r}")
    print(f"  bar = REAL LB 0.70904; per-hand fingerprint AUC={roc_auc_score(y_hand, oof_h):.4f}")
    pl.DataFrame({"pair_id": evf["pair_id"].to_list(), "risk_score": np.clip(escore,0,1).astype(np.float32)}).write_parquet(OUT / "fingerprint_eval_risk.parquet")
    (OUT / "_fingerprint_summary.json").write_text(json.dumps(
        {"per_hand_auc": float(roc_auc_score(y_hand, oof_h)), "eval_honest_ap": r.eval_honest_ap,
         "confirmed_ap": r.confirmed_ap, "auc": r.auc}, indent=2), encoding="utf-8")
    return 0


def _build_dev_perhand_dims() -> pl.DataFrame:
    from anchor_repro.hand_eval_gpu import strength_for_seats
    dev_pairs = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2"])
    V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
    uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"]).unique()
    shared = uni.join(dev_pairs.lazy(), on="pair_id", how="inner").collect()
    seats = pl.read_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "total_contribution", "net_chips", "folded", "went_to_showdown",
         "won_share", "hole_card_1", "hole_card_2"])
    hands = pl.read_parquet(D / "hands.parquet").select(["hand_id", "board_cards", "final_pot", "big_blind"])
    acts = pl.scan_parquet(D / "actions.parquet").group_by(["hand_id", "player_id"]).agg(
        (pl.col("action").is_in(["bet", "raise", "all_in"])).sum().alias("n_agg"),
        (pl.col("action") == "check").sum().alias("n_check"), pl.len().alias("n_act")).collect()
    def side(col):
        return shared.join(dev_pairs.select(["pair_id", col]).rename({col: "player_id"}), on="pair_id").join(
            seats, on=["hand_id", "player_id"], how="left").join(acts, on=["hand_id", "player_id"], how="left").join(
            hands, on="hand_id", how="left")
    a = side("player_1"); b = side("player_2")
    from anchor_repro.hand_eval_gpu import strength_for_seats as sfs
    def strength(df):
        cl = [[x, y] + (z.split() if z else []) for x, y, z in zip(df["hole_card_1"].to_list(), df["hole_card_2"].to_list(), df["board_cards"].to_list())]
        return sfs(cl)
    a = a.with_columns(pl.Series("strength", strength(a))); b = b.with_columns(pl.Series("strength", strength(b)))
    A = a.select(["pair_id", "hand_id", "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share", "n_agg", "n_check", "n_act", "strength", "board_cards", "final_pot", "big_blind"])
    B = b.select(["pair_id", "hand_id", "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share", "n_agg", "n_check", "n_act", "strength"])
    df = A.join(B, on=["pair_id", "hand_id"], suffix="_b")
    bb = pl.col("big_blind")
    return df.with_columns(
        (pl.col("n_agg") + pl.col("n_agg_b")).alias("pair_agg"),
        (pl.col("total_contribution") / bb).alias("a_contrib_bb"), (pl.col("total_contribution_b") / bb).alias("b_contrib_bb"),
        ((pl.col("net_chips") * pl.col("net_chips_b")) < 0).cast(pl.Int8).alias("opposite_flow"),
        (pl.col("net_chips").abs() + pl.col("net_chips_b").abs()).alias("abs_flow"),
        (pl.col("folded").cast(pl.Int8) + pl.col("folded_b").cast(pl.Int8)).alias("n_folded"),
        (pl.col("won_share") + pl.col("won_share_b")).alias("pair_won"),
        (pl.max_horizontal("net_chips", "net_chips_b") / (pl.col("final_pot") + 1)).alias("winner_pot_share"),
        (pl.col("final_pot") / bb).alias("pot_bb"),
        pl.max_horizontal("strength", "strength_b").alias("max_strength"), pl.min_horizontal("strength", "strength_b").alias("min_strength"),
        (pl.col("strength") - pl.col("strength_b")).abs().alias("strength_gap"),
        (pl.col("board_cards").str.split(" ").list.len()).alias("n_board"))


if __name__ == "__main__":
    raise SystemExit(main())
