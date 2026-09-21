"""Named street-sequence TEMPLATES, position-aware (honghanh §9 path to 0.90).

The 0.64 writeup's central thesis: "the gap from 0.64 to 0.90 is matching the synthetic
generator's hand templates." Its §9 names the specific unbuilt templates:
  (a) river heads-up CHECK-CHECK between the pair (soft-play showdown)
  (b) preflop SQUEEZE (pair raises/3bets, outsiders fold) THEN postflop CHECK-DOWN (isolation)
  (c) LOSER commits/all-in with a WEAK hole then loses to partner (directed dump)
plus POSITION (seat_no): "isolation and dumps are not seat-symmetric."

We built generic sequence aggregates (V30 seq_*) and a raw transformer — both weak. We NEVER built
these SPECIFIC named templates as explicit per-hand detectors with position. This does exactly that,
from the ordered action stream, then aggregates per pair.

Judged eval-honest vs V30 0.709 and the fingerprint 0.16.
Run:  python -m anchor_repro.street_templates
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.hand_eval_gpu import strength_for_seats
from anchor_repro.eval_honest_harness import Harness, CACHE, D

OUT = CACHE / "templates"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")


def _templates_for_pairs(pairs: pl.DataFrame, universe: pl.LazyFrame) -> pl.DataFrame:
    """Per (pair_id, hand_id): the named street-template signals, position-aware."""
    uni = universe.select(["pair_id", "hand_id"]).unique()
    pf = pairs.select(["pair_id", "player_1", "player_2"]).lazy()
    ph = uni.join(pf, on="pair_id", how="inner")  # pair_id, hand_id, player_1, player_2

    # ordered actions restricted to these hands, tagged by member role
    acts = pl.scan_parquet(D / "actions.parquet").select(
        ["hand_id", "action_no", "street", "player_id", "action", "amount", "to_call"])
    ph_hands = ph.select(["pair_id", "hand_id", "player_1", "player_2"])
    j = ph_hands.join(acts, on="hand_id", how="inner").with_columns(
        pl.when(pl.col("player_id") == pl.col("player_1")).then(1)
          .when(pl.col("player_id") == pl.col("player_2")).then(2).otherwise(0).alias("role"),
    )
    # per (pair,hand,street): member action summaries
    def is_agg(): return pl.col("action").is_in(["bet", "raise", "all_in"])
    per_street = j.group_by(["pair_id", "hand_id", "street"]).agg(
        # member1
        (is_agg() & (pl.col("role") == 1)).sum().alias("m1_agg"),
        ((pl.col("action") == "check") & (pl.col("role") == 1)).sum().alias("m1_check"),
        ((pl.col("action") == "fold") & (pl.col("role") == 1)).sum().alias("m1_fold"),
        ((pl.col("action") == "all_in") & (pl.col("role") == 1)).sum().alias("m1_allin"),
        # member2
        (is_agg() & (pl.col("role") == 2)).sum().alias("m2_agg"),
        ((pl.col("action") == "check") & (pl.col("role") == 2)).sum().alias("m2_check"),
        ((pl.col("action") == "fold") & (pl.col("role") == 2)).sum().alias("m2_fold"),
        ((pl.col("action") == "all_in") & (pl.col("role") == 2)).sum().alias("m2_allin"),
        # outsiders
        (is_agg() & (pl.col("role") == 0)).sum().alias("out_agg"),
        ((pl.col("action") == "fold") & (pl.col("role") == 0)).sum().alias("out_fold"),
        pl.len().alias("n_act"),
    ).collect(engine="streaming")

    # pivot streets to columns per (pair,hand)
    def street_col(st, col):
        return pl.col(col).filter(pl.col("street") == st).sum()
    per_hand = per_street.group_by(["pair_id", "hand_id"]).agg(
        # river both members check (heads-up check-check soft-play)
        street_col("river", "m1_check").alias("r_m1_check"),
        street_col("river", "m2_check").alias("r_m2_check"),
        street_col("river", "m1_agg").alias("r_m1_agg"),
        street_col("river", "m2_agg").alias("r_m2_agg"),
        # preflop squeeze: both members aggressive preflop, outsiders fold
        street_col("preflop", "m1_agg").alias("pf_m1_agg"),
        street_col("preflop", "m2_agg").alias("pf_m2_agg"),
        street_col("preflop", "out_fold").alias("pf_out_fold"),
        street_col("preflop", "out_agg").alias("pf_out_agg"),
        # postflop check-down after squeeze
        (street_col("flop", "m1_check") + street_col("turn", "m1_check") + street_col("river", "m1_check")).alias("post_m1_check"),
        (street_col("flop", "m2_check") + street_col("turn", "m2_check") + street_col("river", "m2_check")).alias("post_m2_check"),
        (street_col("flop", "m1_agg") + street_col("turn", "m1_agg") + street_col("river", "m1_agg")).alias("post_m1_agg"),
        (street_col("flop", "m2_agg") + street_col("turn", "m2_agg") + street_col("river", "m2_agg")).alias("post_m2_agg"),
        # all-ins
        (street_col("preflop","m1_allin")+street_col("flop","m1_allin")+street_col("turn","m1_allin")+street_col("river","m1_allin")).alias("m1_allin"),
        (street_col("preflop","m2_allin")+street_col("flop","m2_allin")+street_col("turn","m2_allin")+street_col("river","m2_allin")).alias("m2_allin"),
    )

    # join seat outcomes + hole strength + seat_no (position) + net
    seats = pl.read_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "seat_no", "net_chips", "went_to_showdown", "hole_card_1", "hole_card_2", "folded"])
    hands = pl.read_parquet(D / "hands.parquet").select(["hand_id", "board_cards", "button_seat", "big_blind"])

    def member_seat(col, tag):
        return ph.select(["pair_id", "hand_id", col]).rename({col: "player_id"}).join(
            seats.lazy(), on=["hand_id", "player_id"], how="left").rename({
            "seat_no": f"{tag}_seat", "net_chips": f"{tag}_net", "went_to_showdown": f"{tag}_sd",
            "hole_card_1": f"{tag}_h1", "hole_card_2": f"{tag}_h2", "folded": f"{tag}_fold"}).drop("player_id")
    m1 = member_seat("player_1", "m1").collect()
    m2 = member_seat("player_2", "m2").collect()
    base = per_hand.join(m1, on=["pair_id", "hand_id"], how="left").join(m2, on=["pair_id", "hand_id"], how="left").join(
        hands, on="hand_id", how="left")

    # hole strengths (GPU) for the loser-weak-hole template
    def strength(df, h1, h2, bd):
        cl = [[a, b] + (z.split() if z else []) for a, b, z in zip(df[h1].to_list(), df[h2].to_list(), df[bd].to_list())]
        return strength_for_seats(cl)
    base = base.with_columns(pl.Series("m1_str", strength(base, "m1_h1", "m1_h2", "board_cards")),
                             pl.Series("m2_str", strength(base, "m2_h1", "m2_h2", "board_cards")))

    bb = pl.col("big_blind")
    return base.with_columns(
        # TEMPLATE A: river heads-up check-check (both members check river, neither aggressive)
        ((pl.col("r_m1_check") > 0) & (pl.col("r_m2_check") > 0) & (pl.col("r_m1_agg") == 0) & (pl.col("r_m2_agg") == 0)).cast(pl.Int8).alias("t_river_checkcheck"),
        # TEMPLATE B: preflop squeeze (both agg preflop, outsiders fold, no outsider agg) THEN postflop checkdown
        (((pl.col("pf_m1_agg") > 0) & (pl.col("pf_m2_agg") > 0) & (pl.col("pf_out_fold") > 0) & (pl.col("pf_out_agg") == 0)
          & ((pl.col("post_m1_check") + pl.col("post_m2_check")) > 0) & ((pl.col("post_m1_agg") + pl.col("post_m2_agg")) == 0))).cast(pl.Int8).alias("t_squeeze_checkdown"),
        # TEMPLATE C: loser all-in / big loss with WEAK hole, winner is partner (directed dump)
        (((pl.col("m1_net") < 0) & (pl.col("m2_net") > 0) & (pl.col("m1_str") < 0.25) & ((pl.col("m1_allin") > 0) | (pl.col("m1_net")/bb < -20)))
         | ((pl.col("m2_net") < 0) & (pl.col("m1_net") > 0) & (pl.col("m2_str") < 0.25) & ((pl.col("m2_allin") > 0) | (pl.col("m2_net")/bb < -20)))).cast(pl.Int8).alias("t_weakhole_dump"),
        # POSITION: dump direction relative to button (downhill = loser acts after winner)
        ((pl.col("m1_net") - pl.col("m2_net")).sign() * (pl.col("m1_seat") - pl.col("m2_seat")).sign()).alias("dump_seat_dir"),
        (pl.col("m1_seat") - pl.col("m2_seat")).abs().alias("seat_gap"),
        # loser strong-fold (soft): a member folded WITH a strong hole while the partner was aggressive
        ((pl.col("m1_fold").fill_null(False) & (pl.col("m1_str") > 0.4) & ((pl.col("post_m2_agg") + pl.col("pf_m2_agg")) > 0))
         | (pl.col("m2_fold").fill_null(False) & (pl.col("m2_str") > 0.4) & ((pl.col("post_m1_agg") + pl.col("pf_m1_agg")) > 0))).cast(pl.Int8).alias("t_strongfold_to_partner"),
    ).select(["pair_id", "hand_id", "t_river_checkcheck", "t_squeeze_checkdown", "t_weakhole_dump",
              "t_strongfold_to_partner", "dump_seat_dir", "seat_gap"])


TEMPLATES = ["t_river_checkcheck", "t_squeeze_checkdown", "t_weakhole_dump", "t_strongfold_to_partner"]


def _pair_features(th: pl.DataFrame) -> pl.DataFrame:
    return th.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        *[pl.col(t).sum().alias(f"{t}_n") for t in TEMPLATES],
        *[pl.col(t).mean().alias(f"{t}_rate") for t in TEMPLATES],
        pl.col("dump_seat_dir").mean().alias("dump_dir_mean"),
        pl.col("dump_seat_dir").filter(pl.col("t_weakhole_dump") == 1).mean().alias("dump_dir_on_dumps"),
        pl.col("seat_gap").mean().alias("seat_gap_mean"),
        (pl.col("t_river_checkcheck") + pl.col("t_squeeze_checkdown") + pl.col("t_weakhole_dump")).sum().alias("any_template_n"),
    )


def main() -> int:
    H = Harness()
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])

    t = time.time()
    dev_th = _templates_for_pairs(labels, pl.scan_parquet(V30_DEV))
    dev_pp = _pair_features(dev_th)
    print(f"dev templates ({time.time()-t:.0f}s), pairs={dev_pp.height}")
    t = time.time()
    eval_th = _templates_for_pairs(eval_pairs, pl.scan_parquet(V30_EVAL))
    eval_pp = _pair_features(eval_th)
    print(f"eval templates ({time.time()-t:.0f}s), pairs={eval_pp.height}")

    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    ptbl = (pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
            .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
            .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
            .select(["pair_id", "table_id"]).collect())
    dev = labels.join(dev_pp, on="pair_id", how="left").join(ptbl, on="pair_id", how="left")
    feats = [c for c in dev_pp.columns if c != "pair_id"]
    X = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].fill_null("?").to_numpy()

    print("\n=== template features univariate AUC ===")
    for c in feats:
        v = X[c].to_numpy()
        try:
            a = roc_auc_score(y, v); print(f"  {max(a,1-a):.4f}  {c}")
        except Exception: pass

    oof = np.zeros(len(y), np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    for tr, va in sgkf.split(X, y, g):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=300)
    evf = eval_pairs.select(["pair_id"]).join(eval_pp, on="pair_id", how="left")
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    escore = full.predict(xgb.DMatrix(Xe))
    r = H.judge("street_templates", oof, dev["pair_id"].to_list(), escore, evf["pair_id"].to_list())
    print(f"\n=== STREET TEMPLATES pair risk (eval-honest) ===")
    print(f"  {r}")
    print(f"  vs fingerprint 0.16 | vs V30 real LB 0.70904")
    pl.DataFrame({"pair_id": evf["pair_id"].to_list(), "risk_score": np.clip(escore,0,1).astype(np.float32)}).write_parquet(OUT / "templates_eval_risk.parquet")
    (OUT / "_templates_summary.json").write_text(json.dumps(
        {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
