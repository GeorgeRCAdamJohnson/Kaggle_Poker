"""DISCORDANCE features — action-vs-TRUE-strength, directed at partner (the collusion mechanism).

Uses the GPU-built true_strength cache (showdown seats). For each showdown seat we have how strong
the hand ACTUALLY was + the outcome. Join per-seat aggression from actions. Then:

Per player FIELD BASELINE (all their showdown hands): expected aggression given strength, and
expected net given strength. A colluder DEVIATES from this baseline specifically WITH the partner.

Per pair (shared showdown hands, both members present):
  * strong_passive_excess : member has high true_strength but low aggression / folded / lost chips,
    MORE than their field baseline predicts -> soft-play / dump toward partner
  * loose_commit_excess    : member commits with low true_strength more than baseline -> loose-with-partner
  * directed_yield         : the passive/strong member's chips flow to the PARTNER (won_share alignment)
  * strength_gap_directed  : one member strong + passive while partner wins the pot

Aggregate to pair level. Train risk (confirmed labels, table-grouped OOF). Judge eval-honest.
Run:  python -m anchor_repro.discordance_features
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
from anchor_repro.eval_honest_harness import Harness, CACHE, D

OUT = CACHE / "discordance"
STRENGTH = OUT / "seat_strength.parquet"
SEED = 42


def _seat_aggression() -> pl.LazyFrame:
    """Per (hand_id, player_id): aggression count + committed flag from actions."""
    a = pl.scan_parquet(D / "actions.parquet")
    return a.group_by(["hand_id", "player_id"]).agg(
        (pl.col("action").is_in(["bet", "raise", "all_in"])).sum().alias("n_agg"),
        (pl.col("action") == "call").sum().alias("n_call"),
        (pl.col("action") == "check").sum().alias("n_check"),
        pl.len().alias("n_act"),
    ).with_columns(
        (pl.col("n_agg") / (pl.col("n_act") + 1)).alias("agg_rate"),
    )


def _seat_level() -> pl.DataFrame:
    """Per showdown seat: true_strength + aggression + outcome + player field baselines."""
    st = pl.read_parquet(STRENGTH)  # hand_id, player_id, true_strength, net_chips, total_contribution, won_share
    agg = _seat_aggression()
    df = st.lazy().join(agg, on=["hand_id", "player_id"], how="left").with_columns(
        pl.col("n_agg").fill_null(0), pl.col("agg_rate").fill_null(0.0),
    ).with_columns(
        # per-seat discordance primitives
        (pl.col("true_strength") * (1.0 - pl.col("agg_rate"))).alias("strong_passive"),      # strong but passive
        ((1.0 - pl.col("true_strength")) * (pl.col("total_contribution") > 0).cast(pl.Float64)).alias("weak_commit"),
        (pl.col("net_chips") < 0).cast(pl.Int8).alias("lost"),
    ).collect(engine="streaming")
    # player field baseline: mean strong_passive / weak_commit over ALL their showdown hands
    base = df.group_by("player_id").agg(
        pl.col("strong_passive").mean().alias("bl_strong_passive"),
        pl.col("weak_commit").mean().alias("bl_weak_commit"),
        pl.col("true_strength").mean().alias("bl_strength"),
        pl.len().alias("bl_n"),
    )
    return df.join(base, on="player_id", how="left")


def _pair_discordance(pairs: pl.DataFrame, seat: pl.DataFrame) -> pl.DataFrame:
    """Per pair: DIRECTED discordance in shared showdown hands minus each member's field baseline."""
    # long: for each pair, both members' seat rows on shared hands
    a = pairs.select(["pair_id", pl.col("player_1").alias("player_id")]).join(seat, on="player_id", how="inner")
    b = pairs.select(["pair_id", pl.col("player_2").alias("player_id")]).join(seat, on="player_id", how="inner")
    # shared hands = same hand_id for both members
    j = a.join(b, on=["pair_id", "hand_id"], suffix="_b")
    j = j.with_columns(
        # excess discordance over each member's own field baseline (directed = only in shared hands)
        (pl.col("strong_passive") - pl.col("bl_strong_passive")).alias("a_sp_excess"),
        (pl.col("strong_passive_b") - pl.col("bl_strong_passive_b")).alias("b_sp_excess"),
        (pl.col("weak_commit") - pl.col("bl_weak_commit")).alias("a_wc_excess"),
        (pl.col("weak_commit_b") - pl.col("bl_weak_commit_b")).alias("b_wc_excess"),
        # directed yield: one strong+passive+lost while the OTHER won (won_share)
        ((pl.col("true_strength") > 0.4) & (pl.col("agg_rate") < 0.1) & (pl.col("won_share_b") > 0)).cast(pl.Int8).alias("a_yield_to_b"),
        ((pl.col("true_strength_b") > 0.4) & (pl.col("agg_rate_b") < 0.1) & (pl.col("won_share") > 0)).cast(pl.Int8).alias("b_yield_to_a"),
        # strength gap when one wins: partner much weaker but wins the chips
        (pl.col("net_chips") * pl.col("net_chips_b") < 0).cast(pl.Int8).alias("opposite_flow"),
    )
    per_pair = j.group_by("pair_id").agg(
        pl.len().alias("n_shared_showdown"),
        pl.max_horizontal(pl.col("a_sp_excess").mean(), pl.col("b_sp_excess").mean()).alias("sp_excess_max"),
        (pl.col("a_sp_excess").mean() + pl.col("b_sp_excess").mean()).alias("sp_excess_sum"),
        pl.max_horizontal(pl.col("a_wc_excess").mean(), pl.col("b_wc_excess").mean()).alias("wc_excess_max"),
        (pl.col("a_yield_to_b").mean() + pl.col("b_yield_to_a").mean()).alias("directed_yield_rate"),
        pl.col("a_yield_to_b").sum().alias("a_yield_n"),
        pl.col("b_yield_to_a").sum().alias("b_yield_n"),
        pl.col("opposite_flow").mean().alias("opposite_flow_rate"),
        # directed-yield ASYMMETRY (colluder: consistent direction)
        ((pl.col("a_yield_to_b").sum() - pl.col("b_yield_to_a").sum()).abs()
         / (pl.col("a_yield_to_b").sum() + pl.col("b_yield_to_a").sum() + 1)).alias("yield_asymmetry"),
    )
    return per_pair


def main() -> int:
    H = Harness()
    seat = _seat_level()
    print(f"seat-level: {seat.height:,} showdown seats with strength+aggression")

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])

    t = time.time()
    dev_d = _pair_discordance(labels, seat)
    eval_d = _pair_discordance(eval_pairs, seat)
    print(f"pair discordance: dev={dev_d.height}/{labels.height} eval={eval_d.height}/{eval_pairs.height} "
          f"(pairs with >=1 shared showdown hand) ({time.time()-t:.0f}s)")

    # table groups
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    ptbl = (pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
            .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
            .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
            .select(["pair_id", "table_id"]).collect())

    dev = labels.join(dev_d, on="pair_id", how="left").join(ptbl, on="pair_id", how="left")
    feats = [c for c in dev_d.columns if c != "pair_id"]
    X = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    groups = dev["table_id"].fill_null("?").to_numpy()
    cov = float((dev_d.height) / labels.height)
    print(f"feats={len(feats)} pos={int(y.sum())} coverage={cov:.2%} of dev pairs have showdown discordance")

    print("\n=== univariate AUC (discordance features) ===")
    for c in feats:
        v = X[c].to_numpy()
        try:
            auc = max(roc_auc_score(y, v), roc_auc_score(y, -v))
            print(f"  {auc:.4f}  {c}")
        except Exception:
            pass

    # OOF risk
    oof = np.zeros(len(y), np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                        subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    for tr, va in sgkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)}, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**params, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=300)

    evf = eval_pairs.join(eval_d, on="pair_id", how="left")
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    escore = full.predict(xgb.DMatrix(Xe))
    r = H.judge("discordance", oof, dev["pair_id"].to_list(), escore, evf["pair_id"].to_list())
    print(f"\n=== DISCORDANCE risk (eval-honest; confirmed base -> trustworthy) ===")
    print(f"  {r}")
    print(f"  bar = REAL LB 0.70904 (harness est). confirmed AP {r.confirmed_ap:.4f}")

    pl.DataFrame({"pair_id": evf["pair_id"].to_list(), "risk_score": np.clip(escore,0,1).astype(np.float32)}).write_parquet(OUT / "discordance_eval_risk.parquet")
    (OUT / "_discordance_summary.json").write_text(json.dumps(
        {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc,
         "coverage": cov}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
