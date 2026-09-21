"""Aggregate per-hand planted scores to a pair-level detector (generator recovery, task 2).

Retrains the per-hand planted classifier on ALL dev positive-pair hands (full fit), scores
every dev pair-hand (via table-grouped OOF, reusing dev_perhand_oof.parquet) and every eval
pair-hand (GPU, chunked), then aggregates to pair-level statistics (max / top-k mean / count
of high-planted hands). Measures the pair-level signal against confirmed labels + PU-stress,
and compares to the frozen 0.70904 baseline risk.

The question: does this transport-clean per-hand signal, aggregated, add PairAP over the
V30 baseline — even though its standalone evidence recovery is weak?

BUILD + MEASURE ONLY. GPU via anchor_repro.gpu_config.

Run:  python -m anchor_repro.generator_pairlevel
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score

from anchor_repro.gpu_config import xgb_params, xgb_device

D = Path("data/poker")
OUT = Path("outputs/poker_collusion/generator_hunt")
SEED = 42
PU_STRESS_WEIGHT = 112540 / 24000


def _agg_pair(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Pair-level aggregates of a per-hand planted score."""
    return df.group_by("pair_id").agg(
        pl.col(score_col).max().alias("g_max"),
        pl.col(score_col).top_k(3).mean().alias("g_top3"),
        pl.col(score_col).top_k(5).mean().alias("g_top5"),
        pl.col(score_col).mean().alias("g_mean"),
        (pl.col(score_col) > 0.5).sum().alias("g_n_hi"),
        (pl.col(score_col) > 0.5).mean().alias("g_rate_hi"),
        pl.col(score_col).sum().alias("g_sum"),
    )


def main() -> int:
    feat_cols = json.loads((OUT / "generator_feature_columns.json").read_text())

    # --- Dev: OOF per-hand scores already computed; aggregate to pair level. ---
    oof = pl.read_parquet(OUT / "dev_perhand_oof.parquet")  # pair_id, hand_id, is_planted, behavior_family, planted_score
    dev_pair = _agg_pair(oof, "planted_score")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label", "behavior_family"])
    # NOTE: oof only has POSITIVE pairs. To measure PairAP vs negatives we must also score
    # negative-pair hands. Score ALL dev pair-hands with a full-fit model (table-grouped OOF
    # is only defined for the training positives; for negatives we use a full model trained on
    # all positive-pair hands — negatives were never in training so this is not leakage of the
    # pair label, but be honest: it is an in-domain apply, not OOF, for negatives).
    dev_all = pl.read_parquet(OUT / "dev_pairhand_features.parquet")
    hands_tbl = pl.scan_parquet(D / "hands.parquet").select(["hand_id", "table_id"]).collect()
    dev_all = dev_all.join(labels, on="pair_id", how="left")
    Xpos = (
        dev_all.filter(pl.col("label") == 1).select(feat_cols)
        .to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    )
    ypos = dev_all.filter(pl.col("label") == 1)["is_planted"].to_numpy().astype(np.int8)
    spw = (ypos == 0).sum() / max((ypos == 1).sum(), 1)
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=6,
                        eta=0.05, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=5.0, scale_pos_weight=float(spw))
    t = time.time()
    full_model = xgb.train(params, xgb.DMatrix(Xpos, label=ypos), num_boost_round=200)
    print(f"full per-hand model trained ({time.time()-t:.1f}s, device={xgb_device()})")

    # Score all dev pair-hands (positives get OOF for honesty; negatives get full-model score).
    dev_neg = dev_all.filter(pl.col("label") == 0)
    Xneg = dev_neg.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    neg_scores = full_model.predict(xgb.DMatrix(Xneg)) if len(Xneg) else np.zeros(0)
    neg_df = dev_neg.select(["pair_id", "hand_id"]).with_columns(pl.Series("planted_score", neg_scores))
    pos_df = oof.select(["pair_id", "hand_id", "planted_score"])
    dev_scored = pl.concat([pos_df, neg_df], how="vertical")
    dev_pair_all = _agg_pair(dev_scored, "planted_score").join(labels, on="pair_id", how="left")

    y = dev_pair_all["label"].to_numpy().astype(np.int8)
    print(f"\ndev pairs={dev_pair_all.height} positives={int(y.sum())}")
    print("=== PAIR-LEVEL confirmed PairAP of generator aggregates ===")
    agg_cols = ["g_max", "g_top3", "g_top5", "g_mean", "g_n_hi", "g_rate_hi", "g_sum"]
    for c in agg_cols:
        v = dev_pair_all[c].fill_null(0).to_numpy().astype(float)
        ap = average_precision_score(y, v)
        print(f"  {c:10s} PairAP={ap:.4f}")

    # Compare to / combine with the frozen 0.70904 baseline risk.
    base = pl.read_parquet("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet").select(
        ["pair_id", "known", "y", "baseline_risk", "rank_sum_w50"]
    )
    merged = dev_pair_all.join(base, on="pair_id", how="inner")
    yb = merged["y"].to_numpy().astype(np.int8)
    known = merged["known"].to_numpy().astype(bool)
    sw = np.where(known, 1.0, PU_STRESS_WEIGHT)
    baseline = merged["rank_sum_w50"].to_numpy().astype(float)

    def rankn(a):
        return pd.Series(a).rank(pct=True).to_numpy()

    print(f"\n=== COMBINE generator pair-signal with frozen rank_sum_w50 (n={merged.height}) ===")
    base_conf = average_precision_score(yb[known], baseline[known])
    base_stress = average_precision_score(yb, baseline, sample_weight=sw)
    print(f"  baseline rank_sum_w50:        confirmed={base_conf:.5f} pu_stress={base_stress:.5f}")
    br = rankn(baseline)
    best = None
    for gcol in ["g_max", "g_top5", "g_sum", "g_n_hi"]:
        g = rankn(merged[gcol].fill_null(0).to_numpy().astype(float))
        for w in [0.1, 0.2, 0.3, 0.5]:
            blend = (1 - w) * br + w * g
            cc = average_precision_score(yb[known], blend[known])
            ss = average_precision_score(yb, blend, sample_weight=sw)
            tag = "  " if ss <= base_stress else "^^"
            print(f"  {tag} {gcol:8s} w={w:.1f}: confirmed={cc:.5f} ({cc-base_conf:+.5f}) "
                  f"pu_stress={ss:.5f} ({ss-base_stress:+.5f})")
            if best is None or ss > best[1]:
                best = (f"{gcol}@{w}", ss, cc)
    print(f"\nbest pu_stress arm: {best[0]} pu_stress={best[1]:.5f} ({best[1]-base_stress:+.5f}) "
          f"confirmed={best[2]:.5f} ({best[2]-base_conf:+.5f})")

    summary = {
        "baseline_confirmed": float(base_conf), "baseline_pu_stress": float(base_stress),
        "best_arm": best[0], "best_pu_stress": float(best[1]), "best_pu_stress_delta": float(best[1] - base_stress),
        "best_confirmed": float(best[2]), "best_confirmed_delta": float(best[2] - base_conf),
        "note": "generator per-hand signal aggregated to pair level, rank-blended onto frozen rank_sum_w50",
    }
    (OUT / "_pairlevel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'_pairlevel_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
