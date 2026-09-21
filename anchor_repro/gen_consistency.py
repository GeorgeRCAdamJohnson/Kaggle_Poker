"""Probe: does the planting pattern REPEAT across a pair's shared-hand timeline?

Per-hand evidence plateaus at ~0.28 because a single hand is ambiguous. But a truly
planted pair should exhibit its family signature REPEATEDLY across its shared hands, while
a coincidental (negative) pair should not. This probes pair-level consistency features:
concentration/repetition of per-family per-hand signal, vs the pair label.

If consistency features separate positive vs negative pairs materially BETTER than the
per-hand aggregates alone, that is the structural lever the per-hand view misses (and the
kind of signal that lifts pair-detection + evidence + behavior together).

Run:  python -m anchor_repro.gen_consistency
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")


def main() -> int:
    dev = pl.read_parquet(CACHE / "dev_layer2.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label", "behavior_family"])
    dev = dev.join(labels, on="pair_id", how="left")

    # Per-hand family signals already in Layer-2: dir_signal, soft_signal, iso_squeeze_signal.
    # Consistency = how strongly and REPEATEDLY a pair shows its best family signal.
    sig_cols = ["dir_signal", "soft_signal", "iso_squeeze_signal", "abs_flow_bb",
                "both_commit", "iso_pair_preflop_agg", "soft_both_sd"]

    agg = dev.group_by("pair_id").agg(
        pl.first("label").alias("label"),
        pl.len().alias("n_hands"),
        # per-signal: max, count over threshold, top-k mean, fraction active
        *[pl.col(c).max().alias(f"{c}_max") for c in sig_cols],
        *[pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in sig_cols],
        *[(pl.col(c) > 0).mean().alias(f"{c}_rate") for c in sig_cols],
        *[(pl.col(c) > 0).sum().alias(f"{c}_count") for c in sig_cols],
    )
    # Consistency ratios: repeated signal relative to volume.
    agg = agg.with_columns(
        (pl.col("dir_signal_count") / pl.col("n_hands")).alias("dir_consistency"),
        (pl.col("iso_pair_preflop_agg_count") / pl.col("n_hands")).alias("iso_consistency"),
        (pl.col("soft_both_sd_count") / pl.col("n_hands")).alias("soft_consistency"),
        (pl.col("both_commit_count") / pl.col("n_hands")).alias("commit_consistency"),
    )
    y = agg["label"].to_numpy().astype(np.int8)
    print(f"dev pairs={agg.height} positives={int(y.sum())}")

    print("\n=== PAIR-LEVEL separation: per-hand AGGREGATES vs CONSISTENCY (AP / AUC) ===")
    cand = [c for c in agg.columns if c not in ("pair_id", "label")]
    rows = []
    for c in cand:
        v = agg[c].fill_null(0).to_numpy().astype(float)
        if np.unique(v).size < 2:
            continue
        ap = average_precision_score(y, v)
        auc = roc_auc_score(y, v)
        rows.append((c, ap, auc))
    rows.sort(key=lambda x: -x[1])
    print(f"{'feature':32s} {'AP':>8} {'AUC':>8}")
    for c, ap, auc in rows[:20]:
        print(f"  {c:30s} {ap:8.4f} {auc:8.4f}")
    print(f"\nbase positive rate {y.mean():.4f}")

    # Head-to-head: is a consistency RATIO better than the matched raw max/count?
    print("\n=== consistency ratios vs their raw counterparts ===")
    for ratio, raw in [("dir_consistency", "dir_signal_max"),
                       ("iso_consistency", "iso_pair_preflop_agg_max"),
                       ("commit_consistency", "both_commit_count")]:
        rv = agg[ratio].fill_null(0).to_numpy().astype(float)
        wv = agg[raw].fill_null(0).to_numpy().astype(float)
        print(f"  {ratio:20s} AP={average_precision_score(y,rv):.4f}   vs {raw:24s} AP={average_precision_score(y,wv):.4f}")

    summary = {"top_features": [(c, float(ap), float(auc)) for c, ap, auc in rows[:20]],
               "base_rate": float(y.mean())}
    (CACHE / "_consistency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
