"""Are V30 (0.92) and the E-detectors (0.65-0.68) on the SAME eval-honest scale?

The user is right to suspect a scale mismatch. V30's 0.9203 is eval-honest PairAP (a COMPONENT,
importance-weighted estimate), NOT the LB (0.70904 = full composite). And the E-detectors must be
judged on the IDENTICAL pair set + weights or the comparison is meaningless.

This script judges, through ONE Harness instance, on the IDENTICAL labeled pair set and weights:
  - V30 rank_sum_w50 (the shipped 0.70904 arm)
  - V30 base_risk alone (no specialists)
  - E2 best (cached)
  - a plain do-nothing reference (random)
and prints confirmed AP + eval-honest AP for each so the scales are directly comparable. It also
reports what fraction of the 0.92 is the BASE vs the specialists, and sanity-checks that eval-honest
PairAP ~0.92 is consistent with LB 0.709 given the 0.70/0.20/0.10 weights.

Run:  python -m anchor_repro.scale_check
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")


def main() -> int:
    H = Harness()
    oof = pl.read_parquet(SEQ / "oof_rows.parquet").filter(pl.col("known"))
    ids = oof["pair_id"].to_list()
    y = oof["y"].to_numpy().astype(int)
    # align harness weights to THIS pair order
    wmap = dict(zip(H.dev_pair_ids, H.w))
    w = np.array([wmap[p] for p in ids])

    def score(name, v):
        conf = average_precision_score(y, v)
        eh = weighted_ap(y, v, w)
        print(f"  {name:<26} confirmed_AP={conf:.4f}  eval_honest_AP={eh:.4f}")
        return eh

    print(f"=== SAME pair set (n={len(y)}, pos={y.sum()}), SAME weights, SAME metric ===\n")
    print("All numbers below are PairAP COMPONENT (not the LB composite).")
    print("LB reference: real leaderboard = 0.70904 = 0.70*PairAP + 0.20*Evidence + 0.10*Behavior.\n")

    v30 = oof["rank_sum_w50"].to_numpy()
    base = oof["baseline_risk"].to_numpy()
    spec = np.column_stack([oof["specialist_directed"].to_numpy(), oof["specialist_soft"].to_numpy(),
                            oof["specialist_isolation"].to_numpy()])
    union = np.clip(spec.sum(1), 0, 1)

    score("V30 rank_sum_w50 (shipped)", v30)
    score("V30 base_risk alone", base)
    score("V30 specialists union (raw)", union)

    e2 = pl.read_parquet(CACHE / "_E2_best_dev_oof.parquet")
    e2m = dict(zip(e2["pair_id"].to_list(), e2["E2_oof"].to_numpy()))
    e2v = np.array([e2m.get(p, np.nan) for p in ids])
    if not np.isnan(e2v).any():
        score("E2 best (PU, generic feats)", e2v)

    rng = np.random.default_rng(0)
    score("random reference", rng.random(len(y)))

    # Sanity: is eval-honest PairAP 0.92 consistent with LB 0.709?
    print("\n=== consistency check: does PairAP~0.92 square with LB 0.70904? ===")
    for ev, beh in [(0.35, 0.60), (0.40, 0.89), (0.30, 0.50)]:
        pairap_implied = (0.70904 - 0.20 * ev - 0.10 * beh) / 0.70
        print(f"  if Evidence={ev:.2f}, Behavior={beh:.2f}  ->  implied LB-PairAP = {pairap_implied:.4f}")
    print("  (LB-PairAP is over the PRIVATE eval labels; our 0.92 is the importance-weighted")
    print("   ESTIMATE of eval-PairAP from dev. A gap between the two = residual dev->eval error")
    print("   the importance weights don't fully correct. So 0.92 (estimate) and ~0.75-0.85")
    print("   (implied from LB) can BOTH be 'true' at different fidelity — the harness is an")
    print("   estimator, not the LB. The E-vs-V30 COMPARISON is valid because SAME estimator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
