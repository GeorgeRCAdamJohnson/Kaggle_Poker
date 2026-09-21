"""What does our SHIPPED 0.70904 submission score per-component on the DEV holdout?

v30_component_decomp proved our lineage has NO dev behavior/evidence head (PairAP 0.98, Evid 0.00,
Behav 0.08). But the SHIPPED eval CSV carries borrowed honghanh heads. To know our true current
per-component baseline, score the shipped behavior+evidence columns on the DEV confirmed pairs:
join the shipped submission's (predicted_behavior, evidence_hand_*) by pair_id for pairs that ARE in
dev, put V30 dev risk, and score all three. That tells us where the borrowed heads actually land for
OUR risk ranking, and thus the exact upside of building/optimizing our own heads.

NOTE: shipped CSV is EVAL pairs (disjoint from dev), so its rows won't intersect dev pair_ids. If so,
we instead measure the honghanh heads on dev via the component_decomposition cache (already computed
0.3455/0.8935) and report the CLEAN arithmetic of the upside on the combined metric.

Run: python -m anchor_repro.shipped_component_decomp
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SHIPPED = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")


def main() -> int:
    from poker_collusion.config import PipelineConfig
    cfg = PipelineConfig()
    labels = pd.read_csv(Path(cfg.input_dir) / "development_labels.csv")
    dev_ids = set(labels["pair_id"].astype(str))

    sub = pd.read_csv(SHIPPED)
    sub["pair_id"] = sub["pair_id"].astype(str)
    inter = sub[sub["pair_id"].isin(dev_ids)]
    print(f"shipped rows: {len(sub)}   intersect dev confirmed pairs: {len(inter)}")

    # behavior head distribution on the shipped (eval) submission
    print("\nshipped predicted_behavior distribution (eval):")
    print(sub["predicted_behavior"].value_counts().to_string())
    nonneu = (sub["predicted_behavior"] != "none").mean()
    print(f"  non-'none' fraction: {nonneu:.4f}  ({(sub['predicted_behavior']!='none').sum()} rows)")

    # evidence fill
    ev_fill = (sub["evidence_hand_1"] != "NO_EVIDENCE").mean()
    print(f"  evidence_hand_1 filled (not NO_EVIDENCE): {ev_fill:.4f}")

    # ---- the upside arithmetic (metric = 0.70 PairAP + 0.20 Evid + 0.10 Behav) ----
    print("\n=== UPSIDE ARITHMETIC on the combined metric ===")
    print("Our lineage own dev heads:     PairAP 0.9806  Evid 0.0000  Behav 0.0806  -> combined 0.6945")
    print("honghanh dev heads (measured): PairAP 0.9747  Evid 0.3455  Behav 0.8935  -> combined 0.8407")
    # If we keep OUR PairAP 0.9806 and adopt honghanh-quality heads:
    for evid, behav, tag in [(0.0, 0.0806, "current-own-heads"),
                             (0.3455, 0.8935, "honghanh-quality-heads"),
                             (0.46, 0.8935, "honghanh-best-evid(0.46)+behav")]:
        combined = 0.70 * 0.9806 + 0.20 * evid + 0.10 * behav
        print(f"  PairAP 0.9806 + Evid {evid:.4f} + Behav {behav:.4f} = combined {combined:.4f}   [{tag}]")
    print("\nNOTE: dev combined != LB (dev PairAP 0.98 vs eval ~0.79). But the DELTA from adding heads")
    print("      (Evid 0->0.35 = +0.070; Behav 0.08->0.89 = +0.081) is a PairAP-independent lift that")
    print("      applies on TOP of whatever eval PairAP we have. Potential +0.10-0.15 combined, and it")
    print("      is NOT planted-overfit risk (behavior classification + evidence ranking generalize).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
