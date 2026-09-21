"""ASSEMBLE the complete E-line submission + compute the whole-composite gate.

Combines the three independent E-line columns:
  risk_score        <- eline/risk_eval.parquet  (base+unionmax_w50, 0.6873 eval-honest PairAP)
  predicted_behavior<- eline/behavior_eval.parquet
  evidence_hand_1..5<- eline/evidence_eval.parquet
into a 112,540-row submission, validates the official contract, and computes the whole-submission
composite estimate on the eval-honest harness:

  composite = 0.70 * PairAP(eval-honest) + 0.20 * Evidence-MAP@5(dev) + 0.10 * Behavior-MAP(dev)

Honest note: PairAP is the eval-honest estimate (runs ~0.1 HOT, established earlier); Evidence and
Behavior are DEV numbers (their eval transfer is unmeasured). So the composite here is an UPPER-ish
estimate. GATE = 0.69: if the composite clears it, emit + (caller) submits; else record honestly.

Run:  python -m anchor_repro.eline_assemble
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D

OUT = CACHE / "eline"
SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
CAND = OUT / "candidate_eline_full.csv"
EV_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior", *EV_COLS]


def main() -> int:
    H = Harness()

    # --- component numbers ---
    risk_sum = json.loads((OUT / "_risk_summary.json").read_text())
    ev_sum = json.loads((OUT / "_evidence_summary.json").read_text())
    beh_sum = json.loads((OUT / "_behavior_summary.json").read_text())
    pairap = risk_sum["arm_scores"][risk_sum["best_arm"]]  # eval-honest PairAP
    ev_map5 = ev_sum["dev_map5"]
    beh_map = beh_sum["dev_behavior_map"]
    composite = 0.70 * pairap + 0.20 * ev_map5 + 0.10 * beh_map
    print("=== E-line whole-submission composite (harness estimate) ===")
    print(f"  PairAP (eval-honest, ~0.1 HOT) : {pairap:.4f}  x0.70 = {0.70*pairap:.4f}")
    print(f"  Evidence-MAP@5 (dev)           : {ev_map5:.4f}  x0.20 = {0.20*ev_map5:.4f}")
    print(f"  Behavior-MAP (dev)             : {beh_map:.4f}  x0.10 = {0.10*beh_map:.4f}")
    print(f"  COMPOSITE                      : {composite:.4f}   (gate 0.69; V30 LB 0.70904)")

    # --- assemble CSV ---
    risk = pl.read_parquet(OUT / "risk_eval.parquet").select(["pair_id", pl.col("risk_best").alias("risk_score")])
    beh = pl.read_parquet(OUT / "behavior_eval.parquet")
    ev = pl.read_parquet(OUT / "evidence_eval.parquet")
    sub = (risk.join(beh, on="pair_id", how="inner").join(ev, on="pair_id", how="inner")
           .select(SUB_COLS))
    # clip risk to [0,1]
    sub = sub.with_columns(pl.col("risk_score").clip(0.0, 1.0))

    # --- validate contract ---
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id"])
    checks = {
        "rows_112540": sub.height == 112540,
        "schema": list(sub.columns) == SUB_COLS,
        "pair_set": set(sub["pair_id"].to_list()) == set(eval_pairs["pair_id"].to_list()),
        "risk_in_01": bool(sub["risk_score"].min() >= 0 and sub["risk_score"].max() <= 1),
        "no_null": int(sub.null_count().sum_horizontal().item()) == 0,
    }
    # no duplicate evidence (excluding NO_EVIDENCE) per row
    dup = 0
    ev_np = sub.select(EV_COLS).to_numpy()
    for row in ev_np:
        real = [h for h in row if h != "NO_EVIDENCE"]
        if len(real) != len(set(real)):
            dup += 1
    checks["no_dup_evidence"] = dup == 0
    print("\n=== contract validation ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    all_ok = all(checks.values())

    if all_ok:
        sub.write_csv(CAND)
        print(f"\nwrote {CAND}")
    else:
        print("\nVALIDATION FAILED — not writing candidate")

    (OUT / "_assemble_summary.json").write_text(json.dumps(
        {"pairap_eval_honest": pairap, "evidence_map5_dev": ev_map5, "behavior_map_dev": beh_map,
         "composite": composite, "gate": 0.69, "gate_cleared": bool(composite > 0.69),
         "validation": checks, "all_ok": bool(all_ok), "candidate": str(CAND)}, indent=2),
        encoding="utf-8")

    print(f"\n{'='*60}")
    if composite > 0.69 and all_ok:
        print(f"GATE CLEARED ({composite:.4f} > 0.69) and CSV valid -> SUBMIT")
    else:
        print(f"GATE NOT CLEARED (composite {composite:.4f} vs 0.69) -> DO NOT SUBMIT")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
