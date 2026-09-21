"""Isolate the behavior head: shipped 0.70904 with ONLY predicted_behavior changed (evidence kept).

The combined heads submission scored 0.67666 (DOWN from 0.70904). Feature parity is clean, so the
drop is real signal loss, most likely from the evidence swap (our dev-planted ranker transfers worse
than honghanh's borrowed evidence head). This variant KEEPS the shipped (honghanh) evidence columns
byte-identical and changes ONLY predicted_behavior (argmax V30 specialists, route top-RATE by risk).
Isolates whether the behavior routing alone helps or hurts vs shipped.

Run: python -m anchor_repro.assemble_behavior_only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHIPPED = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
EVAL_SPEC = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/eval_specialist_scores.parquet")
OUT = Path("outputs/poker_collusion/heads_behavior_only.csv")
FAMILY = {0: "directed_transfer", 1: "soft_play", 2: "coordinated_isolation"}
# dev BehaviorMAP is flat ~0.96 above rate 0.20 (dense dev can't guide the eval rate). On sparse eval,
# over-routing risks misassigning a high-risk true positive's family. 0.05 concentrates on the
# highest-risk pairs (~5600) where the specialist argmax is most reliable, well above shipped's 1.1%.
RATE = 0.05


def main() -> int:
    sub = pd.read_csv(SHIPPED)
    sub["pair_id"] = sub["pair_id"].astype(str)
    spec = pd.read_parquet(EVAL_SPEC)
    spec["pair_id"] = spec["pair_id"].astype(str)
    m = sub.merge(spec, on="pair_id", how="left")
    S = m[["specialist_directed", "specialist_soft", "specialist_isolation"]].fillna(0).to_numpy()
    fam = [FAMILY[i] for i in S.argmax(axis=1)]
    risk = m["risk_score"].astype(float).to_numpy()
    thr = np.quantile(risk, 1 - RATE)
    behavior = np.where(risk >= thr, fam, "none")

    out = sub.copy()
    out["predicted_behavior"] = behavior  # ONLY this changes; evidence kept from shipped
    assert np.allclose(out["risk_score"].astype(float), sub["risk_score"].astype(float))
    assert (out["evidence_hand_1"] == sub["evidence_hand_1"]).all(), "evidence changed!"
    print(f"behavior changed on {(out['predicted_behavior']!=sub['predicted_behavior']).sum()} rows; "
          f"evidence IDENTICAL; risk IDENTICAL")
    print(f"behavior dist: {pd.Series(behavior).value_counts().to_dict()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
