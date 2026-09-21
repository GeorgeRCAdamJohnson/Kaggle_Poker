"""E-LINE BEHAVIOR column — family routing from the E-line's OWN specialists.

Per metric contract §1.3, BehaviorMAP is OvR AP over the 3 disclosed families where the OvR
score = risk_score for pairs predicted as that family (else 0). So the behavior column is a
LABELING: route each pair to the family its strongest specialist fires on, above a threshold;
below threshold -> "none". No new model — reuses the persisted E-line specialists (eline/*).

We pick a per-family threshold that maximizes dev Behavior-MAP (the specialists' argmax among
directed/soft/isolation, gated so we don't over-label negatives). Emits eval behavior labels.

Run:  python -m anchor_repro.eline_behavior
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from anchor_repro.eval_honest_harness import CACHE

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
OUT = CACHE / "eline"
FAMILIES = ["directed_transfer", "soft_play", "coordinated_isolation"]
FAM_COL = {"directed_transfer": "spec_directed", "soft_play": "spec_soft",
           "coordinated_isolation": "spec_isolation"}


def _behavior_map(risk, behavior_y, pred_family):
    """Mean OvR AP over the 3 families: behavior_risk = risk where pred==family else 0."""
    aps = []
    for fid, fam in enumerate(FAMILIES, start=1):
        truth = (behavior_y == fid).astype(int)
        if truth.sum() == 0:
            aps.append(0.0); continue
        brisk = np.where(pred_family == fam, risk, 0.0)
        aps.append(average_precision_score(truth, brisk))
    return float(np.mean(aps)), aps


def main() -> int:
    oof = pl.read_parquet(SEQ / "oof_rows.parquet").select(["pair_id", "known", "behavior_y", "y"])
    rd = pl.read_parquet(OUT / "risk_dev.parquet")
    dev = oof.join(rd, on="pair_id", how="inner").filter(pl.col("known"))
    behavior_y = dev["behavior_y"].to_numpy().astype(int)
    risk = dev["risk_best"].to_numpy()
    spec = {f: dev[FAM_COL[f]].to_numpy() for f in FAMILIES}
    S = np.column_stack([spec[f] for f in FAMILIES])
    argmax_fam = np.array(FAMILIES)[S.argmax(1)]
    argmax_val = S.max(1)

    # sweep a global threshold on the winning specialist; below -> none
    print("=== E-line behavior routing (dev Behavior-MAP) ===")
    best = (-1, None)
    for thr in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        pred = np.where(argmax_val >= thr, argmax_fam, "none")
        bm, aps = _behavior_map(risk, behavior_y, pred)
        n_lab = int((pred != "none").sum())
        print(f"  thr={thr:.2f}: Behavior-MAP={bm:.4f}  (directed={aps[0]:.3f} soft={aps[1]:.3f} iso={aps[2]:.3f}, n_labeled={n_lab})")
        if bm > best[0]:
            best = (bm, thr)
    bm, thr = best
    print(f"BEST: thr={thr} Behavior-MAP={bm:.4f}")

    # emit eval behavior labels at the chosen threshold
    re = pl.read_parquet(OUT / "risk_eval.parquet")
    Se = np.column_stack([re[FAM_COL[f]].to_numpy() for f in FAMILIES])
    e_argmax = np.array(FAMILIES)[Se.argmax(1)]
    e_val = Se.max(1)
    e_pred = np.where(e_val >= thr, e_argmax, "none")
    out = pl.DataFrame({"pair_id": re["pair_id"].to_list(), "predicted_behavior": e_pred})
    out.write_parquet(OUT / "behavior_eval.parquet")
    dist = out["predicted_behavior"].value_counts().sort("count", descending=True)
    print("eval behavior distribution:")
    print(dist)
    (OUT / "_behavior_summary.json").write_text(json.dumps(
        {"threshold": thr, "dev_behavior_map": bm}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}/behavior_eval.parquet, _behavior_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
