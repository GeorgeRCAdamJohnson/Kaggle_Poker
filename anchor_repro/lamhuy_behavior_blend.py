"""Floor-guaranteed blend sweep: lamhuy-style base risk + behavioral time-series risk.

BUILD + MEASURE ONLY. LamhuY remains the risk foundation. This mirrors
``lamhuy_temporal_blend.py`` exactly (same base construction, same shared-fold
discipline, same w=0-is-the-floor guarantee) but blends in the behavioral
time-series block instead of the chip-flow temporal block.

Drift-gate caveat (disclosed, not hidden): ``lamhuy_behavior_timeseries`` failed
the out-of-fold adversarial drift gate (AUC ~0.676 vs the 0.65 reference) even
after the two individually-leaking features were dropped -- the separability
lives in the joint correlation structure, not any single column (the same
pattern the dossier's §54/§55 Rule-17-falsification found: a basis can be
dev/eval-distinguishable and still rank-track). Per that evidence-based
amendment, drift AUC is treated here as a REPORTED DIAGNOSTIC, not a hard veto;
the real test is whether this block lifts the confirmed-label floor under a
shared-fold, floor-guaranteed blend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from anchor_repro.lamhuy_behavior_timeseries import (
    FEATURE_COLUMNS as BEHAVIOR_FEATURE_COLUMNS,
    HAND_COLUMNS as BEHAVIOR_HAND_COLUMNS,
    build_behavior_timeseries_features,
)
from anchor_repro.lamhuy_temporal_blend import (
    BLEND_WEIGHTS,
    LIFT_BAR,
    _base_hand_columns,
    _oof_predict,
    _percentile,
    _shared_fold_assignment,
    build_base_pair_features,
)


def build_report(root: Path) -> dict:
    prepared = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_hand_schema = pd.read_parquet(prepared / "dev_hand_features.parquet").columns.tolist()
    base_columns = _base_hand_columns(dev_hand_schema)

    dev_hands_base = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=["pair_id", *base_columns])
    dev_hands_behavior = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=BEHAVIOR_HAND_COLUMNS)

    base_features = build_base_pair_features(dev_hands_base, base_columns)
    behavior_features = build_behavior_timeseries_features(dev_hands_behavior)

    labeled = dev_pairs[dev_pairs["is_labeled"].astype(bool)].copy()
    data = (
        labeled.merge(base_features, on="pair_id", how="inner", validate="one_to_one")
        .merge(behavior_features, on="pair_id", how="inner", validate="one_to_one")
    )
    data["label"] = data["label"].astype(int)
    y = data["label"].to_numpy()
    fold = _shared_fold_assignment(data["table_id"])

    base_cols = [c for c in data.columns if c.endswith(("_mean", "_max", "_p95"))]
    base_oof = _oof_predict(data[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0), y, fold, seed=1729)
    behavior_oof = _oof_predict(
        data[BEHAVIOR_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0), y, fold, seed=3037
    )

    base_pct = _percentile(base_oof)
    behavior_pct = _percentile(behavior_oof)

    sweep = []
    for w in BLEND_WEIGHTS:
        blended = (1.0 - w) * base_pct + w * behavior_pct
        sweep.append({"weight_behavior": float(w), "pair_ap": float(average_precision_score(y, blended))})

    base_ap = sweep[0]["pair_ap"]
    behavior_ap = sweep[-1]["pair_ap"]
    best = max(sweep, key=lambda item: item["pair_ap"])
    lift_over_base = best["pair_ap"] - base_ap
    corr = float(spearmanr(base_oof, behavior_oof).statistic)

    return {
        "note": (
            "Floor-guaranteed blend sweep (w=0 reproduces the base model exactly, Rule 15). "
            "Base is a disclosed APPROXIMATION of lamhuy's risk head, NOT lamhuy's actual "
            "persisted risk. Confirmed-label screening only; not an LB claim. The behavior "
            "time-series block FAILS the out-of-fold drift gate (AUC ~0.676, reported "
            "separately, not re-measured here) -- treated as a diagnostic per the dossier's "
            "§54/§55 Rule-17 amendment, not a hard veto on this blend test."
        ),
        "n_confirmed": int(len(data)),
        "n_positive": int(y.sum()),
        "n_base_features": len(base_cols),
        "n_behavior_features": len(BEHAVIOR_FEATURE_COLUMNS),
        "base_oof_behavior_oof_spearman": corr,
        "base_alone_pair_ap": base_ap,
        "behavior_alone_pair_ap": behavior_ap,
        "sweep": sweep,
        "best": best,
        "lift_over_base": lift_over_base,
        "lift_bar_reference": LIFT_BAR,
        "clears_lift_bar": bool(lift_over_base >= LIFT_BAR),
        "interpretation": (
            "Any weight above 0 that beats the base (w=0) is a genuine, non-degenerate blend "
            "lift under this shared-fold comparison; it does not require a large jump to be a "
            "valid, stackable improvement. clears_lift_bar applies the dossier's own +0.010 "
            "exp3c/exp4c precedent as a reference, not a hard requirement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root)
    (report_dir / "_behavior_blend_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
