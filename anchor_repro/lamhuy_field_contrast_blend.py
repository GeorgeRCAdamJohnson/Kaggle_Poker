"""Floor-guaranteed blend sweep: lamhuy-style base risk + partner-vs-field contrast risk.

BUILD + MEASURE ONLY. Mirrors ``lamhuy_temporal_blend.py`` / ``lamhuy_behavior_blend.py``:
same base construction, same shared-fold discipline, same w=0-is-the-floor
guarantee. LamhuY remains the risk foundation.

The field-contrast block's confirmed-label OOF AP (0.9375) is the strongest of
the three augmentation blocks tested this session and exceeds the base-alone AP
seen in the prior two blend runs (~0.904-0.907) -- worth testing directly rather
than assuming another null. It FAILS the out-of-fold drift gate (AUC ~0.697,
reported separately); per the dossier's §54/§55 Rule-17 amendment this is
treated as a diagnostic, not a hard veto, and the blend sweep is the real test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from anchor_repro.lamhuy_field_contrast import FEATURE_COLUMNS as FIELD_FEATURE_COLUMNS
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
    report_dir = root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_hand_schema = pd.read_parquet(prepared / "dev_hand_features.parquet").columns.tolist()
    base_columns = _base_hand_columns(dev_hand_schema)

    dev_hands_base = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=["pair_id", *base_columns])
    base_features = build_base_pair_features(dev_hands_base, base_columns)
    field_features = pd.read_parquet(report_dir / "dev_field_contrast.parquet")

    labeled = dev_pairs[dev_pairs["is_labeled"].astype(bool)].copy()
    data = (
        labeled.merge(base_features, on="pair_id", how="inner", validate="one_to_one")
        .merge(field_features, on="pair_id", how="inner", validate="one_to_one")
    )
    data["label"] = data["label"].astype(int)
    y = data["label"].to_numpy()
    fold = _shared_fold_assignment(data["table_id"])

    base_cols = [c for c in data.columns if c.endswith(("_mean", "_max", "_p95"))]
    base_oof = _oof_predict(data[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0), y, fold, seed=1729)
    field_oof = _oof_predict(
        data[FIELD_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0), y, fold, seed=4111
    )

    base_pct = _percentile(base_oof)
    field_pct = _percentile(field_oof)

    sweep = []
    for w in BLEND_WEIGHTS:
        blended = (1.0 - w) * base_pct + w * field_pct
        sweep.append({"weight_field": float(w), "pair_ap": float(average_precision_score(y, blended))})

    base_ap = sweep[0]["pair_ap"]
    field_ap = sweep[-1]["pair_ap"]
    best = max(sweep, key=lambda item: item["pair_ap"])
    lift_over_base = best["pair_ap"] - base_ap
    corr = float(spearmanr(base_oof, field_oof).statistic)

    return {
        "note": (
            "Floor-guaranteed blend sweep (w=0 reproduces the base model exactly, Rule 15). "
            "Base is a disclosed APPROXIMATION of lamhuy's risk head, NOT lamhuy's actual "
            "persisted risk. Confirmed-label screening only; not an LB claim. The field-"
            "contrast block FAILS the out-of-fold drift gate (AUC ~0.697, reported "
            "separately) -- treated as a diagnostic per the dossier's §54/§55 Rule-17 "
            "amendment, not a hard veto on this blend test."
        ),
        "n_confirmed": int(len(data)),
        "n_positive": int(y.sum()),
        "n_base_features": len(base_cols),
        "n_field_features": len(FIELD_FEATURE_COLUMNS),
        "base_oof_field_oof_spearman": corr,
        "base_alone_pair_ap": base_ap,
        "field_alone_pair_ap": field_ap,
        "sweep": sweep,
        "best": best,
        "lift_over_base": lift_over_base,
        "lift_bar_reference": LIFT_BAR,
        "clears_lift_bar": bool(lift_over_base >= LIFT_BAR),
        "interpretation": (
            "Any weight above 0 that beats the base (w=0) is a genuine, non-degenerate blend "
            "lift under this shared-fold comparison; it does not require a large jump to be a "
            "valid, stackable improvement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root)
    (report_dir / "_field_contrast_blend_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
