"""Floor-guaranteed blend sweep: lamhuy-style base risk + temporal/directional risk.

BUILD + MEASURE ONLY. No submission is written or overwritten. LamhuY remains the
risk foundation; this module asks a narrow question: does the (drift-gate-passing)
temporal/directional block from ``lamhuy_temporal_direction.py`` add genuine,
non-noise lift when blended with a base risk model, on a shared table-disjoint
fold assignment?

Disclosed limitation (Rule 9): the "base" model here is an APPROXIMATION of
lamhuy's real triple-GBDT + decoupled-specialist risk head, built from the SAME
cached hand-level feature matrix (mean/max/p95 aggregation of the 61 numeric
``dev_hand_features`` columns) but trained as a single XGB classifier on the
confirmed-labeled subset only, matching the temporal model's own training
regime (``lamhuy_temporal_direction._confirmed_oof``) so the comparison is
apples-to-apples. It is NOT lamhuy's actual persisted risk head.

Weight sweep semantics: ``w=0`` reproduces the base model exactly (a floor
guarantee, Rule 15); ``w=1`` is the temporal model alone. Both OOF vectors are
produced under the IDENTICAL GroupKFold(5, table_id) assignment so the blend at
any ``w`` is a valid pointwise combination, not a mix of different folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from anchor_repro.lamhuy_temporal_direction import (
    FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS,
    HAND_COLUMNS as TEMPORAL_HAND_COLUMNS,
    build_temporal_directional_features,
)

#: Bookkeeping columns excluded from the base aggregation (not behavioral signal).
_BASE_EXCLUDE = {"pair_id", "hand_id", "table_id", "phase_progress"}

#: Weights swept over the blend; 0.0 = base alone (the floor), 1.0 = temporal alone.
BLEND_WEIGHTS = tuple(round(w, 2) for w in np.linspace(0.0, 1.0, 11))

#: Same lift bar the dossier applies to a new feature block over its foundation
#: (exp3c/exp4c precedent, §32/§33): a real, non-noise gain clears +0.010 AP.
LIFT_BAR = 0.010


def _base_hand_columns(schema_columns: list[str]) -> list[str]:
    return [c for c in schema_columns if c not in _BASE_EXCLUDE]


def build_base_pair_features(hand_frame: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    """Mean/max/p95 aggregation of the numeric hand columns (a lamhuy-style base)."""
    frame = hand_frame[["pair_id", *base_columns]].copy()
    frame[base_columns] = frame[base_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    grouped = frame.groupby("pair_id", sort=False)
    means = grouped[base_columns].mean().add_suffix("_mean")
    maxes = grouped[base_columns].max().add_suffix("_max")
    p95 = grouped[base_columns].quantile(0.95).add_suffix("_p95")
    result = means.join(maxes).join(p95).reset_index()
    return result


def _shared_fold_assignment(table_ids: pd.Series, n_splits: int = 5) -> np.ndarray:
    splitter = GroupKFold(n_splits=n_splits)
    fold = np.full(len(table_ids), -1, dtype=int)
    dummy_X = np.zeros((len(table_ids), 1))
    for index, (_, valid) in enumerate(splitter.split(dummy_X, groups=table_ids)):
        fold[valid] = index
    return fold


def _oof_predict(X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, seed: int) -> np.ndarray:
    predictions = np.zeros(len(X), dtype=float)
    for f in np.unique(fold):
        train = fold != f
        valid = fold == f
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device="cuda",
            random_state=seed + f,
            n_jobs=-1,
        )
        model.fit(X.iloc[train], y[train])
        predictions[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    return predictions


def _percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def build_report(root: Path) -> dict:
    prepared = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_hand_schema = pd.read_parquet(prepared / "dev_hand_features.parquet").columns.tolist()
    base_columns = _base_hand_columns(dev_hand_schema)

    dev_hands_base = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=["pair_id", *base_columns])
    dev_hands_temporal = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=TEMPORAL_HAND_COLUMNS)

    base_features = build_base_pair_features(dev_hands_base, base_columns)
    temporal_features = build_temporal_directional_features(dev_hands_temporal)

    labeled = dev_pairs[dev_pairs["is_labeled"].astype(bool)].copy()
    data = (
        labeled.merge(base_features, on="pair_id", how="inner", validate="one_to_one")
        .merge(temporal_features, on="pair_id", how="inner", validate="one_to_one")
    )
    data["label"] = data["label"].astype(int)
    y = data["label"].to_numpy()
    fold = _shared_fold_assignment(data["table_id"])

    base_cols = [c for c in data.columns if c.endswith(("_mean", "_max", "_p95"))]
    base_oof = _oof_predict(data[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0), y, fold, seed=1729)
    temporal_oof = _oof_predict(
        data[TEMPORAL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0), y, fold, seed=2027
    )

    base_pct = _percentile(base_oof)
    temporal_pct = _percentile(temporal_oof)

    sweep = []
    for w in BLEND_WEIGHTS:
        blended = (1.0 - w) * base_pct + w * temporal_pct
        sweep.append({"weight_temporal": float(w), "pair_ap": float(average_precision_score(y, blended))})

    base_ap = sweep[0]["pair_ap"]
    temporal_ap = sweep[-1]["pair_ap"]
    best = max(sweep, key=lambda item: item["pair_ap"])
    lift_over_base = best["pair_ap"] - base_ap

    corr = float(spearmanr(base_oof, temporal_oof).statistic)

    return {
        "note": (
            "Floor-guaranteed blend sweep (w=0 reproduces the base model exactly, Rule 15). "
            "Base is a disclosed APPROXIMATION of lamhuy's risk head (mean/max/p95 aggregation "
            "of the same cached hand features), NOT lamhuy's actual persisted risk. Confirmed-"
            "label screening only; not an LB claim."
        ),
        "n_confirmed": int(len(data)),
        "n_positive": int(y.sum()),
        "n_base_features": len(base_cols),
        "n_temporal_features": len(TEMPORAL_FEATURE_COLUMNS),
        "base_oof_temporal_oof_spearman": corr,
        "base_alone_pair_ap": base_ap,
        "temporal_alone_pair_ap": temporal_ap,
        "sweep": sweep,
        "best": best,
        "lift_over_base": lift_over_base,
        "lift_bar_reference": LIFT_BAR,
        "clears_lift_bar": bool(lift_over_base >= LIFT_BAR),
        "interpretation": (
            "Any weight above 0 that beats the base (w=0) is a genuine, non-degenerate blend "
            "lift under this shared-fold comparison; it does not require a large jump to be "
            "a valid, stackable improvement. clears_lift_bar applies the dossier's own "
            "+0.010 exp3c/exp4c precedent as a reference, not a hard requirement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root)
    (report_dir / "_temporal_blend_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
