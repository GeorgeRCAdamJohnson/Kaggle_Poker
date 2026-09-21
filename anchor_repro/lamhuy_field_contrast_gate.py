"""Drift gate + confirmed-label OOF screen for the partner-vs-field contrast block.

BUILD + MEASURE ONLY. Same discipline as the other lamhuy augmentation
diagnostics: out-of-fold adversarial drift AUC, a per-feature solo-drift audit
(names the exact leaking column), and a confirmed-label table-disjoint OOF
screen. LamhuY remains the risk foundation; no submission is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from anchor_repro.lamhuy_field_contrast import FEATURE_COLUMNS


def _adversarial_auc_oof(dev: pd.DataFrame, evaluation: pd.DataFrame) -> float:
    combined = pd.concat([dev.assign(_domain=0), evaluation.assign(_domain=1)], ignore_index=True)
    X = combined[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = combined["_domain"].to_numpy()
    predictions = np.zeros(len(combined), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=1729)
    for train, valid in splitter.split(X, y):
        model = XGBClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=8.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            random_state=1729,
            n_jobs=-1,
        )
        model.fit(X.iloc[train], y[train])
        predictions[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    return float(roc_auc_score(y, predictions))


def _per_feature_solo_drift_auc(dev: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, float]:
    combined = pd.concat([dev.assign(_domain=0), evaluation.assign(_domain=1)], ignore_index=True)
    y = combined["_domain"].to_numpy()
    result: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        values = combined[[column]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result[column] = float(roc_auc_score(y, values[column]))
    return result


def _confirmed_oof(dev_features: pd.DataFrame, dev_pairs: pd.DataFrame) -> dict:
    labeled = dev_pairs[dev_pairs["is_labeled"].astype(bool)].copy()
    data = labeled.merge(dev_features, on="pair_id", how="inner", validate="one_to_one")
    data["label"] = data["label"].astype(int)
    predictions = np.zeros(len(data), dtype=float)
    splitter = GroupKFold(n_splits=5)
    X = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = data["label"].to_numpy()
    for train, valid in splitter.split(X, y, groups=data["table_id"]):
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
            random_state=1729 + len(train),
            n_jobs=-1,
        )
        model.fit(X.iloc[train], y[train])
        predictions[valid] = model.predict_proba(X.iloc[valid])[:, 1]
    return {
        "n_confirmed": int(len(data)),
        "n_positive": int(y.sum()),
        "confirmed_oof_pair_ap": float(average_precision_score(y, predictions)),
    }


def build_report(root: Path) -> dict:
    prepared = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    report_dir = root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_features = pd.read_parquet(report_dir / "dev_field_contrast.parquet")
    eval_features = pd.read_parquet(report_dir / "eval_field_contrast.parquet")

    drift_auc_oof = _adversarial_auc_oof(dev_features, eval_features)
    solo_drift = _per_feature_solo_drift_auc(dev_features, eval_features)
    oof = _confirmed_oof(dev_features, dev_pairs)
    leaking = {name: auc for name, auc in solo_drift.items() if abs(auc - 0.5) > 0.15}

    return {
        "note": "LamhuY remains the risk foundation; this is the partner-vs-field contrast diagnostic only.",
        "n_features": len(FEATURE_COLUMNS),
        "n_dev_pairs": int(len(dev_features)),
        "n_eval_pairs": int(len(eval_features)),
        "drift_auc_oof": drift_auc_oof,
        "drift_bar_reference": 0.65,
        "drift_gate_passes_reference": bool(drift_auc_oof < 0.65),
        "n_leaking_features_gt_0.65_equiv": len(leaking),
        "leaking_features": dict(sorted(leaking.items(), key=lambda item: -abs(item[1] - 0.5))),
        "confirmed_oof": oof,
        "interpretation": "No leaderboard or lamhuy-composition claim is made by this report.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root)
    (report_dir / "_field_contrast_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
