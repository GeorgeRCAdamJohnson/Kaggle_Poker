"""Temporal/directional augmentation experiment on top of the lamhuy feature cache.

This module is BUILD + MEASURE ONLY. It does not overwrite a submission. Dev and
eval features use the same aggregation function and the experiment reports:

* adversarial dev-vs-eval AUC (distribution diagnostic);
* table-disjoint OOF AP on confirmed development labels;
* feature-to-feature diagnostics for forward/reverse flow and temporal evolution;
* correlation with the persisted lamhuy evaluation risk ranking.

The module intentionally does not claim that a confirmed-label OOF score is an LB
prediction. A later composition runner can add these features to the lamhuy risk
foundation once this diagnostic is understood.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBClassifier


HAND_COLUMNS = [
    "pair_id",
    "phase_progress",
    "transfer_1_to_2_bb",
    "transfer_2_to_1_bb",
    "net_gap_bb",
    "transfer_any_bb",
    "directed_signal",
    "soft_signal",
    "isolation_signal",
]

# Raw cumulative sums (forward_flow, reverse_flow, flow_total, n_hands) are
# EXCLUDED: they scale directly with shared-hand opportunity, which is known
# to drift dev-vs-eval (dossier §51/§52). Only opportunity-normalized rates
# and bounded ratios enter the model.
FEATURE_COLUMNS = [
    "forward_rate",
    "reverse_rate",
    "flow_rate",
    "directional_imbalance",
    "forward_share",
    "early_net_flow",
    "late_net_flow",
    "late_minus_early_net",
    "early_transfer",
    "late_transfer",
    "late_minus_early_transfer",
    "net_flow_slope",
    "direction_persistence",
    "direction_switch_rate",
    "burst_centroid_net",
    "burst_entropy_net",
    "burst_max_net",
    "burst_excess_net",
    "burst_max_transfer",
    "burst_excess_transfer",
    "burst_max_directed",
    "burst_excess_directed",
    "burst_max_isolation",
    "burst_excess_isolation",
]


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) <= 1e-12:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _burst_stats(frame: pd.DataFrame, column: str) -> dict[str, float]:
    bins = frame.groupby("_phase_bin", sort=True)[column].mean()
    values = bins.to_numpy(dtype=float)
    if len(values) == 0:
        return {
            f"burst_centroid_{column}": 0.0,
            f"burst_entropy_{column}": 0.0,
            f"burst_max_{column}": 0.0,
            f"burst_excess_{column}": 0.0,
        }
    positive = np.maximum(values, 0.0)
    total = float(positive.sum())
    weights = positive / total if total > 0.0 else np.full(len(values), 1.0 / len(values))
    positions = bins.index.to_numpy(dtype=float)
    entropy = float(-(weights * np.log(weights + 1e-12)).sum() / np.log(max(len(values), 2)))
    return {
        f"burst_centroid_{column}": float((positions * weights).sum() / 7.0),
        f"burst_entropy_{column}": entropy,
        f"burst_max_{column}": float(values.max()),
        f"burst_excess_{column}": float(values.max() - values.mean()),
    }


def build_temporal_directional_features(hand_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the same pair-level temporal/directional features for dev and eval."""
    missing = [column for column in HAND_COLUMNS if column not in hand_frame.columns]
    if missing:
        raise ValueError(f"missing hand feature columns: {missing}")
    frame = hand_frame[HAND_COLUMNS].copy()
    frame["phase_progress"] = frame["phase_progress"].astype(float).clip(0.0, 1.0)
    numeric = [column for column in HAND_COLUMNS if column != "pair_id"]
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame = frame.sort_values(["pair_id", "phase_progress"], kind="stable")
    frame["_phase_bin"] = np.floor(frame["phase_progress"].clip(0.0, 0.999999) * 8).astype(int)

    rows: list[dict[str, float | str]] = []
    for pair_id, group in frame.groupby("pair_id", sort=False):
        group = group.reset_index(drop=True)
        forward = float(group["transfer_1_to_2_bb"].sum())
        reverse = float(group["transfer_2_to_1_bb"].sum())
        total = forward + reverse
        # ``net_gap_bb`` is a non-negative magnitude in the lamhuy cache, not a
        # signed direction. Reconstruct the antisymmetric flow from the two
        # directional transfer columns instead.
        group = group.assign(
            _signed_flow_bb=group["transfer_1_to_2_bb"] - group["transfer_2_to_1_bb"]
        )
        midpoint = max(len(group) // 2, 1)
        early = group.iloc[:midpoint]
        late = group.iloc[midpoint:]
        signed = group["_signed_flow_bb"].to_numpy(dtype=float)
        nonzero = np.sign(signed[np.abs(signed) > 1e-9])
        switches = float(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0.0
        n_hands = float(len(group))
        row: dict[str, float | str] = {
            "pair_id": str(pair_id),
            "forward_rate": forward / n_hands,
            "reverse_rate": reverse / n_hands,
            "flow_rate": total / n_hands,
            "directional_imbalance": (forward - reverse) / (total + 1e-3),
            "forward_share": forward / (total + 1e-3),
            "early_net_flow": float(early["_signed_flow_bb"].mean()),
            "late_net_flow": float(late["_signed_flow_bb"].mean()) if len(late) else 0.0,
            "early_transfer": float(early["transfer_any_bb"].mean()),
            "late_transfer": float(late["transfer_any_bb"].mean()) if len(late) else 0.0,
            "net_flow_slope": _safe_slope(group["phase_progress"].to_numpy(float), signed),
            "direction_persistence": float(abs(nonzero.mean())) if len(nonzero) else 0.0,
            "direction_switch_rate": switches / max(len(nonzero) - 1, 1),
        }
        row["late_minus_early_net"] = float(row["late_net_flow"] - row["early_net_flow"])
        row["late_minus_early_transfer"] = float(row["late_transfer"] - row["early_transfer"])
        for column, output_name in (
            ("_signed_flow_bb", "net"),
            ("transfer_any_bb", "transfer"),
            ("directed_signal", "directed"),
            ("isolation_signal", "isolation"),
        ):
            stats = _burst_stats(group, column)
            for key, value in stats.items():
                row[key.replace(column, output_name)] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    return result[["pair_id", *FEATURE_COLUMNS]]


def _per_feature_solo_drift_auc(dev: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, float]:
    """Single-feature dev-vs-eval AUC per column (names the leaking feature, Rule 11)."""
    combined = pd.concat(
        [dev.assign(_domain=0), evaluation.assign(_domain=1)], ignore_index=True
    )
    y = combined["_domain"].to_numpy()
    result: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        values = combined[[column]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result[column] = float(roc_auc_score(y, values[column]))
    return result


def _adversarial_auc(dev: pd.DataFrame, evaluation: pd.DataFrame) -> float:
    combined = pd.concat(
        [dev.assign(_domain=0), evaluation.assign(_domain=1)], ignore_index=True
    )
    X = combined[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = combined["_domain"].to_numpy()
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
    model.fit(X, y)
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def _adversarial_auc_oof(dev: pd.DataFrame, evaluation: pd.DataFrame) -> float:
    """Measure dev/eval separability out of fold, avoiding in-sample optimism."""
    combined = pd.concat(
        [dev.assign(_domain=0), evaluation.assign(_domain=1)], ignore_index=True
    )
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
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_hands = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=HAND_COLUMNS)
    eval_hands = pd.read_parquet(prepared / "eval_hand_features.parquet", columns=HAND_COLUMNS)
    dev_features = build_temporal_directional_features(dev_hands)
    eval_features = build_temporal_directional_features(eval_hands)
    drift_auc = _adversarial_auc(dev_features, eval_features)
    drift_auc_oof = _adversarial_auc_oof(dev_features, eval_features)
    solo_drift = _per_feature_solo_drift_auc(dev_features, eval_features)
    oof = _confirmed_oof(dev_features, dev_pairs)

    lamhuy_path = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "submission.csv"
    lamhuy = pd.read_csv(lamhuy_path, usecols=["pair_id", "risk_score"])
    comparison = eval_features.merge(lamhuy, on="pair_id", how="inner", validate="one_to_one")
    comparison_spearman = float(spearmanr(comparison[FEATURE_COLUMNS].mean(axis=1), comparison["risk_score"]).statistic)
    return {
        "note": "LamhuY remains the risk foundation; this is a temporal/directional specialist diagnostic only.",
        "feature_columns": FEATURE_COLUMNS,
        "n_dev_pairs": int(len(dev_features)),
        "n_eval_pairs": int(len(eval_features)),
        "drift_auc_in_sample_diagnostic": drift_auc,
        "drift_auc_oof": drift_auc_oof,
        "drift_bar_reference": 0.65,
        "drift_gate_passes_reference": bool(drift_auc_oof < 0.65),
        "solo_feature_drift_auc": dict(sorted(solo_drift.items(), key=lambda item: -abs(item[1] - 0.5))),
        "confirmed_oof": oof,
        "eval_mean_feature_vs_lamhuy_risk_spearman": comparison_spearman,
        "interpretation": "No leaderboard or lamhuy-composition claim is made by this report.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root)
    prepared = args.poker_root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    build_temporal_directional_features(
        pd.read_parquet(prepared / "dev_hand_features.parquet", columns=HAND_COLUMNS)
    ).to_parquet(report_dir / "dev_temporal_directional.parquet", index=False)
    build_temporal_directional_features(
        pd.read_parquet(prepared / "eval_hand_features.parquet", columns=HAND_COLUMNS)
    ).to_parquet(report_dir / "eval_temporal_directional.parquet", index=False)
    (report_dir / "_temporal_direction_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
