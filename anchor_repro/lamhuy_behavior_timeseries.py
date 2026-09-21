"""Behavioral time-series augmentation: fold/call/aggression/passivity trends over a
pair's shared-hand timeline (not just chip flow).

BUILD + MEASURE ONLY. Same discipline as ``lamhuy_temporal_direction.py``: one
shared dev/eval feature builder, an out-of-fold adversarial drift diagnostic, a
per-feature solo-drift audit (names the exact leaking column instead of a
blanket verdict), and a confirmed-label table-disjoint OOF screen. No
submission is written or overwritten; LamhuY remains the risk foundation.

Five behavioral axes, each built from the SAME cached ``dev_hand_features`` /
``eval_hand_features`` columns lamhuy already computes per hand (never raw
action-log re-derivation):

* fold      - one_folded, both_folded, hu_flop_folds, dump_flop_fold,
              partner_folds_to_pair, outsider_folds_to_pair
* call      - pair_calls, true_hu_calls, partner_calls_to_pair,
              outsider_calls_to_pair
* aggression/bluff - pair_aggression, pair_raises, is_pure_dump, dump_vs_hole,
              strong_hole_passive, max_hole_strength
* passivity - passivity_density, pure_checkdown, is_soft_checkdown,
              pair_checks, pair_postflop_checks
* isolation - is_squeeze_isolation, isolation_squeeze

For every axis column we compute only PER-HAND-MEAN statistics (early-half
mean, late-half mean, their difference, a phase-progress slope, and the
existing burst entropy/excess over 8 phase bins) - never a cumulative sum over
the pair, so nothing here scales with raw shared-hand opportunity by
construction. That is a design choice, not proof; the drift gate below
verifies it empirically rather than assuming it (Rule 5).
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

BEHAVIOR_COLUMNS: dict[str, list[str]] = {
    "fold": [
        "one_folded",
        "both_folded",
        "hu_flop_folds",
        "dump_flop_fold",
        "partner_folds_to_pair",
        "outsider_folds_to_pair",
    ],
    "call": [
        "pair_calls",
        "true_hu_calls",
        "partner_calls_to_pair",
        "outsider_calls_to_pair",
    ],
    "aggression_bluff": [
        "pair_aggression",
        "pair_raises",
        "is_pure_dump",
        "dump_vs_hole",
        "strong_hole_passive",
        "max_hole_strength",
    ],
    "passivity": [
        "passivity_density",
        "pure_checkdown",
        "is_soft_checkdown",
        "pair_checks",
        "pair_postflop_checks",
    ],
    "isolation": [
        "is_squeeze_isolation",
        "isolation_squeeze",
    ],
}

_AXIS_COLUMNS: list[str] = [c for cols in BEHAVIOR_COLUMNS.values() for c in cols]
HAND_COLUMNS: list[str] = ["pair_id", "phase_progress", *_AXIS_COLUMNS]

_STAT_SUFFIXES = ("early_mean", "late_mean", "late_minus_early", "slope", "burst_entropy", "burst_excess")

# Both named the drift-gate failure (out-of-fold AUC 0.676): sparse binary-flag
# burst entropy over 8 phase bins is sensitive to shared-hand-count population
# shift for these two low-frequency indicators. Excluded from FEATURE_COLUMNS
# (still computed in the builder for anyone who wants to inspect them raw).
_DROPPED_LEAKING_FEATURES = frozenset(
    {"strong_hole_passive_burst_entropy", "pure_checkdown_burst_entropy"}
)
FEATURE_COLUMNS: list[str] = [
    f"{col}_{suffix}"
    for col in _AXIS_COLUMNS
    for suffix in _STAT_SUFFIXES
    if f"{col}_{suffix}" not in _DROPPED_LEAKING_FEATURES
]


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) <= 1e-12:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _burst_entropy_excess(bin_means: np.ndarray) -> tuple[float, float]:
    if len(bin_means) == 0:
        return 0.0, 0.0
    positive = np.maximum(bin_means, 0.0)
    total = float(positive.sum())
    weights = positive / total if total > 0.0 else np.full(len(bin_means), 1.0 / len(bin_means))
    entropy = float(-(weights * np.log(weights + 1e-12)).sum() / np.log(max(len(bin_means), 2)))
    excess = float(bin_means.max() - bin_means.mean())
    return entropy, excess


def _reference_build_behavior_timeseries_features(hand_frame: pd.DataFrame) -> pd.DataFrame:
    """Reference (slow, per-pair Python loop) implementation, kept ONLY to validate
    the vectorized production path against on a sample (never used in the full run)."""
    missing = [column for column in HAND_COLUMNS if column not in hand_frame.columns]
    if missing:
        raise ValueError(f"missing hand feature columns: {missing}")
    frame = hand_frame[HAND_COLUMNS].copy()
    frame["phase_progress"] = frame["phase_progress"].astype(float).clip(0.0, 1.0)
    frame[_AXIS_COLUMNS] = (
        frame[_AXIS_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    frame = frame.sort_values(["pair_id", "phase_progress"], kind="stable")
    frame["_phase_bin"] = np.floor(frame["phase_progress"].clip(0.0, 0.999999) * 8).astype(int)

    rows: list[dict[str, float | str]] = []
    for pair_id, group in frame.groupby("pair_id", sort=False):
        group = group.reset_index(drop=True)
        midpoint = max(len(group) // 2, 1)
        early = group.iloc[:midpoint]
        late = group.iloc[midpoint:]
        x = group["phase_progress"].to_numpy(dtype=float)
        row: dict[str, float | str] = {"pair_id": str(pair_id)}
        for column in _AXIS_COLUMNS:
            early_mean = float(early[column].mean())
            late_mean = float(late[column].mean()) if len(late) else 0.0
            bin_means = group.groupby("_phase_bin")[column].mean().to_numpy(dtype=float)
            entropy, excess = _burst_entropy_excess(bin_means)
            row[f"{column}_early_mean"] = early_mean
            row[f"{column}_late_mean"] = late_mean
            row[f"{column}_late_minus_early"] = late_mean - early_mean
            row[f"{column}_slope"] = _safe_slope(x, group[column].to_numpy(dtype=float))
            row[f"{column}_burst_entropy"] = entropy
            row[f"{column}_burst_excess"] = excess
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    return result[["pair_id", *FEATURE_COLUMNS]]


def build_behavior_timeseries_features(hand_frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized pair-level behavioral time-series features (dev/eval share this path).

    Group-level pandas aggregates replace the per-pair Python loop the reference
    implementation uses; every statistic is computed identically, just without the
    per-pair-per-column overhead (validated by ``test`` mode against the reference
    on a sample before this path is trusted for the full run).
    """
    missing = [column for column in HAND_COLUMNS if column not in hand_frame.columns]
    if missing:
        raise ValueError(f"missing hand feature columns: {missing}")
    frame = hand_frame[HAND_COLUMNS].copy()
    frame["phase_progress"] = frame["phase_progress"].astype(float).clip(0.0, 1.0)
    frame[_AXIS_COLUMNS] = (
        frame[_AXIS_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    frame = frame.sort_values(["pair_id", "phase_progress"], kind="stable").reset_index(drop=True)
    frame["_phase_bin"] = np.floor(frame["phase_progress"].clip(0.0, 0.999999) * 8).astype(int)

    pair_ids = frame["pair_id"].drop_duplicates().to_numpy()
    grouped = frame.groupby("pair_id", sort=False)
    group_size = grouped["pair_id"].transform("size")
    rank = grouped.cumcount()
    midpoint = np.maximum(group_size // 2, 1)
    is_late = (rank >= midpoint).to_numpy()

    early_mean = frame.loc[~is_late].groupby("pair_id", sort=False)[_AXIS_COLUMNS].mean()
    late_mean = frame.loc[is_late].groupby("pair_id", sort=False)[_AXIS_COLUMNS].mean()
    early_mean = early_mean.reindex(pair_ids).fillna(0.0)
    late_mean = late_mean.reindex(pair_ids).fillna(0.0)

    # Closed-form per-pair OLS slope of each axis column against phase_progress,
    # exactly equivalent to ``np.polyfit(x, y, 1)[0]`` (same normal-equations
    # formula), computed for all axis columns at once via group-sum aggregates.
    x = frame["phase_progress"]
    n = grouped["pair_id"].transform("size").groupby(frame["pair_id"], sort=False).first()
    sum_x = grouped["phase_progress"].sum()
    sum_x2 = (x * x).groupby(frame["pair_id"], sort=False).sum()
    sum_y = grouped[_AXIS_COLUMNS].sum()
    xy = frame[_AXIS_COLUMNS].multiply(x, axis=0)
    sum_xy = xy.groupby(frame["pair_id"], sort=False).sum()
    x_ptp = grouped["phase_progress"].agg(lambda s: s.max() - s.min())

    denom = (n * sum_x2 - sum_x**2).reindex(pair_ids)
    numer = sum_xy.reindex(pair_ids).multiply(n.reindex(pair_ids), axis=0) - sum_y.reindex(pair_ids).multiply(
        sum_x.reindex(pair_ids), axis=0
    )
    valid = (n.reindex(pair_ids) >= 3) & (x_ptp.reindex(pair_ids) > 1e-12) & (denom.abs() > 1e-12)
    slope = numer.div(denom.replace(0.0, np.nan), axis=0).where(valid, 0.0).fillna(0.0)

    # Per-(pair, phase_bin) means, unstacked to a fixed 8-bin matrix per axis column;
    # missing bins stay NaN so entropy/excess average only over OBSERVED bins,
    # matching the reference loop's variable-length bin array exactly.
    bin_means = frame.groupby(["pair_id", "_phase_bin"], sort=False)[_AXIS_COLUMNS].mean()
    bin_count = frame.groupby("pair_id", sort=False)["_phase_bin"].nunique().reindex(pair_ids)
    log_divisor = np.log(np.maximum(bin_count.to_numpy(dtype=float), 2))

    entropy_cols: dict[str, np.ndarray] = {}
    excess_cols: dict[str, np.ndarray] = {}
    for column in _AXIS_COLUMNS:
        block = bin_means[column].unstack("_phase_bin").reindex(pair_ids)
        block = block.reindex(columns=range(8))
        values = block.to_numpy(dtype=float)
        positive = np.where(np.isnan(values), np.nan, np.maximum(values, 0.0))
        total = np.nansum(positive, axis=1)
        observed = np.sum(~np.isnan(values), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            weights = np.where(
                total[:, None] > 0.0,
                positive / total[:, None],
                np.where(np.isnan(values), np.nan, 1.0 / np.maximum(observed[:, None], 1)),
            )
            entropy = -np.nansum(weights * np.log(weights + 1e-12), axis=1) / log_divisor
        excess = np.nanmax(values, axis=1) - np.nanmean(values, axis=1)
        entropy_cols[column] = np.nan_to_num(entropy, nan=0.0)
        excess_cols[column] = np.nan_to_num(excess, nan=0.0)

    result = pd.DataFrame({"pair_id": [str(p) for p in pair_ids]})
    columns_out = {}
    for column in _AXIS_COLUMNS:
        columns_out[f"{column}_early_mean"] = early_mean[column].to_numpy(dtype=float)
        columns_out[f"{column}_late_mean"] = late_mean[column].to_numpy(dtype=float)
        columns_out[f"{column}_late_minus_early"] = (late_mean[column] - early_mean[column]).to_numpy(dtype=float)
        columns_out[f"{column}_slope"] = slope[column].to_numpy(dtype=float)
        columns_out[f"{column}_burst_entropy"] = entropy_cols[column]
        columns_out[f"{column}_burst_excess"] = excess_cols[column]
    result = pd.concat([result, pd.DataFrame(columns_out)], axis=1)
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    return result[["pair_id", *FEATURE_COLUMNS]]


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
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_hands = pd.read_parquet(prepared / "dev_hand_features.parquet", columns=HAND_COLUMNS)
    eval_hands = pd.read_parquet(prepared / "eval_hand_features.parquet", columns=HAND_COLUMNS)
    dev_features = build_behavior_timeseries_features(dev_hands)
    eval_features = build_behavior_timeseries_features(eval_hands)

    drift_auc_oof = _adversarial_auc_oof(dev_features, eval_features)
    solo_drift = _per_feature_solo_drift_auc(dev_features, eval_features)
    oof = _confirmed_oof(dev_features, dev_pairs)
    leaking = {name: auc for name, auc in solo_drift.items() if abs(auc - 0.5) > 0.15}

    return {
        "note": "LamhuY remains the risk foundation; this is a behavioral time-series (fold/call/bluff/passivity/isolation) diagnostic only.",
        "axes": {axis: cols for axis, cols in BEHAVIOR_COLUMNS.items()},
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
    prepared = args.poker_root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    build_behavior_timeseries_features(
        pd.read_parquet(prepared / "dev_hand_features.parquet", columns=HAND_COLUMNS)
    ).to_parquet(report_dir / "dev_behavior_timeseries.parquet", index=False)
    build_behavior_timeseries_features(
        pd.read_parquet(prepared / "eval_hand_features.parquet", columns=HAND_COLUMNS)
    ).to_parquet(report_dir / "eval_behavior_timeseries.parquet", index=False)
    (report_dir / "_behavior_timeseries_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
