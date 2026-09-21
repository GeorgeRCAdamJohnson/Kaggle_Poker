"""Behavior-head audit for the leaderboard-validated sequence-specialist risk.

Uses already out-of-fold family specialist scores; no model is retrained. A positive
rate is selected on the discovery folds and checked unchanged on the independent
seed-137 confirmation folds. If conservative gates pass, writes a NO-SUBMIT candidate
that changes only ``predicted_behavior`` on the 0.70904 risk artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from anchor_repro.lamhuy_recipe import validate_submission_csv

BEHAVIORS = np.array(
    ["directed_transfer", "soft_play", "coordinated_isolation"], dtype=object
)
RATE_GRID = np.array([0.0040, 0.0055, 0.0070, 0.0080, 0.0090, 0.0100, 0.0110, 0.0120, 0.0150, 0.0200])
# Measured from the faithful local V30 notebook run; used only as a reference floor.
V30_LABELED_BEHAVIOR_MAP = 0.6969
V30_STRESS_BEHAVIOR_MAP = 0.5636
MIN_DISCOVERY_STRESS_LIFT = 0.010
CONFIRMATION_STRESS_FLOOR_DELTA = -0.005
MIN_FAMILY_ACCURACY = 0.97


def _top_mask(scores: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, int(round(len(scores) * rate)))
    mask = np.zeros(len(scores), dtype=bool)
    order = np.lexsort((np.arange(len(scores)), -scores))
    mask[order[:count]] = True
    return mask


def _behavior_metrics(frame: pd.DataFrame, rate: float) -> Dict[str, object]:
    risk = frame["rank_sum_w50"].to_numpy(dtype=np.float64)
    behavior_y = frame["behavior_y"].to_numpy(dtype=np.int8)
    known = frame["known"].to_numpy(dtype=bool)
    y = frame["y"].to_numpy(dtype=np.int8)
    family_scores = frame[
        ["specialist_directed", "specialist_soft", "specialist_isolation"]
    ].to_numpy(dtype=np.float64)
    family = family_scores.argmax(axis=1) + 1
    active = _top_mask(risk, rate)
    stress_weight = np.where(known, 1.0, 112_540 / 24_000)

    def score(mask: np.ndarray, weights: np.ndarray | None) -> Tuple[float, list]:
        values = []
        for family_id in (1, 2, 3):
            family_score = np.where(
                active[mask] & (family[mask] == family_id), risk[mask], 0.0
            )
            values.append(
                float(
                    average_precision_score(
                        behavior_y[mask] == family_id,
                        family_score,
                        sample_weight=None if weights is None else weights[mask],
                    )
                )
            )
        return float(np.mean(values)), values

    labeled_map, labeled_classes = score(known, None)
    all_rows = np.ones(len(frame), dtype=bool)
    stress_map, stress_classes = score(all_rows, stress_weight)
    positive = known & (y == 1)
    return {
        "rate": float(rate),
        "active_rows": int(active.sum()),
        "labeled_behavior_map": labeled_map,
        "labeled_class_ap": labeled_classes,
        "pu_stress_behavior_map": stress_map,
        "pu_stress_class_ap": stress_classes,
        "positive_family_accuracy": float(np.mean(family[positive] == behavior_y[positive])),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    discovery_path = (
        root / "outputs" / "poker_collusion" / "lamhuy_sequence_specialists" / "oof_rows.parquet"
    )
    confirmation_path = (
        root / "outputs" / "poker_collusion" / "lamhuy_sequence_confirmation" / "oof_rows.parquet"
    )
    eval_specialists_path = (
        root
        / "outputs"
        / "poker_collusion"
        / "lamhuy_sequence_confirmation"
        / "eval_specialist_scores.parquet"
    )
    submitted_path = (
        root
        / "outputs"
        / "poker_collusion"
        / "lamhuy_sequence_confirmation"
        / "candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv"
    )
    eval_pairs_path = root / "data" / "poker" / "evaluation_pairs.csv"
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_sequence_behavior"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "candidate_070904_sequence_behavior_no_submit.csv"
    candidate_path.unlink(missing_ok=True)

    discovery = pd.read_parquet(discovery_path)
    confirmation = pd.read_parquet(confirmation_path)
    discovery_grid = [_behavior_metrics(discovery, rate) for rate in RATE_GRID]
    best_stress = max(row["pu_stress_behavior_map"] for row in discovery_grid)
    eligible = [
        row for row in discovery_grid
        if row["pu_stress_behavior_map"] >= best_stress - 0.005
    ]
    selected_discovery = min(eligible, key=lambda row: row["rate"])
    selected_rate = float(selected_discovery["rate"])
    confirmation_result = _behavior_metrics(confirmation, selected_rate)

    gates = {
        "discovery_stress_lift": selected_discovery["pu_stress_behavior_map"]
        - V30_STRESS_BEHAVIOR_MAP
        >= MIN_DISCOVERY_STRESS_LIFT,
        "discovery_labeled_nonregression": selected_discovery["labeled_behavior_map"]
        >= V30_LABELED_BEHAVIOR_MAP - 0.005,
        "confirmation_stress_floor": confirmation_result["pu_stress_behavior_map"]
        >= V30_STRESS_BEHAVIOR_MAP + CONFIRMATION_STRESS_FLOOR_DELTA,
        "confirmation_family_accuracy": confirmation_result["positive_family_accuracy"]
        >= MIN_FAMILY_ACCURACY,
    }

    source = pd.read_csv(submitted_path, dtype={"pair_id": str})
    eval_specialists = pd.read_parquet(eval_specialists_path)
    merged = source[["pair_id", "risk_score"]].merge(
        eval_specialists, on="pair_id", how="left", validate="one_to_one"
    )
    if merged.isna().any().any() or len(merged) != len(source):
        raise ValueError("Evaluation specialist join failed")
    family = merged[
        ["specialist_directed", "specialist_soft", "specialist_isolation"]
    ].to_numpy().argmax(axis=1)
    active = _top_mask(merged["risk_score"].to_numpy(), selected_rate)
    predicted = np.full(len(source), "none", dtype=object)
    predicted[active] = BEHAVIORS[family[active]]

    report: Dict[str, object] = {
        "note": "BUILD + MEASURE ONLY; no Kaggle submission was made.",
        "leaderboard_anchor": 0.70904,
        "v30_reference": {
            "labeled_behavior_map": V30_LABELED_BEHAVIOR_MAP,
            "pu_stress_behavior_map": V30_STRESS_BEHAVIOR_MAP,
        },
        "rate_grid": discovery_grid,
        "selected_rate": selected_rate,
        "discovery_selected": selected_discovery,
        "confirmation": confirmation_result,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "eval_old_distribution": source["predicted_behavior"].value_counts().to_dict(),
        "eval_new_distribution": pd.Series(predicted).value_counts().to_dict(),
        "source_path": str(submitted_path),
        "source_sha256": _sha256(submitted_path),
    }

    if all(gates.values()):
        candidate = source.copy(deep=True)
        candidate["predicted_behavior"] = predicted
        unchanged = [column for column in source.columns if column != "predicted_behavior"]
        if not candidate[unchanged].equals(source[unchanged]):
            raise AssertionError("Behavior candidate changed a non-behavior column")
        candidate.to_csv(candidate_path, index=False)
        validation = validate_submission_csv(candidate_path, eval_pairs_path)
        if not validation["all_passed"]:
            candidate_path.unlink(missing_ok=True)
            raise RuntimeError(f"Behavior candidate validation failed: {validation}")
        report["candidate"] = {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "validation": validation,
            "non_behavior_columns_exact": True,
        }
    else:
        report["candidate"] = None

    (out_dir / "_sequence_behavior_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
