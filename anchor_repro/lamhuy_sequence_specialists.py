"""Family-decoupled ordered-sequence risk specialists on the full V30 frame.

The original V30 notebook trains one-vs-rest LightGBM heads for directed transfer,
soft play, and coordinated isolation, but its final executable risk path uses only the
monolithic triple-GBDT score. This module tests the omitted specialist-union route with
the ordered-action block from ``lamhuy_ordered_sequence`` under the exact persisted,
table-group-disjoint folds. It never submits automatically and emits a validated
risk-only CSV only when material, fold-robust promotion gates pass.

Run from the Poker workspace root::

    python -m anchor_repro.lamhuy_sequence_specialists
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

from anchor_repro.lamhuy_hard_negative import (
    EXPECTED_DEV_ROWS,
    EXPECTED_EVAL_ROWS,
    N_SPLITS,
    PROMOTION_BASELINE_AP,
    PROMOTION_CONFIRMED_DELTA,
    PROMOTION_STRESS_DELTA_FLOOR,
    SEED,
    _candidate_source,
    _emit_candidate,
    _fold_metrics,
    _json_ready,
    _load_full_frames,
    _prepare_training_data,
    _rank_comparison,
)
from anchor_repro.lamhuy_ordered_sequence import (
    _append_sequence_block,
    _load_baseline_oof,
    _validate_persisted_folds,
)
from anchor_repro.lamhuy_recipe import default_lamhuy_paths

RECIPE_ID = "lamhuy_v30_sequence_family_specialists_v1"
FAMILY_NAMES = ("directed_transfer", "soft_play", "coordinated_isolation")
FAMILY_IDS = (1, 2, 3)
FAMILY_POSITIVE_WEIGHTS = {1: 3.0, 2: 3.0, 3: 10.0}
MIN_POSITIVE_CONFIRMED_FOLDS = 3

# Fixed before observing specialist OOF results. These are deliberately few and coarse.
ARM_DEFINITIONS = {
    "specialist_sum": {"kind": "specialist_sum"},
    "specialist_max": {"kind": "specialist_max"},
    "raw_half_sum": {"kind": "raw_blend_sum", "weight": 0.50},
    "rank_sum_w25": {"kind": "rank_blend_sum", "weight": 0.25},
    "rank_sum_w50": {"kind": "rank_blend_sum", "weight": 0.50},
    "rank_sum_w75": {"kind": "rank_blend_sum", "weight": 0.75},
    "rank_max_w25": {"kind": "rank_blend_max", "weight": 0.25},
    "rank_max_w50": {"kind": "rank_blend_max", "weight": 0.50},
}


def _write_run_files(out_dir: Path, summary: dict, log_lines: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_sequence_specialists_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "_sequence_specialists.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def _load_augmented_frames(
    hnm_dir: Path,
    sequence_dir: Path,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
    base_dev, groups, base_eval = _load_full_frames(hnm_dir)
    dev_path = sequence_dir / "dev_sequence_features.parquet"
    eval_path = sequence_dir / "eval_sequence_features.parquet"
    if not dev_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(
            "Run python -m anchor_repro.lamhuy_ordered_sequence first; sequence caches are missing"
        )
    sequence_dev = pl.read_parquet(dev_path)
    sequence_eval = pl.read_parquet(eval_path)
    if sequence_dev.height != EXPECTED_DEV_ROWS or sequence_eval.height != EXPECTED_EVAL_ROWS:
        raise ValueError(
            f"Unexpected sequence cache shapes: {sequence_dev.shape}, {sequence_eval.shape}"
        )
    dev, columns = _append_sequence_block(base_dev, sequence_dev)
    evaluation, eval_columns = _append_sequence_block(base_eval, sequence_eval)
    if columns != eval_columns:
        raise ValueError("Development/evaluation sequence feature order differs")
    return dev, groups, evaluation, columns


def _specialist_params() -> dict:
    """Verbatim family-head structure from V30; only family labels/weights differ."""
    return {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 35,
        "min_child_samples": 12,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 6.0,
        "objective": "binary",
        "n_jobs": -1,
        "verbose": -1,
    }


def _run_specialists(
    data: Dict[str, object],
    fold_id: np.ndarray,
    log: Callable[[str], None],
    model_seed: int = SEED,
) -> dict:
    from lightgbm import LGBMClassifier

    X: pd.DataFrame = data["X"]
    X_eval: pd.DataFrame = data["X_eval"]
    behavior_y = data["behavior_y"]
    known = data["known"]
    groups = data["groups"]
    stress_weight = data["stress_weight"]

    oof = np.zeros((len(X), 3), dtype=np.float32)
    eval_parts: Dict[int, List[np.ndarray]] = {family_id: [] for family_id in FAMILY_IDS}
    fold_family_metrics: List[dict] = []

    for fold in range(N_SPLITS):
        tr = np.flatnonzero(fold_id != fold)
        va = np.flatnonzero(fold_id == fold)
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"Fold {fold}: table leakage detected")

        fold_row = {"fold": fold, "families": {}}
        for family_id, family_name in zip(FAMILY_IDS, FAMILY_NAMES):
            target = (behavior_y == family_id).astype(np.int8)
            fit_weight = np.where(
                behavior_y == family_id,
                FAMILY_POSITIVE_WEIGHTS[family_id],
                np.where(known, 1.0, 0.35),
            ).astype(np.float64)
            model = LGBMClassifier(**_specialist_params(), random_state=model_seed + fold)
            model.fit(
                X.iloc[tr],
                target[tr],
                sample_weight=fit_weight[tr],
                eval_set=[(X.iloc[va], target[va])],
                eval_sample_weight=[stress_weight[va]],
            )
            oof[va, family_id - 1] = model.predict_proba(X.iloc[va])[:, 1]
            eval_parts[family_id].append(model.predict_proba(X_eval)[:, 1])

            known_va = va[known[va]]
            family_confirmed_ap = float(
                average_precision_score(
                    target[known_va], oof[known_va, family_id - 1]
                )
            )
            family_stress_ap = float(
                average_precision_score(
                    target[va],
                    oof[va, family_id - 1],
                    sample_weight=stress_weight[va],
                )
            )
            fold_row["families"][family_name] = {
                "confirmed_ap": family_confirmed_ap,
                "pu_stress_ap": family_stress_ap,
            }
            del model
            gc.collect()
        fold_family_metrics.append(fold_row)
        log(
            f"Fold {fold + 1}: "
            + ", ".join(
                f"{name} AP={fold_row['families'][name]['confirmed_ap']:.4f}"
                for name in FAMILY_NAMES
            )
        )

    evaluation = np.column_stack(
        [
            np.mean(eval_parts[family_id], axis=0).astype(np.float32)
            for family_id in FAMILY_IDS
        ]
    )
    family_metrics = {}
    for family_id, family_name in zip(FAMILY_IDS, FAMILY_NAMES):
        target = (behavior_y == family_id).astype(np.int8)
        family_metrics[family_name] = {
            "confirmed_ap": float(
                average_precision_score(target[known], oof[known, family_id - 1])
            ),
            "pu_stress_ap": float(
                average_precision_score(
                    target,
                    oof[:, family_id - 1],
                    sample_weight=stress_weight,
                )
            ),
        }
    positive = known & (behavior_y > 0)
    family_accuracy = float(
        np.mean((np.argmax(oof[positive], axis=1) + 1) == behavior_y[positive])
    )
    return {
        "oof": oof,
        "eval": evaluation,
        "family_metrics": family_metrics,
        "family_accuracy": family_accuracy,
        "fold_family_metrics": fold_family_metrics,
    }


def _build_arms(
    base_risk: np.ndarray,
    family_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    base = np.asarray(base_risk, dtype=np.float32)
    scores = np.asarray(family_scores, dtype=np.float32)
    union_sum = np.clip(scores.sum(axis=1), 0.0, 1.0).astype(np.float32)
    union_max = scores.max(axis=1).astype(np.float32)
    base_rank = _percentile_rank(base)
    sum_rank = _percentile_rank(union_sum)
    max_rank = _percentile_rank(union_max)

    arms: Dict[str, np.ndarray] = {}
    for name, definition in ARM_DEFINITIONS.items():
        kind = definition["kind"]
        weight = float(definition.get("weight", 0.0))
        if kind == "specialist_sum":
            risk = union_sum
        elif kind == "specialist_max":
            risk = union_max
        elif kind == "raw_blend_sum":
            risk = (1.0 - weight) * base + weight * union_sum
        elif kind == "rank_blend_sum":
            risk = (1.0 - weight) * base_rank + weight * sum_rank
        elif kind == "rank_blend_max":
            risk = (1.0 - weight) * base_rank + weight * max_rank
        else:
            raise ValueError(f"Unknown arm kind: {kind}")
        arms[name] = np.clip(risk, 0.0, 1.0).astype(np.float32)
    return arms


def _score_arms(
    arms: Dict[str, np.ndarray],
    baseline: np.ndarray,
    y: np.ndarray,
    known: np.ndarray,
    stress_weight: np.ndarray,
    fold_id: np.ndarray,
) -> Tuple[dict, dict]:
    baseline_confirmed = float(average_precision_score(y[known], baseline[known]))
    baseline_stress = float(
        average_precision_score(y, baseline, sample_weight=stress_weight)
    )
    baseline_folds = [
        _fold_metrics(
            np.flatnonzero(fold_id == fold),
            y,
            known,
            stress_weight,
            baseline,
        )
        for fold in range(N_SPLITS)
    ]
    baseline_metrics = {
        "confirmed_ap": baseline_confirmed,
        "pu_stress_ap": baseline_stress,
        "folds": [
            {"fold": fold, **metrics} for fold, metrics in enumerate(baseline_folds)
        ],
    }

    metrics = {}
    for name, score in arms.items():
        confirmed = float(average_precision_score(y[known], score[known]))
        stress = float(average_precision_score(y, score, sample_weight=stress_weight))
        folds = []
        for fold in range(N_SPLITS):
            current = _fold_metrics(
                np.flatnonzero(fold_id == fold),
                y,
                known,
                stress_weight,
                score,
            )
            current["confirmed_ap_delta_vs_baseline"] = (
                current["confirmed_ap"] - baseline_folds[fold]["confirmed_ap"]
            )
            current["pu_stress_ap_delta_vs_baseline"] = (
                current["pu_stress_ap"] - baseline_folds[fold]["pu_stress_ap"]
            )
            folds.append({"fold": fold, **current})
        positive_confirmed_folds = sum(
            row["confirmed_ap_delta_vs_baseline"] > 0 for row in folds
        )
        positive_stress_folds = sum(
            row["pu_stress_ap_delta_vs_baseline"] > 0 for row in folds
        )
        metrics[name] = {
            "definition": ARM_DEFINITIONS[name],
            "confirmed_ap": confirmed,
            "confirmed_ap_delta_vs_baseline": confirmed - baseline_confirmed,
            "pu_stress_ap": stress,
            "pu_stress_ap_delta_vs_baseline": stress - baseline_stress,
            "positive_confirmed_folds": positive_confirmed_folds,
            "positive_pu_stress_folds": positive_stress_folds,
            "folds": folds,
        }
    return baseline_metrics, metrics


def _gate_arm(metrics: dict, baseline_metrics: dict) -> dict:
    checks = {
        "baseline_confirmed_ap": baseline_metrics["confirmed_ap"]
        >= PROMOTION_BASELINE_AP,
        "candidate_confirmed_delta": metrics["confirmed_ap_delta_vs_baseline"]
        >= PROMOTION_CONFIRMED_DELTA,
        "candidate_pu_stress_delta": metrics["pu_stress_ap_delta_vs_baseline"]
        >= PROMOTION_STRESS_DELTA_FLOOR,
        "positive_confirmed_folds": metrics["positive_confirmed_folds"]
        >= MIN_POSITIVE_CONFIRMED_FOLDS,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _load_original_eval_risk(
    source_path: Path,
    eval_pair_ids: np.ndarray,
) -> np.ndarray:
    source = pd.read_csv(source_path, usecols=["pair_id", "risk_score"], dtype={"pair_id": str})
    if not source["pair_id"].is_unique:
        raise ValueError("V30 source pair IDs are not unique")
    indexed = source.set_index("pair_id")["risk_score"]
    try:
        values = indexed.loc[eval_pair_ids.astype(str)].to_numpy(dtype=np.float32)
    except KeyError as exc:
        raise ValueError("V30 source does not exactly cover eval pair IDs") from exc
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("V30 source risk is invalid")
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = default_lamhuy_paths(root)
    hnm_dir = root / "outputs" / "poker_collusion" / "lamhuy_hnm"
    sequence_dir = root / "outputs" / "poker_collusion" / "lamhuy_ordered_sequence"
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_sequence_specialists"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    summary: Dict[str, object] = {
        "recipe_id": RECIPE_ID,
        "status": "RUNNING",
        "output_dir": str(out_dir),
        "source_frame": str(hnm_dir),
        "sequence_features": str(sequence_dir),
        "seed": SEED,
        "cv": "exact persisted HNM outer folds, table-group-disjoint",
        "families": list(FAMILY_NAMES),
        "family_positive_weights": FAMILY_POSITIVE_WEIGHTS,
        "arms": ARM_DEFINITIONS,
        "promotion_gate": {
            "baseline_confirmed_ap_min": PROMOTION_BASELINE_AP,
            "candidate_confirmed_ap_delta_min": PROMOTION_CONFIRMED_DELTA,
            "candidate_pu_stress_ap_delta_min": PROMOTION_STRESS_DELTA_FLOOR,
            "positive_confirmed_folds_min": MIN_POSITIVE_CONFIRMED_FOLDS,
        },
        "transport_note": (
            "OOF rank compositions use the persisted local V30 baseline; eval rank "
            "compositions use the recovered original V30 risk because it is the "
            "validated higher-LB production source. Raw specialist-only arms have no "
            "baseline transport dependency."
        ),
    }
    started = time.time()
    for stale in out_dir.glob("candidate_v30_*sequence_specialists_*.csv"):
        stale.unlink()

    try:
        dev, pair_groups, evaluation, sequence_columns = _load_augmented_frames(
            hnm_dir, sequence_dir
        )
        summary["frame_shapes"] = {
            "dev": dev.shape,
            "pair_groups": pair_groups.shape,
            "eval": evaluation.shape,
        }
        summary["sequence_feature_count"] = len(sequence_columns)
        data = _prepare_training_data(paths, dev, pair_groups, evaluation, out_dir)
        baseline_rows = _load_baseline_oof(hnm_dir, data["pair_ids"])
        fold_id = baseline_rows["outer_fold"].to_numpy(dtype=np.int8)
        summary["fold_manifest"] = _validate_persisted_folds(
            data["pair_ids"],
            data["groups"],
            fold_id,
            hnm_dir / "fold_manifest.json",
        )
        baseline_oof = baseline_rows["baseline_risk"].to_numpy(dtype=np.float32)
        log("Training 3 family specialists across 5 persisted table-disjoint folds...")
        specialist = _run_specialists(data, fold_id, log)
        summary["family_metrics"] = specialist["family_metrics"]
        summary["family_accuracy"] = specialist["family_accuracy"]
        summary["fold_family_metrics"] = specialist["fold_family_metrics"]

        oof_arms = _build_arms(baseline_oof, specialist["oof"])
        baseline_metrics, arm_metrics = _score_arms(
            oof_arms,
            baseline_oof,
            data["y"],
            data["known"],
            data["stress_weight"],
            fold_id,
        )
        summary["baseline_metrics"] = baseline_metrics
        summary["arm_metrics"] = arm_metrics
        gates = {
            name: _gate_arm(metrics, baseline_metrics)
            for name, metrics in arm_metrics.items()
        }
        summary["gate_results"] = gates
        passing = [name for name, gate in gates.items() if gate["passed"]]
        selected = (
            max(passing, key=lambda name: arm_metrics[name]["confirmed_ap"])
            if passing
            else None
        )
        summary["passing_arms"] = passing
        summary["selected_arm"] = selected

        for name in ARM_DEFINITIONS:
            metrics = arm_metrics[name]
            log(
                f"{name}: confirmed {metrics['confirmed_ap']:.7f} "
                f"({metrics['confirmed_ap_delta_vs_baseline']:+.7f}), "
                f"stress {metrics['pu_stress_ap']:.7f} "
                f"({metrics['pu_stress_ap_delta_vs_baseline']:+.7f}), "
                f"positive confirmed folds={metrics['positive_confirmed_folds']}/5, "
                f"gate={gates[name]['passed']}."
            )

        oof_path = out_dir / "oof_rows.parquet"
        oof_frame = pd.DataFrame(
            {
                "pair_id": data["pair_ids"],
                "cv_group": data["groups"].astype(str),
                "outer_fold": fold_id,
                "known": data["known"],
                "behavior_y": data["behavior_y"],
                "y": data["y"],
                "baseline_risk": baseline_oof,
                "specialist_directed": specialist["oof"][:, 0],
                "specialist_soft": specialist["oof"][:, 1],
                "specialist_isolation": specialist["oof"][:, 2],
            }
        )
        for name, score in oof_arms.items():
            oof_frame[name] = score
        oof_frame.to_parquet(oof_path, index=False)

        eval_path = out_dir / "eval_specialist_scores.parquet"
        eval_frame = pd.DataFrame(
            {
                "pair_id": data["eval_pair_ids"],
                "specialist_directed": specialist["eval"][:, 0],
                "specialist_soft": specialist["eval"][:, 1],
                "specialist_isolation": specialist["eval"][:, 2],
            }
        )
        eval_frame.to_parquet(eval_path, index=False)
        summary["artifacts"] = {
            "oof_rows": str(oof_path),
            "eval_specialist_scores": str(eval_path),
            "feature_columns": str(out_dir / "feature_columns.json"),
        }

        source_path, source_kind = _candidate_source(root)
        eval_base = _load_original_eval_risk(source_path, data["eval_pair_ids"])
        eval_arms = _build_arms(eval_base, specialist["eval"])
        summary["rank_comparisons"] = {
            name: _rank_comparison(
                data["eval_pair_ids"], score, source_path
            )
            for name, score in eval_arms.items()
        }

        if selected is not None:
            emitted = _emit_candidate(
                f"sequence_specialists_{selected}",
                eval_arms[selected],
                data["eval_pair_ids"],
                source_path,
                source_kind,
                paths.data_dir / "evaluation_pairs.csv",
                out_dir,
            )
            summary["candidate"] = emitted
            summary["status"] = "COMPLETED_PROMOTED"
            log(f"PROMOTED {selected}: {emitted['path']}")
        else:
            summary["candidate"] = None
            summary["status"] = "COMPLETED_NULL"
            log("No family-specialist arm cleared all material promotion gates; no CSV emitted.")

        summary["duration_s"] = time.time() - started
        _write_run_files(out_dir, summary, log_lines)
        return 0
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        summary["duration_s"] = time.time() - started
        log(f"FAILED: {type(exc).__name__}: {exc}")
        _write_run_files(out_dir, summary, log_lines)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
