"""Independent confirmation of one frozen sequence-specialist V30 composition.

This is not an arm search. It cleanly rebuilds sequence features from the raw ordered
action artifact, creates a new table-group split with seed 137, freshly trains the V30
triple-GBDT baseline and the three family specialists, and evaluates exactly one frozen
transport-safe rule::

    0.50 * percentile_rank(V30 risk)
  + 0.50 * percentile_rank(clipped sum of family specialist scores)

A risk-only candidate using recovered-original V30 risk is emitted only if this new
split independently clears the material AP and fold-robustness gates.

Run from the Poker workspace root::

    python -m anchor_repro.lamhuy_sequence_confirmation
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.graph_retest import _probe_cuda
from anchor_repro.lamhuy_hard_negative import (
    BLEND_WEIGHTS,
    CB_PARAMS,
    LGB_PARAMS,
    N_SPLITS,
    PROMOTION_BASELINE_AP,
    PROMOTION_CONFIRMED_DELTA,
    PROMOTION_STRESS_DELTA_FLOOR,
    _candidate_source,
    _emit_candidate,
    _json_ready,
    _load_full_frames,
    _prepare_training_data,
    _rank_comparison,
    _sha256,
    _xgb_params,
)
from anchor_repro.lamhuy_ordered_sequence import (
    _append_sequence_block,
    _build_sequence_features,
)
from anchor_repro.lamhuy_recipe import default_lamhuy_paths
from anchor_repro.lamhuy_sequence_specialists import (
    MIN_POSITIVE_CONFIRMED_FOLDS,
    _build_arms,
    _gate_arm,
    _load_original_eval_risk,
    _run_specialists,
    _score_arms,
)

CONFIRMATION_SEED = 137
FROZEN_ARM = "rank_sum_w50"
RECIPE_ID = "lamhuy_v30_sequence_specialists_rank_sum_w50_confirmation_seed137"


def _write_run_files(out_dir: Path, summary: dict, log_lines: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_sequence_confirmation_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "_sequence_confirmation.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )


def _make_confirmation_folds(data: Dict[str, object]) -> tuple[np.ndarray, List[dict]]:
    pair_ids = data["pair_ids"]
    groups = data["groups"]
    behavior_y = data["behavior_y"]
    fold_id = np.full(len(pair_ids), -1, dtype=np.int8)
    manifest: List[dict] = []
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CONFIRMATION_SEED
    )
    for fold, (tr, va) in enumerate(cv.split(data["X"], behavior_y, groups)):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"Confirmation fold {fold}: table leakage detected")
        fold_id[va] = fold
        manifest.append(
            {
                "fold": fold,
                "train_rows": len(tr),
                "validation_rows": len(va),
                "train_groups": len(set(groups[tr])),
                "validation_groups": len(set(groups[va])),
                "validation_pair_id_sha256": hashlib.sha256(
                    "\n".join(pair_ids[va]).encode("utf-8")
                ).hexdigest(),
            }
        )
    if np.any(fold_id < 0):
        raise RuntimeError("Confirmation folds did not assign every row")
    return fold_id, manifest


def _run_baseline_triple(
    data: Dict[str, object],
    fold_id: np.ndarray,
    use_cuda: bool,
    log: Callable[[str], None],
) -> dict:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X: pd.DataFrame = data["X"]
    X_eval: pd.DataFrame = data["X_eval"]
    y = data["y"]
    groups = data["groups"]
    fit_weight = data["fit_weight"]
    stress_weight = data["stress_weight"]
    components = {
        name: np.zeros(len(X), dtype=np.float32)
        for name in ("xgb", "lgbm", "catboost")
    }
    eval_parts = {name: [] for name in components}

    for fold in range(N_SPLITS):
        tr = np.flatnonzero(fold_id != fold)
        va = np.flatnonzero(fold_id == fold)
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"Baseline fold {fold}: table leakage detected")

        xgb = XGBClassifier(
            **_xgb_params(use_cuda), random_state=CONFIRMATION_SEED + fold
        )
        xgb.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])],
            sample_weight_eval_set=[stress_weight[va]],
            verbose=False,
        )
        components["xgb"][va] = xgb.predict_proba(X.iloc[va])[:, 1]
        eval_parts["xgb"].append(xgb.predict_proba(X_eval)[:, 1])
        del xgb
        gc.collect()

        lgbm = LGBMClassifier(
            **LGB_PARAMS, random_state=CONFIRMATION_SEED + fold
        )
        lgbm.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_sample_weight=[stress_weight[va]],
        )
        components["lgbm"][va] = lgbm.predict_proba(X.iloc[va])[:, 1]
        eval_parts["lgbm"].append(lgbm.predict_proba(X_eval)[:, 1])
        del lgbm
        gc.collect()

        catboost = CatBoostClassifier(
            **CB_PARAMS, random_seed=CONFIRMATION_SEED + fold
        )
        catboost.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])],
            early_stopping_rounds=80,
            verbose=0,
        )
        components["catboost"][va] = catboost.predict_proba(X.iloc[va])[:, 1]
        eval_parts["catboost"].append(catboost.predict_proba(X_eval)[:, 1])
        del catboost
        gc.collect()
        log(f"Confirmation baseline fold {fold + 1}/5 complete.")

    oof = (
        BLEND_WEIGHTS["xgb"] * components["xgb"]
        + BLEND_WEIGHTS["lgbm"] * components["lgbm"]
        + BLEND_WEIGHTS["catboost"] * components["catboost"]
    ).astype(np.float32)
    eval_components = {
        name: np.mean(parts, axis=0).astype(np.float32)
        for name, parts in eval_parts.items()
    }
    evaluation = (
        BLEND_WEIGHTS["xgb"] * eval_components["xgb"]
        + BLEND_WEIGHTS["lgbm"] * eval_components["lgbm"]
        + BLEND_WEIGHTS["catboost"] * eval_components["catboost"]
    ).astype(np.float32)
    return {
        "oof": oof,
        "eval": evaluation,
        "oof_components": components,
        "eval_components": eval_components,
    }


def _provenance_manifest(
    root: Path,
    paths,
    hnm_dir: Path,
    out_dir: Path,
    dev_sequence: pl.DataFrame,
    eval_sequence: pl.DataFrame,
) -> dict:
    files = {
        "ordered_sequence_source": root / "anchor_repro" / "lamhuy_ordered_sequence.py",
        "specialist_source": root / "anchor_repro" / "lamhuy_sequence_specialists.py",
        "confirmation_source": Path(__file__).resolve(),
        "action_context": paths.prepared_dir / "action_context.parquet",
        "hands": paths.data_dir / "hands.parquet",
        "dev_pairs": paths.prepared_dir / "dev_pairs.parquet",
        "eval_pairs": paths.prepared_dir / "eval_pairs.parquet",
        "base_dev": hnm_dir / "dev_features.parquet",
        "base_eval": hnm_dir / "eval_features.parquet",
        "sequence_dev": out_dir / "dev_sequence_features.parquet",
        "sequence_eval": out_dir / "eval_sequence_features.parquet",
        "ordered_state": out_dir / "ordered_action_state.parquet",
    }
    missing = [path for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Provenance inputs missing: " + ", ".join(map(str, missing)))
    return {
        "files": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in files.items()
        },
        "dev_pair_id_sha256": hashlib.sha256(
            "\n".join(dev_sequence["pair_id"].cast(pl.Utf8).to_list()).encode("utf-8")
        ).hexdigest(),
        "eval_pair_id_sha256": hashlib.sha256(
            "\n".join(eval_sequence["pair_id"].cast(pl.Utf8).to_list()).encode("utf-8")
        ).hexdigest(),
        "dev_sequence_shape": dev_sequence.shape,
        "eval_sequence_shape": eval_sequence.shape,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = default_lamhuy_paths(root)
    hnm_dir = root / "outputs" / "poker_collusion" / "lamhuy_hnm"
    discovery_dir = root / "outputs" / "poker_collusion" / "lamhuy_sequence_specialists"
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_sequence_confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest_dir = out_dir / "baseline_manifest"
    augmented_manifest_dir = out_dir / "augmented_manifest"
    baseline_manifest_dir.mkdir(exist_ok=True)
    augmented_manifest_dir.mkdir(exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    summary: Dict[str, object] = {
        "recipe_id": RECIPE_ID,
        "status": "RUNNING",
        "output_dir": str(out_dir),
        "confirmation_seed": CONFIRMATION_SEED,
        "frozen_arm": FROZEN_ARM,
        "arm_selection": "none; exactly one formula fixed before this run",
        "frozen_formula": (
            "0.50 * percentile_rank(V30 risk) + 0.50 * "
            "percentile_rank(clip(sum(three family scores), 0, 1))"
        ),
        "promotion_gate": {
            "baseline_confirmed_ap_min": PROMOTION_BASELINE_AP,
            "candidate_confirmed_ap_delta_min": PROMOTION_CONFIRMED_DELTA,
            "candidate_pu_stress_ap_delta_min": PROMOTION_STRESS_DELTA_FLOOR,
            "positive_confirmed_folds_min": MIN_POSITIVE_CONFIRMED_FOLDS,
        },
        "transport": (
            "The formula uses percentile ranks only. OOF uses a freshly trained V30 "
            "triple baseline; eval uses recovered-original V30 ranks, avoiding raw-score "
            "scale substitution. Non-risk columns remain recovered-original V30."
        ),
    }
    started = time.time()
    candidate_pattern = "candidate_v30_hnm_sequence_confirmation_*.csv"
    for stale in out_dir.glob(candidate_pattern):
        stale.unlink()

    try:
        discovery_summary_path = discovery_dir / "_sequence_specialists_summary.json"
        discovery_summary = json.loads(
            discovery_summary_path.read_text(encoding="utf-8")
        )
        discovery_arm = discovery_summary["arm_metrics"][FROZEN_ARM]
        discovery_gate = discovery_summary["gate_results"][FROZEN_ARM]
        summary["discovery_result"] = {
            "metrics": discovery_arm,
            "gate": discovery_gate,
        }
        if not discovery_gate["passed"]:
            raise RuntimeError(f"Frozen discovery arm {FROZEN_ARM} did not pass discovery")

        log("Clean-building sequence features into the independent confirmation directory...")
        dev_sequence, eval_sequence, feature_build = _build_sequence_features(
            paths, out_dir, log
        )
        summary["feature_build"] = feature_build
        base_dev, pair_groups, base_eval = _load_full_frames(hnm_dir)
        augmented_dev, sequence_columns = _append_sequence_block(
            base_dev, dev_sequence
        )
        augmented_eval, eval_sequence_columns = _append_sequence_block(
            base_eval, eval_sequence
        )
        if sequence_columns != eval_sequence_columns:
            raise ValueError("Confirmation sequence columns differ by phase")

        baseline_data = _prepare_training_data(
            paths, base_dev, pair_groups, base_eval, baseline_manifest_dir
        )
        augmented_data = _prepare_training_data(
            paths,
            augmented_dev,
            pair_groups,
            augmented_eval,
            augmented_manifest_dir,
        )
        for key in ("pair_ids", "eval_pair_ids", "behavior_y", "y", "groups"):
            if not np.array_equal(baseline_data[key], augmented_data[key]):
                raise ValueError(f"Baseline/augmented confirmation data differ on {key}")
        fold_id, fold_manifest = _make_confirmation_folds(baseline_data)
        summary["fold_manifest"] = fold_manifest
        use_cuda = bool(_probe_cuda(log))
        summary["xgboost_cuda"] = use_cuda

        log("Training fresh V30 triple baseline on confirmation folds...")
        baseline = _run_baseline_triple(
            baseline_data, fold_id, use_cuda, log
        )
        log("Training fresh sequence family specialists on the same confirmation folds...")
        specialist = _run_specialists(
            augmented_data,
            fold_id,
            log,
            model_seed=CONFIRMATION_SEED,
        )
        summary["family_metrics"] = specialist["family_metrics"]
        summary["family_accuracy"] = specialist["family_accuracy"]

        confirmation_arm = _build_arms(
            baseline["oof"], specialist["oof"]
        )[FROZEN_ARM]
        baseline_metrics, arm_metrics = _score_arms(
            {FROZEN_ARM: confirmation_arm},
            baseline["oof"],
            baseline_data["y"],
            baseline_data["known"],
            baseline_data["stress_weight"],
            fold_id,
        )
        arm_metric = arm_metrics[FROZEN_ARM]
        gate = _gate_arm(arm_metric, baseline_metrics)
        summary["baseline_metrics"] = baseline_metrics
        summary["confirmation_metrics"] = arm_metric
        summary["confirmation_gate"] = gate
        summary["discovery_and_confirmation_passed"] = bool(
            discovery_gate["passed"] and gate["passed"]
        )
        log(
            f"Frozen {FROZEN_ARM}: confirmed {arm_metric['confirmed_ap']:.7f} "
            f"({arm_metric['confirmed_ap_delta_vs_baseline']:+.7f}), "
            f"stress {arm_metric['pu_stress_ap']:.7f} "
            f"({arm_metric['pu_stress_ap_delta_vs_baseline']:+.7f}), "
            f"positive confirmed folds={arm_metric['positive_confirmed_folds']}/5, "
            f"gate={gate['passed']}."
        )

        oof_path = out_dir / "oof_rows.parquet"
        pd.DataFrame(
            {
                "pair_id": baseline_data["pair_ids"],
                "cv_group": baseline_data["groups"].astype(str),
                "outer_fold": fold_id,
                "known": baseline_data["known"],
                "behavior_y": baseline_data["behavior_y"],
                "y": baseline_data["y"],
                "baseline_risk": baseline["oof"],
                "specialist_directed": specialist["oof"][:, 0],
                "specialist_soft": specialist["oof"][:, 1],
                "specialist_isolation": specialist["oof"][:, 2],
                FROZEN_ARM: confirmation_arm,
            }
        ).to_parquet(oof_path, index=False)
        eval_scores_path = out_dir / "eval_specialist_scores.parquet"
        pd.DataFrame(
            {
                "pair_id": augmented_data["eval_pair_ids"],
                "specialist_directed": specialist["eval"][:, 0],
                "specialist_soft": specialist["eval"][:, 1],
                "specialist_isolation": specialist["eval"][:, 2],
            }
        ).to_parquet(eval_scores_path, index=False)

        summary["provenance"] = _provenance_manifest(
            root,
            paths,
            hnm_dir,
            out_dir,
            dev_sequence,
            eval_sequence,
        )
        summary["artifacts"] = {
            "oof_rows": str(oof_path),
            "eval_specialist_scores": str(eval_scores_path),
            "baseline_feature_manifest": str(
                baseline_manifest_dir / "feature_columns.json"
            ),
            "augmented_feature_manifest": str(
                augmented_manifest_dir / "feature_columns.json"
            ),
        }

        if gate["passed"]:
            source_path, source_kind = _candidate_source(root)
            eval_base = _load_original_eval_risk(
                source_path, augmented_data["eval_pair_ids"]
            )
            eval_arm = _build_arms(eval_base, specialist["eval"])[FROZEN_ARM]
            summary["rank_comparison"] = _rank_comparison(
                augmented_data["eval_pair_ids"], eval_arm, source_path
            )
            emitted = _emit_candidate(
                "sequence_confirmation_rank_sum_w50",
                eval_arm,
                augmented_data["eval_pair_ids"],
                source_path,
                source_kind,
                paths.data_dir / "evaluation_pairs.csv",
                out_dir,
            )
            summary["candidate"] = emitted
            summary["status"] = "COMPLETED_PROMOTED"
            log(f"INDEPENDENTLY CONFIRMED: {emitted['path']}")
        else:
            summary["candidate"] = None
            summary["status"] = "COMPLETED_NULL"
            log("Independent confirmation failed; no candidate emitted.")

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
