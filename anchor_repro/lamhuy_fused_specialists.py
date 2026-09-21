"""Fused-basis ordered-sequence family specialists on the full V30 frame.

The LB-proven route (dossier §95, real LB 0.70904) trains family-decoupled LightGBM
specialists on the V30 frame augmented with the ordered-action sequence block, then
rank-blends their union with the baseline risk. This module tests dossier §99: widen
that SAME discriminative specialist basis with two persisted, drift-clean, pair-level
feature blocks that were previously only tried as weak post-hoc risk blends:

* temporal-directional pair features (24 columns), and
* partner-vs-field contrast features.

Everything else is reused verbatim from :mod:`anchor_repro.lamhuy_sequence_specialists`
(exact persisted table-group-disjoint folds, per-family LGBM structure, PU-stress
weighting, the eight pre-registered composition arms, gates, and the frozen
``rank_sum_w50`` promotion lineage). The only change is the appended feature blocks,
joined identically for development and evaluation. A discovery pass plus an independent
seed-137 confirmation pass are both run. A validated risk-only CSV is emitted only when
a material, fold-robust arm clears the gate on BOTH passes.

Run from the Poker workspace root::

    python -m anchor_repro.lamhuy_fused_specialists
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

from anchor_repro.lamhuy_hard_negative import (
    N_SPLITS,
    SEED,
    _candidate_source,
    _emit_candidate,
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
from anchor_repro.lamhuy_sequence_specialists import (
    ARM_DEFINITIONS,
    _build_arms,
    _gate_arm,
    _load_original_eval_risk,
    _run_specialists,
    _score_arms,
)

RECIPE_ID = "lamhuy_v30_fused_family_specialists_v1"
CONFIRMATION_SEED = 137
EXPECTED_DEV_ROWS = 25860
EXPECTED_EVAL_ROWS = 112540

# Persisted, drift-clean pair-level blocks appended to the specialist basis. Each parquet
# is keyed by pair_id and covers the exact dev/eval pair sets. Prefixes keep the appended
# columns collision-free and auditable inside the shared feature-selection contract.
FUSED_BLOCKS = (
    ("temporal_directional", "fdir_"),
    ("field_contrast", "ffc_"),
)


def _write_run_files(out_dir: Path, summary: dict, log_lines: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_fused_specialists_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    (out_dir / "_fused_specialists.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )


def _load_block(path: Path, prefix: str, expected_rows: int) -> Tuple[pl.DataFrame, List[str]]:
    frame = pl.read_parquet(path)
    if frame.height != expected_rows:
        raise ValueError(f"{path} has {frame.height} rows, expected {expected_rows}")
    if "pair_id" not in frame.columns:
        raise ValueError(f"{path} lacks pair_id")
    numeric = [
        column
        for column, dtype in frame.schema.items()
        if dtype.is_numeric() and column != "pair_id"
    ]
    if not numeric:
        raise ValueError(f"{path} has no numeric feature columns")
    renamed = frame.select(
        ["pair_id", *[pl.col(column).alias(f"{prefix}{column}") for column in numeric]]
    )
    return renamed, [f"{prefix}{column}" for column in numeric]


def _append_fused_blocks(
    dev: pl.DataFrame,
    evaluation: pl.DataFrame,
    block_dir: Path,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, List[str]]]:
    """Join the fused blocks onto dev + eval via the SAME code path (transport fidelity)."""
    block_columns: Dict[str, List[str]] = {}
    existing = set(dev.columns) | set(evaluation.columns)
    for block_name, prefix in FUSED_BLOCKS:
        dev_path = block_dir / f"dev_{block_name}.parquet"
        eval_path = block_dir / f"eval_{block_name}.parquet"
        if not dev_path.is_file() or not eval_path.is_file():
            raise FileNotFoundError(f"Missing fused block parquet(s): {dev_path}, {eval_path}")
        dev_block, dev_cols = _load_block(dev_path, prefix, EXPECTED_DEV_ROWS)
        eval_block, eval_cols = _load_block(eval_path, prefix, EXPECTED_EVAL_ROWS)
        if dev_cols != eval_cols:
            raise ValueError(f"{block_name}: dev/eval feature columns differ")
        if set(dev_cols) & existing:
            raise ValueError(f"{block_name}: appended feature names collide with base frame")
        existing |= set(dev_cols)

        dev = dev.join(dev_block, on="pair_id", how="left", validate="1:1")
        evaluation = evaluation.join(eval_block, on="pair_id", how="left", validate="1:1")
        # A pair absent from a block genuinely has no such interaction -> fill 0.0.
        dev = dev.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in dev_cols])
        evaluation = evaluation.with_columns(
            [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in eval_cols]
        )
        block_columns[block_name] = dev_cols

    if dev.height != EXPECTED_DEV_ROWS or evaluation.height != EXPECTED_EVAL_ROWS:
        raise ValueError("Fused join changed pair row counts")
    return dev, evaluation, block_columns


def _load_augmented_frames(hnm_dir: Path, sequence_dir: Path):
    base_dev, groups, base_eval = _load_full_frames(hnm_dir)
    dev_seq = sequence_dir / "dev_sequence_features.parquet"
    eval_seq = sequence_dir / "eval_sequence_features.parquet"
    if not dev_seq.is_file() or not eval_seq.is_file():
        raise FileNotFoundError(
            "Run python -m anchor_repro.lamhuy_ordered_sequence first; sequence caches missing"
        )
    sequence_dev = pl.read_parquet(dev_seq)
    sequence_eval = pl.read_parquet(eval_seq)
    dev, columns = _append_sequence_block(base_dev, sequence_dev)
    evaluation, eval_columns = _append_sequence_block(base_eval, sequence_eval)
    if columns != eval_columns:
        raise ValueError("Development/evaluation sequence feature order differs")
    return dev, groups, evaluation, columns


def _pass_metrics(
    data: Dict[str, object],
    fold_id: np.ndarray,
    baseline_oof: np.ndarray,
    model_seed: int,
    log: Callable[[str], None],
    label: str,
) -> Dict[str, object]:
    specialist = _run_specialists(data, fold_id, log, model_seed=model_seed)
    oof_arms = _build_arms(baseline_oof, specialist["oof"])
    baseline_metrics, arm_metrics = _score_arms(
        oof_arms,
        baseline_oof,
        data["y"],
        data["known"],
        data["stress_weight"],
        fold_id,
    )
    gates = {name: _gate_arm(metrics, baseline_metrics) for name, metrics in arm_metrics.items()}
    passing = [name for name, gate in gates.items() if gate["passed"]]
    for name in ARM_DEFINITIONS:
        metrics = arm_metrics[name]
        log(
            f"[{label}] {name}: confirmed {metrics['confirmed_ap']:.7f} "
            f"({metrics['confirmed_ap_delta_vs_baseline']:+.7f}), "
            f"stress {metrics['pu_stress_ap']:.7f} "
            f"({metrics['pu_stress_ap_delta_vs_baseline']:+.7f}), "
            f"+folds={metrics['positive_confirmed_folds']}/5, gate={gates[name]['passed']}"
        )
    return {
        "specialist": specialist,
        "oof_arms": oof_arms,
        "baseline_metrics": baseline_metrics,
        "arm_metrics": arm_metrics,
        "gates": gates,
        "passing_arms": passing,
        "family_metrics": specialist["family_metrics"],
        "family_accuracy": specialist["family_accuracy"],
        "fold_family_metrics": specialist["fold_family_metrics"],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = default_lamhuy_paths(root)
    hnm_dir = root / "outputs" / "poker_collusion" / "lamhuy_hnm"
    sequence_dir = root / "outputs" / "poker_collusion" / "lamhuy_ordered_sequence"
    block_dir = root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_fused_specialists"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    summary: Dict[str, object] = {
        "recipe_id": RECIPE_ID,
        "status": "RUNNING",
        "note": "BUILD + MEASURE ONLY; no Kaggle submission is made automatically.",
        "pre_registration": "RESEARCH_DOSSIER.md §99",
        "output_dir": str(out_dir),
        "discovery_seed": SEED,
        "confirmation_seed": CONFIRMATION_SEED,
        "frozen_lineage_arm": "rank_sum_w50",
        "fused_blocks": [name for name, _ in FUSED_BLOCKS],
    }
    started = time.time()
    for stale in out_dir.glob("candidate_v30_*fused_specialists_*.csv"):
        stale.unlink()

    try:
        dev, pair_groups, evaluation, sequence_columns = _load_augmented_frames(hnm_dir, sequence_dir)
        dev, evaluation, block_columns = _append_fused_blocks(dev, evaluation, block_dir)
        summary["sequence_feature_count"] = len(sequence_columns)
        summary["fused_feature_counts"] = {name: len(cols) for name, cols in block_columns.items()}
        summary["frame_shapes"] = {"dev": list(dev.shape), "eval": list(evaluation.shape)}

        data = _prepare_training_data(paths, dev, pair_groups, evaluation, out_dir)
        model_feature_count = int(
            json.loads((out_dir / "feature_columns.json").read_text(encoding="utf-8"))[
                "model_feature_count"
            ]
        )
        summary["model_feature_count"] = model_feature_count

        baseline_rows = _load_baseline_oof(hnm_dir, data["pair_ids"])
        fold_id = baseline_rows["outer_fold"].to_numpy(dtype=np.int8)
        summary["fold_manifest"] = _validate_persisted_folds(
            data["pair_ids"], data["groups"], fold_id, hnm_dir / "fold_manifest.json"
        )
        baseline_oof = baseline_rows["baseline_risk"].to_numpy(dtype=np.float32)

        log("Discovery pass: training fused-basis specialists (seed %d)..." % SEED)
        discovery = _pass_metrics(data, fold_id, baseline_oof, SEED, log, "discovery")
        log("Confirmation pass: independent seed %d..." % CONFIRMATION_SEED)
        confirmation = _pass_metrics(data, fold_id, baseline_oof, CONFIRMATION_SEED, log, "confirm")

        def _arm_view(pass_result: Dict[str, object]) -> Dict[str, object]:
            return {
                "baseline_metrics": pass_result["baseline_metrics"],
                "arm_metrics": pass_result["arm_metrics"],
                "gates": pass_result["gates"],
                "passing_arms": pass_result["passing_arms"],
                "family_metrics": pass_result["family_metrics"],
                "family_accuracy": pass_result["family_accuracy"],
                "fold_family_metrics": pass_result["fold_family_metrics"],
            }

        summary["discovery"] = _arm_view(discovery)
        summary["confirmation"] = _arm_view(confirmation)

        robust = [
            name for name in ARM_DEFINITIONS
            if name in discovery["passing_arms"] and name in confirmation["passing_arms"]
        ]
        selected = (
            max(robust, key=lambda name: discovery["arm_metrics"][name]["confirmed_ap"])
            if robust
            else None
        )
        summary["robust_passing_arms"] = robust
        summary["selected_arm"] = selected

        # Persist discovery OOF for downstream reuse / audit parity with the 0.70904 run.
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
                "specialist_directed": discovery["specialist"]["oof"][:, 0],
                "specialist_soft": discovery["specialist"]["oof"][:, 1],
                "specialist_isolation": discovery["specialist"]["oof"][:, 2],
            }
        )
        for name, score in discovery["oof_arms"].items():
            oof_frame[name] = score
        oof_frame.to_parquet(oof_path, index=False)
        summary["oof_path"] = str(oof_path)

        if selected is None:
            summary["status"] = "COMPLETED_NULL"
            summary["candidate"] = None
            log("No arm cleared the gate on BOTH discovery and confirmation; no CSV emitted.")
            summary["duration_s"] = time.time() - started
            _write_run_files(out_dir, summary, log_lines)
            print(json.dumps({
                "status": summary["status"],
                "robust_passing_arms": robust,
                "discovery_rank_sum_w50": discovery["arm_metrics"]["rank_sum_w50"]["confirmed_ap_delta_vs_baseline"],
                "confirmation_rank_sum_w50": confirmation["arm_metrics"]["rank_sum_w50"]["confirmed_ap_delta_vs_baseline"],
            }, indent=2))
            return 0

        # Selected arm cleared BOTH passes -> build the eval candidate on the frozen source.
        source_path, source_kind = _candidate_source(root)
        eval_base = _load_original_eval_risk(source_path, data["eval_pair_ids"])
        eval_arms = _build_arms(eval_base, discovery["specialist"]["eval"])
        summary["rank_comparison_selected"] = _rank_comparison(
            data["eval_pair_ids"], eval_arms[selected], source_path
        )
        emitted = _emit_candidate(
            f"fused_specialists_{selected}",
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
        summary["duration_s"] = time.time() - started
        _write_run_files(out_dir, summary, log_lines)
        print(json.dumps({
            "status": summary["status"],
            "selected_arm": selected,
            "candidate_path": emitted["path"],
        }, indent=2))
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
