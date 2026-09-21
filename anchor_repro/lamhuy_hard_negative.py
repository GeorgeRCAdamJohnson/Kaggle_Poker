"""Nested confirmed-negative hard mining on the full LamHuy V30 feature frame.

BUILD + MEASURE ONLY. This module never submits and never modifies the established
``repro_lamhuy`` or recovered-original V30 artifacts. It extracts the notebook's full
417-column development/evaluation pair frames into an isolated cache, evaluates a
paired baseline and two confirmed-negative reweighting arms under identical outer
folds, and emits risk-only candidates only when the pre-registered promotion gates
pass.

Run from the Poker workspace root::

    python -m anchor_repro.lamhuy_hard_negative
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.graph_retest import _probe_cuda
from anchor_repro.lamhuy_recipe import (
    build_child_script,
    default_lamhuy_paths,
    validate_submission_csv,
)

SEED = 42
N_SPLITS = 5
EXPECTED_DEV_ROWS = 25_860
EXPECTED_EVAL_ROWS = 112_540
EXPECTED_FRAME_COLUMNS = 417
EXPECTED_PU_ROWS = 24_000
PU_STRESS_WEIGHT = 112_540 / 24_000
BLEND_WEIGHTS = {"xgb": 0.40, "lgbm": 0.35, "catboost": 0.25}
PROMOTION_BASELINE_AP = 0.96
PROMOTION_CONFIRMED_DELTA = 0.003
PROMOTION_STRESS_DELTA_FLOOR = -0.005
TOP_KS = (100, 500, 1000)

ARMS: Dict[str, Dict[str, float]] = {
    "baseline": {"fraction": 0.0, "negative_weight": 1.0},
    "hn20_w2": {"fraction": 0.20, "negative_weight": 2.0},
    "hn40_w4": {"fraction": 0.40, "negative_weight": 4.0},
}

SUBMISSION_COLUMNS = [
    "pair_id",
    "risk_score",
    "predicted_behavior",
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
]
NON_RISK_COLUMNS = [c for c in SUBMISSION_COLUMNS if c != "risk_score"]
WARM_CACHE_NAMES = [
    "dev_pairs.parquet",
    "eval_pairs.parquet",
    "dev_pair_hands.parquet",
    "eval_pair_hands.parquet",
    "action_context.parquet",
    "player_hand_features.parquet",
    "player_baselines.parquet",
    "dev_hand_features.parquet",
    "eval_hand_features.parquet",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_snapshot(paths: Iterable[Path]) -> Dict[str, Tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_summary(out_dir: Path, summary: dict, log_lines: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_lamhuy_hnm_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "_lamhuy_hnm.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _feature_cache_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "dev": out_dir / "dev_features.parquet",
        "groups": out_dir / "pair_groups.parquet",
        "eval": out_dir / "eval_features.parquet",
    }


def _extract_full_feature_frames(
    root: Path,
    out_dir: Path,
    log: Callable[[str], None],
    timeout_s: int = 7200,
) -> Dict[str, object]:
    """Run the notebook only through Cell 15 and persist its full pair frames."""
    cache_paths = _feature_cache_paths(out_dir)
    present = {name: path.is_file() for name, path in cache_paths.items()}
    if all(present.values()):
        log("Full feature caches already exist; extraction child not run.")
        return {"ran": False, "returncode": None, "duration_s": 0.0}
    if any(present.values()):
        raise RuntimeError(
            "Refusing to mix partial full-feature cache generations; either all three "
            f"caches must exist or all must be absent. Present map: {present}"
        )

    paths = default_lamhuy_paths(root)
    missing_inputs = paths.missing()
    if missing_inputs:
        raise FileNotFoundError("Missing LamHuy inputs: " + ", ".join(map(str, missing_inputs)))

    # The generated notebook program points /kaggle/working at repro_lamhuy. Requiring
    # every warm artifact makes all pre-injection cache branches read-only; if anything
    # is absent we refuse to launch rather than letting notebook code create/overwrite it.
    warm_paths = [paths.prepared_dir / name for name in WARM_CACHE_NAMES]
    missing_warm = [path for path in warm_paths if not path.is_file()]
    if missing_warm:
        raise FileNotFoundError(
            "Refusing feature extraction because repro_lamhuy warm cache is incomplete: "
            + ", ".join(map(str, missing_warm))
        )
    protected = [*warm_paths, paths.submission_csv, paths.output_dir / "_repro_summary.json"]
    protected = [path for path in protected if path.exists()]
    before = _file_snapshot(protected)

    marker = (
        'print(f"Development matrix shape: {dev_features.shape} | Hand features: '
        '{len(HAND_FEATURES)} | Tail features: {len(TAIL_FEATURES)}")'
    )
    script = build_child_script(paths)
    # Cell 1 is plot-wrapped by lamhuy_recipe. If optional seaborn is absent, that
    # cell exits before reaching its tqdm import even though feature construction
    # itself does not require seaborn. Restore tqdm outside the plot-only wrapper.
    cell9_marker = "# --- notebook code cell 9 ---"
    if script.count(cell9_marker) != 1:
        raise RuntimeError("Could not locate notebook code cell 9 for tqdm restoration")
    script = script.replace(
        cell9_marker, "from tqdm.auto import tqdm\n" + cell9_marker, 1
    )
    if script.count(marker) != 1:
        raise RuntimeError(
            f"Expected exactly one development-matrix print marker, found {script.count(marker)}"
        )

    out_literal = repr(str(out_dir.resolve()))
    injection = f'''

# --- injected by anchor_repro.lamhuy_hard_negative ---
_HNM_OUT = Path({out_literal})
_HNM_OUT.mkdir(parents=True, exist_ok=True)
if not (_HNM_OUT / "dev_features.parquet").exists():
    dev_features.write_parquet(_HNM_OUT / "dev_features.parquet")
if not (_HNM_OUT / "pair_groups.parquet").exists():
    pair_groups.write_parquet(_HNM_OUT / "pair_groups.parquet")
print("lamhuy_hnm: building full evaluation feature frame...", flush=True)
eval_features = aggregate_pair_features(EVAL_HAND_FEATURES_PATH, eval_pairs_prep)
eval_features = add_partner_field_contrasts(
    eval_features, seats, hands, PLAYER_HAND_PATH, "evaluation"
)
if not (_HNM_OUT / "eval_features.parquet").exists():
    eval_features.write_parquet(_HNM_OUT / "eval_features.parquet")
print(
    f"lamhuy_hnm: persisted dev={{dev_features.shape}}, "
    f"groups={{pair_groups.shape}}, eval={{eval_features.shape}}",
    flush=True,
)
raise SystemExit(0)
# --- end injection ---
'''
    script = script.replace(marker, marker + injection, 1)
    script_path = out_dir / "_lamhuy_hnm_feature_child.py"
    child_log_path = out_dir / "_lamhuy_hnm_feature_child.log"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(out_dir / ".mpl_cache"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    (out_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    log("Full feature cache incomplete; running isolated notebook extraction child.")
    start = time.time()
    timed_out = False
    returncode: Optional[int]
    with child_log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("# LamHuy HNM full-frame extraction (stops after Cell 15)\n")
        handle.flush()
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(out_dir),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    duration = time.time() - start
    try:
        script_path.unlink()
    except OSError:
        pass

    after = _file_snapshot(protected)
    if before != after:
        raise RuntimeError("Protected repro_lamhuy artifacts changed during feature extraction")
    if timed_out or returncode != 0:
        tail = child_log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(
            f"Feature child failed: timed_out={timed_out}, returncode={returncode}\n{tail}"
        )
    missing_outputs = [path for path in cache_paths.values() if not path.is_file()]
    if missing_outputs:
        raise RuntimeError("Feature child omitted caches: " + ", ".join(map(str, missing_outputs)))
    log(f"Feature extraction completed in {duration:.1f}s; repro_lamhuy snapshot unchanged.")
    return {
        "ran": True,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": duration,
        "child_log": str(child_log_path),
        "protected_snapshot_unchanged": True,
    }


def _load_full_frames(out_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cache = _feature_cache_paths(out_dir)
    dev = pl.read_parquet(cache["dev"])
    groups = pl.read_parquet(cache["groups"])
    evaluation = pl.read_parquet(cache["eval"])
    if dev.shape != (EXPECTED_DEV_ROWS, EXPECTED_FRAME_COLUMNS):
        raise ValueError(f"Unexpected full dev frame shape: {dev.shape}")
    if evaluation.shape != (EXPECTED_EVAL_ROWS, EXPECTED_FRAME_COLUMNS):
        raise ValueError(f"Unexpected full eval frame shape: {evaluation.shape}")
    if groups.height != EXPECTED_DEV_ROWS or set(groups.columns) != {"pair_id", "cv_group"}:
        raise ValueError(f"Unexpected pair_groups frame: {groups.shape}, {groups.columns}")
    if dev["pair_id"].n_unique() != EXPECTED_DEV_ROWS:
        raise ValueError("Development pair_id values are not unique")
    if evaluation["pair_id"].n_unique() != EXPECTED_EVAL_ROWS:
        raise ValueError("Evaluation pair_id values are not unique")
    return dev, groups, evaluation


def _prepare_training_data(
    paths,
    dev_features: pl.DataFrame,
    pair_groups: pl.DataFrame,
    eval_features: pl.DataFrame,
    out_dir: Path,
) -> Dict[str, object]:
    dev_labels = pl.read_csv(paths.data_dir / "development_labels.csv")
    known_info = dev_labels.select(["pair_id", "behavior_family"]).with_columns(
        pl.lit(True).alias("is_known")
    )
    train_df = (
        dev_features.join(known_info, on="pair_id", how="left")
        .join(pair_groups, on="pair_id", how="left")
        .sort("pair_id")
    )
    behavior_to_id = {
        "none": 0,
        "directed_transfer": 1,
        "soft_play": 2,
        "coordinated_isolation": 3,
    }
    known = train_df["is_known"].fill_null(False).to_numpy().astype(bool)
    behavior_y = np.array(
        [behavior_to_id.get(value, 0) for value in train_df["behavior_family"].fill_null("none").to_list()],
        dtype=np.int8,
    )
    y = (behavior_y > 0).astype(np.int8)
    exclude = {"pair_id", "player_1", "player_2", "cv_group", "is_known", "table_id"}
    feature_cols = [
        column
        for column, dtype in train_df.schema.items()
        if dtype.is_numeric() and column not in exclude
    ]
    missing_eval = [column for column in feature_cols if column not in eval_features.columns]
    if missing_eval:
        eval_features = eval_features.with_columns(
            [pl.lit(0.0).cast(pl.Float32).alias(column) for column in missing_eval]
        )

    X = (
        train_df.select(feature_cols)
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    X_eval = (
        eval_features.select(feature_cols)
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    groups = train_df["cv_group"].to_numpy()
    pair_ids = np.asarray(train_df["pair_id"].cast(pl.Utf8).to_list(), dtype=object)
    eval_pair_ids = np.asarray(eval_features["pair_id"].cast(pl.Utf8).to_list(), dtype=object)

    if (~known).sum() != EXPECTED_PU_ROWS:
        raise ValueError(f"Expected {EXPECTED_PU_ROWS} PU rows, found {(~known).sum()}")
    if known.sum() != len(dev_labels):
        raise ValueError("Known marker does not exactly match development_labels")
    if np.any(pd.isna(groups)):
        raise ValueError("Missing cv_group values")
    if len(X_eval) != EXPECTED_EVAL_ROWS:
        raise ValueError("Evaluation feature row count mismatch")

    fit_weight = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35)).astype(np.float64)
    stress_weight = np.where(known, 1.0, PU_STRESS_WEIGHT).astype(np.float64)
    feature_manifest = {
        "frame_columns": EXPECTED_FRAME_COLUMNS,
        "model_feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "excluded_fields": sorted(exclude),
        "missing_eval_filled_zero": missing_eval,
        "dtype": "float32",
        "infinity_policy": "replace +/-inf with nan, then fill 0",
    }
    (out_dir / "feature_columns.json").write_text(
        json.dumps(feature_manifest, indent=2), encoding="utf-8"
    )
    return {
        "train_df": train_df,
        "X": X,
        "X_eval": X_eval,
        "known": known,
        "behavior_y": behavior_y,
        "y": y,
        "groups": groups,
        "pair_ids": pair_ids,
        "eval_pair_ids": eval_pair_ids,
        "fit_weight": fit_weight,
        "stress_weight": stress_weight,
        "feature_manifest": feature_manifest,
    }


def _xgb_params(use_cuda: bool) -> dict:
    params = dict(
        n_estimators=1600,
        learning_rate=0.022,
        max_depth=6,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=120,
        n_jobs=-1,
    )
    if use_cuda:
        params["device"] = "cuda"
    return params


LGB_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.025,
    num_leaves=45,
    min_child_samples=18,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.2,
    reg_lambda=8.0,
    objective="binary",
    n_jobs=-1,
    verbose=-1,
)
CB_PARAMS = dict(
    iterations=1100,
    learning_rate=0.030,
    depth=6,
    l2_leaf_reg=6.0,
    loss_function="Logloss",
    eval_metric="Logloss",
    verbose=0,
    thread_count=-1,
)


def _fold_metrics(
    indices: np.ndarray,
    y: np.ndarray,
    known: np.ndarray,
    stress_weight: np.ndarray,
    score: np.ndarray,
) -> Dict[str, float]:
    confirmed = indices[known[indices]]
    return {
        "confirmed_ap": float(average_precision_score(y[confirmed], score[confirmed])),
        "pu_stress_ap": float(
            average_precision_score(y[indices], score[indices], sample_weight=stress_weight[indices])
        ),
    }


def _mine_outer_training_negatives(
    outer_fold: int,
    outer_train: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    behavior_y: np.ndarray,
    groups: np.ndarray,
    known: np.ndarray,
    fit_weight: np.ndarray,
    stress_weight: np.ndarray,
    pair_ids: np.ndarray,
    use_cuda: bool,
    log: Callable[[str], None],
) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Produce group-disjoint inner OOF XGB scores and confirmed-negative selections."""
    from xgboost import XGBClassifier

    inner_scores = np.full(len(outer_train), np.nan, dtype=np.float32)
    inner_fold_id = np.full(len(outer_train), -1, dtype=np.int8)
    inner_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    X_outer = X.iloc[outer_train]
    y_outer = y[outer_train]
    behavior_outer = behavior_y[outer_train]
    groups_outer = groups[outer_train]

    for inner_fold, (inner_tr_local, inner_va_local) in enumerate(
        inner_cv.split(X_outer, behavior_outer, groups_outer)
    ):
        if set(groups_outer[inner_tr_local]) & set(groups_outer[inner_va_local]):
            raise AssertionError("Inner miner table leakage detected")
        inner_tr = outer_train[inner_tr_local]
        inner_va = outer_train[inner_va_local]
        model = XGBClassifier(
            **_xgb_params(use_cuda), random_state=SEED + inner_fold
        )
        model.fit(
            X.iloc[inner_tr],
            y[inner_tr],
            sample_weight=fit_weight[inner_tr],
            eval_set=[(X.iloc[inner_va], y[inner_va])],
            sample_weight_eval_set=[stress_weight[inner_va]],
            verbose=False,
        )
        inner_scores[inner_va_local] = model.predict_proba(X.iloc[inner_va])[:, 1]
        inner_fold_id[inner_va_local] = inner_fold
        del model
        gc.collect()

    if np.isnan(inner_scores).any() or np.any(inner_fold_id < 0):
        raise RuntimeError(f"Outer fold {outer_fold}: incomplete inner OOF mining scores")

    eligible_local = np.flatnonzero(known[outer_train] & (y[outer_train] == 0))
    # Explicit deterministic ordering: descending mining score, then pair_id. Python's
    # sort is stable, so rows identical on both keys retain their original order.
    ordered_local = np.asarray(
        sorted(
            eligible_local.tolist(),
            key=lambda local: (-float(inner_scores[local]), str(pair_ids[outer_train[local]])),
        ),
        dtype=np.int64,
    )
    rank_by_local = {int(local): rank + 1 for rank, local in enumerate(ordered_local)}
    selections: Dict[str, np.ndarray] = {"baseline": np.empty(0, dtype=np.int64)}
    selected_sets: Dict[str, set] = {"baseline": set()}
    for arm in ("hn20_w2", "hn40_w4"):
        count = int(math.ceil(len(ordered_local) * ARMS[arm]["fraction"]))
        selected = outer_train[ordered_local[:count]]
        if not np.all(known[selected] & (y[selected] == 0)):
            raise AssertionError(f"{arm} selected a non-confirmed-negative row")
        selections[arm] = selected
        selected_sets[arm] = set(map(int, selected))

    audit_rows: List[dict] = []
    eligible_set = set(map(int, eligible_local))
    for local, absolute in enumerate(outer_train):
        is_eligible = local in eligible_set
        audit_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": int(inner_fold_id[local]),
                "pair_id": str(pair_ids[absolute]),
                "cv_group": str(groups[absolute]),
                "known": bool(known[absolute]),
                "y": int(y[absolute]),
                "miner_score": float(inner_scores[local]),
                "eligible_confirmed_negative": bool(is_eligible),
                "eligible_rank": rank_by_local.get(local),
                "hn20_w2_selected": int(absolute) in selected_sets["hn20_w2"],
                "hn40_w4_selected": int(absolute) in selected_sets["hn40_w4"],
            }
        )
    log(
        f"Outer fold {outer_fold + 1}: nested miner scored {len(outer_train):,} outer-train rows; "
        f"eligible confirmed negatives={len(eligible_local):,}, "
        f"hn20={len(selections['hn20_w2']):,}, hn40={len(selections['hn40_w4']):,}."
    )
    return selections, audit_rows


def _run_nested_experiment(
    data: Dict[str, object],
    use_cuda: bool,
    log: Callable[[str], None],
) -> Dict[str, object]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X: pd.DataFrame = data["X"]
    X_eval: pd.DataFrame = data["X_eval"]
    y = data["y"]
    behavior_y = data["behavior_y"]
    groups = data["groups"]
    known = data["known"]
    fit_weight = data["fit_weight"]
    stress_weight = data["stress_weight"]
    pair_ids = data["pair_ids"]

    oof = {
        arm: {
            "xgb": np.zeros(len(X), dtype=np.float32),
            "lgbm": np.zeros(len(X), dtype=np.float32),
            "catboost": np.zeros(len(X), dtype=np.float32),
            "risk": np.zeros(len(X), dtype=np.float32),
        }
        for arm in ARMS
    }
    eval_components = {
        arm: {engine: [] for engine in ("xgb", "lgbm", "catboost")} for arm in ARMS
    }
    fold_id = np.full(len(X), -1, dtype=np.int8)
    fold_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_manifest: List[dict] = []
    mining_audit: List[dict] = []

    outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for outer_fold, (outer_train, outer_valid) in enumerate(
        outer_cv.split(X, behavior_y, groups)
    ):
        if set(groups[outer_train]) & set(groups[outer_valid]):
            raise AssertionError("Outer-fold table leakage detected")
        fold_id[outer_valid] = outer_fold
        fold_indices.append((outer_train, outer_valid))
        valid_hash = hashlib.sha256("\n".join(pair_ids[outer_valid]).encode("utf-8")).hexdigest()
        fold_manifest.append(
            {
                "fold": outer_fold,
                "train_rows": len(outer_train),
                "validation_rows": len(outer_valid),
                "train_groups": len(set(groups[outer_train])),
                "validation_groups": len(set(groups[outer_valid])),
                "validation_pair_id_sha256": valid_hash,
            }
        )
        selections, audit = _mine_outer_training_negatives(
            outer_fold,
            outer_train,
            X,
            y,
            behavior_y,
            groups,
            known,
            fit_weight,
            stress_weight,
            pair_ids,
            use_cuda,
            log,
        )
        mining_audit.extend(audit)

        for arm, arm_config in ARMS.items():
            arm_weight = fit_weight.copy()
            selected = selections[arm]
            if len(selected):
                arm_weight[selected] = arm_config["negative_weight"]
            if np.any((~known[selected]) | (y[selected] != 0)):
                raise AssertionError(f"{arm}: PU or positive row was reweighted")

            xgb = XGBClassifier(**_xgb_params(use_cuda), random_state=SEED + outer_fold)
            xgb.fit(
                X.iloc[outer_train],
                y[outer_train],
                sample_weight=arm_weight[outer_train],
                eval_set=[(X.iloc[outer_valid], y[outer_valid])],
                sample_weight_eval_set=[stress_weight[outer_valid]],
                verbose=False,
            )
            oof[arm]["xgb"][outer_valid] = xgb.predict_proba(X.iloc[outer_valid])[:, 1]
            eval_components[arm]["xgb"].append(xgb.predict_proba(X_eval)[:, 1])
            del xgb
            gc.collect()

            lgbm = LGBMClassifier(**LGB_PARAMS, random_state=SEED + outer_fold)
            lgbm.fit(
                X.iloc[outer_train],
                y[outer_train],
                sample_weight=arm_weight[outer_train],
                eval_set=[(X.iloc[outer_valid], y[outer_valid])],
                eval_sample_weight=[stress_weight[outer_valid]],
            )
            oof[arm]["lgbm"][outer_valid] = lgbm.predict_proba(X.iloc[outer_valid])[:, 1]
            eval_components[arm]["lgbm"].append(lgbm.predict_proba(X_eval)[:, 1])
            del lgbm
            gc.collect()

            catboost = CatBoostClassifier(**CB_PARAMS, random_seed=SEED + outer_fold)
            catboost.fit(
                X.iloc[outer_train],
                y[outer_train],
                sample_weight=arm_weight[outer_train],
                eval_set=[(X.iloc[outer_valid], y[outer_valid])],
                early_stopping_rounds=80,
                verbose=0,
            )
            oof[arm]["catboost"][outer_valid] = catboost.predict_proba(
                X.iloc[outer_valid]
            )[:, 1]
            eval_components[arm]["catboost"].append(catboost.predict_proba(X_eval)[:, 1])
            del catboost
            gc.collect()

            oof[arm]["risk"][outer_valid] = (
                BLEND_WEIGHTS["xgb"] * oof[arm]["xgb"][outer_valid]
                + BLEND_WEIGHTS["lgbm"] * oof[arm]["lgbm"][outer_valid]
                + BLEND_WEIGHTS["catboost"] * oof[arm]["catboost"][outer_valid]
            )
            current = _fold_metrics(
                outer_valid, y, known, stress_weight, oof[arm]["risk"]
            )
            log(
                f"Outer fold {outer_fold + 1} {arm}: confirmed AP "
                f"{current['confirmed_ap']:.6f}, PU-stress AP {current['pu_stress_ap']:.6f}."
            )

    if np.any(fold_id < 0):
        raise RuntimeError("Incomplete outer OOF assignment")

    eval_risk: Dict[str, np.ndarray] = {}
    metrics: Dict[str, dict] = {}
    for arm in ARMS:
        component_means = {
            engine: np.mean(eval_components[arm][engine], axis=0)
            for engine in ("xgb", "lgbm", "catboost")
        }
        eval_risk[arm] = (
            BLEND_WEIGHTS["xgb"] * component_means["xgb"]
            + BLEND_WEIGHTS["lgbm"] * component_means["lgbm"]
            + BLEND_WEIGHTS["catboost"] * component_means["catboost"]
        ).astype(np.float32)
        fold_metrics = [
            {"fold": fold, **_fold_metrics(va, y, known, stress_weight, oof[arm]["risk"])}
            for fold, (_, va) in enumerate(fold_indices)
        ]
        metrics[arm] = {
            "confirmed_ap": float(
                average_precision_score(y[known], oof[arm]["risk"][known])
            ),
            "pu_stress_ap": float(
                average_precision_score(y, oof[arm]["risk"], sample_weight=stress_weight)
            ),
            "folds": fold_metrics,
        }

    baseline = metrics["baseline"]
    for arm in ("hn20_w2", "hn40_w4"):
        metrics[arm]["confirmed_ap_delta_vs_baseline"] = (
            metrics[arm]["confirmed_ap"] - baseline["confirmed_ap"]
        )
        metrics[arm]["pu_stress_ap_delta_vs_baseline"] = (
            metrics[arm]["pu_stress_ap"] - baseline["pu_stress_ap"]
        )
        for fold in range(N_SPLITS):
            metrics[arm]["folds"][fold]["confirmed_ap_delta_vs_baseline"] = (
                metrics[arm]["folds"][fold]["confirmed_ap"]
                - baseline["folds"][fold]["confirmed_ap"]
            )
            metrics[arm]["folds"][fold]["pu_stress_ap_delta_vs_baseline"] = (
                metrics[arm]["folds"][fold]["pu_stress_ap"]
                - baseline["folds"][fold]["pu_stress_ap"]
            )

    return {
        "oof": oof,
        "eval_risk": eval_risk,
        "metrics": metrics,
        "fold_id": fold_id,
        "fold_manifest": fold_manifest,
        "mining_audit": mining_audit,
    }


def _persist_experiment_rows(
    out_dir: Path,
    data: Dict[str, object],
    result: Dict[str, object],
) -> Dict[str, str]:
    train_df: pl.DataFrame = data["train_df"]
    oof_frame = pd.DataFrame(
        {
            "pair_id": data["pair_ids"],
            "cv_group": data["groups"].astype(str),
            "outer_fold": result["fold_id"],
            "known": data["known"],
            "behavior_y": data["behavior_y"],
            "y": data["y"],
            "fit_weight": data["fit_weight"],
            "stress_weight": data["stress_weight"],
        }
    )
    for arm in ARMS:
        for component in ("xgb", "lgbm", "catboost", "risk"):
            oof_frame[f"{arm}_{component}"] = result["oof"][arm][component]
    oof_path = out_dir / "oof_rows.parquet"
    oof_frame.to_parquet(oof_path, index=False)

    audit_path = out_dir / "mining_audit.parquet"
    pd.DataFrame(result["mining_audit"]).to_parquet(audit_path, index=False)
    fold_path = out_dir / "fold_manifest.json"
    fold_path.write_text(json.dumps(result["fold_manifest"], indent=2), encoding="utf-8")
    return {"oof_rows": str(oof_path), "mining_audit": str(audit_path), "fold_manifest": str(fold_path)}


def _rank_comparison(
    pair_ids: np.ndarray,
    risk: np.ndarray,
    reference_path: Path,
) -> Optional[dict]:
    if not reference_path.is_file():
        return None
    reference = pd.read_csv(reference_path, dtype={"pair_id": str})
    proposed = pd.DataFrame({"pair_id": pair_ids.astype(str), "risk_new": risk})
    merged = proposed.merge(
        reference[["pair_id", "risk_score"]].rename(columns={"risk_score": "risk_reference"}),
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_EVAL_ROWS:
        raise ValueError(f"Rank comparison pair mismatch for {reference_path}")
    spearman = float(merged["risk_new"].corr(merged["risk_reference"], method="spearman"))
    overlaps = {}
    for k in TOP_KS:
        new_top = set(merged.nlargest(k, "risk_new")["pair_id"])
        reference_top = set(merged.nlargest(k, "risk_reference")["pair_id"])
        overlap = len(new_top & reference_top)
        overlaps[str(k)] = {"count": overlap, "rate": overlap / k}
    return {"path": str(reference_path), "spearman": spearman, "top_k_overlap": overlaps}


def _candidate_source(root: Path) -> Tuple[Path, str]:
    original = root / "outputs" / "poker_collusion" / "lamhuy_original_v30" / "submission.csv"
    local = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "submission.csv"
    if original.is_file():
        return original, "recovered_original_v30"
    return local, "local_repro_v30"


def _emit_candidate(
    arm: str,
    risk: np.ndarray,
    eval_pair_ids: np.ndarray,
    source_path: Path,
    source_kind: str,
    eval_pairs_csv: Path,
    out_dir: Path,
) -> dict:
    dtype_map = {column: str for column in NON_RISK_COLUMNS}
    source = pd.read_csv(source_path, dtype=dtype_map)
    if list(source.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Candidate source schema mismatch: {source.columns.tolist()}")
    risk_by_pair = pd.Series(risk.astype(np.float64), index=eval_pair_ids.astype(str))
    if not source["pair_id"].is_unique or not risk_by_pair.index.is_unique:
        raise ValueError("Candidate source/eval pair_id values must be unique")
    candidate = source.copy(deep=True)
    candidate["risk_score"] = candidate["pair_id"].map(risk_by_pair)
    if candidate["risk_score"].isna().any():
        raise ValueError("Candidate risk mapping left null rows")
    if not np.isfinite(candidate["risk_score"]).all():
        raise ValueError("Candidate risk contains non-finite values")
    if not candidate["risk_score"].between(0.0, 1.0).all():
        raise ValueError("Raw averaged candidate risk is outside [0,1]")
    exact_before_write = bool(candidate[NON_RISK_COLUMNS].equals(source[NON_RISK_COLUMNS]))
    if not exact_before_write:
        raise AssertionError("Non-risk source columns changed before candidate write")

    path = out_dir / f"candidate_v30_hnm_{arm}.csv"
    candidate.to_csv(path, index=False)
    validation = validate_submission_csv(path, eval_pairs_csv)
    written = pd.read_csv(path, dtype=dtype_map)
    source_check = source.set_index("pair_id").loc[written["pair_id"]].reset_index()
    exact_after_write = bool(written[NON_RISK_COLUMNS].equals(source_check[NON_RISK_COLUMNS]))
    uniqueness = bool(written["pair_id"].is_unique)
    if not validation.get("all_passed") or not uniqueness or not exact_after_write:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Candidate validation failed: contract={validation}, uniqueness={uniqueness}, "
            f"exact_non_risk={exact_after_write}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "source_path": str(source_path),
        "source_kind": source_kind,
        "raw_risk_min": float(risk.min()),
        "raw_risk_max": float(risk.max()),
        "risk_normalized": False,
        "pair_id_unique": uniqueness,
        "exact_non_risk_preservation": exact_after_write,
        "validation": validation,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = default_lamhuy_paths(root)
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_hnm"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    summary: Dict[str, object] = {
        "recipe_id": "lamhuy_v30_nested_confirmed_negative_hnm",
        "status": "RUNNING",
        "output_dir": str(out_dir),
        "seed": SEED,
        "outer_cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "inner_cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), outer-train tables only",
        "arms": ARMS,
        "blend_weights": BLEND_WEIGHTS,
        "fit_weights": {"positive": 2.0, "confirmed_negative": 1.0, "pu": 0.35},
        "stress_weights": {"known": 1.0, "pu": PU_STRESS_WEIGHT},
        "promotion_gate": {
            "baseline_confirmed_ap_min": PROMOTION_BASELINE_AP,
            "candidate_confirmed_ap_delta_min": PROMOTION_CONFIRMED_DELTA,
            "candidate_pu_stress_ap_delta_min": PROMOTION_STRESS_DELTA_FLOOR,
        },
        "deviations": [
            {
                "scope": "feature-extraction child compatibility only",
                "detail": (
                    "lamhuy_recipe plot-wraps notebook Cell 1; because optional seaborn "
                    "is unavailable, the generated child restores tqdm immediately before "
                    "Cell 9 so feature progress loops remain defined"
                ),
                "model_or_data_change": False,
                "full_v30_parameters_retained": True,
            }
        ],
    }
    started = time.time()

    # A failed/rerun experiment must never leave a stale promoted artifact behind.
    for arm in ("hn20_w2", "hn40_w4"):
        stale = out_dir / f"candidate_v30_hnm_{arm}.csv"
        if stale.exists():
            stale.unlink()

    try:
        log("Preparing full 417-column LamHuy development/evaluation feature caches.")
        extraction = _extract_full_feature_frames(root, out_dir, log)
        summary["feature_extraction"] = extraction
        dev_features, pair_groups, eval_features = _load_full_frames(out_dir)
        summary["feature_frame_shapes"] = {
            "dev": list(dev_features.shape),
            "pair_groups": list(pair_groups.shape),
            "eval": list(eval_features.shape),
        }
        log(
            f"Loaded full frames: dev={dev_features.shape}, groups={pair_groups.shape}, "
            f"eval={eval_features.shape}."
        )

        data = _prepare_training_data(paths, dev_features, pair_groups, eval_features, out_dir)
        summary["feature_manifest"] = data["feature_manifest"]
        summary["label_counts"] = {
            "known": int(data["known"].sum()),
            "pu": int((~data["known"]).sum()),
            "positive": int(data["y"].sum()),
            "confirmed_negative": int((data["known"] & (data["y"] == 0)).sum()),
        }
        log(
            f"Prepared {data['X'].shape[1]} numeric float32 model features; "
            f"labels={summary['label_counts']}."
        )

        use_cuda = _probe_cuda(log)
        summary["xgboost_cuda_probe_passed"] = bool(use_cuda)
        summary["device_policy"] = {
            "xgboost": "cuda" if use_cuda else "cpu hist",
            "lightgbm": "cpu",
            "catboost": "cpu",
        }
        log("Running paired nested-mining experiment (25 inner XGB + 45 outer arm models).")
        experiment = _run_nested_experiment(data, use_cuda, log)
        summary["metrics"] = experiment["metrics"]
        summary["fold_manifest"] = experiment["fold_manifest"]
        summary["artifacts"] = _persist_experiment_rows(out_dir, data, experiment)

        local_v30 = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "submission.csv"
        original_v30 = root / "outputs" / "poker_collusion" / "lamhuy_original_v30" / "submission.csv"
        rank_comparisons = {}
        for arm, risk in experiment["eval_risk"].items():
            rank_comparisons[arm] = {
                "local_v30": _rank_comparison(data["eval_pair_ids"], risk, local_v30),
                "original_v30": _rank_comparison(data["eval_pair_ids"], risk, original_v30),
            }
        summary["eval_rank_comparisons"] = rank_comparisons

        baseline_metrics = experiment["metrics"]["baseline"]
        baseline_gate = baseline_metrics["confirmed_ap"] >= PROMOTION_BASELINE_AP
        summary["baseline_gate_passed"] = bool(baseline_gate)

        candidates = {}
        source: Optional[Tuple[Path, str]] = None
        for arm in ("hn20_w2", "hn40_w4"):
            arm_metrics = experiment["metrics"][arm]
            gate = {
                "baseline_confirmed_ap_pass": bool(baseline_gate),
                "confirmed_ap_delta_pass": bool(
                    arm_metrics["confirmed_ap_delta_vs_baseline"] >= PROMOTION_CONFIRMED_DELTA
                ),
                "pu_stress_delta_pass": bool(
                    arm_metrics["pu_stress_ap_delta_vs_baseline"]
                    >= PROMOTION_STRESS_DELTA_FLOOR
                ),
            }
            gate["all_passed"] = bool(all(gate.values()))
            if gate["all_passed"]:
                if source is None:
                    source = _candidate_source(root)
                    if not source[0].is_file():
                        raise FileNotFoundError(
                            "A promotion gate passed, but neither recovered original nor "
                            "local V30 submission exists for non-risk column preservation"
                        )
                source_path, source_kind = source
                candidate = _emit_candidate(
                    arm,
                    experiment["eval_risk"][arm],
                    data["eval_pair_ids"],
                    source_path,
                    source_kind,
                    paths.data_dir / "evaluation_pairs.csv",
                    out_dir,
                )
                log(f"PROMOTED {arm}: {candidate['path']} sha256={candidate['sha256']}")
            else:
                candidate = None
                log(f"NOT PROMOTED {arm}: gate={gate}")
            candidates[arm] = {"gate": gate, "candidate": candidate}
        summary["candidates"] = candidates
        emitted = [item["candidate"] for item in candidates.values() if item["candidate"]]
        summary["status"] = "COMPLETED_PROMOTED" if emitted else "COMPLETED_NULL"
        summary["verdict"] = (
            f"{len(emitted)} hard-negative arm(s) cleared all promotion gates."
            if emitted
            else "Neither confirmed-negative hard-mining arm cleared the promotion gates; no submission emitted."
        )
        summary["duration_s"] = time.time() - started
        _write_summary(out_dir, summary, log_lines)

        log("=" * 72)
        for arm, metrics in experiment["metrics"].items():
            log(
                f"{arm}: confirmed AP={metrics['confirmed_ap']:.6f}, "
                f"PU-stress AP={metrics['pu_stress_ap']:.6f}, "
                f"confirmed delta={metrics.get('confirmed_ap_delta_vs_baseline', 0.0):+.6f}, "
                f"stress delta={metrics.get('pu_stress_ap_delta_vs_baseline', 0.0):+.6f}"
            )
        log(summary["verdict"])
        _write_summary(out_dir, summary, log_lines)
        return 0

    except Exception as exc:  # noqa: BLE001 - persist an honest blocker with traceback.
        tb = traceback.format_exc()
        summary["status"] = "ERROR"
        summary["error"] = repr(exc)
        summary["traceback_tail"] = tb[-8000:]
        summary["duration_s"] = time.time() - started
        log("ERROR: " + repr(exc))
        log(tb[-8000:])
        _write_summary(out_dir, summary, log_lines)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
