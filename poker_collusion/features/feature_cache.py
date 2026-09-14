"""FEATURE-MATRIX CACHE: compute each pair's feature vector ONCE and persist it.

Why this module exists
----------------------
Every fast generator / CV gate pays the same up-front tax: ONE bounded streaming pass over the
18.6M-row ``actions.parquet`` (the ``build_rows_by_hand`` scan) plus a per-pair signal ->
feature build over ALL 112,540 evaluation pairs (or the ~1,860 confirmed development pairs). That
actions scan is the ~10-minute cost that dominates wall-clock. It is IDENTICAL on every run
because the inputs are identical, so it is pure waste for model iteration.

This module computes that feature vector ONCE per pair and persists it, so downstream experiments
load a matrix + labels/pools and fit/score in SECONDS instead of re-scanning actions:

* :func:`build_and_cache_eval_matrix` — ONE eval actions pass + per-pair feature build for ALL
  evaluation pairs -> ``eval_features.parquet``.
* :func:`build_and_cache_dev_matrix` — the confirmed-development matrix (reusing
  :func:`poker_collusion.discovery.signal_separation.build_confirmed_feature_matrix`) ->
  ``dev_features.parquet``.
* :func:`load_eval_matrix` / :func:`load_dev_matrix` — fast loaders returning
  ``(DataFrame, schema_dict)`` with a soft staleness check against the sidecar's input hash.

The record schema is the SAME one the LEARNED risk head (CV AUC ~0.85) trains and scores on
------------------------------------------------------------------------------------------
Both matrices use :func:`poker_collusion.models.learned_risk.pair_record` (eval) /
:func:`poker_collusion.discovery.signal_separation.build_confirmed_feature_matrix` (dev), which
build IDENTICAL per-pair records: every numeric ``PairFeatureSet`` feature (finite float;
NaN/+-inf -> 0.0) PLUS the classical-risk pseudo-feature (:data:`CLASSICAL_RISK_KEY`) PLUS the
cheap derived candidates. The persisted feature columns are therefore exactly what a downstream
``StandardScaler + LogisticRegression`` needs to reproduce the ~0.85 group-by-pool CV AUC directly
from the cache — no actions rescan.

DRY reuse (nothing here reinvents the feature math)
---------------------------------------------------
* Eval precompute: :func:`poker_collusion.pipeline_fast.build_eval_hands_by_player` +
  :func:`build_rows_by_hand` (the single actions pass), :func:`poker_collusion.pipeline._net_chips_by_hand`
  + :func:`_hands_meta_for`, then the per-pair model-free build via the shared driver
  :func:`poker_collusion.parallel_scoring.score_pairs_sequential` /
  :func:`score_pairs_parallel` (which calls ``compute_pair_signals -> build_pair_feature_set``
  with ``phase="evaluation"`` — the exact eval feature flow).
* Per-pair record: :func:`poker_collusion.models.learned_risk.pair_record`.
* Dev matrix: :func:`poker_collusion.discovery.signal_separation.build_confirmed_feature_matrix`
  over the cached ``_prepare_labelled_inputs`` single-pass dev inputs, plus
  :func:`poker_collusion.validation.cv.derive_pool_map` for the pool column.

Determinism
-----------
The classical baseline and the vectorized precompute are order-independent; the feature-column
schema is a ``sorted(...)`` union; the parquet rows are written in the deterministic pair order
(evaluation-pair file order for eval, confirmed-pair order for dev). Two runs on identical
inputs/config produce identical matrices and identical sidecar hashes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from poker_collusion.config import PipelineConfig, get_config
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.io import DataLoader
from poker_collusion.models.learned_risk import (
    CLASSICAL_RISK_KEY,
    CONFIRMED_NON_TARGET,
    CONFIRMED_TARGET,
    pair_record,
)
from poker_collusion.parallel_scoring import (
    score_pairs_parallel,
    score_pairs_sequential,
)
from poker_collusion.pipeline import (
    _canonical_players,
    _hands_meta_for,
    _net_chips_by_hand,
    _obtain_eda_artifact,
)
from poker_collusion.pipeline_fast import build_eval_hands_by_player, build_rows_by_hand
from poker_collusion.pipeline_phase2_fast import _meta_record_map

__all__ = [
    "SCHEMA_VERSION",
    "EVAL_MATRIX_NAME",
    "DEV_MATRIX_NAME",
    "HAS_EVAL_SIGNAL_COLUMN",
    "PAIR_ID_COLUMN",
    "LABEL_COLUMN",
    "POOL_COLUMN",
    "default_cache_dir",
    "build_and_cache_eval_matrix",
    "build_and_cache_dev_matrix",
    "load_eval_matrix",
    "load_dev_matrix",
]

#: Bump when the cache row/sidecar layout changes (independent of the feature schema_version).
SCHEMA_VERSION = "feature_cache_v1"

#: Persisted file names (parquet matrix + JSON sidecar) for each matrix.
EVAL_MATRIX_NAME = "eval_features"
DEV_MATRIX_NAME = "dev_features"

#: Reserved (non-feature) column names in the persisted matrices.
PAIR_ID_COLUMN = "pair_id"
HAS_EVAL_SIGNAL_COLUMN = "has_eval_signal"
LABEL_COLUMN = "label"
POOL_COLUMN = "pool"

#: The real competition input files whose bytes define the content hash (staleness check).
_INPUT_FILE_STEMS = ("actions", "hands", "seats", "players", "development_labels", "evaluation_pairs")
_PARQUET_STEMS = frozenset({"actions", "hands", "seats", "players"})


# --------------------------------------------------------------------------- #
# Paths + provenance helpers
# --------------------------------------------------------------------------- #
def default_cache_dir(config: Optional[PipelineConfig] = None) -> Path:
    """Return the default cache directory ``<output_dir>/feature_cache``."""
    cfg = config or get_config()
    return Path(cfg.output_dir) / "feature_cache"


def _matrix_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / f"{name}.parquet"


def _sidecar_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / f"{name}.meta.json"


def _content_hash(config: PipelineConfig) -> str:
    """SHA-256 over the (name, size, mtime_ns) of every real input file present.

    A cheap, deterministic fingerprint of the inputs the matrices were built from. We hash file
    metadata (not full contents) so the check stays fast on the 18.6M-row actions file; the size +
    mtime pair changes whenever the data is regenerated, which is all the soft staleness check
    needs. Missing files are skipped (recorded as ``<stem>:MISSING``) so a synthetic fixture that
    omits, say, players still produces a stable hash.
    """
    h = hashlib.sha256()
    base = Path(config.input_dir)
    for stem in _INPUT_FILE_STEMS:
        ext = ".parquet" if stem in _PARQUET_STEMS else ".csv"
        path = base / f"{stem}{ext}"
        if path.exists():
            st = path.stat()
            h.update(f"{stem}:{st.st_size}:{st.st_mtime_ns}".encode("ascii"))
        else:
            h.update(f"{stem}:MISSING".encode("ascii"))
    return h.hexdigest()


def _write_sidecar(
    out_dir: Path,
    name: str,
    *,
    feature_columns: Sequence[str],
    row_count: int,
    config: PipelineConfig,
    matrix_kind: str,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Write the JSON sidecar (ordered feature columns + provenance) atomically."""
    sidecar = {
        "matrix_kind": matrix_kind,
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": config.schema_version,
        "threshold_version": config.threshold_version,
        "feature_columns": list(feature_columns),
        "n_features": len(feature_columns),
        "row_count": int(row_count),
        "content_hash": _content_hash(config),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        sidecar.update(extra)
    target = _sidecar_path(out_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="ascii")
    tmp.replace(target)
    return target


def _write_matrix(out_dir: Path, name: str, frame: pd.DataFrame) -> Path:
    """Write the matrix parquet atomically and return its path."""
    target = _matrix_path(out_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------- #
# (1) EVAL matrix: one eval actions pass + per-pair feature build for ALL eval pairs
# --------------------------------------------------------------------------- #
def build_and_cache_eval_matrix(
    config: Optional[PipelineConfig] = None,
    out_dir: Optional[Path] = None,
    *,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    parallel: bool = False,
    n_workers: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Build + persist the evaluation feature matrix for ALL evaluation pairs.

    ONE bounded actions pass over the union of every eval pair's shared evaluation hands (the
    Phase-1 verbatim precompute), then the model-free per-pair feature build, then one
    :func:`pair_record` per scoreable pair. Pairs with no shared evaluation hands get an all-zero
    feature row with ``has_eval_signal = False`` (so the matrix covers the EXACT eval pair set).

    Persists ``eval_features.parquet`` (columns: ``pair_id`` + every numeric feature in a fixed
    sorted schema + boolean ``has_eval_signal``) plus a JSON sidecar (ordered feature columns,
    schema/threshold version, row count, build timestamp, content hash of the inputs).

    Args:
        config: Pipeline config (defaults to :func:`get_config`).
        out_dir: Cache directory (defaults to ``<output_dir>/feature_cache``).
        data_loader: Optional pre-built loader.
        eda_artifact: Optional pre-computed artifact (the pool prior / min-shared-hands / classical
            risk column source — MUST match the artifact downstream fitting uses).
        parallel: Route the per-pair build through the deterministic multiprocess driver.
        n_workers: Worker count when ``parallel=True``.

    Returns:
        ``(matrix_path, sidecar_path)``.
    """
    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)
    out = Path(out_dir) if out_dir is not None else default_cache_dir(cfg)
    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)

    evaluation_pairs = loader.load_evaluation_pairs()
    seats = loader.load_seats()
    hands = loader.load_hands()

    # Eval-hand sets per player + per-pair shared eval hands (Phase-1 verbatim).
    eval_hands_by_player = build_eval_hands_by_player(seats, hands)
    pair_rows: List[dict] = []
    all_eval_hand_ids: Set[str] = set()
    for row in evaluation_pairs.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        player_a, player_b = _canonical_players(
            getattr(row, "player_1"), getattr(row, "player_2")
        )
        set_a = eval_hands_by_player.get(player_a, frozenset())
        set_b = eval_hands_by_player.get(player_b, frozenset())
        eval_hands = sorted(set_a & set_b)
        all_eval_hand_ids.update(eval_hands)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "player_a": player_a,
                "player_b": player_b,
                "eval_hands": eval_hands,
            }
        )

    # ONE bounded actions pass + net/meta over the whole eval-hand union.
    ordered_eval_hand_ids = sorted(all_eval_hand_ids)
    rows_by_hand = build_rows_by_hand(loader, ordered_eval_hand_ids)
    net_by_hand = _net_chips_by_hand(seats, ordered_eval_hand_ids)
    eval_meta = _hands_meta_for(hands, ordered_eval_hand_ids)
    meta_cols, meta_record_by_hand = _meta_record_map(eval_meta)

    # Model-free per-pair build (signals -> PairFeatureSet phase="evaluation"), via shared driver.
    if parallel:
        scored = score_pairs_parallel(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=cfg,
            n_workers=n_workers,
        )
    else:
        scored = score_pairs_sequential(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=cfg,
        )

    # One record per scoreable pair (the 0.85-schema record); build the sorted-union column set.
    record_by_pair: Dict[str, Dict[str, float]] = {}
    feature_names: Set[str] = set()
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        res = scored.get(pair_id)
        if res is None:
            continue
        rec = pair_record(res.feature_set, eda_artifact=artifact)
        record_by_pair[pair_id] = rec
        feature_names.update(rec.keys())

    feature_columns = sorted(feature_names)

    # Assemble rows in the deterministic evaluation-pair order; cover the EXACT pair set.
    out_records: List[Dict[str, object]] = []
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        rec = record_by_pair.get(pair_id)
        row: Dict[str, object] = {PAIR_ID_COLUMN: pair_id}
        if rec is None:
            for col in feature_columns:
                row[col] = 0.0
            row[HAS_EVAL_SIGNAL_COLUMN] = False
        else:
            for col in feature_columns:
                row[col] = float(rec.get(col, 0.0))
            row[HAS_EVAL_SIGNAL_COLUMN] = True
        out_records.append(row)

    ordered_columns = [PAIR_ID_COLUMN] + feature_columns + [HAS_EVAL_SIGNAL_COLUMN]
    frame = pd.DataFrame(out_records, columns=ordered_columns)

    matrix_path = _write_matrix(out, EVAL_MATRIX_NAME, frame)
    sidecar_path = _write_sidecar(
        out,
        EVAL_MATRIX_NAME,
        feature_columns=feature_columns,
        row_count=len(frame),
        config=cfg,
        matrix_kind="eval",
        extra={
            "n_pairs_total": int(len(pair_rows)),
            "n_pairs_with_signal": int(sum(1 for r in out_records if r[HAS_EVAL_SIGNAL_COLUMN])),
        },
    )
    return matrix_path, sidecar_path


# --------------------------------------------------------------------------- #
# (2) DEV matrix: the confirmed-development matrix (reuse build_confirmed_feature_matrix)
# --------------------------------------------------------------------------- #
def build_and_cache_dev_matrix(
    config: Optional[PipelineConfig] = None,
    out_dir: Optional[Path] = None,
    *,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
) -> Tuple[Path, Path]:
    """Build + persist the confirmed-development feature matrix.

    Reuses :func:`poker_collusion.discovery.signal_separation.build_confirmed_feature_matrix` over
    the cached single-pass development inputs (:func:`_prepare_labelled_inputs`), so the measured
    population is IDENTICAL to the one the ~0.85 multivariate CV AUC was measured on. One row per
    confirmed dev pair that has development shared hands.

    Persists ``dev_features.parquet`` (columns: ``pair_id`` + every numeric feature in a fixed
    sorted schema + ``label`` 0/1 + ``pool``) plus the same JSON sidecar as the eval matrix.

    Returns:
        ``(matrix_path, sidecar_path)``.
    """
    from poker_collusion.discovery.signal_separation import build_confirmed_feature_matrix
    from poker_collusion.validation.cv import derive_pool_map
    from poker_collusion.validation.phase1_baseline_cv import _prepare_labelled_inputs

    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)
    out = Path(out_dir) if out_dir is not None else default_cache_dir(cfg)
    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)

    labels = loader.load_labels_frame()
    confirmed_pair_ids = [
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}
    ]

    # ONE bounded actions pass over the confirmed dev pairs' development shared hands.
    inputs = _prepare_labelled_inputs(loader, labels, confirmed_pair_ids)
    pool_map = derive_pool_map(labels, config=cfg)

    frame, y, groups, feature_columns, _derived = build_confirmed_feature_matrix(
        inputs, labels, pool_map, artifact, cfg
    )

    # build_confirmed_feature_matrix drops pairs with no dev shared hands (its row order == the
    # status_by_pair iteration order). Recover the SAME pair-id order so we can attach ids/pools.
    status_by_pair: Dict[str, str] = {}
    for r in labels.itertuples(index=False):
        st = str(getattr(r, "label_status", ""))
        if st in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}:
            status_by_pair[str(getattr(r, "pair_id"))] = st

    kept_pair_ids: List[str] = []
    from poker_collusion.validation.phase1_baseline_cv import _signals_for_pair

    for pair_id in status_by_pair:
        if _signals_for_pair(inputs, pair_id):
            kept_pair_ids.append(pair_id)

    # Sanity: the recovered id order must line up with the matrix rows.
    if len(kept_pair_ids) != len(frame):
        raise RuntimeError(
            f"dev matrix row/id mismatch: {len(kept_pair_ids)} ids vs {len(frame)} rows"
        )

    out_frame = pd.DataFrame({PAIR_ID_COLUMN: kept_pair_ids})
    for col in feature_columns:
        out_frame[col] = frame[col].to_numpy(dtype=float) if col in frame.columns else 0.0
    out_frame[LABEL_COLUMN] = list(y)
    out_frame[POOL_COLUMN] = [str(g) for g in groups]

    ordered_columns = [PAIR_ID_COLUMN] + list(feature_columns) + [LABEL_COLUMN, POOL_COLUMN]
    out_frame = out_frame[ordered_columns]

    n_pos = int((out_frame[LABEL_COLUMN] == 1).sum())
    n_neg = int((out_frame[LABEL_COLUMN] == 0).sum())
    n_pools = int(out_frame[POOL_COLUMN].nunique())

    matrix_path = _write_matrix(out, DEV_MATRIX_NAME, out_frame)
    sidecar_path = _write_sidecar(
        out,
        DEV_MATRIX_NAME,
        feature_columns=feature_columns,
        row_count=len(out_frame),
        config=cfg,
        matrix_kind="dev",
        extra={"n_pos": n_pos, "n_neg": n_neg, "n_pools": n_pools},
    )
    return matrix_path, sidecar_path


# --------------------------------------------------------------------------- #
# (3) Loaders: fast round-trip with a soft staleness check
# --------------------------------------------------------------------------- #
def _load_matrix(
    out_dir: Path,
    name: str,
    *,
    config: Optional[PipelineConfig],
    check_staleness: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load a cached matrix + its sidecar; soft-warn (never raise) if the input hash drifted."""
    out = Path(out_dir)
    matrix_path = _matrix_path(out, name)
    sidecar_path = _sidecar_path(out, name)
    if not matrix_path.exists():
        raise FileNotFoundError(f"cached matrix not found: {matrix_path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"cache sidecar not found: {sidecar_path}")

    frame = pd.read_parquet(matrix_path)
    schema: Dict[str, object] = json.loads(sidecar_path.read_text(encoding="ascii"))

    schema["is_stale"] = False
    schema["staleness_reason"] = ""
    if check_staleness and config is not None:
        current = _content_hash(config)
        cached = str(schema.get("content_hash", ""))
        if cached and current != cached:
            schema["is_stale"] = True
            schema["staleness_reason"] = (
                "input content hash differs from the hash recorded when this matrix was built; "
                "the cache MAY be stale (rebuild to be certain)."
            )
    return frame, schema


def load_eval_matrix(
    out_dir: Optional[Path] = None,
    *,
    config: Optional[PipelineConfig] = None,
    check_staleness: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load the cached evaluation matrix -> ``(DataFrame, schema)``.

    The returned schema is the sidecar dict augmented with ``is_stale`` / ``staleness_reason`` (a
    SOFT check — a hash mismatch only annotates the schema, it never raises, so an experiment can
    still opt to use a slightly-stale cache knowingly).
    """
    cfg = config or (get_config() if check_staleness else None)
    out = Path(out_dir) if out_dir is not None else default_cache_dir(cfg or get_config())
    return _load_matrix(out, EVAL_MATRIX_NAME, config=cfg, check_staleness=check_staleness)


def load_dev_matrix(
    out_dir: Optional[Path] = None,
    *,
    config: Optional[PipelineConfig] = None,
    check_staleness: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load the cached development matrix -> ``(DataFrame, schema)`` (soft staleness check)."""
    cfg = config or (get_config() if check_staleness else None)
    out = Path(out_dir) if out_dir is not None else default_cache_dir(cfg or get_config())
    return _load_matrix(out, DEV_MATRIX_NAME, config=cfg, check_staleness=check_staleness)
