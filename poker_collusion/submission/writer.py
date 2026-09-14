"""Submission_Writer: schema-matching submission with self-validation + atomic write.

Implements task 13.1 / Requirements 9.1-9.7. The writer emits EXACTLY the
``sample_submission.csv`` schema and column order to the configured submission
path (``/kaggle/working/submission.csv`` in the Kaggle notebook, a configurable
local path otherwise), one row per evaluation ``pair_id`` copied unchanged.

Design contract (mirrors ``reference_public_metric.py`` so a locally-valid file
is leaderboard-valid):

* Column order is fixed to :data:`SUBMISSION_COLUMNS` (Req 9.1, 9.2).
* One row per sample-submission ``pair_id`` (the evaluation pair set), pair ids
  copied unchanged and row order following the sample submission (Req 9.6).
* ``risk_score`` is a real number in ``[0, 1]`` (Req 9.3).
* ``predicted_behavior`` is in
  :data:`~poker_collusion.metric.reference_public_metric.ALLOWED_BEHAVIORS`
  (Req 9.4).
* Every one of the 5 evidence cells is non-empty (a hand id or the literal
  ``NO_EVIDENCE``) and no hand id repeats within a row (Req 9.7).

Coverage: any evaluation pair missing from ``predictions`` is filled with a safe
default row (risk ``0.0``, behavior ``"none"``, all evidence ``NO_EVIDENCE``) so
the emitted row set always matches the sample submission exactly.

Self-validation runs before the file is considered complete: identical headers,
identical row count, exact pair-id SET equality, non-empty evidence cells, no
repeated hand id per row, ``risk_score`` in ``[0, 1]``, allowed behaviors. Any
violation raises :class:`SubmissionValidationError` naming the first offending
pair/column.

The write is ATOMIC (Req 9.5): the frame is written to a temp file in the target
directory and then ``os.replace``-renamed onto the final path, so a partial file
is never observed even on crash.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from poker_collusion.config import NO_EVIDENCE, PipelineConfig, get_config
from poker_collusion.metric.reference_public_metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
)
from poker_collusion.types import PairPrediction

__all__ = [
    "SUBMISSION_COLUMNS",
    "SubmissionValidationError",
    "build_submission_frame",
    "validate_submission",
    "write_submission",
]

#: Exact submission schema and column ORDER (Req 9.1, 9.2). Matches
#: ``sample_submission.csv`` and ``reference_public_metric.REQUIRED_COLUMNS``.
SUBMISSION_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "risk_score",
    "predicted_behavior",
    *EVIDENCE_COLUMNS,
)

#: Safe default row values for an evaluation pair absent from ``predictions``.
_DEFAULT_RISK: float = 0.0
_DEFAULT_BEHAVIOR: str = "none"

# A prediction mapping value is (risk_score, predicted_behavior, evidence-list).
PredictionTuple = tuple  # (float, str, Sequence[str])
PredictionsInput = Union[
    Iterable[PairPrediction],
    Mapping[str, "PredictionTuple"],
]


class SubmissionValidationError(Exception):
    """Raised when the submission frame violates the sample-submission contract.

    The message names the offending pair/column so a failure is self-describing
    (Req 9.6, 9.7).
    """


# --------------------------------------------------------------------------- #
# Prediction normalization
# --------------------------------------------------------------------------- #
def _normalize_predictions(
    predictions: PredictionsInput,
) -> dict[str, tuple[float, str, list[str]]]:
    """Normalize accepted prediction inputs into ``pair_id -> (risk, behavior, evidence5)``.

    Accepts either an iterable of :class:`PairPrediction` or a mapping
    ``pair_id -> (risk_score, predicted_behavior, evidence-list)``. Evidence is
    normalized to a length-5 list, padding short lists with ``NO_EVIDENCE`` and
    truncating long ones; empty/blank cells become ``NO_EVIDENCE``.
    """
    normalized: dict[str, tuple[float, str, list[str]]] = {}

    if isinstance(predictions, Mapping):
        items = predictions.items()
        for pair_id, value in items:
            risk, behavior, evidence = value
            normalized[str(pair_id)] = (
                float(risk),
                str(behavior),
                _normalize_evidence(evidence),
            )
        return normalized

    for pred in predictions:
        if not isinstance(pred, PairPrediction):
            raise TypeError(
                "predictions must be PairPrediction objects or a "
                f"pair_id->(risk, behavior, evidence) mapping; got {type(pred)!r}."
            )
        normalized[str(pred.pair_id)] = (
            float(pred.risk_score),
            str(pred.predicted_behavior),
            _normalize_evidence(pred.evidence),
        )
    return normalized


def _normalize_evidence(evidence: Optional[Sequence[object]]) -> list[str]:
    """Coerce an evidence sequence to exactly 5 non-empty string cells."""
    cells: list[str] = []
    for value in list(evidence or [])[:5]:
        text = "" if value is None else str(value).strip()
        cells.append(text if text else NO_EVIDENCE)
    while len(cells) < 5:
        cells.append(NO_EVIDENCE)
    return cells


def _default_row(pair_id: str) -> dict[str, object]:
    """Return a safe default submission row for a missing pair (coverage)."""
    row: dict[str, object] = {
        "pair_id": pair_id,
        "risk_score": _DEFAULT_RISK,
        "predicted_behavior": _DEFAULT_BEHAVIOR,
    }
    for col in EVIDENCE_COLUMNS:
        row[col] = NO_EVIDENCE
    return row


# --------------------------------------------------------------------------- #
# Frame construction
# --------------------------------------------------------------------------- #
def build_submission_frame(
    predictions: PredictionsInput,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Build a validated submission frame in the exact sample-submission schema.

    One row per ``sample_submission`` ``pair_id`` in the sample's row order, pair
    ids copied unchanged (Req 9.1, 9.6). Any pair absent from ``predictions`` is
    filled with a safe default row so the row set always matches the sample
    submission exactly (coverage). The result is self-validated before return
    (Req 9.7); a violation raises :class:`SubmissionValidationError`.

    Args:
        predictions: Iterable of :class:`PairPrediction` or a mapping
            ``pair_id -> (risk_score, predicted_behavior, evidence-list)``.
        sample_submission: The template frame (must contain ``pair_id``).

    Returns:
        A deterministic ``DataFrame`` with columns :data:`SUBMISSION_COLUMNS`.
    """
    if "pair_id" not in sample_submission.columns:
        raise SubmissionValidationError(
            "sample_submission is missing the required 'pair_id' column."
        )

    pred_map = _normalize_predictions(predictions)

    # Row order follows the sample submission's pair_id order, unchanged (Req 9.6).
    sample_pair_ids = [str(pid) for pid in sample_submission["pair_id"].tolist()]

    rows: list[dict[str, object]] = []
    for pair_id in sample_pair_ids:
        if pair_id in pred_map:
            risk, behavior, evidence = pred_map[pair_id]
            row: dict[str, object] = {
                "pair_id": pair_id,
                "risk_score": risk,
                "predicted_behavior": behavior,
            }
            for col, cell in zip(EVIDENCE_COLUMNS, evidence):
                row[col] = cell
        else:
            row = _default_row(pair_id)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
    validate_submission(frame, sample_submission)
    return frame


# --------------------------------------------------------------------------- #
# Self-validation
# --------------------------------------------------------------------------- #
def validate_submission(
    frame: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Validate a submission frame against the sample submission (Req 9.7).

    Checks, raising :class:`SubmissionValidationError` naming the first offending
    pair/column on any mismatch:

    * identical headers in the exact order of :data:`SUBMISSION_COLUMNS`;
    * identical row COUNT to the sample submission;
    * exact pair-id SET equality (no missing/extra) with the sample submission;
    * ``risk_score`` numeric and in ``[0, 1]``;
    * ``predicted_behavior`` in :data:`ALLOWED_BEHAVIORS`;
    * every evidence cell non-empty;
    * no repeated hand id within a row (``NO_EVIDENCE`` cleaned out first).
    """
    # (1) Headers: exact set + exact order (Req 9.2).
    actual_columns = tuple(frame.columns)
    if actual_columns != SUBMISSION_COLUMNS:
        raise SubmissionValidationError(
            "submission header mismatch: expected columns in order "
            f"{list(SUBMISSION_COLUMNS)} but got {list(actual_columns)}."
        )

    # (2) Row count (Req 9.6).
    if len(frame) != len(sample_submission):
        raise SubmissionValidationError(
            f"submission row count mismatch: expected {len(sample_submission)} "
            f"rows (sample submission) but got {len(frame)}."
        )

    # (3) Pair-id SET equality (Req 9.1, 9.6).
    frame_ids = frame["pair_id"].astype(str)
    if frame_ids.duplicated().any():
        dup = frame_ids[frame_ids.duplicated()].iloc[0]
        raise SubmissionValidationError(
            f"submission contains a duplicate pair_id: {dup!r}."
        )
    sample_ids = set(sample_submission["pair_id"].astype(str))
    submission_ids = set(frame_ids)
    if submission_ids != sample_ids:
        missing = sorted(sample_ids - submission_ids)
        extra = sorted(submission_ids - sample_ids)
        raise SubmissionValidationError(
            "submission pair_id set mismatch with sample submission: "
            f"missing {missing[:5]} (+{max(len(missing) - 5, 0)} more), "
            f"extra {extra[:5]} (+{max(len(extra) - 5, 0)} more)."
        )

    # (4-7) Per-row field-domain checks (Req 9.3, 9.4, 9.7).
    for row in frame.itertuples(index=False):
        row_map = dict(zip(SUBMISSION_COLUMNS, row))
        pair_id = row_map["pair_id"]

        # risk_score numeric in [0, 1] (Req 9.3).
        risk = row_map["risk_score"]
        try:
            risk_value = float(risk)
        except (TypeError, ValueError) as exc:
            raise SubmissionValidationError(
                f"pair {pair_id!r}: risk_score {risk!r} is not numeric."
            ) from exc
        if not math.isfinite(risk_value) or not (0.0 <= risk_value <= 1.0):
            raise SubmissionValidationError(
                f"pair {pair_id!r}: risk_score {risk_value!r} is outside [0, 1]."
            )

        # predicted_behavior allowed (Req 9.4).
        behavior = str(row_map["predicted_behavior"])
        if behavior not in ALLOWED_BEHAVIORS:
            raise SubmissionValidationError(
                f"pair {pair_id!r}: predicted_behavior {behavior!r} is not one "
                f"of {sorted(ALLOWED_BEHAVIORS)}."
            )

        # Evidence cells non-empty + no repeats within the row (Req 9.7).
        seen: set[str] = set()
        for col in EVIDENCE_COLUMNS:
            cell = row_map[col]
            text = "" if cell is None else str(cell).strip()
            if not text:
                raise SubmissionValidationError(
                    f"pair {pair_id!r}: evidence column {col!r} is empty; every "
                    "evidence cell must be a hand id or NO_EVIDENCE."
                )
            if text == NO_EVIDENCE:
                continue
            if text in seen:
                raise SubmissionValidationError(
                    f"pair {pair_id!r}: evidence hand id {text!r} repeats within "
                    "the row."
                )
            seen.add(text)


# --------------------------------------------------------------------------- #
# Atomic write
# --------------------------------------------------------------------------- #
def _resolve_path(config: Optional[PipelineConfig], path: Optional[Union[str, Path]]) -> Path:
    """Resolve the submission output path.

    An explicit ``path`` wins; otherwise ``config.submission_path`` is used
    (``/kaggle/working/submission.csv`` in Kaggle, else the configured local
    path) (Req 9.5).
    """
    if path is not None:
        return Path(path)
    cfg = config if config is not None else get_config()
    return cfg.submission_path


def write_submission(
    frame_or_predictions: Union[pd.DataFrame, PredictionsInput],
    *,
    sample_submission: Optional[pd.DataFrame] = None,
    config: Optional[PipelineConfig] = None,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """Validate then ATOMICALLY write the submission to the resolved path.

    Accepts either a pre-built submission ``DataFrame`` or raw ``predictions``
    (iterable of :class:`PairPrediction` or a ``pair_id -> (risk, behavior,
    evidence)`` mapping). When given raw predictions a ``sample_submission`` is
    required to fix the row set and order.

    The write is atomic (Req 9.5): the frame is written to a temp file in the
    destination directory and ``os.replace``-renamed onto the final path, so a
    partial ``submission.csv`` is never observed.

    Args:
        frame_or_predictions: A validated/validatable frame, or raw predictions.
        sample_submission: Template frame (required when passing raw predictions;
            also used to re-validate a passed-in frame when provided).
        config: Pipeline configuration; defaults to :func:`get_config`.
        path: Explicit output path override; wins over ``config.submission_path``.

    Returns:
        The :class:`~pathlib.Path` the submission was written to.
    """
    if isinstance(frame_or_predictions, pd.DataFrame):
        frame = frame_or_predictions
        if sample_submission is not None:
            validate_submission(frame, sample_submission)
    else:
        if sample_submission is None:
            raise SubmissionValidationError(
                "write_submission requires a sample_submission when given raw "
                "predictions (to fix the row set and order)."
            )
        frame = build_submission_frame(frame_or_predictions, sample_submission)

    target = _resolve_path(config, path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: temp file in the destination dir, then os.replace rename.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".submission_", suffix=".csv.tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)  # atomic rename on the same filesystem
    except BaseException:
        # Never leave the temp artifact behind on failure.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    return target
