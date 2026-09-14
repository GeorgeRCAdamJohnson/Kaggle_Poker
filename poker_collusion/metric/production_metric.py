"""Production three-part competition metric for local CV and bootstrap resampling.

This is the metric the ``poker_collusion`` pipeline uses for local cross-validation.
It reproduces the OFFICIAL public competition metric **exactly** (identical numbers)
but is implemented efficiently (vectorized numpy/pandas) so it can score the full
112,540-pair evaluation set quickly and be reused across CV folds and bootstrap
resamples without re-paying Python-loop overhead on every call.

Contract (RESEARCH_DOSSIER.md section 1 "Metric Contract", CONFIRMED-CODE) and the
verbatim reference at :mod:`poker_collusion.metric.reference_public_metric` are the
ground truth. Any divergence from the reference is a release blocker (task 4.4 does
the exhaustive bit-for-bit reconciliation; this module is validated against the
reference in ``tests/test_production_metric.py``).

Confirmed behavior reproduced here
----------------------------------
* Combined score = ``0.70 * PairAP + 0.20 * EvidenceMAP@5 + 0.10 * BehaviorMAP``.
* **Pair AP**: Average Precision of the binary truth ranked by ``risk_score``
  descending. The official code sorts truth by ``pair_id`` ascending, then does a
  *stable* mergesort of ``-risk``; ties therefore break toward the smaller
  ``pair_id``. Normalized by the number of positives; ``0.0`` when there are none.
* **Evidence MAP@5**: over TRUE positives only; per-pair denominator is
  ``min(#relevant, 5)``; a missed positive contributes ``0`` but is still counted;
  ``NO_EVIDENCE``/blank/NaN cells are cleaned out; hand IDs compared as strings.
* **Behavior MAP**: mean over exactly the three :data:`TARGET_BEHAVIORS`; each family
  is a one-vs-rest AP scored with ``risk`` where ``predicted_behavior == family``
  else ``0``; a family with zero true positives contributes ``0.0`` and stays in the
  3-way denominator; ``other_coordination`` and ``none`` are excluded.
* **Submission validation** (raises :class:`ParticipantVisibleError`, mirroring the
  official ``ParticipantVisibleError`` semantics): required columns present;
  ``pair_id`` unique and its set exactly equal to the solution/evaluation set;
  ``risk_score`` numeric in ``[0, 1]`` with no clipping; ``predicted_behavior`` in
  :data:`ALLOWED_BEHAVIORS`; no repeated evidence hand within a pair.

Public API
----------
* :func:`score_components` -> :class:`ScoreComponents` with the three component
  scores plus the combined 0.70/0.20/0.10 summary (Req 10.5).
* :func:`score` -> the combined float (drop-in for the reference ``score()``).

Sources of truth:
  * ``.kiro/specs/poker-collusion-detection/RESEARCH_DOSSIER.md`` section 1.
  * :mod:`poker_collusion.metric.reference_public_metric`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the reference constants/exception so the two implementations can never
# drift on the allowed sets, the sentinel, the required schema, or the error type.
from poker_collusion.metric.reference_public_metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    REQUIRED_COLUMNS,
    TARGET_BEHAVIORS,
    ParticipantVisibleError,
)

__all__ = [
    "ScoreComponents",
    "score_components",
    "score",
    "ParticipantVisibleError",
    "ALLOWED_BEHAVIORS",
    "TARGET_BEHAVIORS",
    "NO_EVIDENCE",
]

# Combining weights, confirmed by the metric contract and reference code.
PAIR_AP_WEIGHT = 0.70
EVIDENCE_MAP_WEIGHT = 0.20
BEHAVIOR_MAP_WEIGHT = 0.10

# Evidence MAP is truncated at rank 5 (MAP@5).
EVIDENCE_CUTOFF = 5


@dataclass(frozen=True)
class ScoreComponents:
    """The three metric components plus the weighted combined summary (Req 10.5).

    Attributes:
        pair_ap: Pair Average Precision (weight 0.70).
        evidence_map: Evidence MAP@5 over true positives (weight 0.20).
        behavior_map: Behavior macro one-vs-rest AP over the 3 families (weight 0.10).
        combined: ``0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map``.
    """

    pair_ap: float
    evidence_map: float
    behavior_map: float
    combined: float

    def as_dict(self) -> dict[str, float]:
        """Return the components + combined summary as a plain dict."""
        return {
            "pair_ap": self.pair_ap,
            "evidence_map": self.evidence_map,
            "behavior_map": self.behavior_map,
            "combined": self.combined,
        }


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision with the official stable-mergesort tie-break.

    Mirrors ``reference_public_metric._average_precision`` exactly: a stable
    ``mergesort`` of ``-scores`` preserves the caller's row order among ties, so
    the caller is responsible for pre-ordering rows by ``pair_id`` ascending to get
    the deterministic ``pair_id`` tie-break. Normalized by the positive count; 0.0
    when there are no positives.
    """
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = y_true[order]
    true_positives = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((true_positives / ranks) * ranked) / positives)


def _clean_evidence_matrix(evidence: pd.DataFrame) -> np.ndarray:
    """Vectorized clean of the 5 evidence columns into an object matrix.

    Replaces NaN/blank/``NO_EVIDENCE`` cells with ``None`` and strips + stringifies
    the rest, matching ``reference_public_metric._clean_evidence`` cell-for-cell
    (order preserved). Returns an ``(n_rows, 5)`` object array where kept cells are
    ``str`` and dropped cells are ``None``.
    """
    values = evidence.to_numpy(dtype=object)
    isna = pd.isna(values)
    cleaned = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if isna[i, j]:
                cleaned[i, j] = None
                continue
            text = str(values[i, j]).strip()
            cleaned[i, j] = text if (text and text != NO_EVIDENCE) else None
    return cleaned


def _clean_row(row: np.ndarray) -> list[str]:
    """Ordered list of kept (non-None) evidence hand IDs for a single row."""
    return [cell for cell in row if cell is not None]


def _validate(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> None:
    """Run the full submission/solution validation, matching the reference.

    Raises :class:`ParticipantVisibleError` for participant-facing problems and
    :class:`ValueError` for private-solution schema problems, exactly as the
    official metric does.
    """
    if row_id_column_name != "pair_id":
        raise ParticipantVisibleError("The row ID column must be pair_id.")
    if not REQUIRED_COLUMNS.issubset(solution.columns):
        raise ValueError("The private solution has an invalid schema.")
    if not REQUIRED_COLUMNS.issubset(submission.columns):
        missing = sorted(REQUIRED_COLUMNS - set(submission.columns))
        raise ParticipantVisibleError(
            f"submission.csv is missing columns: {missing}"
        )
    if solution["pair_id"].duplicated().any():
        raise ValueError("The private solution contains duplicate pair IDs.")
    if submission["pair_id"].duplicated().any():
        raise ParticipantVisibleError("pair_id values must be unique.")

    solution_ids = set(solution["pair_id"].astype(str))
    submission_ids = set(submission["pair_id"].astype(str))
    if solution_ids != submission_ids:
        missing = len(solution_ids - submission_ids)
        extra = len(submission_ids - solution_ids)
        raise ParticipantVisibleError(
            f"pair_id coverage mismatch: {missing} missing and {extra} extra."
        )


def score_components(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "pair_id",
) -> ScoreComponents:
    """Compute the three metric components and the combined summary.

    Args:
        solution: Private solution frame with the ground-truth ``risk_score``
            (binary), ``predicted_behavior``, and the five evidence columns.
        submission: Participant submission with the same schema.
        row_id_column_name: Must be ``"pair_id"`` (kept for signature parity with
            the official metric).

    Returns:
        A :class:`ScoreComponents` with ``pair_ap``, ``evidence_map``,
        ``behavior_map`` and the weighted ``combined`` summary (Req 10.5).

    Raises:
        ParticipantVisibleError: on any participant-facing validation failure or a
            non-finite final score.
        ValueError: on a malformed private solution.
    """
    _validate(solution, submission, row_id_column_name)

    # Align both frames to the solution ordered by pair_id ascending. This ordering
    # is what produces the deterministic pair_id tie-break under the stable sort in
    # `_average_precision` (identical to the reference `.set_index().sort_index()`).
    truth = solution.set_index("pair_id").sort_index()
    predictions = submission.set_index("pair_id").loc[truth.index]

    risk = pd.to_numeric(predictions["risk_score"], errors="coerce")
    if risk.isna().any() or not risk.between(0, 1).all():
        raise ParticipantVisibleError(
            "risk_score must be numeric and between 0 and 1."
        )
    risk_values = risk.to_numpy(dtype=float)

    predicted_behavior = predictions["predicted_behavior"].astype(str)
    invalid = set(predicted_behavior) - ALLOWED_BEHAVIORS
    if invalid:
        raise ParticipantVisibleError(
            f"Invalid predicted_behavior values: {sorted(invalid)}"
        )

    # Clean evidence once (vectorized) and reuse for validation + Evidence MAP.
    submitted_clean = _clean_evidence_matrix(predictions.loc[:, list(EVIDENCE_COLUMNS)])
    for row in submitted_clean:
        kept = _clean_row(row)
        if len(kept) != len(set(kept)):
            raise ParticipantVisibleError(
                "Evidence hand IDs must not repeat within a pair."
            )

    # --- Ground-truth labels -------------------------------------------------
    y_true = pd.to_numeric(truth["risk_score"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Private risk_score values must be binary labels.")

    # --- Pair AP (weight 0.70) ----------------------------------------------
    pair_ap = _average_precision(y_true, risk_values)

    # --- Behavior MAP (weight 0.10) -----------------------------------------
    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    predicted_behavior_values = predicted_behavior.to_numpy()
    behavior_scores: list[float] = []
    for behavior in TARGET_BEHAVIORS:
        behavior_truth = (true_behavior == behavior).astype(int)
        if behavior_truth.sum() == 0:
            behavior_scores.append(0.0)
            continue
        behavior_risk = np.where(
            predicted_behavior_values == behavior,
            risk_values,
            0.0,
        )
        behavior_scores.append(_average_precision(behavior_truth, behavior_risk))
    behavior_map = float(np.mean(behavior_scores))

    # --- Evidence MAP@5 (weight 0.20) ---------------------------------------
    truth_clean = _clean_evidence_matrix(truth.loc[:, list(EVIDENCE_COLUMNS)])
    positive_positions = np.flatnonzero(y_true == 1)
    evidence_scores: list[float] = []
    for position in positive_positions:
        relevant = set(_clean_row(truth_clean[position]))
        submitted = _clean_row(submitted_clean[position])
        if not relevant:
            evidence_scores.append(0.0)
            continue
        hits = 0
        precision_sum = 0.0
        for rank, hand_id in enumerate(submitted[:EVIDENCE_CUTOFF], start=1):
            if hand_id in relevant:
                hits += 1
                precision_sum += hits / rank
        evidence_scores.append(precision_sum / min(len(relevant), EVIDENCE_CUTOFF))
    evidence_map = float(np.mean(evidence_scores)) if evidence_scores else 0.0

    combined = (
        PAIR_AP_WEIGHT * pair_ap
        + EVIDENCE_MAP_WEIGHT * evidence_map
        + BEHAVIOR_MAP_WEIGHT * behavior_map
    )
    if not np.isfinite(combined):
        raise ParticipantVisibleError("The metric produced a non-finite score.")

    return ScoreComponents(
        pair_ap=pair_ap,
        evidence_map=evidence_map,
        behavior_map=behavior_map,
        combined=float(combined),
    )


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "pair_id",
) -> float:
    """Combined three-part score (drop-in for the reference ``score``).

    Convenience wrapper around :func:`score_components` returning only the weighted
    ``combined`` float. Identical numerically to
    :func:`poker_collusion.metric.reference_public_metric.score`.
    """
    return score_components(solution, submission, row_id_column_name).combined
