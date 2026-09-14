"""Table-disjoint PU-stress AP holdout scorer (design: ``holdout/scorer.py``).

This is the internal judge of the layered-tuning harness: the **PU-stress AP** a
candidate scores on the pair-disjoint, table-disjoint 40% holdout. It is a THIN
WRAPPER — it REUSES, and never reimplements, two substrate pieces of the existing
:mod:`poker_collusion` package:

* the **split** — :class:`poker_collusion.validation.cv.CVHarness` and
  :func:`poker_collusion.validation.cv.build_fold_solution`. The harness folds
  are group-by-``table_id`` (a pool = one ``table_id``: 30 disjoint players, so a
  pair belongs to exactly one pool). Folding at the pool level makes every
  validation fold **table-disjoint and pair-disjoint** from its train set — the
  eval-mimic holdout the design requires. A five-fold split yields ~20% per fold;
  a ``holdout_fraction`` of ~0.40 is realised with the matching ``n_folds`` (the
  design's "40% table-disjoint" split), and the pooled-across-held-out-folds Pair
  AP is the PU-stress AP.
* the **metric** — :func:`poker_collusion.metric.production_metric.score_components`
  (numerically identical to the verbatim official
  :mod:`poker_collusion.metric.reference_public_metric`). The **Pair AP**
  component (the 0.70-weight ``pair_ap``) measured on the PU-correct holdout
  solution IS the PU-stress AP the harness ranks candidates by. We do NOT
  re-derive AP here — we read ``score_components(...).pair_ap``.

"PU-stress" = the solution frames the split hands us are PU-correct
(:func:`build_fold_solution`): only confirmed positives / confirmed negatives are
materialised, unknown pairs are never fabricated as negatives. Scoring the Pair
AP over that PU-correct, table-disjoint solution is exactly the stressed holdout
AP recorded as each LB_Anchor's ``holdout_ap``.

Interface (design "Holdout scorer"):

* :func:`holdout_pair_ap` — the thin scorer. Given a candidate ranking
  (a :class:`~poker_collusion.tuning_harness.models.RankVector` of per-``pair_id``
  risk) OR a per-fold ``predict_fn`` callback, run the reused split + metric and
  return the pooled PU-stress AP in ``[0, 1]``.

Nothing in this module owns data, recomputes features, or re-derives the split or
the metric (Req 3.1 reuse mandate; accountability contract Rule 6 "reuse, do not
rebuild").

Requirements: 3.1 (the holdout is the reused table-disjoint split + reference
metric), 5.4 (the measured holdout AP the tuner/report keep distinct from the
projected LB).
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from poker_collusion.metric.production_metric import score_components
from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.tuning_harness.models import RankVector
from poker_collusion.validation.cv import CVHarness, Fold, build_fold_solution

__all__ = [
    "HoldoutScore",
    "holdout_pair_ap",
    "ranking_predict_fn",
]

#: Neutral fill for the fields the Pair-AP holdout does not stress. Pair AP is a
#: function of ``risk_score`` ranking only; ``predicted_behavior``/evidence do not
#: enter it, so a candidate ranking scores its PU-stress AP with these neutral
#: values (the metric still validates them, so they must be in-domain).
_NEUTRAL_BEHAVIOR = "none"
_NO_EVIDENCE = "NO_EVIDENCE"


class HoldoutScore(float):
    """The pooled PU-stress AP, a plain float carrying its provenance.

    Subclasses ``float`` so it drops into arithmetic/comparison unchanged while
    exposing how many held-out folds and confirmed pairs it was pooled over — the
    honest-provenance the harness records alongside every internal (non-external)
    number (accountability contract Rule 9).
    """

    n_folds: int
    n_pairs: int
    n_positive_pairs: int

    def __new__(
        cls,
        value: float,
        *,
        n_folds: int = 0,
        n_pairs: int = 0,
        n_positive_pairs: int = 0,
    ) -> "HoldoutScore":
        obj = super().__new__(cls, value)
        obj.n_folds = int(n_folds)
        obj.n_pairs = int(n_pairs)
        obj.n_positive_pairs = int(n_positive_pairs)
        return obj


def ranking_predict_fn(
    ranking: RankVector,
    *,
    default_risk: float = 0.0,
) -> Callable[[Fold, pd.DataFrame], pd.DataFrame]:
    """Adapt a candidate :class:`RankVector` into a per-fold ``predict_fn``.

    The returned callback is exactly the shape
    :meth:`poker_collusion.validation.cv.CVHarness.run` expects: given a fold and
    its PU-correct solution frame, it emits a metric-valid submission over EXACTLY
    the solution's ``pair_id`` set, filling ``risk_score`` from the candidate
    ranking (a pair the ranking does not cover falls back to ``default_risk``) and
    leaving the non-Pair-AP fields neutral (Pair AP keys off ``risk_score`` order
    only).

    The ranking's risk is read from ``RankVector.scores`` when present; otherwise a
    monotone rank-derived risk in ``(0, 1]`` is synthesised from the ordering
    (index 0 = highest risk) so the ordering alone is enough to score Pair AP.
    """

    risk_by_pair = _risk_map(ranking)

    def predict_fn(fold: Fold, solution: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for pair_id in solution["pair_id"].astype(str):
            risk = float(risk_by_pair.get(pair_id, default_risk))
            # Clamp into the metric's accepted [0, 1] domain (no clipping surprises).
            risk = min(max(risk, 0.0), 1.0)
            rows.append(
                {
                    "pair_id": pair_id,
                    "risk_score": risk,
                    "predicted_behavior": _NEUTRAL_BEHAVIOR,
                    "evidence_hand_1": _NO_EVIDENCE,
                    "evidence_hand_2": _NO_EVIDENCE,
                    "evidence_hand_3": _NO_EVIDENCE,
                    "evidence_hand_4": _NO_EVIDENCE,
                    "evidence_hand_5": _NO_EVIDENCE,
                }
            )
        return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))

    return predict_fn


def _risk_map(ranking: RankVector) -> dict[str, float]:
    """Build ``{pair_id: risk}`` from a RankVector.

    Uses the explicit ``scores`` when provided; otherwise derives a strictly
    decreasing risk from the ordering (highest-risk-first) so a bare ordering is
    enough to reproduce the same Pair AP ranking.
    """
    n = len(ranking.pair_ids)
    if ranking.scores:
        return {pid: float(s) for pid, s in zip(ranking.pair_ids, ranking.scores)}
    if n == 0:
        return {}
    # Rank-derived risk in (0, 1]: index 0 (highest risk) -> 1.0, last -> 1/n.
    return {pid: (n - i) / n for i, pid in enumerate(ranking.pair_ids)}


def holdout_pair_ap(
    harness: CVHarness,
    candidate: RankVector | Callable[[Fold, pd.DataFrame], pd.DataFrame],
    *,
    default_risk: float = 0.0,
) -> HoldoutScore:
    """Return a candidate's PU-stress AP on the reused table-disjoint holdout.

    This is the thin wrapper the design's "Holdout scorer" specifies. It:

    1. REUSES the split — iterates ``harness.folds()`` (group-by-``table_id``,
       hence table-disjoint and pair-disjoint) and builds each fold's PU-correct
       solution via :func:`build_fold_solution` (unknown pairs never fabricated as
       negatives — the "PU" in PU-stress AP).
    2. produces each fold's submission from the ``candidate`` — either a
       :class:`RankVector` (adapted by :func:`ranking_predict_fn`) or a caller
       ``predict_fn`` passed straight through.
    3. pools the per-fold (solution, submission) frames across the held-out folds
       (pairs are globally unique across folds) and REUSES
       :func:`score_components` to read the pooled **Pair AP** — the PU-stress AP.

    No split logic and no AP arithmetic live here; both are the reused substrate.

    Args:
        harness: A constructed :class:`CVHarness` (its folds ARE the table-disjoint
            holdout split; the caller chooses ``n_folds`` to size the held-out
            fraction, e.g. the design's ~40%).
        candidate: The candidate ranking (:class:`RankVector`) or a per-fold
            ``predict_fn`` with the :meth:`CVHarness.run` signature.
        default_risk: Risk assigned to a solution pair the ranking omits (only used
            when ``candidate`` is a RankVector).

    Returns:
        A :class:`HoldoutScore` (a float in ``[0, 1]``) carrying the pooled Pair AP
        and its held-out-fold / pair provenance. ``0.0`` when no held-out fold has a
        confirmed pair (the metric defines Pair AP as 0 with no positives).
    """
    predict_fn = (
        ranking_predict_fn(candidate, default_risk=default_risk)
        if isinstance(candidate, RankVector)
        else candidate
    )

    solutions: list[pd.DataFrame] = []
    submissions: list[pd.DataFrame] = []
    n_folds = 0
    for fold in harness.folds():
        solution = build_fold_solution(
            harness.labels, fold.validation_pairs, evidence=harness.evidence
        )
        if len(solution) == 0:
            continue
        submission = predict_fn(fold, solution)
        solutions.append(solution)
        submissions.append(submission)
        n_folds += 1

    if not solutions:
        return HoldoutScore(0.0, n_folds=0, n_pairs=0, n_positive_pairs=0)

    pooled_solution = pd.concat(solutions, ignore_index=True)
    pooled_submission = pd.concat(submissions, ignore_index=True)

    components = score_components(pooled_solution, pooled_submission)
    n_positive = int(
        pd.to_numeric(pooled_solution["risk_score"], errors="coerce")
        .fillna(0)
        .astype(int)
        .sum()
    )
    return HoldoutScore(
        float(components.pair_ap),
        n_folds=n_folds,
        n_pairs=int(len(pooled_solution)),
        n_positive_pairs=n_positive,
    )
