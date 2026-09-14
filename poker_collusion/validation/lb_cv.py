"""LB-vs-CV correlation diagnostic and the submission-selection rule.

This module operationalizes Design Principle 1 — *trust local CV, distrust the noisy
public leaderboard* (Requirement 12.4; RESEARCH_DOSSIER §1.4). It builds on the CV
harness (task 5.1, :mod:`poker_collusion.validation.cv`) and the bootstrap noise band
(task 5.2, :mod:`poker_collusion.validation.bootstrap`).

Why the public LB is not trusted
---------------------------------
The public/private split is only ~30/70 and family-stratified, so the public LB is
scored on a small, uneven slice of pairs and is *noisy* (dossier §1.4). Optimizing
against it — a mistake that cost real placement on prior competitions — overfits the
public slice. Every decision here is therefore gated on local CV, and the public LB is
recorded only as a *diagnostic* signal: if it correlates well with CV we gain a little
confidence, and if it disagrees we explicitly trust CV.

Two tools
---------
1. :func:`lb_cv_correlation` / :class:`LbCvDiagnostic` — given an experiment log (each
   experiment's recorded local-CV combined score, ideally its three components, and the
   observed public-LB score), compute the Spearman rank correlation and Pearson
   correlation between CV and LB, and raise a ``warn`` flag when the correlation is weak
   or negative (the public LB is too noisy to trust — lean on CV alone).

2. :func:`select_submission` — the submission-selection rule. Among candidate
   submissions, pick the one that maximizes the combined three-component summary on
   local CV, but only treat a challenger as better than the incumbent when it improves
   **beyond the bootstrap noise band** (reusing
   :func:`poker_collusion.validation.bootstrap.is_improvement_beyond_band`). Within-band
   ties break toward the more conservative/robust choice — the narrower CV band, then
   the incumbent (fewest moving parts). The public LB never enters the decision.

Dependencies: numpy only for the statistics (a tiny average-rank helper gives Spearman
without scipy); pandas is accepted for the experiment log but not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from poker_collusion.metric.production_metric import score_components
from poker_collusion.validation.bootstrap import (
    ConfidenceInterval,
    bootstrap_components,
    confidence_interval,
    is_improvement_beyond_band,
)

__all__ = [
    "WEAK_CORRELATION_THRESHOLD",
    "LbCvDiagnostic",
    "SelectionResult",
    "SubmissionCandidate",
    "spearman_corr",
    "pearson_corr",
    "lb_cv_correlation",
    "select_submission",
]

# Below this Spearman rank correlation the public LB is judged too noisy to trust: the
# diagnostic raises its warning flag and callers should lean on local CV alone. A
# moderate 0.5 threshold means "the LB does not even track CV monotonically well".
WEAK_CORRELATION_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Pure-numpy correlation helpers                                              #
# --------------------------------------------------------------------------- #
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average (fractional) ranks of ``values``; ties share the mean rank.

    Matches the tie handling used by Spearman's rho (``scipy.stats.rankdata`` with the
    default ``"average"`` method), implemented with numpy only.
    """
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")  # stable
    ranks = np.empty(len(arr), dtype=float)
    sorted_arr = arr[order]
    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        # positions i..j (inclusive) are tied; assign them the average 1-based rank.
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson product-moment correlation of two equal-length sequences.

    Returns ``0.0`` when either input has zero variance (a constant series has no
    linear relationship to define) — this keeps the diagnostic well-behaved instead of
    emitting NaN.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("x and y must have the same length.")
    if a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson correlation of the average ranks."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("x and y must have the same length.")
    if a.size < 2:
        return 0.0
    return pearson_corr(_average_ranks(a), _average_ranks(b))


# --------------------------------------------------------------------------- #
# LB-vs-CV correlation diagnostic                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LbCvDiagnostic:
    """Report on how well the public LB tracks local CV across logged experiments.

    Attributes:
        n_experiments: Number of experiments in the log.
        spearman: Spearman rank correlation between CV-combined and LB.
        pearson: Pearson correlation between CV-combined and LB.
        warn: ``True`` when the LB is too noisy to trust (Spearman below the weak
            threshold, or too few experiments to judge) — trust CV alone.
        message: A short human-readable summary of the finding.
    """

    n_experiments: int
    spearman: float
    pearson: float
    warn: bool
    message: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_experiments": self.n_experiments,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "warn": self.warn,
            "message": self.message,
        }


# Column names expected in a DataFrame experiment log.
CV_COMBINED_COLUMN = "cv_combined"
LB_SCORE_COLUMN = "lb_score"


def _extract_cv_lb(
    experiments: Union[pd.DataFrame, Sequence[Mapping[str, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pull the CV-combined and LB arrays from a DataFrame or a sequence of dicts."""
    if isinstance(experiments, pd.DataFrame):
        for col in (CV_COMBINED_COLUMN, LB_SCORE_COLUMN):
            if col not in experiments.columns:
                raise ValueError(
                    f"experiment log is missing required column {col!r}."
                )
        cv = experiments[CV_COMBINED_COLUMN].to_numpy(dtype=float)
        lb = experiments[LB_SCORE_COLUMN].to_numpy(dtype=float)
        return cv, lb

    cv_list: List[float] = []
    lb_list: List[float] = []
    for row in experiments:
        if CV_COMBINED_COLUMN not in row or LB_SCORE_COLUMN not in row:
            raise ValueError(
                f"each experiment must define {CV_COMBINED_COLUMN!r} and "
                f"{LB_SCORE_COLUMN!r}."
            )
        cv_list.append(float(row[CV_COMBINED_COLUMN]))
        lb_list.append(float(row[LB_SCORE_COLUMN]))
    return np.asarray(cv_list, dtype=float), np.asarray(lb_list, dtype=float)


def lb_cv_correlation(
    experiments: Union[pd.DataFrame, Sequence[Mapping[str, float]]],
    *,
    threshold: float = WEAK_CORRELATION_THRESHOLD,
) -> LbCvDiagnostic:
    """Diagnose how well the public LB tracks local CV across logged experiments.

    Each experiment records its local-CV combined score (``cv_combined``) and the
    observed public-LB score (``lb_score``); the three CV components
    (``cv_pair_ap``, ``cv_evidence_map``, ``cv_behavior_map``) may also be present but
    are not required by the correlation. Accepts either a pandas ``DataFrame`` or a
    sequence of mappings.

    The diagnostic reports the Spearman rank correlation (the primary signal — we care
    whether the LB ranks experiments the same way CV does) and the Pearson correlation.
    It raises ``warn=True`` when there are fewer than two experiments (nothing to
    correlate) or the Spearman correlation is below ``threshold`` — either way the
    public LB is too noisy to trust and decisions should rest on local CV alone.

    Args:
        experiments: Experiment log (DataFrame with ``cv_combined``/``lb_score``
            columns, or a sequence of dicts with those keys).
        threshold: Weak-correlation cutoff on Spearman (default
            :data:`WEAK_CORRELATION_THRESHOLD`).

    Returns:
        An :class:`LbCvDiagnostic`.
    """
    cv, lb = _extract_cv_lb(experiments)
    n = int(cv.size)
    if n < 2:
        return LbCvDiagnostic(
            n_experiments=n,
            spearman=0.0,
            pearson=0.0,
            warn=True,
            message=(
                f"Only {n} experiment(s) logged — too few to judge LB↔CV agreement; "
                "trust local CV."
            ),
        )

    rho = spearman_corr(cv, lb)
    r = pearson_corr(cv, lb)
    warn = rho < threshold
    if warn:
        message = (
            f"Public LB tracks CV weakly (Spearman={rho:.3f} < {threshold:.2f}) over "
            f"{n} experiments — LB is too noisy to trust; gate decisions on local CV."
        )
    else:
        message = (
            f"Public LB tracks CV (Spearman={rho:.3f}, Pearson={r:.3f}) over {n} "
            "experiments; decisions still gated on local CV."
        )
    return LbCvDiagnostic(
        n_experiments=n, spearman=rho, pearson=r, warn=warn, message=message
    )


# --------------------------------------------------------------------------- #
# Submission-selection rule                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubmissionCandidate:
    """One candidate submission scored against a shared CV solution frame.

    A candidate carries its predicted submission frame; its CV combined score and
    bootstrap band are computed on demand from the shared solution. The ``lb_score`` is
    recorded for logging/diagnostics only and is deliberately never used to decide.

    Attributes:
        candidate_id: Stable identifier for the candidate.
        submission: The candidate's submission frame (production-metric schema).
        lb_score: Optional observed public-LB score (diagnostic only; not used to
            select).
    """

    candidate_id: str
    submission: pd.DataFrame
    lb_score: Optional[float] = None


@dataclass(frozen=True)
class SelectionResult:
    """The chosen submission plus the rationale.

    Attributes:
        chosen_id: The selected candidate id.
        rationale: Human-readable explanation of why it was chosen.
        cv_combined: Chosen candidate's CV combined point estimate.
        cv_band: Chosen candidate's combined-score bootstrap CI (the noise band).
        considered: Ordered ``[(candidate_id, cv_combined, band_width)]`` for every
            candidate, best CV first (for logging).
    """

    chosen_id: str
    rationale: str
    cv_combined: float
    cv_band: ConfidenceInterval
    considered: List[tuple]


def select_submission(
    solution: pd.DataFrame,
    candidates: Sequence[SubmissionCandidate],
    *,
    incumbent_id: Optional[str] = None,
    key: str = "combined",
    n_boot: int = 500,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    rule: str = "paired",
) -> SelectionResult:
    """Select which submission to submit, gated on local CV (never the public LB).

    The rule (design line: "Submission-selection rule"; Req 12.4):

    1. Score every candidate's combined three-component summary on the shared local-CV
       ``solution`` and rank them best-first.
    2. Start from the incumbent (``incumbent_id`` if given, else the best-CV candidate)
       and walk the remaining candidates in best-CV order. A challenger only *replaces*
       the current best when it improves **beyond the bootstrap noise band**
       (:func:`is_improvement_beyond_band`, paired bootstrap of the difference by
       default). Improvements within the band are treated as ties — a noisy wiggle, not
       a real gain.
    3. **Tie-break (within band): prefer the more conservative / robust choice.** Keep
       whichever of the two has the *narrower* combined-score CI (lower variance); if the
       bands are equally wide, keep the incumbent (fewest moving parts). The public LB is
       never consulted.

    Args:
        solution: Shared CV ground-truth frame (production-metric schema).
        candidates: The candidate submissions.
        incumbent_id: The currently-shipping candidate to defend; defaults to the
            best-CV candidate.
        key: Component the noise band is tested on (default ``"combined"``).
        n_boot: Bootstrap replicates for the bands / paired test.
        seed: RNG seed; defaults to ``config.Seeds.bootstrap`` inside the bootstrap.
        alpha: Significance level for the band (default 0.05 → 95%).
        rule: Improvement-beyond-band rule passed through to
            :func:`is_improvement_beyond_band` (default ``"paired"``).

    Returns:
        A :class:`SelectionResult` with the chosen id, the rationale, and per-candidate
        CV summaries.

    Raises:
        ValueError: if ``candidates`` is empty or ``incumbent_id`` is unknown.
    """
    if len(candidates) == 0:
        raise ValueError("select_submission requires at least one candidate.")

    by_id = {c.candidate_id: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate_id values must be unique.")
    if incumbent_id is not None and incumbent_id not in by_id:
        raise ValueError(f"incumbent_id {incumbent_id!r} is not among the candidates.")

    # Bootstrap band per candidate and the true CV point estimate for the tested
    # component (all on the shared solution). The band gives the noise width used for
    # the conservative tie-break; the point estimate drives best-first ranking.
    bands: Dict[str, ConfidenceInterval] = {}
    point: Dict[str, float] = {}
    for c in candidates:
        boot = bootstrap_components(solution, c.submission, n_boot=n_boot, seed=seed)
        bands[c.candidate_id] = confidence_interval(boot.samples[key], alpha=alpha)
        point[c.candidate_id] = score_components(solution, c.submission).as_dict()[key]

    # Rank best-CV first; ties in CV break toward the narrower band, then id for
    # determinism.
    ranked_ids = sorted(
        by_id.keys(),
        key=lambda cid: (
            -point[cid],
            bands[cid].high - bands[cid].low,
            str(cid),
        ),
    )

    best_id = incumbent_id if incumbent_id is not None else ranked_ids[0]
    incumbent_was = best_id
    replaced: List[str] = []

    for cid in ranked_ids:
        if cid == best_id:
            continue
        challenger = by_id[cid]
        current = by_id[best_id]
        beyond = is_improvement_beyond_band(
            solution,
            baseline_submission=current.submission,
            candidate_submission=challenger.submission,
            key=key,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            rule=rule,
        )
        if beyond:
            best_id = cid
            replaced.append(cid)
        else:
            # Within-band tie: keep the more conservative/robust choice.
            cur_w = bands[best_id].high - bands[best_id].low
            chl_w = bands[cid].high - bands[cid].low
            # Prefer the challenger only if it is strictly higher on CV AND has a
            # strictly narrower band; otherwise keep the current best (incumbent /
            # fewest moving parts).
            if point[cid] > point[best_id] and chl_w < cur_w:
                best_id = cid
                replaced.append(cid)

    chosen_band = bands[best_id]
    considered = [
        (cid, point[cid], bands[cid].high - bands[cid].low) for cid in ranked_ids
    ]

    if best_id == incumbent_was and replaced == []:
        if incumbent_id is not None:
            rationale = (
                f"Kept incumbent {best_id!r}: no challenger improved beyond the "
                f"bootstrap noise band on local CV (combined={point[best_id]:.4f}, "
                f"95% band width={chosen_band.high - chosen_band.low:.4f}). "
                "Public LB not used."
            )
        else:
            rationale = (
                f"Chose {best_id!r}: best local-CV combined score "
                f"({point[best_id]:.4f}) and no other candidate beat it beyond the "
                "bootstrap noise band. Public LB not used."
            )
    else:
        rationale = (
            f"Selected {best_id!r}: it improves the local-CV combined score to "
            f"{point[best_id]:.4f} beyond the bootstrap noise band vs. the incumbent "
            f"{incumbent_was!r}. Public LB not used."
        )

    return SelectionResult(
        chosen_id=best_id,
        rationale=rationale,
        cv_combined=point[best_id],
        cv_band=chosen_band,
        considered=considered,
    )
