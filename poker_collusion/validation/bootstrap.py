"""Bootstrap confidence intervals over pairs and the noise-band acceptance rule.

This module implements the variance band that turns "looks better" into "is better
beyond noise" — the acceptance gate every phase and every adopted exploit must clear
(Design Principle 1, "Validation Discipline"; Requirements 12.2 and 12.4). It builds
on the CV harness (task 5.1, :mod:`poker_collusion.validation.cv`) and the production
metric (:func:`poker_collusion.metric.production_metric.score_components`).

Why resample **pairs**
----------------------
The three metric components all aggregate over the per-pair set:

* **Pair AP** ranks the binary truth over the full pair set.
* **Behavior MAP** is a one-vs-rest AP over the same pair set per family.
* **Evidence MAP@5** averages over the *true-positive* pairs only.

The pair is therefore the natural, exchangeable resampling unit: resampling pairs
with replacement approximates the sampling distribution of each component under the
same generative unit the metric aggregates over. (Contrast with resampling hands or
actions, which would not preserve the estimator's per-pair structure.)

Resampling method (documented, defensible)
-------------------------------------------
For each of ``B`` bootstrap replicates we draw ``n`` pair *positions* with
replacement (``n`` = number of pairs in the solution) using a single seeded RNG
(``config.Seeds.bootstrap``). A resample is a multiset of the original pairs.

The metric requires ``pair_id`` to be **unique** and the submission/solution
``pair_id`` sets to match exactly, so we cannot feed a multiset of the original ids
directly. We instead **rebuild a valid solution/submission for the resample by
issuing fresh, unique ``pair_id``s** (``b0``, ``b1``, …) to the drawn positions while
copying every scored field (``risk_score``, ``predicted_behavior``, the five evidence
columns) from the corresponding original row. This is exactly the standard bootstrap
over the pair set: the estimator ``score_components`` is recomputed on a sample of
``n`` pairs drawn with replacement, and renaming the row id is a no-op for all three
components because none of them depend on the *value* of ``pair_id`` beyond its use
as a unique key and the ascending tie-break. To keep the tie-break deterministic and
faithful, the fresh ids are zero-padded and assigned in the drawn order, and both the
resampled solution and submission are relabelled identically so they still align.

This preserves each estimator's meaning:

* Pair AP / Behavior MAP re-rank over the resampled ``n`` pairs (duplicates simply
  appear multiple times, exactly as in a standard AP bootstrap).
* Evidence MAP@5 re-averages over whichever resampled pairs are true positives.

Public API
----------
* :func:`bootstrap_components` — draw ``B`` seeded pair-resamples and return the
  bootstrap samples of each component (+ combined) as a :class:`BootstrapSamples`.
* :func:`confidence_interval` / :meth:`BootstrapSamples.intervals` — percentile CIs
  (default 95%) plus mean/std for every component.
* :func:`paired_bootstrap_diff` — the paired bootstrap of the *difference* between a
  candidate and a baseline on a shared resample (higher power than comparing two
  independent CIs).
* :func:`is_improvement_beyond_band` — the acceptance rule used as the phase gate.

The exact improvement-beyond-band rule
---------------------------------------
:func:`is_improvement_beyond_band` implements a **paired bootstrap of the difference**
by default (the preferred, higher-power test): on each replicate we draw one pair
resample and score *both* the baseline and the candidate on that same resample, then
build the distribution of ``candidate - baseline``. A candidate is an improvement
beyond the noise band iff the two-sided ``(1 - alpha)`` percentile CI of that paired
difference lies strictly **above zero** (its lower bound ``> 0``). Pairing cancels the
shared pair-sampling variance, so a real improvement that is small relative to the
absolute component variance is still detected — this is why it is preferred over the
weaker "candidate mean above the baseline CI upper bound" / "non-overlapping CIs"
rules (both of which are also offered via ``rule=`` for reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from poker_collusion.config import get_config
from poker_collusion.metric.production_metric import ScoreComponents, score_components
from poker_collusion.metric.reference_public_metric import EVIDENCE_COLUMNS

__all__ = [
    "COMPONENT_KEYS",
    "BootstrapSamples",
    "ConfidenceInterval",
    "bootstrap_components",
    "confidence_interval",
    "paired_bootstrap_diff",
    "is_improvement_beyond_band",
]

# The four reported quantities: the three components plus the weighted combined.
COMPONENT_KEYS = ("pair_ap", "evidence_map", "behavior_map", "combined")

# Columns copied verbatim from the original row into a resampled row.
_SCORED_COLUMNS = ("risk_score", "predicted_behavior", *EVIDENCE_COLUMNS)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A percentile CI plus point-estimate summaries for one quantity."""

    mean: float
    std: float
    low: float
    high: float
    alpha: float

    def contains(self, value: float) -> bool:
        """True if ``value`` lies within ``[low, high]`` (inclusive)."""
        return self.low <= value <= self.high

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "low": self.low,
            "high": self.high,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class BootstrapSamples:
    """Bootstrap replicate arrays for each component (+ combined).

    Attributes:
        samples: ``{component_key: np.ndarray of length B}`` bootstrap replicates.
        n_pairs: Number of pairs in the resampled solution (the resample size).
        n_boot: Number of bootstrap replicates ``B``.
        seed: The seed used to draw the resamples (for reproducibility).
    """

    samples: Dict[str, np.ndarray]
    n_pairs: int
    n_boot: int
    seed: int

    def interval(self, key: str = "combined", *, alpha: float = 0.05) -> ConfidenceInterval:
        """Percentile CI + mean/std for a single component."""
        return confidence_interval(self.samples[key], alpha=alpha)

    def intervals(self, *, alpha: float = 0.05) -> Dict[str, ConfidenceInterval]:
        """Percentile CIs + mean/std for every component."""
        return {k: confidence_interval(v, alpha=alpha) for k, v in self.samples.items()}


def _resample_frames(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    positions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild a valid (solution, submission) pair for the drawn pair positions.

    ``positions`` is an array of row indices (with replacement) into the aligned
    solution/submission. Fresh unique ``pair_id``s are minted in draw order and
    applied identically to both frames so they stay aligned and satisfy the metric's
    uniqueness + set-equality validation. All scored fields are copied verbatim.
    """
    sol = solution.reset_index(drop=True)
    sub = submission.set_index("pair_id").loc[solution["pair_id"].to_numpy()].reset_index(drop=True)

    new_ids = [f"b{i}" for i in range(len(positions))]
    sol_rows = sol.iloc[positions].reset_index(drop=True).copy()
    sub_rows = sub.iloc[positions].reset_index(drop=True).copy()
    sol_rows["pair_id"] = new_ids
    sub_rows["pair_id"] = new_ids
    return sol_rows, sub_rows


def bootstrap_components(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    *,
    n_boot: int = 500,
    seed: Optional[int] = None,
    row_id_column_name: str = "pair_id",
) -> BootstrapSamples:
    """Bootstrap the three metric components (+ combined) by resampling pairs.

    Draws ``n_boot`` resamples of the pair set with replacement (seeded by
    ``config.Seeds.bootstrap`` unless ``seed`` is given), recomputes
    :func:`score_components` on each, and collects the per-component replicate arrays.

    Args:
        solution: Ground-truth frame (production-metric schema).
        submission: Candidate submission (same ``pair_id`` set).
        n_boot: Number of bootstrap replicates ``B``.
        seed: RNG seed; defaults to ``config.Seeds.bootstrap`` for determinism.
        row_id_column_name: Kept for signature parity; must be ``"pair_id"``.

    Returns:
        A :class:`BootstrapSamples` with the replicate arrays for every component.
    """
    if row_id_column_name != "pair_id":
        raise ValueError("The row ID column must be pair_id.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if seed is None:
        seed = get_config().seeds.bootstrap

    n = len(solution)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty solution.")

    # Align the submission to the solution order once, so resample indices refer to
    # a single, consistent row ordering for both frames.
    aligned_sub = submission.set_index("pair_id").loc[solution["pair_id"].to_numpy()].reset_index()
    aligned_sol = solution.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    collected: Dict[str, List[float]] = {k: [] for k in COMPONENT_KEYS}
    for _ in range(n_boot):
        positions = rng.integers(0, n, size=n)
        sol_b, sub_b = _resample_frames(aligned_sol, aligned_sub, positions)
        comp = score_components(sol_b, sub_b)
        d = comp.as_dict()
        for k in COMPONENT_KEYS:
            collected[k].append(d[k])

    samples = {k: np.asarray(v, dtype=float) for k, v in collected.items()}
    return BootstrapSamples(samples=samples, n_pairs=n, n_boot=n_boot, seed=seed)


def confidence_interval(samples: np.ndarray, *, alpha: float = 0.05) -> ConfidenceInterval:
    """Percentile CI + mean/std for a bootstrap replicate array.

    Args:
        samples: 1-D array of bootstrap replicates.
        alpha: Significance level; the CI is the ``[alpha/2, 1 - alpha/2]``
            percentile interval (default ``alpha=0.05`` -> 95% CI).

    Returns:
        A :class:`ConfidenceInterval`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute a CI from an empty sample array.")
    low = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return ConfidenceInterval(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        low=low,
        high=high,
        alpha=alpha,
    )


def paired_bootstrap_diff(
    solution: pd.DataFrame,
    baseline_submission: pd.DataFrame,
    candidate_submission: pd.DataFrame,
    *,
    key: str = "combined",
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Paired bootstrap of ``candidate - baseline`` on a shared pair resample.

    On each replicate a single pair resample is drawn and BOTH submissions are
    scored on that same resample; the returned array is the distribution of the
    per-replicate difference ``candidate[key] - baseline[key]``. Pairing removes the
    shared pair-sampling variance, giving a much more powerful test of a real
    improvement than comparing two independent CIs.

    Args:
        solution: Shared ground-truth frame.
        baseline_submission: The incumbent submission.
        candidate_submission: The proposed submission (same ``pair_id`` set).
        key: Which component to difference (default ``"combined"``).
        n_boot: Number of replicates.
        seed: RNG seed; defaults to ``config.Seeds.bootstrap``.

    Returns:
        A 1-D array of length ``n_boot`` of paired differences.
    """
    if key not in COMPONENT_KEYS:
        raise ValueError(f"key must be one of {COMPONENT_KEYS}, got {key!r}.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if seed is None:
        seed = get_config().seeds.bootstrap

    n = len(solution)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty solution.")

    aligned_sol = solution.reset_index(drop=True)
    base = baseline_submission.set_index("pair_id").loc[solution["pair_id"].to_numpy()].reset_index()
    cand = candidate_submission.set_index("pair_id").loc[solution["pair_id"].to_numpy()].reset_index()

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        positions = rng.integers(0, n, size=n)
        sol_b, base_b = _resample_frames(aligned_sol, base, positions)
        _, cand_b = _resample_frames(aligned_sol, cand, positions)
        base_score = score_components(sol_b, base_b).as_dict()[key]
        cand_score = score_components(sol_b, cand_b).as_dict()[key]
        diffs[i] = cand_score - base_score
    return diffs


def is_improvement_beyond_band(
    solution: pd.DataFrame,
    baseline_submission: pd.DataFrame,
    candidate_submission: pd.DataFrame,
    *,
    key: str = "combined",
    n_boot: int = 500,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    rule: str = "paired",
) -> bool:
    """Decide whether the candidate improves beyond the bootstrap noise band.

    Default rule (``rule="paired"``, preferred): run :func:`paired_bootstrap_diff`
    and return ``True`` iff the two-sided ``(1 - alpha)`` percentile CI of the paired
    difference ``candidate - baseline`` lies strictly above zero (lower bound ``> 0``).

    Alternative rules (offered for reference / diagnostics):

    * ``rule="upper"`` — candidate's bootstrap mean exceeds the baseline's CI upper
      bound. Weaker: ignores the pairing and the candidate's own variance.
    * ``rule="disjoint"`` — the candidate CI lies entirely above the baseline CI
      (non-overlapping, candidate higher). The most conservative / least powerful.

    Args:
        solution: Shared ground-truth frame.
        baseline_submission: The incumbent submission.
        candidate_submission: The proposed submission.
        key: Component to test (default ``"combined"``).
        n_boot: Number of bootstrap replicates.
        seed: RNG seed; defaults to ``config.Seeds.bootstrap``.
        alpha: Significance level for the band (default 0.05 -> 95%).
        rule: One of ``"paired"`` (default), ``"upper"``, ``"disjoint"``.

    Returns:
        ``True`` if the candidate is an improvement beyond the noise band.
    """
    if key not in COMPONENT_KEYS:
        raise ValueError(f"key must be one of {COMPONENT_KEYS}, got {key!r}.")

    if rule == "paired":
        diffs = paired_bootstrap_diff(
            solution,
            baseline_submission,
            candidate_submission,
            key=key,
            n_boot=n_boot,
            seed=seed,
        )
        ci = confidence_interval(diffs, alpha=alpha)
        return ci.low > 0.0

    base_samples = bootstrap_components(
        solution, baseline_submission, n_boot=n_boot, seed=seed
    )
    cand_samples = bootstrap_components(
        solution, candidate_submission, n_boot=n_boot, seed=seed
    )
    base_ci = base_samples.interval(key, alpha=alpha)
    cand_ci = cand_samples.interval(key, alpha=alpha)

    if rule == "upper":
        return cand_ci.mean > base_ci.high
    if rule == "disjoint":
        return cand_ci.low > base_ci.high
    raise ValueError(f"Unknown rule {rule!r}; expected 'paired', 'upper', or 'disjoint'.")
