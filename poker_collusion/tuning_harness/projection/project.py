"""LB projection with an enforced fit range (no extrapolation).

This module turns the ``LB_Anchor`` store into a linear ``LB <- PU-stress-AP``
fit and projects a holdout AP onto a leaderboard estimate WITHOUT ever
extrapolating past the sampled AP regime.

Grounding (design "Projection + submission gate" / dossier §17): the calibrated
predictor is ``LB ~= 1.4235 * PU_stress_AP - 0.1414`` with residual std
~0.0022, but it is valid ONLY within the sampled AP range. Applying it outside
that range is banned by the reverse-engineering-accountability contract Rule 4
("no extrapolation - an estimator is valid ONLY within the range it was fit
on"). ``project_lb`` therefore always returns a value but flags
``regime_verified = False`` whenever the input AP falls outside
``[fit_min_ap, fit_max_ap]`` - it never extrapolates silently.

The :class:`ProjectionFit` and :class:`Projection` data models are ALREADY
defined in :mod:`poker_collusion.tuning_harness.models`; this module reuses them
and only supplies the constructor (``fit_projection``) and the projection
function (``project_lb``).

Requirements:
- 5.4: reports distinguish the measured holdout AP from the projected LB with
  its uncertainty - ``project_lb`` returns a :class:`Projection` whose ``point``
  / ``uncertainty`` are the projected value and its spread, never the holdout AP.
- 7.1: the submission report's LB_Projection + regime-verified flag come from
  here.
- Enforces accountability contract Rule 4 (no extrapolation) and design
  Assumption A3 (the LB<-PU-stress-AP line holds only within the sampled range).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from poker_collusion.tuning_harness.anchor_store import load_anchors
from poker_collusion.tuning_harness.models import LBAnchor, Projection, ProjectionFit

__all__ = [
    "GROUNDING_SLOPE",
    "GROUNDING_INTERCEPT",
    "GROUNDING_RESID_STD",
    "fit_projection",
    "project_lb",
]

#: Grounding line from dossier §17 (LB ~= slope * PU_stress_AP + intercept).
#: Used as the fallback when the anchor store has a single point (a slope cannot
#: be estimated from one point) and as the seed for the recorded resid_std.
GROUNDING_SLOPE: float = 1.4235
GROUNDING_INTERCEPT: float = -0.1414
GROUNDING_RESID_STD: float = 0.0022

#: Minimum ``sum((x - mean_x)^2)`` for a slope to be considered identifiable.
#: APs live in ``[0, 1]``; a spread whose variance falls below this is a
#: numerically-degenerate (denormal / near-zero) difference, not a real regime,
#: so the fit falls back to grounding rather than dividing by ~0. Chosen well
#: below any realistic AP spread (a spread of ~1e-12 gives sxx ~1e-24) yet far
#: above the denormal-scale values (~1e-308) Hypothesis surfaces.
_MIN_SXX: float = 1e-24


def _least_squares_line(
    xs: Sequence[float], ys: Sequence[float]
) -> Optional[tuple[float, float, float]]:
    """Ordinary least-squares fit of ``y = slope * x + intercept``.

    Returns ``(slope, intercept, resid_std)`` where ``resid_std`` is the
    population standard deviation of the residuals, or ``None`` when the slope
    is unidentifiable because the ``xs`` have (numerically) no spread.

    Assumes ``len(xs) >= 2``. The ``sxx`` guard defends against a spread that is
    positive but denormal/near-zero (e.g. distinct-by-~2.2e-308 AP values that
    slip past an exact ``fit_min_ap == fit_max_ap`` equality check): dividing by
    such an ``sxx`` yields a ``ZeroDivisionError`` or an inf/nan slope. When the
    spread is not comfortably positive (or non-finite) we report the fit as
    unidentifiable so the caller can fall back to grounding.
    """

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if not math.isfinite(sxx) or sxx <= _MIN_SXX:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        return None
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    resid_std = (sum(r * r for r in residuals) / n) ** 0.5
    return slope, intercept, resid_std


def fit_projection(anchors: Iterable[LBAnchor] | None = None) -> ProjectionFit:
    """Fit the ``LB <- holdout_ap`` line and record the regime it is valid in.

    ``anchors`` is an explicit list of :class:`LBAnchor` records; when ``None``
    the seeded append-only LB_Anchor store is read via
    :func:`~poker_collusion.tuning_harness.anchor_store.load_anchors`.

    Behaviour by anchor count (design + contract Rule 4):

    - **0 anchors** - no data to fit. Fall back to the known grounding
      slope/intercept and record an empty regime (``fit_min_ap == fit_max_ap ==
      0.0``, ``n_points == 0``) so every projection is flagged unverified.
    - **1 anchor** - a slope cannot be estimated from a single point, so fall
      back to the grounding slope/intercept. Record
      ``fit_min_ap == fit_max_ap == that anchor's holdout_ap`` with
      ``n_points == 1`` and the grounding residual std. Only that exact AP is
      "in regime".
    - **>= 2 anchors with distinct APs** - least-squares fit; record the actual
      min/max AP as the regime and the empirical residual std.
    - **>= 2 anchors all at the SAME AP** - a slope is still unidentifiable, so
      fall back to the grounding slope/intercept anchored at that AP; the regime
      is the single point.

    The returned :class:`ProjectionFit` carries the fit range so
    :func:`project_lb` can refuse to extrapolate.
    """

    if anchors is None:
        anchors = load_anchors()
    anchor_list = list(anchors)

    if not anchor_list:
        return ProjectionFit(
            slope=GROUNDING_SLOPE,
            intercept=GROUNDING_INTERCEPT,
            resid_std=GROUNDING_RESID_STD,
            fit_min_ap=0.0,
            fit_max_ap=0.0,
            n_points=0,
        )

    aps = [a.holdout_ap for a in anchor_list]
    lbs = [a.real_lb for a in anchor_list]
    fit_min_ap = min(aps)
    fit_max_ap = max(aps)

    def _grounded() -> ProjectionFit:
        # Slope unidentifiable (too few points, no AP spread, or a spread so
        # small the fit is numerically degenerate): fall back to the grounding
        # line anchored at the observed regime. n_points still records the
        # anchor count so callers see how much data was present.
        return ProjectionFit(
            slope=GROUNDING_SLOPE,
            intercept=GROUNDING_INTERCEPT,
            resid_std=GROUNDING_RESID_STD,
            fit_min_ap=fit_min_ap,
            fit_max_ap=fit_max_ap,
            n_points=len(anchor_list),
        )

    # A slope needs >= 2 distinct AP values; otherwise fall back to grounding.
    if len(anchor_list) < 2 or fit_min_ap == fit_max_ap:
        return _grounded()

    fit = _least_squares_line(aps, lbs)
    if fit is None:
        # APs differ only by a numerically-degenerate amount -> no identifiable
        # slope; treat exactly like the single-distinct-AP branch.
        return _grounded()

    slope, intercept, resid_std = fit
    return ProjectionFit(
        slope=slope,
        intercept=intercept,
        resid_std=resid_std,
        fit_min_ap=fit_min_ap,
        fit_max_ap=fit_max_ap,
        n_points=len(anchor_list),
    )


def project_lb(holdout_ap: float, fit: ProjectionFit) -> Projection:
    """Project a holdout AP onto a leaderboard estimate, honestly flagged.

    Returns a :class:`Projection`:

    - ``point`` = ``fit.slope * holdout_ap + fit.intercept`` (the calibrated
      line's estimate).
    - ``uncertainty`` = ``fit.resid_std`` - the honest spread carried from the
      fit, kept distinct from the measured holdout AP (Req 5.4).
    - ``regime_verified`` = ``True`` ONLY when ``holdout_ap`` lies within the
      inclusive fit range ``[fit_min_ap, fit_max_ap]``; otherwise ``False``.

    The point is ALWAYS computed and returned - but an off-regime AP is flagged
    ``regime_verified = False`` rather than silently extrapolated (accountability
    contract Rule 4 / design Property 23). Callers (the submission gate) use the
    flag to refuse to trust an unverified-regime projection.
    """

    point = fit.slope * holdout_ap + fit.intercept
    regime_verified = fit.fit_min_ap <= holdout_ap <= fit.fit_max_ap
    return Projection(
        point=point,
        uncertainty=fit.resid_std,
        regime_verified=regime_verified,
    )
