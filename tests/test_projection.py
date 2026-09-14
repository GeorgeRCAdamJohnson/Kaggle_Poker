"""Tests for the LB_Projection with an enforced fit range (task 15.1).

Covers ``fit_projection`` (single-anchor grounding fallback + multi-anchor
least-squares fit with the actual regime recorded) and ``project_lb`` (the
point equals ``slope*ap+intercept``, and ``regime_verified`` is True only within
the fit range - never extrapolating, accountability contract Rule 4 / design
Property 23).

Requirements: 5.4, 7.1.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.models import LBAnchor, ProjectionFit
from poker_collusion.tuning_harness.projection.project import (
    GROUNDING_INTERCEPT,
    GROUNDING_RESID_STD,
    GROUNDING_SLOPE,
    fit_projection,
    project_lb,
)


def _anchor(config_id: str, real_lb: float, holdout_ap: float, drift: float = 0.6) -> LBAnchor:
    return LBAnchor(
        config_id=config_id,
        real_lb=real_lb,
        measured_drift=drift,
        holdout_ap=holdout_ap,
        source="test",
        verified=True,
    )


# --------------------------------------------------------------------------- #
# fit_projection
# --------------------------------------------------------------------------- #
def test_fit_single_anchor_falls_back_to_grounding() -> None:
    """One anchor: grounding slope/intercept, regime = that anchor's AP, n=1."""
    anchors = [_anchor("known_good_floor", real_lb=0.44519, holdout_ap=0.3839)]
    fit = fit_projection(anchors)

    assert fit.slope == GROUNDING_SLOPE
    assert fit.intercept == GROUNDING_INTERCEPT
    assert fit.resid_std == GROUNDING_RESID_STD
    assert fit.fit_min_ap == 0.3839
    assert fit.fit_max_ap == 0.3839
    assert fit.n_points == 1


def test_fit_no_anchors_falls_back_with_empty_regime() -> None:
    """Zero anchors: grounding line, empty regime so everything is unverified."""
    fit = fit_projection([])

    assert fit.slope == GROUNDING_SLOPE
    assert fit.intercept == GROUNDING_INTERCEPT
    assert fit.fit_min_ap == 0.0
    assert fit.fit_max_ap == 0.0
    assert fit.n_points == 0


def test_fit_two_distinct_anchors_uses_least_squares() -> None:
    """>=2 distinct APs: least-squares recovers the exact line through them."""
    # Two points on the line y = 2x + 0.1 -> slope 2, intercept 0.1.
    anchors = [
        _anchor("a", real_lb=2 * 0.30 + 0.1, holdout_ap=0.30),
        _anchor("b", real_lb=2 * 0.40 + 0.1, holdout_ap=0.40),
    ]
    fit = fit_projection(anchors)

    assert abs(fit.slope - 2.0) < 1e-9
    assert abs(fit.intercept - 0.1) < 1e-9
    assert fit.resid_std < 1e-9  # exact fit through 2 points
    assert fit.fit_min_ap == 0.30
    assert fit.fit_max_ap == 0.40
    assert fit.n_points == 2


def test_fit_multiple_anchors_records_actual_min_max() -> None:
    anchors = [
        _anchor("a", real_lb=0.40, holdout_ap=0.30),
        _anchor("b", real_lb=0.45, holdout_ap=0.38),
        _anchor("c", real_lb=0.50, holdout_ap=0.42),
    ]
    fit = fit_projection(anchors)

    assert fit.fit_min_ap == 0.30
    assert fit.fit_max_ap == 0.42
    assert fit.n_points == 3


def test_fit_two_anchors_same_ap_falls_back_to_grounding() -> None:
    """>=2 anchors but identical AP: slope unidentifiable -> grounding fallback."""
    anchors = [
        _anchor("a", real_lb=0.44, holdout_ap=0.38),
        _anchor("b", real_lb=0.45, holdout_ap=0.38),
    ]
    fit = fit_projection(anchors)

    assert fit.slope == GROUNDING_SLOPE
    assert fit.intercept == GROUNDING_INTERCEPT
    assert fit.fit_min_ap == 0.38
    assert fit.fit_max_ap == 0.38
    assert fit.n_points == 2


def test_fit_default_reads_seeded_store() -> None:
    """No explicit anchors -> reads the seeded store (the known_good_floor)."""
    fit = fit_projection()
    # The seeded store has a single anchor at holdout_ap 0.3839.
    assert fit.n_points >= 1
    assert fit.fit_min_ap <= 0.3839 <= fit.fit_max_ap


# --------------------------------------------------------------------------- #
# project_lb
# --------------------------------------------------------------------------- #
def test_project_in_range_is_verified_and_matches_line() -> None:
    fit = ProjectionFit(
        slope=1.4235,
        intercept=-0.1414,
        resid_std=0.0022,
        fit_min_ap=0.30,
        fit_max_ap=0.45,
        n_points=3,
    )
    proj = project_lb(0.3839, fit)

    assert proj.regime_verified is True
    assert abs(proj.point - (1.4235 * 0.3839 - 0.1414)) < 1e-12
    assert proj.uncertainty == 0.0022


def test_project_below_range_is_unverified() -> None:
    fit = ProjectionFit(
        slope=1.4235,
        intercept=-0.1414,
        resid_std=0.0022,
        fit_min_ap=0.30,
        fit_max_ap=0.45,
        n_points=3,
    )
    proj = project_lb(0.20, fit)  # below fit_min_ap

    assert proj.regime_verified is False
    # Point is still returned (never silently dropped), just flagged unverified.
    assert abs(proj.point - (1.4235 * 0.20 - 0.1414)) < 1e-12


def test_project_above_range_is_unverified() -> None:
    fit = ProjectionFit(
        slope=1.4235,
        intercept=-0.1414,
        resid_std=0.0022,
        fit_min_ap=0.30,
        fit_max_ap=0.45,
        n_points=3,
    )
    proj = project_lb(0.60, fit)  # above fit_max_ap

    assert proj.regime_verified is False
    assert abs(proj.point - (1.4235 * 0.60 - 0.1414)) < 1e-12


def test_project_at_boundaries_is_verified() -> None:
    """The fit range is inclusive at both endpoints."""
    fit = ProjectionFit(
        slope=1.0,
        intercept=0.0,
        resid_std=0.0022,
        fit_min_ap=0.30,
        fit_max_ap=0.45,
        n_points=3,
    )
    assert project_lb(0.30, fit).regime_verified is True
    assert project_lb(0.45, fit).regime_verified is True


# --------------------------------------------------------------------------- #
# Property 23: projections outside the fit range are flagged unverified
# Feature: poker-layered-tuning, Property 23: Projections outside the fit range
# are flagged unverified (no extrapolation)
# Validates: Requirements 5.4, 7.1
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(
    holdout_ap=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    fit_min_ap=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    span=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    slope=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    intercept=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    resid_std=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_property_no_extrapolation_flagging(
    holdout_ap: float,
    fit_min_ap: float,
    span: float,
    slope: float,
    intercept: float,
    resid_std: float,
) -> None:
    fit_max_ap = fit_min_ap + span
    fit = ProjectionFit(
        slope=slope,
        intercept=intercept,
        resid_std=resid_std,
        fit_min_ap=fit_min_ap,
        fit_max_ap=fit_max_ap,
        n_points=3,
    )
    proj = project_lb(holdout_ap, fit)

    in_range = fit_min_ap <= holdout_ap <= fit_max_ap
    # regime_verified is True iff and only iff the AP lies within the fit range.
    assert proj.regime_verified == in_range
    # The point is always the calibrated line's value; uncertainty is the fit's.
    assert proj.point == slope * holdout_ap + intercept
    assert proj.uncertainty == resid_std
