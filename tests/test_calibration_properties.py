"""Property-based tests for the Calibrator (task 7.2, ``gate/calibration.py``).

Consolidated home for the calibrator's Hypothesis properties. This task adds
exactly ONE property:

* **Property 1: Calibrated threshold is derived from anchors, never the fixed
  constant** (Validates Requirements 1.1) - for any separable set of >=3
  LB_Anchors, ``calibrate`` returns an ``LB_CALIBRATED`` threshold that lies
  strictly inside the open gap ``(max_pass_drift, min_fail_drift)`` and equals
  the anchor-derived midpoint, NOT the fixed provisional constant 0.65 (unless
  0.65 happens to fall inside the calibrated gap).

This is the invariant that fixes the motivating bug: the deciding threshold is
computed from REAL leaderboard outcomes (the anchors), never guessed as a
constant.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.gate.calibration import (
    PROVISIONAL_THRESHOLD,
    calibrate,
)
from poker_collusion.tuning_harness.models import LBAnchor

FLOOR_LB = 0.44519


def _anchor(config_id: str, real_lb: float, drift: float, holdout_ap: float = 0.38) -> LBAnchor:
    """Build a verified ``LBAnchor`` with the given real LB and measured drift."""

    return LBAnchor(
        config_id=config_id,
        real_lb=real_lb,
        measured_drift=drift,
        holdout_ap=holdout_ap,
        source="property-test",
        verified=True,
    )


@st.composite
def separable_anchor_sets(draw):
    """Draw a SEPARABLE anchor set of >=3 anchors with a guaranteed open gap.

    Strategy (matches the task's construction so the gap is always open):

    - Draw a ``floor_lb`` in a realistic LB range.
    - Draw a PASS drift band ``[lo, mid)`` and a strictly higher FAIL drift band
      ``(mid2, hi]`` with ``mid < mid2``, so every PASS drift is strictly below
      every FAIL drift and ``max_pass_drift < min_fail_drift`` by construction.
    - Draw >=1 PASS anchors (``real_lb >= floor_lb``) with drifts in the PASS
      band and >=1 FAIL anchors (``real_lb < floor_lb``) with drifts in the FAIL
      band, with the total count >= 3.
    """

    floor_lb = draw(st.floats(min_value=0.10, max_value=0.80))

    # Four ordered cut points lo < mid < mid2 < hi carve two disjoint drift
    # bands with a strictly positive gap (mid2 - mid) between them.
    cuts = draw(
        st.lists(
            st.floats(min_value=0.05, max_value=0.99, allow_nan=False, allow_infinity=False),
            min_size=4,
            max_size=4,
            unique=True,
        )
    )
    lo, mid, mid2, hi = sorted(cuts)

    pass_band = st.floats(min_value=lo, max_value=mid, exclude_max=True)
    fail_band = st.floats(min_value=mid2, max_value=hi, exclude_min=True)

    # >=1 PASS and >=1 FAIL, total >= 3.
    n_pass = draw(st.integers(min_value=1, max_value=4))
    n_fail = draw(st.integers(min_value=max(1, 3 - n_pass), max_value=4))

    pass_lb = st.floats(min_value=floor_lb, max_value=1.0)  # real_lb >= floor -> PASS
    fail_lb = st.floats(min_value=0.0, max_value=floor_lb, exclude_max=True)  # < floor -> FAIL

    anchors = []
    for i in range(n_pass):
        anchors.append(_anchor(f"p{i}", draw(pass_lb), draw(pass_band)))
    for j in range(n_fail):
        anchors.append(_anchor(f"f{j}", draw(fail_lb), draw(fail_band)))

    return floor_lb, anchors


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 1: Calibrated threshold is derived
# from anchors, never the fixed constant
# Validates: Requirements 1.1
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(case=separable_anchor_sets())
def test_property_calibrated_threshold_is_derived_from_anchors(case) -> None:
    floor_lb, anchors = case
    assert len(anchors) >= 3

    calib = calibrate(anchors, floor_lb)
    sep = calib.separation

    # The set is separable by construction, and there are >=3 anchors, so the
    # calibrator MUST trust an LB-calibrated (anchor-derived) threshold.
    assert sep.separable is True
    assert calib.threshold_kind == "LB_CALIBRATED"
    assert calib.trusted is True
    assert calib.null_surfaced is False

    # The threshold lies strictly inside the open gap (max_pass, min_fail).
    assert sep.max_pass_drift < calib.threshold < sep.min_fail_drift

    # It equals the anchor-derived midpoint of that gap - the deciding value is
    # COMPUTED from the anchors, not a guessed constant.
    expected_midpoint = (sep.max_pass_drift + sep.min_fail_drift) / 2.0
    assert calib.threshold == expected_midpoint

    # It is NOT the fixed provisional constant 0.65 unless 0.65 legitimately
    # falls inside the calibrated gap (in which case equality is a coincidence,
    # not a hardcode).
    if not (sep.max_pass_drift < PROVISIONAL_THRESHOLD < sep.min_fail_drift):
        assert calib.threshold != PROVISIONAL_THRESHOLD
