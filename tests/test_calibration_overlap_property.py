"""Property-based test for the Calibrator overlap/null-surfacing invariant
(task 7.5, ``gate/calibration.py``).

Adds exactly ONE Hypothesis property:

* **Property 4: Overlapping anchor ranges invalidate the gate and surface the
  null** (Validates Requirements 1.5) - for any set of LB_Anchors whose
  PASS-anchor drift range and FAIL-anchor drift range OVERLAP (i.e. NOT
  ``max_pass_drift < min_fail_drift``), ``calibrate`` returns
  ``trusted == False``, sets ``null_surfaced == True``, and reports NO
  LB_CALIBRATED (trustworthy) threshold - it degrades to PROVISIONAL.

This is the honest-null branch (accountability contract Rule 3): when drift does
NOT cleanly separate real leaderboard outcomes, the gate must say so rather than
manufacture a trusted bar.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.gate.calibration import (
    calibrate,
    clean_separation_check,
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
def overlapping_anchor_sets(draw):
    """Draw an anchor set whose PASS and FAIL drift ranges are guaranteed to OVERLAP.

    Construction (forces ``NOT (max_pass_drift < min_fail_drift)``):

    - Draw a ``floor_lb`` in a realistic LB range.
    - Draw a single ``shared_drift`` value and assign it to BOTH a PASS anchor
      (``real_lb >= floor_lb``) and a FAIL anchor (``real_lb < floor_lb``). That
      alone forces a tie (``max_pass_drift >= shared_drift == min_fail_drift``),
      which is non-separable under the STRICT-order predicate.
    - Additionally sprinkle extra PASS anchors drawn from a band that reaches at
      or above ``shared_drift`` and extra FAIL anchors from a band reaching at or
      below it, so overlap is robust regardless of how many extra anchors land.

    Overlap is then explicitly verified with
    ``clean_separation_check(...).separable is False`` (assumed/guarded) so a
    stray coincidence can never let a separable set through.
    """

    floor_lb = draw(st.floats(min_value=0.10, max_value=0.80))

    # A drift value shared by at least one PASS and one FAIL anchor -> a tie,
    # which is NOT separable under the strict predicate.
    shared_drift = draw(
        st.floats(min_value=0.10, max_value=0.90, allow_nan=False, allow_infinity=False)
    )

    pass_lb = st.floats(min_value=floor_lb, max_value=1.0)  # real_lb >= floor -> PASS
    fail_lb = st.floats(min_value=0.0, max_value=floor_lb, exclude_max=True)  # < floor -> FAIL

    anchors = [
        _anchor("p_shared", draw(pass_lb), shared_drift),
        _anchor("f_shared", draw(fail_lb), shared_drift),
    ]

    # Extra PASS anchors whose drift can reach at/above the shared value, and
    # extra FAIL anchors whose drift can reach at/below it -> keeps the ranges
    # interleaved (overlapping) rather than merely tied at one point.
    pass_band = st.floats(min_value=shared_drift, max_value=0.99)
    fail_band = st.floats(min_value=0.01, max_value=shared_drift)

    for i in range(draw(st.integers(min_value=0, max_value=3))):
        anchors.append(_anchor(f"p{i}", draw(pass_lb), draw(pass_band)))
    for j in range(draw(st.integers(min_value=0, max_value=3))):
        anchors.append(_anchor(f"f{j}", draw(fail_lb), draw(fail_band)))

    return floor_lb, anchors


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 4: Overlapping anchor ranges invalidate
# the gate and surface the null
# Validates: Requirements 1.5
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(case=overlapping_anchor_sets())
def test_property_overlap_invalidates_gate_and_surfaces_null(case) -> None:
    floor_lb, anchors = case

    # Guard: confirm the constructed set actually OVERLAPS (non-separable) before
    # asserting the overlap behaviour - never test the wrong regime.
    sep = clean_separation_check(anchors, floor_lb)
    assume(sep.separable is False)

    calib = calibrate(anchors, floor_lb)

    # Overlap invalidates the gate: no trusted, anchor-calibrated threshold.
    assert calib.trusted is False
    # The honest null is surfaced (Req 1.5, contract Rule 3).
    assert calib.null_surfaced is True
    # No LB_CALIBRATED threshold is reported as trustworthy - it degrades to
    # PROVISIONAL (and stays untrusted, asserted above).
    assert calib.threshold_kind != "LB_CALIBRATED"
    assert calib.separation.separable is False
