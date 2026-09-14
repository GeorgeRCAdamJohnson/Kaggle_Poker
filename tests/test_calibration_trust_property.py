"""Property-based test for the Calibrator trust branch (task 7.6).

Adds exactly ONE Hypothesis property:

* **Property 5: Trust is granted only when separable and sufficiently anchored**
  (Validates Requirements 1.6, 1.7) - for ANY set of LB_Anchors, ``calibrate``
  reports ``trusted == True`` with an ``LB_CALIBRATED`` threshold IF AND ONLY IF
  the Clean_Separation_Check passes AND there are at least
  ``MIN_ANCHORS_FOR_TRUST`` (3) anchors; otherwise it reports a ``PROVISIONAL``
  threshold with ``trusted == False`` (the reduced-confidence flag).

The generator draws ARBITRARY anchor sets (count 0..8, PASS/FAIL mixed freely,
drift bands overlapping or separated, arbitrary floor_lb) so BOTH branches of the
iff are exercised: sometimes separable & >=3, sometimes not.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.gate.calibration import (
    MIN_ANCHORS_FOR_TRUST,
    calibrate,
    clean_separation_check,
)
from poker_collusion.tuning_harness.models import LBAnchor


@st.composite
def arbitrary_anchor_sets(draw):
    """Draw an ARBITRARY anchor set (count 0..8) plus an arbitrary ``floor_lb``.

    No structure is imposed: real_lb and measured_drift are drawn independently
    across the whole realistic range, so PASS/FAIL membership and the ordering of
    the PASS/FAIL drift bands are unconstrained. This makes the set separable &
    >=3 on some draws and non-separable and/or too small on others, exercising
    both sides of the iff.
    """

    floor_lb = draw(st.floats(min_value=0.10, max_value=0.80))
    n = draw(st.integers(min_value=0, max_value=8))

    lb_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    drift_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ap_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

    anchors = []
    for i in range(n):
        anchors.append(
            LBAnchor(
                config_id=f"a{i}",
                real_lb=draw(lb_strategy),
                measured_drift=draw(drift_strategy),
                holdout_ap=draw(ap_strategy),
                source="property-test",
                verified=True,
            )
        )

    return floor_lb, anchors


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 5: Trust is granted only when
# separable and sufficiently anchored
# Validates: Requirements 1.6, 1.7
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(case=arbitrary_anchor_sets())
def test_property_trust_only_when_separable_and_anchored(case) -> None:
    floor_lb, anchors = case

    calib = calibrate(anchors, floor_lb)

    separable = clean_separation_check(anchors, floor_lb).separable
    enough = len(anchors) >= MIN_ANCHORS_FOR_TRUST

    # The iff: a trusted LB_CALIBRATED threshold is granted exactly when the
    # anchors separate cleanly AND there are enough of them.
    trusted_and_calibrated = calib.trusted and calib.threshold_kind == "LB_CALIBRATED"
    assert trusted_and_calibrated == (separable and enough)

    # The other direction of the iff, spelled out: whenever trust is NOT granted,
    # the calibrator degrades to a reduced-confidence PROVISIONAL threshold.
    if not (separable and enough):
        assert calib.threshold_kind == "PROVISIONAL"
        assert calib.trusted is False
