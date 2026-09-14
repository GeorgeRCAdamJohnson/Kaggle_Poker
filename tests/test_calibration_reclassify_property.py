"""Property test for the Calibrator anchor re-classification (task 7.3).

Property 2 (design): For any SEPARABLE set of at least 3 LB_Anchors, classifying
each anchor's measured drift with the derived ``LB_Calibrated_Threshold`` SHALL
label every PASS anchor (real LB at or above the floor) as PASS and every FAIL
anchor (regressed) as FAIL.

The generator builds SEPARABLE sets: PASS anchors draw their drift from a low
band, FAIL anchors from a strictly higher band, with a guaranteed OPEN gap
between the two bands so a midpoint threshold discriminates them exactly. Sets
carry >=1 PASS and >=1 FAIL anchor and total >=3 so the calibrator produces an
``LB_CALIBRATED`` / trusted :class:`Calibration` (separable AND >= 3 anchors),
whose ``threshold`` sits strictly inside the open gap ``(max_pass_drift,
min_fail_drift)``.

Classification mirrors ``evaluate_drift`` (see ``_classify`` in
``poker_collusion/tests/test_tuning_harness_calibration.py``): a drift AUC is
FAIL if ``auc >= threshold`` and PASS if ``auc < threshold``.

Validates: Requirements 1.2, 1.3.
"""

from __future__ import annotations

from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.gate.calibration import calibrate
from poker_collusion.tuning_harness.models import LBAnchor

FLOOR_LB = 0.44519


def _classify(auc: float, threshold: float) -> str:
    """Mirror ``evaluate_drift``: FAIL at/above the threshold, PASS strictly below."""

    return "FAIL" if auc >= threshold else "PASS"


# PASS anchors drift in the low band; FAIL anchors in a strictly higher band.
# An OPEN gap is inserted between the two bands so the two classes are separable
# by a single threshold placed in the gap (design's SEPARABLE generator).
_PASS_BAND_LO = 0.50
_PASS_BAND_HI = 0.60
_GAP = 0.05  # strictly positive gap guarantees max_pass_drift < min_fail_drift
_FAIL_BAND_LO = _PASS_BAND_HI + _GAP  # 0.65
_FAIL_BAND_HI = 0.75

_pass_drift = st.floats(
    min_value=_PASS_BAND_LO, max_value=_PASS_BAND_HI, allow_nan=False, allow_infinity=False
)
_fail_drift = st.floats(
    min_value=_FAIL_BAND_LO, max_value=_FAIL_BAND_HI, allow_nan=False, allow_infinity=False
)
# PASS anchors: real_lb at or above the floor. FAIL anchors: strictly below it.
_pass_lb = st.floats(
    min_value=FLOOR_LB, max_value=FLOOR_LB + 0.2, allow_nan=False, allow_infinity=False
)
_fail_lb = st.floats(
    min_value=0.0, max_value=FLOOR_LB - 0.01, allow_nan=False, allow_infinity=False
)


@st.composite
def separable_anchor_sets(draw) -> List[LBAnchor]:
    """A SEPARABLE set of >= 3 anchors with a guaranteed open PASS/FAIL drift gap."""

    n_pass = draw(st.integers(min_value=1, max_value=5))
    n_fail = draw(st.integers(min_value=1, max_value=5))
    # Guarantee at least 3 anchors total so the calibrator reaches the trusted
    # LB_CALIBRATED branch (separable AND >= 3 anchors).
    if n_pass + n_fail < 3:
        n_fail += 3 - (n_pass + n_fail)

    anchors: List[LBAnchor] = []
    for i in range(n_pass):
        anchors.append(
            LBAnchor(
                config_id=f"pass_{i}",
                real_lb=draw(_pass_lb),
                measured_drift=draw(_pass_drift),
                holdout_ap=0.38,
                source="pbt",
                verified=True,
            )
        )
    for j in range(n_fail):
        anchors.append(
            LBAnchor(
                config_id=f"fail_{j}",
                real_lb=draw(_fail_lb),
                measured_drift=draw(_fail_drift),
                holdout_ap=0.35,
                source="pbt",
                verified=True,
            )
        )
    return anchors


# Feature: poker-layered-tuning, Property 2: The calibrated threshold correctly re-classifies its own anchors
@settings(max_examples=200)
@given(anchors=separable_anchor_sets())
def test_property_calibrated_threshold_reclassifies_anchors(anchors: List[LBAnchor]) -> None:
    calib = calibrate(anchors, FLOOR_LB)

    # A separable set of >= 3 anchors yields a trusted, LB-calibrated threshold.
    assert calib.threshold_kind == "LB_CALIBRATED"
    assert calib.trusted is True

    for anchor in anchors:
        verdict = _classify(anchor.measured_drift, calib.threshold)
        if anchor.real_lb >= FLOOR_LB:
            # Req 1.2: a PASS anchor (held/improved) classifies PASS.
            assert verdict == "PASS", (
                f"PASS anchor {anchor.config_id} (drift {anchor.measured_drift}) "
                f"misclassified {verdict} at threshold {calib.threshold}"
            )
        else:
            # Req 1.3: a FAIL anchor (regressed) classifies FAIL.
            assert verdict == "FAIL", (
                f"FAIL anchor {anchor.config_id} (drift {anchor.measured_drift}) "
                f"misclassified {verdict} at threshold {calib.threshold}"
            )
