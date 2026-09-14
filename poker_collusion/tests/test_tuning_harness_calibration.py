"""Unit tests for the Calibrator (task 7.1, ``gate/calibration.py``).

Covers the four behaviours the design/requirements name for the Calibrator:

* **separable & >= 3 anchors** -> ``LB_CALIBRATED`` threshold strictly inside the
  open gap ``(max_pass_drift, min_fail_drift)``, ``trusted = True`` (Req 1.6).
* **overlapping ranges** -> gate INVALID: ``trusted = False``, threshold kind
  ``PROVISIONAL``, ``null_surfaced = True`` (Req 1.5).
* **separable but < 3 anchors** -> ``PROVISIONAL`` (~0.65), reduced confidence,
  ``trusted = False`` (Req 1.7).
* **the motivating bug** (Req 1.8): anchors 0.639 PASS / 0.644 PASS / 0.865 FAIL
  yield a calibrated gap between 0.644 and 0.865 so the foundation's whole-stack
  drift 0.679 classifies PASS - the exact regression the fixed 0.65 caused.

Plus the documented separation-check edge conventions (no FAIL anchors, no PASS
anchors) and the never-raise-on-degenerate-input contract.

Small synthetic ``LBAnchor`` tuples only - no store, no real caches - so the test
is fast and hermetic.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8.
"""

from __future__ import annotations

import math

from poker_collusion.tuning_harness.gate.calibration import (
    MIN_ANCHORS_FOR_TRUST,
    PROVISIONAL_THRESHOLD,
    calibrate,
    clean_separation_check,
)
from poker_collusion.tuning_harness.gate.drift_gate import PROVISIONAL_THRESHOLD as GATE_PROVISIONAL
from poker_collusion.tuning_harness.models import Calibration, SeparationCheck

FLOOR_LB = 0.44519


def _anchor(config_id: str, real_lb: float, drift: float, holdout_ap: float = 0.38):
    """Build a verified ``LBAnchor`` with the given real LB and measured drift."""

    from poker_collusion.tuning_harness.models import LBAnchor

    return LBAnchor(
        config_id=config_id,
        real_lb=real_lb,
        measured_drift=drift,
        holdout_ap=holdout_ap,
        source="unit-test",
        verified=True,
    )


def _classify(auc: float, threshold: float) -> str:
    """Mirror ``evaluate_drift``: FAIL at/above the threshold, PASS strictly below."""

    return "FAIL" if auc >= threshold else "PASS"


# --------------------------------------------------------------------------- #
# clean_separation_check
# --------------------------------------------------------------------------- #
def test_separation_strict_gap_is_separable():
    anchors = [
        _anchor("p1", 0.50, 0.60),
        _anchor("p2", 0.46, 0.64),
        _anchor("f1", 0.40, 0.80),
    ]
    sep = clean_separation_check(anchors, FLOOR_LB)
    assert sep.max_pass_drift == 0.64
    assert sep.min_fail_drift == 0.80
    assert sep.separable is True


def test_separation_tie_is_not_separable():
    # A PASS and a FAIL anchor at the SAME drift -> no strict gap -> not separable.
    anchors = [
        _anchor("p1", 0.50, 0.70),
        _anchor("f1", 0.40, 0.70),
    ]
    sep = clean_separation_check(anchors, FLOOR_LB)
    assert sep.max_pass_drift == 0.70
    assert sep.min_fail_drift == 0.70
    assert sep.separable is False


def test_separation_overlap_is_not_separable():
    # A PASS anchor drifts higher than a FAIL anchor -> ranges overlap.
    anchors = [
        _anchor("p1", 0.50, 0.75),
        _anchor("f1", 0.40, 0.70),
    ]
    sep = clean_separation_check(anchors, FLOOR_LB)
    assert sep.separable is False


def test_separation_no_fail_anchors_is_separable():
    # No FAIL anchors -> no upper conflict -> +inf min_fail_drift -> separable.
    anchors = [_anchor("p1", 0.50, 0.60), _anchor("p2", 0.46, 0.64)]
    sep = clean_separation_check(anchors, FLOOR_LB)
    assert sep.max_pass_drift == 0.64
    assert math.isinf(sep.min_fail_drift) and sep.min_fail_drift > 0
    assert sep.separable is True


def test_separation_no_pass_anchors_is_separable():
    # No PASS anchors -> no lower conflict -> -inf max_pass_drift -> separable.
    anchors = [_anchor("f1", 0.40, 0.80), _anchor("f2", 0.30, 0.85)]
    sep = clean_separation_check(anchors, FLOOR_LB)
    assert math.isinf(sep.max_pass_drift) and sep.max_pass_drift < 0
    assert sep.min_fail_drift == 0.80
    assert sep.separable is True


# --------------------------------------------------------------------------- #
# calibrate - the three trust branches
# --------------------------------------------------------------------------- #
def test_calibrate_separable_ge3_is_lb_calibrated_in_gap():
    anchors = [
        _anchor("p1", 0.50, 0.60),
        _anchor("p2", 0.46, 0.64),
        _anchor("f1", 0.40, 0.80),
    ]
    calib = calibrate(anchors, FLOOR_LB)
    assert isinstance(calib, Calibration)
    assert calib.threshold_kind == "LB_CALIBRATED"
    assert calib.trusted is True
    assert calib.null_surfaced is False
    # Threshold strictly inside the open gap (max_pass, min_fail).
    assert calib.separation.max_pass_drift < calib.threshold < calib.separation.min_fail_drift
    # Midpoint by default.
    assert calib.threshold == (0.64 + 0.80) / 2.0
    # Re-classifies its own anchors correctly (Req 1.2/1.3).
    assert _classify(0.60, calib.threshold) == "PASS"
    assert _classify(0.64, calib.threshold) == "PASS"
    assert _classify(0.80, calib.threshold) == "FAIL"


def test_calibrate_overlap_is_invalid_and_surfaces_null():
    anchors = [
        _anchor("p1", 0.50, 0.75),  # PASS but drifts high
        _anchor("p2", 0.47, 0.60),
        _anchor("f1", 0.40, 0.70),  # FAIL but drifts below a PASS anchor
    ]
    calib = calibrate(anchors, FLOOR_LB)
    assert calib.threshold_kind == "PROVISIONAL"
    assert calib.trusted is False
    assert calib.null_surfaced is True
    assert calib.separation.separable is False
    assert calib.threshold == PROVISIONAL_THRESHOLD


def test_calibrate_fewer_than_three_is_provisional():
    anchors = [
        _anchor("p1", 0.50, 0.60),
        _anchor("f1", 0.40, 0.80),
    ]
    assert len(anchors) < MIN_ANCHORS_FOR_TRUST
    calib = calibrate(anchors, FLOOR_LB)
    assert calib.threshold_kind == "PROVISIONAL"
    assert calib.trusted is False
    # Separable, so the null is NOT surfaced (that is reserved for overlap).
    assert calib.null_surfaced is False
    assert calib.separation.separable is True
    assert calib.threshold == PROVISIONAL_THRESHOLD


def test_calibrate_never_raises_on_empty_or_degenerate():
    # Empty and single-anchor sets must degrade, never raise.
    assert calibrate([], FLOOR_LB).threshold_kind == "PROVISIONAL"
    single = calibrate([_anchor("p1", 0.50, 0.60)], FLOOR_LB)
    assert single.threshold_kind == "PROVISIONAL"
    assert single.trusted is False


def test_calibrate_populates_projection_fit():
    anchors = [
        _anchor("p1", 0.50, 0.60, holdout_ap=0.38),
        _anchor("p2", 0.46, 0.64, holdout_ap=0.37),
        _anchor("f1", 0.40, 0.80, holdout_ap=0.35),
    ]
    calib = calibrate(anchors, FLOOR_LB)
    assert calib.projection is not None
    assert calib.projection.n_points == 3


# --------------------------------------------------------------------------- #
# The motivating bug (Req 1.8)
# --------------------------------------------------------------------------- #
def test_motivating_bug_foundation_drift_0679_classifies_pass():
    # dossier anchors: DIRc 0.639 PASS, MFg 0.644 PASS, leaky exp4b 0.865 FAIL.
    anchors = [
        _anchor("dirc", 0.44519, 0.639),
        _anchor("mfg", 0.44519, 0.644),
        _anchor("exp4b", 0.30, 0.865),
    ]
    calib = calibrate(anchors, FLOOR_LB)
    assert calib.threshold_kind == "LB_CALIBRATED"
    assert calib.trusted is True
    # Gap is (0.644, 0.865); midpoint 0.7545.
    assert calib.separation.max_pass_drift == 0.644
    assert calib.separation.min_fail_drift == 0.865
    assert 0.644 < calib.threshold < 0.865

    foundation_drift = 0.679
    # The whole point: 0.679 PASSes under the calibrated gap although it is
    # ABOVE the fixed 0.65 that wrongly FAILed it.
    assert _classify(foundation_drift, calib.threshold) == "PASS"
    assert _classify(foundation_drift, GATE_PROVISIONAL) == "FAIL"
