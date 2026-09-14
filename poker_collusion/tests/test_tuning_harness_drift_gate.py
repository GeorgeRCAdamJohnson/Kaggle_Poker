"""Unit tests for the adversarial dev-vs-eval Drift_Gate (task 6.1).

Confirms the two behaviours the design names for ``gate/drift_gate.py``:

* :func:`adversarial_auc` reads HIGH on a feature set that cleanly separates
  eval rows from dev rows (the features describe "which pool" - drift), and LOW
  (near chance) on a feature set that does not separate the two pools.
* :func:`evaluate_drift` classifies against the deciding threshold: PASS when the
  measured AUC is strictly below it and FAIL at/above it, using the
  LB_CALIBRATED threshold when the calibration is trusted and the PROVISIONAL
  threshold otherwise (Req 1.1, 1.6, 1.7).

These use small synthetic :class:`FeatureFrame` fixtures - no real caches - so the
test is fast and hermetic.

Requirements: 1.1.
"""

from __future__ import annotations

import pytest

from poker_collusion.tuning_harness.gate.drift_gate import (
    PROVISIONAL_THRESHOLD,
    adversarial_auc,
    evaluate_drift,
)
from poker_collusion.tuning_harness.models import (
    Calibration,
    FeatureFrame,
    SeparationCheck,
)


# --------------------------------------------------------------------------- #
# Synthetic FeatureFrame builders
# --------------------------------------------------------------------------- #
def _separating_frame(n_per_pool: int = 40) -> FeatureFrame:
    """A frame whose single feature cleanly separates eval from dev rows.

    Dev rows get feature ~0.0, eval rows get feature ~10.0, so a dev-vs-eval
    classifier separates them almost perfectly -> HIGH adversarial AUC (drift).
    A tiny deterministic jitter keeps folds non-degenerate without blurring the
    classes.
    """

    rows = {}
    is_eval = {}
    for i in range(n_per_pool):
        jitter = (i % 5) * 0.01
        rows[f"dev_{i}"] = (0.0 + jitter,)
        is_eval[f"dev_{i}"] = False
        rows[f"eval_{i}"] = (10.0 + jitter,)
        is_eval[f"eval_{i}"] = True
    return FeatureFrame(feature_keys=("f0",), rows=rows, is_eval=is_eval)


def _non_separating_frame(n_per_pool: int = 40) -> FeatureFrame:
    """A frame whose feature is identically distributed across dev and eval rows.

    Both pools draw the SAME repeating pattern of values, so no classifier can
    tell dev from eval -> adversarial AUC near chance (0.5), i.e. no drift.
    """

    pattern = [0.0, 1.0, 2.0, 3.0, 4.0]
    rows = {}
    is_eval = {}
    for i in range(n_per_pool):
        val = pattern[i % len(pattern)]
        rows[f"dev_{i}"] = (val,)
        is_eval[f"dev_{i}"] = False
        rows[f"eval_{i}"] = (val,)
        is_eval[f"eval_{i}"] = True
    return FeatureFrame(feature_keys=("f0",), rows=rows, is_eval=is_eval)


def _calibration(threshold: float, *, trusted: bool, kind: str) -> Calibration:
    """A minimal Calibration carrier (task 7.1 produces the real one)."""

    return Calibration(
        threshold=threshold,
        threshold_kind=kind,
        trusted=trusted,
        separation=SeparationCheck(max_pass_drift=0.6, min_fail_drift=0.7, separable=True),
        null_surfaced=False,
        projection=None,
    )


# --------------------------------------------------------------------------- #
# adversarial_auc
# --------------------------------------------------------------------------- #
def test_adversarial_auc_high_when_features_separate_pools():
    """Features that separate eval from dev rows yield a high adversarial AUC."""
    auc = adversarial_auc(_separating_frame())
    assert auc > 0.9


def test_adversarial_auc_near_chance_when_features_do_not_separate():
    """Identically-distributed features across pools yield an AUC near chance."""
    auc = adversarial_auc(_non_separating_frame())
    assert auc < 0.65
    assert 0.4 <= auc <= 0.65


def test_adversarial_auc_degenerate_frames_return_chance():
    """Empty, single-pool, or featureless frames report chance (no measurable drift)."""
    empty = FeatureFrame(feature_keys=("f0",), rows={}, is_eval={})
    assert adversarial_auc(empty) == pytest.approx(0.5)

    single_pool = FeatureFrame(
        feature_keys=("f0",),
        rows={"dev_0": (1.0,), "dev_1": (2.0,)},
        is_eval={"dev_0": False, "dev_1": False},
    )
    assert adversarial_auc(single_pool) == pytest.approx(0.5)

    no_features = FeatureFrame(
        feature_keys=(),
        rows={"dev_0": (), "eval_0": ()},
        is_eval={"dev_0": False, "eval_0": True},
    )
    assert adversarial_auc(no_features) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# evaluate_drift
# --------------------------------------------------------------------------- #
def test_evaluate_drift_pass_below_threshold_trusted():
    """A low-drift frame PASSes when its AUC is strictly below the calibrated bar."""
    calib = _calibration(0.75, trusted=True, kind="LB_CALIBRATED")
    result = evaluate_drift(_non_separating_frame(), calib)
    assert result.verdict == "PASS"
    assert result.adversarial_auc < result.threshold
    assert result.threshold == pytest.approx(0.75)
    assert result.threshold_kind == "LB_CALIBRATED"


def test_evaluate_drift_fail_at_or_above_threshold_trusted():
    """A high-drift frame FAILs when its AUC is at/above the calibrated bar."""
    calib = _calibration(0.75, trusted=True, kind="LB_CALIBRATED")
    result = evaluate_drift(_separating_frame(), calib)
    assert result.verdict == "FAIL"
    assert result.adversarial_auc >= result.threshold
    assert result.threshold_kind == "LB_CALIBRATED"


def test_evaluate_drift_boundary_is_fail():
    """AUC exactly at the threshold is FAIL (>= is high drift)."""
    # Set the calibrated threshold to the measured AUC so auc == threshold.
    frame = _separating_frame()
    auc = adversarial_auc(frame)
    calib = _calibration(auc, trusted=True, kind="LB_CALIBRATED")
    result = evaluate_drift(frame, calib)
    assert result.adversarial_auc == pytest.approx(auc)
    assert result.verdict == "FAIL"


def test_evaluate_drift_uses_provisional_when_untrusted():
    """An untrusted calibration classifies with the PROVISIONAL threshold."""
    # trusted=False and no threshold populated -> fall back to PROVISIONAL_THRESHOLD.
    calib = Calibration(
        threshold=None,  # type: ignore[arg-type]
        threshold_kind="PROVISIONAL",
        trusted=False,
        separation=SeparationCheck(0.0, 0.0, False),
        null_surfaced=True,
        projection=None,
    )
    result = evaluate_drift(_non_separating_frame(), calib)
    assert result.threshold_kind == "PROVISIONAL"
    assert result.threshold == pytest.approx(PROVISIONAL_THRESHOLD)
    # non-separating frame is near chance, below 0.65 -> PASS
    assert result.verdict == "PASS"


def test_evaluate_drift_untrusted_prefers_calibrator_provisional_value():
    """When untrusted but a provisional value is present, that value is used."""
    calib = _calibration(0.60, trusted=False, kind="PROVISIONAL")
    result = evaluate_drift(_separating_frame(), calib)
    assert result.threshold_kind == "PROVISIONAL"
    assert result.threshold == pytest.approx(0.60)
    # separating frame is high drift, above 0.60 -> FAIL
    assert result.verdict == "FAIL"
