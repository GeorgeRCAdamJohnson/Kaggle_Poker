"""Unit tests for the drift -> holdout gate sequence (task 9.1).

Confirms the sequencing the design's "The gate sequence" section and
accountability contract Rule 17 require: the Drift_Gate runs FIRST, and the
holdout AP is a trustworthy leaderboard predictor ONLY if drift PASSes. A drift
FAIL inverts trust in the holdout (Rule 17), and a RETRAIN composition is trusted
only when its COMBINED base+layer feature set clears the Drift_Gate (Req 4.2).

Two cases per the task's acceptance:
  (a) a drift-PASS feature set  -> holdout_trustworthy True,  RETRAIN trusted True
  (b) a drift-FAIL feature set  -> holdout_trustworthy False, RETRAIN trusted False

Uses small synthetic :class:`FeatureFrame` fixtures (mirroring the drift-gate
tests) so the test is fast and hermetic.

Requirements: 4.2.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.gate.sequence import (
    GateSequenceResult,
    evaluate_retrain,
    evaluate_sequence,
)
from poker_collusion.tuning_harness.models import (
    Calibration,
    FeatureFrame,
    SeparationCheck,
)


# --------------------------------------------------------------------------- #
# Synthetic FeatureFrame builders (mirror the drift-gate test fixtures)
# --------------------------------------------------------------------------- #
def _separating_frame(n_per_pool: int = 40, *, key: str = "f0") -> FeatureFrame:
    """A frame whose feature cleanly separates eval from dev rows -> HIGH drift."""
    rows = {}
    is_eval = {}
    for i in range(n_per_pool):
        jitter = (i % 5) * 0.01
        rows[f"dev_{i}"] = (0.0 + jitter,)
        is_eval[f"dev_{i}"] = False
        rows[f"eval_{i}"] = (10.0 + jitter,)
        is_eval[f"eval_{i}"] = True
    return FeatureFrame(feature_keys=(key,), rows=rows, is_eval=is_eval)


def _non_separating_frame(n_per_pool: int = 40, *, key: str = "f0") -> FeatureFrame:
    """A frame whose feature is identically distributed across pools -> ~chance."""
    pattern = [0.0, 1.0, 2.0, 3.0, 4.0]
    rows = {}
    is_eval = {}
    for i in range(n_per_pool):
        val = pattern[i % len(pattern)]
        rows[f"dev_{i}"] = (val,)
        is_eval[f"dev_{i}"] = False
        rows[f"eval_{i}"] = (val,)
        is_eval[f"eval_{i}"] = True
    return FeatureFrame(feature_keys=(key,), rows=rows, is_eval=is_eval)


def _base_frame(n_per_pool: int = 40) -> FeatureFrame:
    """A benign, non-drifting base (foundation) feature view over the same pairs."""
    return _non_separating_frame(n_per_pool, key="base0")


def _calibration(threshold: float = 0.75) -> Calibration:
    """A minimal TRUSTED LB_CALIBRATED calibration carrier (task 7.1 makes the real one)."""
    return Calibration(
        threshold=threshold,
        threshold_kind="LB_CALIBRATED",
        trusted=True,
        separation=SeparationCheck(max_pass_drift=0.6, min_fail_drift=0.7, separable=True),
        null_surfaced=False,
        projection=None,
    )


# --------------------------------------------------------------------------- #
# (a) drift PASS -> holdout trustworthy, RETRAIN trusted
# --------------------------------------------------------------------------- #
def test_sequence_drift_pass_makes_holdout_trustworthy():
    """A non-drifting feature set PASSes the gate, so its holdout is trustworthy."""
    calib = _calibration()
    result = evaluate_sequence(_non_separating_frame(), calib)

    assert isinstance(result, GateSequenceResult)
    assert result.drift.verdict == "PASS"
    assert result.drift_passed is True
    assert result.holdout_trustworthy is True
    # A plain sequence evaluation makes no RETRAIN claim.
    assert result.retrain_trusted is None


def test_retrain_trusted_when_combined_features_pass():
    """A RETRAIN whose COMBINED base+layer features do not drift is trusted (Req 4.2)."""
    calib = _calibration()
    base = _base_frame()
    layer = _non_separating_frame(key="layer0")

    result = evaluate_retrain(base, layer, calib)

    assert result.drift.verdict == "PASS"
    assert result.holdout_trustworthy is True
    assert result.retrain_trusted is True


# --------------------------------------------------------------------------- #
# (b) drift FAIL -> holdout NOT trustworthy, RETRAIN NOT trusted
# --------------------------------------------------------------------------- #
def test_sequence_drift_fail_makes_holdout_untrustworthy():
    """A drifting feature set FAILs the gate, so its holdout is NOT trustworthy (Rule 17)."""
    calib = _calibration()
    result = evaluate_sequence(_separating_frame(), calib)

    assert result.drift.verdict == "FAIL"
    assert result.drift_passed is False
    assert result.holdout_trustworthy is False
    assert result.retrain_trusted is None


def test_retrain_not_trusted_when_combined_features_fail():
    """A RETRAIN whose COMBINED features drift is never trusted, even with a base that doesn't (Req 4.2)."""
    calib = _calibration()
    base = _base_frame()
    layer = _separating_frame(key="layer0")  # the layer drags the combined set into drift

    result = evaluate_retrain(base, layer, calib)

    assert result.drift.verdict == "FAIL"
    assert result.holdout_trustworthy is False
    assert result.retrain_trusted is False
