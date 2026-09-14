"""Property test for RETRAIN trust gating on the combined feature set (task 9.2).

Property 12 (design.md): For any RETRAIN composition, the result SHALL be marked
trusted only if the combined (base + layer) feature set clears the LB-calibrated
Drift_Gate; a RETRAIN whose combined features FAIL the Drift_Gate SHALL never be
trusted.

The invariant under test is the biconditional wired in
``gate/sequence.py::evaluate_retrain``: ``retrain_trusted is True`` **iff** the
combined feature set PASSes the Drift_Gate. Equivalently, a drift ``FAIL`` can
never yield ``retrain_trusted`` True. This is why a RETRAIN that improves the
holdout but whose combined features DRIFT is never believed (accountability
contract Rule 17: on drifting features the holdout is inverted).

Strategy: build a benign, non-drifting base (foundation) view over a fixed set
of shared pairs, then draw a boolean ``combined_drifts``. When True the layer
strongly separates the eval pool from the dev pool (so the COMBINED base+layer
set drifts -> Drift_Gate FAIL); when False the layer is identically distributed
across pools (so the combined set does not drift -> Drift_Gate PASS). We use a
TRUSTED LB_CALIBRATED calibration with a fixed 0.75 threshold, exactly as the
sequence unit tests do, so the gate decision is the calibrated bar and not the
provisional fallback. Frames are kept small (few rows per pool) because
``adversarial_auc`` trains a classifier per example.

Validates: Requirements 4.2.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.gate.sequence import evaluate_retrain
from poker_collusion.tuning_harness.models import (
    Calibration,
    FeatureFrame,
    SeparationCheck,
)


# A TRUSTED, LB_CALIBRATED calibration with a fixed threshold (mirrors the
# sequence unit-test fixture): the gate decides against 0.75, not the provisional
# fallback, so drift PASS/FAIL is a genuine calibrated verdict.
_CALIB_THRESHOLD = 0.75


def _calibration() -> Calibration:
    return Calibration(
        threshold=_CALIB_THRESHOLD,
        threshold_kind="LB_CALIBRATED",
        trusted=True,
        separation=SeparationCheck(max_pass_drift=0.6, min_fail_drift=0.7, separable=True),
        null_surfaced=False,
        projection=None,
    )


def _base_frame(n_per_pool: int) -> FeatureFrame:
    """A benign, non-drifting foundation view over shared dev/eval pairs.

    Both pools draw the SAME repeating pattern, so the base feature carries no
    dev-vs-eval signal on its own. ``_combine_features`` keeps only pairs present
    in both frames and carries ``is_eval`` from the base, so the layer frame must
    reuse these exact pair ids.
    """
    pattern = [0.0, 1.0, 2.0, 3.0, 4.0]
    rows: dict[str, tuple[float, ...]] = {}
    is_eval: dict[str, bool] = {}
    for i in range(n_per_pool):
        val = pattern[i % len(pattern)]
        rows[f"dev_{i}"] = (val,)
        is_eval[f"dev_{i}"] = False
        rows[f"eval_{i}"] = (val,)
        is_eval[f"eval_{i}"] = True
    return FeatureFrame(feature_keys=("base0",), rows=rows, is_eval=is_eval)


def _separating_layer(n_per_pool: int) -> FeatureFrame:
    """A layer that cleanly separates eval from dev rows -> combined set DRIFTS.

    Dev rows ~0.0, eval rows ~10.0, over the same pair ids as the base, so a
    dev-vs-eval classifier on the combined frame separates the pools almost
    perfectly (HIGH adversarial AUC, well above 0.75 -> Drift_Gate FAIL). A tiny
    deterministic jitter keeps CV folds non-degenerate.
    """
    rows: dict[str, tuple[float, ...]] = {}
    for i in range(n_per_pool):
        jitter = (i % 5) * 0.01
        rows[f"dev_{i}"] = (0.0 + jitter,)
        rows[f"eval_{i}"] = (10.0 + jitter,)
    return FeatureFrame(feature_keys=("layer0",), rows=rows, is_eval={})


def _non_separating_layer(n_per_pool: int) -> FeatureFrame:
    """A layer identically distributed across pools -> combined set does NOT drift.

    Both pools draw the same repeating pattern, so neither the layer nor the
    combined base+layer set carries dev-vs-eval signal (AUC near chance, below
    0.75 -> Drift_Gate PASS).
    """
    pattern = [4.0, 3.0, 2.0, 1.0, 0.0]
    rows: dict[str, tuple[float, ...]] = {}
    for i in range(n_per_pool):
        val = pattern[i % len(pattern)]
        rows[f"dev_{i}"] = (val,)
        rows[f"eval_{i}"] = (val,)
    return FeatureFrame(feature_keys=("layer0",), rows=rows, is_eval={})


# Feature: poker-layered-tuning, Property 12: A RETRAIN result is trusted only if its combined feature set passes the Drift_Gate
@settings(max_examples=120, deadline=None)
@given(
    combined_drifts=st.booleans(),
    # Small frames: adversarial_auc trains a classifier per example. Enough rows
    # per pool that the grouped CV has representable folds.
    n_per_pool=st.integers(min_value=10, max_value=20),
)
def test_property_retrain_trusted_iff_combined_features_pass(
    combined_drifts: bool,
    n_per_pool: int,
) -> None:
    calib = _calibration()
    base = _base_frame(n_per_pool)
    layer = (
        _separating_layer(n_per_pool)
        if combined_drifts
        else _non_separating_layer(n_per_pool)
    )

    result = evaluate_retrain(base, layer, calib)

    # The invariant: a RETRAIN is trusted IF AND ONLY IF its combined base+layer
    # feature set PASSes the Drift_Gate (Req 4.2). Stated as a biconditional so
    # both directions are exercised across the drawn examples.
    assert result.retrain_trusted is (result.drift.verdict == "PASS")

    # And the never-trusted-on-FAIL half stated explicitly: a drift FAIL can
    # never yield trust, regardless of any holdout gain.
    if result.drift.verdict == "FAIL":
        assert result.retrain_trusted is False
    else:
        assert result.drift.verdict == "PASS"
        assert result.retrain_trusted is True

    # holdout_trustworthy is the same PASS-gated value the RETRAIN trust rides on.
    assert result.holdout_trustworthy is result.retrain_trusted
