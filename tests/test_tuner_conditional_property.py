"""Property test for the two-level tuner's conditional/RETRAIN classification.

Task 13.4 / design Property 13 (Requirement 4.3). One property, one test.

Property 13 (design.md): *For any* layer, the system SHALL record it as
conditional and use RETRAIN if and only if RANK_BLEND reduces holdout AP at every
positive weight AND RETRAIN of the same layer increases holdout AP; where these
conditions do not both hold, no conditional/RETRAIN classification is mandated.

This mirrors the three conditional-classification unit tests in
``test_tuner.py`` (single ON layer ``"iso"``, ``score_fn`` driven by the
``CompositionSpec`` mode/weight, ``drift_fn`` PASS everywhere, in-regime ``_FIT``)
but drives the two load-bearing booleans -- whether RANK_BLEND corrupts at every
positive weight and whether RETRAIN improves -- across the full 2x2 space with
Hypothesis, plus a randomly drawn floor AP, asserting the exact iff.

Requirements: 4.3.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.models import (
    Calibration,
    CandidateConfig,
    CompositionSpec,
    DriftResult,
    ProjectionFit,
    SeparationCheck,
)
from poker_collusion.tuning_harness.tuning.tuner import tune_with_report


# --------------------------------------------------------------------------- #
# Fixtures mirrored from tests/test_tuner.py (in-regime so drift never
# interferes with the classification under test).
# --------------------------------------------------------------------------- #
_CALIB = Calibration(
    threshold=0.70,
    threshold_kind="LB_CALIBRATED",
    trusted=True,
    separation=SeparationCheck(max_pass_drift=0.644, min_fail_drift=0.865, separable=True),
    null_surfaced=False,
)

#: In-regime projection fit so project_lb always verifies the regime and never
#: perturbs the classification path.
_FIT = ProjectionFit(
    slope=1.4235,
    intercept=-0.1414,
    resid_std=0.0022,
    fit_min_ap=0.0,
    fit_max_ap=1.0,
    n_points=5,
)


def _pass_drift(_cfg: CandidateConfig) -> DriftResult:
    # PASS for ALL configs so drift never interferes with the classification
    # (the only variable under test is the blend/retrain holdout arithmetic).
    return DriftResult(
        adversarial_auc=0.60, verdict="PASS", threshold=0.70, threshold_kind="LB_CALIBRATED"
    )


# Feature: poker-layered-tuning, Property 13: Conditional/RETRAIN classification holds exactly when both conditions hold
@settings(max_examples=200)
@given(
    blend_corrupts=st.booleans(),
    retrain_improves=st.booleans(),
    # A floor AP with head-room on both sides so a below-floor and an above-floor
    # score always exist inside [0, 1].
    floor=st.floats(min_value=0.20, max_value=0.80),
)
def test_conditional_retrain_iff_blend_corrupts_and_retrain_improves(
    blend_corrupts: bool, retrain_improves: bool, floor: float
) -> None:
    # Distinct below/above-floor sentinels bracketing the drawn floor.
    below = floor - 0.10  # strictly below the floor
    above = floor + 0.10  # strictly above the floor

    def score_fn(cfg: CandidateConfig) -> float:
        spec = cfg.composition.get("iso")
        if spec is None:
            return floor
        if spec.mode == "RETRAIN":
            # RETRAIN score is above the floor IFF retrain_improves (else <= floor).
            return above if retrain_improves else floor
        if spec.mode == "RANK_BLEND" and spec.weight > 0.0:
            # RANK_BLEND positive-weight score is below the floor IFF
            # blend_corrupts; otherwise it clears the floor at that weight.
            return below if blend_corrupts else above
        # weight 0.0 == floor by construction.
        return floor

    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.5)},
    )
    _tuned, report = tune_with_report(
        cfg, _CALIB, score_fn=score_fn, drift_fn=_pass_drift, projection_fit=_FIT
    )

    decision = report.layer_decisions["iso"]

    # The two conditions are recorded exactly as driven.
    assert decision.rank_blend_corrupting == blend_corrupts
    assert decision.retrain_improves == retrain_improves

    # The iff: conditional IFF both conditions hold.
    expected_conditional = blend_corrupts and retrain_improves
    assert decision.conditional == expected_conditional

    # chosen_mode == "RETRAIN" iff conditional, else "RANK_BLEND".
    assert decision.chosen_mode == ("RETRAIN" if expected_conditional else "RANK_BLEND")
