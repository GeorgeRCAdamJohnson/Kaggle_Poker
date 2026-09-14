"""Property test 16.3: a drift-failing candidate is never recommended.

One Hypothesis property (design Property 20, Req 7.2 / 8.3): for ANY candidate,
ANY holdout AP across the FULL [0, 1] range (including 0.99+), and ANY drift
result whose verdict is FAIL, :func:`submission_report` MUST set
``recommend_submit`` to False. A drifting feature set's holdout is inverted
(accountability contract Rule 17), so a high holdout number can never rescue a
drift FAIL. This is the universal companion to the example test in
``test_submission_gate.py`` -- it proves the invariant holds across the whole
input space, not just for one hand-picked high holdout value.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.models import (
    Calibration,
    Candidate,
    CandidateConfig,
    DriftResult,
    ProjectionFit,
    RankVector,
    SeparationCheck,
)
from poker_collusion.tuning_harness.submission.gate import submission_report

# The Drift_Gate threshold carried on the calibration below. A FAIL candidate's
# adversarial AUC is drawn strictly ABOVE this so the input is a genuine FAIL.
_THRESHOLD = 0.74

# The optional layers that may be switched ON for a candidate (foundation is
# always on and never listed). The generator picks a random subset.
_OPTIONAL_LAYERS = ("board_equity", "directional", "iso", "whipsaw", "negative_space")


def _calibration(fit: ProjectionFit) -> Calibration:
    """A minimal TRUSTED calibration carrying an explicit projection fit."""

    sep = SeparationCheck(max_pass_drift=0.68, min_fail_drift=0.80, separable=True)
    return Calibration(
        threshold=_THRESHOLD,
        threshold_kind="LB_CALIBRATED",
        trusted=True,
        separation=sep,
        null_surfaced=False,
        projection=fit,
    )


@st.composite
def _driftfail_inputs(draw):
    """Generate (candidate, holdout_ap, drift_FAIL, calibration).

    - candidate: a random subset of optional layers ON.
    - holdout_ap: drawn across the FULL [0, 1] range INCLUDING very high values
      (up to 1.0) to prove a high holdout cannot rescue a drift FAIL.
    - drift: verdict "FAIL" with adversarial_auc strictly ABOVE the threshold.
    - calibration: carries a ProjectionFit whose regime may be in OR out of range
      relative to the drawn holdout_ap (so recommendation cannot lean on regime).
    """

    layers = frozenset(
        draw(st.lists(st.sampled_from(_OPTIONAL_LAYERS), unique=True).map(frozenset))
    )
    candidate = Candidate(
        config=CandidateConfig(layers_on=layers),
        ranking=RankVector(pair_ids=("p1", "p2", "p3")),
        layers_reported=tuple(sorted(layers)),
    )

    # Full [0, 1] holdout AP range, including 0.99+ high values.
    holdout_ap = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))

    # A genuine drift FAIL: adversarial_auc strictly above the calibrated gate.
    adversarial_auc = draw(
        st.floats(min_value=_THRESHOLD + 1e-6, max_value=1.0, allow_nan=False)
    )
    drift = DriftResult(
        adversarial_auc=adversarial_auc,
        verdict="FAIL",
        threshold=_THRESHOLD,
        threshold_kind="LB_CALIBRATED",
    )

    # A projection fit whose regime may or may not cover holdout_ap.
    fit_min = draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False))
    fit_max = draw(st.floats(min_value=fit_min, max_value=1.0, allow_nan=False))
    fit = ProjectionFit(
        slope=1.4235,
        intercept=-0.1414,
        resid_std=0.0022,
        fit_min_ap=fit_min,
        fit_max_ap=fit_max,
        n_points=3,
    )

    return candidate, holdout_ap, drift, _calibration(fit)


# Feature: poker-layered-tuning, Property 20: A drift-failing candidate is never recommended for submission
@settings(max_examples=200)
@given(_driftfail_inputs())
def test_driftfail_candidate_never_recommended(inputs) -> None:
    """Validates: Requirements 7.2, 8.3

    For every drift-FAIL input the report's ``recommend_submit`` is False, even
    when the holdout AP is 0.99+ -- a high holdout can never rescue a drift FAIL.
    """

    candidate, holdout_ap, drift, calib = inputs

    report = submission_report(candidate, holdout_ap=holdout_ap, drift=drift, calib=calib)

    assert drift.verdict == "FAIL"
    assert report.gate_verdict == "FAIL"
    assert report.recommend_submit is False
