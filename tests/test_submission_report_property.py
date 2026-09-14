"""Property test for Property 19 (task 16.2, design ``submission/gate.py``).

Property 19 (design.md): *The submission report is complete.* For any candidate
proposed for submission, the emitted report SHALL contain all of: layers ON,
holdout AP, measured drift, gate verdict, LB_Projection, and whether the
projection regime is verified.

This exercises the completeness contract of
:func:`poker_collusion.tuning_harness.submission.gate.submission_report`: for an
arbitrary candidate (a random subset of optional layers ON), an arbitrary
holdout AP in [0, 1], an arbitrary Drift_Gate verdict (PASS/FAIL) with an
adversarial AUC, and a calibration carrying a ProjectionFit, the returned
:class:`SubmissionReport` must have every Req 7.1 field populated with the right
type and consistent with the inputs. No field is ``None`` or missing.

Completeness is asserted structurally (each field present + correctly typed)
and by consistency with the inputs (``layers_on`` reflects the candidate,
``holdout_ap`` echoes the input, ``measured_drift`` echoes the adversarial AUC,
``gate_verdict`` echoes the drift verdict, ``projection`` is a proper
``Projection`` with float point/uncertainty, and ``regime_verified`` mirrors the
projection's flag). It does NOT assert the *recommendation* logic -- that is
Property 20 (task 16.3).

Requirements: 7.1.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.models import (
    Calibration,
    Candidate,
    CandidateConfig,
    DriftResult,
    Projection,
    ProjectionFit,
    RankVector,
    SeparationCheck,
    SubmissionReport,
)
from poker_collusion.tuning_harness.submission.gate import submission_report


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
#: The optional layers that may be switched ON (foundation is always on and is
#: not part of this set); a candidate turns ON an arbitrary subset of these.
_OPTIONAL_LAYERS = (
    "board_equity",
    "whipsaw",
    "iso",
    "directional",
    "negative_space",
    "coordinated_isolation",
)

_probs = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def _candidates(draw: st.DrawFn) -> Candidate:
    """A candidate with a random subset of optional layers ON.

    ``layers_reported`` is drawn independently as either the best-effort composer
    report (the sorted ON layers) or ``None`` (a reporting failure), so both
    reporting branches of ``submission_report`` are exercised.
    """

    layers = draw(st.sets(st.sampled_from(_OPTIONAL_LAYERS)))
    frozen = frozenset(layers)
    cfg = CandidateConfig(layers_on=frozen)
    ranking = RankVector(pair_ids=("p1", "p2", "p3"))
    reported = draw(st.sampled_from([tuple(sorted(frozen)), None]))
    return Candidate(config=cfg, ranking=ranking, layers_reported=reported)


@st.composite
def _drifts(draw: st.DrawFn) -> DriftResult:
    """A Drift_Gate result with an arbitrary verdict and adversarial AUC."""

    verdict = draw(st.sampled_from(["PASS", "FAIL"]))
    auc = draw(_probs)
    return DriftResult(
        adversarial_auc=auc,
        verdict=verdict,
        threshold=0.74,
        threshold_kind="LB_CALIBRATED",
    )


@st.composite
def _calibrations(draw: st.DrawFn) -> Calibration:
    """A calibration carrying an arbitrary (valid) ProjectionFit.

    The fit range is an ordered pair drawn from [0, 1] so that ``project_lb`` can
    classify the drawn holdout AP as in- or out-of-regime; both outcomes must
    still yield a complete report.
    """

    a = draw(_probs)
    b = draw(_probs)
    fit_min_ap, fit_max_ap = (a, b) if a <= b else (b, a)
    fit = ProjectionFit(
        slope=draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
        intercept=draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
        resid_std=draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)),
        fit_min_ap=fit_min_ap,
        fit_max_ap=fit_max_ap,
        n_points=draw(st.integers(min_value=2, max_value=10)),
    )
    sep = SeparationCheck(max_pass_drift=0.68, min_fail_drift=0.80, separable=True)
    return Calibration(
        threshold=0.74,
        threshold_kind="LB_CALIBRATED",
        trusted=True,
        separation=sep,
        null_surfaced=False,
        projection=fit,
    )


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 19: The submission report is complete
# --------------------------------------------------------------------------- #
# Validates: Requirements 7.1
@settings(max_examples=200)
@given(
    candidate=_candidates(),
    holdout_ap=_probs,
    drift=_drifts(),
    calib=_calibrations(),
)
def test_submission_report_is_complete(
    candidate: Candidate,
    holdout_ap: float,
    drift: DriftResult,
    calib: Calibration,
) -> None:
    report = submission_report(candidate, holdout_ap=holdout_ap, drift=drift, calib=calib)

    # The report itself exists and is the right kind.
    assert isinstance(report, SubmissionReport)

    # --- layers ON: a tuple, and consistent with the candidate ------------- #
    assert isinstance(report.layers_on, tuple)
    assert all(isinstance(layer, str) for layer in report.layers_on)
    expected_layers = (
        tuple(candidate.layers_reported)
        if candidate.layers_reported is not None
        else tuple(sorted(candidate.config.layers_on))
    )
    assert report.layers_on == expected_layers

    # --- holdout AP: present, a float, echoes the input -------------------- #
    assert report.holdout_ap is not None
    assert isinstance(report.holdout_ap, float)
    assert report.holdout_ap == holdout_ap

    # --- measured drift: present, a float, echoes the adversarial AUC ------ #
    assert report.measured_drift is not None
    assert isinstance(report.measured_drift, float)
    assert report.measured_drift == drift.adversarial_auc

    # --- gate verdict: present, echoes the drift verdict ------------------- #
    assert report.gate_verdict is not None
    assert report.gate_verdict == drift.verdict

    # --- LB_Projection: present, a Projection with float point/uncertainty - #
    assert report.projection is not None
    assert isinstance(report.projection, Projection)
    assert isinstance(report.projection.point, float)
    assert isinstance(report.projection.uncertainty, float)

    # --- regime-verified flag: present, a bool, mirrors the projection ----- #
    assert report.regime_verified is not None
    assert isinstance(report.regime_verified, bool)
    assert report.regime_verified == report.projection.regime_verified

    # recommend_submit is also populated (a bool) -- the report is fully built.
    assert isinstance(report.recommend_submit, bool)
