"""Property test for Property 16 (task 13.5, design ``tuning/tuner.py``).

Property 16 (design.md): *Reports distinguish measured holdout from projected
leaderboard.* For any tuned configuration, the report SHALL contain the measured
holdout AP and the LB_Projection with its uncertainty as DISTINCT fields, so
measured and projected values are never conflated.

This exercises the reporting contract of
:func:`poker_collusion.tuning_harness.tuning.tuner.tune`: given an injected
``score_fn`` that returns a measured holdout AP drawn from WITHIN the projection
fit's regime and a ``drift_fn`` that PASSes, the returned :class:`TunedConfig`
must carry (a) ``holdout_ap`` equal to the measured AP, (b) ``projection.point``
equal to ``slope * ap + intercept`` (the projected LB, a DIFFERENT quantity),
(c) ``projection.uncertainty`` equal to the fit's ``resid_std``, and (d)
``projection.regime_verified is True`` because the AP is in-regime. The measured
field is never overwritten by the projected value: both stay separately
accessible on the tuned config.

Fixtures mirror ``tests/test_tuner.py``: the wide-regime ``_FIT`` (slope 1.4235,
intercept -0.1414, resid_std 0.0022, regime 0.0..1.0), the calibrated ``_CALIB``,
the PASSing ``_pass_drift``, and the ``_floor_cfg`` (no optional layers).

Requirements: 5.4.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.models import (
    Calibration,
    CandidateConfig,
    DriftResult,
    ProjectionFit,
    SeparationCheck,
)
from poker_collusion.tuning_harness.tuning.tuner import tune


# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirroring tests/test_tuner.py)
# --------------------------------------------------------------------------- #
_CALIB = Calibration(
    threshold=0.70,
    threshold_kind="LB_CALIBRATED",
    trusted=True,
    separation=SeparationCheck(max_pass_drift=0.644, min_fail_drift=0.865, separable=True),
    null_surfaced=False,
)

#: A WIDE-regime projection fit: because the regime is [0.0, 1.0], any AP drawn in
#: that range is in-regime, so ``project_lb`` verifies the regime deterministically
#: and the measured/projected distinction is exercised without extrapolation.
_FIT = ProjectionFit(
    slope=1.4235,
    intercept=-0.1414,
    resid_std=0.0022,
    fit_min_ap=0.0,
    fit_max_ap=1.0,
    n_points=5,
)


def _pass_drift(auc: float = 0.60) -> DriftResult:
    return DriftResult(
        adversarial_auc=auc, verdict="PASS", threshold=0.70, threshold_kind="LB_CALIBRATED"
    )


def _floor_cfg() -> CandidateConfig:
    return CandidateConfig(layers_on=frozenset())


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 16: Reports distinguish measured
# holdout from projected leaderboard
# --------------------------------------------------------------------------- #
# Validates: Requirements 5.4
@settings(max_examples=200)
@given(
    # Draw a measured holdout AP strictly inside the wide _FIT regime [0.0, 1.0]
    # so the projection is regime-verified (no extrapolation).
    measured_ap=st.floats(
        min_value=_FIT.fit_min_ap,
        max_value=_FIT.fit_max_ap,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_report_distinguishes_measured_holdout_from_projected_lb(measured_ap: float) -> None:
    # score_fn reports the drawn measured holdout AP; drift_fn PASSes so the floor
    # config is the selected config and its measured AP flows straight through.
    def score_fn(_cfg: CandidateConfig) -> float:
        return measured_ap

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift()

    tuned = tune(
        _floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    # (a) The MEASURED holdout AP is carried verbatim -- never overwritten by the
    #     projected value.
    assert tuned.holdout_ap == pytest.approx(measured_ap)

    # (b) The PROJECTED LB is a DIFFERENT quantity: slope * ap + intercept.
    expected_point = _FIT.slope * measured_ap + _FIT.intercept
    assert tuned.projection.point == pytest.approx(expected_point)

    # (c) The projection carries its own uncertainty (the fit's residual std).
    assert tuned.projection.uncertainty == pytest.approx(_FIT.resid_std)

    # (d) In-regime draw -> the projection is regime-verified.
    assert tuned.projection.regime_verified is True

    # Measured and projected are SEPARATELY accessible distinct fields: the
    # measured holdout is not the projected point (except at the single
    # coincidental crossing where slope*ap + intercept == ap).
    crossing = _FIT.intercept / (1.0 - _FIT.slope)  # ap where projection == ap
    if abs(measured_ap - crossing) > 1e-6:
        assert tuned.projection.point != pytest.approx(tuned.holdout_ap)
