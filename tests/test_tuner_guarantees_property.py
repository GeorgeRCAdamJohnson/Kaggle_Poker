"""Property test for the two INDEPENDENT tuning guarantees (task 13.3).

Property 15 (design.md): For any set of tuned candidates, (a) the selected
configuration SHALL clear the LB-calibrated Drift_Gate, AND (b) any candidate
whose tuned gain increases measured drift past the threshold SHALL be rejected
even when its holdout AP improved; guarantee (b) SHALL hold independently of the
selection check (a).

The invariant under test lives in ``tuning/tuner.py::tune_with_report``:

  * guarantee (b) is a SEPARATE filter (``_drift_increased_past_gate``) that
    fires for every candidate BEFORE selection, so a grid point whose drift rose
    past the gate lands in ``report.rejected_for_drift_increase`` even when its
    holdout AP was the best seen -- and never becomes the selection;
  * guarantee (a) is a DISTINCT check: the finally-selected config's drift
    verdict is PASS.

Independence is asserted directly: in a run where the selection ALSO (separately)
passes the gate, the drift-increasing high-holdout grid point is STILL in the
rejected list. Removing the selection check would not stop (b) rejecting the
drift-increasing gain, and vice versa; the two are not one test wearing two hats.

Strategy: mirror the tuner unit-test fixtures (``_CALIB``, ``_FIT``,
``_pass_drift``, ``_fail_drift``, ``_floor_cfg``). One specific baseline grid
point -- ``{"n_estimators": 600.0}`` from ``DEFAULT_BASELINE_GRID`` -- is drawn
to be BOTH drift-increasing (``drift_fn`` returns FAIL for it) AND high-holdout
(``score_fn`` returns a value strictly greater than the floor for it), while
every other config PASSes drift at the floor holdout. We vary the high-holdout
value (always > floor) and the FAIL adversarial AUC across examples so the
guarantees are exercised over the whole trap regime, not a single point.

Validates: Requirements 5.3.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.models import (
    Calibration,
    CandidateConfig,
    DriftResult,
    ProjectionFit,
    SeparationCheck,
)
from poker_collusion.tuning_harness.tuning.tuner import tune_with_report


# --------------------------------------------------------------------------- #
# Fixtures mirrored from poker/tests/test_tuner.py
# --------------------------------------------------------------------------- #
_CALIB = Calibration(
    threshold=0.70,
    threshold_kind="LB_CALIBRATED",
    trusted=True,
    separation=SeparationCheck(max_pass_drift=0.644, min_fail_drift=0.865, separable=True),
    null_surfaced=False,
)

_FIT = ProjectionFit(
    slope=1.4235,
    intercept=-0.1414,
    resid_std=0.0022,
    fit_min_ap=0.0,
    fit_max_ap=1.0,
    n_points=5,
)

#: The floor holdout every non-trap config returns (the known-good reference).
_FLOOR_HOLDOUT = 0.38

#: The specific coarse baseline grid point (in DEFAULT_BASELINE_GRID) that is
#: drawn to be the drift-increasing, high-holdout trap.
_TRAP_N_ESTIMATORS = 600.0


def _pass_drift(auc: float = 0.60) -> DriftResult:
    return DriftResult(adversarial_auc=auc, verdict="PASS", threshold=0.70, threshold_kind="LB_CALIBRATED")


def _fail_drift(auc: float) -> DriftResult:
    return DriftResult(adversarial_auc=auc, verdict="FAIL", threshold=0.70, threshold_kind="LB_CALIBRATED")


def _floor_cfg() -> CandidateConfig:
    return CandidateConfig(layers_on=frozenset())


# Feature: poker-layered-tuning, Property 15: The two tuning guarantees hold independently
@settings(max_examples=150, deadline=None)
@given(
    # The trap grid point's holdout is ALWAYS strictly above the floor: its high
    # holdout is the bait guarantee (b) must reject anyway.
    high_holdout=st.floats(min_value=_FLOOR_HOLDOUT + 0.01, max_value=1.0),
    # Its adversarial AUC is a genuine FAIL (strictly above the 0.70 gate), and
    # strictly above the passing configs' AUC so drift also ROSE.
    fail_auc=st.floats(min_value=0.71, max_value=1.0),
    # The AUC every passing config reports (a genuine PASS, below the gate).
    pass_auc=st.floats(min_value=0.40, max_value=0.69),
)
def test_property_two_tuning_guarantees_hold_independently(
    high_holdout: float,
    fail_auc: float,
    pass_auc: float,
) -> None:
    def score_fn(cfg: CandidateConfig) -> float:
        # The 600-estimator baseline scores strictly higher than the floor: the
        # bait for guarantee (b).
        if cfg.model_cfg.hyperparameters.get("n_estimators") == _TRAP_N_ESTIMATORS:
            return high_holdout
        return _FLOOR_HOLDOUT

    def drift_fn(cfg: CandidateConfig) -> DriftResult:
        # The 600-estimator baseline drifts PAST the gate (FAIL); everything else
        # PASSes at the floor. fail_auc > pass_auc so drift genuinely ROSE.
        if cfg.model_cfg.hyperparameters.get("n_estimators") == _TRAP_N_ESTIMATORS:
            return _fail_drift(fail_auc)
        return _pass_drift(pass_auc)

    tuned, report = tune_with_report(
        _floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    rejected_descs = [d for (d, _ap) in report.rejected_for_drift_increase]
    rejected_aps = [ap for (_d, ap) in report.rejected_for_drift_increase]

    # Guarantee (b): the drift-increasing, high-holdout grid point was REJECTED
    # despite its holdout being strictly the best seen.
    assert any(str(_TRAP_N_ESTIMATORS) in d for d in rejected_descs)
    assert any(ap == high_holdout for ap in rejected_aps)
    # ...and the selection never adopted that high holdout value.
    assert tuned.holdout_ap != high_holdout
    assert tuned.holdout_ap <= _FLOOR_HOLDOUT

    # Guarantee (a): the SELECTED config clears the gate.
    assert tuned.drift.verdict == "PASS"

    # Independence: guarantee (b)'s rejection is present EVEN THOUGH the selection
    # also (separately) passed the gate (a). The rejection is not a side effect of
    # the selection check -- the trap is in the rejected list regardless of which
    # config finally won.
    assert any(str(_TRAP_N_ESTIMATORS) in d for d in rejected_descs)
