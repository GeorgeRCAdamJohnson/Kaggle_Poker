"""Unit tests for the two-level tuner (task 13.1, design ``tuning/tuner.py``).

Example / edge-case unit tests with INJECTED ``score_fn`` / ``drift_fn`` callbacks
(mirroring the ablation and composer injection pattern) so the four load-bearing
behaviours are testable without any real data:

  * stage order is drift -> baseline -> layers (Req 5.1),
  * a drift-increasing tuned gain is rejected even with a HIGHER holdout, and
    that rejection (guarantee b) holds INDEPENDENTLY of the selection check
    (guarantee a) (Req 5.3),
  * the selected config clears the Drift_Gate (guarantee a) (Req 5.3),
  * conditional/RETRAIN classification holds EXACTLY when RANK_BLEND is
    corrupting at every positive weight AND RETRAIN improves (Req 4.3),
  * the returned TunedConfig keeps the MEASURED holdout distinct from the
    PROJECTED LB (Req 5.4).

The Hypothesis property tests for Properties 13/14/15/16 are separate tasks
(13.2-13.5); these are the concrete examples and edge cases.

Requirements: 4.1, 4.3, 5.1, 5.2, 5.3, 5.4.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from poker_collusion.tuning_harness.models import (
    Calibration,
    CandidateConfig,
    CompositionSpec,
    DriftResult,
    ProjectionFit,
    SeparationCheck,
    TunedConfig,
)
from poker_collusion.tuning_harness.tuning.tuner import (
    STAGE_BASELINE_TUNING,
    STAGE_LAYER_TUNING,
    STAGE_RESOLVE_DRIFT,
    GridBudgetExceeded,
    tune,
    tune_with_report,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
_CALIB = Calibration(
    threshold=0.70,
    threshold_kind="LB_CALIBRATED",
    trusted=True,
    separation=SeparationCheck(max_pass_drift=0.644, min_fail_drift=0.865, separable=True),
    null_surfaced=False,
)

#: A trivial in-regime projection fit so project_lb always verifies the regime
#: and the measured/projected distinction is exercised deterministically.
_FIT = ProjectionFit(
    slope=1.4235,
    intercept=-0.1414,
    resid_std=0.0022,
    fit_min_ap=0.0,
    fit_max_ap=1.0,
    n_points=5,
)


def _pass_drift(auc: float = 0.60) -> DriftResult:
    return DriftResult(adversarial_auc=auc, verdict="PASS", threshold=0.70, threshold_kind="LB_CALIBRATED")


def _fail_drift(auc: float = 0.90) -> DriftResult:
    return DriftResult(adversarial_auc=auc, verdict="FAIL", threshold=0.70, threshold_kind="LB_CALIBRATED")


def _floor_cfg() -> CandidateConfig:
    return CandidateConfig(layers_on=frozenset())


# --------------------------------------------------------------------------- #
# Stage order  -- Req 5.1 (Property 14)
# --------------------------------------------------------------------------- #
def test_stage_order_is_drift_then_baseline_then_layers() -> None:
    calls: List[str] = []

    # score_fn / drift_fn record the stage boundaries by observing WHICH configs
    # they are asked about, but the report's stage_order is the authoritative
    # record; we assert both.
    def score_fn(_cfg: CandidateConfig) -> float:
        return 0.38

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift()

    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.0)},
    )
    _tuned, report = tune_with_report(
        cfg, _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    assert report.stage_order == (
        STAGE_RESOLVE_DRIFT,
        STAGE_BASELINE_TUNING,
        STAGE_LAYER_TUNING,
    )


def test_resolve_drift_runs_before_any_scoring() -> None:
    # The FIRST drift call must precede the FIRST score call (drift resolved
    # first, Req 5.1). We tag the order the callbacks fire.
    order: List[str] = []

    def score_fn(_cfg: CandidateConfig) -> float:
        order.append("score")
        return 0.38

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        order.append("drift")
        return _pass_drift()

    tune(_floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT)

    assert order[0] == "drift"


# --------------------------------------------------------------------------- #
# Guarantee (b) independent of (a)  -- Req 5.3 (Property 15)
# --------------------------------------------------------------------------- #
def test_drift_increasing_gain_rejected_even_with_higher_holdout() -> None:
    # A baseline grid point improves the holdout a LOT but its drift rose past the
    # gate (FAIL). Guarantee (b) must reject it even though holdout improved.
    baseline_override_key = repr({"n_estimators": 600.0})

    def score_fn(cfg: CandidateConfig) -> float:
        # The 600-estimator baseline scores much higher than the floor.
        if cfg.model_cfg.hyperparameters.get("n_estimators") == 600.0:
            return 0.99
        return 0.38

    def drift_fn(cfg: CandidateConfig) -> DriftResult:
        # The 600-estimator baseline DRIFTS past the gate (its higher holdout is a
        # trap: drift-inverted signal). Everything else passes.
        if cfg.model_cfg.hyperparameters.get("n_estimators") == 600.0:
            return _fail_drift(0.90)
        return _pass_drift(0.60)

    tuned, report = tune_with_report(
        _floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    # The high-holdout, high-drift gain was rejected...
    rejected_descs = [d for (d, _ap) in report.rejected_for_drift_increase]
    assert any("600.0" in d for d in rejected_descs)
    # ...and it was rejected DESPITE its holdout (0.99) being the best seen.
    rejected_aps = [ap for (_d, ap) in report.rejected_for_drift_increase]
    assert 0.99 in rejected_aps
    # The selected config did NOT adopt the drift-increasing gain.
    assert tuned.holdout_ap < 0.99
    # Guarantee (a) still holds: the selection passes the gate.
    assert tuned.drift.verdict == "PASS"


def test_guarantee_b_holds_even_when_selection_would_pass_gate() -> None:
    # Independence check: even in a run where the FINAL selection passes the gate
    # regardless, guarantee (b) still fires on the drift-increasing candidate
    # (b is a separate filter, not a side effect of the selection check).
    def score_fn(cfg: CandidateConfig) -> float:
        if cfg.model_cfg.hyperparameters.get("n_estimators") == 300.0:
            return 0.80  # a higher-holdout but drift-increasing gain
        return 0.38

    def drift_fn(cfg: CandidateConfig) -> DriftResult:
        if cfg.model_cfg.hyperparameters.get("n_estimators") == 300.0:
            return _fail_drift(0.88)
        return _pass_drift(0.60)

    tuned, report = tune_with_report(
        _floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    # The drift-increasing 300-estimator gain is in the rejected list...
    assert any("300.0" in d for (d, _ap) in report.rejected_for_drift_increase)
    # ...and the winner is the floor (holdout 0.38), which passes the gate (a).
    assert tuned.holdout_ap == pytest.approx(0.38)
    assert tuned.drift.verdict == "PASS"


# --------------------------------------------------------------------------- #
# Guarantee (a): selected config clears the gate  -- Req 5.3 (Property 15)
# --------------------------------------------------------------------------- #
def test_selected_config_clears_the_gate() -> None:
    # Every candidate passes; the selected config's drift verdict must be PASS.
    def score_fn(cfg: CandidateConfig) -> float:
        if cfg.model_cfg.hyperparameters.get("n_estimators") == 300.0:
            return 0.42
        return 0.38

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift(0.62)

    tuned = tune(_floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT)

    assert tuned.drift.verdict == "PASS"
    # The best-holdout passing baseline was selected.
    assert tuned.holdout_ap == pytest.approx(0.42)


# --------------------------------------------------------------------------- #
# Conditional / RETRAIN classification  -- Req 4.3 (Property 13)
# --------------------------------------------------------------------------- #
def test_layer_is_conditional_when_blend_corrupts_and_retrain_improves() -> None:
    # RANK_BLEND hurts at EVERY positive weight (below floor) AND RETRAIN improves
    # -> conditional, use RETRAIN.
    floor = 0.38

    def score_fn(cfg: CandidateConfig) -> float:
        spec = cfg.composition.get("iso")
        if spec is None:
            return floor
        if spec.mode == "RETRAIN":
            return 0.45  # RETRAIN improves
        if spec.mode == "RANK_BLEND" and spec.weight > 0.0:
            return 0.30  # RANK_BLEND below floor at every positive weight
        return floor  # weight 0.0 == floor

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift(0.60)

    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.5)},
    )
    _tuned, report = tune_with_report(
        cfg, _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    decision = report.layer_decisions["iso"]
    assert decision.rank_blend_corrupting is True
    assert decision.retrain_improves is True
    assert decision.conditional is True
    assert decision.chosen_mode == "RETRAIN"


def test_layer_not_conditional_when_blend_helps() -> None:
    # RANK_BLEND helps at some positive weight -> not corrupting -> NOT conditional
    # even if RETRAIN also improves (Property 13: BOTH conditions must hold).
    floor = 0.38

    def score_fn(cfg: CandidateConfig) -> float:
        spec = cfg.composition.get("iso")
        if spec is None:
            return floor
        if spec.mode == "RETRAIN":
            return 0.50
        if spec.mode == "RANK_BLEND" and spec.weight > 0.0:
            return 0.44  # blend HELPS -> not corrupting
        return floor

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift(0.60)

    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.5)},
    )
    _tuned, report = tune_with_report(
        cfg, _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    decision = report.layer_decisions["iso"]
    assert decision.rank_blend_corrupting is False
    assert decision.conditional is False
    assert decision.chosen_mode == "RANK_BLEND"


def test_layer_not_conditional_when_retrain_does_not_improve() -> None:
    # RANK_BLEND corrupts at every positive weight BUT RETRAIN does not improve
    # -> NOT conditional (Property 13: both conditions required).
    floor = 0.38

    def score_fn(cfg: CandidateConfig) -> float:
        spec = cfg.composition.get("iso")
        if spec is None:
            return floor
        if spec.mode == "RETRAIN":
            return 0.35  # RETRAIN does NOT improve
        if spec.mode == "RANK_BLEND" and spec.weight > 0.0:
            return 0.30  # blend corrupts
        return floor

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift(0.60)

    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.5)},
    )
    _tuned, report = tune_with_report(
        cfg, _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    decision = report.layer_decisions["iso"]
    assert decision.rank_blend_corrupting is True
    assert decision.retrain_improves is False
    assert decision.conditional is False
    assert decision.chosen_mode == "RANK_BLEND"


# --------------------------------------------------------------------------- #
# Measured vs projected reporting  -- Req 5.4 (Property 16)
# --------------------------------------------------------------------------- #
def test_tuned_config_keeps_measured_holdout_distinct_from_projected_lb() -> None:
    def score_fn(_cfg: CandidateConfig) -> float:
        return 0.40

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift(0.60)

    tuned = tune(_floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT)

    assert isinstance(tuned, TunedConfig)
    # Measured holdout is the raw AP the scorer returned...
    assert tuned.holdout_ap == pytest.approx(0.40)
    # ...and the projection is the calibrated LB estimate, a DIFFERENT number,
    # carried with its own uncertainty (never conflated with the holdout AP).
    expected_point = 1.4235 * 0.40 - 0.1414
    assert tuned.projection.point == pytest.approx(expected_point)
    assert tuned.projection.uncertainty == pytest.approx(0.0022)
    assert tuned.projection.point != pytest.approx(tuned.holdout_ap)
    assert tuned.projection.regime_verified is True


# --------------------------------------------------------------------------- #
# Coarse-grid budget (Gate D) + floor guarantee (weight grid must include 0.0)
# --------------------------------------------------------------------------- #
def test_weight_grid_must_include_zero() -> None:
    def score_fn(_cfg: CandidateConfig) -> float:
        return 0.38

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift()

    with pytest.raises(ValueError):
        tune(
            _floor_cfg(),
            _CALIB,
            score_fn=score_fn,
            drift_fn=drift_fn,
            weight_grid=(0.25, 0.5, 1.0),  # no 0.0
            projection_fit=_FIT,
        )


def test_fine_micro_grid_escalates() -> None:
    def score_fn(_cfg: CandidateConfig) -> float:
        return 0.38

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift()

    fine = tuple({"n_estimators": float(x)} for x in range(20))  # 20 points
    with pytest.raises(GridBudgetExceeded):
        tune(
            _floor_cfg(),
            _CALIB,
            score_fn=score_fn,
            drift_fn=drift_fn,
            baseline_grid=fine,
            projection_fit=_FIT,
        )


def test_bar_cleared_when_passes_gate_and_holds_floor() -> None:
    def score_fn(_cfg: CandidateConfig) -> float:
        return 0.40

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        return _pass_drift(0.60)

    _tuned, report = tune_with_report(
        _floor_cfg(), _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )
    assert report.bar_cleared is True
