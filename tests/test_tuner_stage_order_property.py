"""Hypothesis property test for tuning stage order (task 13.2, design Property 14).

# Feature: poker-layered-tuning, Property 14: Tuning executes drift, then baseline, then layers, in that order

Property statement (design.md): For any candidate configuration, the tuning
routine SHALL invoke its stages in exactly the order: resolve feature-set drift,
then Baseline_Tuning, then Layer_Tuning.

The property is checked over randomly-generated candidate configs: any subset of
the optional layers (board_equity, directional, iso, whipsaw, negative_space) is
switched ON, each composed with a RANK_BLEND spec at weight ``0.0`` so no
``layer_ranking_fn`` is required and scoring stays trivial (the floor is a
reachable, no-op point at every layer). ``score_fn`` returns a constant and
``drift_fn`` returns a PASS ``DriftResult``; the in-regime ``_FIT`` makes the
projection deterministic.

For EVERY generated config we assert
``report.stage_order == (STAGE_RESOLVE_DRIFT, STAGE_BASELINE_TUNING,
STAGE_LAYER_TUNING)``. We additionally record the order in which the callbacks
fire and assert the FIRST callback invoked is ``drift`` (drift resolved first),
reinforcing the "drift first" half of the ordering.

Validates: Requirements 5.1.
"""

from __future__ import annotations

from typing import List

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
from poker_collusion.tuning_harness.tuning.tuner import (
    STAGE_BASELINE_TUNING,
    STAGE_LAYER_TUNING,
    STAGE_RESOLVE_DRIFT,
    tune_with_report,
)

# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirroring poker/tests/test_tuner.py)
# --------------------------------------------------------------------------- #
_CALIB = Calibration(
    threshold=0.70,
    threshold_kind="LB_CALIBRATED",
    trusted=True,
    separation=SeparationCheck(max_pass_drift=0.644, min_fail_drift=0.865, separable=True),
    null_surfaced=False,
)

#: A trivial in-regime projection fit so project_lb always verifies the regime
#: and the tune run stays deterministic across generated configs.
_FIT = ProjectionFit(
    slope=1.4235,
    intercept=-0.1414,
    resid_std=0.0022,
    fit_min_ap=0.0,
    fit_max_ap=1.0,
    n_points=5,
)

#: The optional layers the harness registers (registry.py); foundation is always
#: on and never listed here.
_OPTIONAL_LAYERS = ("board_equity", "directional", "iso", "whipsaw", "negative_space")


def _pass_drift(auc: float = 0.60) -> DriftResult:
    return DriftResult(
        adversarial_auc=auc,
        verdict="PASS",
        threshold=0.70,
        threshold_kind="LB_CALIBRATED",
    )


#: Generator: any subset of the optional layers ON, each with a RANK_BLEND spec
#: at weight 0.0 (no layer_ranking_fn needed, scoring trivial).
def _candidate_configs() -> st.SearchStrategy[CandidateConfig]:
    def _build(names: List[str]) -> CandidateConfig:
        layers_on = frozenset(names)
        composition = {
            name: CompositionSpec(mode="RANK_BLEND", weight=0.0) for name in names
        }
        return CandidateConfig(layers_on=layers_on, composition=composition)

    return st.lists(
        st.sampled_from(_OPTIONAL_LAYERS), unique=True, min_size=0, max_size=len(_OPTIONAL_LAYERS)
    ).map(_build)


# --------------------------------------------------------------------------- #
# Property 14 -- Req 5.1
# --------------------------------------------------------------------------- #
@given(cfg=_candidate_configs())
@settings(max_examples=200)
def test_tuning_executes_drift_then_baseline_then_layers(cfg: CandidateConfig) -> None:
    # Feature: poker-layered-tuning, Property 14: Tuning executes drift, then baseline, then layers, in that order
    call_order: List[str] = []

    def score_fn(_cfg: CandidateConfig) -> float:
        call_order.append("score")
        return 0.38  # trivial constant holdout

    def drift_fn(_cfg: CandidateConfig) -> DriftResult:
        call_order.append("drift")
        return _pass_drift()

    _tuned, report = tune_with_report(
        cfg, _CALIB, score_fn=score_fn, drift_fn=drift_fn, projection_fit=_FIT
    )

    # The authoritative record: stages ran in exactly drift -> baseline -> layers.
    assert report.stage_order == (
        STAGE_RESOLVE_DRIFT,
        STAGE_BASELINE_TUNING,
        STAGE_LAYER_TUNING,
    )

    # Reinforce "drift first": the very first callback invoked resolves drift.
    assert call_order, "tuning must invoke at least one callback"
    assert call_order[0] == "drift"
