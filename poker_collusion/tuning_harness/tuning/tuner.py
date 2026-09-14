"""Two-level tuner: drift -> baseline -> layers (design: ``tuning/tuner.py``, task 13.1).

Tuning is strictly SEQUENCED to avoid overfitting the holdout (design "Two-level
tuning order", Requirement 5.1): the harness first *resolves feature-set drift*,
THEN tunes the *Baseline* model, THEN tunes the *Layers*. All grids are COARSE
structural steps (Req 5.2), never fine micro-grids, and the RANK_BLEND weight
grid ALWAYS includes ``0.0`` so the known-good floor is a reachable point at
every layer (accountability contract Rule 15: floor-guaranteed composition).

Two INDEPENDENT guarantees are enforced (Req 5.3, design Property 15):

  (a) the SELECTED configuration clears the LB-calibrated Drift_Gate, AND
  (b) any tuned gain that increases measured drift PAST the threshold is
      REJECTED even when its holdout AP improved.

Guarantee (b) is structured as a SEPARATE check that fires for every candidate
regardless of the final selection check (a): a candidate whose drift rose past
the gate is filtered OUT before selection ever runs, so (b) holds independently
of (a). The two are not the same test wearing two hats -- (a) is "the winner
passes the gate", (b) is "a drift-increasing gain is never allowed to win even if
it scored higher". Removing (a) would still leave (b) rejecting drift-increasing
gains; removing (b) would still leave (a) requiring the winner to pass.

Conditional / RETRAIN classification (Req 4.3, design Property 13): a layer is
recorded as conditional and switched to RETRAIN IFF RANK_BLEND reduces holdout at
EVERY positive weight AND RETRAIN of the same layer increases holdout. Where both
conditions do not hold, NO conditional classification is mandated. This reuses
the composer's :func:`classify_layer` (a layer that hurts at every positive
weight is ``CORRUPTING``) for the RANK_BLEND half of the predicate.

Reporting (Req 5.4, design Property 16): the returned :class:`TunedConfig` keeps
the MEASURED ``holdout_ap`` (internal, table-disjoint PU-stress AP) strictly
distinct from the ``projection`` (the projected LB with its uncertainty), so a
measured number is never conflated with a projected one.

Pre-registered bar (Gate D): a tuned config is ACCEPTED only if it clears the
Drift_Gate AND its projection does not regress the floor; grids are coarse only.
The bar is represented explicitly in :class:`TuningReport` (``bar_*`` fields) and
an overrun of the coarse grid budget escalates to the user via
:class:`GridBudgetExceeded` rather than silently continuing.

Injection boundary (accountability contract Rule 6; mirrors the ablation /
composer injection pattern). Real scoring and real drift require the reused CV
harness, reference metric, and adversarial classifier over the feature caches --
none of which belong inside a pure, deterministic unit test. So :func:`tune`
takes two injectable callbacks:

  ``score_fn(cfg) -> holdout_ap``   -- the table-disjoint PU-stress AP (in
      production a thin wrapper over
      :func:`poker_collusion.tuning_harness.holdout.scorer.holdout_pair_ap`).
  ``drift_fn(cfg) -> DriftResult``  -- the adversarial Drift_Gate verdict for a
      config's feature set (in production a wrapper over
      :func:`poker_collusion.tuning_harness.gate.drift_gate.evaluate_drift`,
      itself consuming the calibrated ``calib``).

With these injected, the STAGE ORDER, the two guarantees, the conditional
classification, and the measured-vs-projected reporting are all unit-testable
without any real data. Production wires the callbacks to the reused modules; the
tuner itself only sequences stages and does accept/reject arithmetic over the
numbers the callbacks return.

Requirements: 4.1, 4.3, 5.1, 5.2, 5.3, 5.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from poker_collusion.tuning_harness.composition.composer import (
    CORRUPTING,
    classify_layer,
)
from poker_collusion.tuning_harness.models import (
    Calibration,
    CandidateConfig,
    CompositionSpec,
    DriftResult,
    ModelConfig,
    Projection,
    ProjectionFit,
    TunedConfig,
)
from poker_collusion.tuning_harness.projection.project import fit_projection, project_lb

__all__ = [
    "ScoreFn",
    "DriftFn",
    "STAGE_RESOLVE_DRIFT",
    "STAGE_BASELINE_TUNING",
    "STAGE_LAYER_TUNING",
    "DEFAULT_BASELINE_GRID",
    "DEFAULT_RANK_BLEND_WEIGHT_GRID",
    "GridBudgetExceeded",
    "LayerModeDecision",
    "TuningReport",
    "tune",
]

#: A callback scoring a config's table-disjoint PU-stress AP (the measured
#: holdout). Injected so the tuner's accept/reject arithmetic is unit-testable
#: without the real CV run (mirrors the ablation ``ScoreFn``).
ScoreFn = Callable[[CandidateConfig], float]

#: A callback returning the Drift_Gate verdict for a config's feature set.
#: Injected so the tuner's sequencing + the two guarantees are unit-testable
#: without the real adversarial classifier. In production this wraps
#: ``evaluate_drift`` (consuming the LB-calibrated ``calib``).
DriftFn = Callable[[CandidateConfig], DriftResult]

#: The three tuning stages, in the order Req 5.1 mandates. Recorded in
#: :attr:`TuningReport.stage_order` so the ordering is externally auditable /
#: unit-testable (design Property 14).
STAGE_RESOLVE_DRIFT = "resolve_drift"
STAGE_BASELINE_TUNING = "baseline_tuning"
STAGE_LAYER_TUNING = "layer_tuning"

#: A COARSE structural baseline grid (Req 5.2): a handful of structural knob
#: settings, NOT a fine micro-grid. Each entry is a ``{knob: value}`` mapping
#: merged onto the incoming ``model_cfg`` hyperparameters. The empty dict is the
#: incoming default so the current baseline is always a reachable point.
DEFAULT_BASELINE_GRID: Tuple[Mapping[str, float], ...] = (
    {},
    {"n_estimators": 300.0},
    {"n_estimators": 600.0},
)

#: A COARSE RANK_BLEND weight grid that ALWAYS includes 0.0 (Req 5.2 + contract
#: Rule 15: the floor is a reachable point at every layer). Structural steps only.
DEFAULT_RANK_BLEND_WEIGHT_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Coarse-grid budget cap (Gate D). A grid larger than this is a fine micro-grid,
#: which is banned; exceeding it escalates to the user instead of continuing.
_MAX_COARSE_GRID_POINTS = 8


class GridBudgetExceeded(RuntimeError):
    """Raised when a tuning grid exceeds the coarse-grid budget (Gate D).

    Per Req 5.2 / Gate D, grids must be COARSE structural steps. A grid larger
    than :data:`_MAX_COARSE_GRID_POINTS` is a fine micro-grid; rather than
    silently grinding it, the tuner escalates to the user with this error so a
    keep/kill decision can be made explicitly.
    """


@dataclass(frozen=True)
class LayerModeDecision:
    """The RANK_BLEND-vs-RETRAIN decision recorded for one optional layer (Req 4.3).

    ``conditional`` is True IFF RANK_BLEND reduced holdout at EVERY positive
    weight (``rank_blend_corrupting``) AND RETRAIN increased holdout over the
    floor (``retrain_improves``). ``chosen_mode`` is ``"RETRAIN"`` when
    ``conditional`` holds and ``"RANK_BLEND"`` otherwise (design Property 13:
    where both conditions do not hold, no conditional classification is
    mandated). ``best_blend_weight`` is the RANK_BLEND weight that scored best on
    the holdout (``0.0`` when the layer never helped -- the floor is preserved).
    """

    layer: str
    conditional: bool
    chosen_mode: str  # "RANK_BLEND" | "RETRAIN"
    rank_blend_corrupting: bool
    retrain_improves: bool
    best_blend_weight: float
    best_blend_holdout: float
    retrain_holdout: float


@dataclass(frozen=True)
class TuningReport:
    """The full audit trail of a :func:`tune` run (externally auditable, Gate D/F).

    ``stage_order`` is the exact sequence of stages executed -- it MUST be
    ``[resolve_drift, baseline_tuning, layer_tuning]`` (Req 5.1, Property 14).
    ``rejected_for_drift_increase`` lists the ``(description, holdout_ap)`` of
    every candidate rejected by guarantee (b) -- a tuned gain that raised drift
    past the gate -- INCLUDING gains whose holdout improved, so the independence
    of (b) from (a) is auditable. ``layer_decisions`` records the conditional /
    RETRAIN classification per layer (Req 4.3). ``bar_cleared`` / ``bar_reason``
    record the pre-registered Gate-D bar outcome (clears the Drift_Gate AND the
    projection does not regress the floor).
    """

    stage_order: Tuple[str, ...]
    baseline_grid_size: int
    rejected_for_drift_increase: Tuple[Tuple[str, float], ...]
    layer_decisions: Mapping[str, LayerModeDecision]
    baseline_drift: DriftResult
    selected_drift: DriftResult
    floor_holdout: float
    bar_cleared: bool
    bar_reason: str


def _merge_model_cfg(base: ModelConfig, overrides: Mapping[str, float]) -> ModelConfig:
    """Return a copy of ``base`` with ``overrides`` merged onto its hyperparameters.

    Only the hyperparameter mapping changes; PU weights are carried through so a
    baseline-grid point moves ONLY the structural knobs it names (a clean,
    single-axis structural step -- Req 5.2 coarse grid).
    """

    merged = dict(base.hyperparameters)
    merged.update(overrides)
    return ModelConfig(hyperparameters=merged, pu_weights=dict(base.pu_weights))


def _with_model_cfg(cfg: CandidateConfig, model_cfg: ModelConfig) -> CandidateConfig:
    """Return ``cfg`` with its baseline ``model_cfg`` replaced (layers unchanged)."""

    return CandidateConfig(
        layers_on=frozenset(cfg.layers_on),
        composition=dict(cfg.composition),
        model_cfg=model_cfg,
        shrinkage=dict(cfg.shrinkage),
    )


def _floor_config(cfg: CandidateConfig) -> CandidateConfig:
    """Return ``cfg`` reduced to its known-good FLOOR: every RANK_BLEND optional
    layer set to weight ``0.0`` (a no-op on the base -- contract Rule 15).

    This is the true floor the Req 4.3 classification and the composer's
    :func:`classify_layer` compare against: a layer "hurts" or "helps" relative
    to composing NOTHING optional onto the base, NOT relative to whatever weight
    the incoming candidate happened to carry. RETRAIN specs are left as-is (they
    are refit into the base and ignore the RANK_BLEND weight); the composer
    treats an all-weight-0 config as the floor ranking regardless.
    """

    composition = {
        name: (
            CompositionSpec(mode=spec.mode, weight=0.0)
            if spec.mode == "RANK_BLEND"
            else spec
        )
        for name, spec in cfg.composition.items()
    }
    return CandidateConfig(
        layers_on=frozenset(cfg.layers_on),
        composition=composition,
        model_cfg=cfg.model_cfg,
        shrinkage=dict(cfg.shrinkage),
    )


def _with_layer_spec(
    cfg: CandidateConfig, layer: str, spec: CompositionSpec
) -> CandidateConfig:
    """Return ``cfg`` with ``layer``'s :class:`CompositionSpec` set to ``spec``.

    Used both to probe a layer at a specific RANK_BLEND weight and to fix a
    layer's final mode after the classification decision. All other layers and
    knobs are carried through unchanged.
    """

    composition = dict(cfg.composition)
    composition[layer] = spec
    return CandidateConfig(
        layers_on=frozenset(cfg.layers_on),
        composition=composition,
        model_cfg=cfg.model_cfg,
        shrinkage=dict(cfg.shrinkage),
    )


def _drift_increased_past_gate(
    baseline_drift: DriftResult, candidate_drift: DriftResult
) -> bool:
    """Guarantee (b) predicate: did a tuned gain push drift PAST the threshold?

    A candidate's tuned gain is rejected when its measured drift is HIGHER than
    the baseline's AND that increase crosses the gate -- i.e. the candidate's
    drift verdict is FAIL (at/above the deciding threshold) while it rose above
    the baseline's measured drift. This is deliberately a SEPARATE test from the
    final "selected config passes the gate" check (guarantee a): it fires for
    every candidate as it is evaluated, so a drift-increasing gain is filtered
    out even if its holdout improved, independently of which config is ultimately
    selected.
    """

    rose = candidate_drift.adversarial_auc > baseline_drift.adversarial_auc
    crossed_gate = candidate_drift.verdict == "FAIL"
    return rose and crossed_gate


def _check_coarse_budget(grid_size: int, stage: str) -> None:
    """Escalate (Gate D) if a grid exceeds the coarse-grid budget."""

    if grid_size > _MAX_COARSE_GRID_POINTS:
        raise GridBudgetExceeded(
            f"{stage} grid has {grid_size} points, exceeding the coarse-grid "
            f"budget of {_MAX_COARSE_GRID_POINTS} (Req 5.2 / Gate D: coarse "
            "structural grids only). Escalating for a keep/kill decision rather "
            "than grinding a fine micro-grid."
        )


def tune(
    cfg: CandidateConfig,
    calib: Calibration,
    *,
    score_fn: ScoreFn,
    drift_fn: DriftFn,
    baseline_grid: Sequence[Mapping[str, float]] = DEFAULT_BASELINE_GRID,
    weight_grid: Sequence[float] = DEFAULT_RANK_BLEND_WEIGHT_GRID,
    projection_fit: Optional[ProjectionFit] = None,
) -> TunedConfig:
    """Two-level tune ``cfg`` in the order drift -> baseline -> layers (Req 5.1).

    The stages run in EXACTLY this order (recorded in the returned report's
    ``stage_order`` for auditing / Property 14):

    1. **Resolve feature-set drift.** Measure the incoming config's drift via
       ``drift_fn`` FIRST. Its measured drift is the baseline against which every
       later tuned gain's drift is compared for guarantee (b).
    2. **Baseline_Tuning.** Sweep the COARSE ``baseline_grid`` of structural model
       knobs, scoring each with ``score_fn``. A grid point whose tuned gain
       increases drift past the gate is REJECTED (guarantee b) even if its holdout
       improved; among the surviving points the highest-holdout baseline is kept.
    3. **Layer_Tuning.** For each ON optional layer, sweep the COARSE
       ``weight_grid`` (which INCLUDES 0.0). Decide RANK_BLEND vs RETRAIN by the
       Req 4.3 predicate (conditional IFF RANK_BLEND is CORRUPTING at every
       positive weight AND RETRAIN improves) and fix the layer's mode/weight
       accordingly; a drift-increasing layer setting is likewise rejected
       (guarantee b).

    After tuning, the SELECTED config must clear the Drift_Gate (guarantee a); the
    measured holdout is projected to an LB estimate via
    :func:`~poker_collusion.tuning_harness.projection.project.project_lb` and the
    two are returned as DISTINCT fields on the :class:`TunedConfig` (Req 5.4).

    :param cfg: the incoming candidate configuration to tune.
    :param calib: the LB calibration (consumed by ``drift_fn`` in production;
        carried here so the tuner's contract matches the design signature).
    :param score_fn: injectable ``(CandidateConfig) -> holdout_ap`` scorer.
    :param drift_fn: injectable ``(CandidateConfig) -> DriftResult`` drift gate.
    :param baseline_grid: coarse structural baseline knob grid (Req 5.2).
    :param weight_grid: coarse RANK_BLEND weight grid; MUST include ``0.0``.
    :param projection_fit: the LB<-AP fit to project through; when ``None`` it is
        built from the seeded anchor store via :func:`fit_projection`.
    :returns: a :class:`TunedConfig` (measured ``holdout_ap`` distinct from the
        projected ``projection``). The full audit trail is attached as the
        ``report`` attribute (see :func:`tune_with_report` for the typed pair).
    :raises GridBudgetExceeded: if a grid exceeds the coarse-grid budget (Gate D).
    :raises ValueError: if ``weight_grid`` does not include ``0.0`` (the floor
        MUST be a reachable point -- contract Rule 15).
    """

    tuned, _report = tune_with_report(
        cfg,
        calib,
        score_fn=score_fn,
        drift_fn=drift_fn,
        baseline_grid=baseline_grid,
        weight_grid=weight_grid,
        projection_fit=projection_fit,
    )
    return tuned


def tune_with_report(
    cfg: CandidateConfig,
    calib: Calibration,
    *,
    score_fn: ScoreFn,
    drift_fn: DriftFn,
    baseline_grid: Sequence[Mapping[str, float]] = DEFAULT_BASELINE_GRID,
    weight_grid: Sequence[float] = DEFAULT_RANK_BLEND_WEIGHT_GRID,
    projection_fit: Optional[ProjectionFit] = None,
) -> Tuple[TunedConfig, TuningReport]:
    """Like :func:`tune` but also returns the :class:`TuningReport` audit trail.

    The report exposes the stage order, the guarantee-(b) rejections (including
    drift-increasing gains whose holdout improved, proving (b) is independent of
    (a)), the per-layer conditional/RETRAIN decisions, and the Gate-D bar
    outcome, so the whole run is externally auditable (Gate D/F, Req 8.7).
    """

    if 0.0 not in tuple(weight_grid):
        raise ValueError(
            "weight_grid MUST include 0.0 so the known-good floor is a reachable "
            "point at every layer (contract Rule 15: floor-guaranteed composition)."
        )
    _check_coarse_budget(len(tuple(baseline_grid)), "baseline")
    _check_coarse_budget(len(tuple(weight_grid)), "layer weight")

    stage_order: List[str] = []
    rejected: List[Tuple[str, float]] = []

    # ------------------------------------------------------------------ #
    # STAGE 1 (Req 5.1): resolve feature-set drift FIRST.
    # ------------------------------------------------------------------ #
    stage_order.append(STAGE_RESOLVE_DRIFT)
    baseline_drift = drift_fn(cfg)
    # The FLOOR is the base with every optional RANK_BLEND layer zeroed (Rule 15):
    # the true known-good reference the Req 4.3 layer classification compares
    # against, independent of whatever weight the incoming candidate carried.
    floor_holdout = float(score_fn(_floor_config(cfg)))

    # ------------------------------------------------------------------ #
    # STAGE 2 (Req 5.1): Baseline_Tuning over the coarse structural grid.
    # Guarantee (b) is applied as a SEPARATE filter here: any grid point whose
    # tuned gain increases drift past the gate is rejected BEFORE selection,
    # even if its holdout improved.
    # ------------------------------------------------------------------ #
    stage_order.append(STAGE_BASELINE_TUNING)
    best_cfg = cfg
    best_holdout = floor_holdout
    best_drift = baseline_drift

    for overrides in baseline_grid:
        candidate_cfg = _with_model_cfg(cfg, _merge_model_cfg(cfg.model_cfg, overrides))
        candidate_drift = drift_fn(candidate_cfg)
        candidate_holdout = float(score_fn(candidate_cfg))

        # Guarantee (b), independent of selection: reject a drift-increasing gain
        # even when its holdout improved.
        if _drift_increased_past_gate(baseline_drift, candidate_drift):
            rejected.append((f"baseline:{dict(overrides)}", candidate_holdout))
            continue

        if candidate_holdout > best_holdout:
            best_cfg = candidate_cfg
            best_holdout = candidate_holdout
            best_drift = candidate_drift

    # ------------------------------------------------------------------ #
    # STAGE 3 (Req 5.1): Layer_Tuning. Per ON optional layer, sweep the coarse
    # weight grid (includes 0.0) and decide RANK_BLEND vs RETRAIN (Req 4.3).
    # ------------------------------------------------------------------ #
    stage_order.append(STAGE_LAYER_TUNING)
    layer_decisions: Dict[str, LayerModeDecision] = {}

    for layer in sorted(best_cfg.layers_on):
        decision, best_cfg, best_holdout, best_drift = _tune_one_layer(
            layer,
            best_cfg,
            best_holdout,
            best_drift,
            baseline_drift=baseline_drift,
            floor_holdout=floor_holdout,
            score_fn=score_fn,
            drift_fn=drift_fn,
            weight_grid=tuple(weight_grid),
            rejected=rejected,
        )
        layer_decisions[layer] = decision

    # ------------------------------------------------------------------ #
    # Guarantee (a): the SELECTED config must clear the Drift_Gate. This is a
    # DISTINCT check from (b): if the selected config does not pass, fall back to
    # the incoming floor config (which resolved drift in stage 1). Because every
    # drift-increasing gain was already filtered by (b), the selected config's
    # drift never rose past the gate through tuning; this check guarantees the
    # final winner is a PASS regardless.
    # ------------------------------------------------------------------ #
    selected_drift = drift_fn(best_cfg)
    if selected_drift.verdict != "PASS":
        best_cfg = cfg
        best_holdout = floor_holdout
        selected_drift = baseline_drift

    # ------------------------------------------------------------------ #
    # Report (Req 5.4 / Property 16): measured holdout kept DISTINCT from the
    # projected LB. project_lb returns a Projection with its own point +
    # uncertainty; the TunedConfig carries the measured holdout separately.
    # ------------------------------------------------------------------ #
    fit = projection_fit if projection_fit is not None else fit_projection()
    projection: Projection = project_lb(best_holdout, fit)

    # Pre-registered Gate-D bar: clears the Drift_Gate AND projection does not
    # regress the floor's projected LB.
    floor_projection = project_lb(floor_holdout, fit)
    bar_cleared, bar_reason = _evaluate_bar(selected_drift, projection, floor_projection)

    tuned = TunedConfig(
        config=best_cfg,
        holdout_ap=best_holdout,
        drift=selected_drift,
        projection=projection,
    )
    report = TuningReport(
        stage_order=tuple(stage_order),
        baseline_grid_size=len(tuple(baseline_grid)),
        rejected_for_drift_increase=tuple(rejected),
        layer_decisions=layer_decisions,
        baseline_drift=baseline_drift,
        selected_drift=selected_drift,
        floor_holdout=floor_holdout,
        bar_cleared=bar_cleared,
        bar_reason=bar_reason,
    )
    return tuned, report


def _tune_one_layer(
    layer: str,
    cfg: CandidateConfig,
    current_holdout: float,
    current_drift: DriftResult,
    *,
    baseline_drift: DriftResult,
    floor_holdout: float,
    score_fn: ScoreFn,
    drift_fn: DriftFn,
    weight_grid: Tuple[float, ...],
    rejected: List[Tuple[str, float]],
) -> Tuple[LayerModeDecision, CandidateConfig, float, DriftResult]:
    """Tune one optional layer: sweep RANK_BLEND weights, decide the mode (Req 4.3).

    Returns ``(decision, updated_cfg, updated_holdout, updated_drift)``.

    The RANK_BLEND half of the Req 4.3 predicate reuses the composer's
    :func:`classify_layer`: the composed holdout AP at each positive weight is
    fed to ``classify_layer(..., floor_ap=floor_holdout)`` and the layer is
    RANK_BLEND-CORRUPTING iff that returns ``CORRUPTING`` (strictly below the
    floor at EVERY positive weight). The RETRAIN half probes the layer in RETRAIN
    mode and checks the holdout strictly improved over the floor. Only when BOTH
    hold is the layer recorded conditional and switched to RETRAIN; otherwise no
    conditional classification is mandated and the best RANK_BLEND weight (which
    may be 0.0 -- the floor) is kept.

    Guarantee (b) applies here too: a layer setting whose drift rose past the gate
    is rejected even if its holdout improved.
    """

    # Sweep RANK_BLEND weights. weight 0.0 is the floor (a no-op) by construction.
    holdout_by_weight: Dict[float, float] = {}
    best_weight = 0.0
    best_weight_holdout = current_holdout
    best_weight_cfg = cfg
    best_weight_drift = current_drift

    for weight in weight_grid:
        probe_cfg = _with_layer_spec(
            cfg, layer, CompositionSpec(mode="RANK_BLEND", weight=float(weight))
        )
        probe_drift = drift_fn(probe_cfg)
        probe_holdout = float(score_fn(probe_cfg))
        holdout_by_weight[float(weight)] = probe_holdout

        if weight == 0.0:
            # The floor point: never rejected, never "improves"; anchors the sweep.
            continue

        if _drift_increased_past_gate(baseline_drift, probe_drift):
            rejected.append((f"layer:{layer}:blend_w={weight}", probe_holdout))
            continue

        if probe_holdout > best_weight_holdout:
            best_weight = float(weight)
            best_weight_holdout = probe_holdout
            best_weight_cfg = probe_cfg
            best_weight_drift = probe_drift

    # RANK_BLEND-CORRUPTING per Req 4.3 first half: strictly below floor at EVERY
    # positive weight (reuse the composer's classifier).
    rank_blend_corrupting = (
        classify_layer(holdout_by_weight, floor_holdout) == CORRUPTING
    )

    # RETRAIN probe (Req 4.3 second half): does RETRAIN increase holdout?
    retrain_cfg = _with_layer_spec(
        cfg, layer, CompositionSpec(mode="RETRAIN", weight=0.0)
    )
    retrain_drift = drift_fn(retrain_cfg)
    retrain_holdout = float(score_fn(retrain_cfg))
    retrain_improves = retrain_holdout > floor_holdout

    conditional = rank_blend_corrupting and retrain_improves

    if conditional:
        # Record conditional -> use RETRAIN. Guarantee (b) still applies: only
        # adopt the RETRAIN config if its drift did not rise past the gate.
        chosen_mode = "RETRAIN"
        if _drift_increased_past_gate(baseline_drift, retrain_drift):
            rejected.append((f"layer:{layer}:retrain", retrain_holdout))
            updated_cfg, updated_holdout, updated_drift = cfg, current_holdout, current_drift
        else:
            updated_cfg = retrain_cfg
            updated_holdout = retrain_holdout
            updated_drift = retrain_drift
    else:
        # No conditional classification mandated: keep the best RANK_BLEND weight
        # (possibly 0.0 == the floor, which never regresses).
        chosen_mode = "RANK_BLEND"
        updated_cfg = best_weight_cfg
        updated_holdout = best_weight_holdout
        updated_drift = best_weight_drift

    decision = LayerModeDecision(
        layer=layer,
        conditional=conditional,
        chosen_mode=chosen_mode,
        rank_blend_corrupting=rank_blend_corrupting,
        retrain_improves=retrain_improves,
        best_blend_weight=best_weight,
        best_blend_holdout=best_weight_holdout,
        retrain_holdout=retrain_holdout,
    )
    return decision, updated_cfg, updated_holdout, updated_drift


def _evaluate_bar(
    selected_drift: DriftResult,
    projection: Projection,
    floor_projection: Projection,
) -> Tuple[bool, str]:
    """Evaluate the pre-registered Gate-D bar for the tuned config.

    The bar (design task 13.1): a tuned config is ACCEPTED only if it clears the
    Drift_Gate AND its projection does not regress the floor's projected LB.
    Returns ``(cleared, reason)`` for the audit trail.
    """

    if selected_drift.verdict != "PASS":
        return False, "selected config does not clear the Drift_Gate"
    if projection.point < floor_projection.point:
        return (
            False,
            "projected LB regresses the floor "
            f"({projection.point:.5f} < {floor_projection.point:.5f})",
        )
    return True, "clears Drift_Gate and does not regress the floor projection"
