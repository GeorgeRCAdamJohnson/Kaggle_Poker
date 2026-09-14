"""Floor-guaranteed composer (design: ``composition/composer.py``).

This module composes optional layers onto the known-good base with a HARD FLOOR
GUARANTEE (accountability contract Rule 15, Requirement 2). Nothing here can
silently regress the LB-verified 0.44519 floor: a layer either contributes or is
flagged, and the neutral (all-OFF / weight-0) path reproduces the floor exactly.

Four pieces live here, mirroring the design "Composer" section:

``rank_blend(base, layer, weight)``
    A convex combination of *ranks* (not raw scores):
    ``blended = (1 - w) * rank(base) + w * rank(layer)``, then re-ranked. The
    CRITICAL invariant (Req 2.2, design Property 7, Assumption A5): ``weight ==
    0.0`` returns the base ordering EXACTLY (order-identical). Composing at w=0
    is therefore a no-op on the floor, which is what makes a correct layer unable
    to regress it.

``retrain(base_features, layer_features, model_cfg, fit_fn)``
    Refit the reused baseline model on the concatenated base+layer features and
    return the resulting :class:`RankVector`. The result is trusted ONLY after
    the combined feature set clears the Drift_Gate (Req 4.2) - that wiring is
    task 9.1; here we only PRODUCE the ranking. ``fit_fn`` is an injectable fit
    callback so the pure composition logic (feature concatenation + ranking
    shape) is unit-testable without loading the real model/caches; when omitted
    the function raises a clear error rather than guessing a refit.

``compose(cfg, base_ranking_fn, layer_ranking_fn)``
    Assemble a :class:`Candidate`. When every optional layer is OFF the produced
    ranking reproduces :func:`known_good_floor_config`'s ranking EXACTLY (Req
    2.1, design Property 6, Assumption A4). Which layers are ON is reported on a
    best-effort basis: ANY exception during reporting yields ``layers_reported =
    None`` and NEVER blocks assembly (Req 2.3, 2.4).

``classify_layer(holdout_by_weight, floor_ap)``
    Corrupting-layer detection (Req 2.5, design Property 8): a layer whose
    composed holdout AP is strictly below the floor at EVERY positive weight in
    the tuning grid is ``CORRUPTING`` (it displaces known-good signal), not
    merely ``NULL``.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

from poker_collusion.tuning_harness.layers.registry import known_good_floor_config
from poker_collusion.tuning_harness.models import (
    Candidate,
    CandidateConfig,
    FeatureFrame,
    ModelConfig,
    RankVector,
)

__all__ = [
    "CORRUPTING",
    "NULL",
    "CONTRIBUTING",
    "rank_blend",
    "retrain",
    "compose",
    "classify_layer",
]

#: Layer-classification verdicts (Req 2.5).
CORRUPTING = "CORRUPTING"
NULL = "NULL"
CONTRIBUTING = "CONTRIBUTING"

#: A callback that produces the base (floor) ranking, given the baseline model
#: config. Injected into :func:`compose` so the composition logic is testable
#: without loading the real feature caches / model.
BaseRankingFn = Callable[[ModelConfig], RankVector]

#: A callback that produces one optional layer's per-pair ranking, given the
#: layer name and the config. Injected into :func:`compose`.
LayerRankingFn = Callable[[str, CandidateConfig], RankVector]

#: A callback that refits the baseline model on a combined FeatureFrame and
#: returns a RankVector. Injected into :func:`retrain`.
FitFn = Callable[[FeatureFrame, ModelConfig], RankVector]


def _ranks_by_pair(ranking: RankVector) -> dict[str, int]:
    """Map each ``pair_id`` to its 0-based rank position (0 = highest risk).

    The RankVector's ``pair_ids`` tuple IS the ordering (index 0 = highest
    risk), so the rank is simply the index. Duplicate pair ids would make the
    ranking ill-defined, so they are rejected loudly.
    """

    ranks: dict[str, int] = {}
    for position, pair_id in enumerate(ranking.pair_ids):
        if pair_id in ranks:
            raise ValueError(
                f"RankVector contains duplicate pair_id {pair_id!r}; a ranking "
                "must be a strict ordering."
            )
        ranks[pair_id] = position
    return ranks


def rank_blend(
    base_ranking: RankVector,
    layer_ranking: RankVector,
    weight: float,
) -> RankVector:
    """Blend two rankings by a convex combination of their RANK POSITIONS.

    ``blended_rank(pair) = (1 - weight) * rank_base(pair) + weight * rank_layer(pair)``
    where ``rank_*`` is the 0-based position in each ranking (0 = highest risk).
    The pairs are then re-sorted by ascending blended rank (lower blended rank =
    higher risk) to form the returned ordering.

    FLOOR GUARANTEE (Req 2.2, design Property 7): when ``weight == 0.0`` the
    layer contributes nothing and the result is the base ordering EXACTLY
    (order-identical, same ``pair_ids`` tuple). This is enforced by an explicit
    early return so it is exact for ALL inputs, independent of any tie-breaking
    or float arithmetic - the invariant Assumption A5 falsifies.

    Symmetrically, ``weight == 1.0`` returns the layer ordering restricted to the
    base's pair set.

    Ties in the blended rank are broken by the pair's BASE rank first (keeping
    the result as close to the floor as possible) and then by ``pair_id`` for
    determinism.

    :param base_ranking: the base (floor) ordering; the returned ordering covers
        exactly this ranking's pair set.
    :param layer_ranking: the layer's ordering; pairs it does not cover fall back
        to the base rank (the layer is neutral about them).
    :param weight: blend weight in ``[0, 1]``; ``0.0`` reproduces the base.
    :returns: the blended :class:`RankVector` (ordering only; no scores).
    :raises ValueError: if ``weight`` is outside ``[0, 1]`` or either ranking has
        duplicate pair ids.
    """

    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"rank_blend weight must be in [0, 1], got {weight!r}")

    # FLOOR GUARANTEE: weight 0 is a no-op on the base ordering, exactly.
    if weight == 0.0:
        return base_ranking

    base_ranks = _ranks_by_pair(base_ranking)
    layer_ranks = _ranks_by_pair(layer_ranking)

    def blended_key(pair_id: str) -> tuple[float, int, str]:
        rb = base_ranks[pair_id]
        # A pair the layer does not cover: the layer is neutral -> use its base
        # rank so an incomplete layer never invents an ordering it did not state.
        rl = layer_ranks.get(pair_id, rb)
        blended = (1.0 - weight) * rb + weight * rl
        return (blended, rb, pair_id)

    ordered = sorted(base_ranking.pair_ids, key=blended_key)
    return RankVector(pair_ids=tuple(ordered))


def retrain(
    base_features: FeatureFrame,
    layer_features: FeatureFrame,
    model_cfg: ModelConfig,
    fit_fn: Optional[FitFn] = None,
) -> RankVector:
    """Refit the reused baseline model on base + layer features -> a RankVector.

    The base and layer :class:`FeatureFrame` views (keyed by ``pair_id``) are
    concatenated column-wise into a single combined frame over the pairs they
    share, then handed to ``fit_fn`` which refits the reused baseline model and
    returns the per-pair ranking.

    TRUST BOUNDARY (Req 4.2): the ranking produced here is trusted ONLY after the
    combined feature set clears the Drift_Gate - that gating is wired in task
    9.1. This function only PRODUCES the ranking; it makes no trust claim.

    ``fit_fn`` is injectable so the composition logic (the combined FeatureFrame
    assembly and the ranking's shape) is unit-testable without loading the real
    baseline model or the feature caches. In production the caller passes the
    reused ``PURanker`` fit adapter; in tests a lightweight callback stands in.
    When ``fit_fn`` is omitted the function raises rather than silently guessing
    a refit (a full model refit is impractical - and non-deterministic - inside a
    pure unit test, so the callback is required).

    :param base_features: the foundation feature view (keyed by ``pair_id``).
    :param layer_features: the layer feature view (keyed by ``pair_id``); its
        columns are appended to the base's for the pairs present in BOTH frames.
    :param model_cfg: the baseline hyperparameters + PU weights to refit under.
    :param fit_fn: an injectable ``(FeatureFrame, ModelConfig) -> RankVector``
        refit callback. Required.
    :returns: the refit model's per-pair :class:`RankVector`.
    :raises ValueError: if ``fit_fn`` is not supplied.
    """

    if fit_fn is None:
        raise ValueError(
            "retrain requires a fit_fn callback: a full baseline refit is "
            "produced by the reused model (wired in task 9.1) or by an injected "
            "test callback. Refusing to guess a refit."
        )

    combined = _combine_features(base_features, layer_features)
    return fit_fn(combined, model_cfg)


def _combine_features(
    base_features: FeatureFrame, layer_features: FeatureFrame
) -> FeatureFrame:
    """Column-concatenate two per-pair FeatureFrames over their shared pairs.

    The combined frame's ``feature_keys`` are the base keys followed by the layer
    keys; each row is the base row values followed by the layer row values. Only
    pairs present in BOTH frames are kept (a refit needs a full feature vector
    per pair). ``is_eval`` is carried from the base frame (the pool membership of
    a pair does not change when a layer is added).
    """

    combined_keys = tuple(base_features.feature_keys) + tuple(layer_features.feature_keys)
    rows: dict[str, tuple[float, ...]] = {}
    for pair_id, base_row in base_features.rows.items():
        layer_row = layer_features.rows.get(pair_id)
        if layer_row is None:
            continue
        rows[pair_id] = tuple(base_row) + tuple(layer_row)

    is_eval = {pid: base_features.is_eval.get(pid, False) for pid in rows}
    return FeatureFrame(feature_keys=combined_keys, rows=rows, is_eval=is_eval)


def _safe_report_layers(cfg: CandidateConfig) -> Optional[tuple[str, ...]]:
    """Best-effort report of which optional layers are ON (Req 2.3, 2.4).

    Returns the ON optional layer names in a stable sorted order. ANY exception
    while gathering the report is swallowed and yields ``None`` so a reporting
    failure NEVER blocks candidate assembly (Req 2.4). This is the only place a
    broad ``except`` is justified: reporting is explicitly best-effort.
    """

    try:
        return tuple(sorted(cfg.layers_on))
    except Exception:  # noqa: BLE001 - Req 2.4: reporting must never block assembly.
        return None


def compose(
    cfg: CandidateConfig,
    base_ranking_fn: BaseRankingFn,
    layer_ranking_fn: Optional[LayerRankingFn] = None,
    *,
    report_layers_fn: Optional[Callable[[CandidateConfig], Optional[tuple[str, ...]]]] = None,
) -> Candidate:
    """Assemble a :class:`Candidate` with the hard floor guarantee.

    Composition proceeds:

    1. Produce the base (floor) ranking via ``base_ranking_fn(cfg.model_cfg)``.
    2. For each optional layer that is ON (``cfg.layers_on``), obtain its ranking
       via ``layer_ranking_fn`` and RANK_BLEND it onto the running ranking with
       the layer's configured weight (RETRAIN layers are refit upstream and are
       not re-blended here; a RETRAIN layer with no explicit ranking is left to
       the base). A layer composed at weight 0 is a no-op (the floor guarantee),
       so an all-``weight==0`` config reproduces the base ranking regardless of
       which layers are nominally ON.
    3. FLOOR (Req 2.1, Property 6): when NO optional layer is ON, the running
       ranking is exactly the base ranking - identical to
       :func:`known_good_floor_config`'s ranking when ``base_ranking_fn`` yields
       the floor ranking. No blending is applied, so it is order-identical.
    4. Report which layers are ON best-effort; any failure -> ``layers_reported =
       None`` and assembly still returns a valid Candidate (Req 2.3, 2.4).

    ``base_ranking_fn`` and ``layer_ranking_fn`` are injected because the real
    rankings require the feature caches and the reused model; injecting them
    keeps the composition ORDERING logic and the floor guarantee unit-testable in
    isolation (design Assumption A4/A5 falsifying tests).

    :param cfg: the candidate configuration (which optional layers are ON, each
        layer's :class:`CompositionSpec`, model config).
    :param base_ranking_fn: produces the base/floor ranking from the model config.
    :param layer_ranking_fn: produces an optional layer's ranking; required only
        when ``cfg`` switches on a RANK_BLEND layer with a positive weight.
    :param report_layers_fn: override for the best-effort layer reporter (used to
        exercise the reporting-failure path in tests); defaults to
        :func:`_safe_report_layers`.
    :returns: the assembled :class:`Candidate`.
    :raises ValueError: if a positive-weight RANK_BLEND layer is ON but no
        ``layer_ranking_fn`` was supplied to rank it.
    """

    ranking = base_ranking_fn(cfg.model_cfg)

    # Blend each ON optional layer onto the running ranking. Deterministic order.
    for layer_name in sorted(cfg.layers_on):
        spec = cfg.composition.get(layer_name)
        if spec is None:
            # No composition spec -> the layer contributes nothing (neutral).
            continue
        if spec.mode != "RANK_BLEND":
            # RETRAIN layers are refit into the base upstream (retrain()); they
            # are not re-blended here. Leaving the ranking untouched keeps the
            # floor guarantee for the all-RETRAIN floor stack.
            continue
        if spec.weight == 0.0:
            # Floor guarantee: weight-0 blend is a no-op; skip without needing a
            # layer ranking at all (an all-OFF-effect config needs no layer fn).
            continue
        if layer_ranking_fn is None:
            raise ValueError(
                f"layer {layer_name!r} is ON via RANK_BLEND at weight "
                f"{spec.weight!r} but no layer_ranking_fn was supplied to rank it."
            )
        layer_ranking = layer_ranking_fn(layer_name, cfg)
        ranking = rank_blend(ranking, layer_ranking, spec.weight)

    # Best-effort layer reporting (Req 2.3, 2.4). The guarantee that a reporting
    # failure NEVER blocks assembly is enforced HERE, at the composition
    # boundary, so it holds for ANY reporter - the default AND any injected one.
    reporter = report_layers_fn if report_layers_fn is not None else _safe_report_layers
    try:
        layers_reported = reporter(cfg)
    except Exception:  # noqa: BLE001 - Req 2.4: reporting must never block assembly.
        layers_reported = None

    return Candidate(config=cfg, ranking=ranking, layers_reported=layers_reported)


def classify_layer(
    holdout_by_weight: Mapping[float, float] | Sequence[tuple[float, float]],
    floor_ap: float,
) -> str:
    """Classify a layer as CORRUPTING, NULL, or CONTRIBUTING (Req 2.5).

    Given the layer's composed holdout AP at each POSITIVE weight in the tuning
    grid (``{weight: holdout_ap}`` or ``[(weight, holdout_ap), ...]``) and the
    floor's holdout AP:

    - ``CORRUPTING`` - the composed holdout AP is STRICTLY below ``floor_ap`` at
      EVERY positive weight. Adding the layer at any strength displaces
      known-good signal; this is worse than a null layer, which merely fails to
      help (design Property 8).
    - ``CONTRIBUTING`` - at least one positive weight lifts the composed holdout
      AP strictly above ``floor_ap`` (the layer adds signal at some strength).
    - ``NULL`` - neither: the layer neither strictly beats nor strictly hurts at
      every weight (e.g. it matches the floor, or is mixed). It is redundant, not
      corrupting.

    Only POSITIVE weights are considered (weight 0 is the floor by construction
    and carries no information about the layer). If no positive weights are
    provided the layer cannot be classified as corrupting or contributing and is
    reported ``NULL``.

    :param holdout_by_weight: composed holdout AP per weight; only strictly
        positive weights are used.
    :param floor_ap: the floor configuration's holdout AP to compare against.
    :returns: one of :data:`CORRUPTING`, :data:`NULL`, :data:`CONTRIBUTING`.
    """

    items = (
        list(holdout_by_weight.items())
        if isinstance(holdout_by_weight, Mapping)
        else list(holdout_by_weight)
    )
    positive = [(w, ap) for (w, ap) in items if w > 0.0]

    if not positive:
        return NULL

    below_at_every_weight = all(ap < floor_ap for (_w, ap) in positive)
    if below_at_every_weight:
        return CORRUPTING

    above_at_some_weight = any(ap > floor_ap for (_w, ap) in positive)
    if above_at_some_weight:
        return CONTRIBUTING

    return NULL
