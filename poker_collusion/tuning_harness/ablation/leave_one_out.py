"""Leave-one-out ablation for the layered-tuning harness (design: ``ablation/leave_one_out.py``).

This module answers three questions about a multi-layer candidate, all against the
SAME external judge the rest of the harness trusts -- the table-disjoint PU-stress
AP holdout scorer from task 8.1 (:func:`poker_collusion.tuning_harness.holdout.scorer.holdout_pair_ap`):

``ablate(cfg, score_fn, tolerance)``
    1. Each ON layer's UNIQUE contribution = ``full_candidate_holdout -
       holdout_with_that_single_layer_removed`` (Req 3.1, design Property 9). A
       positive contribution means the candidate scores WORSE without the layer,
       so the layer is pulling its weight; a non-positive contribution means
       removing it did not hurt (or helped).
    2. Any layer whose unique contribution is ``<= 0`` is RECOMMENDED FOR REMOVAL
       (Req 3.2, design Property 10): it is redundant or corrupting, never
       load-bearing.
    3. The LEANEST SUBSET within a stated ``tolerance`` of the best observed
       holdout (Req 3.3, design Property 11): the minimum-cardinality ON-layer
       subset whose holdout AP is within ``tolerance`` of the best holdout AP over
       all evaluated subsets. Ties in cardinality break deterministically.

Boundary / reuse (accountability contract Rule 6, Req 3.1 "reuse, do not rebuild"):
scoring a configuration requires the reused CV harness + reference metric wired
through :func:`holdout_pair_ap`. To keep the ablation ARITHMETIC (leave-one-out
differences, removal recommendations, leanest-subset search) unit-testable in
isolation -- exactly as the composer and holdout tests inject their scoring -- this
module takes an INJECTABLE ``score_fn(cfg) -> holdout_ap`` callback. Production
callers pass a thin adapter that composes ``cfg`` and calls ``holdout_pair_ap``;
tests pass a deterministic stub. This module NEVER introduces a new metric or a
new split; it only calls the injected scorer and does difference/subset arithmetic
over the numbers it returns.

Every distinct configuration is scored at most once (the leave-one-out subsets and
the leanest-subset search share a memoised cache keyed by the ON-layer subset), so
the injected ``score_fn`` -- which in production is an expensive CV run -- is called
the minimum number of times.

Requirements:
- 3.1: report each ON layer's unique leave-one-out contribution.
- 3.2: recommend removing any layer whose unique contribution is ``<= 0``.
- 3.3: identify the leanest subset within a stated tolerance of the best holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, FrozenSet, Mapping, Tuple

from poker_collusion.tuning_harness.models import CandidateConfig

__all__ = [
    "ScoreFn",
    "AblationResult",
    "ablate",
]

#: A callback that scores a :class:`CandidateConfig` and returns its holdout AP
#: (the PU-stress AP on the table-disjoint holdout). Injected into :func:`ablate`
#: so the leave-one-out arithmetic is unit-testable without the real CV run; in
#: production this wraps :func:`poker_collusion.tuning_harness.holdout.scorer.holdout_pair_ap`.
ScoreFn = Callable[[CandidateConfig], float]


@dataclass(frozen=True)
class AblationResult:
    """The result of a leave-one-out ablation over a multi-layer candidate.

    ``full_holdout`` is the holdout AP of the full candidate (all ON layers).
    ``contributions`` maps each ON layer to its unique leave-one-out contribution
    (``full_holdout - holdout_without_that_layer``); a value ``<= 0`` means the
    layer is not load-bearing (Req 3.1). ``removal_recommended`` is the tuple of
    layers whose contribution is ``<= 0`` (Req 3.2), in stable sorted order.

    ``best_holdout`` is the highest holdout AP observed over every subset the
    ablation evaluated. ``leanest_subset`` is the minimum-cardinality ON-layer
    subset whose holdout AP is within ``tolerance`` of ``best_holdout`` (Req 3.3);
    ``leanest_holdout`` is that subset's holdout AP and ``tolerance`` records the
    tolerance the selection used, so the choice is externally auditable.
    """

    full_holdout: float
    contributions: Mapping[str, float]
    removal_recommended: Tuple[str, ...]
    leanest_subset: FrozenSet[str]
    leanest_holdout: float
    best_holdout: float
    tolerance: float


def _config_with_layers(cfg: CandidateConfig, layers_on: FrozenSet[str]) -> CandidateConfig:
    """Return a copy of ``cfg`` restricted to the given ON optional layers.

    Only the ``layers_on`` set and the ``composition`` map (filtered to the kept
    layers) change; the baseline ``model_cfg`` and per-layer ``shrinkage`` knobs
    are carried through unchanged so a removed layer changes ONLY its own presence
    (a clean leave-one-out -- nothing else about the candidate moves). ``foundation``
    is always on and is never listed in ``layers_on``, so it is unaffected.
    """

    composition = {
        name: spec for name, spec in cfg.composition.items() if name in layers_on
    }
    shrinkage = {
        name: k for name, k in cfg.shrinkage.items() if name in layers_on
    }
    return CandidateConfig(
        layers_on=frozenset(layers_on),
        composition=composition,
        model_cfg=cfg.model_cfg,
        shrinkage=shrinkage,
    )


def _memoised_scorer(
    cfg: CandidateConfig, score_fn: ScoreFn
) -> Callable[[FrozenSet[str]], float]:
    """Wrap ``score_fn`` so each distinct ON-layer subset is scored at most once.

    The ablation evaluates the full set, every leave-one-out subset, and (for the
    leanest-subset search) potentially every subset; memoising on the frozenset of
    ON layers means the expensive injected scorer runs the minimum number of times.
    """

    cache: Dict[FrozenSet[str], float] = {}

    def score(layers_on: FrozenSet[str]) -> float:
        key = frozenset(layers_on)
        if key not in cache:
            cache[key] = float(score_fn(_config_with_layers(cfg, key)))
        return cache[key]

    return score


def ablate(
    cfg: CandidateConfig,
    score_fn: ScoreFn,
    *,
    tolerance: float = 0.0,
) -> AblationResult:
    """Leave-one-out ablation of every ON layer in ``cfg`` against the holdout.

    For each ON optional layer ``L`` in ``cfg.layers_on`` (Req 3.1, Property 9)::

        contribution(L) = full_holdout - holdout(cfg without L)

    where ``full_holdout = score_fn(cfg)`` and ``holdout(cfg without L)`` scores
    the same candidate with ONLY ``L`` switched off. A positive contribution means
    the layer is load-bearing (the candidate is worse without it); a non-positive
    contribution triggers a removal recommendation (Req 3.2, Property 10).

    The leanest subset (Req 3.3, Property 11) is the minimum-cardinality subset of
    the ON layers whose holdout AP is within ``tolerance`` of the best holdout AP
    observed over the searched subsets. When several subsets tie on cardinality
    AND all lie within tolerance, the highest-scoring one is chosen; further ties
    break by the sorted layer tuple, so the choice is deterministic.

    Scoring is delegated ENTIRELY to the injected ``score_fn`` (Req 3.1 reuse
    mandate): this function introduces no metric and no split, it only differences
    and compares the numbers ``score_fn`` returns. Every distinct subset is scored
    at most once via an internal memoised cache.

    :param cfg: the multi-layer candidate to ablate. ``cfg.layers_on`` names the
        ON optional layers; ``foundation`` is always on and is not ablated.
    :param score_fn: an injectable ``(CandidateConfig) -> holdout_ap`` callback
        (in production a thin wrapper over the reused holdout scorer).
    :param tolerance: how far below the best holdout AP a leaner subset may fall
        and still be accepted (Req 3.3). Must be ``>= 0``. ``0.0`` (the default)
        means the leanest subset must match the best holdout exactly.
    :returns: an :class:`AblationResult` with per-layer contributions, the removal
        recommendations, and the chosen leanest subset.
    :raises ValueError: if ``tolerance`` is negative.
    """

    if tolerance < 0.0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")

    on_layers = frozenset(cfg.layers_on)
    score = _memoised_scorer(cfg, score_fn)

    full_holdout = score(on_layers)

    # Req 3.1 / Property 9: each ON layer's unique leave-one-out contribution.
    contributions: Dict[str, float] = {}
    for layer in sorted(on_layers):
        without_layer = on_layers - {layer}
        holdout_without = score(without_layer)
        contributions[layer] = full_holdout - holdout_without

    # Req 3.2 / Property 10: a non-positive contribution => recommend removal.
    removal_recommended = tuple(
        layer for layer in sorted(on_layers) if contributions[layer] <= 0.0
    )

    # Req 3.3 / Property 11: the leanest subset within tolerance of the best.
    leanest_subset, leanest_holdout, best_holdout = _leanest_within_tolerance(
        on_layers, score, tolerance
    )

    return AblationResult(
        full_holdout=full_holdout,
        contributions=contributions,
        removal_recommended=removal_recommended,
        leanest_subset=leanest_subset,
        leanest_holdout=leanest_holdout,
        best_holdout=best_holdout,
        tolerance=tolerance,
    )


def _leanest_within_tolerance(
    on_layers: FrozenSet[str],
    score: Callable[[FrozenSet[str]], float],
    tolerance: float,
) -> tuple[FrozenSet[str], float, float]:
    """Find the minimum-cardinality subset within ``tolerance`` of the best holdout.

    Enumerates every subset of ``on_layers`` (the full candidate down to the
    empty set = foundation-only), scoring each through the memoised ``score``. The
    best holdout AP is the maximum over all subsets. Among every subset whose
    holdout AP is ``>= best - tolerance``, the returned subset is the one with the
    FEWEST layers; ties on cardinality break by highest holdout AP, then by the
    sorted layer tuple, so the result is deterministic.

    Returns ``(subset, subset_holdout, best_holdout)``.
    """

    layers_sorted = sorted(on_layers)
    n = len(layers_sorted)

    # Score every subset once (memoised). Enumerate small subsets first so the
    # natural order already favours leaner subsets.
    scored: list[tuple[int, float, tuple[str, ...], FrozenSet[str]]] = []
    best_holdout = float("-inf")
    for size in range(n + 1):
        for combo in combinations(layers_sorted, size):
            subset = frozenset(combo)
            ap = score(subset)
            best_holdout = max(best_holdout, ap)
            scored.append((size, ap, combo, subset))

    cutoff = best_holdout - tolerance
    # Candidates within tolerance of the best. Choose fewest layers, then highest
    # AP, then lexicographically smallest layer tuple for determinism.
    within = [entry for entry in scored if entry[1] >= cutoff]
    size, ap, _combo, subset = min(within, key=lambda e: (e[0], -e[1], e[2]))
    return subset, ap, best_holdout
