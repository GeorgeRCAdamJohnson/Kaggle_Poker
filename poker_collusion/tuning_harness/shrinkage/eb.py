"""Empirical-Bayes sample-size shrinkage for per-pair statistics (design: ``shrinkage/eb.py``).

Eval pairs are low-co-hand (median shared-hand count ~76), so a raw per-pair rate
/ z-score / excess is mostly noise: a pair that shared five hands and "always
clashed" is not evidence, while the same rate over 300 shared hands is. Before any
per-pair statistic enters a ranking it MUST be shrunk toward a population prior by
its shared-hand count ``n`` (accountability contract Rule 16).

Two pieces live here:

``shrink(value, n, k, prior)``
    The empirical-Bayes estimator ``prior + (value - prior) * n / (n + k)``. At
    ``n = 0`` it returns the prior exactly (no data -> no update); it moves
    monotonically from the prior toward the raw ``value`` as ``n`` grows; and it
    approaches ``value`` as ``n`` grows large (Req 6.1, design Property 17). The
    shrinkage constant ``k`` is the pseudo-count: larger ``k`` means more data is
    needed before the raw value is trusted.

``sweep_shrinkage(layer, k_grid)``
    An EXHAUSTIVE sweep over a grid of shrinkage constants. It evaluates every
    constant in the grid and NEVER stops early, even once an earlier constant
    passes verification (Req 6.2, design Property 18). A candidate ``k`` is
    *verified* when the pairs it ranks into the top-K by shrunk score have a
    median shared-hand count that is NOT below the population median shared-hand
    count -- i.e. the shrinkage has stopped promoting small-sample noise into the
    top of the ranking (design assumption A6). The sweep returns a
    :class:`ShrinkageChoice` carrying the chosen ``k``, the per-``k`` verification
    record for the whole grid, and whether a verified choice was found.

Requirements:
- 6.1: shrinkage is monotone in the shared-hand count; ``n = 0`` returns the prior;
  large ``n`` approaches the raw value.
- 6.2: the sweep is exhaustive (never stops early) and its choice is verified
  against the population median shared-hand count.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence, Tuple

__all__ = [
    "shrink",
    "PairStat",
    "ShrinkageLayer",
    "KEvaluation",
    "ShrinkageChoice",
    "sweep_shrinkage",
]


def shrink(value: float, n: int, k: float, prior: float) -> float:
    """Empirical-Bayes shrinkage of a per-pair statistic toward a prior.

    Computes ``prior + (value - prior) * n / (n + k)`` where ``n`` is the
    shared-hand count backing ``value`` and ``k >= 0`` is the shrinkage
    pseudo-count.

    Behaviour (design Property 17 / Req 6.1):

    - ``n == 0`` returns ``prior`` exactly -- no data means no update. This holds
      even when ``k == 0`` (the ``0 / 0`` case is defined to be "no data", so the
      prior is returned rather than dividing by zero).
    - The result moves monotonically from ``prior`` toward ``value`` as ``n``
      increases (the weight ``n / (n + k)`` is non-decreasing in ``n`` for
      ``k >= 0``).
    - As ``n`` grows large the weight approaches 1, so the result approaches the
      raw ``value``.

    :param value: the raw per-pair statistic (rate / z-score / excess).
    :param n: shared-hand count backing ``value``; must be ``>= 0``.
    :param k: shrinkage pseudo-count; must be ``>= 0``. ``k = 0`` means no
        shrinkage for any ``n > 0`` (the raw value is returned).
    :param prior: the population prior the statistic is shrunk toward.
    :returns: the shrunk statistic.
    :raises ValueError: if ``n < 0`` or ``k < 0``.
    """

    if n < 0:
        raise ValueError(f"shared-hand count n must be >= 0, got {n!r}")
    if k < 0:
        raise ValueError(f"shrinkage constant k must be >= 0, got {k!r}")

    # No data -> no update. Guards the n=0, k=0 -> 0/0 case explicitly so we
    # never divide by zero; the estimator's limit as n->0 is the prior anyway.
    if n == 0:
        return prior

    weight = n / (n + k)
    return prior + (value - prior) * weight


@dataclass(frozen=True)
class PairStat:
    """One pair's raw per-pair statistic and the shared-hand count backing it.

    ``value`` is the raw statistic (higher = more suspicious in the layer's
    convention); ``n`` is the number of hands the pair actually shared, which is
    what shrinkage weights by (accountability contract Rule 16).
    """

    pair_id: str
    value: float
    n: int


@dataclass(frozen=True)
class ShrinkageLayer:
    """The input to :func:`sweep_shrinkage`: a layer's raw per-pair statistics.

    A thin, immutable carrier so the sweep does not depend on the concrete layer
    implementation -- any layer that can express its per-pair statistic as
    ``(pair_id, value, n)`` triples plus a population ``prior`` and a top-K size
    can be swept.

    :param name: the layer name (for reporting / the ``ShrinkageChoice``).
    :param stats: the per-pair statistics; each carries its own shared-hand count.
    :param prior: the population prior every statistic is shrunk toward.
    :param top_k: how many top-by-shrunk-score pairs define the "top-K" whose
        median shared-hand count is compared against the population median.
    """

    name: str
    stats: Tuple[PairStat, ...]
    prior: float
    top_k: int

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k!r}")
        object.__setattr__(self, "stats", tuple(self.stats))


@dataclass(frozen=True)
class KEvaluation:
    """The verification record for one shrinkage constant in the swept grid.

    Records the constant ``k`` that was evaluated, the median shared-hand count
    of the top-K pairs it produced (``topk_median_n``), the population median
    shared-hand count it was compared against (``population_median_n``), and the
    resulting ``verified`` verdict (top-K median NOT below population median).
    """

    k: float
    topk_median_n: float
    population_median_n: float
    verified: bool


@dataclass(frozen=True)
class ShrinkageChoice:
    """The result of an exhaustive shrinkage-constant sweep (design Property 18).

    ``chosen_k`` is the selected shrinkage constant (``None`` when no constant in
    the grid produced a verified top-K). ``evaluations`` holds one
    :class:`KEvaluation` for EVERY constant in the swept grid, in grid order --
    the exhaustiveness of the sweep is auditable from ``len(evaluations)``.
    ``verified`` is True iff a verified constant was chosen.
    """

    layer_name: str
    chosen_k: float | None
    evaluations: Tuple[KEvaluation, ...]
    verified: bool


def _topk_pair_ids(layer: ShrinkageLayer, k: float) -> Sequence[str]:
    """Rank pairs by shrunk score (descending) and return the top-K pair ids.

    Ties are broken by ``pair_id`` so the ranking is deterministic.
    """

    scored = [
        (shrink(stat.value, stat.n, k, layer.prior), stat.pair_id)
        for stat in layer.stats
    ]
    # Sort by shrunk score descending, then pair_id ascending for determinism.
    scored.sort(key=lambda sp: (-sp[0], sp[1]))
    top = scored[: layer.top_k]
    return [pair_id for _score, pair_id in top]


def sweep_shrinkage(layer: ShrinkageLayer, k_grid: Sequence[float]) -> ShrinkageChoice:
    """Exhaustively sweep a shrinkage-constant grid and pick a verified constant.

    For EACH constant ``k`` in ``k_grid`` (in order, with no early stop -- Req
    6.2 / Property 18):

    1. Shrink every pair's statistic with :func:`shrink` at that ``k``.
    2. Take the top-K pairs by shrunk score.
    3. Verify: the median shared-hand count of those top-K pairs must be NOT
       below the population median shared-hand count. If it is at or above, the
       shrinkage has stopped promoting small-sample noise into the top of the
       ranking and the constant is ``verified``.

    The full per-``k`` record is retained in the returned
    :class:`ShrinkageChoice` regardless of how early a verified constant appears,
    so the exhaustiveness is externally auditable. Among verified constants the
    smallest ``k`` is chosen (the least shrinkage that still passes -- keep as
    much signal as verification allows). If no constant verifies, ``chosen_k`` is
    ``None`` and ``verified`` is False (an honest null -- accountability contract
    Rule 3).

    :param layer: the layer's raw per-pair statistics + prior + top-K size.
    :param k_grid: the grid of shrinkage constants to evaluate; must be non-empty.
    :returns: a :class:`ShrinkageChoice` with the chosen ``k`` (or ``None``) and
        the per-``k`` verification record for the whole grid.
    :raises ValueError: if ``k_grid`` is empty or the layer has no stats.
    """

    grid = tuple(k_grid)
    if not grid:
        raise ValueError("k_grid must contain at least one shrinkage constant")
    if not layer.stats:
        raise ValueError(f"layer {layer.name!r} has no per-pair statistics to sweep")

    population_median_n = float(median([stat.n for stat in layer.stats]))
    all_n: Mapping[str, int] = {stat.pair_id: stat.n for stat in layer.stats}

    evaluations = []
    for k in grid:
        top_ids = _topk_pair_ids(layer, k)
        topk_median_n = float(median([all_n[pid] for pid in top_ids]))
        # Verified: the top-K's median shared-hand count is NOT below the
        # population median (small-sample noise is no longer over-represented).
        verified = topk_median_n >= population_median_n
        evaluations.append(
            KEvaluation(
                k=k,
                topk_median_n=topk_median_n,
                population_median_n=population_median_n,
                verified=verified,
            )
        )

    # Choose the smallest verified k (least shrinkage that still passes). The
    # full grid was evaluated first (above); selection happens only afterwards,
    # so the sweep never stopped early.
    verified_ks = [ev.k for ev in evaluations if ev.verified]
    chosen_k = min(verified_ks) if verified_ks else None

    return ShrinkageChoice(
        layer_name=layer.name,
        chosen_k=chosen_k,
        evaluations=tuple(evaluations),
        verified=chosen_k is not None,
    )
