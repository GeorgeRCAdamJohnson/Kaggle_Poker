"""Property test for the exhaustive verified shrinkage sweep (task 11.3).

Hypothesis test for design Property 18: ``sweep_shrinkage`` evaluates EVERY
constant in the grid in order (never stopping early even once an earlier
constant verifies), and any chosen constant is one whose top-K-by-score median
shared-hand count is not below the population median shared-hand count. When no
constant verifies the choice is an honest null (``chosen_k is None``,
``verified is False``).

Requirements: 6.2.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.shrinkage import (
    PairStat,
    ShrinkageLayer,
    sweep_shrinkage,
)


@st.composite
def _layer_and_grid(draw: st.DrawFn) -> tuple[ShrinkageLayer, list[float]]:
    """Draw a random ShrinkageLayer with distinct pair_ids and a non-empty grid.

    - stats: 1..8 PairStats, each with a random value and n >= 0; pair_ids are
      distinct (a set of index-derived ids).
    - prior: any finite float.
    - top_k: 1 .. len(stats).
    - k_grid: 1..6 distinct k >= 0 constants (kept in draw order).
    """
    n_stats = draw(st.integers(min_value=1, max_value=8))
    values = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_stats,
            max_size=n_stats,
        )
    )
    ns = draw(
        st.lists(st.integers(min_value=0, max_value=500), min_size=n_stats, max_size=n_stats)
    )
    stats = tuple(
        PairStat(pair_id=f"p{i}", value=values[i], n=ns[i]) for i in range(n_stats)
    )
    prior = draw(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    top_k = draw(st.integers(min_value=1, max_value=n_stats))
    layer = ShrinkageLayer(name="L", stats=stats, prior=prior, top_k=top_k)

    k_grid = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    return layer, k_grid


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 18: The shrinkage sweep is exhaustive
# and its choice is verified
# Validates: Requirements 6.2
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(_layer_and_grid())
def test_property_sweep_exhaustive_and_choice_verified(
    layer_and_grid: tuple[ShrinkageLayer, list[float]],
) -> None:
    layer, k_grid = layer_and_grid
    choice = sweep_shrinkage(layer, k_grid)

    # Exhaustiveness: one evaluation per grid constant, in grid order, no early stop.
    assert len(choice.evaluations) == len(k_grid)
    assert [ev.k for ev in choice.evaluations] == list(k_grid)

    if choice.verified:
        # A verified choice: chosen_k is a real grid constant whose top-K median
        # shared-hand count is NOT below the population median.
        assert choice.chosen_k is not None
        assert choice.chosen_k in k_grid
        chosen_ev = next(ev for ev in choice.evaluations if ev.k == choice.chosen_k)
        assert chosen_ev.topk_median_n >= chosen_ev.population_median_n
    else:
        # Honest null: no constant verified, so no k is chosen.
        assert choice.chosen_k is None
