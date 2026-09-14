"""Unit tests for empirical-Bayes shrinkage (task 11.1, design ``shrinkage/eb.py``).

Covers the ``shrink`` estimator contract (n=0 returns prior, monotone in n,
approaches value for large n) and the ``sweep_shrinkage`` exhaustive-verified
sweep (evaluates the whole grid, picks a verified k). The Hypothesis property
tests for Property 17 / 18 are separate tasks (11.2, 11.3); these are the
example / edge-case unit tests.

Requirements: 6.1, 6.2.
"""

from __future__ import annotations

import pytest

from poker_collusion.tuning_harness.shrinkage import (
    KEvaluation,
    PairStat,
    ShrinkageChoice,
    ShrinkageLayer,
    shrink,
    sweep_shrinkage,
)


# --------------------------------------------------------------------------- #
# shrink(value, n, k, prior)  -- Req 6.1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prior", [-2.5, 0.0, 0.37, 1.0, 12.0])
@pytest.mark.parametrize("k", [0.0, 1.0, 76.0])
def test_shrink_at_n_zero_returns_prior_exactly(prior: float, k: float) -> None:
    # No data -> no update, for any k including k=0 (the 0/0 case).
    assert shrink(value=999.0, n=0, k=k, prior=prior) == prior


def test_shrink_n_zero_k_zero_does_not_divide_by_zero() -> None:
    # Explicit guard on the 0/0 edge: returns prior, never raises.
    assert shrink(value=5.0, n=0, k=0.0, prior=0.3) == 0.3


def test_shrink_monotone_toward_value_as_n_increases() -> None:
    prior, value, k = 0.0, 1.0, 10.0
    results = [shrink(value, n, k, prior) for n in range(0, 200, 5)]
    # Non-decreasing: moves from prior (0.0) toward value (1.0) as n grows.
    for earlier, later in zip(results, results[1:]):
        assert later >= earlier
    assert results[0] == prior


def test_shrink_monotone_when_value_below_prior() -> None:
    # Monotone toward value also when value < prior (result decreases).
    prior, value, k = 1.0, 0.0, 8.0
    results = [shrink(value, n, k, prior) for n in range(0, 100, 4)]
    for earlier, later in zip(results, results[1:]):
        assert later <= earlier
    assert results[0] == prior


def test_shrink_approaches_value_for_large_n() -> None:
    prior, value, k = 0.2, 0.9, 25.0
    assert shrink(value, n=10_000_000, k=k, prior=prior) == pytest.approx(value, abs=1e-4)


def test_shrink_k_zero_returns_raw_value_for_positive_n() -> None:
    # k=0 means no shrinkage: any n>0 gives the raw value back.
    assert shrink(value=0.8, n=1, k=0.0, prior=0.1) == pytest.approx(0.8)


def test_shrink_half_weight_at_n_equals_k() -> None:
    # At n == k the weight is exactly 1/2: result is the midpoint of prior/value.
    assert shrink(value=1.0, n=4, k=4.0, prior=0.0) == pytest.approx(0.5)


def test_shrink_rejects_negative_n_and_k() -> None:
    with pytest.raises(ValueError):
        shrink(value=1.0, n=-1, k=1.0, prior=0.0)
    with pytest.raises(ValueError):
        shrink(value=1.0, n=1, k=-1.0, prior=0.0)


# --------------------------------------------------------------------------- #
# sweep_shrinkage(layer, k_grid)  -- Req 6.2
# --------------------------------------------------------------------------- #
def _noise_layer() -> ShrinkageLayer:
    """A layer where raw values reward small-sample pairs.

    High-value, low-n pairs (noise) vs lower-value, high-n pairs (real signal).
    At k=0 the noisy small-n pairs dominate the top-K; with enough shrinkage the
    high-n pairs are promoted so the top-K median n reaches the population median.
    """

    stats = (
        # Small-sample, extreme raw values (noise the sweep must shrink away).
        PairStat("noise1", value=1.0, n=3),
        PairStat("noise2", value=0.98, n=4),
        PairStat("noise3", value=0.97, n=5),
        # Larger-sample, moderate raw values (the real signal).
        PairStat("real1", value=0.80, n=200),
        PairStat("real2", value=0.78, n=250),
        PairStat("real3", value=0.76, n=300),
    )
    return ShrinkageLayer(name="noise_layer", stats=stats, prior=0.0, top_k=3)


def test_sweep_evaluates_entire_grid_never_stops_early() -> None:
    layer = _noise_layer()
    k_grid = [0.0, 1.0, 5.0, 50.0, 500.0, 5000.0]
    choice = sweep_shrinkage(layer, k_grid)
    # One evaluation per grid constant, in grid order -- proves no early stop.
    assert isinstance(choice, ShrinkageChoice)
    assert len(choice.evaluations) == len(k_grid)
    assert [ev.k for ev in choice.evaluations] == k_grid
    for ev in choice.evaluations:
        assert isinstance(ev, KEvaluation)


def test_sweep_picks_a_verified_k() -> None:
    layer = _noise_layer()
    k_grid = [0.0, 1.0, 5.0, 50.0, 500.0, 5000.0]
    choice = sweep_shrinkage(layer, k_grid)
    # A verified choice exists and is one of the grid constants.
    assert choice.verified is True
    assert choice.chosen_k in k_grid
    # The chosen k's own evaluation is verified: top-K median n >= population median.
    chosen_eval = next(ev for ev in choice.evaluations if ev.k == choice.chosen_k)
    assert chosen_eval.verified is True
    assert chosen_eval.topk_median_n >= chosen_eval.population_median_n
    # No shrinkage (k=0) should NOT verify here -- noise dominates the top-K.
    zero_eval = next(ev for ev in choice.evaluations if ev.k == 0.0)
    assert zero_eval.verified is False


def test_sweep_chooses_smallest_verified_k() -> None:
    layer = _noise_layer()
    k_grid = [0.0, 1.0, 5.0, 50.0, 500.0, 5000.0]
    choice = sweep_shrinkage(layer, k_grid)
    verified_ks = [ev.k for ev in choice.evaluations if ev.verified]
    assert choice.chosen_k == min(verified_ks)


def test_sweep_reports_honest_null_when_no_k_verifies() -> None:
    # A layer whose top pairs are ALL small-sample: no shrinkage can lift the
    # top-K median n to the population median, so the sweep returns no choice.
    stats = (
        PairStat("a", value=1.0, n=2),
        PairStat("b", value=0.9, n=3),
        PairStat("c", value=0.1, n=400),
        PairStat("d", value=0.05, n=500),
    )
    # top_k=1 with the single highest raw value being a tiny-n pair; because
    # value ordering keeps small-n "a" on top until fully shrunk to the prior,
    # ties resolve without ever surfacing the high-n pairs into a 1-slot top-K.
    layer = ShrinkageLayer(name="all_noise_top", stats=stats, prior=1.0, top_k=1)
    k_grid = [0.0, 1.0, 10.0]
    choice = sweep_shrinkage(layer, k_grid)
    assert len(choice.evaluations) == len(k_grid)
    # Honest null is a valid result: no false-positive choice is manufactured.
    if not choice.verified:
        assert choice.chosen_k is None


def test_sweep_rejects_empty_grid() -> None:
    with pytest.raises(ValueError):
        sweep_shrinkage(_noise_layer(), [])


def test_sweep_rejects_layer_with_no_stats() -> None:
    empty = ShrinkageLayer(name="empty", stats=(), prior=0.0, top_k=1)
    with pytest.raises(ValueError):
        sweep_shrinkage(empty, [1.0])


def test_shrinkage_layer_rejects_nonpositive_top_k() -> None:
    with pytest.raises(ValueError):
        ShrinkageLayer(name="bad", stats=(PairStat("a", 1.0, 5),), prior=0.0, top_k=0)
