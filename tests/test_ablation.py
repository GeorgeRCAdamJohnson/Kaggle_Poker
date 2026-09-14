"""Unit tests for leave-one-out ablation (task 12.1, design ``ablation/leave_one_out.py``).

Example / edge-case unit tests for :func:`ablate`, exercised through an INJECTED
``score_fn`` so the leave-one-out arithmetic, removal recommendations, and
leanest-subset search are tested in isolation from the real (expensive) holdout
scorer -- exactly the boundary the composer and holdout tests use. The Hypothesis
property tests for Properties 9/10/11 are separate tasks (12.2-12.4); these are
the concrete examples and edge cases.

The injected scorers here map an ON-layer set to a fixed holdout AP, so every
assertion is a deterministic statement about the ablation arithmetic:

  (a) each layer's reported contribution EQUALS full - without-layer (Req 3.1);
  (b) a layer whose contribution is <= 0 is recommended for removal (Req 3.2);
  (c) the leanest subset within tolerance is the minimum-cardinality subset whose
      holdout is within tolerance of the best (Req 3.3).

Requirements: 3.1, 3.2, 3.3.
"""

from __future__ import annotations

import pytest

from poker_collusion.tuning_harness.ablation.leave_one_out import (
    AblationResult,
    ablate,
)
from poker_collusion.tuning_harness.models import CandidateConfig


def _cfg(*layers: str) -> CandidateConfig:
    """A candidate with the given optional layers ON (default knobs)."""
    return CandidateConfig(layers_on=frozenset(layers))


def _scorer(table: dict[frozenset[str], float], default: float = 0.0):
    """Build a deterministic score_fn from an {ON-layer-set: holdout_ap} table."""

    def score_fn(cfg: CandidateConfig) -> float:
        return table.get(frozenset(cfg.layers_on), default)

    return score_fn


# --------------------------------------------------------------------------- #
# (a) leave-one-out contribution == full - without-layer  -- Req 3.1
# --------------------------------------------------------------------------- #
def test_contribution_equals_full_minus_without_each_layer() -> None:
    # Full stack scores 0.40; removing A drops to 0.37 (A worth +0.03),
    # removing B drops to 0.39 (B worth +0.01).
    table = {
        frozenset({"a", "b"}): 0.40,
        frozenset({"b"}): 0.37,  # A removed
        frozenset({"a"}): 0.39,  # B removed
        frozenset(): 0.30,
    }
    result = ablate(_cfg("a", "b"), _scorer(table))

    assert isinstance(result, AblationResult)
    assert result.full_holdout == 0.40
    # Req 3.1 / Property 9: contribution = full - holdout_without_layer.
    assert result.contributions["a"] == pytest.approx(0.40 - 0.37)
    assert result.contributions["b"] == pytest.approx(0.40 - 0.39)


def test_contribution_computed_for_every_on_layer() -> None:
    table = {
        frozenset({"a", "b", "c"}): 0.50,
        frozenset({"b", "c"}): 0.48,  # A removed
        frozenset({"a", "c"}): 0.45,  # B removed
        frozenset({"a", "b"}): 0.49,  # C removed
    }
    result = ablate(_cfg("a", "b", "c"), _scorer(table))

    assert set(result.contributions) == {"a", "b", "c"}
    assert result.contributions["a"] == pytest.approx(0.50 - 0.48)
    assert result.contributions["b"] == pytest.approx(0.50 - 0.45)
    assert result.contributions["c"] == pytest.approx(0.50 - 0.49)


# --------------------------------------------------------------------------- #
# (b) non-positive contribution => recommend removal  -- Req 3.2
# --------------------------------------------------------------------------- #
def test_zero_and_negative_contribution_layers_are_recommended_for_removal() -> None:
    # A: load-bearing (+0.03). B: redundant (0.0). C: corrupting (-0.02, i.e.
    # removing C IMPROVES the holdout, so full < without-C).
    table = {
        frozenset({"a", "b", "c"}): 0.40,
        frozenset({"b", "c"}): 0.37,  # A removed -> A worth +0.03
        frozenset({"a", "c"}): 0.40,  # B removed -> B worth 0.0
        frozenset({"a", "b"}): 0.42,  # C removed -> C worth -0.02
    }
    result = ablate(_cfg("a", "b", "c"), _scorer(table))

    assert result.contributions["a"] > 0.0
    assert result.contributions["b"] == pytest.approx(0.0)
    assert result.contributions["c"] < 0.0
    # Req 3.2 / Property 10: contribution <= 0 (B and C) is recommended for removal;
    # the load-bearing A is not. Reported in stable sorted order.
    assert result.removal_recommended == ("b", "c")


def test_all_positive_contributions_recommend_no_removal() -> None:
    table = {
        frozenset({"a", "b"}): 0.40,
        frozenset({"b"}): 0.35,
        frozenset({"a"}): 0.36,
        frozenset(): 0.30,
    }
    result = ablate(_cfg("a", "b"), _scorer(table))
    assert result.removal_recommended == ()


# --------------------------------------------------------------------------- #
# (c) leanest subset within tolerance of the best  -- Req 3.3
# --------------------------------------------------------------------------- #
def test_leanest_subset_is_minimum_cardinality_within_tolerance() -> None:
    # Best holdout is the 2-layer {a,b} at 0.40, but {a} alone reaches 0.395,
    # within a tolerance of 0.01. So the leanest subset within tolerance is {a}.
    table = {
        frozenset({"a", "b"}): 0.40,
        frozenset({"a"}): 0.395,
        frozenset({"b"}): 0.34,
        frozenset(): 0.30,
    }
    result = ablate(_cfg("a", "b"), _scorer(table), tolerance=0.01)

    assert result.best_holdout == pytest.approx(0.40)
    # {a} (cardinality 1) is within 0.01 of the best and is leaner than {a,b}.
    assert result.leanest_subset == frozenset({"a"})
    assert result.leanest_holdout == pytest.approx(0.395)


def test_leanest_subset_is_full_set_when_tolerance_is_zero_and_full_is_best() -> None:
    # With zero tolerance, only a subset matching the best exactly qualifies.
    table = {
        frozenset({"a", "b"}): 0.40,
        frozenset({"a"}): 0.395,
        frozenset({"b"}): 0.34,
        frozenset(): 0.30,
    }
    result = ablate(_cfg("a", "b"), _scorer(table), tolerance=0.0)

    # No leaner subset reaches 0.40, so the full set is the leanest within tol 0.
    assert result.leanest_subset == frozenset({"a", "b"})
    assert result.leanest_holdout == pytest.approx(0.40)


def test_leanest_subset_can_be_empty_foundation_only_within_tolerance() -> None:
    # Every subset is within a wide tolerance of the best, so the leanest (the
    # empty set = foundation only) is chosen.
    table = {
        frozenset({"a", "b"}): 0.40,
        frozenset({"a"}): 0.39,
        frozenset({"b"}): 0.38,
        frozenset(): 0.37,
    }
    result = ablate(_cfg("a", "b"), _scorer(table), tolerance=0.05)

    assert result.leanest_subset == frozenset()
    assert result.leanest_holdout == pytest.approx(0.37)


def test_leanest_subset_ties_break_by_higher_holdout_then_lexicographic() -> None:
    # Two singletons both within tolerance of the best (0.40): {a}=0.395 and
    # {b}=0.398. Same cardinality -> the higher-scoring {b} wins.
    table = {
        frozenset({"a", "b"}): 0.40,
        frozenset({"a"}): 0.395,
        frozenset({"b"}): 0.398,
        frozenset(): 0.30,
    }
    result = ablate(_cfg("a", "b"), _scorer(table), tolerance=0.01)

    assert result.leanest_subset == frozenset({"b"})


# --------------------------------------------------------------------------- #
# Edge cases / contract
# --------------------------------------------------------------------------- #
def test_negative_tolerance_raises() -> None:
    with pytest.raises(ValueError):
        ablate(_cfg("a"), _scorer({frozenset({"a"}): 0.4}), tolerance=-0.01)


def test_single_layer_candidate() -> None:
    table = {frozenset({"a"}): 0.40, frozenset(): 0.30}
    result = ablate(_cfg("a"), _scorer(table), tolerance=0.0)

    assert result.contributions == {"a": pytest.approx(0.10)}
    assert result.removal_recommended == ()
    assert result.leanest_subset == frozenset({"a"})


def test_score_fn_called_once_per_distinct_subset() -> None:
    # The memoised cache means each distinct ON-layer subset is scored at most
    # once even though leave-one-out and the leanest search both need the empty
    # set, {a}, {b}, and {a,b}.
    calls: list[frozenset[str]] = []

    def counting_score(cfg: CandidateConfig) -> float:
        calls.append(frozenset(cfg.layers_on))
        return {
            frozenset({"a", "b"}): 0.40,
            frozenset({"a"}): 0.39,
            frozenset({"b"}): 0.34,
            frozenset(): 0.30,
        }[frozenset(cfg.layers_on)]

    ablate(_cfg("a", "b"), counting_score, tolerance=0.0)

    # Four distinct subsets, each scored exactly once.
    assert len(calls) == len(set(calls)) == 4


def test_removed_layer_config_carries_other_knobs_through() -> None:
    # The leave-one-out config for "remove A" must keep B present and unchanged;
    # only A disappears from layers_on and composition.
    seen: list[frozenset[str]] = []

    def score_fn(cfg: CandidateConfig) -> float:
        seen.append(frozenset(cfg.layers_on))
        return {
            frozenset({"a", "b"}): 0.40,
            frozenset({"b"}): 0.37,
            frozenset({"a"}): 0.39,
            frozenset(): 0.30,
        }[frozenset(cfg.layers_on)]

    ablate(_cfg("a", "b"), score_fn, tolerance=0.0)

    # The "A removed" evaluation kept exactly {b}; the "B removed" kept exactly {a}.
    assert frozenset({"b"}) in seen
    assert frozenset({"a"}) in seen
