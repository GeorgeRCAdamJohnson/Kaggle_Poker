"""Property-based tests for the floor-guaranteed composer (task 5.2).

Covers the RANK_BLEND weight-zero identity: composing any layer onto any base at
weight 0.0 is a no-op that returns the base ordering EXACTLY (order-identical).
This is the hard floor guarantee (Requirement 2.2, design Property 7) - the
falsifying test for Assumption A5. If this ever failed, a "neutral" layer could
silently perturb the LB-verified floor ordering, which the accountability
contract Rule 15 forbids.

Requirements: 2.2.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.composition.composer import rank_blend
from poker_collusion.tuning_harness.models import RankVector


# A pool of candidate pair ids to draw distinct permutations from. Kept small so
# base and layer sets overlap for some examples and stay disjoint for others.
_PAIR_ID_POOL = [f"p{i}" for i in range(12)]


def _rankings():
    """Draw (base_ranking, layer_ranking): each a permutation of a distinct pair
    subset. The two subsets may overlap fully, partially, or be disjoint, so the
    weight-0 identity is exercised across covered and uncovered pairs alike."""

    return st.tuples(
        st.permutations(_PAIR_ID_POOL).flatmap(
            lambda perm: st.integers(min_value=1, max_value=len(perm)).map(
                lambda k: RankVector(pair_ids=tuple(perm[:k]))
            )
        ),
        st.permutations(_PAIR_ID_POOL).flatmap(
            lambda perm: st.integers(min_value=0, max_value=len(perm)).map(
                lambda k: RankVector(pair_ids=tuple(perm[:k]))
            )
        ),
    )


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 7: RANK_BLEND at weight zero equals
# the base ranking exactly
# Validates: Requirements 2.2
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(rankings=_rankings())
def test_property_rank_blend_weight_zero_is_base_identity(
    rankings: tuple[RankVector, RankVector],
) -> None:
    base_ranking, layer_ranking = rankings

    blended = rank_blend(base_ranking, layer_ranking, 0.0)

    # Order-identical to the base ranking for ALL generated inputs.
    assert blended.pair_ids == base_ranking.pair_ids
