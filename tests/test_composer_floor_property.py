"""Property test for the composer's floor guarantee (task 5.3, design Property 6).

One Hypothesis property, one test: for ANY baseline model state (any base
ranking the injected ``base_ranking_fn`` returns), composing a candidate with
EVERY optional layer switched OFF reproduces that floor ranking ORDER-IDENTICALLY.

The real floor ranking needs the feature caches and the reused model, which are
not available in a pure unit test; following the existing ``test_composer.py``
convention, the floor ranking is injected via ``base_ranking_fn``. The generator
varies that injected ranking (a random permutation of distinct pair ids), so
"for any baseline model state" is exercised across >=100 examples.

Requirements: 2.1 (Assumption A4 falsifying test).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.composition.composer import compose
from poker_collusion.tuning_harness.models import (
    CandidateConfig,
    ModelConfig,
    RankVector,
)


# --------------------------------------------------------------------------- #
# Property 6: All optional layers OFF reproduces the floor exactly
# Feature: poker-layered-tuning, Property 6: All optional layers OFF reproduces the floor exactly
# Validates: Requirements 2.1
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(
    # A random "baseline model state": a non-empty ranking of DISTINCT pair ids
    # (distinctness is required - a RankVector must be a strict ordering).
    base_pair_ids=st.lists(
        st.text(min_size=1, max_size=8),
        min_size=1,
        max_size=40,
        unique=True,
    ),
)
def test_property_all_optional_off_reproduces_floor(base_pair_ids: list[str]) -> None:
    base_ranking = RankVector(pair_ids=tuple(base_pair_ids))

    def base_ranking_fn(_model_cfg: ModelConfig) -> RankVector:
        """Stand in for the floor ranking produced by the reused model + caches."""
        return base_ranking

    # All optional layers OFF: an empty layers_on set is the canonical "every
    # optional layer switched OFF" per the property text.
    cfg = CandidateConfig(layers_on=frozenset())

    candidate = compose(cfg, base_ranking_fn=base_ranking_fn)

    # Floor guarantee: the composed ranking is order-identical to the base
    # (floor) ranking for ANY baseline model state.
    assert candidate.ranking.pair_ids == base_ranking.pair_ids
