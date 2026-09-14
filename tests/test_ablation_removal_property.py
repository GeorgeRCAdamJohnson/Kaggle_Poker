"""Property test for the ablation removal recommendation (task 12.3).

Property 10 (design.md): For any ablation result, every layer whose unique
leave-one-out contribution is less than or equal to zero SHALL be recommended
for removal (Req 3.2). Equivalently, the removal set is EXACTLY the set of
non-positive-contribution ON layers -- every non-positive-contribution layer is
recommended, and every positive-contribution layer is not.

The candidate's ON layers are ablated through an INJECTED deterministic
``score_fn`` (the same boundary the ablation unit tests use): a full score table
over EVERY subset of the ON layers with random holdout AP per subset. Because
each subset's AP is independent, the leave-one-out difference
``full - holdout_without_layer`` spans positive, zero, and negative values, so
the iff below is exercised across all three regimes.

Validates: Requirements 3.2.
"""

from __future__ import annotations

from itertools import chain, combinations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.ablation.leave_one_out import ablate
from poker_collusion.tuning_harness.models import CandidateConfig


# A small pool of layer names; 1..5 are drawn ON for each example.
_LAYER_POOL = ("board_equity", "directional", "iso", "whipsaw", "negative_space")
_FINITE = dict(allow_nan=False, allow_infinity=False)


def _all_subsets(layers: tuple[str, ...]):
    """Every subset of ``layers`` (the full set down to the empty foundation-only set)."""
    return chain.from_iterable(
        combinations(layers, r) for r in range(len(layers) + 1)
    )


@st.composite
def _cfg_and_scorer(draw):
    """A candidate (1..5 ON layers) plus a deterministic score_fn over ALL subsets.

    Each subset of the ON layers is assigned an independent random holdout AP, so
    the leave-one-out contribution ``full - holdout_without_layer`` for each layer
    can be positive (load-bearing), zero (redundant), or negative (corrupting).
    A dense band of AP values (rounded to make exact ties plausible) makes the
    contribution == 0 boundary reachable, not just measure-zero.
    """
    n = draw(st.integers(min_value=1, max_value=5))
    layers = tuple(_LAYER_POOL[:n])

    subsets = list(_all_subsets(layers))
    # Round to 2 decimals over a small band so adjacent subsets can tie exactly,
    # making zero-contribution layers (contribution == 0) genuinely reachable.
    table: dict[frozenset[str], float] = {}
    for subset in subsets:
        ap = draw(st.floats(min_value=0.0, max_value=0.5, **_FINITE))
        table[frozenset(subset)] = round(ap, 2)

    def score_fn(cfg: CandidateConfig) -> float:
        return table[frozenset(cfg.layers_on)]

    return CandidateConfig(layers_on=frozenset(layers)), score_fn


# Feature: poker-layered-tuning, Property 10: Non-positive contribution triggers a removal recommendation
@settings(max_examples=200)
@given(_cfg_and_scorer())
def test_property_non_positive_contribution_triggers_removal(cfg_and_scorer) -> None:
    cfg, score_fn = cfg_and_scorer
    result = ablate(cfg, score_fn)

    # Req 3.2 / Property 10: the removal set is EXACTLY the non-positive-contribution
    # ON layers -- every layer with contribution <= 0 is recommended for removal,
    # and every layer with contribution > 0 is not.
    expected = {
        layer
        for layer in cfg.layers_on
        if result.contributions[layer] <= 0.0
    }
    assert set(result.removal_recommended) == expected

    # And the recommendation is reported in stable sorted order (design contract).
    assert list(result.removal_recommended) == sorted(result.removal_recommended)
