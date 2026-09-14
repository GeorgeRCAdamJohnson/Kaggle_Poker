"""Property-based test for leave-one-out ablation contributions (task 12.2).

# Feature: poker-layered-tuning, Property 9: Ablation reports each layer's leave-one-out contribution

Design Property 9 (Validates: Requirements 3.1):
    For any multi-layer candidate, the reported unique contribution of each ON
    layer SHALL equal the full-candidate holdout AP minus the holdout AP of the
    candidate with that single layer removed.

Strategy
--------
We draw a set of 1..5 distinct ON layers, then assign a deterministic holdout AP
to EVERY subset of those layers (a full lookup table). An injected ``score_fn``
just reads that table for a config's ``layers_on``. Because every subset's score
is pre-drawn, the expected leave-one-out arithmetic
``table[full] - table[full - {L}]`` is independently computable, so the assertion
does not re-derive it from the implementation under test.
"""

from __future__ import annotations

from itertools import chain, combinations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.ablation.leave_one_out import ablate
from poker_collusion.tuning_harness.models import CandidateConfig


def _all_subsets(layers: frozenset[str]) -> list[frozenset[str]]:
    """Every subset of ``layers`` (including the empty set)."""
    items = sorted(layers)
    return [
        frozenset(combo)
        for r in range(len(items) + 1)
        for combo in combinations(items, r)
    ]


@st.composite
def _on_layers_and_table(draw):
    """Draw 1..5 distinct ON layers and a holdout AP for every subset of them."""
    on_layers = draw(
        st.sets(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=4),
            min_size=1,
            max_size=5,
        )
    )
    on_layers = frozenset(on_layers)
    ap = st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
    table = {subset: draw(ap) for subset in _all_subsets(on_layers)}
    return on_layers, table


@given(_on_layers_and_table())
@settings(max_examples=200)
def test_ablation_reports_leave_one_out_contribution(data) -> None:
    on_layers, table = data
    full_set = on_layers

    def score_fn(cfg: CandidateConfig) -> float:
        return table[frozenset(cfg.layers_on)]

    result = ablate(CandidateConfig(layers_on=full_set), score_fn)

    # The contribution map's keys are EXACTLY the ON layers.
    assert set(result.contributions) == set(full_set)

    # Property 9: each ON layer's reported contribution equals
    # full_holdout - holdout(full_set without that layer), computed independently
    # from the pre-drawn table.
    for layer in full_set:
        expected = table[full_set] - table[full_set - {layer}]
        assert result.contributions[layer] == pytest.approx(expected)
