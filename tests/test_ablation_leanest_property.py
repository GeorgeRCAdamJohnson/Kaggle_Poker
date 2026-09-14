"""Property test for leanest-subset selection in leave-one-out ablation (task 12.4).

Hypothesis test for design Property 11: given any mapping of ON-layer subsets to
holdout scores and any non-negative tolerance, the subset :func:`ablate`
identifies (``AblationResult.leanest_subset``) SHALL be a minimum-cardinality
subset whose holdout AP is within ``tolerance`` of the best observed holdout AP
over all subsets.

The property is asserted INDEPENDENTLY of the implementation: this test
recomputes the best holdout, the within-tolerance set, and the minimum
cardinality directly from the drawn score table, then checks that the chosen
subset is a valid minimum-cardinality within-tolerance subset. The exact
tie-break winner is not asserted (the property text only requires "a
minimum-cardinality subset", not a specific one).

Scoring is INJECTED via a deterministic ``score_fn`` built from the drawn table
(the same boundary the ablation unit tests use), so this is a pure statement
about the leanest-subset arithmetic, isolated from the real (expensive) holdout
scorer.

Requirements: 3.3.
"""

from __future__ import annotations

from itertools import combinations

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.ablation.leave_one_out import ablate
from poker_collusion.tuning_harness.models import CandidateConfig

# The universe of optional layer names the generator draws ON layers from.
_LAYER_NAMES = ("a", "b", "c", "d", "e")


@st.composite
def _config_scores_and_tolerance(
    draw: st.DrawFn,
) -> tuple[CandidateConfig, dict[frozenset[str], float], float]:
    """Draw 1..5 ON layers, a full score table over ALL their subsets, and a tolerance.

    - ``layers_on``: a random subset of 1..5 distinct layer names.
    - ``table``: a deterministic holdout AP for EVERY subset of ``layers_on``
      (including the empty foundation-only set), each drawn independently so the
      best subset is not systematically the full set.
    - ``tolerance``: a random tolerance >= 0.
    """
    n_on = draw(st.integers(min_value=1, max_value=len(_LAYER_NAMES)))
    layers_on = frozenset(draw(st.sampled_from(_LAYER_NAMES)) for _ in range(n_on))
    layers_sorted = sorted(layers_on)

    ap = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
    table: dict[frozenset[str], float] = {}
    for size in range(len(layers_sorted) + 1):
        for combo in combinations(layers_sorted, size):
            table[frozenset(combo)] = draw(ap)

    tolerance = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return CandidateConfig(layers_on=layers_on), table, tolerance


# --------------------------------------------------------------------------- #
# Feature: poker-layered-tuning, Property 11: The chosen subset is the leanest
# within tolerance of the best
# Validates: Requirements 3.3
# --------------------------------------------------------------------------- #
@settings(max_examples=200)
@given(_config_scores_and_tolerance())
def test_property_leanest_subset_is_minimum_cardinality_within_tolerance(
    case: tuple[CandidateConfig, dict[frozenset[str], float], float],
) -> None:
    cfg, table, tolerance = case

    def score_fn(candidate: CandidateConfig) -> float:
        return table[frozenset(candidate.layers_on)]

    result = ablate(cfg, score_fn, tolerance=tolerance)

    # Recompute the property's terms INDEPENDENTLY from the drawn score table.
    best = max(table.values())
    within = {subset for subset, value in table.items() if value >= best - tolerance}
    min_card = min(len(subset) for subset in within)

    # best_holdout is the max over all subsets.
    assert result.best_holdout == best

    # The chosen subset is within tolerance of the best...
    assert result.leanest_subset in within
    assert result.leanest_holdout >= best - tolerance
    # ...and IS a minimum-cardinality within-tolerance subset (the property text).
    assert len(result.leanest_subset) == min_card
