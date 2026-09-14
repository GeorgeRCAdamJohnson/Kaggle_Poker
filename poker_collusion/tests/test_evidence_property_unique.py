# Feature: poker-collusion-detection, Property 11: No repeated evidence hand within a pair
"""Property 11 (task 12.4): no hand id repeats among a pair's selected evidence slots.

**Validates: Requirements 8.8**

For any candidate list handed to
:func:`poker_collusion.evidence.retriever.select_evidence` — even one containing DUPLICATE hand
ids — the non-``NO_EVIDENCE`` slots of the result MUST contain each hand id at most once. The
``NO_EVIDENCE`` padding literal MAY repeat (that is the intended fill for empty positions); only
real hand ids must be unique.

To stress de-duplication, the generator deliberately draws hand ids from a SMALL pool so the same
id appears many times across candidates (with differing phase/flags/strength). No matter which
duplicate is kept, the id must appear once at most in the slots.

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable it transparently
falls back to a seeded randomized loop (200 seeds) drawing random candidate lists (0..12 hands)
over a small hand-id pool so duplicates are frequent. The active backend is recorded in
``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random
from typing import Dict, List

import pytest

from poker_collusion.config import DISCLOSED_FAMILY_NAMES, NO_EVIDENCE
from poker_collusion.evidence.retriever import select_evidence
from poker_collusion.types import NOT_APPLICABLE, HandSignal

PAIR_ID = "1_2"
PHASES = ("development", "evaluation")

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"


# --------------------------------------------------------------------------- #
# Synthetic HandSignal builder                                                #
# --------------------------------------------------------------------------- #
def _make_signal(
    hand_id: int,
    *,
    phase: str,
    value_flow: float,
    aggression_asymmetry,
    isolation,
    flags: Dict[str, bool],
) -> HandSignal:
    return HandSignal(
        hand_id=hand_id,
        pair_id=PAIR_ID,
        phase=phase,
        value_flow=value_flow,
        aggression_asymmetry=aggression_asymmetry,
        isolation=isolation,
        mi_conflict=0.0,
        behavior_action_flags=dict(flags),
    )


def _draw_signal(rng: random.Random) -> HandSignal:
    # Small id pool (1..4) so duplicate hand ids appear often across the list.
    hand_id = rng.randint(1, 4)
    phase = rng.choice(PHASES)
    value_flow = rng.uniform(-500.0, 500.0)
    aggression_asymmetry = NOT_APPLICABLE if rng.random() < 0.35 else rng.uniform(0.0, 1.0)
    isolation = NOT_APPLICABLE if rng.random() < 0.35 else rng.uniform(0.0, 10.0)
    if rng.random() < 0.30:
        flags = {fam: False for fam in DISCLOSED_FAMILY_NAMES}
    else:
        flags = {fam: (rng.random() < 0.5) for fam in DISCLOSED_FAMILY_NAMES}
        if not any(flags.values()):
            flags[rng.choice(DISCLOSED_FAMILY_NAMES)] = True
    return _make_signal(
        hand_id,
        phase=phase,
        value_flow=value_flow,
        aggression_asymmetry=aggression_asymmetry,
        isolation=isolation,
        flags=flags,
    )


def _draw_candidates(rng: random.Random) -> List[HandSignal]:
    n = rng.randint(0, 12)
    return [_draw_signal(rng) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Core property assertion                                                     #
# --------------------------------------------------------------------------- #
def _assert_no_repeated_hand(candidates: List[HandSignal]) -> None:
    sel = select_evidence(PAIR_ID, candidates)
    hand_ids = [cell for cell in sel.slots if cell != NO_EVIDENCE]
    assert len(hand_ids) == len(set(hand_ids)), (
        f"repeated evidence hand id in slots {sel.slots!r}"
    )


# --------------------------------------------------------------------------- #
# Explicit corner cases                                                       #
# --------------------------------------------------------------------------- #
def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 11] generator in use: {GENERATOR_IN_USE}")


def test_duplicate_hand_ids_deduplicated_example():
    """Many valid copies of the same hand id collapse to a single slot entry."""
    dups = [
        _make_signal(
            2, phase="evaluation", value_flow=float(i), aggression_asymmetry=NOT_APPLICABLE,
            isolation=NOT_APPLICABLE,
            flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
        )
        for i in range(1, 6)
    ]
    other = _make_signal(
        3, phase="evaluation", value_flow=3.0, aggression_asymmetry=NOT_APPLICABLE,
        isolation=NOT_APPLICABLE,
        flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
    )
    _assert_no_repeated_hand(dups + [other])
    sel = select_evidence(PAIR_ID, dups + [other])
    real = [c for c in sel.slots if c != NO_EVIDENCE]
    assert sorted(real) == ["2", "3"]


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback         #
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _flag_strategy = st.fixed_dictionaries(
        {fam: st.booleans() for fam in DISCLOSED_FAMILY_NAMES}
    )
    _ratio_strategy = st.one_of(
        st.just(NOT_APPLICABLE),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    _signal_strategy = st.builds(
        _make_signal,
        hand_id=st.integers(min_value=1, max_value=4),  # small pool -> frequent duplicates
        phase=st.sampled_from(PHASES),
        value_flow=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        aggression_asymmetry=_ratio_strategy,
        isolation=_ratio_strategy,
        flags=_flag_strategy,
    )
    _candidates_strategy = st.lists(_signal_strategy, min_size=0, max_size=12)

    @settings(max_examples=200, deadline=None)
    @given(candidates=_candidates_strategy)
    def test_property_unique_hypothesis(candidates):
        _assert_no_repeated_hand(list(candidates))

else:

    _N_SEEDS = 200

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_unique_seeded_loop(seed: int):
        rng = random.Random(seed)
        candidates = _draw_candidates(rng)
        _assert_no_repeated_hand(candidates)
