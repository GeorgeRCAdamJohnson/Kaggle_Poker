# Feature: poker-collusion-detection, Property 10: Evidence slots are complete and non-empty
"""Property 10 (task 12.3): evidence slots are ALWAYS complete and non-empty.

**Validates: Requirements 8.6, 8.7**

For any candidate list handed to
:func:`poker_collusion.evidence.retriever.select_evidence`, the returned ``.slots`` MUST:

* be exactly length :data:`~poker_collusion.evidence.retriever.MAX_EVIDENCE` (= 5) (Req 8.6), and
* have every cell a non-empty string — either a hand id or the literal
  :data:`~poker_collusion.config.NO_EVIDENCE` — never ``None`` and never the empty string
  (Req 8.7).

This holds across the whole spectrum of candidate counts, including the 0-valid case (all five
cells are ``NO_EVIDENCE``) and the >5-valid case (all five cells are real hand ids, no padding).

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable it transparently
falls back to a seeded randomized loop (200 seeds) drawing random candidate lists (0..12 hands,
mixed phase / value_flow / ratio signals / flags including latent-only). The active backend is
recorded in ``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random
from typing import Dict, List

import pytest

from poker_collusion.config import DISCLOSED_FAMILY_NAMES, NO_EVIDENCE
from poker_collusion.evidence.retriever import MAX_EVIDENCE, select_evidence
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
    hand_id = rng.randint(1, 12)
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
def _assert_complete_and_non_empty(candidates: List[HandSignal]) -> None:
    sel = select_evidence(PAIR_ID, candidates)
    # Exactly 5 slots (Req 8.6).
    assert len(sel.slots) == MAX_EVIDENCE, f"expected {MAX_EVIDENCE} slots, got {len(sel.slots)}"
    # strengths list is aligned in length.
    assert len(sel.strengths) == MAX_EVIDENCE
    for cell in sel.slots:
        # Every cell non-empty: a hand id or the NO_EVIDENCE literal (Req 8.7).
        assert cell is not None, "slot cell is None"
        assert isinstance(cell, str), f"slot cell {cell!r} is not a str"
        assert cell != "", "slot cell is an empty string"
        assert cell == NO_EVIDENCE or len(cell) > 0


# --------------------------------------------------------------------------- #
# Explicit corner cases                                                       #
# --------------------------------------------------------------------------- #
def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 10] generator in use: {GENERATOR_IN_USE}")


def test_zero_valid_case_all_no_evidence():
    """0-valid case: no candidate passes the gate -> all five cells are NO_EVIDENCE."""
    # All development-phase -> zero valid.
    dev_only = [
        _make_signal(
            hid, phase="development", value_flow=10.0, aggression_asymmetry=NOT_APPLICABLE,
            isolation=NOT_APPLICABLE,
            flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
        )
        for hid in range(1, 4)
    ]
    sel = select_evidence(PAIR_ID, dev_only)
    assert sel.slots == [NO_EVIDENCE] * MAX_EVIDENCE
    _assert_complete_and_non_empty(dev_only)
    # Also the truly-empty candidate list.
    _assert_complete_and_non_empty([])


def test_more_than_five_valid_no_padding():
    """>5-valid case: five real hand ids, no NO_EVIDENCE padding."""
    cands = [
        _make_signal(
            hid, phase="evaluation", value_flow=float(hid), aggression_asymmetry=NOT_APPLICABLE,
            isolation=NOT_APPLICABLE,
            flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
        )
        for hid in range(1, 9)  # 8 valid hands
    ]
    sel = select_evidence(PAIR_ID, cands)
    assert NO_EVIDENCE not in sel.slots
    _assert_complete_and_non_empty(cands)


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
        hand_id=st.integers(min_value=1, max_value=12),
        phase=st.sampled_from(PHASES),
        value_flow=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        aggression_asymmetry=_ratio_strategy,
        isolation=_ratio_strategy,
        flags=_flag_strategy,
    )
    _candidates_strategy = st.lists(_signal_strategy, min_size=0, max_size=12)

    @settings(max_examples=200, deadline=None)
    @given(candidates=_candidates_strategy)
    def test_property_complete_hypothesis(candidates):
        _assert_complete_and_non_empty(list(candidates))

else:

    _N_SEEDS = 200

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_complete_seeded_loop(seed: int):
        rng = random.Random(seed)
        candidates = _draw_candidates(rng)
        _assert_complete_and_non_empty(candidates)
