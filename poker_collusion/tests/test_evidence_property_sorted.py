# Feature: poker-collusion-detection, Property 12: Evidence positions are sorted by strength then hand id
"""Property 12 (task 12.5): selected evidence positions are sorted by strength then hand id.

**Validates: Requirements 8.5**

For any candidate list handed to
:func:`poker_collusion.evidence.retriever.select_evidence`, the non-``NO_EVIDENCE`` slots MUST be
ordered by :func:`~poker_collusion.evidence.retriever.evidence_strength` DESCENDING, with ties
broken by ASCENDING hand id (using the same numeric-if-possible ordering the retriever applies:
"10" sorts after "2"). Concretely, for consecutive selected slots i, i+1:

* ``strength[i] >= strength[i+1]``, and
* when ``strength[i] == strength[i+1]`` then ``hand_key(id[i]) < hand_key(id[i+1])``.

Expected per-slot strengths are recomputed independently via ``retriever.evidence_strength`` on the
originating candidate hand, and are asserted to match the ``.strengths`` the selector reports.

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable it transparently
falls back to a seeded randomized loop (200 seeds) drawing random candidate lists (0..12 hands)
with mixed phase / value_flow / ratio signals (some ``NOT_APPLICABLE``) / flags (some latent-only)
and, to force strength ties, occasionally clamps value_flow to a small shared set. The active
backend is recorded in ``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random
from typing import Dict, List, Tuple

import pytest

from poker_collusion.config import DISCLOSED_FAMILY_NAMES, NO_EVIDENCE
from poker_collusion.evidence.retriever import evidence_strength, select_evidence
from poker_collusion.types import NOT_APPLICABLE, HandSignal

PAIR_ID = "1_2"
PHASES = ("development", "evaluation")
TOL = 1e-9

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


def _hand_key(hid: str) -> Tuple[int, float, str]:
    """Mirror the retriever's numeric-if-possible hand-id ordering key (Req 8.5 tie-break)."""
    try:
        return (0, float(hid), "")
    except (TypeError, ValueError):
        return (1, 0.0, hid)


def _draw_signal(rng: random.Random) -> HandSignal:
    hand_id = rng.randint(1, 12)
    phase = rng.choice(PHASES)
    # Occasionally clamp value_flow to a small shared set to provoke strength ties.
    if rng.random() < 0.4:
        value_flow = rng.choice([1.0, -1.0, 3.0, -3.0, 5.0])
    else:
        value_flow = rng.uniform(-500.0, 500.0)
    aggression_asymmetry = NOT_APPLICABLE if rng.random() < 0.35 else rng.choice([0.2, 0.5, 0.8])
    isolation = NOT_APPLICABLE if rng.random() < 0.35 else rng.choice([2.0, 5.0, 8.0])
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
def _assert_sorted_by_strength_then_id(candidates: List[HandSignal]) -> None:
    sel = select_evidence(PAIR_ID, candidates)
    # First-occurrence map (matches the retriever's dedup) to recompute expected strengths.
    by_id: Dict[str, HandSignal] = {}
    for sig in candidates:
        by_id.setdefault(str(sig.hand_id), sig)

    selected = [
        (slot, strength)
        for slot, strength in zip(sel.slots, sel.strengths)
        if slot != NO_EVIDENCE
    ]

    # Reported strength must equal the independently recomputed evidence_strength.
    for slot, reported in selected:
        expected = evidence_strength(by_id[slot])
        assert reported == pytest.approx(expected, abs=TOL), (
            f"reported strength {reported} != recomputed {expected} for hand {slot!r}"
        )

    # Consecutive ordering: strength descending, ties broken by ascending hand id.
    for (id_a, s_a), (id_b, s_b) in zip(selected, selected[1:]):
        assert s_a >= s_b - TOL, (
            f"strength not descending: {id_a}={s_a} then {id_b}={s_b}"
        )
        if abs(s_a - s_b) <= TOL:
            assert _hand_key(id_a) < _hand_key(id_b), (
                f"tie not broken by ascending hand id: {id_a} then {id_b} at strength {s_a}"
            )


# --------------------------------------------------------------------------- #
# Explicit corner cases                                                       #
# --------------------------------------------------------------------------- #
def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 12] generator in use: {GENERATOR_IN_USE}")


def test_descending_strength_example():
    """Distinct strengths -> strictly descending order by strength."""
    cands = [
        _make_signal(
            hid, phase="evaluation", value_flow=float(hid), aggression_asymmetry=NOT_APPLICABLE,
            isolation=NOT_APPLICABLE,
            flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
        )
        for hid in (2, 7, 4)
    ]
    sel = select_evidence(PAIR_ID, cands)
    assert sel.slots[:3] == ["7", "4", "2"]
    _assert_sorted_by_strength_then_id(cands)


def test_tie_break_ascending_hand_id_example():
    """Equal strengths -> ascending numeric hand id ("2" before "10")."""
    cands = [
        _make_signal(
            hid, phase="evaluation", value_flow=3.0, aggression_asymmetry=NOT_APPLICABLE,
            isolation=NOT_APPLICABLE,
            flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
        )
        for hid in (10, 2, 5)
    ]
    sel = select_evidence(PAIR_ID, cands)
    assert sel.slots[:3] == ["2", "5", "10"]
    _assert_sorted_by_strength_then_id(cands)


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
        st.sampled_from([0.2, 0.5, 0.8, 2.0, 5.0, 8.0]),
    )
    # value_flow strategy mixes a small shared set (ties) with a wide range (distinct).
    _value_flow_strategy = st.one_of(
        st.sampled_from([1.0, -1.0, 3.0, -3.0, 5.0]),
        st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    _signal_strategy = st.builds(
        _make_signal,
        hand_id=st.integers(min_value=1, max_value=12),
        phase=st.sampled_from(PHASES),
        value_flow=_value_flow_strategy,
        aggression_asymmetry=_ratio_strategy,
        isolation=_ratio_strategy,
        flags=_flag_strategy,
    )
    _candidates_strategy = st.lists(_signal_strategy, min_size=0, max_size=12)

    @settings(max_examples=200, deadline=None)
    @given(candidates=_candidates_strategy)
    def test_property_sorted_hypothesis(candidates):
        _assert_sorted_by_strength_then_id(list(candidates))

else:

    _N_SEEDS = 200

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_sorted_seeded_loop(seed: int):
        rng = random.Random(seed)
        candidates = _draw_candidates(rng)
        _assert_sorted_by_strength_then_id(candidates)
