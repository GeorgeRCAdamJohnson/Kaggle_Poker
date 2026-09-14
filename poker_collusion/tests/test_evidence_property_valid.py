# Feature: poker-collusion-detection, Property 9: Every selected evidence hand is valid
"""Property 9 (task 12.2): every SELECTED evidence hand passes the hard validity gate.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

For any candidate list handed to
:func:`poker_collusion.evidence.retriever.select_evidence`, every non-``NO_EVIDENCE``
slot in the result MUST correspond to a candidate hand that passes ALL clauses of the
hard validity gate:

1. both players present (Req 8.1, 8.2) — the hand is a shared hand of the pair, i.e. the
   ``is_shared`` predicate is ``True`` for it,
2. evaluation period (Req 8.3) — ``phase == "evaluation"``; a development-period hand is
   NEVER selected,
3. at least one behavior-specific public action (Req 8.4) — at least one disclosed-family
   flag in ``behavior_action_flags`` is ``True``; a latent-only hand (all flags ``False``)
   is NEVER selected.

Concretely the test asserts, across the whole generated space, that no selected slot is a
development-phase hand, no selected slot is a latent-only (all-flags-``False``) hand, and no
selected slot is a hand for which the ``is_shared`` predicate returned ``False``.

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable it transparently
falls back to a seeded randomized loop (200 seeds) that draws the same input space — random
hand ids with duplicates, random phase in {development, evaluation}, random value_flow, ratio
signals sometimes ``NOT_APPLICABLE``, behavior flags sometimes all-``False`` (latent-only), and
a variable candidate count (0..12) — plus a random ``is_shared`` predicate. The active backend is
recorded in ``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random
from typing import Callable, Dict, List, Optional

import pytest

from poker_collusion.config import DISCLOSED_FAMILY_NAMES, NO_EVIDENCE
from poker_collusion.evidence.retriever import (
    EVALUATION_PHASE,
    is_valid_evidence,
    select_evidence,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal

PAIR_ID = "1_2"
PHASES = ("development", "evaluation")

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"


# --------------------------------------------------------------------------- #
# Synthetic HandSignal builder (mirrors test_evidence_retriever conventions)  #
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
    """Draw one random candidate HandSignal spanning the whole input space."""
    # Random hand id from a small pool so duplicates occur frequently.
    hand_id = rng.randint(1, 8)
    phase = rng.choice(PHASES)
    value_flow = rng.uniform(-500.0, 500.0)
    aggression_asymmetry = NOT_APPLICABLE if rng.random() < 0.35 else rng.uniform(0.0, 1.0)
    isolation = NOT_APPLICABLE if rng.random() < 0.35 else rng.uniform(0.0, 10.0)
    # ~30% of hands are latent-only (all flags False).
    if rng.random() < 0.30:
        flags = {fam: False for fam in DISCLOSED_FAMILY_NAMES}
    else:
        flags = {fam: (rng.random() < 0.5) for fam in DISCLOSED_FAMILY_NAMES}
        # Guarantee at least one True in the non-latent branch.
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


def _draw_is_shared(rng: random.Random) -> Optional[Callable[[HandSignal], bool]]:
    """Sometimes return a predicate that marks a random subset of hands non-shared."""
    choice = rng.random()
    if choice < 0.4:
        return None  # trust upstream: all shared
    if choice < 0.7:
        return lambda s: False  # nothing shared
    # Per-hand-id random shared set.
    shared_ids = {hid for hid in range(1, 9) if rng.random() < 0.5}
    return lambda s: int(s.hand_id) in shared_ids


# --------------------------------------------------------------------------- #
# Core property assertion                                                     #
# --------------------------------------------------------------------------- #
def _assert_all_selected_valid(
    candidates: List[HandSignal],
    is_shared: Optional[Callable[[HandSignal], bool]],
) -> None:
    sel = select_evidence(PAIR_ID, candidates, is_shared=is_shared)
    by_id: Dict[str, HandSignal] = {}
    for sig in candidates:
        by_id.setdefault(str(sig.hand_id), sig)  # first occurrence, matches dedup

    def _shared_of(sig: HandSignal) -> bool:
        return True if is_shared is None else bool(is_shared(sig))

    for slot in sel.slots:
        if slot == NO_EVIDENCE:
            continue
        assert slot in by_id, f"selected slot {slot!r} is not a candidate hand id"
        sig = by_id[slot]
        # Must pass the hard validity gate as a whole.
        assert is_valid_evidence(sig, is_shared=_shared_of(sig)), (
            f"selected hand {slot!r} fails the validity gate"
        )
        # And each individual clause:
        assert sig.phase == EVALUATION_PHASE, f"development-phase hand {slot!r} was selected"
        assert any(sig.behavior_action_flags.values()), (
            f"latent-only hand {slot!r} (no flag) was selected"
        )
        assert _shared_of(sig), f"non-shared hand {slot!r} was selected"


# --------------------------------------------------------------------------- #
# Explicit corner cases                                                       #
# --------------------------------------------------------------------------- #
def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 9] generator in use: {GENERATOR_IN_USE}")


def test_development_and_latent_never_selected_example():
    """Hand-picked: dev-phase and latent-only hands are excluded; valid ones selected."""
    valid = _make_signal(
        3, phase="evaluation", value_flow=5.0, aggression_asymmetry=NOT_APPLICABLE,
        isolation=NOT_APPLICABLE,
        flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
    )
    dev = _make_signal(
        4, phase="development", value_flow=99.0, aggression_asymmetry=NOT_APPLICABLE,
        isolation=NOT_APPLICABLE,
        flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
    )
    latent = _make_signal(
        5, phase="evaluation", value_flow=99.0, aggression_asymmetry=0.1, isolation=9.0,
        flags={"directed_transfer": False, "soft_play": False, "coordinated_isolation": False},
    )
    _assert_all_selected_valid([valid, dev, latent], None)


def test_non_shared_never_selected_example():
    """A hand marked non-shared by the predicate is never selected (Req 8.1/8.2)."""
    valid = _make_signal(
        3, phase="evaluation", value_flow=5.0, aggression_asymmetry=NOT_APPLICABLE,
        isolation=NOT_APPLICABLE,
        flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
    )
    _assert_all_selected_valid([valid], lambda s: False)


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
        hand_id=st.integers(min_value=1, max_value=8),
        phase=st.sampled_from(PHASES),
        value_flow=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        aggression_asymmetry=_ratio_strategy,
        isolation=_ratio_strategy,
        flags=_flag_strategy,
    )
    _candidates_strategy = st.lists(_signal_strategy, min_size=0, max_size=12)
    _is_shared_strategy = st.one_of(
        st.none(),
        st.just(lambda s: False),
        st.builds(
            lambda ids: (lambda s: int(s.hand_id) in ids),
            st.sets(st.integers(min_value=1, max_value=8)),
        ),
    )

    @settings(max_examples=200, deadline=None)
    @given(candidates=_candidates_strategy, is_shared=_is_shared_strategy)
    def test_property_all_selected_valid_hypothesis(candidates, is_shared):
        _assert_all_selected_valid(list(candidates), is_shared)

else:

    _N_SEEDS = 200

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_all_selected_valid_seeded_loop(seed: int):
        rng = random.Random(seed)
        candidates = _draw_candidates(rng)
        is_shared = _draw_is_shared(rng)
        _assert_all_selected_valid(candidates, is_shared)
