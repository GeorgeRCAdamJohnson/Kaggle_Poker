"""Unit tests for poker_collusion.evidence.retriever (task 12.1, Requirement 8).

These are *hand-computable example* tests on synthetic :class:`HandSignal` lists verifying the
Evidence_Retriever's hard validity gate (Req 8.1-8.4), deterministic strength ranking with the
ascending-hand-id tie-break (Req 8.5), ``NO_EVIDENCE`` padding to exactly 5 slots (Req 8.6, 8.7),
top-5 truncation, hand-id de-duplication (Req 8.8), completeness, and determinism.

The Hypothesis property tests (Properties 9/10/11/12, tasks 12.2-12.5) are separate tasks and are
deliberately NOT duplicated here.
"""

from __future__ import annotations

from typing import Dict, Optional

from poker_collusion.config import NO_EVIDENCE
from poker_collusion.evidence.retriever import (
    MAX_EVIDENCE,
    EvidenceSelection,
    evidence_strength,
    is_valid_evidence,
    retrieve_evidence_for_pair,
    select_evidence,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal

PAIR_ID = "1_2"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _signal(
    hand_id: int,
    *,
    phase: str = "evaluation",
    value_flow: float = 0.0,
    aggression_asymmetry=NOT_APPLICABLE,
    isolation=NOT_APPLICABLE,
    mi_conflict: float = 0.0,
    flags: Optional[Dict[str, bool]] = None,
) -> HandSignal:
    """Build a synthetic HandSignal with explicit family flags."""
    return HandSignal(
        hand_id=hand_id,
        pair_id=PAIR_ID,
        phase=phase,
        value_flow=value_flow,
        aggression_asymmetry=aggression_asymmetry,
        isolation=isolation,
        mi_conflict=mi_conflict,
        behavior_action_flags=flags
        or {"directed_transfer": False, "soft_play": False, "coordinated_isolation": False},
    )


def _dt(hand_id: int, value_flow: float, phase: str = "evaluation") -> HandSignal:
    """A valid directed_transfer evidence hand (strength = |value_flow|)."""
    return _signal(
        hand_id,
        phase=phase,
        value_flow=value_flow,
        flags={"directed_transfer": True, "soft_play": False, "coordinated_isolation": False},
    )


# --------------------------------------------------------------------------- #
# Validity gate (Req 8.1-8.4)
# --------------------------------------------------------------------------- #
def test_only_valid_hands_selected():
    """Both-present + eval + >=1 behavior flag hands are selected; others excluded."""
    valid = _dt(10, value_flow=5.0)
    dev_hand = _dt(11, value_flow=99.0, phase="development")  # wrong phase
    no_flag = _signal(12, phase="evaluation", value_flow=99.0)  # no behavior flag
    sel = select_evidence(PAIR_ID, [valid, dev_hand, no_flag])
    assert sel.slots[0] == "10"
    assert "11" not in sel.slots  # dev-phase excluded (Req 8.3)
    assert "12" not in sel.slots  # no-flag excluded (Req 8.4)
    assert sel.slots[1:] == [NO_EVIDENCE] * 4


def test_development_phase_hands_excluded():
    """A development-period hand is never selected even with a fired flag (Req 8.3)."""
    dev = _dt(10, value_flow=100.0, phase="development")
    assert is_valid_evidence(dev) is False
    sel = select_evidence(PAIR_ID, [dev])
    assert sel.slots == [NO_EVIDENCE] * MAX_EVIDENCE


def test_no_behavior_flag_excluded():
    """A hand with no disclosed-family flag (latent-only) is excluded (Req 8.4)."""
    latent = _signal(10, phase="evaluation", value_flow=50.0, isolation=9.0)
    assert is_valid_evidence(latent) is False
    sel = select_evidence(PAIR_ID, [latent])
    assert sel.slots == [NO_EVIDENCE] * MAX_EVIDENCE


def test_not_shared_hand_excluded():
    """A non-shared hand is rejected when is_shared predicate returns False (Req 8.1/8.2)."""
    valid = _dt(10, value_flow=5.0)
    sel = select_evidence(PAIR_ID, [valid], is_shared=lambda s: False)
    assert sel.slots == [NO_EVIDENCE] * MAX_EVIDENCE


# --------------------------------------------------------------------------- #
# Ranking (Req 8.5)
# --------------------------------------------------------------------------- #
def test_ranking_by_strength_descending():
    """Slots ordered by evidence-strength descending."""
    weak = _dt(10, value_flow=1.0)
    strong = _dt(11, value_flow=8.0)
    mid = _dt(12, value_flow=4.0)
    sel = select_evidence(PAIR_ID, [weak, strong, mid])
    assert sel.slots[:3] == ["11", "12", "10"]
    assert sel.strengths[:3] == [8.0, 4.0, 1.0]


def test_tie_break_ascending_hand_id():
    """Equal strength -> ascending hand id ordering (Req 8.5)."""
    # All three have |value_flow| == 3.0 -> equal strength; ids given out of order.
    a = _dt(30, value_flow=3.0)
    b = _dt(10, value_flow=-3.0)  # magnitude 3.0
    c = _dt(20, value_flow=3.0)
    sel = select_evidence(PAIR_ID, [a, b, c])
    assert sel.slots[:3] == ["10", "20", "30"]


def test_strength_combines_families_via_max():
    """evidence_strength takes the max over flagged families' contributions."""
    # soft_play contribution = 1 - aggression_asymmetry = 1 - 0.2 = 0.8
    # coordinated_isolation contribution = isolation = 2.5 -> max is 2.5
    sig = _signal(
        10,
        aggression_asymmetry=0.2,
        isolation=2.5,
        flags={"directed_transfer": False, "soft_play": True, "coordinated_isolation": True},
    )
    assert evidence_strength(sig) == 2.5


def test_soft_play_strength_inverts_asymmetry():
    """soft_play strength = 1 - aggression_asymmetry (more avoidance -> stronger)."""
    strong_soft = _signal(
        10,
        aggression_asymmetry=0.1,
        flags={"directed_transfer": False, "soft_play": True, "coordinated_isolation": False},
    )
    weak_soft = _signal(
        11,
        aggression_asymmetry=0.9,
        flags={"directed_transfer": False, "soft_play": True, "coordinated_isolation": False},
    )
    assert abs(evidence_strength(strong_soft) - 0.9) < 1e-9
    assert abs(evidence_strength(weak_soft) - 0.1) < 1e-9
    sel = select_evidence(PAIR_ID, [weak_soft, strong_soft])
    assert sel.slots[:2] == ["10", "11"]


# --------------------------------------------------------------------------- #
# Slot filling / truncation / completeness (Req 8.6, 8.7)
# --------------------------------------------------------------------------- #
def test_fewer_than_five_padded_with_no_evidence():
    """Fewer than 5 valid hands -> padded to exactly 5 with NO_EVIDENCE."""
    sel = select_evidence(PAIR_ID, [_dt(10, 5.0), _dt(11, 4.0)])
    assert len(sel.slots) == MAX_EVIDENCE
    assert sel.slots[:2] == ["10", "11"]
    assert sel.slots[2:] == [NO_EVIDENCE] * 3


def test_more_than_five_keeps_top_five():
    """More than 5 valid hands -> only the top 5 by strength are kept."""
    cands = [_dt(hid, value_flow=float(hid)) for hid in range(10, 20)]  # 10 valid hands
    sel = select_evidence(PAIR_ID, cands)
    assert len(sel.slots) == MAX_EVIDENCE
    # strongest are the largest value_flow: 19,18,17,16,15
    assert sel.slots == ["19", "18", "17", "16", "15"]
    assert NO_EVIDENCE not in sel.slots


def test_all_five_slots_non_empty():
    """Every one of the 5 cells is non-empty (a hand id or NO_EVIDENCE) (Req 8.7)."""
    sel = select_evidence(PAIR_ID, [_dt(10, 5.0)])
    assert len(sel.slots) == MAX_EVIDENCE
    assert all(cell for cell in sel.slots)
    assert all(cell == NO_EVIDENCE or cell.isdigit() for cell in sel.slots)


def test_empty_candidates_all_no_evidence():
    """Zero candidates -> all five slots NO_EVIDENCE (Req 8.1 lower bound 0)."""
    sel = select_evidence(PAIR_ID, [])
    assert sel.slots == [NO_EVIDENCE] * MAX_EVIDENCE
    assert sel.strengths == [None] * MAX_EVIDENCE


# --------------------------------------------------------------------------- #
# De-duplication (Req 8.8)
# --------------------------------------------------------------------------- #
def test_no_duplicate_hand_id():
    """A repeated hand id appears at most once within the pair's slots (Req 8.8)."""
    dup_a = _dt(10, value_flow=5.0)
    dup_b = _dt(10, value_flow=1.0)  # same hand id, different strength
    other = _dt(11, value_flow=3.0)
    sel = select_evidence(PAIR_ID, [dup_a, dup_b, other])
    hand_ids = [c for c in sel.slots if c != NO_EVIDENCE]
    assert hand_ids.count("10") == 1
    assert sorted(hand_ids) == ["10", "11"]
    # First occurrence (strength 5.0) is kept, so hand 10 ranks above hand 11.
    assert sel.slots[:2] == ["10", "11"]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_input_same_output():
    """Selecting twice on the same signals yields identical slots and strengths."""
    cands = [_dt(12, 3.0), _dt(10, 3.0), _dt(11, 7.0), _dt(13, 1.0)]
    first = select_evidence(PAIR_ID, cands)
    second = select_evidence(PAIR_ID, cands)
    assert first == second


def test_determinism_independent_of_input_order():
    """Ranking is independent of candidate input order (Req 8.5 repeatability)."""
    cands = [_dt(12, 3.0), _dt(10, 3.0), _dt(11, 7.0), _dt(13, 1.0)]
    forward = select_evidence(PAIR_ID, cands)
    backward = select_evidence(PAIR_ID, list(reversed(cands)))
    assert forward.slots == backward.slots


# --------------------------------------------------------------------------- #
# Thin helper
# --------------------------------------------------------------------------- #
def test_retrieve_helper_infers_pair_id():
    """The I/O-free helper infers pair_id from the first signal and forwards selection."""
    sel = retrieve_evidence_for_pair([_dt(10, 5.0)])
    assert isinstance(sel, EvidenceSelection)
    assert sel.pair_id == PAIR_ID
    assert sel.slots[0] == "10"
