"""Correctness unit tests for poker_collusion.features.signals (task 9.1, Requirement 3).

These are *hand-computable example* tests on tiny constructed hands that verify each signal's
formula, the all-in classification rule (Req 3.4), the NOT_APPLICABLE zero-denominator handling
(Req 3.5), the value-flow sign convention / antisymmetry (Req 3.1), and the per-family
behavior-action flags (Req 3.8). The exhaustive Hypothesis property tests for antisymmetry,
all-in classification, and the sentinel are separate tasks (9.2 / 9.3 / 9.4) and are not
duplicated here.

All fixtures use only the decision-time-context columns the requirement permits.
"""

from __future__ import annotations

import pandas as pd

from poker_collusion.features.signals import (
    compute_hand_signal,
    compute_pair_signals,
    compute_value_flow,
    hand_signals_to_frame,
    is_aggressive_action,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _actions(rows: list[dict]) -> pd.DataFrame:
    """Build an ordered action-log frame; auto-assigns action_no if absent."""
    for i, r in enumerate(rows):
        r.setdefault("action_no", i)
        r.setdefault("amount", 0.0)
        r.setdefault("to_call", 0.0)
        r.setdefault("street", "preflop")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# all_in classification (Req 3.4)
# --------------------------------------------------------------------------- #
def test_all_in_is_call_when_amount_le_to_call():
    assert is_aggressive_action("all_in", amount=50, to_call=50) is False
    assert is_aggressive_action("all_in", amount=30, to_call=50) is False


def test_all_in_is_aggressive_when_amount_gt_to_call():
    assert is_aggressive_action("all_in", amount=200, to_call=50) is True


def test_bet_and_raise_always_aggressive_call_check_fold_never():
    assert is_aggressive_action("bet", 10, 0) is True
    assert is_aggressive_action("raise", 40, 20) is True
    assert is_aggressive_action("call", 20, 20) is False
    assert is_aggressive_action("check", 0, 0) is False
    assert is_aggressive_action("fold", 0, 20) is False


def test_all_in_missing_amounts_treated_as_call():
    assert is_aggressive_action("all_in", amount=None, to_call=50) is False


# --------------------------------------------------------------------------- #
# value_flow (Req 3.1) — sign convention and antisymmetry on a hand-computable example
# --------------------------------------------------------------------------- #
def test_value_flow_sign_convention_a_loses_to_b():
    # A loses 100 chips, B gains 100 chips; bb = 10 => 10 bb flow A->B (positive).
    vf = compute_value_flow(net_a=-100, net_b=100, big_blind=10)
    assert vf == 10.0


def test_value_flow_antisymmetric_on_example():
    vf_ab = compute_value_flow(net_a=-100, net_b=100, big_blind=10)
    vf_ba = compute_value_flow(net_a=100, net_b=-100, big_blind=10)
    assert vf_ab == -vf_ba == 10.0


def test_value_flow_uses_min_of_loss_and_gain():
    # A lost 100 but B only gained 30 (a third player took the rest): flow A->B capped at 30.
    vf = compute_value_flow(net_a=-100, net_b=30, big_blind=1)
    assert vf == 30.0


def test_value_flow_zero_big_blind_falls_back_to_raw_chips():
    vf = compute_value_flow(net_a=-40, net_b=40, big_blind=0)
    assert vf == 40.0  # divided by fallback 1.0, not a crash


# --------------------------------------------------------------------------- #
# aggression_asymmetry (Req 3.2)
# --------------------------------------------------------------------------- #
def test_aggression_asymmetry_not_applicable_when_no_aggression():
    # Neither A nor B ever bets/raises -> denominator zero -> NOT_APPLICABLE (Req 3.5).
    actions = _actions(
        [
            {"player_id": "A", "action": "check"},
            {"player_id": "B", "action": "check"},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    assert sig.aggression_asymmetry is NOT_APPLICABLE


def test_aggression_asymmetry_soft_play_low_ratio():
    # A bets while B still active, then C is the only active opponent when A bets again:
    # A folds partner-directed aggression by pushing the field once B has folded.
    actions = _actions(
        [
            {"player_id": "B", "action": "fold"},   # partner folds first
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},  # only C active
            {"player_id": "C", "action": "call", "amount": 20, "to_call": 20},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    # A made 1 aggressive action, 0 while partner active -> ratio 0.0 (max avoidance).
    assert sig.aggression_asymmetry == 0.0


def test_aggression_asymmetry_full_when_partner_always_active():
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},  # B still active
            {"player_id": "B", "action": "call", "amount": 20, "to_call": 20},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    # A's single aggressive action is partner-directed -> ratio 1.0.
    assert sig.aggression_asymmetry == 1.0


# --------------------------------------------------------------------------- #
# isolation (Req 3.3)
# --------------------------------------------------------------------------- #
def test_isolation_not_applicable_when_no_mutual_opportunity():
    # Only A acts aggressively and B folds immediately -> no "at each other" -> NOT_APPLICABLE.
    actions = _actions(
        [
            {"player_id": "B", "action": "fold"},
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},
            {"player_id": "C", "action": "call", "amount": 20, "to_call": 20},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    assert sig.isolation is NOT_APPLICABLE


def test_isolation_ratio_pair_pressures_third_player():
    # Both A and B bet/raise while C (third) and each other are active.
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},   # B & C active
            {"player_id": "B", "action": "raise", "amount": 60, "to_call": 20},  # A & C active
            {"player_id": "C", "action": "call", "amount": 60, "to_call": 60},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    # at_each_other = 2 (both aggress while partner active), at_others = 2 (C active both times).
    assert sig.isolation == 1.0


# --------------------------------------------------------------------------- #
# mi_conflict (documented proxy)
# --------------------------------------------------------------------------- #
def test_mi_conflict_high_when_pair_avoids_each_other():
    # No pair member ever aggresses while the other is active -> no conflict opportunity -> 0.0.
    actions = _actions(
        [
            {"player_id": "A", "action": "check"},
            {"player_id": "B", "action": "check"},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    assert sig.mi_conflict == 0.0


def test_mi_conflict_zero_when_pair_always_contests():
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},   # B active -> contest
            {"player_id": "B", "action": "raise", "amount": 60, "to_call": 20},  # A active -> contest
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    # 2 opportunities, 2 conflicts -> 1 - 2/2 = 0.0.
    assert sig.mi_conflict == 0.0


# --------------------------------------------------------------------------- #
# behavior_action_flags (Req 3.8)
# --------------------------------------------------------------------------- #
def test_directed_transfer_flag_requires_public_action_and_nonzero_flow():
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 100, "to_call": 0},  # public commit
            {"player_id": "B", "action": "call", "amount": 100, "to_call": 100},
        ]
    )
    sig = compute_hand_signal(
        "H1", "P", "evaluation", "A", "B", actions, net_a=-100, net_b=100, big_blind=10
    )
    assert sig.behavior_action_flags["directed_transfer"] is True


def test_directed_transfer_flag_false_without_flow():
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 100, "to_call": 0},
            {"player_id": "B", "action": "call", "amount": 100, "to_call": 100},
        ]
    )
    # net flow zero -> not a directed transfer even though a public action exists.
    sig = compute_hand_signal(
        "H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0, big_blind=10
    )
    assert sig.behavior_action_flags["directed_transfer"] is False


def test_coordinated_isolation_flag_when_both_pressure_third():
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},
            {"player_id": "B", "action": "raise", "amount": 60, "to_call": 20},
            {"player_id": "C", "action": "call", "amount": 60, "to_call": 60},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    assert sig.behavior_action_flags["coordinated_isolation"] is True


def test_soft_play_flag_when_partner_directed_aggression_declined():
    actions = _actions(
        [
            {"player_id": "B", "action": "fold"},
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},
            {"player_id": "C", "action": "call", "amount": 20, "to_call": 20},
        ]
    )
    sig = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=0, net_b=0)
    # aggression_asymmetry == 0.0 (< 1.0) -> soft_play signature present.
    assert sig.behavior_action_flags["soft_play"] is True


# --------------------------------------------------------------------------- #
# Determinism + frame flattening + pair driver
# --------------------------------------------------------------------------- #
def test_compute_hand_signal_is_deterministic():
    actions = _actions(
        [
            {"player_id": "A", "action": "bet", "amount": 20, "to_call": 0},
            {"player_id": "B", "action": "call", "amount": 20, "to_call": 20},
        ]
    )
    s1 = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=-20, net_b=20)
    s2 = compute_hand_signal("H1", "P", "evaluation", "A", "B", actions, net_a=-20, net_b=20)
    assert s1 == s2


def test_compute_pair_signals_orders_by_hand_id_and_flattens():
    hands_meta = pd.DataFrame(
        [
            {"hand_id": "H2", "phase": "evaluation", "big_blind": 10},
            {"hand_id": "H1", "phase": "development", "big_blind": 10},
        ]
    )
    actions_by_hand = {
        "H1": _actions([{"player_id": "A", "action": "bet", "amount": 100, "to_call": 0}]),
        "H2": _actions([{"player_id": "B", "action": "bet", "amount": 100, "to_call": 0}]),
    }
    net_by_hand = {
        "H1": {"A": -100, "B": 100},
        "H2": {"A": 100, "B": -100},
    }
    signals = compute_pair_signals("P", "A", "B", hands_meta, actions_by_hand, net_by_hand)
    assert [s.hand_id for s in signals] == ["H1", "H2"]  # ascending hand_id
    # H1: A loses to B -> +10 bb ; H2: B loses to A -> -10 bb (sign convention).
    assert signals[0].value_flow == 10.0
    assert signals[1].value_flow == -10.0

    frame = hand_signals_to_frame(signals)
    assert list(frame["hand_id"]) == ["H1", "H2"]
    assert {"flag_directed_transfer", "flag_soft_play", "flag_coordinated_isolation"} <= set(
        frame.columns
    )
    # NOT_APPLICABLE preserved as an object, never coerced to NaN.
    empty = compute_hand_signal("H3", "P", "evaluation", "A", "B", _actions([]), net_a=0, net_b=0)
    assert isinstance(empty, HandSignal)
    assert empty.aggression_asymmetry is NOT_APPLICABLE
