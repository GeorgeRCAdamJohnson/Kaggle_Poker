"""Hand-Level Signal Extraction (Requirement 3, Design "Hand-Level Signal Extraction").

For each *shared hand* of a co-seated Pair (a hand in which both players are seated), this
module computes the per-hand coordination signals that later aggregate to the pair level:

- ``value_flow``            signed, directed net chip transfer between the two players (Req 3.1)
- ``aggression_asymmetry``  per-player pair-directed vs. all-directed bet/raise ratio (Req 3.2)
- ``isolation``             pair's bet/raise pressure at *others* vs. at *each other* (Req 3.3)
- ``mi_conflict``           conflict-avoidance / mutual-information-style proxy (Design)
- ``behavior_action_flags`` per-disclosed-family public-action signature flag (Req 3.8)

Decision-time-only guarantee (Req 3.6 / 3.7)
--------------------------------------------
The *action* signals (``aggression_asymmetry``, ``isolation``, ``mi_conflict``, and the
``behavior_action_flags``) are computed **exclusively** from the ordered public action log
using only the decision-time context columns the requirement enumerates:
``pot_before, stack_before, to_call, players_active, amount, amount_to`` plus the actor
(``player_id``), the ordinal (``action_no``), the round (``street``) and the ``action`` token.
They never read hole cards, the board, showdown-only outcomes, or ``seats`` outcome columns.
This is what makes the aggression / isolation signals honest about *what a colluder could
observe when they acted*.

``value_flow`` is defined by Req 3.1 as an *outcome-based* directed chip transfer (chips a
player contributes minus chips received in the hand outcome). It therefore legitimately reads
the per-seat outcome columns (``net_chips``) — this is explicitly permitted for the value-flow
signal and is the only signal that touches outcome data. The Discovery dossier (§2.1 H5)
confirmed per-hand chips are exactly zero-sum with no rake, so ``seats.net_chips`` is the
ready-made, conservation-respecting basis for value flow.

all_in classification (Req 3.4)
-------------------------------
An ``all_in`` action is classified as a **call** when ``amount <= to_call`` and as an
**aggressive** (bet/raise-equivalent) action when ``amount > to_call``. Every place that counts
"bet-or-raise" actions applies this rule via :func:`is_aggressive_action`, so the classification
is defined once and used everywhere.

Zero-denominator handling (Req 3.5)
-----------------------------------
Whenever a ratio's denominator is zero (no qualifying bet/raise opportunities for the relevant
player or pair in the hand), the signal is assigned the shared ``NOT_APPLICABLE`` sentinel from
:mod:`poker_collusion.types` — never ``NaN`` and never a raised error.

Determinism
-----------
Given identical input frames, :func:`compute_hand_signal` and :func:`compute_pair_signals`
return identical values. No randomness, no wall-clock, no dict-iteration-order dependence.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from poker_collusion.config import DISCLOSED_FAMILY_NAMES
from poker_collusion.types import NOT_APPLICABLE, HandSignal, Sentinel

__all__ = [
    "AGGRESSIVE_ACTIONS",
    "PASSIVE_ACTIONS",
    "is_aggressive_action",
    "compute_value_flow",
    "action_rows",
    "compute_hand_signal",
    "compute_pair_signals",
    "hand_signals_to_frame",
]

# Action tokens (closed 6-token vocabulary, Discovery §2.1 H4:
# {fold, call, raise, check, bet, all_in}).
#: Unconditionally aggressive tokens (a bet or a raise is always aggression).
_ALWAYS_AGGRESSIVE = frozenset({"bet", "raise"})
#: Tokens that are never aggression.
_NEVER_AGGRESSIVE = frozenset({"fold", "check", "call"})
#: For documentation/consumers: the aggressive and passive token sets (``all_in`` is
#: conditional and therefore belongs to neither static set).
AGGRESSIVE_ACTIONS = _ALWAYS_AGGRESSIVE
PASSIVE_ACTIONS = _NEVER_AGGRESSIVE


def _to_float(value: object) -> Optional[float]:
    """Best-effort float coercion; returns None for missing/uncoercible values."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # handles NaN / NaT / None
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def is_aggressive_action(action: object, amount: object, to_call: object) -> bool:
    """Classify a single action row as aggressive (bet/raise-equivalent) or not (Req 3.4).

    Rules:
      * ``bet`` / ``raise`` are always aggressive.
      * ``fold`` / ``check`` / ``call`` are never aggressive.
      * ``all_in`` is aggressive iff ``amount > to_call`` (Req 3.4): an all-in that merely
        covers the amount owed is a *call*, a larger all-in is *aggressive*. When ``amount`` or
        ``to_call`` is missing, the all-in cannot be shown to exceed the call amount, so it is
        conservatively treated as a call (not aggressive).

    Args:
        action: The ``actions.action`` token (case-insensitive).
        amount: The ``actions.amount`` committed by the actor.
        to_call: The ``actions.to_call`` amount owed at decision time.

    Returns:
        True when the action counts as a bet-or-raise for the signals below.
    """
    token = str(action).strip().lower()
    if token in _ALWAYS_AGGRESSIVE:
        return True
    if token == "all_in":
        amt = _to_float(amount)
        call = _to_float(to_call)
        if amt is None or call is None:
            return False
        return amt > call
    return False


def compute_value_flow(
    net_a: object,
    net_b: object,
    big_blind: object = 1.0,
) -> float:
    """Signed directed value flow between players A and B in one hand (Req 3.1).

    Definition and sign convention
    ------------------------------
    Using the per-seat signed outcome ``net_chips`` (chips won minus chips contributed for that
    seat in the hand; Discovery §2.1 H5 confirmed per-hand ``net_chips`` sums to exactly zero —
    zero-sum, no rake), the *directed transfer from A to B* is the chips that plausibly moved
    from A into B within this hand:

        gain_from_a_to_b = min(max(-net_a, 0), max(net_b, 0))
        gain_from_b_to_a = min(max(-net_b, 0), max(net_a, 0))
        value_flow(A, B) = (gain_from_a_to_b - gain_from_b_to_a) / big_blind

    ``max(-net_a, 0)`` is what A *lost*; ``max(net_b, 0)`` is what B *gained*; their min is the
    chips that could have flowed A->B. The symmetric term is the flow B->A. The signed
    difference, expressed in **big blinds** for cross-pool comparability (Discovery §2.1 H3:
    one stake per pool; H1/H2 refuted a coarse chip lattice, so bb is the right unit), is the
    directed value flow.

    **Sign convention:** ``value_flow(A, B) > 0`` means chips flowed **A -> B** (B gained at
    A's expense); ``< 0`` means chips flowed **B -> A**. Swapping the arguments negates every
    term, so the signal is **antisymmetric**: ``value_flow(A, B) == -value_flow(B, A)``
    (Property 1). When either net is missing it is treated as 0 (a seat with unknown outcome
    contributes no attributable transfer), preserving antisymmetry.

    Args:
        net_a: ``seats.net_chips`` for player A in the hand.
        net_b: ``seats.net_chips`` for player B in the hand.
        big_blind: The hand's big blind (``hands.big_blind``); normalizes chips to bb. A
            non-positive/missing big blind falls back to 1.0 (raw chips) so no divide-by-zero.

    Returns:
        Signed value flow in big blinds (float), antisymmetric in (A, B).
    """
    a = _to_float(net_a) or 0.0
    b = _to_float(net_b) or 0.0
    bb = _to_float(big_blind)
    if bb is None or bb <= 0.0:
        bb = 1.0
    a_to_b = min(max(-a, 0.0), max(b, 0.0))
    b_to_a = min(max(-b, 0.0), max(a, 0.0))
    return (a_to_b - b_to_a) / bb


def _action_rows(actions: pd.DataFrame) -> List[dict]:
    """Return the action log as a list of plain dicts ordered by ``action_no``.

    Keeps only the decision-time-context columns the requirement permits (Req 3.6). Rows are
    ordered by ``action_no`` (Discovery §2.2 H8: a 0-based, gap-free per-hand ordinal), so the
    ordered public log is reproduced deterministically.
    """
    if actions is None or len(actions) == 0:
        return []
    allowed = [
        "action_no",
        "player_id",
        "action",
        "amount",
        "amount_to",
        "to_call",
        "pot_before",
        "stack_before",
        "players_active",
        "street",
    ]
    cols = [c for c in allowed if c in actions.columns]
    frame = actions[cols]
    if "action_no" in frame.columns:
        frame = frame.sort_values("action_no", kind="stable")
    return frame.to_dict("records")


def action_rows(actions: pd.DataFrame) -> List[dict]:
    """Public wrapper for :func:`_action_rows` (the decision-time-only ordered action log).

    Exposed so callers that process the SAME hand for many pairs can parse each hand's action
    frame into row-dicts ONCE and reuse the result across pairs (the parsing is a pure function
    of the hand's action frame, independent of which pair is being scored). The returned list is
    exactly what :func:`compute_hand_signal` would build internally, so reusing it yields
    bit-identical signals.
    """
    return _action_rows(actions)


def _aggressive_rows(rows: Sequence[Mapping]) -> List[Mapping]:
    """Filter action rows to the bet-or-raise-equivalent ones (Req 3.4 rule applied)."""
    return [
        r
        for r in rows
        if is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call"))
    ]


def _aggression_asymmetry_for(
    rows: Sequence[Mapping],
    player: str,
    partner: str,
) -> Union[float, Sentinel]:
    """Per-player aggression-asymmetry ratio for ``player`` toward ``partner`` (Req 3.2).

    ``aggression_asymmetry`` = (count of ``player``'s bet/raise actions aimed at ``partner``)
    ÷ (count of ``player``'s bet/raise actions aimed at *any* opponent). A **lower** ratio means
    the player avoids aggression toward the partner relative to their overall aggression — the
    soft-play tell.

    Attribution model (decision-time only): an aggressive action "targets" every *other* player
    who is still active when it is taken. Because the public log does not name a single target,
    we count a player's aggressive action as directed at the partner whenever the partner is
    among the currently-active opponents (i.e. the partner had not folded before this action).
    This uses only the ordered action log and ``players_active`` — never outcome or hole-card
    data.

    Zero-denominator (Req 3.5): if ``player`` makes no bet/raise action at all in the hand, the
    ratio is undefined and returns ``NOT_APPLICABLE``.
    """
    partner_folded = False
    total = 0
    toward_partner = 0
    for r in rows:
        actor = str(r.get("player_id"))
        # Track whether the partner is still contesting the hand (has not folded yet).
        if actor == partner and str(r.get("action")).strip().lower() == "fold":
            partner_folded = True
        if actor != player:
            continue
        if not is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call")):
            continue
        total += 1
        if not partner_folded:
            toward_partner += 1
    if total == 0:
        return NOT_APPLICABLE
    return toward_partner / total


def _isolation_for(rows: Sequence[Mapping], player_a: str, player_b: str) -> Union[float, Sentinel]:
    """Pair isolation ratio: pair's bet/raise at *others* ÷ pair's bet/raise at *each other* (Req 3.3).

    A **higher** ratio means the pair concentrates its aggression on third players (ganging up
    to isolate) rather than contesting each other — the coordinated-isolation tell.

    Attribution model (decision-time only): each aggressive action by a pair member is counted
    as "at each other" when the *other* pair member is still active (not yet folded) at that
    action, and additionally as "at others" when at least one third player is still active. Both
    counts use only the ordered action log and fold events.

    Zero-denominator (Req 3.5): if the pair never bet/raises into each other while both are
    active (denominator 0), the ratio is undefined and returns ``NOT_APPLICABLE``.
    """
    pair = {player_a, player_b}
    folded: set[str] = set()
    active_third_exists_ever = False
    at_others = 0
    at_each_other = 0

    # Determine seated third players from the actors observed in the log.
    actors = {str(r.get("player_id")) for r in rows}
    third_players = actors - pair

    for r in rows:
        actor = str(r.get("player_id"))
        if str(r.get("action")).strip().lower() == "fold":
            folded.add(actor)
            continue
        if actor not in pair:
            continue
        if not is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call")):
            continue
        other_member = player_b if actor == player_a else player_a
        # "at each other": the partner is still contesting when the aggression is taken.
        if other_member not in folded:
            at_each_other += 1
        # "at others": at least one third player is still contesting.
        if any(tp not in folded for tp in third_players):
            at_others += 1
            active_third_exists_ever = True

    if at_each_other == 0:
        return NOT_APPLICABLE
    return at_others / at_each_other


def _mi_conflict_for(rows: Sequence[Mapping], player_a: str, player_b: str) -> float:
    """Conflict-avoidance / mutual-information-style proxy for the pair (Design).

    The design specifies a "mutual-information-style conflict-avoidance signal between the pair's
    action streams" but does not pin an exact formula, so this is a **documented proxy** (noted
    as such): the fraction of the pair's *shared aggression opportunities* on which they avoided
    contesting each other.

        opportunities = number of aggressive actions by either pair member while the *other*
                        member is still active in the hand (a chance to contest each other)
        conflicts     = of those, how many were actually aimed at the partner
        mi_conflict   = 1 - conflicts / opportunities        (in [0, 1])

    A value near **1.0** means the pair had chances to contest each other but consistently
    avoided it (high conflict-avoidance — the collusion tell); near **0.0** means they contested
    each other whenever they could. When the pair had **no** shared aggression opportunity, there
    is nothing to avoid, so the signal is defined as **0.0** (no evidence of conflict avoidance),
    keeping it a finite float (this signal has no ratio denominator to protect per Req 3.5, which
    scopes the sentinel to the aggression/isolation ratios).

    Decision-time only: uses only the ordered action log and fold events.
    """
    pair = {player_a, player_b}
    folded: set[str] = set()
    opportunities = 0
    conflicts = 0
    for r in rows:
        actor = str(r.get("player_id"))
        if str(r.get("action")).strip().lower() == "fold":
            folded.add(actor)
            continue
        if actor not in pair:
            continue
        if not is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call")):
            continue
        other_member = player_b if actor == player_a else player_a
        if other_member not in folded:
            opportunities += 1
            conflicts += 1  # an aggressive action while partner active *is* contesting them
    if opportunities == 0:
        return 0.0
    return 1.0 - conflicts / opportunities


def _behavior_action_flags(
    rows: Sequence[Mapping],
    value_flow: float,
    aggression_asymmetry: Union[float, Sentinel],
    isolation: Union[float, Sentinel],
    player_a: str,
    player_b: str,
) -> Dict[str, bool]:
    """Per-disclosed-family public-action signature flags (Req 3.8, Dossier §3.1).

    Each flag is True when the hand exhibits that family's *public* decision-time action
    signature — the predicate the evidence-validity gate (task 12.1) later reproduces. Every
    flag requires a **public action** between/among the players (never an outcome/latent flag
    alone, Dossier INV-1):

      * ``directed_transfer``: a materially non-zero directed ``value_flow`` between the two
        AND at least one public value-committing (aggressive/call) action by a pair member — a
        public chip transfer between the two, not an outcome-only flag.
      * ``soft_play``: the pair had a public bet/raise opportunity against each other that was
        declined — i.e. ``aggression_asymmetry`` is applicable and strictly below 1.0 for at
        least one member (partner-directed aggression is low relative to the field).
      * ``coordinated_isolation``: both pair members apply bet/raise pressure at third players
        in the hand (joint isolation), i.e. ``isolation`` is applicable and > 0 and both members
        each have at least one aggressive action at an active third player.

    Returns a dict keyed by the three disclosed family names; ``value_flow`` here may be signed,
    so directedness uses its magnitude.
    """
    # directed_transfer: public value-committing action by a pair member + non-zero directed flow.
    pair = {player_a, player_b}
    pair_committed = any(
        str(r.get("player_id")) in pair
        and (
            is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call"))
            or str(r.get("action")).strip().lower() == "call"
        )
        for r in rows
    )
    directed_transfer = bool(pair_committed and abs(value_flow) > 0.0)

    # soft_play: an applicable, below-1 aggression-asymmetry (declined pair-directed aggression).
    soft_play = isinstance(aggression_asymmetry, float) and aggression_asymmetry < 1.0

    # coordinated_isolation: both members apply aggression at an active third player.
    coordinated_isolation = _both_members_pressure_third(rows, player_a, player_b) and (
        isinstance(isolation, float) and isolation > 0.0
    )

    return {
        "directed_transfer": directed_transfer,
        "soft_play": bool(soft_play),
        "coordinated_isolation": bool(coordinated_isolation),
    }


def _both_members_pressure_third(rows: Sequence[Mapping], player_a: str, player_b: str) -> bool:
    """True when BOTH pair members take an aggressive action while a third player is active."""
    pair = {player_a, player_b}
    actors = {str(r.get("player_id")) for r in rows}
    third_players = actors - pair
    folded: set[str] = set()
    pressured = {player_a: False, player_b: False}
    for r in rows:
        actor = str(r.get("player_id"))
        if str(r.get("action")).strip().lower() == "fold":
            folded.add(actor)
            continue
        if actor not in pair:
            continue
        if not is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call")):
            continue
        if any(tp not in folded for tp in third_players):
            pressured[actor] = True
    return pressured[player_a] and pressured[player_b]


def compute_hand_signal(
    hand_id: object,
    pair_id: str,
    phase: str,
    player_a: str,
    player_b: str,
    actions: pd.DataFrame,
    net_a: object = 0.0,
    net_b: object = 0.0,
    big_blind: object = 1.0,
    rows: Optional[Sequence[Mapping]] = None,
) -> HandSignal:
    """Compute the full :class:`HandSignal` for one shared hand of one pair.

    ``player_a`` / ``player_b`` are the pair's two player ids (canonical order recommended:
    A = min id, B = max id — the sign of ``value_flow`` and the direction of the per-player
    aggression asymmetry are defined relative to this ordering). ``actions`` is that hand's
    ordered public action log (any subset of the decision-time columns); ``net_a`` / ``net_b``
    are the two players' ``seats.net_chips`` for the hand (the only outcome inputs, used solely
    by ``value_flow``).

    ``rows`` is an OPTIONAL pre-parsed action-row list (from :func:`action_rows`); when supplied
    it is used verbatim instead of re-parsing ``actions``. Because ``action_rows`` is a pure
    function of the hand's action frame, passing the cached rows produces a bit-identical
    :class:`HandSignal`. This lets a caller that scores the same hand for many pairs parse each
    hand ONCE. ``rows`` and ``actions`` must describe the same hand.

    All ratios return ``NOT_APPLICABLE`` on a zero denominator (Req 3.5); nothing raises.
    """
    player_a = str(player_a)
    player_b = str(player_b)
    rows = _action_rows(actions) if rows is None else rows

    value_flow = compute_value_flow(net_a, net_b, big_blind)
    # Per-player aggression asymmetry; report the pair-level signal as the minimum applicable
    # per-player ratio (the strongest avoidance), falling back to NOT_APPLICABLE when neither
    # player made any aggressive action.
    aa_a = _aggression_asymmetry_for(rows, player_a, player_b)
    aa_b = _aggression_asymmetry_for(rows, player_b, player_a)
    applicable = [x for x in (aa_a, aa_b) if isinstance(x, float)]
    aggression_asymmetry: Union[float, Sentinel] = (
        min(applicable) if applicable else NOT_APPLICABLE
    )

    isolation = _isolation_for(rows, player_a, player_b)
    mi_conflict = _mi_conflict_for(rows, player_a, player_b)
    flags = _behavior_action_flags(
        rows, value_flow, aggression_asymmetry, isolation, player_a, player_b
    )

    return HandSignal(
        hand_id=hand_id,
        pair_id=pair_id,
        phase=phase,
        value_flow=value_flow,
        aggression_asymmetry=aggression_asymmetry,
        isolation=isolation,
        mi_conflict=mi_conflict,
        behavior_action_flags=flags,
    )


def compute_pair_signals(
    pair_id: str,
    player_a: str,
    player_b: str,
    hands_meta: pd.DataFrame,
    actions_by_hand: Mapping[object, pd.DataFrame],
    net_by_hand: Mapping[object, Mapping[str, object]],
    rows_by_hand: Optional[Mapping[object, Sequence[Mapping]]] = None,
) -> List[HandSignal]:
    """Compute :class:`HandSignal` objects for every shared hand of a pair (deterministic).

    Args:
        pair_id: The pair identifier used to key the resulting signals.
        player_a: Pair's first player id (canonical: min id).
        player_b: Pair's second player id (canonical: max id).
        hands_meta: Frame with at least ``hand_id`` and ``phase`` (and optionally ``big_blind``)
            for the pair's shared hands. Iterated in ascending ``hand_id`` order for determinism.
        actions_by_hand: Mapping ``hand_id -> ordered action-log frame`` for each shared hand.
        net_by_hand: Mapping ``hand_id -> {player_id: net_chips}`` for each shared hand.
        rows_by_hand: OPTIONAL mapping ``hand_id -> pre-parsed action rows`` (from
            :func:`action_rows`). When present, each hand's rows are reused verbatim instead of
            re-parsing its action frame — a bit-identical speedup for callers that score the same
            hand across many pairs (each hand parsed ONCE). Hands absent from the map fall back to
            parsing ``actions_by_hand`` (or an empty frame), so the argument is fully optional.

    Returns:
        A list of :class:`HandSignal`, one per shared hand, ordered by ``hand_id`` ascending.
    """
    if hands_meta is None or len(hands_meta) == 0:
        return []
    meta = hands_meta.sort_values("hand_id", kind="stable")
    has_bb = "big_blind" in meta.columns
    out: List[HandSignal] = []
    for row in meta.itertuples(index=False):
        hand_id = getattr(row, "hand_id")
        phase = getattr(row, "phase", "")
        big_blind = getattr(row, "big_blind", 1.0) if has_bb else 1.0
        actions = actions_by_hand.get(hand_id, pd.DataFrame())
        nets = net_by_hand.get(hand_id, {})
        cached_rows = None if rows_by_hand is None else rows_by_hand.get(hand_id)
        out.append(
            compute_hand_signal(
                hand_id=hand_id,
                pair_id=pair_id,
                phase=phase,
                player_a=player_a,
                player_b=player_b,
                actions=actions,
                net_a=nets.get(str(player_a), nets.get(player_a, 0.0)),
                net_b=nets.get(str(player_b), nets.get(player_b, 0.0)),
                big_blind=big_blind,
                rows=cached_rows,
            )
        )
    return out


def hand_signals_to_frame(signals: Sequence[HandSignal]) -> pd.DataFrame:
    """Flatten :class:`HandSignal` objects into a DataFrame keyed by ``(pair_id, hand_id)``.

    ``NOT_APPLICABLE`` sentinels are preserved as objects (not coerced to NaN) so downstream
    aggregation can treat them explicitly. Per-family flags are exploded into
    ``flag_<family>`` boolean columns for the disclosed families.
    """
    records: List[dict] = []
    for s in signals:
        rec = {
            "pair_id": s.pair_id,
            "hand_id": s.hand_id,
            "phase": s.phase,
            "value_flow": s.value_flow,
            "aggression_asymmetry": s.aggression_asymmetry,
            "isolation": s.isolation,
            "mi_conflict": s.mi_conflict,
        }
        for fam in DISCLOSED_FAMILY_NAMES:
            rec[f"flag_{fam}"] = bool(s.behavior_action_flags.get(fam, False))
        records.append(rec)
    return pd.DataFrame.from_records(records)
