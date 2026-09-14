# Feature: poker-collusion-detection, Property 3: Zero-denominator signals yield the sentinel, never an error
"""Property 3 (task 9.4): zero-denominator signals yield the sentinel, never an error.

**Validates: Requirements 3.5**

For every shared hand, :func:`poker_collusion.features.signals.compute_hand_signal`
must be *total*: it never raises. Whenever a ratio signal's denominator is zero it
must carry the shared ``NOT_APPLICABLE`` sentinel from :mod:`poker_collusion.types`
(identity, ``is NOT_APPLICABLE``) — never a raised error and never a float ``NaN``:

* ``aggression_asymmetry`` is ``NOT_APPLICABLE`` when *neither* pair member makes a
  single bet/raise-equivalent action in the hand (no per-player denominator).
* ``isolation`` is ``NOT_APPLICABLE`` when the pair never bet/raises into each other
  while both are active (zero "at each other" denominator).

When a signal *is* applicable it must instead be a finite float in its sensible
range (``aggression_asymmetry`` and ``isolation`` are non-negative; asymmetry lies
in ``[0, 1]``).

Generator: this test prefers the `hypothesis` library. If hypothesis is not
importable in the running interpreter, it transparently falls back to a seeded
randomized generator loop (300 seeds) that fabricates small random action logs for
a pair — random players, random action tokens, random amounts / to_call, random
fold orderings, and edge cases (empty logs, logs where no pair member ever
bet/raises, and logs where the pair never bet/raises into each other). The active
backend is recorded in ``GENERATOR_IN_USE`` and reported by
``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import math
import random

import pandas as pd

from poker_collusion.features.signals import compute_hand_signal, is_aggressive_action
from poker_collusion.types import NOT_APPLICABLE, HandSignal

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

# The closed 6-token action vocabulary (Discovery §2.1 H4).
ACTION_TOKENS = ["fold", "check", "call", "bet", "raise", "all_in"]


# --------------------------------------------------------------------------- #
# Random hand fabrication (shared by both backends)
# --------------------------------------------------------------------------- #
def _build_actions(rows: list[dict]) -> pd.DataFrame:
    """Build an ordered action-log frame, auto-assigning action_no by order."""
    for i, r in enumerate(rows):
        r.setdefault("action_no", i)
    if not rows:
        return pd.DataFrame(
            columns=["action_no", "player_id", "action", "amount", "to_call", "street"]
        )
    return pd.DataFrame(rows)


def _make_random_hand(rng: random.Random) -> dict:
    """Fabricate a random small hand for a pair (A, B) plus some third players.

    Returns a dict with the ``player_a`` / ``player_b`` ids and the ordered action
    log frame. Deliberately biased to frequently hit the zero-denominator cases:
    empty logs, all-passive logs, and logs where the pair only aggresses at others.
    """
    # Random distinct player ids; A/B are two of them, canonical order.
    n_players = rng.randint(2, 5)
    players = rng.sample(range(1000), n_players)
    player_a, player_b = sorted(rng.sample(players, 2))[:2]
    player_a, player_b = str(min(player_a, player_b)), str(max(player_a, player_b))

    n_actions = rng.randint(0, 8)  # 0 exercises the empty-log edge case

    # With some probability, restrict pair members to passive-only tokens so that
    # the aggression / isolation denominators are guaranteed zero.
    pair_passive_only = rng.random() < 0.35
    pair_tokens = ["fold", "check", "call"] if pair_passive_only else ACTION_TOKENS

    rows: list[dict] = []
    for _ in range(n_actions):
        actor = rng.choice([str(p) for p in players])
        if actor in (player_a, player_b):
            token = rng.choice(pair_tokens)
        else:
            token = rng.choice(ACTION_TOKENS)
        amount = rng.choice([0, 0, 10, 25, 50, 100])
        to_call = rng.choice([0, 10, 25, 50, 100])
        rows.append(
            {
                "player_id": actor,
                "action": token,
                "amount": amount,
                "to_call": to_call,
                "street": rng.choice(["preflop", "flop", "turn", "river"]),
            }
        )

    return {
        "player_a": player_a,
        "player_b": player_b,
        "actions": _build_actions(rows),
    }


# --------------------------------------------------------------------------- #
# The single-hand invariant, asserted for every generated hand
# --------------------------------------------------------------------------- #
def _assert_sentinel_invariant(player_a: str, player_b: str, actions: pd.DataFrame) -> None:
    """compute_hand_signal must not raise; zero-denominator ratios ARE the sentinel."""
    try:
        sig: HandSignal = compute_hand_signal(
            hand_id=1,
            pair_id="P",
            phase="development",
            player_a=player_a,
            player_b=player_b,
            actions=actions,
        )
    except Exception as exc:  # noqa: BLE001 - any exception is a hard failure of the property
        rows = actions.to_dict("records") if actions is not None else None
        raise AssertionError(
            f"compute_hand_signal raised {type(exc).__name__}: {exc}; "
            f"pair=({player_a},{player_b}) rows={rows}"
        ) from exc

    rows = actions.to_dict("records")

    # --- aggression_asymmetry ------------------------------------------------ #
    a_aggr = any(
        str(r.get("player_id")) == player_a
        and is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call"))
        for r in rows
    )
    b_aggr = any(
        str(r.get("player_id")) == player_b
        and is_aggressive_action(r.get("action"), r.get("amount"), r.get("to_call"))
        for r in rows
    )
    aa = sig.aggression_asymmetry
    if not (a_aggr or b_aggr):
        # No pair member made any bet/raise -> every per-player denominator is 0.
        assert aa is NOT_APPLICABLE, (
            f"aggression_asymmetry must be the sentinel when neither pair member "
            f"aggresses; got {aa!r}; rows={rows}"
        )
    else:
        assert isinstance(aa, float) and math.isfinite(aa), (
            f"applicable aggression_asymmetry must be a finite float; got {aa!r}; rows={rows}"
        )
        assert 0.0 <= aa <= 1.0, f"aggression_asymmetry out of [0,1]: {aa!r}; rows={rows}"

    # Never a NaN float regardless of branch.
    if isinstance(aa, float):
        assert not math.isnan(aa), f"aggression_asymmetry is NaN; rows={rows}"

    # --- isolation ----------------------------------------------------------- #
    iso = sig.isolation
    assert iso is NOT_APPLICABLE or isinstance(iso, float), (
        f"isolation must be a float or the sentinel; got {iso!r}; rows={rows}"
    )
    if isinstance(iso, float):
        assert math.isfinite(iso) and not math.isnan(iso), (
            f"applicable isolation must be finite (never NaN); got {iso!r}; rows={rows}"
        )
        assert iso >= 0.0, f"isolation must be non-negative; got {iso!r}; rows={rows}"

    # mi_conflict is always a finite float (never sentinel, never NaN).
    assert isinstance(sig.mi_conflict, float) and math.isfinite(sig.mi_conflict), (
        f"mi_conflict must be a finite float; got {sig.mi_conflict!r}; rows={rows}"
    )


# --------------------------------------------------------------------------- #
# hypothesis-preferring path
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:
    from hypothesis import given, settings, strategies as st  # type: ignore

    _player_ids = st.integers(min_value=0, max_value=999)

    @st.composite
    def _hands(draw):  # type: ignore[no-untyped-def]
        players = draw(
            st.lists(_player_ids, min_size=2, max_size=5, unique=True)
        )
        a, b = sorted(draw(st.permutations(players))[:2])
        player_a, player_b = str(min(a, b)), str(max(a, b))
        pair_passive_only = draw(st.booleans())
        pair_tokens = ["fold", "check", "call"] if pair_passive_only else ACTION_TOKENS

        def _row(actor_is_pair):
            token = st.sampled_from(pair_tokens if actor_is_pair else ACTION_TOKENS)
            return token

        n = draw(st.integers(min_value=0, max_value=8))
        rows = []
        for _ in range(n):
            actor = str(draw(st.sampled_from(players)))
            is_pair = actor in (player_a, player_b)
            token = draw(st.sampled_from(pair_tokens if is_pair else ACTION_TOKENS))
            amount = draw(st.sampled_from([0, 10, 25, 50, 100]))
            to_call = draw(st.sampled_from([0, 10, 25, 50, 100]))
            rows.append(
                {
                    "player_id": actor,
                    "action": token,
                    "amount": amount,
                    "to_call": to_call,
                    "street": draw(st.sampled_from(["preflop", "flop", "turn", "river"])),
                }
            )
        return player_a, player_b, _build_actions(rows)

    @settings(max_examples=300)
    @given(_hands())
    def test_zero_denominator_signals_are_sentinel(hand):
        player_a, player_b, actions = hand
        _assert_sentinel_invariant(player_a, player_b, actions)

else:

    def test_zero_denominator_signals_are_sentinel():
        """Seeded randomized fallback (300 seeds) for the sentinel property."""
        for seed in range(300):
            rng = random.Random(seed)
            hand = _make_random_hand(rng)
            _assert_sentinel_invariant(hand["player_a"], hand["player_b"], hand["actions"])


# --------------------------------------------------------------------------- #
# Explicit edge cases (always run, both backends)
# --------------------------------------------------------------------------- #
def test_all_check_hand_aggression_asymmetry_is_not_applicable():
    """A hand with no aggression at all -> aggression_asymmetry IS the sentinel."""
    actions = _build_actions(
        [
            {"player_id": "1", "action": "check", "amount": 0, "to_call": 0, "street": "flop"},
            {"player_id": "2", "action": "check", "amount": 0, "to_call": 0, "street": "flop"},
            {"player_id": "3", "action": "check", "amount": 0, "to_call": 0, "street": "flop"},
        ]
    )
    sig = compute_hand_signal(
        hand_id=1,
        pair_id="P",
        phase="development",
        player_a="1",
        player_b="2",
        actions=actions,
    )
    assert sig.aggression_asymmetry is NOT_APPLICABLE
    # And it must be the sentinel identity, not a NaN masquerading as one.
    assert not isinstance(sig.aggression_asymmetry, float)


def test_empty_log_yields_sentinels_and_no_error():
    """An empty action log -> both ratio signals are the sentinel, nothing raises."""
    actions = _build_actions([])
    sig = compute_hand_signal(
        hand_id=1,
        pair_id="P",
        phase="development",
        player_a="1",
        player_b="2",
        actions=actions,
    )
    assert sig.aggression_asymmetry is NOT_APPLICABLE
    assert sig.isolation is NOT_APPLICABLE


def test_pair_never_aggresses_into_each_other_isolation_is_not_applicable():
    """Both pair members only aggress at a third player -> isolation denominator 0 -> sentinel."""
    # A and B both bet while third player 3 is active; neither aggresses while the
    # other pair member is still active AND being contested as a partner target...
    # Construct so the partner has folded before each pair member's aggression.
    actions = _build_actions(
        [
            {"player_id": "2", "action": "fold", "amount": 0, "to_call": 0, "street": "preflop"},
            {"player_id": "1", "action": "bet", "amount": 50, "to_call": 0, "street": "preflop"},
            {"player_id": "3", "action": "call", "amount": 50, "to_call": 50, "street": "preflop"},
        ]
    )
    sig = compute_hand_signal(
        hand_id=1,
        pair_id="P",
        phase="development",
        player_a="1",
        player_b="2",
        actions=actions,
    )
    # Player 2 folded first, so player 1's bet is not "at each other" -> denom 0.
    assert sig.isolation is NOT_APPLICABLE


def test_report_generator_in_use():
    """Surface which generator backend the property test is running under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 3] generator in use: {GENERATOR_IN_USE}")
