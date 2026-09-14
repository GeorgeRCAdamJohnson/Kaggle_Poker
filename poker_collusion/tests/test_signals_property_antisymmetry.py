# Feature: poker-collusion-detection, Property 1: Directed value-flow is antisymmetric
"""Property 1 (task 9.2): the directed value-flow signal is ANTISYMMETRIC.

**Validates: Requirements 3.1**

The directed value-flow signal defined in
:func:`poker_collusion.features.signals.compute_value_flow` must satisfy

    value_flow(A, B) == -value_flow(B, A)   for all inputs,

so that "chips flowed A -> B" and "chips flowed B -> A" are exact negatives of one
another. Swapping the two players' per-seat outcomes negates every term of the
signal; this test asserts that identity holds across the whole generated input
space, including the corner cases the requirement cares about:

* ``a == b`` (no attributable transfer -> flow 0, and 0 == -0),
* one-sided outcomes (``a < 0, b > 0`` and the mirror),
* both-positive / both-negative outcomes,
* ``big_blind`` in {1, various positive stakes, and the 0/None fallback to 1.0},
* missing/``None`` nets (a seat with unknown outcome is treated as 0, which still
  preserves antisymmetry).

The same antisymmetry must also hold at the :class:`~poker_collusion.types.HandSignal`
level: swapping ``player_a``/``player_b`` and ``net_a``/``net_b`` through
:func:`~poker_collusion.features.signals.compute_hand_signal` negates the
``value_flow`` field.

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable in
the running interpreter it transparently falls back to a seeded randomized loop
(300 seeds) that draws the same input space — negatives, zeros, large magnitudes,
and missing/``None`` nets — achieving equivalent coverage. The active backend is
recorded in ``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import math
import random

import pandas as pd
import pytest

from poker_collusion.features.signals import compute_hand_signal, compute_value_flow

# Antisymmetry is exact for finite arithmetic here (min/max/subtraction of the same
# floats, then a single divide by the same bb), so we can demand a very tight bound.
TOL = 1e-9

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"


# --------------------------------------------------------------------------- #
# Core assertions (shared by both generator backends)                         #
# --------------------------------------------------------------------------- #
def _assert_value_flow_antisymmetric(net_a: object, net_b: object, big_blind: object) -> None:
    """Assert ``value_flow(a, b, bb) == -value_flow(b, a, bb)`` to within ``TOL``.

    Also asserts the diagonal ``a == b`` case yields exactly 0.0 flow.
    """
    vf_ab = compute_value_flow(net_a, net_b, big_blind)
    vf_ba = compute_value_flow(net_b, net_a, big_blind)

    assert math.isfinite(vf_ab), f"value_flow not finite for ({net_a!r},{net_b!r},{big_blind!r})"
    assert math.isfinite(vf_ba), f"value_flow not finite for ({net_b!r},{net_a!r},{big_blind!r})"

    assert vf_ab == pytest.approx(-vf_ba, abs=TOL), (
        f"antisymmetry violated: value_flow({net_a!r},{net_b!r})={vf_ab} "
        f"!= -value_flow({net_b!r},{net_a!r})={-vf_ba} (bb={big_blind!r})"
    )


def _coerce(v: object) -> float:
    """Mirror the module's None->0 treatment for building the expected diagonal."""
    return 0.0 if v is None else float(v)


def _assert_hand_signal_antisymmetric(
    net_a: object, net_b: object, big_blind: object
) -> None:
    """Assert swapping players + nets through ``compute_hand_signal`` negates value_flow.

    Uses an empty action log (value_flow depends only on the net chips + big blind,
    not on the action stream), so this isolates the value-flow field's antisymmetry
    at the HandSignal level (Req 3.1).
    """
    empty_actions = pd.DataFrame()
    sig_ab = compute_hand_signal(
        hand_id="H",
        pair_id="P",
        phase="evaluation",
        player_a="A",
        player_b="B",
        actions=empty_actions,
        net_a=net_a,
        net_b=net_b,
        big_blind=big_blind,
    )
    sig_ba = compute_hand_signal(
        hand_id="H",
        pair_id="P",
        phase="evaluation",
        player_a="B",
        player_b="A",
        actions=empty_actions,
        net_a=net_b,
        net_b=net_a,
        big_blind=big_blind,
    )
    assert sig_ab.value_flow == pytest.approx(-sig_ba.value_flow, abs=TOL), (
        f"HandSignal value_flow antisymmetry violated: {sig_ab.value_flow} "
        f"!= -{sig_ba.value_flow} for ({net_a!r},{net_b!r},bb={big_blind!r})"
    )


def _assert_all(net_a: object, net_b: object, big_blind: object) -> None:
    """Run both the function-level and HandSignal-level antisymmetry assertions."""
    _assert_value_flow_antisymmetric(net_a, net_b, big_blind)
    _assert_hand_signal_antisymmetric(net_a, net_b, big_blind)
    # Diagonal: equal nets => exactly zero flow (and 0 == -0), regardless of bb.
    if _coerce(net_a) == _coerce(net_b):
        assert compute_value_flow(net_a, net_b, big_blind) == 0.0


# --------------------------------------------------------------------------- #
# Explicit corner cases (always run under both backends)                      #
# --------------------------------------------------------------------------- #
_CORNER_CASES: list[tuple[object, object, object]] = [
    # a == b -> flow 0
    (0.0, 0.0, 1.0),
    (50.0, 50.0, 10.0),
    (-50.0, -50.0, 2.0),
    # one-sided: a loses, b gains
    (-100.0, 100.0, 10.0),
    (-100.0, 100.0, 1.0),
    # one-sided mirror handled by swap inside the assertion
    (-100.0, 30.0, 1.0),   # loss > gain: transfer capped at partner's gain
    (-30.0, 100.0, 1.0),   # gain > loss: transfer capped at own loss
    # both positive / both negative (no attributable transfer -> 0 flow)
    (100.0, 200.0, 5.0),
    (-100.0, -200.0, 5.0),
    # big_blind fallbacks
    (-100.0, 100.0, 0.0),     # bb <= 0 -> fallback 1.0
    (-100.0, 100.0, None),    # bb None -> fallback 1.0
    (-100.0, 100.0, -5.0),    # negative bb -> fallback 1.0
    # large magnitudes
    (-1_000_000.0, 1_000_000.0, 100.0),
    (1e9, -1e9, 1e3),
    # missing / None nets treated as 0
    (None, 100.0, 10.0),
    (100.0, None, 10.0),
    (None, None, 10.0),
    (None, -250.0, 25.0),
]


@pytest.mark.parametrize("net_a,net_b,big_blind", _CORNER_CASES)
def test_corner_cases_antisymmetric(net_a, net_b, big_blind):
    """Hand-picked corners: value-flow antisymmetry at both levels."""
    _assert_all(net_a, net_b, big_blind)


def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 1] generator in use: {GENERATOR_IN_USE}")


# --------------------------------------------------------------------------- #
# Shared random draw of one (net_a, net_b, big_blind) input                   #
# --------------------------------------------------------------------------- #
def _draw_net(rng: random.Random) -> object:
    """Draw a net-chips value spanning negatives, zeros, large magnitudes, and None."""
    kind = rng.random()
    if kind < 0.12:
        return None  # missing outcome -> treated as 0
    if kind < 0.24:
        return 0.0
    # Mixed magnitudes including large ones; both signs.
    magnitude = rng.choice([1.0, 10.0, 100.0, 5_000.0, 1_000_000.0, 1e9])
    val = rng.uniform(-magnitude, magnitude)
    # Occasionally force an integer-typed value to exercise coercion paths.
    if rng.random() < 0.3:
        return int(val)
    return val


def _draw_big_blind(rng: random.Random) -> object:
    """Draw a big blind in {1, various positive, 0/None/negative fallback-to-1.0}."""
    kind = rng.random()
    if kind < 0.15:
        return 0.0        # fallback to 1.0
    if kind < 0.30:
        return None       # fallback to 1.0
    if kind < 0.40:
        return rng.uniform(-100.0, -0.001)  # negative -> fallback to 1.0
    if kind < 0.55:
        return 1.0        # unit stake
    return rng.choice([2.0, 5.0, 10.0, 25.0, 100.0, rng.uniform(0.5, 500.0)])


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback         #
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _finite_floats = st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e12,
        max_value=1e12,
    )
    # nets: finite floats, small ints, or None (missing outcome treated as 0).
    _net_strategy = st.one_of(
        _finite_floats,
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.none(),
    )
    # big blind: positive stakes, plus 0/None/negative that must fall back to 1.0.
    _bb_strategy = st.one_of(
        st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.sampled_from([1.0, 2.0, 5.0, 10.0, 25.0, 100.0]),
        st.floats(max_value=0.0, min_value=-1e6, allow_nan=False, allow_infinity=False),
        st.none(),
    )

    @settings(max_examples=300, deadline=None)
    @given(net_a=_net_strategy, net_b=_net_strategy, big_blind=_bb_strategy)
    def test_property_antisymmetry_hypothesis(net_a, net_b, big_blind):
        _assert_all(net_a, net_b, big_blind)

else:

    # Seeded randomized loop achieving equivalent coverage: 300 seeds over random
    # net_a/net_b/big_blind including negatives, zeros, large magnitudes, and
    # missing/None values, plus forced diagonal (a == b) insertions so the
    # zero-flow boundary is hit regularly.
    _N_SEEDS = 300

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_antisymmetry_seeded_loop(seed: int):
        rng = random.Random(seed)
        net_a = _draw_net(rng)
        net_b = _draw_net(rng)
        big_blind = _draw_big_blind(rng)
        # Every 7th seed forces the equal-nets diagonal to guarantee flow-0 coverage.
        if seed % 7 == 0:
            net_b = net_a
        _assert_all(net_a, net_b, big_blind)
