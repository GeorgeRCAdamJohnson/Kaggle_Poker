# Feature: poker-collusion-detection, Property 2: All-in classification follows the amount/to_call rule
"""Property test for the ``all_in`` classification rule (Requirement 3.4, Property 2).

Rule under test (:func:`poker_collusion.features.signals.is_aggressive_action`):

    * ``all_in`` is classified as a CALL (not aggressive) when ``amount <= to_call``.
    * ``all_in`` is classified as AGGRESSIVE when ``amount > to_call``.
    * The boundary ``amount == to_call`` is NOT aggressive (it merely covers the amount owed).
    * ``bet`` / ``raise`` are always aggressive regardless of ``amount`` / ``to_call``.
    * ``fold`` / ``check`` / ``call`` are never aggressive.
    * A missing ``amount`` or ``to_call`` on an ``all_in`` cannot be shown to exceed the call
      amount, so it is conservatively treated as a call (not aggressive) — documented behavior.

Backend
-------
The property is expressed once in :func:`_check_all_in_property` and driven by two harnesses:
a ``hypothesis`` path (preferred, used when the library is importable) and a seeded
randomized-loop fallback (300 seeds) so the property is still exercised where ``hypothesis``
is unavailable. The active backend is reported via a captured module-level flag.
"""

from __future__ import annotations

import importlib.util
import random

from poker_collusion.features.signals import is_aggressive_action

# ---------------------------------------------------------------------------
# Backend detection (reported to the runner via -s / captured stdout).
# ---------------------------------------------------------------------------
_HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
ACTIVE_BACKEND = "hypothesis" if _HYPOTHESIS_AVAILABLE else "seeded-random-fallback"


def test_report_active_backend(capsys=None):
    """Emit which backend drives the property (visible with ``pytest -s``)."""
    print(f"[test_signals_property_all_in] active backend: {ACTIVE_BACKEND}")
    assert ACTIVE_BACKEND in {"hypothesis", "seeded-random-fallback"}


# ---------------------------------------------------------------------------
# The property, expressed once and reused by both backends.
# ---------------------------------------------------------------------------
def _check_all_in_property(amount: float, to_call: float) -> None:
    """Assert the amount/to_call rule holds for one (amount, to_call) pair."""
    result = is_aggressive_action("all_in", amount, to_call)
    # Core rule: all_in aggressive iff amount > to_call.
    assert result == (amount > to_call), (
        f"all_in classification mismatch for amount={amount!r}, to_call={to_call!r}: "
        f"got aggressive={result}, expected {amount > to_call}"
    )
    # Boundary: amount == to_call must be a call (not aggressive).
    if amount == to_call:
        assert result is False, (
            f"boundary amount==to_call should be a call, got aggressive={result} "
            f"(amount={amount!r})"
        )


# ---------------------------------------------------------------------------
# Static invariants that do not depend on the (amount, to_call) draw.
# ---------------------------------------------------------------------------
def test_bet_and_raise_always_aggressive():
    """bet/raise are aggressive regardless of amount/to_call (including missing values)."""
    for token in ("bet", "raise", "BET", "Raise"):
        for amount, to_call in ((0, 0), (5, 100), (100, 5), (None, None), (10, None), (None, 3)):
            assert is_aggressive_action(token, amount, to_call) is True, (
                f"{token!r} should always be aggressive (amount={amount!r}, to_call={to_call!r})"
            )


def test_fold_check_call_never_aggressive():
    """fold/check/call are never aggressive regardless of amount/to_call."""
    for token in ("fold", "check", "call", "FOLD", "Check", "CALL"):
        for amount, to_call in ((0, 0), (5, 100), (100, 5), (None, None), (10, None)):
            assert is_aggressive_action(token, amount, to_call) is False, (
                f"{token!r} should never be aggressive (amount={amount!r}, to_call={to_call!r})"
            )


def test_all_in_boundary_equal_is_call():
    """The exact boundary amount == to_call classifies as a call (not aggressive)."""
    for value in (0.0, 1.0, 2.5, 50.0, 1_000_000.0):
        assert is_aggressive_action("all_in", value, value) is False, (
            f"all_in with amount==to_call=={value!r} should be a call, not aggressive"
        )


def test_all_in_just_above_and_below_boundary():
    """Just-above the boundary is aggressive; just-below (and equal) is a call."""
    for base in (0.0, 1.0, 10.0, 250.0, 1_000_000.0):
        assert is_aggressive_action("all_in", base + 1.0, base) is True
        if base >= 1.0:
            assert is_aggressive_action("all_in", base - 1.0, base) is False
        assert is_aggressive_action("all_in", base, base) is False


def test_all_in_missing_amount_or_to_call_is_not_aggressive():
    """Missing amount or to_call on an all_in => conservative call (not aggressive)."""
    assert is_aggressive_action("all_in", None, None) is False
    assert is_aggressive_action("all_in", None, 0.0) is False
    assert is_aggressive_action("all_in", None, 100.0) is False
    assert is_aggressive_action("all_in", 100.0, None) is False
    assert is_aggressive_action("all_in", 0.0, None) is False


# ---------------------------------------------------------------------------
# Seeded randomized-loop fallback: always runs, and is the sole driver when
# hypothesis is unavailable. 300 seeds over non-negative amount/to_call, with
# explicit coverage of the equality boundary, just-above/below, zeros, and
# large values.
# ---------------------------------------------------------------------------
def test_all_in_property_seeded_fallback():
    """Exercise the amount/to_call property across 300 seeded random draws."""
    # Fixed boundary / edge cases that must always be covered.
    fixed_cases = [
        (0.0, 0.0),          # zeros, boundary
        (0.0, 1.0),          # amount below to_call
        (1.0, 0.0),          # amount above to_call (aggressive)
        (5.0, 5.0),          # equality boundary
        (5.0, 4.0),          # just above
        (4.0, 5.0),          # just below
        (1e12, 1e12),        # large equal
        (1e12, 0.0),         # large aggressive
        (0.0, 1e12),         # large call
    ]
    for amount, to_call in fixed_cases:
        _check_all_in_property(amount, to_call)

    for seed in range(300):
        rng = random.Random(seed)
        # Draw non-negative amounts/to_call across a wide magnitude range.
        magnitude = 10 ** rng.randint(0, 9)
        amount = rng.random() * magnitude
        to_call = rng.random() * magnitude
        _check_all_in_property(amount, to_call)

        # Deliberately hit the equality boundary on a subset of seeds.
        if seed % 3 == 0:
            shared = rng.random() * magnitude
            _check_all_in_property(shared, shared)
        # And integer/zero draws to cover discrete chip amounts.
        if seed % 5 == 0:
            ia = rng.randint(0, 10_000)
            ic = rng.randint(0, 10_000)
            _check_all_in_property(float(ia), float(ic))


# ---------------------------------------------------------------------------
# hypothesis-preferring path: registered only when the library is importable,
# so the file imports and runs cleanly whether or not hypothesis is installed.
# ---------------------------------------------------------------------------
if _HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only when hypothesis present
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _nonneg = st.floats(
        min_value=0.0,
        max_value=1e12,
        allow_nan=False,
        allow_infinity=False,
    )

    @settings(max_examples=300)
    @given(amount=_nonneg, to_call=_nonneg)
    def test_all_in_property_hypothesis(amount, to_call):
        """all_in aggressive iff amount > to_call, across generated non-negative pairs."""
        _check_all_in_property(amount, to_call)

    @settings(max_examples=200)
    @given(value=_nonneg)
    def test_all_in_equality_boundary_hypothesis(value):
        """The equality boundary amount == to_call is always a call (not aggressive)."""
        assert is_aggressive_action("all_in", value, value) is False
