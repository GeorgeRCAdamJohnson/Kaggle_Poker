# Feature: poker-collusion-detection, Property 4: Episodic concentration never lowers the coordination signal
"""Property 4 (task 10.2): episodic CONCENTRATION never lowers the coordination signal.

**Validates: Requirements 4.1, 4.2**

The episodic coordination score computed by
:func:`poker_collusion.features.aggregation.aggregate_pair` must be **monotone under
concentration**: holding the *multiset* of flagged hands fixed (same flagged count ``k`` and the
same per-hand ``|value_flow|`` magnitudes), any rearrangement in time that packs the flagged
hands closer together may only *raise or hold* the episodic score — never lower it.

Why this holds (see the module docstring's monotonicity proof):

    peak            = max over flagged hands of |value_flow|      (arrangement-INVARIANT)
    max_burst       = length of the longest run of consecutive flagged hands
    episodic_score  = peak * (1 + BURST_GAIN * (max_burst - 1)),  BURST_GAIN >= 0

``peak`` depends only on *which* hands are flagged and their magnitudes, not on their order.
Concentrating the same flagged multiset can only *increase* ``max_burst`` (most-spread => 1,
fully-contiguous => k), and ``episodic_score`` is affine-increasing in ``max_burst``. Hence

    concentrated.episodic_score >= spread.episodic_score.

This test builds, for a fixed flagged multiset over a timeline of ``n`` hands, a **spread**
arrangement (flags placed as far apart / interleaved as possible) and a **concentrated**
arrangement (flags packed contiguously) that share the SAME per-hand ``|value_flow|`` magnitudes
— only the temporal ORDER differs — and asserts:

  * ``concentrated.episodic_score >= spread.episodic_score``   (the core Property-4 invariant),
  * ``concentrated.max_burst      >= spread.max_burst``,
  * both share the same ``flagged_count`` and the same ``episodic_peak`` (flagged mass preserved).

A **monotone-chain** case verifies a sequence of increasingly concentrated arrangements yields a
non-decreasing episodic score. Edge cases ``k == 0`` (score 0, equal), ``k == 1`` (single flag,
equal), and ``k == n`` (all flagged, equal) are exercised explicitly. A **strict** inequality is
asserted only where guaranteed: ``k >= 2`` with a positive peak, where the fully-spread
arrangement has ``max_burst == 1`` while the contiguous arrangement has ``max_burst == k`` and
``BURST_GAIN > 0``.

Ordering control
----------------
:func:`aggregate_pair` orders hands by ``(started_at, str(hand_id))``. :class:`HandSignal` carries
no ``started_at``, so ordering falls to the lexicographic order of ``str(hand_id)``. We therefore
name hands with zero-padded ids (``"h000"``, ``"h001"``, ...) so lexicographic order equals the
intended timeline position, giving us exact control over the flag stream.

Generator backend
-----------------
Prefers ``hypothesis``; if it is not importable it transparently falls back to a seeded
randomized loop (250 seeds) drawing the same space (random ``n`` in 3..30, random ``k`` in 1..n,
random per-flag magnitudes). The active backend is recorded in ``GENERATOR_IN_USE`` and surfaced
by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random
from typing import List, Sequence

import pytest

from poker_collusion.features.aggregation import BURST_GAIN, aggregate_pair
from poker_collusion.types import HandSignal

TOL = 1e-9

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"


# --------------------------------------------------------------------------- #
# HandSignal / timeline construction helpers                                  #
# --------------------------------------------------------------------------- #
def _make_signal(pos: int, *, flagged: bool, magnitude: float) -> HandSignal:
    """Build one HandSignal at timeline position ``pos``.

    ``hand_id`` is a zero-padded string so lexicographic sort == timeline order (see module
    docstring). A flagged hand sets ``behavior_action_flags["directed_transfer"] = True`` and
    carries the given signed ``|value_flow|`` magnitude; a non-flagged hand has all flags False.
    The unflagged ``value_flow`` is irrelevant to the episodic score (peak is anchored to flagged
    hands), but we still set it to 0.0 for clarity.
    """
    return HandSignal(
        hand_id=f"h{pos:04d}",
        pair_id="P",
        phase="evaluation",
        value_flow=float(magnitude) if flagged else 0.0,
        aggression_asymmetry=0.0,
        isolation=0.0,
        mi_conflict=0.0,
        behavior_action_flags={"directed_transfer": bool(flagged)},
    )


def _timeline(flag_positions: Sequence[int], magnitudes: Sequence[float], n: int) -> List[HandSignal]:
    """Assemble a timeline of ``n`` hands with flags at ``flag_positions``.

    ``magnitudes`` gives the ``|value_flow|`` for each flagged hand IN THE ORDER they appear in
    ``flag_positions`` (so the same multiset of magnitudes can be assigned to different position
    sets). Non-flagged positions get magnitude 0.0.
    """
    flag_set = {p: magnitudes[i] for i, p in enumerate(flag_positions)}
    signals: List[HandSignal] = []
    for pos in range(n):
        if pos in flag_set:
            signals.append(_make_signal(pos, flagged=True, magnitude=flag_set[pos]))
        else:
            signals.append(_make_signal(pos, flagged=False, magnitude=0.0))
    return signals


def _spread_positions(n: int, k: int) -> List[int]:
    """Place ``k`` flags as far apart as possible over ``[0, n)`` (max isolation).

    Uses an evenly spaced pattern; when ``k < n`` this yields isolated flags (max_burst == 1).
    When ``k == n`` every position is flagged (unavoidably contiguous).
    """
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # Evenly spaced indices rounded to the timeline; dedupe defensively.
    positions = sorted({min(n - 1, round(i * (n - 1) / (k - 1))) if k > 1 else 0 for i in range(k)})
    # If rounding collided (rare for small n), fall back to a guaranteed-spread stride layout.
    if len(positions) < k:
        positions = list(range(0, 2 * k, 2))  # 0,2,4,... guaranteed isolated within [0, n)
        positions = [p for p in positions if p < n][:k]
        # top up if still short (n too small for full stride) by filling remaining slots
        if len(positions) < k:
            remaining = [p for p in range(n) if p not in set(positions)]
            positions = sorted(positions + remaining[: k - len(positions)])
    return positions


def _concentrated_positions(n: int, k: int, start: int = 0) -> List[int]:
    """Place ``k`` flags contiguously starting at ``start`` (max concentration => max_burst == k)."""
    if k <= 0:
        return []
    start = max(0, min(start, n - k))
    return list(range(start, start + k))


# --------------------------------------------------------------------------- #
# Core invariant assertion (shared by both backends)                          #
# --------------------------------------------------------------------------- #
def _assert_concentration_monotone(n: int, k: int, magnitudes: Sequence[float]) -> None:
    """Assert Property 4 for a fixed flagged multiset over a timeline of ``n`` hands.

    Builds a spread and a concentrated arrangement of the SAME flagged multiset (same ``k`` and
    same magnitudes, only order differs) and checks the score/burst/mass invariants.
    """
    assert len(magnitudes) == k, "magnitude multiset size must equal flagged count k"

    spread_pos = _spread_positions(n, k)
    conc_pos = _concentrated_positions(n, k)

    # SAME multiset of magnitudes assigned to each arrangement (sorted so the multiset is
    # identical regardless of how positions are ordered).
    mags = sorted(float(m) for m in magnitudes)

    spread = aggregate_pair("P", _timeline(spread_pos, mags, n))
    conc = aggregate_pair("P", _timeline(conc_pos, mags, n))

    # Mass preserved: same flagged count and same episodic peak (peak is arrangement-invariant).
    assert spread.flagged_count == k
    assert conc.flagged_count == k
    assert spread.features["episodic_peak"] == pytest.approx(
        conc.features["episodic_peak"], abs=TOL
    ), (
        f"episodic_peak differs across arrangements (mass not preserved): "
        f"spread={spread.features['episodic_peak']} conc={conc.features['episodic_peak']}"
    )

    # Core invariants (>=): concentration never lowers the score, never shortens the top burst.
    assert conc.max_burst >= spread.max_burst, (
        f"max_burst regressed under concentration: conc={conc.max_burst} < spread={spread.max_burst} "
        f"(n={n}, k={k})"
    )
    assert conc.episodic_score >= spread.episodic_score - TOL, (
        f"Property 4 violated: concentrated episodic_score {conc.episodic_score} < spread "
        f"{spread.episodic_score} (n={n}, k={k}, mags={mags})"
    )

    # Strict inequality is GUARANTEED only when k>=2, the peak is positive, and the spread
    # arrangement is genuinely maximally isolated (max_burst == 1) while concentration packs all
    # k together (max_burst == k). BURST_GAIN > 0 then forces a strict increase.
    peak_positive = conc.features["episodic_peak"] > 0.0
    if (
        k >= 2
        and peak_positive
        and BURST_GAIN > 0.0
        and spread.max_burst == 1
        and conc.max_burst == k
    ):
        assert conc.episodic_score > spread.episodic_score + TOL, (
            f"expected STRICT increase (k={k}, spread max_burst=1, conc max_burst={k}, "
            f"BURST_GAIN={BURST_GAIN}) but conc={conc.episodic_score} <= spread="
            f"{spread.episodic_score}"
        )


# --------------------------------------------------------------------------- #
# Explicit edge cases (always run under both backends)                        #
# --------------------------------------------------------------------------- #
def test_edge_k_zero_score_zero_and_equal():
    """k == 0: no flagged hands -> episodic_score 0 for any arrangement (equal)."""
    n = 8
    a = aggregate_pair("P", _timeline([], [], n))
    b = aggregate_pair("P", _timeline([], [], n))
    assert a.flagged_count == 0 and b.flagged_count == 0
    assert a.episodic_score == 0.0
    assert b.episodic_score == 0.0
    assert a.max_burst == 0 and b.max_burst == 0
    assert a.episodic_score == pytest.approx(b.episodic_score, abs=TOL)


def test_edge_k_one_single_flag_equal():
    """k == 1: a single flag can never form a burst > 1, so spread == concentrated."""
    n = 10
    mags = [42.0]
    spread = aggregate_pair("P", _timeline(_spread_positions(n, 1), mags, n))
    conc = aggregate_pair("P", _timeline(_concentrated_positions(n, 1), mags, n))
    assert spread.flagged_count == 1 and conc.flagged_count == 1
    assert spread.max_burst == 1 and conc.max_burst == 1
    assert conc.episodic_score == pytest.approx(spread.episodic_score, abs=TOL)
    assert conc.episodic_score >= spread.episodic_score - TOL


def test_edge_k_equals_n_all_flagged_equal():
    """k == n: every hand flagged -> only one arrangement exists (already contiguous)."""
    n = 6
    mags = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    spread = aggregate_pair("P", _timeline(_spread_positions(n, n), mags, n))
    conc = aggregate_pair("P", _timeline(_concentrated_positions(n, n), mags, n))
    assert spread.flagged_count == n and conc.flagged_count == n
    assert spread.max_burst == n and conc.max_burst == n
    assert conc.episodic_score == pytest.approx(spread.episodic_score, abs=TOL)


def test_strict_increase_when_fully_isolated_vs_contiguous():
    """k >= 2 with positive peak: fully-spread (max_burst 1) < fully-contiguous (max_burst k)."""
    n = 12
    k = 4
    mags = [10.0, 20.0, 30.0, 40.0]  # peak 40 preserved across arrangements
    spread = aggregate_pair("P", _timeline(_spread_positions(n, k), mags, n))
    conc = aggregate_pair("P", _timeline(_concentrated_positions(n, k), mags, n))
    assert spread.max_burst == 1, f"expected isolated spread, got max_burst={spread.max_burst}"
    assert conc.max_burst == k
    assert conc.episodic_score > spread.episodic_score + TOL


def test_monotone_chain_increasing_concentration():
    """A chain of increasingly concentrated arrangements yields non-decreasing episodic_score.

    Fixed multiset (k flags, fixed magnitudes) over n hands. We build arrangements whose longest
    contiguous burst increases 1, 2, ..., k and assert the episodic score is non-decreasing along
    the chain (with a strict rise from the isolated arrangement to any burst>=2 arrangement).
    """
    n = 16
    k = 5
    mags = [5.0, 5.0, 5.0, 5.0, 5.0]

    scores: List[float] = []
    bursts: List[int] = []
    # burst length b => one contiguous block of length b plus (k-b) isolated flags packed after
    # the block with a gap, so max_burst == b exactly.
    for b in range(1, k + 1):
        positions = _positions_with_max_burst(n, k, b)
        res = aggregate_pair("P", _timeline(positions, mags, n))
        assert res.flagged_count == k
        assert res.max_burst == b, f"constructed max_burst {res.max_burst} != target {b}"
        scores.append(res.episodic_score)
        bursts.append(res.max_burst)

    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1] - TOL, (
            f"episodic_score decreased along concentration chain at step {i}: "
            f"{scores[i]} < {scores[i - 1]} (bursts={bursts})"
        )
    # First (max_burst 1) strictly below last (max_burst k) because BURST_GAIN > 0 and peak > 0.
    assert scores[-1] > scores[0] + TOL


def _positions_with_max_burst(n: int, k: int, b: int) -> List[int]:
    """Return flag positions with EXACTLY ``k`` flags whose longest run is EXACTLY ``b``.

    Layout: one contiguous block of length ``b`` at the start, then the remaining ``k - b`` flags
    placed isolated (each separated by a gap) after the block so no other run reaches ``b``.
    Requires enough room; ``n`` is chosen large enough by callers.
    """
    assert 1 <= b <= k <= n
    positions = list(range(0, b))  # block of length b
    remaining = k - b
    # Place isolated flags starting two past the block, stride 2, ensuring gaps (run length 1).
    cursor = b + 1
    while remaining > 0 and cursor < n:
        positions.append(cursor)
        cursor += 2
        remaining -= 1
    assert remaining == 0, f"not enough room to place {k} flags with burst {b} in n={n}"
    return positions


def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 4] generator in use: {GENERATOR_IN_USE}")


# --------------------------------------------------------------------------- #
# Shared random draw of one (n, k, magnitudes) case                           #
# --------------------------------------------------------------------------- #
def _draw_case(rng: random.Random) -> tuple[int, int, List[float]]:
    """Draw a random timeline size ``n`` (3..30), flag count ``k`` (1..n) and k magnitudes."""
    n = rng.randint(3, 30)
    k = rng.randint(1, n)
    magnitudes = [rng.uniform(0.5, 1000.0) for _ in range(k)]
    return n, k, magnitudes


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback         #
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @st.composite
    def _cases(draw):
        n = draw(st.integers(min_value=3, max_value=30))
        k = draw(st.integers(min_value=1, max_value=n))
        magnitudes = draw(
            st.lists(
                st.floats(min_value=0.5, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=k,
                max_size=k,
            )
        )
        return n, k, magnitudes

    @settings(max_examples=300, deadline=None)
    @given(case=_cases())
    def test_property_concentration_monotone_hypothesis(case):
        n, k, magnitudes = case
        _assert_concentration_monotone(n, k, magnitudes)

else:

    _N_SEEDS = 250

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_concentration_monotone_seeded_loop(seed: int):
        rng = random.Random(seed)
        n, k, magnitudes = _draw_case(rng)
        _assert_concentration_monotone(n, k, magnitudes)
