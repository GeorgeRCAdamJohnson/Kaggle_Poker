# Feature: poker-collusion-detection, Property 5: Feature computation is deterministic
"""Property 5 (task 10.4): feature computation is DETERMINISTIC and the dev/eval definition is
identical.

**Validates: Requirements 5.3, 5.5**

:func:`poker_collusion.features.pair_features.build_pair_feature_set` folds a pair's per-hand
:class:`~poker_collusion.types.HandSignal` stream into the single
:class:`~poker_collusion.types.PairFeatureSet` every downstream consumer reads. This module
exercises the two guarantees the design's Property 5 pins on it:

Determinism / order-invariance (Req 5.5)
----------------------------------------
Building the feature set twice from the SAME inputs must yield an EQUAL result (identical
``features`` dict, identical ``schema_version`` / ``threshold_version``), and building on a
SHUFFLED copy of the same signals must yield an IDENTICAL ``features`` dict. There is no
randomness, no wall-clock, and no dict-iteration-order dependence in the assembly: the signals
are aggregated through a deterministic, order-invariant pass, the confounder block is pure
arithmetic over the same signals + injected baseline, and the final dict is materialised in a
fixed sorted key order. We compare via :func:`pair_feature_set_to_dict` for a stable, hashable
comparison surface.

Identical dev/eval definition (Req 5.3)
---------------------------------------
The feature FORMULAS are identical for the Development_Period and the Evaluation_Period — the
ONLY thing that legitimately differs dev-vs-eval is *which* shared hands feed the aggregation
(``phase=`` selects the slice). So if we take the SAME multiset of signals and relabel every
signal's phase to ``"development"`` in one copy and to ``"evaluation"`` in the other (keeping
everything else identical), then build with the matching ``phase`` argument, every feature EXCEPT
the provenance ``phase_is_development`` flag must be byte-identical. That isolates "the
definitions/formulas don't differ dev vs eval" from "only the included set differs" — here the
included set is deliberately the same, so all real features must match.

Stable schema + finite values
------------------------------
Across arbitrary random inputs the ``features`` dict must always expose the SAME stable sorted set
of keys (a fixed schema; no feature appears/disappears based on the data), and every value must be
a finite float (no NaN, no inf, and no ``NOT_APPLICABLE`` sentinel leaking into the numeric dict).

Generator backend
-----------------
Prefers ``hypothesis`` (guarded via ``importlib.util.find_spec``); when it is not importable the
test transparently falls back to a seeded randomized loop (250 seeds) drawing the same input
space. The active backend is recorded in ``GENERATOR_IN_USE`` and surfaced by
``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import math
import random
from typing import Dict, List, Optional, Sequence

import pytest

from poker_collusion.features.pair_features import (
    CONFOUNDER_FEATURE_NAMES,
    FIELD_BASELINE_KEYS,
    build_pair_feature_set,
    pair_feature_set_to_dict,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal, PairFeatureSet

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

#: The disclosed behavior-action flag families that may appear on a HandSignal.
_FLAG_FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")

#: The lone provenance marker that is ALLOWED to differ between a dev-labelled and an
#: eval-labelled build of the same multiset (everything else must be identical, Req 5.3).
_PROVENANCE_PHASE_KEY = "phase_is_development"


# --------------------------------------------------------------------------- #
# Random input construction (shared by both backends)
# --------------------------------------------------------------------------- #
def _rand_signal_field(rng: random.Random) -> object:
    """A ratio-style signal value that is SOMETIMES the NOT_APPLICABLE sentinel.

    Roughly a quarter of the draws return the zero-denominator sentinel so the determinism and
    finiteness assertions genuinely exercise sentinel handling (the sentinel must never leak into
    the numeric feature dict).
    """
    if rng.random() < 0.25:
        return NOT_APPLICABLE
    return rng.uniform(-2.0, 2.0)


def _rand_flags(rng: random.Random) -> Dict[str, bool]:
    """A random behavior-action-flag map over the disclosed families (some True, some absent)."""
    flags: Dict[str, bool] = {}
    for fam in _FLAG_FAMILIES:
        # sometimes omit the key entirely, sometimes True, sometimes False
        r = rng.random()
        if r < 0.34:
            continue
        flags[fam] = r > 0.67
    return flags


def _draw_signals(rng: random.Random, pair_id: str) -> List[HandSignal]:
    """Draw a random list of HandSignal for one pair (random size 0..25).

    hand_ids are random (and may collide — the aggregation tie-breaks deterministically on the
    string form); value_flow spans positive and negative; aggression_asymmetry / isolation are
    sometimes the NOT_APPLICABLE sentinel; mi_conflict and phase are random.
    """
    n = rng.randint(0, 25)
    signals: List[HandSignal] = []
    for _ in range(n):
        signals.append(
            HandSignal(
                hand_id=rng.randint(0, 10_000),
                pair_id=pair_id,
                phase=rng.choice(("development", "evaluation")),
                value_flow=rng.uniform(-500.0, 500.0),
                aggression_asymmetry=_rand_signal_field(rng),
                isolation=_rand_signal_field(rng),
                mi_conflict=rng.uniform(-1.0, 1.0),
                behavior_action_flags=_rand_flags(rng),
            )
        )
    return signals


def _draw_field_baseline(
    rng: random.Random, players: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """A random per-player field-baseline mapping over the documented FIELD_BASELINE_KEYS.

    Occasionally omits a player entirely (the assembler must default the absent baseline to a
    neutral 0.0 without changing the schema).
    """
    baseline: Dict[str, Dict[str, float]] = {}
    for p in players:
        if rng.random() < 0.2:
            continue  # missing player -> neutral defaults downstream
        baseline[str(p)] = {
            "mean_value_flow_vs_field": rng.uniform(-300.0, 300.0),
            "mean_aggression_vs_field": rng.uniform(0.0, 1.0),
            "coseat_hands": float(rng.randint(0, 500)),
        }
    return baseline


def _draw_case(rng: random.Random):
    """Draw one full random case: (pair_id, signals, player_a, player_b, field_baseline)."""
    a = rng.randint(1, 100)
    b = rng.randint(1, 100)
    player_a, player_b = str(min(a, b)), str(max(a, b))
    pair_id = f"{player_a}__{player_b}"
    signals = _draw_signals(rng, pair_id)
    field_baseline = _draw_field_baseline(rng, (player_a, player_b))
    return pair_id, signals, player_a, player_b, field_baseline


def _with_phase(signals: Sequence[HandSignal], phase: str) -> List[HandSignal]:
    """Return a copy of ``signals`` with EVERY signal relabelled to ``phase`` (all else identical)."""
    out: List[HandSignal] = []
    for s in signals:
        out.append(
            HandSignal(
                hand_id=s.hand_id,
                pair_id=s.pair_id,
                phase=phase,
                value_flow=s.value_flow,
                aggression_asymmetry=s.aggression_asymmetry,
                isolation=s.isolation,
                mi_conflict=s.mi_conflict,
                behavior_action_flags=dict(s.behavior_action_flags),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Expected stable schema (computed once from a non-trivial case)
# --------------------------------------------------------------------------- #
def _reference_feature_keys() -> frozenset:
    """The sorted set of feature keys the assembler emits, derived from a representative build.

    Used to assert the schema is STABLE across arbitrary inputs. Includes the confounder feature
    names, the provenance markers, and every aggregation feature.
    """
    rng = random.Random(0xC0FFEE)
    _, signals, pa, pb, fb = _draw_case(rng)
    # ensure a non-empty, mixed stream so all feature families are present
    signals = _with_phase(_draw_signals(random.Random(1), "1__2"), "development")
    fs = build_pair_feature_set("1__2", signals, "1", "2", phase="development", field_baseline=fb)
    return frozenset(fs.features.keys())


_EXPECTED_KEYS = _reference_feature_keys()


# --------------------------------------------------------------------------- #
# Core assertion (shared by both backends)
# --------------------------------------------------------------------------- #
def _assert_property(
    pair_id: str,
    signals: Sequence[HandSignal],
    player_a: str,
    player_b: str,
    field_baseline: Dict[str, Dict[str, float]],
) -> None:
    """Assert determinism, order-invariance, identical dev/eval definition, stable finite schema."""
    phase = "development"

    # --- 1. Determinism: same inputs twice => EQUAL PairFeatureSet (Req 5.5). ---------------- #
    fs1 = build_pair_feature_set(
        pair_id, list(signals), player_a, player_b, phase=phase, field_baseline=field_baseline
    )
    fs2 = build_pair_feature_set(
        pair_id, list(signals), player_a, player_b, phase=phase, field_baseline=field_baseline
    )
    assert isinstance(fs1, PairFeatureSet) and isinstance(fs2, PairFeatureSet)
    assert pair_feature_set_to_dict(fs1) == pair_feature_set_to_dict(fs2), (
        "non-deterministic: two builds on identical inputs produced different feature sets"
    )
    assert fs1.schema_version == fs2.schema_version
    assert fs1.threshold_version == fs2.threshold_version

    # --- 2. Order-invariance: a SHUFFLED copy => identical features dict (Req 5.5). ---------- #
    shuffled = list(signals)
    _shuffle_rng.shuffle(shuffled)
    fs_shuf = build_pair_feature_set(
        pair_id, shuffled, player_a, player_b, phase=phase, field_baseline=field_baseline
    )
    assert pair_feature_set_to_dict(fs_shuf)["features"] == pair_feature_set_to_dict(fs1)["features"], (
        "feature dict changed under input reordering (should be order-invariant)"
    )

    # --- 3. Identical dev/eval DEFINITION (Req 5.3). ---------------------------------------- #
    # Same multiset, relabelled entirely to development vs entirely to evaluation, built with the
    # matching phase arg. Every feature except the provenance phase flag must be identical.
    dev_signals = _with_phase(signals, "development")
    eval_signals = _with_phase(signals, "evaluation")
    fs_dev = build_pair_feature_set(
        pair_id, dev_signals, player_a, player_b, phase="development", field_baseline=field_baseline
    )
    fs_eval = build_pair_feature_set(
        pair_id, eval_signals, player_a, player_b, phase="evaluation", field_baseline=field_baseline
    )
    dev_feats = fs_dev.features
    eval_feats = fs_eval.features
    assert set(dev_feats.keys()) == set(eval_feats.keys()), (
        "dev/eval builds emit different feature keys (schema must be identical)"
    )
    # The provenance flag is the ONLY permitted difference.
    assert dev_feats[_PROVENANCE_PHASE_KEY] == 1.0
    assert eval_feats[_PROVENANCE_PHASE_KEY] == 0.0
    for key in dev_feats:
        if key == _PROVENANCE_PHASE_KEY:
            continue
        assert dev_feats[key] == pytest.approx(eval_feats[key], rel=0, abs=0), (
            f"feature '{key}' differs dev-vs-eval definition: "
            f"dev={dev_feats[key]} eval={eval_feats[key]} (only formulas via phase-slice may differ)"
        )

    # --- 4. Stable schema + finite values (no NaN/inf, no sentinel leakage). ---------------- #
    keys = frozenset(fs1.features.keys())
    assert keys == _EXPECTED_KEYS, (
        f"feature schema drifted for this input: "
        f"missing={sorted(_EXPECTED_KEYS - keys)} extra={sorted(keys - _EXPECTED_KEYS)}"
    )
    # Confounder features and provenance markers are always present.
    for name in CONFOUNDER_FEATURE_NAMES:
        assert name in keys, f"confounder feature {name!r} missing from schema"
    assert "n_shared_hands" in keys
    assert _PROVENANCE_PHASE_KEY in keys
    # Every value is a finite float.
    for name, val in fs1.features.items():
        assert isinstance(val, float), f"feature {name!r} is not a float: {type(val)}"
        assert math.isfinite(val), f"feature {name!r} is not finite: {val!r}"


#: A fixed-seed RNG used ONLY to shuffle within the assertion (kept deterministic so the test
#: itself is reproducible).
_shuffle_rng = random.Random(0x5EED)


# --------------------------------------------------------------------------- #
# Explicit edge cases (always run under both backends)
# --------------------------------------------------------------------------- #
def test_empty_signal_stream_is_deterministic_and_finite():
    """No shared hands: the feature set is still deterministic, finite, and schema-stable."""
    fs1 = build_pair_feature_set("1__2", [], "1", "2", phase="development", field_baseline={})
    fs2 = build_pair_feature_set("1__2", [], "1", "2", phase="development", field_baseline={})
    assert pair_feature_set_to_dict(fs1) == pair_feature_set_to_dict(fs2)
    assert frozenset(fs1.features.keys()) == _EXPECTED_KEYS
    for name, val in fs1.features.items():
        assert isinstance(val, float) and math.isfinite(val), f"{name}={val!r}"


def test_all_not_applicable_signals_no_sentinel_leak():
    """A stream where every ratio signal is the sentinel must not leak NOT_APPLICABLE into features."""
    signals = [
        HandSignal(
            hand_id=h,
            pair_id="1__2",
            phase="evaluation",
            value_flow=float(10 * (h + 1)),
            aggression_asymmetry=NOT_APPLICABLE,
            isolation=NOT_APPLICABLE,
            mi_conflict=0.0,
            behavior_action_flags={"soft_play": True},
        )
        for h in range(5)
    ]
    fs = build_pair_feature_set("1__2", signals, "1", "2", phase="evaluation", field_baseline={})
    for name, val in fs.features.items():
        assert isinstance(val, float) and math.isfinite(val), f"{name}={val!r}"
    assert frozenset(fs.features.keys()) == _EXPECTED_KEYS


def test_dev_eval_definition_identical_explicit():
    """A concrete mixed stream: dev vs eval builds agree on every feature but the phase flag."""
    base = [
        HandSignal(
            hand_id=h,
            pair_id="3__7",
            phase="development",  # will be relabelled
            value_flow=float(h - 2) * 25.0,
            aggression_asymmetry=(NOT_APPLICABLE if h % 2 else 0.3),
            isolation=0.1 * h,
            mi_conflict=0.05 * h,
            behavior_action_flags={"directed_transfer": h % 3 == 0},
        )
        for h in range(6)
    ]
    fb = {"3": {"mean_value_flow_vs_field": 5.0, "mean_aggression_vs_field": 0.2, "coseat_hands": 40.0}}
    fs_dev = build_pair_feature_set(
        "3__7", _with_phase(base, "development"), "3", "7", phase="development", field_baseline=fb
    )
    fs_eval = build_pair_feature_set(
        "3__7", _with_phase(base, "evaluation"), "3", "7", phase="evaluation", field_baseline=fb
    )
    assert fs_dev.features[_PROVENANCE_PHASE_KEY] == 1.0
    assert fs_eval.features[_PROVENANCE_PHASE_KEY] == 0.0
    for key in fs_dev.features:
        if key == _PROVENANCE_PHASE_KEY:
            continue
        assert fs_dev.features[key] == fs_eval.features[key], f"{key} differs dev-vs-eval"


def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 5] generator in use: {GENERATOR_IN_USE}")


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _sentinel_or_float = st.one_of(
        st.just(NOT_APPLICABLE),
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )

    @st.composite
    def _hand_signals(draw, pair_id: str):
        n = draw(st.integers(min_value=0, max_value=25))
        signals = []
        for _ in range(n):
            flags = draw(
                st.dictionaries(
                    keys=st.sampled_from(_FLAG_FAMILIES),
                    values=st.booleans(),
                    max_size=len(_FLAG_FAMILIES),
                )
            )
            signals.append(
                HandSignal(
                    hand_id=draw(st.integers(min_value=0, max_value=10_000)),
                    pair_id=pair_id,
                    phase=draw(st.sampled_from(("development", "evaluation"))),
                    value_flow=draw(
                        st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
                    ),
                    aggression_asymmetry=draw(_sentinel_or_float),
                    isolation=draw(_sentinel_or_float),
                    mi_conflict=draw(
                        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
                    ),
                    behavior_action_flags=flags,
                )
            )
        return signals

    @st.composite
    def _cases(draw):
        a = draw(st.integers(min_value=1, max_value=100))
        b = draw(st.integers(min_value=1, max_value=100))
        player_a, player_b = str(min(a, b)), str(max(a, b))
        pair_id = f"{player_a}__{player_b}"
        signals = draw(_hand_signals(pair_id))
        baseline = {}
        for p in (player_a, player_b):
            if draw(st.booleans()):
                baseline[p] = {
                    "mean_value_flow_vs_field": draw(
                        st.floats(min_value=-300.0, max_value=300.0, allow_nan=False, allow_infinity=False)
                    ),
                    "mean_aggression_vs_field": draw(
                        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
                    ),
                    "coseat_hands": float(draw(st.integers(min_value=0, max_value=500))),
                }
        return pair_id, signals, player_a, player_b, baseline

    @settings(max_examples=250, deadline=None)
    @given(case=_cases())
    def test_property_feature_determinism_hypothesis(case):
        pair_id, signals, player_a, player_b, baseline = case
        _assert_property(pair_id, signals, player_a, player_b, baseline)

else:

    _N_SEEDS = 250

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_feature_determinism_seeded_loop(seed: int):
        rng = random.Random(seed)
        pair_id, signals, player_a, player_b, baseline = _draw_case(rng)
        _assert_property(pair_id, signals, player_a, player_b, baseline)
