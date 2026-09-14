# Feature: poker-collusion-detection, Property 6: Risk score coverage and range
"""Property-based test for the classical baseline scorer's risk-score RANGE + COVERAGE.

This test asserts the two guarantees the classical baseline promises for every pair, no matter
how hostile the input feature vector is (Requirements 6.3, 6.4; Property 6):

* RANGE (Req 6.3): ``score_pair(...).risk_score`` is ALWAYS a finite float in ``[0.0, 1.0]`` —
  including feature vectors with extreme magnitudes (``+/-1e9``), zeros, and ``NaN`` / ``inf``
  values that the scorer must tolerate by treating them as neutral.
* COVERAGE (Req 6.4): ``score_pairs`` over a LIST of feature sets returns exactly one
  :class:`ClassicalScore` per input, in input order, with EVERY input ``pair_id`` present in the
  output and every ``risk_score`` in ``[0, 1]`` (no pair is dropped or left unscored). Pairs that
  are UNSCOREABLE (too few shared hands / all-neutral inputs) still receive a score, and that score
  equals the INJECTED pool prior (itself clamped into ``[0, 1]``).
* DETERMINISM: the same feature set always yields the same ``risk_score``.

Generator backend
-----------------
Prefers the ``hypothesis`` library, guarded via ``importlib.util.find_spec``. When hypothesis is
not importable in the running interpreter (the case here), the test transparently falls back to a
seeded randomized loop (~220 seeds) drawing the same input space — every feature the detectors read
gets random values (including the extreme / ``NaN`` / ``inf`` inputs), plus a random schema /
threshold version, a random injected ``pool_prior`` in ``[0, 1]`` and a random ``min_shared_hands``.
The active backend is recorded in ``GENERATOR_IN_USE`` and surfaced by
``test_report_generator_in_use``.

Fast + hermetic: no I/O, injected pool-prior / min-shared-hands only.
"""

from __future__ import annotations

import importlib.util
import math
import random
from typing import Dict, List, Optional

from poker_collusion.models.classical import (
    ClassicalScore,
    score_pair,
    score_pairs,
)
from poker_collusion.types import PairFeatureSet

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

#: Every feature name the five detectors read (from classical.py). The generator randomizes each of
#: these; missing/NaN/inf are documented to be treated as neutral by the scorer.
FEATURE_NAMES: tuple[str, ...] = (
    "value_flow_abs_p95",
    "value_flow_abs_p90",
    "value_flow_max_abs",
    "value_flow_max_pos",
    "value_flow_mean",
    "value_flow_absmean",
    "aggression_asymmetry_mean",
    "aggression_asymmetry_max",
    "conf_avoidance_gap",
    "conf_joint_isolation_rate",
    "isolation_max",
    "mi_conflict_max",
    "mi_conflict_mean",
    "episodic_score",
    "conf_directedness_contrast",
    "n_shared_hands",
)

#: "Adversarial" feature values the scorer must tolerate as neutral (Req 6.3).
_EXTREME_VALUES: tuple[float, ...] = (
    float("nan"),
    float("inf"),
    float("-inf"),
    1e9,
    -1e9,
    0.0,
)


# --------------------------------------------------------------------------- #
# Random input construction (shared by both backends)
# --------------------------------------------------------------------------- #
def _rand_feature_value(rng: random.Random) -> float:
    """Draw one feature value: sometimes an extreme/NaN/inf, otherwise a random finite float."""
    if rng.random() < 0.35:
        return rng.choice(_EXTREME_VALUES)
    # Random finite magnitude across many orders of magnitude, both signs.
    mag = 10.0 ** rng.uniform(-3.0, 9.0)
    return rng.choice((-1.0, 1.0)) * mag * rng.random()


def _rand_feature_set(rng: random.Random, pair_id: str) -> PairFeatureSet:
    """Build a PairFeatureSet with random values for every feature the detectors read.

    Occasionally drops a feature entirely (missing -> read as neutral) and occasionally forces the
    all-neutral / few-shared-hands unscoreable regimes so coverage covers those explicitly.
    """
    features: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        if rng.random() < 0.1:
            continue  # leave feature absent (must be tolerated)
        features[name] = _rand_feature_value(rng)

    regime = rng.random()
    if regime < 0.2:
        # Force "few shared hands" unscoreable regime.
        features["n_shared_hands"] = float(rng.randint(0, 3))
    elif regime < 0.4:
        # Force "all-neutral" unscoreable regime: every detector input zero/absent.
        features = {"n_shared_hands": float(rng.randint(10, 100))}

    return PairFeatureSet(
        pair_id=pair_id,
        schema_version=f"schema_v{rng.randint(0, 9)}",
        features=features,
        threshold_version=f"thr_v{rng.randint(0, 9)}",
    )


# --------------------------------------------------------------------------- #
# Core assertions (shared by both generator backends)
# --------------------------------------------------------------------------- #
def _assert_range(score: ClassicalScore) -> None:
    """RANGE (Req 6.3): risk_score is a finite float in [0, 1]."""
    rs = score.risk_score
    assert isinstance(rs, float), f"risk_score must be a float, got {type(rs)!r}"
    assert math.isfinite(rs), f"risk_score must be finite, got {rs!r}"
    assert 0.0 <= rs <= 1.0, f"risk_score must be in [0, 1], got {rs!r}"


def _assert_determinism(
    fs: PairFeatureSet, pool_prior: float, min_shared_hands: int
) -> None:
    """DETERMINISM: same feature set -> same risk_score."""
    a = score_pair(fs, pool_prior=pool_prior, min_shared_hands=min_shared_hands)
    b = score_pair(fs, pool_prior=pool_prior, min_shared_hands=min_shared_hands)
    assert a.risk_score == b.risk_score, (
        f"non-deterministic risk_score: {a.risk_score!r} != {b.risk_score!r}"
    )


def _assert_coverage(
    feature_sets: List[PairFeatureSet], pool_prior: float, min_shared_hands: int
) -> None:
    """COVERAGE (Req 6.4): one score per input, in order; every pair scored in [0, 1]."""
    scores = score_pairs(
        feature_sets, pool_prior=pool_prior, min_shared_hands=min_shared_hands
    )
    # Exactly one ClassicalScore per input, in input order.
    assert len(scores) == len(feature_sets), (
        f"expected {len(feature_sets)} scores, got {len(scores)}"
    )
    for fs, sc in zip(feature_sets, scores):
        assert sc.pair_id == fs.pair_id, (
            f"order/coverage mismatch: input {fs.pair_id!r} -> output {sc.pair_id!r}"
        )
        _assert_range(sc)
        # Unscoreable pairs fall back to the injected pool prior (clamped into [0, 1]).
        if sc.unscoreable:
            expected = min(max(float(pool_prior), 0.0), 1.0)
            assert sc.risk_score == expected, (
                f"unscoreable pair {sc.pair_id!r} should get pool prior {expected!r}, "
                f"got {sc.risk_score!r}"
            )

    # Every input pair_id appears in the output (no pair dropped).
    assert {fs.pair_id for fs in feature_sets} <= {sc.pair_id for sc in scores}


def _run_one_case(rng: random.Random) -> None:
    """One property trial: build a random batch, assert range + coverage + determinism."""
    pool_prior = rng.random()  # random injected prior in [0, 1]
    min_shared_hands = rng.randint(0, 10)
    n = rng.randint(1, 8)
    feature_sets = [_rand_feature_set(rng, f"P{i}") for i in range(n)]

    for fs in feature_sets:
        single = score_pair(fs, pool_prior=pool_prior, min_shared_hands=min_shared_hands)
        _assert_range(single)
        _assert_determinism(fs, pool_prior, min_shared_hands)

    _assert_coverage(feature_sets, pool_prior, min_shared_hands)


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) + seeded-loop fallback
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - hypothesis is not importable in this environment
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _feature_value_strat = st.one_of(
        st.sampled_from(_EXTREME_VALUES),
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    _feature_dict_strat = st.dictionaries(
        keys=st.sampled_from(FEATURE_NAMES), values=_feature_value_strat
    )

    @st.composite
    def _feature_set_strat(draw, pair_id: str = "P"):
        return PairFeatureSet(
            pair_id=pair_id,
            schema_version=draw(st.text(min_size=0, max_size=8)),
            features=draw(_feature_dict_strat),
            threshold_version=draw(st.text(min_size=0, max_size=8)),
        )

    @settings(max_examples=250)
    @given(
        feats=_feature_dict_strat,
        pool_prior=st.floats(min_value=0.0, max_value=1.0),
        min_shared_hands=st.integers(min_value=0, max_value=10),
    )
    def test_risk_score_range_hypothesis(feats, pool_prior, min_shared_hands):
        fs = PairFeatureSet(
            pair_id="P",
            schema_version="s",
            features=dict(feats),
            threshold_version="t",
        )
        score = score_pair(fs, pool_prior=pool_prior, min_shared_hands=min_shared_hands)
        _assert_range(score)
        _assert_determinism(fs, pool_prior, min_shared_hands)

    @settings(max_examples=250)
    @given(
        batch=st.lists(_feature_dict_strat, min_size=1, max_size=8),
        pool_prior=st.floats(min_value=0.0, max_value=1.0),
        min_shared_hands=st.integers(min_value=0, max_value=10),
    )
    def test_risk_score_coverage_hypothesis(batch, pool_prior, min_shared_hands):
        feature_sets = [
            PairFeatureSet(
                pair_id=f"P{i}",
                schema_version="s",
                features=dict(feats),
                threshold_version="t",
            )
            for i, feats in enumerate(batch)
        ]
        _assert_coverage(feature_sets, pool_prior, min_shared_hands)

else:

    def test_risk_score_range_and_coverage_seeded_loop():
        """Seeded randomized-loop fallback (~220 seeds) exercising range + coverage + determinism."""
        for seed in range(220):
            rng = random.Random(seed)
            _run_one_case(rng)


# --------------------------------------------------------------------------- #
# Explicit edge cases (always run, both backends)
# --------------------------------------------------------------------------- #
def _fs(pair_id: str, **features: float) -> PairFeatureSet:
    return PairFeatureSet(
        pair_id=pair_id,
        schema_version="s",
        features={k: float(v) for k, v in features.items()},
        threshold_version="t",
    )


def test_all_nan_inf_features_stay_in_range():
    """A feature vector of only NaN/inf values is tolerated and scored in [0, 1] (Req 6.3)."""
    fs = _fs(
        "P",
        value_flow_max_abs=float("nan"),
        value_flow_mean=float("inf"),
        value_flow_absmean=float("-inf"),
        aggression_asymmetry_max=float("nan"),
        conf_avoidance_gap=float("inf"),
        conf_joint_isolation_rate=float("nan"),
        isolation_max=float("inf"),
        mi_conflict_max=float("nan"),
        mi_conflict_mean=float("inf"),
        episodic_score=float("nan"),
        conf_directedness_contrast=float("inf"),
        n_shared_hands=float("nan"),
    )
    _assert_range(score_pair(fs, pool_prior=0.5, min_shared_hands=1))


def test_extreme_magnitudes_stay_in_range():
    """Extreme +/-1e9 magnitudes still map into [0, 1] (Req 6.3)."""
    fs = _fs(
        "P",
        value_flow_max_abs=1e9,
        value_flow_mean=-1e9,
        value_flow_absmean=1e9,
        aggression_asymmetry_max=1e9,
        conf_avoidance_gap=1e9,
        conf_joint_isolation_rate=1e9,
        isolation_max=1e9,
        mi_conflict_max=1e9,
        mi_conflict_mean=1e9,
        episodic_score=1e9,
        conf_directedness_contrast=-1e9,
        n_shared_hands=1e9,
    )
    _assert_range(score_pair(fs, pool_prior=0.5, min_shared_hands=1))


def test_unscoreable_few_shared_hands_gets_pool_prior():
    """Few shared hands -> unscoreable -> injected pool prior (Req 6.4)."""
    fs = _fs("P", value_flow_max_abs=50.0, n_shared_hands=0.0)
    sc = score_pair(fs, pool_prior=0.03, min_shared_hands=5)
    assert sc.unscoreable is True
    assert sc.risk_score == 0.03
    _assert_range(sc)


def test_unscoreable_all_neutral_gets_pool_prior():
    """All-neutral inputs -> unscoreable -> injected pool prior (Req 6.4)."""
    fs = _fs("P", n_shared_hands=100.0)  # plenty of hands, but no coordination signal
    sc = score_pair(fs, pool_prior=0.07, min_shared_hands=1)
    assert sc.unscoreable is True
    assert sc.risk_score == 0.07
    _assert_range(sc)


def test_coverage_no_pair_dropped_mixed_batch():
    """Batch mixing scoreable + unscoreable pairs: one score each, in order, all in [0, 1] (Req 6.4)."""
    feature_sets = [
        _fs("P0", value_flow_max_abs=80.0, value_flow_mean=40.0, n_shared_hands=50.0),
        _fs("P1", n_shared_hands=0.0),  # few shared hands -> unscoreable
        _fs("P2", n_shared_hands=100.0),  # all neutral -> unscoreable
        _fs("P3", conf_avoidance_gap=float("inf"), n_shared_hands=float("nan")),
    ]
    _assert_coverage(feature_sets, pool_prior=0.02, min_shared_hands=3)


def test_pool_prior_clamped_into_range():
    """An out-of-range injected pool prior is clamped so the unscoreable score stays in [0, 1]."""
    fs = _fs("P", n_shared_hands=0.0)
    hi = score_pair(fs, pool_prior=5.0, min_shared_hands=5)
    lo = score_pair(fs, pool_prior=-2.0, min_shared_hands=5)
    assert hi.risk_score == 1.0
    assert lo.risk_score == 0.0
    _assert_range(hi)
    _assert_range(lo)


def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 6] generator in use: {GENERATOR_IN_USE}")
