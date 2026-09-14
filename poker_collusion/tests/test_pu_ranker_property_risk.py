# Feature: poker-collusion-detection, Property 6: Risk score coverage and range
"""Property-based test for the PU-aware risk model's risk-score RANGE + COVERAGE.

This test asserts the two guarantees the :class:`PURanker` promises for every evaluation pair,
no matter how hostile the input feature vector is (Requirements 6.3, 6.4; Property 6):

* COVERAGE (Req 6.4): :meth:`PURanker.predict_risk` over a LIST of eval feature sets returns
  exactly one ``(pair_id, risk_score)`` per input, in input order, with EVERY input ``pair_id``
  present in the output (no pair is dropped or left unscored).
* RANGE (Req 6.3): every ``risk_score`` is ALWAYS a finite float in ``[0.0, 1.0]`` — including
  feature vectors with extreme magnitudes (``+/-1e9``), zeros, and ``NaN`` / ``inf`` values that
  the model must tolerate by treating them as neutral.
* UNSCOREABLE default (Req 6.4): pairs that are unscoreable (too few shared hands / all-neutral
  inputs) still receive a score, and that score equals the INJECTED pool prior (itself in
  ``[0, 1]``).
* DETERMINISM (Req 6.5/6.7): scoring the same eval list twice yields identical scores.

Runtime discipline
-------------------
The :class:`PURanker` is fit ONCE per test on a small fixed synthetic training set (7 pairs from
the ``test_pu_ranker`` conventions), then the generated cases vary only the EVAL inputs. sklearn
is never re-fit per example.

Generator backend
-----------------
Prefers the ``hypothesis`` library, guarded via ``importlib.util.find_spec``. When hypothesis is
not importable in the running interpreter (the case here), the test transparently falls back to a
seeded randomized loop (~180 seeds) drawing the same input space — every trained feature gets
random values (including the extreme / ``NaN`` / ``inf`` inputs), a random ``n_shared_hands`` that
sometimes dips below the min-shared threshold, a variable list length ``1..12``, and random
``pair_ids``. The active backend is recorded in ``GENERATOR_IN_USE`` and surfaced by
``test_report_generator_in_use``.

Fast + hermetic: no I/O, injected EDA artifact only, model fit once.
"""

from __future__ import annotations

import importlib.util
import math
import random
from typing import Dict, List, Optional

from poker_collusion.models.pu_ranker import (
    CLASSICAL_SCORE_COLUMN,
    PURanker,
    train_pu_ranker,
)
from poker_collusion.types import LabelTable, PairFeatureSet

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

# --------------------------------------------------------------------------- #
# Shared tiny training fixture (test_pu_ranker conventions)
# --------------------------------------------------------------------------- #
POOL_PRIOR = 0.02
MIN_SHARED = 2

#: The feature names the tiny synthetic training set defines. Eval pairs MUST NOT introduce a
#: feature name outside this set (that is a schema mismatch -> HALT, covered elsewhere in the unit
#: tests). The generator therefore draws values only for these trained columns.
FEATURE_NAMES: tuple[str, ...] = (
    "n_shared_hands",
    "value_flow_max_abs",
    "value_flow_mean",
    "conf_avoidance_gap",
    "conf_joint_isolation_rate",
    "episodic_score",
)

#: "Adversarial" feature values the model must tolerate as neutral (Req 6.3).
_EXTREME_VALUES: tuple[float, ...] = (
    float("nan"),
    float("inf"),
    float("-inf"),
    1e9,
    -1e9,
    0.0,
)


def _artifact(pool_prior: float = POOL_PRIOR, min_shared_hands: int = MIN_SHARED) -> Dict[str, object]:
    return {
        "thresholds": {
            "pool_prior_positive_prevalence": pool_prior,
            "min_shared_hands": min_shared_hands,
        },
        "threshold_version": "test_thresholds",
    }


def _fs(
    pair_id: str,
    *,
    n_shared_hands: float = 10.0,
    value_flow_max_abs: float = 0.0,
    value_flow_mean: float = 0.0,
    conf_avoidance_gap: float = 0.0,
    conf_joint_isolation_rate: float = 0.0,
    episodic_score: float = 0.0,
) -> PairFeatureSet:
    """Tiny synthetic PairFeatureSet with the fixed, shared feature schema."""
    feats: Dict[str, float] = {
        "n_shared_hands": float(n_shared_hands),
        "value_flow_max_abs": float(value_flow_max_abs),
        "value_flow_mean": float(value_flow_mean),
        "conf_avoidance_gap": float(conf_avoidance_gap),
        "conf_joint_isolation_rate": float(conf_joint_isolation_rate),
        "episodic_score": float(episodic_score),
    }
    return PairFeatureSet(
        pair_id=pair_id,
        schema_version="test_schema",
        features=feats,
        threshold_version="test_thresholds",
    )


def _positive_fs(pair_id: str) -> PairFeatureSet:
    return _fs(
        pair_id,
        n_shared_hands=20.0,
        value_flow_max_abs=30.0,
        value_flow_mean=20.0,
        conf_avoidance_gap=0.6,
        conf_joint_isolation_rate=0.5,
        episodic_score=15.0,
    )


def _benign_fs(pair_id: str) -> PairFeatureSet:
    return _fs(
        pair_id,
        n_shared_hands=20.0,
        value_flow_max_abs=1.0,
        value_flow_mean=0.1,
        conf_avoidance_gap=0.01,
        conf_joint_isolation_rate=0.02,
        episodic_score=0.2,
    )


def _train_set() -> List[PairFeatureSet]:
    """Small fixed dev set: 2 positives, 2 confirmed negatives, 3 unknowns (~7 pairs)."""
    return [
        _positive_fs("POS1"),
        _positive_fs("POS2"),
        _benign_fs("NEG1"),
        _benign_fs("NEG2"),
        _benign_fs("UNK1"),
        _fs("UNK2", value_flow_max_abs=5.0, value_flow_mean=2.0, episodic_score=3.0),
        _benign_fs("UNK3"),
    ]


def _label_table() -> LabelTable:
    return LabelTable(
        trusted_positive={"POS1": "directed_transfer", "POS2": "soft_play"},
        confirmed_negative={"NEG1", "NEG2"},
    )


def _fit_ranker(seed: Optional[int] = 101) -> PURanker:
    """Fit the PURanker ONCE on the tiny fixed training set (sklearn fit happens here only)."""
    return train_pu_ranker(
        _train_set(),
        _label_table(),
        eda_artifact=_artifact(),
        seed=seed,
    )


# A single shared fitted model reused across every generated case (fit once, vary eval inputs).
_RANKER = _fit_ranker()
_ARTIFACT = _artifact()


# --------------------------------------------------------------------------- #
# Random EVAL input construction (shared by both backends)
# --------------------------------------------------------------------------- #
def _rand_feature_value(rng: random.Random) -> float:
    """Draw one feature value: sometimes an extreme/NaN/inf, otherwise a random finite float."""
    if rng.random() < 0.35:
        return rng.choice(_EXTREME_VALUES)
    mag = 10.0 ** rng.uniform(-3.0, 9.0)
    return rng.choice((-1.0, 1.0)) * mag * rng.random()


def _rand_eval_fs(rng: random.Random, pair_id: str) -> PairFeatureSet:
    """Build an eval PairFeatureSet over the TRAINED columns with random (possibly extreme) values.

    ``n_shared_hands`` is drawn to sometimes dip below the min-shared threshold, and the all-neutral
    unscoreable regime is forced occasionally so coverage covers those explicitly. Feature names
    never leave the trained schema (an unseen column is a HALT case, tested in the unit suite).
    """
    feats: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name == "n_shared_hands":
            continue
        if rng.random() < 0.1:
            continue  # leave feature absent (read as neutral)
        feats[name] = _rand_feature_value(rng)

    regime = rng.random()
    if regime < 0.2:
        # Force "few shared hands" unscoreable regime (below the min-shared threshold).
        feats["n_shared_hands"] = float(rng.randint(0, MIN_SHARED - 1))
    elif regime < 0.4:
        # Force "all-neutral" unscoreable regime: only a healthy shared-hand count, no signal.
        feats = {"n_shared_hands": float(rng.randint(MIN_SHARED, 100))}
    else:
        feats["n_shared_hands"] = float(rng.randint(0, 100))

    return PairFeatureSet(
        pair_id=pair_id,
        schema_version="test_schema",
        features=feats,
        threshold_version="test_thresholds",
    )


# --------------------------------------------------------------------------- #
# Core assertions (shared by both generator backends)
# --------------------------------------------------------------------------- #
def _assert_range(risk: float) -> None:
    """RANGE (Req 6.3): risk is a finite float in [0, 1]."""
    assert isinstance(risk, float), f"risk must be a float, got {type(risk)!r}"
    assert math.isfinite(risk), f"risk must be finite, got {risk!r}"
    assert 0.0 <= risk <= 1.0, f"risk must be in [0, 1], got {risk!r}"


def _is_unscoreable(fs: PairFeatureSet, min_shared_hands: int) -> bool:
    """Mirror of PURanker's unscoreable rule for asserting the pool-prior default."""
    n_shared = fs.features.get("n_shared_hands", 0.0)
    try:
        n_shared = float(n_shared)
    except (TypeError, ValueError):
        n_shared = 0.0
    if math.isnan(n_shared) or math.isinf(n_shared):
        n_shared = 0.0
    if n_shared < float(min_shared_hands):
        return True
    for name, val in fs.features.items():
        if name in ("n_shared_hands", "phase_is_development"):
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            f = 0.0
        if math.isnan(f) or math.isinf(f):
            f = 0.0
        if abs(f) > 1e-12:
            return False
    return True


def _assert_coverage_and_range(
    ranker: PURanker,
    eval_sets: List[PairFeatureSet],
    artifact: Dict[str, object],
    pool_prior: float,
    min_shared_hands: int,
) -> None:
    """COVERAGE (Req 6.4) + RANGE (Req 6.3) + unscoreable default (Req 6.4)."""
    scored = ranker.predict_risk(eval_sets, eda_artifact=artifact)

    # Exactly one (pair_id, risk) per input, in input order.
    assert len(scored) == len(eval_sets), (
        f"expected {len(eval_sets)} scores, got {len(scored)}"
    )
    for fs, (pid, risk) in zip(eval_sets, scored):
        assert pid == fs.pair_id, (
            f"order/coverage mismatch: input {fs.pair_id!r} -> output {pid!r}"
        )
        _assert_range(risk)
        if _is_unscoreable(fs, min_shared_hands):
            expected = min(max(float(pool_prior), 0.0), 1.0)
            assert risk == expected, (
                f"unscoreable pair {pid!r} should get pool prior {expected!r}, got {risk!r}"
            )

    # Every input pair_id appears in the output (no pair dropped).
    assert {fs.pair_id for fs in eval_sets} == {pid for pid, _ in scored}


def _assert_determinism(
    ranker: PURanker, eval_sets: List[PairFeatureSet], artifact: Dict[str, object]
) -> None:
    """DETERMINISM (Req 6.5/6.7): same eval list -> identical scores."""
    a = ranker.predict_risk(eval_sets, eda_artifact=artifact)
    b = ranker.predict_risk(eval_sets, eda_artifact=artifact)
    assert a == b, f"non-deterministic scores: {a!r} != {b!r}"


def _run_one_case(rng: random.Random) -> None:
    """One property trial: build a random eval batch, assert range + coverage + determinism."""
    n = rng.randint(1, 12)  # variable list length 1..12
    eval_sets = [_rand_eval_fs(rng, f"E{i}_{rng.randint(0, 10_000)}") for i in range(n)]
    _assert_coverage_and_range(_RANKER, eval_sets, _ARTIFACT, POOL_PRIOR, MIN_SHARED)
    _assert_determinism(_RANKER, eval_sets, _ARTIFACT)


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) + seeded-loop fallback
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - hypothesis is not importable in this environment
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _feature_value_strat = st.one_of(
        st.sampled_from(_EXTREME_VALUES),
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    # Feature dict keys constrained to the TRAINED columns (unseen columns are a HALT case).
    _eval_feat_names = tuple(n for n in FEATURE_NAMES if n != "n_shared_hands")
    _feature_dict_strat = st.dictionaries(
        keys=st.sampled_from(_eval_feat_names), values=_feature_value_strat
    )

    def _build_eval_fs(pair_id: str, feats: dict, n_shared: float) -> PairFeatureSet:
        merged = dict(feats)
        merged["n_shared_hands"] = float(n_shared)
        return PairFeatureSet(
            pair_id=pair_id,
            schema_version="test_schema",
            features=merged,
            threshold_version="test_thresholds",
        )

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        batch=st.lists(
            st.tuples(
                _feature_dict_strat,
                st.integers(min_value=0, max_value=100),  # n_shared_hands (crosses threshold)
            ),
            min_size=1,
            max_size=12,
        ),
    )
    def test_pu_risk_coverage_range_hypothesis(batch):
        eval_sets = [
            _build_eval_fs(f"E{i}", feats, n_shared)
            for i, (feats, n_shared) in enumerate(batch)
        ]
        _assert_coverage_and_range(_RANKER, eval_sets, _ARTIFACT, POOL_PRIOR, MIN_SHARED)
        _assert_determinism(_RANKER, eval_sets, _ARTIFACT)

else:

    def test_pu_risk_coverage_range_seeded_loop():
        """Seeded randomized-loop fallback (~180 seeds) exercising range + coverage + determinism."""
        for seed in range(180):
            rng = random.Random(seed)
            _run_one_case(rng)


# --------------------------------------------------------------------------- #
# Explicit edge cases (always run, both backends)
# --------------------------------------------------------------------------- #
def test_all_nan_inf_features_stay_in_range():
    """An eval pair of only NaN/inf feature values is tolerated and scored in [0, 1] (Req 6.3)."""
    fs = _fs(
        "P",
        n_shared_hands=float("nan"),
        value_flow_max_abs=float("inf"),
        value_flow_mean=float("-inf"),
        conf_avoidance_gap=float("nan"),
        conf_joint_isolation_rate=float("inf"),
        episodic_score=float("nan"),
    )
    scored = _RANKER.predict_risk([fs], eda_artifact=_ARTIFACT)
    assert scored[0][0] == "P"
    _assert_range(scored[0][1])
    # NaN n_shared_hands -> unscoreable -> pool prior default.
    assert scored[0][1] == POOL_PRIOR


def test_extreme_magnitudes_stay_in_range():
    """Extreme +/-1e9 magnitudes still map into [0, 1] (Req 6.3)."""
    fs = _fs(
        "P",
        n_shared_hands=1e9,
        value_flow_max_abs=1e9,
        value_flow_mean=-1e9,
        conf_avoidance_gap=1e9,
        conf_joint_isolation_rate=-1e9,
        episodic_score=1e9,
    )
    scored = _RANKER.predict_risk([fs], eda_artifact=_ARTIFACT)
    _assert_range(scored[0][1])


def test_unscoreable_few_shared_hands_gets_pool_prior():
    """Few shared hands -> unscoreable -> injected pool prior (Req 6.4)."""
    fs = _fs("P", n_shared_hands=1.0, value_flow_mean=99.0)
    scored = _RANKER.predict_risk([fs], eda_artifact=_ARTIFACT)
    assert scored[0][1] == POOL_PRIOR
    _assert_range(scored[0][1])


def test_unscoreable_all_neutral_gets_pool_prior():
    """All-neutral inputs (plenty of hands, no signal) -> unscoreable -> pool prior (Req 6.4)."""
    fs = _fs("P", n_shared_hands=100.0)
    scored = _RANKER.predict_risk([fs], eda_artifact=_ARTIFACT)
    assert scored[0][1] == POOL_PRIOR
    _assert_range(scored[0][1])


def test_coverage_no_pair_dropped_mixed_batch():
    """Mixed scoreable + unscoreable batch: one score each, in order, all in [0, 1] (Req 6.4)."""
    eval_sets = [
        _positive_fs("P0"),
        _fs("P1", n_shared_hands=1.0),  # few shared hands -> unscoreable
        _fs("P2", n_shared_hands=100.0),  # all neutral -> unscoreable
        _fs("P3", n_shared_hands=float("nan"), conf_avoidance_gap=float("inf")),
        _benign_fs("P4"),
    ]
    _assert_coverage_and_range(_RANKER, eval_sets, _ARTIFACT, POOL_PRIOR, MIN_SHARED)


def test_single_element_list_is_covered():
    """List length 1 still yields exactly one (pair_id, risk) in range (Req 6.4)."""
    _assert_coverage_and_range(_RANKER, [_positive_fs("SOLO")], _ARTIFACT, POOL_PRIOR, MIN_SHARED)


def test_injected_pool_prior_default_in_range():
    """A model fit with an injected pool prior uses it (in [0, 1]) for unscoreable pairs (Req 6.4).

    The unscoreable default is baked into the model at FIT time (from the training EDA artifact),
    so we fit a dedicated ranker with pool_prior=0.5 and confirm an unscoreable eval pair receives
    that default and stays in range.
    """
    art = _artifact(pool_prior=0.5, min_shared_hands=5)
    ranker = train_pu_ranker(_train_set(), _label_table(), eda_artifact=art, seed=101)
    fs = _fs("P", n_shared_hands=0.0)
    scored = ranker.predict_risk([fs], eda_artifact=art)
    assert scored[0][1] == 0.5
    _assert_range(scored[0][1])


def test_determinism_same_eval_list_identical_scores():
    """Same eval list scored twice -> identical scores (Req 6.5/6.7)."""
    eval_sets = [_positive_fs("E1"), _benign_fs("E2"), _fs("E3", value_flow_mean=4.0)]
    _assert_determinism(_RANKER, eval_sets, _ARTIFACT)


def test_classical_score_column_in_trained_schema():
    """Sanity: the fitted model carries the classical-score feature column (fit-once fixture)."""
    assert CLASSICAL_SCORE_COLUMN in _RANKER.columns


def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 6 / PU model] generator in use: {GENERATOR_IN_USE}")
