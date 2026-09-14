# Feature: poker-collusion-detection, Property 7: Ranking and evidence ordering are independent of input row order
"""Property-based test for the PU-aware risk model's ORDER-INDEPENDENCE (Property 7, Req 6.5).

The :class:`PURanker` scores every pair from its own feature row, so neither the per-pair risk
scores nor the induced ranking may depend on the order the eval pairs are presented in. This test
asserts, for a fitted model and a random eval list:

* PER-PAIR INVARIANCE (Req 6.5): scoring the list and scoring a SHUFFLED copy of the SAME list
  yields IDENTICAL per-pair risk scores. We build a ``{pair_id -> risk}`` map from each ordering
  and assert the two maps are equal.
* RANKING INVARIANCE (Req 6.5): the induced ranking — pairs sorted by ``risk`` DESCENDING with
  ties broken by ``pair_id`` ASCENDING (the competition's deterministic ``pair_id`` tie-break) —
  is IDENTICAL regardless of input row order.

Since the model scores per row, both must hold EXACTLY. Duplicate-risk cases are included to
exercise the deterministic tie-break (Requirement 8.5 evidence ordering is covered by the
Evidence_Retriever property tests; here we focus on the risk RANKING order-independence).

Runtime discipline
-------------------
The :class:`PURanker` is fit ONCE per test on a small fixed synthetic training set (7 pairs from
the ``test_pu_ranker`` conventions); the generated cases only re-order/vary EVAL inputs. sklearn is
never re-fit per example.

Generator backend
-----------------
Prefers the ``hypothesis`` library, guarded via ``importlib.util.find_spec``. When hypothesis is
not importable (the case here), the test falls back to a seeded randomized loop (~180 seeds). The
active backend is recorded in ``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.

Fast + hermetic: no I/O, injected EDA artifact only, model fit once.
"""

from __future__ import annotations

import importlib.util
import math
import random
from typing import Dict, List, Optional, Tuple

from poker_collusion.models.pu_ranker import PURanker, train_pu_ranker
from poker_collusion.types import LabelTable, PairFeatureSet

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

# --------------------------------------------------------------------------- #
# Shared tiny training fixture (test_pu_ranker conventions)
# --------------------------------------------------------------------------- #
POOL_PRIOR = 0.02
MIN_SHARED = 2

FEATURE_NAMES: tuple[str, ...] = (
    "n_shared_hands",
    "value_flow_max_abs",
    "value_flow_mean",
    "conf_avoidance_gap",
    "conf_joint_isolation_rate",
    "episodic_score",
)

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


_RANKER = _fit_ranker()
_ARTIFACT = _artifact()


# --------------------------------------------------------------------------- #
# The competition ranking: sort by risk DESC, tie-break by pair_id ASC.
# --------------------------------------------------------------------------- #
def _ranking(scores: Dict[str, float]) -> List[str]:
    """Induced ranking: risk descending, ties broken by ascending pair_id (competition rule)."""
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


# --------------------------------------------------------------------------- #
# Random EVAL input construction (shared by both backends)
# --------------------------------------------------------------------------- #
def _rand_feature_value(rng: random.Random) -> float:
    if rng.random() < 0.35:
        return rng.choice(_EXTREME_VALUES)
    mag = 10.0 ** rng.uniform(-3.0, 9.0)
    return rng.choice((-1.0, 1.0)) * mag * rng.random()


def _rand_eval_fs(rng: random.Random, pair_id: str) -> PairFeatureSet:
    """Build an eval PairFeatureSet over the TRAINED columns with random (possibly extreme) values."""
    feats: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name == "n_shared_hands":
            continue
        if rng.random() < 0.1:
            continue
        feats[name] = _rand_feature_value(rng)
    feats["n_shared_hands"] = float(rng.randint(0, 100))
    return PairFeatureSet(
        pair_id=pair_id,
        schema_version="test_schema",
        features=feats,
        threshold_version="test_thresholds",
    )


def _rand_eval_list(rng: random.Random) -> List[PairFeatureSet]:
    """A random eval list with UNIQUE pair_ids. Sometimes injects duplicate-risk archetypes.

    To exercise the deterministic tie-break, a fraction of pairs reuse one of a few fixed feature
    archetypes (which therefore produce identical risk scores), while still carrying distinct
    pair_ids.
    """
    n = rng.randint(1, 12)
    archetypes = [_positive_fs, _benign_fs, lambda pid: _fs(pid, n_shared_hands=1.0)]
    out: List[PairFeatureSet] = []
    used: set = set()
    for i in range(n):
        pid = f"E{i:03d}"
        while pid in used:
            pid = f"E{i:03d}_{rng.randint(0, 99999)}"
        used.add(pid)
        if rng.random() < 0.5:
            # Duplicate-risk archetype (identical features -> identical risk -> tie).
            out.append(rng.choice(archetypes)(pid))
        else:
            out.append(_rand_eval_fs(rng, pid))
    return out


# --------------------------------------------------------------------------- #
# Core assertion (shared by both generator backends)
# --------------------------------------------------------------------------- #
def _assert_order_independent(ranker: PURanker, eval_sets: List[PairFeatureSet]) -> None:
    """Per-pair scores AND the induced ranking are identical under input reordering (Req 6.5)."""
    ref_map = ranker.score_pairs(eval_sets, eda_artifact=_ARTIFACT)

    # Shuffle a copy of the SAME list (distinct pair_ids are assumed; the map is keyed by pair_id).
    shuffled = list(eval_sets)
    random.Random(hash(tuple(fs.pair_id for fs in eval_sets)) & 0xFFFFFFFF).shuffle(shuffled)
    got_map = ranker.score_pairs(shuffled, eda_artifact=_ARTIFACT)

    # PER-PAIR INVARIANCE: {pair_id -> risk} maps are equal.
    assert got_map == ref_map, (
        f"per-pair risk changed under reorder:\n  ref={ref_map!r}\n  got={got_map!r}"
    )

    # RANKING INVARIANCE: risk desc, pair_id asc tie-break identical regardless of row order.
    assert _ranking(got_map) == _ranking(ref_map), (
        f"ranking changed under reorder:\n  ref={_ranking(ref_map)!r}\n  got={_ranking(got_map)!r}"
    )


def _run_one_case(rng: random.Random) -> None:
    """One property trial: build a random eval list, assert order-independence."""
    eval_sets = _rand_eval_list(rng)
    _assert_order_independent(_RANKER, eval_sets)


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
    _eval_feat_names = tuple(n for n in FEATURE_NAMES if n != "n_shared_hands")
    _feature_dict_strat = st.dictionaries(
        keys=st.sampled_from(_eval_feat_names), values=_feature_value_strat
    )

    def _build_eval_fs(pair_id: str, feats: dict, n_shared: int) -> PairFeatureSet:
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
            st.tuples(_feature_dict_strat, st.integers(min_value=0, max_value=100)),
            min_size=1,
            max_size=12,
        ),
    )
    def test_pu_ranking_order_independent_hypothesis(batch):
        # Distinct pair_ids (map keyed by pair_id requires uniqueness).
        eval_sets = [
            _build_eval_fs(f"E{i:03d}", feats, n_shared)
            for i, (feats, n_shared) in enumerate(batch)
        ]
        _assert_order_independent(_RANKER, eval_sets)

else:

    def test_pu_ranking_order_independent_seeded_loop():
        """Seeded randomized-loop fallback (~180 seeds) exercising order-independence + tie-break."""
        for seed in range(180):
            rng = random.Random(seed)
            _run_one_case(rng)


# --------------------------------------------------------------------------- #
# Explicit edge cases (always run, both backends)
# --------------------------------------------------------------------------- #
def test_shuffle_preserves_per_pair_scores():
    """A shuffled eval list yields identical per-pair scores (Req 6.5)."""
    eval_sets = [
        _positive_fs("E1"),
        _benign_fs("E2"),
        _fs("E3", value_flow_mean=4.0),
        _benign_fs("E4"),
        _fs("E5", n_shared_hands=1.0),  # unscoreable -> pool prior
    ]
    _assert_order_independent(_RANKER, eval_sets)


def test_duplicate_risk_tie_break_is_pair_id_ascending():
    """Pairs with identical features (=> identical risk) rank by ascending pair_id, order-free."""
    # Three identical-feature pairs => a 3-way tie; tie-break must be pair_id ascending.
    eval_sets = [_benign_fs("B_c"), _benign_fs("B_a"), _benign_fs("B_b")]
    fwd = _RANKER.score_pairs(eval_sets, eda_artifact=_ARTIFACT)
    rev = _RANKER.score_pairs(list(reversed(eval_sets)), eda_artifact=_ARTIFACT)
    assert fwd == rev
    # All three share the same risk => the tied block is ordered by ascending pair_id.
    tied = [pid for pid in _ranking(fwd) if pid.startswith("B_")]
    assert tied == sorted(tied), f"tie-break not pair_id-ascending: {tied!r}"
    assert _ranking(fwd) == _ranking(rev)


def test_all_unscoreable_tie_ranks_by_pair_id():
    """All-unscoreable pairs share the pool prior -> a full tie ordered by ascending pair_id."""
    eval_sets = [
        _fs("Z", n_shared_hands=0.0),
        _fs("A", n_shared_hands=0.0),
        _fs("M", n_shared_hands=100.0),  # all-neutral -> also pool prior
    ]
    scores = _RANKER.score_pairs(eval_sets, eda_artifact=_ARTIFACT)
    assert set(scores.values()) == {POOL_PRIOR}
    assert _ranking(scores) == ["A", "M", "Z"]
    _assert_order_independent(_RANKER, eval_sets)


def test_reverse_order_matches_forward_ranking():
    """Reversing the input list changes nothing about the ranking (Req 6.5)."""
    eval_sets = [
        _positive_fs("P1"),
        _benign_fs("P2"),
        _fs("P3", value_flow_mean=8.0, value_flow_max_abs=12.0),
        _fs("P4", n_shared_hands=1.0),
    ]
    fwd = _RANKER.score_pairs(eval_sets, eda_artifact=_ARTIFACT)
    rev = _RANKER.score_pairs(list(reversed(eval_sets)), eda_artifact=_ARTIFACT)
    assert _ranking(fwd) == _ranking(rev)
    assert fwd == rev


def test_single_pair_ranking_trivial():
    """A single eval pair is trivially order-independent (Req 6.5)."""
    _assert_order_independent(_RANKER, [_positive_fs("SOLO")])


def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 7 / PU ranking order] generator in use: {GENERATOR_IN_USE}")
