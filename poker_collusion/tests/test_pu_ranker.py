"""Unit tests for poker_collusion.models.pu_ranker (task 16.1, Req 6.1-6.8).

Hand-checkable example tests on tiny synthetic :class:`PairFeatureSet`\\ s plus a small PU
label map / :class:`LabelTable`, verifying:

* every risk_score is in ``[0, 1]`` and every eval pair is scored (coverage; Req 6.3/6.4);
* an UNSCOREABLE pair (too few shared hands) falls back to the injected pool-prior default (Req 6.4);
* determinism — same seed/input -> identical scores, and shuffled input rows -> identical
  per-pair scores (Req 6.5/6.7);
* HALT (raise, no partial output) on empty/missing required input and on a feature-schema
  mismatch at predict time (Req 6.6/6.8);
* a positive-leaning feature vector scores higher than a benign one (sanity);
* save/load round-trips reproduce identical scores (Req 6.7).

The Property-6/7 Hypothesis tests are tasks 16.2/16.3 and are intentionally NOT written here.
Fast and hermetic: tiny synthetic inputs, injected artifact only, artifacts written to tmp_path.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import pytest

from poker_collusion.exceptions import SchemaError
from poker_collusion.models.pu_ranker import (
    CLASSICAL_SCORE_COLUMN,
    PURanker,
    train_pu_ranker,
)
from poker_collusion.types import LabelTable, PairFeatureSet


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
POOL_PRIOR = 0.02
MIN_SHARED = 2


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
    """Tiny synthetic PairFeatureSet with a fixed, shared feature schema."""
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
    """A strongly coordination-leaning feature vector."""
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
    """A benign feature vector (small, near-neutral but scoreable)."""
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
    """A small dev set: 2 positives, 2 confirmed negatives, 3 unknowns."""
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


def _fit_ranker(seed: Optional[int] = None) -> PURanker:
    art = _artifact()
    return train_pu_ranker(
        _train_set(),
        _label_table(),
        eda_artifact=art,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Coverage + range (Req 6.3, 6.4)
# --------------------------------------------------------------------------- #
def test_risk_in_range_and_full_coverage():
    ranker = _fit_ranker()
    art = _artifact()
    eval_sets = [_positive_fs("E1"), _benign_fs("E2"), _fs("E3", value_flow_mean=4.0)]
    scored = ranker.predict_risk(eval_sets, eda_artifact=art)

    assert len(scored) == len(eval_sets)  # coverage: one score per pair
    assert [pid for pid, _ in scored] == ["E1", "E2", "E3"]  # order preserved
    for _, risk in scored:
        assert 0.0 <= risk <= 1.0  # Req 6.3


def test_unscoreable_pair_gets_pool_prior_default():
    ranker = _fit_ranker()
    art = _artifact(pool_prior=0.02, min_shared_hands=2)
    # Too few shared hands => unscoreable => defined default = pool prior.
    unscoreable = _fs("U", n_shared_hands=1.0, value_flow_mean=99.0)
    scored = ranker.predict_risk([unscoreable], eda_artifact=art)
    assert scored[0][0] == "U"
    assert scored[0][1] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# Determinism + order-independence (Req 6.5, 6.7)
# --------------------------------------------------------------------------- #
def test_determinism_same_seed_identical_scores():
    art = _artifact()
    eval_sets = [_positive_fs("E1"), _benign_fs("E2")]
    a = _fit_ranker(seed=101).score_pairs(eval_sets, eda_artifact=art)
    b = _fit_ranker(seed=101).score_pairs(eval_sets, eda_artifact=art)
    assert a == b


def test_ranking_independent_of_input_row_order():
    art = _artifact()
    ranker = _fit_ranker()
    eval_sets = [_positive_fs("E1"), _benign_fs("E2"), _fs("E3", value_flow_mean=4.0), _benign_fs("E4")]
    ref = ranker.score_pairs(eval_sets, eda_artifact=art)

    shuffled = list(eval_sets)
    rng = random.Random(7)
    rng.shuffle(shuffled)
    got = ranker.score_pairs(shuffled, eda_artifact=art)
    assert got == ref  # per-pair scores identical regardless of row order


def test_training_row_order_does_not_change_scores():
    art = _artifact()
    eval_sets = [_positive_fs("E1"), _benign_fs("E2")]

    train_a = _train_set()
    ranker_a = train_pu_ranker(train_a, _label_table(), eda_artifact=art, seed=101)

    train_b = list(train_a)
    random.Random(3).shuffle(train_b)
    ranker_b = train_pu_ranker(train_b, _label_table(), eda_artifact=art, seed=101)

    assert ranker_a.score_pairs(eval_sets, eda_artifact=art) == ranker_b.score_pairs(
        eval_sets, eda_artifact=art
    )


# --------------------------------------------------------------------------- #
# Sanity: positive-leaning scores higher than benign (Req 6.1/6.2 intent)
# --------------------------------------------------------------------------- #
def test_positive_scores_higher_than_benign():
    ranker = _fit_ranker()
    art = _artifact()
    scored = ranker.score_pairs([_positive_fs("P"), _benign_fs("B")], eda_artifact=art)
    assert scored["P"] > scored["B"]


# --------------------------------------------------------------------------- #
# HALT on missing/empty required input (Req 6.6, 6.8) — no partial output
# --------------------------------------------------------------------------- #
def test_fit_halts_on_empty_training_set():
    with pytest.raises(SchemaError):
        train_pu_ranker([], _label_table(), eda_artifact=_artifact())


def test_fit_halts_when_no_positives():
    labels = LabelTable(trusted_positive={}, confirmed_negative={"NEG1", "NEG2"})
    with pytest.raises(SchemaError):
        train_pu_ranker(_train_set(), labels, eda_artifact=_artifact())


def test_predict_halts_on_empty_eval_set():
    ranker = _fit_ranker()
    with pytest.raises(SchemaError):
        ranker.predict_risk([], eda_artifact=_artifact())


def test_predict_halts_on_feature_schema_mismatch():
    ranker = _fit_ranker()
    art = _artifact()
    bad = _fs("X")
    bad.features["a_brand_new_feature"] = 1.0  # unseen column at predict time
    with pytest.raises(SchemaError):
        ranker.predict_risk([bad], eda_artifact=art)


def test_predict_before_fit_halts():
    ranker = PURanker()
    with pytest.raises(SchemaError):
        ranker.predict_risk([_benign_fs("B")], eda_artifact=_artifact())


def test_save_before_fit_halts(tmp_path):
    ranker = PURanker()
    with pytest.raises(SchemaError):
        ranker.save(tmp_path / "model.joblib")


# --------------------------------------------------------------------------- #
# Persistence round-trip (Req 6.7)
# --------------------------------------------------------------------------- #
def test_save_load_round_trip_reproduces_scores(tmp_path):
    ranker = _fit_ranker()
    art = _artifact()
    eval_sets = [_positive_fs("E1"), _benign_fs("E2"), _fs("E3", value_flow_mean=4.0)]
    before = ranker.score_pairs(eval_sets, eda_artifact=art)

    path = tmp_path / "pu_model.joblib"
    ranker.save(path)
    assert path.exists()

    loaded = PURanker.load(path)
    after = loaded.score_pairs(eval_sets, eda_artifact=art)
    assert after == before

    # persisted schema/columns are carried through
    assert loaded.columns == ranker.columns
    assert CLASSICAL_SCORE_COLUMN in loaded.columns


def test_load_missing_file_halts(tmp_path):
    with pytest.raises(SchemaError):
        PURanker.load(tmp_path / "does_not_exist.joblib")
