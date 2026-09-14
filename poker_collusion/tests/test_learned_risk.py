"""Hermetic unit tests for poker_collusion.models.learned_risk.LearnedRiskModel.

Tiny synthetic :class:`PairFeatureSet`\\ s + a small ``{pair_id -> label_status}`` map (the same
shape as ``test_pu_ranker.py``), verifying the learned risk combiner:

* every ``risk_score`` is in ``[0, 1]`` and every eval pair is scored (coverage);
* determinism — two fits on identical data produce identical per-pair scores;
* trained on synthetic positives/negatives it separates a clearly-coordinated pair ABOVE a benign
  one (the learned logistic actually learns the separation);
* with the pool-prior clamp OFF (the DEFAULT for the learned head) a low-shared-hands pair still
  gets a feature-derived learned score — NOT the constant pool prior — and that score is still in
  ``[0, 1]`` and deterministic across fits;
* with the clamp ON (``apply_unscoreable_prior=True``, opt-in) the old PURanker-style
  unscoreable->pool-prior fallback still works (explicit coverage of the option);
* HALT (raise, no partial state) on empty input and on a single-class label set;
* save/load round-trips reproduce identical scores.

Never reads the real ``data/poker`` files. Fast and hermetic: injected artifact only, artifacts
written to tmp_path. No property-based testing here.
"""

from __future__ import annotations

from typing import Dict

import pytest

from poker_collusion.exceptions import SchemaError
from poker_collusion.models.learned_risk import (
    LearnedRiskModel,
    train_learned_risk,
)
from poker_collusion.types import PairFeatureSet


# --------------------------------------------------------------------------- #
# Helpers (mirror test_pu_ranker.py's tiny synthetic pattern)
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


def _positive_fs(pair_id: str, jitter: float = 0.0) -> PairFeatureSet:
    """A strongly coordination-leaning feature vector (with a little per-pair jitter)."""
    return _fs(
        pair_id,
        n_shared_hands=20.0,
        value_flow_max_abs=30.0 + jitter,
        value_flow_mean=20.0 + jitter,
        conf_avoidance_gap=0.6,
        conf_joint_isolation_rate=0.5,
        episodic_score=15.0 + jitter,
    )


def _negative_fs(pair_id: str, jitter: float = 0.0) -> PairFeatureSet:
    """A benign feature vector (near-neutral signals, still enough shared hands to be scoreable)."""
    return _fs(
        pair_id,
        n_shared_hands=20.0,
        value_flow_max_abs=1.0 + jitter,
        value_flow_mean=0.0 + jitter,
        conf_avoidance_gap=0.0,
        conf_joint_isolation_rate=0.0,
        episodic_score=0.5 + jitter,
    )


def _training_set():
    """Several confirmed positives + negatives (multiple examples so the logistic fits cleanly)."""
    fs_list = []
    status: Dict[str, str] = {}
    for i in range(6):
        pid = f"POS{i}"
        fs_list.append(_positive_fs(pid, jitter=0.1 * i))
        status[pid] = "confirmed_target"
    for i in range(6):
        pid = f"NEG{i}"
        fs_list.append(_negative_fs(pid, jitter=0.1 * i))
        status[pid] = "confirmed_non_target"
    return fs_list, status


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_risk_scores_in_unit_interval_and_full_coverage():
    art = _artifact()
    train_fs, status = _training_set()
    model = train_learned_risk(train_fs, status, eda_artifact=art)

    eval_fs = [_positive_fs("EP"), _negative_fs("EN"), _fs("EU", n_shared_hands=0.0)]
    scored = model.score_pairs(eval_fs, eda_artifact=art)

    assert set(scored.keys()) == {"EP", "EN", "EU"}  # coverage: one score per pair
    for v in scored.values():
        assert 0.0 <= v <= 1.0


def test_deterministic_two_fits_same_scores():
    art = _artifact()
    train_fs, status = _training_set()
    eval_fs = [_positive_fs("EP"), _negative_fs("EN")]

    a = train_learned_risk(train_fs, status, eda_artifact=art).score_pairs(eval_fs, eda_artifact=art)
    b = train_learned_risk(train_fs, status, eda_artifact=art).score_pairs(eval_fs, eda_artifact=art)
    assert a == b


def test_separates_coordinated_above_benign():
    art = _artifact()
    train_fs, status = _training_set()
    model = train_learned_risk(train_fs, status, eda_artifact=art)

    scored = model.score_pairs([_positive_fs("EP"), _negative_fs("EN")], eda_artifact=art)
    assert scored["EP"] > scored["EN"], (
        f"learned combiner failed to rank the coordinated pair above the benign one: {scored}"
    )


def test_default_clamp_off_low_shared_hands_gets_feature_score_not_prior():
    """DEFAULT (apply_unscoreable_prior=False): a low-shared-hands, high-signal pair is scored from
    its calibrated features, NOT clamped to the constant pool prior. This is the fix: the old clamp
    collapsed ~30% of confirmed positives to one constant and destroyed the learned ranking."""
    prior = 0.033
    art = _artifact(pool_prior=prior, min_shared_hands=5)
    train_fs, status = _training_set()
    model = train_learned_risk(train_fs, status, eda_artifact=art)  # default: clamp OFF
    assert model._apply_unscoreable_prior is False

    # Shares fewer than min_shared_hands but carries strong coordination signal.
    low_shared_strong = _fs(
        "EU",
        n_shared_hands=1.0,
        value_flow_max_abs=30.0,
        value_flow_mean=20.0,
        conf_avoidance_gap=0.6,
        conf_joint_isolation_rate=0.5,
        episodic_score=15.0,
    )
    scored = model.score_pairs([low_shared_strong], eda_artifact=art)
    # NOT the constant prior — a genuine feature-derived score.
    assert scored["EU"] != pytest.approx(prior)
    # Still a valid, high-signal risk in [0, 1].
    assert 0.0 <= scored["EU"] <= 1.0
    assert scored["EU"] > 0.5, f"strong-signal low-shared pair should score high, got {scored['EU']}"


def test_default_clamp_off_low_shared_hands_score_is_deterministic():
    """The feature-derived score for a low-shared-hands pair is identical across two fits."""
    art = _artifact(pool_prior=0.033, min_shared_hands=5)
    train_fs, status = _training_set()
    low_shared = _fs("EU", n_shared_hands=1.0, value_flow_max_abs=30.0, value_flow_mean=20.0)

    a = train_learned_risk(train_fs, status, eda_artifact=art).score_pairs([low_shared], eda_artifact=art)
    b = train_learned_risk(train_fs, status, eda_artifact=art).score_pairs([low_shared], eda_artifact=art)
    assert a == b
    assert 0.0 <= a["EU"] <= 1.0


def test_flag_on_unscoreable_pair_gets_pool_prior():
    """OPT-IN (apply_unscoreable_prior=True): the old PURanker-style clamp still works — a
    too-few-shared-hands pair falls back to the injected pool-prior default."""
    prior = 0.033
    art = _artifact(pool_prior=prior, min_shared_hands=2)
    train_fs, status = _training_set()
    model = train_learned_risk(
        train_fs, status, eda_artifact=art, apply_unscoreable_prior=True
    )
    assert model._apply_unscoreable_prior is True

    # Too few shared hands -> unscoreable -> pool prior (clamp ON).
    too_few = _fs("EU", n_shared_hands=1.0, value_flow_max_abs=30.0, value_flow_mean=20.0)
    scored = model.score_pairs([too_few], eda_artifact=art)
    assert scored["EU"] == pytest.approx(prior)


def test_halt_on_empty_training_input():
    with pytest.raises(SchemaError):
        LearnedRiskModel().fit([], {}, eda_artifact=_artifact())


def test_halt_on_single_class_labels():
    art = _artifact()
    # Only positives -> cannot learn a separation -> HALT (no partial state).
    fs_list = [_positive_fs(f"P{i}") for i in range(4)]
    status = {f"P{i}": "confirmed_target" for i in range(4)}
    model = LearnedRiskModel()
    with pytest.raises(SchemaError):
        model.fit(fs_list, status, eda_artifact=art)
    assert model.is_fitted is False  # no partial state


def test_halt_on_empty_eval_input():
    art = _artifact()
    train_fs, status = _training_set()
    model = train_learned_risk(train_fs, status, eda_artifact=art)
    with pytest.raises(SchemaError):
        model.predict_risk([], eda_artifact=art)


def test_save_load_roundtrip_reproduces_scores(tmp_path):
    art = _artifact()
    train_fs, status = _training_set()
    model = train_learned_risk(train_fs, status, eda_artifact=art)
    eval_fs = [_positive_fs("EP"), _negative_fs("EN")]
    before = model.score_pairs(eval_fs, eda_artifact=art)

    path = model.save(tmp_path / "learned_risk.joblib")
    loaded = LearnedRiskModel.load(path)
    after = loaded.score_pairs(eval_fs, eda_artifact=art)
    assert before == after
