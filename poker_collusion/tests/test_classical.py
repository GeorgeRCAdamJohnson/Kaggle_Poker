"""Unit tests for poker_collusion.models.classical (task 11.1, Req 5.1/5.2, 6.3/6.4).

Hand-checkable example tests on tiny synthetic :class:`PairFeatureSet` vectors verifying:

* each detector's ``(score, reason)`` on a hand-computable feature vector — the reason is
  non-empty and references the driving feature/value (Req 5.1, 5.2);
* ``risk_score`` is always in ``[0, 1]`` — including extreme feature values (Req 6.3);
* EVERY input pair gets a score in the batch scorer (coverage, Req 6.4);
* an UNSCOREABLE pair (few shared hands / all-neutral features) falls back to the INJECTED pool
  prior default (Req 6.4);
* determinism — the same input yields an identical :class:`ClassicalScore`.

The Property-6 Hypothesis test (risk-score coverage and range) is task 11.2 and is intentionally
NOT written here. Fast and hermetic: no I/O, injected artifact only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import pytest

from poker_collusion.models.classical import (
    DEFAULT_POOL_PRIOR,
    ClassicalScore,
    aggression_asymmetry_softplay_test,
    collusion_table_advantage,
    isolation_pressure_ratio,
    mi_conflict_avoidance,
    score_pair,
    score_pairs,
    score_pairs_ranked,
    value_flow_graph_score,
)
from poker_collusion.types import PairFeatureSet


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fs(pair_id: str = "P", **features: float) -> PairFeatureSet:
    """Build a PairFeatureSet from explicit feature overrides (missing -> absent -> read as 0.0)."""
    feats: Dict[str, float] = {k: float(v) for k, v in features.items()}
    return PairFeatureSet(
        pair_id=pair_id,
        schema_version="test_schema",
        features=feats,
        threshold_version="test_thresholds",
    )


def _artifact(pool_prior: float, min_shared_hands: int) -> Dict[str, object]:
    """A minimal injected EDA-artifact-shaped dict (thresholds only)."""
    return {
        "thresholds": {
            "pool_prior_positive_prevalence": pool_prior,
            "min_shared_hands": min_shared_hands,
        }
    }


# A "clearly suspicious", scoreable feature vector reused across tests (plenty of shared hands,
# strong directed flow, avoidance gap, isolation, MI, and episodic concentration).
def _suspicious_fs(pair_id: str = "SUS") -> PairFeatureSet:
    return _fs(
        pair_id,
        n_shared_hands=200.0,
        value_flow_abs_p95=40.0,
        value_flow_abs_p90=20.0,
        value_flow_max_abs=40.0,
        value_flow_max_pos=40.0,
        value_flow_mean=12.3,
        value_flow_absmean=15.0,
        aggression_asymmetry_mean=0.8,
        conf_avoidance_gap=0.6,
        aggression_asymmetry_max=0.8,
        conf_joint_isolation_rate=0.5,
        isolation_max=0.7,
        mi_conflict_max=1.2,
        mi_conflict_mean=0.8,
        episodic_score=30.0,
        conf_directedness_contrast=10.0,
    )


# A clearly BENIGN scoreable pair (plenty of hands, negligible directed-flow tail) — used to
# assert the coordinated pair scores strictly higher than a benign one.
def _benign_fs(pair_id: str = "BEN") -> PairFeatureSet:
    return _fs(
        pair_id,
        n_shared_hands=200.0,
        value_flow_abs_p95=0.5,
        value_flow_abs_p90=0.2,
        value_flow_max_abs=1.0,
        value_flow_max_pos=0.5,
        value_flow_mean=0.05,
        value_flow_absmean=0.2,
        aggression_asymmetry_mean=0.05,
    )


# --------------------------------------------------------------------------- #
# Per-detector: score + non-empty reason referencing the feature
# --------------------------------------------------------------------------- #
def test_value_flow_graph_score_formula_and_reason():
    # score = value_flow_abs_p95 + 0.5 * value_flow_abs_p90 = 40 + 0.5*20 = 50
    fs = _fs(
        value_flow_abs_p95=40.0,
        value_flow_abs_p90=20.0,
        value_flow_max_abs=40.0,
        value_flow_mean=12.3,
    )
    score, reason = value_flow_graph_score(fs)
    assert score == pytest.approx(50.0)
    assert reason  # non-empty
    assert "40.0" in reason and "20.0" in reason  # p95 and p90 named
    assert "toward partner" in reason  # mean >= 0


def test_value_flow_direction_toward_self_when_mean_negative():
    _, reason = value_flow_graph_score(_fs(value_flow_abs_p95=10.0, value_flow_mean=-5.0))
    assert "toward self" in reason


def test_aggression_asymmetry_softplay_references_mean():
    # score = max(aggression_asymmetry_mean, 0) = 0.6
    fs = _fs(aggression_asymmetry_mean=0.6, conf_avoidance_gap=0.2)
    score, reason = aggression_asymmetry_softplay_test(fs)
    assert score == pytest.approx(0.6)
    assert reason and "0.600" in reason and "H15" in reason


def test_aggression_mean_clamped_at_zero():
    score, _ = aggression_asymmetry_softplay_test(_fs(aggression_asymmetry_mean=-0.4))
    assert score == pytest.approx(0.0)


def test_isolation_pressure_ratio_references_rate():
    fs = _fs(conf_joint_isolation_rate=0.5, isolation_max=0.7)
    score, reason = isolation_pressure_ratio(fs)
    # score = rate + 0.5 * iso_max = 0.5 + 0.35
    assert score == pytest.approx(0.85)
    assert reason and "0.500" in reason and "H16" in reason


def test_mi_conflict_avoidance_references_mi():
    fs = _fs(mi_conflict_max=1.2, mi_conflict_mean=0.8)
    score, reason = mi_conflict_avoidance(fs)
    # score = mi_max + 0.5 * mi_mean = 1.2 + 0.4
    assert score == pytest.approx(1.6)
    assert reason and "1.200" in reason


def test_collusion_table_advantage_backbone_composite():
    fs = _fs(value_flow_absmean=15.0, episodic_score=30.0, conf_directedness_contrast=-10.0)
    score, reason = collusion_table_advantage(fs)
    # score = absmean + 0.05*episodic + 0.05*|directedness| = 15 + 1.5 + 0.5
    assert score == pytest.approx(17.0)
    assert reason and "other_coordination" in reason
    assert "15.0" in reason and "30.0" in reason


def test_all_detectors_emit_nonempty_reason_on_neutral_input():
    fs = _fs(n_shared_hands=200.0)  # scoreable count, but neutral signals
    for fn in (
        value_flow_graph_score,
        aggression_asymmetry_softplay_test,
        isolation_pressure_ratio,
        mi_conflict_avoidance,
        collusion_table_advantage,
    ):
        score, reason = fn(fs)
        assert isinstance(score, float)
        assert isinstance(reason, str) and reason.strip()


# --------------------------------------------------------------------------- #
# risk_score range [0, 1] including extremes
# --------------------------------------------------------------------------- #
def test_risk_score_in_range_for_suspicious_pair():
    res = score_pair(_suspicious_fs(), pool_prior=0.01, min_shared_hands=1)
    assert 0.0 <= res.risk_score <= 1.0
    assert not res.unscoreable
    # a strongly suspicious pair should score well above the pool prior
    assert res.risk_score > 0.5


def test_risk_score_in_range_for_extreme_values():
    # Absurdly large feature values must NOT push risk_score out of [0, 1].
    fs = _fs(
        n_shared_hands=10_000.0,
        value_flow_max_abs=1e9,
        value_flow_mean=1e9,
        value_flow_absmean=1e9,
        conf_avoidance_gap=1e9,
        aggression_asymmetry_max=1e9,
        conf_joint_isolation_rate=1e9,
        isolation_max=1e9,
        mi_conflict_max=1e9,
        mi_conflict_mean=1e9,
        episodic_score=1e9,
        conf_directedness_contrast=1e9,
    )
    res = score_pair(fs, pool_prior=0.01, min_shared_hands=1)
    assert 0.0 <= res.risk_score <= 1.0


def test_risk_score_in_range_for_negative_extremes():
    fs = _fs(
        n_shared_hands=500.0,
        value_flow_max_abs=-1e9,
        value_flow_mean=-1e9,
        conf_avoidance_gap=-1e9,
        mi_conflict_max=-1e9,
    )
    res = score_pair(fs, pool_prior=0.02, min_shared_hands=1)
    assert 0.0 <= res.risk_score <= 1.0


# --------------------------------------------------------------------------- #
# Coverage: every input pair gets a score
# --------------------------------------------------------------------------- #
def test_batch_scores_every_pair():
    feature_sets = [
        _suspicious_fs("A"),
        _fs("B", n_shared_hands=0.0),                       # unscoreable (few hands)
        _fs("C", n_shared_hands=200.0),                     # scoreable but neutral -> unscoreable
        _fs("D", n_shared_hands=50.0, value_flow_max_abs=20.0, value_flow_mean=5.0),
    ]
    results = score_pairs(feature_sets, pool_prior=0.03, min_shared_hands=5)
    assert len(results) == len(feature_sets)
    assert [r.pair_id for r in results] == ["A", "B", "C", "D"]
    for r in results:
        assert 0.0 <= r.risk_score <= 1.0
        assert isinstance(r, ClassicalScore)


# --------------------------------------------------------------------------- #
# Unscoreable -> injected pool-prior default
# --------------------------------------------------------------------------- #
def test_unscoreable_few_shared_hands_defaults_to_injected_pool_prior():
    artifact = _artifact(pool_prior=0.037, min_shared_hands=10)
    # strong signals but only 3 shared hands (< min 10) => unscoreable => pool prior
    fs = _fs(
        "few",
        n_shared_hands=3.0,
        value_flow_max_abs=40.0,
        value_flow_mean=12.0,
    )
    res = score_pair(fs, eda_artifact=artifact)
    assert res.unscoreable
    assert res.risk_score == pytest.approx(0.037)
    assert res.pool_prior == pytest.approx(0.037)
    assert any("UNSCOREABLE" in r for r in res.reasons)


def test_unscoreable_neutral_features_defaults_to_injected_pool_prior():
    artifact = _artifact(pool_prior=0.02, min_shared_hands=1)
    fs = _fs("neutral", n_shared_hands=500.0)  # plenty of hands but every signal neutral
    res = score_pair(fs, eda_artifact=artifact)
    assert res.unscoreable
    assert res.risk_score == pytest.approx(0.02)


def test_unscoreable_falls_back_to_default_constant_without_artifact():
    fs = _fs("np", n_shared_hands=0.0)
    res = score_pair(fs)  # no artifact, no override
    assert res.unscoreable
    assert res.risk_score == pytest.approx(DEFAULT_POOL_PRIOR)


def test_explicit_pool_prior_override_beats_artifact():
    artifact = _artifact(pool_prior=0.5, min_shared_hands=1)
    fs = _fs("neutral", n_shared_hands=500.0)
    res = score_pair(fs, eda_artifact=artifact, pool_prior=0.09)
    assert res.risk_score == pytest.approx(0.09)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_input_same_output():
    fs = _suspicious_fs("det")
    a = score_pair(fs, pool_prior=0.01, min_shared_hands=1)
    b = score_pair(fs, pool_prior=0.01, min_shared_hands=1)
    assert a == b
    assert a.risk_score == b.risk_score
    assert a.reasons == b.reasons
    assert a.detector_scores == b.detector_scores


def test_determinism_independent_of_feature_dict_insertion_order():
    base = _suspicious_fs("order")
    # Rebuild with a reversed-insertion-order features dict; risk must be identical.
    reordered = replace(base, features={k: base.features[k] for k in reversed(list(base.features))})
    a = score_pair(base, pool_prior=0.01, min_shared_hands=1)
    b = score_pair(reordered, pool_prior=0.01, min_shared_hands=1)
    assert a.risk_score == b.risk_score
    assert a.detector_scores == b.detector_scores


# --------------------------------------------------------------------------- #
# Discrimination + spread (the Phase-1 anti-saturation fix)
# --------------------------------------------------------------------------- #
def test_coordinated_pair_scores_higher_than_benign():
    """A clearly coordinated pair must score strictly above a benign one (both scoreable)."""
    sus = score_pair(_suspicious_fs("SUS"), pool_prior=0.01, min_shared_hands=1)
    ben = score_pair(_benign_fs("BEN"), pool_prior=0.01, min_shared_hands=1)
    assert not sus.unscoreable and not ben.unscoreable
    assert sus.risk_score > ben.risk_score
    # And the monotone raw combined score orders them the same way (drives Pair AP ranking).
    assert sus.raw_combined_score > ben.raw_combined_score


def test_score_pair_does_not_saturate_near_one():
    """A strong-but-finite pair stays well below 1.0 (no logistic saturation clump)."""
    res = score_pair(_suspicious_fs(), pool_prior=0.01, min_shared_hands=1)
    assert res.risk_score < 0.95  # non-saturating rational squash, not the old ~0.955 clump


def test_score_pairs_ranked_preserves_ordering_of_score_pair():
    """Ranked risk is a MONOTONE transform of the per-pair raw score => identical pair ORDER."""
    feature_sets = [
        _benign_fs("B0"),
        _suspicious_fs("S0"),
        _fs("MID", n_shared_hands=200.0, value_flow_abs_p95=8.0, value_flow_abs_p90=4.0,
            value_flow_absmean=3.0, aggression_asymmetry_mean=0.4),
    ]
    ranked = score_pairs_ranked(feature_sets, pool_prior=0.01, min_shared_hands=1)
    by_id = {r.pair_id: r for r in ranked}
    # Raw combined order: S0 (strong) > MID > B0 (benign) => ranked risk must follow.
    assert by_id["S0"].risk_score > by_id["MID"].risk_score > by_id["B0"].risk_score


def test_score_pairs_ranked_spreads_and_keeps_high_risk_rare():
    """Ranked risks fill (0,1) with only a small high-risk fraction (no saturation clump)."""
    import random

    rng = random.Random(7)
    feature_sets = []
    for i in range(400):
        # random but scoreable pairs with varied value-flow tails
        p95 = 10.0 ** rng.uniform(-2.0, 2.0)
        feature_sets.append(
            _fs(
                f"P{i:04d}",
                n_shared_hands=200.0,
                value_flow_abs_p95=p95,
                value_flow_abs_p90=p95 * 0.5,
                value_flow_absmean=p95 * 0.2,
                aggression_asymmetry_mean=rng.random(),
            )
        )
    ranked = score_pairs_ranked(feature_sets, pool_prior=0.002, min_shared_hands=1)
    risks = [r.risk_score for r in ranked]
    # Every pair scored, all in range.
    assert len(risks) == len(feature_sets)
    assert all(0.0 <= r <= 1.0 for r in risks)
    # Well-spread ranking: the risks are (near-)distinct across the population — NOT the old
    # saturation clump where ~half the pairs shared one value. (Rank percentiles are distinct;
    # the power curve is strictly monotone, so distinct raw scores => distinct risks.)
    assert len(set(risks)) >= 0.95 * len(risks)
    # No high-end tie clump: only a small fraction sits above 0.5, and high risk is rare under the
    # k=6 spreading curve (~1.7% > 0.9 by construction, independent of the score distribution).
    frac_gt5 = sum(1 for r in risks if r > 0.5) / len(risks)
    frac_hi = sum(1 for r in risks if r > 0.9) / len(risks)
    assert frac_gt5 < 0.20
    assert frac_hi < 0.05


def test_score_pairs_ranked_unscoreable_below_scoreable():
    """Unscoreable pairs keep the pool prior and rank BELOW every scoreable pair."""
    feature_sets = [
        _suspicious_fs("SUS"),
        _fs("NOHANDS", n_shared_hands=0.0, value_flow_abs_p95=99.0),  # few hands -> unscoreable
        _fs("NEUTRAL", n_shared_hands=200.0),  # all-neutral -> unscoreable
    ]
    ranked = score_pairs_ranked(feature_sets, pool_prior=0.01, min_shared_hands=5)
    by_id = {r.pair_id: r for r in ranked}
    assert by_id["NOHANDS"].unscoreable and by_id["NOHANDS"].risk_score == pytest.approx(0.01)
    assert by_id["NEUTRAL"].unscoreable and by_id["NEUTRAL"].risk_score == pytest.approx(0.01)
    assert by_id["SUS"].risk_score > by_id["NOHANDS"].risk_score


def test_score_pairs_ranked_deterministic_and_order_independent():
    """Same inputs (any order) -> same {pair_id -> ranked risk}."""
    fss = [_suspicious_fs("A"), _benign_fs("B"),
           _fs("C", n_shared_hands=200.0, value_flow_abs_p95=5.0)]
    a = {r.pair_id: r.risk_score for r in score_pairs_ranked(fss, pool_prior=0.01, min_shared_hands=1)}
    b = {r.pair_id: r.risk_score
         for r in score_pairs_ranked(list(reversed(fss)), pool_prior=0.01, min_shared_hands=1)}
    assert a == b
