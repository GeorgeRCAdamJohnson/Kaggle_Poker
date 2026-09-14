"""Correctness unit tests for poker_collusion.features.aggregation (task 10.1, Requirement 4).

Hand-computable example tests on tiny synthetic :class:`HandSignal` lists verifying:

* peak-preserving stats (max / p90 / p95 / mean) on inputs whose stats we can compute by hand
  (Req 4.1);
* NOT_APPLICABLE hands are EXCLUDED from a signal's stats, never coerced to 0 (Req 3.5 / 4.1);
* the burst / temporal-concentration feature increases (or stays equal) when the same flagged
  mass is concentrated vs spread — the seed of Property 4 (Req 4.2, 4.3);
* empirical-Bayes shrinkage pulls a low-count pair toward the prior and a high-count pair toward
  its empirical rate (Req 4.4);
* determinism (identical inputs -> identical output).

The exhaustive Hypothesis property test for Property 4 is task 10.2 and is intentionally NOT
written here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from poker_collusion.features.aggregation import (
    BURST_GAIN,
    DEFAULT_PRIOR_STRENGTH,
    aggregate_pair,
    empirical_bayes_shrink,
    gap_gini,
    longest_flagged_burst,
    peak_preserving_stats,
    prior_from_artifact,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal, Sentinel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sig(
    hand_id: int,
    *,
    value_flow: float = 0.0,
    aggression_asymmetry: Union[float, Sentinel] = 0.0,
    isolation: Union[float, Sentinel] = 0.0,
    mi_conflict: float = 0.0,
    flagged: bool = False,
    phase: str = "evaluation",
    pair_id: str = "P",
) -> HandSignal:
    """Build a HandSignal; ``flagged`` sets the directed_transfer flag True."""
    flags: Dict[str, bool] = {
        "directed_transfer": bool(flagged),
        "soft_play": False,
        "coordinated_isolation": False,
    }
    return HandSignal(
        hand_id=hand_id,
        pair_id=pair_id,
        phase=phase,
        value_flow=value_flow,
        aggression_asymmetry=aggression_asymmetry,
        isolation=isolation,
        mi_conflict=mi_conflict,
        behavior_action_flags=flags,
    )


# --------------------------------------------------------------------------- #
# Peak-preserving statistics (Req 4.1)
# --------------------------------------------------------------------------- #
def test_peak_preserving_stats_hand_computable():
    # values 0..10 (11 points). numpy-default percentiles:
    #   p90 -> rank 0.9*10 = 9.0 -> value 9.0 ; p95 -> rank 9.5 -> 9.5 ; max 10 ; mean 5.
    vals = [float(i) for i in range(11)]
    stats = peak_preserving_stats(vals)
    assert stats["max"] == 10.0
    assert stats["mean"] == 5.0
    assert stats["p90"] == 9.0
    assert stats["p95"] == 9.5


def test_peak_preserving_stats_empty_is_neutral_not_error():
    stats = peak_preserving_stats([])
    assert stats == {"mean": 0.0, "max": 0.0, "p90": 0.0, "p95": 0.0}


def test_aggregate_value_flow_directed_and_abs_stats():
    # signed flows: +10, -6, +2  -> abs {10,6,2}; pos {10,2}
    signals = [
        _sig(1, value_flow=10.0),
        _sig(2, value_flow=-6.0),
        _sig(3, value_flow=2.0),
    ]
    res = aggregate_pair("P", signals)
    f = res.features
    assert f["value_flow_max_pos"] == 10.0
    assert f["value_flow_max_abs"] == 10.0
    # signed mean = (10 - 6 + 2)/3 = 2.0 ; abs mean = 18/3 = 6.0
    assert abs(f["value_flow_mean"] - 2.0) < 1e-12
    assert abs(f["value_flow_absmean"] - 6.0) < 1e-12
    assert f["value_flow_abs_max"] == 10.0


# --------------------------------------------------------------------------- #
# NOT_APPLICABLE exclusion (Req 3.5 / 4.1)
# --------------------------------------------------------------------------- #
def test_not_applicable_excluded_from_signal_stats():
    # isolation applicable only on two hands (values 4 and 8); the third is NOT_APPLICABLE and
    # must be dropped, NOT coerced to 0 (which would drag mean to 4.0 instead of 6.0).
    signals = [
        _sig(1, isolation=4.0),
        _sig(2, isolation=NOT_APPLICABLE),
        _sig(3, isolation=8.0),
    ]
    res = aggregate_pair("P", signals)
    f = res.features
    assert f["isolation_max"] == 8.0
    assert abs(f["isolation_mean"] - 6.0) < 1e-12  # mean of {4,8}, sentinel excluded


def test_all_not_applicable_yields_neutral_fill():
    signals = [
        _sig(1, aggression_asymmetry=NOT_APPLICABLE),
        _sig(2, aggression_asymmetry=NOT_APPLICABLE),
    ]
    res = aggregate_pair("P", signals)
    f = res.features
    assert f["aggression_asymmetry_mean"] == 0.0
    assert f["aggression_asymmetry_max"] == 0.0


# --------------------------------------------------------------------------- #
# Burst / temporal-concentration feature (seed for Property 4, Req 4.2 / 4.3)
# --------------------------------------------------------------------------- #
def test_longest_flagged_burst_counts_runs():
    # T F T T F T T T  -> max run 3, three runs
    flags = [True, False, True, True, False, True, True, True]
    max_burst, burst_count = longest_flagged_burst(flags)
    assert max_burst == 3
    assert burst_count == 3


def test_longest_flagged_burst_empty_and_all_false():
    assert longest_flagged_burst([]) == (0, 0)
    assert longest_flagged_burst([False, False]) == (0, 0)


def test_concentration_increases_burst_and_episodic_score():
    # Same flagged MASS (k=3 flagged hands, each |value_flow|=5) and same n=6.
    # Spread: flagged at positions 0,2,4 (isolated) -> max_burst 1.
    spread = [
        _sig(0, value_flow=5.0, flagged=True),
        _sig(1, value_flow=0.0, flagged=False),
        _sig(2, value_flow=5.0, flagged=True),
        _sig(3, value_flow=0.0, flagged=False),
        _sig(4, value_flow=5.0, flagged=True),
        _sig(5, value_flow=0.0, flagged=False),
    ]
    # Concentrated: flagged at positions 0,1,2 (adjacent) -> max_burst 3.
    concentrated = [
        _sig(0, value_flow=5.0, flagged=True),
        _sig(1, value_flow=5.0, flagged=True),
        _sig(2, value_flow=5.0, flagged=True),
        _sig(3, value_flow=0.0, flagged=False),
        _sig(4, value_flow=0.0, flagged=False),
        _sig(5, value_flow=0.0, flagged=False),
    ]
    r_spread = aggregate_pair("P", spread)
    r_conc = aggregate_pair("P", concentrated)

    # Same mass -> same flagged_count and same episodic peak.
    assert r_spread.flagged_count == r_conc.flagged_count == 3
    assert r_spread.features["episodic_peak"] == r_conc.features["episodic_peak"] == 5.0

    # Concentration raises max_burst and therefore the episodic score (>=, Property 4 seed).
    assert r_conc.max_burst >= r_spread.max_burst
    assert r_conc.max_burst == 3 and r_spread.max_burst == 1
    assert r_conc.episodic_score >= r_spread.episodic_score
    assert r_conc.episodic_score > r_spread.episodic_score  # strict here since BURST_GAIN > 0
    # exact: spread multiplier 1.0 -> 5.0 ; concentrated 1 + 0.5*2 = 2.0 -> 10.0
    assert abs(r_spread.episodic_score - 5.0) < 1e-12
    assert abs(r_conc.episodic_score - 5.0 * (1.0 + BURST_GAIN * 2)) < 1e-12


def test_gap_gini_even_vs_clustered():
    # even spacing -> gini ~ 0
    even = [True, False, True, False, True, False, True]
    assert gap_gini(even) == 0.0
    # clustered flags then a long tail gap -> gini > 0
    clustered = [True, True, True, False, False, False, False, False, True]
    assert gap_gini(clustered) > 0.0


# --------------------------------------------------------------------------- #
# Empirical-Bayes shrinkage (Req 4.4)
# --------------------------------------------------------------------------- #
def test_shrinkage_low_count_pulled_to_prior():
    prior = 0.02
    # Low-count pair: 1 flagged of 2 hands (empirical 0.5) but only 2 hands -> pulled toward prior.
    shrunk = empirical_bayes_shrink(k=1, n=2, prior=prior, alpha=DEFAULT_PRIOR_STRENGTH)
    empirical = 0.5
    # closer to prior than to its own empirical rate
    assert abs(shrunk - prior) < abs(shrunk - empirical)


def test_shrinkage_high_count_pulled_to_empirical():
    prior = 0.02
    # High-count pair: 500 flagged of 1000 (empirical 0.5) -> stays near its empirical rate.
    shrunk = empirical_bayes_shrink(k=500, n=1000, prior=prior, alpha=DEFAULT_PRIOR_STRENGTH)
    empirical = 0.5
    assert abs(shrunk - empirical) < abs(shrunk - prior)


def test_shrinkage_weight_decreases_with_n():
    prior = 0.1
    # zero hands -> exactly the prior
    assert empirical_bayes_shrink(k=0, n=0, prior=prior) == prior
    # the weight on the prior, alpha/(n+alpha), strictly decreases as n grows
    a = DEFAULT_PRIOR_STRENGTH
    w_small = a / (2 + a)
    w_large = a / (1000 + a)
    assert w_large < w_small


def test_prior_from_artifact_variants():
    # bare thresholds dict
    assert prior_from_artifact({"pool_prior_positive_prevalence": 0.03}) == 0.03
    # nested under "thresholds"
    assert prior_from_artifact({"thresholds": {"pool_prior_positive_prevalence": 0.05}}) == 0.05
    # None -> default
    assert prior_from_artifact(None, default=0.07) == 0.07

    class _Artifact:
        thresholds = {"pool_prior_positive_prevalence": 0.09}

    assert prior_from_artifact(_Artifact()) == 0.09


def test_aggregate_uses_injected_prior_for_shrinkage():
    signals = [_sig(1, value_flow=3.0, flagged=True), _sig(2, value_flow=0.0)]
    res = aggregate_pair(
        "P", signals, eda_artifact={"thresholds": {"pool_prior_positive_prevalence": 0.25}}
    )
    assert res.features["pool_prior_used"] == 0.25
    # explicit prior overrides artifact
    res2 = aggregate_pair("P", signals, prior=0.5, eda_artifact={"thresholds": {"pool_prior_positive_prevalence": 0.25}})
    assert res2.features["pool_prior_used"] == 0.5


# --------------------------------------------------------------------------- #
# Phase filtering & determinism
# --------------------------------------------------------------------------- #
def test_phase_filtering():
    signals = [
        _sig(1, value_flow=9.0, phase="development", flagged=True),
        _sig(2, value_flow=1.0, phase="evaluation"),
    ]
    dev = aggregate_pair("P", signals, phase="development")
    ev = aggregate_pair("P", signals, phase="evaluation")
    assert dev.n_hands == 1 and dev.flagged_count == 1
    assert ev.n_hands == 1 and ev.flagged_count == 0
    assert dev.features["value_flow_max_abs"] == 9.0
    assert ev.features["value_flow_max_abs"] == 1.0


def test_determinism_reordered_input_same_output():
    signals = [
        _sig(3, value_flow=2.0, flagged=True),
        _sig(1, value_flow=5.0, flagged=True),
        _sig(2, value_flow=-4.0),
    ]
    a = aggregate_pair("P", signals)
    b = aggregate_pair("P", list(reversed(signals)))
    assert a.features == b.features
    assert a.max_burst == b.max_burst
    assert a.episodic_score == b.episodic_score


def test_empty_input_is_zero_not_error():
    res = aggregate_pair("P", [])
    assert res.n_hands == 0
    assert res.flagged_count == 0
    assert res.max_burst == 0
    assert res.episodic_score == 0.0
    assert res.features["value_flow_max_abs"] == 0.0
