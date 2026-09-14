"""Property test for the rank-tracking gate monotonicity, both metrics (task 5.3).

# Feature: poker-anchor-reproduction, Property 4: For any ladder of
# ``(real_lb, local_pair_ap, local_combined)`` points, EACH per-metric
# ``MetricRankTracking.passed`` flag (primary PairAP and secondary combined) SHALL be a
# monotone function of that metric's local ordering agreement with the real-LB
# ordering: making one artifact's local value move strictly toward its real-LB rank
# never decreases that metric's Spearman, and a metric is marked TRUSTED only when
# ``spearman >= threshold`` AND ``spearman > 0.0``. The overall
# ``RankTrackingVerdict.passed``/``verdict`` SHALL equal the PRIMARY (PairAP) result.

Property 4 (design.md) is the load-bearing external-judge gate invariant. This test
exercises three sub-claims over random ladders (min 100 iterations each):

(a) MONOTONICITY. Sorting the points by ``real_lb`` and applying a *rank-improving
    perturbation* — an adjacent-in-real-order transposition of two locally-inverted
    values — moves the local ordering strictly toward the real-LB ordering and fixes
    exactly one discordant pair while changing no other pair's concordance. Spearman is
    the Pearson correlation of ranks, so such a single-inversion-fixing adjacent swap
    can only INCREASE (or hold) the correlation. We assert the recomputed per-metric
    Spearman is non-decreasing after the perturbation, for BOTH metrics independently.

(b) PASS-FLAG RULE + verdict tiers. For every scored ladder we assert
    ``passed == (spearman >= 0.7 AND spearman > 0.0)`` and that the verdict tier obeys
    the pre-registered boundaries (TRUSTED at ``>= 0.7``, WEAK on ``0 < s < 0.7``, NULL
    at ``s <= 0.0``) — per metric.

(c) OVERALL == PRIMARY. The overall ``RankTrackingVerdict.passed``/``verdict`` equal
    the PRIMARY (PairAP) metric's result, independent of the secondary (contract
    Rules 2 & 12 — the pre-registered primary gate is never swapped for the secondary
    because it looks better).

Governed by the reverse-engineering-accountability contract: the gate constants
(OPERATIONAL_BAR=0.7, CONTRACT_MINIMUM_BAR=0.0) and the Spearman statistic
(``poker_collusion.validation.lb_cv.spearman_corr``) are REUSED verbatim, never
re-implemented here.

Validates: Requirements 2.2, 2.3
"""

from __future__ import annotations

from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.config import PipelineConfig
from poker_collusion.validation.lb_cv import spearman_corr

from anchor_repro.ladder import (
    CONTRACT_MINIMUM_BAR,
    NULL,
    OPERATIONAL_BAR,
    SCORABLE,
    TRUSTED,
    WEAK,
    LadderPoint,
    LadderValidator,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer


# --------------------------------------------------------------------------- #
# A validator whose scorer is never invoked. ``rank_tracking`` reads only the  #
# ladder points passed to it (never ``self.scorer``), so an un-fed scorer is a #
# safe, data-file-free driver for the gate logic (mirrors the models test).    #
# --------------------------------------------------------------------------- #
def _validator() -> LadderValidator:
    return LadderValidator(CanonicalScorer(ScoringRecipe(), PipelineConfig()))


def _make_points(
    real_lb: List[float], pair_ap: List[float], combined: List[float]
) -> List[LadderPoint]:
    """Build SCORABLE ladder points directly from three aligned value lists."""
    return [
        LadderPoint(
            config_id=f"ladder_{i}",
            real_lb=r,
            local_pair_ap=pa,
            local_combined=cb,
            recipe_id="canonical_v1",
            source=f"synthetic[{i}]",
            status=SCORABLE,
        )
        for i, (r, pa, cb) in enumerate(zip(real_lb, pair_ap, combined))
    ]


def _expected_tier(spearman: float) -> str:
    if spearman >= OPERATIONAL_BAR:
        return TRUSTED
    if spearman > CONTRACT_MINIMUM_BAR:
        return WEAK
    return NULL


# --------------------------------------------------------------------------- #
# Strategies: DISTINCT real_lb and DISTINCT per-metric local values so the      #
# rank orderings are unambiguous (no shared ranks) and an adjacent-in-real-     #
# order swap fixes EXACTLY one discordant pair. Values are drawn as distinct    #
# integers then used as floats — exact, well-conditioned, tie-free.             #
# --------------------------------------------------------------------------- #
_distinct_ints = st.lists(
    st.integers(min_value=-10_000, max_value=10_000),
    min_size=3,
    max_size=9,
    unique=True,
)


@st.composite
def _ladders(draw):
    """Draw ``n`` points (3..9) with DISTINCT real_lb and DISTINCT local metrics.

    Returns ``(real_lb, pair_ap, combined)`` as three equal-length float lists whose
    entries are pairwise distinct within each list, so ranks are a strict permutation.
    """
    n = draw(st.integers(min_value=3, max_value=9))
    real_lb = draw(st.lists(st.integers(-10_000, 10_000), min_size=n, max_size=n, unique=True))
    pair_ap = draw(st.lists(st.integers(-10_000, 10_000), min_size=n, max_size=n, unique=True))
    combined = draw(st.lists(st.integers(-10_000, 10_000), min_size=n, max_size=n, unique=True))
    return (
        [float(x) for x in real_lb],
        [float(x) for x in pair_ap],
        [float(x) for x in combined],
    )


# --------------------------------------------------------------------------- #
# (a) MONOTONICITY under a rank-improving perturbation, for BOTH metrics        #
# --------------------------------------------------------------------------- #
def _rank_improve_once(real_lb: List[float], local: List[float]) -> List[float]:
    """Apply ONE rank-improving perturbation to ``local`` w.r.t. the ``real_lb`` order.

    Sort indices by ``real_lb`` ascending; walk adjacent pairs in that order and, at the
    first pair whose local values are INVERTED relative to the real order (local[a] >
    local[b] while real_lb[a] < real_lb[b]), swap those two local values. This is an
    adjacent-in-real-order transposition that fixes exactly one discordant pair and
    leaves every other pair's concordance unchanged, so it can only raise/hold Spearman.

    Returns the (possibly) perturbed local list. If already perfectly ordered w.r.t.
    real_lb (no inverted adjacent pair), returns an unchanged copy.
    """
    order = sorted(range(len(real_lb)), key=lambda i: real_lb[i])
    perturbed = list(local)
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        if perturbed[a] > perturbed[b]:
            perturbed[a], perturbed[b] = perturbed[b], perturbed[a]
            break
    return perturbed


# Feature: poker-anchor-reproduction, Property 4: rank-improving perturbation never
# decreases a metric's Spearman (monotonicity), for BOTH metrics independently.
@settings(max_examples=200, deadline=None)
@given(ladder=_ladders())
def test_rank_improving_perturbation_never_decreases_spearman(ladder) -> None:
    real_lb, pair_ap, combined = ladder

    for local in (pair_ap, combined):
        before = spearman_corr(local, real_lb)
        improved = _rank_improve_once(real_lb, local)
        after = spearman_corr(improved, real_lb)
        # A single adjacent-in-real-order concordant swap fixes one inversion and
        # changes no other pair's concordance -> Spearman is non-decreasing.
        assert after >= before - 1e-9, (
            f"rank-improving perturbation decreased Spearman: {before} -> {after}"
        )


# Feature: poker-anchor-reproduction, Property 4: driving a metric's local values ALL
# the way to the real-LB ordering maximises Spearman at +1.0 (the monotone endpoint).
@settings(max_examples=200, deadline=None)
@given(ladder=_ladders())
def test_perfect_alignment_reaches_and_holds_the_maximum(ladder) -> None:
    real_lb, pair_ap, _combined = ladder
    # Repeatedly apply the rank-improving swap until no adjacent inversion remains;
    # each step is non-decreasing (asserted below), and the fixed point is the
    # perfectly-aligned ordering with Spearman == +1.0 (distinct values, tie-free).
    local = list(pair_ap)
    prev = spearman_corr(local, real_lb)
    for _ in range(len(local) ** 2 + 1):
        nxt_local = _rank_improve_once(real_lb, local)
        if nxt_local == local:
            break
        nxt = spearman_corr(nxt_local, real_lb)
        assert nxt >= prev - 1e-9  # every step monotone non-decreasing
        local, prev = nxt_local, nxt
    # Fixed point: local now orders exactly like real_lb -> perfect rank correlation.
    assert spearman_corr(local, real_lb) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# (b) PASS-FLAG RULE + verdict tiers, per metric                              #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 4: each per-metric passed flag ==
# (spearman >= 0.7 AND spearman > 0.0) and the verdict tier obeys the boundaries.
@settings(max_examples=200, deadline=None)
@given(ladder=_ladders())
def test_per_metric_pass_flag_and_verdict_tier_rule(ladder) -> None:
    real_lb, pair_ap, combined = ladder
    verdict = _validator().rank_tracking(_make_points(real_lb, pair_ap, combined))

    for mrt, local in ((verdict.primary, pair_ap), (verdict.secondary, combined)):
        s = spearman_corr(local, real_lb)
        # The gate reused the same statistic verbatim.
        assert mrt.spearman == pytest.approx(s)
        assert mrt.threshold == OPERATIONAL_BAR
        # passed == (spearman >= 0.7 AND spearman > 0.0)
        expected_passed = (s >= OPERATIONAL_BAR) and (s > CONTRACT_MINIMUM_BAR)
        assert mrt.passed == expected_passed
        # verdict tier boundaries: TRUSTED >= 0.7, WEAK on (0, 0.7), NULL <= 0.
        assert mrt.verdict == _expected_tier(s)
        # TRUSTED <=> passed (the two ways of stating the operational bar agree).
        assert (mrt.verdict == TRUSTED) == mrt.passed
        # A metric is TRUSTED only when strictly positive AND >= threshold.
        if mrt.verdict == TRUSTED:
            assert s >= OPERATIONAL_BAR and s > CONTRACT_MINIMUM_BAR


# --------------------------------------------------------------------------- #
# (c) OVERALL verdict == PRIMARY (PairAP), independent of the secondary        #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 4: overall passed/verdict equal the
# PRIMARY (PairAP) metric's result and never the secondary.
@settings(max_examples=200, deadline=None)
@given(ladder=_ladders())
def test_overall_verdict_equals_primary(ladder) -> None:
    real_lb, pair_ap, combined = ladder
    verdict = _validator().rank_tracking(_make_points(real_lb, pair_ap, combined))

    # The pre-registered gate: overall == PRIMARY (PairAP), never the secondary.
    assert verdict.passed == verdict.primary.passed
    assert verdict.verdict == verdict.primary.verdict
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"

    # Both metrics are always present (never cherry-picked); the secondary is reported
    # but never overrides the overall verdict.
    assert verdict.n_points == len(real_lb)
    prov = verdict.to_provenance()
    assert prov["primary"]["metric_name"] == "pair_ap"
    assert prov["secondary"]["metric_name"] == "combined"

    # When the two metrics land on the same tier, agree is True; otherwise agree is
    # False with a non-empty disagreement note (recorded, never dropped).
    same_tier = verdict.primary.verdict == verdict.secondary.verdict
    assert verdict.agree == same_tier
    if not verdict.agree:
        assert verdict.disagreement_note


# Feature: poker-anchor-reproduction, Property 4: a PairAP-TRUSTED / combined-not-TRUSTED
# ladder still passes overall (primary decides), and the reverse never passes overall.
@settings(max_examples=200, deadline=None)
@given(ladder=_ladders())
def test_secondary_never_overrides_primary(ladder) -> None:
    real_lb, pair_ap, combined = ladder
    verdict = _validator().rank_tracking(_make_points(real_lb, pair_ap, combined))
    # Overall passed is decided solely by the primary, regardless of the secondary's
    # own pass flag (the secondary can pass or fail without moving the gate).
    assert verdict.passed == verdict.primary.passed
    # If the primary failed, the overall must fail even if the secondary passed.
    if not verdict.primary.passed:
        assert verdict.passed is False
    # If the primary passed, the overall must pass even if the secondary failed.
    if verdict.primary.passed:
        assert verdict.passed is True
