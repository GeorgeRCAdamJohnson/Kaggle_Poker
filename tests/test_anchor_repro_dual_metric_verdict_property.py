"""Property test for the dual-metric verdict — always reported, never cherry-picked (task 5.7).

# Feature: poker-anchor-reproduction, Property 8: For any scored ladder, the
# ``RankTrackingVerdict`` SHALL contain BOTH the primary (PairAP) and the secondary
# (combined) ``MetricRankTracking`` results; the overall ``passed``/``verdict`` SHALL
# equal the primary (PairAP) result independent of the secondary result (the primary
# gate is pre-registered and cannot be swapped for the secondary because it looks
# better), and whenever the two metrics disagree on their pass flag the verdict SHALL
# set ``agree=False`` and record a non-empty ``disagreement_note``.

This exercises ``LadderValidator.rank_tracking`` (task 5.2, ``anchor_repro.ladder``).
It builds random ladders of ``(real_lb, local_pair_ap, local_combined)`` points and
feeds them through the gate, asserting the three Property-8 guarantees.

Design vs task-wording note (honest, contract Rule 9 — read this before the asserts):

    The design ("Data Models" / "Pre-registered rank-tracking gate") and the shipped
    implementation decide ``agree`` on the verdict TIER (TRUSTED / WEAK / NULL), i.e.
    ``agree == (primary.verdict == secondary.verdict)`` and ``disagreement_note`` is
    populated iff the tiers differ. The task text for Property 8 says "disagree on
    their pass flag". These are NOT identical: two metrics can share a pass flag
    (both ``passed=False``) yet sit in different tiers (WEAK vs NULL) — that is a tier
    disagreement but a pass-flag AGREEMENT.

    The relationship is a strict implication, and it runs in the direction the task
    cares about: a pass-flag DISAGREEMENT (one metric ``passed=True``, the other
    ``passed=False``) FORCES a tier disagreement, because ``passed=True`` ⇒ tier
    ``TRUSTED`` while ``passed=False`` ⇒ tier ``WEAK`` or ``NULL``. So whenever the two
    metrics disagree on the pass flag, the tier-based implementation ALREADY sets
    ``agree=False`` with a non-empty ``disagreement_note`` — the Property-8 requirement
    as literally worded is satisfied by the tier-based implementation.

    This test therefore asserts BOTH readings so the trail is honest:
      * the LITERAL Property-8 pass-flag clause (pass-flag disagreement ⇒ agree=False +
        non-empty note), and
      * the tier-based clause the implementation actually uses (tier disagreement ⇒
        agree=False + non-empty note, and agree ⇒ tiers equal + empty note).
    Because the pass-flag clause is implied by the tier clause, both hold. This is a
    genuine wording-vs-implementation nuance surfaced rather than papered over; it is
    NOT a bug — the implementation is STRICTER (flags more disagreements) than the
    literal pass-flag clause demands, which is the safe direction for "never drop a
    disagreement".

Generators deliberately cover ladders where PairAP and combined AGREE and where they
CONFLICT (PairAP rank-tracks the real LB well while combined does not, and vice
versa), so the disagreement branch is exercised, not just the agreeing one. Min 100
iterations (``max_examples=200``).

Governed by the reverse-engineering-accountability contract: a PairAP-vs-combined
disagreement is a first-class recorded finding (Rules 2 & 12), the PRIMARY gate is
pre-registered and never swapped for the secondary because it looks better, and every
number here is a MEASURED local property of the gate — never an absolute-LB claim.

Validates: Requirements 2.2, 2.3, 4.1, 4.2
"""

from __future__ import annotations

from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from anchor_repro.ladder import (
    BLOCKED,
    CONTRACT_MINIMUM_BAR,
    NULL,
    OPERATIONAL_BAR,
    SCORABLE,
    TRUSTED,
    WEAK,
    LadderPoint,
    LadderValidator,
    MetricRankTracking,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer

import pandas as pd
from poker_collusion.config import PipelineConfig


# --------------------------------------------------------------------------- #
# A validator whose scorer never touches real data (score_ladder is not used  #
# here — we build LadderPoints directly and only call rank_tracking).         #
# --------------------------------------------------------------------------- #
def _dev_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": [f"P{i}" for i in range(6)],
            "label_status": ["confirmed_target"] * 3 + ["confirmed_non_target"] * 3,
            "behavior_family": [
                "directed_transfer",
                "soft_play",
                "coordinated_isolation",
                "none",
                "none",
                "none",
            ],
        }
    )


def _validator() -> LadderValidator:
    scorer = CanonicalScorer(
        ScoringRecipe(), PipelineConfig(), labels=_dev_labels(), evidence=None
    )
    return LadderValidator(scorer)


# --------------------------------------------------------------------------- #
# Helpers to build a scorable LadderPoint with chosen local values            #
# --------------------------------------------------------------------------- #
def _point(config_id: str, real_lb: float, pair_ap: float, combined: float) -> LadderPoint:
    return LadderPoint(
        config_id=config_id,
        real_lb=real_lb,
        local_pair_ap=pair_ap,
        local_combined=combined,
        recipe_id="canonical_v1",
        source=f"synthetic:{config_id}",
        status=SCORABLE,
    )


# --------------------------------------------------------------------------- #
# Generators                                                                  #
# --------------------------------------------------------------------------- #
# A monotonically increasing real-LB ladder of n points (mimics an ordered LB axis;
# the fuller real trajectory is non-monotone but a distinct increasing axis is a valid
# and adversarially-neutral choice for exercising the gate).
@st.composite
def _real_lb_ladder(draw, min_n: int = 3, max_n: int = 9) -> List[float]:
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    # Distinct increasing values in [0, 1).
    steps = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    out: List[float] = []
    acc = 0.0
    for s in steps:
        acc += s
        out.append(acc)
    return out


def _perfect_order(real_lb: List[float]) -> List[float]:
    """A local metric that rank-tracks the real LB PERFECTLY (identical order)."""
    # Any strictly-increasing transform of the rank gives Spearman == 1.0.
    ranks = _rank(real_lb)
    return [float(r) for r in ranks]


def _reverse_order(real_lb: List[float]) -> List[float]:
    """A local metric that rank-tracks the real LB PERFECTLY INVERSELY (Spearman -1)."""
    ranks = _rank(real_lb)
    n = len(real_lb)
    return [float(n - r) for r in ranks]


def _rank(values: List[float]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for pos, idx in enumerate(order):
        ranks[idx] = pos + 1
    return ranks


# --------------------------------------------------------------------------- #
# Core assertions shared by every property (Property 8, the three guarantees) #
# --------------------------------------------------------------------------- #
def _assert_property_8(verdict) -> None:
    """Assert the three Property-8 guarantees on a computed RankTrackingVerdict."""
    # (1) BOTH metric results are present and correctly named — never cherry-picked.
    assert isinstance(verdict.primary, MetricRankTracking)
    assert isinstance(verdict.secondary, MetricRankTracking)
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"
    # to_provenance carries both too (append-only audit trail records both, Req 4.1/4.2).
    prov = verdict.to_provenance()
    assert prov["primary"]["metric_name"] == "pair_ap"
    assert prov["secondary"]["metric_name"] == "combined"

    # (2) The overall verdict EQUALS the PRIMARY (PairAP) result, independent of the
    # secondary — the pre-registered gate is never swapped for the secondary.
    assert verdict.passed == verdict.primary.passed
    assert verdict.verdict == verdict.primary.verdict

    # (3a) LITERAL Property-8 clause: a pass-flag disagreement ⇒ agree=False + note.
    pass_flags_disagree = verdict.primary.passed != verdict.secondary.passed
    if pass_flags_disagree:
        assert verdict.agree is False
        assert verdict.disagreement_note  # non-empty

    # (3b) Tier-based clause the implementation actually uses (design "Data Models").
    tiers_disagree = verdict.primary.verdict != verdict.secondary.verdict
    assert verdict.agree == (not tiers_disagree)
    if tiers_disagree:
        assert verdict.disagreement_note  # non-empty first-class finding
    else:
        assert verdict.disagreement_note == ""

    # Cross-check the implication that reconciles the two readings: a pass-flag
    # disagreement FORCES a tier disagreement (so the literal clause is subsumed).
    if pass_flags_disagree:
        assert tiers_disagree

    # Verdict/pass consistency inside each MetricRankTracking (the gate's own rule).
    for m in (verdict.primary, verdict.secondary):
        expected_pass = m.spearman >= OPERATIONAL_BAR and m.spearman > CONTRACT_MINIMUM_BAR
        assert m.passed == expected_pass
        if m.spearman >= OPERATIONAL_BAR:
            assert m.verdict == TRUSTED
        elif m.spearman > CONTRACT_MINIMUM_BAR:
            assert m.verdict == WEAK
        else:
            assert m.verdict == NULL


# --------------------------------------------------------------------------- #
# Property 8a: random ladders — both results present, overall == primary,      #
# disagreements always surfaced                                               #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 8: dual-metric verdict, never cherry-picked
@settings(max_examples=200, deadline=None)
@given(
    real_lb=_real_lb_ladder(),
    pair_noise=st.data(),
    combined_noise=st.data(),
)
def test_dual_metric_verdict_random_ladders(real_lb, pair_noise, combined_noise) -> None:
    """For arbitrary (independently random) PairAP and combined orderings over a real-LB
    ladder, the verdict carries BOTH metrics, mirrors the PRIMARY, and never drops a
    disagreement (both the literal pass-flag clause and the tier clause)."""
    n = len(real_lb)
    # Independent random local orderings for the two metrics (may agree or conflict by
    # chance) — the "any scored ladder" universal quantifier.
    pair_ap = [
        pair_noise.draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            label=f"pair_{i}",
        )
        for i in range(n)
    ]
    combined = [
        combined_noise.draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            label=f"comb_{i}",
        )
        for i in range(n)
    ]
    points = [_point(f"c{i}", real_lb[i], pair_ap[i], combined[i]) for i in range(n)]
    verdict = _validator().rank_tracking(points)
    assert verdict.n_points == n
    _assert_property_8(verdict)


# --------------------------------------------------------------------------- #
# Property 8b: DELIBERATE conflict — PairAP tracks well, combined does NOT      #
# (and vice versa). Forces the disagreement branch and pins overall==primary.  #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 8: primary decides under conflict
@settings(max_examples=200, deadline=None)
@given(
    real_lb=_real_lb_ladder(min_n=4),
    flip=st.booleans(),
)
def test_primary_decides_when_metrics_conflict(real_lb, flip) -> None:
    """Construct a ladder where ONE metric perfectly rank-tracks the real LB (TRUSTED)
    and the OTHER perfectly inverts it (NULL). The tiers differ ⇒ pass flags differ ⇒
    agree=False with a non-empty note, and the overall verdict EQUALS whichever the
    PRIMARY (PairAP) is — even when the SECONDARY looks better (no cherry-picking)."""
    n = len(real_lb)
    tracking = _perfect_order(real_lb)   # Spearman +1 → TRUSTED, passed=True
    inverting = _reverse_order(real_lb)  # Spearman -1 → NULL, passed=False

    if flip:
        # PairAP is the GOOD metric, combined is the BAD one.
        pair_ap, combined = tracking, inverting
        expected_verdict = TRUSTED
        expected_passed = True
    else:
        # PairAP is the BAD metric, combined is the GOOD one — the secondary looks
        # better, but the pre-registered PRIMARY gate must still decide the overall.
        pair_ap, combined = inverting, tracking
        expected_verdict = NULL
        expected_passed = False

    points = [_point(f"c{i}", real_lb[i], pair_ap[i], combined[i]) for i in range(n)]
    verdict = _validator().rank_tracking(points)

    _assert_property_8(verdict)
    # Overall mirrors the PRIMARY regardless of which side is stronger.
    assert verdict.verdict == expected_verdict
    assert verdict.passed is expected_passed
    # A genuine disagreement was surfaced, not smoothed over.
    assert verdict.agree is False
    assert verdict.disagreement_note
    # The pass flags actually differ here (one TRUSTED, one NULL) — the literal
    # Property-8 pass-flag clause is exercised, not just the tier clause.
    assert verdict.primary.passed != verdict.secondary.passed


# --------------------------------------------------------------------------- #
# Property 8c: AGREEMENT — both metrics perfectly track ⇒ agree=True, empty note #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 8: agreement carries an empty note
@settings(max_examples=200, deadline=None)
@given(real_lb=_real_lb_ladder())
def test_agreement_when_both_metrics_track(real_lb) -> None:
    """When BOTH metrics perfectly rank-track the real LB (both TRUSTED), agree=True,
    the disagreement_note is empty, both results are still present, and the overall
    verdict is TRUSTED (mirrors the primary)."""
    n = len(real_lb)
    pair_ap = _perfect_order(real_lb)
    combined = _perfect_order(real_lb)
    points = [_point(f"c{i}", real_lb[i], pair_ap[i], combined[i]) for i in range(n)]
    verdict = _validator().rank_tracking(points)

    _assert_property_8(verdict)
    assert verdict.agree is True
    assert verdict.disagreement_note == ""
    assert verdict.verdict == TRUSTED
    assert verdict.passed is True
    # Both metrics present and both TRUSTED — the secondary is reported alongside even
    # when it agrees (never dropped).
    assert verdict.primary.verdict == TRUSTED
    assert verdict.secondary.verdict == TRUSTED


# --------------------------------------------------------------------------- #
# Property 8d: BLOCKED points never enter the gate, but the verdict STILL       #
# carries both metric results (never cherry-picked, even in the null case).    #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 8: both results present even with BLOCKED points
@settings(max_examples=200, deadline=None)
@given(
    real_lb=_real_lb_ladder(),
    n_blocked=st.integers(min_value=0, max_value=4),
)
def test_blocked_points_excluded_but_both_metrics_still_reported(real_lb, n_blocked) -> None:
    """Adding BLOCKED points (no local values) must not change the scorable gate result
    and must not drop either metric result: the verdict always carries BOTH primary and
    secondary, and n_points counts only SCORABLE points."""
    n = len(real_lb)
    pair_ap = _perfect_order(real_lb)
    combined = _reverse_order(real_lb)  # deliberate conflict again
    scorable = [_point(f"c{i}", real_lb[i], pair_ap[i], combined[i]) for i in range(n)]

    blocked = [
        LadderPoint(
            config_id=f"b{j}",
            real_lb=0.9 + 0.001 * j,
            local_pair_ap=None,
            local_combined=None,
            recipe_id="canonical_v1",
            source=f"synthetic-blocked:{j}",
            status=BLOCKED,
            blocked_reason="no re-runnable recipe (synthetic)",
        )
        for j in range(n_blocked)
    ]

    verdict = _validator().rank_tracking(scorable + blocked)
    # Only SCORABLE points enter the gate.
    assert verdict.n_points == n
    _assert_property_8(verdict)
    # Both metric results are present regardless of how many BLOCKED points were added.
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"
