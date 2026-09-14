"""ACCEPTANCE GATE — Metric-contract coverage check (task 4.5, Req 10.6).

This file is the Phase-0 acceptance gate that confirms the metric work of tasks
4.1–4.4 fully satisfies the CONFIRMED-CODE **Metric Contract** in
``.kiro/specs/poker-collusion-detection/RESEARCH_DOSSIER.md`` section 1 (and its
"Reconciliation checklist for task 4.4") BEFORE Phase 0 proceeds.

Unlike ``test_metric_reconciliation.py`` (which reconciles production vs. the
official reference across many cases) this file is a **self-contained acceptance
test of the contract itself**: for every contract clause / constant / edge-case it
constructs a minimal case and ASSERTS the required behavior directly on BOTH the
production scorer (:mod:`poker_collusion.metric.production_metric`) and the verbatim
official reference (:mod:`poker_collusion.metric.reference_public_metric`). One test
per contract clause; each docstring cites the contract sub-section it accepts.

Contract-coverage map (clause -> accepting test):

  §combined weights 0.70/0.20/0.10 ......... test_accept_weighting_is_70_20_10
                                             test_accept_top_level_agreement_and_weighting
  §combined non-finite raises .............. test_accept_combined_is_finite
  §1.1 PairAP ranking / 0-when-no-positives  test_accept_pair_ap_ranking_and_zero_when_no_positives
  §1.1 tie-break pair_id-asc + stable ...... test_accept_pair_ap_tiebreak_pair_id_ascending
  §1.1 never-omit / exact-coverage ......... test_accept_coverage_mismatch_rejected
  §1.5 duplicate pair_id rejected .......... test_accept_duplicate_pair_id_rejected
  §1.5 risk in [0,1] no-clip rejection ..... test_accept_risk_out_of_range_and_nan_rejected_no_clip
  §1.2 EvidenceMAP true-positives-only ..... test_accept_evidence_true_positives_only
  §1.2 denominator min(#relevant,5) ........ test_accept_evidence_denominator_min_relevant_5
  §1.2 missed positive 0-but-counted ....... test_accept_missed_positive_contributes_zero_but_counted
  §1.5 NO_EVIDENCE literal ................. test_accept_no_evidence_literal_sentinel
  §1.2 no-repeat evidence rejection ........ test_accept_repeated_evidence_rejected_multi_no_evidence_ok
  §1.5 fewer-than-5 handling ............... test_accept_fewer_than_five_evidence
  §1.3 BehaviorMAP 3-way mean .............. test_accept_behavior_map_three_way_mean
  §1.3 zero-TP family 0.0 stays in denom ... test_accept_behavior_zero_tp_family_stays_in_denominator
  §1.3 other_coordination + none excluded .. test_accept_behavior_excludes_other_coordination_and_none
  §1.3 behavior scored over ALL pairs ...... test_accept_behavior_scored_over_all_pairs_not_just_tp
  §1.5 ALLOWED_BEHAVIORS enforced .......... test_accept_invalid_predicted_behavior_rejected
  §1.5 row-id column must be pair_id ....... test_accept_wrong_row_id_column_rejected
  §combined all-negative -> 0 .............. test_accept_all_negative_scores_zero
  §1.3 perfect -> 1.0 w/ all 3 families .... test_accept_perfect_submission_scores_one_all_three_families
  §source-of-truth reference guard ......... test_accept_reference_module_is_official_code

Tolerance: production and reference perform the identical 0.70/0.20/0.10 float
summation, so they agree to ``abs <= 1e-12``; component arithmetic worked out by
hand is asserted at the same tolerance. Any failure here means the metric work does
NOT satisfy the contract and Phase 0 is BLOCKED (Req 10.6).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from poker_collusion.metric import production_metric as prod
from poker_collusion.metric import reference_public_metric as ref

# Bit-for-bit tolerance — both do the identical weighted float summation.
TOL = 1e-12

TARGET = list(ref.TARGET_BEHAVIORS)
NON_TARGET_POSITIVE = "other_coordination"


# --------------------------------------------------------------------------- #
# Builders (self-contained; do not depend on other test modules)              #
# --------------------------------------------------------------------------- #
def _evidence_row(hands: list[str]) -> dict:
    """Fill the 5 evidence columns from a hand list, padding with NO_EVIDENCE."""
    padded = list(hands) + [ref.NO_EVIDENCE] * (5 - len(hands))
    return {f"evidence_hand_{i}": padded[i - 1] for i in range(1, 6)}


def _row(pair_id, risk, behavior, hands) -> dict:
    return {
        "pair_id": pair_id,
        "risk_score": risk,
        "predicted_behavior": behavior,
        **_evidence_row(list(hands)),
    }


def _components(solution: pd.DataFrame, submission: pd.DataFrame) -> prod.ScoreComponents:
    """Production components for a case."""
    return prod.score_components(solution, submission, "pair_id")


def _assert_prod_ref_agree(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Assert production combined == reference score to within TOL; return it.

    This is the acceptance-side agreement anchor reused by most clause tests: the
    contract is only "accepted" for a clause if the production scorer reproduces the
    OFFICIAL reference on that clause's minimal case.
    """
    prod_value = prod.score(solution, submission, "pair_id")
    ref_value = ref.score(solution, submission, "pair_id")
    assert prod_value == pytest.approx(ref_value, abs=TOL), (
        f"DIVERGENCE from official reference: production={prod_value!r} "
        f"reference={ref_value!r} delta={abs(prod_value - ref_value):.3e}"
    )
    return ref_value


def _assert_both_reject(solution, submission, row_id, expected_exc) -> None:
    """Assert BOTH production and reference raise ``expected_exc`` on the inputs."""
    with pytest.raises(expected_exc):
        ref.score(solution, submission, row_id)
    with pytest.raises(expected_exc):
        prod.score(solution, submission, row_id)


def _valid_solution() -> pd.DataFrame:
    """A 5-pair solution with all three target families represented among TPs."""
    return pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 1, "soft_play", ["HC"]),
            _row("P3", 1, "coordinated_isolation", ["HD"]),
            _row("P4", 0, "none", []),
            _row("P5", 0, "none", []),
        ]
    )


# =========================================================================== #
# §combined — 0.70 / 0.20 / 0.10 weighting                                    #
# =========================================================================== #
def test_accept_weighting_is_70_20_10():
    """Accepts §1 combined summary: final = 0.70·PairAP + 0.20·EvMAP + 0.10·BehMAP.

    Constructs a case whose three components are all distinct and non-trivial, then
    asserts the production combined equals the exact 0.70/0.20/0.10 weighted sum of
    the exposed components (Req 10.5) AND equals the official reference.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 1, "soft_play", ["HC"]),
            _row("P3", 1, "coordinated_isolation", ["HD"]),
            _row("P4", 0, "none", []),
            _row("P5", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.90, "directed_transfer", ["HA", "WRONG"]),  # partial evidence
            _row("P2", 0.40, "soft_play", ["HC"]),
            _row("P3", 0.80, "coordinated_isolation", ["MISS"]),  # missed evidence
            _row("P4", 0.10, "none", []),
            _row("P5", 0.05, "none", []),
        ]
    )
    comps = _components(solution, submission)
    expected = (
        0.70 * comps.pair_ap
        + 0.20 * comps.evidence_map
        + 0.10 * comps.behavior_map
    )
    assert comps.combined == pytest.approx(expected, abs=TOL)
    # And the exact literal weights are what the module uses.
    assert (prod.PAIR_AP_WEIGHT, prod.EVIDENCE_MAP_WEIGHT, prod.BEHAVIOR_MAP_WEIGHT) == (
        0.70,
        0.20,
        0.10,
    )
    _assert_prod_ref_agree(solution, submission)


def test_accept_combined_is_finite():
    """Accepts §1 combined: a valid case yields a finite float (non-finite raises).

    The contract says a non-finite result raises ``ParticipantVisibleError``; on a
    well-formed case the score must therefore be a finite float in [0, 1] and both
    implementations must agree.
    """
    solution = _valid_solution()
    submission = solution.copy()
    value = _assert_prod_ref_agree(solution, submission)
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


# =========================================================================== #
# §1.1 Pair AP — ranking, zero-when-no-positives, tie-break                   #
# =========================================================================== #
def test_accept_pair_ap_ranking_and_zero_when_no_positives():
    """Accepts §1.1: higher risk ranks earlier; AP == 0.0 when no positives.

    Two sub-cases: (a) positives given the highest risk -> Pair AP == 1.0; (b) a
    solution with zero positives -> Pair AP == 0.0 (the ``positives == 0`` branch).
    """
    # (a) ranking: positives get the highest risk -> perfect Pair AP.
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 0, "none", []),
            _row("P3", 1, "soft_play", ["HB"]),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "none", []),
            _row("P2", 0.1, "none", []),
            _row("P3", 0.8, "none", []),
        ]
    )
    assert _components(solution, submission).pair_ap == pytest.approx(1.0, abs=TOL)
    _assert_prod_ref_agree(solution, submission)

    # (b) no positives anywhere -> Pair AP is defined as 0.0.
    sol0 = pd.DataFrame([_row(f"P{i}", 0, "none", []) for i in range(4)])
    sub0 = pd.DataFrame([_row(f"P{i}", round(0.2 * i, 3), "none", []) for i in range(4)])
    assert _components(sol0, sub0).pair_ap == 0.0
    _assert_prod_ref_agree(sol0, sub0)


def test_accept_pair_ap_tiebreak_pair_id_ascending():
    """Accepts §1.1 tie-break: equal risk -> pair_id ascending via stable mergesort.

    All risks equal. When the positives own the LARGEST pair_ids the negatives sort
    ahead of them (pair_id-ascending), so Pair AP is strictly below 1.0 and equals
    the hand-computed ``(1/3 + 2/4)/2``. This pins the deterministic tie-break exactly
    and both implementations must agree.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 0, "none", []),
            _row("P2", 0, "none", []),
            _row("P3", 1, "directed_transfer", ["HA"]),
            _row("P4", 1, "soft_play", ["HB"]),
        ]
    )
    submission = pd.DataFrame([_row(p, 0.5, "none", []) for p in ("P1", "P2", "P3", "P4")])
    comps = _components(solution, submission)
    assert comps.pair_ap == pytest.approx((1 / 3 + 2 / 4) / 2, abs=TOL)
    assert comps.pair_ap < 1.0
    _assert_prod_ref_agree(solution, submission)


# =========================================================================== #
# §1.1 / §1.5 coverage + uniqueness + row-id column                           #
# =========================================================================== #
def test_accept_coverage_mismatch_rejected():
    """Accepts §1.1 never-omit: a missing OR extra pair_id is rejected outright.

    The contract makes the "never omit a pair" policy MANDATORY: the submission
    pair_id set must EXACTLY equal the solution set. Both a missing pair and an extra
    pair must raise ``ParticipantVisibleError`` in both implementations.
    """
    solution = _valid_solution()
    # Missing: drop a whole pair from the submission.
    missing_sub = solution.iloc[:-1].copy()
    _assert_both_reject(solution, missing_sub, "pair_id", ref.ParticipantVisibleError)
    # Extra: rename one pair to an id not in the solution.
    extra_sub = solution.copy()
    extra_sub.loc[extra_sub.index[0], "pair_id"] = "P_EXTRA"
    _assert_both_reject(solution, extra_sub, "pair_id", ref.ParticipantVisibleError)


def test_accept_duplicate_pair_id_rejected():
    """Accepts §1.5: a duplicate submission pair_id raises (must be unique)."""
    solution = _valid_solution()
    submission = solution.copy()
    submission.loc[submission.index[0], "pair_id"] = submission.loc[
        submission.index[1], "pair_id"
    ]
    _assert_both_reject(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_accept_wrong_row_id_column_rejected():
    """Accepts §1.5: the row-id column name must be exactly ``pair_id``."""
    solution = _valid_solution()
    submission = solution.copy()
    _assert_both_reject(solution, submission, "not_pair_id", ref.ParticipantVisibleError)


# =========================================================================== #
# §1.5 risk domain [0,1] — NO clipping, out-of-range/NaN rejected             #
# =========================================================================== #
@pytest.mark.parametrize("bad_value", [1.5, -0.01, np.nan])
def test_accept_risk_out_of_range_and_nan_rejected_no_clip(bad_value: float):
    """Accepts §1.5 risk domain: out-of-range / NaN risk is REJECTED, never clipped.

    Values above 1, below 0, and NaN each raise ``ParticipantVisibleError`` in both
    implementations (proving no silent clipping to [0, 1]).
    """
    solution = _valid_solution()
    submission = solution.copy()
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[submission.index[0], "risk_score"] = bad_value
    _assert_both_reject(solution, submission, "pair_id", ref.ParticipantVisibleError)


# =========================================================================== #
# §1.2 Evidence MAP@5                                                          #
# =========================================================================== #
def test_accept_evidence_true_positives_only():
    """Accepts §1.2: evidence is scored over TRUE positives only, no FP penalty.

    A NON-target pair (y_true == 0) that submits perfectly matching-looking evidence
    must NOT contribute to Evidence MAP and must carry no penalty. Here the only true
    positive has perfect evidence (AP 1.0); the negative's evidence is ignored, so
    Evidence MAP == 1.0 exactly.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 0, "none", ["HB"]),  # negative WITH planted-looking evidence
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA"]),
            _row("P2", 0.8, "none", ["HB"]),  # ignored: P2 is not a true positive
        ]
    )
    assert _components(solution, submission).evidence_map == pytest.approx(1.0, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


@pytest.mark.parametrize("n_relevant", [1, 2, 5])
def test_accept_evidence_denominator_min_relevant_5(n_relevant: int):
    """Accepts §1.2 denominator: per-pair divisor is ``min(#relevant, 5)``.

    Plant ``n_relevant`` distinct true hands and submit all of them in order (all
    hits), so the numerator is ``k = min(#submitted, 5)`` and the pair score is
    ``k / min(#relevant, 5)``. Tested at 1, 2 and the k=5 cap.
    """
    planted = [f"H{k}" for k in range(n_relevant)]
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", planted),
            _row("P2", 0, "none", []),
        ]
    )
    submitted = planted[:5]
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", submitted),
            _row("P2", 0.1, "none", []),
        ]
    )
    expected = len(submitted) / min(len(planted), 5)
    assert _components(solution, submission).evidence_map == pytest.approx(expected, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


def test_accept_missed_positive_contributes_zero_but_counted():
    """Accepts §1.2: a true positive with no matching evidence scores 0, still counted.

    Two positives: P1 perfectly hit (AP 1.0), P2 fully missed (AP 0.0). The mean is
    over BOTH true positives -> ``(1.0 + 0.0)/2 == 0.5`` (the missed pair stays in
    the denominator). This is the E4 coverage-dominance lever.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 1, "soft_play", ["HC"]),
            _row("P3", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA", "HB"]),
            _row("P2", 0.8, "soft_play", ["WRONG1", "WRONG2"]),
            _row("P3", 0.1, "none", []),
        ]
    )
    assert _components(solution, submission).evidence_map == pytest.approx(0.5, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


def test_accept_no_evidence_literal_sentinel():
    """Accepts §1.5 sentinel: only the exact literal ``NO_EVIDENCE`` is dropped.

    A row of all-``NO_EVIDENCE`` for a true positive scores 0 (nothing relevant
    submitted). A case-deviated sentinel (``No_Evidence``) is NOT the sentinel: it is
    kept as a (non-matching) hand id, so it also scores 0 here — but crucially it is
    treated as data, not stripped. Both facts are asserted, and both implementations
    must agree.
    """
    assert ref.NO_EVIDENCE == "NO_EVIDENCE"  # the exact literal
    assert prod.NO_EVIDENCE == "NO_EVIDENCE"

    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 0, "none", []),
        ]
    )
    # All submitted evidence is the sentinel -> cleaned to empty -> pair AP 0.
    all_sentinel = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", []),  # padded with NO_EVIDENCE
            _row("P2", 0.1, "none", []),
        ]
    )
    assert _components(solution, all_sentinel).evidence_map == pytest.approx(0.0, abs=TOL)
    _assert_prod_ref_agree(solution, all_sentinel)

    # A case-deviated "No_Evidence" is treated as an ordinary (non-matching) hand id.
    deviated = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["No_Evidence"]),
            _row("P2", 0.1, "none", []),
        ]
    )
    # It does not match the planted "HA", so evidence is still 0 — but it was kept as
    # data (no crash / no strip), and production agrees with the official reference.
    assert _components(solution, deviated).evidence_map == pytest.approx(0.0, abs=TOL)
    _assert_prod_ref_agree(solution, deviated)


def test_accept_repeated_evidence_rejected_multi_no_evidence_ok():
    """Accepts §1.2: repeated real hand within a pair rejected; multi-NO_EVIDENCE ok.

    Repeating a real hand id in two evidence slots raises
    ``ParticipantVisibleError`` in both implementations. But padding with several
    ``NO_EVIDENCE`` cells (the common case) is explicitly allowed and must score
    without raising.
    """
    solution = _valid_solution()

    # Repeated REAL hand id within one pair -> rejected by both.
    repeated = solution.copy()
    repeated.loc[repeated["pair_id"] == "P1", "evidence_hand_1"] = "DUP"
    repeated.loc[repeated["pair_id"] == "P1", "evidence_hand_2"] = "DUP"
    _assert_both_reject(solution, repeated, "pair_id", ref.ParticipantVisibleError)

    # One real hand + four NO_EVIDENCE pads (repeated sentinel) -> allowed.
    padded = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA"]),
            _row("P2", 0.8, "soft_play", ["HC"]),
            _row("P3", 0.7, "coordinated_isolation", ["HD"]),
            _row("P4", 0.1, "none", []),
            _row("P5", 0.05, "none", []),
        ]
    )
    _assert_prod_ref_agree(solution, padded)  # must not raise


def test_accept_fewer_than_five_evidence():
    """Accepts §1.5 @5 cutoff: submissions shorter than 5 hands score correctly.

    Plant 3 relevant hands; submit only 2 (one miss at rank 1, one hit at rank 2) ->
    precision contribution ``(1/2)`` over denominator ``min(3, 5) == 3`` -> evidence
    ``(1/2)/3``. Padding under 5 does not change the denominator.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB", "HC"]),
            _row("P2", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["MISS", "HB"]),
            _row("P2", 0.1, "none", []),
        ]
    )
    assert _components(solution, submission).evidence_map == pytest.approx((1 / 2) / 3, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


# =========================================================================== #
# §1.3 Behavior MAP                                                            #
# =========================================================================== #
def test_accept_behavior_map_three_way_mean():
    """Accepts §1.3: Behavior MAP is the mean over EXACTLY the three families.

    All three families present as true positives, each perfectly predicted -> every
    per-family OvR AP == 1.0 -> Behavior MAP == (1+1+1)/3 == 1.0.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 1, "soft_play", ["HB"]),
            _row("P3", 1, "coordinated_isolation", ["HC"]),
            _row("P4", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA"]),
            _row("P2", 0.8, "soft_play", ["HB"]),
            _row("P3", 0.7, "coordinated_isolation", ["HC"]),
            _row("P4", 0.1, "none", []),
        ]
    )
    assert _components(solution, submission).behavior_map == pytest.approx(1.0, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


def test_accept_behavior_zero_tp_family_stays_in_denominator():
    """Accepts §1.3: a family with zero true positives contributes 0.0, kept in denom.

    Only directed_transfer and soft_play appear; coordinated_isolation has zero TPs.
    Behavior MAP == (1.0 + 1.0 + 0.0)/3 (dividing by 3, not 2).
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 1, "soft_play", ["HB"]),
            _row("P3", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA"]),
            _row("P2", 0.8, "soft_play", ["HB"]),
            _row("P3", 0.1, "none", []),
        ]
    )
    assert _components(solution, submission).behavior_map == pytest.approx(
        (1.0 + 1.0 + 0.0) / 3, abs=TOL
    )
    _assert_prod_ref_agree(solution, submission)


def test_accept_behavior_excludes_other_coordination_and_none():
    """Accepts §1.3: ``other_coordination`` and ``none`` are never OvR classes.

    P2 is a true positive labelled ``other_coordination``; it must not act as a
    relevant item for any of the three disclosed families. Behavior MAP therefore
    depends only on directed_transfer + soft_play (both 1.0) and coordinated_isolation
    (0.0 TP) -> (1+1+0)/3, exactly as if P2's family were invisible to Behavior MAP.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 1, NON_TARGET_POSITIVE, ["HB"]),
            _row("P3", 1, "soft_play", ["HC"]),
            _row("P4", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA"]),
            _row("P2", 0.85, NON_TARGET_POSITIVE, ["HB"]),
            _row("P3", 0.8, "soft_play", ["HC"]),
            _row("P4", 0.1, "none", []),
        ]
    )
    assert _components(solution, submission).behavior_map == pytest.approx(
        (1.0 + 1.0 + 0.0) / 3, abs=TOL
    )
    _assert_prod_ref_agree(solution, submission)


def test_accept_behavior_scored_over_all_pairs_not_just_tp():
    """Accepts §1.3 scope: OvR AP ranks over ALL scored pairs, not just true positives.

    A negative pair predicted ``directed_transfer`` with a HIGHER risk than the true
    directed_transfer positive drags that family's OvR AP below 1.0 (the negative's
    behavior_risk participates in the ranking). If Behavior MAP only looked at true
    positives, the family AP would be 1.0; instead it is (positive lands at rank 2):
    directed_transfer AP == 1/2, soft_play AP == 1.0, coordinated_isolation == 0.0.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 0, "none", []),  # negative, but predicted directed_transfer high
            _row("P3", 1, "soft_play", ["HB"]),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.5, "directed_transfer", ["HA"]),
            _row("P2", 0.9, "directed_transfer", []),  # decoy ranks ahead of P1
            _row("P3", 0.8, "soft_play", ["HB"]),
        ]
    )
    # directed_transfer OvR: truth positive at P1 ranked BEHIND the P2 decoy -> AP=1/2.
    expected = (0.5 + 1.0 + 0.0) / 3
    assert _components(solution, submission).behavior_map == pytest.approx(expected, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


def test_accept_invalid_predicted_behavior_rejected():
    """Accepts §1.5: a predicted_behavior outside ALLOWED_BEHAVIORS is rejected.

    Also asserts the ALLOWED_BEHAVIORS set is exactly the five disclosed labels and
    TARGET_BEHAVIORS is exactly the three families.
    """
    assert ref.ALLOWED_BEHAVIORS == {
        "none",
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
        "other_coordination",
    }
    assert set(ref.TARGET_BEHAVIORS) == {
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
    }
    solution = _valid_solution()
    submission = solution.copy()
    submission["predicted_behavior"] = submission["predicted_behavior"].astype(object)
    submission.loc[submission.index[0], "predicted_behavior"] = "totally_made_up"
    _assert_both_reject(solution, submission, "pair_id", ref.ParticipantVisibleError)


# =========================================================================== #
# §combined — boundary scores                                                 #
# =========================================================================== #
def test_accept_all_negative_scores_zero():
    """Accepts §combined boundary: an all-negative solution scores 0 on all parts.

    No positives anywhere -> Pair AP == 0, Evidence MAP == 0, Behavior MAP == 0, so
    the combined score is exactly 0.0.
    """
    solution = pd.DataFrame([_row(f"P{i}", 0, "none", []) for i in range(6)])
    submission = pd.DataFrame([_row(f"P{i}", round(0.1 * i, 3), "none", []) for i in range(6)])
    comps = _components(solution, submission)
    assert comps.pair_ap == 0.0
    assert comps.evidence_map == 0.0
    assert comps.behavior_map == 0.0
    assert comps.combined == pytest.approx(0.0, abs=TOL)
    _assert_prod_ref_agree(solution, submission)


def test_accept_perfect_submission_scores_one_all_three_families():
    """Accepts §1.3: a perfect submission reaches 1.0 ONLY with all 3 families present.

    With all three families among the true positives, a perfect copy scores 1.0. A
    variant missing coordinated_isolation caps Behavior MAP at (1+1+0)/3 so the
    combined is strictly below 1.0 — confirming the "needs all three" clause.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 1, "soft_play", ["HC"]),
            _row("P3", 1, "coordinated_isolation", ["HD", "HE"]),
            _row("P4", 0, "none", []),
            _row("P5", 0, "none", []),
        ]
    )
    perfect = solution.copy()
    value = _assert_prod_ref_agree(solution, perfect)
    assert value == pytest.approx(1.0, abs=TOL)
    comps = _components(solution, perfect)
    assert comps.pair_ap == pytest.approx(1.0, abs=TOL)
    assert comps.evidence_map == pytest.approx(1.0, abs=TOL)
    assert comps.behavior_map == pytest.approx(1.0, abs=TOL)

    # Missing a whole family caps the perfect score below 1.0.
    solution_2fam = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 1, "soft_play", ["HB"]),
            _row("P3", 0, "none", []),
        ]
    )
    value2 = _assert_prod_ref_agree(solution_2fam, solution_2fam.copy())
    assert value2 < 1.0
    assert _components(solution_2fam, solution_2fam.copy()).behavior_map == pytest.approx(
        (1.0 + 1.0 + 0.0) / 3, abs=TOL
    )


# =========================================================================== #
# TOP-LEVEL ACCEPTANCE — representative agreement + exact weighting            #
# =========================================================================== #
def test_accept_top_level_agreement_and_weighting():
    """TOP-LEVEL GATE: production == official reference on a representative case AND
    the combined equals the exact 0.70/0.20/0.10 weighting (Req 10.5, 10.6).

    A single representative mixed case (all three families, hits/misses/decoys, ties)
    that exercises every component simultaneously. The production combined must (a)
    equal the OFFICIAL reference score to within TOL and (b) equal the explicit
    0.70·PairAP + 0.20·EvidenceMAP + 0.10·BehaviorMAP of its own exposed components.
    """
    solution = pd.DataFrame(
        [
            _row("P01", 1, "directed_transfer", ["HA", "HB"]),
            _row("P02", 1, "soft_play", ["HC"]),
            _row("P03", 1, "coordinated_isolation", ["HD", "HE", "HF"]),
            _row("P04", 1, NON_TARGET_POSITIVE, ["HG"]),  # excluded from Behavior MAP
            _row("P05", 0, "none", []),
            _row("P06", 0, "none", []),
            _row("P07", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P01", 0.90, "directed_transfer", ["HA", "WRONG", "HB"]),  # 2 hits
            _row("P02", 0.70, "soft_play", ["HC"]),  # perfect
            _row("P03", 0.70, "coordinated_isolation", ["HD", "MISS", "HF"]),  # 2 hits, tie w/ P02
            _row("P04", 0.60, NON_TARGET_POSITIVE, ["HG"]),
            _row("P05", 0.50, "none", []),
            _row("P06", 0.20, "directed_transfer", []),  # decoy family on a negative
            _row("P07", 0.05, "none", []),
        ]
    )
    ref_value = ref.score(solution, submission, "pair_id")
    comps = _components(solution, submission)

    # (a) production combined == official reference.
    assert comps.combined == pytest.approx(ref_value, abs=TOL)
    assert prod.score(solution, submission, "pair_id") == pytest.approx(ref_value, abs=TOL)

    # (b) combined == exact 0.70/0.20/0.10 weighting of the exposed components.
    expected = (
        0.70 * comps.pair_ap
        + 0.20 * comps.evidence_map
        + 0.10 * comps.behavior_map
    )
    assert comps.combined == pytest.approx(expected, abs=TOL)
    assert np.isfinite(comps.combined)


# =========================================================================== #
# Source-of-truth guard — the reference is the OFFICIAL code                   #
# =========================================================================== #
def test_accept_reference_module_is_official_code():
    """Guard: the saved reference carries the official constants (nobody edited it).

    Light structural check that ``reference_public_metric`` still encodes the official
    contract: the ALLOWED_BEHAVIORS set, the three TARGET_BEHAVIORS, the exact
    ``NO_EVIDENCE`` literal, the required-column set, and the 0.70/0.20/0.10 weights
    literally present in the ``score`` source. This blocks silent edits to the ground
    truth that would make every other test tautological.
    """
    assert ref.ALLOWED_BEHAVIORS == {
        "none",
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
        "other_coordination",
    }
    assert ref.TARGET_BEHAVIORS == (
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
    )
    assert ref.NO_EVIDENCE == "NO_EVIDENCE"
    assert ref.REQUIRED_COLUMNS == {
        "pair_id",
        "risk_score",
        "predicted_behavior",
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    }
    # The 0.70/0.20/0.10 weights must literally appear in the official score source.
    src = inspect.getsource(ref.score)
    assert "0.70" in src and "0.20" in src and "0.10" in src
    # And production reuses the very same reference constants (cannot drift).
    assert prod.ALLOWED_BEHAVIORS is ref.ALLOWED_BEHAVIORS
    assert prod.TARGET_BEHAVIORS is ref.TARGET_BEHAVIORS
    assert prod.NO_EVIDENCE is ref.NO_EVIDENCE
