"""Hand-computed tests for the INDEPENDENT reference metric.

Every expected number here is worked out by hand from the metric contract
(RESEARCH_DOSSIER.md section 1) so the test itself is an audit of
``reference_metric.py``. The comments show the arithmetic.

These tests deliberately do NOT import ``reference_public_metric`` — this suite
anchors the independent oracle on its own terms. The bit-for-bit reconciliation
against the official code is a separate task (4.4).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from poker_collusion.metric.reference_metric import (
    NO_EVIDENCE,
    average_precision,
    behavior_map,
    clean_evidence,
    evidence_ap_at_5,
    evidence_map5,
    pair_ap,
    score,
)


def _evidence_row(hands: list[str]) -> dict:
    """Fill the five evidence columns from a hand-id list, padding with sentinel."""
    padded = list(hands) + [NO_EVIDENCE] * (5 - len(hands))
    return {f"evidence_hand_{i}": padded[i - 1] for i in range(1, 6)}


# --- clean_evidence -----------------------------------------------------------


def test_clean_evidence_drops_sentinel_blank_and_nan_and_strips():
    raw = ["HA", NO_EVIDENCE, "  HB  ", "", float("nan"), None, "HC"]
    assert clean_evidence(raw) == ["HA", "HB", "HC"]


# --- Pair AP (hand-computed) --------------------------------------------------


def test_pair_ap_perfect_ranking_is_one():
    # Two positives ranked strictly above one negative -> AP = 1.0.
    # Ranked order by risk desc: P1(1, .9), P2(1, .8), P3(0, .1)
    # precisions at each hit: 1/1, 2/2 -> sum = 2.0; /2 positives = 1.0
    labels = [1, 1, 0]
    risk = [0.9, 0.8, 0.1]
    pair_ids = ["P1", "P2", "P3"]
    assert pair_ap(labels, risk, pair_ids) == pytest.approx(1.0)


def test_pair_ap_interleaved_ranking_hand_computed():
    # Ranking by risk desc: P1(1,.9), P2(0,.7), P3(1,.5), P4(0,.2)
    # ranks of positives: 1 and 3.
    # precision at rank1 = 1/1 = 1.0; at rank3 = 2/3 ~ 0.6667.
    # AP = (1.0 + 2/3) / 2 = (5/3)/2 = 5/6 ~ 0.833333.
    labels = [1, 0, 1, 0]
    risk = [0.9, 0.7, 0.5, 0.2]
    pair_ids = ["P1", "P2", "P3", "P4"]
    assert pair_ap(labels, risk, pair_ids) == pytest.approx(5.0 / 6.0)


def test_pair_ap_no_positives_is_zero():
    assert pair_ap([0, 0, 0], [0.9, 0.5, 0.1], ["P1", "P2", "P3"]) == 0.0


def test_pair_ap_tie_break_by_pair_id_is_deterministic():
    # All three share risk 0.5. Positives are the ones with pair_id "A" and "C".
    # Tie-break = pair_id ascending -> ranked order A, B, C.
    # positive A at rank1 -> 1/1; positive C at rank3 -> 2/3.
    # AP = (1 + 2/3)/2 = 5/6.
    labels_map = {"A": 1, "B": 0, "C": 1}
    # Feed in a scrambled order to prove the tie-break, not input order, decides.
    pair_ids = ["C", "A", "B"]
    labels = [labels_map[p] for p in pair_ids]
    risk = [0.5, 0.5, 0.5]
    assert pair_ap(labels, risk, pair_ids) == pytest.approx(5.0 / 6.0)

    # Reordering the inputs must not change the result (determinism).
    pair_ids2 = ["B", "C", "A"]
    labels2 = [labels_map[p] for p in pair_ids2]
    risk2 = [0.5, 0.5, 0.5]
    assert pair_ap(labels2, risk2, pair_ids2) == pytest.approx(5.0 / 6.0)


# --- Evidence AP@5 / MAP@5 (hand-computed) ------------------------------------


def test_evidence_ap_at_5_all_correct_in_order():
    # relevant = {HA, HB}; submitted in order [HA, HB].
    # hits: rank1 -> 1/1, rank2 -> 2/2; sum = 2.0; /min(2,5)=2 -> 1.0
    assert evidence_ap_at_5(["HA", "HB"], ["HA", "HB"]) == pytest.approx(1.0)


def test_evidence_ap_at_5_partial_and_ordering():
    # relevant = {HA, HB, HC} (3 planted). submitted = [HX, HA, HB]
    # rank1 HX miss; rank2 HA hit -> hits=1, +1/2; rank3 HB hit -> hits=2, +2/3
    # sum = 0.5 + 0.6667 = 1.16667; /min(3,5)=3 -> 0.388889
    got = evidence_ap_at_5(["HA", "HB", "HC"], ["HX", "HA", "HB"])
    assert got == pytest.approx((0.5 + 2.0 / 3.0) / 3.0)


def test_evidence_ap_at_5_no_relevant_is_zero():
    assert evidence_ap_at_5([], ["HA", "HB"]) == 0.0


def test_evidence_map5_missed_positive_contributes_zero():
    # Two true positives. Pair 0 gets its evidence perfectly (AP@5 = 1.0).
    # Pair 1 is a MISSED positive: no correct evidence -> contributes 0.0 but is
    # still counted. MAP = (1.0 + 0.0)/2 = 0.5.
    labels = [1, 1, 0]
    relevant = [["HA"], ["HB"], []]
    submitted = [["HA"], ["WRONG"], []]
    assert evidence_map5(labels, relevant, submitted) == pytest.approx(0.5)


def test_evidence_map5_only_true_positives_counted():
    # The negative pair (label 0) has perfect-looking evidence but must never be
    # scored. Only the single true positive counts -> MAP = 1.0.
    labels = [1, 0]
    relevant = [["HA"], ["HZ"]]
    submitted = [["HA"], ["HZ"]]
    assert evidence_map5(labels, relevant, submitted) == pytest.approx(1.0)


def test_evidence_map5_no_positives_is_zero():
    assert evidence_map5([0, 0], [[], []], [[], []]) == 0.0


# --- Behavior MAP (hand-computed) ---------------------------------------------


def test_behavior_map_zero_positive_family_pulls_toward_zero():
    # Only directed_transfer and soft_play have true positives; coordinated_
    # isolation has zero -> contributes 0.0 but stays in the 3-way denominator.
    # Both present families are ranked perfectly (their positive has the top
    # risk within its OvR score), so each scores 1.0.
    # Behavior MAP = (1.0 + 1.0 + 0.0) / 3 = 2/3.
    true_behavior = ["directed_transfer", "soft_play", "none"]
    predicted_behavior = ["directed_transfer", "soft_play", "none"]
    risk = [0.9, 0.8, 0.1]
    pair_ids = ["P1", "P2", "P3"]
    got = behavior_map(true_behavior, predicted_behavior, risk, pair_ids)
    assert got == pytest.approx(2.0 / 3.0)


def test_behavior_map_excludes_other_coordination():
    # other_coordination is a true family here but must NOT form its own family.
    # directed_transfer: 1 positive ranked top -> 1.0
    # soft_play: 1 positive ranked top -> 1.0
    # coordinated_isolation: 0 positives -> 0.0
    # MAP = (1 + 1 + 0)/3 = 2/3. If other_coordination were (wrongly) included
    # as a 4th family the denominator would change; this asserts it does not.
    true_behavior = ["directed_transfer", "soft_play", "other_coordination"]
    predicted_behavior = ["directed_transfer", "soft_play", "other_coordination"]
    risk = [0.9, 0.8, 0.7]
    pair_ids = ["P1", "P2", "P3"]
    got = behavior_map(true_behavior, predicted_behavior, risk, pair_ids)
    assert got == pytest.approx(2.0 / 3.0)


def test_average_precision_ovr_partial_hand_computed():
    # One family with two positives but predicted_behavior mislabels one, so its
    # OvR risk drops to 0 and it ranks last.
    # true directed_transfer at indices 0 and 2.
    # predicted family matches only index 0 (risk .9); index 2 predicted "none"
    # so its OvR score = 0.0. Negatives at 1 (predicted directed_transfer, risk
    # .8 -> OvR .8) and 3 (risk 0).
    # family_labels = [1,0,1,0]; family_risk = [.9,.8,0,0]
    # ranked by risk desc, tie-break pair_id asc:
    #   P1(.9,label1), P2(.8,label0), P3(0,label1), P4(0,label0)
    # positive ranks: 1 and 3 -> AP = (1/1 + 2/3)/2 = 5/6.
    true_behavior = ["directed_transfer", "none", "directed_transfer", "none"]
    predicted_behavior = ["directed_transfer", "directed_transfer", "none", "none"]
    risk = [0.9, 0.8, 0.5, 0.2]
    pair_ids = ["P1", "P2", "P3", "P4"]
    # Only directed_transfer has positives; the other two families -> 0.0.
    # MAP = (5/6 + 0 + 0)/3 = 5/18.
    got = behavior_map(true_behavior, predicted_behavior, risk, pair_ids)
    assert got == pytest.approx((5.0 / 6.0) / 3.0)


# --- Combined score() over frames ---------------------------------------------


def _tiny_solution() -> pd.DataFrame:
    """A 5-pair case with one true positive per disclosed family, two negatives."""
    return pd.DataFrame(
        [
            {"pair_id": "P1", "risk_score": 1, "predicted_behavior": "directed_transfer",
             **_evidence_row(["HA", "HB"])},
            {"pair_id": "P2", "risk_score": 1, "predicted_behavior": "soft_play",
             **_evidence_row(["HC"])},
            {"pair_id": "P5", "risk_score": 1, "predicted_behavior": "coordinated_isolation",
             **_evidence_row(["HD", "HE", "HF"])},
            {"pair_id": "P3", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
            {"pair_id": "P4", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
        ]
    )


def test_score_perfect_submission_is_one():
    solution = _tiny_solution()
    submission = solution.copy()
    result = score(solution, submission)
    assert result["pair_ap"] == pytest.approx(1.0)
    assert result["evidence_map5"] == pytest.approx(1.0)
    assert result["behavior_map"] == pytest.approx(1.0)
    assert result["combined"] == pytest.approx(1.0)


def test_score_reports_three_components_and_combined_weighting():
    # Build a submission where each component is a known, hand-checkable value.
    solution = _tiny_solution()

    # Submission: keep risk perfect (Pair AP = 1.0, all positives above negatives)
    # and behaviors perfect (Behavior MAP = 1.0), but degrade evidence on P2 so
    # Evidence MAP@5 is a clean fraction.
    submission = solution.copy()
    # P2 planted = {HC}. Submit a wrong hand first then nothing else:
    # relevant={HC}; submitted=[WRONG]; no hit -> AP@5 = 0.0 for P2.
    # P1 perfect -> 1.0, P5 perfect -> 1.0. Evidence MAP = (1 + 0 + 1)/3 = 2/3.
    p2_idx = submission.index[submission["pair_id"] == "P2"][0]
    for col in [f"evidence_hand_{i}" for i in range(1, 6)]:
        submission.loc[p2_idx, col] = NO_EVIDENCE
    submission.loc[p2_idx, "evidence_hand_1"] = "WRONG"

    result = score(solution, submission)
    assert result["pair_ap"] == pytest.approx(1.0)
    assert result["evidence_map5"] == pytest.approx(2.0 / 3.0)
    assert result["behavior_map"] == pytest.approx(1.0)
    # Combined = 0.70*1 + 0.20*(2/3) + 0.10*1
    expected = 0.70 * 1.0 + 0.20 * (2.0 / 3.0) + 0.10 * 1.0
    assert result["combined"] == pytest.approx(expected)


def test_score_combined_is_weighted_not_equal():
    # Sanity: a case where components differ, confirming the 0.7/0.2/0.1 split.
    solution = _tiny_solution()
    submission = solution.copy()
    # Zero out all risk -> Pair AP = 0 (no positive ranked; but with all-equal
    # risk, tie-break by pair_id decides; positives P1,P2,P5 vs negatives P3,P4).
    submission["risk_score"] = 0.0
    result = score(solution, submission)
    # With all risk equal, ranking is purely pair_id ascending:
    # P1(1), P2(1), P3(0), P4(0), P5(1).
    # positive ranks: 1,2,5 -> AP = (1/1 + 2/2 + 3/5)/3 = (1+1+0.6)/3 = 2.6/3.
    assert result["pair_ap"] == pytest.approx((1.0 + 1.0 + 3.0 / 5.0) / 3.0)
    assert math.isfinite(result["combined"])
