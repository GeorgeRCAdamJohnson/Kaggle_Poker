"""Complementary edge-case tests for EDA/label-audit REPORTING (task 8.2, Req 2.2-2.4).

Task 8.1 (:mod:`poker_collusion.tests.test_eda`) already covers the *core* fixture
cases: the basic PU partition, the disclosed-only family distribution, a single
undisclosed-positive flag flip, valid/non-shared evidence, leakage checks, the
end-to-end run and the artifact round-trip.

This file adds the AUDIT-REPORTING ROBUSTNESS edge cases that 8.1 did NOT cover, all on
tiny hermetic frames (no real data, no DataLoader, no ``actions.parquet``):

  * PU counts (Req 2.2): labels with ONLY positives, ONLY negatives, EMPTY labels, an
    unexpected/extra ``label_status`` token counted as unknown *alongside* eval pairs,
    and confirmation that evaluation pairs all count as unknown (never negative) — with
    the prevalence arithmetic asserted in each case.
  * Family distribution (Req 2.3): all three disclosed families present in DIFFERENT
    proportions; a positive with an undisclosed family bucketed as ``other`` that flips
    ``other_coordination_absent_in_public`` to False while the disclosed counts survive;
    the zero-positive case.
  * Evidence-validity reporting (Req 2.4): a MIX of valid and invalid evidence rows
    (a hand missing one player, and a hand absent from ``seats`` entirely), asserting
    valid/invalid counts, the ``violations`` list contents, CONFIRMED vs REFUTED status,
    the behavior-action clause reported as un-computable, and the data-absent (empty
    evidence) PENDING path.

None of these duplicate an 8.1 test name or body; they exercise distinct inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.discovery.label_evidence_audit import PENDING
from poker_collusion.features.eda import (
    audit_evidence_validity,
    compute_family_distribution,
    compute_pu_counts,
)


# --------------------------------------------------------------------------- #
# Small builders
# --------------------------------------------------------------------------- #
def _labels(statuses, families=None, pairs=None, players=None):
    """Build a minimal development_labels-shaped frame."""
    n = len(statuses)
    families = families if families is not None else ["none"] * n
    pairs = pairs if pairs is not None else [f"P{i}" for i in range(n)]
    data = {
        "pair_id": pairs,
        "label_status": statuses,
        "behavior_family": families,
    }
    if players is not None:
        data["player_1"] = [p[0] for p in players]
        data["player_2"] = [p[1] for p in players]
    return pd.DataFrame(data)


def _eval_pairs(n):
    return pd.DataFrame({"pair_id": [f"E{i}" for i in range(n)]})


# --------------------------------------------------------------------------- #
# 2.2 PU counts — edge cases (only-pos, only-neg, empty, extra token, eval-only)
# --------------------------------------------------------------------------- #
def test_pu_counts_only_positives_prevalence_is_one() -> None:
    labels = _labels(["confirmed_target", "confirmed_target"],
                     ["soft_play", "directed_transfer"])
    pu = compute_pu_counts(labels, None)
    assert pu["trusted_positive"] == 2
    assert pu["confirmed_negative"] == 0
    assert pu["unknown"] == 0
    assert pu["unknown_from_evaluation_pairs"] == 0
    assert pu["unknown_from_unlabelled_status"] == 0
    assert pu["labelled_positive_prevalence"] == pytest.approx(1.0)


def test_pu_counts_only_negatives_prevalence_is_zero() -> None:
    labels = _labels(["confirmed_non_target"] * 3, ["none", "none", "none"])
    pu = compute_pu_counts(labels, None)
    assert pu["trusted_positive"] == 0
    assert pu["confirmed_negative"] == 3
    assert pu["unknown"] == 0
    assert pu["labelled_positive_prevalence"] == pytest.approx(0.0)


def test_pu_counts_empty_labels_no_division_error() -> None:
    empty = _labels([], [])
    pu = compute_pu_counts(empty, None)
    assert pu["trusted_positive"] == 0
    assert pu["confirmed_negative"] == 0
    assert pu["unknown"] == 0
    assert pu["labelled_rows"] == 0
    # Empty labels must not raise on the prevalence division.
    assert pu["labelled_positive_prevalence"] == pytest.approx(0.0)


def test_pu_counts_empty_labels_with_eval_pairs_all_unknown() -> None:
    empty = _labels([], [])
    pu = compute_pu_counts(empty, _eval_pairs(5))
    # No labelled rows, but the 5 eval pairs are all unknown (never negative).
    assert pu["trusted_positive"] == 0
    assert pu["confirmed_negative"] == 0
    assert pu["unknown"] == 5
    assert pu["unknown_from_evaluation_pairs"] == 5
    assert pu["unknown_from_unlabelled_status"] == 0


def test_pu_counts_extra_token_counted_as_unknown_with_eval_pairs() -> None:
    # An unexpected label_status token must fall into unknown, added to eval-pair unknowns.
    labels = _labels(
        ["confirmed_target", "confirmed_non_target", "under_review", "weird_token"],
        ["soft_play", "none", "x", "y"],
    )
    pu = compute_pu_counts(labels, _eval_pairs(3))
    assert pu["trusted_positive"] == 1
    assert pu["confirmed_negative"] == 1
    # 2 unrecognised-status rows + 3 eval pairs = 5 unknown.
    assert pu["unknown"] == 5
    assert pu["unknown_from_unlabelled_status"] == 2
    assert pu["unknown_from_evaluation_pairs"] == 3
    # Prevalence is over labelled rows only (1 positive / 4 labelled rows).
    assert pu["labelled_positive_prevalence"] == pytest.approx(1 / 4)


def test_pu_counts_evaluation_pairs_never_reduce_negatives() -> None:
    # Sanity: eval pairs only ever add to unknown; they never become negatives.
    labels = _labels(["confirmed_target"], ["soft_play"])
    pu = compute_pu_counts(labels, _eval_pairs(10))
    assert pu["confirmed_negative"] == 0
    assert pu["unknown"] == 10
    assert pu["unknown_from_evaluation_pairs"] == 10


# --------------------------------------------------------------------------- #
# 2.3 family distribution — different proportions, undisclosed bucketing, zero-positive
# --------------------------------------------------------------------------- #
def test_family_distribution_all_three_in_different_proportions() -> None:
    labels = _labels(
        ["confirmed_target"] * 6 + ["confirmed_non_target"],
        [
            "directed_transfer",
            "soft_play", "soft_play",
            "coordinated_isolation", "coordinated_isolation", "coordinated_isolation",
            "none",  # negative row, must be ignored by the distribution
        ],
    )
    fam = compute_family_distribution(labels)
    assert fam["n_positive"] == 6
    assert fam["by_family"] == {
        "directed_transfer": 1,
        "soft_play": 2,
        "coordinated_isolation": 3,
    }
    assert fam["other_or_undisclosed"] == 0
    assert fam["other_coordination_absent_in_public"] is True


def test_family_distribution_undisclosed_family_buckets_to_other_and_flips_flag() -> None:
    # A positive tagged other_coordination is bucketed as `other` and flips the flag,
    # while the disclosed-family counts are still reported correctly.
    labels = _labels(
        ["confirmed_target"] * 3,
        ["directed_transfer", "soft_play", "other_coordination"],
    )
    fam = compute_family_distribution(labels)
    assert fam["n_positive"] == 3
    assert fam["by_family"] == {
        "directed_transfer": 1,
        "soft_play": 1,
        "coordinated_isolation": 0,
    }
    assert fam["other_or_undisclosed"] == 1
    assert fam["other_coordination_absent_in_public"] is False


def test_family_distribution_zero_positive() -> None:
    labels = _labels(["confirmed_non_target", "confirmed_non_target"], ["none", "none"])
    fam = compute_family_distribution(labels)
    assert fam["n_positive"] == 0
    assert fam["by_family"] == {
        "directed_transfer": 0,
        "soft_play": 0,
        "coordinated_isolation": 0,
    }
    assert fam["other_or_undisclosed"] == 0
    # Vacuously true: no positives => no undisclosed positives.
    assert fam["other_coordination_absent_in_public"] is True


# --------------------------------------------------------------------------- #
# 2.4 evidence-validity reporting — mixed valid/invalid, absent hand, PENDING path
# --------------------------------------------------------------------------- #
def test_evidence_validity_mixed_valid_and_two_invalid_kinds() -> None:
    # PPOS = (UA, UB). Evidence rows:
    #   H2 -> both seated (VALID)
    #   H5 -> only UA seated (INVALID: hand missing one player)
    #   H9 -> not in seats at all (INVALID: hand absent entirely)
    labels = _labels(
        ["confirmed_target"], ["soft_play"], pairs=["PPOS"], players=[("UA", "UB")]
    )
    seats = pd.DataFrame(
        {
            "hand_id": ["H2", "H2", "H5"],
            "player_id": ["UA", "UB", "UA"],
        }
    )
    evidence = pd.DataFrame(
        {
            "pair_id": ["PPOS", "PPOS", "PPOS"],
            "evidence_rank": [1, 2, 3],
            "hand_id": ["H2", "H5", "H9"],
            "behavior_family": ["soft_play"] * 3,
        }
    )
    res = audit_evidence_validity(evidence, labels, seats)

    assert res["data_present"] is True
    assert res["checkable"] is True
    assert res["rows_checked"] == 3
    assert res["valid_shared_hand"] == 1
    assert res["invalid"] == 2
    assert res["status"] == "REFUTED"

    # violations list content: both invalid hands present, with correct diagnostics.
    viols = {v["hand_id"]: v for v in res["violations"]}
    assert set(viols) == {"H5", "H9"}
    # H5: the hand exists in seats (UA seated) but not both players.
    assert viols["H5"]["hand_exists"] is True
    assert viols["H5"]["both_players_seated"] is False
    # H9: absent from seats entirely.
    assert viols["H9"]["hand_exists"] is False
    assert viols["H9"]["both_players_seated"] is False
    assert res["n_violations"] == 2

    # The behavior-action clause is reported as un-computable from public data.
    assert "un-computable" in res["behavior_action_clause"]


def test_evidence_validity_all_valid_reports_confirmed() -> None:
    labels = _labels(
        ["confirmed_target"], ["directed_transfer"], pairs=["PX"], players=[("PA", "PB")]
    )
    seats = pd.DataFrame(
        {"hand_id": ["HA", "HA", "HB", "HB"], "player_id": ["PA", "PB", "PA", "PB"]}
    )
    evidence = pd.DataFrame(
        {
            "pair_id": ["PX", "PX"],
            "evidence_rank": [1, 2],
            "hand_id": ["HA", "HB"],
            "behavior_family": ["directed_transfer", "directed_transfer"],
        }
    )
    res = audit_evidence_validity(evidence, labels, seats)
    assert res["valid_shared_hand"] == 2
    assert res["invalid"] == 0
    assert res["n_violations"] == 0
    assert res["status"] == "CONFIRMED"


def test_evidence_validity_empty_evidence_is_data_absent_pending() -> None:
    labels = _labels(
        ["confirmed_target"], ["soft_play"], pairs=["PPOS"], players=[("UA", "UB")]
    )
    seats = pd.DataFrame({"hand_id": ["H2", "H2"], "player_id": ["UA", "UB"]})
    empty_evidence = pd.DataFrame(
        columns=["pair_id", "evidence_rank", "hand_id", "behavior_family"]
    )
    res = audit_evidence_validity(empty_evidence, labels, seats)
    assert res["data_present"] is False
    assert res["status"] == PENDING
    # The behavior-action clause note is always present, even on the absent-data path.
    assert "un-computable" in res["behavior_action_clause"]
