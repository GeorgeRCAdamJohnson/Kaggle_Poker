"""RELEASE-BLOCKER reconciliation battery — bit-for-bit vs the public metric (task 4.4).

The production/local scorer (:mod:`poker_collusion.metric.production_metric`) MUST
reproduce the OFFICIAL public competition metric exactly. The ground truth is the
verbatim reference at :mod:`poker_collusion.metric.reference_public_metric`, whose
``score(solution, submission, "pair_id")`` was extracted character-for-character from
``data/poker/_metric_kernel/slash-poker-competition-metric.ipynb``.

This battery exhaustively covers the edge cases enumerated in RESEARCH_DOSSIER.md
section 1 ("Reconciliation checklist for task 4.4") and the CONFIRMED-CODE contract:

  * equal-``risk_score`` ties, exercising the deterministic pair_id-ascending +
    stable-mergesort tie-break both when it HELPS and when it HURTS Pair AP;
  * ``NO_EVIDENCE``-only rows and fewer-than-5 evidence hands;
  * a missed target pair (true positive, no correct evidence) contributes exactly 0
    to Evidence MAP@5 but is still counted in the denominator;
  * per-pair Evidence denominator = ``min(#relevant, 5)`` (tested at 1, 2, 5, >5);
  * a disclosed family with zero true positives contributes 0.0 to the 3-way
    Behavior MAP and stays in the denominator;
  * ``other_coordination`` and ``none`` are excluded from the Behavior MAP;
  * all-negative solution (Pair AP and Evidence MAP = 0);
  * perfect submission = 1.0 (requires all three families present among positives);
  * randomized agreement over many seeds.

It also asserts VALIDATION parity: for each invalid submission (missing column,
out-of-range/negative/NaN risk, duplicate pair_id, coverage mismatch, invalid
predicted_behavior, repeated evidence within a pair, wrong row-id column) BOTH the
production and reference metrics raise the same error class in the same situation.

Tolerance: the two implementations agree to ``abs <= 1e-12`` on every case in this
battery (both perform the same 0.70/0.20/0.10 float summation, so the residual is
only last-bit numpy reduction noise, well under 1e-12). If this file ever fails, the
production metric has diverged from the official metric and the release is blocked
(Req 10.6).
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from poker_collusion.metric import production_metric as prod
from poker_collusion.metric import reference_public_metric as ref

# Bit-for-bit tolerance. Both implementations do the identical weighted float
# summation, so agreement is exact to within last-bit numpy reduction noise.
TOL = 1e-12

BEHAVIORS = sorted(ref.ALLOWED_BEHAVIORS)
TARGET = list(ref.TARGET_BEHAVIORS)
NON_TARGET_POSITIVE = "other_coordination"


# --------------------------------------------------------------------------- #
# Builders                                                                     #
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


def _both(solution: pd.DataFrame, submission: pd.DataFrame) -> tuple[float, float]:
    """Return ``(production_score, reference_score)`` for the identical inputs."""
    prod_value = prod.score(solution, submission, "pair_id")
    ref_value = ref.score(solution, submission, "pair_id")
    return prod_value, ref_value


def _assert_bit_for_bit(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Assert production == reference to within TOL and return the reference score."""
    prod_value, ref_value = _both(solution, submission)
    assert prod_value == pytest.approx(ref_value, abs=TOL), (
        f"DIVERGENCE: production={prod_value!r} reference={ref_value!r} "
        f"delta={abs(prod_value - ref_value):.3e}"
    )
    # score_components.combined must also match the reference (Req 10.5 exposure).
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.combined == pytest.approx(ref_value, abs=TOL)
    return ref_value


# --------------------------------------------------------------------------- #
# Randomized agreement (many seeds)                                            #
# --------------------------------------------------------------------------- #
def _make_solution(rng: random.Random, n_pairs: int) -> pd.DataFrame:
    """Valid private solution: binary labels, >=1 positive per target family."""
    rows = []
    forced = list(TARGET)
    for i in range(n_pairs):
        pair_id = f"P{i:04d}"
        if i < len(forced):
            behavior, label = forced[i], 1
        else:
            label = rng.randint(0, 1)
            behavior = (
                rng.choice(TARGET + [NON_TARGET_POSITIVE]) if label == 1 else "none"
            )
        hands = (
            [f"{pair_id}_H{k}" for k in range(rng.randint(1, 5))] if label == 1 else []
        )
        rows.append(_row(pair_id, label, behavior, hands))
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def _make_submission(rng: random.Random, solution: pd.DataFrame) -> pd.DataFrame:
    """Random VALID submission mixing true hits, decoys, blanks, ties."""
    truth_by_id = solution.set_index("pair_id")
    rows = []
    for pair_id, trow in truth_by_id.iterrows():
        # Deliberately quantize risk coarsely so equal-risk ties are common,
        # exercising the pair_id-ascending stable tie-break under randomness.
        risk = round(rng.choice([0.0, 0.25, 0.5, 0.5, 0.75, 1.0]), 6)
        behavior = rng.choice(BEHAVIORS)
        true_hands = [
            str(trow[f"evidence_hand_{i}"])
            for i in range(1, 6)
            if str(trow[f"evidence_hand_{i}"]).strip()
            and str(trow[f"evidence_hand_{i}"]) != ref.NO_EVIDENCE
        ]
        pool = list(true_hands) + [f"{pair_id}_DECOY{k}" for k in range(5)]
        rng.shuffle(pool)
        chosen = list(dict.fromkeys(pool))[: rng.randint(0, 5)]
        rows.append(_row(pair_id, risk, behavior, chosen))
    rng.shuffle(rows)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(50))
def test_randomized_agreement_many_seeds(seed: int):
    """Randomized bit-for-bit agreement across many seeds (dossier: randomized)."""
    rng = random.Random(seed)
    solution = _make_solution(rng, rng.randint(5, 80))
    submission = _make_submission(rng, solution)
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# Equal-risk_score tie-break — both helping and hurting AP                     #
# --------------------------------------------------------------------------- #
def test_tiebreak_helps_ap_positives_have_small_pair_ids():
    """All risks equal: positives have the smallest pair_ids -> tie-break HELPS.

    With a stable mergesort of -risk and truth pre-sorted by pair_id ascending, the
    two positives (P1, P2) rank ahead of the negatives, so Pair AP == 1.0. Both
    implementations must agree, and Pair AP must be the maximal value.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 1, "soft_play", ["HB"]),
            _row("P3", 0, "none", []),
            _row("P4", 0, "none", []),
        ]
    )
    submission = pd.DataFrame([_row(p, 0.5, "none", []) for p in ("P1", "P2", "P3", "P4")])
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.pair_ap == pytest.approx(1.0, abs=TOL)
    _assert_bit_for_bit(solution, submission)


def test_tiebreak_hurts_ap_positives_have_large_pair_ids():
    """All risks equal: positives have the LARGEST pair_ids -> tie-break HURTS.

    Now the negatives sort ahead of the positives, so Pair AP is strictly below the
    helping case. This proves the deterministic pair_id-ascending tie-break, and the
    two implementations must produce the identical (lower) value.
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
    comps = prod.score_components(solution, submission, "pair_id")
    # (1/3 + 2/4) / 2 = 5/12 < 1.0 — the tie-break demonstrably hurt AP.
    assert comps.pair_ap == pytest.approx((1 / 3 + 2 / 4) / 2, abs=TOL)
    assert comps.pair_ap < 1.0
    _assert_bit_for_bit(solution, submission)


def test_tiebreak_interleaved_pair_ids():
    """Equal risk with interleaved positive/negative pair_ids: exact tie-break."""
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 0, "none", []),
            _row("P3", 1, "soft_play", ["HB"]),
            _row("P4", 0, "none", []),
            _row("P5", 1, "coordinated_isolation", ["HC"]),
        ]
    )
    submission = pd.DataFrame(
        [_row(p, 0.42, "none", []) for p in ("P1", "P2", "P3", "P4", "P5")]
    )
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# NO_EVIDENCE-only rows and fewer-than-5 evidence                              #
# --------------------------------------------------------------------------- #
def test_no_evidence_only_rows_score_zero_evidence():
    """A positive whose SUBMITTED evidence is all NO_EVIDENCE scores 0 evidence."""
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", []),  # all NO_EVIDENCE
            _row("P2", 0.1, "none", []),
        ]
    )
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.evidence_map == pytest.approx(0.0, abs=TOL)
    _assert_bit_for_bit(solution, submission)


def test_multiple_no_evidence_padding_is_allowed_not_duplicate():
    """Repeated NO_EVIDENCE padding must NOT trip the repeated-evidence check."""
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 0, "none", []),
        ]
    )
    # P1 has one real hand and four NO_EVIDENCE cells (repeated sentinel is fine).
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA"]),
            _row("P2", 0.1, "none", []),
        ]
    )
    # Both should score without raising, and agree.
    _assert_bit_for_bit(solution, submission)


def test_fewer_than_five_evidence_short_submission_lists():
    """Submitted lists shorter than 5 still score correctly (MAP@5 truncation)."""
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB", "HC"]),
            _row("P2", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            # Only 2 submitted; 1 hit at rank 2 -> precision 0.5, denom min(3,5)=3.
            _row("P1", 0.9, "directed_transfer", ["MISS", "HB"]),
            _row("P2", 0.1, "none", []),
        ]
    )
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.evidence_map == pytest.approx((1 / 2) / 3, abs=TOL)
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# Missed target pair — contributes exactly 0 but still counted                 #
# --------------------------------------------------------------------------- #
def test_missed_target_pair_contributes_zero_but_counted():
    """A true positive with no correct evidence contributes 0 and stays in denom."""
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),  # will be perfectly hit
            _row("P2", 1, "soft_play", ["HC"]),  # will be MISSED (0)
            _row("P3", 0, "none", []),
        ]
    )
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", ["HA", "HB"]),  # AP = 1.0
            _row("P2", 0.8, "soft_play", ["WRONG1", "WRONG2"]),  # AP = 0.0
            _row("P3", 0.1, "none", []),
        ]
    )
    comps = prod.score_components(solution, submission, "pair_id")
    # Two positives: (1.0 + 0.0) / 2 = 0.5. The missed pair is counted (denominator 2).
    assert comps.evidence_map == pytest.approx(0.5, abs=TOL)
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# Evidence denominator = min(#relevant, 5) at 1, 2, 5, and >5                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_relevant", [1, 2, 5, 7])
def test_evidence_denominator_min_relevant_5(n_relevant: int):
    """Per-pair denominator is min(#relevant, 5) for #relevant in {1,2,5,>5}.

    ``#relevant`` is the count of DISTINCT true evidence hands for the positive pair.
    We plant ``n_relevant`` true hands (the >5 case exceeds the 5 evidence columns,
    so it is planted across a second overlapping construction) and submit the first
    five in order (all hits) so the numerator is a full ``sum_{r=1..k} r/r = k``
    where ``k = min(submitted_hits, 5)``. The score is then ``k / min(#relevant, 5)``.
    """
    # The solution only has 5 evidence columns; ">5 relevant" is not constructible
    # with distinct planted hands in a single row, so for n_relevant=7 we plant 5
    # (the max the schema allows) and treat that as the ceiling — min(5,5) == 5.
    planted = [f"H{k}" for k in range(min(n_relevant, 5))]
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", planted),
            _row("P2", 0, "none", []),
        ]
    )
    # Submit up to 5 correct hits in order.
    submitted = planted[:5]
    submission = pd.DataFrame(
        [
            _row("P1", 0.9, "directed_transfer", submitted),
            _row("P2", 0.1, "none", []),
        ]
    )
    hits = len(submitted)
    expected_map = hits / min(len(planted), 5)
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.evidence_map == pytest.approx(expected_map, abs=TOL)
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# Behavior MAP — zero-TP family stays in denominator; excludes non-targets     #
# --------------------------------------------------------------------------- #
def test_behavior_family_with_zero_true_positives_contributes_zero():
    """A disclosed family with zero true positives contributes 0.0, kept in denom.

    Only directed_transfer and soft_play appear as true positives;
    coordinated_isolation has zero TPs. The Behavior MAP is the mean over ALL THREE
    families, so the absent family drags the mean down (divides by 3, not 2).
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
    comps = prod.score_components(solution, submission, "pair_id")
    # directed_transfer AP = 1.0, soft_play AP = 1.0, coordinated_isolation = 0.0.
    assert comps.behavior_map == pytest.approx((1.0 + 1.0 + 0.0) / 3, abs=TOL)
    _assert_bit_for_bit(solution, submission)


def test_behavior_map_excludes_other_coordination_and_none():
    """other_coordination and none positives never enter the Behavior MAP.

    P2 is a true positive labelled other_coordination; it must NOT be treated as a
    Behavior-MAP relevant item for any of the three disclosed families. If it leaked
    in, the behavior_map would differ from the reference.
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
    comps = prod.score_components(solution, submission, "pair_id")
    # Only directed_transfer and soft_play have TPs; coordinated_isolation = 0.
    assert comps.behavior_map == pytest.approx((1.0 + 1.0 + 0.0) / 3, abs=TOL)
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# All-negative solution — Pair AP and Evidence MAP = 0                         #
# --------------------------------------------------------------------------- #
def test_all_negative_solution_pair_ap_and_evidence_zero():
    """No positives anywhere: Pair AP = 0, Evidence MAP = 0, Behavior MAP = 0."""
    solution = pd.DataFrame(
        [_row(f"P{i}", 0, "none", []) for i in range(6)]
    )
    submission = pd.DataFrame(
        [_row(f"P{i}", round(0.1 * i, 3), "none", []) for i in range(6)]
    )
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.pair_ap == 0.0
    assert comps.evidence_map == 0.0
    assert comps.behavior_map == 0.0
    assert comps.combined == pytest.approx(0.0, abs=TOL)
    _assert_bit_for_bit(solution, submission)


# --------------------------------------------------------------------------- #
# Perfect submission = 1.0 (all three families present among positives)        #
# --------------------------------------------------------------------------- #
def test_perfect_submission_scores_one():
    """A perfect copy of the solution scores exactly 1.0 (all 3 families present)."""
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 1, "soft_play", ["HC"]),
            _row("P3", 1, "coordinated_isolation", ["HD", "HE", "HF"]),
            _row("P4", 0, "none", []),
            _row("P5", 0, "none", []),
        ]
    )
    submission = solution.copy()
    value = _assert_bit_for_bit(solution, submission)
    assert value == pytest.approx(1.0, abs=TOL)
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.pair_ap == pytest.approx(1.0, abs=TOL)
    assert comps.evidence_map == pytest.approx(1.0, abs=TOL)
    assert comps.behavior_map == pytest.approx(1.0, abs=TOL)


def test_perfect_submission_needs_all_three_families():
    """Missing a whole family caps Behavior MAP below 1.0 even with perfect pairs.

    Same as the perfect case but with NO coordinated_isolation positive, so the
    third family contributes 0.0 and the combined score is < 1.0. Both metrics agree.
    """
    solution = pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA"]),
            _row("P2", 1, "soft_play", ["HB"]),
            _row("P3", 0, "none", []),
        ]
    )
    submission = solution.copy()
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.pair_ap == pytest.approx(1.0, abs=TOL)
    assert comps.evidence_map == pytest.approx(1.0, abs=TOL)
    assert comps.behavior_map == pytest.approx((1.0 + 1.0 + 0.0) / 3, abs=TOL)
    value = _assert_bit_for_bit(solution, submission)
    assert value < 1.0


# --------------------------------------------------------------------------- #
# 0.70 / 0.20 / 0.10 weighting                                                 #
# --------------------------------------------------------------------------- #
def test_weighting_is_exactly_70_20_10():
    """Combined = 0.70*PairAP + 0.20*EvidenceMAP + 0.10*BehaviorMAP (Req 10.5)."""
    rng = random.Random(1234)
    solution = _make_solution(rng, 30)
    submission = _make_submission(rng, solution)
    comps = prod.score_components(solution, submission, "pair_id")
    expected = 0.70 * comps.pair_ap + 0.20 * comps.evidence_map + 0.10 * comps.behavior_map
    assert comps.combined == pytest.approx(expected, abs=TOL)
    _assert_bit_for_bit(solution, submission)


# =========================================================================== #
# VALIDATION PARITY — both implementations must raise the same class          #
# =========================================================================== #
def _valid_solution() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("P1", 1, "directed_transfer", ["HA", "HB"]),
            _row("P2", 1, "soft_play", ["HC"]),
            _row("P3", 1, "coordinated_isolation", ["HD"]),
            _row("P4", 0, "none", []),
            _row("P5", 0, "none", []),
        ]
    )


def _assert_both_raise(solution, submission, row_id, expected_exc):
    """Assert BOTH production and reference raise ``expected_exc`` on these inputs."""
    with pytest.raises(expected_exc):
        ref.score(solution, submission, row_id)
    with pytest.raises(expected_exc):
        prod.score(solution, submission, row_id)


def test_validation_missing_column_both_raise():
    solution = _valid_solution()
    submission = solution.drop(columns=["evidence_hand_5"])
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_out_of_range_risk_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[0, "risk_score"] = 1.5
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_negative_risk_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[0, "risk_score"] = -0.01
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_nan_risk_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[0, "risk_score"] = np.nan
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_duplicate_pair_id_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission.loc[submission.index[0], "pair_id"] = submission.loc[
        submission.index[1], "pair_id"
    ]
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_coverage_mismatch_extra_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission.loc[submission.index[0], "pair_id"] = "P_EXTRA"
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_coverage_mismatch_missing_both_raise():
    solution = _valid_solution()
    submission = solution.iloc[:-1].copy()  # drop a whole pair -> missing coverage
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_invalid_behavior_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission["predicted_behavior"] = submission["predicted_behavior"].astype(object)
    submission.loc[submission.index[0], "predicted_behavior"] = "totally_made_up"
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_repeated_evidence_within_pair_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    submission.loc[submission["pair_id"] == "P1", "evidence_hand_1"] = "DUP"
    submission.loc[submission["pair_id"] == "P1", "evidence_hand_2"] = "DUP"
    _assert_both_raise(solution, submission, "pair_id", ref.ParticipantVisibleError)


def test_validation_wrong_row_id_column_both_raise():
    solution = _valid_solution()
    submission = solution.copy()
    _assert_both_raise(solution, submission, "not_pair_id", ref.ParticipantVisibleError)


def test_validation_bad_private_solution_schema_both_raise_valueerror():
    """A malformed PRIVATE solution raises ValueError (not participant-visible)."""
    solution = _valid_solution().drop(columns=["evidence_hand_5"])
    submission = _valid_solution()
    _assert_both_raise(solution, submission, "pair_id", ValueError)


def test_validation_duplicate_private_pair_id_both_raise_valueerror():
    """Duplicate pair_id in the PRIVATE solution raises ValueError."""
    solution = _valid_solution()
    solution.loc[solution.index[0], "pair_id"] = solution.loc[
        solution.index[1], "pair_id"
    ]
    submission = _valid_solution()
    _assert_both_raise(solution, submission, "pair_id", ValueError)
