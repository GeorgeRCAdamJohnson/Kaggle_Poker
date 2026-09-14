"""Agreement + validation tests for the production three-part metric (task 4.2).

The production metric (:mod:`poker_collusion.metric.production_metric`) is a
vectorized re-implementation of the OFFICIAL metric. These tests assert it AGREES
with the verbatim reference (:mod:`poker_collusion.metric.reference_public_metric`)
to within ~1e-9 on randomized and hand-constructed solution/submission pairs, and
that it enforces the same submission validation (missing column, out-of-range risk,
duplicate pair_id, repeated evidence within a pair).

The exhaustive bit-for-bit reconciliation battery lives in task 4.4; this file only
establishes solid agreement + validation coverage. The reference metric is the
ground truth and is never modified.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from poker_collusion.metric import reference_public_metric as ref
from poker_collusion.metric import production_metric as prod

TOL = 1e-9

BEHAVIORS = sorted(ref.ALLOWED_BEHAVIORS)
TARGET = list(ref.TARGET_BEHAVIORS)


# --------------------------------------------------------------------------- #
# Valid-submission builders                                                   #
# --------------------------------------------------------------------------- #
def _evidence_row(hands: list[str]) -> dict:
    """Fill the 5 evidence columns from a hand list, padding with NO_EVIDENCE."""
    padded = list(hands) + [ref.NO_EVIDENCE] * (5 - len(hands))
    return {f"evidence_hand_{i}": padded[i - 1] for i in range(1, 6)}


def _make_solution(rng: random.Random, n_pairs: int) -> pd.DataFrame:
    """Build a valid private solution with binary labels and planted evidence.

    Ensures at least one positive per target family (so Behavior MAP can vary) and
    a mix of positives/negatives so Pair AP is well defined.
    """
    rows = []
    # Force one positive per target family up front.
    forced = list(TARGET)
    for i in range(n_pairs):
        pair_id = f"P{i:04d}"
        if i < len(forced):
            behavior = forced[i]
            label = 1
        else:
            label = rng.randint(0, 1)
            if label == 1:
                behavior = rng.choice(TARGET + ["other_coordination"])
            else:
                behavior = "none"
        if label == 1:
            n_ev = rng.randint(1, 5)
            hands = [f"{pair_id}_H{k}" for k in range(n_ev)]
        else:
            hands = []
        rows.append(
            {
                "pair_id": pair_id,
                "risk_score": label,
                "predicted_behavior": behavior,
                **_evidence_row(hands),
            }
        )
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def _make_submission(rng: random.Random, solution: pd.DataFrame) -> pd.DataFrame:
    """Build a random VALID submission for the given solution.

    Valid means: same pair_id set (unique), risk in [0, 1], allowed behavior
    labels, and non-repeating evidence hands per pair (NO_EVIDENCE padding). The
    evidence deliberately mixes truly-relevant hands, decoys, and blanks so
    Evidence MAP@5 exercises hits, misses, and short lists.
    """
    truth_by_id = solution.set_index("pair_id")
    rows = []
    for pair_id, trow in truth_by_id.iterrows():
        risk = round(rng.random(), 6)
        behavior = rng.choice(BEHAVIORS)
        # Gather the pair's true evidence to optionally reuse some as hits.
        true_hands = [
            str(trow[f"evidence_hand_{i}"])
            for i in range(1, 6)
            if str(trow[f"evidence_hand_{i}"]).strip()
            and str(trow[f"evidence_hand_{i}"]) != ref.NO_EVIDENCE
        ]
        n_ev = rng.randint(0, 5)
        pool = list(true_hands) + [f"{pair_id}_DECOY{k}" for k in range(5)]
        rng.shuffle(pool)
        # Unique selection to satisfy the no-repeat rule.
        chosen = list(dict.fromkeys(pool))[:n_ev]
        rows.append(
            {
                "pair_id": pair_id,
                "risk_score": risk,
                "predicted_behavior": behavior,
                **_evidence_row(chosen),
            }
        )
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def _tiny_solution() -> pd.DataFrame:
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


# --------------------------------------------------------------------------- #
# Agreement tests                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(25))
def test_production_agrees_with_reference_random(seed: int):
    rng = random.Random(seed)
    n_pairs = rng.randint(5, 60)
    solution = _make_solution(rng, n_pairs)
    submission = _make_submission(rng, solution)

    ref_value = ref.score(solution, submission, "pair_id")
    prod_value = prod.score(solution, submission, "pair_id")
    assert prod_value == pytest.approx(ref_value, abs=TOL)

    # The combined summary from score_components must equal both the reference and
    # the convenience `score`.
    comps = prod.score_components(solution, submission, "pair_id")
    assert comps.combined == pytest.approx(ref_value, abs=TOL)
    assert comps.combined == pytest.approx(prod_value, abs=TOL)
    # Combined must be the exact weighted sum of the exposed components (Req 10.5).
    expected = 0.70 * comps.pair_ap + 0.20 * comps.evidence_map + 0.10 * comps.behavior_map
    assert comps.combined == pytest.approx(expected, abs=TOL)


def test_production_agrees_perfect_submission():
    solution = _tiny_solution()
    submission = solution.copy()
    assert prod.score(solution, submission, "pair_id") == pytest.approx(1.0, abs=TOL)
    assert prod.score(solution, submission, "pair_id") == pytest.approx(
        ref.score(solution, submission, "pair_id"), abs=TOL
    )


def test_production_agrees_all_negative_no_positives():
    """Pair AP and Evidence MAP are 0 with no positives; must match reference."""
    solution = pd.DataFrame(
        [
            {"pair_id": f"P{i}", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])}
            for i in range(6)
        ]
    )
    submission = pd.DataFrame(
        [
            {"pair_id": f"P{i}", "risk_score": round(0.1 * i, 3),
             "predicted_behavior": "none", **_evidence_row([])}
            for i in range(6)
        ]
    )
    prod_c = prod.score_components(solution, submission, "pair_id")
    assert prod_c.pair_ap == 0.0
    assert prod_c.evidence_map == 0.0
    assert prod_c.behavior_map == 0.0
    assert prod_c.combined == pytest.approx(
        ref.score(solution, submission, "pair_id"), abs=TOL
    )


def test_production_agrees_equal_risk_tiebreak():
    """All-equal risk: reference breaks ties by pair_id asc via the stable sort."""
    solution = pd.DataFrame(
        [
            {"pair_id": "P1", "risk_score": 1, "predicted_behavior": "directed_transfer",
             **_evidence_row(["HA"])},
            {"pair_id": "P2", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
            {"pair_id": "P3", "risk_score": 1, "predicted_behavior": "soft_play",
             **_evidence_row(["HB"])},
            {"pair_id": "P4", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
        ]
    )
    submission = pd.DataFrame(
        [
            {"pair_id": p, "risk_score": 0.5, "predicted_behavior": "none",
             **_evidence_row([])}
            for p in ("P1", "P2", "P3", "P4")
        ]
    )
    assert prod.score(solution, submission, "pair_id") == pytest.approx(
        ref.score(solution, submission, "pair_id"), abs=TOL
    )


def test_production_agrees_missed_target_and_other_coordination():
    """Missed positive contributes 0 to Evidence MAP; other_coordination excluded."""
    solution = pd.DataFrame(
        [
            {"pair_id": "P1", "risk_score": 1, "predicted_behavior": "directed_transfer",
             **_evidence_row(["HA", "HB"])},
            {"pair_id": "P2", "risk_score": 1, "predicted_behavior": "other_coordination",
             **_evidence_row(["HC"])},
            {"pair_id": "P3", "risk_score": 1, "predicted_behavior": "soft_play",
             **_evidence_row(["HD"])},
            {"pair_id": "P4", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
        ]
    )
    submission = pd.DataFrame(
        [
            # P1: correct family, but WRONG evidence -> missed positive (0).
            {"pair_id": "P1", "risk_score": 0.9, "predicted_behavior": "directed_transfer",
             **_evidence_row(["WRONG1", "WRONG2"])},
            {"pair_id": "P2", "risk_score": 0.8, "predicted_behavior": "other_coordination",
             **_evidence_row(["HC"])},
            {"pair_id": "P3", "risk_score": 0.7, "predicted_behavior": "soft_play",
             **_evidence_row(["HD"])},
            {"pair_id": "P4", "risk_score": 0.1, "predicted_behavior": "none",
             **_evidence_row([])},
        ]
    )
    assert prod.score(solution, submission, "pair_id") == pytest.approx(
        ref.score(solution, submission, "pair_id"), abs=TOL
    )


# --------------------------------------------------------------------------- #
# Validation tests                                                            #
# --------------------------------------------------------------------------- #
def test_missing_column_raises():
    solution = _tiny_solution()
    submission = solution.drop(columns=["evidence_hand_5"])
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_out_of_range_risk_raises_no_clipping():
    solution = _tiny_solution()
    submission = solution.copy()
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[0, "risk_score"] = 1.5
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_negative_risk_raises():
    solution = _tiny_solution()
    submission = solution.copy()
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[0, "risk_score"] = -0.01
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_duplicate_pair_id_raises():
    solution = _tiny_solution()
    submission = solution.copy()
    submission.loc[submission.index[0], "pair_id"] = submission.loc[
        submission.index[1], "pair_id"
    ]
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_pair_id_coverage_mismatch_raises():
    solution = _tiny_solution()
    submission = solution.copy()
    submission.loc[submission.index[0], "pair_id"] = "P_EXTRA"
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_invalid_behavior_raises():
    solution = _tiny_solution()
    submission = solution.copy()
    submission["predicted_behavior"] = submission["predicted_behavior"].astype(object)
    submission.loc[submission.index[0], "predicted_behavior"] = "totally_made_up"
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_repeated_evidence_within_pair_raises():
    solution = _tiny_solution()
    submission = solution.copy()
    # Repeat a real hand id within P1's evidence slots.
    submission.loc[submission["pair_id"] == "P1", "evidence_hand_1"] = "DUP"
    submission.loc[submission["pair_id"] == "P1", "evidence_hand_2"] = "DUP"
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "pair_id")


def test_wrong_row_id_column_raises():
    solution = _tiny_solution()
    submission = solution.copy()
    with pytest.raises(prod.ParticipantVisibleError):
        prod.score(solution, submission, "not_pair_id")


def test_score_components_returns_dict_and_dataclass():
    solution = _tiny_solution()
    submission = solution.copy()
    comps = prod.score_components(solution, submission, "pair_id")
    d = comps.as_dict()
    assert set(d) == {"pair_ap", "evidence_map", "behavior_map", "combined"}
    assert all(np.isfinite(v) for v in d.values())
