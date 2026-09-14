"""Unit tests for the verbatim official-metric reference (task 4.4 seed).

These tests do not re-derive the metric; they only assert the extracted
``score()`` is importable and exhibits the two anchor properties every later
reconciliation test builds on:

* a well-formed solution/submission scores to a finite value in ``[0, 1]``;
* a perfect submission (risk 1.0 on true positives / 0.0 on negatives, correct
  behavior labels, and exactly the planted evidence hands) scores 1.0.

Source of truth: ``data/poker/_metric_kernel/slash-poker-competition-metric.ipynb``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from poker_collusion.metric.reference_public_metric import (
    NO_EVIDENCE,
    ParticipantVisibleError,
    score,
)


def _evidence_row(hands: list[str]) -> dict:
    """Fill the five evidence columns from a hand-id list, padding with sentinel."""
    padded = list(hands) + [NO_EVIDENCE] * (5 - len(hands))
    return {f"evidence_hand_{i}": padded[i - 1] for i in range(1, 6)}


def _tiny_case():
    """A 5-pair case: one true positive per disclosed family, two negatives.

    All three ``TARGET_BEHAVIORS`` are represented among the positives so a
    perfect submission can reach 1.0 (an absent family would otherwise pin
    Behavior MAP below 1 by contributing 0.0 to the 3-way mean).
    """
    solution = pd.DataFrame(
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
    return solution


def test_score_returns_finite_value_in_unit_interval():
    solution = _tiny_case()
    # A mediocre-but-valid submission: reversed-ish risk, wrong evidence order.
    submission = pd.DataFrame(
        [
            {"pair_id": "P1", "risk_score": 0.6, "predicted_behavior": "directed_transfer",
             **_evidence_row(["HB", "HX"])},
            {"pair_id": "P2", "risk_score": 0.4, "predicted_behavior": "soft_play",
             **_evidence_row(["HY", "HC"])},
            {"pair_id": "P5", "risk_score": 0.55, "predicted_behavior": "coordinated_isolation",
             **_evidence_row(["HD", "HZ"])},
            {"pair_id": "P3", "risk_score": 0.3, "predicted_behavior": "none",
             **_evidence_row([])},
            {"pair_id": "P4", "risk_score": 0.1, "predicted_behavior": "none",
             **_evidence_row([])},
        ]
    )
    value = score(solution, submission, "pair_id")
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_perfect_submission_scores_one():
    solution = _tiny_case()
    # Perfect: exact risk labels, exact families, exact planted evidence.
    submission = solution.copy()
    value = score(solution, submission, "pair_id")
    assert value == pytest.approx(1.0)


def test_out_of_range_risk_is_rejected():
    solution = _tiny_case()
    submission = solution.copy()
    # Rebuild risk_score as float so an out-of-range value can be assigned, then
    # confirm the metric rejects it (no clipping) rather than scoring it.
    submission["risk_score"] = submission["risk_score"].astype(float)
    submission.loc[0, "risk_score"] = 1.5
    try:
        score(solution, submission, "pair_id")
        raised = False
    except ParticipantVisibleError:
        raised = True
    assert raised
