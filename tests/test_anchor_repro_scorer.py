"""Unit tests for the poker-anchor-reproduction CanonicalScorer (task 2.2).

Covers :meth:`CanonicalScorer.score_dev_predictions`:
- it reuses ``build_fold_solution`` + ``score_components`` (PU-correct: unknown pairs
  never scored as negatives) and returns a ``CanonicalScore`` whose ``pair_ap`` and
  ``combined`` both come from a SINGLE metric pass and equal the reused metric's
  components exactly (Req 1.2, 2.5, 2.6);
- it refuses to run when the official-metric self-check FAILS (the task-2.1 hard gate
  wired in via ``require_metric_matches_kernel``) (Req 1.2, 1.6);
- it surfaces the official metric's own coverage-mismatch raise (never a silent zero /
  fabricated number) when the prediction pair set does not match the built solution.

Tests inject synthetic dev labels/evidence so no real data files are touched. The
self-check gate runs against the real (first-party) metric kernel, which is present in
this repo; the FAIL path is exercised by monkeypatching the gate.

Validates: Requirements 1.2, 1.6, 2.5, 2.6
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.metric.production_metric import score_components
from poker_collusion.metric.reference_public_metric import (
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    ParticipantVisibleError,
)
from poker_collusion.validation.cv import build_fold_solution

import anchor_repro.scorer as scorer_module
from anchor_repro.models import CanonicalScore, ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.self_check import MetricSelfCheckError, SelfCheckResult


# --------------------------------------------------------------------------- #
# Fixtures / builders                                                         #
# --------------------------------------------------------------------------- #
def _evidence_slots(hands=()):
    padded = list(hands)[:5] + [NO_EVIDENCE] * (5 - min(len(hands), 5))
    return {col: padded[i] for i, col in enumerate(EVIDENCE_COLUMNS)}


def _dev_labels() -> pd.DataFrame:
    """Three confirmed positives (one per target family) + two confirmed negatives."""
    return pd.DataFrame(
        [
            {"pair_id": "P1", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
            {"pair_id": "P2", "label_status": "confirmed_target",
             "behavior_family": "soft_play"},
            {"pair_id": "P3", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "P4", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "P5", "label_status": "confirmed_target",
             "behavior_family": "coordinated_isolation"},
        ]
    )


def _dev_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "P1", "evidence_rank": 1, "hand_id": "HA"},
            {"pair_id": "P1", "evidence_rank": 2, "hand_id": "HB"},
            {"pair_id": "P2", "evidence_rank": 1, "hand_id": "HC"},
            {"pair_id": "P5", "evidence_rank": 1, "hand_id": "HD"},
            {"pair_id": "P5", "evidence_rank": 2, "hand_id": "HE"},
        ]
    )


def _dev_predictions() -> pd.DataFrame:
    """An imperfect submission over the same five confirmed pairs (score in (0,1))."""
    return pd.DataFrame(
        [
            {"pair_id": "P1", "risk_score": 0.90, "predicted_behavior": "directed_transfer",
             **_evidence_slots(["HA", "HX"])},
            {"pair_id": "P2", "risk_score": 0.55, "predicted_behavior": "soft_play",
             **_evidence_slots(["HY", "HC"])},
            {"pair_id": "P3", "risk_score": 0.20, "predicted_behavior": "none",
             **_evidence_slots([])},
            {"pair_id": "P4", "risk_score": 0.30, "predicted_behavior": "none",
             **_evidence_slots([])},
            {"pair_id": "P5", "risk_score": 0.80, "predicted_behavior": "coordinated_isolation",
             **_evidence_slots(["HD", "HZ", "HE"])},
        ]
    )


def _make_scorer() -> CanonicalScorer:
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(),
        labels=_dev_labels(),
        evidence=_dev_evidence(),
    )


# --------------------------------------------------------------------------- #
# Core scoring behaviour                                                      #
# --------------------------------------------------------------------------- #
def test_score_dev_predictions_returns_canonical_score_from_single_pass():
    scorer = _make_scorer()
    preds = _dev_predictions()

    result = scorer.score_dev_predictions(preds)

    assert isinstance(result, CanonicalScore)

    # Reconstruct the SAME reused metric pass independently and assert bit-equality:
    # this proves the scorer reused build_fold_solution + score_components (no
    # re-implementation) and that pair_ap/combined come from one identical evaluation.
    solution = build_fold_solution(_dev_labels(), list(preds["pair_id"]), evidence=_dev_evidence())
    expected = score_components(solution.copy(), preds.copy(), "pair_id")

    assert result.pair_ap == expected.pair_ap          # PRIMARY gate == local_holdout_ap
    assert result.combined == expected.combined        # SECONDARY diagnostic
    assert result.components == expected               # full breakdown, same pass
    # combined is the metric's own weighting of the SAME components (single pass).
    assert result.combined == result.components.combined
    assert result.pair_ap == result.components.pair_ap


def test_score_dev_predictions_is_deterministic():
    scorer = _make_scorer()
    preds = _dev_predictions()
    first = scorer.score_dev_predictions(preds)
    second = scorer.score_dev_predictions(preds)
    assert first == second


def test_score_dev_predictions_pu_unknown_never_scored_as_negative():
    """A pair absent from the confirmed labels is UNKNOWN — the solution must not
    fabricate a negative for it, so predicting only confirmed pairs is what scores."""
    scorer = _make_scorer()
    preds = _dev_predictions()
    # The built solution covers exactly the confirmed pairs, none fabricated.
    solution = build_fold_solution(_dev_labels(), list(preds["pair_id"]), evidence=_dev_evidence())
    assert set(solution["pair_id"]) == set(preds["pair_id"])
    # And scoring succeeds (coverage matches) rather than raising.
    assert isinstance(scorer.score_dev_predictions(preds), CanonicalScore)


def test_score_dev_predictions_does_not_mutate_inputs():
    scorer = _make_scorer()
    preds = _dev_predictions()
    before = preds.copy(deep=True)
    scorer.score_dev_predictions(preds)
    pd.testing.assert_frame_equal(preds, before)


# --------------------------------------------------------------------------- #
# Hard gate: refuse to run when the metric self-check fails                   #
# --------------------------------------------------------------------------- #
def test_scorer_refuses_to_run_when_self_check_fails(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise MetricSelfCheckError("simulated metric drift")

    monkeypatch.setattr(scorer_module, "require_metric_matches_kernel", _boom)

    scorer = _make_scorer()
    with pytest.raises(MetricSelfCheckError):
        scorer.score_dev_predictions(_dev_predictions())


def test_self_check_runs_once_and_is_cached(monkeypatch):
    calls = {"n": 0}
    passing = SelfCheckResult(
        passed=True, local_score=1.0, kernel_score=1.0, abs_diff=0.0,
        kernel_notebook=__import__("pathlib").Path("kernel.ipynb"),
        example_name="stub", detail="stub-pass",
    )

    def _counted(*_args, **_kwargs):
        calls["n"] += 1
        return passing

    monkeypatch.setattr(scorer_module, "require_metric_matches_kernel", _counted)

    scorer = _make_scorer()
    preds = _dev_predictions()
    scorer.score_dev_predictions(preds)
    scorer.score_dev_predictions(preds)
    assert calls["n"] == 1  # gate verified once, then cached


# --------------------------------------------------------------------------- #
# Coverage mismatch surfaces the metric's own raise (never a fabricated number) #
# --------------------------------------------------------------------------- #
def test_missing_pair_id_column_raises_valueerror():
    scorer = _make_scorer()
    bad = _dev_predictions().drop(columns=["pair_id"])
    with pytest.raises(ValueError):
        scorer.score_dev_predictions(bad)


def test_coverage_mismatch_surfaces_metric_raise_not_silent_zero():
    """Predicting a pair set that does not match the built solution must surface the
    official metric's ParticipantVisibleError coverage-mismatch — never a silent 0."""
    scorer = _make_scorer()
    # Add an extra pair with no confirmed label: build_fold_solution skips it (PU),
    # so the solution set != submission set and the metric raises coverage mismatch.
    preds = _dev_predictions()
    extra = pd.DataFrame([{
        "pair_id": "P_UNKNOWN", "risk_score": 0.5, "predicted_behavior": "none",
        **_evidence_slots([]),
    }])
    preds = pd.concat([preds, extra], ignore_index=True)
    with pytest.raises(ParticipantVisibleError):
        scorer.score_dev_predictions(preds)
