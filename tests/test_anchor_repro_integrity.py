"""Unit tests for CanonicalScorer.integrity_report + the coverage-mismatch guard (task 2.3).

Covers, without touching the real 112,540-row artifacts (a synthetic eval-like frame /
a real-artifact COPY on a tmp path is used instead — never the artifact itself):

- ``integrity_report`` mirrors ``gate_a_verify_floor``: hash (from a file path), row
  count, pair-set-equals-sample, and non-degenerate ranking, WITHOUT dev-scoring the
  eval artifact (impossible: disjoint pairs) — it surfaces the disjoint-set sizes
  between the artifact and the dev holdout explicitly (Req 1.2, 2.4, 4.1);
- the impossibility guard names the eval-vs-dev disjoint sizes (``intersection == 0``)
  rather than returning a silent zero or a fabricated number;
- the belt-and-suspenders coverage-mismatch guard on ``score_dev_predictions`` raises
  an explicit ``CoverageMismatchError`` (a ``ParticipantVisibleError`` subclass) whose
  message names how many pair-ids are missing and how many are extra — on BOTH the
  pre-check (belt) path and the metric-raise (suspenders) path.

Validates: Requirements 1.2, 2.4, 4.1
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.metric.reference_public_metric import (
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    ParticipantVisibleError,
)
from poker_collusion.submission.writer import SUBMISSION_COLUMNS

from anchor_repro.models import DisjointSetDiagnostic, IntegrityReport
from anchor_repro.scorer import CanonicalScorer, CoverageMismatchError
from anchor_repro.models import ScoringRecipe


# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #
def _evidence_slots(hands=()):
    padded = list(hands)[:5] + [NO_EVIDENCE] * (5 - min(len(hands), 5))
    return {col: padded[i] for i, col in enumerate(EVIDENCE_COLUMNS)}


def _dev_labels() -> pd.DataFrame:
    """Confirmed DEV pairs (D*) — DISJOINT from the eval pairs (E*) below."""
    return pd.DataFrame(
        [
            {"pair_id": "D1", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
            {"pair_id": "D2", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "D3", "label_status": "confirmed_target",
             "behavior_family": "soft_play"},
        ]
    )


def _eval_artifact(n: int = 1500) -> pd.DataFrame:
    """A synthetic EVAL-like artifact: n eval pairs, a non-degenerate risk ranking
    (>1000 distinct values in (0,1)), the exact submission schema/column order, and
    pair-ids DISJOINT from the dev labels."""
    rows = []
    for i in range(n):
        rows.append({
            "pair_id": f"E{i}",
            "risk_score": (i + 1) / (n + 1),  # n distinct values, strictly in (0,1)
            "predicted_behavior": "none",
            **_evidence_slots([]),
        })
    return pd.DataFrame(rows)[list(SUBMISSION_COLUMNS)]


def _make_scorer(sample: pd.DataFrame | None = None) -> CanonicalScorer:
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(),
        labels=_dev_labels(),
        evidence=None,
        sample_submission=sample,
    )


# --------------------------------------------------------------------------- #
# integrity_report: frozen EVAL artifact scoreability (mirrors gate_a)        #
# --------------------------------------------------------------------------- #
def test_integrity_report_on_frame_surfaces_disjoint_and_is_scoreable():
    artifact = _eval_artifact(1500)
    scorer = _make_scorer(sample=artifact.copy())  # sample == artifact pair set

    report = scorer.integrity_report(artifact, expected_row_count=1500)

    assert isinstance(report, IntegrityReport)
    # Integrity / scoreability checks mirror gate_a_verify_floor.
    assert report.n_rows == 1500
    assert report.row_count_ok is True
    assert report.columns_match is True
    assert report.risk_in_unit_interval is True
    assert report.ranking_non_degenerate is True
    assert report.pair_set_equals_sample is True
    assert report.scoreable is True

    # It NEVER dev-scores the eval artifact; instead it surfaces the eval-vs-dev
    # disjoint sizes explicitly (intersection == 0, the impossibility).
    assert isinstance(report.dev_disjoint_diagnostic, DisjointSetDiagnostic)
    assert report.dev_disjoint_diagnostic.is_disjoint is True
    assert report.dev_disjoint_diagnostic.n_intersection == 0
    assert report.dev_disjoint_diagnostic.n_missing == 3   # 3 dev pairs absent
    assert report.dev_disjoint_diagnostic.n_extra == 1500  # 1500 eval pairs extra
    assert any("IMPOSSIBILITY" in note for note in report.notes)


def test_integrity_report_from_path_computes_hash(tmp_path):
    artifact = _eval_artifact(1200)
    # A COPY of an artifact on a tmp path — never a real artifact file itself.
    path = tmp_path / "submission_best_012345.csv"
    artifact.to_csv(path, index=False)
    scorer = _make_scorer(sample=artifact.copy())

    report = scorer.integrity_report(path, expected_row_count=1200)

    assert len(report.sha256) == 64  # sha256 hex digest computed from the file
    assert report.n_bytes > 0
    assert report.n_rows == 1200
    assert report.scoreable is True


def test_integrity_report_flags_pair_set_mismatch_with_sample():
    artifact = _eval_artifact(1500)
    # Sample submission has an extra pair the artifact lacks -> not scoreable.
    sample = pd.concat(
        [artifact, pd.DataFrame([{
            "pair_id": "E_EXTRA", "risk_score": 0.5, "predicted_behavior": "none",
            **_evidence_slots([]),
        }])[list(SUBMISSION_COLUMNS)]],
        ignore_index=True,
    )
    scorer = _make_scorer(sample=sample)

    report = scorer.integrity_report(artifact, expected_row_count=1500)

    assert report.pair_set_equals_sample is False
    assert report.sample_diagnostic.n_missing == 1  # E_EXTRA missing from artifact
    assert report.sample_diagnostic.n_extra == 0
    assert report.scoreable is False


def test_integrity_report_flags_degenerate_ranking():
    n = 1500
    artifact = _eval_artifact(n)
    artifact["risk_score"] = 0.5  # every risk identical -> degenerate ranking
    scorer = _make_scorer(sample=artifact.copy())

    report = scorer.integrity_report(artifact, expected_row_count=n)

    assert report.ranking_non_degenerate is False
    assert report.scoreable is False


def test_integrity_report_wrong_row_count_not_scoreable():
    artifact = _eval_artifact(1500)
    scorer = _make_scorer(sample=artifact.copy())
    # Check against the real eval count (112,540); the synthetic frame won't match.
    report = scorer.integrity_report(artifact)  # default expected_row_count = 112540
    assert report.row_count_ok is False
    assert report.scoreable is False


# --------------------------------------------------------------------------- #
# The coverage-mismatch guard: belt (pre-check) and suspenders (metric catch) #
# --------------------------------------------------------------------------- #
def _dev_predictions(pairs) -> pd.DataFrame:
    rows = []
    for pid in pairs:
        rows.append({
            "pair_id": pid, "risk_score": 0.5, "predicted_behavior": "none",
            **_evidence_slots([]),
        })
    return pd.DataFrame(rows)[list(SUBMISSION_COLUMNS)]


def test_belt_precheck_raises_coverage_mismatch_naming_disjoint_sizes():
    """Predicting an EVAL pair set (disjoint from dev) hits the belt pre-check: the
    built solution is empty (all pairs unknown), so the guard refuses with a
    CoverageMismatchError naming the disjoint-set sizes — never a silent zero."""
    scorer = _make_scorer()
    # E* pairs are all unknown to the dev labels -> solution is empty -> mismatch.
    preds = _dev_predictions(["E1", "E2", "E3"])

    with pytest.raises(CoverageMismatchError) as exc_info:
        scorer.score_dev_predictions(preds)

    err = exc_info.value
    # It is a ParticipantVisibleError subclass (existing callers still catch it).
    assert isinstance(err, ParticipantVisibleError)
    assert err.diagnostic.n_intersection == 0
    assert err.diagnostic.n_extra == 3      # 3 predicted eval pairs are extra
    assert err.diagnostic.n_missing == 0    # empty solution has nothing missing
    # The message names the disjoint-set sizes explicitly (Property 5).
    assert "3 extra" in str(err)
    assert "0 missing" in str(err)


def test_belt_precheck_partial_mismatch_names_missing_and_extra():
    """A partial mismatch (one confirmed pair predicted, one confirmed pair dropped,
    one extra unknown) reports both missing and extra counts."""
    scorer = _make_scorer()
    # Solution will contain D1 + D2 (confirmed); we predict D1 + an unknown extra,
    # so D2 is missing and the unknown is extra.
    preds = _dev_predictions(["D1", "D2", "E_UNK"])
    # D1,D2 are confirmed -> solution={D1,D2}; preds pair set={D1,D2,E_UNK}.
    with pytest.raises(CoverageMismatchError) as exc_info:
        scorer.score_dev_predictions(preds)
    err = exc_info.value
    assert err.diagnostic.n_missing == 0     # D1,D2 both present
    assert err.diagnostic.n_extra == 1       # E_UNK is extra vs the solution
    assert "1 extra" in str(err)


def test_suspenders_catch_resurfaces_metric_raise(monkeypatch):
    """If the belt is bypassed, the metric's own coverage-mismatch ParticipantVisibleError
    is re-surfaced as the same explicit CoverageMismatchError, never swallowed."""
    import anchor_repro.scorer as scorer_module

    scorer = _make_scorer()
    preds = _dev_predictions(["D1", "D2"])  # both confirmed -> belt passes

    # Bypass the belt by forcing is_equal True, then make the metric raise coverage.
    monkeypatch.setattr(
        scorer_module.DisjointSetDiagnostic, "is_equal", property(lambda self: True)
    )

    def _raise_coverage(*_a, **_k):
        raise ParticipantVisibleError(
            "pair_id coverage mismatch: 2 missing and 4 extra."
        )

    monkeypatch.setattr(scorer_module, "score_components", _raise_coverage)

    with pytest.raises(CoverageMismatchError):
        scorer.score_dev_predictions(preds)


def test_suspenders_reraises_non_coverage_metric_error_untouched(monkeypatch):
    """A genuine, non-coverage ParticipantVisibleError from the metric is re-raised
    untouched (not misclassified as a coverage mismatch)."""
    import anchor_repro.scorer as scorer_module

    scorer = _make_scorer()
    preds = _dev_predictions(["D1", "D2"])

    monkeypatch.setattr(
        scorer_module.DisjointSetDiagnostic, "is_equal", property(lambda self: True)
    )

    def _raise_other(*_a, **_k):
        raise ParticipantVisibleError("risk_score must be numeric and between 0 and 1.")

    monkeypatch.setattr(scorer_module, "score_components", _raise_other)

    with pytest.raises(ParticipantVisibleError) as exc_info:
        scorer.score_dev_predictions(preds)
    # Re-raised untouched: NOT wrapped as a CoverageMismatchError.
    assert not isinstance(exc_info.value, CoverageMismatchError)
