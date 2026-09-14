"""Unit test for CanonicalScorer.integrity_report on a COPY of a REAL artifact (task 2.7).

This complements ``test_anchor_repro_integrity.py`` (which drives ``integrity_report``
with synthetic eval-like frames and a real-artifact COPY on a tmp path). Here we run the
report against a byte-for-byte COPY of one of the nine REAL on-disk ground-truth
artifacts (``outputs/poker_collusion/submission_best_<score>.csv``), scored against the
REAL eval ``sample_submission.csv`` and the REAL ``development_labels.csv``, and assert
the same falsifiable bar the existing ``poker_collusion.experiments.gate_a_verify_floor``
enforces on the 0.44519 floor:

* row count == 112,540 (``EVAL_ROW_COUNT``),
* the artifact's ``pair_id`` SET equals the eval sample submission's set (Kaggle-scoreable),
* the risk column is a NON-DEGENERATE ranking (>1000 distinct values, spread in (0,1)),
* and therefore the report is ``scoreable``.

Accountability contract (Rules 1, 9):

* We NEVER touch the real artifact in place — the test COPIES it to a pytest ``tmp_path``
  and runs ``integrity_report`` on the copy. The original file is asserted byte-unchanged
  (sha256) at the end. No read-lock, no mutation of ground truth.
* ``integrity_report`` does NOT dev-score the eval artifact (impossible: its ~112,540 eval
  pairs are DISJOINT from the 1,860 confirmed dev pairs). The test asserts that the
  eval-vs-dev disjoint sizes are surfaced explicitly (``n_intersection == 0``) rather than
  a fabricated absolute score.
* If the real artifacts, the eval sample submission, or the dev labels are NOT present in
  this environment, the test ``pytest.skip()``s with a clear reason — it never fabricates
  data to manufacture a pass.

Marked ``@pytest.mark.integration`` (same convention as ``test_integration_real_holdout.py``)
so the fast unit/property suite can exclude the real-data path.

Validates: Requirements 1.2
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig

from anchor_repro.models import DisjointSetDiagnostic, IntegrityReport
from anchor_repro.scorer import EVAL_ROW_COUNT, CanonicalScorer
from anchor_repro.models import ScoringRecipe

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------- #
# Real-data / real-artifact locations (resolved relative to this file so the   #
# test runs from anywhere), mirroring gate_a_verify_floor's layout.            #
# --------------------------------------------------------------------------- #
_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_ARTIFACT_DIR = _POKER_ROOT / "outputs" / "poker_collusion"

_SAMPLE_SUBMISSION = _DATA_DIR / "sample_submission.csv"
_DEV_LABELS = _DATA_DIR / "development_labels.csv"

#: The nine known-LB artifacts (design.md load-bearing fact); any one is a valid
#: subject for the integrity check. We prefer the recorded 0.44519 floor (the same
#: artifact gate_a_verify_floor checks) and fall back to whichever is present.
_PREFERRED_ARTIFACT = _ARTIFACT_DIR / "submission_best_044519.csv"
_ARTIFACT_GLOB = "submission_best_*.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate_real_artifact() -> Path:
    """Return a real on-disk artifact, or skip cleanly if none is present."""
    if _PREFERRED_ARTIFACT.is_file():
        return _PREFERRED_ARTIFACT
    candidates = sorted(_ARTIFACT_DIR.glob(_ARTIFACT_GLOB))
    if candidates:
        return candidates[0]
    pytest.skip(
        "No real submission_best_<score>.csv artifact present under "
        f"{_ARTIFACT_DIR}; cannot run integrity_report on a real-artifact copy "
        "(skipping rather than fabricating data)."
    )


def _require_real_data() -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Skip unless the real artifact, eval sample submission, and dev labels exist.

    Honesty gate (contract Rules 1, 9): the integrity/scoreability bar
    (pair-set-equals-sample) needs the REAL eval ``sample_submission.csv``, and the
    eval-vs-dev disjoint surfacing needs the REAL ``development_labels.csv``. Without
    them we SKIP with a clear reason rather than invent a passing frame.
    """
    artifact = _locate_real_artifact()
    missing = [str(p) for p in (_SAMPLE_SUBMISSION, _DEV_LABELS) if not p.is_file()]
    if missing:
        pytest.skip(
            "Real eval sample submission / dev labels unavailable for the "
            f"real-artifact integrity test; missing: {missing}"
        )
    sample = pd.read_csv(_SAMPLE_SUBMISSION, dtype={"pair_id": str})
    labels = pd.read_csv(_DEV_LABELS)
    return artifact, sample, labels


def _make_scorer(sample: pd.DataFrame, labels: pd.DataFrame) -> CanonicalScorer:
    """Canonical scorer wired to the REAL eval sample submission + REAL dev labels.

    ``PipelineConfig`` still points at the real data dir, but injecting ``labels`` /
    ``sample_submission`` keeps the lazy loaders from doing any extra disk reads and
    makes the data the report scores against explicit and auditable.
    """
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=None,
        sample_submission=sample,
    )


def test_integrity_report_on_real_artifact_copy_matches_gate_a(tmp_path) -> None:
    """integrity_report on a COPY of a real artifact clears the gate_a bar.

    Copies a real ``submission_best_<score>.csv`` to a tmp path (never the artifact
    itself), runs ``integrity_report`` on the copy, and asserts:

    * row count == 112,540 (``EVAL_ROW_COUNT``),
    * pair-set equals the eval sample submission,
    * a non-degenerate ranking (>1000 distinct risk values with spread),
    * ``scoreable is True`` — the same conjunction gate_a_verify_floor enforces,
    * the eval-vs-dev disjoint sizes are surfaced (``n_intersection == 0``) instead
      of any fabricated dev score, and the impossibility is noted,
    * the ORIGINAL artifact is byte-unchanged (sha256) — ground truth is never
      mutated or read-locked in place.

    Skips cleanly if the real artifacts / sample submission / dev labels are absent.
    """
    artifact_path, sample, labels = _require_real_data()

    # Record the original artifact's digest so we can prove we never mutated it.
    original_sha = _sha256(artifact_path)

    # COPY the real artifact to a tmp path and run the report on the COPY only.
    copy_path = tmp_path / artifact_path.name
    shutil.copy2(artifact_path, copy_path)
    assert _sha256(copy_path) == original_sha, "copy must be byte-identical to the real artifact"

    scorer = _make_scorer(sample, labels)
    report = scorer.integrity_report(copy_path)  # default expected_row_count = 112,540

    assert isinstance(report, IntegrityReport)

    # ---- Integrity / scoreability bar (mirrors gate_a_verify_floor) ---- #
    assert report.expected_row_count == EVAL_ROW_COUNT
    assert report.n_rows == EVAL_ROW_COUNT, (
        f"real artifact should have {EVAL_ROW_COUNT} eval rows; got {report.n_rows}"
    )
    assert report.row_count_ok is True
    assert report.columns_match is True
    assert report.risk_in_unit_interval is True

    # Non-degenerate ranking: gate_a's clause (>1000 distinct risk values + spread).
    assert report.ranking_non_degenerate is True
    assert report.risk_n_distinct > 1000
    assert report.risk_max > report.risk_min

    # Pair-set equals the eval sample submission (Kaggle-scoreable).
    assert report.pair_set_equals_sample is True
    assert isinstance(report.sample_diagnostic, DisjointSetDiagnostic)
    assert report.sample_diagnostic.is_equal is True
    assert report.sample_diagnostic.n_missing == 0
    assert report.sample_diagnostic.n_extra == 0

    # The overall scoreability verdict (same conjunction as gate_a's bar clauses).
    assert report.scoreable is True

    # The hash/bytes were computed from the file COPY (not an in-memory frame).
    assert len(report.sha256) == 64
    assert report.n_bytes > 0

    # ---- IMPOSSIBILITY: eval-vs-dev disjoint surfaced, never a fabricated score ---- #
    assert isinstance(report.dev_disjoint_diagnostic, DisjointSetDiagnostic)
    assert report.dev_disjoint_diagnostic.is_disjoint is True
    assert report.dev_disjoint_diagnostic.n_intersection == 0
    assert any("IMPOSSIBILITY" in note for note in report.notes)

    # ---- Ground truth untouched: the ORIGINAL real artifact is byte-unchanged ---- #
    assert _sha256(artifact_path) == original_sha, (
        "the real artifact must never be mutated or read-locked in place"
    )
