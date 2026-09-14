"""Unit tests for poker_collusion.submission.writer (task 13.1, Requirements 9.1-9.7).

Hermetic, ``tmp_path``-based example tests on a tiny synthetic sample submission and
predictions verifying the Submission_Writer's schema/column-order emission (Req 9.1, 9.2),
one-row-per-evaluation-pair coverage with unchanged pair ids (Req 9.6), risk-score range
validation (Req 9.3), allowed behaviors (Req 9.4), evidence completeness + no-repeat gate
(Req 9.7), the missing-pair default row (coverage), atomic write producing a complete file
that round-trips (Req 9.5), and determinism.

The Hypothesis property tests (Properties 13/14, tasks 13.2/13.3) are separate tasks and are
deliberately NOT duplicated here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE, PipelineConfig
from poker_collusion.submission.writer import (
    SUBMISSION_COLUMNS,
    SubmissionValidationError,
    build_submission_frame,
    validate_submission,
    write_submission,
)
from poker_collusion.types import PairPrediction


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _sample_submission(pair_ids=("3_7", "1_2", "5_9")) -> pd.DataFrame:
    """A tiny sample_submission.csv template in the exact schema/column order."""
    rows = []
    for pid in pair_ids:
        row = {
            "pair_id": pid,
            "risk_score": 0.0,
            "predicted_behavior": "none",
        }
        for i in range(1, 6):
            row[f"evidence_hand_{i}"] = NO_EVIDENCE
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))


def _pred(pair_id, risk, behavior, evidence) -> PairPrediction:
    return PairPrediction(
        pair_id=pair_id,
        risk_score=risk,
        predicted_behavior=behavior,
        evidence=list(evidence),
    )


# --------------------------------------------------------------------------- #
# Schema / column order / coverage
# --------------------------------------------------------------------------- #
def test_exact_header_column_order_and_row_count():
    sample = _sample_submission()
    preds = [
        _pred("1_2", 0.9, "directed_transfer", ["101", "102", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
        _pred("3_7", 0.4, "soft_play", ["201", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
        _pred("5_9", 0.1, "none", [NO_EVIDENCE] * 5),
    ]
    frame = build_submission_frame(preds, sample)

    assert tuple(frame.columns) == SUBMISSION_COLUMNS
    assert len(frame) == len(sample)
    # Row order follows the sample submission's pair_id order, unchanged (Req 9.6).
    assert list(frame["pair_id"]) == list(sample["pair_id"].astype(str))
    assert set(frame["pair_id"]) == set(sample["pair_id"].astype(str))


def test_missing_pair_gets_safe_default_row():
    sample = _sample_submission(("a", "b", "c"))
    # Only predict for "b"; "a" and "c" must be filled with the default row.
    preds = [_pred("b", 0.7, "soft_play", ["55", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE])]
    frame = build_submission_frame(preds, sample).set_index("pair_id")

    assert set(frame.index) == {"a", "b", "c"}
    for missing in ("a", "c"):
        assert frame.loc[missing, "risk_score"] == 0.0
        assert frame.loc[missing, "predicted_behavior"] == "none"
        assert all(frame.loc[missing, f"evidence_hand_{i}"] == NO_EVIDENCE for i in range(1, 6))


def test_mapping_input_accepted():
    sample = _sample_submission(("1_2", "3_7", "5_9"))
    mapping = {
        "1_2": (0.5, "coordinated_isolation", ["7", "8"]),
        "3_7": (0.2, "none", []),
        "5_9": (0.9, "other_coordination", ["1", "2", "3", "4", "5"]),
    }
    frame = build_submission_frame(mapping, sample).set_index("pair_id")
    # Short evidence lists are padded to 5 with NO_EVIDENCE.
    assert frame.loc["1_2", "evidence_hand_3"] == NO_EVIDENCE
    assert frame.loc["1_2", "evidence_hand_1"] == "7"
    assert frame.loc["5_9", "evidence_hand_5"] == "5"


# --------------------------------------------------------------------------- #
# Field-domain validation (Req 9.3, 9.4, 9.7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_risk", [-0.01, 1.5, float("nan"), float("inf")])
def test_risk_out_of_range_raises(bad_risk):
    sample = _sample_submission(("1_2",))
    preds = [_pred("1_2", bad_risk, "none", [NO_EVIDENCE] * 5)]
    with pytest.raises(SubmissionValidationError) as exc:
        build_submission_frame(preds, sample)
    assert "1_2" in str(exc.value)


def test_invalid_behavior_raises():
    sample = _sample_submission(("1_2",))
    preds = [_pred("1_2", 0.5, "totally_made_up", [NO_EVIDENCE] * 5)]
    with pytest.raises(SubmissionValidationError) as exc:
        build_submission_frame(preds, sample)
    assert "predicted_behavior" in str(exc.value)


def test_duplicate_evidence_within_row_raises():
    sample = _sample_submission(("1_2",))
    preds = [_pred("1_2", 0.5, "soft_play", ["42", "42", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE])]
    with pytest.raises(SubmissionValidationError) as exc:
        build_submission_frame(preds, sample)
    assert "42" in str(exc.value)


def test_empty_evidence_cell_raises_via_validate():
    # build_submission_frame normalizes blanks to NO_EVIDENCE, so exercise the
    # validator directly on a hand-crafted frame with an empty cell.
    sample = _sample_submission(("1_2",))
    frame = _sample_submission(("1_2",))
    frame.loc[0, "evidence_hand_2"] = ""
    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(frame, sample)
    assert "evidence_hand_2" in str(exc.value)


def test_pair_id_set_mismatch_raises():
    sample = _sample_submission(("1_2", "3_7"))
    frame = _sample_submission(("1_2", "9_9"))  # 3_7 missing, 9_9 extra
    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(frame, sample)
    msg = str(exc.value)
    assert "3_7" in msg and "9_9" in msg


def test_header_order_mismatch_raises():
    sample = _sample_submission(("1_2",))
    frame = _sample_submission(("1_2",))
    reordered = frame[["risk_score", "pair_id", "predicted_behavior", *[f"evidence_hand_{i}" for i in range(1, 6)]]]
    with pytest.raises(SubmissionValidationError):
        validate_submission(reordered, sample)


# --------------------------------------------------------------------------- #
# Atomic write + round-trip + determinism (Req 9.5)
# --------------------------------------------------------------------------- #
def test_write_submission_atomic_roundtrip(tmp_path):
    sample = _sample_submission(("1_2", "3_7", "5_9"))
    preds = [
        _pred("1_2", 0.9, "directed_transfer", ["101", "102", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
        _pred("3_7", 0.4, "soft_play", ["201", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
        _pred("5_9", 0.1, "none", [NO_EVIDENCE] * 5),
    ]
    out = tmp_path / "sub" / "submission.csv"
    written = write_submission(preds, sample_submission=sample, path=out)

    assert written == out
    assert out.exists()
    # No temp artifacts left behind.
    leftovers = [p.name for p in out.parent.iterdir() if p.name != "submission.csv"]
    assert leftovers == []

    # Round-trip: reading back equals the built frame (dtype=str for exactness).
    read_back = pd.read_csv(out, dtype=str)
    expected = build_submission_frame(preds, sample).astype(str)
    assert tuple(read_back.columns) == SUBMISSION_COLUMNS
    pd.testing.assert_frame_equal(read_back.reset_index(drop=True), expected.reset_index(drop=True))


def test_write_submission_uses_config_submission_path(tmp_path):
    sample = _sample_submission(("1_2",))
    preds = [_pred("1_2", 0.3, "none", [NO_EVIDENCE] * 5)]
    config = PipelineConfig(output_dir=tmp_path / "outdir")
    written = write_submission(preds, sample_submission=sample, config=config)
    assert written == config.submission_path
    assert written.exists()


def test_write_submission_deterministic(tmp_path):
    sample = _sample_submission(("1_2", "3_7", "5_9"))
    preds = [
        _pred("5_9", 0.1, "none", [NO_EVIDENCE] * 5),
        _pred("1_2", 0.9, "directed_transfer", ["101", "102", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
        _pred("3_7", 0.4, "soft_play", ["201", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
    ]
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    write_submission(preds, sample_submission=sample, path=a)
    write_submission(preds, sample_submission=sample, path=b)
    assert a.read_bytes() == b.read_bytes()


def test_write_prevalidated_frame(tmp_path):
    sample = _sample_submission(("1_2",))
    frame = build_submission_frame(
        [_pred("1_2", 0.5, "soft_play", ["9", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE])],
        sample,
    )
    out = tmp_path / "submission.csv"
    written = write_submission(frame, sample_submission=sample, path=out)
    assert written.exists()
    read_back = pd.read_csv(out, dtype=str)
    assert read_back.loc[0, "evidence_hand_1"] == "9"


def test_write_raw_predictions_without_sample_raises(tmp_path):
    preds = [_pred("1_2", 0.5, "none", [NO_EVIDENCE] * 5)]
    with pytest.raises(SubmissionValidationError):
        write_submission(preds, path=tmp_path / "submission.csv")
