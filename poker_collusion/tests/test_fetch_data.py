"""Unit tests for poker_collusion.discovery.fetch_data (task 2.1).

Builds tiny constructed parquet/CSV fixtures in a temp directory and verifies:
- verify_schema passes on valid fixtures (all 8 files, correct join keys,
  correct hands.phase values, ordered actions, matching row count sentinel);
- SchemaError is raised on a missing column and on a missing file;
- absent data is reported gracefully (no crash);
- findings are recorded to a dossier file.

Requirements exercised: 1.1, 1.2, 1.3, 1.6.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.discovery import fetch_data
from poker_collusion.discovery.fetch_data import (
    EXPECTED_FILES,
    kaggle_download_command,
    record_findings,
    verify_schema,
)
from poker_collusion.exceptions import SchemaError


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _write_valid_fixtures(input_dir: Path) -> None:
    """Write all eight required files as tiny, well-formed fixtures."""
    input_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"player_id": [1, 2, 3]}).to_parquet(input_dir / "players.parquet")

    pd.DataFrame(
        {
            "hand_id": [10, 11, 12],
            "table_id": [1, 1, 1],
            "phase": ["development", "development", "evaluation"],
        }
    ).to_parquet(input_dir / "hands.parquet")

    pd.DataFrame(
        {
            "hand_id": [10, 10, 11, 12],
            "player_id": [1, 2, 1, 3],
            "seat": [0, 1, 0, 2],
        }
    ).to_parquet(input_dir / "seats.parquet")

    # Actions ordered by (hand_id, action_no).
    pd.DataFrame(
        {
            "hand_id": [10, 10, 11, 12, 12],
            "action_no": [0, 1, 0, 0, 1],
            "actor": [1, 2, 1, 3, 1],
            "action": ["bet", "call", "fold", "raise", "call"],
            "amount": [10, 10, 0, 20, 20],
        }
    ).to_parquet(input_dir / "actions.parquet")

    pd.DataFrame({"pair_id": ["1_2"], "target_behavior": ["soft_play"]}).to_csv(
        input_dir / "development_labels.csv", index=False
    )
    pd.DataFrame({"pair_id": ["1_2"], "hand_id": [10]}).to_csv(
        input_dir / "development_evidence.csv", index=False
    )
    pd.DataFrame({"pair_id": ["1_3", "2_3"]}).to_csv(
        input_dir / "evaluation_pairs.csv", index=False
    )
    pd.DataFrame(
        {
            "pair_id": ["1_3", "2_3"],
            "risk_score": [0.0, 0.0],
            "predicted_behavior": ["none", "none"],
            "evidence_hand_1": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_2": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_3": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_4": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_5": ["NO_EVIDENCE", "NO_EVIDENCE"],
        }
    ).to_csv(input_dir / "sample_submission.csv", index=False)


def _config_for(input_dir: Path, output_dir: Path) -> PipelineConfig:
    return PipelineConfig(input_dir=input_dir, output_dir=output_dir)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_verify_schema_passes_on_valid_fixtures(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    _write_valid_fixtures(input_dir)
    cfg = _config_for(input_dir, tmp_path / "out")

    report = verify_schema(cfg, strict=True)

    assert report.data_present is True
    assert report.ok is True
    assert report.errors == []
    # Every expected file present with no missing columns.
    for name in EXPECTED_FILES:
        fr = report.files[name]
        assert fr.present is True, name
        assert fr.missing_columns == [], name
    # Join-key / structural findings recorded.
    assert report.actions_ordering_ok is True
    assert report.actions_row_count == 5
    assert set(report.hands_phase_values) == {"development", "evaluation"}


def test_verify_schema_raises_on_missing_column(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    _write_valid_fixtures(input_dir)
    # Drop the join key 'phase' from hands.parquet -> should raise SchemaError.
    hands = pd.read_parquet(input_dir / "hands.parquet")
    hands.drop(columns=["phase"]).to_parquet(input_dir / "hands.parquet")
    cfg = _config_for(input_dir, tmp_path / "out")

    with pytest.raises(SchemaError) as exc:
        verify_schema(cfg, strict=True)

    assert exc.value.file == "hands.parquet"
    assert exc.value.missing_key == "phase"


def test_verify_schema_raises_on_missing_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    _write_valid_fixtures(input_dir)
    # Remove a required file (but leave others) -> strict mode raises.
    (input_dir / "actions.parquet").unlink()
    cfg = _config_for(input_dir, tmp_path / "out")

    with pytest.raises(SchemaError) as exc:
        verify_schema(cfg, strict=True)

    assert exc.value.file == "actions.parquet"


def test_verify_schema_missing_column_non_strict_collects_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    _write_valid_fixtures(input_dir)
    # Remove player_id from seats -> non-strict should collect, not raise.
    seats = pd.read_parquet(input_dir / "seats.parquet")
    seats.drop(columns=["player_id"]).to_parquet(input_dir / "seats.parquet")
    cfg = _config_for(input_dir, tmp_path / "out")

    report = verify_schema(cfg, strict=False)

    assert report.ok is False
    assert report.files["seats.parquet"].missing_columns == ["player_id"]
    assert any("player_id" in e for e in report.errors)


def test_verify_schema_reports_absent_data_gracefully(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    cfg = _config_for(input_dir, tmp_path / "out")

    # Even in strict mode, absent data must not raise.
    report = verify_schema(cfg, strict=True)

    assert report.data_present is False
    assert report.ok is False
    assert all(report.files[name].present is False for name in EXPECTED_FILES)
    assert any("No competition files found" in e for e in report.errors)


def test_record_findings_writes_dossier(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    _write_valid_fixtures(input_dir)
    cfg = _config_for(input_dir, tmp_path / "out")
    report = verify_schema(cfg, strict=False)

    dossier = tmp_path / "RESEARCH_DOSSIER.md"
    written = record_findings(report, dossier_path=dossier)

    assert written == dossier
    text = dossier.read_text(encoding="utf-8")
    assert "Schema & Join-Key Verification" in text
    assert "actions.parquet" in text
    assert "development" in text  # phase value recorded

    # Appending a second time keeps prior content (living dossier).
    record_findings(report, dossier_path=dossier)
    text2 = dossier.read_text(encoding="utf-8")
    assert text2.count("Schema & Join-Key Verification") == 2


def test_kaggle_download_command_is_documented(tmp_path: Path) -> None:
    cfg = _config_for(tmp_path / "data", tmp_path / "out")
    cmd = kaggle_download_command(cfg)
    assert "kaggle competitions download" in cmd
    assert fetch_data.COMPETITION_SLUG in cmd
