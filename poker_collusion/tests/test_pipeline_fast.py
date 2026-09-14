"""Hermetic equality test: ``run_baseline_pipeline_fast`` == ``run_baseline_pipeline``.

Reuses the tiny synthetic competition dataset from ``test_pipeline.py`` (real schema, hex-ish
string ids, a handful of hands/pairs/actions) and proves the vectorized generator produces a
submission that is IDENTICAL to the original slow pipeline:

  * same evaluation-pair row set;
  * per-pair ``risk_score`` within 1e-9, exact ``predicted_behavior`` and evidence strings;
  * byte-for-byte identical ``submission.csv`` (the strongest form of the correctness claim);
  * schema-valid (the writer's own validator passes);
  * deterministic (two fast runs are byte-identical).

Never reads the real ``data/poker`` files. Small and fast; no property-based testing here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.io import DataLoader
from poker_collusion.pipeline import run_baseline_pipeline
from poker_collusion.pipeline_fast import (
    build_eval_hands_by_player,
    run_baseline_pipeline_fast,
)
from poker_collusion.submission.writer import validate_submission

# Reuse the exact synthetic fixture builder + config helper from the slow-pipeline test so the
# two pipelines are compared on identical inputs (single source of truth for the fixture).
from poker_collusion.tests.test_pipeline import _config_for, _write_fixture


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    _write_fixture(tmp_path)
    return tmp_path


def _run(fn, root: Path) -> pd.DataFrame:
    """Write the fixture into ``root``, run ``fn``, return the submission frame."""
    cfg = _config_for(root)
    out = fn(cfg)
    assert out == cfg.submission_path
    assert out.exists()
    return pd.read_csv(out)


# --------------------------------------------------------------------------- #
# Correctness: fast == slow
# --------------------------------------------------------------------------- #
def test_fast_matches_slow_byte_identical(dataset: Path, tmp_path_factory) -> None:
    # Slow pipeline into its own dir.
    cfg_slow = _config_for(dataset)
    out_slow = run_baseline_pipeline(cfg_slow)
    bytes_slow = out_slow.read_bytes()

    # Fast pipeline into a fresh dir seeded with the SAME fixture.
    fast_root = tmp_path_factory.mktemp("fast_run")
    _write_fixture(fast_root)
    cfg_fast = _config_for(fast_root)
    out_fast = run_baseline_pipeline_fast(cfg_fast)
    bytes_fast = out_fast.read_bytes()

    assert bytes_fast == bytes_slow


def test_fast_matches_slow_per_pair(dataset: Path, tmp_path_factory) -> None:
    slow_frame = _run(run_baseline_pipeline, dataset)

    fast_root = tmp_path_factory.mktemp("fast_run_perpair")
    _write_fixture(fast_root)
    fast_frame = _run(run_baseline_pipeline_fast, fast_root)

    # Same pair set / row count.
    assert set(fast_frame["pair_id"]) == set(slow_frame["pair_id"])
    assert len(fast_frame) == len(slow_frame)

    slow = slow_frame.set_index("pair_id")
    fast = fast_frame.set_index("pair_id")

    evidence_cols = [f"evidence_hand_{i}" for i in range(1, 6)]
    for pair_id in slow.index:
        # risk_score within 1e-9
        assert abs(
            float(fast.loc[pair_id, "risk_score"])
            - float(slow.loc[pair_id, "risk_score"])
        ) <= 1e-9, f"risk mismatch for {pair_id}"
        # exact behavior string
        assert str(fast.loc[pair_id, "predicted_behavior"]) == str(
            slow.loc[pair_id, "predicted_behavior"]
        ), f"behavior mismatch for {pair_id}"
        # exact evidence strings, in order
        for col in evidence_cols:
            assert str(fast.loc[pair_id, col]) == str(
                slow.loc[pair_id, col]
            ), f"evidence mismatch for {pair_id}/{col}"


# --------------------------------------------------------------------------- #
# Schema validity + determinism of the fast path itself
# --------------------------------------------------------------------------- #
def test_fast_writes_schema_valid_submission(dataset: Path) -> None:
    frame = _run(run_baseline_pipeline_fast, dataset)
    loader = DataLoader(config=_config_for(dataset))
    sample = loader.load_sample_submission()
    validate_submission(frame, sample)
    assert set(frame["pair_id"]) == {"PAB", "PCD", "PAE"}
    assert len(frame) == 3


def test_fast_is_deterministic_byte_identical(dataset: Path, tmp_path_factory) -> None:
    cfg1 = _config_for(dataset)
    out1 = run_baseline_pipeline_fast(cfg1)
    bytes1 = out1.read_bytes()

    second_root = tmp_path_factory.mktemp("fast_second_run")
    _write_fixture(second_root)
    cfg2 = _config_for(second_root)
    out2 = run_baseline_pipeline_fast(cfg2)
    bytes2 = out2.read_bytes()

    assert bytes1 == bytes2


def test_fast_limit_pairs_still_emits_full_submission(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_baseline_pipeline_fast(cfg, limit_pairs=1)
    frame = pd.read_csv(out)
    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()
    validate_submission(frame, sample)
    assert set(frame["pair_id"]) == {"PAB", "PCD", "PAE"}


# --------------------------------------------------------------------------- #
# Unit test for the vectorized precompute step
# --------------------------------------------------------------------------- #
def test_build_eval_hands_by_player_matches_shared_hands(dataset: Path) -> None:
    """The precomputed per-player eval-hand sets reproduce DataLoader.shared_hands exactly."""
    cfg = _config_for(dataset)
    loader = DataLoader(config=cfg)
    seats = loader.load_seats()
    hands = loader.load_hands()

    by_player = build_eval_hands_by_player(seats, hands)

    # PAB (UA, UB): coordinated eval hands H02, H03, H04 (H01/H07 are development).
    assert by_player.get("UA", frozenset()) & by_player.get("UB", frozenset()) == {
        "H02",
        "H03",
        "H04",
    }
    # Cross-check against the authoritative slow resolver for every eval pair.
    for player_1, player_2 in (("UA", "UB"), ("UC", "UD"), ("UA", "UE")):
        expected = set(loader.shared_hands((player_1, player_2)).evaluation)
        got = by_player.get(player_1, frozenset()) & by_player.get(player_2, frozenset())
        assert set(got) == expected, f"mismatch for pair ({player_1}, {player_2})"
