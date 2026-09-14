"""Hermetic equality test: the PARALLEL fast path == the SEQUENTIAL fast path (byte-identical).

Protects the reproducibility contract (Property 16): the deterministic multiprocess scoring driver
(:mod:`poker_collusion.parallel_scoring`) MUST produce a ``submission.csv`` that is byte-for-byte
identical to the sequential :func:`run_baseline_pipeline_fast`. Chunks are contiguous pair slices
and the workers call the SAME per-pair functions, so order and values are preserved.

Reuses the tiny synthetic competition fixture from ``test_pipeline.py`` (real schema, hex-ish string
ids, a handful of hands/pairs/actions) — never touches the real ``data/poker`` files. Small and
fast; no property-based testing here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from poker_collusion.io import DataLoader
from poker_collusion.pipeline_fast import run_baseline_pipeline_fast
from poker_collusion.submission.writer import validate_submission

# Reuse the exact synthetic fixture builder + config helper (single source of truth).
from poker_collusion.tests.test_pipeline import _config_for, _write_fixture


# --------------------------------------------------------------------------- #
# Core contract: parallel == sequential, byte-for-byte
# --------------------------------------------------------------------------- #
def test_parallel_matches_sequential_byte_identical(tmp_path_factory) -> None:
    # Sequential fast run into its own dir.
    seq_root = tmp_path_factory.mktemp("seq_run")
    _write_fixture(seq_root)
    cfg_seq = _config_for(seq_root)
    out_seq = run_baseline_pipeline_fast(cfg_seq, parallel=False)
    bytes_seq = out_seq.read_bytes()

    # Parallel fast run (n_workers=2) into a fresh dir seeded with the SAME fixture.
    par_root = tmp_path_factory.mktemp("par_run")
    _write_fixture(par_root)
    cfg_par = _config_for(par_root)
    out_par = run_baseline_pipeline_fast(cfg_par, parallel=True, n_workers=2)
    bytes_par = out_par.read_bytes()

    assert bytes_par == bytes_seq


def test_parallel_matches_sequential_per_pair(tmp_path_factory) -> None:
    seq_root = tmp_path_factory.mktemp("seq_run_pp")
    _write_fixture(seq_root)
    seq_frame = pd.read_csv(run_baseline_pipeline_fast(_config_for(seq_root), parallel=False))

    par_root = tmp_path_factory.mktemp("par_run_pp")
    _write_fixture(par_root)
    par_frame = pd.read_csv(
        run_baseline_pipeline_fast(_config_for(par_root), parallel=True, n_workers=2)
    )

    # Exact pair set + row count.
    assert set(par_frame["pair_id"]) == set(seq_frame["pair_id"])
    assert len(par_frame) == len(seq_frame)

    seq = seq_frame.set_index("pair_id")
    par = par_frame.set_index("pair_id")
    evidence_cols = [f"evidence_hand_{i}" for i in range(1, 6)]
    for pair_id in seq.index:
        assert float(par.loc[pair_id, "risk_score"]) == float(seq.loc[pair_id, "risk_score"])
        assert str(par.loc[pair_id, "predicted_behavior"]) == str(
            seq.loc[pair_id, "predicted_behavior"]
        )
        for col in evidence_cols:
            assert str(par.loc[pair_id, col]) == str(seq.loc[pair_id, col])


# --------------------------------------------------------------------------- #
# Domain + schema validity of the parallel path
# --------------------------------------------------------------------------- #
def test_parallel_risk_in_unit_interval_and_schema_valid(tmp_path_factory) -> None:
    root = tmp_path_factory.mktemp("par_domain")
    _write_fixture(root)
    cfg = _config_for(root)
    frame = pd.read_csv(run_baseline_pipeline_fast(cfg, parallel=True, n_workers=2))

    # risk in [0, 1] for every row.
    assert (frame["risk_score"] >= 0.0).all()
    assert (frame["risk_score"] <= 1.0).all()

    # Schema-valid per the writer's own validator, exact pair set.
    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()
    validate_submission(frame, sample)
    assert set(frame["pair_id"]) == {"PAB", "PCD", "PAE"}


def test_parallel_single_worker_falls_back_and_matches(tmp_path_factory) -> None:
    """n_workers=1 resolves to the sequential path and stays byte-identical."""
    seq_root = tmp_path_factory.mktemp("seq_one")
    _write_fixture(seq_root)
    bytes_seq = run_baseline_pipeline_fast(_config_for(seq_root), parallel=False).read_bytes()

    one_root = tmp_path_factory.mktemp("par_one")
    _write_fixture(one_root)
    bytes_one = run_baseline_pipeline_fast(
        _config_for(one_root), parallel=True, n_workers=1
    ).read_bytes()

    assert bytes_one == bytes_seq
