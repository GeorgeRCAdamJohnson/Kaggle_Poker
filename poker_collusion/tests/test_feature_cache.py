"""Hermetic tests for the FEATURE-MATRIX CACHE (``poker_collusion.features.feature_cache``).

Uses the SAME tiny synthetic competition dataset as ``test_pipeline_phase2_fast.py`` (real schema,
hex-ish string ids, a handful of hands/pairs/actions with development-period confirmed labels), so
BOTH matrices can actually be built and measured:

  * eval pairs PAB (UA,UB coordinated), PCD (UC,UD benign), PAE (UA,UE — DEV-only shared hand, so
    NO eval shared hands -> the has_eval_signal=False default row);
  * dev labels DAB (confirmed_target/directed_transfer) and DCD (confirmed_non_target/none).

Never reads the real ``data/poker`` files. Small and fast; no property-based testing here.

Assertions:
  (a) build -> load round-trip preserves the eval matrix values exactly + writes the sidecar;
  (b) the eval matrix covers the EXACT evaluation pair set (incl. the no-eval-hands default row);
  (c) build -> load round-trip preserves the dev matrix + the dev matrix has correct labels/pools;
  (d) both sidecars carry the ordered feature columns, versions, row count, hash, timestamp, and
      the loader's soft staleness check flags a changed input hash without raising.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.features.feature_cache import (
    DEV_MATRIX_NAME,
    EVAL_MATRIX_NAME,
    HAS_EVAL_SIGNAL_COLUMN,
    LABEL_COLUMN,
    PAIR_ID_COLUMN,
    POOL_COLUMN,
    build_and_cache_dev_matrix,
    build_and_cache_eval_matrix,
    load_dev_matrix,
    load_eval_matrix,
)

# --------------------------------------------------------------------------- #
# Synthetic dataset (identical to test_pipeline_phase2_fast.py).
# --------------------------------------------------------------------------- #
_HANDS = pd.DataFrame(
    {
        "hand_id": ["D01", "D02", "D03", "D04", "H02", "H03", "H04", "H05", "H06", "H07"],
        "table_id": ["T1"] * 10,
        "phase": [
            "development",
            "development",
            "development",
            "development",
            "evaluation",
            "evaluation",
            "evaluation",
            "evaluation",
            "evaluation",
            "development",
        ],
        "big_blind": [2] * 10,
        "small_blind": [1] * 10,
        "started_at": list(range(1, 11)),
        "button_seat": [0] * 10,
    }
)

_PLAYERS = pd.DataFrame(
    {
        "player_id": ["UA", "UB", "UC", "UD", "UE"],
        "account_age_days": [10, 20, 30, 40, 50],
        "experience_hands_bucket": ["a", "b", "c", "d", "e"],
        "preferred_stake": ["s"] * 5,
        "region_bucket": ["r"] * 5,
        "client_family": ["cf"] * 5,
    }
)


def _seat(hand_id: str, player_id: str, seat_no: int, net_chips: float, folded: bool = False):
    return {
        "hand_id": hand_id,
        "player_id": player_id,
        "seat_no": seat_no,
        "starting_stack": 200,
        "total_contribution": abs(net_chips) if net_chips < 0 else 0,
        "net_chips": net_chips,
        "folded": folded,
        "went_to_showdown": not folded,
        "won_share": 1.0 if net_chips > 0 else 0.0,
    }


_SEATS = pd.DataFrame(
    [
        _seat("D01", "UA", 0, -20),
        _seat("D01", "UB", 1, 20),
        _seat("D01", "UC", 2, 0, folded=True),
        _seat("D02", "UA", 0, -20),
        _seat("D02", "UB", 1, 20),
        _seat("D02", "UC", 2, 0, folded=True),
        _seat("D03", "UC", 0, 2),
        _seat("D03", "UD", 1, -2),
        _seat("D03", "UA", 2, 0, folded=True),
        _seat("D04", "UC", 0, -2),
        _seat("D04", "UD", 1, 2),
        _seat("D04", "UB", 2, 0, folded=True),
        _seat("H02", "UA", 0, -20),
        _seat("H02", "UB", 1, 20),
        _seat("H02", "UC", 2, 0, folded=True),
        _seat("H03", "UA", 0, -20),
        _seat("H03", "UB", 1, 20),
        _seat("H03", "UC", 2, 0, folded=True),
        _seat("H04", "UA", 0, -18),
        _seat("H04", "UB", 1, 18),
        _seat("H04", "UC", 2, 0, folded=True),
        _seat("H05", "UC", 0, 2),
        _seat("H05", "UD", 1, -2),
        _seat("H05", "UA", 2, 0, folded=True),
        _seat("H06", "UC", 0, -2),
        _seat("H06", "UD", 1, 2),
        _seat("H06", "UB", 2, 0, folded=True),
        _seat("H07", "UA", 0, -2),
        _seat("H07", "UE", 1, 2),
    ]
)


def _act(hand_id, action_no, player_id, action, amount, to_call, players_active):
    return {
        "hand_id": hand_id,
        "action_no": action_no,
        "street": "pre",
        "player_id": player_id,
        "action": action,
        "amount": amount,
        "amount_to": amount,
        "to_call": to_call,
        "pot_before": 0,
        "stack_before": 200,
        "players_active": players_active,
    }


_ACTIONS = pd.DataFrame(
    [
        _act("D01", 0, "UC", "fold", 0, 2, 3),
        _act("D01", 1, "UA", "bet", 20, 0, 2),
        _act("D01", 2, "UB", "call", 20, 20, 2),
        _act("D02", 0, "UC", "fold", 0, 2, 3),
        _act("D02", 1, "UA", "bet", 20, 0, 2),
        _act("D02", 2, "UB", "call", 20, 20, 2),
        _act("D03", 0, "UA", "fold", 0, 2, 3),
        _act("D03", 1, "UC", "bet", 2, 0, 2),
        _act("D03", 2, "UD", "raise", 4, 2, 2),
        _act("D03", 3, "UC", "call", 4, 4, 2),
        _act("D04", 0, "UB", "fold", 0, 2, 3),
        _act("D04", 1, "UD", "bet", 2, 0, 2),
        _act("D04", 2, "UC", "raise", 4, 2, 2),
        _act("D04", 3, "UD", "call", 4, 4, 2),
        _act("H02", 0, "UC", "fold", 0, 2, 3),
        _act("H02", 1, "UA", "bet", 20, 0, 2),
        _act("H02", 2, "UB", "call", 20, 20, 2),
        _act("H03", 0, "UC", "fold", 0, 2, 3),
        _act("H03", 1, "UA", "bet", 20, 0, 2),
        _act("H03", 2, "UB", "call", 20, 20, 2),
        _act("H04", 0, "UC", "fold", 0, 2, 3),
        _act("H04", 1, "UA", "bet", 18, 0, 2),
        _act("H04", 2, "UB", "call", 18, 18, 2),
        _act("H05", 0, "UA", "fold", 0, 2, 3),
        _act("H05", 1, "UC", "bet", 2, 0, 2),
        _act("H05", 2, "UD", "raise", 4, 2, 2),
        _act("H05", 3, "UC", "call", 4, 4, 2),
        _act("H06", 0, "UB", "fold", 0, 2, 3),
        _act("H06", 1, "UD", "bet", 2, 0, 2),
        _act("H06", 2, "UC", "raise", 4, 2, 2),
        _act("H06", 3, "UD", "call", 4, 4, 2),
        _act("H07", 0, "UA", "bet", 2, 0, 2),
    ]
)

_DEV_LABELS = pd.DataFrame(
    {
        "pair_id": ["DAB", "DCD"],
        "player_1": ["UA", "UC"],
        "player_2": ["UB", "UD"],
        "label": [1, 0],
        "label_status": ["confirmed_target", "confirmed_non_target"],
        "behavior_family": ["directed_transfer", "none"],
    }
)

_DEV_EVIDENCE = pd.DataFrame(
    {
        "pair_id": pd.Series([], dtype=str),
        "evidence_rank": pd.Series([], dtype=int),
        "hand_id": pd.Series([], dtype=str),
        "behavior_family": pd.Series([], dtype=str),
    }
)

_EVAL_PAIRS = pd.DataFrame(
    {
        "pair_id": ["PAB", "PCD", "PAE"],
        "player_1": ["UA", "UC", "UA"],
        "player_2": ["UB", "UD", "UE"],
        "shared_hands": [3, 2, 0],
    }
)

_SAMPLE_SUBMISSION = pd.DataFrame(
    {
        "pair_id": ["PAB", "PCD", "PAE"],
        "risk_score": [0.0, 0.0, 0.0],
        "predicted_behavior": ["none", "none", "none"],
        "evidence_hand_1": ["NO_EVIDENCE"] * 3,
        "evidence_hand_2": ["NO_EVIDENCE"] * 3,
        "evidence_hand_3": ["NO_EVIDENCE"] * 3,
        "evidence_hand_4": ["NO_EVIDENCE"] * 3,
        "evidence_hand_5": ["NO_EVIDENCE"] * 3,
    }
)


def _write_fixture(root: Path) -> None:
    _HANDS.to_parquet(root / "hands.parquet")
    _PLAYERS.to_parquet(root / "players.parquet")
    _SEATS.to_parquet(root / "seats.parquet")
    _ACTIONS.to_parquet(root / "actions.parquet")
    _DEV_LABELS.to_csv(root / "development_labels.csv", index=False)
    _DEV_EVIDENCE.to_csv(root / "development_evidence.csv", index=False)
    _EVAL_PAIRS.to_csv(root / "evaluation_pairs.csv", index=False)
    _SAMPLE_SUBMISSION.to_csv(root / "sample_submission.csv", index=False)


def _config_for(root: Path, out: Path) -> PipelineConfig:
    cfg = PipelineConfig(input_dir=root, output_dir=out)
    cfg.row_group_size = 4  # force multi-batch streaming of actions
    return cfg


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    _write_fixture(root)
    return tmp_path


# --------------------------------------------------------------------------- #
# (a) EVAL build -> load round-trip preserves values + writes the sidecar
# --------------------------------------------------------------------------- #
def test_eval_build_load_round_trip_preserves_values(dataset: Path) -> None:
    out = dataset / "cache"
    cfg = _config_for(dataset / "data", out)

    matrix_path, sidecar_path = build_and_cache_eval_matrix(cfg, out)
    assert matrix_path.exists()
    assert sidecar_path.exists()
    assert matrix_path.name == f"{EVAL_MATRIX_NAME}.parquet"

    on_disk = pd.read_parquet(matrix_path)
    loaded, schema = load_eval_matrix(out, config=cfg)

    # Round-trip is exact (values + column order).
    pd.testing.assert_frame_equal(loaded, on_disk)

    # The sidecar's feature columns are exactly the matrix's feature columns, in order.
    feature_cols = list(schema["feature_columns"])
    expected_cols = [PAIR_ID_COLUMN] + feature_cols + [HAS_EVAL_SIGNAL_COLUMN]
    assert list(loaded.columns) == expected_cols
    assert schema["row_count"] == len(loaded)
    assert schema["n_features"] == len(feature_cols)
    assert schema["matrix_kind"] == "eval"
    assert schema["content_hash"]
    assert schema["build_timestamp"]
    assert schema["feature_schema_version"] == cfg.schema_version
    assert schema["threshold_version"] == cfg.threshold_version


# --------------------------------------------------------------------------- #
# (b) EVAL matrix covers the EXACT evaluation pair set (incl. no-eval-hands row)
# --------------------------------------------------------------------------- #
def test_eval_matrix_covers_exact_eval_pair_set(dataset: Path) -> None:
    out = dataset / "cache"
    cfg = _config_for(dataset / "data", out)
    build_and_cache_eval_matrix(cfg, out)
    loaded, _schema = load_eval_matrix(out, config=cfg)

    assert set(loaded[PAIR_ID_COLUMN]) == {"PAB", "PCD", "PAE"}
    assert len(loaded) == 3

    by_pair = loaded.set_index(PAIR_ID_COLUMN)
    # PAB / PCD share eval hands -> has_eval_signal True; PAE has none -> False + all-zero features.
    assert bool(by_pair.loc["PAB", HAS_EVAL_SIGNAL_COLUMN]) is True
    assert bool(by_pair.loc["PCD", HAS_EVAL_SIGNAL_COLUMN]) is True
    assert bool(by_pair.loc["PAE", HAS_EVAL_SIGNAL_COLUMN]) is False

    feature_cols = [c for c in loaded.columns if c not in (PAIR_ID_COLUMN, HAS_EVAL_SIGNAL_COLUMN)]
    pae_feats = by_pair.loc["PAE", feature_cols].to_numpy(dtype=float)
    assert (pae_feats == 0.0).all(), "no-eval-hands pair must be an all-zero feature row"
    # The coordinated pair should have at least one non-zero feature (real signal built).
    pab_feats = by_pair.loc["PAB", feature_cols].to_numpy(dtype=float)
    assert (pab_feats != 0.0).any()


# --------------------------------------------------------------------------- #
# (c) DEV build -> load round-trip preserves values + correct labels/pools
# --------------------------------------------------------------------------- #
def test_dev_build_load_round_trip_labels_and_pools(dataset: Path) -> None:
    out = dataset / "cache"
    cfg = _config_for(dataset / "data", out)

    matrix_path, sidecar_path = build_and_cache_dev_matrix(cfg, out)
    assert matrix_path.exists()
    assert sidecar_path.exists()
    assert matrix_path.name == f"{DEV_MATRIX_NAME}.parquet"

    on_disk = pd.read_parquet(matrix_path)
    loaded, schema = load_dev_matrix(out, config=cfg)
    pd.testing.assert_frame_equal(loaded, on_disk)

    # Both confirmed pairs have development shared hands -> both rows present.
    assert set(loaded[PAIR_ID_COLUMN]) == {"DAB", "DCD"}
    by_pair = loaded.set_index(PAIR_ID_COLUMN)

    # Correct labels: DAB confirmed_target -> 1, DCD confirmed_non_target -> 0.
    assert int(by_pair.loc["DAB", LABEL_COLUMN]) == 1
    assert int(by_pair.loc["DCD", LABEL_COLUMN]) == 0

    # Correct pools: both pairs live in table T1.
    assert str(by_pair.loc["DAB", POOL_COLUMN]) == "T1"
    assert str(by_pair.loc["DCD", POOL_COLUMN]) == "T1"

    # Sidecar records the split counts + pool count.
    assert schema["matrix_kind"] == "dev"
    assert schema["n_pos"] == 1
    assert schema["n_neg"] == 1
    assert schema["n_pools"] == 1
    assert schema["row_count"] == 2

    feature_cols = list(schema["feature_columns"])
    expected_cols = [PAIR_ID_COLUMN] + feature_cols + [LABEL_COLUMN, POOL_COLUMN]
    assert list(loaded.columns) == expected_cols


# --------------------------------------------------------------------------- #
# (d) soft staleness check flags a changed input hash without raising
# --------------------------------------------------------------------------- #
def test_soft_staleness_flag_on_changed_inputs(dataset: Path) -> None:
    out = dataset / "cache"
    data_dir = dataset / "data"
    cfg = _config_for(data_dir, out)

    build_and_cache_eval_matrix(cfg, out)

    # Fresh load: not stale.
    _loaded, schema = load_eval_matrix(out, config=cfg)
    assert schema["is_stale"] is False

    # Mutate an input file's bytes -> the content hash changes -> soft-flagged (never raises).
    (data_dir / "hands.parquet").touch()  # bumps mtime_ns
    import time

    time.sleep(0.01)
    _HANDS.to_parquet(data_dir / "hands.parquet")  # rewrite to change size/mtime deterministically

    _loaded2, schema2 = load_eval_matrix(out, config=cfg)
    assert schema2["is_stale"] is True
    assert schema2["staleness_reason"]

    # check_staleness=False disables the check entirely.
    _loaded3, schema3 = load_eval_matrix(out, config=cfg, check_staleness=False)
    assert schema3["is_stale"] is False
