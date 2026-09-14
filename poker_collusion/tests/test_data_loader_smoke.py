"""Light smoke tests for the DataLoader (task 7.1).

These use tiny synthetic parquet/CSV fixtures written to a temp directory so they run
fast and never touch the 18M-row real ``actions.parquet``. Exhaustive fixture / edge-case
tests are the separate task 7.2; this file only sanity-checks the core behaviors:
chunked+ordered actions, per-pool filtering, single-hand actions, PU label structure,
shared-hand dev/eval partitioning, and SchemaError on missing file/column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.exceptions import SchemaError
from poker_collusion.io import DataLoader
from poker_collusion.types import LabelTable, Pair


def _write_fixture(root: Path) -> None:
    """Write a tiny but schema-complete competition dataset under ``root``."""
    # Two pools (tables). Hands H1,H2 in pool T1 (dev, eval); H3 in pool T2 (dev).
    hands = pd.DataFrame(
        {
            "hand_id": ["H1", "H2", "H3"],
            "table_id": ["T1", "T1", "T2"],
            "started_at": [1, 2, 3],
            "phase": ["development", "evaluation", "development"],
            "button_seat": [0, 1, 2],
            "small_blind": [1, 1, 1],
            "big_blind": [2, 2, 2],
            "board_cards": ["", "", ""],
            "final_pot": [10, 20, 30],
            "players_dealt": [3, 3, 2],
            "players_at_showdown": [2, 2, 2],
        }
    )
    hands.to_parquet(root / "hands.parquet")

    players = pd.DataFrame(
        {
            "player_id": ["UA", "UB", "UC"],
            "account_age_days": [10, 20, 30],
            "experience_hands_bucket": ["a", "b", "c"],
            "preferred_stake": ["s", "s", "s"],
            "region_bucket": ["r", "r", "r"],
            "client_family": ["cf", "cf", "cf"],
        }
    )
    players.to_parquet(root / "players.parquet")

    # Seats: UA+UB share H1 and H2 (dev + eval); UA+UC share H3 (dev only).
    seats = pd.DataFrame(
        {
            "hand_id": ["H1", "H1", "H2", "H2", "H3", "H3"],
            "player_id": ["UA", "UB", "UA", "UB", "UA", "UC"],
            "seat_no": [0, 1, 0, 1, 0, 1],
            "starting_stack": [100] * 6,
            "hole_card_1": ["Ah"] * 6,
            "hole_card_2": ["Kh"] * 6,
            "total_contribution": [5, 3, 4, 6, 2, 8],
            "net_chips": [2, -2, -3, 3, 5, -5],
            "folded": [False] * 6,
            "went_to_showdown": [True] * 6,
            "won_share": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        }
    )
    seats.to_parquet(root / "seats.parquet")

    # Actions: deliberately store rows out of action_no order within a hand and with
    # hand blocks not globally sorted, to exercise the (hand_id, action_no) sort.
    actions = pd.DataFrame(
        {
            "hand_id": ["H2", "H2", "H1", "H1", "H1", "H3"],
            "action_no": [1, 0, 2, 0, 1, 0],
            "street": ["pre"] * 6,
            "player_id": ["UA", "UB", "UA", "UB", "UA", "UC"],
            "action": ["bet", "call", "raise", "bet", "call", "bet"],
            "amount": [4, 2, 6, 2, 4, 3],
            "amount_to": [4, 2, 6, 2, 4, 3],
            "pot_before": [2, 0, 4, 0, 2, 0],
            "stack_before": [100] * 6,
            "to_call": [2, 2, 4, 0, 4, 0],
            "players_active": [2, 2, 2, 2, 2, 2],
        }
    )
    actions.to_parquet(root / "actions.parquet")

    pd.DataFrame(
        {
            "pair_id": ["PPOS", "PNEG"],
            "player_1": ["UA", "UB"],
            "player_2": ["UB", "UC"],
            "label": [1, 0],
            "label_status": ["confirmed_target", "confirmed_non_target"],
            "behavior_family": ["soft_play", "none"],
        }
    ).to_csv(root / "development_labels.csv", index=False)

    pd.DataFrame(
        {
            "pair_id": ["PPOS"],
            "evidence_rank": [1],
            "hand_id": ["H1"],
            "behavior_family": ["soft_play"],
        }
    ).to_csv(root / "development_evidence.csv", index=False)

    pd.DataFrame(
        {
            "pair_id": ["PEVAL"],
            "player_1": ["UA"],
            "player_2": ["UB"],
            "shared_hands": [2],
        }
    ).to_csv(root / "evaluation_pairs.csv", index=False)

    pd.DataFrame(
        {
            "pair_id": ["PEVAL"],
            "risk_score": [0.0],
            "predicted_behavior": ["none"],
            "evidence_hand_1": ["NO_EVIDENCE"],
            "evidence_hand_2": ["NO_EVIDENCE"],
            "evidence_hand_3": ["NO_EVIDENCE"],
            "evidence_hand_4": ["NO_EVIDENCE"],
            "evidence_hand_5": ["NO_EVIDENCE"],
        }
    ).to_csv(root / "sample_submission.csv", index=False)


@pytest.fixture()
def loader(tmp_path: Path) -> DataLoader:
    _write_fixture(tmp_path)
    cfg = PipelineConfig(input_dir=tmp_path, output_dir=tmp_path)
    cfg.row_group_size = 2  # tiny batches to force multi-batch streaming
    return DataLoader(config=cfg)


def test_column_projected_loads(loader: DataLoader) -> None:
    players = loader.load_players()
    assert list(players.columns) == loader.columns["players"]
    hands = loader.load_hands()
    assert {"hand_id", "table_id", "phase"} <= set(hands.columns)
    seats = loader.load_seats()
    assert {"hand_id", "player_id"} <= set(seats.columns)


def test_iter_actions_batches_are_ordered_within_hand(loader: DataLoader) -> None:
    batches = list(loader.iter_actions())
    assert batches, "expected at least one batch"
    # The contract: each yielded BATCH is sorted by (hand_id, action_no). (A hand's rows
    # are contiguous in the real file; global cross-batch hand order is not promised, so
    # per-hand order across the whole stream is obtained via iter_hand_actions.)
    for batch in batches:
        keys = list(zip(batch["hand_id"].tolist(), batch["action_no"].tolist()))
        assert keys == sorted(keys)
    # All rows are still present exactly once across batches.
    combined = pd.concat(batches, ignore_index=True)
    assert len(combined) == 6

    # Larger batch size => whole hands land together and are per-hand ordered.
    loader.row_group_size = 100
    combined_big = pd.concat(list(loader.iter_actions()), ignore_index=True)
    for _, grp in combined_big.groupby("hand_id", sort=False):
        assert grp["action_no"].is_monotonic_increasing


def test_iter_actions_per_pool_filter(loader: DataLoader) -> None:
    pool_hands = pd.concat(list(loader.iter_actions(pool_id="T1")), ignore_index=True)
    assert set(pool_hands["hand_id"]) == {"H1", "H2"}  # T2's H3 excluded
    # Unknown pool yields nothing.
    assert list(loader.iter_actions(pool_id="TZ")) == []


def test_iter_hand_actions_single_hand_ordered(loader: DataLoader) -> None:
    h1 = loader.iter_hand_actions("H1")
    assert list(h1["hand_id"].unique()) == ["H1"]
    assert h1["action_no"].tolist() == [0, 1, 2]
    assert loader.iter_hand_actions("NOPE").empty


def test_load_labels_pu_structure(loader: DataLoader) -> None:
    labels = loader.load_labels()
    assert isinstance(labels, LabelTable)
    assert labels.trusted_positive == {"PPOS": "soft_play"}
    assert labels.confirmed_negative == {"PNEG"}
    # Unknown pairs are absent from both (not treated as negative).
    frame = loader.load_labels_frame()
    assert list(frame.columns) == [
        "pair_id",
        "player_1",
        "player_2",
        "label",
        "label_status",
        "behavior_family",
    ]


def test_shared_hands_dev_eval_partition(loader: DataLoader) -> None:
    pair = Pair(pair_id="PEVAL", player_a="UA", player_b="UB", pool_id="T1")
    sh = loader.shared_hands(pair)
    assert sh.development == ["H1"]  # H1 is development phase
    assert sh.evaluation == ["H2"]  # H2 is evaluation phase
    assert set(sh.all_hands) == {"H1", "H2"}
    # Sequence form works too.
    sh2 = loader.shared_hands(("UA", "UC"))
    assert sh2.development == ["H3"] and sh2.evaluation == []


def test_evaluation_pairs_and_sample_submission(loader: DataLoader) -> None:
    ev = loader.load_evaluation_pairs()
    assert list(ev.columns) == ["pair_id", "player_1", "player_2", "shared_hands"]
    sub = loader.load_sample_submission()
    assert list(sub.columns)[:3] == ["pair_id", "risk_score", "predicted_behavior"]


def test_missing_file_raises_schema_error(tmp_path: Path) -> None:
    cfg = PipelineConfig(input_dir=tmp_path, output_dir=tmp_path)
    dl = DataLoader(config=cfg)
    with pytest.raises(SchemaError) as exc:
        dl.load_players()
    assert "players.parquet" in str(exc.value)


def test_missing_column_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    # Corrupt evaluation_pairs to drop a required column.
    pd.DataFrame({"pair_id": ["P1"]}).to_csv(tmp_path / "evaluation_pairs.csv", index=False)
    cfg = PipelineConfig(input_dir=tmp_path, output_dir=tmp_path)
    dl = DataLoader(config=cfg)
    with pytest.raises(SchemaError) as exc:
        dl.load_evaluation_pairs()
    assert exc.value.missing_key in {"player_1", "player_2", "shared_hands"}
