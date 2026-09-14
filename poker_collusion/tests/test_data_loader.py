"""Thorough fixture and edge-case tests for the DataLoader (task 7.2).

These complement the lighter ``test_data_loader_smoke.py`` (task 7.1) without
duplicating it. Every test builds a TINY, hermetic parquet/CSV dataset in
``tmp_path`` using the REAL competition schema and hex-string IDs, points a
``DataLoader`` at a ``PipelineConfig(input_dir=tmp_path, output_dir=tmp_path)``, and
never reads the real ``data/poker`` files (in particular never the 18M-row
``actions.parquet``).

Coverage (Requirements 1.1–1.5):
  * Joins & column projection — players/hands/seats project to configured columns;
    ``hand_id`` links gameplay tables, ``player_id`` links player tables.
  * ``iter_actions`` — chunked iteration yields every row for a pool, column-projected,
    each batch ordered by ``(hand_id, action_no)``; pool filtering resolved via
    ``hands.table_id``; unknown pool -> no rows; ``iter_hand_actions`` returns one hand
    in ``action_no`` order.
  * ``shared_hands`` — dev/eval partition via ``hands.phase``; a hand seating only one
    player is excluded; players seated in different tables share nothing; empty set OK.
  * Labels PU structure — ``confirmed_target`` -> trusted_positive (with behavior_family),
    ``confirmed_non_target`` -> confirmed_negative, absent pair is neither.
  * ``SchemaError`` — missing file and missing required column across several files, each
    naming the offending file / missing_key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from poker_collusion.config import DEFAULT_COLUMN_SELECTION, PipelineConfig
from poker_collusion.exceptions import SchemaError
from poker_collusion.io import DataLoader, SharedHands
from poker_collusion.types import LabelTable, Pair


# --------------------------------------------------------------------------- #
# Fixture construction — richer than the smoke fixture on purpose.
#
# Three pools (table_id): T1, T2, T3.
#   Players: UA, UB, UC, UD, UE (hex-string ids, "U..." per real schema).
#   Hands (phase, table):
#     H01 dev  T1 : seats UA, UB, UC   (UA+UB shared, dev)
#     H02 eval T1 : seats UA, UB       (UA+UB shared, eval)
#     H03 dev  T1 : seats UA, UB       (UA+UB shared, dev)
#     H04 eval T1 : seats UA, UC       (UA+UC shared, eval; UA+UB NOT shared here)
#     H05 dev  T2 : seats UC, UD       (UC+UD shared, dev)
#     H06 eval T2 : seats UD           (single-seat hand -> excluded from any pair)
#     H07 dev  T3 : seats UE           (UE seated alone in a different table)
#
# So UA+UB share {H01(dev), H02(eval), H03(dev)}: dev=[H01,H03], eval=[H02].
#    UA+UC share {H01(dev), H04(eval)}: dev=[H01], eval=[H04].
#    UC+UD share {H05(dev)}: dev=[H05], eval=[].
#    UA+UD share {} (never co-seated) -> empty shared set.
#    UB+UE share {} (different tables) -> empty shared set.
# --------------------------------------------------------------------------- #

_HANDS = pd.DataFrame(
    {
        "hand_id": ["H01", "H02", "H03", "H04", "H05", "H06", "H07"],
        "table_id": ["T1", "T1", "T1", "T1", "T2", "T2", "T3"],
        "phase": [
            "development",
            "evaluation",
            "development",
            "evaluation",
            "development",
            "evaluation",
            "development",
        ],
        "big_blind": [2, 2, 2, 2, 2, 2, 2],
        "small_blind": [1, 1, 1, 1, 1, 1, 1],
        "started_at": [1, 2, 3, 4, 5, 6, 7],
        "button_seat": [0, 1, 2, 0, 1, 0, 0],
        # extra real-schema columns to prove projection drops them:
        "board_cards": ["", "", "", "", "", "", ""],
        "final_pot": [10, 20, 30, 40, 50, 60, 70],
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
        # extra column to prove projection drops it:
        "vip_flag": [False] * 5,
    }
)

# seats: (hand_id, player_id) rows describing who sits where.
_SEATS = pd.DataFrame(
    {
        "hand_id": [
            "H01", "H01", "H01",  # UA, UB, UC
            "H02", "H02",         # UA, UB
            "H03", "H03",         # UA, UB
            "H04", "H04",         # UA, UC
            "H05", "H05",         # UC, UD
            "H06",                # UD alone
            "H07",                # UE alone
        ],
        "player_id": [
            "UA", "UB", "UC",
            "UA", "UB",
            "UA", "UB",
            "UA", "UC",
            "UC", "UD",
            "UD",
            "UE",
        ],
        "seat_no": [0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        "starting_stack": [100] * 13,
        "total_contribution": [5, 3, 1, 4, 6, 2, 2, 7, 1, 3, 9, 0, 0],
        "net_chips": [2, -2, 0, -3, 3, 1, -1, 4, -4, 5, -5, 0, 0],
        "folded": [False] * 13,
        "went_to_showdown": [True] * 13,
        "won_share": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        # extra column to prove projection drops it:
        "hole_card_1": ["Ah"] * 13,
    }
)

# actions: intentionally stored OUT of (hand_id, action_no) order and with hand blocks
# NOT globally sorted, so the loader's stable (hand_id, action_no) batch-sort is exercised.
#   H02: action_no 1,0 ; H01: 2,0,1 ; H05: 1,0 ; H04: 0 ; H03: 0 ; H07 has no actions.
_ACTIONS = pd.DataFrame(
    {
        "hand_id": ["H02", "H02", "H01", "H01", "H01", "H05", "H05", "H04", "H03"],
        "action_no": [1, 0, 2, 0, 1, 1, 0, 0, 0],
        "street": ["pre"] * 9,
        "player_id": ["UA", "UB", "UA", "UB", "UC", "UC", "UD", "UA", "UA"],
        "action": ["bet", "call", "raise", "bet", "call", "bet", "call", "bet", "bet"],
        "amount": [4, 2, 6, 2, 4, 3, 3, 5, 5],
        "amount_to": [4, 2, 6, 2, 4, 3, 3, 5, 5],
        "to_call": [2, 2, 4, 0, 4, 0, 3, 0, 0],
        "pot_before": [2, 0, 4, 0, 2, 0, 3, 0, 0],
        "stack_before": [100] * 9,
        "players_active": [2, 2, 3, 3, 3, 2, 2, 2, 2],
    }
)

_DEV_LABELS = pd.DataFrame(
    {
        # PAB: confirmed target (soft_play). PCD: confirmed non-target. PAC: unknown pair
        # is deliberately ABSENT from this file (never listed -> neither pos nor neg).
        "pair_id": ["PAB", "PCD"],
        "player_1": ["UA", "UC"],
        "player_2": ["UB", "UD"],
        "label": [1, 0],
        "label_status": ["confirmed_target", "confirmed_non_target"],
        "behavior_family": ["soft_play", "none"],
    }
)

_DEV_EVIDENCE = pd.DataFrame(
    {
        "pair_id": ["PAB"],
        "evidence_rank": [1],
        "hand_id": ["H01"],
        "behavior_family": ["soft_play"],
    }
)

_EVAL_PAIRS = pd.DataFrame(
    {
        "pair_id": ["PAC"],
        "player_1": ["UA"],
        "player_2": ["UC"],
        "shared_hands": [2],
    }
)

_SAMPLE_SUBMISSION = pd.DataFrame(
    {
        "pair_id": ["PAC"],
        "risk_score": [0.0],
        "predicted_behavior": ["none"],
        "evidence_hand_1": ["NO_EVIDENCE"],
        "evidence_hand_2": ["NO_EVIDENCE"],
        "evidence_hand_3": ["NO_EVIDENCE"],
        "evidence_hand_4": ["NO_EVIDENCE"],
        "evidence_hand_5": ["NO_EVIDENCE"],
    }
)


def _write_fixture(root: Path) -> None:
    """Write the full hermetic dataset (real schema, hex ids) under ``root``."""
    _HANDS.to_parquet(root / "hands.parquet")
    _PLAYERS.to_parquet(root / "players.parquet")
    _SEATS.to_parquet(root / "seats.parquet")
    _ACTIONS.to_parquet(root / "actions.parquet")
    _DEV_LABELS.to_csv(root / "development_labels.csv", index=False)
    _DEV_EVIDENCE.to_csv(root / "development_evidence.csv", index=False)
    _EVAL_PAIRS.to_csv(root / "evaluation_pairs.csv", index=False)
    _SAMPLE_SUBMISSION.to_csv(root / "sample_submission.csv", index=False)


def _make_loader(root: Path, row_group_size: int = 2) -> DataLoader:
    """Return a DataLoader pointed at ``root`` with a tiny row-group size.

    A small ``row_group_size`` forces ``iter_actions`` to stream several batches,
    exercising the multi-batch path.
    """
    cfg = PipelineConfig(input_dir=root, output_dir=root)
    cfg.row_group_size = row_group_size
    return DataLoader(config=cfg)


@pytest.fixture()
def loader(tmp_path: Path) -> DataLoader:
    _write_fixture(tmp_path)
    return _make_loader(tmp_path)


# --------------------------------------------------------------------------- #
# Requirement 1.1 / 1.2 / 1.6 — column-projected loads and join keys.
# --------------------------------------------------------------------------- #
def test_loads_project_to_configured_columns_exactly(loader: DataLoader) -> None:
    players = loader.load_players()
    assert list(players.columns) == DEFAULT_COLUMN_SELECTION["players"]
    assert "vip_flag" not in players.columns  # extra source column dropped

    hands = loader.load_hands()
    assert list(hands.columns) == DEFAULT_COLUMN_SELECTION["hands"]
    assert "final_pot" not in hands.columns

    seats = loader.load_seats()
    assert list(seats.columns) == DEFAULT_COLUMN_SELECTION["seats"]
    assert "hole_card_1" not in seats.columns


def test_join_keys_link_the_tables(loader: DataLoader) -> None:
    """hand_id links gameplay tables; player_id links player tables (Req 1.2)."""
    hands = loader.load_hands()
    seats = loader.load_seats()
    players = loader.load_players()

    # Every seat's hand_id exists in hands; every seat's player_id exists in players.
    assert set(seats["hand_id"]).issubset(set(hands["hand_id"]))
    assert set(seats["player_id"]).issubset(set(players["player_id"]))

    # A concrete join round-trips: seats ⋈ hands on hand_id, seats ⋈ players on player_id.
    seats_hands = seats.merge(hands[["hand_id", "table_id", "phase"]], on="hand_id", how="left")
    assert seats_hands["table_id"].notna().all()
    assert seats_hands["phase"].notna().all()
    seats_players = seats.merge(players[["player_id"]], on="player_id", how="left")
    assert len(seats_players) == len(seats)


def test_load_hands_always_carries_join_and_phase_keys(loader: DataLoader) -> None:
    hands = loader.load_hands()
    assert {"hand_id", "table_id", "phase"} <= set(hands.columns)
    assert set(hands["phase"].unique()) <= {"development", "evaluation"}


# --------------------------------------------------------------------------- #
# Requirement 1.2 / 1.6 — iter_actions chunking, ordering, projection, pooling.
# --------------------------------------------------------------------------- #
def test_iter_actions_yields_all_rows_column_projected(loader: DataLoader) -> None:
    batches = list(loader.iter_actions())
    assert batches, "expected at least one streamed batch"
    for batch in batches:
        # Column projection: exactly the configured actions columns, in order.
        assert list(batch.columns) == DEFAULT_COLUMN_SELECTION["actions"]
    combined = pd.concat(batches, ignore_index=True)
    assert len(combined) == len(_ACTIONS)  # every row delivered exactly once


def test_iter_actions_each_batch_sorted_by_hand_then_action_no(loader: DataLoader) -> None:
    for batch in loader.iter_actions():
        keys = list(zip(batch["hand_id"].tolist(), batch["action_no"].tolist()))
        assert keys == sorted(keys), "each batch must be ordered by (hand_id, action_no)"


def test_iter_actions_custom_column_projection(loader: DataLoader) -> None:
    cols = ["hand_id", "player_id", "action"]
    for batch in loader.iter_actions(columns=cols):
        assert list(batch.columns) == cols


def test_iter_actions_pool_filter_resolves_via_hands_table_id(loader: DataLoader) -> None:
    # T1 owns H01..H04; T2 owns H05,H06 (H06 has no actions); T3 owns H07 (no actions).
    t1 = pd.concat(list(loader.iter_actions(pool_id="T1")), ignore_index=True)
    assert set(t1["hand_id"]) == {"H01", "H02", "H03", "H04"}

    t2_batches = list(loader.iter_actions(pool_id="T2"))
    t2 = pd.concat(t2_batches, ignore_index=True) if t2_batches else pd.DataFrame()
    assert set(t2["hand_id"]) == {"H05"}  # H06 has no action rows


def test_iter_actions_unknown_pool_yields_no_rows(loader: DataLoader) -> None:
    assert list(loader.iter_actions(pool_id="TZZ")) == []


def test_iter_actions_pool_with_hands_but_no_actions_yields_no_rows(loader: DataLoader) -> None:
    # T3 has hand H07 but H07 has no rows in actions.parquet.
    assert list(loader.iter_actions(pool_id="T3")) == []


def test_iter_hand_actions_returns_single_hand_in_action_no_order(loader: DataLoader) -> None:
    h01 = loader.iter_hand_actions("H01")
    assert list(h01["hand_id"].unique()) == ["H01"]
    assert h01["action_no"].tolist() == [0, 1, 2]
    assert list(h01.columns) == DEFAULT_COLUMN_SELECTION["actions"]


def test_iter_hand_actions_absent_hand_is_empty(loader: DataLoader) -> None:
    empty = loader.iter_hand_actions("HZZ")
    assert empty.empty
    assert list(empty.columns) == DEFAULT_COLUMN_SELECTION["actions"]


# --------------------------------------------------------------------------- #
# Requirement 1.3 / 1.4 — shared_hands partitioned dev vs eval.
# --------------------------------------------------------------------------- #
def test_shared_hands_partitions_dev_and_eval(loader: DataLoader) -> None:
    sh = loader.shared_hands(("UA", "UB"))
    assert isinstance(sh, SharedHands)
    # UA+UB share H01(dev), H02(eval), H03(dev). H04 seats UA+UC (not UB) -> excluded.
    assert sh.development == ["H01", "H03"]
    assert sh.evaluation == ["H02"]
    assert sh.all_hands == ["H01", "H03", "H02"]  # dev first, then eval
    # The attached [hand_id, phase] frame is consistent and phase-labelled.
    assert set(sh.table.columns) == {"hand_id", "phase"}
    assert dict(zip(sh.table["hand_id"], sh.table["phase"])) == {
        "H01": "development",
        "H02": "evaluation",
        "H03": "development",
    }


def test_shared_hands_excludes_hand_seating_only_one_player(loader: DataLoader) -> None:
    # H04 seats UA and UC only. For UA+UB, H04 must NOT appear anywhere.
    sh = loader.shared_hands(("UA", "UB"))
    assert "H04" not in sh.all_hands
    # For UA+UC, H04 IS shared (both seated) and is evaluation phase.
    sh_ac = loader.shared_hands(("UA", "UC"))
    assert sh_ac.development == ["H01"]
    assert sh_ac.evaluation == ["H04"]


def test_shared_hands_different_tables_have_none(loader: DataLoader) -> None:
    # UB sits only in table T1; UE sits only in table T3 -> never co-seated.
    sh = loader.shared_hands(("UB", "UE"))
    assert sh.development == []
    assert sh.evaluation == []
    assert sh.all_hands == []
    assert sh.table.empty


def test_shared_hands_empty_set_handled(loader: DataLoader) -> None:
    # UA and UD are never seated together (UD only in H05/H06, UA never there).
    sh = loader.shared_hands(("UA", "UD"))
    assert sh.all_hands == []
    assert sh.table.empty


def test_shared_hands_dev_only_pair(loader: DataLoader) -> None:
    # UC+UD share only H05 which is development -> empty evaluation list.
    sh = loader.shared_hands(("UC", "UD"))
    assert sh.development == ["H05"]
    assert sh.evaluation == []


def test_shared_hands_accepts_pair_object_and_keeps_pair_id(loader: DataLoader) -> None:
    pair = Pair(pair_id="PAB", player_a="UA", player_b="UB", pool_id="T1")
    sh = loader.shared_hands(pair)
    assert sh.pair_id == "PAB"
    assert sh.development == ["H01", "H03"]
    assert sh.evaluation == ["H02"]


def test_shared_hands_rejects_wrong_arity_sequence(loader: DataLoader) -> None:
    with pytest.raises(ValueError):
        loader.shared_hands(("UA", "UB", "UC"))


# --------------------------------------------------------------------------- #
# Requirement 6.1 (PU structure surfaced by the loader) — labels.
# --------------------------------------------------------------------------- #
def test_load_labels_pu_structure(loader: DataLoader) -> None:
    labels = loader.load_labels()
    assert isinstance(labels, LabelTable)
    # confirmed_target -> trusted_positive keyed to its behavior_family.
    assert labels.trusted_positive == {"PAB": "soft_play"}
    # confirmed_non_target -> confirmed_negative.
    assert labels.confirmed_negative == {"PCD"}


def test_absent_pair_is_neither_positive_nor_negative(loader: DataLoader) -> None:
    labels = loader.load_labels()
    # PAC (the evaluation pair) is not listed in development_labels -> unknown, never neg.
    assert "PAC" not in labels.trusted_positive
    assert "PAC" not in labels.confirmed_negative


def test_load_labels_frame_keeps_pu_columns(loader: DataLoader) -> None:
    frame = loader.load_labels_frame()
    assert list(frame.columns) == [
        "pair_id",
        "player_1",
        "player_2",
        "label",
        "label_status",
        "behavior_family",
    ]
    assert set(frame["pair_id"]) == {"PAB", "PCD"}


def test_evaluation_pairs_and_sample_submission_schema(loader: DataLoader) -> None:
    ev = loader.load_evaluation_pairs()
    assert list(ev.columns) == ["pair_id", "player_1", "player_2", "shared_hands"]
    sub = loader.load_sample_submission()
    assert list(sub.columns) == [
        "pair_id",
        "risk_score",
        "predicted_behavior",
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    ]


# --------------------------------------------------------------------------- #
# Requirement 1.5 — SchemaError on missing files, naming the file.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("file_name", "call"),
    [
        ("players.parquet", lambda dl: dl.load_players()),
        ("hands.parquet", lambda dl: dl.load_hands()),
        ("seats.parquet", lambda dl: dl.load_seats()),
        ("development_labels.csv", lambda dl: dl.load_labels()),
        ("evaluation_pairs.csv", lambda dl: dl.load_evaluation_pairs()),
        ("sample_submission.csv", lambda dl: dl.load_sample_submission()),
    ],
)
def test_missing_file_raises_schema_error(tmp_path: Path, file_name: str, call) -> None:
    _write_fixture(tmp_path)
    (tmp_path / file_name).unlink()  # delete the required file
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        call(dl)
    assert file_name in str(exc.value)
    assert exc.value.missing_key == f"file:{file_name}"


def test_missing_actions_file_raises_on_iteration(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "actions.parquet").unlink()
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        list(dl.iter_actions())
    assert "actions.parquet" in str(exc.value)


# --------------------------------------------------------------------------- #
# Requirement 1.5 — SchemaError on a missing required COLUMN, naming file + key.
# --------------------------------------------------------------------------- #
def _rewrite_parquet_without(root: Path, name: str, drop: str) -> None:
    df = pd.read_parquet(root / name).drop(columns=[drop])
    df.to_parquet(root / name)


def test_hands_missing_table_id_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _rewrite_parquet_without(tmp_path, "hands.parquet", "table_id")
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_hands()
    assert "hands.parquet" in exc.value.file
    assert exc.value.missing_key == "table_id"


def test_hands_missing_phase_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _rewrite_parquet_without(tmp_path, "hands.parquet", "phase")
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_hands()
    assert exc.value.missing_key == "phase"


def test_seats_missing_player_id_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _rewrite_parquet_without(tmp_path, "seats.parquet", "player_id")
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_seats()
    assert "seats.parquet" in exc.value.file
    assert exc.value.missing_key == "player_id"


def test_players_missing_projected_column_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _rewrite_parquet_without(tmp_path, "players.parquet", "region_bucket")
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_players()
    assert "players.parquet" in exc.value.file
    assert exc.value.missing_key == "region_bucket"


def test_actions_missing_column_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _rewrite_parquet_without(tmp_path, "actions.parquet", "action_no")
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        list(dl.iter_actions())
    assert "actions.parquet" in exc.value.file
    assert exc.value.missing_key == "action_no"


def test_labels_missing_required_column_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _DEV_LABELS.drop(columns=["behavior_family"]).to_csv(
        tmp_path / "development_labels.csv", index=False
    )
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_labels()
    assert "development_labels.csv" in exc.value.file
    assert exc.value.missing_key == "behavior_family"


def test_evaluation_pairs_missing_column_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _EVAL_PAIRS.drop(columns=["shared_hands"]).to_csv(
        tmp_path / "evaluation_pairs.csv", index=False
    )
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_evaluation_pairs()
    assert exc.value.missing_key == "shared_hands"


def test_sample_submission_missing_column_raises_schema_error(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _SAMPLE_SUBMISSION.drop(columns=["evidence_hand_3"]).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    dl = _make_loader(tmp_path)
    with pytest.raises(SchemaError) as exc:
        dl.load_sample_submission()
    assert exc.value.missing_key == "evidence_hand_3"
