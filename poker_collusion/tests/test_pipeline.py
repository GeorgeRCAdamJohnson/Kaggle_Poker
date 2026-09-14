"""Hermetic end-to-end test for the Phase-1 baseline pipeline (task 14.2 companion / task 14.1).

Builds a TINY synthetic competition dataset (real schema, hex-string ids, a handful of
hands/pairs/actions) in ``tmp_path``, points a ``PipelineConfig(input_dir=tmp_path,
output_dir=tmp_path)`` at it, and runs :func:`poker_collusion.pipeline.run_baseline_pipeline`
end-to-end. It NEVER reads the real ``data/poker`` files (in particular never the 18M-row
``actions.parquet``).

Assertions:
  * the pipeline writes a ``submission.csv`` that passes the writer's own validation against the
    sample submission (exact schema/column order, one row per evaluation pair, ``risk_score`` in
    [0, 1], allowed behaviors, every evidence cell non-empty, no repeated hand id per row);
  * a known coordinated pair receives a HIGHER risk than a benign pair (sanity of the wiring);
  * a pair with NO evaluation shared hands gets the documented default row (pool-prior risk,
    behavior ``none``, all ``NO_EVIDENCE``);
  * determinism: two runs produce byte-identical submission files.

The dataset is intentionally small and fast; no property-based testing here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.io import DataLoader
from poker_collusion.pipeline import run_baseline_pipeline
from poker_collusion.submission.writer import validate_submission


# --------------------------------------------------------------------------- #
# Synthetic dataset (real schema, hex-ish string ids).
#
# One pool T1. Players: UA, UB (COORDINATED), UC, UD (BENIGN), UE (partner with no eval hands).
#
# Hands (phase, table):
#   H01 dev  T1 : UA, UB, UC              (dev warm-up, not scored/evidence)
#   H02 eval T1 : UA, UB, UC              coordinated: UA dumps ~10bb -> UB, public bet/call
#   H03 eval T1 : UA, UB, UC              coordinated: UA dumps ~10bb -> UB, public bet/call
#   H04 eval T1 : UA, UB, UC              coordinated: UA dumps ~9bb  -> UB, public bet/call
#   H05 eval T1 : UC, UD, UA              benign: symmetric small pot, UC wins a tiny bit
#   H06 eval T1 : UC, UD, UB              benign: symmetric small pot, UD wins a tiny bit
#   H07 dev  T1 : UA, UE                  UA+UE share only a DEV hand -> no eval shared hands
#
# Pairs evaluated: PAB (UA,UB) coordinated; PCD (UC,UD) benign; PAE (UA,UE) no-eval-hands.
#
# big_blind = 2 throughout, so net_chips 20 == 10 bb.
# --------------------------------------------------------------------------- #

_HANDS = pd.DataFrame(
    {
        "hand_id": ["H01", "H02", "H03", "H04", "H05", "H06", "H07"],
        "table_id": ["T1"] * 7,
        "phase": [
            "development",
            "evaluation",
            "evaluation",
            "evaluation",
            "evaluation",
            "evaluation",
            "development",
        ],
        "big_blind": [2] * 7,
        "small_blind": [1] * 7,
        "started_at": [1, 2, 3, 4, 5, 6, 7],
        "button_seat": [0, 0, 0, 0, 0, 0, 0],
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
        # H01 dev warm-up (UA, UB, UC), roughly neutral.
        _seat("H01", "UA", 0, -2),
        _seat("H01", "UB", 1, 2),
        _seat("H01", "UC", 2, 0),
        # H02 eval: UA -> UB directed dump of 20 chips (10bb); UC folds, loses nothing here.
        _seat("H02", "UA", 0, -20),
        _seat("H02", "UB", 1, 20),
        _seat("H02", "UC", 2, 0, folded=True),
        # H03 eval: UA -> UB dump of 20 chips (10bb).
        _seat("H03", "UA", 0, -20),
        _seat("H03", "UB", 1, 20),
        _seat("H03", "UC", 2, 0, folded=True),
        # H04 eval: UA -> UB dump of 18 chips (9bb).
        _seat("H04", "UA", 0, -18),
        _seat("H04", "UB", 1, 18),
        _seat("H04", "UC", 2, 0, folded=True),
        # H05 eval benign (UC, UD, UA): tiny symmetric pot, UC nets +2, UD -2.
        _seat("H05", "UC", 0, 2),
        _seat("H05", "UD", 1, -2),
        _seat("H05", "UA", 2, 0, folded=True),
        # H06 eval benign (UC, UD, UB): tiny symmetric pot, UD nets +2, UC -2.
        _seat("H06", "UC", 0, -2),
        _seat("H06", "UD", 1, 2),
        _seat("H06", "UB", 2, 0, folded=True),
        # H07 dev (UA, UE): dev-only shared hand for PAE.
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


# Coordinated eval hands: UC folds early (so pair "isolates"/soft-plays), UA bets and UB calls —
# a public value-committing transfer between the pair (directed_transfer + soft_play flags).
_ACTIONS = pd.DataFrame(
    [
        # H02
        _act("H02", 0, "UC", "fold", 0, 2, 3),
        _act("H02", 1, "UA", "bet", 20, 0, 2),
        _act("H02", 2, "UB", "call", 20, 20, 2),
        # H03
        _act("H03", 0, "UC", "fold", 0, 2, 3),
        _act("H03", 1, "UA", "bet", 20, 0, 2),
        _act("H03", 2, "UB", "call", 20, 20, 2),
        # H04
        _act("H04", 0, "UC", "fold", 0, 2, 3),
        _act("H04", 1, "UA", "bet", 18, 0, 2),
        _act("H04", 2, "UB", "call", 18, 18, 2),
        # H05 benign: UA folds, UC bets small, UD calls (pair UC/UD both aggress each other).
        _act("H05", 0, "UA", "fold", 0, 2, 3),
        _act("H05", 1, "UC", "bet", 2, 0, 2),
        _act("H05", 2, "UD", "raise", 4, 2, 2),
        _act("H05", 3, "UC", "call", 4, 4, 2),
        # H06 benign: UB folds, UD bets small, UC calls (pair UC/UD contest each other).
        _act("H06", 0, "UB", "fold", 0, 2, 3),
        _act("H06", 1, "UD", "bet", 2, 0, 2),
        _act("H06", 2, "UC", "raise", 4, 2, 2),
        _act("H06", 3, "UD", "call", 4, 4, 2),
        # H01 dev warm-up actions (not in eval; should be ignored by the eval-only pipeline).
        _act("H01", 0, "UA", "bet", 2, 0, 3),
        _act("H01", 1, "UB", "call", 2, 2, 3),
        # H07 dev action.
        _act("H07", 0, "UA", "bet", 2, 0, 2),
    ]
)

_DEV_LABELS = pd.DataFrame(
    {
        "pair_id": ["PXY"],
        "player_1": ["UB"],
        "player_2": ["UC"],
        "label": [0],
        "label_status": ["confirmed_non_target"],
        "behavior_family": ["none"],
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


def _config_for(root: Path) -> PipelineConfig:
    cfg = PipelineConfig(input_dir=root, output_dir=root)
    cfg.row_group_size = 4  # force multi-batch streaming of actions
    return cfg


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    _write_fixture(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# End-to-end tests
# --------------------------------------------------------------------------- #
def test_pipeline_writes_schema_valid_submission(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_baseline_pipeline(cfg)

    assert out == cfg.submission_path
    assert out.exists()

    frame = pd.read_csv(out)
    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()

    # The writer's own contract validator: exact schema/order, one row per eval pair, risk in
    # [0,1], allowed behaviors, non-empty evidence, no repeated hand id per row.
    validate_submission(frame, sample)

    # One row per evaluation pair, exact pair-id set.
    assert set(frame["pair_id"]) == {"PAB", "PCD", "PAE"}
    assert len(frame) == 3


def test_coordinated_pair_outranks_benign_pair(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_baseline_pipeline(cfg)
    frame = pd.read_csv(out).set_index("pair_id")

    risk_coordinated = float(frame.loc["PAB", "risk_score"])
    risk_benign = float(frame.loc["PCD", "risk_score"])

    # Sanity of the wiring: the directed-dump pair must score strictly higher than the benign one.
    assert risk_coordinated > risk_benign

    # The coordinated pair should carry real evidence in at least the first slot.
    assert str(frame.loc["PAB", "evidence_hand_1"]) != "NO_EVIDENCE"


def test_pair_with_no_eval_hands_gets_default_row(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_baseline_pipeline(cfg)
    frame = pd.read_csv(out).set_index("pair_id")

    # PAE shares only a development hand -> documented default row.
    assert str(frame.loc["PAE", "predicted_behavior"]) == "none"
    for col in (
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    ):
        assert str(frame.loc["PAE", col]) == "NO_EVIDENCE"
    # Risk is the pool-prior default and within [0, 1].
    assert 0.0 <= float(frame.loc["PAE", "risk_score"]) <= 1.0


def test_pipeline_is_deterministic_byte_identical(dataset: Path, tmp_path_factory) -> None:
    cfg1 = _config_for(dataset)
    out1 = run_baseline_pipeline(cfg1)
    bytes1 = out1.read_bytes()

    # Run again into a fresh output dir from the SAME inputs; bytes must match exactly.
    second_root = tmp_path_factory.mktemp("second_run")
    _write_fixture(second_root)
    cfg2 = _config_for(second_root)
    out2 = run_baseline_pipeline(cfg2)
    bytes2 = out2.read_bytes()

    assert bytes1 == bytes2


def test_limit_pairs_still_emits_full_submission(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_baseline_pipeline(cfg, limit_pairs=1)
    frame = pd.read_csv(out)

    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()
    # Even limiting to the first pair, the full evaluation-pair set must be present (rest default).
    validate_submission(frame, sample)
    assert set(frame["pair_id"]) == {"PAB", "PCD", "PAE"}
