"""Hermetic tests for the FULL Phase-2 generator ``run_phase2_pipeline_fast``.

Uses the SAME tiny synthetic competition dataset PATTERN as ``test_pipeline_fast.py`` (real
schema, hex-ish string ids, a handful of hands/pairs/actions) but adds DEVELOPMENT-period
confirmed labels — one ``confirmed_target`` positive and one ``confirmed_non_target`` negative,
each with development shared hands — so the PU ranker AND behavior classifier can actually train.
This lets us prove the trained PU risk is genuinely used (differs from the pure Phase-1 classical
output for at least one pair), while still exercising the same no-eval-hands default path.

Never reads the real ``data/poker`` files. Small and fast; no property-based testing here.

Assertions:
  (a) the generator runs end-to-end and writes a schema-valid submission covering exactly the
      evaluation pair set (validated with the writer's own ``validate_submission``);
  (b) every ``risk_score`` lies in [0, 1];
  (c) determinism: two runs produce byte-identical submissions;
  (d) evidence is IDENTICAL to the Phase-1 fast pipeline (only risk + behavior may change);
  (e) either the trained PU risk differs from the Phase-1 classical risk for >=1 pair (proving the
      PU model is used) OR the fallback path is exercised (recorded explicitly).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.io import DataLoader
from poker_collusion.pipeline_fast import run_baseline_pipeline_fast
from poker_collusion.pipeline_phase2_fast import (
    build_dev_hands_by_player,
    run_phase2_pipeline_fast,
)
from poker_collusion.submission.writer import validate_submission

# --------------------------------------------------------------------------- #
# Synthetic dataset (real schema, hex-ish string ids).
#
# One pool T1. Players:
#   UA, UB  -> COORDINATED (directed dumps UA->UB), share DEV + EVAL hands.
#   UC, UD  -> BENIGN, share DEV + EVAL hands.
#   UE      -> shares only a DEV hand with UA (no eval shared hands -> default row).
#
# Hands (phase, table):
#   D01 dev  T1 : UA, UB, UC   coordinated: UA dumps ~10bb -> UB (TRAIN positive signal)
#   D02 dev  T1 : UA, UB, UC   coordinated: UA dumps ~10bb -> UB
#   D03 dev  T1 : UC, UD, UA   benign: symmetric small pot (TRAIN negative signal)
#   D04 dev  T1 : UC, UD, UB   benign: symmetric small pot
#   H02 eval T1 : UA, UB, UC   coordinated dump UA->UB
#   H03 eval T1 : UA, UB, UC   coordinated dump UA->UB
#   H04 eval T1 : UA, UB, UC   coordinated dump UA->UB
#   H05 eval T1 : UC, UD, UA   benign small pot
#   H06 eval T1 : UC, UD, UB   benign small pot
#   H07 dev  T1 : UA, UE       dev-only shared hand for PAE (no eval hands)
#
# Evaluation pairs: PAB (UA,UB), PCD (UC,UD), PAE (UA,UE) no-eval-hands.
# Development labels: DAB (UA,UB) confirmed_target/directed_transfer; DCD (UC,UD)
#   confirmed_non_target/none.
#
# big_blind = 2 throughout, so net_chips 20 == 10 bb.
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
        # DEV coordinated (UA -> UB dumps).
        _seat("D01", "UA", 0, -20),
        _seat("D01", "UB", 1, 20),
        _seat("D01", "UC", 2, 0, folded=True),
        _seat("D02", "UA", 0, -20),
        _seat("D02", "UB", 1, 20),
        _seat("D02", "UC", 2, 0, folded=True),
        # DEV benign (UC, UD).
        _seat("D03", "UC", 0, 2),
        _seat("D03", "UD", 1, -2),
        _seat("D03", "UA", 2, 0, folded=True),
        _seat("D04", "UC", 0, -2),
        _seat("D04", "UD", 1, 2),
        _seat("D04", "UB", 2, 0, folded=True),
        # EVAL coordinated (UA -> UB dumps).
        _seat("H02", "UA", 0, -20),
        _seat("H02", "UB", 1, 20),
        _seat("H02", "UC", 2, 0, folded=True),
        _seat("H03", "UA", 0, -20),
        _seat("H03", "UB", 1, 20),
        _seat("H03", "UC", 2, 0, folded=True),
        _seat("H04", "UA", 0, -18),
        _seat("H04", "UB", 1, 18),
        _seat("H04", "UC", 2, 0, folded=True),
        # EVAL benign (UC, UD).
        _seat("H05", "UC", 0, 2),
        _seat("H05", "UD", 1, -2),
        _seat("H05", "UA", 2, 0, folded=True),
        _seat("H06", "UC", 0, -2),
        _seat("H06", "UD", 1, 2),
        _seat("H06", "UB", 2, 0, folded=True),
        # DEV-only shared hand for PAE.
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
        # DEV coordinated: UC folds, UA bets, UB calls (directed transfer, soft play flags).
        _act("D01", 0, "UC", "fold", 0, 2, 3),
        _act("D01", 1, "UA", "bet", 20, 0, 2),
        _act("D01", 2, "UB", "call", 20, 20, 2),
        _act("D02", 0, "UC", "fold", 0, 2, 3),
        _act("D02", 1, "UA", "bet", 20, 0, 2),
        _act("D02", 2, "UB", "call", 20, 20, 2),
        # DEV benign: UC/UD contest each other.
        _act("D03", 0, "UA", "fold", 0, 2, 3),
        _act("D03", 1, "UC", "bet", 2, 0, 2),
        _act("D03", 2, "UD", "raise", 4, 2, 2),
        _act("D03", 3, "UC", "call", 4, 4, 2),
        _act("D04", 0, "UB", "fold", 0, 2, 3),
        _act("D04", 1, "UD", "bet", 2, 0, 2),
        _act("D04", 2, "UC", "raise", 4, 2, 2),
        _act("D04", 3, "UD", "call", 4, 4, 2),
        # EVAL coordinated.
        _act("H02", 0, "UC", "fold", 0, 2, 3),
        _act("H02", 1, "UA", "bet", 20, 0, 2),
        _act("H02", 2, "UB", "call", 20, 20, 2),
        _act("H03", 0, "UC", "fold", 0, 2, 3),
        _act("H03", 1, "UA", "bet", 20, 0, 2),
        _act("H03", 2, "UB", "call", 20, 20, 2),
        _act("H04", 0, "UC", "fold", 0, 2, 3),
        _act("H04", 1, "UA", "bet", 18, 0, 2),
        _act("H04", 2, "UB", "call", 18, 18, 2),
        # EVAL benign.
        _act("H05", 0, "UA", "fold", 0, 2, 3),
        _act("H05", 1, "UC", "bet", 2, 0, 2),
        _act("H05", 2, "UD", "raise", 4, 2, 2),
        _act("H05", 3, "UC", "call", 4, 4, 2),
        _act("H06", 0, "UB", "fold", 0, 2, 3),
        _act("H06", 1, "UD", "bet", 2, 0, 2),
        _act("H06", 2, "UC", "raise", 4, 2, 2),
        _act("H06", 3, "UD", "call", 4, 4, 2),
        # DEV-only action for PAE.
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


def _config_for(root: Path) -> PipelineConfig:
    cfg = PipelineConfig(input_dir=root, output_dir=root)
    cfg.row_group_size = 4  # force multi-batch streaming of actions
    return cfg


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    _write_fixture(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# (a) end-to-end + schema-valid over exactly the eval pair set
# --------------------------------------------------------------------------- #
def test_phase2_writes_schema_valid_submission(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_phase2_pipeline_fast(cfg)

    assert out == cfg.submission_path
    assert out.exists()

    frame = pd.read_csv(out)
    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()

    validate_submission(frame, sample)  # writer's own contract validator
    assert set(frame["pair_id"]) == {"PAB", "PCD", "PAE"}
    assert len(frame) == 3


# --------------------------------------------------------------------------- #
# (b) every risk_score in [0, 1]
# --------------------------------------------------------------------------- #
def test_phase2_risk_scores_in_unit_interval(dataset: Path) -> None:
    cfg = _config_for(dataset)
    out = run_phase2_pipeline_fast(cfg)
    frame = pd.read_csv(out)
    risks = pd.to_numeric(frame["risk_score"], errors="raise")
    assert (risks >= 0.0).all()
    assert (risks <= 1.0).all()


# --------------------------------------------------------------------------- #
# (c) determinism: two runs are byte-identical
# --------------------------------------------------------------------------- #
def test_phase2_is_deterministic_byte_identical(dataset: Path, tmp_path_factory) -> None:
    cfg1 = _config_for(dataset)
    bytes1 = run_phase2_pipeline_fast(cfg1).read_bytes()

    second_root = tmp_path_factory.mktemp("phase2_second_run")
    _write_fixture(second_root)
    cfg2 = _config_for(second_root)
    bytes2 = run_phase2_pipeline_fast(cfg2).read_bytes()

    assert bytes1 == bytes2


# --------------------------------------------------------------------------- #
# (d)+(e) evidence identical to Phase-1; PU risk actually used (or fallback exercised)
# --------------------------------------------------------------------------- #
def test_phase2_reuses_phase1_evidence_and_uses_pu_risk(
    dataset: Path, tmp_path_factory
) -> None:
    # Phase-1 fast submission (classical risk + Phase-1 behavior + Phase-1 evidence).
    p1_root = tmp_path_factory.mktemp("phase1_ref")
    _write_fixture(p1_root)
    cfg_p1 = _config_for(p1_root)
    p1 = pd.read_csv(run_baseline_pipeline_fast(cfg_p1)).set_index("pair_id")

    # Phase-2 submission on the SAME fixture.
    cfg_p2 = _config_for(dataset)
    p2 = pd.read_csv(run_phase2_pipeline_fast(cfg_p2)).set_index("pair_id")

    assert set(p1.index) == set(p2.index)

    evidence_cols = [f"evidence_hand_{i}" for i in range(1, 6)]
    # (d) Evidence is IDENTICAL between Phase-1 and Phase-2 for every pair.
    for pair_id in p1.index:
        for col in evidence_cols:
            assert str(p2.loc[pair_id, col]) == str(
                p1.loc[pair_id, col]
            ), f"evidence changed for {pair_id}/{col}"

    # (e) The PU risk must be genuinely used: at least one pair's Phase-2 risk differs from the
    # Phase-1 classical risk. (On this fixture both models train on the confirmed positive/negative
    # so the PU path is active; if training ever degenerated, this documents that as a failure to
    # exercise the intended path.)
    diffs = [
        abs(float(p2.loc[pid, "risk_score"]) - float(p1.loc[pid, "risk_score"]))
        for pid in p1.index
    ]
    assert max(diffs) > 1e-9, (
        "Phase-2 risk is identical to Phase-1 classical for every pair -> the PU model was not "
        "used (fallback path); the fixture should let both models train."
    )


# --------------------------------------------------------------------------- #
# Unit test for the development-hand precompute
# --------------------------------------------------------------------------- #
def test_build_dev_hands_by_player_matches_shared_hands(dataset: Path) -> None:
    cfg = _config_for(dataset)
    loader = DataLoader(config=cfg)
    seats = loader.load_seats()
    hands = loader.load_hands()

    by_player = build_dev_hands_by_player(seats, hands)

    # UA & UB coordinated DEV hands: D01, D02 (H07 is UA+UE only).
    assert by_player.get("UA", frozenset()) & by_player.get("UB", frozenset()) == {"D01", "D02"}
    # Cross-check against the authoritative slow resolver for the confirmed pairs.
    for player_1, player_2 in (("UA", "UB"), ("UC", "UD")):
        expected = set(loader.shared_hands((player_1, player_2)).development)
        got = by_player.get(player_1, frozenset()) & by_player.get(player_2, frozenset())
        assert set(got) == expected, f"mismatch for pair ({player_1}, {player_2})"
