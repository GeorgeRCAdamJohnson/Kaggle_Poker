"""Gate-machinery test for the Phase-2 gate (task 19, Req 12.2, 12.4).

Covers the Phase-2 gate WIRING (:func:`poker_collusion.validation.phase2_gate.run_phase2_gate`) on
a TINY, hermetic synthetic dataset — NOT the real ``data/poker`` files (the 18M-row
``actions.parquet`` is never touched). Its job is to prove the gate machinery is sound and
deterministic, independent of the slow real run:

* the gate returns Phase-1 and Phase-2 pooled component dicts (all four keys, every value in
  ``[0, 1]``);
* it returns a boolean ``beats_floor`` verdict plus a ``"BEATS-FLOOR"`` / ``"DOES-NOT-BEAT"``
  string, and the paired-difference CIs for every component;
* the whole measurement is DETERMINISTIC — two runs on identical inputs/config return identical
  numbers (classical baseline is deterministic; PU/behavior estimators are seeded; the CV split and
  bootstrap are seeded).

Fixture: FOUR pools T1..T4, each with a coordinated (confirmed_target) pair that dumps chips in
development hands and a benign (confirmed_non_target) pair, mirroring the Phase-1 floor test so the
group-by-pool splitter can form multiple folds and the PU/behavior heads have train positives.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.validation.bootstrap import COMPONENT_KEYS
from poker_collusion.validation.phase2_gate import run_phase2_gate


# --------------------------------------------------------------------------- #
# Tiny synthetic multi-pool fixture (mirrors test_phase1_baseline_cv).
# --------------------------------------------------------------------------- #
def _pool_players(idx: int) -> dict:
    base = f"U{idx}"
    return {k: f"{base}{k}" for k in "ABCDEF"}


def _seat(hand_id, player_id, seat_no, net_chips, folded=False):
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


def _build_fixture(n_pools: int = 4) -> dict:
    hands, seats, actions = [], [], []
    dev_labels, dev_evidence, eval_pairs = [], [], []

    action_no_counter = 0
    # Alternate the disclosed family across pools so the behavior head sees >=2 families.
    families = ["directed_transfer", "soft_play", "coordinated_isolation"]
    for i in range(1, n_pools + 1):
        table = f"T{i}"
        p = _pool_players(i)
        for j, (hid, kind) in enumerate(
            [(f"H{i}01", "dump"), (f"H{i}02", "dump"), (f"H{i}03", "benign"), (f"H{i}04", "benign")]
        ):
            hands.append(
                {
                    "hand_id": hid,
                    "table_id": table,
                    "phase": "development",
                    "big_blind": 2,
                    "small_blind": 1,
                    "started_at": j + 1,
                    "button_seat": 0,
                }
            )
            if kind == "dump":
                seats += [
                    _seat(hid, p["A"], 0, -20),
                    _seat(hid, p["B"], 1, 20),
                    _seat(hid, p["E"], 2, 0, folded=True),
                ]
                actions += [
                    _act(hid, action_no_counter + 0, p["E"], "fold", 0, 2, 3),
                    _act(hid, action_no_counter + 1, p["A"], "bet", 20, 0, 2),
                    _act(hid, action_no_counter + 2, p["B"], "call", 20, 20, 2),
                ]
            else:
                seats += [
                    _seat(hid, p["C"], 0, 2),
                    _seat(hid, p["D"], 1, -2),
                    _seat(hid, p["F"], 2, 0, folded=True),
                ]
                actions += [
                    _act(hid, action_no_counter + 0, p["F"], "fold", 0, 2, 3),
                    _act(hid, action_no_counter + 1, p["C"], "bet", 2, 0, 2),
                    _act(hid, action_no_counter + 2, p["D"], "raise", 4, 2, 2),
                    _act(hid, action_no_counter + 3, p["C"], "call", 4, 4, 2),
                ]
            action_no_counter += 4

        pos_id, neg_id = f"P{i}POS", f"P{i}NEG"
        fam = families[(i - 1) % len(families)]
        dev_labels += [
            {
                "pair_id": pos_id,
                "player_1": p["A"],
                "player_2": p["B"],
                "label": 1,
                "label_status": "confirmed_target",
                "behavior_family": fam,
            },
            {
                "pair_id": neg_id,
                "player_1": p["C"],
                "player_2": p["D"],
                "label": 0,
                "label_status": "confirmed_non_target",
                "behavior_family": "none",
            },
        ]
        dev_evidence += [
            {"pair_id": pos_id, "evidence_rank": 1, "hand_id": f"H{i}01", "behavior_family": fam},
            {"pair_id": pos_id, "evidence_rank": 2, "hand_id": f"H{i}02", "behavior_family": fam},
        ]
        eval_pairs.append(
            {"pair_id": f"P{i}EVAL", "player_1": p["E"], "player_2": p["F"], "shared_hands": 0}
        )

    players = pd.DataFrame({"player_id": sorted({s["player_id"] for s in seats})})
    players["account_age_days"] = 10
    players["experience_hands_bucket"] = "a"
    players["preferred_stake"] = "s"
    players["region_bucket"] = "r"
    players["client_family"] = "cf"

    return {
        "hands": pd.DataFrame(hands),
        "seats": pd.DataFrame(seats),
        "actions": pd.DataFrame(actions),
        "players": players,
        "development_labels": pd.DataFrame(dev_labels),
        "development_evidence": pd.DataFrame(dev_evidence),
        "evaluation_pairs": pd.DataFrame(eval_pairs),
    }


def _write_fixture(root: Path, data: dict) -> None:
    data["hands"].to_parquet(root / "hands.parquet")
    data["seats"].to_parquet(root / "seats.parquet")
    data["actions"].to_parquet(root / "actions.parquet")
    data["players"].to_parquet(root / "players.parquet")
    data["development_labels"].to_csv(root / "development_labels.csv", index=False)
    data["development_evidence"].to_csv(root / "development_evidence.csv", index=False)
    data["evaluation_pairs"].to_csv(root / "evaluation_pairs.csv", index=False)
    sample = data["evaluation_pairs"][["pair_id"]].copy()
    sample["risk_score"] = 0.0
    sample["predicted_behavior"] = "none"
    for k in range(1, 6):
        sample[f"evidence_hand_{k}"] = "NO_EVIDENCE"
    sample.to_csv(root / "sample_submission.csv", index=False)


def _config_for(root: Path) -> PipelineConfig:
    cfg = PipelineConfig(input_dir=root, output_dir=root)
    cfg.row_group_size = 4  # force multi-batch streaming of the tiny actions parquet
    return cfg


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    _write_fixture(tmp_path, _build_fixture(n_pools=4))
    return tmp_path


def _run(root: Path):
    cfg = _config_for(root)
    return run_phase2_gate(config=cfg, n_folds=3, n_boot=64)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_gate_returns_component_dicts_in_unit_interval(synthetic_root: Path) -> None:
    """Phase-1 and Phase-2 pooled component dicts exist, all four keys, every value in [0, 1]."""
    result = _run(synthetic_root)

    assert result.n_pairs > 0
    assert result.n_folds >= 2

    for scores in (result.phase1, result.phase2):
        for k in COMPONENT_KEYS:
            assert k in scores
            v = scores[k]
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_gate_returns_boolean_verdict_and_cis(synthetic_root: Path) -> None:
    """The gate returns a boolean verdict, a matching string, and paired-diff CIs per component."""
    result = _run(synthetic_root)

    assert isinstance(result.beats_floor, bool)
    assert result.verdict in ("BEATS-FLOOR", "DOES-NOT-BEAT")
    assert result.beats_floor == (result.verdict == "BEATS-FLOOR")

    for k in COMPONENT_KEYS:
        assert k in result.cis
        ci = result.cis[k]
        assert ci.low <= ci.high  # a well-formed CI (differences may be negative)

    # Verdict logic is internally consistent.
    assert result.beats_floor == (result.combined_improves and not result.regressed_components)


def test_gate_is_deterministic(synthetic_root: Path) -> None:
    """Two runs on identical inputs/config return identical numbers (determinism)."""
    first = _run(synthetic_root)
    second = _run(synthetic_root)

    assert first.phase1 == second.phase1
    assert first.phase2 == second.phase2
    assert first.deltas == second.deltas
    assert first.verdict == second.verdict
    assert first.beats_floor == second.beats_floor
    for k in COMPONENT_KEYS:
        assert first.cis[k].as_dict() == second.cis[k].as_dict()
    assert first.deterministic is True
