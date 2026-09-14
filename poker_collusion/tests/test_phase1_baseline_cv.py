"""Gate-machinery test for the Phase-1 baseline floor measurement (task 15, Req 9.6, 11.1).

This test covers the FLOOR-measurement WIRING
(:func:`poker_collusion.validation.phase1_baseline_cv.measure_baseline_floor`) on a TINY,
hermetic synthetic dataset — NOT the real ``data/poker`` files (esp. the 18M-row
``actions.parquet``, which is never touched). Its job is to prove the gate machinery itself is
sound and deterministic, independent of the slow real run:

* the harness runs the classical ``predict_fn`` over several group-by-pool folds and returns the
  four components (``pair_ap``, ``evidence_map``, ``behavior_map``, ``combined``);
* every component (mean and per-fold) is a finite value in ``[0, 1]``;
* the combined summary equals ``0.70*pair_ap + 0.20*evidence_map + 0.10*behavior_map``;
* the whole measurement is DETERMINISTIC — two runs on identical inputs/config return identical
  numbers (the classical baseline is deterministic; the CV split and bootstrap are seeded).

Fixture shape (real competition schema, hex-ish string ids), FOUR pools T1..T4
==============================================================================
Each pool has a coordinated (confirmed_target) pair that dumps chips in DEVELOPMENT hands with a
public value commitment, plus a benign (confirmed_non_target) pair contesting each other. Four
pools give the group-by-pool splitter enough distinct groups to form multiple folds. Development
evidence points at each positive pair's development hands so Evidence MAP@5 is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.io import DataLoader
from poker_collusion.validation.phase1_baseline_cv import measure_baseline_floor


# --------------------------------------------------------------------------- #
# Tiny synthetic multi-pool fixture builders.
# --------------------------------------------------------------------------- #
def _pool_players(idx: int) -> dict:
    """Six player ids for pool ``idx``: colluders A/B, benign C/D, outsiders E/F."""
    base = f"U{idx}"
    return {k: f"{base}{k}" for k in "ABCDEF"}


def _build_fixture(n_pools: int = 4) -> dict:
    hands, seats, actions = [], [], []
    dev_labels, dev_evidence, eval_pairs = [], [], []

    action_no_counter = 0
    for i in range(1, n_pools + 1):
        table = f"T{i}"
        p = _pool_players(i)
        # Two development hands where colluders A->B dump ~10bb (bb=2 -> net 20), outsider folds.
        # Two development hands where benign C/D contest each other.
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

        # PU labels: one confirmed_target (colluders), one confirmed_non_target (benign).
        pos_id, neg_id = f"P{i}POS", f"P{i}NEG"
        dev_labels += [
            {
                "pair_id": pos_id,
                "player_1": p["A"],
                "player_2": p["B"],
                "label": 1,
                "label_status": "confirmed_target",
                "behavior_family": "directed_transfer",
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
        # Development evidence for the positive pair (its two development dump hands).
        dev_evidence += [
            {"pair_id": pos_id, "evidence_rank": 1, "hand_id": f"H{i}01", "behavior_family": "directed_transfer"},
            {"pair_id": pos_id, "evidence_rank": 2, "hand_id": f"H{i}02", "behavior_family": "directed_transfer"},
        ]
        # Evaluation pairs are unrelated ids (the floor only needs labelled pairs; keep it minimal).
        eval_pairs.append(
            {"pair_id": f"P{i}EVAL", "player_1": p["E"], "player_2": p["F"], "shared_hands": 0}
        )

    players = pd.DataFrame(
        {
            "player_id": sorted({s["player_id"] for s in seats}),
        }
    )
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


def _write_fixture(root: Path, data: dict) -> None:
    data["hands"].to_parquet(root / "hands.parquet")
    data["seats"].to_parquet(root / "seats.parquet")
    data["actions"].to_parquet(root / "actions.parquet")
    data["players"].to_parquet(root / "players.parquet")
    data["development_labels"].to_csv(root / "development_labels.csv", index=False)
    data["development_evidence"].to_csv(root / "development_evidence.csv", index=False)
    data["evaluation_pairs"].to_csv(root / "evaluation_pairs.csv", index=False)
    # A minimal sample_submission for completeness (not used by the floor measurement).
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


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _measure(root: Path):
    cfg = _config_for(root)
    # Neutral (no cached artifact) -> run_eda computes one on the fly from the tiny tables.
    return measure_baseline_floor(config=cfg, n_folds=3, n_boot=64)


def test_floor_returns_four_components_in_unit_interval(synthetic_root: Path) -> None:
    """The gate returns the four components (mean + per-fold), all finite and in [0, 1]."""
    result = _measure(synthetic_root)

    assert result.n_folds >= 2
    assert result.n_pairs_scored > 0

    keys = ("pair_ap", "evidence_map", "behavior_map", "combined")
    for k in keys:
        assert k in result.mean
        v = result.mean[k]
        assert 0.0 <= v <= 1.0, f"mean {k}={v} out of [0,1]"

    for fold_scores in result.per_fold:
        for k in keys:
            v = fold_scores[k]
            assert 0.0 <= v <= 1.0, f"per-fold {k}={v} out of [0,1]"


def test_floor_combined_matches_weighting(synthetic_root: Path) -> None:
    """Combined summary equals the 0.70/0.20/0.10 weighted sum of the three components."""
    result = _measure(synthetic_root)
    for scores in result.per_fold:
        expected = (
            0.70 * scores["pair_ap"]
            + 0.20 * scores["evidence_map"]
            + 0.10 * scores["behavior_map"]
        )
        assert scores["combined"] == pytest.approx(expected, abs=1e-9)


def test_floor_intervals_present_and_bounded(synthetic_root: Path) -> None:
    """Bootstrap CIs exist for every component with low <= mean <= high, all in [0, 1]."""
    result = _measure(synthetic_root)
    tol = 1e-9  # a near-degenerate CI (constant replicates) can put mean ~1 ULP outside [low, high]
    for k in ("pair_ap", "evidence_map", "behavior_map", "combined"):
        ci = result.intervals[k]
        assert 0.0 <= ci.low <= ci.high <= 1.0
        assert ci.low - tol <= ci.mean <= ci.high + tol


def test_floor_measurement_is_deterministic(synthetic_root: Path) -> None:
    """Two runs on identical inputs/config return identical floor numbers (determinism)."""
    first = _measure(synthetic_root)
    second = _measure(synthetic_root)

    assert first.mean == second.mean
    assert first.per_fold == second.per_fold
    for k in ("pair_ap", "evidence_map", "behavior_map", "combined"):
        assert first.intervals[k].as_dict() == second.intervals[k].as_dict()
    assert first.deterministic is True
