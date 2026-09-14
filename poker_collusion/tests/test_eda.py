"""Fixture tests for the EDA & label audit + threshold artifact (task 8.1, Req 2).

These tests build tiny synthetic competition tables (hex string IDs matching the real
schema) and drive :mod:`poker_collusion.features.eda` end-to-end through a temp-config
:class:`DataLoader`, plus unit-test each report function on constructed frames.

They verify, on hermetic data:
  * PU counts (trusted-positive / confirmed-negative / unknown incl. eval pairs) — Req 2.2
  * disclosed-family distribution among positives, incl. no-other_coordination — Req 2.3
  * evidence-validity reporting (both-players-seated; behavior-action clause documented
    as un-computable) — Req 2.4
  * evaluation-pair leakage checks EXC-1 / EXC-2 on clean and violating data — Req 2.5
  * shared-hand-count distribution stats — Req 2.1
  * the artifact round-trips (write then read) carrying the threshold_version — Req 2.6

They are fast and never read the real ``actions.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.features import eda
from poker_collusion.features.eda import (
    EDA_ARTIFACT_SCHEMA,
    EdaArtifact,
    audit_evaluation_leakage,
    audit_evidence_validity,
    compute_family_distribution,
    compute_pu_counts,
    compute_shared_hand_count_stats,
    load_eda_artifact,
    run_eda,
    write_eda_artifact,
)
from poker_collusion.io import DataLoader


# --------------------------------------------------------------------------- #
# Synthetic real-schema fixture written to a temp dir
# --------------------------------------------------------------------------- #
def _write_fixture(root: Path) -> None:
    """Write a small schema-complete dataset (hex string IDs) under ``root``.

    Layout:
      * One pool T1 with hands H1 (dev), H2 (eval), H3 (dev), H4 (eval).
      * Players UA, UB, UC, UD.
      * UA+UB co-seated in all four hands (2 dev + 2 eval shared).
      * UC+UD co-seated in H1 (dev) and H2 (eval) (1 dev + 1 eval shared).
    Labels: PPOS=(UA,UB) confirmed_target soft_play; PNEG=(UC,UD) confirmed_non_target.
    Evidence: PPOS evidence on H2 (valid: both seated + eval) and H4 (valid).
    Eval pairs: PE1=(UA,UC) clean, PE2=(UB,UD) clean — no labelled pair id, and neither
    reuses a labelled *positive* player? NB UA/UB ARE positive players, so we use
    non-positive players for the clean eval set below.
    """
    hands = pd.DataFrame(
        {
            "hand_id": ["H1", "H2", "H3", "H4"],
            "table_id": ["T1", "T1", "T1", "T1"],
            "phase": ["development", "evaluation", "development", "evaluation"],
            "big_blind": [2, 2, 2, 2],
            "small_blind": [1, 1, 1, 1],
            "started_at": [1, 2, 3, 4],
            "button_seat": [0, 1, 2, 0],
        }
    )
    hands.to_parquet(root / "hands.parquet")

    players = pd.DataFrame(
        {
            "player_id": ["UA", "UB", "UC", "UD", "UE", "UF"],
            "account_age_days": [10, 20, 30, 40, 50, 60],
            "experience_hands_bucket": ["a"] * 6,
            "preferred_stake": ["s"] * 6,
            "region_bucket": ["r"] * 6,
            "client_family": ["cf"] * 6,
        }
    )
    players.to_parquet(root / "players.parquet")

    # Seats: UA+UB in H1..H4; UC+UD in H1,H2; UE in H2 (filler), UF in H4 (filler).
    seat_rows = []
    for h in ["H1", "H2", "H3", "H4"]:
        seat_rows.append({"hand_id": h, "player_id": "UA"})
        seat_rows.append({"hand_id": h, "player_id": "UB"})
    for h in ["H1", "H2"]:
        seat_rows.append({"hand_id": h, "player_id": "UC"})
        seat_rows.append({"hand_id": h, "player_id": "UD"})
    seat_rows.append({"hand_id": "H2", "player_id": "UE"})
    seat_rows.append({"hand_id": "H4", "player_id": "UF"})
    seats = pd.DataFrame(seat_rows)
    seats["seat_no"] = range(len(seats))
    seats["starting_stack"] = 100
    seats["total_contribution"] = 1
    seats["net_chips"] = 0
    seats["folded"] = False
    seats["went_to_showdown"] = True
    seats["won_share"] = 0.0
    seats.to_parquet(root / "seats.parquet")

    # actions.parquet must exist (schema-complete) but is never read by EDA.
    pd.DataFrame(
        {
            "hand_id": ["H1"],
            "action_no": [0],
            "street": ["pre"],
            "player_id": ["UA"],
            "action": ["bet"],
            "amount": [2],
            "amount_to": [2],
            "to_call": [0],
            "pot_before": [0],
            "stack_before": [100],
            "players_active": [2],
        }
    ).to_parquet(root / "actions.parquet")

    pd.DataFrame(
        {
            "pair_id": ["PPOS", "PNEG"],
            "player_1": ["UA", "UC"],
            "player_2": ["UB", "UD"],
            "label": [1, 0],
            "label_status": ["confirmed_target", "confirmed_non_target"],
            "behavior_family": ["soft_play", "none"],
        }
    ).to_csv(root / "development_labels.csv", index=False)

    # Long-form evidence: PPOS evidence hands H2 (eval) and H4 (eval) — both shared.
    pd.DataFrame(
        {
            "pair_id": ["PPOS", "PPOS"],
            "evidence_rank": [1, 2],
            "hand_id": ["H2", "H4"],
            "behavior_family": ["soft_play", "soft_play"],
        }
    ).to_csv(root / "development_evidence.csv", index=False)

    # Clean eval pairs: use non-positive players UE/UF (and UC/UD are negatives, allowed).
    pd.DataFrame(
        {
            "pair_id": ["PE1", "PE2"],
            "player_1": ["UE", "UC"],
            "player_2": ["UF", "UD"],
            "shared_hands": [0, 2],
        }
    ).to_csv(root / "evaluation_pairs.csv", index=False)

    pd.DataFrame(
        {
            "pair_id": ["PE1", "PE2"],
            "risk_score": [0.0, 0.0],
            "predicted_behavior": ["none", "none"],
            "evidence_hand_1": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_2": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_3": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_4": ["NO_EVIDENCE", "NO_EVIDENCE"],
            "evidence_hand_5": ["NO_EVIDENCE", "NO_EVIDENCE"],
        }
    ).to_csv(root / "sample_submission.csv", index=False)


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    _write_fixture(tmp_path)
    return PipelineConfig(input_dir=tmp_path, output_dir=tmp_path)


@pytest.fixture()
def loader(cfg: PipelineConfig) -> DataLoader:
    return DataLoader(config=cfg)


# --------------------------------------------------------------------------- #
# 2.1 shared-hand-count distribution
# --------------------------------------------------------------------------- #
def test_shared_hand_count_stats_basic() -> None:
    stats = compute_shared_hand_count_stats([1, 2, 3, 4, 100])
    assert stats.n_pairs == 5
    assert stats.min == 1.0
    assert stats.max == 100.0
    assert stats.median == 3.0
    assert 50 in stats.percentiles and stats.percentiles[50] == 3.0


def test_shared_hand_count_stats_empty() -> None:
    stats = compute_shared_hand_count_stats([])
    assert stats.n_pairs == 0
    assert stats.min == 0.0 and stats.max == 0.0


# --------------------------------------------------------------------------- #
# 2.2 PU counts
# --------------------------------------------------------------------------- #
def test_pu_counts_partition() -> None:
    labels = pd.DataFrame(
        {
            "pair_id": ["P1", "P2", "P3"],
            "label_status": ["confirmed_target", "confirmed_non_target", "confirmed_target"],
            "behavior_family": ["soft_play", "none", "directed_transfer"],
        }
    )
    eval_pairs = pd.DataFrame({"pair_id": ["E1", "E2", "E3", "E4"]})
    pu = compute_pu_counts(labels, eval_pairs)
    assert pu["trusted_positive"] == 2
    assert pu["confirmed_negative"] == 1
    # Unknown = 0 unlabelled-status rows + 4 eval pairs.
    assert pu["unknown"] == 4
    assert pu["unknown_from_evaluation_pairs"] == 4
    assert pu["labelled_positive_prevalence"] == pytest.approx(2 / 3)


def test_pu_counts_unlabelled_status_counts_as_unknown() -> None:
    labels = pd.DataFrame(
        {
            "pair_id": ["P1", "P2"],
            "label_status": ["confirmed_target", "some_other_status"],
            "behavior_family": ["soft_play", "x"],
        }
    )
    pu = compute_pu_counts(labels, None)
    assert pu["trusted_positive"] == 1
    assert pu["confirmed_negative"] == 0
    assert pu["unknown"] == 1
    assert pu["unknown_from_unlabelled_status"] == 1


# --------------------------------------------------------------------------- #
# 2.3 family distribution
# --------------------------------------------------------------------------- #
def test_family_distribution_disclosed_only() -> None:
    labels = pd.DataFrame(
        {
            "pair_id": ["P1", "P2", "P3", "P4"],
            "label_status": ["confirmed_target"] * 3 + ["confirmed_non_target"],
            "behavior_family": ["directed_transfer", "soft_play", "coordinated_isolation", "none"],
        }
    )
    fam = compute_family_distribution(labels)
    assert fam["n_positive"] == 3
    assert fam["by_family"] == {
        "directed_transfer": 1,
        "soft_play": 1,
        "coordinated_isolation": 1,
    }
    assert fam["other_or_undisclosed"] == 0
    assert fam["other_coordination_absent_in_public"] is True


def test_family_distribution_flags_undisclosed_positive() -> None:
    labels = pd.DataFrame(
        {
            "pair_id": ["P1", "P2"],
            "label_status": ["confirmed_target", "confirmed_target"],
            "behavior_family": ["soft_play", "other_coordination"],
        }
    )
    fam = compute_family_distribution(labels)
    assert fam["other_or_undisclosed"] == 1
    assert fam["other_coordination_absent_in_public"] is False


# --------------------------------------------------------------------------- #
# 2.4 evidence validity
# --------------------------------------------------------------------------- #
def test_evidence_validity_confirms_shared_hands() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["PPOS"], "player_1": ["UA"], "player_2": ["UB"],
         "label_status": ["confirmed_target"], "behavior_family": ["soft_play"]}
    )
    seats = pd.DataFrame(
        {"hand_id": ["H2", "H2", "H4", "H4"], "player_id": ["UA", "UB", "UA", "UB"]}
    )
    evidence = pd.DataFrame(
        {"pair_id": ["PPOS", "PPOS"], "evidence_rank": [1, 2], "hand_id": ["H2", "H4"],
         "behavior_family": ["soft_play", "soft_play"]}
    )
    res = audit_evidence_validity(evidence, labels, seats)
    assert res["data_present"] is True
    assert res["checkable"] is True
    assert res["valid_shared_hand"] == 2
    assert res["invalid"] == 0
    assert res["status"] == "CONFIRMED"
    # The action-log clause is documented as un-computable from public data.
    assert "un-computable" in res["behavior_action_clause"]


def test_evidence_validity_flags_non_shared_hand() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["PPOS"], "player_1": ["UA"], "player_2": ["UB"],
         "label_status": ["confirmed_target"], "behavior_family": ["soft_play"]}
    )
    # H9 seats only UA => not a shared hand => invalid evidence row.
    seats = pd.DataFrame(
        {"hand_id": ["H2", "H2", "H9"], "player_id": ["UA", "UB", "UA"]}
    )
    evidence = pd.DataFrame(
        {"pair_id": ["PPOS", "PPOS"], "evidence_rank": [1, 2], "hand_id": ["H2", "H9"],
         "behavior_family": ["soft_play", "soft_play"]}
    )
    res = audit_evidence_validity(evidence, labels, seats)
    assert res["valid_shared_hand"] == 1
    assert res["invalid"] == 1
    assert res["status"] == "REFUTED"
    assert res["violations"][0]["hand_id"] == "H9"
    assert res["violations"][0]["both_players_seated"] is False


# --------------------------------------------------------------------------- #
# 2.5 leakage checks
# --------------------------------------------------------------------------- #
def test_leakage_clean_eval_set() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["PPOS"], "player_1": ["UA"], "player_2": ["UB"],
         "label_status": ["confirmed_target"], "behavior_family": ["soft_play"]}
    )
    eval_pairs = pd.DataFrame(
        {"pair_id": ["PE1"], "player_1": ["UE"], "player_2": ["UF"], "shared_hands": [0]}
    )
    res = audit_evaluation_leakage(eval_pairs, labels)
    assert res["EXC-1"]["status"] == "CONFIRMED"
    assert res["EXC-2"]["status"] == "CONFIRMED"
    assert res["EXC-2"]["n_violations"] == 0


def test_leakage_flags_labelled_pair_id() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["PPOS"], "player_1": ["UA"], "player_2": ["UB"],
         "label_status": ["confirmed_target"], "behavior_family": ["soft_play"]}
    )
    eval_pairs = pd.DataFrame(
        {"pair_id": ["PPOS"], "player_1": ["UX"], "player_2": ["UY"], "shared_hands": [0]}
    )
    res = audit_evaluation_leakage(eval_pairs, labels)
    assert res["EXC-1"]["status"] == "REFUTED"
    assert "PPOS" in res["EXC-1"]["violations"]


def test_leakage_flags_positive_player_reuse() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["PPOS"], "player_1": ["UA"], "player_2": ["UB"],
         "label_status": ["confirmed_target"], "behavior_family": ["directed_transfer"]}
    )
    # UA is a labelled-positive player; reusing it in an eval pair is an EXC-2 violation.
    eval_pairs = pd.DataFrame(
        {"pair_id": ["PE9"], "player_1": ["UA"], "player_2": ["UZ"], "shared_hands": [0]}
    )
    res = audit_evaluation_leakage(eval_pairs, labels)
    assert res["EXC-2"]["status"] == "REFUTED"
    assert res["EXC-2"]["violations"][0]["pair_id"] == "PE9"
    assert "UA" in res["EXC-2"]["violations"][0]["offending_players"]


def test_leakage_negative_player_reuse_is_allowed() -> None:
    # A negative labelled player reused in eval is NOT an EXC-2 violation (positives only).
    labels = pd.DataFrame(
        {"pair_id": ["PPOS", "PNEG"], "player_1": ["UA", "UC"], "player_2": ["UB", "UD"],
         "label_status": ["confirmed_target", "confirmed_non_target"],
         "behavior_family": ["soft_play", "none"]}
    )
    eval_pairs = pd.DataFrame(
        {"pair_id": ["PE1"], "player_1": ["UC"], "player_2": ["UZ"], "shared_hands": [0]}
    )
    res = audit_evaluation_leakage(eval_pairs, labels)
    assert res["EXC-2"]["status"] == "CONFIRMED"


# --------------------------------------------------------------------------- #
# End-to-end run + artifact round-trip (2.1-2.6)
# --------------------------------------------------------------------------- #
def test_run_eda_end_to_end(loader: DataLoader, cfg: PipelineConfig) -> None:
    artifact = run_eda(loader, cfg)
    assert isinstance(artifact, EdaArtifact)
    assert artifact.schema == EDA_ARTIFACT_SCHEMA
    assert artifact.threshold_version == cfg.threshold_version

    report = artifact.report
    # PU: 1 positive (PPOS), 1 negative (PNEG), unknown = 2 eval pairs.
    assert report["pu_counts"]["trusted_positive"] == 1
    assert report["pu_counts"]["confirmed_negative"] == 1
    assert report["pu_counts"]["unknown"] == 2

    # Family: 1 soft_play positive, no other_coordination.
    assert report["family_distribution"]["by_family"]["soft_play"] == 1
    assert report["family_distribution"]["other_coordination_absent_in_public"] is True

    # Evidence validity: both PPOS evidence hands (H2, H4) are shared eval hands.
    assert report["evidence_validity"]["valid_shared_hand"] == 2
    assert report["evidence_validity"]["status"] == "CONFIRMED"

    # Leakage: clean eval set (PE1=UE/UF, PE2=UC/UD negatives) => both confirmed.
    assert report["evaluation_leakage"]["EXC-1"]["status"] == "CONFIRMED"
    assert report["evaluation_leakage"]["EXC-2"]["status"] == "CONFIRMED"

    # Shared-hand-count stats: UA+UB share 4 hands; the labelled set includes PPOS(4),PNEG(2).
    labelled = report["shared_hand_counts"]["labelled_pairs"]
    assert labelled["max"] == 4.0

    # Thresholds present and versioned.
    assert artifact.thresholds["min_shared_hands"] >= 1
    assert "shared_hand_count_anchors" in artifact.thresholds
    assert artifact.thresholds["family_priors"]["other_coordination"] == 0.0


def test_artifact_round_trip(loader: DataLoader, cfg: PipelineConfig, tmp_path: Path) -> None:
    artifact = run_eda(loader, cfg)
    out = write_eda_artifact(artifact, config=cfg)
    assert out == Path(cfg.eda_artifact_path)
    assert out.exists()

    loaded = load_eda_artifact(config=cfg)
    assert loaded.threshold_version == artifact.threshold_version
    assert loaded.schema == artifact.schema
    assert loaded.master_seed == artifact.master_seed
    assert loaded.thresholds == artifact.thresholds
    # report survives the round-trip too.
    assert loaded.report["pu_counts"]["trusted_positive"] == 1


def test_run_eda_is_deterministic(loader: DataLoader, cfg: PipelineConfig) -> None:
    a1 = run_eda(loader, cfg)
    a2 = run_eda(DataLoader(config=cfg), cfg)
    assert a1.thresholds == a2.thresholds
    assert a1.report == a2.report
