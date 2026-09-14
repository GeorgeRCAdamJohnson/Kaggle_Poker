"""Unit tests for poker_collusion.discovery.label_evidence_audit (task 2.4, Workstream C).

These tests build *tiny synthetic fixtures that embed the label/evidence
structure the dossier section 3 predicts* (evidence hands with both players
seated and a public behavior-specific action; a latent-only hand that must be
rejected; an evaluation-pairs set that correctly excludes labelled pairs and
positive players, plus a deliberately violating set), then assert each audit
recovers the planted structure. They also confirm every audit degrades
gracefully to ``data_present=False`` / ``status=PENDING`` on empty input, so the
module is safe to import and exercise with no competition data present.

Requirements exercised: 2.4, 2.5, 8.4.
"""

from __future__ import annotations

import pandas as pd

from poker_collusion.discovery import label_evidence_audit as lea
from poker_collusion.discovery.label_evidence_audit import (
    AuditResult,
    DISCLOSED_FAMILIES,
    FAMILY_SIGNATURES,
    PENDING,
    evaluation_pairs_exclusions,
    family_action_signatures,
    run_all_audits,
    structural_invariants,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _seats() -> pd.DataFrame:
    """Seatings: pair (100,101) share hands 1&2; hand 3 seats only 100 (+201)."""
    rows = [
        {"hand_id": 1, "player_id": 100},
        {"hand_id": 1, "player_id": 101},
        {"hand_id": 1, "player_id": 102},
        {"hand_id": 2, "player_id": 100},
        {"hand_id": 2, "player_id": 101},
        {"hand_id": 3, "player_id": 100},
        {"hand_id": 3, "player_id": 201},
    ]
    return pd.DataFrame(rows)


def _valid_evidence() -> pd.DataFrame:
    """Two well-formed planted evidence hands: both players seated + public action."""
    return pd.DataFrame(
        {
            "pair_id": ["100_101", "100_101"],
            "hand_id": [1, 2],
            "target_behavior": ["directed_transfer", "soft_play"],
            "behavior_action": [True, True],
        }
    )


# --------------------------------------------------------------------------- #
# 3.1 : per-family signatures
# --------------------------------------------------------------------------- #
def test_family_signatures_returns_spec_for_each_family() -> None:
    res = family_action_signatures()
    assert isinstance(res, AuditResult)
    assert res.data_present is False
    assert set(res.checks) == set(DISCLOSED_FAMILIES)
    for fam in DISCLOSED_FAMILIES:
        assert res.checks[fam]["status"] == PENDING
        # The gate-target predicate is present for every family.
        assert "predicate" in res.checks[fam]
        assert "gate_requirement" in res.checks[fam]
    # FAMILY_SIGNATURES covers exactly the three disclosed families.
    assert set(FAMILY_SIGNATURES) == set(DISCLOSED_FAMILIES)


def test_family_signatures_counts_when_evidence_present() -> None:
    res = family_action_signatures(_valid_evidence())
    assert res.data_present is True
    assert res.checks["directed_transfer"]["observed_evidence_hand_count"] == 1
    assert res.checks["soft_play"]["observed_evidence_hand_count"] == 1
    assert res.checks["coordinated_isolation"]["observed_evidence_hand_count"] == 0


# --------------------------------------------------------------------------- #
# 3.2 : structural invariants
# --------------------------------------------------------------------------- #
def test_structural_invariants_confirmed_on_valid_evidence() -> None:
    res = structural_invariants(_valid_evidence(), _seats())
    assert res.data_present is True
    assert res.checks["INV-1"]["status"] == "CONFIRMED"
    assert res.checks["INV-1"]["n_violations"] == 0
    assert res.checks["INV-2"]["status"] == "CONFIRMED"
    assert res.checks["INV-2"]["n_violations"] == 0
    assert res.status == "CONFIRMED"


def test_structural_invariants_flags_latent_only_hand() -> None:
    # Hand 2 has no public behavior-specific action => INV-1 violation (latent only).
    evidence = pd.DataFrame(
        {
            "pair_id": ["100_101", "100_101"],
            "hand_id": [1, 2],
            "target_behavior": ["directed_transfer", "soft_play"],
            "behavior_action": [True, False],
        }
    )
    res = structural_invariants(evidence, _seats())
    assert res.checks["INV-1"]["status"] == "REFUTED"
    assert res.checks["INV-1"]["n_violations"] == 1
    assert res.checks["INV-1"]["violations"][0]["hand_id"] == 2


def test_structural_invariants_flags_missing_player() -> None:
    # Hand 3 does not seat player 101 => INV-2 violation (not both players present).
    evidence = pd.DataFrame(
        {
            "pair_id": ["100_101"],
            "hand_id": [3],
            "target_behavior": ["coordinated_isolation"],
            "behavior_action": [True],
        }
    )
    res = structural_invariants(evidence, _seats())
    assert res.checks["INV-2"]["status"] == "REFUTED"
    assert res.checks["INV-2"]["violations"][0]["both_players_seated"] is False


def test_structural_invariants_empty_is_pending() -> None:
    res = structural_invariants(pd.DataFrame(), None)
    assert res.data_present is False
    assert res.checks["INV-1"]["status"] == PENDING
    assert res.checks["INV-2"]["status"] == PENDING


# --------------------------------------------------------------------------- #
# 3.3 : evaluation_pairs exclusions
# --------------------------------------------------------------------------- #
def test_exclusions_confirmed_on_clean_eval_set() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["100_101"], "target_behavior": ["soft_play"]}
    )
    # Clean eval set: no labelled pair id, no player from the positive pair (100/101).
    eval_pairs = pd.DataFrame({"pair_id": ["200_201", "300_301"]})
    res = evaluation_pairs_exclusions(eval_pairs, labels)
    assert res.data_present is True
    assert res.checks["EXC-1"]["status"] == "CONFIRMED"
    assert res.checks["EXC-2"]["status"] == "CONFIRMED"
    assert res.status == "CONFIRMED"


def test_exclusions_flag_labelled_pair_id() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["100_101"], "target_behavior": ["soft_play"]}
    )
    # Eval set re-lists the labelled pair id => EXC-1 violation.
    eval_pairs = pd.DataFrame({"pair_id": ["100_101", "300_301"]})
    res = evaluation_pairs_exclusions(eval_pairs, labels)
    assert res.checks["EXC-1"]["status"] == "REFUTED"
    assert "100_101" in res.checks["EXC-1"]["violations"]


def test_exclusions_flag_positive_player_reuse() -> None:
    labels = pd.DataFrame(
        {"pair_id": ["100_101"], "target_behavior": ["directed_transfer"]}
    )
    # 100 is a labelled-positive player; pairing it with a fresh id is an EXC-2 violation.
    eval_pairs = pd.DataFrame({"pair_id": ["100_999", "300_301"]})
    res = evaluation_pairs_exclusions(eval_pairs, labels)
    assert res.checks["EXC-2"]["status"] == "REFUTED"
    assert res.checks["EXC-2"]["violations"][0]["pair_id"] == "100_999"
    assert 100 in res.checks["EXC-2"]["violations"][0]["offending_players"]


def test_exclusions_empty_is_pending() -> None:
    res = evaluation_pairs_exclusions(None, None)
    assert res.data_present is False
    assert res.checks["EXC-1"]["status"] == PENDING
    assert res.checks["EXC-2"]["status"] == PENDING


# --------------------------------------------------------------------------- #
# run_all_audits convenience
# --------------------------------------------------------------------------- #
def test_run_all_audits_graceful_without_data() -> None:
    results = run_all_audits({})
    assert set(results) == {
        "family_action_signatures",
        "structural_invariants",
        "evaluation_pairs_exclusions",
    }
    for r in results.values():
        assert isinstance(r, AuditResult)
        assert r.data_present is False


def test_run_all_audits_with_fixtures() -> None:
    tables = {
        "development_evidence": _valid_evidence(),
        "seats": _seats(),
        "evaluation_pairs": pd.DataFrame({"pair_id": ["200_201"]}),
        "development_labels": pd.DataFrame(
            {"pair_id": ["100_101"], "target_behavior": ["soft_play"]}
        ),
    }
    results = run_all_audits(tables)
    assert results["structural_invariants"].data_present is True
    assert results["structural_invariants"].status == "CONFIRMED"
    assert results["evaluation_pairs_exclusions"].status == "CONFIRMED"
