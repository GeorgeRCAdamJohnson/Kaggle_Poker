"""Unit tests for poker_collusion.discovery.datagen_probes (task 2.3, Workstream B).

These tests build *tiny synthetic fixtures that deliberately embed the generator
structure the dossier hypotheses (H1-H19) predict* (a chip lattice, monotone
pool-blocked hand ids, deterministic seat rotation, disjoint pools, a
pair-directed value-flow episode, an untemplated high-advantage positive), then
assert each probe recovers the planted structure. They also confirm every probe
degrades gracefully to ``data_present=False`` when handed empty input, so the
module is safe to import and exercise even though no competition data is present
in this environment.

Requirements exercised: 2.1, 2.2, 2.3.
"""

from __future__ import annotations

import pandas as pd

from poker_collusion.discovery import datagen_probes as dp
from poker_collusion.discovery.datagen_probes import (
    ProbeResult,
    confounder_tells,
    distribution_artifacts,
    id_ordering_regularities,
    other_coordination_signature,
    pool_construction,
    run_all_probes,
)


# --------------------------------------------------------------------------- #
# Fixtures: a tiny world whose structure matches the H1-H19 predictions.
# --------------------------------------------------------------------------- #
def _actions_on_a_lattice() -> pd.DataFrame:
    """Actions with every chip quantity a multiple of 5 (H1 lattice, unit=5)."""
    return pd.DataFrame(
        {
            "hand_id": [1, 1, 1, 2, 2, 3, 3],
            "action_no": [0, 1, 2, 0, 1, 0, 1],
            "phase": ["preflop", "preflop", "flop", "preflop", "flop", "preflop", "turn"],
            "action": ["bet", "call", "raise", "bet", "call", "bet", "call"],
            "amount": [10, 10, 20, 5, 5, 25, 25],
            "amount_to": [10, 0, 20, 5, 0, 25, 0],
            "to_call": [10, 10, 20, 5, 5, 25, 25],
            "pot_before": [0, 10, 20, 0, 5, 0, 25],
            "stack_before": [1000, 1000, 980, 1000, 995, 1000, 975],
        }
    )


def _hands_pool_blocked() -> pd.DataFrame:
    """Two disjoint pools, contiguous non-overlapping hand_id ranges (H6, H10, H11)."""
    return pd.DataFrame(
        {
            "hand_id": [1, 2, 3, 4, 5, 6],
            "pool": [0, 0, 0, 1, 1, 1],
            "table_id": [10, 10, 10, 11, 11, 11],
            "phase": ["development"] * 4 + ["evaluation"] * 2,
        }
    )


def _seats_rotating() -> pd.DataFrame:
    """Deterministic +1 seat rotation per player across hands (H9)."""
    rows = []
    # Pool 0: players 100,101,102 across hands 1-3; seats rotate by +1.
    for h in (1, 2, 3):
        for k, p in enumerate((100, 101, 102)):
            rows.append({"hand_id": h, "player_id": p, "seat": (k + (h - 1)) % 3, "table_id": 10})
    # Pool 1: players 200,201,202 across hands 4-6.
    for h in (4, 5, 6):
        for k, p in enumerate((200, 201, 202)):
            rows.append({"hand_id": h, "player_id": p, "seat": (k + (h - 4)) % 3, "table_id": 11})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# H1-H5 : distribution artifacts
# --------------------------------------------------------------------------- #
def test_distribution_artifacts_detects_chip_lattice() -> None:
    actions = _actions_on_a_lattice()
    res = distribution_artifacts(actions, hands=_hands_pool_blocked())

    assert res.data_present is True
    h1 = res.hypotheses["H1"]
    # Every chip value is a multiple of 5, so unit 5 clears the 99% threshold.
    assert h1["divisible_fraction"][5] == 1.0
    assert h1["dominant_unit_ge_99pct"] in (5, 10, 20, 25)  # largest clearing unit
    assert h1["gcd_nonzero"] % 5 == 0

    # H4: closed, small action + phase vocabulary.
    h4 = res.hypotheses["H4"]
    assert set(h4["action_vocab"]) <= {"bet", "call", "raise", "fold", "check", "all_in"}
    assert h4["action_vocab_size"] <= 6
    assert set(h4["phase_vocab"]) == {"development", "evaluation"}


def test_distribution_artifacts_empty_is_graceful() -> None:
    res = distribution_artifacts(pd.DataFrame())
    assert isinstance(res, ProbeResult)
    assert res.data_present is False


# --------------------------------------------------------------------------- #
# H6-H9 : id / ordering regularities
# --------------------------------------------------------------------------- #
def test_id_ordering_pool_blocking_and_rotation() -> None:
    hands = _hands_pool_blocked()
    actions = _actions_on_a_lattice()
    seats = _seats_rotating()

    res = id_ordering_regularities(hands, actions, seats)

    h6 = res.hypotheses["H6"]
    assert h6["hand_id_strictly_increasing"] is True
    assert h6["pools_contiguous"] is True
    assert h6["pool_ranges_overlap"] is False

    h8 = res.hypotheses["H8"]
    assert h8["action_no_contiguous"] is True
    assert h8["distinct_base"] == [0]
    assert h8["phase_nondecreasing"] is True

    h9 = res.hypotheses["H9"]
    # Deterministic rotation => a very small set of distinct seat steps.
    assert h9["deterministic_rotation_hint"] is True


# --------------------------------------------------------------------------- #
# H10-H13 : pool construction
# --------------------------------------------------------------------------- #
def test_pool_construction_cardinality_and_coseating() -> None:
    hands = _hands_pool_blocked()
    seats = _seats_rotating()

    res = pool_construction(
        hands,
        seats,
        expected_pools=2,
        expected_players_per_pool=3,
        expected_hands_per_pool=3,
    )

    h10 = res.hypotheses["H10"]
    assert h10["distinct_pools"] == 2
    assert h10["matches_expected_pools"] is True
    assert h10["all_pools_have_expected_players"] is True
    assert h10["pools_disjoint"] is True

    h11 = res.hypotheses["H11"]
    assert h11["min"] == 3 and h11["max"] == 3
    assert h11["near_expected"] is True

    h12 = res.hypotheses["H12"]
    # Every within-pool pair (3 per pool * 2 pools = 6) co-seats at least once.
    assert h12["fraction_pairs_coseated"] == 1.0


def test_pool_construction_positive_prevalence_with_labels() -> None:
    hands = _hands_pool_blocked()
    seats = _seats_rotating()
    labels = pd.DataFrame({"pair_id": ["100_101"], "target_behavior": ["soft_play"]})

    res = pool_construction(hands, seats, labels=labels, expected_pools=2)
    assert res.labels_present is True
    assert res.hypotheses["H13"]["n_trusted_positive"] == 1


# --------------------------------------------------------------------------- #
# H14-H17 : confounder tells
# --------------------------------------------------------------------------- #
def test_confounder_tells_directedness_and_concentration() -> None:
    # Player 100 sends chips to 101 (positive directed flow) but is neutral vs 102.
    signals = pd.DataFrame(
        {
            "pair_id": ["100_101"] * 4 + ["100_102"] * 2,
            "player": [100, 100, 100, 100, 100, 100],
            "opponent": [101, 101, 101, 101, 102, 102],
            "hand_id": [1, 2, 3, 4, 1, 2],
            "value_flow": [5.0, 5.0, 5.0, 5.0, 0.0, 0.0],
            "is_flagged": [True, True, False, False, False, False],
        }
    )

    res = confounder_tells(signals)
    assert res.data_present is True

    h14 = res.hypotheses["H14"]
    contrasts = h14["directedness_contrast_by_pair"]
    # Directed flow to 101 (mean 5) minus 100's field baseline (mean of {5, 0}=2.5) > 0.
    assert contrasts["100_101"] > 0

    h17 = res.hypotheses["H17"]
    conc = h17["temporal_concentration_by_pair"]["100_101"]
    assert conc["n_flagged"] == 2
    assert conc["burst_count"] == 1  # the two flagged hands are consecutive => one burst


def test_confounder_tells_empty_is_graceful() -> None:
    res = confounder_tells(pd.DataFrame())
    assert res.data_present is False


# --------------------------------------------------------------------------- #
# H18-H19 : other_coordination signature
# --------------------------------------------------------------------------- #
def test_other_coordination_upper_tail_and_residual() -> None:
    # Three pairs; the untemplated positive has high advantage but low templates.
    pair_features = pd.DataFrame(
        {
            "pair_id": ["a", "b", "c"],
            "table_advantage": [0.1, 0.2, 0.95],
            "directed_transfer_score": [0.0, 0.0, 0.05],
            "soft_play_score": [0.0, 0.0, 0.05],
            "coordinated_isolation_score": [0.0, 0.0, 0.05],
        }
    )
    labels = pd.DataFrame(
        {"pair_id": ["c"], "target_behavior": ["other_coordination"]}
    )

    res = other_coordination_signature(pair_features, labels=labels)
    assert res.labels_present is True

    h18 = res.hypotheses["H18"]
    assert h18["max"] == 0.95
    # The single labeled other_coordination pair sits in the upper decile.
    assert h18["n_other_coordination_labeled"] == 1
    assert h18["n_other_in_upper_decile"] == 1

    h19 = res.hypotheses["H19"]
    # Pair 'c' is high-advantage but low on every template => residual candidate.
    assert h19["residual_cluster_size"] == 1


# --------------------------------------------------------------------------- #
# run_all_probes convenience
# --------------------------------------------------------------------------- #
def test_run_all_probes_graceful_without_data() -> None:
    results = run_all_probes({})
    assert set(results) == {
        "distribution_artifacts",
        "id_ordering_regularities",
        "pool_construction",
        "confounder_tells",
        "other_coordination_signature",
    }
    for r in results.values():
        assert isinstance(r, ProbeResult)
        assert r.data_present is False


def test_run_all_probes_with_fixtures() -> None:
    tables = {
        "actions": _actions_on_a_lattice(),
        "hands": _hands_pool_blocked(),
        "seats": _seats_rotating(),
    }
    results = run_all_probes(tables)
    assert results["distribution_artifacts"].data_present is True
    assert results["id_ordering_regularities"].data_present is True
    assert results["pool_construction"].data_present is True
