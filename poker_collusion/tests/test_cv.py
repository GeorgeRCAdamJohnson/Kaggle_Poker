"""Tests for the group-by-pool, family-stratified, PU-correct CV harness (task 5.1).

All fixtures are small and synthetic (no real competition data required), so the
suite runs fast. The harness is nevertheless designed to run on the real
``development_labels.csv`` when pointed at ``data/poker`` (via
``CVHarness.from_real_data`` / ``derive_pool_map`` with the real seats/hands).

Verified invariants:
  (a) grouping — no pool appears in more than one fold;
  (b) family stratification — each family is spread across folds when the grouping
      allows (not all of a family's positives land in one fold);
  (c) determinism — a fixed seed reproduces the fold assignment exactly;
  (d) PU-correct scoring wiring — per-fold scoring yields the three metric
      components plus the combined summary via the production metric, and unknown
      pairs are never materialised as negatives in the fold solution.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE
from poker_collusion.validation.cv import (
    CVHarness,
    build_fold_solution,
    derive_pool_map,
    make_pool_folds,
)

FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #
def _make_labels(n_pools: int = 12) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Build a synthetic label table with a known pool per pair.

    Each pool gets: one confirmed positive whose family cycles through the three
    disclosed families, two confirmed negatives, and one UNKNOWN pair (present in
    the frame with an unknown label_status that the harness must ignore, or simply
    absent — here we exercise "absent from confirmed" by tagging it 'unknown').
    Returns (labels, pool_map, evidence).
    """
    rows = []
    pool_map: dict[str, object] = {}
    evidence_rows = []
    pid_counter = 0

    def new_pair(pool: int) -> str:
        nonlocal pid_counter
        a = pool * 100 + pid_counter * 2
        b = a + 1
        pid = f"{a}_{b}"
        pool_map[pid] = pool
        pid_counter += 1
        return pid, a, b

    for pool in range(n_pools):
        pid_counter = 0
        # one positive, family cycles across pools -> families spread over pools.
        fam = FAMILIES[pool % len(FAMILIES)]
        pid, a, b = new_pair(pool)
        rows.append(
            {
                "pair_id": pid,
                "player_1": a,
                "player_2": b,
                "label": 1,
                "label_status": "confirmed_target",
                "behavior_family": fam,
            }
        )
        # planted evidence for the positive.
        for rank in range(1, 4):
            evidence_rows.append(
                {"pair_id": pid, "evidence_rank": rank, "hand_id": f"{pid}_H{rank}", "behavior_family": fam}
            )
        # two confirmed negatives.
        for _ in range(2):
            pid, a, b = new_pair(pool)
            rows.append(
                {
                    "pair_id": pid,
                    "player_1": a,
                    "player_2": b,
                    "label": 0,
                    "label_status": "confirmed_non_target",
                    "behavior_family": "none",
                }
            )
        # one unknown pair (label_status not a confirmed token).
        pid, a, b = new_pair(pool)
        rows.append(
            {
                "pair_id": pid,
                "player_1": a,
                "player_2": b,
                "label": 0,
                "label_status": "unknown",
                "behavior_family": "none",
            }
        )

    labels = pd.DataFrame(rows)
    evidence = pd.DataFrame(evidence_rows)
    return labels, pool_map, evidence


# --------------------------------------------------------------------------- #
# (a) Grouping invariant                                                      #
# --------------------------------------------------------------------------- #
def test_no_pool_appears_in_more_than_one_fold():
    labels, pool_map, _ = _make_labels(n_pools=12)
    assignment = make_pool_folds(labels, pool_map, n_folds=4, seed=404, families=FAMILIES)
    # Each pool maps to exactly one fold index.
    assert set(assignment.keys()) == set(pool_map.values())
    for fold in assignment.values():
        assert 0 <= fold < 4

    harness = CVHarness(labels, pool_map, n_folds=4, seed=404, evidence=None)
    folds = harness.folds()
    # Validation pools are disjoint across folds.
    seen: set = set()
    for fold in folds:
        assert seen.isdisjoint(fold.validation_pools)
        seen |= set(fold.validation_pools)
    # Every pool is covered exactly once.
    assert seen == set(pool_map.values())

    # No validation pair leaks into its own train set (implied by pool disjointness).
    for fold in folds:
        assert set(fold.validation_pairs).isdisjoint(fold.train_pairs)


# --------------------------------------------------------------------------- #
# (b) Family stratification                                                   #
# --------------------------------------------------------------------------- #
def test_family_stratification_spreads_each_family_across_folds():
    labels, pool_map, _ = _make_labels(n_pools=12)  # 4 positives per family
    n_folds = 4
    assignment = make_pool_folds(labels, pool_map, n_folds=n_folds, seed=404, families=FAMILIES)

    # Count positives per (fold, family).
    counts = {fam: [0] * n_folds for fam in FAMILIES}
    for row in labels.itertuples(index=False):
        if row.label_status != "confirmed_target":
            continue
        pool = pool_map[row.pair_id]
        fold = assignment[pool]
        counts[row.behavior_family][fold] += 1

    # With 4 positives of each family and 4 folds, a good stratification should
    # NOT dump all of a family into a single fold. Assert every family touches at
    # least two folds.
    for fam in FAMILIES:
        folds_touched = sum(1 for c in counts[fam] if c > 0)
        assert folds_touched >= 2, f"family {fam} concentrated: {counts[fam]}"


# --------------------------------------------------------------------------- #
# (c) Determinism                                                             #
# --------------------------------------------------------------------------- #
def test_fold_assignment_is_deterministic_given_seed():
    labels, pool_map, _ = _make_labels(n_pools=12)
    a1 = make_pool_folds(labels, pool_map, n_folds=5, seed=404, families=FAMILIES)
    a2 = make_pool_folds(labels, pool_map, n_folds=5, seed=404, families=FAMILIES)
    assert a1 == a2

    # Harness folds are also reproducible.
    h1 = CVHarness(labels, pool_map, n_folds=5, seed=404)
    h2 = CVHarness(labels, pool_map, n_folds=5, seed=404)
    assert [f.validation_pairs for f in h1.folds()] == [f.validation_pairs for f in h2.folds()]

    # A different seed may reassign pools (not asserted equal, just that it runs).
    a3 = make_pool_folds(labels, pool_map, n_folds=5, seed=999, families=FAMILIES)
    assert set(a3.keys()) == set(a1.keys())


# --------------------------------------------------------------------------- #
# (d) PU-correct scoring wiring                                               #
# --------------------------------------------------------------------------- #
def test_fold_solution_is_pu_correct_excludes_unknown():
    labels, pool_map, evidence = _make_labels(n_pools=6)
    # Ask for every pair in a pool INCLUDING the unknown one.
    pool0_pairs = [pid for pid, pool in pool_map.items() if pool == 0]
    solution = build_fold_solution(labels, pool0_pairs, evidence=evidence)

    # Unknown pair must be absent (never scored as a negative).
    unknown_ids = set(
        labels.loc[labels["label_status"] == "unknown", "pair_id"].astype(str)
    )
    assert unknown_ids.isdisjoint(set(solution["pair_id"]))

    # Confirmed positive present with risk 1 + its family + real evidence hands.
    pos = solution.loc[solution["risk_score"] == 1]
    assert len(pos) == 1
    assert pos.iloc[0]["predicted_behavior"] in FAMILIES
    assert pos.iloc[0]["evidence_hand_1"] != NO_EVIDENCE

    # Confirmed negatives present with risk 0, behavior none, no evidence.
    neg = solution.loc[solution["risk_score"] == 0]
    assert len(neg) == 2
    assert (neg["predicted_behavior"] == "none").all()
    assert (neg["evidence_hand_1"] == NO_EVIDENCE).all()


def test_run_produces_three_components_plus_combined():
    labels, pool_map, evidence = _make_labels(n_pools=12)
    harness = CVHarness(labels, pool_map, n_folds=4, seed=404, evidence=evidence)

    def perfect_predict(fold, solution: pd.DataFrame) -> pd.DataFrame:
        # A perfect submission = copy the solution (risk 1 for positives, correct
        # behavior, correct evidence). This exercises the full scoring wiring.
        return solution.copy()

    report = harness.run(perfect_predict)

    assert len(report.folds) == 4
    mean = report.mean
    for key in ("pair_ap", "evidence_map", "behavior_map", "combined"):
        assert key in mean
    # Perfect predictions -> each component should be 1.0 (folds all have >=1 pos).
    assert mean["pair_ap"] == pytest.approx(1.0)
    assert mean["evidence_map"] == pytest.approx(1.0)
    assert mean["behavior_map"] == pytest.approx(1.0)
    assert mean["combined"] == pytest.approx(1.0)

    # per_fold exposes each fold's components separately (Req 10.5).
    assert len(report.per_fold) == 4
    for comp in report.per_fold:
        assert set(comp) == {"pair_ap", "evidence_map", "behavior_map", "combined"}


def test_run_scores_imperfect_predictions_below_one():
    labels, pool_map, evidence = _make_labels(n_pools=12)
    harness = CVHarness(labels, pool_map, n_folds=4, seed=404, evidence=evidence)

    def flat_predict(fold, solution: pd.DataFrame) -> pd.DataFrame:
        # Constant risk + wrong behavior + no evidence -> degraded components.
        out = solution.copy()
        out["risk_score"] = 0.5
        out["predicted_behavior"] = "none"
        for i in range(1, 6):
            out[f"evidence_hand_{i}"] = NO_EVIDENCE
        return out

    report = harness.run(flat_predict)
    mean = report.mean
    # No submitted evidence -> Evidence MAP@5 collapses to 0.
    assert mean["evidence_map"] == pytest.approx(0.0)
    # Wrong behavior (all "none") + constant risk degrades Behavior MAP well below
    # the perfect 1.0 (the metric still ranks the degenerate zero-risk scores, so
    # it is not exactly 0). It must at least be strictly below perfect.
    assert mean["behavior_map"] < 1.0
    # Combined stays finite and below the perfect score.
    assert 0.0 <= mean["combined"] < 1.0


# --------------------------------------------------------------------------- #
# pool derivation from injected seats/hands (mirrors the real join, no I/O)   #
# --------------------------------------------------------------------------- #
def test_derive_pool_map_from_seats_and_hands():
    # Two pools; player -> table via seats+hands join (column-projected shape).
    labels = pd.DataFrame(
        {
            "pair_id": ["1_2", "3_4"],
            "player_1": [1, 3],
            "player_2": [2, 4],
            "label_status": ["confirmed_target", "confirmed_non_target"],
            "behavior_family": ["soft_play", "none"],
        }
    )
    seats = pd.DataFrame(
        {
            "hand_id": [10, 10, 20, 20],
            "player_id": [1, 2, 3, 4],
        }
    )
    hands = pd.DataFrame({"hand_id": [10, 20], "table_id": [500, 600]})
    pool_map = derive_pool_map(labels, seats=seats, hands=hands)
    assert pool_map == {"1_2": 500, "3_4": 600}


def test_derive_pool_map_with_explicit_player_pool():
    labels = pd.DataFrame(
        {
            "pair_id": ["1_2"],
            "player_1": [1],
            "player_2": [2],
            "label_status": ["confirmed_target"],
            "behavior_family": ["soft_play"],
        }
    )
    pool_map = derive_pool_map(labels, player_pool={1: 7, 2: 7})
    assert pool_map == {"1_2": 7}
