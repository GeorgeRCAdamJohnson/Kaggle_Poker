"""Unit tests for the table-disjoint PU-stress AP holdout scorer (task 8.1).

These confirm the THIN WRAPPER wiring: ``holdout_pair_ap`` drives the REUSED
group-by-``table_id`` split (``CVHarness`` + ``build_fold_solution``) and the
REUSED reference metric (``production_metric.score_components``, numerically
identical to ``reference_public_metric``) to return a finite PU-stress **Pair AP**
in ``[0, 1]`` — and does not reimplement either the split or the metric.

The tests use a SMALL synthetic label frame with an injected ``player_pool``
(``CVHarness`` mode 1: no parquet I/O), so they run fast and offline. Two of them
spy on the reused boundary (``build_fold_solution`` / ``score_components``) to
assert the wrapper actually calls the substrate rather than computing AP itself.

Requirements: 3.1, 5.4.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.tuning_harness.holdout import scorer as scorer_mod
from poker_collusion.tuning_harness.holdout import (
    HoldoutScore,
    holdout_pair_ap,
    ranking_predict_fn,
)
from poker_collusion.tuning_harness.models import RankVector
from poker_collusion.validation.cv import CVHarness

# The three disclosed families the metric/CV harness expect.
_FAMILY = "directed_transfer"


def _make_labels(n_pos: int = 6, n_neg: int = 6) -> pd.DataFrame:
    """A tiny confirmed-label frame: n_pos positives + n_neg negatives.

    Each pair carries two player ids so the injected ``player_pool`` can assign it
    a ``table_id`` pool; positives are ``confirmed_target``, negatives
    ``confirmed_non_target``.
    """
    rows = []
    for i in range(n_pos):
        rows.append(
            {
                "pair_id": f"pos_{i}",
                "player_1": f"p{i}a",
                "player_2": f"p{i}b",
                "label_status": "confirmed_target",
                "behavior_family": _FAMILY,
            }
        )
    for i in range(n_neg):
        rows.append(
            {
                "pair_id": f"neg_{i}",
                "player_1": f"n{i}a",
                "player_2": f"n{i}b",
                "label_status": "confirmed_non_target",
                "behavior_family": "none",
            }
        )
    return pd.DataFrame(rows)


def _make_pool_map(labels: pd.DataFrame, n_pools: int = 4) -> dict:
    """Spread the pairs' players across ``n_pools`` disjoint table pools.

    Both players of a pair share the same pool (a pair lives in one table), and
    pairs round-robin across pools so folds are non-degenerate and table-disjoint.
    """
    player_pool: dict[str, str] = {}
    for idx, row in enumerate(labels.itertuples(index=False)):
        pool = f"table_{idx % n_pools}"
        player_pool[getattr(row, "player_1")] = pool
        player_pool[getattr(row, "player_2")] = pool
    return player_pool


def _make_harness(n_folds: int = 4) -> CVHarness:
    labels = _make_labels()
    player_pool = _make_pool_map(labels, n_pools=n_folds)
    from poker_collusion.validation.cv import derive_pool_map

    pool_map = derive_pool_map(labels, player_pool=player_pool)
    return CVHarness(labels, pool_map, n_folds=n_folds, seed=0)


def test_returns_finite_pair_ap_in_unit_interval():
    """A candidate ranking scores a finite PU-stress AP in [0, 1] via the reused stack."""
    harness = _make_harness()
    # Rank all positives above all negatives -> a perfect ranking.
    pos = [f"pos_{i}" for i in range(6)]
    neg = [f"neg_{i}" for i in range(6)]
    ranking = RankVector(pair_ids=tuple(pos + neg))

    score = holdout_pair_ap(harness, ranking)

    assert isinstance(score, HoldoutScore)
    assert 0.0 <= float(score) <= 1.0
    assert pd.notna(float(score))
    # Provenance is carried honestly.
    assert score.n_folds >= 1
    assert score.n_pairs >= 1
    assert score.n_positive_pairs >= 1


def test_perfect_ranking_scores_ap_one():
    """Positives-first ranking -> pooled Pair AP is 1.0 (metric correctness through the wrapper)."""
    harness = _make_harness()
    pos = [f"pos_{i}" for i in range(6)]
    neg = [f"neg_{i}" for i in range(6)]
    ranking = RankVector(pair_ids=tuple(pos + neg))

    score = holdout_pair_ap(harness, ranking)
    assert float(score) == pytest.approx(1.0)


def test_worst_ranking_scores_below_perfect():
    """Negatives-first ranking scores strictly below the perfect ranking."""
    harness = _make_harness()
    pos = [f"pos_{i}" for i in range(6)]
    neg = [f"neg_{i}" for i in range(6)]
    worst = RankVector(pair_ids=tuple(neg + pos))
    best = RankVector(pair_ids=tuple(pos + neg))

    assert float(holdout_pair_ap(harness, worst)) < float(holdout_pair_ap(harness, best))


def test_explicit_scores_are_honoured():
    """When the RankVector carries explicit scores, they drive the risk column."""
    harness = _make_harness()
    pairs = tuple([f"pos_{i}" for i in range(6)] + [f"neg_{i}" for i in range(6)])
    # High risk on positives, low on negatives.
    scores = tuple([0.9] * 6 + [0.1] * 6)
    ranking = RankVector(pair_ids=pairs, scores=scores)

    score = holdout_pair_ap(harness, ranking)
    assert float(score) == pytest.approx(1.0)


def test_wrapper_invokes_reused_split_and_metric(monkeypatch):
    """Spy proof: the wrapper calls the REUSED build_fold_solution + score_components."""
    harness = _make_harness()
    ranking = RankVector(pair_ids=tuple([f"pos_{i}" for i in range(6)] + [f"neg_{i}" for i in range(6)]))

    calls = {"build_fold_solution": 0, "score_components": 0}

    real_build = scorer_mod.build_fold_solution
    real_score = scorer_mod.score_components

    def spy_build(*args, **kwargs):
        calls["build_fold_solution"] += 1
        return real_build(*args, **kwargs)

    def spy_score(*args, **kwargs):
        calls["score_components"] += 1
        return real_score(*args, **kwargs)

    monkeypatch.setattr(scorer_mod, "build_fold_solution", spy_build)
    monkeypatch.setattr(scorer_mod, "score_components", spy_score)

    holdout_pair_ap(harness, ranking)

    # The reused split was queried per non-empty fold, and the reused metric scored once (pooled).
    assert calls["build_fold_solution"] >= 1
    assert calls["score_components"] == 1


def test_predict_fn_passthrough_is_supported():
    """A caller predict_fn is passed straight through to the reused CV split."""
    harness = _make_harness()

    def predict_fn(fold, solution):
        # Emit a perfect ranking directly (positives high, negatives low).
        from poker_collusion.submission.writer import SUBMISSION_COLUMNS

        rows = []
        for pid in solution["pair_id"].astype(str):
            risk = 0.9 if pid.startswith("pos_") else 0.1
            rows.append(
                {
                    "pair_id": pid,
                    "risk_score": risk,
                    "predicted_behavior": "none",
                    "evidence_hand_1": "NO_EVIDENCE",
                    "evidence_hand_2": "NO_EVIDENCE",
                    "evidence_hand_3": "NO_EVIDENCE",
                    "evidence_hand_4": "NO_EVIDENCE",
                    "evidence_hand_5": "NO_EVIDENCE",
                }
            )
        return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))

    score = holdout_pair_ap(harness, predict_fn)
    assert float(score) == pytest.approx(1.0)


def test_ranking_predict_fn_covers_only_solution_pairs():
    """The RankVector adapter emits exactly the fold solution's pair set, risk in [0,1]."""
    harness = _make_harness()
    ranking = RankVector(pair_ids=tuple([f"pos_{i}" for i in range(6)] + [f"neg_{i}" for i in range(6)]))
    predict_fn = ranking_predict_fn(ranking)

    fold = harness.folds()[0]
    from poker_collusion.validation.cv import build_fold_solution

    solution = build_fold_solution(harness.labels, fold.validation_pairs, evidence=harness.evidence)
    if len(solution) == 0:
        pytest.skip("degenerate fold in this synthetic split")
    submission = predict_fn(fold, solution)

    assert set(submission["pair_id"]) == set(solution["pair_id"].astype(str))
    assert submission["risk_score"].between(0.0, 1.0).all()
