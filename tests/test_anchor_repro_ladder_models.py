"""Unit tests for the ladder data models + ``LadderValidator.score_ladder`` (task 5.1).

Covers ``LadderPoint`` / ``MetricRankTracking`` / ``RankTrackingVerdict`` (frozen,
correct field names per design "Data Models") and ``LadderValidator.score_ladder``:

- a SCORABLE point is produced by RE-RUNNING the recipe on the dev holdout and scoring
  the dev-pair predictions with the canonical scorer (never dev-scoring the eval CSV);
- an artifact with NO re-runnable recipe is recorded BLOCKED with no fabricated
  ``local_pair_ap`` / ``local_combined`` (contract Rules 1 & 9, Req 2.1);
- a re-run reporting a different ``recipe_id`` is refused (no cross-regime mixing, 2.6);
- both local scores + the recipe id are recorded so points are mutually comparable.

The canonical scorer is driven with an INJECTED synthetic dev holdout so no real data
files are touched (mirrors the existing anchor_repro scorer tests).

Validates: Requirements 2.1, 2.5, 2.6
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.submission.writer import SUBMISSION_COLUMNS

from anchor_repro.artifact_names import parse_artifact_lb
from anchor_repro.ladder import (
    BLOCKED,
    SCORABLE,
    LadderPoint,
    LadderValidator,
    MetricRankTracking,
    RankTrackingVerdict,
    RecipeRerun,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer


# --------------------------------------------------------------------------- #
# Synthetic dev holdout so the canonical scorer never touches real data files  #
# --------------------------------------------------------------------------- #
def _dev_labels() -> pd.DataFrame:
    # 3 confirmed positives + 3 confirmed negatives (PU: confirmed only).
    return pd.DataFrame(
        {
            "pair_id": [f"P{i}" for i in range(6)],
            "label_status": ["confirmed_target"] * 3 + ["confirmed_non_target"] * 3,
            "behavior_family": [
                "directed_transfer",
                "soft_play",
                "coordinated_isolation",
                "none",
                "none",
                "none",
            ],
        }
    )


def _dev_predictions(risk):
    frame = pd.DataFrame({"pair_id": [f"P{i}" for i in range(6)]})
    frame["risk_score"] = risk
    frame["predicted_behavior"] = [
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
        "none",
        "none",
        "none",
    ]
    for col in SUBMISSION_COLUMNS:
        if col not in frame.columns:
            frame[col] = "NO_EVIDENCE"
    return frame[list(SUBMISSION_COLUMNS)]


def _scorer() -> CanonicalScorer:
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(),
        labels=_dev_labels(),
        evidence=None,
    )


# --------------------------------------------------------------------------- #
# LadderPoint honesty invariants                                              #
# --------------------------------------------------------------------------- #
def test_ladder_point_is_frozen():
    pt = LadderPoint("ladder_044519", 0.44519, 0.38, 0.30, "canonical_v1", "src")
    assert dataclasses.is_dataclass(pt)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pt.real_lb = 0.9  # type: ignore[misc]


def test_blocked_point_must_not_carry_a_local_number():
    # Fabricating a number for a BLOCKED artifact is exactly what the contract forbids.
    with pytest.raises(ValueError):
        LadderPoint(
            "ladder_044519", 0.44519, 0.38, None, "canonical_v1", "src",
            status=BLOCKED, blocked_reason="no recipe",
        )


def test_blocked_point_requires_a_reason():
    with pytest.raises(ValueError):
        LadderPoint(
            "ladder_044519", 0.44519, None, None, "canonical_v1", "src", status=BLOCKED,
        )


def test_scorable_point_requires_both_local_scores():
    with pytest.raises(ValueError):
        LadderPoint("ladder_044519", 0.44519, 0.38, None, "canonical_v1", "src")


# --------------------------------------------------------------------------- #
# MetricRankTracking / RankTrackingVerdict models exist with the design fields #
# --------------------------------------------------------------------------- #
def test_metric_rank_tracking_and_verdict_shapes():
    primary = MetricRankTracking("pair_ap", 0.83, 0.7, True, "TRUSTED")
    secondary = MetricRankTracking("combined", 0.5, 0.7, False, "WEAK")
    verdict = RankTrackingVerdict(
        n_points=9,
        ci_note="n=9: wide CI",
        primary=primary,
        secondary=secondary,
        agree=False,
        disagreement_note="combined weaker than PairAP",
        passed=primary.passed,
        verdict=primary.verdict,
        message="primary decides",
    )
    # Overall verdict mirrors the PRIMARY (PairAP) result (pre-registered gate).
    assert verdict.passed == primary.passed
    assert verdict.verdict == primary.verdict
    # Both metric results are present (never cherry-picked).
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"
    assert verdict.to_provenance()["primary"]["metric_name"] == "pair_ap"


# --------------------------------------------------------------------------- #
# score_ladder: honest SCORABLE vs BLOCKED                                    #
# --------------------------------------------------------------------------- #
def test_score_ladder_all_blocked_when_no_recipe_runner():
    # The honest default: without a re-runnable recipe, every point is BLOCKED and
    # NO local number is invented (contract Rules 1 & 9, Req 2.1).
    validator = LadderValidator(_scorer())
    names = ["submission_best_044519.csv", "submission_best_018462.csv"]
    points = validator.score_ladder(names)
    assert [p.status for p in points] == [BLOCKED, BLOCKED]
    for p in points:
        assert p.local_pair_ap is None and p.local_combined is None
        assert p.blocked_reason
        assert p.recipe_id == "canonical_v1"
    # real_lb parsed from the filename without distortion (external judge).
    assert points[0].real_lb == pytest.approx(0.44519)
    assert points[0].config_id == "ladder_044519"


def test_score_ladder_scores_when_recipe_reruns_on_dev_holdout():
    scorer = _scorer()

    def runner(path, artifact):
        # A perfect-ordering dev-holdout re-run (positives ranked above negatives).
        preds = _dev_predictions([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        return RecipeRerun(dev_predictions=preds, recipe_id="canonical_v1", detail="rerun@holdout")

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(["submission_best_044519.csv"])
    (pt,) = points
    assert pt.status == SCORABLE
    assert pt.local_pair_ap is not None and pt.local_combined is not None
    # PairAP is the anchor's local_holdout_ap; perfect ordering ⇒ AP == 1.0.
    assert pt.local_pair_ap == pytest.approx(1.0)
    assert pt.recipe_id == "canonical_v1"
    assert "rerun@holdout" in pt.source


def test_score_ladder_blocks_on_recipe_id_mismatch_no_cross_regime_mixing():
    scorer = _scorer()

    def runner(path, artifact):
        preds = _dev_predictions([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        return RecipeRerun(dev_predictions=preds, recipe_id="some_other_recipe")

    validator = LadderValidator(scorer, recipe_runner=runner)
    (pt,) = validator.score_ladder(["submission_best_044519.csv"])
    assert pt.status == BLOCKED
    assert pt.local_pair_ap is None and pt.local_combined is None
    assert "regime" in pt.blocked_reason.lower() or "recipe_id" in pt.blocked_reason


def test_score_ladder_preserves_order_and_mixes_scorable_and_blocked():
    scorer = _scorer()

    def runner(path, artifact):
        # Only the 0.44519 artifact has a re-runnable recipe; the other is BLOCKED.
        if artifact.digits == "044519":
            return RecipeRerun(_dev_predictions([0.9, 0.8, 0.7, 0.2, 0.1, 0.05]), "canonical_v1")
        return None

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(
        ["submission_best_018462.csv", "submission_best_044519.csv"]
    )
    assert points[0].status == BLOCKED
    assert points[1].status == SCORABLE
    assert points[1].local_pair_ap is not None
