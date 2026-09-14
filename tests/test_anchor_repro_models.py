"""Unit tests for the poker-anchor-reproduction frozen data models (task 1.2).

Covers ``ScoringRecipe`` and ``CanonicalScore``:
- correct defaults per design.md (canonical_v1, pair_ap PRIMARY, combined SECONDARY,
  table_disjoint_holdout_40pct, pu_confirmed_only, holdout_seed == Seeds.cv_split);
- frozen/immutability (both are ``@dataclass(frozen=True)``);
- provenance serialization naming the primary + secondary components so the recipe can
  be embedded in every anchor (Req 2.5, 2.6).

Validates: Requirements 2.5, 2.6
"""

from __future__ import annotations

import dataclasses

import pytest

from poker_collusion.config import Seeds
from poker_collusion.metric.production_metric import ScoreComponents

from anchor_repro.models import CanonicalScore, ScoringRecipe


# --------------------------------------------------------------------------- #
# ScoringRecipe                                                               #
# --------------------------------------------------------------------------- #
def test_scoring_recipe_defaults_match_design():
    recipe = ScoringRecipe()
    assert recipe.recipe_id == "canonical_v1"
    assert recipe.metric == "reference_public_metric.score"
    # PairAP is the PRIMARY rank-tracked component (the metric is 70% PairAP).
    assert recipe.primary_ap_component == "pair_ap"
    # The full official combined score is the SECONDARY diagnostic.
    assert recipe.secondary_component == "combined"
    assert recipe.split == "table_disjoint_holdout_40pct"
    assert recipe.unknown_handling == "pu_confirmed_only"
    # Deterministic split seed reused from poker_collusion — never hard-coded.
    assert recipe.holdout_seed == Seeds.cv_split == 404


def test_scoring_recipe_is_frozen():
    recipe = ScoringRecipe()
    assert dataclasses.is_dataclass(recipe)
    with pytest.raises(dataclasses.FrozenInstanceError):
        recipe.recipe_id = "tampered"  # type: ignore[misc]


def test_scoring_recipe_to_provenance_names_primary_and_secondary():
    prov = ScoringRecipe().to_provenance()
    assert prov == {
        "recipe_id": "canonical_v1",
        "metric": "reference_public_metric.score",
        "primary_ap_component": "pair_ap",
        "secondary_component": "combined",
        "split": "table_disjoint_holdout_40pct",
        "unknown_handling": "pu_confirmed_only",
        "holdout_seed": 404,
    }
    # The provenance must NAME which component is primary (the gate) vs secondary.
    assert prov["primary_ap_component"] == "pair_ap"
    assert prov["secondary_component"] == "combined"


def test_scoring_recipe_provenance_round_trips_into_a_recipe():
    original = ScoringRecipe()
    rebuilt = ScoringRecipe(**original.to_provenance())
    assert rebuilt == original


# --------------------------------------------------------------------------- #
# CanonicalScore                                                              #
# --------------------------------------------------------------------------- #
def _make_components(pair_ap: float = 0.3839) -> ScoreComponents:
    combined = 0.70 * pair_ap + 0.20 * 0.5 + 0.10 * 0.25
    return ScoreComponents(
        pair_ap=pair_ap,
        evidence_map=0.5,
        behavior_map=0.25,
        combined=combined,
    )


def test_canonical_score_holds_both_scores_and_components():
    comps = _make_components()
    score = CanonicalScore(pair_ap=comps.pair_ap, combined=comps.combined, components=comps)
    # pair_ap is the PRIMARY gate quantity (the anchor's local_holdout_ap).
    assert score.pair_ap == comps.pair_ap
    # combined is the SECONDARY diagnostic.
    assert score.combined == comps.combined
    assert score.components is comps


def test_canonical_score_is_frozen():
    comps = _make_components()
    score = CanonicalScore(pair_ap=comps.pair_ap, combined=comps.combined, components=comps)
    assert dataclasses.is_dataclass(score)
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.pair_ap = 0.9  # type: ignore[misc]


def test_canonical_score_to_provenance_labels_primary_secondary():
    comps = _make_components()
    score = CanonicalScore(pair_ap=comps.pair_ap, combined=comps.combined, components=comps)
    prov = score.to_provenance()
    assert prov["pair_ap"] == comps.pair_ap
    assert prov["combined"] == comps.combined
    assert prov["components"] == comps.as_dict()
    # The anchor must record which local score is the gate vs the diagnostic so the
    # numbers are never mistaken for an absolute LB reproduction (Req 2.4, Rules 1/9).
    assert prov["primary_component"] == "pair_ap"
    assert prov["secondary_component"] == "combined"
