"""Property test for scorer–recipe consistency across artifacts (task 5.4).

# Feature: poker-anchor-reproduction, Property 2: For any set of artifacts scored to
# produce anchors, every resulting anchor SHALL record the SAME ``recipe_id``; scoring
# the same dev-prediction frame under two anchors' recorded recipes SHALL yield the
# same local holdout AP (no cross-regime mixing between our-artifact and competitor
# anchors).

Property 2 (design.md "Correctness Properties", Req 2.5, 2.6, 3.5) is the "no
cross-regime mixing" guarantee that makes anchors mutually comparable: the canonical
scorer is a SINGLE, frozen ``ScoringRecipe`` shared by every anchor, so

1. every :class:`LadderPoint` a batch of artifacts produces (SCORABLE or BLOCKED)
   records the identical ``recipe_id`` — the canonical recipe id, never a per-artifact
   or cross-regime one; and
2. because that one recipe id names one recipe, scoring the SAME dev-prediction frame
   under any two anchors' recorded recipes yields the SAME local holdout AP (PairAP) —
   there is no second regime for two anchors to disagree across.

Part (2) is checked directly: the recipe id every ladder point records is looked up
back to its :class:`ScoringRecipe`, a fresh scorer is built per point from that
recipe, and the same random dev-prediction frame is scored under each. All the local
holdout APs (and combined scores) must be identical — a divergence would be exactly
the cross-regime mixing this property forbids.

The generator builds a FIXED synthetic confirmed dev-label set (so no real data files
are touched — mirroring ``test_anchor_repro_scorer_determinism_property.py`` and
``test_anchor_repro_ladder_models.py``) and drives the ladder with an INJECTED
``recipe_runner`` that returns dev-holdout predictions under the canonical recipe id.
Min 100 iterations (``max_examples=150``).

Governed by the reverse-engineering-accountability contract: the equal-AP claim is a
MEASURED local property (identical inputs + identical recipe ⇒ identical PairAP), never
an absolute-LB reproduction claim (impossible; the real_lb is the external judge).

Validates: Requirements 2.5, 2.6, 3.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.config import PipelineConfig
from poker_collusion.metric.reference_public_metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
)

from anchor_repro.artifact_names import ArtifactLB, format_artifact_filename
from anchor_repro.ladder import (
    BLOCKED,
    SCORABLE,
    LadderPoint,
    LadderValidator,
    RecipeRerun,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer


# --------------------------------------------------------------------------- #
# Fixed synthetic dev labels/evidence (no real data files touched)            #
# --------------------------------------------------------------------------- #
# Confirmed pairs whose pair_id set the built PU solution equals, so a prediction
# frame covering EXACTLY these pairs scores cleanly (no coverage-mismatch guard).
_CONFIRMED_PAIRS = ("P1", "P2", "P3", "P4", "P5", "P6")

_BEHAVIORS = sorted(ALLOWED_BEHAVIORS)
_HAND_POOL = ["HA", "HB", "HC", "HD", "HE", "HX", "HY", "HZ"]


def _dev_labels() -> pd.DataFrame:
    """Confirmed targets (one per target family) + confirmed non-targets."""
    return pd.DataFrame(
        [
            {"pair_id": "P1", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
            {"pair_id": "P2", "label_status": "confirmed_target",
             "behavior_family": "soft_play"},
            {"pair_id": "P3", "label_status": "confirmed_target",
             "behavior_family": "coordinated_isolation"},
            {"pair_id": "P4", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "P5", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "P6", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
        ]
    )


def _dev_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "P1", "evidence_rank": 1, "hand_id": "HA"},
            {"pair_id": "P1", "evidence_rank": 2, "hand_id": "HB"},
            {"pair_id": "P2", "evidence_rank": 1, "hand_id": "HC"},
            {"pair_id": "P3", "evidence_rank": 1, "hand_id": "HD"},
            {"pair_id": "P3", "evidence_rank": 2, "hand_id": "HE"},
            {"pair_id": "P6", "evidence_rank": 1, "hand_id": "HX"},
        ]
    )


def _make_scorer(recipe: Optional[ScoringRecipe] = None) -> CanonicalScorer:
    return CanonicalScorer(
        recipe if recipe is not None else ScoringRecipe(),
        PipelineConfig(),
        labels=_dev_labels(),
        evidence=_dev_evidence(),
    )


# --------------------------------------------------------------------------- #
# Smart generators: valid dev-prediction rows over the confirmed pair set     #
# --------------------------------------------------------------------------- #
def _evidence_slots(hands: List[str]) -> Dict[str, str]:
    """Five rank-ordered evidence slots; hands must not repeat within a pair, so the
    drawn hands are de-duplicated (order preserved) before padding with NO_EVIDENCE."""
    seen: List[str] = []
    for h in hands:
        if h not in seen:
            seen.append(h)
    padded = seen[:5] + [NO_EVIDENCE] * (5 - min(len(seen), 5))
    return {col: padded[i] for i, col in enumerate(EVIDENCE_COLUMNS)}


@st.composite
def _prediction_row(draw, pair_id: str) -> Dict[str, object]:
    """A single valid prediction row for ``pair_id`` (risk in [0,1], allowed behavior,
    0..5 distinct evidence hands)."""
    risk = draw(st.floats(min_value=0.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False))
    behavior = draw(st.sampled_from(_BEHAVIORS))
    hands = draw(st.lists(st.sampled_from(_HAND_POOL), min_size=0, max_size=5))
    return {
        "pair_id": pair_id,
        "risk_score": risk,
        "predicted_behavior": behavior,
        **_evidence_slots(hands),
    }


@st.composite
def _dev_predictions(draw) -> pd.DataFrame:
    """A full valid dev-prediction frame covering EXACTLY the confirmed pair set."""
    rows = [draw(_prediction_row(pid)) for pid in _CONFIRMED_PAIRS]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Generator: a batch of distinct known-LB artifact filenames                  #
# --------------------------------------------------------------------------- #
@st.composite
def _artifact_batch(draw) -> List[str]:
    """A non-empty batch of DISTINCT ``submission_best_<d>.csv`` filenames.

    Each is a valid five-digit known-LB artifact name (parseable by
    ``parse_artifact_lb``); distinct so ``score_ladder`` produces one point per
    artifact in a stable, comparable set."""
    scores = draw(
        st.lists(
            st.integers(min_value=0, max_value=99_999),
            min_size=1,
            max_size=9,
            unique=True,
        )
    )
    return [format_artifact_filename(s / 100_000, width=5) for s in scores]


# --------------------------------------------------------------------------- #
# Property 2a: every produced ladder point records the SAME recipe_id          #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 2: single recipe_id across anchors
@settings(max_examples=150, deadline=None)
@given(names=_artifact_batch(), preds=_dev_predictions())
def test_all_ladder_points_share_one_recipe_id(
    names: List[str], preds: pd.DataFrame
) -> None:
    """For any batch of artifacts scored under the canonical scorer, EVERY resulting
    ladder point (SCORABLE or BLOCKED) records the identical ``recipe_id`` — the one
    canonical recipe id, never a per-artifact or cross-regime one (Req 2.5, 2.6)."""
    recipe = ScoringRecipe()
    scorer = _make_scorer(recipe)

    def runner(path: Path, artifact: ArtifactLB) -> RecipeRerun:
        # Every artifact re-runs under the SAME canonical recipe id → SCORABLE.
        return RecipeRerun(
            dev_predictions=preds.copy(),
            recipe_id=recipe.recipe_id,
            detail=f"rerun@{artifact.digits}",
        )

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(names)

    assert len(points) == len(names)
    recipe_ids = {p.recipe_id for p in points}
    # A SINGLE recipe id across the whole batch — the core "no cross-regime mixing"
    # invariant. Anything else means two anchors are not mutually comparable.
    assert recipe_ids == {recipe.recipe_id}
    # And it is exactly the canonical scorer's recipe id.
    assert recipe.recipe_id == "canonical_v1"


# --------------------------------------------------------------------------- #
# Property 2b: same recipe_id holds even when some artifacts are BLOCKED        #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 2: single recipe_id across mixed points
@settings(max_examples=150, deadline=None)
@given(
    names=_artifact_batch(),
    preds=_dev_predictions(),
    block_flags=st.data(),
)
def test_recipe_id_uniform_across_scorable_and_blocked(
    names: List[str], preds: pd.DataFrame, block_flags
) -> None:
    """When a batch mixes SCORABLE points (recipe re-runs) and BLOCKED points (no
    re-runnable recipe / recipe unavailable), EVERY point still records the SAME
    canonical ``recipe_id`` — a BLOCKED point records the recipe that WOULD have been
    used, so the batch never mixes regimes (Req 2.1, 2.6)."""
    recipe = ScoringRecipe()
    scorer = _make_scorer(recipe)

    # Randomly decide, per artifact, whether a re-runnable recipe exists.
    available = {
        name: block_flags.draw(st.booleans(), label=f"avail_{name}") for name in names
    }

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if not available[path.name]:
            return None  # → BLOCKED (no fabricated number, contract Rules 1 & 9)
        return RecipeRerun(preds.copy(), recipe.recipe_id, detail=f"rerun@{artifact.digits}")

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(names)

    assert len(points) == len(names)
    # Uniform recipe id regardless of SCORABLE/BLOCKED status.
    assert {p.recipe_id for p in points} == {recipe.recipe_id}
    for p in points:
        assert p.status in (SCORABLE, BLOCKED)
        if p.status == BLOCKED:
            # BLOCKED points never fabricate a local number.
            assert p.local_pair_ap is None and p.local_combined is None
        else:
            assert p.local_pair_ap is not None and p.local_combined is not None


# --------------------------------------------------------------------------- #
# Property 2c: same frame under two anchors' recorded recipes ⇒ same local AP   #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 2: no cross-regime mixing (equal AP)
@settings(max_examples=150, deadline=None)
@given(names=_artifact_batch(), preds=_dev_predictions())
def test_same_frame_under_recorded_recipes_yields_same_ap(
    names: List[str], preds: pd.DataFrame
) -> None:
    """Score the SAME dev-prediction frame under EACH scorable anchor's recorded
    recipe; every anchor's local holdout AP (PairAP) — and its combined score — is
    identical. There is one recipe, so there is no second regime for two anchors to
    diverge across (Req 2.5, 2.6, 3.5).

    The recorded ``recipe_id`` on each ladder point is looked up back to a
    ``ScoringRecipe``, a fresh scorer is built from it, and ``preds`` is scored. This
    is the direct falsification target of "no cross-regime mixing": if two anchors'
    recipes produced different APs for the same frame, the anchors would not be
    mutually comparable."""
    canonical = ScoringRecipe()
    scorer = _make_scorer(canonical)

    def runner(path: Path, artifact: ArtifactLB) -> RecipeRerun:
        return RecipeRerun(preds.copy(), canonical.recipe_id, detail=f"rerun@{artifact.digits}")

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(names)

    # Registry mapping the one canonical recipe id back to its recipe (models the
    # "anchors record a recipe_id that resolves to one recipe" contract).
    registry: Dict[str, ScoringRecipe] = {canonical.recipe_id: canonical}

    pair_aps: List[float] = []
    combineds: List[float] = []
    for pt in points:
        assert pt.status == SCORABLE  # runner always returns the canonical recipe
        recipe = registry[pt.recipe_id]  # KeyError here would be a cross-regime leak
        # A fresh scorer built from THIS anchor's recorded recipe scores the frame.
        anchor_score = _make_scorer(recipe).score_dev_predictions(preds.copy())
        pair_aps.append(anchor_score.pair_ap)
        combineds.append(anchor_score.combined)
        # The anchor's recorded local_holdout_ap (pair_ap) must match re-scoring under
        # its own recorded recipe — the point is not carrying a stale/foreign number.
        assert anchor_score.pair_ap == pt.local_pair_ap
        assert anchor_score.combined == pt.local_combined

    # Equal local holdout AP across ALL anchors' recorded recipes (no cross-regime
    # mixing): every produced AP is the same single value.
    assert len(set(pair_aps)) == 1
    assert len(set(combineds)) == 1


# --------------------------------------------------------------------------- #
# Property 2d: a re-run reporting a FOREIGN recipe_id is refused (BLOCKED)      #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 2: foreign recipe_id ⇒ no mixing
@settings(max_examples=150, deadline=None)
@given(
    names=_artifact_batch(),
    preds=_dev_predictions(),
    foreign=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz_0123456789", min_size=1, max_size=16
    ).filter(lambda s: s != "canonical_v1"),
)
def test_foreign_recipe_id_rerun_is_blocked_not_mixed(
    names: List[str], preds: pd.DataFrame, foreign: str
) -> None:
    """If an artifact's re-run reports a recipe id DIFFERENT from the canonical one,
    the point is refused (BLOCKED) with no fabricated local number — never silently
    scored into the ladder under a mismatched regime (Req 2.6). The point STILL records
    the canonical ``recipe_id`` (the recipe that WOULD have applied), so the batch's
    recorded recipe id stays uniform."""
    canonical = ScoringRecipe()
    scorer = _make_scorer(canonical)

    def runner(path: Path, artifact: ArtifactLB) -> RecipeRerun:
        return RecipeRerun(preds.copy(), foreign, detail=f"rerun@{artifact.digits}")

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(names)

    assert len(points) == len(names)
    assert {p.recipe_id for p in points} == {canonical.recipe_id}
    for p in points:
        assert p.status == BLOCKED
        assert p.local_pair_ap is None and p.local_combined is None
        assert (
            "regime" in p.blocked_reason.lower()
            or "recipe_id" in p.blocked_reason
        )
