"""Integration test: the 2-point ladder is scorable end-to-end (§47 follow-up).

design.md "Components and Interfaces" §3 (LadderValidator / the injected
``recipe_runner``); Requirements 2.1, 2.5, 2.6, 2.3. Governed by the workspace
``reverse-engineering-accountability`` contract.

This test closes the §42 OPEN follow-up (reach ladder gate n=2 so a Spearman is
computable at all). It injects the two-point ladder runner
(:func:`anchor_repro.competitor_recipe.build_two_point_ladder_runner`), which
reconstructs the ``cand_exp4c_directional`` floor stack for 044519 AND re-runs the
0.64469 competitor recipe on the dev holdout, and asserts the 2-point ladder scores
end-to-end:

1. exactly TWO SCORABLE points (044519 + 064469), each with a FINITE PairAP in [0,1];
2. BOTH rank-tracking metrics (primary PairAP + secondary combined) are computed at
   ``n_points == 2`` (Spearman is defined for n>=2) and a verdict is recorded;
3. the competitor point's local PairAP sits ABOVE the floor point's (matching the
   real LB order 0.64469 > 0.44519) — the single directional bit n=2 carries.

Honesty (contract Rules 9, 12): a +1 Spearman at n=2 is essentially uninformative
(one bit), so this test asserts the gate is COMPUTABLE and the points are recovered
— it does NOT assert the scorer is a trustworthy judge (n=2 cannot establish that).

If the real caches / competition data are absent, the test ``pytest.skip()``s with a
clear reason — it never fabricates data to manufacture a SCORABLE point. The
competitor OOF cache is REUSED when present (no refit); only the floor stack refits.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig

from anchor_repro.competitor_recipe import (
    COMPETITOR_DIGITS,
    CompetitorRerunResult,
    build_two_point_ladder_runner,
    default_competitor_paths,
)
from anchor_repro.ladder import SCORABLE, LadderValidator
from anchor_repro.models import ScoringRecipe
from anchor_repro.recipe_registry import (
    FLOOR_DIGITS,
    FloorRerunResult,
    default_cache_locations,
)
from anchor_repro.scorer import CanonicalScorer

pytestmark = pytest.mark.integration

_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_DEV_LABELS = _DATA_DIR / "development_labels.csv"
_DEV_EVIDENCE = _DATA_DIR / "development_evidence.csv"
_ARTIFACT_DIR = _POKER_ROOT / "outputs" / "poker_collusion"


def _require_real_inputs():
    """Skip unless the floor caches + competitor cache/data + dev labels are present."""
    if not _DEV_LABELS.is_file():
        pytest.skip(f"Real dev labels absent ({_DEV_LABELS}); skipping.")

    caches = default_cache_locations(_POKER_ROOT)
    floor_missing = caches.missing()
    if floor_missing:
        pytest.skip(
            "Floor-recipe feature caches not mounted; missing: "
            + ", ".join(str(p) for p in floor_missing)
        )

    competitor_paths = default_competitor_paths(_POKER_ROOT)
    comp_missing = competitor_paths.missing()
    if comp_missing:
        pytest.skip(
            "Competitor re-run inputs (notebook / data / warm prepared_pu24k cache) "
            "absent; missing: " + ", ".join(str(p) for p in comp_missing)
        )
    return caches, competitor_paths


def _dev_evidence() -> Optional[pd.DataFrame]:
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


@pytest.mark.integration
def test_two_point_ladder_is_scorable_end_to_end() -> None:
    """044519 + 064469 → n=2 ladder; both rank-tracking metrics computed + recorded.

    Closes the §42 n=1 null: with the two-point runner injected, the ladder reaches
    two SCORABLE points and ``rank_tracking`` computes a Spearman for BOTH metrics at
    n=2. The competitor OOF cache is reused if present (no refit); the floor refits.
    """
    caches, competitor_paths = _require_real_inputs()
    labels = pd.read_csv(_DEV_LABELS)
    evidence = _dev_evidence()

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )
    recipe_id = scorer.recipe.recipe_id

    floor_seen: List[FloorRerunResult] = []
    comp_seen: List[CompetitorRerunResult] = []
    runner = build_two_point_ladder_runner(
        caches,
        competitor_paths,
        labels,
        canonical_recipe_id=recipe_id,
        evidence=evidence,
        on_floor_result=floor_seen.append,
        on_competitor_result=comp_seen.append,
    )

    # The floor artifact exists on disk; the competitor artifact is a VIRTUAL name
    # carrying only real_lb=0.64469 via the filename parser (its EVAL submission is
    # disjoint from dev labels and is NEVER dev-scored — the runner reconstructs the
    # dev-holdout OOF from the warm cache instead).
    floor_artifact = _ARTIFACT_DIR / f"submission_best_{FLOOR_DIGITS}.csv"
    competitor_artifact = _ARTIFACT_DIR / f"submission_best_{COMPETITOR_DIGITS}.csv"
    if not floor_artifact.is_file():
        pytest.skip(f"Floor artifact absent ({floor_artifact}); skipping.")

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder([floor_artifact, competitor_artifact])
    verdict = validator.rank_tracking(points)

    # (1) Exactly two SCORABLE points, each with a finite PairAP/combined in [0,1].
    scorable = [p for p in points if p.status == SCORABLE]
    assert len(scorable) == 2, "both 044519 and 064469 must be SCORABLE (ladder n=2)"
    by_digits = {p.config_id.split("_")[-1]: p for p in scorable}
    assert set(by_digits) == {FLOOR_DIGITS, COMPETITOR_DIGITS}
    for p in scorable:
        assert p.local_pair_ap is not None and math.isfinite(p.local_pair_ap)
        assert 0.0 <= p.local_pair_ap <= 1.0
        assert p.local_combined is not None and math.isfinite(p.local_combined)
        assert p.recipe_id == recipe_id  # mutually comparable (Req 2.6)
        assert "MEASURED local re-run" in p.source  # labelled, never an LB claim

    # (2) Both rank-tracking metrics are COMPUTED at n=2 and a verdict is recorded.
    assert verdict.n_points == 2, "Spearman is now computable (n>=2)"
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"
    assert math.isfinite(verdict.primary.spearman)
    assert math.isfinite(verdict.secondary.spearman)
    assert verdict.verdict in {"TRUSTED", "WEAK", "NULL"}
    # The overall verdict mirrors the PRIMARY (PairAP) result (Req 2.3, §40).
    assert verdict.passed == verdict.primary.passed
    assert verdict.verdict == verdict.primary.verdict

    # (3) The single directional bit n=2 carries: the higher-LB competitor point
    # scored higher locally than the floor (matching real LB 0.64469 > 0.44519).
    floor_pt = by_digits[FLOOR_DIGITS]
    comp_pt = by_digits[COMPETITOR_DIGITS]
    assert comp_pt.real_lb > floor_pt.real_lb
    assert comp_pt.local_pair_ap > floor_pt.local_pair_ap, (
        "competitor local PairAP should sit above the floor's, matching LB order — "
        "the one informative bit at n=2"
    )
    # With a perfectly consistent 2-point order, PairAP Spearman is +1 (the ONLY value
    # a 2-point positive-order correlation can take). This is asserted as the computed
    # number, NOT as evidence of a trustworthy judge (n=2 is uninformative — §47).
    assert verdict.primary.spearman == pytest.approx(1.0)

    print(
        "\n[two-point ladder] n=2 gate: "
        f"044519 PairAP={floor_pt.local_pair_ap:.4f} / "
        f"064469 PairAP={comp_pt.local_pair_ap:.4f}; "
        f"PRIMARY spearman={verdict.primary.spearman:.4f} ({verdict.primary.verdict}), "
        f"SECONDARY spearman={verdict.secondary.spearman:.4f} ({verdict.secondary.verdict}). "
        "NOTE: +1 at n=2 is one bit, NOT a trustworthy judge (contract Rules 9, 12)."
    )
    if comp_seen:
        print(
            f"[two-point ladder] competitor combiner={comp_seen[0].selected_combiner} "
            f"PU-stress OOF AP={comp_seen[0].pu_stress_ap:.4f} "
            f"(reused_cache={comp_seen[0].reused_cache})"
        )
