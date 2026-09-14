"""§54/§55 integration test: falsify (or confirm) contract Rule 17 on THIS data.

Pre-registered bar: RESEARCH_DOSSIER §54 (BINDING). Governed by the workspace
``reverse-engineering-accountability`` contract (Rules 2, 3, 6, 9, 12, 17).

THE QUESTION (§54). Rule 17 asserts a DRIFT-FAILING feature basis makes the
dev-holdout PairAP an INVERTED / untrustworthy LB predictor. That was imported as law,
never measured here. The drift gate asks a DISTRIBUTIONAL question (dev-vs-eval
separability); the ladder asks a RANK-ORDERING question (does local PairAP order
recipes like the LB?). This test measures whether they coincide.

THE DECISIVE TEST (§54). Build the 3-point ladder at three DISTINCT, EXTERNALLY-
VERIFIED LB levels, scoring EACH point with a DRIFT-FAILING dev-holdout basis:

* floor      real LB 0.44519  (submission_best_044519.csv — floor stack, drift ~0.7621)
* nomannic   real LB 0.56202  (self-earned; own 129-col pair matrix, drift ~0.7594)
* honghanh   real LB 0.64262  (self-earned competitor 064469 recipe)

Then check whether local PairAP orders these three the SAME as the LB
(0.44519 < 0.56202 < 0.64262) and compute ``spearman_corr(local_pair_ap, real_lb)``
at n=3 + its ~95% CI.

This DELIBERATELY scores drift-failing bases ANYWAY (overriding the §52 Rule-17 block)
— §54 authorizes it explicitly as the pre-registered experiment, not a contract
violation.

PRE-REGISTERED DECISION RULE (§54, FIXED — do NOT move the 0.7 Spearman bar, Rule 12):
* Spearman >= 0.7 AND orders all 3 correctly -> Rule 17 FALSIFIED for the ladder's
  purpose (drift AUC should be demoted from hard gate to reported diagnostic).
* Spearman < 0.7 or mis-orders -> Rule 17 CONFIRMED on this data (gate justified).
* borderline at n=3 -> INCONCLUSIVE (need more distinct LB points, not a threshold tweak).

Skips cleanly if the real caches are absent — never fabricates data (Rules 1, 9).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.validation.lb_cv import spearman_corr

from anchor_repro.competitor_recipe import (
    default_competitor_paths,
    rerun_competitor_recipe,
)
from anchor_repro.ladder import (
    OPERATIONAL_BAR,
    SCORABLE,
    LadderPoint,
    LadderValidator,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.nomannic_recipe import (
    default_nomannic_cache_locations,
    rerun_nomannic_recipe,
)
from anchor_repro.recipe_registry import default_cache_locations, rerun_floor_stack
from anchor_repro.scorer import CanonicalScorer

pytestmark = pytest.mark.integration

_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_DEV_LABELS = _DATA_DIR / "development_labels.csv"
_DEV_EVIDENCE = _DATA_DIR / "development_evidence.csv"

# §54 externally-verified LB coordinates (self-earned this session where noted). These
# are the THREE distinct LB levels the ladder is ordered against. They are read as
# EXTERNAL-JUDGE ground truth, never fabricated (contract Rule 1).
_FLOOR_LB = 0.44519
_NOMANNIC_LB = 0.56202
_HONGHANH_LB = 0.64262


def _spearman_ci_t(r: float, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Approx 95% CI for Spearman via t-approx SE=sqrt((1-r^2)/(n-2)); nan for n<3."""
    if n < 3 or abs(r) >= 1.0:
        if abs(r) >= 1.0 and n >= 3:
            return (r, r, 0.0)
        return (float("nan"), float("nan"), float("nan"))
    se = math.sqrt((1.0 - r * r) / (n - 2))
    return (max(-1.0, r - z * se), min(1.0, r + z * se), se)


def _dev_evidence() -> Optional[pd.DataFrame]:
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


def _require_inputs():
    """Skip unless the floor + competitor(OOF cache) + nomannic caches + labels exist."""
    if not _DEV_LABELS.is_file():
        pytest.skip(f"Real dev labels absent ({_DEV_LABELS}); skipping.")

    floor_caches = default_cache_locations(_POKER_ROOT)
    if floor_caches.missing():
        pytest.skip(
            "Floor caches absent; missing: "
            + ", ".join(str(p) for p in floor_caches.missing())
        )

    competitor_paths = default_competitor_paths(_POKER_ROOT)
    if competitor_paths.missing():
        pytest.skip(
            "Competitor re-run inputs / warm OOF cache absent; missing: "
            + ", ".join(str(p) for p in competitor_paths.missing())
        )

    nomannic_caches = default_nomannic_cache_locations(_POKER_ROOT)
    if nomannic_caches.missing():
        pytest.skip(
            "nomannic (prepared_v2) caches absent; missing: "
            + ", ".join(str(p) for p in nomannic_caches.missing())
        )
    return floor_caches, competitor_paths, nomannic_caches


@pytest.mark.integration
def test_rule17_falsify_three_point_drift_failing_ladder() -> None:
    """Score 3 drift-failing bases at 3 distinct LB levels; measure rank-tracking.

    Computes all THREE dev-holdout PairAPs (floor reuse, honghanh reuse from OOF
    cache, nomannic NEW OOF), assembles the 3-point ladder under ONE canonical recipe
    id, runs rank_tracking, and records the §54 Rule-17 verdict. Asserts the three
    PairAPs are finite in [0,1] and a verdict is recorded — never asserts a particular
    direction (the MEASURED result is the deliverable, §55).
    """
    floor_caches, competitor_paths, nomannic_caches = _require_inputs()
    labels = pd.read_csv(_DEV_LABELS)
    evidence = _dev_evidence()

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )
    recipe_id = scorer.recipe.recipe_id

    # (1) THREE dev-holdout PairAPs, each from a DRIFT-FAILING basis (scored ANYWAY
    #     per §54). Report each PairAP value AND the confirmed-pair count.
    floor = rerun_floor_stack(floor_caches, labels, evidence=evidence)
    floor_score = scorer.score_dev_predictions(floor.dev_predictions)

    honghanh = rerun_competitor_recipe(competitor_paths, labels, evidence=evidence)
    honghanh_score = scorer.score_dev_predictions(honghanh.dev_predictions)

    nomannic = rerun_nomannic_recipe(nomannic_caches, labels, evidence=evidence)
    nomannic_score = scorer.score_dev_predictions(nomannic.dev_predictions)

    for s in (floor_score, honghanh_score, nomannic_score):
        assert s.pair_ap is not None and math.isfinite(s.pair_ap)
        assert 0.0 <= s.pair_ap <= 1.0

    # (2) Assemble the 3-point ladder at the §54 externally-verified LB levels, all
    #     under ONE canonical recipe id (mutually comparable — Req 2.6). We build the
    #     LadderPoints directly (the honghanh self-earned 0.64262 and nomannic 0.56202
    #     coordinates have no on-disk artifact filename; the LB is EXTERNAL ground
    #     truth read here, never parsed from a fabricated file — contract Rule 1).
    points: List[LadderPoint] = [
        LadderPoint(
            config_id="ladder_044519_floor",
            real_lb=_FLOOR_LB,
            local_pair_ap=float(floor_score.pair_ap),
            local_combined=float(floor_score.combined),
            recipe_id=recipe_id,
            source="floor stack (v5+MFg+DIRc); drift ~0.7621 (DRIFT-FAILING, scored per §54)",
        ),
        LadderPoint(
            config_id="ladder_056202_nomannic",
            real_lb=_NOMANNIC_LB,
            local_pair_ap=float(nomannic_score.pair_ap),
            local_combined=float(nomannic_score.combined),
            recipe_id=recipe_id,
            source="nomannic own 129-col pair matrix; drift ~0.7594 (DRIFT-FAILING, scored per §54)",
        ),
        LadderPoint(
            config_id="ladder_064262_honghanh",
            real_lb=_HONGHANH_LB,
            local_pair_ap=float(honghanh_score.pair_ap),
            local_combined=float(honghanh_score.combined),
            recipe_id=recipe_id,
            source="honghanh competitor 064469 recipe (self-earned LB 0.64262)",
        ),
    ]

    validator = LadderValidator(scorer)
    verdict = validator.rank_tracking(points)

    scorable = [p for p in points if p.status == SCORABLE]
    assert len(scorable) == 3
    assert verdict.n_points == 3
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"
    assert verdict.verdict in {"TRUSTED", "WEAK", "NULL"}

    # (3) PRIMARY (PairAP) Spearman + ~95% CI; SECONDARY (combined) Spearman.
    real_lb = [p.real_lb for p in points]
    pair_ap = [float(p.local_pair_ap) for p in points]
    combined = [float(p.local_combined) for p in points]
    r_primary = spearman_corr(pair_ap, real_lb)
    r_secondary = spearman_corr(combined, real_lb)
    lo, hi, se = _spearman_ci_t(r_primary, 3)

    # The ACTUAL local ordering (auditable, §54 guardrail): the LB order is fixed
    # ascending [floor, nomannic, honghanh]; check whether PairAP orders them the same.
    lb_order = [_FLOOR_LB, _NOMANNIC_LB, _HONGHANH_LB]
    local_order_pairs = sorted(zip(lb_order, pair_ap))  # sort by LB ascending
    pair_ap_in_lb_order = [ap for _lb, ap in local_order_pairs]
    orders_correctly = pair_ap_in_lb_order == sorted(pair_ap_in_lb_order)

    # (4) Apply the §54 PRE-REGISTERED decision rule EXACTLY (bars NOT moved, Rule 12).
    if r_primary >= OPERATIONAL_BAR and orders_correctly:
        rule17 = "FALSIFIED"
    elif r_primary < OPERATIONAL_BAR or not orders_correctly:
        rule17 = "CONFIRMED"
    else:  # unreachable given the two branches above; kept for completeness
        rule17 = "INCONCLUSIVE"
    # n=3 borderline note: a positive-but-<0.7 Spearman is INCONCLUSIVE in strength
    # even when the CONFIRMED branch fires — reported in §55, not forced either way.

    print(
        "\n[§54 Rule-17 falsify] MEASURED dev-holdout PairAPs (DRIFT-FAILING bases, "
        "scored ANYWAY per §54):"
        f"\n  floor    real_lb={_FLOOR_LB:.5f}  PairAP={floor_score.pair_ap:.4f}  "
        f"(confirmed n={floor.n_holdout_confirmed})"
        f"\n  nomannic real_lb={_NOMANNIC_LB:.5f}  PairAP={nomannic_score.pair_ap:.4f}  "
        f"(confirmed n={nomannic.n_confirmed})"
        f"\n  honghanh real_lb={_HONGHANH_LB:.5f}  PairAP={honghanh_score.pair_ap:.4f}  "
        f"(confirmed n={honghanh.n_confirmed})"
        f"\n  local PairAP order (by ascending LB) = {[round(a, 4) for a in pair_ap_in_lb_order]}"
        f"; orders_correctly={orders_correctly}"
        f"\n  PRIMARY(PairAP) spearman={r_primary:.4f}  ~95% CI~[{lo:.4f},{hi:.4f}] (n=3)"
        f"\n  SECONDARY(combined) spearman={r_secondary:.4f}"
        f"\n  §54 verdict: Rule 17 {rule17} (bar {OPERATIONAL_BAR}, NOT moved — Rule 12)"
    )

    # Assert the experiment RAN and a verdict was recorded — not a direction.
    assert rule17 in {"FALSIFIED", "CONFIRMED", "INCONCLUSIVE"}
    assert math.isfinite(r_primary) and math.isfinite(r_secondary)
