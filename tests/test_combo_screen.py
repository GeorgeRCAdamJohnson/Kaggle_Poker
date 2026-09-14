"""§60/§61 integration test: screen floor-guaranteed rank blends of VERIFIED levers.

Pre-registered bar: RESEARCH_DOSSIER §60 (BINDING). Trusted-scorer scope: §57.
Governed by the workspace ``reverse-engineering-accountability`` contract
(Rules 1, 2, 3, 6, 9, 12, 15).

WHAT THIS TESTS
---------------
Reuses (contract Rule 6) the three EXISTING dev-holdout runners — floor
(``rerun_floor_stack``), honghanh (``rerun_competitor_recipe``, warm OOF cache),
nomannic (``rerun_nomannic_recipe``, OOF cache) — to produce the three verified
prediction frames. Then it runs :func:`anchor_repro.combo_screen.screen_combos`:

1. Computes the COMMON confirmed-pair intersection so every combo is scored on ONE
   common set.
2. Sweeps ``w`` over {0.0, ..., 1.0} for each of the three blends (honghanh↔floor,
   nomannic↔floor, {honghanh+nomannic avg}↔floor), scoring canonical confirmed-only
   PairAP at each w via ``CanonicalScorer.score_dev_predictions``.
3. Asserts the FLOOR-GUARANTEE holds at w=0 (Rule 15) and a WINNER-or-NULL verdict is
   produced (§60 decision rule). Never asserts a particular direction — the MEASURED
   result is the deliverable (Rule 3).

When a WINNER is found, it builds the full 112,540-row eval submission to the NEW
``outputs/poker_collusion/combo_screen/submission.csv`` (NEVER overwriting the live
submission) and validates it.

Skips cleanly if any real cache is absent — never fabricates data (Rules 1, 9).
DOES NOT submit to Kaggle.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig

from anchor_repro.combo_screen import (
    DEFAULT_W_GRID,
    build_combo_eval_submission,
    screen_combos,
    validate_eval_submission,
)
from anchor_repro.competitor_recipe import (
    default_competitor_paths,
    rerun_competitor_recipe,
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
_EVAL_PAIRS = _DATA_DIR / "evaluation_pairs.csv"

_OUT_DIR = _POKER_ROOT / "outputs" / "poker_collusion" / "combo_screen"
_FLOOR_EVAL = _POKER_ROOT / "outputs" / "poker_collusion" / "submission_best_044519.csv"
_HONGHANH_EVAL = (
    _POKER_ROOT / "outputs" / "poker_collusion" / "repro_064469" / "submission.csv"
)
_NOMANNIC_EVAL = (
    _POKER_ROOT / "outputs" / "poker_collusion" / "repro_nomannic" / "submission.csv"
)

_FLOOR_LB = 0.44519
_NOMANNIC_LB = 0.56202
_HONGHANH_LB = 0.64262


def _dev_evidence() -> Optional[pd.DataFrame]:
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


def _require_inputs():
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
            "Competitor OOF cache absent; missing: "
            + ", ".join(str(p) for p in competitor_paths.missing())
        )
    nomannic_caches = default_nomannic_cache_locations(_POKER_ROOT)
    if nomannic_caches.missing():
        pytest.skip(
            "nomannic caches absent; missing: "
            + ", ".join(str(p) for p in nomannic_caches.missing())
        )
    for p in (_FLOOR_EVAL, _HONGHANH_EVAL, _NOMANNIC_EVAL, _EVAL_PAIRS):
        if not p.is_file():
            pytest.skip(f"Eval artifact absent ({p}); skipping.")
    return floor_caches, competitor_paths, nomannic_caches


@pytest.mark.integration
def test_combo_screen_floor_guaranteed_winner_or_null() -> None:
    """Screen the three floor-guaranteed blends; assert floor-guarantee + a verdict."""
    floor_caches, competitor_paths, nomannic_caches = _require_inputs()
    labels = pd.read_csv(_DEV_LABELS)
    evidence = _dev_evidence()

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )

    # (1) Reuse the three verified runners (contract Rule 6).
    floor = rerun_floor_stack(floor_caches, labels, evidence=evidence)
    honghanh = rerun_competitor_recipe(competitor_paths, labels, evidence=evidence)
    nomannic = rerun_nomannic_recipe(nomannic_caches, labels, evidence=evidence)

    # (2) Screen all three floor-guaranteed blends on the common confirmed set.
    result = screen_combos(
        scorer,
        floor.dev_predictions,
        honghanh.dev_predictions,
        nomannic.dev_predictions,
        w_grid=DEFAULT_W_GRID,
    )

    # --- FLOOR-GUARANTEE (Rule 15): every blend's best PairAP >= floor-alone at w=0.
    for s in result.sweeps:
        assert 0.0 in s.w_grid, "w grid must include 0.0 (the floor guarantee)"
        assert s.best_pair_ap >= s.floor_alone_pair_ap - 1e-12, (
            f"{s.blend_id}: best PairAP {s.best_pair_ap} regressed below the floor-alone "
            f"{s.floor_alone_pair_ap} — the floor guarantee is broken"
        )
        for ap in s.pair_ap:
            assert math.isfinite(ap) and 0.0 <= ap <= 1.0

    assert result.verdict in {"WINNER", "NULL"}
    assert result.n_common > 0

    # --- Audit print: the full w-sweep per blend + the verdict.
    print(
        f"\n[§60 combo screen] common confirmed pairs = {result.n_common}"
        f"\n  floor-alone PairAP on common set = {result.floor_alone_pair_ap:.6f}"
    )
    for s in result.sweeps:
        rows = "  ".join(f"w={w:.1f}:{ap:.6f}" for w, ap in s.sweep_rows())
        print(f"  [{s.blend_id}] {rows}")
        print(
            f"    best_w={s.best_w:.1f}  best_PairAP={s.best_pair_ap:.6f}  "
            f"delta_over_floor={s.best_pair_ap - result.floor_alone_pair_ap:+.6f}"
        )
    if result.verdict == "WINNER":
        print(
            f"  WINNER: {result.winner_blend_id} at w={result.winner_w:.1f}  "
            f"PairAP={result.winner_pair_ap:.6f}  "
            f"delta_over_floor={result.delta_over_floor:+.6f}"
        )
    else:
        print("  NULL: no blend beats the floor beyond margin — recommend NO submission.")

    # (3) On a WINNER, build + validate the full eval submission (NEW path only).
    if result.verdict == "WINNER":
        floor_eval = pd.read_csv(_FLOOR_EVAL)
        lever_eval_path = {
            "honghanh": _HONGHANH_EVAL,
            "nomannic": _NOMANNIC_EVAL,
        }.get(result.winner_blend_id)
        eval_pairs = pd.read_csv(_EVAL_PAIRS)

        if lever_eval_path is not None:
            lever_eval = pd.read_csv(lever_eval_path)
            out = build_combo_eval_submission(
                floor_eval, lever_eval, result.winner_w, _OUT_DIR / "submission.csv"
            )
        else:
            # hh+nm_avg winner: blend floor against the rank-avg of both levers' eval.
            hh = pd.read_csv(_HONGHANH_EVAL)
            nm = pd.read_csv(_NOMANNIC_EVAL)
            hh_by = dict(zip(hh["pair_id"].astype(str), hh["risk_score"].astype(float)))
            nm_by = dict(zip(nm["pair_id"].astype(str), nm["risk_score"].astype(float)))
            avg_rank = pd.DataFrame({"pair_id": floor_eval["pair_id"].astype(str)})
            avg_rank["hh"] = avg_rank["pair_id"].map(hh_by)
            avg_rank["nm"] = avg_rank["pair_id"].map(nm_by)
            avg_rank["risk_score"] = 0.5 * (
                avg_rank["hh"].rank(method="average")
                + avg_rank["nm"].rank(method="average")
            )
            out = build_combo_eval_submission(
                floor_eval,
                avg_rank[["pair_id", "risk_score"]],
                result.winner_w,
                _OUT_DIR / "submission.csv",
            )

        checks = validate_eval_submission(out, eval_pairs)
        print(f"  eval submission validation: {checks}")
        assert checks["n_rows_ok"]
        assert checks["schema_ok"]
        assert checks["risk_in_unit"]
        assert checks["pairs_match_eval"]
