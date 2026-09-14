"""Integration test for the ladder recipe-runner adapter (task 5.8).

design.md "Components and Interfaces" §3 (LadderValidator / the injected
``recipe_runner``), tasks.md 5.8. Requirements 2.1, 2.5, 2.6. Governed by the
workspace ``reverse-engineering-accountability`` contract (debug the null; recover
every point honestly; BLOCK only what is genuinely unreconstructable, naming why; a
re-run dev-holdout AP is a MEASURED local number, never an LB claim).

This test closes the task-5.5 execution-gap null (contract Rules 3 & 11): task 5.5
measured 0/9 scorable ONLY because no ``recipe_runner`` was injected. Here we inject
the real :func:`anchor_repro.recipe_registry.build_ladder_recipe_runner`, which
reconstructs the ``cand_exp4c_directional`` floor stack for ``submission_best_044519``
from the on-disk caches (foundation v5 + MFg + DIRc, one XGB, seed-7 60/40
table-disjoint) and re-runs it on the REAL dev holdout, and honestly BLOCKS the other
eight artifacts (each with a specific named missing recipe input).

Asserts (per task 5.8):

1. ≥1 SCORABLE point (044519) with a FINITE canonical dev-holdout PairAP in ``[0,1]``.
2. The re-run consistency check: the loopv-style weighted-all-holdout AP (the number
   directly comparable to the recorded floor) lands in the recorded 0.3833–0.3839
   band. (The canonical confirmed-only PairAP is a DIFFERENT AP definition on a
   DIFFERENT row set, so it is NOT expected to equal the recorded number — that
   deviation is surfaced honestly, contract Rule 9, not asserted equal.)
3. Every BLOCKED artifact names its specific missing recipe input.

If the real caches / competition data are absent, the test ``pytest.skip()``s with a
clear reason — it never fabricates data to manufacture a SCORABLE point.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import math

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig

from anchor_repro.ladder import BLOCKED, SCORABLE, LadderValidator
from anchor_repro.models import ScoringRecipe
from anchor_repro.recipe_registry import (
    FLOOR_DIGITS,
    FLOOR_HOLDOUT_AP_HIGH,
    FLOOR_HOLDOUT_AP_LOW,
    FloorRerunResult,
    build_ladder_recipe_runner,
    default_cache_locations,
    reconstruction_report,
)
from anchor_repro.scorer import CanonicalScorer

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------- #
# Real-data / real-artifact locations, resolved relative to this file.         #
# --------------------------------------------------------------------------- #
_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_ARTIFACT_DIR = _POKER_ROOT / "outputs" / "poker_collusion"

_DEV_LABELS = _DATA_DIR / "development_labels.csv"
_DEV_EVIDENCE = _DATA_DIR / "development_evidence.csv"
_ARTIFACT_GLOB = "submission_best_*.csv"

#: A tolerance band around the recorded floor AP for the re-run consistency check.
#: The recorded number is 0.3833 (§34 whole-stack) / 0.3839 (§37 grounding note); we
#: allow a small tolerance for library/version drift in the XGB fit rather than
#: demanding a bit-exact match (contract Rule 9 — a re-run consistency check, not a
#: pass/fail LB gate).
_FLOOR_AP_TOL: float = 0.02


def _require_real_caches():
    """Skip unless the real caches + dev labels the floor recipe needs are present."""
    artifacts = sorted(_ARTIFACT_DIR.glob(_ARTIFACT_GLOB))
    if not artifacts:
        pytest.skip(
            f"No real submission_best_<score>.csv artifacts under {_ARTIFACT_DIR}; "
            "skipping rather than fabricating the ladder."
        )
    if not _DEV_LABELS.is_file():
        pytest.skip(f"Real dev labels absent ({_DEV_LABELS}); skipping.")
    caches = default_cache_locations(_POKER_ROOT)
    missing = caches.missing()
    if missing:
        pytest.skip(
            "Real feature caches for the floor recipe not mounted; missing: "
            + ", ".join(str(p) for p in missing)
        )
    return artifacts, caches


def _dev_evidence() -> Optional[pd.DataFrame]:
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


def test_reconstruction_report_names_blocked_missing_inputs() -> None:
    """The per-artifact reconstruction report: only the floor is reconstructable.

    A pure unit check (no caches needed): the floor (044519) is reconstructable with a
    recipe id; every other trajectory artifact is honestly non-reconstructable with a
    specific named missing input (contract Rules 3 & 9). This is what lets a BLOCKED
    ladder point cite exactly what is unavailable rather than a blanket give-up.
    """
    report = reconstruction_report()
    assert report[FLOOR_DIGITS].reconstructable is True
    assert report[FLOOR_DIGITS].recipe_id

    blocked = {d: s for d, s in report.items() if not s.reconstructable}
    assert len(blocked) == 8, "exactly the eight non-floor trajectory artifacts BLOCKED"
    for digits, status in blocked.items():
        assert status.missing_input, f"BLOCKED artifact {digits} must name its missing input"
        # The missing input must be specific: it names the artifact and why it is unreconstructable.
        assert digits in status.missing_input
        assert "fabricat" in status.missing_input.lower()


@pytest.mark.integration
def test_floor_recipe_runner_recovers_one_scorable_point_and_blocks_the_rest() -> None:
    """Re-run the floor recipe → ≥1 SCORABLE point; the other eight honestly BLOCKED.

    Closes the task-5.5 execution-gap null (contract Rules 3 & 11): with the real
    recipe runner injected, ``submission_best_044519.csv`` becomes SCORABLE with a
    finite canonical dev-holdout PairAP, the loopv-style weighted AP lands in the
    recorded floor band (re-run consistency), and the other eight artifacts stay
    honestly BLOCKED with specific named missing inputs.
    """
    artifacts, caches = _require_real_caches()
    labels = pd.read_csv(_DEV_LABELS)
    evidence = _dev_evidence()

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )
    recipe_id = scorer.recipe.recipe_id

    captured: List[FloorRerunResult] = []
    runner = build_ladder_recipe_runner(
        caches,
        labels,
        canonical_recipe_id=recipe_id,
        evidence=evidence,
        on_result=captured.append,
    )

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(artifacts)

    scorable = [p for p in points if p.status == SCORABLE]
    blocked = [p for p in points if p.status == BLOCKED]

    # ---- Report the scorable-vs-BLOCKED split (this determines whether the ladder
    # gate has n>=2 real points — here n=1 SCORABLE, honestly BLOCKED rest). ----
    assert len(captured) == 1, "the floor re-run should fit exactly once"
    rerun = captured[0]
    print(
        "\n[recipe registry] ladder split: "
        f"{len(scorable)} SCORABLE / {len(blocked)} BLOCKED (of {len(points)}). "
        f"Floor re-run: {rerun.n_features} feats, "
        f"{rerun.n_holdout_confirmed} confirmed holdout pairs / "
        f"{rerun.n_holdout_all} all-holdout pairs; "
        f"loopv weighted-all AP={rerun.loopv_weighted_ap:.4f} "
        f"(recorded floor {FLOOR_HOLDOUT_AP_LOW}-{FLOOR_HOLDOUT_AP_HIGH}); "
        f"canonical confirmed-only PairAP={scorable[0].local_pair_ap:.4f} "
        "(DIFFERENT AP definition/row set — not expected to equal the recorded number)."
    )

    # (1) At least one SCORABLE point (the floor 044519) with a finite PairAP in [0,1].
    assert len(scorable) >= 1, "the floor recipe re-run must recover >=1 SCORABLE point"
    floor_point = next((p for p in scorable if p.config_id.endswith(FLOOR_DIGITS)), None)
    assert floor_point is not None, "the 044519 floor must be the SCORABLE point"
    assert floor_point.local_pair_ap is not None
    assert math.isfinite(floor_point.local_pair_ap)
    assert 0.0 <= floor_point.local_pair_ap <= 1.0
    assert floor_point.local_combined is not None
    assert math.isfinite(floor_point.local_combined)
    assert floor_point.recipe_id == recipe_id
    assert "MEASURED local re-run" in floor_point.source  # labelled, not an LB claim

    # (2) Re-run consistency: the loopv-style weighted-all AP (the number comparable to
    # the recorded floor) lands in the recorded 0.3833-0.3839 band (tolerance for XGB
    # version drift). The canonical PairAP is a DIFFERENT definition and is NOT asserted
    # equal — its deviation is surfaced honestly (contract Rule 9), never forced.
    assert FLOOR_HOLDOUT_AP_LOW - _FLOOR_AP_TOL <= rerun.loopv_weighted_ap <= FLOOR_HOLDOUT_AP_HIGH + _FLOOR_AP_TOL, (
        f"re-run loopv weighted AP {rerun.loopv_weighted_ap:.4f} is outside the recorded "
        f"floor band [{FLOOR_HOLDOUT_AP_LOW}, {FLOOR_HOLDOUT_AP_HIGH}] (+/-{_FLOOR_AP_TOL}) "
        "— a re-run inconsistency to investigate, not silently accept"
    )

    # (3) Every BLOCKED artifact names its specific missing recipe input (contract Rule 9).
    assert len(blocked) == len(artifacts) - len(scorable)
    for p in blocked:
        assert p.local_pair_ap is None and p.local_combined is None
        assert p.blocked_reason, "a BLOCKED point must record an honest reason"
        assert p.recipe_id == recipe_id
