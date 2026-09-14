"""Integration smoke test for ladder scoring end-to-end (task 5.5).

design.md "Components and Interfaces" §3 (LadderValidator) + "The load-bearing fact"
(the nine eval artifacts have pair sets DISJOINT from the dev labels, so they CANNOT
be dev-scored directly). Requirements 2.1, 2.5. Governed by the workspace
``reverse-engineering-accountability`` contract (honest BLOCKED reporting, no
fabrication, clean skip when the real competition data is absent).

This is ONE representative end-to-end run over the REAL dev holdout via
:class:`poker_collusion.validation.cv.CVHarness`. It is NOT property-tested (a single
smoke run, same convention as ``test_integration_real_holdout.py`` and
``test_anchor_repro_integrity_real_artifact.py``), and is marked
``@pytest.mark.integration`` so the fast unit/property suite can exclude it.

It exercises BOTH honest ladder outcomes on the SAME real machinery:

1. **The honest BLOCKED split (design §"The load-bearing fact", dossier §37/§40).**
   No recipe registry exists yet, so :meth:`LadderValidator.score_ladder` over the
   real nine ``submission_best_<d>.csv`` artifacts with the honest default (no
   ``recipe_runner``) records ALL nine as BLOCKED — the eval CSVs are disjoint from
   the dev labels and cannot be dev-scored, so no local AP is invented. The test
   asserts exactly this honest behavior: all BLOCKED, ``local_pair_ap ==
   local_combined == None`` for every point, ``real_lb`` parsed from the filename,
   and reports the scorable-vs-BLOCKED split (0 scorable / 9 BLOCKED).

2. **At least one SCORABLE path end-to-end on the REAL dev holdout.** Where a
   re-runnable recipe CAN be constructed against the real dev holdout via
   ``CVHarness`` / ``build_fold_solution``, we build one: a real ``CVHarness`` fold's
   PU-correct dev-holdout solution (its ``pair_id``\\s ARE real dev pairs, NOT eval
   pairs) drives a ``recipe_runner`` that emits a dev-holdout prediction frame over
   those real dev pairs. The :class:`CanonicalScorer` then scores it end-to-end,
   producing a REAL local PairAP / combined — the test asserts the point is SCORABLE
   with finite ``local_pair_ap`` / ``local_combined`` in ``[0, 1]`` and the canonical
   ``recipe_id`` recorded.

If the real competition data is not mounted (dev labels / seats / hands absent), the
test ``pytest.skip()``s with a clear reason — it NEVER fabricates a dev holdout to
manufacture a SCORABLE point (contract Rules 1 & 9).

Requirements: 2.1, 2.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.validation.cv import CVHarness, build_fold_solution

from anchor_repro.artifact_names import ArtifactLB
from anchor_repro.ladder import BLOCKED, SCORABLE, LadderValidator, RecipeRerun
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------- #
# Real-data / real-artifact locations (resolved relative to this file so the   #
# test runs from anywhere), mirroring the sibling integration tests.           #
# --------------------------------------------------------------------------- #
_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_ARTIFACT_DIR = _POKER_ROOT / "outputs" / "poker_collusion"

_DEV_LABELS = _DATA_DIR / "development_labels.csv"
_SEATS_PARQUET = _DATA_DIR / "seats.parquet"
_HANDS_PARQUET = _DATA_DIR / "hands.parquet"
_DEV_EVIDENCE = _DATA_DIR / "development_evidence.csv"

#: The nine known-LB artifacts (design.md load-bearing fact).
_ARTIFACT_GLOB = "submission_best_*.csv"


def _require_real_data() -> tuple[list[Path], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Skip unless the nine real artifacts + real dev labels / seats / hands exist.

    Honesty gate (contract Rules 1 & 9): the SCORABLE end-to-end path re-runs a recipe
    on the REAL dev holdout via ``CVHarness`` (needs dev labels + the seats/hands pool
    map), and the BLOCKED split runs over the REAL nine artifacts. Without the real
    competition data we SKIP with a clear reason rather than fabricate a dev holdout or
    a passing number.
    """
    artifacts = sorted(_ARTIFACT_DIR.glob(_ARTIFACT_GLOB))
    if not artifacts:
        pytest.skip(
            "No real submission_best_<score>.csv artifacts present under "
            f"{_ARTIFACT_DIR}; cannot run the ladder end-to-end smoke test "
            "(skipping rather than fabricating data)."
        )
    missing = [
        str(p) for p in (_DEV_LABELS, _SEATS_PARQUET, _HANDS_PARQUET) if not p.is_file()
    ]
    if missing:
        pytest.skip(
            "Real competition data not mounted for the ladder end-to-end smoke "
            f"test; missing: {missing}"
        )
    labels = pd.read_csv(_DEV_LABELS)
    seats = pd.read_parquet(_SEATS_PARQUET, columns=["hand_id", "player_id"])
    hands = pd.read_parquet(_HANDS_PARQUET, columns=["hand_id", "table_id"])
    return artifacts, labels, seats, hands


def _dev_evidence() -> Optional[pd.DataFrame]:
    """Load the real dev evidence if present (positives get real evidence slots)."""
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


def _real_holdout_predictions(
    harness: CVHarness, labels: pd.DataFrame, evidence: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Build a dev-holdout prediction frame over ONE real ``CVHarness`` fold.

    This is the honest reproduction of "a re-runnable recipe on the dev holdout": we
    take a real fold's PU-correct solution (built by the REUSED ``build_fold_solution``
    from the REAL dev labels, so its ``pair_id``\\s ARE real dev pairs — NOT the eval
    pairs of a ``submission_best_*.csv``), and emit a metric-schema submission over
    exactly those dev pairs. The ``risk_score`` is the ground-truth label (a perfect
    ranking) purely so the end-to-end scorer produces a real, finite PairAP/combined —
    the point of this smoke test is that the SCORABLE path runs end-to-end and yields a
    real number, not that any particular recipe is faithful.
    """
    # Use the first fold that actually has confirmed pairs.
    for fold in harness.folds():
        solution = build_fold_solution(labels, fold.validation_pairs, evidence=evidence)
        if len(solution) == 0:
            continue
        preds = solution[["pair_id"]].copy()
        # risk_score = ground-truth label so the AP is real and finite (perfect order).
        preds["risk_score"] = solution["risk_score"].astype(float).clip(0.0, 1.0)
        preds["predicted_behavior"] = solution["predicted_behavior"].astype(str)
        for col in SUBMISSION_COLUMNS:
            if col not in preds.columns:
                preds[col] = "NO_EVIDENCE"
        return preds[list(SUBMISSION_COLUMNS)]
    pytest.skip(
        "No real CVHarness fold produced a non-empty PU-correct solution; cannot "
        "exercise the SCORABLE path (skipping rather than fabricating a holdout)."
    )
    raise AssertionError("unreachable")  # pragma: no cover


def _scorer(labels: pd.DataFrame, evidence: Optional[pd.DataFrame]) -> CanonicalScorer:
    """Canonical scorer wired to the REAL dev labels / evidence (injected, no re-read)."""
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )


@pytest.mark.integration
def test_ladder_end_to_end_blocked_split_and_one_scorable_real_holdout() -> None:
    """One representative end-to-end ladder run over the REAL dev holdout.

    Part 1 (honest BLOCKED split): scoring the real nine artifacts with the honest
    default (no recipe registry) records ALL nine as BLOCKED with no fabricated local
    AP — exactly the honest expected outcome (design §"The load-bearing fact").

    Part 2 (at least one SCORABLE path end-to-end): a re-runnable recipe constructed
    against the REAL dev holdout via ``CVHarness`` / ``build_fold_solution`` scores a
    real dev-holdout prediction frame through the canonical scorer, producing a REAL
    local PairAP / combined.

    Skips cleanly (no invented number) if the real competition data is absent.
    """
    artifacts, labels, seats, hands = _require_real_data()
    evidence = _dev_evidence()

    # ------------------------------------------------------------------ #
    # Part 1: the honest BLOCKED split over the REAL nine eval artifacts. #
    # ------------------------------------------------------------------ #
    scorer = _scorer(labels, evidence)
    validator_blocked = LadderValidator(scorer)  # no recipe_runner ⇒ honest default
    blocked_points = validator_blocked.score_ladder(artifacts)

    n_blocked = sum(1 for p in blocked_points if p.status == BLOCKED)
    n_scorable = sum(1 for p in blocked_points if p.status == SCORABLE)

    print(
        "\n[ladder e2e] BLOCKED split over the real nine eval artifacts: "
        f"{n_scorable} SCORABLE / {n_blocked} BLOCKED (of {len(blocked_points)}). "
        "Honest expected outcome: all BLOCKED (no re-runnable recipe registry; the "
        "eval CSVs are disjoint from the dev labels and cannot be dev-scored)."
    )

    assert len(blocked_points) == len(artifacts)
    # HONEST EXPECTED OUTCOME (design §"The load-bearing fact", contract Rules 1 & 9):
    # with no recipe registry, EVERY eval artifact is BLOCKED and NO local AP invented.
    assert all(p.status == BLOCKED for p in blocked_points), (
        "with no re-runnable recipe registry every eval artifact must be BLOCKED "
        "(the eval CSVs are disjoint from the dev labels — no dev score is possible)"
    )
    for p in blocked_points:
        assert p.local_pair_ap is None and p.local_combined is None, (
            "a BLOCKED ladder point must never carry a fabricated local AP"
        )
        assert p.blocked_reason, "a BLOCKED point must record an honest reason"
        assert p.recipe_id == scorer.recipe.recipe_id
        # real_lb is the external-judge score parsed from the filename (no distortion).
        assert 0.0 <= p.real_lb < 1.0

    # ------------------------------------------------------------------ #
    # Part 2: at least one SCORABLE path end-to-end on the REAL holdout.  #
    # ------------------------------------------------------------------ #
    harness = CVHarness.from_real_data(labels, seats=seats, hands=hands, n_folds=5)
    dev_predictions = _real_holdout_predictions(harness, labels, evidence)
    recipe_id = scorer.recipe.recipe_id

    def real_recipe_runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        # A genuinely re-runnable recipe for exactly one artifact: it returns DEV-holdout
        # predictions (real dev pairs), NEVER the disjoint eval CSV.
        if artifact.digits == "044519":
            return RecipeRerun(
                dev_predictions=dev_predictions,
                recipe_id=recipe_id,
                detail="synthetic re-runnable recipe over the REAL CVHarness dev holdout",
            )
        return None

    validator_scorable = LadderValidator(scorer, recipe_runner=real_recipe_runner)
    points = validator_scorable.score_ladder(artifacts)

    scorable_pts = [p for p in points if p.status == SCORABLE]
    blocked_pts = [p for p in points if p.status == BLOCKED]

    print(
        f"[ladder e2e] with one re-runnable recipe: {len(scorable_pts)} SCORABLE / "
        f"{len(blocked_pts)} BLOCKED. "
        + (
            f"SCORABLE point local_pair_ap={scorable_pts[0].local_pair_ap:.4f}, "
            f"local_combined={scorable_pts[0].local_combined:.4f} "
            f"(recipe_id={scorable_pts[0].recipe_id})"
            if scorable_pts
            else "no SCORABLE point produced"
        )
    )

    # At least one SCORABLE path exercised end-to-end, producing a REAL local AP.
    assert len(scorable_pts) >= 1, (
        "the re-runnable recipe over the real dev holdout should produce at least one "
        "SCORABLE ladder point end-to-end"
    )
    sp = scorable_pts[0]
    assert sp.local_pair_ap is not None and sp.local_combined is not None
    assert 0.0 <= sp.local_pair_ap <= 1.0, sp.local_pair_ap
    assert 0.0 <= sp.local_combined <= 1.0, sp.local_combined
    assert sp.recipe_id == recipe_id
    assert "recipe_id=" in sp.source

    # The remaining artifacts (no recipe) stay honestly BLOCKED — the scorable-vs-BLOCKED
    # split is reported, never fabricated.
    assert len(blocked_pts) == len(artifacts) - len(scorable_pts)
    for p in blocked_pts:
        assert p.local_pair_ap is None and p.local_combined is None
