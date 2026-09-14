"""Ladder validation data models + ``LadderValidator.score_ladder`` (task 5.1).

design.md "Components and Interfaces" §3 (LadderValidator) and "Data Models"
(LadderPoint / MetricRankTracking / RankTrackingVerdict). Requirements 2.1, 2.5, 2.6.

This module defines the three frozen data models the external-judge rank-tracking gate
is built on, plus :meth:`LadderValidator.score_ladder`, which turns the nine on-disk
``submission_best_<d>.csv`` artifacts into :class:`LadderPoint`\\s carrying BOTH the
PRIMARY PairAP and the SECONDARY combined local scores per artifact.

The load-bearing honest constraint (design.md "The load-bearing fact", contract
Rules 1 & 9, reinforced by dossier §40/§41): the nine eval artifacts have ``pair_id``
sets DISJOINT from the dev labels, so an eval artifact CSV **cannot be dev-scored
directly** — there is nothing to score. A ladder point's local AP is obtained by
RE-RUNNING the recipe that produced the artifact on the dev holdout (the PU-stress
table-disjoint split), then scoring those dev-pair predictions with the official
metric via :class:`anchor_repro.scorer.CanonicalScorer` — exactly how the recorded
floor ``holdout_ap = 0.3839/0.3833`` (dossier §37) was produced.

Therefore :meth:`score_ladder`:

* NEVER dev-scores an eval CSV and pretends the result is a dev AP.
* NEVER invents a ``local_pair_ap`` / ``local_combined`` for an artifact whose
  re-runnable recipe is unavailable — that artifact is recorded as **BLOCKED** with
  ``local_pair_ap = local_combined = None`` and a reason, reported honestly (Req 2.1,
  contract Rule 9). A BLOCKED point is a first-class, honest null, not a hidden gap.

The re-runnable recipe for an artifact is provided by an injected ``recipe_runner``
callback: given the parsed artifact it returns the dev-holdout prediction frame that
recipe emits (which the :class:`CanonicalScorer` then scores), or ``None`` when no
re-runnable recipe exists for that artifact — in which case the point is BLOCKED. No
recipe registry is built here (that is downstream work); this task defines the models
and the honest scoring/BLOCKED bookkeeping only. The Spearman rank-tracking gate is
task 5.2 (:meth:`LadderValidator.rank_tracking`, NOT implemented here beyond the data
models it consumes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from anchor_repro.artifact_names import ArtifactLB, parse_artifact_lb
from anchor_repro.models import CanonicalScore, ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from poker_collusion.validation.lb_cv import spearman_corr

__all__ = [
    "SCORABLE",
    "BLOCKED",
    "CONTRACT_MINIMUM_BAR",
    "OPERATIONAL_BAR",
    "TRUSTED",
    "WEAK",
    "NULL",
    "N9_CI_NOTE",
    "LadderPoint",
    "MetricRankTracking",
    "RankTrackingVerdict",
    "RecipeRerun",
    "LadderValidator",
]

# --------------------------------------------------------------------------- #
# Pre-registered rank-tracking gate constants (dossier §40, design "Pre-        #
# registered rank-tracking gate"). Fixed BEFORE any ladder is scored (contract #
# Rule 2); recorded here verbatim so no post-hoc narrative can move the bar.    #
# --------------------------------------------------------------------------- #

#: Contract-minimum bar (Req 2.2), applied strictly per metric: ``spearman > 0.0``.
#: Exactly 0.0 or negative ⇒ NULL for that metric; a PRIMARY (PairAP) NULL means the
#: scorer is NOT a trustworthy relative judge (Req 2.3).
CONTRACT_MINIMUM_BAR: float = 0.0

#: RECOMMENDED operational bar — the actual pre-registered gate — applied to the
#: PRIMARY PairAP metric (and computed/reported for the SECONDARY combined too, which
#: never overrides the primary): ``spearman >= 0.7``. High because at n=9 a small
#: positive Spearman is indistinguishable from noise.
OPERATIONAL_BAR: float = 0.7

#: Verdict tiers, decided by the PRIMARY (PairAP) metric.
TRUSTED: str = "TRUSTED"
WEAK: str = "WEAK"
NULL: str = "NULL"

#: The pre-registered n=9 significance / CI caveat (dossier §40, contract Rule 9),
#: attached verbatim to every :class:`RankTrackingVerdict`.
N9_CI_NOTE: str = (
    "n=9: with only nine points a small positive Spearman is indistinguishable from "
    "noise (wide confidence interval, non-significant p-value); this is why the "
    "operational bar is set high at 0.7. A bare spearman>0.0 clears the contract "
    "minimum but does NOT, at n=9, establish a trustworthy judge. The real trajectory "
    "is also non-monotone (documented regressions in the fuller 0.05713->...->0.44519 "
    "history), so a perfect Spearman is not expected even for a good scorer."
)

#: A ladder point whose recipe was re-run on the dev holdout and scored (has a local
#: PairAP / combined). This is the only status that contributes a value to the gate.
SCORABLE: str = "SCORABLE"

#: A ladder point whose re-runnable recipe is UNAVAILABLE, so no dev AP could be
#: produced. Recorded honestly with ``local_pair_ap = local_combined = None`` and a
#: reason — never a fabricated number (contract Rules 1 & 9, Req 2.1).
BLOCKED: str = "BLOCKED"


@dataclass(frozen=True)
class RecipeRerun:
    """The output of re-running an artifact's recipe on the dev holdout.

    A ``recipe_runner`` callback returns one of these for an artifact when a
    re-runnable recipe EXISTS, or ``None`` when it does not (⇒ the point is BLOCKED).
    The ``dev_predictions`` frame's ``pair_id``\\s ARE dev-holdout pairs (the eval CSV
    is NOT dev-scored — its pairs are disjoint), so :class:`CanonicalScorer` can score
    it under the canonical recipe.

    Attributes:
        dev_predictions: The dev-holdout prediction frame the recipe emits (production
            metric schema: ``pair_id``, ``risk_score``, ``predicted_behavior``, the
            five ``evidence_hand_*`` columns). Scored by the canonical scorer to yield
            the point's PairAP + combined.
        recipe_id: The recipe id that produced these predictions. It MUST match the
            canonical scorer's ``recipe_id`` so no cross-regime mixing occurs (Req 2.6);
            :meth:`LadderValidator.score_ladder` verifies this.
        detail: Optional human-readable provenance for the re-run (e.g. which config /
            dossier section), recorded in the ladder point's ``source``.
    """

    dev_predictions: pd.DataFrame
    recipe_id: str
    detail: str = ""


#: Signature of the injected recipe-rerun callback. Given the artifact path and its
#: parsed LB, it returns a :class:`RecipeRerun` (recipe available, dev predictions to
#: score) or ``None`` (recipe unavailable ⇒ BLOCKED). It NEVER returns the eval CSV as
#: if it were dev predictions (that would be dev-scoring an eval artifact — banned).
RecipeRunner = Callable[[Path, ArtifactLB], Optional[RecipeRerun]]


@dataclass(frozen=True)
class LadderPoint:
    """One known-LB artifact as a ``(real_lb, local_pair_ap, local_combined)`` point.

    Carries BOTH the PRIMARY PairAP and the SECONDARY combined local scores per
    artifact (design "Data Models"). ``real_lb`` is the external-judge score parsed
    from the filename; the two local scores come from a SINGLE canonical scoring pass
    over the recipe's dev-holdout re-run (never from dev-scoring the eval CSV).

    A BLOCKED point (no re-runnable recipe) records ``local_pair_ap = local_combined =
    None`` and a ``blocked_reason`` — an honest null, never a fabricated number
    (contract Rules 1 & 9, Req 2.1). Only ``SCORABLE`` points carry local values and
    are eligible to enter the rank-tracking gate (task 5.2).

    Attributes:
        config_id: Stable id of the scored configuration (derived from the artifact
            filename, e.g. ``ladder_044519``).
        real_lb: The REAL leaderboard score parsed from ``submission_best_<d>.csv``
            (external judge — the only ground truth; contract Rule 1).
        local_pair_ap: PRIMARY — the canonical scorer's PairAP from the recipe's
            dev-holdout re-run, or ``None`` when BLOCKED.
        local_combined: SECONDARY — the full official combined score from the SAME
            scoring pass, or ``None`` when BLOCKED.
        recipe_id: The canonical ``recipe_id`` under which this point was scored, so
            every point is mutually comparable (Req 2.5, 2.6). Recorded even for
            BLOCKED points (the recipe that WOULD have been used).
        source: Provenance: the artifact filename / sha and the recipe-rerun detail.
        status: :data:`SCORABLE` or :data:`BLOCKED`.
        blocked_reason: Why the point is BLOCKED (empty for SCORABLE points).
    """

    config_id: str
    real_lb: float
    local_pair_ap: Optional[float]
    local_combined: Optional[float]
    recipe_id: str
    source: str
    status: str = SCORABLE
    blocked_reason: str = ""

    def __post_init__(self) -> None:
        # Guard the honest constraint at construction: a BLOCKED point must NOT carry a
        # local number, and a SCORABLE point MUST carry both (contract Rules 1 & 9).
        if self.status not in (SCORABLE, BLOCKED):
            raise ValueError(f"status must be {SCORABLE!r} or {BLOCKED!r}, got {self.status!r}")
        if self.status == BLOCKED:
            if self.local_pair_ap is not None or self.local_combined is not None:
                raise ValueError(
                    "a BLOCKED ladder point must not carry a local_pair_ap/local_combined "
                    "(no fabricated number for an unavailable recipe; contract Rules 1 & 9)"
                )
            if not self.blocked_reason:
                raise ValueError("a BLOCKED ladder point must record a blocked_reason")
        else:  # SCORABLE
            if self.local_pair_ap is None or self.local_combined is None:
                raise ValueError(
                    "a SCORABLE ladder point must carry both local_pair_ap and local_combined"
                )

    @property
    def is_scorable(self) -> bool:
        """``True`` iff this point has a local AP (its recipe was re-run and scored)."""
        return self.status == SCORABLE

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the point for the append-only audit trail (dossier / anchor store)."""
        return {
            "config_id": self.config_id,
            "real_lb": self.real_lb,
            "local_pair_ap": self.local_pair_ap,
            "local_combined": self.local_combined,
            "recipe_id": self.recipe_id,
            "source": self.source,
            "status": self.status,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class MetricRankTracking:
    """Rank-tracking result for ONE metric (PairAP primary, or combined secondary).

    Populated by :meth:`LadderValidator.rank_tracking` (task 5.2). This task defines
    the model only; the fields match design "Data Models" so the gate can fill them.

    Attributes:
        metric_name: ``"pair_ap"`` (PRIMARY gate) or ``"combined"`` (SECONDARY).
        spearman: Spearman rank correlation between this metric's local values and the
            real LB across the scorable ladder points.
        threshold: The pre-registered operational bar for this metric (e.g. 0.7).
        passed: ``spearman >= threshold`` AND strictly ``> 0.0`` (contract-minimum).
        verdict: ``"TRUSTED"`` | ``"WEAK"`` | ``"NULL"``.
    """

    metric_name: str
    spearman: float
    threshold: float
    passed: bool
    verdict: str

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the per-metric rank-tracking result for the audit trail."""
        return {
            "metric_name": self.metric_name,
            "spearman": self.spearman,
            "threshold": self.threshold,
            "passed": self.passed,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class RankTrackingVerdict:
    """Bundle of the PRIMARY (PairAP) + SECONDARY (combined) rank-tracking results.

    Populated by :meth:`LadderValidator.rank_tracking` (task 5.2). Defined here so the
    model exists for the gate. The overall ``passed``/``verdict`` mirror the PRIMARY
    (PairAP) result; the SECONDARY is reported alongside regardless of whether it
    agrees — no cherry-picking (contract Rules 2 & 12). A PairAP-vs-combined
    disagreement sets ``agree=False`` and is recorded verbatim in ``disagreement_note``
    as a first-class finding, never reconciled away.

    Attributes:
        n_points: Number of SCORABLE ladder points that entered the gate.
        ci_note: The n=9 significance / confidence-interval caveat (a near-zero
            Spearman over so few points is indistinguishable from noise; Rule 9).
        primary: The PairAP :class:`MetricRankTracking` — the pre-registered gate.
        secondary: The combined :class:`MetricRankTracking` — the diagnostic.
        agree: Do primary and secondary land on the same pass/verdict tier?
        disagreement_note: Non-empty first-class finding when ``agree`` is ``False``.
        passed: ``== primary.passed`` — the gate verdict is the PRIMARY gate.
        verdict: ``== primary.verdict``.
        message: Human-readable summary of the verdict.
    """

    n_points: int
    ci_note: str
    primary: MetricRankTracking
    secondary: MetricRankTracking
    agree: bool
    disagreement_note: str
    passed: bool
    verdict: str
    message: str

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the bundled verdict (both metrics) for the append-only dossier."""
        return {
            "n_points": self.n_points,
            "ci_note": self.ci_note,
            "primary": self.primary.to_provenance(),
            "secondary": self.secondary.to_provenance(),
            "agree": self.agree,
            "disagreement_note": self.disagreement_note,
            "passed": self.passed,
            "verdict": self.verdict,
            "message": self.message,
        }


class LadderValidator:
    """Score the known-LB ladder under the canonical scorer (task 5.1).

    :meth:`score_ladder` turns the nine ``submission_best_<d>.csv`` artifacts into
    :class:`LadderPoint`\\s, each carrying BOTH the PRIMARY PairAP and SECONDARY
    combined local scores — obtained by RE-RUNNING each artifact's recipe on the dev
    holdout and scoring the dev-pair predictions with the :class:`CanonicalScorer`,
    NEVER by dev-scoring the disjoint eval CSV. Where a re-runnable recipe is
    unavailable, the point is recorded BLOCKED with no fabricated number (Req 2.1,
    contract Rules 1 & 9).

    The Spearman rank-tracking gate (:meth:`rank_tracking`) is task 5.2 and is NOT
    implemented here; only the data models it consumes are defined in this module.

    Args:
        scorer: The :class:`CanonicalScorer` (already gated on the official-metric
            self-check). Its ``recipe.recipe_id`` is recorded on every ladder point so
            all points are mutually comparable (Req 2.5, 2.6).
        recipe_runner: A callback that, given an artifact path and its parsed LB,
            returns a :class:`RecipeRerun` (the dev-holdout predictions the recipe
            emits) when a re-runnable recipe EXISTS, or ``None`` when it does not (⇒
            the point is BLOCKED). If omitted, EVERY artifact is BLOCKED — the honest
            default until a recipe registry is wired (that is downstream work).
    """

    def __init__(
        self,
        scorer: CanonicalScorer,
        *,
        recipe_runner: Optional[RecipeRunner] = None,
    ) -> None:
        self.scorer = scorer
        self._recipe_runner = recipe_runner

    @property
    def recipe(self) -> ScoringRecipe:
        """The canonical recipe every ladder point is (or would be) scored under."""
        return self.scorer.recipe

    @staticmethod
    def _config_id(artifact: ArtifactLB) -> str:
        """Stable ladder config id from the artifact's LB digits (e.g. ``ladder_044519``)."""
        return f"ladder_{artifact.digits}"

    def score_ladder(self, artifacts: List[Union[str, Path]]) -> List[LadderPoint]:
        """Score the known-LB ladder into :class:`LadderPoint`\\s, honestly.

        For each artifact:

        1. Parse the external-judge ``real_lb`` from its filename (no distortion).
        2. Ask the injected ``recipe_runner`` for the recipe's dev-holdout predictions.
           * ``None`` ⇒ record a **BLOCKED** point (no re-runnable recipe): no
             ``local_pair_ap`` / ``local_combined`` is invented (Req 2.1, Rules 1 & 9).
           * a :class:`RecipeRerun` whose ``recipe_id`` != the canonical ``recipe_id``
             ⇒ BLOCKED as well (no cross-regime mixing; Req 2.6). The eval CSV is never
             dev-scored — the runner returns DEV-holdout predictions.
        3. Score the dev-holdout predictions with the :class:`CanonicalScorer` in a
           SINGLE pass, recording BOTH ``pair_ap`` (PRIMARY) and ``combined``
           (SECONDARY) into a **SCORABLE** point, with the recipe id recorded so the
           point is mutually comparable (Req 2.5, 2.6).

        Args:
            artifacts: The list of ``submission_best_<d>.csv`` paths (or names) to
                score. Order is preserved in the returned list.

        Returns:
            One :class:`LadderPoint` per artifact, in the input order, each either
            SCORABLE (recipe re-run + scored) or BLOCKED (recipe unavailable) — the
            scorable-vs-BLOCKED split is thus reported honestly per artifact.
        """
        points: List[LadderPoint] = []
        recipe_id = self.recipe.recipe_id

        for raw in artifacts:
            path = Path(raw)
            artifact = parse_artifact_lb(path)
            config_id = self._config_id(artifact)

            rerun = self._recipe_runner(path, artifact) if self._recipe_runner else None

            if rerun is None:
                points.append(
                    LadderPoint(
                        config_id=config_id,
                        real_lb=artifact.real_lb,
                        local_pair_ap=None,
                        local_combined=None,
                        recipe_id=recipe_id,
                        source=f"artifact={path.name}",
                        status=BLOCKED,
                        blocked_reason=(
                            "no re-runnable recipe available for this artifact; the eval "
                            "CSV cannot be dev-scored (its pairs are disjoint from the dev "
                            "labels), so no local AP is produced (Req 2.1; contract "
                            "Rules 1 & 9)"
                        ),
                    )
                )
                continue

            # Cross-regime guard: the re-run's recipe MUST be the canonical recipe, or
            # the point is not mutually comparable and is recorded BLOCKED (Req 2.6).
            if rerun.recipe_id != recipe_id:
                points.append(
                    LadderPoint(
                        config_id=config_id,
                        real_lb=artifact.real_lb,
                        local_pair_ap=None,
                        local_combined=None,
                        recipe_id=recipe_id,
                        source=f"artifact={path.name}",
                        status=BLOCKED,
                        blocked_reason=(
                            f"recipe re-run reported recipe_id={rerun.recipe_id!r} which "
                            f"differs from the canonical recipe_id={recipe_id!r}; refusing "
                            "to mix regimes (Req 2.6)"
                        ),
                    )
                )
                continue

            # Re-run available under the canonical recipe: score the DEV-holdout
            # predictions (never the eval CSV) in a single canonical pass.
            score: CanonicalScore = self.scorer.score_dev_predictions(rerun.dev_predictions)
            source = f"artifact={path.name}; recipe_id={recipe_id}"
            if rerun.detail:
                source += f"; {rerun.detail}"
            points.append(
                LadderPoint(
                    config_id=config_id,
                    real_lb=artifact.real_lb,
                    local_pair_ap=score.pair_ap,      # PRIMARY gate quantity
                    local_combined=score.combined,    # SECONDARY diagnostic
                    recipe_id=recipe_id,
                    source=source,
                    status=SCORABLE,
                    blocked_reason="",
                )
            )

        return points

    # ------------------------------------------------------------------ #
    # Task 5.2 — the pre-registered external-judge rank-tracking gate     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _verdict_tier(spearman: float) -> str:
        """Map a PairAP-style Spearman onto the pre-registered verdict tier.

        Tiers (dossier §40 / design): ``spearman >= 0.7`` → :data:`TRUSTED`;
        ``0.0 < spearman < 0.7`` → :data:`WEAK`; ``spearman <= 0.0`` → :data:`NULL`.
        Used for BOTH metrics so the secondary's tier is comparable to the primary's
        for the agree/disagreement check.
        """
        if spearman >= OPERATIONAL_BAR:
            return TRUSTED
        if spearman > CONTRACT_MINIMUM_BAR:
            return WEAK
        return NULL

    @classmethod
    def _metric_tracking(
        cls, metric_name: str, local_values: List[float], real_lb: List[float]
    ) -> MetricRankTracking:
        """Build one :class:`MetricRankTracking` for a metric from its aligned values.

        Reuses :func:`poker_collusion.validation.lb_cv.spearman_corr` verbatim (never
        re-implemented). ``passed`` is set only when ``spearman >= 0.7 AND
        spearman > 0.0`` — the operational bar AND the contract-minimum, per the
        pre-registered gate.
        """
        spearman = spearman_corr(local_values, real_lb)
        verdict = cls._verdict_tier(spearman)
        passed = spearman >= OPERATIONAL_BAR and spearman > CONTRACT_MINIMUM_BAR
        return MetricRankTracking(
            metric_name=metric_name,
            spearman=spearman,
            threshold=OPERATIONAL_BAR,
            passed=passed,
            verdict=verdict,
        )

    def rank_tracking(self, points: List[LadderPoint]) -> RankTrackingVerdict:
        """Measure Spearman rank-tracking of BOTH metrics against the real LB order.

        The pre-registered external-judge gate (dossier §40, design "Pre-registered
        rank-tracking gate"), fixed BEFORE any ladder is scored (contract Rule 2):

        * **PRIMARY (the gate): PairAP** — ``spearman_corr(local_pair_ap, real_lb)``.
          The overall ``passed``/``verdict`` EQUAL this primary result. PairAP is
          pre-registered as the gate because the official metric is 70% PairAP; it
          cannot be swapped for the combined metric later because that looks better
          (contract Rules 2, 12).
        * **SECONDARY (diagnostic): combined** — ``spearman_corr(local_combined,
          real_lb)``, ALWAYS computed and reported regardless of whether it agrees
          with the primary. It NEVER overrides the primary.

        Both statistics reuse :func:`poker_collusion.validation.lb_cv.spearman_corr`
        verbatim. When the two metrics land on DIFFERENT verdict tiers, ``agree`` is
        ``False`` and ``disagreement_note`` records it as a first-class finding — it is
        never reconciled away or cherry-picked (contract Rule 12). The n=9 CI caveat
        (:data:`N9_CI_NOTE`) is attached to every verdict (contract Rule 9).

        Only SCORABLE points enter the gate — BLOCKED points carry no local values.
        Honest edge case (contract Rules 1, 9): Spearman is undefined for fewer than
        two points, so if fewer than two SCORABLE points exist this reports a
        :data:`NULL` verdict whose message states the correlation could not be
        computed — it does NOT fabricate a correlation. (Note ``spearman_corr`` itself
        returns ``0.0`` for ``size < 2``; that 0.0 is a sentinel, not a measured
        correlation, so the guard here reports the undefined case explicitly rather
        than passing a spurious 0.0 off as a NULL "result".)

        Args:
            points: The ladder points from :meth:`score_ladder`. Only SCORABLE points
                (those carrying ``local_pair_ap`` / ``local_combined``) contribute.

        Returns:
            The bundled :class:`RankTrackingVerdict` (both metrics, the agree flag /
            disagreement note, the n=9 caveat, and the overall PRIMARY verdict).
        """
        scorable = [p for p in points if p.is_scorable]
        n = len(scorable)

        # Honest edge case: Spearman is undefined for < 2 points — do not fabricate a
        # correlation (contract Rules 1, 9). Report NULL for both metrics with an
        # explicit "could not be computed" message; the spearman field carries the
        # spearman_corr sentinel (0.0) but the message and NULL verdict make clear no
        # correlation was measured.
        if n < 2:
            undefined = MetricRankTracking(
                metric_name="pair_ap",
                spearman=0.0,
                threshold=OPERATIONAL_BAR,
                passed=False,
                verdict=NULL,
            )
            secondary_undefined = MetricRankTracking(
                metric_name="combined",
                spearman=0.0,
                threshold=OPERATIONAL_BAR,
                passed=False,
                verdict=NULL,
            )
            return RankTrackingVerdict(
                n_points=n,
                ci_note=N9_CI_NOTE,
                primary=undefined,
                secondary=secondary_undefined,
                agree=True,
                disagreement_note="",
                passed=False,
                verdict=NULL,
                message=(
                    f"NULL: only {n} scorable ladder point(s) — Spearman is undefined for "
                    "fewer than 2 points, so rank-tracking could NOT be computed and no "
                    "correlation is claimed (contract Rules 1, 9). BLOCKED points carry no "
                    "local values and do not enter the gate."
                ),
            )

        real_lb = [p.real_lb for p in scorable]
        pair_ap = [float(p.local_pair_ap) for p in scorable]  # SCORABLE ⇒ not None
        combined = [float(p.local_combined) for p in scorable]

        primary = self._metric_tracking("pair_ap", pair_ap, real_lb)
        secondary = self._metric_tracking("combined", combined, real_lb)

        # Disagreement is decided on the verdict TIER (TRUSTED/WEAK/NULL). It is a
        # first-class recorded finding, never reconciled away (contract Rule 12).
        agree = primary.verdict == secondary.verdict
        disagreement_note = ""
        if not agree:
            disagreement_note = (
                f"PairAP (PRIMARY) rank-tracks as {primary.verdict} "
                f"(spearman={primary.spearman:.4f}) but combined (SECONDARY) rank-tracks "
                f"as {secondary.verdict} (spearman={secondary.spearman:.4f}). Recorded "
                "verbatim as a first-class finding: the two metrics order the ladder "
                "differently, so the behavior/evidence heads carry LB order not captured "
                "by PairAP alone (or vice versa). The overall gate verdict still equals "
                "the PRIMARY (PairAP) result — the secondary NEVER overrides it and is "
                "not cherry-picked (contract Rules 2, 12)."
            )

        # Overall verdict EQUALS the PRIMARY (PairAP) — pre-registered, not swappable.
        message = (
            f"{primary.verdict}: PairAP (PRIMARY gate) spearman={primary.spearman:.4f} "
            f"over n={n} scorable point(s) (operational bar {OPERATIONAL_BAR}, contract "
            f"minimum >{CONTRACT_MINIMUM_BAR}); combined (SECONDARY) "
            f"spearman={secondary.spearman:.4f} → {secondary.verdict}, reported alongside "
            f"but never overriding. {'Metrics agree.' if agree else 'DISAGREEMENT recorded.'}"
        )

        return RankTrackingVerdict(
            n_points=n,
            ci_note=N9_CI_NOTE,
            primary=primary,
            secondary=secondary,
            agree=agree,
            disagreement_note=disagreement_note,
            passed=primary.passed,
            verdict=primary.verdict,
            message=message,
        )
