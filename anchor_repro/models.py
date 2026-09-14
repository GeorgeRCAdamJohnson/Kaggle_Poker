"""Frozen data models for the poker-anchor-reproduction canonical scorer.

This module defines the two load-bearing, frozen data models the canonical local
scorer produces and is parameterised by (design.md "Components and Interfaces" §1,
"Data Models" §ScoringRecipe/§CanonicalScore):

- ``ScoringRecipe`` — the single, auditable local scoring recipe under which EVERY
  anchor (ours and competitors') is measured, so anchors are mutually comparable
  and never mix regimes (Req 2.5, 2.6). It names PairAP as the PRIMARY rank-tracked
  component and the full official combined score as the SECONDARY diagnostic.
- ``CanonicalScore`` — both local scores plus the full ``ScoreComponents`` breakdown,
  all produced by a SINGLE ``reference_public_metric.score`` pass (never two divergent
  recomputations). ``pair_ap`` is the PRIMARY gate quantity and the anchor's
  ``local_holdout_ap``; ``combined`` is the SECONDARY diagnostic recorded alongside it.

Reuse, not re-implementation (contract Rule 1, Req 1.6): ``ScoreComponents`` and
``Seeds`` are imported verbatim from the existing ``poker_collusion`` package. The
official metric, CV split, and Spearman statistic are NOT re-implemented here.

Accountability contract note: the recipe is serialized into every anchor's provenance
so the exact metric/split/unknown-handling used is externally auditable, and the
provenance explicitly names which component is PRIMARY (the gate) versus SECONDARY
(a reported-alongside diagnostic) — no cherry-picking whichever metric looks better
(contract Rules 2 & 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# Reuse verbatim from the existing poker_collusion package — do NOT re-implement.
from poker_collusion.config import Seeds
from poker_collusion.metric.production_metric import ScoreComponents

__all__ = [
    "ScoringRecipe",
    "CanonicalScore",
    "DisjointSetDiagnostic",
    "IntegrityReport",
]


@dataclass(frozen=True)
class ScoringRecipe:
    """The single agreed local scoring recipe every anchor is measured under.

    Frozen and identical across all anchors this spec produces so our-artifact and
    competitor anchors never mix regimes (Req 2.6). The recipe is serialized into
    every anchor's provenance (see :meth:`to_provenance`) so the metric, split, and
    unknown-handling are auditable.

    The recipe explicitly names the PRIMARY rank-tracked component (PairAP — the
    official metric is 70% PairAP and the canonical scorer targets it) and the
    SECONDARY diagnostic (the full official combined score). Both are produced by a
    single ``reference_public_metric.score`` pass; the gate verdict is decided by the
    PRIMARY, with the SECONDARY reported alongside regardless (contract Rules 2, 12).

    Attributes:
        recipe_id: Stable id for this recipe, embedded in every anchor's provenance.
        metric: The verbatim official metric used (never a re-implementation).
        primary_ap_component: The PRIMARY rank-tracked component (PairAP), the gate.
        secondary_component: The SECONDARY diagnostic (the full official combined score).
        split: The eval-mimic dev holdout split under which predictions are scored.
        unknown_handling: PU unknown-label handling — unknowns never scored as negatives.
        holdout_seed: Deterministic split seed; defaults to ``Seeds.cv_split`` (404).
    """

    recipe_id: str = "canonical_v1"
    metric: str = "reference_public_metric.score"
    #: PRIMARY rank-tracked component: the metric's 70%-weighted PairAP intermediate.
    primary_ap_component: str = "pair_ap"
    #: SECONDARY diagnostic: the full official combined score.
    secondary_component: str = "combined"
    split: str = "table_disjoint_holdout_40pct"
    unknown_handling: str = "pu_confirmed_only"
    holdout_seed: int = Seeds.cv_split

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the recipe to a provenance dict for embedding in every anchor.

        The dict names the PRIMARY (gate) and SECONDARY (diagnostic) components
        explicitly so an auditor can see, from the anchor alone, which local metric
        decided trust and which was merely reported alongside (contract Rules 2, 12).
        Every recipe field is recorded so the exact scoring regime is reproducible.
        """
        return {
            "recipe_id": self.recipe_id,
            "metric": self.metric,
            "primary_ap_component": self.primary_ap_component,
            "secondary_component": self.secondary_component,
            "split": self.split,
            "unknown_handling": self.unknown_handling,
            "holdout_seed": self.holdout_seed,
        }


@dataclass(frozen=True)
class CanonicalScore:
    """The two local scores plus the full component breakdown from a single scoring pass.

    Produced by ``CanonicalScorer.score_dev_predictions`` from a SINGLE
    ``reference_public_metric.score`` evaluation, so ``pair_ap`` and ``combined`` are
    never computed from two divergent recomputations or splits (design §1). ``pair_ap``
    is the PRIMARY gate quantity and the anchor's ``local_holdout_ap``; ``combined`` is
    the SECONDARY diagnostic recorded alongside it.

    Attributes:
        pair_ap: PRIMARY gate quantity — the metric's 70%-weighted PairAP intermediate.
            This is the anchor's ``local_holdout_ap`` and the value the ladder gate
            rank-tracks against the real LB.
        combined: SECONDARY diagnostic — the full official combined score
            ``0.70*pair_ap + 0.20*evidence_map + 0.10*behavior_map``.
        components: The full ``ScoreComponents`` breakdown (pair_ap, evidence_map,
            behavior_map, combined) from the same single metric pass.
    """

    #: PRIMARY gate quantity (the anchor's local_holdout_ap).
    pair_ap: float
    #: SECONDARY diagnostic (the full official combined score).
    combined: float
    #: Full component breakdown from the same single reference_public_metric pass.
    components: ScoreComponents

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize both local scores plus the full component breakdown for an anchor.

        Labels ``pair_ap`` as the PRIMARY (the anchor's ``local_holdout_ap``) and
        ``combined`` as the SECONDARY diagnostic so the recorded numbers are never
        confused with an absolute LB reproduction (impossible; contract Rules 1 & 9,
        Req 2.4) — these are MEASURED local scores, not externally verified LB values.
        """
        return {
            "pair_ap": self.pair_ap,
            "combined": self.combined,
            "components": self.components.as_dict(),
            "primary_component": "pair_ap",
            "secondary_component": "combined",
        }


@dataclass(frozen=True)
class DisjointSetDiagnostic:
    """The MEASURED sizes of the disjoint pair-id sets between two frames.

    This is the load-bearing honest number the impossibility / coverage-mismatch guard
    surfaces (design.md "Coverage-mismatch guard", Property 5): whenever a prediction
    frame's ``pair_id`` set does not equal the reference (dev-holdout solution or eval
    sample) set, the guard names HOW MANY ids are missing and HOW MANY are extra —
    never a silent zero or a fabricated number (Req 2.4, 4.1). ``intersection == 0`` is
    the specific eval-vs-dev impossibility recorded during design (the eval artifacts'
    ~112,540 pairs are disjoint from the 1,860 confirmed dev pairs).

    Attributes:
        n_reference: Size of the reference pair-id set (the dev-holdout solution or the
            eval sample submission the predictions are checked against).
        n_predicted: Size of the prediction frame's pair-id set.
        n_intersection: Number of pair-ids common to both (0 ⇒ fully disjoint).
        n_missing: Pair-ids present in the reference but absent from the predictions
            (``|reference - predicted|``) — mirrors the official metric's ``missing``.
        n_extra: Pair-ids present in the predictions but absent from the reference
            (``|predicted - reference|``) — mirrors the official metric's ``extra``.
    """

    n_reference: int
    n_predicted: int
    n_intersection: int
    n_missing: int
    n_extra: int

    @property
    def is_equal(self) -> bool:
        """``True`` iff the two pair-id sets are identical (nothing missing or extra)."""
        return self.n_missing == 0 and self.n_extra == 0

    @property
    def is_disjoint(self) -> bool:
        """``True`` iff the two pair-id sets share no members at all."""
        return self.n_intersection == 0

    def describe(self) -> str:
        """A human-readable one-line diagnostic naming the disjoint-set sizes.

        Phrased so it reads correctly for both the total-disjoint eval-vs-dev
        impossibility and a partial missing/extra coverage mismatch, always naming the
        missing and extra counts explicitly (Property 5, contract Rule 9).
        """
        base = (
            f"pair_id coverage mismatch: {self.n_missing} missing and "
            f"{self.n_extra} extra "
            f"(reference={self.n_reference}, predicted={self.n_predicted}, "
            f"intersection={self.n_intersection})"
        )
        if self.is_disjoint:
            base += (
                " — the pair-id sets are DISJOINT (intersection=0): these are eval "
                "pairs, not dev-holdout pairs, so no local metric can reproduce an "
                "absolute LB score here (impossible, not zero; Req 2.4)"
            )
        return base

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the measured disjoint-set sizes for the audit trail."""
        return {
            "n_reference": self.n_reference,
            "n_predicted": self.n_predicted,
            "n_intersection": self.n_intersection,
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "is_equal": self.is_equal,
            "is_disjoint": self.is_disjoint,
        }


@dataclass(frozen=True)
class IntegrityReport:
    """Scoreability / integrity of a frozen EVAL artifact (task 2.3, mirrors gate_a).

    Produced by :meth:`anchor_repro.scorer.CanonicalScorer.integrity_report`. It
    mirrors the checks in ``poker_collusion.experiments.gate_a_verify_floor`` — hash,
    row count, pair-set-equals-sample, non-degenerate ranking — WITHOUT attempting to
    dev-score the eval artifact, which is impossible (its pairs are disjoint from the
    dev labels; design.md, contract Rules 1 & 9). The report always surfaces the
    disjoint-set sizes between the artifact and the dev holdout so the impossibility is
    named explicitly rather than hidden (Property 5, Req 2.4, 4.1).

    Attributes:
        sha256: SHA-256 hex digest of the artifact file (byte-integrity anchor).
        n_bytes: Artifact size in bytes.
        n_rows: Row count. The eval artifacts have ``EVAL_ROW_COUNT`` (112,540) rows.
        expected_row_count: The expected eval row count (112,540) this was checked
            against.
        row_count_ok: ``n_rows == expected_row_count``.
        columns_match: The artifact header equals the official submission schema/order.
        pair_set_equals_sample: The artifact's ``pair_id`` SET equals the eval sample
            submission's set (Kaggle-scoreable) — ``None`` when the sample submission
            was unavailable to compare against.
        sample_diagnostic: The measured disjoint-set sizes vs the eval sample
            submission (``None`` when the sample was unavailable).
        risk_min / risk_max / risk_mean: risk-column statistics.
        risk_n_distinct: Number of distinct risk values (a ranking discriminator).
        risk_in_unit_interval: Every risk value is finite and in ``[0, 1]``.
        ranking_non_degenerate: The risk column is a non-degenerate ranking (many
            distinct values with spread), matching ``gate_a_verify_floor``.
        dev_disjoint_diagnostic: The measured disjoint-set sizes between the artifact
            and the dev-holdout labels — surfaces the eval-vs-dev impossibility
            explicitly (intersection is expected to be 0). ``None`` when dev labels
            were unavailable to compare against.
        scoreable: ``True`` iff the artifact passes the integrity/scoreability bar
            (intact schema, correct row count, pair-set matches the eval sample, and a
            non-degenerate ranking) — the necessary condition to be a leaderboard-valid
            file. This is NOT a claim that it can be dev-scored (it cannot).
        notes: Human-readable notes, MEASURED-labelled (contract Rule 9).
    """

    sha256: str
    n_bytes: int
    n_rows: int
    expected_row_count: int
    row_count_ok: bool
    columns_match: bool
    pair_set_equals_sample: Any  # Optional[bool]
    sample_diagnostic: Any       # Optional[DisjointSetDiagnostic]
    risk_min: float
    risk_max: float
    risk_mean: float
    risk_n_distinct: int
    risk_in_unit_interval: bool
    ranking_non_degenerate: bool
    dev_disjoint_diagnostic: Any  # Optional[DisjointSetDiagnostic]
    scoreable: bool
    notes: List[str] = field(default_factory=list)

    def to_provenance(self) -> Dict[str, Any]:
        """Serialize the integrity report for the audit trail (append-only dossier)."""
        return {
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "n_rows": self.n_rows,
            "expected_row_count": self.expected_row_count,
            "row_count_ok": self.row_count_ok,
            "columns_match": self.columns_match,
            "pair_set_equals_sample": self.pair_set_equals_sample,
            "sample_diagnostic": (
                self.sample_diagnostic.to_provenance()
                if self.sample_diagnostic is not None
                else None
            ),
            "risk_min": self.risk_min,
            "risk_max": self.risk_max,
            "risk_mean": self.risk_mean,
            "risk_n_distinct": self.risk_n_distinct,
            "risk_in_unit_interval": self.risk_in_unit_interval,
            "ranking_non_degenerate": self.ranking_non_degenerate,
            "dev_disjoint_diagnostic": (
                self.dev_disjoint_diagnostic.to_provenance()
                if self.dev_disjoint_diagnostic is not None
                else None
            ),
            "scoreable": self.scoreable,
            "notes": list(self.notes),
        }
