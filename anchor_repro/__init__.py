"""anchor_repro: turn REAL code with KNOWN real leaderboard scores into REAL
``(local-holdout-AP, real-LB)`` anchors.

This package is governed end-to-end by the workspace
``reverse-engineering-accountability`` contract (external judge only; pre-register
the bar; honest nulls are results; no extrapolation; state the uncertainty).

Design intent (build-order step 1 — tooling first): this package **reuses** the
existing :mod:`poker_collusion` package verbatim and does **NOT** re-implement the
official metric, the CV split, or the Spearman statistic. All three already exist
and are imported here so downstream modules (CanonicalScorer, LadderValidator,
AnchorStore, SandboxedReproRunner) build on the single audited implementation
rather than a divergent copy:

* :func:`poker_collusion.metric.reference_public_metric.score`
      the verbatim official competition metric (Req 1.2 / 1.3 already satisfied by
      this existing local copy — no second copy is created).
* :func:`poker_collusion.metric.production_metric.score_components`
      the component breakdown (PairAP / EvidenceMAP@5 / BehaviorMAP + combined).
* :class:`poker_collusion.metric.production_metric.ScoreComponents`
      the component dataclass carried inside a ``CanonicalScore``.
* :class:`poker_collusion.validation.cv.CVHarness` and
  :func:`poker_collusion.validation.cv.build_fold_solution`
      the pool-disjoint, PU-correct CV split (unknowns never materialized as
      negatives) reused so PU handling is inherited, not reinvented.
* :func:`poker_collusion.validation.lb_cv.spearman_corr`
      the rank-correlation statistic reused for both the primary (PairAP) and
      secondary (combined) rank-tracking gates.
* :class:`poker_collusion.config.PipelineConfig` and
  :class:`poker_collusion.config.Seeds`
      the input-dir shim and the deterministic seeds (``Seeds.cv_split == 404``).

These names are re-exported so the rest of ``anchor_repro`` imports them from one
place. Importing this module must not pull in ``polars`` or execute any untrusted
notebook code — the notebook dependency is recorded in ``requirements.lock`` and
imported only inside the sandboxed reproduction runner.
"""

from poker_collusion.config import PipelineConfig, Seeds
from poker_collusion.metric import reference_public_metric
from poker_collusion.metric.production_metric import ScoreComponents, score_components
from poker_collusion.metric.reference_public_metric import ParticipantVisibleError
from poker_collusion.validation.cv import CVHarness, build_fold_solution
from poker_collusion.validation.lb_cv import spearman_corr

# Official-metric self-check (task 2.1): re-verifies the local verbatim metric copy
# against the kernel's own score() before the scorer is trusted (Req 1.2 / 1.4). A
# FAIL is a hard release blocker via ``require_metric_matches_kernel``.
from anchor_repro.self_check import (
    MetricSelfCheckError,
    SelfCheckResult,
    require_metric_matches_kernel,
    verify_metric_against_kernel,
)

# Frozen data models (tasks 1.2, 2.3): the scoring recipe, the canonical score, the
# disjoint-set diagnostic and the frozen-EVAL-artifact integrity report.
from anchor_repro.models import (
    CanonicalScore,
    DisjointSetDiagnostic,
    IntegrityReport,
    ScoringRecipe,
)

# Canonical local scorer (task 2.2): the single auditable recipe under which every
# anchor is measured. Reuses the metric/CV/PU-handling re-exported above and is gated
# on the official-metric self-check before it will score anything (Req 1.2, 1.6, 2.5, 2.6).
# Task 2.3 adds ``integrity_report`` (frozen-EVAL-artifact scoreability, mirrors
# gate_a_verify_floor) and the impossibility + coverage-mismatch guard
# (``CoverageMismatchError``, naming the disjoint-set sizes — never a silent zero).
from anchor_repro.scorer import EVAL_ROW_COUNT, CanonicalScorer, CoverageMismatchError

# Ladder validation (task 5.1): the frozen data models for the external-judge
# rank-tracking gate (LadderPoint / MetricRankTracking / RankTrackingVerdict) plus
# ``LadderValidator.score_ladder``, which scores the nine known-LB artifacts by
# RE-RUNNING each recipe on the dev holdout (never dev-scoring the disjoint eval CSV)
# and records scorable-vs-BLOCKED honestly (Req 2.1, 2.5, 2.6; contract Rules 1 & 9).
# The Spearman gate (rank_tracking) is task 5.2.
# Task 5.2 adds ``LadderValidator.rank_tracking`` (the pre-registered Spearman gate)
# plus the pre-registered gate constants (bars / verdict tiers / n=9 CI caveat),
# mirroring dossier §40 verbatim (Req 2.2, 2.3; contract Rules 2, 9, 12).
from anchor_repro.ladder import (
    BLOCKED,
    CONTRACT_MINIMUM_BAR,
    N9_CI_NOTE,
    NULL,
    OPERATIONAL_BAR,
    SCORABLE,
    TRUSTED,
    WEAK,
    LadderPoint,
    LadderValidator,
    MetricRankTracking,
    RankTrackingVerdict,
    RecipeRerun,
)

# Append-only anchor store (task 8.1): records this spec's anchors into a store
# schema-identical to the parent .kiro/specs/poker-collusion-detection/lb_anchors.json
# so the poker-layered-tuning harness consumes them verbatim. Reuses the parent
# LBAnchor schema (no new top-level field); the canonical recipe_id is folded INTO the
# source/provenance string. Enforces append-only (no edit/delete of an existing
# anchor), verified-externals-only (verified=true needs a real positive external LB —
# contract Rule 1), and atomic writes (temp file + os.replace) (Req 3.5, 4.2; Property 6).
from anchor_repro.anchor_store import (
    Anchor,
    AnchorStore,
    AnchorStoreError,
    DEFAULT_ANCHOR_STORE_PATH,
    RECIPE_ID_PROVENANCE_KEY,
)

# Sandboxed reproduction runner — read_as_untrusted + NotebookManifest (task 7.1).
# Statically parses an UNTRUSTED competitor .ipynb WITHOUT executing any cell,
# extracting declared deps + every data-path read and flagging any network /
# subprocess / credential-env / filesystem-escape call for human review. ALL notebook
# content is treated as untrusted regardless of source (Req 1.5, 3.1). The actual
# sandboxed run() (path shim, network disabled, working-dir jail) is task 7.2.
# Task 7.2 adds the actual sandboxed ``run()`` + the ``ReproResult`` RUNNING →
# SUCCEEDED/BLOCKED state machine (path shim, network egress disabled, credential-
# stripped env, working-dir jail + wall-clock timeout). The notebook is executed as a
# CHILD PROCESS (never in-process) and the emitted submission.csv is validated against
# the official submission contract before a run is called SUCCEEDED (Req 1.5, 3.1, 3.3).
from anchor_repro.repro_runner import (
    DataPathRead,
    FlaggedCall,
    NotebookManifest,
    ReproResult,
    SandboxedReproRunner,
    STAGE_COMPUTE,
    STAGE_DATA_PATH,
    STAGE_DEPS,
    STAGE_RUNTIME,
    STATUS_BLOCKED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
)

#: The verbatim official metric callable, re-exported for convenience so the
#: canonical scorer references exactly the same ``score`` the kernel self-check
#: verifies (never a re-implementation).
# Ladder recipe-runner adapter (task 5.8): reconstructs the cand_exp4c_directional
# floor stack (044519) from the on-disk caches and re-runs it on the dev holdout to
# close the task-5.5 execution-gap null (contract Rules 3, 11). The other eight
# trajectory artifacts are honestly BLOCKED with a specific named missing input.
from anchor_repro.recipe_registry import (
    CacheLocations,
    FloorRerunResult,
    ReconstructionStatus,
    build_floor_recipe_runner,
    build_ladder_recipe_runner,
    default_cache_locations,
    reconstruction_report,
    FLOOR_DIGITS,
    FLOOR_RECIPE_ID,
)

reference_score = reference_public_metric.score

__all__ = [
    # config / seeds (reused)
    "PipelineConfig",
    "Seeds",
    # official metric (reused, verbatim — never re-implemented)
    "reference_public_metric",
    "reference_score",
    "ParticipantVisibleError",
    "ScoreComponents",
    "score_components",
    # CV split (reused, PU-correct)
    "CVHarness",
    "build_fold_solution",
    # rank-correlation statistic (reused)
    "spearman_corr",
    # official-metric self-check (task 2.1)
    "verify_metric_against_kernel",
    "require_metric_matches_kernel",
    "SelfCheckResult",
    "MetricSelfCheckError",
    # frozen data models (tasks 1.2, 2.3)
    "ScoringRecipe",
    "CanonicalScore",
    "DisjointSetDiagnostic",
    "IntegrityReport",
    # canonical scorer (task 2.2) + impossibility/coverage-mismatch guard (task 2.3)
    "CanonicalScorer",
    "CoverageMismatchError",
    "EVAL_ROW_COUNT",
    # ladder validation data models + score_ladder (task 5.1)
    "LadderPoint",
    "MetricRankTracking",
    "RankTrackingVerdict",
    "RecipeRerun",
    "LadderValidator",
    "SCORABLE",
    "BLOCKED",
    # pre-registered rank-tracking gate constants (task 5.2, dossier §40)
    "CONTRACT_MINIMUM_BAR",
    "OPERATIONAL_BAR",
    "TRUSTED",
    "WEAK",
    "NULL",
    "N9_CI_NOTE",
    # ladder recipe-runner adapter (task 5.8)
    "CacheLocations",
    "FloorRerunResult",
    "ReconstructionStatus",
    "build_floor_recipe_runner",
    "build_ladder_recipe_runner",
    "default_cache_locations",
    "reconstruction_report",
    "FLOOR_DIGITS",
    "FLOOR_RECIPE_ID",
    # append-only anchor store (task 8.1)
    "AnchorStore",
    "Anchor",
    "AnchorStoreError",
    "DEFAULT_ANCHOR_STORE_PATH",
    "RECIPE_ID_PROVENANCE_KEY",
    # sandboxed reproduction runner: read_as_untrusted (task 7.1)
    "SandboxedReproRunner",
    "NotebookManifest",
    "FlaggedCall",
    "DataPathRead",
    # sandboxed run() + ReproResult state machine (task 7.2)
    "ReproResult",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "STATUS_BLOCKED",
    "STAGE_DEPS",
    "STAGE_DATA_PATH",
    "STAGE_COMPUTE",
    "STAGE_RUNTIME",
]

__version__ = "0.1.0"
