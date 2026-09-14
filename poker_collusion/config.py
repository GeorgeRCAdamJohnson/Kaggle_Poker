"""Central seeded configuration for the poker_collusion pipeline.

Holds the single source of truth for filesystem paths, random seeds, the
row-group / column selections that drive chunked reads of the large
``actions.parquet`` file (Requirement 1.6), and the schema / threshold version
identifiers used for reproducible feature generation (Requirements 5.4, 11.2).

The configuration auto-detects the Kaggle notebook environment so the same code
runs on Windows PowerShell and in the Kaggle kernel. Nothing here performs I/O
at import time; paths are resolved lazily by consumers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

__all__ = [
    "ACTIONS_ROW_COUNT",
    "DEFAULT_COLUMN_SELECTION",
    "ExploitConfig",
    "PipelineConfig",
    "Seeds",
    "get_config",
    "is_kaggle_environment",
]

# Expected number of rows in actions.parquet, recorded during Discovery (task 2.1).
ACTIONS_ROW_COUNT: int = 18_609_028

# Behavior families disclosed publicly (used across the pipeline for stratification).
DISCLOSED_FAMILY_NAMES: tuple[str, ...] = (
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
)

# Sentinel literal written into unused evidence slots (Requirements 8.6, 8.7).
NO_EVIDENCE: str = "NO_EVIDENCE"


def is_kaggle_environment() -> bool:
    """Return True when running inside the Kaggle notebook environment.

    Kaggle mounts competition inputs under ``/kaggle/input`` and expects the
    submission at ``/kaggle/working/submission.csv`` (Requirement 9.5).
    """
    return Path("/kaggle/input").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


@dataclass(frozen=True)
class Seeds:
    """All random seeds, fixed for reproducibility (Requirements 6.7, 11.2).

    A single ``master`` seed plus per-component seeds so that changing one
    component's randomness does not perturb the others.
    """

    master: int = 20240918
    pu_ranker: int = 101
    behavior_classifier: int = 202
    evidence_ranker: int = 303
    cv_split: int = 404
    bootstrap: int = 505
    #: Seed for the LearnedRiskModel (supervised logistic over the confirmed-pair features that
    #: reached CV AUC ~0.85 in the signal-separation measurement). See models/learned_risk.py.
    learned_risk: int = 606


# Column projections per gameplay table. Signal extraction never loads
# actions.parquet whole; it reads only the decision-time-context columns
# (Requirements 1.6, 3.6, 3.7).
#
# Column names are the REAL competition schema, verified against the parquet/CSV
# files during the Discovery-refresh pass (task 2.x re-run). Notes on names that
# differ from earlier DERIVED-SPEC guesses:
#   * seats use ``seat_no`` (not ``seat``) and ``starting_stack`` (not
#     ``stack_start``); there is no ``stack_end`` — per-seat outcome is carried by
#     ``net_chips`` / ``won_share`` / ``total_contribution``.
#   * actions carry the actor in ``player_id`` (not ``actor``) and the betting
#     round in ``street`` (not a per-action ``phase``); ``players_active`` exists.
#   * ``table_id`` on hands is the pool key (400 tables x 30 players x 5,000 hands,
#     verified disjoint); there is no separate ``pool`` column.
DEFAULT_COLUMN_SELECTION: Dict[str, List[str]] = {
    "players": [
        "player_id",
        "account_age_days",
        "experience_hands_bucket",
        "preferred_stake",
        "region_bucket",
        "client_family",
    ],
    "hands": [
        "hand_id",
        "table_id",
        "phase",
        "big_blind",
        "small_blind",
        "started_at",
        "button_seat",
    ],
    "seats": [
        "hand_id",
        "player_id",
        "seat_no",
        "starting_stack",
        "total_contribution",
        "net_chips",
        "folded",
        "went_to_showdown",
        "won_share",
    ],
    "actions": [
        "hand_id",
        "action_no",
        "street",
        "player_id",
        "action",
        "amount",
        "amount_to",
        "to_call",
        "pot_before",
        "stack_before",
        "players_active",
    ],
}


def _default_input_dir() -> Path:
    """Resolve the input directory for the current environment."""
    if is_kaggle_environment():
        # Kaggle mounts the competition dataset under /kaggle/input/<slug>.
        base = Path("/kaggle/input")
        return base
    return Path.cwd() / "data" / "poker"


def _default_output_dir() -> Path:
    """Resolve the output directory for the current environment (Requirement 9.5)."""
    if is_kaggle_environment():
        return Path("/kaggle/working")
    return Path.cwd() / "outputs" / "poker_collusion"


@dataclass(frozen=True)
class ExploitConfig:
    """Config flags for the Discovery Exploit Ledger adoptions (task 18; Req 12.1, 12.3, 12.4).

    Each flag toggles ONE exploit from ``RESEARCH_DOSSIER.md`` §4 that was run through the local
    CV harness (``poker_collusion.validation.exploit_eval``). An exploit is default-**ON** only if
    it was CONFIRMED beyond the bootstrap noise band vs the Phase-1 floor with no component
    regression; REFUTED / OPEN exploits stay default-**OFF**. Every flag is deterministic and pure
    (no randomness, no wall-clock); flipping one changes only the wired behavior it names.

    The wiring is intentionally minimal:

    * ``e1_risk_calibration`` (E1) — apply a monotone recalibration of the shared ``risk_score``
      before behavior/Pair-AP scoring. A strictly-monotone map preserves the pair ranking (so Pair
      AP is unchanged in the tie-free case) while it can lift Behavior MAP because a family's OvR
      score is the ``risk_score`` where that family is predicted. ``e1_calibration_gamma`` is the
      power-map exponent applied to ``risk`` (``risk ** gamma``); ``gamma == 1.0`` is the identity.
    * ``e2_liberal_other_coordination`` (E2) — widen the behavior classifier's
      ``other_coordination`` routing margin by ``e2_extra_margin`` (routing ambiguous coordinated
      pairs to the excluded catch-all family is free in Behavior MAP; the pair keeps its
      ``risk_score`` for Pair AP). Adopted only if the net combined summary does not regress.
    * ``e4_coverage_over_precision`` (E4) — lower the evidence-emission risk floor so MORE true
      target pairs are surfaced under Evidence MAP@5's zero-for-missed-target rule.

    Defaults below reflect the task-18 local-CV outcome (see the dossier §4 ledger and Change Log).
    """

    #: E1 — monotone risk recalibration of the shared risk_score (lifts Behavior MAP; Pair-AP-safe
    #: when strictly monotone). Default reflects the CONFIRMED/REFUTED status recorded in the ledger.
    e1_risk_calibration: bool = False
    #: Power-map exponent for E1 (``risk_calibrated = clip(risk ** gamma, 0, 1)``). 1.0 = identity.
    e1_calibration_gamma: float = 1.0

    #: E2 — liberal other_coordination routing (widen the routing margin by ``e2_extra_margin``).
    e2_liberal_other_coordination: bool = False
    #: Additional decision-margin added to the behavior classifier's other_coordination margin.
    e2_extra_margin: float = 0.0

    #: E4 — coverage-over-precision evidence policy (emit evidence for more surfaced target pairs).
    e4_coverage_over_precision: bool = False

    def calibrate_risk(self, risk: float) -> float:
        """Apply the E1 monotone recalibration to a single ``risk`` value (identity when off).

        A power map ``risk ** gamma`` with ``gamma > 0`` is strictly monotone on ``[0, 1]``, so it
        preserves the pair ranking (Pair-AP-safe in the tie-free case) while re-spacing the levels
        that feed Behavior MAP. Returns ``risk`` unchanged when E1 is disabled or ``gamma == 1``.
        """
        if not self.e1_risk_calibration or self.e1_calibration_gamma == 1.0:
            return float(min(max(risk, 0.0), 1.0))
        r = float(min(max(risk, 0.0), 1.0))
        g = float(self.e1_calibration_gamma)
        if g <= 0.0:
            return r
        return float(min(max(r ** g, 0.0), 1.0))

    @property
    def effective_other_coordination_margin_delta(self) -> float:
        """Extra other_coordination routing margin from E2 (0.0 when the exploit is disabled)."""
        return float(self.e2_extra_margin) if self.e2_liberal_other_coordination else 0.0


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Attributes:
        input_dir: Directory containing the competition input files.
        output_dir: Directory for pipeline outputs (submission, artifacts).
        seeds: Fixed random seeds (Requirements 6.7, 11.2).
        exploits: Discovery Exploit Ledger adoption flags (task 18; Requirements 12.1, 12.3, 12.4).
        columns: Per-table column projections for chunked reads (Requirement 1.6).
        row_group_size: Parquet row-group batch size for chunked reads
            (Requirement 1.6).
        schema_version: Version identifier of the Pair_Feature_Set schema
            (Requirement 5.4).
        threshold_version: Version identifier of the calibrated thresholds
            consumed by feature engineering (Requirements 2.6, 5.4).
    """

    input_dir: Path = field(default_factory=_default_input_dir)
    output_dir: Path = field(default_factory=_default_output_dir)
    seeds: Seeds = field(default_factory=Seeds)
    exploits: ExploitConfig = field(default_factory=ExploitConfig)
    columns: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_COLUMN_SELECTION.items()}
    )
    row_group_size: int = 100_000
    schema_version: str = "pair_features_v1"
    threshold_version: str = "thresholds_v1"

    @property
    def submission_path(self) -> Path:
        """Path where submission.csv is written (Requirement 9.5)."""
        return self.output_dir / "submission.csv"

    @property
    def eda_artifact_path(self) -> Path:
        """Path of the versioned EDA / threshold artifact (Requirement 2.6)."""
        return self.output_dir / "eda_artifact.json"


def get_config() -> PipelineConfig:
    """Return a fresh :class:`PipelineConfig` with environment-appropriate paths."""
    return PipelineConfig()
