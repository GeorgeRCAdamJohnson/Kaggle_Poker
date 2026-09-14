"""Shared data models for the poker-layered-tuning harness.

This module defines the immutable data models declared in the layered-tuning
design ("Components and Interfaces" and "Data Models" sections). The harness is a
thin orchestration layer over the existing :mod:`poker_collusion` package; these
models are the shared vocabulary the later-task modules (registry, composer,
drift gate, calibrator, holdout scorer, shrinkage, projection, submission gate)
import from a single place.

Everything here is a frozen dataclass so candidates and configs are hashable,
comparable, and cannot be mutated after assembly. ``RankVector`` and
``FeatureFrame`` are deliberately thin, read-only *views* keyed by ``pair_id``
over the existing feature caches (``poker_collusion.features.feature_cache``);
they do NOT recompute or own feature data.

Only the data models are defined here. Registry / composer / gate / calibrator /
tuner / shrinkage / projection / submission behaviour is filled in by later
tasks and imports these types.

Requirements:
- 7.1: The submission report distinguishes measured holdout AP from projected
  leaderboard score with uncertainty, plus layers ON, measured drift, gate
  verdict, and the regime-verified flag. ``SubmissionReport`` / ``Projection``
  carry those fields.
- 8.7: The audit trail (LB_Anchor store, calibration provenance) stays
  externally auditable; the models here are the serialisable tuples that store
  records. ``LBAnchor`` mirrors the append-only ``lb_anchors.json`` schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

__all__ = [
    # Configuration + candidate assembly
    "CompositionSpec",
    "ModelConfig",
    "CandidateConfig",
    "Candidate",
    # Cache views (keyed by pair_id)
    "RankVector",
    "FeatureFrame",
    # Calibration / gate models
    "LBAnchor",
    "SeparationCheck",
    "ProjectionFit",
    "Projection",
    "Calibration",
    "DriftResult",
    # Tuning + reporting
    "TunedConfig",
    "SubmissionReport",
]


# --------------------------------------------------------------------------- #
# Cache views (thin, read-only, keyed by pair_id)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RankVector:
    """A per-``pair_id`` risk ordering.

    A thin, immutable view over an ordering produced from the existing feature
    caches / models. ``pair_ids`` is the ordering itself (index 0 = highest
    risk); ``scores`` is an optional parallel tuple of the underlying risk
    scores in the same order. Nothing here recomputes features - it only carries
    the ordering keyed by ``pair_id`` (design: "``RankVector`` is a per-``pair_id``
    ordering").
    """

    pair_ids: Tuple[str, ...]
    scores: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.scores and len(self.scores) != len(self.pair_ids):
            raise ValueError(
                "RankVector.scores, when provided, must align 1:1 with pair_ids "
                f"(got {len(self.scores)} scores for {len(self.pair_ids)} pairs)."
            )


@dataclass(frozen=True)
class FeatureFrame:
    """A thin, read-only view over the existing feature caches keyed by ``pair_id``.

    ``feature_keys`` names the columns (keys into the existing feature caches -
    reuse, not recompute). ``rows`` maps each ``pair_id`` to its feature values
    in ``feature_keys`` order. ``is_eval`` (optional) flags which rows are
    evaluation-pool rows, used by the Drift_Gate's dev-vs-eval adversarial
    classifier in a later task. The mappings are wrapped read-only to preserve
    the "view, not owner" contract.
    """

    feature_keys: Tuple[str, ...]
    rows: Mapping[str, Tuple[float, ...]]
    is_eval: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", MappingProxyType(dict(self.rows)))
        object.__setattr__(self, "is_eval", MappingProxyType(dict(self.is_eval)))


# --------------------------------------------------------------------------- #
# Configuration + candidate assembly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompositionSpec:
    """How a single optional layer is composed onto the base.

    ``mode`` is ``"RANK_BLEND"`` (convex combination of ranks) or ``"RETRAIN"``
    (refit the baseline on base+layer features). ``weight`` is the RANK_BLEND
    weight in [0, 1]; ``0.0`` is allowed and MUST reproduce the base exactly
    (design Req 2.2). Composer behaviour is implemented in a later task.
    """

    mode: str  # "RANK_BLEND" | "RETRAIN"
    weight: float  # RANK_BLEND weight in [0, 1]; 0 allowed


@dataclass(frozen=True)
class ModelConfig:
    """Baseline model hyperparameters + PU weights.

    Stub carrier for the reused baseline risk model's knobs (design references
    ``ModelConfig`` in ``composition/composer.py`` and ``tuning/tuner.py``). The
    concrete grids and defaults are supplied by later tuning tasks; this frozen
    view just carries them so composition/tuning have a typed handle.
    """

    hyperparameters: Mapping[str, float] = field(default_factory=dict)
    pu_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hyperparameters", MappingProxyType(dict(self.hyperparameters)))
        object.__setattr__(self, "pu_weights", MappingProxyType(dict(self.pu_weights)))


@dataclass(frozen=True)
class CandidateConfig:
    """A fully-specified harness configuration.

    ``layers_on`` names the optional layers switched ON (foundation is always on
    and not listed here). ``composition`` maps each ON layer to its
    :class:`CompositionSpec`. ``model_cfg`` carries baseline knobs. ``shrinkage``
    maps each layer to its empirical-Bayes ``k``. All-optional-layers-OFF must
    reproduce the Known_Good_Floor exactly (design Req 2.1).
    """

    layers_on: frozenset[str]
    composition: Mapping[str, CompositionSpec] = field(default_factory=dict)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    shrinkage: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "composition", MappingProxyType(dict(self.composition)))
        object.__setattr__(self, "shrinkage", MappingProxyType(dict(self.shrinkage)))


@dataclass(frozen=True)
class Candidate:
    """An assembled candidate: a config plus its produced ranking.

    ``layers_reported`` is best-effort (design Req 2.3/2.4): a reporting failure
    yields ``None`` and never blocks assembly.
    """

    config: CandidateConfig
    ranking: RankVector
    layers_reported: Tuple[str, ...] | None = None


# --------------------------------------------------------------------------- #
# Calibration / gate models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LBAnchor:
    """One (config, real_lb, measured_drift, holdout_ap) leaderboard anchor.

    Mirrors the append-only ``lb_anchors.json`` schema. An anchor is "real" only
    when confirmed by the external leaderboard judge (accountability contract
    Rule 1); ``verified`` records that. Used to calibrate the Drift_Gate and the
    LB_Projection (design: Calibrator).
    """

    config_id: str
    real_lb: float
    measured_drift: float  # adversarial dev-vs-eval AUC
    holdout_ap: float  # PU-stress AP on the table-disjoint holdout
    source: str = ""
    verified: bool = False


@dataclass(frozen=True)
class SeparationCheck:
    """Result of the Clean_Separation_Check over the anchor set.

    ``separable`` is True IFF the maximum drift among PASS anchors is strictly
    less than the minimum drift among FAIL anchors (design Req 1.4). Populated by
    the calibrator in a later task.
    """

    max_pass_drift: float
    min_fail_drift: float
    separable: bool


@dataclass(frozen=True)
class ProjectionFit:
    """The LB <- PU-stress-AP linear fit and the regime it is valid within.

    Slope ~1.4235 / intercept ~-0.1414 / resid_std ~0.0022 (design). ``fit_min_ap``
    / ``fit_max_ap`` record the sampled AP range; projecting outside it is banned
    (accountability contract Rule 4 - no extrapolation).
    """

    slope: float
    intercept: float
    resid_std: float
    fit_min_ap: float
    fit_max_ap: float
    n_points: int


@dataclass(frozen=True)
class Projection:
    """A projected leaderboard score with honest uncertainty.

    ``regime_verified`` is True only when the source holdout AP lay within the
    fit range (design Property 23 / Rule 4). Kept distinct from the measured
    holdout AP so measured and projected values are never conflated (Req 7.1).
    """

    point: float
    uncertainty: float
    regime_verified: bool


@dataclass(frozen=True)
class Calibration:
    """The calibrated (or provisional) Drift_Gate threshold + projection fit.

    ``threshold_kind`` is ``"LB_CALIBRATED"`` or ``"PROVISIONAL"``. ``trusted`` is
    True only when the separation check passes AND there are >=3 anchors (design
    Req 1.6/1.7). ``null_surfaced`` is set when drift does NOT separate LB
    outcomes (Req 1.5). Populated by the calibrator in a later task.
    """

    threshold: float
    threshold_kind: str  # "LB_CALIBRATED" | "PROVISIONAL"
    trusted: bool
    separation: SeparationCheck
    null_surfaced: bool
    projection: ProjectionFit | None = None


@dataclass(frozen=True)
class DriftResult:
    """Verdict of the adversarial dev-vs-eval Drift_Gate for one feature set.

    ``adversarial_auc`` is the dev-vs-eval classifier AUC; a high value means the
    features describe "which pool" (drift) rather than "is collusion".
    ``threshold_kind`` records whether the deciding threshold was LB-calibrated
    or provisional. Populated by the drift gate in a later task.
    """

    adversarial_auc: float
    verdict: str  # "PASS" | "FAIL"
    threshold: float
    threshold_kind: str  # "LB_CALIBRATED" | "PROVISIONAL"


# --------------------------------------------------------------------------- #
# Tuning + reporting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TunedConfig:
    """A tuned configuration plus its measured/projected outcomes.

    Keeps ``holdout_ap`` (measured, internal) distinct from ``projection``
    (projected LB with uncertainty) so the two are never conflated (design Req
    5.4 / Property 16). Populated by the two-level tuner in a later task.
    """

    config: CandidateConfig
    holdout_ap: float
    drift: DriftResult
    projection: Projection


@dataclass(frozen=True)
class SubmissionReport:
    """The complete report emitted for a candidate proposed for submission.

    Contains every field Req 7.1 requires: layers ON, holdout AP, measured
    drift, gate verdict, LB_Projection, and the regime-verified flag.
    ``recommend_submit`` is False whenever the Drift_Gate FAILs regardless of
    holdout AP (Req 7.2). Populated by the submission gate in a later task.
    """

    layers_on: Tuple[str, ...]
    holdout_ap: float
    measured_drift: float
    gate_verdict: str
    projection: Projection
    regime_verified: bool
    recommend_submit: bool
