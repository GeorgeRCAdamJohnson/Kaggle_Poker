"""poker_collusion.tuning_harness: layered detection + LB-calibrated tuning harness.

A thin orchestration layer that REUSES the existing :mod:`poker_collusion`
package (metric reference, group-by-pool CV harness, PU ranker, feature caches,
submission writer) and wraps its pieces as toggleable layers, a floor-guaranteed
composer, an adversarial Drift_Gate, an anchor-calibrated threshold, a
table-disjoint holdout scorer, sample-size shrinkage, an LB projection, and a
guarded submission gate. Nothing here rebuilds those substrate components.

This package establishes the scaffolding for the ``poker-layered-tuning`` spec.
The sub-packages mirror the design's module names and are filled in by later
tasks:

- ``layers``      - layer registry + known-good-floor config
- ``composition`` - floor-guaranteed composer (RANK_BLEND / RETRAIN)
- ``gate``        - adversarial Drift_Gate + anchor calibrator
- ``holdout``     - table-disjoint PU-stress AP scorer
- ``ablation``    - leave-one-out layer contribution
- ``tuning``      - two-level tuner (drift -> baseline -> layers)
- ``shrinkage``   - empirical-Bayes sample-size shrinkage
- ``projection``  - LB projection with enforced fit range (no extrapolation)
- ``submission``  - submission gate, anchor recording, guarded floor writer

The shared data models and the append-only LB_Anchor store access layer live at
the package root (``models``, ``anchor_store``) so every sub-package imports them
from one place.

Requirements: 7.1 (submission report shape), 8.7 (externally-auditable anchor
store).
"""

from __future__ import annotations

from poker_collusion.tuning_harness.anchor_store import (
    DEFAULT_ANCHOR_STORE_PATH,
    AnchorStoreError,
    append_anchor,
    load_anchors,
    load_store,
)
from poker_collusion.tuning_harness.models import (
    Calibration,
    Candidate,
    CandidateConfig,
    CompositionSpec,
    DriftResult,
    FeatureFrame,
    LBAnchor,
    ModelConfig,
    Projection,
    ProjectionFit,
    RankVector,
    SeparationCheck,
    SubmissionReport,
    TunedConfig,
)

__all__ = [
    # Data models
    "CompositionSpec",
    "ModelConfig",
    "CandidateConfig",
    "Candidate",
    "RankVector",
    "FeatureFrame",
    "LBAnchor",
    "SeparationCheck",
    "ProjectionFit",
    "Projection",
    "Calibration",
    "DriftResult",
    "TunedConfig",
    "SubmissionReport",
    # Anchor store access
    "DEFAULT_ANCHOR_STORE_PATH",
    "AnchorStoreError",
    "load_store",
    "load_anchors",
    "append_anchor",
]
