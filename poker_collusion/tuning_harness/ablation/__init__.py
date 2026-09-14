"""Ablation sub-package (design: ``ablation/leave_one_out.py``).

Leave-one-out layer contribution (full holdout minus holdout without the layer),
removal recommendations for non-positive contributions, and the leanest subset
within tolerance of the best. Reuses the holdout scorer via an injected
``score_fn`` so the arithmetic is unit-testable without the real CV run.

Implemented in :mod:`poker_collusion.tuning_harness.ablation.leave_one_out` (task 12.1).
"""

from __future__ import annotations

from poker_collusion.tuning_harness.ablation.leave_one_out import (
    AblationResult,
    ScoreFn,
    ablate,
)

__all__ = [
    "ablate",
    "AblationResult",
    "ScoreFn",
]
