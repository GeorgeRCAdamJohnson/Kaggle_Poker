"""Sample-size shrinkage sub-package (design: ``shrinkage/eb.py``).

Empirical-Bayes shrinkage ``prior + (value - prior) * n / (n + k)`` (monotone in
the shared-hand count ``n``) and an exhaustive shrinkage-constant sweep whose
choice is verified against the population median shared-hand count.

Implemented in :mod:`poker_collusion.tuning_harness.shrinkage.eb` (task 11.1).
"""

from __future__ import annotations

from poker_collusion.tuning_harness.shrinkage.eb import (
    KEvaluation,
    PairStat,
    ShrinkageChoice,
    ShrinkageLayer,
    shrink,
    sweep_shrinkage,
)

__all__ = [
    "shrink",
    "sweep_shrinkage",
    "PairStat",
    "ShrinkageLayer",
    "KEvaluation",
    "ShrinkageChoice",
]
