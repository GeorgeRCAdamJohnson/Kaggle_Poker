"""Holdout scorer sub-package (design: ``holdout/scorer.py``).

Wraps the existing group-by-``table_id`` CV (``validation/cv.py``) and
``reference_public_metric.py`` (via the numerically identical
``production_metric.score_components``) to produce PU-stress AP on the 40%
table-disjoint, pair-disjoint split. Reuse, do not reimplement the split or
metric.

Implemented by task 8.1.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.holdout.scorer import (
    HoldoutScore,
    holdout_pair_ap,
    ranking_predict_fn,
)

__all__ = [
    "HoldoutScore",
    "holdout_pair_ap",
    "ranking_predict_fn",
]
