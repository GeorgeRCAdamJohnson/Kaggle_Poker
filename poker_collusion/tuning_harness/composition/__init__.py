"""Composer sub-package (design: ``composition/composer.py``).

Floor-guaranteed composition: ``rank_blend`` (convex combination of ranks,
weight 0 == base exactly), ``retrain`` (refit baseline on base+layer features),
and ``compose`` (all optional layers OFF reproduces the Known_Good_Floor). Also
``classify_layer`` corrupting-layer detection.

Implemented by task 5.1.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.composition.composer import (
    CONTRIBUTING,
    CORRUPTING,
    NULL,
    classify_layer,
    compose,
    rank_blend,
    retrain,
)

__all__ = [
    "CORRUPTING",
    "NULL",
    "CONTRIBUTING",
    "rank_blend",
    "retrain",
    "compose",
    "classify_layer",
]
