"""Layer registry sub-package (design: ``layers/registry.py``).

Defines the frozen ``Layer`` block, the always-on ``FOUNDATION``, the optional
toggleable layers (board_equity, directional, iso, whipsaw, negative_space), and
``known_good_floor_config()`` - the exact LB-verified 0.44519 stack. Layers
reference keys into the existing feature caches (reuse, not recompute).

Implemented by task 4.1 in ``registry.py``; re-exported here for convenience.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.layers.registry import (
    ALL_LAYERS,
    DEFAULT_RANK_BLEND_WEIGHT,
    FLOOR_LAYERS,
    FOUNDATION,
    FOUNDATION_NAME,
    OPTIONAL_LAYERS,
    RANK_BLEND,
    RETRAIN,
    V5_KEYS,
    Layer,
    get_layer,
    known_good_floor_config,
    optional_layer_names,
)

__all__ = [
    "Layer",
    "RANK_BLEND",
    "RETRAIN",
    "FOUNDATION_NAME",
    "V5_KEYS",
    "FOUNDATION",
    "OPTIONAL_LAYERS",
    "ALL_LAYERS",
    "FLOOR_LAYERS",
    "DEFAULT_RANK_BLEND_WEIGHT",
    "get_layer",
    "optional_layer_names",
    "known_good_floor_config",
]
