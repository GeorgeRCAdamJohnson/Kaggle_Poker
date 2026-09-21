"""Reusable GPU config for this workspace (dossier §79: RTX 5070 Ti + xgboost cuda).

xgboost device="cuda" is VERIFIED working and faster on the large real matrices; it also
frees system RAM (the real bottleneck, §72). LightGBM needs a GPU build (usually absent in
the pip wheel) so it stays CPU unless probed. This module centralizes the choice so every
model in the generator-recovery pipeline uses the same, auditable device settings.

Usage:
    from anchor_repro.gpu_config import xgb_device, xgb_params
    params = xgb_params(objective="binary:logistic", max_depth=6)  # includes device="cuda"
"""

from __future__ import annotations

import functools
from typing import Any, Dict


#: Verified on this laptop (2026-09-15): nvidia-smi exposes exactly ONE CUDA device at
#: index 0 = the RTX 5070 eGPU (12GB). The internal RTX 4050 Laptop GPU is NOT CUDA-visible
#: here, and XGBoost CUDA never targets the AMD 780M iGPU. So "cuda:0" is the 5070 eGPU.
#: MEASURED: 2M x 61 / 300 rounds -> CPU 26.4s vs cuda:0 4.6s (5.8x), GPU util ~70%.
#: (Dossier §79's "RTX 5070 Ti 16GB" refers to the USER'S DESKTOP, not this machine.)
_CUDA_DEVICE = "cuda:0"


@functools.lru_cache(maxsize=1)
def xgb_device() -> str:
    """Return the pinned CUDA device (the 5070 eGPU) if a fit succeeds, else "cpu" (cached)."""
    try:
        import numpy as np
        import xgboost as xgb

        X = np.random.rand(256, 4).astype("float32")
        y = (np.random.rand(256) > 0.5).astype("int32")
        xgb.train(
            {"device": _CUDA_DEVICE, "tree_method": "hist", "objective": "binary:logistic"},
            xgb.DMatrix(X, label=y),
            num_boost_round=2,
        )
        return _CUDA_DEVICE
    except Exception:
        return "cpu"


def xgb_params(**overrides: Any) -> Dict[str, Any]:
    """Baseline xgboost params with the verified device + hist tree method.

    Pass any overrides (objective, max_depth, eta, etc.). device/tree_method are set for
    you but may be overridden explicitly.
    """
    params: Dict[str, Any] = {
        "device": xgb_device(),
        "tree_method": "hist",
        "n_jobs": -1,
    }
    params.update(overrides)
    return params
