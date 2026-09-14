"""LB projection sub-package (design: ``projection/project.py``).

``ProjectionFit`` (slope ~1.4235, intercept ~-0.1414, resid_std ~0.0022, fit
range) and ``project_lb`` which flags any holdout AP outside the recorded fit
range as ``regime_verified = False`` - never extrapolating (accountability
contract Rule 4).

Implemented in ``project.py`` (task 15.1): ``fit_projection`` builds the fit
from the LB_Anchors, ``project_lb`` projects a holdout AP with an honest,
regime-verified flag.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.projection.project import (
    GROUNDING_INTERCEPT,
    GROUNDING_RESID_STD,
    GROUNDING_SLOPE,
    fit_projection,
    project_lb,
)

__all__ = [
    "GROUNDING_SLOPE",
    "GROUNDING_INTERCEPT",
    "GROUNDING_RESID_STD",
    "fit_projection",
    "project_lb",
]
