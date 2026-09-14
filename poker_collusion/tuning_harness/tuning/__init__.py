"""Two-level tuner sub-package (design: ``tuning/tuner.py``).

Tunes in strict order - resolve feature-set drift, then Baseline_Tuning, then
Layer_Tuning - on coarse structural grids, enforcing the two independent
guarantees (selected config clears the Drift_Gate; a tuned gain that increases
drift past the threshold is rejected). Reports the LB_Projection with
uncertainty.

Skeleton only: ``tuner.py`` is implemented by task 13.1.
"""

from __future__ import annotations

__all__: list[str] = []
