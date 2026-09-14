"""Gate sub-package (design: ``gate/drift_gate.py``, ``gate/calibration.py``).

The adversarial dev-vs-eval Drift_Gate and the anchor-driven Calibrator
(Clean_Separation_Check, LB_Calibrated_Threshold, null surfacing). Drift PASS
first, then holdout becomes a trustworthy LB predictor.

``drift_gate.py`` (task 6.1) is implemented; ``calibration.py`` (task 7.1)
produces the :class:`Calibration` the gate consumes.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.gate.calibration import (
    MIN_ANCHORS_FOR_TRUST,
    OPEN_GAP_MARGIN,
    calibrate,
    clean_separation_check,
)
from poker_collusion.tuning_harness.gate.drift_gate import (
    PROVISIONAL_THRESHOLD,
    adversarial_auc,
    evaluate_drift,
)
from poker_collusion.tuning_harness.gate.sequence import (
    GateSequenceResult,
    evaluate_retrain,
    evaluate_sequence,
)

__all__ = [
    "PROVISIONAL_THRESHOLD",
    "adversarial_auc",
    "evaluate_drift",
    "MIN_ANCHORS_FOR_TRUST",
    "OPEN_GAP_MARGIN",
    "calibrate",
    "clean_separation_check",
    "GateSequenceResult",
    "evaluate_sequence",
    "evaluate_retrain",
]
