"""Submission gate sub-package (design: ``submission/gate.py``).

``submission_report`` (complete report; ``recommend_submit = False`` on a drift
FAIL), ``record_anchor_and_recalibrate`` (append one anchor + recalibrate), the
guarded floor writer (never overwrites ``submission_best_044519.csv`` without a
verified improvement), and the two-consecutive-regression HALT state machine.

Implemented by task 16.1 in ``gate.py``.
"""

from __future__ import annotations

from poker_collusion.tuning_harness.submission.gate import (
    FloorWriteRefused,
    KNOWN_GOOD_FLOOR,
    StopLossState,
    guarded_floor_write,
    record_anchor_and_recalibrate,
    stop_loss_state,
    submission_report,
)

__all__: list[str] = [
    "KNOWN_GOOD_FLOOR",
    "FloorWriteRefused",
    "StopLossState",
    "submission_report",
    "record_anchor_and_recalibrate",
    "guarded_floor_write",
    "stop_loss_state",
]
