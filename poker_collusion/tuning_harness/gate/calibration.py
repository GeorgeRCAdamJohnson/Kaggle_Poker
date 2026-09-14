"""Calibrator - turn the LB_Anchor store into a trusted or provisional gate.

The Calibrator (design "Calibrator - ``gate/calibration.py``", task 7.1) is the
piece that fixes the motivating bug: a *fixed* 0.65 drift bar wrongly rejected
the LB-VERIFIED foundation (``bq+dir``, whole-stack drift ~0.679, real LB
0.44519) while it was calibrated only on single-feature adds. Instead of guessing
a constant, we derive the deciding threshold from REAL leaderboard outcomes - the
``LB_Anchor`` store - and only trust it when those anchors actually separate the
"held / improved" configs from the "regressed" configs.

Two functions, exactly as the design names them:

- :func:`clean_separation_check` - split the anchors into PASS (``real_lb >=
  floor_lb`` - held or improved vs the Known_Good_Floor) and FAIL (regressed),
  compute ``max_pass_drift`` and ``min_fail_drift``, and declare the set
  ``separable`` IFF ``max_pass_drift < min_fail_drift`` STRICTLY (Req 1.4). This
  is the validity precondition: without a strict gap no single drift threshold
  can discriminate the two classes.

- :func:`calibrate` - produce a :class:`Calibration`:

    * **not separable** (the PASS and FAIL drift ranges overlap) -> the gate is
      INVALID: ``trusted = False``, threshold kind ``PROVISIONAL``, no trusted
      threshold, and ``null_surfaced = True`` (Req 1.5 - honestly surface that
      drift does NOT cleanly separate LB outcomes; contract Rule 3 - a null is a
      result).
    * **separable AND >= 3 anchors** -> ``LB_CALIBRATED``: place the threshold in
      the OPEN gap ``(max_pass_drift, min_fail_drift)`` at the MIDPOINT so every
      PASS anchor classifies PASS and every FAIL anchor classifies FAIL by
      construction (Req 1.2, 1.3, 1.6, 1.8). ``trusted = True``.
    * **separable but < 3 anchors** -> ``PROVISIONAL`` (~0.65) with reduced
      confidence: ``trusted = False`` (Req 1.7).

The calibrator NEVER raises on insufficient or non-separable anchors - it always
degrades to a reduced-confidence PROVISIONAL/INVALID :class:`Calibration`
(contract Rule 3/4: report the honest null, never extrapolate a trusted bar past
what the anchors support).

Edge conventions for the separation check (kept consistent with the property
tests in tasks 7.4/7.5):

- **No FAIL anchors** -> there is no upper conflict, so ``min_fail_drift`` is
  ``+inf`` and the set is separable (any PASS drift is below ``+inf``). The
  calibrated gap is open above; the threshold is placed a fixed margin above
  ``max_pass_drift`` (clamped to a valid AUC).
- **No PASS anchors** -> there is no lower conflict, so ``max_pass_drift`` is
  ``-inf`` and the set is separable. The gap is open below; the threshold is
  placed a fixed margin below ``min_fail_drift`` (clamped).
- **Neither PASS nor FAIL anchors** (empty set) -> vacuously separable
  (``-inf < +inf``) but there is nothing to calibrate against, so calibration
  falls through to PROVISIONAL by the ``< 3 anchors`` branch.

The :class:`Calibration`, :class:`SeparationCheck`, and :class:`ProjectionFit`
data models are ALREADY defined in
:mod:`poker_collusion.tuning_harness.models`; this module reuses them and only
supplies the two functions. The projection fit is populated by reusing
:func:`~poker_collusion.tuning_harness.projection.project.fit_projection`
(task 15.1) over the same anchors.

Requirements: 1.1, 1.4, 1.5, 1.6, 1.7 (and 1.2/1.3/1.8 via the midpoint gap
placement).
"""

from __future__ import annotations

import math
from typing import Iterable, List

from poker_collusion.tuning_harness.models import (
    Calibration,
    LBAnchor,
    SeparationCheck,
)
from poker_collusion.tuning_harness.projection.project import fit_projection

__all__ = [
    "PROVISIONAL_THRESHOLD",
    "MIN_ANCHORS_FOR_TRUST",
    "OPEN_GAP_MARGIN",
    "clean_separation_check",
    "calibrate",
]

#: The historical single-feature drift bar (~0.65) used only as the reduced-
#: confidence PROVISIONAL threshold - never as a trusted bar (design: PROVISIONAL
#: fallback, Req 1.7). Mirrors ``drift_gate.PROVISIONAL_THRESHOLD`` so the gate
#: and calibrator agree on the fallback constant.
PROVISIONAL_THRESHOLD: float = 0.65

#: The Clean_Separation_Check must pass AND at least this many anchors must exist
#: before an LB_Calibrated_Threshold is trusted (design Req 1.6).
MIN_ANCHORS_FOR_TRUST: int = 3

#: Margin used to place a calibrated threshold when the open gap is unbounded on
#: one side (no PASS or no FAIL anchors). Kept small so the placed threshold sits
#: just past the single-sided boundary while remaining a valid AUC in [0, 1].
OPEN_GAP_MARGIN: float = 0.01


def clean_separation_check(
    anchors: Iterable[LBAnchor], floor_lb: float
) -> SeparationCheck:
    """Compute the Clean_Separation_Check over the anchor set (Req 1.4).

    Splits ``anchors`` into PASS (``real_lb >= floor_lb`` - held/improved vs the
    Known_Good_Floor) and FAIL (``real_lb < floor_lb`` - regressed), then reports
    ``max_pass_drift``, ``min_fail_drift``, and ``separable``.

    Separability is STRICT: ``separable`` is True IFF ``max_pass_drift <
    min_fail_drift``. A tie (``max_pass_drift == min_fail_drift``) is NOT
    separable - a PASS and a FAIL config sit at the same drift, so no single
    threshold can put one PASS and the other FAIL.

    Edge conventions (documented on the module; consistent with tasks 7.4/7.5):

    - No FAIL anchors -> ``min_fail_drift = +inf`` (no upper conflict) -> separable.
    - No PASS anchors -> ``max_pass_drift = -inf`` (no lower conflict) -> separable.
    - Empty anchor set -> ``-inf < +inf`` -> vacuously separable (calibration then
      falls back to PROVISIONAL on the anchor-count branch).

    Returns:
        A :class:`SeparationCheck` carrying the two boundary drifts and the strict
        separability verdict.
    """

    pass_drifts: List[float] = []
    fail_drifts: List[float] = []
    for anchor in anchors:
        if anchor.real_lb >= floor_lb:
            pass_drifts.append(float(anchor.measured_drift))
        else:
            fail_drifts.append(float(anchor.measured_drift))

    # No PASS anchors -> no lower conflict; -inf keeps the strict-order predicate
    # honest (any finite min_fail_drift is above it).
    max_pass_drift = max(pass_drifts) if pass_drifts else float("-inf")
    # No FAIL anchors -> no upper conflict; +inf keeps the predicate honest.
    min_fail_drift = min(fail_drifts) if fail_drifts else float("inf")

    separable = max_pass_drift < min_fail_drift

    return SeparationCheck(
        max_pass_drift=max_pass_drift,
        min_fail_drift=min_fail_drift,
        separable=separable,
    )


def _clamp_auc(value: float) -> float:
    """Clamp a threshold to the valid orientation-max AUC range ``[0.5, 1.0]``.

    Adversarial AUC is orientation-folded to ``[0.5, 1.0]`` (see
    ``drift_gate._orientation_max_auc``), so a calibrated threshold outside that
    range would be meaningless. Clamping keeps a single-sided open-gap placement
    inside the range the gate actually measures.
    """

    return float(min(1.0, max(0.5, value)))


def _placed_threshold(sep: SeparationCheck) -> float:
    """Place a threshold inside the open gap ``(max_pass_drift, min_fail_drift)``.

    - Both bounds finite -> the MIDPOINT (design: "midpoint by default"), which by
      construction classifies every PASS anchor PASS and every FAIL anchor FAIL.
    - Only the upper bound finite (no PASS anchors) -> a fixed margin below
      ``min_fail_drift``.
    - Only the lower bound finite (no FAIL anchors) -> a fixed margin above
      ``max_pass_drift``.

    The result is clamped to the valid ``[0.5, 1.0]`` AUC range for the
    single-sided cases (the finite-both midpoint already lies between two valid
    AUCs).
    """

    lo = sep.max_pass_drift
    hi = sep.min_fail_drift
    lo_finite = math.isfinite(lo)
    hi_finite = math.isfinite(hi)

    if lo_finite and hi_finite:
        return (lo + hi) / 2.0
    if hi_finite:  # no PASS anchors: gap open below -> just under min_fail_drift
        return _clamp_auc(hi - OPEN_GAP_MARGIN)
    if lo_finite:  # no FAIL anchors: gap open above -> just over max_pass_drift
        return _clamp_auc(lo + OPEN_GAP_MARGIN)
    # Both infinite (empty anchor set): no data to place against; the caller only
    # reaches this via a branch that will not trust it. Use the provisional bar.
    return PROVISIONAL_THRESHOLD


def calibrate(anchors: Iterable[LBAnchor], floor_lb: float) -> Calibration:
    """Turn the anchors into a trusted or provisional :class:`Calibration`.

    Branches (design + Req 1.5/1.6/1.7):

    - **not separable** (PASS/FAIL drift ranges overlap) -> gate INVALID:
      ``trusted = False``, ``threshold_kind = "PROVISIONAL"``, threshold =
      :data:`PROVISIONAL_THRESHOLD` (NOT trusted), ``null_surfaced = True`` - the
      honest null that drift does not cleanly separate LB outcomes (Req 1.5,
      contract Rule 3).
    - **separable AND >= 3 anchors** -> ``LB_CALIBRATED``: threshold at the
      MIDPOINT of the open gap ``(max_pass_drift, min_fail_drift)``,
      ``trusted = True``, ``null_surfaced = False`` (Req 1.6, 1.8).
    - **separable but < 3 anchors** -> ``PROVISIONAL`` (~0.65) with reduced
      confidence: ``trusted = False``, ``null_surfaced = False`` (Req 1.7).

    The projection fit is populated by reusing :func:`fit_projection` over the
    same anchors (task 15.1) so the calibration carries both the gate threshold
    and the LB<-holdout_ap regime in one object; if the anchors cannot support a
    fit that helper degrades to the grounding line rather than raising.

    This function NEVER raises on insufficient/non-separable anchors - it always
    returns a valid :class:`Calibration`, degrading to reduced-confidence
    (contract Rule 3/4).

    Args:
        anchors: The LB_Anchor records (from the append-only store).
        floor_lb: The Known_Good_Floor real leaderboard score; anchors at or
            above it are PASS, below it are FAIL.

    Returns:
        A :class:`Calibration` with the deciding threshold, its kind, the trust
        flag, the :class:`SeparationCheck`, the null-surfaced flag, and the
        projection fit.
    """

    anchor_list = list(anchors)
    separation = clean_separation_check(anchor_list, floor_lb)

    # The projection fit is best-effort over the same anchors; fit_projection
    # degrades to the grounding line for 0/1/degenerate anchors and never raises.
    projection = fit_projection(anchor_list)

    if not separation.separable:
        # Gate INVALID: drift does not cleanly separate LB outcomes. Surface the
        # null; report the provisional bar but do NOT trust it (Req 1.5).
        return Calibration(
            threshold=PROVISIONAL_THRESHOLD,
            threshold_kind="PROVISIONAL",
            trusted=False,
            separation=separation,
            null_surfaced=True,
            projection=projection,
        )

    if len(anchor_list) >= MIN_ANCHORS_FOR_TRUST:
        # Separable AND enough anchors -> trusted, anchor-derived threshold in the
        # open gap (Req 1.6, 1.8). This is the fix for the motivating bug.
        return Calibration(
            threshold=_placed_threshold(separation),
            threshold_kind="LB_CALIBRATED",
            trusted=True,
            separation=separation,
            null_surfaced=False,
            projection=projection,
        )

    # Separable but too few anchors -> PROVISIONAL, reduced confidence (Req 1.7).
    return Calibration(
        threshold=PROVISIONAL_THRESHOLD,
        threshold_kind="PROVISIONAL",
        trusted=False,
        separation=separation,
        null_surfaced=False,
        projection=projection,
    )
