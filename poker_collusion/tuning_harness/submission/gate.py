"""Submission gate, anchor recording, guarded floor writer, and stop-loss HALT.

This module is the final decision surface of the layered-tuning harness (design
"Projection + submission gate", task 16.1). It never rebuilds a metric, split,
model, or writer -- it *sequences* the pieces the earlier tasks produced (the
Drift_Gate verdict, the calibrator, the LB_Projection, the append-only anchor
store) into the four things a submission decision needs:

1. :func:`submission_report` -- emit the COMPLETE report for a candidate proposed
   for submission (layers ON, holdout AP, measured drift, gate verdict,
   LB_Projection, regime-verified flag) and set ``recommend_submit = False``
   whenever the Drift_Gate FAILs, *regardless of holdout AP* (Req 7.1, 7.2). A
   drifting feature set's holdout is inverted (accountability contract Rule 17),
   so a high holdout number can never rescue a drift FAIL.

2. :func:`record_anchor_and_recalibrate` -- after a REAL leaderboard score
   returns, append EXACTLY ONE new :class:`LBAnchor` to the append-only store
   (reusing :func:`append_anchor`) and recalibrate the gate + projection over the
   augmented anchor set (reusing :func:`calibrate`) (Req 7.3). Only a real,
   external judge may enter the store -- the appended anchor is ``verified=True``.

3. :func:`guarded_floor_write` -- the Known_Good_Floor artifact
   (``submission_best_044519.csv``) is overwritten ONLY through a write backed by
   a leaderboard-verified improvement over the current floor score (Req 7.4). On
   refusal the artifact is left byte-for-byte unchanged and the refusal is logged
   to the dossier. The floor path is injectable so tests operate on a temp copy.

4. :func:`stop_loss_state` / :class:`StopLossState` -- the submission-history
   state machine (Gate C / Req 8.3): two CONSECUTIVE regressions versus the
   Known_Good_Floor emit a HALT signal that blocks new candidates until the
   harness itself is re-examined.

Requirements: 7.1, 7.2, 7.3, 7.4, 8.3.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from poker_collusion.tuning_harness.anchor_store import (
    DEFAULT_ANCHOR_STORE_PATH,
    append_anchor,
    load_anchors,
)
from poker_collusion.tuning_harness.gate.calibration import calibrate
from poker_collusion.tuning_harness.models import (
    Calibration,
    Candidate,
    DriftResult,
    LBAnchor,
    Projection,
    SubmissionReport,
)
from poker_collusion.tuning_harness.projection.project import fit_projection, project_lb

__all__ = [
    "KNOWN_GOOD_FLOOR",
    "DEFAULT_FLOOR_ARTIFACT_PATH",
    "DEFAULT_DOSSIER_PATH",
    "FloorWriteRefused",
    "submission_report",
    "record_anchor_and_recalibrate",
    "guarded_floor_write",
    "StopLossState",
    "stop_loss_state",
]

#: The externally-verified Known_Good_Floor leaderboard score. Anchors at or
#: above this held/improved; below it regressed. This is the same floor the
#: calibrator (task 7.1) splits PASS/FAIL against and the stop-loss compares to.
KNOWN_GOOD_FLOOR: float = 0.44519

#: Canonical location of the floor submission artifact. Injectable everywhere so
#: tests operate on a temp copy and never touch the real artifact.
DEFAULT_FLOOR_ARTIFACT_PATH: Path = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "poker_collusion"
    / "submission_best_044519.csv"
)

#: The externally-auditable dossier where a refused floor write is logged (Req
#: 7.4 / 8.7). Injectable so tests log to a temp file.
DEFAULT_DOSSIER_PATH: Path = (
    Path(__file__).resolve().parents[4]
    / ".kiro"
    / "specs"
    / "poker-collusion-detection"
    / "RESEARCH_DOSSIER.md"
)


# --------------------------------------------------------------------------- #
# 1. Submission report (Req 7.1, 7.2)
# --------------------------------------------------------------------------- #
def submission_report(
    candidate: Candidate,
    holdout_ap: float,
    drift: DriftResult,
    calib: Calibration,
) -> SubmissionReport:
    """Emit the complete submission report for a proposed candidate.

    Assembles every field Req 7.1 requires from pieces the earlier gates already
    produced -- this function computes nothing new except the LB_Projection,
    which it derives by reusing the projection fit carried on the calibration:

    - **layers ON** -- from the candidate's config (``layers_reported`` when the
      best-effort composer reporter succeeded, else the raw ``layers_on`` set so
      the report is never empty just because reporting failed).
    - **holdout AP** -- the measured, table-disjoint PU-stress AP (task 8.1),
      passed in; kept DISTINCT from the projected LB (Req 5.4 / 7.1).
    - **measured drift** -- the adversarial dev-vs-eval AUC from ``drift``.
    - **gate verdict** -- ``drift.verdict`` (``"PASS"`` | ``"FAIL"``).
    - **LB_Projection** -- ``project_lb(holdout_ap, fit)`` where ``fit`` is the
      calibration's projection fit (or a fresh fit over the seeded store if the
      calibration carried none), so the point/uncertainty come from the
      LB<-holdout_ap line and are never conflated with the holdout AP.
    - **regime-verified flag** -- mirrors the projection's ``regime_verified``
      (False when the holdout AP is outside the fit range; no extrapolation).

    ``recommend_submit`` is ``True`` ONLY when the Drift_Gate verdict is PASS.
    A drift FAIL forces ``recommend_submit = False`` **regardless of holdout AP**
    (Req 7.2, Gate C): a drifting feature set's holdout is inverted, so no
    holdout number can rescue it. When the projection regime is unverified we do
    NOT auto-recommend either -- an off-regime projection is not trustworthy
    evidence (Rule 4), so recommendation additionally requires a verified regime.

    Args:
        candidate: The assembled candidate (config + ranking) proposed for
            submission.
        holdout_ap: The measured table-disjoint PU-stress AP for this candidate.
        drift: The :class:`DriftResult` from the LB-calibrated Drift_Gate.
        calib: The current :class:`Calibration` (carries the projection fit used
            to project the leaderboard estimate).

    Returns:
        A complete :class:`SubmissionReport`.
    """

    # Layers ON: prefer the best-effort composer report; fall back to the raw
    # configured set so a reporting failure never yields an empty report.
    if candidate.layers_reported is not None:
        layers_on: Tuple[str, ...] = tuple(candidate.layers_reported)
    else:
        layers_on = tuple(sorted(candidate.config.layers_on))

    # LB_Projection from the calibrated line; reuse the calibration's fit when
    # present, else fit over the seeded store. project_lb never extrapolates.
    fit = calib.projection if calib.projection is not None else fit_projection()
    projection: Projection = project_lb(holdout_ap, fit)

    drift_passed = drift.verdict == "PASS"
    # A drift FAIL is never submittable regardless of holdout AP (Req 7.2). An
    # off-regime projection is not trustworthy evidence, so a recommendation also
    # requires the projection regime to be verified (Rule 4).
    recommend_submit = bool(drift_passed and projection.regime_verified)

    return SubmissionReport(
        layers_on=layers_on,
        holdout_ap=float(holdout_ap),
        measured_drift=float(drift.adversarial_auc),
        gate_verdict=drift.verdict,
        projection=projection,
        regime_verified=projection.regime_verified,
        recommend_submit=recommend_submit,
    )


# --------------------------------------------------------------------------- #
# 2. Record a real LB result: append one anchor + recalibrate (Req 7.3)
# --------------------------------------------------------------------------- #
def record_anchor_and_recalibrate(
    real_lb: float,
    candidate: Candidate,
    holdout_ap: float,
    measured_drift: float,
    *,
    config_id: Optional[str] = None,
    floor_lb: float = KNOWN_GOOD_FLOOR,
    store_path: Union[str, Path, None] = None,
    source: str = "leaderboard",
) -> Calibration:
    """Append exactly one verified LB_Anchor, then recalibrate over the augmented set.

    After a candidate is actually submitted and its REAL leaderboard score
    returns, this records the external result and re-derives the gate + projection
    so future decisions use the enlarged evidence base (Req 7.3). It reuses the
    append-only store's :func:`append_anchor` (which enforces verified-only,
    no-duplicate, strictly-append semantics) and the calibrator's
    :func:`calibrate` -- nothing here re-implements either.

    Exactly ONE anchor is appended: the ``(config_id, real_lb, measured_drift,
    holdout_ap)`` tuple for the just-scored candidate, marked ``verified=True``
    (only a real external judge may enter the store -- contract Rule 1).

    Args:
        real_lb: The REAL leaderboard score returned for the candidate.
        candidate: The submitted candidate (used to derive a stable ``config_id``
            when one is not supplied).
        holdout_ap: The candidate's measured holdout AP (stored on the anchor so
            it can re-enter the projection fit).
        measured_drift: The candidate's measured adversarial dev-vs-eval AUC.
        config_id: Explicit anchor id; when omitted a deterministic id is derived
            from the candidate's ON layers.
        floor_lb: The Known_Good_Floor real LB used to split PASS/FAIL when
            recalibrating (defaults to :data:`KNOWN_GOOD_FLOOR`).
        store_path: Injectable anchor-store path (tests pass a temp copy);
            defaults to the seeded store.
        source: Provenance string recorded on the anchor.

    Returns:
        The recalibrated :class:`Calibration` over the augmented anchor set (which
        now includes the just-appended anchor).

    Raises:
        AnchorStoreError: propagated from :func:`append_anchor` if the append
            would violate the append-only / verified invariants (e.g. a duplicate
            ``config_id``).
    """

    if config_id is None:
        config_id = _derive_config_id(candidate)

    anchor = LBAnchor(
        config_id=config_id,
        real_lb=float(real_lb),
        measured_drift=float(measured_drift),
        holdout_ap=float(holdout_ap),
        source=source,
        verified=True,  # a real external LB result; contract Rule 1.
    )

    # Append exactly one anchor (reuse the guarded append-only writer), then
    # recalibrate the gate + projection over the augmented set (reuse calibrate).
    augmented = append_anchor(anchor, path=store_path)
    return calibrate(augmented, floor_lb)


def _derive_config_id(candidate: Candidate) -> str:
    """Derive a deterministic anchor ``config_id`` from a candidate's ON layers.

    Foundation is always on; the id lists the optional layers ON in sorted order
    so the same stack always maps to the same id (and a re-submit of an already-
    recorded stack is correctly rejected as a duplicate by the store).
    """

    layers = sorted(candidate.config.layers_on)
    return "floor" if not layers else "floor+" + "+".join(layers)


# --------------------------------------------------------------------------- #
# 3. Guarded floor writer (Req 7.4)
# --------------------------------------------------------------------------- #
class FloorWriteRefused(Exception):
    """Raised when a floor write is not backed by a verified LB improvement.

    Carries the refusal reason; the artifact is left byte-for-byte unchanged when
    this is raised (Req 7.4).
    """


def guarded_floor_write(
    candidate_csv: Union[str, Path],
    real_lb: float,
    *,
    current_floor_lb: float = KNOWN_GOOD_FLOOR,
    floor_path: Union[str, Path, None] = None,
    dossier_path: Union[str, Path, None] = None,
    verified: bool = True,
) -> Path:
    """Overwrite the floor artifact ONLY on a leaderboard-verified improvement.

    The Known_Good_Floor artifact is the single most valuable file in the
    project; three regressions were shipped by trusting internal metrics, so this
    writer refuses ANY overwrite not backed by a REAL, external leaderboard score
    strictly above the current floor (Req 7.4, accountability contract Rule 1/7).

    A write proceeds ONLY when BOTH hold:

    - ``verified`` is True (the score came from the real leaderboard, not a
      projection -- a projected value may never overwrite the floor), AND
    - ``real_lb`` is STRICTLY greater than ``current_floor_lb`` (a tie or a
      regression is not an improvement).

    On refusal the artifact is left **byte-for-byte unchanged** and the refusal is
    appended to the dossier (externally auditable), then :class:`FloorWriteRefused`
    is raised. On acceptance the candidate CSV is copied over the floor artifact
    and the accepted write is logged.

    Args:
        candidate_csv: Path to the candidate submission CSV to promote to the
            floor on acceptance.
        real_lb: The candidate's leaderboard score.
        current_floor_lb: The floor score to beat (defaults to
            :data:`KNOWN_GOOD_FLOOR`).
        floor_path: The floor artifact path to guard/overwrite (injectable; tests
            pass a temp copy). Defaults to :data:`DEFAULT_FLOOR_ARTIFACT_PATH`.
        dossier_path: Where the accept/refuse decision is logged (injectable).
        verified: Whether ``real_lb`` is a real external result. A non-verified
            (projected) value is always refused.

    Returns:
        The floor artifact path on a successful (accepted) write.

    Raises:
        FloorWriteRefused: when the write is not backed by a verified improvement;
            the artifact is unchanged.
    """

    floor = Path(floor_path) if floor_path is not None else DEFAULT_FLOOR_ARTIFACT_PATH
    candidate = Path(candidate_csv)

    is_improvement = bool(verified) and float(real_lb) > float(current_floor_lb)

    if not is_improvement:
        if not verified:
            reason = (
                f"real_lb={real_lb!r} is not leaderboard-verified; a projected "
                "value may never overwrite the floor"
            )
        else:
            reason = (
                f"real_lb={real_lb!r} does not strictly improve on the current "
                f"floor {current_floor_lb!r}"
            )
        _log_floor_decision(
            dossier_path,
            accepted=False,
            reason=reason,
            floor=floor,
            candidate=candidate,
            real_lb=real_lb,
            current_floor_lb=current_floor_lb,
        )
        raise FloorWriteRefused(
            f"Refusing to overwrite floor artifact {floor.name}: {reason} "
            "(Req 7.4). Artifact left unchanged."
        )

    # Verified improvement: promote the candidate to the floor artifact.
    shutil.copyfile(candidate, floor)
    _log_floor_decision(
        dossier_path,
        accepted=True,
        reason=(
            f"real_lb={real_lb!r} strictly improves on floor "
            f"{current_floor_lb!r}; artifact overwritten"
        ),
        floor=floor,
        candidate=candidate,
        real_lb=real_lb,
        current_floor_lb=current_floor_lb,
    )
    return floor


def _log_floor_decision(
    dossier_path: Union[str, Path, None],
    *,
    accepted: bool,
    reason: str,
    floor: Path,
    candidate: Path,
    real_lb: float,
    current_floor_lb: float,
) -> None:
    """Append a floor accept/refuse decision to the dossier (best-effort).

    Logging is best-effort: a logging failure must never leave the guard in a
    state where the artifact was touched but the decision unrecorded, so the copy
    (or refusal) is decided by the caller and logging only records it. If the
    dossier is unwritable the guard still enforces the invariant (the artifact is
    unchanged on refusal), so a logging error is swallowed rather than masking the
    refusal.
    """

    target = Path(dossier_path) if dossier_path is not None else DEFAULT_DOSSIER_PATH
    verdict = "ACCEPTED" if accepted else "REFUSED"
    stamp = datetime.now(timezone.utc).isoformat()
    line = (
        f"\n- [{stamp}] FLOOR WRITE {verdict}: candidate={candidate.name} "
        f"real_lb={real_lb} floor={current_floor_lb} target={floor.name} -- {reason}\n"
    )
    try:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        # Never let a logging failure mask the guard's real invariant.
        pass


# --------------------------------------------------------------------------- #
# 4. Stop-loss HALT state machine (Gate C, Req 8.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StopLossState:
    """The stop-loss verdict over a submission history (Gate C, Req 8.3).

    ``halt`` is True when the history contains two CONSECUTIVE regressions versus
    the Known_Good_Floor -- at that point new candidates are BLOCKED until the
    harness itself is re-examined. ``consecutive_regressions`` is the length of
    the current trailing regression run (0 when the most recent submission held or
    improved). ``blocks_new_candidates`` mirrors ``halt`` -- when HALT is active,
    no new candidate may proceed.
    """

    halt: bool
    consecutive_regressions: int
    blocks_new_candidates: bool

    @property
    def message(self) -> str:
        """Human-readable status for the report/dossier."""

        if self.halt:
            return (
                "HALT: two consecutive submissions regressed versus the "
                "Known_Good_Floor; new candidates are blocked until the harness "
                "is re-examined (Gate C)."
            )
        return (
            f"OK: {self.consecutive_regressions} consecutive regression(s); "
            "below the two-regression HALT tripwire."
        )


def stop_loss_state(
    history: Iterable[Union[float, str, bool]],
    *,
    floor_lb: float = KNOWN_GOOD_FLOOR,
) -> StopLossState:
    """Compute the stop-loss HALT state from a submission history (Gate C).

    ``history`` is a sequence, in submission order (oldest first), of either:

    - **real LB scores** (``float``/``int``): a submission REGRESSED iff its score
      is strictly below ``floor_lb``; at or above the floor it HELD/improved.
    - **pass/regress markers**: ``"PASS"`` / ``"HOLD"`` / ``True`` (held or
      improved) vs ``"REGRESS"`` / ``"FAIL"`` / ``False`` (regressed). Strings are
      case-insensitive.

    The rule (Req 8.3): **two CONSECUTIVE regressions** versus the floor trip the
    HALT. The count is the trailing run of consecutive regressions -- a hold or
    improvement anywhere resets the run to 0 (two regressions separated by a hold
    do NOT trip HALT; only back-to-back ones do). HALT is latched as soon as the
    trailing run reaches 2.

    Args:
        history: Submission outcomes in order (scores or pass/regress markers).
        floor_lb: The Known_Good_Floor score used to classify numeric scores.

    Returns:
        A :class:`StopLossState` whose ``halt`` / ``blocks_new_candidates`` are
        True iff two consecutive regressions occurred.
    """

    regress_flags = [_is_regression(item, floor_lb) for item in history]

    # Length of the current trailing run of consecutive regressions.
    trailing = 0
    for is_regress in reversed(regress_flags):
        if is_regress:
            trailing += 1
        else:
            break

    # HALT trips the moment ANY two consecutive regressions appear anywhere in the
    # history (latched) -- not only at the tail -- so a hold AFTER two consecutive
    # regressions does not silently un-HALT a run that already tripped Gate C.
    max_run = 0
    run = 0
    for is_regress in regress_flags:
        run = run + 1 if is_regress else 0
        max_run = max(max_run, run)

    halt = max_run >= 2
    return StopLossState(
        halt=halt,
        consecutive_regressions=trailing,
        blocks_new_candidates=halt,
    )


def _is_regression(item: Union[float, str, bool], floor_lb: float) -> bool:
    """Classify one history item as a regression versus the floor.

    Numeric score -> regression iff strictly below ``floor_lb``. Marker -> the
    explicit regress/hold label (case-insensitive for strings). ``bool`` is
    checked before the numeric branch because ``bool`` is a subclass of ``int``:
    ``True`` means held, ``False`` means regressed.
    """

    if isinstance(item, bool):
        return not item
    if isinstance(item, (int, float)):
        return float(item) < float(floor_lb)
    if isinstance(item, str):
        token = item.strip().lower()
        if token in {"regress", "regression", "fail", "failed", "worse", "down"}:
            return True
        if token in {"pass", "hold", "held", "improve", "improved", "better", "up", "ok"}:
            return False
        raise ValueError(f"Unrecognized submission-history marker: {item!r}")
    raise TypeError(f"Unsupported submission-history item type: {type(item)!r}")
