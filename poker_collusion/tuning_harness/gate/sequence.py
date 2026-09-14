"""The drift -> holdout gate sequence (design: "The gate sequence", task 9.1).

Gates run **in sequence, not in parallel** (design "The gate sequence",
accountability contract Rule 17): the table-disjoint holdout AP is a trustworthy
leaderboard predictor ONLY *after* the candidate's feature set clears the
adversarial Drift_Gate. This module wires that ordering so the rest of the
harness (projection, submission gate, tuner) can ask one question -- "is this
candidate's holdout number trustworthy?" -- and get the sequenced answer.

Why the sequence matters (Rule 17, dossier §17/§29). A feature set that drifts
dev->eval describes "which pool a row came from", not "is this pair colluding".
Its holdout AP is then *inverted* -- it rewards dev-only signal that will not
transfer to the leaderboard -- so a HIGH holdout number on a drift-FAILing
feature set is a trap, not evidence. The gate is therefore a **gatekeeper, not a
vote**: we do not average a good holdout against a bad drift; a drift FAIL
disqualifies the candidate and its holdout is explicitly marked NOT trustworthy.

Two entry points, mirroring the design:

``evaluate_sequence(feature_frame, calib, ...)``
    Run the Drift_Gate on a candidate's feature set FIRST, then report whether
    its holdout is trustworthy. Returns a :class:`GateSequenceResult` carrying
    the drift verdict and the ``holdout_trustworthy`` flag. ``holdout_trustworthy
    is True`` **iff** the drift verdict is PASS (Rule 17).

``evaluate_retrain(base_features, layer_features, calib, ...)``
    A RETRAIN composition is trusted ONLY when the COMBINED (base + layer)
    feature set clears the Drift_Gate (Requirement 4.2). This runs the sequence
    on the column-concatenated combined frame and sets ``retrain_trusted`` iff
    that combined feature set PASSes. A RETRAIN whose combined features FAIL is
    never trusted, regardless of any holdout gain.

This module only *sequences* existing pieces -- it reuses
:func:`poker_collusion.tuning_harness.gate.drift_gate.evaluate_drift` and the
composer's combined-feature assembly. It never derives a threshold itself (that
is the calibrator, task 7.1) and never scores the holdout itself (that is the
holdout scorer, task 8.1); it decides *whether the holdout may be trusted*.

Requirements: 4.2 (a RETRAIN result is trusted only if its combined feature set
passes the Drift_Gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from poker_collusion.tuning_harness.composition.composer import _combine_features
from poker_collusion.tuning_harness.gate.drift_gate import evaluate_drift
from poker_collusion.tuning_harness.models import (
    Calibration,
    DriftResult,
    FeatureFrame,
)

__all__ = [
    "GateSequenceResult",
    "evaluate_sequence",
    "evaluate_retrain",
]


@dataclass(frozen=True)
class GateSequenceResult:
    """Outcome of running the drift -> holdout sequence on one candidate.

    ``drift`` is the full :class:`DriftResult` from the Drift_Gate (the FIRST and
    deciding gate). ``holdout_trustworthy`` is the sequenced verdict: the
    table-disjoint holdout AP is a trustworthy leaderboard predictor for this
    candidate **iff** the drift verdict is PASS (accountability contract Rule 17;
    a drift FAIL inverts trust in the holdout). ``retrain_trusted`` is populated
    only by :func:`evaluate_retrain`: it is ``True`` iff the COMBINED base+layer
    feature set passed the Drift_Gate (Requirement 4.2), ``None`` for a plain
    (non-RETRAIN) sequence evaluation where the notion does not apply.
    """

    drift: DriftResult
    holdout_trustworthy: bool
    retrain_trusted: Optional[bool] = None

    @property
    def drift_passed(self) -> bool:
        """``True`` iff the Drift_Gate verdict is PASS (the sequence gatekeeper)."""

        return self.drift.verdict == "PASS"


def evaluate_sequence(
    feature_frame: FeatureFrame,
    calib: Calibration,
    *,
    groups: Optional[Mapping[str, object]] = None,
    n_folds: int = 5,
    seed: int = 0,
) -> GateSequenceResult:
    """Run Drift_Gate first, then report whether the holdout may be trusted.

    This is the core of the design's "gate sequence": the Drift_Gate is the
    external judge that runs BEFORE the (internal) holdout is believed. We
    compute the drift verdict for ``feature_frame`` and derive
    ``holdout_trustworthy`` from it -- it is ``True`` **only** when the drift
    verdict is PASS. On a drift FAIL the holdout number for this candidate is
    explicitly NOT trustworthy (its signal is inverted -- it rewards dev-only
    features, Rule 17), so downstream code must not treat the holdout AP as a
    leaderboard predictor.

    The gate is a gatekeeper, not a vote: we never combine a good holdout with a
    bad drift; the drift verdict alone decides trust.

    Args:
        feature_frame: The candidate's feature set (rows keyed by ``pair_id``,
            dev-vs-eval labels in ``is_eval``).
        calib: The calibration produced by ``gate/calibration.py`` (task 7.1);
            consumed to pick the deciding threshold, never re-derived here.
        groups: Optional ``{pair_id: pool}`` map threaded to the adversarial CV so
            whole pools stay together (design note: the ``FeatureFrame`` carries
            no pool column, so the pool map is threaded through here when
            available).
        n_folds: Grouped-CV fold count forwarded to the Drift_Gate.
        seed: Estimator seed forwarded to the Drift_Gate.

    Returns:
        A :class:`GateSequenceResult` with the drift verdict and the sequenced
        ``holdout_trustworthy`` flag (``retrain_trusted`` left ``None``).
    """

    drift = evaluate_drift(
        feature_frame, calib, groups=groups, n_folds=n_folds, seed=seed
    )
    holdout_trustworthy = drift.verdict == "PASS"
    return GateSequenceResult(
        drift=drift,
        holdout_trustworthy=holdout_trustworthy,
        retrain_trusted=None,
    )


def evaluate_retrain(
    base_features: FeatureFrame,
    layer_features: FeatureFrame,
    calib: Calibration,
    *,
    groups: Optional[Mapping[str, object]] = None,
    n_folds: int = 5,
    seed: int = 0,
) -> GateSequenceResult:
    """Decide whether a RETRAIN composition may be trusted (Requirement 4.2).

    A RETRAIN layer is added as features and the baseline is refit on the
    COMBINED base+layer set. Per Req 4.2 the refit result is trusted ONLY when
    that combined feature set clears the LB-calibrated Drift_Gate -- a RETRAIN
    that improves the holdout but whose combined features DRIFT is never trusted
    (the holdout is inverted on drifting features, Rule 17).

    We column-concatenate ``base_features`` and ``layer_features`` over their
    shared pairs (reusing the composer's :func:`_combine_features`, so the
    combined frame here is byte-identical to the one a real refit would see),
    then run the drift -> holdout sequence on that combined frame.
    ``retrain_trusted`` and ``holdout_trustworthy`` are both set iff the combined
    feature set PASSes the Drift_Gate.

    Args:
        base_features: The foundation feature view (keyed by ``pair_id``).
        layer_features: The RETRAIN layer's feature view (keyed by ``pair_id``);
            its columns are appended to the base's for pairs present in both.
        calib: The calibration consumed by the Drift_Gate.
        groups: Optional ``{pair_id: pool}`` pool map threaded to the adversarial
            CV (see :func:`evaluate_sequence`).
        n_folds: Grouped-CV fold count forwarded to the Drift_Gate.
        seed: Estimator seed forwarded to the Drift_Gate.

    Returns:
        A :class:`GateSequenceResult` whose ``retrain_trusted`` is ``True`` iff the
        combined feature set passed the Drift_Gate; ``holdout_trustworthy``
        carries the same PASS-gated value.
    """

    combined = _combine_features(base_features, layer_features)
    result = evaluate_sequence(
        combined, calib, groups=groups, n_folds=n_folds, seed=seed
    )
    return GateSequenceResult(
        drift=result.drift,
        holdout_trustworthy=result.holdout_trustworthy,
        retrain_trusted=result.holdout_trustworthy,
    )
