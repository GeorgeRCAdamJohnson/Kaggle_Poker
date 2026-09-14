"""Adversarial dev-vs-eval Drift_Gate (design: ``gate/drift_gate.py``, task 6.1).

The Drift_Gate is the single most valuable external judge in this harness
(design "Overview" + dossier §29/§32/§33). It answers one question about a
feature set: *do these features describe "which pool a row came from" (dev vs
eval) rather than "is this pair colluding"?* We answer it exactly the way the
dossier did - train a classifier to distinguish dev rows from eval rows on the
feature set and read its cross-validated AUC. A high AUC means the features
*drift* dev->eval and are untrustworthy regardless of any CV/holdout number; a
low AUC means the features look the same in both pools, so a holdout score
measured on them is a trustworthy leaderboard predictor (Rule 17: drift PASS
first, THEN holdout is trustworthy).

This module implements the two functions the design names:

- :func:`adversarial_auc` - the raw dev-vs-eval separability of a
  :class:`~poker_collusion.tuning_harness.models.FeatureFrame`, computed with a
  leak-safe grouped CV so a pool never spans train and test (reusing the
  grouped-CV machinery style from ``validation/cv.py`` /
  ``experiments/cache_models.py``: group by ``table_id``/pool).
- :func:`evaluate_drift` - turns that AUC into a PASS/FAIL
  :class:`~poker_collusion.tuning_harness.models.DriftResult` against the
  calibrated threshold when the :class:`Calibration` is TRUSTED, else against the
  PROVISIONAL threshold (Req 1.1, 1.6, 1.7). ``verdict == FAIL`` when the AUC is
  at or above the threshold (high drift), ``PASS`` when it is strictly below.

Reuse, not rebuild (design principle 6): the classifier recipe mirrors the
existing package - ``StandardScaler + LogisticRegression(C=1.0)`` - and the
grouped-CV-AUC helper mirrors ``experiments.cache_models.grouped_kfold_cv_auc_model``.
Calibration is produced by ``gate/calibration.py`` (task 7.1); this module only
*consumes* a :class:`Calibration` and never derives thresholds itself.

Requirements: 1.1 (classify against a calibrated, not fixed, threshold).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from poker_collusion.tuning_harness.models import (
    Calibration,
    DriftResult,
    FeatureFrame,
)

__all__ = [
    "PROVISIONAL_THRESHOLD",
    "adversarial_auc",
    "evaluate_drift",
]

#: The historical single-feature drift bar (~0.65) used only when the calibrator
#: has not earned trust (design: PROVISIONAL fallback, Req 1.7). The trusted,
#: anchor-derived threshold comes from the :class:`Calibration`; this constant is
#: the reduced-confidence default the gate falls back to, never the primary bar
#: (the fixed 0.65 is exactly what wrongly rejected the known-good foundation, so
#: it is used only when nothing better is available).
PROVISIONAL_THRESHOLD: float = 0.65

#: Number of grouped folds for the adversarial CV. Matches the fast grouped
#: K-fold used elsewhere in the package (whole pools per fold, tractable).
_DEFAULT_N_FOLDS: int = 5


def _default_logistic_factory(seed: int = 0) -> Callable[[], object]:
    """Fresh ``StandardScaler + LogisticRegression(C=1.0)`` pipeline factory.

    Mirrors the recipe used across the existing package
    (``models.learned_risk`` / ``experiments.cache_models.make_logistic``) so the
    Drift_Gate classifier is the same well-understood, deterministic estimator -
    reuse, not a new model (design principle 6). Imported lazily so importing this
    module stays cheap and hermetic.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _factory() -> object:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )

    return _factory


def _stable_unique(values: Sequence[object]) -> List[object]:
    """First-seen-order unique values (deterministic, order-preserving)."""

    seen: Dict[object, None] = {}
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen.keys())


def _kfold_group_assignment(groups: Sequence[object], k: int) -> Dict[object, int]:
    """Assign each distinct group to one of ``k`` folds (sorted round-robin).

    Whole groups stay together (no group spans train/test), so the CV is
    leak-safe by pool - the same guarantee ``validation/cv.py`` and
    ``experiments.cache_models`` enforce. Deterministic and order-independent.
    """

    distinct = sorted({str(g) for g in groups})
    return {g: (i % k) for i, g in enumerate(distinct)}


def _grouped_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], object],
    *,
    k: int,
) -> float:
    """Grouped K-fold dev-vs-eval AUC (whole groups per fold; leak-safe).

    A FRESH estimator is fit on the other ``k-1`` folds and scored on the held-out
    fold via ``predict_proba[:, 1]``; folds whose train or test set is single-class
    are skipped (a fold with only dev or only eval rows carries no separability
    signal). Returns the mean fold AUC, or ``0.5`` (chance = no measurable drift)
    when nothing is scoreable. Mirrors
    ``experiments.cache_models.grouped_kfold_cv_auc_model``.
    """

    from sklearn.metrics import roc_auc_score

    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)

    n_distinct = len(_stable_unique(list(groups)))
    k_eff = max(2, min(k, n_distinct))
    fold_of = _kfold_group_assignment(list(groups), k_eff)
    fold_ids = np.array([fold_of[str(g)] for g in groups.tolist()], dtype=int)

    aucs: List[float] = []
    for f in range(k_eff):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = estimator_factory()
        try:
            clf.fit(X[train_mask], y_tr)
            proba = clf.predict_proba(X[test_mask])[:, 1]
            aucs.append(float(roc_auc_score(y_te, proba)))
        except Exception:
            # A degenerate fold (e.g. constant features) contributes no signal.
            continue

    if not aucs:
        return 0.5
    return float(np.mean(aucs))


def _orientation_max_auc(mean_auc: float) -> float:
    """Fold the AUC to its orientation-max around chance.

    Separability is symmetric: an AUC of 0.2 means the classifier separates the
    pools just as well as 0.8 (it just labels them the other way). The drift
    magnitude is therefore ``max(auc, 1 - auc)`` so a strongly-separating feature
    is never hidden by an arbitrary label orientation - the same orientation-max
    convention ``discovery/signal_separation`` uses for single features.
    """

    return float(max(mean_auc, 1.0 - mean_auc))


def _feature_matrix(
    feature_frame: FeatureFrame,
) -> tuple[List[str], np.ndarray, np.ndarray]:
    """Extract ``(pair_ids, X, y)`` from a FeatureFrame.

    ``X`` is the ``rows`` matrix in ``feature_keys`` order; ``y`` is the
    dev-vs-eval target from ``is_eval`` (1 = eval-pool row, 0 = dev row). Missing
    ``is_eval`` entries default to ``0`` (dev), matching the "dev unless flagged
    eval" convention of the caches.
    """

    pair_ids = list(feature_frame.rows.keys())
    keys = feature_frame.feature_keys
    X = np.array(
        [
            [float(v) for v in feature_frame.rows[pid]]
            if len(feature_frame.rows[pid]) == len(keys)
            else _pad_row(feature_frame.rows[pid], len(keys))
            for pid in pair_ids
        ],
        dtype=float,
    ).reshape(len(pair_ids), len(keys) if keys else 0)
    y = np.array(
        [1 if bool(feature_frame.is_eval.get(pid, False)) else 0 for pid in pair_ids],
        dtype=int,
    )
    return pair_ids, X, y


def _pad_row(row: Sequence[float], width: int) -> List[float]:
    """Pad/truncate a feature row to ``width`` (defensive; frames are usually aligned)."""

    vals = [float(v) for v in row][:width]
    return vals + [0.0] * (width - len(vals))


def _derive_groups(
    pair_ids: Sequence[str],
    y: np.ndarray,
    groups: Optional[Mapping[str, object]],
) -> np.ndarray:
    """Resolve the CV grouping for the adversarial split.

    When the caller supplies a ``pair_id -> pool`` map (from ``validation/cv.py``
    ``derive_pool_map`` - group by ``table_id``), we use it directly so whole
    tables stay together and no pool leaks across folds. When no map is supplied,
    dev and eval rows are inherently disjoint pools (they come from different data
    partitions), so each row is placed in a group tied to its own pool-side and
    identity; this keeps the split leak-safe while still letting the classifier
    learn the dev-vs-eval boundary.
    """

    if groups is not None:
        return np.array([groups.get(pid, groups.get(str(pid), pid)) for pid in pair_ids], dtype=object)
    # No pool map: group by (side, pair) so folds mix dev and eval pools without
    # any single pair straddling folds. Side keeps the classes representable.
    return np.array(
        [f"{'eval' if y[i] else 'dev'}::{pid}" for i, pid in enumerate(pair_ids)],
        dtype=object,
    )


def adversarial_auc(
    feature_frame: FeatureFrame,
    *,
    groups: Optional[Mapping[str, object]] = None,
    n_folds: int = _DEFAULT_N_FOLDS,
    seed: int = 0,
) -> float:
    """Dev-vs-eval separability of a feature set (the drift magnitude).

    Trains a dev-vs-eval classifier (target = ``FeatureFrame.is_eval``: 1 for
    eval-pool rows, 0 for dev rows) with a leak-safe grouped CV and returns the
    orientation-max mean AUC. A **high** AUC means the features describe "which
    pool" (drift) rather than "is collusion"; a value near ``0.5`` means the two
    pools are indistinguishable on this feature set (no drift), so a holdout score
    measured on it can be trusted (Rule 17).

    The classifier and grouped-CV recipe reuse the existing package's approach
    (``StandardScaler + LogisticRegression``, group by pool so no table spans
    train/test).

    Args:
        feature_frame: The candidate feature set to test (rows keyed by
            ``pair_id``, columns named by ``feature_keys``, dev-vs-eval labels in
            ``is_eval``).
        groups: Optional ``{pair_id: pool}`` map (e.g. from
            ``validation.cv.derive_pool_map``) to group the CV by ``table_id``. If
            omitted, a leak-safe per-pair grouping is derived (dev and eval are
            already disjoint pools).
        n_folds: Grouped-CV fold count (default 5).
        seed: Estimator seed for determinism.

    Returns:
        The dev-vs-eval AUC in ``[0.5, 1.0]`` (orientation-max). Returns ``0.5``
        (chance = no measurable drift) when the frame is empty, single-class
        (all-dev or all-eval), or has no features.
    """

    pair_ids, X, y = _feature_matrix(feature_frame)

    # Degenerate frames carry no drift signal: no rows, no features, or only one
    # pool present. Chance (0.5) is the honest "cannot measure drift" answer.
    if len(pair_ids) == 0 or X.shape[1] == 0 or len(np.unique(y)) < 2:
        return 0.5

    group_arr = _derive_groups(pair_ids, y, groups)
    factory = _default_logistic_factory(seed=seed)
    mean_auc = _grouped_cv_auc(X, y, group_arr, factory, k=n_folds)
    return _orientation_max_auc(mean_auc)


def evaluate_drift(
    feature_frame: FeatureFrame,
    calib: Calibration,
    *,
    groups: Optional[Mapping[str, object]] = None,
    n_folds: int = _DEFAULT_N_FOLDS,
    seed: int = 0,
) -> DriftResult:
    """Classify a feature set's drift into PASS/FAIL against the calibrated gate.

    Computes the dev-vs-eval :func:`adversarial_auc`, then classifies it:

    - If ``calib.trusted`` is True, the deciding threshold is the anchor-derived
      ``calib.threshold`` and ``threshold_kind`` is ``"LB_CALIBRATED"`` (Req 1.1,
      1.6). This is the whole point - a calibrated bar, not the fixed 0.65 that
      wrongly rejected the known-good foundation.
    - Otherwise the gate falls back to the PROVISIONAL threshold with reduced
      confidence (Req 1.7): ``calib.threshold`` if the calibrator populated one,
      else :data:`PROVISIONAL_THRESHOLD`.

    ``verdict`` is ``"FAIL"`` when the AUC is **at or above** the threshold (high
    drift = the features separate the pools too well to trust), and ``"PASS"``
    when it is strictly below.

    Args:
        feature_frame: The candidate feature set.
        calib: The calibration produced by ``gate/calibration.py`` (task 7.1).
            This function only consumes it; it never derives a threshold itself.
        groups: Optional pool grouping forwarded to :func:`adversarial_auc`.
        n_folds: Grouped-CV fold count forwarded to :func:`adversarial_auc`.
        seed: Estimator seed forwarded to :func:`adversarial_auc`.

    Returns:
        A :class:`DriftResult` with the measured AUC, the PASS/FAIL verdict, the
        deciding threshold, and its kind (``"LB_CALIBRATED"`` | ``"PROVISIONAL"``).
    """

    auc = adversarial_auc(feature_frame, groups=groups, n_folds=n_folds, seed=seed)

    if calib.trusted:
        threshold = float(calib.threshold)
        threshold_kind = "LB_CALIBRATED"
    else:
        # Reduced-confidence fallback (Req 1.7): prefer whatever provisional
        # value the calibrator carried, else the historical ~0.65 default.
        threshold = float(calib.threshold) if calib.threshold is not None else PROVISIONAL_THRESHOLD
        threshold_kind = "PROVISIONAL"

    verdict = "FAIL" if auc >= threshold else "PASS"

    return DriftResult(
        adversarial_auc=auc,
        verdict=verdict,
        threshold=threshold,
        threshold_kind=threshold_kind,
    )
