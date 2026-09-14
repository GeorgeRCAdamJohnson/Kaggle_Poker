"""CACHE-BASED risk-model experiments: pure numpy/sklearn utilities over the feature matrices.

Everything here operates on the already-built feature-matrix cache (``dev_features.parquet`` /
``eval_features.parquet``); it never rescans ``actions.parquet``. The goal is loop-engineering the
risk head (the 70% Pair-AP component of the competition metric) to beat the baseline
``StandardScaler + LogisticRegression`` group-by-pool CV AUC of ~0.850.

The module is deliberately pure + deterministic so it is unit-testable on tiny synthetic data:

* :func:`grouped_cv_auc_model` — leave-one-pool-out CV AUC for ANY sklearn-style estimator factory
  (fresh estimator per fold, held-out AUC, single-class folds skipped). This is the primary ranking
  signal (it tracks Pair-AP, the metric's 70% component).
* :func:`enrich_features` — deterministic cheap engineered columns from the existing feature set
  (products / ratios / log1p of heavy-tailed counts). Same transform for dev + eval.
* Estimator factories: :func:`make_logistic`, :func:`make_gbt` (StandardScaler is a no-op for trees
  but kept out of the GBT pipeline for speed).
* :func:`rank_normalize` / :func:`rank_average` — per-pool-safe rank-normalized score averaging for
  the ensemble candidate.

None of these touch ``classical.py`` / ``pu_ranker.py`` / ``tasks.md``; the learned logistic recipe
mirrors :mod:`poker_collusion.models.learned_risk` (StandardScaler + LogisticRegression, C=1.0).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reserved (non-feature) columns in the cached matrices.
RESERVED_COLUMNS = ("pair_id", "label", "pool", "has_eval_signal")

# Features that are phase / bookkeeping artifacts rather than collusion signal. ``phase_is_development``
# is constant 1.0 in the dev matrix and 0.0 in eval (a train/serve mismatch that trees could latch
# onto); ``pool_prior_used`` is the injected pool prior. The linear baseline standardizes a constant
# column to zero (harmless), but trees can split on it, so GBT variants get the option to drop these.
PHASE_ARTIFACT_COLUMNS = ("phase_is_development", "pool_prior_used")


# --------------------------------------------------------------------------- #
# Feature-column helpers
# --------------------------------------------------------------------------- #
def feature_columns(frame: pd.DataFrame) -> List[str]:
    """Return the ordered numeric feature columns (everything that is not reserved)."""
    return [c for c in frame.columns if c not in RESERVED_COLUMNS]


def matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Dense float matrix over ``columns`` with NaN/inf scrubbed to 0.0 (neutral)."""
    X = frame.reindex(columns=list(columns)).to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Deterministic feature enrichment (products / ratios / log1p of heavy tails)
# --------------------------------------------------------------------------- #
def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of ``frame`` with cheap deterministic engineered columns appended.

    Engineered from the top separators the discovery module surfaced. All derived deterministically
    from existing columns so the SAME transform applies identically to dev and eval:

    * ``eng_flow_x_flagged``  = value_flow_abs_p95 * flagged_rate     (heavy flow AND often flagged)
    * ``eng_iso_x_flagged``   = isolation_p95 * flagged_rate
    * ``eng_mi_x_iso``        = mi_conflict_p95 * isolation_p95        (interaction of two separators)
    * ``eng_flow_ratio``      = value_flow_absmean / (value_flow_abs_mean + eps)  (flow directedness)
    * ``eng_agg_x_iso``       = aggression_asymmetry_p95 * isolation_p95
    * ``eng_log_flagged_cnt`` = log1p(flagged_count)                   (heavy-tailed count)
    * ``eng_log_shared``      = log1p(n_shared_hands)                  (heavy-tailed count)
    * ``eng_log_burst``       = log1p(burst_count)

    Missing source columns default to 0.0 so the transform never raises on a partial schema.
    """
    eps = 1e-9
    out = frame.copy()

    def col(name: str) -> np.ndarray:
        if name in out.columns:
            v = out[name].to_numpy(dtype=float)
            return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return np.zeros(len(out), dtype=float)

    out["eng_flow_x_flagged"] = col("value_flow_abs_p95") * col("flagged_rate")
    out["eng_iso_x_flagged"] = col("isolation_p95") * col("flagged_rate")
    out["eng_mi_x_iso"] = col("mi_conflict_p95") * col("isolation_p95")
    out["eng_flow_ratio"] = col("value_flow_absmean") / (col("value_flow_abs_mean") + eps)
    out["eng_agg_x_iso"] = col("aggression_asymmetry_p95") * col("isolation_p95")
    out["eng_log_flagged_cnt"] = np.log1p(np.clip(col("flagged_count"), 0.0, None))
    out["eng_log_shared"] = np.log1p(np.clip(col("n_shared_hands"), 0.0, None))
    out["eng_log_burst"] = np.log1p(np.clip(col("burst_count"), 0.0, None))
    return out


ENGINEERED_COLUMNS = (
    "eng_flow_x_flagged",
    "eng_iso_x_flagged",
    "eng_mi_x_iso",
    "eng_flow_ratio",
    "eng_agg_x_iso",
    "eng_log_flagged_cnt",
    "eng_log_shared",
    "eng_log_burst",
)


# --------------------------------------------------------------------------- #
# Estimator factories (deterministic)
# --------------------------------------------------------------------------- #
def make_logistic(seed: int = 0) -> Callable[[], object]:
    """Factory returning a fresh ``StandardScaler + LogisticRegression(C=1.0)`` pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _factory() -> object:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )

    return _factory


def make_gbt(
    *,
    max_depth: Optional[int] = None,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    l2_regularization: float = 0.0,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    seed: int = 0,
) -> Callable[[], object]:
    """Factory returning a fresh :class:`HistGradientBoostingClassifier` with the given config."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _factory() -> object:
        return HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )

    return _factory


# --------------------------------------------------------------------------- #
# Group-by-pool CV AUC for an estimator (leave-one-pool-out)
# --------------------------------------------------------------------------- #
def _stable_unique(values: np.ndarray) -> List:
    seen: set = set()
    out: List = []
    for v in values.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def grouped_cv_auc_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], object],
) -> Tuple[float, float, int]:
    """Leave-one-pool-out CV AUC for an estimator factory.

    A FRESH estimator is fit on all-but-one pool and scored on the held-out pool via
    ``predict_proba[:, 1]``. Folds whose train OR held-out set is single-class are skipped. Returns
    ``(mean, std, n_folds)``; ``(0.5, 0.0, 0)`` when nothing is scoreable.
    """
    from sklearn.metrics import roc_auc_score

    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)

    aucs: List[float] = []
    for g in _stable_unique(groups):
        test_mask = groups == g
        train_mask = ~test_mask
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = estimator_factory()
        try:
            clf.fit(X[train_mask], y_tr)
            proba = clf.predict_proba(X[test_mask])[:, 1]
            aucs.append(float(roc_auc_score(y_te, proba)))
        except Exception:
            continue
    if not aucs:
        return 0.5, 0.0, 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def _kfold_pool_assignment(groups: np.ndarray, k: int) -> Dict[object, int]:
    """Assign each distinct pool to one of ``k`` folds deterministically (sorted round-robin).

    Pools are sorted by string, then round-robin assigned to folds. Keeps whole pools together
    (no pool split across train/test) while collapsing the ~397 leave-one-out folds to ``k`` folds
    so tree CV stays fast. Deterministic and order-independent.
    """
    distinct = sorted({str(g) for g in groups.tolist()})
    return {g: (i % k) for i, g in enumerate(distinct)}


def grouped_kfold_cv_auc_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], object],
    *,
    k: int = 8,
) -> Tuple[float, float, int]:
    """Grouped K-fold CV AUC: pools partitioned into ``k`` folds (whole pools per fold).

    Same leak-safety as leave-one-pool-out (no pool spans train/test) but only ``k`` model fits, so
    it is tractable for gradient-boosted trees. A FRESH estimator is fit on the other ``k-1`` folds
    and scored on the held-out fold via ``predict_proba[:, 1]``. Folds whose train OR test set is
    single-class are skipped. Returns ``(mean, std, n_folds)``.
    """
    from sklearn.metrics import roc_auc_score

    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)
    fold_of = _kfold_pool_assignment(groups, k)
    fold_ids = np.array([fold_of[str(g)] for g in groups.tolist()], dtype=int)

    aucs: List[float] = []
    for f in range(k):
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
            continue
    if not aucs:
        return 0.5, 0.0, 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def oof_scores_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], object],
    *,
    k: int = 8,
) -> np.ndarray:
    """Out-of-fold ``predict_proba[:, 1]`` using grouped K-fold (whole pools per fold)."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)
    fold_of = _kfold_pool_assignment(groups, k)
    fold_ids = np.array([fold_of[str(g)] for g in groups.tolist()], dtype=int)
    out = np.full(len(y), float(y.mean()), dtype=float)
    for f in range(k):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue
        clf = estimator_factory()
        try:
            clf.fit(X[train_mask], y[train_mask])
            out[test_mask] = clf.predict_proba(X[test_mask])[:, 1]
        except Exception:
            continue
    return out


def oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], object],
) -> np.ndarray:
    """Out-of-fold ``predict_proba[:, 1]`` for every row (leave-one-pool-out).

    Rows in a pool whose training set is single-class (unscoreable fold) get their score from a
    model trained on all other pools; if that also fails they fall back to the global positive rate.
    Used to build the ensemble's OOF scores so the rank-average CV AUC is leak-safe.
    """
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)
    out = np.full(len(y), float(y.mean()), dtype=float)

    for g in _stable_unique(groups):
        test_mask = groups == g
        train_mask = ~test_mask
        y_tr = y[train_mask]
        if len(np.unique(y_tr)) < 2:
            continue
        clf = estimator_factory()
        try:
            clf.fit(X[train_mask], y_tr)
            out[test_mask] = clf.predict_proba(X[test_mask])[:, 1]
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Rank-average ensemble (per-pool-safe rank normalization)
# --------------------------------------------------------------------------- #
def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Rank-normalize scores to ``[0, 1]`` (average ranks / (n-1)); constant -> all 0.5."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n <= 1:
        return np.full(n, 0.5, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    # Average-rank for ties.
    i = 0
    sorted_vals = s[order]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks / (n - 1)


def rank_average(score_list: Sequence[np.ndarray]) -> np.ndarray:
    """Average of the rank-normalized scores across models (each mapped to [0,1] then meaned)."""
    if not score_list:
        raise ValueError("rank_average needs at least one score vector")
    normed = [rank_normalize(np.asarray(s, dtype=float)) for s in score_list]
    return np.mean(np.vstack(normed), axis=0)
