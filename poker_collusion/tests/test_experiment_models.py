"""Hermetic tests for :mod:`poker_collusion.experiments.cache_models`.

Deterministic, tiny synthetic data (no cache, no actions scan). Verifies:
* the GBT + logistic factories train and produce risk strictly in [0, 1];
* enrichment appends exactly the engineered columns and is deterministic;
* grouped K-fold / leave-one-pool-out CV AUC recover a strong signal on separable synthetic data;
* rank normalization / rank-average stay in [0, 1] and are order-preserving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poker_collusion.experiments import cache_models as cm


def _synthetic(n: int = 240, seed: int = 0):
    """Two-feature separable synthetic set with a pool column (whole-pool-safe groups)."""
    rng = np.random.default_rng(seed)
    # Pool from (i mod 12); label from (i // 12) parity so EVERY pool holds both classes.
    pools = np.array([f"pool_{i % 12}" for i in range(n)], dtype=object)
    y = np.array([(i // 12) % 2 for i in range(n)], dtype=int)
    # Feature 1 separates the classes; feature 2 is noise.
    f1 = y + rng.normal(0.0, 0.4, size=n)
    f2 = rng.normal(0.0, 1.0, size=n)
    frame = pd.DataFrame(
        {
            "pair_id": [f"P{i:04d}" for i in range(n)],
            "value_flow_abs_p95": np.abs(f1),
            "flagged_rate": np.clip(f1, 0.0, None),
            "isolation_p95": f2,
            "flagged_count": rng.integers(0, 50, size=n).astype(float),
            "n_shared_hands": rng.integers(1, 100, size=n).astype(float),
            "label": y,
            "pool": pools,
        }
    )
    return frame, y, pools


def test_feature_columns_excludes_reserved():
    frame, _, _ = _synthetic()
    cols = cm.feature_columns(frame)
    assert "pair_id" not in cols and "label" not in cols and "pool" not in cols
    assert "value_flow_abs_p95" in cols


def test_matrix_scrubs_nan_inf():
    frame = pd.DataFrame({"a": [1.0, np.nan, np.inf], "b": [-np.inf, 2.0, 3.0]})
    X = cm.matrix(frame, ["a", "b"])
    assert np.isfinite(X).all()
    assert X[1, 0] == 0.0 and X[0, 1] == 0.0


def test_enrich_features_appends_expected_columns_and_is_deterministic():
    frame, _, _ = _synthetic()
    out1 = cm.enrich_features(frame)
    out2 = cm.enrich_features(frame)
    for c in cm.ENGINEERED_COLUMNS:
        assert c in out1.columns
    # Original columns preserved; engineered columns deterministic.
    assert list(frame.columns) == [c for c in out1.columns if c not in cm.ENGINEERED_COLUMNS]
    for c in cm.ENGINEERED_COLUMNS:
        assert np.allclose(out1[c].to_numpy(), out2[c].to_numpy())
    assert np.isfinite(out1[list(cm.ENGINEERED_COLUMNS)].to_numpy()).all()


def test_logistic_factory_trains_and_risk_in_unit_interval():
    frame, y, _ = _synthetic()
    X = cm.matrix(frame, ["value_flow_abs_p95", "flagged_rate", "isolation_p95"])
    clf = cm.make_logistic(seed=0)()
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_gbt_factory_trains_scores_and_risk_in_unit_interval():
    frame, y, _ = _synthetic()
    X = cm.matrix(frame, ["value_flow_abs_p95", "flagged_rate", "isolation_p95"])
    clf = cm.make_gbt(max_depth=3, learning_rate=0.1, max_iter=50, l2_regularization=1.0, seed=0)()
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_grouped_kfold_cv_auc_recovers_signal():
    frame, y, pools = _synthetic()
    X = cm.matrix(frame, ["value_flow_abs_p95", "flagged_rate", "isolation_p95"])
    m, s, folds = cm.grouped_kfold_cv_auc_model(X, y, pools, cm.make_logistic(seed=0), k=4)
    assert folds >= 2
    assert 0.0 <= m <= 1.0
    assert m > 0.8  # strongly separable synthetic


def test_leave_one_pool_out_cv_auc_matches_shape():
    frame, y, pools = _synthetic()
    X = cm.matrix(frame, ["value_flow_abs_p95", "flagged_rate", "isolation_p95"])
    m, s, folds = cm.grouped_cv_auc_model(X, y, pools, cm.make_logistic(seed=0))
    assert 0.0 <= m <= 1.0 and folds >= 1


def test_gbt_kfold_deterministic():
    frame, y, pools = _synthetic()
    X = cm.matrix(frame, ["value_flow_abs_p95", "flagged_rate", "isolation_p95"])
    fac = cm.make_gbt(max_depth=3, learning_rate=0.1, max_iter=50, l2_regularization=1.0, seed=7)
    m1, _, _ = cm.grouped_kfold_cv_auc_model(X, y, pools, fac, k=4)
    m2, _, _ = cm.grouped_kfold_cv_auc_model(X, y, pools, fac, k=4)
    assert m1 == m2


def test_rank_normalize_in_unit_interval_and_monotone():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    r = cm.rank_normalize(x)
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0
    # Order preserved: largest input -> largest rank.
    assert np.argmax(r) == np.argmax(x)
    assert np.argmin(r) == np.argmin(x)


def test_rank_normalize_constant_is_half():
    r = cm.rank_normalize(np.array([2.0, 2.0, 2.0]))
    assert np.allclose(r, 0.5)


def test_rank_average_in_unit_interval():
    a = np.array([0.1, 0.9, 0.5, 0.3])
    b = np.array([0.8, 0.2, 0.4, 0.6])
    avg = cm.rank_average([a, b])
    assert float(avg.min()) >= 0.0 and float(avg.max()) <= 1.0
    assert avg.shape == a.shape


def test_oof_scores_kfold_covers_all_rows_in_unit_interval():
    frame, y, pools = _synthetic()
    X = cm.matrix(frame, ["value_flow_abs_p95", "flagged_rate", "isolation_p95"])
    oof = cm.oof_scores_kfold(X, y, pools, cm.make_logistic(seed=0), k=4)
    assert oof.shape == (len(y),)
    assert float(oof.min()) >= 0.0 and float(oof.max()) <= 1.0
