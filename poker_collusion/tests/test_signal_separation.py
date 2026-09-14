"""Unit test for the Phase -1 signal-separation measurement (synthetic data only).

Exercises the pure statistics + the report assembly in
:mod:`poker_collusion.discovery.signal_separation` on a TINY, hermetic matrix -- the real
``data/poker`` files are never touched. It proves:

* ``auc_orientation_max`` recovers a perfect separator (AUC ~ 1.0) and picks the correct
  orientation for a "lower = more suspicious" feature;
* ``cohens_d`` has the expected sign and a noise feature is ~0;
* ``grouped_cv_auc`` is leak-safe (per-group AUC) and near 1.0 for a strong feature;
* ``measure_separation`` ranks features by CV AUC descending, computes a multivariate upper bound,
  and reports the honest useful/not-useful verdict against the >= 0.65 bar;
* the ASCII report renders and contains the verdict line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poker_collusion.discovery.signal_separation import (
    USEFUL_AUC_BAR,
    auc_orientation_max,
    cohens_d,
    format_report,
    grouped_cv_auc,
    measure_separation,
)


def _synthetic_matrix(seed: int = 0):
    """Build a small labelled matrix with a strong, a reversed, and a noise feature.

    4 pools x (several positives + negatives). ``strong_hi`` is high for positives; ``strong_lo``
    is LOW for positives (reversed orientation); ``noise`` carries no signal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    ys = []
    groups = []
    for pool in range(4):
        for _ in range(6):  # positives
            rows.append(
                {
                    "strong_hi": float(rng.normal(3.0, 0.5)),
                    "strong_lo": float(rng.normal(-3.0, 0.5)),
                    "noise": float(rng.normal(0.0, 1.0)),
                }
            )
            ys.append(1)
            groups.append(f"T{pool}")
        for _ in range(6):  # negatives
            rows.append(
                {
                    "strong_hi": float(rng.normal(-3.0, 0.5)),
                    "strong_lo": float(rng.normal(3.0, 0.5)),
                    "noise": float(rng.normal(0.0, 1.0)),
                }
            )
            ys.append(0)
            groups.append(f"T{pool}")
    frame = pd.DataFrame(rows)
    return frame, np.asarray(ys, dtype=int), np.asarray(groups, dtype=object)


def test_auc_orientation_max_recovers_separators():
    frame, y, _ = _synthetic_matrix()
    auc_hi, or_hi = auc_orientation_max(y, frame["strong_hi"].to_numpy())
    auc_lo, or_lo = auc_orientation_max(y, frame["strong_lo"].to_numpy())
    assert auc_hi > 0.95 and or_hi == "+"
    assert auc_lo > 0.95 and or_lo == "-"  # lower = more suspicious


def test_auc_constant_feature_is_half():
    y = np.array([0, 1, 0, 1])
    auc, orient = auc_orientation_max(y, np.array([2.0, 2.0, 2.0, 2.0]))
    assert auc == 0.5 and orient == "+"


def test_cohens_d_sign_and_noise():
    frame, y, _ = _synthetic_matrix()
    assert cohens_d(y, frame["strong_hi"].to_numpy()) > 2.0
    assert cohens_d(y, frame["strong_lo"].to_numpy()) < -2.0
    assert abs(cohens_d(y, frame["noise"].to_numpy())) < 1.0


def test_grouped_cv_auc_leaksafe_strong_and_noise():
    frame, y, groups = _synthetic_matrix()
    mean_hi, std_hi, folds_hi = grouped_cv_auc(
        y, frame["strong_hi"].to_numpy(), groups, orientation="+"
    )
    assert folds_hi == 4
    assert mean_hi > 0.95
    mean_no, _, folds_no = grouped_cv_auc(
        y, frame["noise"].to_numpy(), groups, orientation="+"
    )
    assert folds_no == 4
    assert 0.3 <= mean_no <= 0.7  # noise ~ chance


def test_grouped_cv_auc_skips_single_class_group():
    y = np.array([1, 1, 0, 1])
    x = np.array([3.0, 2.0, -1.0, 2.5])
    groups = np.array(["A", "A", "B", "B"], dtype=object)  # group A is single-class
    _, _, folds = grouped_cv_auc(y, x, groups, orientation="+")
    assert folds == 1  # only group B has both classes


def test_measure_separation_ranks_and_verdict_useful():
    frame, y, groups = _synthetic_matrix()
    report = measure_separation(frame, y, groups)

    # Ranked by CV AUC descending: a strong feature is first, noise last.
    assert report.features[0].name in {"strong_hi", "strong_lo"}
    assert report.features[-1].name == "noise"
    assert report.features[0].cv_auc_mean >= report.features[-1].cv_auc_mean

    assert report.n_pos == 24 and report.n_neg == 24 and report.n_pools == 4
    assert report.multivariate_folds == 4
    assert report.multivariate_cv_auc_mean > 0.9

    # Strong synthetic separation clears the useful bar.
    assert report.best_cv_auc >= USEFUL_AUC_BAR
    assert report.verdict_useful is True

    text = format_report(report, top=3)
    assert "VERDICT" in text
    assert "USEFUL SIGNAL FOUND" in text


def test_measure_separation_verdict_not_useful_on_noise_only():
    """A matrix of pure noise yields the honest NO-USEFUL-SIGNAL verdict."""
    rng = np.random.default_rng(1)
    rows = []
    ys = []
    groups = []
    for pool in range(4):
        for _ in range(6):
            rows.append({"n1": float(rng.normal()), "n2": float(rng.normal())})
            ys.append(1)
            groups.append(f"T{pool}")
        for _ in range(6):
            rows.append({"n1": float(rng.normal()), "n2": float(rng.normal())})
            ys.append(0)
            groups.append(f"T{pool}")
    frame = pd.DataFrame(rows)
    report = measure_separation(frame, np.asarray(ys), np.asarray(groups, dtype=object))
    assert report.verdict_useful is False
    text = format_report(report)
    assert "NO USEFUL SIGNAL" in text
