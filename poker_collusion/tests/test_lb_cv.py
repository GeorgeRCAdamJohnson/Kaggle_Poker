"""Tests for the LB-vs-CV correlation diagnostic and selection rule (task 5.3).

All fixtures are small synthetic frames and ``B`` is modest so the suite stays fast.
Verified behavior:

  (a) correlation — Spearman/Pearson computed correctly on a small hand-checkable
      example, and the weak-correlation warning fires when CV and LB disagree;
  (b) selection rule — picks a clearly-better candidate, keeps the incumbent when a
      challenger is only within-band better, and never uses the public LB to decide;
  (c) determinism — a fixed seed reproduces the selection and the correlations exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE
from poker_collusion.validation.lb_cv import (
    WEAK_CORRELATION_THRESHOLD,
    LbCvDiagnostic,
    SubmissionCandidate,
    lb_cv_correlation,
    pearson_corr,
    select_submission,
    spearman_corr,
)

FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")


# --------------------------------------------------------------------------- #
# Synthetic frame builders (mirror the bootstrap-test fixtures)               #
# --------------------------------------------------------------------------- #
def _solution(n_pairs: int = 48, n_pos: int = 15) -> pd.DataFrame:
    rows = []
    for i in range(n_pairs):
        is_pos = i < n_pos
        fam = FAMILIES[i % len(FAMILIES)] if is_pos else "none"
        evid = {}
        for r in range(1, 6):
            evid[f"evidence_hand_{r}"] = f"P{i}_H{r}" if (is_pos and r <= 3) else NO_EVIDENCE
        rows.append(
            {
                "pair_id": f"pair_{i}",
                "risk_score": 1 if is_pos else 0,
                "predicted_behavior": fam,
                **evid,
            }
        )
    return pd.DataFrame(rows)


def _strong_submission(solution: pd.DataFrame, noise: float = 0.05, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub = solution.copy()
    truth = solution["risk_score"].to_numpy(dtype=float)
    risk = np.where(truth == 1, 0.9, 0.1) + rng.uniform(-noise, noise, size=len(sub))
    sub["risk_score"] = np.clip(risk, 0.0, 1.0)
    return sub


def _weak_submission(solution: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub = solution.copy()
    sub["risk_score"] = rng.uniform(0.0, 1.0, size=len(sub))
    sub["predicted_behavior"] = "none"
    for r in range(1, 6):
        sub[f"evidence_hand_{r}"] = NO_EVIDENCE
    return sub


def _perturbed_submission(base: pd.DataFrame, eps: float = 1e-4, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub = base.copy()
    risk = sub["risk_score"].to_numpy(dtype=float) + rng.uniform(-eps, eps, size=len(sub))
    sub["risk_score"] = np.clip(risk, 0.0, 1.0)
    return sub


# --------------------------------------------------------------------------- #
# (a) Correlation diagnostic                                                  #
# --------------------------------------------------------------------------- #
def test_pearson_and_spearman_on_hand_checkable_example():
    # Perfectly monotone but non-linear: y = x**2 for x = 1..5.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 4.0, 9.0, 16.0, 25.0]
    # Spearman of a strictly increasing transform is exactly 1.
    assert spearman_corr(x, y) == pytest.approx(1.0)
    # Pearson is high but < 1 because the relationship is non-linear.
    r = pearson_corr(x, y)
    assert 0.95 < r < 1.0
    assert r == pytest.approx(0.9811049, abs=1e-6)


def test_pearson_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    x = rng.normal(size=25)
    y = 0.6 * x + rng.normal(size=25)
    assert pearson_corr(x, y) == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=1e-12)


def test_spearman_handles_ties_with_average_ranks():
    # Tied values in x share the average rank; perfect monotone agreement -> 1.0.
    x = [1.0, 2.0, 2.0, 3.0]
    y = [10.0, 20.0, 20.0, 30.0]
    assert spearman_corr(x, y) == pytest.approx(1.0)


def test_constant_series_yields_zero_correlation():
    assert pearson_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert spearman_corr([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0


def test_strong_agreement_does_not_warn():
    # LB tracks CV monotonically (a noisy but rank-preserving version).
    experiments = [
        {"experiment_id": "e1", "cv_combined": 0.50, "lb_score": 0.48},
        {"experiment_id": "e2", "cv_combined": 0.55, "lb_score": 0.57},
        {"experiment_id": "e3", "cv_combined": 0.60, "lb_score": 0.61},
        {"experiment_id": "e4", "cv_combined": 0.65, "lb_score": 0.64},
        {"experiment_id": "e5", "cv_combined": 0.70, "lb_score": 0.72},
    ]
    diag = lb_cv_correlation(experiments)
    assert isinstance(diag, LbCvDiagnostic)
    assert diag.n_experiments == 5
    assert diag.spearman == pytest.approx(1.0)
    assert diag.warn is False


def test_weak_or_negative_agreement_fires_warning():
    # CV ascends while LB descends: strong disagreement -> weak/negative correlation.
    experiments = pd.DataFrame(
        {
            "experiment_id": ["e1", "e2", "e3", "e4", "e5"],
            "cv_combined": [0.50, 0.55, 0.60, 0.65, 0.70],
            "lb_score": [0.72, 0.64, 0.61, 0.57, 0.48],
        }
    )
    diag = lb_cv_correlation(experiments)
    assert diag.spearman < WEAK_CORRELATION_THRESHOLD
    assert diag.spearman == pytest.approx(-1.0)
    assert diag.warn is True
    assert "too noisy" in diag.message


def test_too_few_experiments_warns():
    diag = lb_cv_correlation([{"cv_combined": 0.5, "lb_score": 0.5}])
    assert diag.n_experiments == 1
    assert diag.warn is True


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        lb_cv_correlation(pd.DataFrame({"cv_combined": [0.5, 0.6]}))
    with pytest.raises(ValueError):
        lb_cv_correlation([{"cv_combined": 0.5}])


def test_correlation_is_deterministic():
    experiments = [
        {"cv_combined": 0.50, "lb_score": 0.30},
        {"cv_combined": 0.55, "lb_score": 0.90},
        {"cv_combined": 0.60, "lb_score": 0.10},
        {"cv_combined": 0.65, "lb_score": 0.70},
    ]
    d1 = lb_cv_correlation(experiments)
    d2 = lb_cv_correlation(experiments)
    assert d1.as_dict() == d2.as_dict()


# --------------------------------------------------------------------------- #
# (b) Selection rule                                                          #
# --------------------------------------------------------------------------- #
def test_selects_clearly_better_candidate():
    sol = _solution()
    weak = SubmissionCandidate("weak", _weak_submission(sol), lb_score=0.9)
    strong = SubmissionCandidate("strong", _strong_submission(sol), lb_score=0.1)
    # Incumbent is the weak one; the strong candidate beats it beyond the band.
    result = select_submission(
        sol, [weak, strong], incumbent_id="weak", n_boot=300, seed=505
    )
    assert result.chosen_id == "strong"
    assert result.cv_combined > 0.0


def test_keeps_incumbent_when_challenger_only_within_band():
    sol = _solution()
    base = _strong_submission(sol, seed=7)
    incumbent = SubmissionCandidate("incumbent", base, lb_score=0.1)
    # A tiny jitter: same ranking, no gain beyond noise, same band width.
    challenger = SubmissionCandidate(
        "challenger", _perturbed_submission(base, eps=1e-4, seed=3), lb_score=0.99
    )
    result = select_submission(
        sol, [incumbent, challenger], incumbent_id="incumbent", n_boot=300, seed=505
    )
    # Within-band tie with equal-ish bands -> conservative choice keeps the incumbent,
    # despite the challenger having a far higher (irrelevant) LB score.
    assert result.chosen_id == "incumbent"


def test_never_uses_lb_to_decide():
    # A weak submission with a stellar LB must NOT be selected over a strong one.
    sol = _solution()
    weak_high_lb = SubmissionCandidate("weak", _weak_submission(sol), lb_score=0.99)
    strong_low_lb = SubmissionCandidate("strong", _strong_submission(sol), lb_score=0.01)
    # No incumbent given -> defaults to best-CV candidate; LB is ignored entirely.
    result = select_submission(sol, [weak_high_lb, strong_low_lb], n_boot=300, seed=505)
    assert result.chosen_id == "strong"


def test_selection_is_deterministic():
    sol = _solution()
    cands = [
        SubmissionCandidate("weak", _weak_submission(sol)),
        SubmissionCandidate("strong", _strong_submission(sol)),
    ]
    r1 = select_submission(sol, cands, incumbent_id="weak", n_boot=300, seed=505)
    r2 = select_submission(sol, cands, incumbent_id="weak", n_boot=300, seed=505)
    assert r1.chosen_id == r2.chosen_id
    assert r1.cv_combined == r2.cv_combined
    assert r1.cv_band.as_dict() == r2.cv_band.as_dict()


def test_selection_rejects_empty_and_bad_inputs():
    sol = _solution()
    with pytest.raises(ValueError):
        select_submission(sol, [])
    with pytest.raises(ValueError):
        select_submission(
            sol,
            [SubmissionCandidate("a", _strong_submission(sol))],
            incumbent_id="does_not_exist",
        )
    # Duplicate candidate ids are rejected.
    with pytest.raises(ValueError):
        select_submission(
            sol,
            [
                SubmissionCandidate("dup", _strong_submission(sol)),
                SubmissionCandidate("dup", _weak_submission(sol)),
            ],
        )
