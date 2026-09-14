"""Tests for the bootstrap CIs over pairs and the noise-band rule (task 5.2).

All fixtures are small synthetic frames and ``B`` is modest (200-500) so the suite
stays fast. Verified behavior:

  (a) determinism — a fixed seed reproduces the bootstrap samples and CIs exactly;
  (b) decision rule — a clearly-better submission is flagged as an improvement
      beyond the band, while a within-noise perturbation is NOT;
  (c) CI sanity — the CI brackets the point estimate and widens sensibly as the
      pair count shrinks;
  (d) CVHarness integration — the bootstrap consumes a fold's solution+submission
      built by the harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE
from poker_collusion.metric.production_metric import score_components
from poker_collusion.validation.bootstrap import (
    COMPONENT_KEYS,
    bootstrap_components,
    confidence_interval,
    is_improvement_beyond_band,
    paired_bootstrap_diff,
)
from poker_collusion.validation.cv import CVHarness, build_fold_solution

FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")


# --------------------------------------------------------------------------- #
# Synthetic frame builders                                                    #
# --------------------------------------------------------------------------- #
def _solution(n_pairs: int = 40, n_pos: int = 12, seed: int = 1) -> pd.DataFrame:
    """A synthetic solution: ``n_pos`` positives (with evidence + family), rest neg."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        is_pos = i < n_pos
        fam = FAMILIES[i % len(FAMILIES)] if is_pos else "none"
        evid = {}
        for r in range(1, 6):
            if is_pos and r <= 3:
                evid[f"evidence_hand_{r}"] = f"P{i}_H{r}"
            else:
                evid[f"evidence_hand_{r}"] = NO_EVIDENCE
        rows.append(
            {
                "pair_id": f"pair_{i}",
                "risk_score": 1 if is_pos else 0,
                "predicted_behavior": fam,
                **evid,
            }
        )
    return pd.DataFrame(rows)


def _perfect_submission(solution: pd.DataFrame) -> pd.DataFrame:
    """A perfect submission is a copy of the solution."""
    return solution.copy()


def _strong_submission(solution: pd.DataFrame, noise: float = 0.05, seed: int = 7) -> pd.DataFrame:
    """A clearly-good submission: correct family/evidence, risk = truth minus jitter.

    Positives get high risk, negatives low, keeping the ranking essentially correct
    but not identical to the solution.
    """
    rng = np.random.default_rng(seed)
    sub = solution.copy()
    truth = solution["risk_score"].to_numpy(dtype=float)
    risk = np.where(truth == 1, 0.9, 0.1) + rng.uniform(-noise, noise, size=len(sub))
    sub["risk_score"] = np.clip(risk, 0.0, 1.0)
    return sub


def _weak_submission(solution: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """A clearly-bad submission: random risk, wrong behavior, no evidence."""
    rng = np.random.default_rng(seed)
    sub = solution.copy()
    sub["risk_score"] = rng.uniform(0.0, 1.0, size=len(sub))
    sub["predicted_behavior"] = "none"
    for r in range(1, 6):
        sub[f"evidence_hand_{r}"] = NO_EVIDENCE
    return sub


def _perturbed_submission(base: pd.DataFrame, eps: float = 1e-4, seed: int = 3) -> pd.DataFrame:
    """A within-noise perturbation of ``base``: tiny risk jitter, same ranking."""
    rng = np.random.default_rng(seed)
    sub = base.copy()
    risk = sub["risk_score"].to_numpy(dtype=float) + rng.uniform(-eps, eps, size=len(sub))
    sub["risk_score"] = np.clip(risk, 0.0, 1.0)
    return sub


# --------------------------------------------------------------------------- #
# (a) Determinism                                                             #
# --------------------------------------------------------------------------- #
def test_bootstrap_is_deterministic_given_seed():
    sol = _solution()
    sub = _strong_submission(sol)
    b1 = bootstrap_components(sol, sub, n_boot=200, seed=505)
    b2 = bootstrap_components(sol, sub, n_boot=200, seed=505)
    for key in COMPONENT_KEYS:
        assert np.array_equal(b1.samples[key], b2.samples[key])
    # CIs derived from identical samples are identical too.
    assert b1.interval("combined").as_dict() == b2.interval("combined").as_dict()


def test_bootstrap_default_seed_from_config():
    sol = _solution()
    sub = _strong_submission(sol)
    b = bootstrap_components(sol, sub, n_boot=100)
    # config.Seeds.bootstrap == 505.
    assert b.seed == 505


def test_paired_diff_is_deterministic():
    sol = _solution()
    base = _weak_submission(sol)
    cand = _strong_submission(sol)
    d1 = paired_bootstrap_diff(sol, base, cand, n_boot=200, seed=505)
    d2 = paired_bootstrap_diff(sol, base, cand, n_boot=200, seed=505)
    assert np.array_equal(d1, d2)


# --------------------------------------------------------------------------- #
# (b) Decision rule                                                           #
# --------------------------------------------------------------------------- #
def test_clearly_better_submission_is_improvement_beyond_band():
    sol = _solution(n_pairs=48, n_pos=15)
    baseline = _weak_submission(sol)
    candidate = _strong_submission(sol)
    assert is_improvement_beyond_band(
        sol, baseline, candidate, n_boot=300, seed=505
    ) is True


def test_within_noise_perturbation_is_not_improvement():
    sol = _solution(n_pairs=48, n_pos=15)
    baseline = _strong_submission(sol, seed=7)
    # A tiny jitter of the same submission: same ranking, no real gain.
    candidate = _perturbed_submission(baseline, eps=1e-4, seed=3)
    assert is_improvement_beyond_band(
        sol, baseline, candidate, n_boot=300, seed=505
    ) is False


def test_weaker_candidate_is_not_improvement():
    sol = _solution(n_pairs=48, n_pos=15)
    baseline = _strong_submission(sol)
    candidate = _weak_submission(sol)
    assert is_improvement_beyond_band(
        sol, baseline, candidate, n_boot=300, seed=505
    ) is False


def test_alternative_rules_agree_on_clear_improvement():
    sol = _solution(n_pairs=48, n_pos=15)
    baseline = _weak_submission(sol)
    candidate = _strong_submission(sol)
    for rule in ("paired", "upper", "disjoint"):
        assert is_improvement_beyond_band(
            sol, baseline, candidate, n_boot=300, seed=505, rule=rule
        ) is True


# --------------------------------------------------------------------------- #
# (c) CI sanity                                                               #
# --------------------------------------------------------------------------- #
def test_ci_brackets_the_point_estimate():
    sol = _solution(n_pairs=40, n_pos=12)
    sub = _strong_submission(sol)
    point = score_components(sol, sub).as_dict()
    boot = bootstrap_components(sol, sub, n_boot=400, seed=505)
    intervals = boot.intervals()
    for key in COMPONENT_KEYS:
        ci = intervals[key]
        assert ci.low <= ci.high
        # The bootstrap mean sits inside its own CI.
        assert ci.contains(ci.mean)
        # The point estimate is near / within the band (allow a small margin since
        # the bootstrap mean can be slightly biased for AP-type statistics).
        assert ci.low - 0.15 <= point[key] <= ci.high + 0.15


def test_ci_widens_with_fewer_pairs():
    # Same submission "shape", far fewer pairs -> wider combined CI.
    big = _solution(n_pairs=80, n_pos=24, seed=1)
    small = _solution(n_pairs=16, n_pos=5, seed=1)
    big_ci = bootstrap_components(big, _strong_submission(big), n_boot=400, seed=505).interval("combined")
    small_ci = bootstrap_components(small, _strong_submission(small), n_boot=400, seed=505).interval("combined")
    big_width = big_ci.high - big_ci.low
    small_width = small_ci.high - small_ci.low
    assert small_width > big_width


def test_confidence_interval_alpha_controls_width():
    sol = _solution()
    boot = bootstrap_components(sol, _strong_submission(sol), n_boot=400, seed=505)
    narrow = confidence_interval(boot.samples["combined"], alpha=0.20)  # 80% CI
    wide = confidence_interval(boot.samples["combined"], alpha=0.02)    # 98% CI
    assert (wide.high - wide.low) >= (narrow.high - narrow.low)


def test_confidence_interval_rejects_bad_inputs():
    with pytest.raises(ValueError):
        confidence_interval(np.array([1.0, 2.0]), alpha=0.0)
    with pytest.raises(ValueError):
        confidence_interval(np.array([]), alpha=0.05)


def test_bootstrap_rejects_empty_solution():
    empty = _solution(n_pairs=0, n_pos=0)
    with pytest.raises(ValueError):
        bootstrap_components(empty, empty, n_boot=10, seed=505)


# --------------------------------------------------------------------------- #
# (d) CVHarness integration                                                   #
# --------------------------------------------------------------------------- #
def _harness_fixture():
    """Reuse a small synthetic labelled set to build a CVHarness fold solution."""
    rows = []
    pool_map = {}
    evidence_rows = []
    pid = 0
    for pool in range(8):
        # one positive.
        fam = FAMILIES[pool % len(FAMILIES)]
        p = f"pos_{pid}"
        pid += 1
        pool_map[p] = pool
        rows.append(
            {
                "pair_id": p,
                "player_1": pool * 10,
                "player_2": pool * 10 + 1,
                "label_status": "confirmed_target",
                "behavior_family": fam,
            }
        )
        for r in range(1, 4):
            evidence_rows.append(
                {"pair_id": p, "evidence_rank": r, "hand_id": f"{p}_H{r}"}
            )
        # two negatives.
        for _ in range(2):
            n = f"neg_{pid}"
            pid += 1
            pool_map[n] = pool
            rows.append(
                {
                    "pair_id": n,
                    "player_1": pool * 10 + 2,
                    "player_2": pool * 10 + 3,
                    "label_status": "confirmed_non_target",
                    "behavior_family": "none",
                }
            )
    return pd.DataFrame(rows), pool_map, pd.DataFrame(evidence_rows)


def test_bootstrap_consumes_a_cvharness_fold_solution():
    labels, pool_map, evidence = _harness_fixture()
    harness = CVHarness(labels, pool_map, n_folds=4, seed=404, evidence=evidence)

    # Aggregate a fold's confirmed pairs across all folds to get enough pairs.
    all_pairs = [p for fold in harness.folds() for p in fold.validation_pairs]
    solution = build_fold_solution(labels, all_pairs, evidence=evidence)
    assert len(solution) > 0

    perfect = solution.copy()
    weak = _weak_submission(solution)

    # Bootstrap on the fold solution runs and yields the expected component keys.
    boot = bootstrap_components(solution, perfect, n_boot=200, seed=505)
    assert set(boot.samples) == set(COMPONENT_KEYS)
    assert boot.n_pairs == len(solution)

    # The perfect fold submission beats the weak one beyond the band.
    assert is_improvement_beyond_band(
        solution, weak, perfect, n_boot=200, seed=505
    ) is True
