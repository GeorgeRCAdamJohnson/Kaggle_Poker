"""Unit tests for the official-metric self-check (task 2.1).

The self-check re-verifies the local verbatim official-metric copy
(``poker_collusion.metric.reference_public_metric.score``) against the KERNEL's own
``score()`` extracted fresh from
``data/poker/_metric_kernel/slash-poker-competition-metric.ipynb`` (Req 1.2 / 1.4).

These tests assert the load-bearing behaviors the canonical scorer depends on:

* the kernel notebook is locatable and its ``score()`` cell is extractable and callable;
* the local metric equals the kernel score EXACTLY (full float tolerance, not approx)
  on the fixed mixed example — the equality that clears the scorer to run;
* the fixed example is non-degenerate (strictly inside ``(0, 1)``) so a divergence in
  any metric component would surface;
* the hard gate (``require_metric_matches_kernel``) returns on a pass and raises
  ``MetricSelfCheckError`` on a simulated drift (FAIL => release blocker);
* a missing/invalid kernel path is a loud blocker, never a silent pass.

Validates: Requirements 1.2, 1.4
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poker_collusion.metric import reference_public_metric

from anchor_repro.self_check import (
    MetricSelfCheckError,
    SelfCheckResult,
    build_fixed_example,
    extract_kernel_score_fn,
    locate_kernel_notebook,
    require_metric_matches_kernel,
    verify_metric_against_kernel,
)


# --------------------------------------------------------------------------- #
# Kernel location + extraction                                                #
# --------------------------------------------------------------------------- #
def test_kernel_notebook_is_locatable():
    path = locate_kernel_notebook()
    assert path.is_file()
    assert path.name == "slash-poker-competition-metric.ipynb"


def test_missing_kernel_is_a_loud_blocker(tmp_path):
    """A missing kernel notebook raises rather than silently passing (Rules 1, 9)."""
    with pytest.raises(MetricSelfCheckError):
        locate_kernel_notebook(tmp_path / "does_not_exist.ipynb")


def test_kernel_score_fn_is_extractable_and_callable():
    kernel_score = extract_kernel_score_fn()
    assert callable(kernel_score)
    example = build_fixed_example()
    value = kernel_score(
        example["solution"].copy(), example["submission"].copy(), "pair_id"
    )
    assert isinstance(value, float)


def test_extracted_kernel_is_not_the_local_reimport():
    """The extracted score() is the KERNEL's own object, not our local re-import."""
    kernel_score = extract_kernel_score_fn()
    assert kernel_score is not reference_public_metric.score


# --------------------------------------------------------------------------- #
# The self-check itself                                                       #
# --------------------------------------------------------------------------- #
def test_fixed_example_is_non_degenerate():
    """The fixed example scores strictly inside (0, 1) so all components matter."""
    example = build_fixed_example()
    value = reference_public_metric.score(
        example["solution"].copy(), example["submission"].copy(), "pair_id"
    )
    assert 0.0 < value < 1.0


def test_local_metric_equals_kernel_exactly():
    """Property 3 core: local == kernel to FULL float tolerance (exact equality)."""
    result = verify_metric_against_kernel()
    assert isinstance(result, SelfCheckResult)
    assert result.passed is True
    # Full float tolerance means EXACT equality, not merely close.
    assert result.local_score == result.kernel_score
    assert result.abs_diff == 0.0
    assert result.example_name == "fixed_5pair_mixed"
    assert result.kernel_notebook.name == "slash-poker-competition-metric.ipynb"
    assert "MEASURED" in result.detail
    # The audit trail records the compared example.
    assert result.extra_checks and result.extra_checks[0]["abs_diff"] == 0.0


def test_hard_gate_returns_on_pass():
    """require_metric_matches_kernel returns the passing result without raising."""
    result = require_metric_matches_kernel()
    assert result.passed is True


def test_hard_gate_raises_on_simulated_drift(monkeypatch):
    """A drifted local metric is a hard release blocker: the gate MUST raise.

    We simulate drift by monkeypatching the local metric to return a different
    number than the kernel, then assert the self-check reports ``passed=False`` and
    the hard gate raises ``MetricSelfCheckError`` (Req 1.4).
    """
    real_score = reference_public_metric.score

    def _drifted_score(solution, submission, row_id_column_name):
        return real_score(solution, submission, row_id_column_name) + 1e-6

    monkeypatch.setattr(reference_public_metric, "score", _drifted_score)

    # The plain check records the mismatch (does not raise for a genuine mismatch).
    result = verify_metric_against_kernel()
    assert result.passed is False
    assert result.abs_diff > 0.0
    assert "release blocker" in result.detail

    # The hard gate turns that mismatch into a raise (scorer refuses to run).
    with pytest.raises(MetricSelfCheckError):
        require_metric_matches_kernel()
