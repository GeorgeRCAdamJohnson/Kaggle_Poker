"""Official-metric equivalence self-check (task 2.6) — a single fixed-example gate.

# Feature: poker-anchor-reproduction, Property 3: For any fixed (solution, submission)
# input from the metric-kernel example, the local
# ``reference_public_metric.score`` SHALL equal the kernel's own score to full float
# tolerance; if not, the scorer SHALL refuse to run.

This is the INTEGRATION self-check the whole spec is built around (design.md §2
"MetricSelfCheck", Req 1.2 / 1.4). It is deliberately NOT a hypothesis property with
100 iterations — it is one fixed (solution, submission) example, scored once, that
asserts two load-bearing facts:

  1. EQUIVALENCE. The local verbatim official-metric copy
     (``poker_collusion.metric.reference_public_metric.score``) returns EXACTLY the
     same float as the competition metric KERNEL's own ``score()`` — extracted fresh
     from ``data/poker/_metric_kernel/slash-poker-competition-metric.ipynb`` and
     executed in isolation, so this compares our copy against the published source,
     not against itself (accountability contract Rules 1 & 9: the check is confirmed
     by something OUTSIDE the pipeline — the kernel's own code).

  2. REFUSAL ON DRIFT. If the local metric has drifted from the kernel, the
     CanonicalScorer refuses to run: its first scoring call routes through the
     task-2.1 hard gate (``require_metric_matches_kernel``), which raises
     ``MetricSelfCheckError``. A self-referential metric that silently disagrees with
     the external ground truth is exactly the failure this spec exists to prevent, so
     the scorer must NOT judge anything with a drifted metric.

"Full float tolerance" is read strictly: the local copy is meant to be a
character-for-character extraction of the kernel, so the two must return the IDENTICAL
float (exact bit equality), not merely a close one.

Validates: Requirements 1.2, 1.4
"""

from __future__ import annotations

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.metric import reference_public_metric

import anchor_repro.scorer as scorer_module
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.self_check import (
    MetricSelfCheckError,
    build_fixed_example,
    extract_kernel_score_fn,
    verify_metric_against_kernel,
)

_ROW_ID_COLUMN = "pair_id"


# --------------------------------------------------------------------------- #
# Minimal dev labels/evidence so a CanonicalScorer can be constructed without  #
# touching real data files (the refusal-on-drift path does not need to reach   #
# scoring — the gate fires first — but the scorer must construct cleanly).     #
# --------------------------------------------------------------------------- #
def _dev_labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "P1", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
            {"pair_id": "P2", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
        ]
    )


def _make_scorer() -> CanonicalScorer:
    return CanonicalScorer(ScoringRecipe(), PipelineConfig(), labels=_dev_labels())


# --------------------------------------------------------------------------- #
# Property 3 — part 1: EQUIVALENCE on the fixed example (executed once)         #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 3: local == kernel to full float tolerance
def test_local_metric_equals_kernel_on_fixed_example_full_float_tolerance():
    """The fixed (solution, submission) example scores IDENTICALLY under the local
    metric and the kernel's own score() — exact float equality (full tolerance)."""
    example = build_fixed_example()
    solution = example["solution"]
    submission = example["submission"]

    # LOCAL side: the verbatim reference_public_metric copy the scorer will use.
    local_value = float(
        reference_public_metric.score(solution.copy(), submission.copy(), _ROW_ID_COLUMN)
    )
    # KERNEL side: the published metric executed fresh from the notebook (external
    # ground truth, not a re-import of our local copy).
    kernel_score = extract_kernel_score_fn()
    kernel_value = float(
        kernel_score(solution.copy(), submission.copy(), _ROW_ID_COLUMN)
    )

    # Full float tolerance == EXACT equality (byte-identical operations on identical
    # inputs). Any difference at all is a drift and a release blocker.
    assert local_value == kernel_value
    assert abs(local_value - kernel_value) == 0.0
    # Non-degenerate: the example lands strictly inside (0, 1), so a divergence in ANY
    # metric component (PairAP / EvidenceMAP@5 / BehaviorMAP) would have surfaced here.
    assert 0.0 < local_value < 1.0


# Feature: poker-anchor-reproduction, Property 3: the packaged self-check agrees
def test_verify_metric_against_kernel_reports_exact_equivalence():
    """The packaged self-check confirms the same equivalence and marks it passed."""
    result = verify_metric_against_kernel()
    assert result.passed is True
    assert result.local_score == result.kernel_score  # full float tolerance
    assert result.abs_diff == 0.0


# --------------------------------------------------------------------------- #
# Property 3 — part 2: REFUSAL ON DRIFT (the scorer SHALL refuse to run)        #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 3: drift => scorer refuses to run
def test_scorer_refuses_to_run_when_metric_drifts_from_kernel(monkeypatch):
    """If the local metric drifts from the kernel, the scorer's hard gate raises
    MetricSelfCheckError and no scoring happens (Req 1.4: FAIL => release blocker)."""
    real_score = reference_public_metric.score

    def _drifted_score(solution, submission, row_id_column_name):
        # Perturb the local metric so it no longer equals the kernel score.
        return real_score(solution, submission, row_id_column_name) + 1e-6

    monkeypatch.setattr(reference_public_metric, "score", _drifted_score)

    # The self-check now measures a mismatch (records it, does not itself raise).
    drifted = verify_metric_against_kernel()
    assert drifted.passed is False
    assert drifted.abs_diff > 0.0

    # The CanonicalScorer routes its first scoring call through the hard gate, so a
    # drifted metric makes the scorer REFUSE to run rather than judge with it.
    scorer = _make_scorer()
    example = build_fixed_example()
    with pytest.raises(MetricSelfCheckError):
        scorer.score_dev_predictions(example["submission"])


# Feature: poker-anchor-reproduction, Property 3: no drift => scorer is cleared
def test_scorer_gate_clears_when_metric_matches_kernel():
    """With the real (matching) metric, the scorer's self-check gate passes so the
    scorer is cleared to run (the equivalence directly unblocks scoring)."""
    scorer = _make_scorer()
    result = scorer.ensure_metric_verified()
    assert result.passed is True
    assert result.local_score == result.kernel_score
