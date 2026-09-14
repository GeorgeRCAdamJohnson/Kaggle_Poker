"""§58b structural-hypotheses diagnostic — integration test (MEASUREMENT-ONLY).

Runs the pre-registered §58b diagnostic (RESEARCH_DOSSIER.md §58b) that MEASURES
which of three structural hypotheses — RINGS, POSITION, TABLE-LEVEL — carry real
label signal on a false-positive-defense split, BEFORE any detector is built.

This test asserts only that the diagnostic PRODUCES a well-formed, finite result
and a CONFIRM/NULL verdict per hypothesis (per the §58b decision rule); it does
NOT assert a particular verdict, because a null is a valid, pre-registered
deliverable (contract Rule 3). It builds NO detector, trains NO submission model,
and touches NO submission.csv.

DATA-GATED (contract Rules 1, 9 — no invented numbers): the diagnostic needs the
real ``data/poker`` files; if they are absent the test ``pytest.skip()``s with a
clear message rather than fabricating a result.

Run explicitly::

    pytest tests/test_structural_hypotheses_diag.py -m integration -q -s

Requirements: governed by contract Rules 1, 2, 3, 8, 9, 12, 16 (§58b).
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.integration

from poker_collusion.experiments import structural_hypotheses_diag as diag

_VALID_VERDICTS = {"CONFIRM", "NULL"}
_EXPECTED_HYPOTHESES = {"RINGS", "POSITION", "TABLE-LEVEL"}


def _data_available() -> bool:
    return diag._find_data_root() is not None


requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="real data/poker not available; §58b diagnostic is data-gated (Rule 1)",
)


@pytest.fixture(scope="module")
def report():
    if not _data_available():
        pytest.skip("real data/poker not available; §58b diagnostic is data-gated")
    return diag.run_diagnostic(seed=diag.SPLIT_SEED)


@requires_data
def test_split_is_false_positive_defense(report):
    """The split is a real DISCOVER/CONFIRM partition with a nonempty confirm set."""
    s = report.split
    assert s.seed == diag.SPLIT_SEED
    assert s.player_disjoint is True
    assert s.n_discover > 0
    assert s.n_confirm > 0
    # confirm split must contain BOTH classes for any lift to be measurable
    assert s.n_confirm_pos > 0
    assert s.n_confirm_neg > 0
    # player-disjoint => some cross-split pairs are dropped (dense player-sharing)
    assert s.n_shared_players_dropped >= 0


@requires_data
def test_produces_a_verdict_per_hypothesis(report):
    """Every one of the three hypotheses gets a CONFIRM/NULL verdict (§58b rule)."""
    names = {r.name for r in report.results}
    assert names == _EXPECTED_HYPOTHESES
    for r in report.results:
        assert r.verdict in _VALID_VERDICTS, f"{r.name} verdict={r.verdict!r}"


@requires_data
def test_lifts_and_aucs_are_finite_or_declared_nan(report):
    """Reported lifts/AUCs are real floats; NaN is allowed only where a flag set is
    empty by construction (e.g. player-disjoint rings) — never inf leaking into a
    verdict."""
    for r in report.results:
        # AUC, when defined, must be a probability in [0, 1]
        if not math.isnan(r.confirm_auc):
            assert 0.0 <= r.confirm_auc <= 1.0, f"{r.name} AUC out of range"
        # lifts must be finite when not NaN (no inf artifacts driving decisions)
        for lv in (r.lift_before_control, r.lift_after_control):
            assert not math.isinf(lv), f"{r.name} produced an inf lift"
        # base rates are well-formed
        assert r.base_rate_prereg == pytest.approx(diag.PREREG_BASE_RATE)
        assert 0.0 < r.base_rate_confirm < 1.0
        assert isinstance(r.detail, dict) and len(r.detail) > 0


@requires_data
def test_confirmers_ranked_consistently(report):
    """confirmers_ranked() returns only CONFIRM hypotheses, sorted by after-control
    lift descending (build-largest-first, Rule 18). If all NULL, the list is empty
    — a valid pre-registered deliverable (Rule 3)."""
    ranked = report.confirmers_ranked()
    assert all(r.verdict == "CONFIRM" for r in ranked)
    lifts = [r.lift_after_control for r in ranked if not math.isnan(r.lift_after_control)]
    assert lifts == sorted(lifts, reverse=True)


@requires_data
def test_table_level_reports_leakage_contrast(report):
    """TABLE-LEVEL must report BOTH the leakage-safe (discover-only) AUC and the
    circular within-confirm AUC, so a leakage-NULL is diagnosable (§58b Rule 9)."""
    tl = next(r for r in report.results if r.name == "TABLE-LEVEL")
    assert "confirm_discover_only_auc" in tl.detail
    assert "confirm_LEAKAGE_within_confirm_auc" in tl.detail
