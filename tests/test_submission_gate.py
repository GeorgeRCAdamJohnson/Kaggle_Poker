"""Wiring unit tests for the submission gate (task 16.1, ``submission/gate.py``).

These are focused example tests that exercise the module's *wiring* -- how
``submission_report``, ``record_anchor_and_recalibrate``, the guarded floor
writer, and the stop-loss state machine sequence the reused pieces. The 23
correctness properties (16.2-16.5) and the dedicated HALT property (16.6) are
separate tasks and are NOT implemented here.

Covered here:
- The report is complete (every Req 7.1 field present) and a drift FAIL forces
  ``recommend_submit == False`` regardless of holdout AP (Req 7.1, 7.2).
- ``record_anchor_and_recalibrate`` appends EXACTLY one anchor to a temp store
  copy and recalibrates over the augmented set (Req 7.3).
- The guarded writer refuses a non-improving write leaving the artifact
  byte-for-byte unchanged, and allows a verified-improvement write (Req 7.4).
- Two consecutive regressions versus the floor trip HALT (Gate C, Req 8.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_collusion.tuning_harness.anchor_store import (
    AnchorStoreError,
    load_anchors,
)
from poker_collusion.tuning_harness.models import (
    Calibration,
    Candidate,
    CandidateConfig,
    DriftResult,
    LBAnchor,
    ProjectionFit,
    RankVector,
    SeparationCheck,
)
from poker_collusion.tuning_harness.submission.gate import (
    FloorWriteRefused,
    KNOWN_GOOD_FLOOR,
    guarded_floor_write,
    record_anchor_and_recalibrate,
    stop_loss_state,
    submission_report,
)

FLOOR_LB = KNOWN_GOOD_FLOOR


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _candidate(layers: frozenset[str] = frozenset()) -> Candidate:
    cfg = CandidateConfig(layers_on=layers)
    ranking = RankVector(pair_ids=("p1", "p2", "p3"))
    return Candidate(config=cfg, ranking=ranking, layers_reported=tuple(sorted(layers)))


def _calibration(fit: ProjectionFit) -> Calibration:
    """A minimal TRUSTED calibration carrying an explicit projection fit."""

    sep = SeparationCheck(max_pass_drift=0.68, min_fail_drift=0.80, separable=True)
    return Calibration(
        threshold=0.74,
        threshold_kind="LB_CALIBRATED",
        trusted=True,
        separation=sep,
        null_surfaced=False,
        projection=fit,
    )


def _in_regime_fit() -> ProjectionFit:
    """A fit whose regime covers holdout_ap = 0.38 so projections are verified."""

    return ProjectionFit(
        slope=1.4235,
        intercept=-0.1414,
        resid_std=0.0022,
        fit_min_ap=0.30,
        fit_max_ap=0.42,
        n_points=3,
    )


def _seed_store(tmp_path: Path) -> Path:
    """Write a minimal 3-anchor append-only store to a temp file and return it."""

    store = {
        "_schema": {"note": "temp copy for tests"},
        "anchors": [
            {
                "config_id": "floor",
                "real_lb": 0.44519,
                "measured_drift": 0.679,
                "holdout_ap": 0.400,
                "source": "seed",
                "verified": True,
            },
            {
                "config_id": "cfg_a",
                "real_lb": 0.4400,
                "measured_drift": 0.639,
                "holdout_ap": 0.395,
                "source": "seed",
                "verified": True,
            },
            {
                "config_id": "cfg_b",
                "real_lb": 0.4200,
                "measured_drift": 0.865,
                "holdout_ap": 0.360,
                "source": "seed",
                "verified": True,
            },
        ],
    }
    store_path = tmp_path / "lb_anchors.json"
    store_path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    return store_path


# --------------------------------------------------------------------------- #
# submission_report: completeness + drift-FAIL forces recommend_submit False
# --------------------------------------------------------------------------- #
def test_submission_report_is_complete_and_recommends_on_pass() -> None:
    candidate = _candidate(frozenset({"board_equity"}))
    calib = _calibration(_in_regime_fit())
    drift = DriftResult(
        adversarial_auc=0.62, verdict="PASS", threshold=0.74, threshold_kind="LB_CALIBRATED"
    )

    report = submission_report(candidate, holdout_ap=0.38, drift=drift, calib=calib)

    # Every Req 7.1 field is present and consistent with the inputs.
    assert report.layers_on == ("board_equity",)
    assert report.holdout_ap == pytest.approx(0.38)
    assert report.measured_drift == pytest.approx(0.62)
    assert report.gate_verdict == "PASS"
    # Projection is derived from the calibrated line, not the holdout AP.
    assert report.projection.point == pytest.approx(1.4235 * 0.38 - 0.1414)
    assert report.projection.uncertainty == pytest.approx(0.0022)
    assert report.regime_verified is True
    # PASS + in-regime -> recommended.
    assert report.recommend_submit is True


def test_drift_fail_forces_no_recommendation_regardless_of_holdout() -> None:
    candidate = _candidate(frozenset({"whipsaw"}))
    calib = _calibration(_in_regime_fit())
    # A drift FAIL with an absurdly HIGH holdout AP: the holdout must not rescue it.
    drift = DriftResult(
        adversarial_auc=0.92, verdict="FAIL", threshold=0.74, threshold_kind="LB_CALIBRATED"
    )

    report = submission_report(candidate, holdout_ap=0.99, drift=drift, calib=calib)

    assert report.gate_verdict == "FAIL"
    assert report.measured_drift == pytest.approx(0.92)
    # Req 7.2 / Gate C: never recommended on a drift FAIL, regardless of holdout AP.
    assert report.recommend_submit is False


def test_report_falls_back_to_layers_on_when_reporting_failed() -> None:
    # layers_reported None (composer reporting failure) -> report still non-empty.
    cfg = CandidateConfig(layers_on=frozenset({"iso", "directional"}))
    candidate = Candidate(config=cfg, ranking=RankVector(("p1",)), layers_reported=None)
    calib = _calibration(_in_regime_fit())
    drift = DriftResult(0.60, "PASS", 0.74, "LB_CALIBRATED")

    report = submission_report(candidate, holdout_ap=0.38, drift=drift, calib=calib)

    assert report.layers_on == ("directional", "iso")


# --------------------------------------------------------------------------- #
# record_anchor_and_recalibrate: appends ONE anchor + recalibrates (temp store)
# --------------------------------------------------------------------------- #
def test_record_anchor_appends_exactly_one_and_recalibrates(tmp_path: Path) -> None:
    store_path = _seed_store(tmp_path)
    before = load_anchors(store_path)
    assert len(before) == 3

    candidate = _candidate(frozenset({"negative_space"}))
    calib = record_anchor_and_recalibrate(
        real_lb=0.4460,
        candidate=candidate,
        holdout_ap=0.401,
        measured_drift=0.70,
        store_path=store_path,
        floor_lb=FLOOR_LB,
    )

    after = load_anchors(store_path)
    # Exactly ONE new anchor appended (the just-scored candidate).
    assert len(after) == 4
    assert after[-1].config_id == "floor+negative_space"
    assert after[-1].real_lb == pytest.approx(0.4460)
    assert after[-1].verified is True
    # Recalibration ran over the AUGMENTED set and returned a Calibration.
    assert isinstance(calib, Calibration)
    # The projection fit reflects 4 points now (recalibrated over augmented set).
    assert calib.projection is not None
    assert calib.projection.n_points == 4


def test_record_anchor_rejects_duplicate_config(tmp_path: Path) -> None:
    store_path = _seed_store(tmp_path)
    # "floor" already exists in the seeded store -> a re-submit of the same stack
    # must be rejected by the append-only store (reused invariant).
    candidate = _candidate(frozenset())  # -> config_id "floor"
    with pytest.raises(AnchorStoreError):
        record_anchor_and_recalibrate(
            real_lb=0.4460,
            candidate=candidate,
            holdout_ap=0.40,
            measured_drift=0.68,
            store_path=store_path,
        )
    # The store is unchanged (still 3 anchors) after a rejected append.
    assert len(load_anchors(store_path)) == 3


# --------------------------------------------------------------------------- #
# Guarded floor writer: refuses non-improving (bytes unchanged), allows improvement
# --------------------------------------------------------------------------- #
def _make_floor_and_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    floor = tmp_path / "submission_best_044519.csv"
    floor.write_bytes(b"pair_id,risk_score\np1,0.9\np2,0.1\n")
    candidate = tmp_path / "candidate.csv"
    candidate.write_bytes(b"pair_id,risk_score\np1,0.8\np2,0.2\n")
    dossier = tmp_path / "RESEARCH_DOSSIER.md"
    dossier.write_text("# dossier\n", encoding="utf-8")
    return floor, candidate, dossier


def test_guarded_writer_refuses_non_improving_write_bytes_unchanged(tmp_path: Path) -> None:
    floor, candidate, dossier = _make_floor_and_candidate(tmp_path)
    original_bytes = floor.read_bytes()

    with pytest.raises(FloorWriteRefused):
        guarded_floor_write(
            candidate,
            real_lb=0.4400,  # below the current floor -> not an improvement
            current_floor_lb=FLOOR_LB,
            floor_path=floor,
            dossier_path=dossier,
            verified=True,
        )

    # Req 7.4: the artifact is left byte-for-byte unchanged on refusal.
    assert floor.read_bytes() == original_bytes
    # The refusal was logged to the dossier (externally auditable).
    assert "FLOOR WRITE REFUSED" in dossier.read_text(encoding="utf-8")


def test_guarded_writer_refuses_unverified_even_if_higher(tmp_path: Path) -> None:
    floor, candidate, dossier = _make_floor_and_candidate(tmp_path)
    original_bytes = floor.read_bytes()

    # A higher score that is NOT leaderboard-verified (a projection) is refused.
    with pytest.raises(FloorWriteRefused):
        guarded_floor_write(
            candidate,
            real_lb=0.5000,
            current_floor_lb=FLOOR_LB,
            floor_path=floor,
            dossier_path=dossier,
            verified=False,
        )
    assert floor.read_bytes() == original_bytes


def test_guarded_writer_allows_verified_improvement(tmp_path: Path) -> None:
    floor, candidate, dossier = _make_floor_and_candidate(tmp_path)
    candidate_bytes = candidate.read_bytes()

    result = guarded_floor_write(
        candidate,
        real_lb=0.4600,  # strictly above the floor
        current_floor_lb=FLOOR_LB,
        floor_path=floor,
        dossier_path=dossier,
        verified=True,
    )

    # The floor now holds the candidate's bytes (promoted).
    assert result == floor
    assert floor.read_bytes() == candidate_bytes
    assert "FLOOR WRITE ACCEPTED" in dossier.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Stop-loss HALT state machine (Gate C, Req 8.3)
# --------------------------------------------------------------------------- #
def test_two_consecutive_regressions_trip_halt() -> None:
    # Req 8.3 / Gate C: two consecutive scores below the Known_Good_Floor emit a
    # HALT that BLOCKS new candidates. This is the dedicated unit test for the
    # two-consecutive-regression HALT (task 16.6).
    state = stop_loss_state([0.4450, 0.4300], floor_lb=FLOOR_LB)
    assert state.halt is True
    # The HALT must block any new candidate from proceeding (Gate C).
    assert state.blocks_new_candidates is True
    assert state.consecutive_regressions == 2


def test_single_regression_does_not_halt_or_block() -> None:
    # Req 8.3 contrast: a SINGLE regression below the floor is below the
    # two-regression tripwire -> no HALT and new candidates are NOT blocked.
    state = stop_loss_state([0.4600, 0.4300], floor_lb=FLOOR_LB)
    assert state.halt is False
    assert state.blocks_new_candidates is False
    assert state.consecutive_regressions == 1


def test_regressions_separated_by_a_hold_do_not_halt() -> None:
    # regress, hold, regress -> no TWO consecutive regressions -> no HALT.
    state = stop_loss_state([0.4300, 0.4600, 0.4300], floor_lb=FLOOR_LB)
    assert state.halt is False
    assert state.blocks_new_candidates is False
    assert state.consecutive_regressions == 1


def test_halt_latches_even_if_a_later_hold_follows() -> None:
    # Two consecutive regressions early, then a hold: Gate C already tripped.
    state = stop_loss_state([0.4300, 0.4200, 0.4600], floor_lb=FLOOR_LB)
    assert state.halt is True


def test_stop_loss_accepts_markers() -> None:
    state = stop_loss_state(["regress", "regress"], floor_lb=FLOOR_LB)
    assert state.halt is True
    ok = stop_loss_state(["pass", "regress"], floor_lb=FLOOR_LB)
    assert ok.halt is False
