"""Property test for anchor recording + recalibration (task 16.4).

# Feature: poker-layered-tuning, Property 21: Recording a real leaderboard result appends an anchor and recalibrates

Property 21 (design.md): For any candidate whose real leaderboard score returns,
recording it SHALL append exactly one new LB_Anchor to the store and SHALL
recalibrate the gate and projection over the augmented anchor set.

Validates: Requirements 7.3.

Under test:
``poker_collusion.tuning_harness.submission.gate.record_anchor_and_recalibrate``
which appends exactly one verified anchor via ``append_anchor`` and returns
``calibrate(augmented, floor_lb)``.

Each Hypothesis example operates on a FRESH temp store copy (created per example
via ``tempfile.mkdtemp`` and cleaned up), so appends never accumulate across
examples and the real append-only store is never mutated. A unique ``config_id``
is drawn per example so no duplicate-append rejection can occur.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.anchor_store import load_anchors
from poker_collusion.tuning_harness.models import (
    Calibration,
    Candidate,
    CandidateConfig,
    RankVector,
)
from poker_collusion.tuning_harness.submission.gate import (
    KNOWN_GOOD_FLOOR,
    record_anchor_and_recalibrate,
)

FLOOR_LB = KNOWN_GOOD_FLOOR

# A small fixed set of seed anchors written into every fresh temp store. Chosen
# with distinct holdout APs so the projection fit is well defined, and with a mix
# of PASS (real_lb >= floor) / FAIL (real_lb < floor) so recalibration is real.
_SEED_ANCHORS = [
    {
        "config_id": "seed_floor",
        "real_lb": 0.44519,
        "measured_drift": 0.679,
        "holdout_ap": 0.400,
        "source": "seed",
        "verified": True,
    },
    {
        "config_id": "seed_a",
        "real_lb": 0.4400,
        "measured_drift": 0.639,
        "holdout_ap": 0.395,
        "source": "seed",
        "verified": True,
    },
    {
        "config_id": "seed_b",
        "real_lb": 0.4200,
        "measured_drift": 0.865,
        "holdout_ap": 0.360,
        "source": "seed",
        "verified": True,
    },
]


def _fresh_seeded_store() -> tuple[Path, Path]:
    """Create a unique temp dir + a freshly seeded store file inside it.

    Returns ``(store_path, tmp_dir)`` so the caller can remove ``tmp_dir`` after
    the call. A fresh dir/file per call guarantees appends never accumulate
    across Hypothesis examples and the real store is never touched.
    """

    tmp_dir = Path(tempfile.mkdtemp(prefix="lb_anchor_prop_"))
    store_path = tmp_dir / "lb_anchors.json"
    store = {
        "_schema": {"note": "fresh temp copy for property test"},
        "anchors": [dict(a) for a in _SEED_ANCHORS],
    }
    store_path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    return store_path, tmp_dir


def _candidate() -> Candidate:
    cfg = CandidateConfig(layers_on=frozenset())
    return Candidate(config=cfg, ranking=RankVector(("p1", "p2", "p3")), layers_reported=())


@settings(max_examples=150, deadline=None)
@given(
    # Unique config_id per example -> never a duplicate-append rejection.
    config_id=st.uuids().map(lambda u: f"cand_{u.hex}"),
    # A real leaderboard score spanning the plausible LB range (PASS and FAIL).
    real_lb=st.floats(min_value=0.30, max_value=0.55, allow_nan=False, allow_infinity=False),
    # Holdout AP in [0, 1] and measured drift in the orientation-folded [0.5, 1.0].
    holdout_ap=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    measured_drift=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_record_appends_exactly_one_and_recalibrates(
    config_id: str,
    real_lb: float,
    holdout_ap: float,
    measured_drift: float,
) -> None:
    store_path, tmp_dir = _fresh_seeded_store()
    try:
        before = load_anchors(store_path)
        n_before = len(before)
        assert n_before == len(_SEED_ANCHORS)

        calib = record_anchor_and_recalibrate(
            real_lb=real_lb,
            candidate=_candidate(),
            holdout_ap=holdout_ap,
            measured_drift=measured_drift,
            config_id=config_id,
            floor_lb=FLOOR_LB,
            store_path=store_path,
        )

        after = load_anchors(store_path)

        # Exactly ONE new anchor was appended (append-only, +1).
        assert len(after) == n_before + 1

        # The appended anchor is the LAST one, carries the given config_id, the
        # recorded real_lb, and is verified (only a real external judge enters).
        appended = after[-1]
        assert appended.config_id == config_id
        assert appended.real_lb == real_lb
        assert appended.holdout_ap == holdout_ap
        assert appended.measured_drift == measured_drift
        assert appended.verified is True

        # Prior anchors are preserved unchanged (strictly append-only).
        assert [a.config_id for a in after[:-1]] == [a.config_id for a in before]

        # Recalibration ran over the AUGMENTED set and returned a Calibration
        # whose projection fit reflects the augmented count (recalibrated, not
        # the pre-append set).
        assert isinstance(calib, Calibration)
        assert calib.projection is not None
        assert calib.projection.n_points == len(after)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
