"""Unit tests for the append-only LB_Anchor store access layer.

Confirms the store loads the seeded ``known_good_floor`` anchor and that a
verified append round-trips, while the append-only / verified invariants reject
duplicate config ids and non-verified values.

The real seeded store is only READ; every append test runs against a temp-dir
copy so the externally-auditable file is never mutated by tests (Req 8.7).

Requirements: 7.1, 8.7.
"""

from __future__ import annotations

import json
import shutil

import pytest

from poker_collusion.tuning_harness.anchor_store import (
    DEFAULT_ANCHOR_STORE_PATH,
    AnchorStoreError,
    append_anchor,
    load_anchors,
    load_store,
)
from poker_collusion.tuning_harness.models import LBAnchor


def test_real_store_loads_seeded_known_good_floor():
    """The seeded store loads and contains the known_good_floor anchor (read-only)."""
    anchors = load_anchors(DEFAULT_ANCHOR_STORE_PATH)
    assert len(anchors) >= 1
    floor = anchors[0]
    assert floor.config_id == "known_good_floor"
    assert floor.real_lb == pytest.approx(0.44519)
    assert floor.measured_drift == pytest.approx(0.679)
    assert floor.holdout_ap == pytest.approx(0.3839)
    assert floor.verified is True


@pytest.fixture()
def store_copy(tmp_path):
    """A writable temp-dir copy of the real seeded store."""
    dst = tmp_path / "lb_anchors.json"
    shutil.copyfile(DEFAULT_ANCHOR_STORE_PATH, dst)
    return dst


def test_append_verified_anchor_round_trips(store_copy):
    """Appending a verified anchor persists it and preserves the seed + schema."""
    before = load_anchors(store_copy)
    new = LBAnchor(
        config_id="probe_candidate_v1",
        real_lb=0.45,
        measured_drift=0.66,
        holdout_ap=0.39,
        source="unit test round-trip",
        verified=True,
    )

    returned = append_anchor(new, store_copy)

    # Return value and a fresh load agree, and the count grew by exactly one.
    reloaded = load_anchors(store_copy)
    assert [a.config_id for a in returned] == [a.config_id for a in reloaded]
    assert len(reloaded) == len(before) + 1

    # Seed anchor is preserved unchanged at the front (append-only).
    assert reloaded[0].config_id == "known_good_floor"
    assert reloaded[0].real_lb == pytest.approx(0.44519)

    # New anchor round-trips field-for-field at the end.
    appended = reloaded[-1]
    assert appended.config_id == "probe_candidate_v1"
    assert appended.real_lb == pytest.approx(0.45)
    assert appended.measured_drift == pytest.approx(0.66)
    assert appended.holdout_ap == pytest.approx(0.39)
    assert appended.verified is True

    # The _schema provenance block survives the write.
    assert "_schema" in load_store(store_copy)


def test_append_rejects_duplicate_config_id(store_copy):
    """A config_id already in the store cannot be appended again (append-only)."""
    dup = LBAnchor(
        config_id="known_good_floor",
        real_lb=0.44519,
        measured_drift=0.679,
        holdout_ap=0.3839,
        verified=True,
    )
    with pytest.raises(AnchorStoreError, match="duplicate config_id"):
        append_anchor(dup, store_copy)

    # Store is unchanged after the rejected append.
    assert len(load_anchors(store_copy)) == 1


def test_append_rejects_non_verified_anchor(store_copy):
    """A non-verified / projected value is rejected: externals only (Rule 1)."""
    projected = LBAnchor(
        config_id="projected_only",
        real_lb=0.46,
        measured_drift=0.64,
        holdout_ap=0.40,
        verified=False,
    )
    with pytest.raises(AnchorStoreError, match="non-verified"):
        append_anchor(projected, store_copy)

    assert [a.config_id for a in load_anchors(store_copy)] == ["known_good_floor"]


def test_missing_store_raises(tmp_path):
    """A missing store is a real error, not silently created."""
    with pytest.raises(FileNotFoundError):
        load_store(tmp_path / "does_not_exist.json")


def test_stored_json_stays_valid_after_append(store_copy):
    """The persisted file remains valid JSON with schema + anchors (auditable)."""
    append_anchor(
        LBAnchor(
            config_id="probe_candidate_v2",
            real_lb=0.44,
            measured_drift=0.70,
            holdout_ap=0.38,
            verified=True,
        ),
        store_copy,
    )
    with store_copy.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert set(doc.keys()) >= {"_schema", "anchors"}
    assert len(doc["anchors"]) == 2
