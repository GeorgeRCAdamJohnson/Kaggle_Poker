"""Unit test for anchor-store schema compatibility (task 8.3).

The poker-anchor-reproduction ``AnchorStore`` (task 8.1,
:mod:`anchor_repro.anchor_store`) must write records that the parent
``poker-layered-tuning`` harness can consume VERBATIM. The parent access layer is
:func:`poker_collusion.tuning_harness.anchor_store.load_anchors` and the on-disk
record schema is :class:`poker_collusion.tuning_harness.models.LBAnchor` — the six
fields ``config_id / real_lb / measured_drift / holdout_ap / source / verified``.

This test proves an appended anchor stays schema-compatible with the seeded
``.kiro/specs/poker-collusion-detection/lb_anchors.json`` so the parent harness can
read it without modification. Specifically:

  (a) the on-disk record has EXACTLY the six parent schema fields at top level — the
      canonical ``recipe_id`` is NOT a new top-level key, it is embedded in ``source``;
  (b) the parent harness's ``load_anchors`` / ``LBAnchor`` loader consumes the appended
      record verbatim (no error) and round-trips the six fields;
  (c) the ``recipe_id`` is recoverable from ``source`` via the store's ``recipe_ids()``;
  (d) (shape-parity confirmation) the REAL seeded ``lb_anchors.json`` itself loads via
      BOTH the parent loader and ``AnchorStore.load()`` read-only, without mutation.

All writable assertions run against a tmp store seeded with the SAME ``_schema`` shape
as the real file (``init_store(overwrite=True)``). The real seeded store is only ever
READ (contract Rule 1: never mutate the externally-auditable trail).

Validates: Requirements 3.5
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Parent-harness access layer + record schema (what the child store must be compatible with).
from poker_collusion.tuning_harness.anchor_store import (
    DEFAULT_ANCHOR_STORE_PATH as PARENT_STORE_PATH,
)
from poker_collusion.tuning_harness.anchor_store import load_anchors as parent_load_anchors
from poker_collusion.tuning_harness.models import LBAnchor

# The child store under test.
from anchor_repro.anchor_store import Anchor, AnchorStore, RECIPE_ID_PROVENANCE_KEY

#: The exact six top-level fields the parent ``lb_anchors.json`` schema defines. An
#: appended record must carry EXACTLY these keys — no more (``recipe_id`` lives inside
#: ``source``), no fewer (all six are required by ``LBAnchor`` / ``load_anchors``).
PARENT_SCHEMA_FIELDS = {
    "config_id",
    "real_lb",
    "measured_drift",
    "holdout_ap",
    "source",
    "verified",
}


def _real_schema_block() -> dict:
    """The ``_schema`` block from the real seeded store (READ ONLY, never mutated).

    Seeding the tmp store with the identical ``_schema`` shape is what makes this a
    genuine schema-compatibility check rather than a test against an invented shape.
    """
    with PARENT_STORE_PATH.open("r", encoding="utf-8") as fh:
        real_store = json.load(fh)
    return real_store["_schema"]


@pytest.fixture()
def tmp_store(tmp_path: Path) -> AnchorStore:
    """A throwaway store seeded with the real ``_schema`` shape (never the real trail)."""
    store = AnchorStore(tmp_path / "lb_anchors.json")
    store.init_store(schema=_real_schema_block(), overwrite=True)
    return store


def test_appended_anchor_has_exactly_the_six_parent_schema_fields(tmp_store: AnchorStore):
    """(a) The on-disk record carries EXACTLY the six parent fields; recipe_id is in source."""
    anchor = Anchor(
        config_id="competitor_pu_value_transfer_064469",
        real_lb=0.64469,
        holdout_ap=0.3912,
        # Parent load_anchors does float(raw["measured_drift"]) — a real value keeps the
        # record consumable by the parent loader (a JSON null would break it).
        measured_drift=0.681,
        source="reproduced competitor notebook; nb_sha256=deadbeef; ladder verdict=NULL",
        recipe_id="canonical_v1",
        verified=True,
    )
    tmp_store.append(anchor)

    raw_store = json.loads(tmp_store.path.read_text(encoding="utf-8"))
    records = raw_store["anchors"]
    assert len(records) == 1
    record = records[0]

    # EXACTLY the six parent fields — no extra top-level key, none missing.
    assert set(record.keys()) == PARENT_SCHEMA_FIELDS

    # recipe_id is NOT a new top-level key ...
    assert "recipe_id" not in record
    # ... it is embedded inside the `source` provenance string instead.
    assert f"{RECIPE_ID_PROVENANCE_KEY}=canonical_v1" in record["source"]

    # The _schema block is preserved verbatim across the append (append-only shape parity).
    assert raw_store["_schema"] == _real_schema_block()


def test_parent_loader_consumes_appended_record_verbatim(tmp_store: AnchorStore):
    """(b) The parent harness's load_anchors / LBAnchor reads the appended record unchanged."""
    anchor = Anchor(
        config_id="competitor_pu_value_transfer_064469",
        real_lb=0.64469,
        holdout_ap=0.3912,
        measured_drift=0.681,
        source="reproduced competitor notebook; nb_sha256=deadbeef",
        recipe_id="canonical_v1",
        verified=True,
    )
    tmp_store.append(anchor)

    # The PARENT loader (not the child's) must consume the child-written store with no error.
    parent_anchors = parent_load_anchors(tmp_store.path)
    assert len(parent_anchors) == 1
    loaded = parent_anchors[0]
    assert isinstance(loaded, LBAnchor)

    # Round-trips the six schema fields exactly.
    assert loaded.config_id == "competitor_pu_value_transfer_064469"
    assert loaded.real_lb == pytest.approx(0.64469)
    assert loaded.measured_drift == pytest.approx(0.681)
    assert loaded.holdout_ap == pytest.approx(0.3912)
    assert loaded.verified is True
    # recipe_id survives inside source (the parent carries the whole string through).
    assert f"{RECIPE_ID_PROVENANCE_KEY}=canonical_v1" in loaded.source


def test_recipe_id_recoverable_from_source_via_store_helper(tmp_store: AnchorStore):
    """(c) recipe_id is recoverable from source via the store's recipe_ids() helper."""
    anchor = Anchor(
        config_id="competitor_pu_value_transfer_064469",
        real_lb=0.64469,
        holdout_ap=0.3912,
        measured_drift=0.681,
        source="reproduced competitor notebook; nb_sha256=deadbeef",
        recipe_id="canonical_v1",
        verified=True,
    )
    tmp_store.append(anchor)

    assert tmp_store.recipe_ids() == ["canonical_v1"]


def test_real_seeded_store_loads_via_both_loaders_readonly():
    """(d) The REAL seeded lb_anchors.json loads via BOTH loaders — read-only shape parity."""
    before = PARENT_STORE_PATH.read_bytes()

    # Parent loader.
    parent_anchors = parent_load_anchors(PARENT_STORE_PATH)
    # Child AnchorStore.load() over the same real file.
    child_anchors = AnchorStore(PARENT_STORE_PATH).load()

    assert len(parent_anchors) == len(child_anchors) >= 1

    # The two loaders agree on the six schema fields for every anchor (shape parity).
    for p, c in zip(parent_anchors, child_anchors):
        assert p.config_id == c.config_id
        assert p.real_lb == pytest.approx(c.real_lb)
        assert p.measured_drift == pytest.approx(c.measured_drift)
        assert p.holdout_ap == pytest.approx(c.holdout_ap)
        assert p.source == c.source
        assert p.verified == c.verified

    # Reading must NEVER mutate the externally-auditable trail (contract Rule 1).
    assert PARENT_STORE_PATH.read_bytes() == before
