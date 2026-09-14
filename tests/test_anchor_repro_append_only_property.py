"""Property test for append-only AnchorStore integrity (task 8.2).

# Feature: poker-anchor-reproduction, Property 6: For any sequence of ``append``
operations, every previously stored anchor SHALL remain byte-unchanged, no anchor
SHALL be deleted, and an anchor SHALL NOT be persisted with ``verified=true`` unless
backed by a real external LB value.

Validates: Requirements 3.5, 4.2.

Governed by the workspace ``reverse-engineering-accountability`` contract (Rule 1:
``verified=true`` only with a real external judge). This test NEVER touches the real
seeded ``.kiro/specs/poker-collusion-detection/lb_anchors.json`` — every Hypothesis
example runs against an isolated throwaway store created with
``AnchorStore.init_store(overwrite=True)`` on a fresh temp path, so appends never
accumulate across examples and the real trail is untouched.

Under test: :class:`anchor_repro.anchor_store.AnchorStore` (task 8.1). Each generated
append operation is one of four kinds so the property exercises the whole guard
surface in one sequence:

* a fresh, valid anchor (unique ``config_id``) — MUST be appended;
* a duplicate ``config_id`` — MUST raise ``AnchorStoreError`` and NOT edit the store
  (assertion (b));
* a ``verified=True`` anchor with a bad ``real_lb`` (``None`` / 0 / negative / NaN /
  inf) — MUST be refused and NOT persisted (assertion (c), contract Rule 1);
* a ``verified=True`` anchor with a real positive ``real_lb`` — MUST be accepted
  (assertion (d)).

After EVERY operation (accepted or refused) we assert:
  (a) every previously stored anchor record is byte-unchanged (compare the serialized
      JSON records) and the ``_schema`` block is preserved;
      the anchor count only ever GROWS (nothing is deleted).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from anchor_repro.anchor_store import Anchor, AnchorStore, AnchorStoreError


# --------------------------------------------------------------------------- #
# A schema block mirroring the parent lb_anchors.json shape (never the real file). #
# --------------------------------------------------------------------------- #
_SCHEMA = {
    "note": "throwaway temp store for the Property 6 append-only test",
    "fields": ["config_id", "real_lb", "measured_drift", "holdout_ap", "source", "verified"],
}


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #
# A "real, positive external LB" value — the only kind that may back verified=True.
_good_lb = st.floats(min_value=1e-5, max_value=1.0, allow_nan=False, allow_infinity=False)

# A "bad" LB that must NEVER back a verified anchor: non-positive or non-finite.
_bad_lb = st.one_of(
    st.just(0.0),
    st.floats(min_value=-1.0, max_value=-1e-6, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.none(),
)

_holdout_ap = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_drift = st.one_of(
    st.none(),
    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
)
_recipe = st.sampled_from(["canonical_v1", "canonical_v2", None])


def _fresh_anchor(config_id: str, real_lb, holdout_ap, drift, verified, recipe):
    return Anchor(
        config_id=config_id,
        real_lb=real_lb,
        holdout_ap=holdout_ap,
        measured_drift=drift,
        source="property-test",
        recipe_id=recipe,
        verified=verified,
    )


# One operation to apply to the store. ``kind`` selects which invariant it exercises.
_operation = st.fixed_dictionaries(
    {
        "kind": st.sampled_from(["fresh", "dup", "bad_verified", "good_verified"]),
        "real_lb": _good_lb,
        "bad_real_lb": _bad_lb,
        "holdout_ap": _holdout_ap,
        "drift": _drift,
        "recipe": _recipe,
        # verified flag used for the plain "fresh" op (may be a legit verified w/ good lb).
        "verified": st.booleans(),
    }
)


def _records(store: AnchorStore) -> List[dict]:
    """The raw serialized anchor record dicts currently on disk (byte-comparable)."""
    return list(store.load_store().get("anchors", []))


@settings(max_examples=120, deadline=None)
@given(operations=st.lists(_operation, min_size=1, max_size=25))
def test_append_only_integrity(operations) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="anchor_append_prop_"))
    store_path = tmp_dir / "lb_anchors.json"
    try:
        store = AnchorStore(store_path)
        # Isolated throwaway store — NEVER the real seeded trail.
        store.init_store(schema=_SCHEMA, overwrite=True)

        # Baseline: empty anchors, schema preserved.
        doc0 = store.load_store()
        assert doc0["anchors"] == []
        assert doc0["_schema"] == _SCHEMA

        prev_records = _records(store)
        seen_config_ids: List[str] = []
        counter = 0

        for op in operations:
            counter += 1
            kind = op["kind"]

            if kind == "dup" and seen_config_ids:
                # (b) Re-appending an existing config_id must be refused with no edit.
                dup_id = seen_config_ids[0]
                anchor = _fresh_anchor(
                    dup_id, op["real_lb"], op["holdout_ap"], op["drift"], False, op["recipe"]
                )
                _expect_refused(store, anchor, prev_records)

            elif kind == "bad_verified":
                # (c) verified=True with a bad real_lb must be refused, not persisted.
                cfg = f"cfg_bad_{counter}"
                anchor = _fresh_anchor(
                    cfg, op["bad_real_lb"], op["holdout_ap"], op["drift"], True, op["recipe"]
                )
                _expect_refused(store, anchor, prev_records)
                # The refused config_id must NOT have leaked into the store.
                assert all(r["config_id"] != cfg for r in _records(store))

            elif kind == "good_verified":
                # (d) verified=True with a real positive real_lb must be accepted.
                cfg = f"cfg_good_{counter}"
                anchor = _fresh_anchor(
                    cfg, op["real_lb"], op["holdout_ap"], op["drift"], True, op["recipe"]
                )
                prev_records = _expect_appended(store, anchor, prev_records)
                seen_config_ids.append(cfg)
                # An accepted verified anchor is backed by a real positive external LB.
                last = _records(store)[-1]
                assert last["verified"] is True
                assert last["real_lb"] > 0.0

            else:  # "fresh"
                cfg = f"cfg_fresh_{counter}"
                verified = op["verified"]
                # A "fresh" op with verified=True still needs a good LB (else it's a
                # bad_verified case) — pair verified with the good LB here.
                anchor = _fresh_anchor(
                    cfg, op["real_lb"], op["holdout_ap"], op["drift"], verified, op["recipe"]
                )
                prev_records = _expect_appended(store, anchor, prev_records)
                seen_config_ids.append(cfg)

        # Final invariant: the on-disk anchor count equals the number of accepted
        # appends (nothing was deleted; refusals added nothing).
        assert len(_records(store)) == len(seen_config_ids)
        # Schema block survived the whole sequence untouched.
        assert store.load_store()["_schema"] == _SCHEMA
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _expect_refused(store: AnchorStore, anchor: Anchor, prev_records: List[dict]) -> None:
    """Assert the append is refused AND the store is byte-unchanged (a) + (b)/(c)."""
    n_before = len(_records(store))
    try:
        store.append(anchor)
    except AnchorStoreError:
        pass
    else:  # pragma: no cover - a refused op that got accepted is a hard failure.
        raise AssertionError(f"append should have been refused for {anchor.config_id!r}")

    after = _records(store)
    # No anchor added, none deleted.
    assert len(after) == n_before
    # Every previously stored record is byte-unchanged (compare serialized JSON).
    assert _dumps(after) == _dumps(prev_records)


def _expect_appended(
    store: AnchorStore, anchor: Anchor, prev_records: List[dict]
) -> List[dict]:
    """Assert the append is accepted, appended at the END, and all prior records are
    byte-unchanged (a). Returns the new record list for the next comparison."""
    n_before = len(_records(store))
    store.append(anchor)
    after = _records(store)

    # Count grew by exactly one (append-only, +1, nothing deleted).
    assert len(after) == n_before + 1
    # The new anchor is the LAST record; all prior records are byte-unchanged.
    assert _dumps(after[:-1]) == _dumps(prev_records)
    assert after[-1]["config_id"] == anchor.config_id
    return after


def _dumps(records: List[dict]) -> str:
    """Canonical byte-comparable serialization of a record list."""
    return json.dumps(records, sort_keys=True)
