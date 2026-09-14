"""Append-only access layer for the LB_Anchor store.

The LB_Anchor store is the small, externally-auditable JSON file that lives
alongside ``RESEARCH_DOSSIER.md`` (at
``.kiro/specs/poker-collusion-detection/lb_anchors.json``). It records, for every
ACTUALLY-submitted candidate, the ``(config_id, real_lb, measured_drift,
holdout_ap)`` tuple returned by the external leaderboard judge. It is the
substrate the calibrator turns into the LB_Calibrated_Threshold and the
LB_Projection fit.

This module is the *access layer only* - it loads the store and appends new
anchors. It does NOT recreate the store (the seed anchor ``known_good_floor`` is
already present) and it does NOT calibrate anything (that is a later task).

Append semantics (accountability contract Rule 1 + Req 8.7 - the trail must stay
externally auditable and honest):

- **Strictly append-only.** Existing anchors are never edited or deleted. A new
  anchor is added to the end of the ``anchors`` list; the ``_schema`` block and
  every prior anchor are preserved byte-for-byte in meaning.
- **No duplicate ``config_id``.** Appending an anchor whose ``config_id`` already
  exists is rejected - the store must not carry two records for one config.
- **Verified externals only.** An anchor must be ``verified`` (confirmed by the
  real leaderboard). A non-verified / merely-projected value is rejected: only a
  real external judge may enter the store.

Requirements:
- 7.1: anchors feed the calibrated Drift_Gate and the LB_Projection reported in
  the submission report.
- 8.7: the store is the append-only, externally-auditable audit trail; these
  functions enforce that it can only grow with real, verified results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from poker_collusion.tuning_harness.models import LBAnchor

__all__ = [
    "DEFAULT_ANCHOR_STORE_PATH",
    "AnchorStoreError",
    "load_anchors",
    "load_store",
    "append_anchor",
]

#: Canonical location of the seeded, append-only LB_Anchor store.
DEFAULT_ANCHOR_STORE_PATH: Path = (
    Path(__file__).resolve().parents[3]
    / ".kiro"
    / "specs"
    / "poker-collusion-detection"
    / "lb_anchors.json"
)


class AnchorStoreError(Exception):
    """Raised when an append would violate the append-only / verified invariants.

    Covers a duplicate ``config_id`` and a non-verified (projected) value - the
    two ways an append could corrupt the externally-auditable trail (Req 8.7,
    accountability contract Rule 1).
    """


def _resolve(path: str | Path | None) -> Path:
    return Path(path) if path is not None else DEFAULT_ANCHOR_STORE_PATH


def load_store(path: str | Path | None = None) -> dict:
    """Load the raw store document (``_schema`` block + ``anchors`` list).

    Returns the parsed JSON exactly as stored so callers that need the schema
    provenance keep it. Raises ``FileNotFoundError`` if the store is missing -
    the store is seeded, so absence is a real error, not something to paper over.
    """

    store_path = _resolve(path)
    if not store_path.exists():
        raise FileNotFoundError(f"LB_Anchor store not found at {store_path!s}")
    with store_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_anchors(path: str | Path | None = None) -> List[LBAnchor]:
    """Load the store and return its anchors as :class:`LBAnchor` records.

    Preserves store order (the seeded ``known_good_floor`` is first). Unknown
    extra keys in a stored anchor are ignored so the schema can grow without
    breaking the loader.
    """

    store = load_store(path)
    anchors: List[LBAnchor] = []
    for raw in store.get("anchors", []):
        anchors.append(
            LBAnchor(
                config_id=raw["config_id"],
                real_lb=float(raw["real_lb"]),
                measured_drift=float(raw["measured_drift"]),
                holdout_ap=float(raw["holdout_ap"]),
                source=str(raw.get("source", "")),
                verified=bool(raw.get("verified", False)),
            )
        )
    return anchors


def append_anchor(anchor: LBAnchor, path: str | Path | None = None) -> List[LBAnchor]:
    """Append one verified anchor to the store, enforcing the invariants.

    Rejects (raising :class:`AnchorStoreError`) when:
    - ``anchor.verified`` is False - only real external results may enter; a
      projected value is never an anchor (contract Rule 1).
    - an anchor with the same ``config_id`` already exists - the store is
      strictly append-only and holds one record per config (Req 8.7).

    On success the anchor is added to the end of the ``anchors`` list and the
    updated store is written back with the ``_schema`` block and all prior
    anchors preserved. Returns the full anchor list after the append.
    """

    if not anchor.verified:
        raise AnchorStoreError(
            f"Refusing to append non-verified anchor {anchor.config_id!r}: only a "
            "real, external leaderboard result may enter the LB_Anchor store "
            "(accountability contract Rule 1)."
        )

    store_path = _resolve(path)
    store = load_store(store_path)
    existing = store.setdefault("anchors", [])

    for raw in existing:
        if raw.get("config_id") == anchor.config_id:
            raise AnchorStoreError(
                f"Refusing to append duplicate config_id {anchor.config_id!r}: the "
                "LB_Anchor store is append-only and holds one record per config."
            )

    existing.append(
        {
            "config_id": anchor.config_id,
            "real_lb": anchor.real_lb,
            "measured_drift": anchor.measured_drift,
            "holdout_ap": anchor.holdout_ap,
            "source": anchor.source,
            "verified": anchor.verified,
        }
    )

    tmp_path = store_path.with_suffix(store_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
        fh.write("\n")
    tmp_path.replace(store_path)

    return load_anchors(store_path)
