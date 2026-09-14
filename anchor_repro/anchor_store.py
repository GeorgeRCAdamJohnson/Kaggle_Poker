"""Append-only ``AnchorStore`` for the poker-anchor-reproduction spec (task 8.1).

design.md "Components and Interfaces" §5 (AnchorStore) and "Data Models"
§"Anchor tuple (extends the existing ``lb_anchors.json`` schema)". Requirements 3.5,
4.2. Governed by the workspace ``reverse-engineering-accountability`` contract
(Rule 1: ``verified=true`` only with a real external judge; Rule 9: state the
uncertainty, honest nulls are results).

Why this class exists (and why it does NOT re-implement the parent store)
-------------------------------------------------------------------------
The parent ``poker-layered-tuning`` harness already owns the canonical, append-only
``.kiro/specs/poker-collusion-detection/lb_anchors.json`` store and a function-style
access layer at :mod:`poker_collusion.tuning_harness.anchor_store`
(``load_anchors`` / ``append_anchor``). This spec must be able to record its OWN
anchors (our ladder points and the reproduced-competitor 0.64469 point) into a store
that is **schema-identical** to the parent's so the parent harness can consume them
verbatim (Req 3.5).

Rather than fork the schema, this module:

* REUSES :class:`poker_collusion.tuning_harness.models.LBAnchor` verbatim as the
  on-disk record — the parent fields ``config_id``, ``real_lb``, ``measured_drift``,
  ``holdout_ap``, ``source``, ``verified`` are preserved byte-for-byte in meaning.
  No new top-level field is introduced.
* Carries this spec's canonical ``recipe_id`` INSIDE the ``source``/provenance string
  (design.md: "this spec adds ``recipe_id`` inside ``source``/provenance so anchors
  remain schema-compatible") — NOT as a new top-level key — so a parent-harness loader
  that only knows the six schema fields still round-trips the anchor unchanged.
* Presents design.md's ``AnchorStore`` object interface (``load`` / ``append`` +
  atomic write + the refusal guards) as the class the rest of ``anchor_repro`` uses.

Invariants enforced (design.md "Error Handling / Anchor store" + Property 6)
----------------------------------------------------------------------------
1. **Strictly append-only.** An ``append`` never edits or deletes an existing anchor;
   the ``_schema`` block and every prior anchor are preserved. Re-appending an
   existing ``config_id`` is refused (that is the only way an edit could sneak in).
2. **Verified externals only.** An anchor is refused if ``verified=True`` is claimed
   without a real, positive external LB value backing it (contract Rule 1). A
   projected / fabricated number is never persisted as verified.
3. **Atomic writes.** The updated store is written to a temp file in the same
   directory and then ``os.replace``-renamed onto the final path, so a crash
   mid-write can never corrupt or truncate the store (a partial file is never
   observed on the real path).

This module is the store implementation only. The append-only property test is
task 8.2 and the schema-compatibility unit test is task 8.3; both drive this class
but are written separately. The class is built to be testable against a temp-file
store path so those tests never touch the real seeded store.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Reuse the parent-harness schema record verbatim — do NOT re-declare the fields.
from poker_collusion.tuning_harness.anchor_store import (
    DEFAULT_ANCHOR_STORE_PATH,
)
from poker_collusion.tuning_harness.models import LBAnchor

__all__ = [
    "DEFAULT_ANCHOR_STORE_PATH",
    "RECIPE_ID_PROVENANCE_KEY",
    "AnchorStoreError",
    "Anchor",
    "AnchorStore",
]

#: The token used to embed the canonical ``recipe_id`` inside an anchor's ``source``
#: provenance string (design.md: recipe_id lives in ``source``, not as a new top-level
#: field). Written as ``recipe_id=<id>`` so a human auditor and a machine parser can
#: both recover it, while the parent harness — which only reads the six schema fields —
#: simply carries the whole ``source`` string through unchanged.
RECIPE_ID_PROVENANCE_KEY: str = "recipe_id"


class AnchorStoreError(Exception):
    """Raised when an ``append`` would violate an append-only / verified invariant.

    Covers the three ways an append could corrupt the externally-auditable trail:
    a duplicate ``config_id`` (the append-only / no-edit guard), a
    non-verified-yet-``verified=true`` claim, and a ``verified=true`` anchor whose
    ``real_lb`` is not a real, positive external value (contract Rule 1, Req 4.2).
    """


def _embed_recipe_id(source: str, recipe_id: Optional[str]) -> str:
    """Return ``source`` with ``recipe_id=<id>`` embedded once, if not already present.

    The ``recipe_id`` is recorded INSIDE the provenance string rather than as a new
    top-level schema field so the anchor stays schema-compatible with the parent
    ``lb_anchors.json`` (design.md data-model note). If the source already names the
    recipe (a caller may pre-format it), it is left untouched so we never duplicate it.
    """

    if not recipe_id:
        return source
    token = f"{RECIPE_ID_PROVENANCE_KEY}={recipe_id}"
    if token in source:
        return source
    if source:
        return f"{source}; {token}"
    return token


def _extract_recipe_id(source: str) -> Optional[str]:
    """Recover the embedded ``recipe_id`` from a ``source`` provenance string, if any.

    Inverse of :func:`_embed_recipe_id`. Returns ``None`` when the source carries no
    ``recipe_id=`` token (e.g. a seed anchor written by the parent harness before this
    spec existed) — absence is reported honestly, never guessed (contract Rule 9).
    """

    marker = f"{RECIPE_ID_PROVENANCE_KEY}="
    idx = source.find(marker)
    if idx < 0:
        return None
    tail = source[idx + len(marker):]
    # The recipe id runs until the next provenance separator (``;``) or end of string.
    end = tail.find(";")
    value = (tail if end < 0 else tail[:end]).strip()
    return value or None


@dataclass(frozen=True)
class Anchor:
    """A poker-anchor-reproduction anchor: the parent schema plus a canonical recipe.

    This is the value the rest of ``anchor_repro`` constructs and hands to
    :meth:`AnchorStore.append`. It carries the six parent-schema fields verbatim
    (so it maps 1:1 onto :class:`LBAnchor`) PLUS the canonical ``recipe_id`` under
    which the ``holdout_ap`` was measured. On persistence the ``recipe_id`` is folded
    INTO the ``source`` provenance string (see :meth:`to_lb_anchor`) — never a new
    top-level field — keeping the on-disk record schema-compatible with the parent
    ``lb_anchors.json`` (Req 3.5).

    Attributes:
        config_id: Stable id of the scored configuration (our ladder id or the
            reproduced-competitor id).
        real_lb: The REAL leaderboard score from the external judge — the artifact
            filename (ours) or the notebook's known 0.64469 (competitor). Never a
            projection (contract Rule 1).
        holdout_ap: The canonical-scorer local holdout **PairAP** measured under
            ``recipe_id`` — the PRIMARY relative quantity that must rank-track
            ``real_lb``. A MEASURED local number, not an externally verified LB value.
        measured_drift: Adversarial dev-vs-eval AUC when available; ``None`` for a
            competitor repro if not computed (stored as JSON ``null``).
        source: Provenance (artifact / notebook sha256, dossier section). The
            ``recipe_id`` is embedded into this string at persistence time.
        recipe_id: The canonical scoring recipe id (e.g. ``canonical_v1``) under which
            ``holdout_ap`` was measured, embedded into ``source`` on write.
        verified: ``True`` only when confirmed by an external judge (contract Rule 1).
    """

    config_id: str
    real_lb: float
    holdout_ap: float
    measured_drift: Optional[float] = None
    source: str = ""
    recipe_id: Optional[str] = None
    verified: bool = False

    def provenance_source(self) -> str:
        """The ``source`` string with the canonical ``recipe_id`` embedded once."""
        return _embed_recipe_id(self.source, self.recipe_id)

    def to_lb_anchor(self) -> LBAnchor:
        """Map this anchor onto the parent-harness :class:`LBAnchor` schema record.

        ``recipe_id`` is folded into ``source`` (never a new top-level field) so the
        result is byte-for-byte a parent ``lb_anchors.json`` record the
        ``poker-layered-tuning`` harness can consume unchanged (Req 3.5). ``measured_drift``
        is coerced to a float for the frozen ``LBAnchor``; the JSON-``null`` case is
        handled by :class:`AnchorStore` at the serialization boundary.
        """
        drift = float(self.measured_drift) if self.measured_drift is not None else float("nan")
        return LBAnchor(
            config_id=self.config_id,
            real_lb=float(self.real_lb),
            measured_drift=drift,
            holdout_ap=float(self.holdout_ap),
            source=self.provenance_source(),
            verified=bool(self.verified),
        )

    def recorded_recipe_id(self) -> Optional[str]:
        """The recipe id that will be recoverable from the persisted ``source``."""
        return _extract_recipe_id(self.provenance_source())


class AnchorStore:
    """Append-only access layer for a ``lb_anchors.json``-schema anchor store.

    Presents design.md's ``AnchorStore`` object interface (``load`` / ``append``) over
    a JSON store that is schema-identical to the parent
    ``.kiro/specs/poker-collusion-detection/lb_anchors.json`` (Req 3.5). Enforces the
    append-only + verified-externals-only invariants (design.md "Error Handling",
    Property 6) and writes atomically (temp file + ``os.replace``).

    The store path defaults to the parent seeded store but is injectable so tests
    (tasks 8.2 / 8.3) operate on a temp copy and never mutate the real trail.
    """

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        """Bind the store to ``path`` (defaults to the parent seeded ``lb_anchors.json``).

        The file is NOT created here — the parent store is seeded and its absence is a
        real error surfaced by :meth:`load`, not something to paper over. For a fresh
        test store, :meth:`init_store` writes an empty, schema-shaped document.
        """
        self._path: Path = Path(path) if path is not None else DEFAULT_ANCHOR_STORE_PATH

    @property
    def path(self) -> Path:
        """The bound store path."""
        return self._path

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_store(self) -> Dict[str, Any]:
        """Load the raw store document (``_schema`` block + ``anchors`` list) verbatim.

        Returns the parsed JSON exactly as stored so the ``_schema`` provenance block
        is preserved across an append. Raises ``FileNotFoundError`` if the store is
        missing — the parent store is seeded, so absence is a real error.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"anchor store not found at {self._path!s}")
        with self._path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load(self) -> List[LBAnchor]:
        """Load the store and return its anchors as parent-schema :class:`LBAnchor` records.

        Preserves store order. Unknown extra keys in a stored anchor are ignored so the
        schema can grow without breaking the loader (mirrors the parent access layer).
        A missing ``measured_drift`` (JSON ``null``) is loaded as ``NaN`` so it never
        silently reads as a real ``0.0`` drift.
        """
        store = self.load_store()
        anchors: List[LBAnchor] = []
        for raw in store.get("anchors", []):
            drift_raw = raw.get("measured_drift", None)
            drift = float(drift_raw) if drift_raw is not None else float("nan")
            anchors.append(
                LBAnchor(
                    config_id=raw["config_id"],
                    real_lb=float(raw["real_lb"]),
                    measured_drift=drift,
                    holdout_ap=float(raw["holdout_ap"]),
                    source=str(raw.get("source", "")),
                    verified=bool(raw.get("verified", False)),
                )
            )
        return anchors

    def recipe_ids(self) -> List[Optional[str]]:
        """The ``recipe_id`` recovered from each stored anchor's ``source`` (order-preserved).

        ``None`` for any anchor whose source carries no ``recipe_id=`` token (e.g. a
        parent seed anchor). Lets a caller assert every anchor this spec produced shares
        one canonical recipe (Property 2) without a new top-level schema field.
        """
        store = self.load_store()
        return [_extract_recipe_id(str(raw.get("source", ""))) for raw in store.get("anchors", [])]

    # ------------------------------------------------------------------ #
    # Appending (guarded, atomic)
    # ------------------------------------------------------------------ #
    def append(self, anchor: Anchor) -> List[LBAnchor]:
        """Append one anchor, enforcing the append-only + verified-externals invariants.

        Refuses (raising :class:`AnchorStoreError`) when:

        * an anchor with the same ``config_id`` already exists — the store is strictly
          append-only and holds one record per config; re-appending would be an edit
          (design.md "Error Handling"; Property 6 — no existing anchor is ever mutated
          or deleted);
        * ``anchor.verified`` is ``True`` but ``real_lb`` is not a real, positive
          external value (``None`` / non-finite / ``<= 0``) — only a real external
          leaderboard judge may back a verified anchor (contract Rule 1, Req 4.2). A
          projected or fabricated number is never persisted as verified.

        On success the anchor (with ``recipe_id`` folded into ``source``) is appended to
        the END of the ``anchors`` list; the ``_schema`` block and all prior anchors are
        preserved. The write is ATOMIC — a temp file in the same directory is
        ``os.replace``-renamed onto the store — so a crash mid-write cannot corrupt or
        truncate the trail. Returns the full anchor list after the append.
        """

        self._guard_verified(anchor)

        store = self.load_store()
        existing = store.setdefault("anchors", [])

        for raw in existing:
            if raw.get("config_id") == anchor.config_id:
                raise AnchorStoreError(
                    f"Refusing to append duplicate config_id {anchor.config_id!r}: the "
                    "anchor store is strictly append-only and holds one record per "
                    "config — editing or replacing an existing anchor is not permitted "
                    "(Property 6)."
                )

        existing.append(self._to_record(anchor))
        self._atomic_write(store)
        return self.load()

    def _guard_verified(self, anchor: Anchor) -> None:
        """Refuse to persist ``verified=true`` without a real, positive external LB.

        Enforces contract Rule 1 / Req 4.2 at the write boundary: a ``verified`` anchor
        MUST be backed by a real external leaderboard score. ``real_lb`` must be a
        finite, strictly-positive number; ``None``, ``NaN``, ``inf`` or a non-positive
        value means there is no external judge behind the claim, so the append is
        refused rather than silently writing an unverifiable "verified" record.
        """
        if not anchor.verified:
            return
        real_lb = anchor.real_lb
        if (
            real_lb is None
            or isinstance(real_lb, bool)
            or not isinstance(real_lb, (int, float))
            or not math.isfinite(float(real_lb))
            or float(real_lb) <= 0.0
        ):
            raise AnchorStoreError(
                f"Refusing to persist anchor {anchor.config_id!r} as verified=true "
                f"without a real external leaderboard value (got real_lb={real_lb!r}): "
                "only a real, external judge may back a verified anchor (accountability "
                "contract Rule 1)."
            )

    @staticmethod
    def _to_record(anchor: Anchor) -> Dict[str, Any]:
        """Serialize an :class:`Anchor` to the parent ``lb_anchors.json`` record dict.

        ``recipe_id`` is embedded in ``source`` (never a top-level field); ``measured_drift``
        serializes to JSON ``null`` when absent so a missing drift is honest rather than
        a fabricated ``0.0`` (contract Rule 9). The key order mirrors the parent schema.
        """
        return {
            "config_id": anchor.config_id,
            "real_lb": float(anchor.real_lb),
            "measured_drift": (
                float(anchor.measured_drift) if anchor.measured_drift is not None else None
            ),
            "holdout_ap": float(anchor.holdout_ap),
            "source": anchor.provenance_source(),
            "verified": bool(anchor.verified),
        }

    def _atomic_write(self, store: Dict[str, Any]) -> None:
        """Write ``store`` atomically: temp file in the store dir, then ``os.replace``.

        Matches the atomic-write convention used across the poker package
        (``submission/writer.py``, ``features/feature_cache.py``): the JSON is fully
        written and flushed/fsynced to a temp file in the SAME directory, then
        ``os.replace``-renamed onto the final path (an atomic rename on one filesystem).
        A crash before the rename leaves the original store untouched; a partial file is
        never observed on the real path.
        """
        target = self._path
        target.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            prefix=".lb_anchors_", suffix=".json.tmp", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(store, fh, indent=2)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, target)  # atomic rename on the same filesystem
        except BaseException:
            # Never leave the temp artifact behind on failure.
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ #
    # Test / bootstrap helper
    # ------------------------------------------------------------------ #
    def init_store(self, schema: Optional[Dict[str, Any]] = None, *, overwrite: bool = False) -> None:
        """Create an empty, schema-shaped store at the bound path (for fresh test stores).

        Writes ``{"_schema": <schema or {}>, "anchors": []}`` atomically. Refuses to
        overwrite an existing store unless ``overwrite=True`` so a stray call can never
        wipe the real seeded trail (append-only discipline, Property 6). Intended for
        tests (tasks 8.2 / 8.3) that need an isolated temp store.
        """
        if self._path.exists() and not overwrite:
            raise AnchorStoreError(
                f"Refusing to initialize an existing store at {self._path!s}: pass "
                "overwrite=True only for a throwaway test store (never the real trail)."
            )
        self._atomic_write({"_schema": schema or {}, "anchors": []})
