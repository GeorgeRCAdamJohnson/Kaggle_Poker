"""Layer registry: the toggleable feature blocks and the known-good floor config.

Design reference: "Layer registry - ``layers/registry.py``" and Requirement 2
("Toggleable layers with a guaranteed floor").

This module is the single source of truth for WHICH feature blocks exist and
WHICH keys into the existing feature caches each one references. It does NOT
recompute or own any feature data (accountability contract Rule 6, design
principle "Reuse, do not rebuild"): every ``feature_keys`` entry is a stable
IDENTIFIER into an already-built cache artifact under
``poker/outputs/poker_collusion/feature_cache/`` (the ``v5`` / ``v7`` blocks the
LB-verified 0.44519 stack was built from - see ``poker/loopv.py``, ``exp5-7.py``,
and dossier Â§32/Â§33). Downstream (task 5+) resolves these keys to the cached
matrices; nothing here reads them.

The LB-verified floor stack (dossier Â§33, ``cand_exp4c_directional``, real LB
0.44519) is exactly:

    foundation (v5 event-detector pair features)
      + board_equity  (MFg graded board-equity block)
      + directional   (DIRc drift-hardened directional / fold-timing / excess-win)

``known_good_floor_config()`` returns precisely that stack with default knobs, so
that turning every optional layer OFF reproduces the floor (Req 2.1) once the
composer (task 5) is wired.

Grounding for the cache-key mapping (all verified against the on-disk cache and
the experiment scripts that wrote them):

- ``foundation``     -> ``v5/dev_v5.parquet`` / ``v5/eval_v5.parquet`` feature columns
  (every column except ``pair_id`` / ``p_low`` / ``p_high`` / ``label`` /
  ``behavior_family``), the ``feats5`` set used as the foundation in ``loopv.py``.
- ``board_equity``   -> ``v7/_MFg_dev.npy`` / ``v7/_MFg_eval.npy`` (the graded
  board-equity "MFg" block, dossier Â§32).
- ``directional``    -> ``v7/_DIRc_dev.npy`` / ``v7/_DIRc_eval.npy`` (columns in
  ``v7/_DIRc_cols.json``: ``d_feeder_conc`` ... ``d_benef_asym``; dossier Â§33).
- ``iso``            -> ``v7/_ISO_dev.npy`` / ``v7/_ISO_eval.npy`` (raw-isolation
  block; dossier H16 - raw ``isolation`` CONFIRMED at CV AUC 0.732).
- ``whipsaw``        -> ``v7/_LAYER2_dev.npy`` / ``v7/_LAYER2_eval.npy`` (columns in
  ``v7/_LAYER2_cols.json`` include ``l_whip_success`` / ``l_whip_bigpot`` /
  ``l_whip_samest``; the whipsaw signal for ``coordinated_isolation``, requirements
  Introduction - "the strongest signal found").
- ``negative_space`` -> ``v7/_SUPP_dev.npy`` / ``v7/_SUPP_eval.npy`` (the
  suppression / back-off "contest-less-than-expected" block from ``exp7.py`` -
  expected-minus-observed clashes; accountability contract Rule 19 "model the
  hole").

Requirements: 2.1 (all optional OFF reproduces the floor), 2.3 (a candidate can
report which layers are ON - the registry names them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from poker_collusion.tuning_harness.models import (
    CandidateConfig,
    CompositionSpec,
)

__all__ = [
    "Layer",
    "RANK_BLEND",
    "RETRAIN",
    "FOUNDATION_NAME",
    "V5_KEYS",
    "FOUNDATION",
    "OPTIONAL_LAYERS",
    "ALL_LAYERS",
    "FLOOR_LAYERS",
    "DEFAULT_RANK_BLEND_WEIGHT",
    "get_layer",
    "optional_layer_names",
    "known_good_floor_config",
]

#: Composition-mode names (mirrors ``CompositionSpec.mode``; design Req 4).
RANK_BLEND = "RANK_BLEND"
RETRAIN = "RETRAIN"

#: The always-on base layer's name (foundation is not optional).
FOUNDATION_NAME = "foundation"

#: Default RANK_BLEND weight for a monotone optional layer when it is switched ON
#: with no explicit knob. The floor stack does NOT rely on this (its board_equity
#: and directional blocks are conditional/RETRAIN); it is the neutral default a
#: monotone layer carries until the tuner (task 13) sweeps its weight (0 allowed).
DEFAULT_RANK_BLEND_WEIGHT = 0.0


@dataclass(frozen=True)
class Layer:
    """One named, independently-toggleable block of features.

    ``name`` is the stable layer identifier ("foundation" | "board_equity" |
    "directional" | "iso" | "whipsaw" | "negative_space").

    ``kind`` is ``"monotone"`` (a per-pair score composed by RANK_BLEND by
    default) or ``"conditional"`` (a non-linear signal composed by RETRAIN by
    default) - design Req 4.1/4.2. ``default_mode`` derives from ``kind`` so a
    layer's natural composition mode is fixed once, here.

    ``feature_keys`` are stable IDENTIFIERS into the EXISTING feature caches
    (reuse, not recompute - accountability contract Rule 6). They are cache keys,
    not file paths and not recomputed features; the cache resolver (task 5+) maps
    them to the persisted matrices.

    ``optional`` is True for every toggleable layer and False ONLY for
    ``foundation`` (the floor's base, which can never be switched off).
    """

    name: str
    kind: str  # "monotone" -> RANK_BLEND default; "conditional" -> RETRAIN default
    feature_keys: Tuple[str, ...]  # keys into the existing feature caches (reuse, not recompute)
    optional: bool  # foundation is not optional

    def __post_init__(self) -> None:
        if self.kind not in ("monotone", "conditional"):
            raise ValueError(
                f"Layer.kind must be 'monotone' or 'conditional', got {self.kind!r}."
            )
        if not self.feature_keys:
            raise ValueError(f"Layer {self.name!r} must reference at least one feature key.")

    @property
    def default_mode(self) -> str:
        """The natural composition mode: monotone -> RANK_BLEND, conditional -> RETRAIN."""
        return RANK_BLEND if self.kind == "monotone" else RETRAIN


# --------------------------------------------------------------------------- #
# Cache-key sets (stable identifiers into the existing feature caches)
# --------------------------------------------------------------------------- #
#: The v5 foundation feature block. This is a single logical cache key naming the
#: v5 event-detector pair-feature matrices (``feature_cache/v5/{dev,eval}_v5.parquet``);
#: the resolver expands it to that matrix's feature columns (all columns except the
#: id / pool / label / behavior_family bookkeeping columns) - the ``feats5`` set the
#: LB floor was built on. Kept as a named tuple so ``FOUNDATION`` stays declarative.
V5_KEYS: Tuple[str, ...] = ("v5_pair_features",)

#: The graded board-equity block (dossier Â§32, "MFg").
_MFG_KEYS: Tuple[str, ...] = ("MFg",)

#: The drift-hardened directional / fold-timing / excess-win block (dossier Â§33,
#: "DIRc"; the eight columns in ``v7/_DIRc_cols.json``).
_DIRC_KEYS: Tuple[str, ...] = ("DIRc",)

#: The raw-isolation block (dossier H16: raw ``isolation`` CONFIRMED).
_ISO_KEYS: Tuple[str, ...] = ("ISO",)

#: The whipsaw block for ``coordinated_isolation`` (``v7/_LAYER2_*`` - the
#: ``l_whip_*`` columns; requirements Introduction "the strongest signal found").
_WHIPSAW_KEYS: Tuple[str, ...] = ("LAYER2_whipsaw",)

#: The negative-space / suppression block ("contest-less-than-expected",
#: expected-minus-observed clashes from ``exp7.py``; contract Rule 19 "model the
#: hole"). Cached as ``v7/_SUPP_*``.
_NEGATIVE_SPACE_KEYS: Tuple[str, ...] = ("SUPP_negative_space",)


# --------------------------------------------------------------------------- #
# The layers
# --------------------------------------------------------------------------- #
#: The always-on base. It is the floor's base feature set and is NOT optional
#: (Req 2.1). ``kind="conditional"`` because the foundation is the GBDT-fit
#: event-detector feature matrix (a non-linear model over the block), not a single
#: monotone per-pair score.
FOUNDATION = Layer(FOUNDATION_NAME, "conditional", V5_KEYS, optional=False)

#: board_equity (MFg): a graded block fed into the refit foundation model - a
#: conditional (RETRAIN) block, part of the LB-verified floor stack.
BOARD_EQUITY = Layer("board_equity", "conditional", _MFG_KEYS, optional=True)

#: directional (DIRc): drift-hardened directional/fold-timing/excess-win block -
#: conditional (RETRAIN), part of the LB-verified floor stack.
DIRECTIONAL = Layer("directional", "conditional", _DIRC_KEYS, optional=True)

#: iso: raw-isolation per-pair signal - a monotone score (RANK_BLEND default).
ISO = Layer("iso", "monotone", _ISO_KEYS, optional=True)

#: whipsaw: the coordinated_isolation whipsaw block - a conditional (RETRAIN)
#: signal (the l_whip_* interaction columns are non-linear, not a single monotone
#: rank), the strongest signal found for that family.
WHIPSAW = Layer("whipsaw", "conditional", _WHIPSAW_KEYS, optional=True)

#: negative_space: the suppression / expected-minus-observed block - a monotone
#: per-pair "hole" score (RANK_BLEND default): higher = the pair clashes LESS than
#: their co-active opportunity predicts (contract Rule 19).
NEGATIVE_SPACE = Layer("negative_space", "monotone", _NEGATIVE_SPACE_KEYS, optional=True)


#: Every optional (toggleable) layer, keyed by name.
OPTIONAL_LAYERS: Dict[str, Layer] = {
    layer.name: layer
    for layer in (BOARD_EQUITY, DIRECTIONAL, ISO, WHIPSAW, NEGATIVE_SPACE)
}

#: Every layer including the always-on foundation, keyed by name.
ALL_LAYERS: Dict[str, Layer] = {FOUNDATION.name: FOUNDATION, **OPTIONAL_LAYERS}

#: The optional layers that make up the LB-verified 0.44519 floor stack, in order.
FLOOR_LAYERS: Tuple[str, ...] = ("board_equity", "directional")


def get_layer(name: str) -> Layer:
    """Return the registered :class:`Layer` for ``name``.

    Raises ``KeyError`` with the known names if ``name`` is not a registered
    layer, so a typo fails loudly rather than silently dropping a block.
    """
    try:
        return ALL_LAYERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ALL_LAYERS))
        raise KeyError(f"Unknown layer {name!r}; registered layers are: {known}.") from exc


def optional_layer_names() -> Tuple[str, ...]:
    """The optional (toggleable) layer names in a stable, sorted order."""
    return tuple(sorted(OPTIONAL_LAYERS))


def known_good_floor_config() -> CandidateConfig:
    """Return the exact LB-verified floor stack as a :class:`CandidateConfig`.

    The stack (dossier Â§33, real LB 0.44519) is foundation + board_equity (MFg) +
    directional (DIRc), all with DEFAULT knobs. Both optional floor layers are
    conditional blocks composed by RETRAIN (they are refit into the foundation
    model, which is how the 0.44519 stack was built - ``loopv.py`` hstacks
    ``[Xd5, Mgd, DIRd]`` and fits one model), so their :class:`CompositionSpec`
    uses ``mode="RETRAIN"`` and a weight of 0.0 (RETRAIN ignores the RANK_BLEND
    weight). ``foundation`` is always on and is NOT listed in ``layers_on`` (that
    frozenset names only the OPTIONAL layers switched ON, per the model docstring).

    Turning every OTHER optional layer OFF (i.e. this exact config) must reproduce
    the floor once the composer is wired (Req 2.1); default model knobs and no
    shrinkage overrides keep it byte-identical to the shipped 0.44519 artifact.
    """
    composition = {
        name: CompositionSpec(mode=get_layer(name).default_mode, weight=0.0)
        for name in FLOOR_LAYERS
    }
    return CandidateConfig(
        layers_on=frozenset(FLOOR_LAYERS),
        composition=composition,
    )
