"""Unit tests for the layer registry (task 4.1).

Confirms the LB-verified floor config is exactly foundation + board_equity +
directional, that ``foundation`` is the only non-optional layer, and that every
optional layer (board_equity, directional, iso, whipsaw, negative_space) is
registered and references at least one existing feature-cache key.

Requirements: 2.1 (all optional OFF reproduces the floor stack), 2.3 (a candidate
can report which layers are ON - the registry names them).
"""

from __future__ import annotations

from poker_collusion.tuning_harness.layers.registry import (
    ALL_LAYERS,
    FLOOR_LAYERS,
    FOUNDATION,
    FOUNDATION_NAME,
    OPTIONAL_LAYERS,
    RETRAIN,
    Layer,
    get_layer,
    known_good_floor_config,
    optional_layer_names,
)
from poker_collusion.tuning_harness.models import CandidateConfig, CompositionSpec

#: The five optional layers named by the design/requirements glossary.
EXPECTED_OPTIONAL = {"board_equity", "directional", "iso", "whipsaw", "negative_space"}


def test_floor_config_is_foundation_plus_board_equity_and_directional():
    """known_good_floor_config() is exactly foundation + board_equity + directional."""
    cfg = known_good_floor_config()
    assert isinstance(cfg, CandidateConfig)

    # foundation is always ON and NOT listed in layers_on (it names optional layers only);
    # the two optional layers ON are precisely the floor stack.
    assert cfg.layers_on == frozenset({"board_equity", "directional"})
    assert FOUNDATION_NAME not in cfg.layers_on

    # Every ON layer has a composition spec; the floor blocks compose by RETRAIN
    # (they are refit into the foundation model - dossier Â§33).
    assert set(cfg.composition) == {"board_equity", "directional"}
    for name in ("board_equity", "directional"):
        spec = cfg.composition[name]
        assert isinstance(spec, CompositionSpec)
        assert spec.mode == RETRAIN
        assert spec.weight == 0.0


def test_foundation_is_the_only_non_optional_layer():
    """FOUNDATION is non-optional; every other registered layer is optional."""
    assert FOUNDATION.name == FOUNDATION_NAME
    assert FOUNDATION.optional is False
    assert FOUNDATION.feature_keys  # references the v5 foundation cache key

    non_optional = [l for l in ALL_LAYERS.values() if not l.optional]
    assert non_optional == [FOUNDATION]


def test_all_optional_layers_are_registered():
    """board_equity, directional, iso, whipsaw, negative_space are all registered."""
    assert set(OPTIONAL_LAYERS) == EXPECTED_OPTIONAL
    assert set(optional_layer_names()) == EXPECTED_OPTIONAL
    # ALL_LAYERS is foundation + the five optional layers.
    assert set(ALL_LAYERS) == EXPECTED_OPTIONAL | {FOUNDATION_NAME}


def test_every_layer_references_existing_cache_keys():
    """Each layer names at least one feature-cache key and a valid kind/mode."""
    for name, layer in ALL_LAYERS.items():
        assert isinstance(layer, Layer)
        assert layer.name == name
        assert layer.kind in ("monotone", "conditional")
        assert layer.default_mode in ("RANK_BLEND", RETRAIN)
        assert len(layer.feature_keys) >= 1
        assert all(isinstance(k, str) and k for k in layer.feature_keys)


def test_floor_layers_are_registered_optional_layers():
    """The floor's optional layers are registered and optional."""
    assert FLOOR_LAYERS == ("board_equity", "directional")
    for name in FLOOR_LAYERS:
        layer = get_layer(name)
        assert layer.optional is True


def test_get_layer_rejects_unknown_name():
    """A typo'd layer name fails loudly rather than silently dropping a block."""
    import pytest

    with pytest.raises(KeyError, match="Unknown layer"):
        get_layer("not_a_layer")


def test_layer_is_frozen_and_hashable():
    """Layer is a frozen dataclass (hashable, immutable) so configs stay hashable."""
    import pytest

    assert hash(FOUNDATION) == hash(FOUNDATION)
    with pytest.raises(Exception):
        FOUNDATION.name = "mutated"  # type: ignore[misc]
