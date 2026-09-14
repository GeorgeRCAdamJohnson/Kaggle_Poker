"""Unit tests for the floor-guaranteed composer (task 5.1, design ``composition/composer.py``).

Example / edge-case unit tests for ``rank_blend`` (weight-0 floor guarantee),
``compose`` (all optional layers OFF reproduces the floor + best-effort layer
reporting + reporting-failure edge), ``classify_layer`` (CORRUPTING / NULL /
CONTRIBUTING), and ``retrain`` (combined-feature assembly via an injected fit
callback). The Hypothesis property tests for Properties 6/7/8 are separate tasks
(5.2-5.4); these are the concrete examples and edge cases.

Boundary note (compose all-OFF): the REAL floor ranking requires the feature
caches and the reused model, which are not available in a pure unit test. We
therefore test the guarantee STRUCTURALLY: ``compose`` is injected a
``base_ranking_fn`` standing in for "the floor ranking", and the assertion is
that composing ``known_good_floor_config()`` (which turns every OTHER optional
layer OFF and composes its floor layers via RETRAIN, i.e. no RANK_BLEND) yields a
ranking ORDER-IDENTICAL to that base ranking, and that a config with no optional
layers ON does likewise. The end-to-end "== the shipped 0.44519 artifact
ranking" check is the integration test in task 17.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1.
"""

from __future__ import annotations

import pytest

from poker_collusion.tuning_harness.composition.composer import (
    CONTRIBUTING,
    CORRUPTING,
    NULL,
    classify_layer,
    compose,
    rank_blend,
    retrain,
)
from poker_collusion.tuning_harness.layers.registry import known_good_floor_config
from poker_collusion.tuning_harness.models import (
    Candidate,
    CandidateConfig,
    CompositionSpec,
    FeatureFrame,
    ModelConfig,
    RankVector,
)


# --------------------------------------------------------------------------- #
# rank_blend  -- Req 2.2 (floor guarantee), 4.1
# --------------------------------------------------------------------------- #
def test_rank_blend_weight_zero_is_order_identical_to_base() -> None:
    base = RankVector(pair_ids=("a", "b", "c", "d"))
    layer = RankVector(pair_ids=("d", "c", "b", "a"))  # exact reverse

    blended = rank_blend(base, layer, 0.0)

    # Floor guarantee: weight 0 returns the base ordering EXACTLY.
    assert blended.pair_ids == base.pair_ids


def test_rank_blend_weight_zero_returns_base_even_for_disjoint_layer() -> None:
    base = RankVector(pair_ids=("a", "b", "c"))
    # Layer covers pairs the base does not; at weight 0 they are irrelevant.
    layer = RankVector(pair_ids=("x", "y", "z"))

    assert rank_blend(base, layer, 0.0).pair_ids == base.pair_ids


def test_rank_blend_weight_one_takes_layer_order_over_base_pairs() -> None:
    base = RankVector(pair_ids=("a", "b", "c"))
    layer = RankVector(pair_ids=("c", "b", "a"))

    blended = rank_blend(base, layer, 1.0)

    assert blended.pair_ids == ("c", "b", "a")


def test_rank_blend_half_weight_averages_ranks() -> None:
    # base ranks: a=0,b=1,c=2 ; layer ranks: a=2,b=1,c=0
    # blended at w=0.5: a=1.0, b=1.0, c=1.0 -> all tie; tie broken by base rank
    # then pair_id -> a, b, c (base order preserved on ties).
    base = RankVector(pair_ids=("a", "b", "c"))
    layer = RankVector(pair_ids=("c", "b", "a"))

    blended = rank_blend(base, layer, 0.5)

    assert blended.pair_ids == ("a", "b", "c")


def test_rank_blend_shifts_order_toward_layer_at_high_weight() -> None:
    # base: a=0,b=1,c=2,d=3 ; layer: d=0,c=1,b=2,a=3
    # w=0.9 -> blended: a=0.1*0+0.9*3=2.7, b=0.1*1+0.9*2=1.9,
    #                   c=0.1*2+0.9*1=1.1, d=0.1*3+0.9*0=0.3
    # ascending blended rank -> d, c, b, a
    base = RankVector(pair_ids=("a", "b", "c", "d"))
    layer = RankVector(pair_ids=("d", "c", "b", "a"))

    assert rank_blend(base, layer, 0.9).pair_ids == ("d", "c", "b", "a")


def test_rank_blend_rejects_out_of_range_weight() -> None:
    base = RankVector(pair_ids=("a", "b"))
    layer = RankVector(pair_ids=("b", "a"))
    with pytest.raises(ValueError):
        rank_blend(base, layer, 1.5)
    with pytest.raises(ValueError):
        rank_blend(base, layer, -0.1)


def test_rank_blend_rejects_duplicate_pair_ids() -> None:
    base = RankVector(pair_ids=("a", "a", "b"))
    layer = RankVector(pair_ids=("a", "b", "c"))
    with pytest.raises(ValueError):
        rank_blend(base, layer, 0.5)


# --------------------------------------------------------------------------- #
# compose  -- Req 2.1 (floor), 2.3 / 2.4 (best-effort reporting)
# --------------------------------------------------------------------------- #
_FLOOR_RANKING = RankVector(pair_ids=("p1", "p2", "p3", "p4", "p5"))


def _floor_base_fn(_model_cfg: ModelConfig) -> RankVector:
    """Stand-in for 'the floor ranking' produced by the reused model + caches."""
    return _FLOOR_RANKING


def test_compose_all_optional_off_reproduces_base_ranking_exactly() -> None:
    cfg = CandidateConfig(layers_on=frozenset())  # no optional layers ON

    candidate = compose(cfg, base_ranking_fn=_floor_base_fn)

    # Req 2.1 / Property 6 (structural): all-OFF -> the base (floor) ranking,
    # order-identical.
    assert candidate.ranking.pair_ids == _FLOOR_RANKING.pair_ids
    assert candidate.layers_reported == ()


def test_compose_known_good_floor_config_reproduces_base_ranking_exactly() -> None:
    # The floor config turns every OTHER optional layer OFF and composes its two
    # floor layers via RETRAIN (no RANK_BLEND), so compose must not blend and the
    # ranking is order-identical to the base ranking.
    cfg = known_good_floor_config()

    candidate = compose(cfg, base_ranking_fn=_floor_base_fn)

    assert candidate.ranking.pair_ids == _FLOOR_RANKING.pair_ids
    # RETRAIN floor layers are reported ON best-effort.
    assert candidate.layers_reported == ("board_equity", "directional")


def test_compose_weight_zero_rank_blend_layer_is_a_noop() -> None:
    # A RANK_BLEND optional layer switched ON but at weight 0 must not change the
    # ranking (floor guarantee) and must not even require a layer_ranking_fn.
    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.0)},
    )

    candidate = compose(cfg, base_ranking_fn=_floor_base_fn)

    assert candidate.ranking.pair_ids == _FLOOR_RANKING.pair_ids
    assert candidate.layers_reported == ("iso",)


def test_compose_positive_weight_rank_blend_uses_layer_fn() -> None:
    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=1.0)},
    )
    layer_ranking = RankVector(pair_ids=("p5", "p4", "p3", "p2", "p1"))

    def layer_fn(name: str, _cfg: CandidateConfig) -> RankVector:
        assert name == "iso"
        return layer_ranking

    candidate = compose(cfg, base_ranking_fn=_floor_base_fn, layer_ranking_fn=layer_fn)

    # weight 1.0 -> the layer order (over the base pair set).
    assert candidate.ranking.pair_ids == layer_ranking.pair_ids


def test_compose_positive_weight_without_layer_fn_raises() -> None:
    cfg = CandidateConfig(
        layers_on=frozenset({"iso"}),
        composition={"iso": CompositionSpec(mode="RANK_BLEND", weight=0.5)},
    )
    with pytest.raises(ValueError):
        compose(cfg, base_ranking_fn=_floor_base_fn)  # no layer_ranking_fn


# --------------------------------------------------------------------------- #
# compose best-effort layer reporting  -- Task 5.5 (Req 2.3, 2.4)
#
# Two dedicated unit tests pinning the best-effort reporting contract:
#   * success -> layers_reported EQUALS the ON layer set (Req 2.3)
#   * injected failure -> valid Candidate, layers_reported is None, ranking and
#     config intact, assembly NOT blocked (Req 2.4)
# --------------------------------------------------------------------------- #
def test_compose_reporting_success_reports_exactly_the_layers_on() -> None:  # Req 2.3
    layers_on = frozenset({"whipsaw", "iso"})
    cfg = CandidateConfig(
        layers_on=layers_on,
        composition={
            "whipsaw": CompositionSpec(mode="RETRAIN", weight=0.0),
            "iso": CompositionSpec(mode="RANK_BLEND", weight=0.0),
        },
    )

    candidate = compose(cfg, base_ranking_fn=_floor_base_fn)

    # Req 2.3: when reporting succeeds, layers_reported is EXACTLY the ON set
    # (as a stable sorted tuple), no more, no less.
    assert candidate.layers_reported is not None
    assert set(candidate.layers_reported) == set(layers_on)
    assert candidate.layers_reported == tuple(sorted(layers_on))
    # The ranking is unaffected by reporting (both weight-0 layers are no-ops).
    assert candidate.ranking.pair_ids == _FLOOR_RANKING.pair_ids


def test_compose_reporting_failure_yields_none_and_does_not_block(  # Req 2.4
) -> None:
    cfg = CandidateConfig(layers_on=frozenset({"iso"}))

    def boom(_cfg: CandidateConfig):
        raise RuntimeError("reporting is broken")

    # Req 2.4: a reporting failure must NOT block assembly. compose still returns
    # a valid Candidate with its config and ranking intact; only layers_reported
    # degrades to None.
    candidate = compose(
        cfg, base_ranking_fn=_floor_base_fn, report_layers_fn=boom
    )

    assert isinstance(candidate, Candidate)
    assert candidate.config is cfg
    assert candidate.ranking.pair_ids == _FLOOR_RANKING.pair_ids
    assert candidate.layers_reported is None


# --------------------------------------------------------------------------- #
# classify_layer  -- Req 2.5 (corrupting-layer detection)
# --------------------------------------------------------------------------- #
_FLOOR_AP = 0.3839


def test_classify_layer_corrupting_when_below_floor_at_every_weight() -> None:
    # Strictly below the floor at every positive weight -> CORRUPTING.
    holdout = {0.25: 0.37, 0.5: 0.36, 0.75: 0.35, 1.0: 0.30}
    assert classify_layer(holdout, _FLOOR_AP) == CORRUPTING


def test_classify_layer_contributing_when_above_floor_somewhere() -> None:
    holdout = {0.25: 0.385, 0.5: 0.39, 0.75: 0.386, 1.0: 0.38}
    assert classify_layer(holdout, _FLOOR_AP) == CONTRIBUTING


def test_classify_layer_null_when_matches_floor() -> None:
    # Never strictly above and never strictly below at all weights -> NULL.
    holdout = {0.25: _FLOOR_AP, 0.5: _FLOOR_AP, 1.0: _FLOOR_AP}
    assert classify_layer(holdout, _FLOOR_AP) == NULL


def test_classify_layer_null_when_mixed_but_never_above() -> None:
    # Below at one weight, equal at another, never strictly above -> NULL
    # (not corrupting: not below at EVERY weight).
    holdout = {0.5: 0.37, 1.0: _FLOOR_AP}
    assert classify_layer(holdout, _FLOOR_AP) == NULL


def test_classify_layer_ignores_weight_zero() -> None:
    # Weight 0 is the floor by construction; only positive weights inform the
    # verdict. Here every POSITIVE weight is below floor -> CORRUPTING despite
    # the w=0 entry equalling the floor.
    holdout = {0.0: _FLOOR_AP, 0.5: 0.37, 1.0: 0.36}
    assert classify_layer(holdout, _FLOOR_AP) == CORRUPTING


def test_classify_layer_no_positive_weights_is_null() -> None:
    assert classify_layer({0.0: 0.30}, _FLOOR_AP) == NULL
    assert classify_layer([], _FLOOR_AP) == NULL


def test_classify_layer_accepts_sequence_of_pairs() -> None:
    holdout = [(0.5, 0.36), (1.0, 0.30)]
    assert classify_layer(holdout, _FLOOR_AP) == CORRUPTING


# --------------------------------------------------------------------------- #
# retrain  -- Req 4.2 (produces ranking; trust is gated in task 9.1)
# --------------------------------------------------------------------------- #
def test_retrain_combines_features_and_calls_fit_fn() -> None:
    base = FeatureFrame(
        feature_keys=("f0", "f1"),
        rows={"p1": (1.0, 2.0), "p2": (3.0, 4.0)},
        is_eval={"p1": False, "p2": True},
    )
    layer = FeatureFrame(
        feature_keys=("g0",),
        rows={"p1": (9.0,), "p2": (8.0,)},
    )
    seen: dict[str, FeatureFrame] = {}

    def fit_fn(combined: FeatureFrame, _cfg: ModelConfig) -> RankVector:
        seen["combined"] = combined
        # Rank by first layer feature descending: p1(9) before p2(8).
        return RankVector(pair_ids=("p1", "p2"))

    ranking = retrain(base, layer, ModelConfig(), fit_fn=fit_fn)

    combined = seen["combined"]
    # Column-concatenated keys and rows.
    assert combined.feature_keys == ("f0", "f1", "g0")
    assert combined.rows["p1"] == (1.0, 2.0, 9.0)
    assert combined.rows["p2"] == (3.0, 4.0, 8.0)
    # is_eval carried from the base frame.
    assert combined.is_eval["p2"] is True
    assert ranking.pair_ids == ("p1", "p2")


def test_retrain_drops_pairs_missing_from_either_frame() -> None:
    base = FeatureFrame(feature_keys=("f0",), rows={"p1": (1.0,), "p2": (2.0,)})
    layer = FeatureFrame(feature_keys=("g0",), rows={"p1": (9.0,)})  # no p2

    captured: dict[str, FeatureFrame] = {}

    def fit_fn(combined: FeatureFrame, _cfg: ModelConfig) -> RankVector:
        captured["combined"] = combined
        return RankVector(pair_ids=tuple(combined.rows))

    retrain(base, layer, ModelConfig(), fit_fn=fit_fn)

    # Only the shared pair survives (a refit needs a full vector per pair).
    assert set(captured["combined"].rows) == {"p1"}


def test_retrain_without_fit_fn_raises() -> None:
    base = FeatureFrame(feature_keys=("f0",), rows={"p1": (1.0,)})
    layer = FeatureFrame(feature_keys=("g0",), rows={"p1": (9.0,)})
    with pytest.raises(ValueError):
        retrain(base, layer, ModelConfig())
