"""Correctness unit tests for poker_collusion.features.pair_features (task 10.3, Requirement 5).

Hand-computable example tests on tiny synthetic :class:`HandSignal` streams verifying:

* the assembled :class:`PairFeatureSet` contains the expected named feature GROUPS — core
  value-flow / aggression-asymmetry / isolation / mi_conflict + episodic + the five confounder
  features (Req 5.1, 5.2);
* dev and eval use the IDENTICAL feature definition — feeding the SAME hands under either phase
  label yields identical features; the only real difference is which hands are in the slice
  (Req 5.3);
* the confounder directedness-contrast is ~0 when the pair flow equals the players' field
  baseline and LARGE when the pair flow exceeds it — the hand-checkable H14 discriminator
  (Req 5.2);
* the avoidance-gap and joint-isolation-rate confounder features compute on a checkable example;
* ``schema_version`` / ``threshold_version`` are attached, taking the version from an injected EDA
  artifact when present (Req 5.4);
* determinism — reordering the input signals yields byte-identical output (Req 5.5).

The exhaustive Hypothesis determinism property test (Property 5) is task 10.4 and is intentionally
NOT written here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from poker_collusion.config import PipelineConfig
from poker_collusion.features.aggregation import aggregate_pair
from poker_collusion.features.pair_features import (
    CONFOUNDER_FEATURE_NAMES,
    FIELD_BASELINE_KEYS,
    build_pair_feature_set,
    confounder_separating_features,
    pair_feature_set_to_dict,
    pair_feature_set_to_frame,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal, PairFeatureSet, Sentinel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sig(
    hand_id: int,
    *,
    value_flow: float = 0.0,
    aggression_asymmetry: Union[float, Sentinel] = 0.0,
    isolation: Union[float, Sentinel] = 0.0,
    mi_conflict: float = 0.0,
    flagged: bool = False,
    phase: str = "evaluation",
    pair_id: str = "P",
) -> HandSignal:
    flags: Dict[str, bool] = {
        "directed_transfer": bool(flagged),
        "soft_play": False,
        "coordinated_isolation": False,
    }
    return HandSignal(
        hand_id=hand_id,
        pair_id=pair_id,
        phase=phase,
        value_flow=value_flow,
        aggression_asymmetry=aggression_asymmetry,
        isolation=isolation,
        mi_conflict=mi_conflict,
        behavior_action_flags=flags,
    )


def _artifact(threshold_version: str = "thresholds_test", prior: float = 0.02):
    """A minimal injected EDA-artifact-like object (dict form) for hermetic tests."""
    return {
        "threshold_version": threshold_version,
        "thresholds": {"pool_prior_positive_prevalence": prior},
    }


# --------------------------------------------------------------------------- #
# Feature-group presence (Req 5.1, 5.2)
# --------------------------------------------------------------------------- #
def test_feature_set_contains_expected_named_groups():
    signals = [
        _sig(1, value_flow=3.0, aggression_asymmetry=0.2, isolation=1.5, mi_conflict=0.4, flagged=True),
        _sig(2, value_flow=1.0, aggression_asymmetry=0.5, isolation=0.5),
        _sig(3, value_flow=-2.0, aggression_asymmetry=NOT_APPLICABLE, isolation=NOT_APPLICABLE),
    ]
    fs = build_pair_feature_set("P", signals, player_a="1", player_b="2")
    feats = fs.features

    # Core value-flow group (directed max/abs/mean + tails).
    for name in ("value_flow_max_pos", "value_flow_max_abs", "value_flow_mean", "value_flow_absmean"):
        assert name in feats
    for stat in ("mean", "max", "p90", "p95"):
        assert f"value_flow_abs_{stat}" in feats
        assert f"aggression_asymmetry_{stat}" in feats
        assert f"isolation_{stat}" in feats
        assert f"mi_conflict_{stat}" in feats

    # Episodic group.
    for name in ("flagged_count", "flagged_rate", "max_burst", "burst_count",
                 "episodic_score", "flagged_rate_shrunk", "concentration_ratio"):
        assert name in feats

    # Confounder-separating group (exactly the documented five).
    for name in CONFOUNDER_FEATURE_NAMES:
        assert name in feats
    assert len(CONFOUNDER_FEATURE_NAMES) == 5


def test_features_dict_is_sorted_and_all_float():
    signals = [_sig(1, value_flow=2.0, flagged=True), _sig(2, value_flow=-1.0)]
    fs = build_pair_feature_set("P", signals, player_a="1", player_b="2")
    keys = list(fs.features.keys())
    assert keys == sorted(keys)
    assert all(isinstance(v, float) for v in fs.features.values())


# --------------------------------------------------------------------------- #
# Dev / eval identical definition (Req 5.3)
# --------------------------------------------------------------------------- #
def test_dev_and_eval_use_identical_definitions_same_inputs():
    # Same hands, but relabel the phase; feeding the identical slice under either phase label must
    # produce identical features (the definition does not depend on the phase label itself).
    dev_signals = [
        _sig(1, value_flow=3.0, aggression_asymmetry=0.3, isolation=1.0, flagged=True, phase="development"),
        _sig(2, value_flow=1.0, aggression_asymmetry=0.6, isolation=0.5, phase="development"),
    ]
    eval_signals = [
        _sig(1, value_flow=3.0, aggression_asymmetry=0.3, isolation=1.0, flagged=True, phase="evaluation"),
        _sig(2, value_flow=1.0, aggression_asymmetry=0.6, isolation=0.5, phase="evaluation"),
    ]
    base = {"1": {"mean_value_flow_vs_field": 0.5, "mean_aggression_vs_field": 0.4, "coseat_hands": 4},
            "2": {"mean_value_flow_vs_field": 0.5, "mean_aggression_vs_field": 0.4, "coseat_hands": 4}}

    dev = build_pair_feature_set("P", dev_signals, "1", "2", phase="development", field_baseline=base)
    ev = build_pair_feature_set("P", eval_signals, "1", "2", phase="evaluation", field_baseline=base)

    # Every feature EXCEPT the provenance phase flag must be identical.
    dev_feats = {k: v for k, v in dev.features.items() if k != "phase_is_development"}
    ev_feats = {k: v for k, v in ev.features.items() if k != "phase_is_development"}
    assert dev_feats == ev_feats
    assert dev.features["phase_is_development"] == 1.0
    assert ev.features["phase_is_development"] == 0.0


def test_phase_only_selects_which_hands_feed_it():
    # Mixed-phase stream: the dev slice must equal building on just the dev hands.
    signals = [
        _sig(1, value_flow=4.0, flagged=True, phase="development"),
        _sig(2, value_flow=2.0, phase="development"),
        _sig(3, value_flow=9.0, flagged=True, phase="evaluation"),
    ]
    only_dev = [s for s in signals if s.phase == "development"]

    mixed_dev = build_pair_feature_set("P", signals, "1", "2", phase="development")
    just_dev = build_pair_feature_set("P", only_dev, "1", "2", phase="development")
    assert mixed_dev.features == just_dev.features
    # And the eval hand's larger flow does not leak into the dev slice.
    assert mixed_dev.features["value_flow_max_abs"] == 4.0


# --------------------------------------------------------------------------- #
# Confounder directedness-contrast (H14) hand-checkable behaviour (Req 5.2)
# --------------------------------------------------------------------------- #
def test_directedness_contrast_zero_when_pair_flow_equals_field_baseline():
    # pair signed-mean flow = 2.0; field baseline mean = 2.0 for both players -> contrast ~0.
    signals = [_sig(1, value_flow=2.0, flagged=True), _sig(2, value_flow=2.0, flagged=True)]
    base = {"1": {"mean_value_flow_vs_field": 2.0},
            "2": {"mean_value_flow_vs_field": 2.0}}
    fs = build_pair_feature_set("P", signals, "1", "2", field_baseline=base)
    assert abs(fs.features["conf_directedness_contrast"]) < 1e-9


def test_directedness_contrast_large_when_pair_flow_exceeds_field_baseline():
    # pair signed-mean flow = 10.0; field baseline mean = 0.5 -> contrast ~= 9.5 (large positive).
    signals = [_sig(1, value_flow=10.0, flagged=True), _sig(2, value_flow=10.0, flagged=True)]
    base = {"1": {"mean_value_flow_vs_field": 0.5},
            "2": {"mean_value_flow_vs_field": 0.5}}
    fs = build_pair_feature_set("P", signals, "1", "2", field_baseline=base)
    assert abs(fs.features["conf_directedness_contrast"] - 9.5) < 1e-9


def test_avoidance_gap_matches_hand_computation():
    # pair-directed aggression mean = mean(0.2, 0.4) = 0.3; field aggression = 0.9 -> gap = 0.6.
    signals = [
        _sig(1, aggression_asymmetry=0.2, flagged=True),
        _sig(2, aggression_asymmetry=0.4),
    ]
    base = {"1": {"mean_aggression_vs_field": 0.9},
            "2": {"mean_aggression_vs_field": 0.9}}
    fs = build_pair_feature_set("P", signals, "1", "2", field_baseline=base)
    assert abs(fs.features["conf_avoidance_gap"] - 0.6) < 1e-9


def test_joint_isolation_rate_conditioned_on_coseating():
    # High co-seating exposure (repeated-opponent-selection confounder) drives the rate DOWN vs a
    # low-exposure pair with the same joint-isolation mass.
    signals = [
        _sig(1, isolation=2.0, flagged=True),
        _sig(2, isolation=2.0, flagged=True),
    ]
    low_expo = {"1": {"coseat_hands": 2}, "2": {"coseat_hands": 2}}
    high_expo = {"1": {"coseat_hands": 100}, "2": {"coseat_hands": 100}}
    fs_low = build_pair_feature_set("P", signals, "1", "2", field_baseline=low_expo)
    fs_high = build_pair_feature_set("P", signals, "1", "2", field_baseline=high_expo)
    assert fs_low.features["conf_joint_isolation_rate"] > fs_high.features["conf_joint_isolation_rate"]
    assert fs_high.features["conf_joint_isolation_rate"] >= 0.0


def test_field_baseline_change_is_documented_stub_zero():
    signals = [_sig(1, value_flow=5.0, flagged=True)]
    fs = build_pair_feature_set("P", signals, "1", "2")
    assert fs.features["conf_field_baseline_change_stub"] == 0.0


def test_confounder_contrasts_reduce_to_raw_stat_without_baseline():
    # With no injected baseline, directedness_contrast == pair signed-mean flow (baseline 0).
    signals = [_sig(1, value_flow=3.0, flagged=True), _sig(2, value_flow=1.0)]
    fs = build_pair_feature_set("P", signals, "1", "2", field_baseline=None)
    assert abs(fs.features["conf_directedness_contrast"] - fs.features["value_flow_mean"]) < 1e-9


# --------------------------------------------------------------------------- #
# Schema / threshold versioning (Req 5.4)
# --------------------------------------------------------------------------- #
def test_schema_and_threshold_version_from_config_default():
    cfg = PipelineConfig()
    fs = build_pair_feature_set("P", [_sig(1, value_flow=1.0)], "1", "2", config=cfg)
    assert fs.schema_version == cfg.schema_version
    assert fs.threshold_version == cfg.threshold_version


def test_threshold_version_taken_from_injected_artifact():
    art = _artifact(threshold_version="thresholds_from_artifact", prior=0.03)
    fs = build_pair_feature_set("P", [_sig(1, value_flow=1.0, flagged=True)], "1", "2", eda_artifact=art)
    assert fs.threshold_version == "thresholds_from_artifact"
    # The artifact's pool prior also drives the shrinkage feature.
    assert fs.features["pool_prior_used"] == 0.03


# --------------------------------------------------------------------------- #
# Determinism (Req 5.5)
# --------------------------------------------------------------------------- #
def test_reordered_input_yields_identical_output():
    signals = [
        _sig(3, value_flow=-2.0, aggression_asymmetry=0.7, isolation=1.0),
        _sig(1, value_flow=3.0, aggression_asymmetry=0.2, isolation=2.0, flagged=True),
        _sig(2, value_flow=1.0, aggression_asymmetry=NOT_APPLICABLE, isolation=0.5, flagged=True),
    ]
    base = {"1": {"mean_value_flow_vs_field": 0.5, "mean_aggression_vs_field": 0.4, "coseat_hands": 6},
            "2": {"mean_value_flow_vs_field": 0.5, "mean_aggression_vs_field": 0.4, "coseat_hands": 6}}
    a = build_pair_feature_set("P", signals, "1", "2", field_baseline=base)
    b = build_pair_feature_set("P", list(reversed(signals)), "1", "2", field_baseline=base)
    assert a == b
    assert pair_feature_set_to_dict(a) == pair_feature_set_to_dict(b)


def test_repeated_build_is_identical():
    signals = [_sig(1, value_flow=2.0, flagged=True), _sig(2, value_flow=-1.0)]
    a = build_pair_feature_set("P", signals, "1", "2")
    b = build_pair_feature_set("P", signals, "1", "2")
    assert a.features == b.features


# --------------------------------------------------------------------------- #
# Serialization (Req 5.4 stable/ordered persist path)
# --------------------------------------------------------------------------- #
def test_to_dict_is_stable_and_ordered():
    fs = build_pair_feature_set("P", [_sig(1, value_flow=2.0, flagged=True)], "1", "2")
    d = pair_feature_set_to_dict(fs)
    assert d["pair_id"] == "P"
    assert d["schema_version"] == fs.schema_version
    assert list(d["features"].keys()) == sorted(d["features"].keys())


def test_to_frame_one_row_per_pair_ordered_by_pair_id():
    s1 = [_sig(1, value_flow=2.0, flagged=True)]
    s2 = [_sig(1, value_flow=1.0)]
    fs_b = build_pair_feature_set("B", s1, "1", "2")
    fs_a = build_pair_feature_set("A", s2, "1", "2")
    frame = pair_feature_set_to_frame([fs_b, fs_a])
    assert list(frame["pair_id"]) == ["A", "B"]
    assert list(frame.columns[:3]) == ["pair_id", "schema_version", "threshold_version"]
    assert "conf_directedness_contrast" in frame.columns


# --------------------------------------------------------------------------- #
# Direct confounder helper on a raw episodic result (contract check)
# --------------------------------------------------------------------------- #
def test_confounder_helper_emits_exactly_the_named_features():
    signals = [_sig(1, value_flow=2.0, flagged=True), _sig(2, value_flow=1.0)]
    epi = aggregate_pair("P", signals)
    conf = confounder_separating_features(signals, "1", "2", epi, field_baseline=None)
    assert set(conf.keys()) == set(CONFOUNDER_FEATURE_NAMES)
    assert all(isinstance(v, float) for v in conf.values())
