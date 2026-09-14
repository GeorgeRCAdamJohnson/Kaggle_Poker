"""Smoke tests for the project scaffolding and shared foundations (task 1).

Verifies the package imports cleanly, the core data models exist with the
declared fields, the NOT_APPLICABLE sentinel behaves as a singleton, the
behavior-label enum has the required members, and SchemaError names the file
and missing key.
"""

from __future__ import annotations

import dataclasses

import pytest

import poker_collusion
from poker_collusion import (
    NOT_APPLICABLE,
    BehaviorLabel,
    HandSignal,
    LabelTable,
    Pair,
    PairFeatureSet,
    PairPrediction,
    SchemaError,
)
from poker_collusion.config import PipelineConfig, get_config
from poker_collusion.types import NotApplicable


def test_package_imports_cleanly() -> None:
    assert poker_collusion.__version__


def test_pair_is_frozen_and_hashable() -> None:
    pair = Pair(pair_id="p1", player_a=1, player_b=2, pool_id=7)
    assert hash(pair)  # hashable
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.player_a = 99  # type: ignore[misc]


def test_behavior_label_members() -> None:
    values = {member.value for member in BehaviorLabel}
    assert values == {
        "none",
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
        "other_coordination",
    }
    # inherits from str -> compares as its wire value
    assert BehaviorLabel.SOFT_PLAY == "soft_play"


def test_not_applicable_is_singleton_and_falsy() -> None:
    assert NotApplicable() is NOT_APPLICABLE
    assert not NOT_APPLICABLE
    assert repr(NOT_APPLICABLE) == "NOT_APPLICABLE"


def test_hand_signal_accepts_sentinel() -> None:
    sig = HandSignal(
        hand_id=10,
        pair_id="p1",
        phase="evaluation",
        value_flow=-5.0,
        aggression_asymmetry=NOT_APPLICABLE,
        isolation=0.5,
        mi_conflict=0.1,
        behavior_action_flags={"soft_play": True},
    )
    assert sig.aggression_asymmetry is NOT_APPLICABLE
    assert sig.isolation == 0.5


def test_pair_feature_set_versions() -> None:
    fs = PairFeatureSet(pair_id="p1", schema_version="v1", features={"x": 1.0}, threshold_version="t1")
    assert fs.schema_version == "v1"
    assert fs.threshold_version == "t1"


def test_label_table_pu_structure() -> None:
    lt = LabelTable(trusted_positive={"p1": "soft_play"}, confirmed_negative={"p2"})
    assert lt.trusted_positive["p1"] == "soft_play"
    assert "p2" in lt.confirmed_negative


def test_pair_prediction_defaults() -> None:
    pred = PairPrediction(pair_id="p1", risk_score=0.9, predicted_behavior="soft_play")
    assert pred.evidence == []
    assert pred.reasons == []


def test_schema_error_names_file_and_key() -> None:
    err = SchemaError(file="hands.parquet", missing_key="hand_id")
    assert err.file == "hands.parquet"
    assert err.missing_key == "hand_id"
    assert "hands.parquet" in str(err)
    assert "hand_id" in str(err)


def test_config_defaults() -> None:
    cfg = get_config()
    assert isinstance(cfg, PipelineConfig)
    assert cfg.schema_version == "pair_features_v1"
    assert cfg.threshold_version == "thresholds_v1"
    assert cfg.seeds.master == 20240918
    assert "actions" in cfg.columns
    assert cfg.submission_path.name == "submission.csv"
