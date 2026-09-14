"""Unit tests for poker_collusion.models.behavior_classifier (task 17.1, Req 7.1-7.6, 12.3).

Hand-checkable example tests on tiny synthetic :class:`PairFeatureSet`\\ s, a small
:class:`LabelTable` of Trusted_Positive disclosed families, and an injected
``{pair_id -> risk_score}`` map (the SHARED risk from the PU ranker), verifying:

* below-threshold risk -> ``none`` (Req 7.6);
* a clearly ``directed_transfer``-leaning pair with high risk -> ``directed_transfer``;
* an ambiguous high-risk pair -> ``other_coordination`` (Req 7.3 routing);
* :meth:`per_family_scores` returns the shared risk for the predicted family and 0 for others,
  and its keys are exactly the three disclosed families (``other_coordination`` / ``none``
  excluded; Req 7.4, 7.5, 12.3);
* coverage — every pair labelled, every label in the allowed set (Req 7.1);
* determinism — same seed/input -> identical labels;
* HALT (raise) on empty/missing required input;
* save/load round-trip reproduces identical labels.

The Property-8 Hypothesis test is task 17.2 and is intentionally NOT written here.
Fast and hermetic: tiny synthetic inputs, injected artifact only, artifacts to tmp_path.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import pytest

from poker_collusion.exceptions import SchemaError
from poker_collusion.models.behavior_classifier import (
    DEFAULT_COORDINATION_THRESHOLD,
    DEFAULT_OTHER_COORDINATION_MARGIN,
    DISCLOSED_FAMILY_TOKENS,
    BehaviorClassifier,
    train_behavior_classifier,
)
from poker_collusion.types import BehaviorLabel, LabelTable, PairFeatureSet

ALLOWED = {
    BehaviorLabel.NONE,
    BehaviorLabel.DIRECTED_TRANSFER,
    BehaviorLabel.SOFT_PLAY,
    BehaviorLabel.COORDINATED_ISOLATION,
    BehaviorLabel.OTHER_COORDINATION,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _artifact() -> Dict[str, object]:
    return {
        "thresholds": {"pool_prior_positive_prevalence": 0.02, "min_shared_hands": 2},
        "threshold_version": "test_thresholds",
    }


def _fs(pair_id: str, *, transfer: float, softplay: float, isolation: float) -> PairFeatureSet:
    """Tiny synthetic PairFeatureSet with three family-discriminating features.

    Each family "owns" one feature axis; a pair's dominant axis names its family. The three axes
    are otherwise on the same scale so the head must learn to separate them from the training data.
    """
    feats: Dict[str, float] = {
        "n_shared_hands": 20.0,
        "sig_transfer": float(transfer),
        "sig_softplay": float(softplay),
        "sig_isolation": float(isolation),
    }
    return PairFeatureSet(
        pair_id=pair_id,
        schema_version="test_schema",
        features=feats,
        threshold_version="test_thresholds",
    )


def _train_set() -> List[PairFeatureSet]:
    """Trusted positives with clearly separated per-family signatures (several per family)."""
    rows: List[PairFeatureSet] = []
    for i in range(4):
        rows.append(_fs(f"DT{i}", transfer=8.0 + i, softplay=0.2, isolation=0.2))
    for i in range(4):
        rows.append(_fs(f"SP{i}", transfer=0.2, softplay=8.0 + i, isolation=0.2))
    for i in range(4):
        rows.append(_fs(f"CI{i}", transfer=0.2, softplay=0.2, isolation=8.0 + i))
    return rows


def _labels() -> LabelTable:
    tp: Dict[str, str] = {}
    for i in range(4):
        tp[f"DT{i}"] = "directed_transfer"
        tp[f"SP{i}"] = "soft_play"
        tp[f"CI{i}"] = "coordinated_isolation"
    return LabelTable(trusted_positive=tp)


def _fit(seed: Optional[int] = None) -> BehaviorClassifier:
    return train_behavior_classifier(_train_set(), _labels(), eda_artifact=_artifact(), seed=seed)


# --------------------------------------------------------------------------- #
# none-thresholding (Req 7.6)
# --------------------------------------------------------------------------- #
def test_below_threshold_risk_is_none():
    clf = _fit()
    pair = _fs("E", transfer=9.0, softplay=0.1, isolation=0.1)  # clearly directed_transfer-leaning
    # ...but risk below the coordination threshold -> none regardless of features.
    labels = clf.predict_labels(
        [pair], {"E": 0.10}, coordination_threshold=0.5, eda_artifact=_artifact()
    )
    assert labels["E"] == BehaviorLabel.NONE


# --------------------------------------------------------------------------- #
# Confident disclosed family (Req 7.1, 7.2)
# --------------------------------------------------------------------------- #
def test_high_risk_directed_transfer_leaning_pair():
    clf = _fit()
    pair = _fs("E", transfer=10.0, softplay=0.1, isolation=0.1)
    labels = clf.predict_labels(
        [pair],
        {"E": 0.95},
        coordination_threshold=0.5,
        other_coordination_margin=DEFAULT_OTHER_COORDINATION_MARGIN,
        eda_artifact=_artifact(),
    )
    assert labels["E"] == BehaviorLabel.DIRECTED_TRANSFER


# --------------------------------------------------------------------------- #
# Ambiguous high-risk pair -> other_coordination (Req 7.3 routing)
# --------------------------------------------------------------------------- #
def test_ambiguous_high_risk_pair_routes_to_other_coordination():
    clf = _fit()
    # All three signatures equal => head is maximally ambiguous => small margin => catch-all.
    pair = _fs("E", transfer=5.0, softplay=5.0, isolation=5.0)
    labels = clf.predict_labels(
        [pair],
        {"E": 0.95},
        coordination_threshold=0.5,
        other_coordination_margin=0.5,  # generous margin so the ambiguous pair is diverted
        eda_artifact=_artifact(),
    )
    assert labels["E"] == BehaviorLabel.OTHER_COORDINATION


# --------------------------------------------------------------------------- #
# Per-family scoring for Behavior MAP (Req 7.4, 7.5, 12.3)
# --------------------------------------------------------------------------- #
def test_per_family_scores_shared_risk_for_predicted_else_zero():
    clf = _fit()
    dt = _fs("DT", transfer=10.0, softplay=0.1, isolation=0.1)
    low = _fs("LOW", transfer=10.0, softplay=0.1, isolation=0.1)  # high features but low risk
    risk = {"DT": 0.9, "LOW": 0.1}
    maps = clf.per_family_scores(
        [dt, low], risk, coordination_threshold=0.5, eda_artifact=_artifact()
    )

    # keys are exactly the three disclosed families (other_coordination / none excluded)
    assert set(maps.keys()) == set(DISCLOSED_FAMILY_TOKENS)
    assert "other_coordination" not in maps
    assert "none" not in maps

    # DT predicted directed_transfer: shared risk in that family, hard 0 in others.
    assert maps["directed_transfer"]["DT"] == pytest.approx(0.9)
    assert maps["soft_play"]["DT"] == 0.0
    assert maps["coordinated_isolation"]["DT"] == 0.0

    # LOW routed to none (below threshold): 0 in every disclosed family (scoring loss).
    for fam in DISCLOSED_FAMILY_TOKENS:
        assert maps[fam]["LOW"] == 0.0


# --------------------------------------------------------------------------- #
# Coverage + allowed labels (Req 7.1)
# --------------------------------------------------------------------------- #
def test_coverage_every_pair_labelled_in_allowed_set():
    clf = _fit()
    eval_sets = [
        _fs("A", transfer=10.0, softplay=0.1, isolation=0.1),
        _fs("B", transfer=0.1, softplay=10.0, isolation=0.1),
        _fs("C", transfer=0.1, softplay=0.1, isolation=10.0),
        _fs("D", transfer=5.0, softplay=5.0, isolation=5.0),
        _fs("E", transfer=9.0, softplay=0.1, isolation=0.1),
    ]
    risk = {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.9, "E": 0.05}
    labels = clf.predict(eval_sets, risk, other_coordination_margin=0.5, eda_artifact=_artifact())
    assert [pid for pid, _ in labels] == ["A", "B", "C", "D", "E"]  # coverage + order
    for _, lab in labels:
        assert lab in ALLOWED


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_seed_identical_labels():
    eval_sets = [
        _fs("A", transfer=10.0, softplay=0.1, isolation=0.1),
        _fs("B", transfer=0.1, softplay=10.0, isolation=0.1),
    ]
    risk = {"A": 0.9, "B": 0.9}
    a = _fit(seed=202).predict_labels(eval_sets, risk, eda_artifact=_artifact())
    b = _fit(seed=202).predict_labels(eval_sets, risk, eda_artifact=_artifact())
    assert a == b


def test_labels_independent_of_input_row_order():
    clf = _fit()
    eval_sets = [
        _fs("A", transfer=10.0, softplay=0.1, isolation=0.1),
        _fs("B", transfer=0.1, softplay=10.0, isolation=0.1),
        _fs("C", transfer=0.1, softplay=0.1, isolation=10.0),
    ]
    risk = {"A": 0.9, "B": 0.9, "C": 0.9}
    ref = clf.predict_labels(eval_sets, risk, eda_artifact=_artifact())

    shuffled = list(eval_sets)
    random.Random(7).shuffle(shuffled)
    got = clf.predict_labels(shuffled, risk, eda_artifact=_artifact())
    assert got == ref


# --------------------------------------------------------------------------- #
# HALT on missing/empty required input
# --------------------------------------------------------------------------- #
def test_fit_halts_on_empty_training_set():
    with pytest.raises(SchemaError):
        train_behavior_classifier([], _labels(), eda_artifact=_artifact())


def test_fit_halts_when_no_disclosed_family_positives():
    # Only other_coordination / none tokens => no disclosed family to learn.
    labels = LabelTable(trusted_positive={"X": "other_coordination", "Y": "none"})
    with pytest.raises(SchemaError):
        train_behavior_classifier(_train_set(), labels, eda_artifact=_artifact())


def test_predict_halts_on_empty_eval_set():
    clf = _fit()
    with pytest.raises(SchemaError):
        clf.predict([], {}, eda_artifact=_artifact())


def test_predict_halts_on_missing_risk_score():
    clf = _fit()
    pair = _fs("E", transfer=10.0, softplay=0.1, isolation=0.1)
    with pytest.raises(SchemaError):
        clf.predict([pair], {}, eda_artifact=_artifact())  # no risk for "E"


def test_predict_halts_on_feature_schema_mismatch():
    clf = _fit()
    bad = _fs("X", transfer=10.0, softplay=0.1, isolation=0.1)
    bad.features["a_brand_new_feature"] = 1.0
    with pytest.raises(SchemaError):
        clf.predict([bad], {"X": 0.9}, eda_artifact=_artifact())


def test_predict_before_fit_halts():
    clf = BehaviorClassifier()
    pair = _fs("E", transfer=10.0, softplay=0.1, isolation=0.1)
    with pytest.raises(SchemaError):
        clf.predict([pair], {"E": 0.9})


def test_save_before_fit_halts(tmp_path):
    clf = BehaviorClassifier()
    with pytest.raises(SchemaError):
        clf.save(tmp_path / "model.joblib")


def test_load_missing_file_halts(tmp_path):
    with pytest.raises(SchemaError):
        BehaviorClassifier.load(tmp_path / "does_not_exist.joblib")


# --------------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------------- #
def test_save_load_round_trip_reproduces_labels(tmp_path):
    clf = _fit()
    eval_sets = [
        _fs("A", transfer=10.0, softplay=0.1, isolation=0.1),
        _fs("B", transfer=0.1, softplay=10.0, isolation=0.1),
        _fs("C", transfer=5.0, softplay=5.0, isolation=5.0),
        _fs("D", transfer=9.0, softplay=0.1, isolation=0.1),
    ]
    risk = {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.05}
    before = clf.predict_labels(eval_sets, risk, other_coordination_margin=0.5, eda_artifact=_artifact())

    path = tmp_path / "behavior_model.joblib"
    clf.save(path)
    assert path.exists()

    loaded = BehaviorClassifier.load(path)
    after = loaded.predict_labels(eval_sets, risk, other_coordination_margin=0.5, eda_artifact=_artifact())
    assert after == before
    assert loaded.columns == clf.columns
    assert set(loaded.classes) == set(clf.classes)
