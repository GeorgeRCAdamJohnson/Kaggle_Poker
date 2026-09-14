# Feature: poker-collusion-detection, Property 8: Below-threshold pairs are routed to none
"""Property test for behavior-classifier none-thresholding (task 17.2, Req 7.6, Property 8).

Property 8 (Req 7.6): for a fitted :class:`BehaviorClassifier`, given a shared
``{pair_id -> risk_score}`` mapping and a ``coordination_threshold``, **every** pair whose
``risk_score < coordination_threshold`` is labelled EXACTLY :data:`BehaviorLabel.NONE`,
irrespective of its features; and every pair whose ``risk_score >= coordination_threshold`` is
labelled a NON-``none`` coordination label (one of the three disclosed families or
``other_coordination``) — i.e. the none-thresholding rule never forces an at/above-threshold pair
to ``none``.

The classifier is fit **once** on a tiny fixed synthetic training set (module scope). Only the
evaluation inputs (features, per-pair ``risk_score`` in ``[0, 1]``, and the
``coordination_threshold`` in ``(0, 1)``) vary across generated cases; the estimator is never
re-fit per case (keeps the property loop fast).

Backend: prefers ``hypothesis`` when importable (guarded via ``importlib.util.find_spec``); when it
is not present, a seeded randomized loop (~200 seeds) exercises the same property. The active
backend is reported by :func:`test_report_active_backend`.
"""

from __future__ import annotations

import importlib.util
import random
from typing import Dict, List, Tuple

from poker_collusion.models.behavior_classifier import (
    BehaviorClassifier,
    train_behavior_classifier,
)
from poker_collusion.types import BehaviorLabel, LabelTable, PairFeatureSet

# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
_HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
ACTIVE_BACKEND = "hypothesis" if _HYPOTHESIS_AVAILABLE else "seeded-loop"

# The allowed label set (Req 7.1).
ALLOWED = {
    BehaviorLabel.NONE,
    BehaviorLabel.DIRECTED_TRANSFER,
    BehaviorLabel.SOFT_PLAY,
    BehaviorLabel.COORDINATED_ISOLATION,
    BehaviorLabel.OTHER_COORDINATION,
}
# Non-none coordination labels (a pair at/above threshold must land in one of these).
NON_NONE = ALLOWED - {BehaviorLabel.NONE}


# --------------------------------------------------------------------------- #
# Tiny fixed synthetic builders (reuse the style of test_behavior_classifier.py)
# --------------------------------------------------------------------------- #
def _artifact() -> Dict[str, object]:
    return {
        "thresholds": {"pool_prior_positive_prevalence": 0.02, "min_shared_hands": 2},
        "threshold_version": "test_thresholds",
    }


def _fs(pair_id: str, *, transfer: float, softplay: float, isolation: float) -> PairFeatureSet:
    """Tiny synthetic PairFeatureSet with three family-discriminating feature axes."""
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


# Fit the classifier ONCE for the whole module (only eval inputs vary across cases).
_CLASSIFIER = train_behavior_classifier(_train_set(), _labels(), eda_artifact=_artifact(), seed=7)


# --------------------------------------------------------------------------- #
# Random-eval-case generation (features + per-pair risk + threshold)
# --------------------------------------------------------------------------- #
def _random_eval_case(
    rng: random.Random,
) -> Tuple[List[PairFeatureSet], Dict[str, float], float]:
    """Return (eval_feature_sets, risk_scores, coordination_threshold) for one random case.

    Threshold in ``(0, 1)``; per-pair risk in ``[0, 1]`` chosen to land on BOTH sides of the
    threshold (and occasionally exactly on it) so each case exercises the routing rule broadly.
    Features are random across the three axes so we prove the rule holds *regardless of features*.
    """
    threshold = rng.uniform(0.05, 0.95)
    n = rng.randint(1, 6)
    eval_sets: List[PairFeatureSet] = []
    risk_scores: Dict[str, float] = {}
    for k in range(n):
        pid = f"P{k}"
        eval_sets.append(
            _fs(
                pid,
                transfer=rng.uniform(0.0, 10.0),
                softplay=rng.uniform(0.0, 10.0),
                isolation=rng.uniform(0.0, 10.0),
            )
        )
        # Draw risk to straddle the threshold: sometimes below, sometimes at/above, sometimes edge.
        pick = rng.random()
        if pick < 0.4:
            r = rng.uniform(0.0, threshold)  # likely below (may equal 0)
        elif pick < 0.8:
            r = rng.uniform(threshold, 1.0)  # at/above
        elif pick < 0.9:
            r = threshold  # exactly on the boundary (>= => NOT none)
        else:
            r = rng.choice([0.0, 1.0])  # explicit extremes
        risk_scores[pid] = min(1.0, max(0.0, float(r)))
    return eval_sets, risk_scores, threshold


def _check_case(
    clf: BehaviorClassifier,
    eval_sets: List[PairFeatureSet],
    risk_scores: Dict[str, float],
    threshold: float,
) -> None:
    """Assert Property 8 (+ coverage / allowed-set / order-independence) for one case."""
    labels = clf.predict_labels(
        eval_sets,
        risk_scores,
        coordination_threshold=threshold,
        eda_artifact=_artifact(),
    )

    # Coverage: exactly one label per input pair (Req 7.1).
    assert set(labels.keys()) == {fs.pair_id for fs in eval_sets}
    assert len(labels) == len(eval_sets)

    for fs in eval_sets:
        pid = fs.pair_id
        label = labels[pid]
        risk = risk_scores[pid]

        # Every label is in the allowed set (Req 7.1).
        assert label in ALLOWED, f"{label!r} not allowed for {pid}"

        if risk < threshold:
            # Below threshold -> EXACTLY none, regardless of features (Req 7.6, Property 8).
            assert label == BehaviorLabel.NONE, (
                f"pair {pid} risk={risk} < threshold={threshold} must be none, got {label!r}"
            )
        else:
            # At/above threshold -> the none-thresholding rule never forces none; it gets a
            # coordination label (a disclosed family or other_coordination).
            assert label in NON_NONE, (
                f"pair {pid} risk={risk} >= threshold={threshold} must be a coordination label, "
                f"got {label!r}"
            )

    # Order-independence: a shuffled eval list yields the same per-pair labels.
    shuffled = list(eval_sets)
    random.Random(len(eval_sets) * 31 + 5).shuffle(shuffled)
    reshuffled = clf.predict_labels(
        shuffled,
        risk_scores,
        coordination_threshold=threshold,
        eda_artifact=_artifact(),
    )
    assert reshuffled == labels


# --------------------------------------------------------------------------- #
# Backend report
# --------------------------------------------------------------------------- #
def test_report_active_backend():
    """Surface which property-testing backend is active (hypothesis vs seeded loop)."""
    assert ACTIVE_BACKEND in {"hypothesis", "seeded-loop"}
    print(f"[Property 8] active backend: {ACTIVE_BACKEND}")


# --------------------------------------------------------------------------- #
# Property 8 — randomized
# --------------------------------------------------------------------------- #
if _HYPOTHESIS_AVAILABLE:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _pair_strategy = st.fixed_dictionaries(
        {
            "transfer": st.floats(min_value=0.0, max_value=10.0),
            "softplay": st.floats(min_value=0.0, max_value=10.0),
            "isolation": st.floats(min_value=0.0, max_value=10.0),
            "risk": st.floats(min_value=0.0, max_value=1.0),
        }
    )

    @settings(max_examples=200, deadline=None)
    @given(
        threshold=st.floats(min_value=0.01, max_value=0.99),
        pairs=st.lists(_pair_strategy, min_size=1, max_size=6),
    )
    def test_property8_below_threshold_routed_to_none(threshold, pairs):
        eval_sets: List[PairFeatureSet] = []
        risk_scores: Dict[str, float] = {}
        for k, p in enumerate(pairs):
            pid = f"P{k}"
            eval_sets.append(
                _fs(pid, transfer=p["transfer"], softplay=p["softplay"], isolation=p["isolation"])
            )
            risk_scores[pid] = float(p["risk"])
        _check_case(_CLASSIFIER, eval_sets, risk_scores, float(threshold))

else:

    def test_property8_below_threshold_routed_to_none():
        """Seeded randomized-loop fallback (~200 seeds) exercising Property 8."""
        for seed in range(200):
            rng = random.Random(seed)
            eval_sets, risk_scores, threshold = _random_eval_case(rng)
            _check_case(_CLASSIFIER, eval_sets, risk_scores, threshold)


# --------------------------------------------------------------------------- #
# Explicit boundary cases (Req 7.6)
# --------------------------------------------------------------------------- #
def test_boundary_risk_exactly_at_threshold_is_not_none():
    """risk == threshold uses ``>=`` semantics, so it is NOT routed to none."""
    threshold = 0.5
    pair = _fs("B", transfer=9.0, softplay=0.1, isolation=0.1)
    label = _CLASSIFIER.predict_labels(
        [pair], {"B": threshold}, coordination_threshold=threshold, eda_artifact=_artifact()
    )["B"]
    assert label != BehaviorLabel.NONE
    assert label in NON_NONE


def test_boundary_risk_just_below_threshold_is_none():
    """risk just below the threshold is routed to none (regardless of strong features)."""
    threshold = 0.5
    pair = _fs("B", transfer=9.0, softplay=0.1, isolation=0.1)
    label = _CLASSIFIER.predict_labels(
        [pair],
        {"B": threshold - 1e-9},
        coordination_threshold=threshold,
        eda_artifact=_artifact(),
    )["B"]
    assert label == BehaviorLabel.NONE


def test_boundary_risk_zero_is_none_for_any_positive_threshold():
    """risk == 0 is below any positive threshold -> none."""
    pair = _fs("B", transfer=9.0, softplay=0.1, isolation=0.1)
    for threshold in (0.01, 0.25, 0.5, 0.99):
        label = _CLASSIFIER.predict_labels(
            [pair], {"B": 0.0}, coordination_threshold=threshold, eda_artifact=_artifact()
        )["B"]
        assert label == BehaviorLabel.NONE


def test_boundary_risk_one_is_non_none_for_threshold_below_one():
    """risk == 1 is at/above any threshold < 1 -> a coordination label (never forced none)."""
    pair = _fs("B", transfer=9.0, softplay=0.1, isolation=0.1)
    for threshold in (0.01, 0.5, 0.99):
        label = _CLASSIFIER.predict_labels(
            [pair], {"B": 1.0}, coordination_threshold=threshold, eda_artifact=_artifact()
        )["B"]
        assert label != BehaviorLabel.NONE
        assert label in NON_NONE
