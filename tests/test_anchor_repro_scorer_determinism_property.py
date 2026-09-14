"""Property test for CanonicalScorer determinism (task 2.4).

# Feature: poker-anchor-reproduction, Property 1: For any dev-prediction frame scored
# twice under the same ``ScoringRecipe``, the ``CanonicalScorer`` SHALL return an
# identical ``CanonicalScore`` — identical ``pair_ap``, identical ``combined``, and
# identical ``ScoreComponents``.

Property 1 (design.md "Correctness Properties"): both local scores are a deterministic
function of the inputs + recipe (fixed seed), and BOTH are produced by the SAME single
``reference_public_metric.score`` pass. Scoring the identical frame twice therefore must
return byte-identical ``pair_ap``, ``combined``, and the full ``ScoreComponents``
breakdown — never two divergent recomputations. This is the load-bearing determinism
guarantee that lets a single ``CanonicalScore`` anchor a ladder point without the
recorded number drifting between reads (contract Rules 1, 9).

The generator builds a FIXED synthetic confirmed dev-label set (so no real data files
are touched — mirroring ``test_anchor_repro_scorer.py``) and draws random VALID
dev-prediction frames over exactly those confirmed pairs: the ``pair_id`` set matches
the built PU solution (so scoring never hits the coverage-mismatch guard), while
``risk_score``, ``predicted_behavior``, and the five evidence-hand slots vary freely
across the metric's allowed input space. Min 200 iterations.

Validates: Requirements 1.2, 2.5, 2.6
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.config import PipelineConfig
from poker_collusion.metric.reference_public_metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
)

from anchor_repro.models import CanonicalScore, ScoringRecipe
from anchor_repro.scorer import CanonicalScorer


# --------------------------------------------------------------------------- #
# Fixed synthetic dev labels/evidence (no real data files touched)            #
# --------------------------------------------------------------------------- #
# Confirmed pairs whose pair_id set the built PU solution equals, so a prediction
# frame covering EXACTLY these pairs scores cleanly (no coverage mismatch).
_CONFIRMED_PAIRS = ("P1", "P2", "P3", "P4", "P5", "P6")

_BEHAVIORS = sorted(ALLOWED_BEHAVIORS)
# Pools of distinct hand-ids to draw evidence slots from (some overlap the labelled
# evidence below so both hit and miss cases are exercised).
_HAND_POOL = ["HA", "HB", "HC", "HD", "HE", "HX", "HY", "HZ"]


def _dev_labels() -> pd.DataFrame:
    """Confirmed targets (one per target family) + confirmed non-targets."""
    return pd.DataFrame(
        [
            {"pair_id": "P1", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
            {"pair_id": "P2", "label_status": "confirmed_target",
             "behavior_family": "soft_play"},
            {"pair_id": "P3", "label_status": "confirmed_target",
             "behavior_family": "coordinated_isolation"},
            {"pair_id": "P4", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "P5", "label_status": "confirmed_non_target",
             "behavior_family": "none"},
            {"pair_id": "P6", "label_status": "confirmed_target",
             "behavior_family": "directed_transfer"},
        ]
    )


def _dev_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "P1", "evidence_rank": 1, "hand_id": "HA"},
            {"pair_id": "P1", "evidence_rank": 2, "hand_id": "HB"},
            {"pair_id": "P2", "evidence_rank": 1, "hand_id": "HC"},
            {"pair_id": "P3", "evidence_rank": 1, "hand_id": "HD"},
            {"pair_id": "P3", "evidence_rank": 2, "hand_id": "HE"},
            {"pair_id": "P6", "evidence_rank": 1, "hand_id": "HX"},
        ]
    )


def _make_scorer() -> CanonicalScorer:
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(),
        labels=_dev_labels(),
        evidence=_dev_evidence(),
    )


# --------------------------------------------------------------------------- #
# Smart generators: valid dev-prediction rows over the confirmed pair set     #
# --------------------------------------------------------------------------- #
def _evidence_slots(hands: list[str]) -> dict:
    """Five rank-ordered evidence slots; hands must not repeat within a pair, so the
    drawn hands are de-duplicated (order preserved) before padding with NO_EVIDENCE."""
    seen: list[str] = []
    for h in hands:
        if h not in seen:
            seen.append(h)
    padded = seen[:5] + [NO_EVIDENCE] * (5 - min(len(seen), 5))
    return {col: padded[i] for i, col in enumerate(EVIDENCE_COLUMNS)}


@st.composite
def _prediction_row(draw, pair_id: str) -> dict:
    """A single valid prediction row for ``pair_id`` (risk in [0,1], allowed behavior,
    0..5 distinct evidence hands)."""
    risk = draw(st.floats(min_value=0.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False))
    behavior = draw(st.sampled_from(_BEHAVIORS))
    hands = draw(st.lists(st.sampled_from(_HAND_POOL), min_size=0, max_size=5))
    return {
        "pair_id": pair_id,
        "risk_score": risk,
        "predicted_behavior": behavior,
        **_evidence_slots(hands),
    }


@st.composite
def _dev_predictions(draw) -> pd.DataFrame:
    """A full valid dev-prediction frame covering EXACTLY the confirmed pair set."""
    rows = [draw(_prediction_row(pid)) for pid in _CONFIRMED_PAIRS]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Property 1: double-scoring the same frame is byte-identical                 #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 1: canonical scorer determinism
@settings(max_examples=200, deadline=None)
@given(preds=_dev_predictions())
def test_canonical_scorer_is_deterministic(preds: pd.DataFrame) -> None:
    """Scoring the identical dev-prediction frame twice returns an identical
    CanonicalScore — identical pair_ap, combined, AND the full ScoreComponents."""
    scorer = _make_scorer()

    first = scorer.score_dev_predictions(preds)
    second = scorer.score_dev_predictions(preds)

    assert isinstance(first, CanonicalScore)
    assert isinstance(second, CanonicalScore)

    # PRIMARY gate quantity (the anchor's local_holdout_ap) is identical.
    assert first.pair_ap == second.pair_ap
    # SECONDARY diagnostic (full official combined score) is identical.
    assert first.combined == second.combined
    # The full component breakdown is identical (same single metric pass).
    assert first.components == second.components
    # And the whole frozen record compares equal.
    assert first == second


# Feature: poker-anchor-reproduction, Property 1: determinism across scorer instances
@settings(max_examples=200, deadline=None)
@given(preds=_dev_predictions())
def test_canonical_scorer_deterministic_across_instances(preds: pd.DataFrame) -> None:
    """Two independent scorers built from the same recipe + labels produce an identical
    CanonicalScore for the same frame (determinism is a function of inputs + recipe,
    not scorer-instance state)."""
    a = _make_scorer().score_dev_predictions(preds)
    b = _make_scorer().score_dev_predictions(preds)

    assert a.pair_ap == b.pair_ap
    assert a.combined == b.combined
    assert a.components == b.components
    assert a == b
