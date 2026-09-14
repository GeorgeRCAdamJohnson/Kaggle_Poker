# Feature: poker-collusion-detection, Property 14: Submission field domains
"""Property 14 (task 13.3): submission field domains (positive and negative directions).

**Validates: Requirements 9.3, 9.4**

Positive direction — for random VALID predictions, every row of the built frame has:

* ``risk_score`` a float in ``[0, 1]`` (Req 9.3);
* ``predicted_behavior`` in
  :data:`~poker_collusion.metric.reference_public_metric.ALLOWED_BEHAVIORS` (Req 9.4);
* all 5 evidence cells non-empty; and
* no repeated hand id within a row.

Negative direction — for random INVALID predictions
(:func:`build_submission_frame` / :func:`validate_submission` must reject them):

* ``risk_score`` outside ``[0, 1]`` including ``NaN`` / ``inf`` (Req 9.3);
* ``predicted_behavior`` outside ``ALLOWED_BEHAVIORS`` (Req 9.4);
* a duplicate evidence hand within a row.

Note: :func:`build_submission_frame` normalizes blank/short evidence to ``NO_EVIDENCE``, so the
empty-cell rejection is exercised by calling :func:`validate_submission` directly on a
hand-crafted frame containing an empty evidence cell.

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable it transparently
falls back to a seeded randomized loop (200 seeds). The active backend is recorded in
``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import math
import random
from typing import List, Sequence

import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE
from poker_collusion.metric.reference_public_metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
)
from poker_collusion.submission.writer import (
    SUBMISSION_COLUMNS,
    SubmissionValidationError,
    build_submission_frame,
    validate_submission,
)
from poker_collusion.types import PairPrediction

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

_BEHAVIORS = tuple(sorted(ALLOWED_BEHAVIORS))
_BAD_BEHAVIORS = ("", "None", "colluding", "transfer", "soft-play", "UNKNOWN", "directed transfer")


# --------------------------------------------------------------------------- #
# Builders (mirror test_submission_writer.py conventions)                     #
# --------------------------------------------------------------------------- #
def _sample_submission(pair_ids: Sequence[str]) -> pd.DataFrame:
    """A sample_submission.csv template in the exact schema/column order."""
    rows = []
    for pid in pair_ids:
        row = {
            "pair_id": pid,
            "risk_score": 0.0,
            "predicted_behavior": "none",
        }
        for i in range(1, 6):
            row[f"evidence_hand_{i}"] = NO_EVIDENCE
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))


def _pred(pair_id: str, risk: float, behavior: str, evidence: Sequence[str]) -> PairPrediction:
    return PairPrediction(
        pair_id=pair_id,
        risk_score=risk,
        predicted_behavior=behavior,
        evidence=list(evidence),
    )


def _valid_evidence(rng: random.Random) -> List[str]:
    """0..5 DISTINCT hand ids (writer pads to 5 with NO_EVIDENCE)."""
    k = rng.randint(0, 5)
    return [str(h) for h in rng.sample(range(1, 10_000), k)]


# --------------------------------------------------------------------------- #
# Property assertions                                                         #
# --------------------------------------------------------------------------- #
def _assert_valid_frame_domains(preds: Sequence[PairPrediction]) -> None:
    pair_ids = [p.pair_id for p in preds]
    sample = _sample_submission(pair_ids)
    frame = build_submission_frame(preds, sample)

    for row in frame.itertuples(index=False):
        row_map = dict(zip(SUBMISSION_COLUMNS, row))

        # risk_score is a float in [0, 1] (Req 9.3).
        risk = float(row_map["risk_score"])
        assert math.isfinite(risk) and 0.0 <= risk <= 1.0, f"risk {risk!r} out of [0,1]"

        # predicted_behavior in ALLOWED_BEHAVIORS (Req 9.4).
        assert row_map["predicted_behavior"] in ALLOWED_BEHAVIORS

        # All 5 evidence cells non-empty and no repeated hand id within the row.
        seen: set[str] = set()
        for col in EVIDENCE_COLUMNS:
            cell = "" if row_map[col] is None else str(row_map[col]).strip()
            assert cell, f"empty evidence cell {col!r}"
            if cell == NO_EVIDENCE:
                continue
            assert cell not in seen, f"repeated evidence hand {cell!r}"
            seen.add(cell)


def _assert_invalid_rejected(preds: Sequence[PairPrediction]) -> None:
    pair_ids = [p.pair_id for p in preds]
    sample = _sample_submission(pair_ids)
    with pytest.raises(SubmissionValidationError):
        build_submission_frame(preds, sample)


# --------------------------------------------------------------------------- #
# Backend report + explicit corner cases                                      #
# --------------------------------------------------------------------------- #
def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 14] generator in use: {GENERATOR_IN_USE}")


def test_empty_evidence_cell_rejected_via_validate():
    """A hand-crafted frame with an empty evidence cell is rejected by validate_submission."""
    sample = _sample_submission(["1_2"])
    frame = _sample_submission(["1_2"])
    frame.loc[0, "evidence_hand_3"] = ""
    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(frame, sample)
    assert "evidence_hand_3" in str(exc.value)


def test_valid_example_domains():
    """A hand-picked valid prediction set satisfies all domain checks."""
    preds = [
        _pred("1_2", 0.0, "none", []),
        _pred("3_4", 1.0, "directed_transfer", ["7", "8", "9"]),
        _pred("5_6", 0.5, "other_coordination", ["10", "11", "12", "13", "14"]),
    ]
    _assert_valid_frame_domains(preds)


@pytest.mark.parametrize("bad_risk", [-0.01, 1.5, float("nan"), float("inf"), -float("inf")])
def test_bad_risk_example_rejected(bad_risk):
    _assert_invalid_rejected([_pred("1_2", bad_risk, "none", [])])


def test_bad_behavior_example_rejected():
    _assert_invalid_rejected([_pred("1_2", 0.5, "made_up_behavior", [])])


def test_duplicate_evidence_example_rejected():
    _assert_invalid_rejected(
        [_pred("1_2", 0.5, "soft_play", ["42", "42", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE])]
    )


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback         #
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _pair_id_strategy = st.text(alphabet="0123456789abcdef_", min_size=2, max_size=8)

    @settings(max_examples=200, deadline=None)
    @given(data=st.data())
    def test_property_valid_domains_hypothesis(data):
        n = data.draw(st.integers(min_value=1, max_value=12))
        pair_ids = data.draw(
            st.lists(_pair_id_strategy, min_size=n, max_size=n, unique=True)
        )
        preds = []
        for pid in pair_ids:
            risk = data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
            behavior = data.draw(st.sampled_from(_BEHAVIORS))
            k = data.draw(st.integers(min_value=0, max_value=5))
            ev = data.draw(
                st.lists(st.integers(min_value=1, max_value=10_000), min_size=k, max_size=k, unique=True)
            )
            preds.append(_pred(pid, risk, behavior, [str(h) for h in ev]))
        _assert_valid_frame_domains(preds)

    @settings(max_examples=200, deadline=None)
    @given(data=st.data())
    def test_property_invalid_rejected_hypothesis(data):
        pid = data.draw(_pair_id_strategy)
        kind = data.draw(st.sampled_from(["risk", "behavior", "dup_evidence"]))
        if kind == "risk":
            bad_risk = data.draw(
                st.one_of(
                    st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
                    st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
                    st.just(float("nan")),
                    st.just(float("inf")),
                    st.just(-float("inf")),
                )
            )
            preds = [_pred(pid, bad_risk, "none", [])]
        elif kind == "behavior":
            bad = data.draw(st.sampled_from(_BAD_BEHAVIORS))
            preds = [_pred(pid, 0.5, bad, [])]
        else:
            hand = data.draw(st.integers(min_value=1, max_value=9999))
            preds = [_pred(pid, 0.5, "soft_play", [str(hand), str(hand)])]
        _assert_invalid_rejected(preds)

else:

    _N_SEEDS = 200

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_valid_domains_seeded_loop(seed: int):
        rng = random.Random(seed)
        n = rng.randint(1, 12)
        pair_ids: List[str] = []
        seen: set[str] = set()
        while len(pair_ids) < n:
            token = "".join(rng.choice("0123456789abcdef_") for _ in range(rng.randint(2, 8)))
            if token and token not in seen:
                seen.add(token)
                pair_ids.append(token)
        preds = [
            _pred(pid, rng.uniform(0.0, 1.0), rng.choice(_BEHAVIORS), _valid_evidence(rng))
            for pid in pair_ids
        ]
        _assert_valid_frame_domains(preds)

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_invalid_rejected_seeded_loop(seed: int):
        rng = random.Random(seed)
        pid = "".join(rng.choice("0123456789abcdef_") for _ in range(rng.randint(2, 8))) or "x"
        kind = rng.choice(["risk", "behavior", "dup_evidence"])
        if kind == "risk":
            bad_risk = rng.choice(
                [
                    rng.uniform(-100.0, -0.0001),
                    rng.uniform(1.0001, 100.0),
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                ]
            )
            preds = [_pred(pid, bad_risk, "none", [])]
        elif kind == "behavior":
            preds = [_pred(pid, rng.uniform(0.0, 1.0), rng.choice(_BAD_BEHAVIORS), [])]
        else:
            hand = str(rng.randint(1, 9999))
            preds = [_pred(pid, rng.uniform(0.0, 1.0), "soft_play", [hand, hand])]
        _assert_invalid_rejected(preds)
