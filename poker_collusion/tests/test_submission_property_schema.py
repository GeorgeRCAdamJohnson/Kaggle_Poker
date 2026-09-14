# Feature: poker-collusion-detection, Property 13: Submission matches the sample schema (round-trip)
"""Property 13 (task 13.2): the built submission matches the sample schema and round-trips.

**Validates: Requirements 9.1, 9.2, 9.6**

For a random ``sample_submission`` (a random set of unique hex-ish ``pair_id`` values, size
1..30) and random ``predictions`` covering a random SUBSET of those pairs (the remainder must
be default-filled by the writer), :func:`build_submission_frame` MUST produce a frame whose:

* columns equal :data:`SUBMISSION_COLUMNS` in the exact order (Req 9.1, 9.2);
* row count equals ``len(sample_submission)`` (Req 9.6);
* ``pair_id`` SET and ORDER exactly match the sample submission, unchanged (Req 9.1, 9.6).

Then the frame is ROUND-TRIPPED through :func:`write_submission` to a ``tmp_path`` and read back
with ``dtype=str``. The reloaded frame MUST equal the built frame (same columns/order/rows),
proving the on-disk file matches the sample schema. A written file must exist and no temp
artifacts may remain in the destination directory.

Generator backend
-----------------
This test PREFERS the ``hypothesis`` library. If hypothesis is not importable it transparently
falls back to a seeded randomized loop (200 seeds). The active backend is recorded in
``GENERATOR_IN_USE`` and surfaced by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random
from typing import List, Sequence

import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE
from poker_collusion.metric.reference_public_metric import ALLOWED_BEHAVIORS
from poker_collusion.submission.writer import (
    SUBMISSION_COLUMNS,
    build_submission_frame,
    write_submission,
)
from poker_collusion.types import PairPrediction

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"

_BEHAVIORS = tuple(sorted(ALLOWED_BEHAVIORS))
_HEX = "0123456789abcdef"


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


# --------------------------------------------------------------------------- #
# Random draws for the seeded-loop fallback                                   #
# --------------------------------------------------------------------------- #
def _draw_pair_ids(rng: random.Random) -> List[str]:
    """A random set of unique hex-ish pair ids (size 1..30)."""
    n = rng.randint(1, 30)
    ids: List[str] = []
    seen: set[str] = set()
    while len(ids) < n:
        token = "".join(rng.choice(_HEX) for _ in range(rng.randint(2, 8)))
        # give it a pair-ish shape occasionally to stress str handling
        if rng.random() < 0.5:
            token = f"{token}_{rng.randint(0, 9999)}"
        if token not in seen:
            seen.add(token)
            ids.append(token)
    return ids


def _draw_valid_evidence(rng: random.Random) -> List[str]:
    """0..5 unique hand ids padded/normalized by the writer; keep them distinct."""
    k = rng.randint(0, 5)
    pool = rng.sample(range(1, 10_000), k)
    cells = [str(h) for h in pool]
    # pad with NO_EVIDENCE up to a random length <= 5 (writer normalizes anyway)
    while len(cells) < rng.randint(k, 5):
        cells.append(NO_EVIDENCE)
    return cells


def _draw_predictions(rng: random.Random, pair_ids: Sequence[str]) -> List[PairPrediction]:
    """Predictions covering a random SUBSET of the sample's pairs."""
    subset_size = rng.randint(0, len(pair_ids))
    covered = rng.sample(list(pair_ids), subset_size)
    preds: List[PairPrediction] = []
    for pid in covered:
        preds.append(
            _pred(
                pid,
                risk=rng.uniform(0.0, 1.0),
                behavior=rng.choice(_BEHAVIORS),
                evidence=_draw_valid_evidence(rng),
            )
        )
    rng.shuffle(preds)  # prediction order must not affect the output row order
    return preds


# --------------------------------------------------------------------------- #
# Core property assertion                                                     #
# --------------------------------------------------------------------------- #
def _assert_schema_and_roundtrip(
    pair_ids: Sequence[str],
    preds: Sequence[PairPrediction],
    tmp_path,
    tag: str,
) -> None:
    sample = _sample_submission(pair_ids)
    frame = build_submission_frame(preds, sample)

    # Columns == SUBMISSION_COLUMNS in exact order (Req 9.1, 9.2).
    assert tuple(frame.columns) == SUBMISSION_COLUMNS

    # Row count == len(sample) (Req 9.6).
    assert len(frame) == len(sample)

    # pair_id SET and ORDER exactly match the sample submission, unchanged (Req 9.1, 9.6).
    assert list(frame["pair_id"]) == list(sample["pair_id"].astype(str))
    assert set(frame["pair_id"]) == set(sample["pair_id"].astype(str))

    # Round-trip through the atomic writer.
    out = tmp_path / tag / "submission.csv"
    written = write_submission(preds, sample_submission=sample, path=out)
    assert written == out
    assert out.exists()

    # No temp artifacts remain in the destination directory.
    leftovers = [p.name for p in out.parent.iterdir() if p.name != "submission.csv"]
    assert leftovers == [], f"unexpected leftover artifacts: {leftovers!r}"

    # Reloaded frame equals the built frame (same columns/order/rows).
    read_back = pd.read_csv(out, dtype=str)
    expected = frame.astype(str)
    assert tuple(read_back.columns) == SUBMISSION_COLUMNS
    pd.testing.assert_frame_equal(
        read_back.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Backend report + explicit corner cases                                      #
# --------------------------------------------------------------------------- #
def test_report_generator_in_use():
    """Surface which generator backend this property test runs under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 13] generator in use: {GENERATOR_IN_USE}")


def test_single_pair_no_predictions_default_filled(tmp_path):
    """A lone pair with no prediction is default-filled and round-trips."""
    _assert_schema_and_roundtrip(["ab12_3"], [], tmp_path, "single")


def test_order_preserved_when_predictions_shuffled(tmp_path):
    """Output row order follows the sample, not the prediction order."""
    pair_ids = ["ff_1", "0a_2", "7c_3", "12ab_4"]
    preds = [
        _pred("7c_3", 0.3, "soft_play", ["10", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
        _pred("ff_1", 0.9, "directed_transfer", ["11", "12", NO_EVIDENCE, NO_EVIDENCE, NO_EVIDENCE]),
    ]
    _assert_schema_and_roundtrip(pair_ids, preds, tmp_path, "order")


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback         #
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _pair_id_strategy = st.text(alphabet=_HEX + "_", min_size=2, max_size=10)

    @settings(max_examples=200, deadline=None)
    @given(data=st.data())
    def test_property_schema_roundtrip_hypothesis(data, tmp_path):
        pair_ids = data.draw(
            st.lists(_pair_id_strategy, min_size=1, max_size=30, unique=True)
        )
        subset = data.draw(st.lists(st.sampled_from(pair_ids), unique=True, max_size=len(pair_ids)))
        preds = []
        for i, pid in enumerate(subset):
            risk = data.draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
            behavior = data.draw(st.sampled_from(_BEHAVIORS))
            k = data.draw(st.integers(min_value=0, max_value=5))
            ev_ids = data.draw(
                st.lists(st.integers(min_value=1, max_value=10_000), min_size=k, max_size=k, unique=True)
            )
            preds.append(_pred(pid, risk, behavior, [str(h) for h in ev_ids]))
        _assert_schema_and_roundtrip(pair_ids, preds, tmp_path, "hyp")

else:

    _N_SEEDS = 200

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_schema_roundtrip_seeded_loop(seed: int, tmp_path):
        rng = random.Random(seed)
        pair_ids = _draw_pair_ids(rng)
        preds = _draw_predictions(rng, pair_ids)
        _assert_schema_and_roundtrip(pair_ids, preds, tmp_path, f"seed_{seed}")
