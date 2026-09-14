"""Property test for the guarded floor writer (task 16.5).

One property, one test. This is the ONLY property implemented here.

# Feature: poker-layered-tuning, Property 22: The floor artifact is never overwritten without a verified improvement

Property statement (design.md, Property 22 / Req 7.4):
    For any attempt to write the Known_Good_Floor submission artifact, the
    artifact SHALL remain byte-for-byte unchanged unless the write is backed by a
    leaderboard-verified improvement over the current floor.

The guard's accept condition is exactly ``verified AND real_lb > current_floor_lb``.
This test draws ``real_lb`` spanning below / equal to / above the current floor and
``verified`` as a boolean so all four quadrants are exercised, and asserts the
byte-level invariant in every quadrant:

  - improvement quadrant (verified AND real_lb > floor): the write succeeds and the
    floor bytes now equal the candidate bytes;
  - every other quadrant: ``FloorWriteRefused`` is raised AND the floor bytes are
    byte-for-byte identical to the original.

Per-example temp files are used (``tempfile.mkdtemp`` per call, cleaned up in a
``finally``) because ``pytest``'s ``tmp_path`` fixture is created once per test and
would NOT re-run for each Hypothesis ``@given`` example.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from poker_collusion.tuning_harness.submission.gate import (
    FloorWriteRefused,
    KNOWN_GOOD_FLOOR,
    guarded_floor_write,
)

# Fixed, known bytes for the floor artifact. Every example starts from exactly
# these bytes so "unchanged" can be checked byte-for-byte.
_FLOOR_BYTES = b"pair_id,risk_score\np1,0.9\np2,0.1\n"


def _make_temp_case(candidate_seed: int) -> tuple[Path, Path, Path, bytes, bytes]:
    """Create a fresh temp dir with a fixed-bytes floor and a DISTINCT candidate.

    Returns ``(tmp_dir, floor, candidate, floor_bytes, candidate_bytes)``. The
    candidate bytes are seeded to differ from the floor (and vary per example) so a
    successful promotion is observable as a real byte change.
    """

    tmp_dir = Path(tempfile.mkdtemp(prefix="floor_guard_prop_"))
    floor = tmp_dir / "submission_best_044519.csv"
    floor.write_bytes(_FLOOR_BYTES)

    # A candidate whose bytes differ from the floor's, varied by the drawn seed so
    # promotion is a genuine, observable change rather than an accidental no-op.
    candidate_bytes = f"pair_id,risk_score\np1,0.{800 + (candidate_seed % 100)}\np2,0.2\n".encode()
    candidate = tmp_dir / "candidate.csv"
    candidate.write_bytes(candidate_bytes)

    dossier = tmp_dir / "RESEARCH_DOSSIER.md"
    dossier.write_text("# dossier\n", encoding="utf-8")
    return tmp_dir, floor, candidate, _FLOOR_BYTES, candidate_bytes


@settings(max_examples=200)
@given(
    # real_lb spans clearly below, exactly equal to, and clearly above the floor so
    # the < / == / > branches are all sampled densely.
    real_lb=st.floats(
        min_value=KNOWN_GOOD_FLOOR - 0.05,
        max_value=KNOWN_GOOD_FLOOR + 0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    verified=st.booleans(),
    candidate_seed=st.integers(min_value=0, max_value=10_000),
)
def test_floor_never_overwritten_without_verified_improvement(
    real_lb: float, verified: bool, candidate_seed: int
) -> None:
    tmp_dir, floor, candidate, floor_bytes, candidate_bytes = _make_temp_case(candidate_seed)
    try:
        # The invariant's definition of "improvement": verified AND strictly above.
        is_improvement = verified and (real_lb > KNOWN_GOOD_FLOOR)

        if is_improvement:
            result = guarded_floor_write(
                candidate,
                real_lb=real_lb,
                current_floor_lb=KNOWN_GOOD_FLOOR,
                floor_path=floor,
                dossier_path=tmp_dir / "RESEARCH_DOSSIER.md",
                verified=verified,
            )
            # Accepted: the floor now holds the candidate's bytes (promoted).
            assert result == floor
            assert floor.read_bytes() == candidate_bytes
            # And the promoted bytes really did differ from the original floor.
            assert candidate_bytes != floor_bytes
        else:
            # Not an improvement (regression, tie, or unverified): must refuse AND
            # leave the artifact byte-for-byte unchanged.
            with pytest.raises(FloorWriteRefused):
                guarded_floor_write(
                    candidate,
                    real_lb=real_lb,
                    current_floor_lb=KNOWN_GOOD_FLOOR,
                    floor_path=floor,
                    dossier_path=tmp_dir / "RESEARCH_DOSSIER.md",
                    verified=verified,
                )
            assert floor.read_bytes() == floor_bytes
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
