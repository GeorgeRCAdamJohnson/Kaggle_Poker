"""Integration test for the guarded floor writer against the REAL floor artifact.

Task 17.2 (poker-layered-tuning) / Requirement 7.4:
    "The system SHALL NEVER overwrite the Known_Good_Floor submission artifact
     without a leaderboard-verified improvement."

Unlike the property test (``test_submission_floor_guard_property.py``), which
uses synthetic fixed bytes, this is a genuine INTEGRATION test against the real
artifact's *actual content*. The real Known_Good_Floor artifact
``submission_best_044519.csv`` (real LB 0.44519) is the single most valuable file
in the project, so the test NEVER points the guard at it. Instead it:

  1. reads the real artifact's actual bytes,
  2. copies those exact bytes into an isolated temp dir,
  3. points ``guarded_floor_write`` at the temp COPY via ``floor_path``,
  4. attempts NON-IMPROVING writes (a regression, a tie, and an unverified
     projection), and
  5. asserts ``FloorWriteRefused`` is raised AND the temp copy's bytes are
     byte-for-byte identical to the real artifact's original bytes.

If the real artifact does not exist in this environment the test SKIPs with a
clear message rather than fabricating a stand-in -- an honest null, not a
manufactured pass.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from poker_collusion.tuning_harness.submission.gate import (
    DEFAULT_FLOOR_ARTIFACT_PATH,
    FloorWriteRefused,
    KNOWN_GOOD_FLOOR,
    guarded_floor_write,
)


def _read_real_floor_bytes() -> bytes:
    """Return the real floor artifact's bytes, or skip if it is absent here.

    The real artifact lives at :data:`DEFAULT_FLOOR_ARTIFACT_PATH`. This
    environment may not carry the large output CSVs, so we skip (never fabricate)
    when it is missing.
    """

    if not DEFAULT_FLOOR_ARTIFACT_PATH.is_file():
        pytest.skip(
            "Real floor artifact not present in this environment: "
            f"{DEFAULT_FLOOR_ARTIFACT_PATH}. Skipping the real-artifact "
            "integration test (no fabricated stand-in)."
        )
    return DEFAULT_FLOOR_ARTIFACT_PATH.read_bytes()


# Non-improving write scenarios. Each is a (real_lb, verified, label) tuple that
# the guard must REFUSE, per the accept rule ``verified AND real_lb > floor``:
#   - regression: verified but strictly below the floor,
#   - tie: verified but exactly equal to the floor (a tie is not an improvement),
#   - unverified: numerically above the floor but NOT leaderboard-verified
#                 (a projected value may never overwrite the floor).
_NON_IMPROVING_CASES = [
    (KNOWN_GOOD_FLOOR - 0.001, True, "regression_below_floor"),
    (KNOWN_GOOD_FLOOR, True, "tie_equal_to_floor"),
    (KNOWN_GOOD_FLOOR + 0.010, False, "unverified_projection_above_floor"),
]


@pytest.mark.parametrize(
    "real_lb, verified, label",
    _NON_IMPROVING_CASES,
    ids=[c[2] for c in _NON_IMPROVING_CASES],
)
def test_non_improving_write_leaves_real_artifact_copy_unchanged(
    real_lb: float, verified: bool, label: str
) -> None:
    """A non-improving write against a copy of the REAL artifact refuses + no-op.

    Genuine integration check: the temp copy is seeded with the real artifact's
    actual bytes, so we prove the guard protects the real content, not a synthetic
    placeholder.
    """

    original_bytes = _read_real_floor_bytes()

    tmp_dir = Path(tempfile.mkdtemp(prefix="floor_guard_integration_"))
    try:
        # Copy the REAL artifact's exact bytes into an isolated temp location and
        # confirm the copy is a faithful reproduction before we test the guard.
        floor_copy = tmp_dir / "submission_best_044519.csv"
        floor_copy.write_bytes(original_bytes)
        assert floor_copy.read_bytes() == original_bytes

        # A candidate whose bytes DIFFER from the floor, so that any accidental
        # promotion would be detectable as a real byte change.
        candidate = tmp_dir / "candidate.csv"
        candidate.write_bytes(original_bytes + b"\nPZZZZZZZZZZZZZ,0.99,none,,,,,\n")

        dossier = tmp_dir / "RESEARCH_DOSSIER.md"
        dossier.write_text("# integration dossier\n", encoding="utf-8")

        # The non-improving write MUST be refused...
        with pytest.raises(FloorWriteRefused):
            guarded_floor_write(
                candidate,
                real_lb=real_lb,
                current_floor_lb=KNOWN_GOOD_FLOOR,
                floor_path=floor_copy,
                dossier_path=dossier,
                verified=verified,
            )

        # ...and the guarded copy must be byte-for-byte identical to the real
        # artifact's original bytes (unchanged), proving the no-op invariant.
        assert floor_copy.read_bytes() == original_bytes
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Defense-in-depth: the real artifact itself must never have been touched.
        assert DEFAULT_FLOOR_ARTIFACT_PATH.read_bytes() == original_bytes
