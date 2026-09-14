"""Unit test for the filename->real-LB parser on the exact nine real artifacts.

Task 1.5 of the ``poker-anchor-reproduction`` spec. The nine on-disk artifacts
are our OWN submission ladder; each filename encodes the REAL leaderboard score
Kaggle returned (the only external-judge ground truth -- accountability contract
Rule 1). This test pins the parser (``parse_artifact_lb``) against those nine
concrete filenames so a regression in the digit->score mapping is caught
immediately rather than silently mis-scoring an anchor.

Where the property test (task 1.4) checks the round-trip over random inputs, this
example-based test asserts the exact known values ``0.18462 ... 0.44519``.

Validates: Requirements 2.1
"""

from __future__ import annotations

import pytest

from anchor_repro.artifact_names import ArtifactNameError, parse_artifact_lb

# The nine known-LB artifacts: (filename, expected real_lb from the dossier).
# The digit string is the zero-padded integer whose value / 1e5 is the real LB.
NINE_ARTIFACTS = [
    ("submission_best_018462.csv", 0.18462),
    ("submission_best_019029.csv", 0.19029),
    ("submission_best_019281.csv", 0.19281),
    ("submission_best_032580.csv", 0.32580),
    ("submission_best_033131.csv", 0.33131),
    ("submission_best_037063.csv", 0.37063),
    ("submission_best_039372.csv", 0.39372),
    ("submission_best_041289.csv", 0.41289),
    ("submission_best_044519.csv", 0.44519),
]


@pytest.mark.parametrize("filename, expected_lb", NINE_ARTIFACTS)
def test_parse_nine_real_filenames(filename: str, expected_lb: float):
    """Each of the nine real filenames parses to its known real-LB score."""
    parsed = parse_artifact_lb(filename)
    # Real LB equals int(digits) / 1e5 exactly (same float the literal produces).
    assert parsed.real_lb == expected_lb
    # The verbatim digits and their zero-pad width are preserved for round-trip.
    assert parsed.digits == filename[len("submission_best_"):-len(".csv")]
    assert parsed.width == 6


def test_nine_filenames_span_expected_ladder():
    """The parsed ladder spans exactly 0.18462 .. 0.44519 as the dossier records."""
    parsed = [parse_artifact_lb(name) for name, _ in NINE_ARTIFACTS]
    scores = [p.real_lb for p in parsed]
    assert min(scores) == pytest.approx(0.18462)
    assert max(scores) == pytest.approx(0.44519)
    # All nine are distinct external-judge points (no accidental collision).
    assert len(set(scores)) == 9


def test_nine_filenames_round_trip_exactly():
    """Re-formatting each parsed artifact reproduces the original filename byte-for-byte."""
    for filename, _ in NINE_ARTIFACTS:
        parsed = parse_artifact_lb(filename)
        assert parsed.format_filename() == filename
