"""Property test for the filename<->real-LB parse round-trip (task 1.4).

# Feature: poker-anchor-reproduction, Property 7: For any artifact filename
# ``submission_best_<d>.csv``, the parsed ``real_lb`` SHALL satisfy
# ``real_lb == int(d)/1e5`` and re-formatting SHALL reproduce ``<d>``.

Property 7 (design.md): the external-judge LB score is read from the artifact
filename WITHOUT distortion. For any ``submission_best_<d>.csv`` the parsed
``real_lb == int(d)/1e5`` and re-formatting ``real_lb`` reproduces ``<d>``
byte-for-byte (including its zero-padding width). This is the ONLY external-judge
ground truth for a ladder point, so a lossless round-trip is the contract-level
guarantee that we never fabricate or drift the real number (Rules 1, 9).

Because ``real_lb`` is a binary float, the inverse must recover the digits with
``round(real_lb * SCALE)`` rather than a raw ``int(...)`` truncation; this test
exercises the full 5-digit score range (min 100 iterations) to confirm no
representation drift corrupts the recovered digit string.

Validates: Requirements 2.1, 4.1
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from anchor_repro.artifact_names import (
    DEFAULT_DIGIT_WIDTH,
    FILENAME_PREFIX,
    FILENAME_SUFFIX,
    SCALE,
    format_artifact_filename,
    format_artifact_lb,
    parse_artifact_lb,
)


# The nine real artifacts are five-digit scores; SCALE == 1e5 means the largest
# five-digit value is 99999 -> 0.99999. Draw the full 5-digit integer range so
# every representable ladder score (and beyond the nine we hold) is exercised.
_MAX_5_DIGIT = SCALE - 1  # 99999


# Feature: poker-anchor-reproduction, Property 7: filename<->real-LB round-trip
@settings(max_examples=200, deadline=None)
@given(scaled=st.integers(min_value=0, max_value=_MAX_5_DIGIT))
def test_filename_real_lb_round_trip(scaled: int) -> None:
    """A random 5-digit score -> filename -> parse -> real_lb -> re-format -> <d>."""
    # Build the canonical zero-padded 5-digit digit string and its filename.
    digits = str(scaled).zfill(DEFAULT_DIGIT_WIDTH)
    filename = f"{FILENAME_PREFIX}{digits}{FILENAME_SUFFIX}"

    parsed = parse_artifact_lb(filename)

    # (1) real_lb == int(d) / 1e5 exactly (the parse reads the external judge).
    assert parsed.real_lb == int(digits) / 1e5
    # Guard the scale is the design's 1e5 (not silently redefined).
    assert parsed.real_lb == scaled / SCALE
    assert parsed.digits == digits
    assert parsed.width == DEFAULT_DIGIT_WIDTH
    assert parsed.scaled_int == scaled

    # (2) re-formatting reproduces <d> byte-for-byte (round-trip, no drift).
    assert parsed.format_digits() == digits
    assert format_artifact_lb(parsed.real_lb, width=DEFAULT_DIGIT_WIDTH) == digits

    # (3) the full filename round-trips too.
    assert parsed.format_filename() == filename
    assert format_artifact_filename(parsed.real_lb, width=DEFAULT_DIGIT_WIDTH) == filename


# Feature: poker-anchor-reproduction, Property 7: round-trip is width-preserving
@settings(max_examples=200, deadline=None)
@given(
    scaled=st.integers(min_value=0, max_value=_MAX_5_DIGIT),
    # Extra zero-padding beyond the natural width must be preserved verbatim so
    # a wider-than-5 digit string still round-trips exactly.
    width=st.integers(min_value=DEFAULT_DIGIT_WIDTH, max_value=9),
)
def test_round_trip_preserves_arbitrary_zero_pad_width(scaled: int, width: int) -> None:
    """Round-trip holds at the parsed width for any zero-pad width >= 5 digits."""
    digits = str(scaled).zfill(width)
    filename = f"{FILENAME_PREFIX}{digits}{FILENAME_SUFFIX}"

    parsed = parse_artifact_lb(filename)

    assert parsed.real_lb == int(digits) / 1e5
    assert parsed.width == len(digits)
    # Re-format at the SAME width the filename carried -> exact digit reproduction.
    assert parsed.format_digits() == digits
    assert parsed.format_filename() == filename
