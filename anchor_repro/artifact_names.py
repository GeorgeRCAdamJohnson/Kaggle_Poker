"""Filename <-> real-LB parser for the known-LB submission ladder.

Task 1.3 of the ``poker-anchor-reproduction`` spec. The nine on-disk artifacts
are named ``submission_best_<d>.csv`` where ``<d>`` is a zero-padded integer whose
value divided by ``1e5`` is the REAL leaderboard score Kaggle returned for that
submission. The filename is therefore the ONLY external-judge ground truth for a
ladder point (accountability contract Rule 1) and must be read WITHOUT distortion
(Rule 9 / Req 4.1).

This module provides:

- ``parse_artifact_lb(name) -> ArtifactLB`` : filename -> (real_lb, digits, width).
- ``format_artifact_lb(...)``               : the inverse, real_lb -> ``<d>`` digit
  string and full ``submission_best_<d>.csv`` filename.

The pair round-trips EXACTLY (Property 7): for any ``submission_best_<d>.csv`` the
parsed ``real_lb == int(d)/1e5`` and re-formatting reproduces ``<d>`` byte-for-byte
(including its zero-padding width). Because ``real_lb`` is a binary float, the
inverse recovers the integer with ``round(real_lb * SCALE)`` rather than a raw
``int(...)`` truncation, so representation drift can never corrupt the recovered
digit string.

Requirements: 2.1 (read real LB from the nine artifact filenames), 4.1 (report
the external-judge number honestly, no distortion).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

__all__ = [
    "SCALE",
    "DEFAULT_DIGIT_WIDTH",
    "FILENAME_PREFIX",
    "FILENAME_SUFFIX",
    "ArtifactLB",
    "ArtifactNameError",
    "parse_artifact_lb",
    "format_artifact_lb",
    "format_artifact_filename",
]

# ``real_lb = int(digits) / SCALE``. Five-digit scores => SCALE = 1e5, expressed
# as an int so the inverse is exact integer arithmetic (no float divisor drift).
SCALE: int = 100_000
DEFAULT_DIGIT_WIDTH: int = 5
FILENAME_PREFIX: str = "submission_best_"
FILENAME_SUFFIX: str = ".csv"

# The digit run between the fixed prefix and the ``.csv`` suffix. Anchored so a
# stray ``submission_best_x044519y.csv`` is rejected rather than silently parsed.
_FILENAME_RE = re.compile(
    r"^" + re.escape(FILENAME_PREFIX) + r"(?P<digits>\d+)" + re.escape(FILENAME_SUFFIX) + r"$"
)


class ArtifactNameError(ValueError):
    """Raised when a filename is not a valid ``submission_best_<d>.csv`` name.

    Failing loudly (rather than returning a fabricated score) upholds the
    accountability contract: an unreadable external-judge value is a null, never
    a guessed number (Rules 1, 9).
    """


@dataclass(frozen=True)
class ArtifactLB:
    """A parsed known-LB artifact filename.

    Attributes:
        real_lb: The external-judge leaderboard score, ``int(digits) / SCALE``.
        digits: The original ``<d>`` digit string, preserved verbatim (with its
            zero padding) so re-formatting round-trips exactly.
        width: ``len(digits)`` -- the zero-pad width used when re-formatting.
    """

    real_lb: float
    digits: str
    width: int

    @property
    def scaled_int(self) -> int:
        """The integer ``<d>`` represents (``int(self.digits)``)."""
        return int(self.digits)

    def format_digits(self) -> str:
        """Inverse of the parse: reproduce ``<d>`` from ``real_lb`` exactly."""
        return format_artifact_lb(self.real_lb, width=self.width)

    def format_filename(self) -> str:
        """Reproduce the full ``submission_best_<d>.csv`` filename exactly."""
        return format_artifact_filename(self.real_lb, width=self.width)


def _basename(name: Union[str, Path]) -> str:
    """Accept a bare filename, a relative path, or a full ``Path`` and return the
    filename component. Only the basename carries the score; directories are
    ignored so ``outputs/.../submission_best_044519.csv`` parses the same as the
    bare name."""
    return Path(name).name


def parse_artifact_lb(name: Union[str, Path]) -> ArtifactLB:
    """Parse ``submission_best_<d>.csv`` into its external-judge LB score.

    Args:
        name: An artifact filename or path ending in ``submission_best_<d>.csv``.

    Returns:
        An :class:`ArtifactLB` with ``real_lb == int(d) / SCALE``, the verbatim
        ``digits`` string, and its ``width``.

    Raises:
        ArtifactNameError: If ``name`` is not a valid artifact filename. No score
            is fabricated for an unparseable name (contract Rules 1, 9).
    """
    basename = _basename(name)
    match = _FILENAME_RE.match(basename)
    if match is None:
        raise ArtifactNameError(
            f"not a submission_best_<d>.csv artifact filename: {basename!r}"
        )
    digits = match.group("digits")
    real_lb = int(digits) / SCALE
    return ArtifactLB(real_lb=real_lb, digits=digits, width=len(digits))


def format_artifact_lb(real_lb: float, *, width: int = DEFAULT_DIGIT_WIDTH) -> str:
    """Inverse of :func:`parse_artifact_lb`: real LB score -> ``<d>`` digit string.

    Recovers the integer with ``round(real_lb * SCALE)`` (not ``int(...)``) so the
    binary-float representation of ``real_lb`` cannot corrupt the recovered digits,
    then zero-pads to ``width``.

    Args:
        real_lb: A leaderboard score in ``[0, 1)`` such as ``0.44519``.
        width: Zero-pad width for the emitted digit string (default 5).

    Returns:
        The ``<d>`` digit string, e.g. ``"044519"`` for ``0.44519`` at width 6.

    Raises:
        ArtifactNameError: If ``real_lb`` is negative or ``width`` is too small to
            hold the scaled integer without truncation (which would break the
            round-trip).
    """
    if real_lb < 0:
        raise ArtifactNameError(f"real_lb must be non-negative, got {real_lb!r}")
    scaled = round(real_lb * SCALE)
    digits = str(scaled)
    if len(digits) > width:
        raise ArtifactNameError(
            f"real_lb {real_lb!r} needs {len(digits)} digits but width={width}"
        )
    return digits.zfill(width)


def format_artifact_filename(real_lb: float, *, width: int = DEFAULT_DIGIT_WIDTH) -> str:
    """Inverse producing the full ``submission_best_<d>.csv`` filename."""
    return f"{FILENAME_PREFIX}{format_artifact_lb(real_lb, width=width)}{FILENAME_SUFFIX}"
