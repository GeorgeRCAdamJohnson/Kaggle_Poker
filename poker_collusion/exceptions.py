"""Shared exception types for the poker_collusion pipeline.

Requirements:
- 1.5: When a referenced ``hand_id``, ``player_id``, or expected column is missing
  from an input file, the Data_Loader raises a descriptive error identifying the
  file and the missing key. ``SchemaError`` is that error.
"""

from __future__ import annotations

__all__ = ["SchemaError"]


class SchemaError(Exception):
    """Raised when an input file is missing an expected key, column, or reference.

    Carries the offending ``file`` and ``missing_key`` so the failure is
    self-describing (Requirement 1.5). The rendered message names both, which is
    what downstream error handling and tests assert on.

    Attributes:
        file: The input file (path or logical name) that failed validation.
        missing_key: The missing column name, ``hand_id``, ``player_id``, or other
            expected key.
    """

    def __init__(self, file: str, missing_key: str) -> None:
        self.file = str(file)
        self.missing_key = str(missing_key)
        message = (
            f"Schema validation failed for input file {self.file!r}: "
            f"missing expected key {self.missing_key!r}."
        )
        super().__init__(message)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SchemaError(file={self.file!r}, missing_key={self.missing_key!r})"
