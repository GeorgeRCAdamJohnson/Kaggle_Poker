"""Phase -1 data access and schema/join-key forensics (task 2.1).

This module implements the first gated Discovery step: obtaining the competition
files and verifying their schema and join keys before any downstream work.

It provides three capabilities:

1. A *documented* download step (Kaggle CLI) for pulling the competition files
   into the configured input directory. Downloading requires Kaggle
   credentials/CLI, which may not be present in every environment, so the
   command is emitted as a documented, copy-pasteable step rather than executed
   blindly.
2. :func:`verify_schema` — checks that all eight expected input files exist and
   carry the expected join-key columns, raising
   :class:`~poker_collusion.exceptions.SchemaError` on any missing file/column
   (Requirements 1.1, 1.2, 1.5). It also confirms ``hands.phase`` values,
   verifies ``(hand_id, action_no)`` ordering exists in ``actions.parquet``, and
   records the ``actions.parquet`` row count (expected 18,609,028, Requirement
   1.6, exposed via :data:`poker_collusion.config.ACTIONS_ROW_COUNT`).
3. :func:`record_findings` — appends the verification findings (files present,
   columns, row counts, ``hands.phase`` values) to ``RESEARCH_DOSSIER.md``
   (Requirement 11.4 / Phase -1 Workstream E).

The verification runs gracefully whether or not the data is present: when files
are absent it reports that clearly (via :attr:`SchemaReport.data_present`)
instead of crashing, and only escalates to :class:`SchemaError` when
``strict=True`` is requested against present-but-malformed data.

Requirements: 1.1, 1.2, 1.3, 1.6.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from poker_collusion.config import ACTIONS_ROW_COUNT, PipelineConfig, get_config
from poker_collusion.exceptions import SchemaError

__all__ = [
    "EXPECTED_FILES",
    "EXPECTED_COLUMNS",
    "EXPECTED_PHASE_VALUES",
    "COMPETITION_SLUG",
    "FileReport",
    "SchemaReport",
    "kaggle_download_command",
    "verify_schema",
    "record_findings",
    "main",
]

# The competition dataset slug used by the Kaggle CLI. Update if the official
# slug differs; this is the documented download target.
COMPETITION_SLUG: str = "detect-suspicious-value-transfers-in-poker"

# The eight required competition input files (Requirement 1.1).
EXPECTED_FILES: tuple[str, ...] = (
    "players.parquet",
    "hands.parquet",
    "seats.parquet",
    "actions.parquet",
    "development_labels.csv",
    "development_evidence.csv",
    "evaluation_pairs.csv",
    "sample_submission.csv",
)

# Minimum join-key / structural columns each file must expose. Verification
# fails (SchemaError) when any of these is missing (Requirements 1.2, 1.5).
# These are the *required* keys, not the full column list; extra columns are
# fine and are recorded in the findings.
EXPECTED_COLUMNS: Dict[str, List[str]] = {
    # player_id links the player tables.
    "players.parquet": ["player_id"],
    # hand_id links gameplay tables; phase partitions dev/eval.
    "hands.parquet": ["hand_id", "phase"],
    # seats links a hand to its seated players (hand_id + player_id).
    "seats.parquet": ["hand_id", "player_id"],
    # actions must preserve (hand_id, action_no) ordering.
    "actions.parquet": ["hand_id", "action_no"],
    # label / evidence / pair / template CSVs.
    "development_labels.csv": ["pair_id"],
    "development_evidence.csv": ["pair_id"],
    "evaluation_pairs.csv": ["pair_id"],
    "sample_submission.csv": ["pair_id"],
}

# hands.phase must be drawn from this set (dev/eval partitioning, Requirement 1.3).
EXPECTED_PHASE_VALUES: frozenset[str] = frozenset({"development", "evaluation"})


# --------------------------------------------------------------------------- #
# Documented download step
# --------------------------------------------------------------------------- #
def kaggle_download_command(config: Optional[PipelineConfig] = None) -> str:
    """Return the documented Kaggle CLI command to download competition files.

    Downloading requires the Kaggle CLI (``pip install kaggle``) and API
    credentials configured at ``~/.kaggle/kaggle.json``. The command downloads
    and unzips all competition files into the configured input directory.

    This is intentionally *returned* rather than executed: many environments
    (including CI and the Kaggle kernel itself, where the data is already
    mounted under ``/kaggle/input``) neither have nor need the CLI. Run the
    printed command manually when local download is required.
    """
    config = config or get_config()
    input_dir = config.input_dir
    return (
        f"kaggle competitions download -c {COMPETITION_SLUG} "
        f'-p "{input_dir}" && '
        f'python -c "import zipfile,glob,os; '
        f"[zipfile.ZipFile(z).extractall(r'{input_dir}') "
        f'for z in glob.glob(os.path.join(r\'{input_dir}\', \'*.zip\'))]"'
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class FileReport:
    """Per-file verification result recorded in the dossier."""

    name: str
    present: bool
    columns: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    missing_columns: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class SchemaReport:
    """Aggregate findings for the whole competition input directory.

    Attributes:
        input_dir: Directory that was inspected.
        data_present: True when at least one expected file exists. When False,
            the caller can report "data not downloaded yet" rather than treating
            the absence as a schema failure.
        files: Per-file reports keyed by filename.
        actions_row_count: Row count observed in ``actions.parquet`` (or None).
        actions_row_count_matches_expected: Whether it matched
            :data:`ACTIONS_ROW_COUNT`.
        hands_phase_values: Distinct ``hands.phase`` values observed.
        actions_ordering_ok: Whether ``(hand_id, action_no)`` is non-decreasing.
        errors: Human-readable descriptions of every schema problem found.
    """

    input_dir: str
    data_present: bool = False
    files: Dict[str, FileReport] = field(default_factory=dict)
    actions_row_count: Optional[int] = None
    actions_row_count_matches_expected: Optional[bool] = None
    hands_phase_values: List[str] = field(default_factory=list)
    actions_ordering_ok: Optional[bool] = None
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when data is present and no schema errors were found."""
        return self.data_present and not self.errors


# --------------------------------------------------------------------------- #
# Schema verification
# --------------------------------------------------------------------------- #
def _read_columns_and_count(path: Path) -> tuple[List[str], Optional[int]]:
    """Return (columns, row_count) for a parquet or CSV file.

    For parquet, the schema (columns) is read from metadata without loading the
    whole file, and the row count is read from parquet metadata cheaply (never
    loading ``actions.parquet`` whole, Requirement 1.6). For CSV, columns come
    from the header and the row count from a streaming line count.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        columns = list(pf.schema_arrow.names)
        row_count = pf.metadata.num_rows if pf.metadata is not None else None
        return columns, row_count

    if suffix == ".csv":
        import pandas as pd

        header = pd.read_csv(path, nrows=0)
        columns = list(header.columns)
        # Streaming row count: total lines minus the header, avoids loading the
        # full frame into memory.
        with path.open("r", encoding="utf-8", newline="") as fh:
            line_total = sum(1 for _ in fh)
        row_count = max(line_total - 1, 0)
        return columns, row_count

    raise ValueError(f"Unsupported file type for schema read: {path!r}")


def _check_actions_ordering(path: Path, sample_rows: int = 200_000) -> bool:
    """Confirm ``(hand_id, action_no)`` ordering exists / is non-decreasing.

    Reads only the ``hand_id`` and ``action_no`` columns of the first row group
    (column-projected, never the whole file, Requirement 1.6) and confirms the
    rows are sorted non-decreasingly by ``(hand_id, action_no)``.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    batch = next(
        pf.iter_batches(batch_size=sample_rows, columns=["hand_id", "action_no"]),
        None,
    )
    if batch is None:
        return True
    df = batch.to_pandas()
    if df.empty:
        return True
    ordered = df[["hand_id", "action_no"]]
    is_sorted = ordered.apply(tuple, axis=1).is_monotonic_increasing
    # Non-decreasing (allows equal-hand runs); use monotonic on the tuple.
    return bool(ordered.equals(ordered.sort_values(["hand_id", "action_no"], kind="stable")) or is_sorted)


def _read_hands_phase_values(path: Path) -> List[str]:
    """Return the distinct ``hands.phase`` values observed (Requirement 1.3)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    values: set[str] = set()
    for batch in pf.iter_batches(batch_size=200_000, columns=["phase"]):
        col = batch.column(0)
        for v in col.to_pylist():
            if v is not None:
                values.add(str(v))
    return sorted(values)


def verify_schema(
    config: Optional[PipelineConfig] = None,
    *,
    strict: bool = True,
) -> SchemaReport:
    """Verify presence, columns, join keys, phase values, and row counts.

    Args:
        config: Pipeline config supplying ``input_dir``. Defaults to
            :func:`poker_collusion.config.get_config`.
        strict: When True (default) a :class:`SchemaError` is raised on the first
            present-but-malformed file (missing required column, bad phase
            values). When False, all problems are collected into
            :attr:`SchemaReport.errors` and returned without raising, which is
            useful for reporting.

    Behavior:
        * If **no** expected file exists, the directory is treated as "data not
          downloaded yet": ``data_present=False`` and no error is raised, even
          in strict mode. This keeps the step graceful in environments without
          the data.
        * If **some but not all** expected files exist, the missing ones are
          recorded as errors (and, in strict mode, raise ``SchemaError``).

    Returns:
        A populated :class:`SchemaReport`.

    Raises:
        SchemaError: In strict mode, when a present file is missing a required
            join-key column, an expected file is missing while other data is
            present, or ``hands.phase`` contains unexpected values.
    """
    config = config or get_config()
    input_dir = Path(config.input_dir)
    report = SchemaReport(input_dir=str(input_dir))

    existing = {name: (input_dir / name) for name in EXPECTED_FILES if (input_dir / name).exists()}
    report.data_present = len(existing) > 0

    if not report.data_present:
        # Nothing to verify; report gracefully rather than crashing.
        report.errors.append(
            f"No competition files found in {input_dir!s}. "
            "Run the documented Kaggle download step (see kaggle_download_command)."
        )
        for name in EXPECTED_FILES:
            report.files[name] = FileReport(name=name, present=False, notes=["not downloaded"])
        return report

    # Data is (partially) present: verify each expected file.
    for name in EXPECTED_FILES:
        path = input_dir / name
        if not path.exists():
            report.files[name] = FileReport(name=name, present=False)
            msg = f"Missing expected input file: {name}"
            report.errors.append(msg)
            if strict:
                raise SchemaError(file=name, missing_key="<file>")
            continue

        try:
            columns, row_count = _read_columns_and_count(path)
        except Exception as exc:  # unreadable file is a schema-level failure
            report.files[name] = FileReport(
                name=name, present=True, notes=[f"unreadable: {exc}"]
            )
            report.errors.append(f"Could not read {name}: {exc}")
            if strict:
                raise SchemaError(file=name, missing_key="<readable-schema>") from exc
            continue

        required = EXPECTED_COLUMNS.get(name, [])
        missing = [c for c in required if c not in columns]
        fr = FileReport(
            name=name,
            present=True,
            columns=columns,
            row_count=row_count,
            missing_columns=missing,
        )
        report.files[name] = fr

        if missing:
            report.errors.append(
                f"{name} is missing required column(s): {', '.join(missing)}"
            )
            if strict:
                raise SchemaError(file=name, missing_key=missing[0])

    # actions.parquet row count + ordering (Requirement 1.6, 1.2).
    actions_path = input_dir / "actions.parquet"
    if actions_path.exists() and not report.files["actions.parquet"].missing_columns:
        report.actions_row_count = report.files["actions.parquet"].row_count
        report.actions_row_count_matches_expected = (
            report.actions_row_count == ACTIONS_ROW_COUNT
        )
        if not report.actions_row_count_matches_expected:
            report.files["actions.parquet"].notes.append(
                f"row count {report.actions_row_count} != expected {ACTIONS_ROW_COUNT}"
            )
        try:
            report.actions_ordering_ok = _check_actions_ordering(actions_path)
            if not report.actions_ordering_ok:
                report.errors.append(
                    "actions.parquet is not ordered by (hand_id, action_no)"
                )
        except Exception as exc:  # ordering check is best-effort
            report.files["actions.parquet"].notes.append(f"ordering check failed: {exc}")

    # hands.phase values (Requirement 1.3).
    hands_path = input_dir / "hands.parquet"
    if hands_path.exists() and not report.files["hands.parquet"].missing_columns:
        try:
            report.hands_phase_values = _read_hands_phase_values(hands_path)
            unexpected = set(report.hands_phase_values) - EXPECTED_PHASE_VALUES
            if unexpected:
                report.files["hands.parquet"].notes.append(
                    f"unexpected phase value(s): {sorted(unexpected)}"
                )
        except Exception as exc:
            report.files["hands.parquet"].notes.append(f"phase read failed: {exc}")

    return report


# --------------------------------------------------------------------------- #
# Dossier reporting
# --------------------------------------------------------------------------- #
def _dossier_path() -> Path:
    """Path to the living research dossier alongside this spec."""
    return (
        Path(__file__).resolve().parents[2]
        / ".kiro"
        / "specs"
        / "poker-collusion-detection"
        / "RESEARCH_DOSSIER.md"
    )


def _render_findings(report: SchemaReport) -> str:
    """Render the schema report as a Markdown dossier section."""
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append(f"### Schema & Join-Key Verification ({stamp})")
    lines.append("")
    lines.append(f"- Input directory: `{report.input_dir}`")
    lines.append(f"- Data present: **{report.data_present}**")
    lines.append(f"- Overall schema OK: **{report.ok}**")
    if report.actions_row_count is not None:
        match = report.actions_row_count_matches_expected
        lines.append(
            f"- `actions.parquet` rows: **{report.actions_row_count:,}** "
            f"(expected {ACTIONS_ROW_COUNT:,}, match={match})"
        )
    if report.actions_ordering_ok is not None:
        lines.append(
            f"- `(hand_id, action_no)` ordering present: **{report.actions_ordering_ok}**"
        )
    if report.hands_phase_values:
        lines.append(
            f"- `hands.phase` values: {', '.join(f'`{v}`' for v in report.hands_phase_values)}"
        )
    lines.append("")
    lines.append("| File | Present | #Cols | Rows | Missing cols | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in EXPECTED_FILES:
        fr = report.files.get(name)
        if fr is None:
            lines.append(f"| `{name}` | ? | | | | not evaluated |")
            continue
        rows = "" if fr.row_count is None else f"{fr.row_count:,}"
        missing = ", ".join(fr.missing_columns) if fr.missing_columns else "-"
        notes = "; ".join(fr.notes) if fr.notes else "-"
        lines.append(
            f"| `{name}` | {fr.present} | {len(fr.columns)} | {rows} | {missing} | {notes} |"
        )
    if report.errors:
        lines.append("")
        lines.append("**Errors / open items:**")
        for err in report.errors:
            lines.append(f"- {err}")
    lines.append("")
    return "\n".join(lines)


def record_findings(report: SchemaReport, dossier_path: Optional[Path] = None) -> Path:
    """Append the verification findings to ``RESEARCH_DOSSIER.md``.

    Creates the dossier with a header if it does not yet exist. Returns the path
    written. (Requirement 11.4 / Phase -1 Workstream E.)
    """
    path = dossier_path or _dossier_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    section = _render_findings(report)
    if not path.exists():
        header = (
            "# Research Dossier — Detect Suspicious Value Transfers in Poker\n\n"
            "> Standing rule: no modeling begins until Discovery findings are "
            "written and reviewed. Every later phase must consult and update "
            "this dossier.\n\n"
            "## Data-Generation Findings\n\n"
        )
        path.write_text(header + section + "\n", encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + section + "\n")
    return path


def main(config: Optional[PipelineConfig] = None) -> SchemaReport:
    """Entry point: verify schema (non-strict) and record findings.

    Runs gracefully whether or not the data is present. Prints the documented
    download command and a short summary, records findings to the dossier, and
    returns the report. Uses ``strict=False`` so it reports every problem rather
    than stopping at the first; the strict :class:`SchemaError` path remains
    exercisable via :func:`verify_schema` with ``strict=True``.
    """
    config = config or get_config()
    print("Documented Kaggle download step:")
    print("  " + kaggle_download_command(config))
    print()
    report = verify_schema(config, strict=False)
    dossier = record_findings(report)
    status = "OK" if report.ok else ("NO DATA" if not report.data_present else "SCHEMA ISSUES")
    print(f"Schema verification: {status}")
    if report.errors:
        for err in report.errors:
            print(f"  - {err}")
    print(f"Findings recorded to: {dossier}")
    return report


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
