"""Data_Loader: chunked, column-projected loading and joining of the competition files.

Implements Requirement 1 (Data Ingestion and Joining) from the poker-collusion-detection
spec and the ``DataLoader`` interface declared in the design document.

Design decisions grounded in the confirmed Discovery findings
-------------------------------------------------------------
* **Real schema.** Column names come from ``config.DEFAULT_COLUMN_SELECTION`` which was
  corrected to the REAL competition schema (verified against the parquet/CSV files):
  players/hands/seats/actions parquet plus development_labels / development_evidence /
  evaluation_pairs / sample_submission CSVs.
* **Hex string IDs throughout.** ``player_id`` (``U...``), ``hand_id`` (``H...``),
  ``table_id`` (``T...``) and ``pair_id`` (``P...``) are opaque hex strings, never ints.
* **Pool == ``table_id``.** There is no ``pool`` column anywhere. A pool is one persistent
  ``table_id`` (400 tables x 30 players x ~5,000 hands, phase-split ~3000 dev / ~2000 eval).
* **actions.parquet is never loaded whole** (18,609,028 rows). ``iter_actions`` streams it
  in parquet-row-group-sized batches with column projection, driven by
  ``config.row_group_size``.
* **actions ordering.** Within each hand block, rows are ordered by ``action_no`` (verified:
  100% of hands have monotonic ``action_no``). Hand blocks are NOT globally sorted by
  ``hand_id`` (verified). Because a single hand's rows live in one contiguous block, each
  yielded batch is sorted by ``(hand_id, action_no)`` so that per-hand consumers see actions
  in decision order; global cross-batch ``hand_id`` order is intentionally not promised.
* **Per-pool actions filtering.** ``actions.parquet`` carries no pool/table column, so pool
  filtering is resolved by first computing the pool's ``hand_id`` set from
  ``hands.table_id`` and then filtering each streamed batch to those hand ids. This is why
  per-pool iteration requires the hands table; it is documented on ``iter_actions``.

Missing files or required columns raise :class:`poker_collusion.exceptions.SchemaError`
carrying the offending file and key (Requirement 1.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set, Union

import pandas as pd
import pyarrow.parquet as pq

from poker_collusion.config import (
    DEFAULT_COLUMN_SELECTION,
    PipelineConfig,
    get_config,
)
from poker_collusion.exceptions import SchemaError
from poker_collusion.types import LabelTable, Pair

__all__ = ["DataLoader", "SharedHands"]

# ---------------------------------------------------------------------------
# Input file logical names -> on-disk file names.
# ---------------------------------------------------------------------------
_PARQUET_FILES = {
    "players": "players.parquet",
    "hands": "hands.parquet",
    "seats": "seats.parquet",
    "actions": "actions.parquet",
}
_CSV_FILES = {
    "development_labels": "development_labels.csv",
    "development_evidence": "development_evidence.csv",
    "evaluation_pairs": "evaluation_pairs.csv",
    "sample_submission": "sample_submission.csv",
}

# Phase tokens carried by hands.phase (Requirement 1.3).
PHASE_DEVELOPMENT = "development"
PHASE_EVALUATION = "evaluation"

# Label-status tokens in development_labels.csv (confirmed against the real file).
LABEL_STATUS_POSITIVE = "confirmed_target"
LABEL_STATUS_NEGATIVE = "confirmed_non_target"

# Required CSV columns (Requirement 1.5 validation targets).
_REQUIRED_LABEL_COLUMNS = (
    "pair_id",
    "player_1",
    "player_2",
    "label",
    "label_status",
    "behavior_family",
)
_REQUIRED_EVAL_PAIR_COLUMNS = ("pair_id", "player_1", "player_2", "shared_hands")


@dataclass(frozen=True)
class SharedHands:
    """The hand ids shared by a pair's two players, partitioned by phase.

    Attributes:
        pair_id: The pair this result describes.
        development: Hand ids (hex strings) shared in the Development_Period.
        evaluation: Hand ids (hex strings) shared in the Evaluation_Period.
        table: A column-projected frame ``[hand_id, phase]`` for the shared hands,
            useful when the caller wants phase attached without re-joining.
    """

    pair_id: str
    development: List[str]
    evaluation: List[str]
    table: pd.DataFrame

    @property
    def all_hands(self) -> List[str]:
        """All shared hand ids (development + evaluation), development first."""
        return list(self.development) + list(self.evaluation)


class DataLoader:
    """Reads and joins the competition files into analysis-ready tables.

    The loader is deliberately stateless about heavy data: parquet tables are read
    on demand with column projection, and ``actions.parquet`` is streamed rather than
    materialized (Requirement 1.6). Small metadata tables (hands, seats) may be cached
    to make repeated per-pool queries cheap.

    Args:
        input_dir: Directory containing the competition input files. Defaults to
            ``config.input_dir`` which auto-detects the local ``data/poker`` layout and
            the Kaggle ``/kaggle/input`` layout.
        columns: Optional per-table column projection overriding
            ``config.DEFAULT_COLUMN_SELECTION``.
        config: Optional :class:`PipelineConfig`; supplies ``row_group_size`` and the
            default paths / column selection when the explicit args are omitted.
    """

    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        columns: Optional[dict] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._config = config or get_config()
        self.input_dir = Path(input_dir) if input_dir is not None else Path(self._config.input_dir)
        self.columns = columns if columns is not None else self._config.columns
        self.row_group_size = int(self._config.row_group_size)
        # Lightweight caches for the small metadata tables used by pool/pair queries.
        self._hands_cache: Optional[pd.DataFrame] = None
        self._seats_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Path resolution + schema validation helpers
    # ------------------------------------------------------------------ #
    def _resolve_path(self, logical_name: str, file_name: str) -> Path:
        """Return the on-disk path for a logical input, raising if it is missing.

        Handles both the flat local layout (``data/poker/<file>``) and the Kaggle
        layout where the competition dataset is mounted under a slug directory
        inside ``/kaggle/input`` (``/kaggle/input/<slug>/<file>``).
        """
        direct = self.input_dir / file_name
        if direct.exists():
            return direct
        # Kaggle: search one level of sub-directories under input_dir.
        if self.input_dir.exists():
            for child in sorted(self.input_dir.iterdir()):
                if child.is_dir():
                    candidate = child / file_name
                    if candidate.exists():
                        return candidate
        raise SchemaError(str(direct), missing_key=f"file:{file_name}")

    def _projection(self, table: str, columns: Optional[Sequence[str]]) -> List[str]:
        """Resolve the column projection for a table."""
        if columns is not None:
            return list(columns)
        return list(self.columns.get(table, DEFAULT_COLUMN_SELECTION.get(table, [])))

    @staticmethod
    def _validate_columns(available: Sequence[str], required: Sequence[str], file: str) -> None:
        """Raise SchemaError for the first required column absent from ``available``."""
        available_set = set(available)
        for col in required:
            if col not in available_set:
                raise SchemaError(file, missing_key=col)

    def _read_parquet_projected(self, table: str, columns: Optional[Sequence[str]]) -> pd.DataFrame:
        """Column-projected read of a parquet table with schema validation."""
        path = self._resolve_path(table, _PARQUET_FILES[table])
        projection = self._projection(table, columns)
        schema_names = pq.read_schema(path).names
        self._validate_columns(schema_names, projection, str(path))
        return pd.read_parquet(path, columns=projection)

    def _read_csv(self, logical_name: str, required: Sequence[str]) -> pd.DataFrame:
        """Read a CSV input with required-column validation."""
        path = self._resolve_path(logical_name, _CSV_FILES[logical_name])
        # dtype=str would be safest for hex ids, but numeric columns (label, shared_hands)
        # must stay numeric; read normally then validate.
        df = pd.read_csv(path)
        self._validate_columns(df.columns, required, str(path))
        return df

    # ------------------------------------------------------------------ #
    # Column-projected table loads (Requirements 1.1, 1.6)
    # ------------------------------------------------------------------ #
    def load_players(self) -> pd.DataFrame:
        """Load ``players.parquet`` projected to the configured columns."""
        return self._read_parquet_projected("players", None)

    def load_hands(self) -> pd.DataFrame:
        """Load ``hands.parquet`` projected to the configured columns.

        Always includes ``hand_id``, ``table_id`` (the pool key) and ``phase`` so each
        hand is labelled Development_Period vs Evaluation_Period (Requirement 1.3).
        """
        df = self._read_parquet_projected("hands", None)
        # Guarantee the join/phase keys are present regardless of a narrow override.
        self._validate_columns(df.columns, ("hand_id", "table_id", "phase"), _PARQUET_FILES["hands"])
        return df

    def load_seats(self) -> pd.DataFrame:
        """Load ``seats.parquet`` projected to the configured columns."""
        df = self._read_parquet_projected("seats", None)
        self._validate_columns(df.columns, ("hand_id", "player_id"), _PARQUET_FILES["seats"])
        return df

    # ------------------------------------------------------------------ #
    # Cached metadata accessors (used by pool/pair queries)
    # ------------------------------------------------------------------ #
    def _hands_meta(self) -> pd.DataFrame:
        """Cached minimal hands frame ``[hand_id, table_id, phase]``."""
        if self._hands_cache is None:
            path = self._resolve_path("hands", _PARQUET_FILES["hands"])
            required = ["hand_id", "table_id", "phase"]
            schema_names = pq.read_schema(path).names
            self._validate_columns(schema_names, required, str(path))
            self._hands_cache = pd.read_parquet(path, columns=required)
        return self._hands_cache

    def _seats_meta(self) -> pd.DataFrame:
        """Cached minimal seats frame ``[hand_id, player_id]`` for co-seating queries."""
        if self._seats_cache is None:
            path = self._resolve_path("seats", _PARQUET_FILES["seats"])
            required = ["hand_id", "player_id"]
            schema_names = pq.read_schema(path).names
            self._validate_columns(schema_names, required, str(path))
            self._seats_cache = pd.read_parquet(path, columns=required)
        return self._seats_cache

    def _pool_hand_ids(self, pool_id: str) -> Set[str]:
        """Return the set of hand ids belonging to a pool (``table_id``).

        Pool filtering requires the hands table because actions.parquet has no pool
        column (documented on ``iter_actions``).
        """
        hands = self._hands_meta()
        return set(hands.loc[hands["table_id"] == pool_id, "hand_id"].tolist())

    # ------------------------------------------------------------------ #
    # Chunked, column-projected actions iterator (Requirements 1.2, 1.6)
    # ------------------------------------------------------------------ #
    def iter_actions(
        self,
        pool_id: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield chunked, column-projected batches of ``actions.parquet``.

        The full file (18,609,028 rows) is NEVER loaded whole. Batches are read using
        parquet row groups coalesced to ``config.row_group_size`` rows, projected to
        ``columns`` (default: ``config.DEFAULT_COLUMN_SELECTION['actions']``).

        Ordering: each yielded batch is sorted by ``(hand_id, action_no)``. A single
        hand's actions live in one contiguous block within the file, so within a batch a
        hand's actions appear in ``action_no`` (decision) order. Hand blocks are not
        globally sorted across the file, so the sequence of ``hand_id`` values across
        successive batches is not globally ascending — callers that need per-hand order
        should group by ``hand_id`` (see :meth:`iter_hand_actions`).

        Per-pool filtering: ``actions.parquet`` has no pool/table column. When
        ``pool_id`` is given, the pool's hand-id set is resolved from ``hands.table_id``
        (hence the hands table is required) and each batch is filtered to those hand ids.
        Batches that become empty after filtering are skipped.

        Args:
            pool_id: Optional pool key (``table_id`` hex string). ``None`` streams all
                actions unfiltered.
            columns: Optional column projection override.

        Yields:
            ``pd.DataFrame`` batches ordered by ``(hand_id, action_no)``.
        """
        path = self._resolve_path("actions", _PARQUET_FILES["actions"])
        projection = self._projection("actions", columns)
        parquet_file = pq.ParquetFile(path)
        self._validate_columns(parquet_file.schema_arrow.names, projection, str(path))

        # Sorting/filtering need hand_id and action_no in the frame even if the caller
        # projected them out; read them, then hand back exactly the requested columns.
        read_cols = list(projection)
        for helper in ("hand_id", "action_no"):
            if helper not in read_cols and helper in parquet_file.schema_arrow.names:
                read_cols.append(helper)

        hand_filter: Optional[Set[str]] = None
        if pool_id is not None:
            hand_filter = self._pool_hand_ids(pool_id)
            if not hand_filter:
                return  # unknown/empty pool -> no actions

        sort_keys = [c for c in ("hand_id", "action_no") if c in read_cols]
        for batch in parquet_file.iter_batches(batch_size=self.row_group_size, columns=read_cols):
            frame = batch.to_pandas()
            if hand_filter is not None:
                frame = frame[frame["hand_id"].isin(hand_filter)]
                if frame.empty:
                    continue
            if sort_keys:
                frame = frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
            # Return only the columns the caller asked for (drop helper columns).
            yield frame[projection]

    def iter_hand_actions(
        self,
        hand_id: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a single hand's actions ordered by ``(hand_id, action_no)``.

        Streams ``actions.parquet`` and collects only the rows for ``hand_id`` (the file
        is never loaded whole). Because a hand's rows are contiguous, at most a couple of
        batches contribute. The result is sorted by ``action_no``.

        Args:
            hand_id: The hex hand id to fetch.
            columns: Optional column projection override.

        Returns:
            A ``pd.DataFrame`` of that hand's actions in decision order (may be empty if
            the hand id is absent).
        """
        path = self._resolve_path("actions", _PARQUET_FILES["actions"])
        projection = self._projection("actions", columns)
        parquet_file = pq.ParquetFile(path)
        self._validate_columns(parquet_file.schema_arrow.names, projection, str(path))

        read_cols = list(projection)
        for helper in ("hand_id", "action_no"):
            if helper not in read_cols and helper in parquet_file.schema_arrow.names:
                read_cols.append(helper)

        collected: List[pd.DataFrame] = []
        for batch in parquet_file.iter_batches(batch_size=self.row_group_size, columns=read_cols):
            frame = batch.to_pandas()
            match = frame[frame["hand_id"] == hand_id]
            if not match.empty:
                collected.append(match)
        if not collected:
            return pd.DataFrame(columns=projection)
        result = pd.concat(collected, ignore_index=True)
        sort_keys = [c for c in ("hand_id", "action_no") if c in result.columns]
        if sort_keys:
            result = result.sort_values(sort_keys, kind="stable").reset_index(drop=True)
        return result[projection]

    # ------------------------------------------------------------------ #
    # Labels with explicit PU structure (Requirement 6.1, 6.2)
    # ------------------------------------------------------------------ #
    def load_labels(self) -> LabelTable:
        """Load development labels with an explicit Positive-Unlabelled structure.

        Returns a :class:`LabelTable` where:
          * ``trusted_positive`` maps each ``label_status == 'confirmed_target'`` pair id
            to its ``behavior_family`` (the target behavior).
          * ``confirmed_negative`` is the set of ``label_status == 'confirmed_non_target'``
            pair ids.
          * every development pair NOT listed is UNKNOWN (unlabelled) and is deliberately
            absent from both collections — it is never treated as negative (Requirement 6.1).

        A missing required column raises :class:`SchemaError`.
        """
        df = self._read_csv("development_labels", _REQUIRED_LABEL_COLUMNS)
        positives = df[df["label_status"] == LABEL_STATUS_POSITIVE]
        negatives = df[df["label_status"] == LABEL_STATUS_NEGATIVE]
        trusted_positive = {
            str(row.pair_id): str(row.behavior_family) for row in positives.itertuples(index=False)
        }
        confirmed_negative = {str(pid) for pid in negatives["pair_id"].tolist()}
        return LabelTable(
            trusted_positive=trusted_positive,
            confirmed_negative=confirmed_negative,
        )

    def load_labels_frame(self) -> pd.DataFrame:
        """Return the raw development-labels frame with the PU-relevant columns.

        Keeps ``pair_id, player_1, player_2, label, label_status, behavior_family`` so
        callers that need the full table (e.g. EDA/audit) get the hex ids and status
        without re-parsing. Unknown pairs are simply those absent from this file.
        """
        return self._read_csv("development_labels", _REQUIRED_LABEL_COLUMNS)[
            list(_REQUIRED_LABEL_COLUMNS)
        ].copy()

    # ------------------------------------------------------------------ #
    # Evaluation pairs + submission template (Requirements 1.1, 9.x)
    # ------------------------------------------------------------------ #
    def load_evaluation_pairs(self) -> pd.DataFrame:
        """Load ``evaluation_pairs.csv`` (``pair_id, player_1, player_2, shared_hands``)."""
        return self._read_csv("evaluation_pairs", _REQUIRED_EVAL_PAIR_COLUMNS)[
            list(_REQUIRED_EVAL_PAIR_COLUMNS)
        ].copy()

    def load_sample_submission(self) -> pd.DataFrame:
        """Load ``sample_submission.csv``, the submission template.

        Validates the exact expected schema/column order so downstream writers can rely
        on it (Requirement 9.2).
        """
        required = (
            "pair_id",
            "risk_score",
            "predicted_behavior",
            "evidence_hand_1",
            "evidence_hand_2",
            "evidence_hand_3",
            "evidence_hand_4",
            "evidence_hand_5",
        )
        return self._read_csv("sample_submission", required)

    # ------------------------------------------------------------------ #
    # Shared hands for a pair, partitioned dev/eval (Requirement 1.4)
    # ------------------------------------------------------------------ #
    def shared_hands(self, pair: Union[Pair, Sequence[str]]) -> SharedHands:
        """Return the hand ids shared by a pair's two players, partitioned dev/eval.

        Shared hands are derived from ``seats`` (both players seated in the same hand),
        then phase is attached from ``hands.phase`` (Requirement 1.3) and the ids are
        split into Development_Period and Evaluation_Period (Requirement 1.4). This is
        column-projected and never loads ``actions.parquet``.

        Args:
            pair: Either a :class:`Pair` (uses ``player_a``/``player_b``) or a
                2-sequence ``(player_1, player_2)`` of hex player ids. A ``pair_id`` is
                taken from ``Pair.pair_id`` when available, else synthesized.

        Returns:
            :class:`SharedHands` with development / evaluation hand-id lists (each sorted
            ascending for determinism) and a ``[hand_id, phase]`` frame.
        """
        if isinstance(pair, Pair):
            player_1, player_2 = str(pair.player_a), str(pair.player_b)
            pair_id = pair.pair_id
        else:
            seq = list(pair)
            if len(seq) != 2:
                raise ValueError("pair must be a Pair or a (player_1, player_2) sequence")
            player_1, player_2 = str(seq[0]), str(seq[1])
            pair_id = f"{player_1}|{player_2}"

        seats = self._seats_meta()
        hands_of_1 = set(seats.loc[seats["player_id"] == player_1, "hand_id"].tolist())
        hands_of_2 = set(seats.loc[seats["player_id"] == player_2, "hand_id"].tolist())
        shared = hands_of_1 & hands_of_2

        hands = self._hands_meta()
        shared_frame = (
            hands.loc[hands["hand_id"].isin(shared), ["hand_id", "phase"]]
            .sort_values("hand_id", kind="stable")
            .reset_index(drop=True)
        )
        development = shared_frame.loc[
            shared_frame["phase"] == PHASE_DEVELOPMENT, "hand_id"
        ].tolist()
        evaluation = shared_frame.loc[
            shared_frame["phase"] == PHASE_EVALUATION, "hand_id"
        ].tolist()
        return SharedHands(
            pair_id=pair_id,
            development=development,
            evaluation=evaluation,
            table=shared_frame,
        )
