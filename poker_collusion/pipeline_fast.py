"""Vectorized full-submission generator for the Phase-1 poker-collusion baseline.

:func:`run_baseline_pipeline_fast` produces the SAME Phase-1 ``submission.csv`` as
:func:`poker_collusion.pipeline.run_baseline_pipeline` but removes the profiled bottleneck:
the original calls :meth:`DataLoader.shared_hands` once per evaluation pair, and each call
rescans the full ~12M-row ``seats`` frame (≈311 ms/pair × 112,540 pairs ≈ 585 min).

Root-cause fix (the ONLY behavioral difference from the original is *how* shared eval hands
are resolved — the resolved sets are identical):

  1. Load ``seats[hand_id, player_id]`` and ``hands[hand_id, table_id, phase]`` once
     (column-projected). Restrict seats to EVALUATION-phase hands via a single hand→phase
     join, then build ``player_id -> frozenset(eval hand_id)`` in ONE groupby pass.
  2. For each eval pair, ``shared_eval_hands = players_eval_hands[p1] ∩ players_eval_hands[p2]``
     — a microsecond set intersection replacing the full-seats scan. This yields exactly the
     same eval-hand set (sorted ascending) that ``DataLoader.shared_hands(pair).evaluation``
     would return, so all downstream signal/detector/evidence math is bit-for-bit identical.
  3. Stream actions ONCE over the union of all shared eval hand ids
     (:func:`actions_for_hands_bulk` — a faster, output-equivalent form of
     :func:`poker_collusion.pipeline._actions_for_hands`), build ``net_chips`` per hand
     (:func:`~poker_collusion.pipeline._net_chips_by_hand`) and hands meta
     (:func:`~poker_collusion.pipeline._hands_meta_for`) — the exact same helpers the original
     uses, so those caches are identical.
  4. Per pair: :func:`compute_pair_signals` → :func:`build_pair_feature_set` (phase="evaluation")
     → :func:`score_pair` → :func:`select_evidence` → :func:`assign_behavior_label`, assembled
     into a :class:`PairPrediction`. Pairs with no shared eval hands get the documented
     pool-prior default row (risk = pool prior, behavior ``none``, all ``NO_EVIDENCE``),
     exactly as the original.
  5. :func:`write_submission` — same schema, deterministic, same row order (from
     ``sample_submission.csv``).

The per-pair signal/detector/feature work is retained UNCHANGED (it is cheap relative to the
seats scan and reusing it guarantees identical outputs). Correctness is proven by
``poker_collusion/tests/test_pipeline_fast.py`` which asserts byte-for-identical submissions
against :func:`run_baseline_pipeline` on the same synthetic dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from poker_collusion.config import NO_EVIDENCE, PipelineConfig, get_config
from poker_collusion.evidence.retriever import MAX_EVIDENCE, select_evidence
from poker_collusion.features.aggregation import prior_from_artifact
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.features.pair_features import build_pair_feature_set
from poker_collusion.features.signals import compute_pair_signals
from poker_collusion.io import DataLoader
from poker_collusion.io.data_loader import PHASE_EVALUATION
from poker_collusion.models.classical import score_pair
from poker_collusion.parallel_scoring import (
    score_pairs_parallel,
    score_pairs_sequential,
)
from poker_collusion.pipeline import (
    _canonical_players,
    _hands_meta_for,
    _net_chips_by_hand,
    _obtain_eda_artifact,
    assign_behavior_label,
)
from poker_collusion.submission.writer import write_submission
from poker_collusion.types import PairPrediction

__all__ = [
    "run_baseline_pipeline_fast",
    "build_eval_hands_by_player",
    "actions_for_hands_bulk",
    "build_rows_by_hand",
    "main",
]

#: Shared empty actions map: the per-pair signal path uses pre-parsed ``rows_by_hand`` for every
#: union eval hand, so it never reads an action FRAME; this constant avoids per-pair dict rebuilds.
_EMPTY_ACTIONS: Dict[str, "pd.DataFrame"] = {}


def _split_by_hand(df: pd.DataFrame, has_action_no: bool) -> Dict[str, pd.DataFrame]:
    """Split a filtered actions frame into ``hand_id -> per-hand frame`` sorted by ``action_no``.

    Sorts the whole frame ONCE by ``[hand_id, action_no]`` (stable) then slices contiguous
    same-``hand_id`` runs with a numpy boundary scan — far cheaper than a Python-level
    ``groupby`` that re-sorts and re-concatenates each of the ~800k groups. Each per-hand frame
    carries exactly the input columns with a fresh ``RangeIndex``, reproducing the shape and
    ordering that :func:`poker_collusion.pipeline._actions_for_hands` yields.
    """
    if len(df) == 0:
        return {}
    sort_keys = ["hand_id", "action_no"] if has_action_no else ["hand_id"]
    ordered = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    hid = ordered["hand_id"].astype(str).to_numpy()
    change = np.empty(len(hid), dtype=bool)
    change[0] = True
    if len(hid) > 1:
        change[1:] = hid[1:] != hid[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(hid))
    out: Dict[str, pd.DataFrame] = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        out[str(hid[s])] = ordered.iloc[s:e].reset_index(drop=True)
    return out


_ALLOWED_ROW_COLS = (
    "action_no",
    "player_id",
    "action",
    "amount",
    "amount_to",
    "to_call",
    "pot_before",
    "stack_before",
    "players_active",
    "street",
)


def _rows_by_hand_from_combined(
    df: pd.DataFrame,
    has_action_no: bool,
) -> Dict[str, List[dict]]:
    """Build ``hand_id -> [action-row dict]`` from a combined filtered actions frame in ONE pass.

    This is the vectorized equivalent of calling
    :func:`poker_collusion.features.action_rows` per hand. ``action_rows`` (a) keeps only the
    decision-time ``allowed`` columns present, (b) stable-sorts each hand's rows by ``action_no``,
    (c) returns ``DataFrame.to_dict("records")``. Doing that per hand means ~800k pandas
    ``to_dict`` calls plus ~800k per-hand ``sort_values`` — the profiled bottleneck once the
    per-pair signal build is de-duplicated across pairs.

    Here we instead:

      1. sort the WHOLE combined frame once by ``[hand_id, action_no]`` (stable) — within each
         contiguous ``hand_id`` block this reproduces ``action_rows``' per-hand ``action_no`` order;
      2. project to exactly the ``allowed`` columns present (same set/precedence as ``action_rows``);
      3. call ``to_dict("records")`` ONCE on the whole frame;
      4. slice that single record list into per-hand lists on ``hand_id`` boundaries (numpy scan).

    Each resulting per-hand list is element-for-element identical to ``action_rows(hand_frame)``
    (same rows, same order, same dict values), so signals built from it are BIT-IDENTICAL. Column
    ordering inside a record dict is irrelevant: the signal code reads rows only via ``.get(key)``.
    """
    if len(df) == 0:
        return {}
    sort_keys = ["hand_id", "action_no"] if has_action_no else ["hand_id"]
    ordered = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    hid = ordered["hand_id"].astype(str).to_numpy()
    # Project to the decision-time columns action_rows keeps, in the same precedence order.
    cols = [c for c in _ALLOWED_ROW_COLS if c in ordered.columns]
    records = ordered[cols].to_dict("records")  # ONE conversion for the whole union
    change = np.empty(len(hid), dtype=bool)
    change[0] = True
    if len(hid) > 1:
        change[1:] = hid[1:] != hid[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(hid))
    out: Dict[str, List[dict]] = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        out[str(hid[s])] = records[s:e]
    return out


def _scan_filtered_actions(
    loader: DataLoader,
    wanted: Set[str],
) -> tuple:
    """Single column-projected scan of ``actions.parquet`` filtered to ``wanted`` hand ids.

    Returns ``(combined_df, has_action_no)`` where ``combined_df`` is the concatenation of every
    kept row (unsorted; callers sort as needed) and ``has_action_no`` records whether the
    projection carries ``action_no``. Uses vectorized ``pyarrow.compute.is_in`` filtering on
    row-group batches; falls back to the loader's ``iter_actions`` stream (hermetic tests) with an
    equivalent pandas mask. This is the ONE pass over the 18.6M-row file (Req 1.6): the file is
    never materialised whole — only the kept eval-hand rows survive in memory.
    """
    projection = loader._projection("actions", None)
    has_action_no = "action_no" in projection

    kept_tables: List[pa.Table] = []
    try:
        path = loader._resolve_path("actions", "actions.parquet")
        parquet_file = pq.ParquetFile(path)
        loader._validate_columns(parquet_file.schema_arrow.names, projection, str(path))
        wanted_arr = pa.array(list(wanted), type=pa.large_string())
        batch_size = max(int(loader.row_group_size), 1)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=projection):
            table = pa.Table.from_batches([record_batch])
            hand_col = table.column("hand_id")
            if not (pa.types.is_string(hand_col.type) or pa.types.is_large_string(hand_col.type)):
                hand_col = pc.cast(hand_col, pa.large_string())
                table = table.set_column(
                    table.schema.get_field_index("hand_id"), "hand_id", hand_col
                )
            elif pa.types.is_string(hand_col.type):
                casted = pc.cast(hand_col, pa.large_string())
                table = table.set_column(
                    table.schema.get_field_index("hand_id"), "hand_id", casted
                )
            mask = pc.is_in(table.column("hand_id"), value_set=wanted_arr)
            filtered = table.filter(mask)
            if filtered.num_rows:
                kept_tables.append(filtered)
        if not kept_tables:
            return pd.DataFrame(), has_action_no
        return pa.concat_tables(kept_tables).to_pandas(), has_action_no
    except Exception:
        pass

    # Fallback: loader.iter_actions stream (hermetic path).
    kept_frames: List[pd.DataFrame] = []
    for batch in loader.iter_actions():
        if batch.empty or "hand_id" not in batch.columns:
            continue
        mask = batch["hand_id"].astype(str).isin(wanted)
        if not mask.any():
            continue
        keep = batch[mask.to_numpy()]
        if not keep.empty:
            kept_frames.append(keep)
    if not kept_frames:
        return pd.DataFrame(), has_action_no
    return pd.concat(kept_frames, ignore_index=True), has_action_no


def build_rows_by_hand(
    loader: DataLoader,
    hand_ids: Sequence[str],
) -> Dict[str, List[dict]]:
    """Return ``hand_id -> pre-parsed action rows`` for ``hand_ids`` in ONE scan + ONE conversion.

    Combines :func:`_scan_filtered_actions` (the single parquet pass) with
    :func:`_rows_by_hand_from_combined` (the single bulk ``to_dict`` + boundary slice) to produce
    exactly the ``rows_by_hand`` map the per-pair signal path consumes — element-for-element
    identical to calling :func:`poker_collusion.features.action_rows` on each hand's frame, but
    without the ~800k per-hand pandas ``to_dict``/``sort_values`` calls. Feeding this map to
    :func:`~poker_collusion.features.signals.compute_pair_signals` yields BIT-IDENTICAL signals.
    """
    wanted = set(str(h) for h in hand_ids)
    if not wanted:
        return {}
    df, has_action_no = _scan_filtered_actions(loader, wanted)
    return _rows_by_hand_from_combined(df, has_action_no)


def actions_for_hands_bulk(
    loader: DataLoader,
    hand_ids: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Return ``hand_id -> ordered action frame`` for ``hand_ids`` — a faster, EQUIVALENT form of
    :func:`poker_collusion.pipeline._actions_for_hands`.

    The original streams ``actions.parquet`` once but, per 100k-row batch, does a pandas
    ``.astype(str).isin(wanted)`` against an ~800k-id set AND a stable ``(hand_id, action_no)``
    re-sort, then appends per-hand slices into lists and ``pd.concat``-s each of the ~800k hands
    (≈9+ min on the real data → the new bottleneck once the per-pair seats scan is removed).

    This version keeps the SAME single scan and the SAME filtering semantics but pushes the hot
    work down to Arrow / numpy:

      1. Column-projected scan of ``actions.parquet`` via ``pyarrow`` row-group batches.
      2. Per batch, filter with ``pyarrow.compute.is_in`` (vectorized C, ~11s total) — no pandas
         ``astype``/``isin`` and no per-batch sort.
      3. Concatenate the kept Arrow tables once, convert to pandas once, then split into per-hand
         frames with a single stable sort + contiguous-run slice (:func:`_split_by_hand`).

    The result is identical to :func:`_actions_for_hands`: a dict mapping each present hand id
    (as ``str``) to that hand's action frame sorted by ``action_no`` (stable) with exactly the
    projected columns. If the direct parquet scan is unavailable for any reason, it falls back to
    the loader's ``iter_actions`` stream (used by the hermetic tests) with the same split. Proven
    equal by the byte-identical submission equality test.

    Args:
        loader: The :class:`DataLoader` (supplies the resolved parquet path + column projection).
        hand_ids: The evaluation hand ids to collect (the union across all pairs).

    Returns:
        ``{hand_id -> action DataFrame}`` sorted per hand by ``action_no``.
    """
    wanted = set(str(h) for h in hand_ids)
    if not wanted:
        return {}

    projection = loader._projection("actions", None)  # same columns iter_actions would yield
    has_action_no = "action_no" in projection

    kept_tables: List[pa.Table] = []
    used_arrow = False
    try:
        path = loader._resolve_path("actions", "actions.parquet")
        parquet_file = pq.ParquetFile(path)
        # Validate the projection exists (mirrors iter_actions' schema check).
        loader._validate_columns(
            parquet_file.schema_arrow.names, projection, str(path)
        )
        wanted_arr = pa.array(list(wanted), type=pa.large_string())
        batch_size = max(int(loader.row_group_size), 1)
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=projection
        ):
            table = pa.Table.from_batches([record_batch])
            hand_col = table.column("hand_id")
            # Cast to large_string only if needed so is_in's value_set type matches.
            if not (pa.types.is_string(hand_col.type) or pa.types.is_large_string(hand_col.type)):
                hand_col = pc.cast(hand_col, pa.large_string())
                table = table.set_column(
                    table.schema.get_field_index("hand_id"), "hand_id", hand_col
                )
            elif pa.types.is_string(hand_col.type):
                # is_in needs matching value-set type; cast the (small) string column up.
                casted = pc.cast(hand_col, pa.large_string())
                table = table.set_column(
                    table.schema.get_field_index("hand_id"), "hand_id", casted
                )
            mask = pc.is_in(table.column("hand_id"), value_set=wanted_arr)
            filtered = table.filter(mask)
            if filtered.num_rows:
                kept_tables.append(filtered)
        used_arrow = True
    except Exception:
        used_arrow = False

    if used_arrow:
        if not kept_tables:
            return {}
        combined = pa.concat_tables(kept_tables)
        df = combined.to_pandas()
        return _split_by_hand(df, has_action_no)

    # --- Fallback: loader.iter_actions stream (hermetic path) + single split ------------------ #
    kept_frames: List[pd.DataFrame] = []
    for batch in loader.iter_actions():
        if batch.empty or "hand_id" not in batch.columns:
            continue
        mask = batch["hand_id"].astype(str).isin(wanted)
        if not mask.any():
            continue
        keep = batch[mask.to_numpy()]
        if not keep.empty:
            kept_frames.append(keep)
    if not kept_frames:
        return {}
    df = pd.concat(kept_frames, ignore_index=True)
    return _split_by_hand(df, has_action_no)


def build_eval_hands_by_player(
    seats: pd.DataFrame,
    hands: pd.DataFrame,
) -> Dict[str, FrozenSet[str]]:
    """Build ``player_id -> frozenset(evaluation-phase hand_id)`` in ONE pass.

    This is the vectorized replacement for the per-pair full-seats scan performed by
    :meth:`DataLoader.shared_hands`. Steps:

      1. Resolve the set of evaluation-phase hand ids from ``hands`` (``phase == "evaluation"``).
      2. Filter ``seats`` to only those hands (one ``isin`` pass over the seats frame).
      3. Group the surviving ``(player_id -> hand_id)`` rows into per-player frozensets.

    The result is exact: for any pair ``(p1, p2)`` the intersection of their frozensets is the
    same eval-hand set that ``DataLoader.shared_hands((p1, p2)).evaluation`` produces (both are
    "hands where both players are seated AND the hand is evaluation-phase").

    Args:
        seats: Column-projected seats frame with at least ``hand_id`` and ``player_id``.
        hands: Column-projected hands frame with at least ``hand_id`` and ``phase``.

    Returns:
        A dict mapping each player id (as ``str``) to the frozenset of evaluation-phase hand ids
        (as ``str``) that player is seated in. Players never seated in an eval hand are absent.
    """
    if hands is None or len(hands) == 0 or seats is None or len(seats) == 0:
        return {}

    eval_hand_ids: Set[str] = set(
        hands.loc[hands["phase"] == PHASE_EVALUATION, "hand_id"].astype(str).tolist()
    )
    if not eval_hand_ids:
        return {}

    seat_hand = seats["hand_id"].astype(str)
    mask = seat_hand.isin(eval_hand_ids)
    if not mask.any():
        return {}

    sub = pd.DataFrame(
        {
            "player_id": seats["player_id"].astype(str)[mask].to_numpy(),
            "hand_id": seat_hand[mask].to_numpy(),
        }
    )

    out: Dict[str, FrozenSet[str]] = {}
    for player_id, grp in sub.groupby("player_id", sort=False):
        out[str(player_id)] = frozenset(grp["hand_id"].tolist())
    return out


def run_baseline_pipeline_fast(
    config: Optional[PipelineConfig] = None,
    *,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    limit_pairs: Optional[int] = None,
    parallel: bool = False,
    n_workers: Optional[int] = None,
) -> Path:
    """Run the Phase-1 classical baseline end-to-end, vectorizing the shared-hand resolution.

    Produces a ``submission.csv`` IDENTICAL to
    :func:`poker_collusion.pipeline.run_baseline_pipeline` (same row set; per-pair
    ``risk_score`` within 1e-9, exact ``predicted_behavior`` / evidence strings) but resolves
    every pair's evaluation-period shared hands from a single precomputed
    ``player_id -> eval-hand-set`` map instead of rescanning the full seats frame per pair.

    Args mirror :func:`run_baseline_pipeline`, plus:
        parallel: Route the independent per-pair loop through the deterministic multiprocess
            driver (:func:`poker_collusion.parallel_scoring.score_pairs_parallel`). The result is
            BYTE-IDENTICAL to the sequential (default) path — chunks are contiguous pair slices,
            so order is preserved and the same per-pair functions are called. Default ``False``
            keeps the existing sequential behavior and all existing tests unchanged.
        n_workers: Worker count when ``parallel=True`` (``None`` -> ``min(cpu_count, 8)``).

    Returns:
        The :class:`~pathlib.Path` the ``submission.csv`` was written to.
    """
    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)

    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)
    pool_prior = prior_from_artifact(artifact, default=0.0)

    evaluation_pairs = loader.load_evaluation_pairs()
    sample_submission = loader.load_sample_submission()
    seats = loader.load_seats()
    hands = loader.load_hands()

    if limit_pairs is not None:
        evaluation_pairs = evaluation_pairs.head(int(limit_pairs)).copy()

    # ---- (1) PRECOMPUTE eval-hand sets for ALL players ONCE (no per-pair seats scan) --------- #
    eval_hands_by_player = build_eval_hands_by_player(seats, hands)

    # ---- (2) resolve each pair's eval shared hands by cheap set intersection ----------------- #
    pair_rows: List[dict] = []
    all_eval_hand_ids: Set[str] = set()
    for row in evaluation_pairs.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        player_a, player_b = _canonical_players(
            getattr(row, "player_1"), getattr(row, "player_2")
        )
        set_a = eval_hands_by_player.get(player_a, frozenset())
        set_b = eval_hands_by_player.get(player_b, frozenset())
        # Sorted ascending to match DataLoader.shared_hands(...).evaluation ordering exactly.
        eval_hands = sorted(set_a & set_b)
        all_eval_hand_ids.update(eval_hands)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "player_a": player_a,
                "player_b": player_b,
                "eval_hands": eval_hands,
            }
        )

    # ---- (3) ONE streaming pass over actions + shared caches (identical to the original) ----- #
    ordered_eval_hand_ids = sorted(all_eval_hand_ids)
    # Parse the action log for EVERY union eval hand exactly once into row-dicts (single scan +
    # single bulk to_dict). This is the pre-parsed form ``compute_pair_signals`` consumes via
    # ``rows_by_hand``; building it here (instead of per-pair, per-hand) removes the ~12x
    # redundant per-hand parsing that dominated the loop, while staying bit-identical.
    rows_by_hand = build_rows_by_hand(loader, ordered_eval_hand_ids)
    net_by_hand = _net_chips_by_hand(seats, ordered_eval_hand_ids)
    eval_hands_meta = _hands_meta_for(hands, ordered_eval_hand_ids)

    # Build a per-hand meta record dict ONCE so each pair's tiny meta frame is assembled from just
    # its own ~handful of hands. The original code re-filtered the whole (~800k-row) meta frame
    # per pair (``meta["hand_id"].astype(str).isin(...)``), which is O(pairs x union) and becomes
    # the dominant cost at full scale. ``compute_pair_signals`` sorts the meta by ``hand_id`` and
    # reads only ``hand_id``/``phase``/``big_blind``, so a small per-pair frame is equivalent.
    _meta_cols = list(eval_hands_meta.columns)
    meta_record_by_hand: Dict[str, dict] = {
        str(getattr(r, "hand_id")): {c: getattr(r, c) for c in _meta_cols}
        for r in eval_hands_meta.itertuples(index=False)
    }



    # ---- (4) per-pair: signals -> features -> risk -> evidence -> PairPrediction ------------- #
    # The independent per-pair work runs through the shared driver: sequential by default, or the
    # deterministic multiprocess path when ``parallel=True``. Both call the SAME per-pair functions
    # and preserve the ``pair_rows`` order, so the emitted submission is byte-identical either way.
    if parallel:
        scored = score_pairs_parallel(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=_meta_cols,
            artifact=artifact,
            config=cfg,
            n_workers=n_workers,
        )
    else:
        scored = score_pairs_sequential(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=_meta_cols,
            artifact=artifact,
            config=cfg,
        )

    predictions: List[PairPrediction] = []
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        res = scored.get(pair_id)
        if res is None:
            # No evaluation-period shared hands -> documented pool-prior default row.
            predictions.append(
                PairPrediction(
                    pair_id=pair_id,
                    risk_score=float(min(max(pool_prior, 0.0), 1.0)),
                    predicted_behavior="none",
                    evidence=[NO_EVIDENCE] * MAX_EVIDENCE,
                    reasons=["no evaluation-period shared hands -> pool-prior default row"],
                )
            )
            continue

        predictions.append(
            PairPrediction(
                pair_id=pair_id,
                risk_score=float(res.classical_risk),
                predicted_behavior=res.phase1_behavior,
                evidence=list(res.evidence_slots),
                reasons=list(res.classical_reasons),
            )
        )

    # ---- (5) route into the Submission_Writer (self-validating, atomic, deterministic) ------- #
    return write_submission(
        predictions,
        sample_submission=sample_submission,
        config=cfg,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m poker_collusion.pipeline_fast``.

    Mirrors :func:`poker_collusion.pipeline.main` but runs the vectorized generator.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m poker_collusion.pipeline_fast",
        description="Run the vectorized Phase-1 poker-collusion baseline end-to-end.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Smoke run: process only the first N evaluation pairs (rest get the default row).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Route the per-pair loop through the deterministic multiprocess driver "
        "(byte-identical to the sequential path).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker count when --parallel is set (default: min(cpu_count, 8)).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = run_baseline_pipeline_fast(
        limit_pairs=args.limit_pairs,
        parallel=args.parallel,
        n_workers=args.n_workers,
    )
    print(f"submission written to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
