"""Ordered street/action sequence ablation on the full LamHuy V30 frame.

BUILD + MEASURE ONLY. The module derives deterministic, target-free sequence motifs
from the prepared 18.6M-row action stream, appends one pair-level feature block to the
persisted full V30 frames, and evaluates it on the exact table-disjoint outer folds
saved by ``lamhuy_hard_negative``. The recovered V30 and HNM artifacts are read-only.
A risk-only submission candidate is emitted only when the pre-registered AP gates pass.

Run from the Poker workspace root::

    python -m anchor_repro.lamhuy_ordered_sequence
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

from anchor_repro.graph_retest import _probe_cuda
from anchor_repro.lamhuy_hard_negative import (
    BLEND_WEIGHTS,
    CB_PARAMS,
    EXPECTED_DEV_ROWS,
    EXPECTED_EVAL_ROWS,
    LGB_PARAMS,
    N_SPLITS,
    PROMOTION_BASELINE_AP,
    PROMOTION_CONFIRMED_DELTA,
    PROMOTION_STRESS_DELTA_FLOOR,
    SEED,
    _candidate_source,
    _emit_candidate,
    _fold_metrics,
    _json_ready,
    _load_full_frames,
    _prepare_training_data,
    _rank_comparison,
    _sha256,
    _xgb_params,
)
from anchor_repro.lamhuy_recipe import default_lamhuy_paths

RECIPE_ID = "lamhuy_v30_ordered_action_sequences_v1"
STATE_COLUMNS = [
    "phase",
    "hand_id",
    "action_no",
    "street",
    "player_id",
    "action",
    "to_call",
    "players_active",
    "is_aggressive",
    "last_aggressor",
    "amount_bb",
    "to_call_bb",
    "amount_pot_ratio",
    "_prev_player",
    "_prev_action",
    "_prev_street",
    "_prev_is_aggressive",
    "_prev_action_no",
    "_next_player",
    "_next_action",
    "_next_street",
    "_next_is_aggressive",
    "_last_agg_no",
]
STREETS = ("preflop", "flop", "turn", "river")


def _write_run_files(out_dir: Path, summary: dict, log_lines: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_ordered_sequence_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "_ordered_sequence.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )


def _canonical_pair_exprs(left: str, right: str) -> List[pl.Expr]:
    """Canonical string pair without relying on pair_id internals."""
    return [
        pl.when(pl.col(left) <= pl.col(right))
        .then(pl.col(left))
        .otherwise(pl.col(right))
        .alias("p_low"),
        pl.when(pl.col(left) <= pl.col(right))
        .then(pl.col(right))
        .otherwise(pl.col(left))
        .alias("p_high"),
    ]


def _build_ordered_state(paths, state_path: Path, log: Callable[[str], None]) -> dict:
    """Materialize chronology fields once so independent motif scans stay cheap."""
    if state_path.is_file():
        schema = pl.scan_parquet(state_path).collect_schema()
        missing = [column for column in STATE_COLUMNS if column not in schema]
        if missing:
            raise ValueError(f"Ordered-state cache is missing columns: {missing}")
        log(f"Reusing ordered-state cache: {state_path}")
        return {
            "built": False,
            "path": str(state_path),
            "size_bytes": state_path.stat().st_size,
        }

    action_path = paths.prepared_dir / "action_context.parquet"
    hands_path = paths.data_dir / "hands.parquet"
    if not action_path.is_file() or not hands_path.is_file():
        raise FileNotFoundError(f"Missing action inputs: {action_path}, {hands_path}")

    log("Building explicitly sorted action-state cache (hand_id, action_no)...")
    started = time.time()
    ordered = (
        pl.scan_parquet(action_path)
        .sort(["hand_id", "action_no"])
        .with_columns(
            pl.when(pl.col("is_aggressive"))
            .then(pl.col("action_no"))
            .otherwise(None)
            .alias("_agg_no")
        )
        .with_columns(
            pl.col("player_id").shift(1).over("hand_id").alias("_prev_player"),
            pl.col("action").shift(1).over("hand_id").alias("_prev_action"),
            pl.col("street").shift(1).over("hand_id").alias("_prev_street"),
            pl.col("is_aggressive")
            .shift(1)
            .over("hand_id")
            .fill_null(False)
            .alias("_prev_is_aggressive"),
            pl.col("action_no").shift(1).over("hand_id").alias("_prev_action_no"),
            pl.col("player_id").shift(-1).over("hand_id").alias("_next_player"),
            pl.col("action").shift(-1).over("hand_id").alias("_next_action"),
            pl.col("street").shift(-1).over("hand_id").alias("_next_street"),
            pl.col("is_aggressive")
            .shift(-1)
            .over("hand_id")
            .fill_null(False)
            .alias("_next_is_aggressive"),
            pl.col("_agg_no")
            .shift(1)
            .forward_fill()
            .over(["hand_id", "street"])
            .alias("_last_agg_no"),
        )
        .join(
            pl.scan_parquet(hands_path).select(["hand_id", "phase"]),
            on="hand_id",
            how="left",
        )
        .select(STATE_COLUMNS)
    )
    ordered.sink_parquet(state_path, compression="zstd")
    elapsed = time.time() - started

    audit = (
        pl.scan_parquet(state_path)
        .select(
            pl.len().alias("rows"),
            pl.col("hand_id").n_unique().alias("hands"),
            pl.col("phase").null_count().alias("missing_phase"),
            (
                pl.col("last_aggressor").is_not_null()
                & pl.col("_last_agg_no").is_null()
            )
            .sum()
            .alias("missing_last_agg_no"),
        )
        .collect()
        .row(0, named=True)
    )
    if audit["missing_phase"] or audit["missing_last_agg_no"]:
        raise RuntimeError(f"Ordered-state audit failed: {audit}")
    log(
        f"Ordered-state cache built in {elapsed:.1f}s: "
        f"{audit['rows']:,} actions across {audit['hands']:,} hands."
    )
    return {
        "built": True,
        "path": str(state_path),
        "size_bytes": state_path.stat().st_size,
        "duration_s": elapsed,
        **audit,
    }


def _response_stats(state_path: Path) -> pl.LazyFrame:
    state = pl.scan_parquet(state_path)
    response = (
        state.filter(
            pl.col("last_aggressor").is_not_null()
            & (pl.col("last_aggressor") != pl.col("player_id"))
            & (pl.col("to_call") > 0)
            & pl.col("_last_agg_no").is_not_null()
        )
        .with_columns(
            *_canonical_pair_exprs("last_aggressor", "player_id"),
            (pl.col("action_no") - pl.col("_last_agg_no"))
            .clip(1, 50)
            .alias("_gap"),
            (pl.col("action") == "fold").alias("_fold"),
            (
                pl.col("action").is_in(["call", "all_in"])
                & ~pl.col("is_aggressive")
            ).alias("_call"),
            pl.col("is_aggressive").alias("_raise"),
        )
        .with_columns(
            (pl.col("last_aggressor") == pl.col("p_low")).alias("_forward")
        )
    )
    agg: List[pl.Expr] = [
        pl.len().alias("seq_resp_n"),
        pl.col("hand_id").n_unique().alias("seq_resp_hands"),
        pl.col("_fold").sum().alias("seq_resp_fold_n"),
        pl.col("_call").sum().alias("seq_resp_call_n"),
        pl.col("_raise").sum().alias("seq_resp_raise_n"),
        (pl.col("_gap") == 1).sum().alias("seq_resp_direct_n"),
        (pl.col("_gap") > 1).sum().alias("seq_resp_delayed_n"),
        (pl.col("_fold") & (pl.col("_gap") == 1)).sum().alias("seq_resp_fold_direct_n"),
        (pl.col("_fold") & (pl.col("_gap") > 1)).sum().alias("seq_resp_fold_delayed_n"),
        (pl.col("_call") & (pl.col("_gap") == 1)).sum().alias("seq_resp_call_direct_n"),
        (pl.col("_call") & (pl.col("_gap") > 1)).sum().alias("seq_resp_call_delayed_n"),
        (pl.col("_raise") & (pl.col("_gap") == 1)).sum().alias("seq_resp_raise_direct_n"),
        (pl.col("_raise") & (pl.col("_gap") > 1)).sum().alias("seq_resp_raise_delayed_n"),
        (pl.col("_fold") & (pl.col("players_active") == 2)).sum().alias("seq_resp_fold_hu_n"),
        (pl.col("_fold") & (pl.col("players_active") > 2)).sum().alias("seq_resp_fold_mw_n"),
        (pl.col("_call") & (pl.col("players_active") == 2)).sum().alias("seq_resp_call_hu_n"),
        (pl.col("_raise") & (pl.col("players_active") == 2)).sum().alias("seq_resp_raise_hu_n"),
        pl.col("_gap").mean().alias("seq_resp_gap_mean"),
        pl.col("_gap").max().alias("seq_resp_gap_max"),
        pl.col("_gap").quantile(0.95, interpolation="nearest").alias("seq_resp_gap_p95"),
        pl.when(pl.col("_fold")).then(pl.col("_gap")).otherwise(None).mean().alias("seq_resp_fold_gap_mean"),
        pl.when(pl.col("_fold")).then(pl.col("_gap")).otherwise(None).max().alias("seq_resp_fold_gap_max"),
        pl.when(pl.col("_fold")).then(pl.col("to_call_bb")).otherwise(None).mean().alias("seq_resp_fold_to_call_mean"),
        pl.when(pl.col("_fold")).then(pl.col("to_call_bb")).otherwise(None).max().alias("seq_resp_fold_to_call_max"),
        pl.when(pl.col("_fold")).then(pl.col("to_call_bb")).otherwise(None).quantile(0.95, interpolation="nearest").alias("seq_resp_fold_to_call_p95"),
        pl.when(pl.col("_call")).then(pl.col("to_call_bb")).otherwise(None).max().alias("seq_resp_call_to_call_max"),
        pl.when(pl.col("_raise")).then(pl.col("amount_pot_ratio")).otherwise(None).max().alias("seq_resp_raise_pot_ratio_max"),
        (pl.col("_fold") & pl.col("_forward")).sum().alias("seq_resp_fold_forward_n"),
        (pl.col("_fold") & ~pl.col("_forward")).sum().alias("seq_resp_fold_reverse_n"),
        (pl.col("_call") & pl.col("_forward")).sum().alias("seq_resp_call_forward_n"),
        (pl.col("_call") & ~pl.col("_forward")).sum().alias("seq_resp_call_reverse_n"),
        (pl.col("_raise") & pl.col("_forward")).sum().alias("seq_resp_raise_forward_n"),
        (pl.col("_raise") & ~pl.col("_forward")).sum().alias("seq_resp_raise_reverse_n"),
    ]
    for street in STREETS:
        tag = "pre" if street == "preflop" else street
        agg.extend(
            [
                (pl.col("_fold") & (pl.col("street") == street)).sum().alias(f"seq_resp_fold_{tag}_n"),
                (pl.col("_call") & (pl.col("street") == street)).sum().alias(f"seq_resp_call_{tag}_n"),
                (pl.col("_raise") & (pl.col("street") == street)).sum().alias(f"seq_resp_raise_{tag}_n"),
            ]
        )
    return response.group_by(["phase", "p_low", "p_high"]).agg(agg)


def _adjacency_stats(state_path: Path) -> pl.LazyFrame:
    adjacent = (
        pl.scan_parquet(state_path)
        .filter(
            pl.col("_prev_player").is_not_null()
            & (pl.col("_prev_player") != pl.col("player_id"))
        )
        .with_columns(
            *_canonical_pair_exprs("_prev_player", "player_id"),
            (pl.col("street") == pl.col("_prev_street")).alias("_same_street"),
            (
                (pl.col("street") == pl.col("_prev_street"))
                & (pl.col("street") != "preflop")
            ).alias("_postflop"),
        )
        .with_columns(
            (
                pl.col("_postflop")
                & (pl.col("_prev_action") == "check")
                & (pl.col("action") == "check")
            ).alias("_check_check"),
            (
                pl.col("_postflop")
                & pl.col("_prev_action").is_in(["check", "call"])
                & pl.col("action").is_in(["check", "call"])
            ).alias("_passive_chain"),
            (
                pl.col("_same_street")
                & pl.col("_prev_is_aggressive")
                & (pl.col("action") == "fold")
            ).alias("_agg_fold"),
            (
                pl.col("_same_street")
                & pl.col("_prev_is_aggressive")
                & pl.col("action").is_in(["call", "all_in"])
                & ~pl.col("is_aggressive")
            ).alias("_agg_call"),
            (
                pl.col("_same_street")
                & pl.col("_prev_is_aggressive")
                & pl.col("is_aggressive")
            ).alias("_agg_reagg"),
            (
                (pl.col("street") != pl.col("_prev_street"))
                & pl.col("_prev_action").is_in(["check", "call"])
                & (pl.col("action") == "check")
            ).alias("_cross_street_passive"),
            (pl.col("_prev_player") == pl.col("p_low")).alias("_forward"),
        )
    )
    return adjacent.group_by(["phase", "p_low", "p_high"]).agg(
        pl.len().alias("seq_adj_n"),
        pl.col("hand_id").n_unique().alias("seq_adj_hands"),
        pl.col("_same_street").sum().alias("seq_adj_same_street_n"),
        pl.col("_postflop").sum().alias("seq_adj_postflop_n"),
        pl.col("_check_check").sum().alias("seq_adj_check_check_n"),
        pl.col("_passive_chain").sum().alias("seq_adj_passive_chain_n"),
        pl.col("_agg_fold").sum().alias("seq_adj_agg_fold_n"),
        pl.col("_agg_call").sum().alias("seq_adj_agg_call_n"),
        pl.col("_agg_reagg").sum().alias("seq_adj_agg_reagg_n"),
        pl.col("_cross_street_passive").sum().alias("seq_adj_cross_street_passive_n"),
        (pl.col("_check_check") & (pl.col("players_active") == 2)).sum().alias("seq_adj_check_check_hu_n"),
        (pl.col("_check_check") & pl.col("street").is_in(["turn", "river"])).sum().alias("seq_adj_check_check_late_n"),
        (pl.col("_check_check") & pl.col("_forward")).sum().alias("seq_adj_check_check_forward_n"),
        (pl.col("_check_check") & ~pl.col("_forward")).sum().alias("seq_adj_check_check_reverse_n"),
        pl.when(pl.col("_agg_fold")).then(pl.col("to_call_bb")).otherwise(None).max().alias("seq_adj_agg_fold_to_call_max"),
        pl.when(pl.col("_agg_fold")).then(pl.col("to_call_bb")).otherwise(None).mean().alias("seq_adj_agg_fold_to_call_mean"),
    )


def _sandwich_stats(state_path: Path) -> pl.LazyFrame:
    """Aggressor -> outsider response -> candidate partner's next action motifs."""
    sandwich = (
        pl.scan_parquet(state_path)
        .filter(
            pl.col("last_aggressor").is_not_null()
            & pl.col("_next_player").is_not_null()
            & (pl.col("_next_street") == pl.col("street"))
            & (pl.col("last_aggressor") != pl.col("player_id"))
            & (pl.col("_next_player") != pl.col("player_id"))
            & (pl.col("_next_player") != pl.col("last_aggressor"))
            & pl.col("_last_agg_no").is_not_null()
        )
        .with_columns(
            *_canonical_pair_exprs("last_aggressor", "_next_player"),
            (pl.col("action") == "fold").alias("_outsider_fold"),
            (
                pl.col("action").is_in(["call", "all_in"])
                & ~pl.col("is_aggressive")
            ).alias("_outsider_call"),
            pl.col("is_aggressive").alias("_outsider_raise"),
            pl.col("_next_action").is_in(["check", "call"]).alias("_partner_passive"),
            (pl.col("_next_action") == "fold").alias("_partner_fold"),
            pl.col("_next_is_aggressive").alias("_partner_raise"),
        )
        .with_columns(
            (pl.col("last_aggressor") == pl.col("p_low")).alias("_forward")
        )
    )
    return sandwich.group_by(["phase", "p_low", "p_high"]).agg(
        pl.len().alias("seq_sand_n"),
        pl.col("hand_id").n_unique().alias("seq_sand_hands"),
        pl.col("_outsider_fold").sum().alias("seq_sand_outsider_fold_n"),
        pl.col("_outsider_call").sum().alias("seq_sand_outsider_call_n"),
        pl.col("_outsider_raise").sum().alias("seq_sand_outsider_raise_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_passive")).sum().alias("seq_sand_fold_then_partner_passive_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_fold")).sum().alias("seq_sand_fold_then_partner_fold_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_raise")).sum().alias("seq_sand_fold_then_partner_raise_n"),
        (pl.col("_outsider_call") & pl.col("_partner_passive")).sum().alias("seq_sand_call_then_partner_passive_n"),
        (pl.col("_outsider_call") & pl.col("_partner_fold")).sum().alias("seq_sand_call_then_partner_fold_n"),
        (pl.col("_outsider_call") & pl.col("_partner_raise")).sum().alias("seq_sand_call_then_partner_raise_n"),
        (pl.col("_outsider_raise") & pl.col("_partner_passive")).sum().alias("seq_sand_raise_then_partner_passive_n"),
        (pl.col("_outsider_raise") & pl.col("_partner_fold")).sum().alias("seq_sand_raise_then_partner_fold_n"),
        (pl.col("_outsider_raise") & pl.col("_partner_raise")).sum().alias("seq_sand_raise_then_partner_raise_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_passive") & pl.col("street").is_in(["flop", "turn", "river"])).sum().alias("seq_sand_iso_yield_post_n"),
        (pl.col("_outsider_raise") & (pl.col("_partner_passive") | pl.col("_partner_fold"))).sum().alias("seq_sand_squeeze_surrender_n"),
        (pl.col("_outsider_fold") & pl.col("_forward")).sum().alias("seq_sand_outsider_fold_forward_n"),
        (pl.col("_outsider_fold") & ~pl.col("_forward")).sum().alias("seq_sand_outsider_fold_reverse_n"),
        pl.when(pl.col("_outsider_fold")).then(pl.col("to_call_bb")).otherwise(None).mean().alias("seq_sand_outsider_fold_to_call_mean"),
        pl.when(pl.col("_outsider_fold")).then(pl.col("to_call_bb")).otherwise(None).max().alias("seq_sand_outsider_fold_to_call_max"),
        pl.when(pl.col("_outsider_raise")).then(pl.col("amount_pot_ratio")).otherwise(None).max().alias("seq_sand_outsider_raise_pot_ratio_max"),
    )


def _add_derived_sequence_features(frame: pl.DataFrame) -> pl.DataFrame:
    id_columns = {"pair_id", "player_1", "player_2", "phase", "table_id"}
    numeric = [
        column
        for column, dtype in frame.schema.items()
        if dtype.is_numeric() and column not in id_columns
    ]
    frame = frame.with_columns([pl.col(column).fill_null(0) for column in numeric])
    eps = 1e-3
    frame = frame.with_columns(
        (pl.col("seq_resp_fold_n") / (pl.col("seq_resp_n") + eps)).alias("seq_resp_fold_rate"),
        (pl.col("seq_resp_call_n") / (pl.col("seq_resp_n") + eps)).alias("seq_resp_call_rate"),
        (pl.col("seq_resp_raise_n") / (pl.col("seq_resp_n") + eps)).alias("seq_resp_raise_rate"),
        (pl.col("seq_resp_delayed_n") / (pl.col("seq_resp_n") + eps)).alias("seq_resp_delayed_rate"),
        (pl.col("seq_resp_fold_delayed_n") / (pl.col("seq_resp_fold_n") + eps)).alias("seq_resp_fold_delayed_rate"),
        (pl.col("seq_resp_fold_n") / (pl.col("shared_hands") + eps)).alias("seq_resp_folds_per_hand"),
        (pl.col("seq_resp_hands") / (pl.col("shared_hands") + eps)).alias("seq_resp_support_rate"),
        ((pl.col("seq_resp_fold_forward_n") - pl.col("seq_resp_fold_reverse_n")).abs() / (pl.col("seq_resp_fold_n") + eps)).alias("seq_resp_fold_direction_imbalance"),
        ((pl.col("seq_resp_call_forward_n") - pl.col("seq_resp_call_reverse_n")).abs() / (pl.col("seq_resp_call_n") + eps)).alias("seq_resp_call_direction_imbalance"),
        ((pl.col("seq_resp_raise_forward_n") - pl.col("seq_resp_raise_reverse_n")).abs() / (pl.col("seq_resp_raise_n") + eps)).alias("seq_resp_raise_direction_imbalance"),
        (pl.col("seq_adj_check_check_n") / (pl.col("seq_adj_postflop_n") + eps)).alias("seq_adj_check_check_rate"),
        (pl.col("seq_adj_passive_chain_n") / (pl.col("seq_adj_postflop_n") + eps)).alias("seq_adj_passive_chain_rate"),
        (pl.col("seq_adj_agg_fold_n") / (pl.col("seq_adj_same_street_n") + eps)).alias("seq_adj_agg_fold_rate"),
        (pl.col("seq_adj_agg_reagg_n") / (pl.col("seq_adj_same_street_n") + eps)).alias("seq_adj_agg_reagg_rate"),
        (pl.col("seq_adj_check_check_n") / (pl.col("shared_hands") + eps)).alias("seq_adj_check_checks_per_hand"),
        (pl.col("seq_adj_cross_street_passive_n") / (pl.col("shared_hands") + eps)).alias("seq_adj_cross_street_passive_per_hand"),
        ((pl.col("seq_adj_check_check_forward_n") - pl.col("seq_adj_check_check_reverse_n")).abs() / (pl.col("seq_adj_check_check_n") + eps)).alias("seq_adj_check_direction_imbalance"),
        (pl.col("seq_sand_outsider_fold_n") / (pl.col("seq_sand_n") + eps)).alias("seq_sand_outsider_fold_rate"),
        (pl.col("seq_sand_fold_then_partner_passive_n") / (pl.col("seq_sand_outsider_fold_n") + eps)).alias("seq_sand_fold_then_partner_passive_rate"),
        (pl.col("seq_sand_squeeze_surrender_n") / (pl.col("seq_sand_outsider_raise_n") + eps)).alias("seq_sand_squeeze_surrender_rate"),
        (pl.col("seq_sand_iso_yield_post_n") / (pl.col("shared_hands") + eps)).alias("seq_sand_iso_yield_post_per_hand"),
        (pl.col("seq_sand_hands") / (pl.col("shared_hands") + eps)).alias("seq_sand_support_rate"),
        ((pl.col("seq_sand_outsider_fold_forward_n") - pl.col("seq_sand_outsider_fold_reverse_n")).abs() / (pl.col("seq_sand_outsider_fold_n") + eps)).alias("seq_sand_fold_direction_imbalance"),
    )
    frame = frame.with_columns(
        (
            pl.col("seq_resp_fold_rate")
            * (pl.col("seq_resp_n") / (pl.col("seq_resp_n") + 10.0))
        ).alias("seq_resp_fold_rate_shrunk"),
        (
            pl.col("seq_adj_check_check_rate")
            * (pl.col("seq_adj_postflop_n") / (pl.col("seq_adj_postflop_n") + 10.0))
        ).alias("seq_adj_check_check_rate_shrunk"),
        (
            pl.col("seq_sand_fold_then_partner_passive_rate")
            * (pl.col("seq_sand_outsider_fold_n") / (pl.col("seq_sand_outsider_fold_n") + 8.0))
        ).alias("seq_sand_fold_then_partner_passive_rate_shrunk"),
    )
    frame = frame.with_columns(
        (
            (
                2.0 * pl.col("seq_resp_fold_delayed_n")
                + pl.col("seq_resp_fold_to_call_max")
                + 3.0 * pl.col("seq_resp_fold_direction_imbalance")
            ).log1p()
        ).alias("seq_directed_signal"),
        (
            (
                3.0 * pl.col("seq_adj_check_check_late_n")
                + 2.0 * pl.col("seq_adj_cross_street_passive_n")
                + 4.0 * pl.col("seq_adj_check_check_rate_shrunk")
            ).log1p()
        ).alias("seq_soft_signal"),
        (
            (
                3.0 * pl.col("seq_sand_iso_yield_post_n")
                + 2.0 * pl.col("seq_sand_squeeze_surrender_n")
                + 4.0 * pl.col("seq_sand_fold_then_partner_passive_rate_shrunk")
            ).log1p()
        ).alias("seq_isolation_signal"),
    )

    table_rank_columns = [
        "seq_resp_fold_rate_shrunk",
        "seq_resp_fold_to_call_max",
        "seq_resp_fold_direction_imbalance",
        "seq_adj_check_check_rate_shrunk",
        "seq_adj_check_checks_per_hand",
        "seq_sand_fold_then_partner_passive_rate_shrunk",
        "seq_sand_iso_yield_post_per_hand",
        "seq_sand_squeeze_surrender_rate",
        "seq_directed_signal",
        "seq_soft_signal",
        "seq_isolation_signal",
    ]
    frame = frame.with_columns(
        [
            (
                pl.col(column).rank(method="average").over("table_id")
                / (pl.len().over("table_id") + eps)
            )
            .cast(pl.Float32)
            .alias(f"{column}_table_pct")
            for column in table_rank_columns
        ]
    )
    return frame


def _build_phase_sequence_frame(
    pairs_path: Path,
    phase: str,
    response: pl.LazyFrame,
    adjacency: pl.LazyFrame,
    sandwich: pl.LazyFrame,
) -> pl.DataFrame:
    pairs = (
        pl.scan_parquet(pairs_path)
        .select(["pair_id", "player_1", "player_2", "shared_hands", "table_id"])
        .with_columns(
            pl.lit(phase).alias("phase"),
            *_canonical_pair_exprs("player_1", "player_2"),
        )
    )
    frame = (
        pairs.join(response, on=["phase", "p_low", "p_high"], how="left")
        .join(adjacency, on=["phase", "p_low", "p_high"], how="left")
        .join(sandwich, on=["phase", "p_low", "p_high"], how="left")
        .drop(["p_low", "p_high"])
        .collect(engine="streaming")
    )
    frame = _add_derived_sequence_features(frame)
    if frame["pair_id"].n_unique() != frame.height:
        raise ValueError(f"{phase}: sequence pair_id values are not unique")
    return frame


def _build_sequence_features(
    paths,
    out_dir: Path,
    log: Callable[[str], None],
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    dev_path = out_dir / "dev_sequence_features.parquet"
    eval_path = out_dir / "eval_sequence_features.parquet"
    state_path = out_dir / "ordered_action_state.parquet"
    if dev_path.is_file() and eval_path.is_file():
        dev = pl.read_parquet(dev_path)
        evaluation = pl.read_parquet(eval_path)
        if dev.height != EXPECTED_DEV_ROWS or evaluation.height != EXPECTED_EVAL_ROWS:
            raise ValueError(
                f"Unexpected cached sequence shapes: dev={dev.shape}, eval={evaluation.shape}"
            )
        log("Reusing cached development/evaluation sequence feature frames.")
        return dev, evaluation, {
            "built": False,
            "dev_path": str(dev_path),
            "eval_path": str(eval_path),
            "state_path": str(state_path),
        }
    if dev_path.is_file() != eval_path.is_file():
        raise RuntimeError("Refusing to mix partial sequence feature caches")

    state_audit = _build_ordered_state(paths, state_path, log)
    log("Aggregating response, adjacency, and outsider-sandwich motifs by phase/player pair...")
    started = time.time()
    response = _response_stats(state_path)
    adjacency = _adjacency_stats(state_path)
    sandwich = _sandwich_stats(state_path)
    dev = _build_phase_sequence_frame(
        paths.prepared_dir / "dev_pairs.parquet",
        "development",
        response,
        adjacency,
        sandwich,
    )
    evaluation = _build_phase_sequence_frame(
        paths.prepared_dir / "eval_pairs.parquet",
        "evaluation",
        response,
        adjacency,
        sandwich,
    )
    if dev.height != EXPECTED_DEV_ROWS or evaluation.height != EXPECTED_EVAL_ROWS:
        raise ValueError(f"Unexpected sequence shapes: dev={dev.shape}, eval={evaluation.shape}")

    dev.write_parquet(dev_path, compression="zstd")
    evaluation.write_parquet(eval_path, compression="zstd")
    elapsed = time.time() - started
    sequence_columns = [column for column in dev.columns if column.startswith("seq_")]
    if not sequence_columns or set(sequence_columns) != {
        column for column in evaluation.columns if column.startswith("seq_")
    }:
        raise RuntimeError("Development/evaluation sequence schemas do not match")
    log(
        f"Sequence frames built in {elapsed:.1f}s: dev={dev.shape}, "
        f"eval={evaluation.shape}, new_features={len(sequence_columns)}."
    )
    return dev, evaluation, {
        "built": True,
        "duration_s": elapsed,
        "state": state_audit,
        "dev_path": str(dev_path),
        "eval_path": str(eval_path),
        "feature_count": len(sequence_columns),
        "feature_columns": sequence_columns,
    }


def _append_sequence_block(
    base: pl.DataFrame,
    sequence: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[str]]:
    sequence_columns = [column for column in sequence.columns if column.startswith("seq_")]
    if set(sequence_columns) & set(base.columns):
        raise ValueError("Sequence feature names collide with V30 columns")
    joined = base.join(
        sequence.select(["pair_id", *sequence_columns]),
        on="pair_id",
        how="left",
        validate="1:1",
    )
    if joined.height != base.height:
        raise ValueError("Sequence join changed pair row count")
    nulls = joined.select([pl.col(column).null_count() for column in sequence_columns]).row(0)
    if any(nulls):
        raise ValueError("Sequence join left null values")
    return joined, sequence_columns


def _load_baseline_oof(hnm_dir: Path, pair_ids: np.ndarray) -> pd.DataFrame:
    path = hnm_dir / "oof_rows.parquet"
    baseline = pd.read_parquet(path)
    required = {
        "pair_id",
        "outer_fold",
        "baseline_xgb",
        "baseline_lgbm",
        "baseline_catboost",
        "baseline_risk",
    }
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"HNM OOF cache is missing baseline fields: {sorted(missing)}")
    if not baseline["pair_id"].astype(str).is_unique:
        raise ValueError("Baseline OOF pair IDs are not unique")
    indexed = baseline.assign(pair_id=baseline["pair_id"].astype(str)).set_index("pair_id")
    try:
        aligned = indexed.loc[pair_ids.astype(str)].reset_index()
    except KeyError as exc:
        raise ValueError("Baseline OOF does not cover the sequence training rows") from exc
    return aligned


def _validate_persisted_folds(
    pair_ids: np.ndarray,
    groups: np.ndarray,
    fold_id: np.ndarray,
    manifest_path: Path,
) -> List[dict]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed: List[dict] = []
    if set(np.unique(fold_id)) != set(range(N_SPLITS)):
        raise ValueError(f"Unexpected persisted fold IDs: {np.unique(fold_id)}")
    for fold in range(N_SPLITS):
        va = np.flatnonzero(fold_id == fold)
        tr = np.flatnonzero(fold_id != fold)
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"Fold {fold}: table leakage detected")
        digest = hashlib.sha256("\n".join(pair_ids[va]).encode("utf-8")).hexdigest()
        row = {
            "fold": fold,
            "train_rows": len(tr),
            "validation_rows": len(va),
            "train_groups": len(set(groups[tr])),
            "validation_groups": len(set(groups[va])),
            "validation_pair_id_sha256": digest,
        }
        observed.append(row)
    if observed != expected:
        raise ValueError("Persisted OOF fold assignments do not match fold_manifest.json")
    return observed


def _score_predictions(
    y: np.ndarray,
    known: np.ndarray,
    stress_weight: np.ndarray,
    predictions: Dict[str, np.ndarray],
    fold_id: np.ndarray,
) -> Dict[str, dict]:
    metrics: Dict[str, dict] = {}
    for component, scores in predictions.items():
        metrics[component] = {
            "confirmed_ap": float(average_precision_score(y[known], scores[known])),
            "pu_stress_ap": float(
                average_precision_score(y, scores, sample_weight=stress_weight)
            ),
            "folds": [
                {
                    "fold": fold,
                    **_fold_metrics(
                        np.flatnonzero(fold_id == fold),
                        y,
                        known,
                        stress_weight,
                        scores,
                    ),
                }
                for fold in range(N_SPLITS)
            ],
        }
    return metrics


def _run_sequence_risk_head(
    data: Dict[str, object],
    baseline: pd.DataFrame,
    use_cuda: bool,
    log: Callable[[str], None],
) -> dict:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X: pd.DataFrame = data["X"]
    X_eval: pd.DataFrame = data["X_eval"]
    y = data["y"]
    known = data["known"]
    groups = data["groups"]
    fit_weight = data["fit_weight"]
    stress_weight = data["stress_weight"]
    fold_id = baseline["outer_fold"].to_numpy(dtype=np.int8)

    oof = {
        "xgb": np.zeros(len(X), dtype=np.float32),
        "lgbm": np.zeros(len(X), dtype=np.float32),
        "catboost": np.zeros(len(X), dtype=np.float32),
        "risk": np.zeros(len(X), dtype=np.float32),
    }
    eval_parts = {component: [] for component in ("xgb", "lgbm", "catboost")}

    for fold in range(N_SPLITS):
        tr = np.flatnonzero(fold_id != fold)
        va = np.flatnonzero(fold_id == fold)
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"Fold {fold}: table leakage detected before fit")

        xgb = XGBClassifier(**_xgb_params(use_cuda), random_state=SEED + fold)
        xgb.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])],
            sample_weight_eval_set=[stress_weight[va]],
            verbose=False,
        )
        oof["xgb"][va] = xgb.predict_proba(X.iloc[va])[:, 1]
        eval_parts["xgb"].append(xgb.predict_proba(X_eval)[:, 1])
        del xgb
        gc.collect()

        lgbm = LGBMClassifier(**LGB_PARAMS, random_state=SEED + fold)
        lgbm.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_sample_weight=[stress_weight[va]],
        )
        oof["lgbm"][va] = lgbm.predict_proba(X.iloc[va])[:, 1]
        eval_parts["lgbm"].append(lgbm.predict_proba(X_eval)[:, 1])
        del lgbm
        gc.collect()

        catboost = CatBoostClassifier(**CB_PARAMS, random_seed=SEED + fold)
        catboost.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])],
            early_stopping_rounds=80,
            verbose=0,
        )
        oof["catboost"][va] = catboost.predict_proba(X.iloc[va])[:, 1]
        eval_parts["catboost"].append(catboost.predict_proba(X_eval)[:, 1])
        del catboost
        gc.collect()

        oof["risk"][va] = (
            BLEND_WEIGHTS["xgb"] * oof["xgb"][va]
            + BLEND_WEIGHTS["lgbm"] * oof["lgbm"][va]
            + BLEND_WEIGHTS["catboost"] * oof["catboost"][va]
        )
        current = _fold_metrics(va, y, known, stress_weight, oof["risk"])
        log(
            f"Fold {fold + 1}: confirmed AP {current['confirmed_ap']:.6f}, "
            f"PU-stress AP {current['pu_stress_ap']:.6f}."
        )

    eval_components = {
        component: np.mean(parts, axis=0).astype(np.float32)
        for component, parts in eval_parts.items()
    }
    eval_risk = (
        BLEND_WEIGHTS["xgb"] * eval_components["xgb"]
        + BLEND_WEIGHTS["lgbm"] * eval_components["lgbm"]
        + BLEND_WEIGHTS["catboost"] * eval_components["catboost"]
    ).astype(np.float32)
    return {
        "oof": oof,
        "eval_components": eval_components,
        "eval_risk": eval_risk,
        "fold_id": fold_id,
    }


def _feature_diagnostics(
    train_df: pl.DataFrame,
    sequence_columns: Sequence[str],
    y: np.ndarray,
    known: np.ndarray,
) -> dict:
    numeric = train_df.select(sequence_columns).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0)
    diagnostics = []
    for column in sequence_columns:
        values = numeric[column].to_numpy(dtype=np.float64)
        unique = int(pd.Series(values).nunique())
        univariate_ap = None
        if unique > 1:
            univariate_ap = float(average_precision_score(y[known], values[known]))
        diagnostics.append(
            {
                "feature": column,
                "nonzero_rate": float(np.mean(values != 0)),
                "unique": unique,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "confirmed_univariate_ap": univariate_ap,
            }
        )
    ranked = sorted(
        diagnostics,
        key=lambda item: item["confirmed_univariate_ap"]
        if item["confirmed_univariate_ap"] is not None
        else -1.0,
        reverse=True,
    )
    return {
        "feature_count": len(sequence_columns),
        "constant_features": [item["feature"] for item in diagnostics if item["unique"] <= 1],
        "top_confirmed_univariate_ap": ranked[:20],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = default_lamhuy_paths(root)
    hnm_dir = root / "outputs" / "poker_collusion" / "lamhuy_hnm"
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_ordered_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_lines.append(line)

    summary: Dict[str, object] = {
        "recipe_id": RECIPE_ID,
        "status": "RUNNING",
        "seed": SEED,
        "source_frame": str(hnm_dir),
        "output_dir": str(out_dir),
        "cv": "exact persisted 5-fold table-group-disjoint assignments from lamhuy_hnm/oof_rows.parquet",
        "fit_weights": {"positive": 2.0, "confirmed_negative": 1.0, "pu": 0.35},
        "blend_weights": BLEND_WEIGHTS,
        "promotion_gate": {
            "baseline_confirmed_ap_min": PROMOTION_BASELINE_AP,
            "candidate_confirmed_ap_delta_min": PROMOTION_CONFIRMED_DELTA,
            "candidate_pu_stress_ap_delta_min": PROMOTION_STRESS_DELTA_FLOOR,
        },
        "sequence_scope": [
            "last-aggressor response type, latency, street, severity, direction",
            "adjacent same-street and cross-street pair action motifs",
            "aggressor -> outsider response -> partner next-action sandwich motifs",
            "opportunity-normalized, per-hand, shrinkage, direction, and table-rank summaries",
        ],
    }
    started = time.time()
    candidate_path = out_dir / "candidate_v30_ordered_sequence.csv"
    if candidate_path.exists():
        candidate_path.unlink()

    try:
        missing = paths.missing()
        if missing:
            raise FileNotFoundError("Missing LamHuy inputs: " + ", ".join(map(str, missing)))
        required_hnm = [
            hnm_dir / "dev_features.parquet",
            hnm_dir / "eval_features.parquet",
            hnm_dir / "pair_groups.parquet",
            hnm_dir / "oof_rows.parquet",
            hnm_dir / "fold_manifest.json",
        ]
        absent_hnm = [path for path in required_hnm if not path.is_file()]
        if absent_hnm:
            raise FileNotFoundError("Missing HNM caches: " + ", ".join(map(str, absent_hnm)))

        base_dev, pair_groups, base_eval = _load_full_frames(hnm_dir)
        sequence_dev, sequence_eval, sequence_build = _build_sequence_features(
            paths, out_dir, log
        )
        summary["sequence_build"] = sequence_build
        dev, sequence_columns = _append_sequence_block(base_dev, sequence_dev)
        evaluation, eval_sequence_columns = _append_sequence_block(base_eval, sequence_eval)
        if sequence_columns != eval_sequence_columns:
            raise ValueError("Development/evaluation sequence column order differs")
        summary["frame_shapes"] = {
            "base_dev": base_dev.shape,
            "base_eval": base_eval.shape,
            "sequence_dev": sequence_dev.shape,
            "sequence_eval": sequence_eval.shape,
            "augmented_dev": dev.shape,
            "augmented_eval": evaluation.shape,
        }
        summary["sequence_feature_count"] = len(sequence_columns)
        summary["sequence_feature_columns"] = sequence_columns

        data = _prepare_training_data(paths, dev, pair_groups, evaluation, out_dir)
        baseline = _load_baseline_oof(hnm_dir, data["pair_ids"])
        fold_id = baseline["outer_fold"].to_numpy(dtype=np.int8)
        summary["fold_manifest"] = _validate_persisted_folds(
            data["pair_ids"],
            data["groups"],
            fold_id,
            hnm_dir / "fold_manifest.json",
        )
        summary["feature_diagnostics"] = _feature_diagnostics(
            data["train_df"], sequence_columns, data["y"], data["known"]
        )

        baseline_predictions = {
            "xgb": baseline["baseline_xgb"].to_numpy(dtype=np.float32),
            "lgbm": baseline["baseline_lgbm"].to_numpy(dtype=np.float32),
            "catboost": baseline["baseline_catboost"].to_numpy(dtype=np.float32),
            "risk": baseline["baseline_risk"].to_numpy(dtype=np.float32),
        }
        baseline_metrics = _score_predictions(
            data["y"], data["known"], data["stress_weight"], baseline_predictions, fold_id
        )
        summary["baseline_metrics"] = baseline_metrics
        log(
            f"Persisted baseline: confirmed AP {baseline_metrics['risk']['confirmed_ap']:.7f}, "
            f"PU-stress AP {baseline_metrics['risk']['pu_stress_ap']:.7f}."
        )

        use_cuda = bool(_probe_cuda(log))
        summary["xgboost_cuda"] = use_cuda
        log(f"XGBoost CUDA enabled: {use_cuda}")
        result = _run_sequence_risk_head(data, baseline, use_cuda, log)
        candidate_metrics = _score_predictions(
            data["y"],
            data["known"],
            data["stress_weight"],
            result["oof"],
            fold_id,
        )
        for component in ("xgb", "lgbm", "catboost", "risk"):
            candidate_metrics[component]["confirmed_ap_delta_vs_baseline"] = (
                candidate_metrics[component]["confirmed_ap"]
                - baseline_metrics[component]["confirmed_ap"]
            )
            candidate_metrics[component]["pu_stress_ap_delta_vs_baseline"] = (
                candidate_metrics[component]["pu_stress_ap"]
                - baseline_metrics[component]["pu_stress_ap"]
            )
            for fold in range(N_SPLITS):
                candidate_metrics[component]["folds"][fold]["confirmed_ap_delta_vs_baseline"] = (
                    candidate_metrics[component]["folds"][fold]["confirmed_ap"]
                    - baseline_metrics[component]["folds"][fold]["confirmed_ap"]
                )
                candidate_metrics[component]["folds"][fold]["pu_stress_ap_delta_vs_baseline"] = (
                    candidate_metrics[component]["folds"][fold]["pu_stress_ap"]
                    - baseline_metrics[component]["folds"][fold]["pu_stress_ap"]
                )
        summary["candidate_metrics"] = candidate_metrics

        oof_frame = pd.DataFrame(
            {
                "pair_id": data["pair_ids"],
                "cv_group": data["groups"].astype(str),
                "outer_fold": fold_id,
                "known": data["known"],
                "behavior_y": data["behavior_y"],
                "y": data["y"],
                "fit_weight": data["fit_weight"],
                "stress_weight": data["stress_weight"],
            }
        )
        for component in ("xgb", "lgbm", "catboost", "risk"):
            oof_frame[f"baseline_{component}"] = baseline_predictions[component]
            oof_frame[f"sequence_{component}"] = result["oof"][component]
        oof_path = out_dir / "oof_rows.parquet"
        oof_frame.to_parquet(oof_path, index=False)
        summary["artifacts"] = {
            "oof_rows": str(oof_path),
            "feature_columns": str(out_dir / "feature_columns.json"),
            "dev_sequence_features": str(out_dir / "dev_sequence_features.parquet"),
            "eval_sequence_features": str(out_dir / "eval_sequence_features.parquet"),
        }

        risk_metrics = candidate_metrics["risk"]
        gates = {
            "baseline_confirmed_ap": baseline_metrics["risk"]["confirmed_ap"]
            >= PROMOTION_BASELINE_AP,
            "candidate_confirmed_delta": risk_metrics["confirmed_ap_delta_vs_baseline"]
            >= PROMOTION_CONFIRMED_DELTA,
            "candidate_pu_stress_delta": risk_metrics["pu_stress_ap_delta_vs_baseline"]
            >= PROMOTION_STRESS_DELTA_FLOOR,
        }
        summary["gate_results"] = gates
        summary["all_gates_passed"] = all(gates.values())
        source_path, source_kind = _candidate_source(root)
        summary["rank_comparison"] = {
            "local_v30": _rank_comparison(
                data["eval_pair_ids"],
                result["eval_risk"],
                root / "outputs" / "poker_collusion" / "repro_lamhuy" / "submission.csv",
            ),
            "original_v30": _rank_comparison(
                data["eval_pair_ids"],
                result["eval_risk"],
                root
                / "outputs"
                / "poker_collusion"
                / "lamhuy_original_v30"
                / "submission.csv",
            ),
        }

        log(
            "Sequence candidate: confirmed AP "
            f"{risk_metrics['confirmed_ap']:.7f} "
            f"(delta {risk_metrics['confirmed_ap_delta_vs_baseline']:+.7f}); "
            f"PU-stress AP {risk_metrics['pu_stress_ap']:.7f} "
            f"(delta {risk_metrics['pu_stress_ap_delta_vs_baseline']:+.7f})."
        )
        if all(gates.values()):
            emitted = _emit_candidate(
                "ordered_sequence",
                result["eval_risk"],
                data["eval_pair_ids"],
                source_path,
                source_kind,
                paths.data_dir / "evaluation_pairs.csv",
                out_dir,
            )
            emitted_path = Path(emitted["path"])
            if emitted_path != candidate_path:
                emitted_path.replace(candidate_path)
                emitted["path"] = str(candidate_path)
                emitted["sha256"] = _sha256(candidate_path)
            summary["candidate"] = emitted
            summary["status"] = "COMPLETED_PROMOTED"
            log(f"PROMOTED: {candidate_path}")
        else:
            summary["candidate"] = None
            summary["status"] = "COMPLETED_NULL"
            log(f"NOT PROMOTED. Gate results: {gates}")

        summary["duration_s"] = time.time() - started
        _write_run_files(out_dir, summary, log_lines)
        return 0
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        summary["duration_s"] = time.time() - started
        log(f"FAILED: {type(exc).__name__}: {exc}")
        _write_run_files(out_dir, summary, log_lines)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
