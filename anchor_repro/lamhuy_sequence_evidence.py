"""Paired V30 evidence ablation with hand-level ordered-action motifs.

BUILD + MEASURE ONLY.  This module reuses the validated ordered action-state cache,
creates sparse response/adjacency/sandwich features keyed by ``(pair_id, hand_id)``,
and compares the exact V30 dual-engine evidence architecture with and without that
new block.  Both arms use identical inherited pair folds and identical family
routing on discovery (seed 42) and confirmation (seed 137) artifacts.

No Kaggle submission is made and no eval candidate is written here.  Production
evidence inference is intentionally deferred until both OOF gates pass.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRanker

from anchor_repro.lamhuy_ordered_sequence import _canonical_pair_exprs

BEHAVIORS: Mapping[int, str] = {
    1: "directed_transfer",
    2: "soft_play",
    3: "coordinated_isolation",
}
N_FOLDS = 5
V30_EVIDENCE_MAP5 = 0.5067
MIN_DISCOVERY_DELTA = 0.015
MIN_CONFIRMATION_DELTA = 0.0
MAX_FAMILY_REGRESSION = 0.02
EPS = 1e-4

# Preregistered, concept-driven sequence block.  These are generated without labels.
SEQUENCE_MODEL_FEATURES = [
    "seq_resp_n",
    "seq_resp_fold_n",
    "seq_resp_call_n",
    "seq_resp_raise_n",
    "seq_resp_direct_n",
    "seq_resp_delayed_n",
    "seq_resp_fold_direct_n",
    "seq_resp_fold_delayed_n",
    "seq_resp_call_direct_n",
    "seq_resp_raise_direct_n",
    "seq_resp_fold_hu_n",
    "seq_resp_fold_mw_n",
    "seq_resp_gap_mean",
    "seq_resp_gap_max",
    "seq_resp_fold_gap_mean",
    "seq_resp_fold_to_call_mean",
    "seq_resp_fold_to_call_max",
    "seq_resp_fold_pre_n",
    "seq_resp_fold_flop_n",
    "seq_resp_fold_turn_n",
    "seq_resp_fold_river_n",
    "seq_resp_fold_direction_imbalance",
    "seq_adj_n",
    "seq_adj_postflop_n",
    "seq_adj_check_check_n",
    "seq_adj_passive_chain_n",
    "seq_adj_agg_fold_n",
    "seq_adj_agg_call_n",
    "seq_adj_agg_reagg_n",
    "seq_adj_cross_street_passive_n",
    "seq_adj_check_check_hu_n",
    "seq_adj_check_check_late_n",
    "seq_adj_agg_fold_to_call_max",
    "seq_adj_check_direction_imbalance",
    "seq_sand_n",
    "seq_sand_outsider_fold_n",
    "seq_sand_outsider_call_n",
    "seq_sand_outsider_raise_n",
    "seq_sand_fold_then_partner_passive_n",
    "seq_sand_fold_then_partner_raise_n",
    "seq_sand_call_then_partner_passive_n",
    "seq_sand_call_then_partner_fold_n",
    "seq_sand_call_then_partner_raise_n",
    "seq_sand_raise_then_partner_passive_n",
    "seq_sand_raise_then_partner_fold_n",
    "seq_sand_iso_yield_post_n",
    "seq_sand_squeeze_surrender_n",
    "seq_sand_outsider_fold_to_call_max",
    "seq_sand_fold_direction_imbalance",
    "seq_hand_directed_signal",
    "seq_hand_soft_signal",
    "seq_hand_isolation_signal",
    "seq_ev_directed_transfer_combo",
    "seq_ev_directed_gap_combo",
    "seq_ev_soft_showdown_combo",
    "seq_ev_soft_pot_combo",
    "seq_ev_isolation_fold_combo",
    "seq_ev_isolation_priority_combo",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _pair_map(dev_pairs: Path, eval_pairs: Path) -> pl.LazyFrame:
    def one(path: Path, phase: str) -> pl.LazyFrame:
        return (
            pl.scan_parquet(path)
            .select(["pair_id", "player_1", "player_2"])
            .with_columns(
                pl.lit(phase).alias("phase"),
                *_canonical_pair_exprs("player_1", "player_2"),
            )
            .select(["phase", "p_low", "p_high", "pair_id"])
        )

    return pl.concat(
        [one(dev_pairs, "development"), one(eval_pairs, "evaluation")],
        how="vertical",
    )


def _response_hand_stats(state_path: Path, pairs: pl.LazyFrame) -> pl.LazyFrame:
    frame = (
        pl.scan_parquet(state_path)
        .filter(
            pl.col("last_aggressor").is_not_null()
            & (pl.col("last_aggressor") != pl.col("player_id"))
            & (pl.col("to_call") > 0)
            & pl.col("_last_agg_no").is_not_null()
        )
        .with_columns(
            *_canonical_pair_exprs("last_aggressor", "player_id"),
            (pl.col("action_no") - pl.col("_last_agg_no")).clip(1, 50).alias("_gap"),
            (pl.col("action") == "fold").alias("_fold"),
            (pl.col("action").is_in(["call", "all_in"]) & ~pl.col("is_aggressive")).alias("_call"),
            pl.col("is_aggressive").alias("_raise"),
        )
        .join(pairs, on=["phase", "p_low", "p_high"], how="inner")
        .with_columns((pl.col("last_aggressor") == pl.col("p_low")).alias("_forward"))
    )
    agg: List[pl.Expr] = [
        pl.len().alias("seq_resp_n"),
        pl.col("_fold").sum().alias("seq_resp_fold_n"),
        pl.col("_call").sum().alias("seq_resp_call_n"),
        pl.col("_raise").sum().alias("seq_resp_raise_n"),
        (pl.col("_gap") == 1).sum().alias("seq_resp_direct_n"),
        (pl.col("_gap") > 1).sum().alias("seq_resp_delayed_n"),
        (pl.col("_fold") & (pl.col("_gap") == 1)).sum().alias("seq_resp_fold_direct_n"),
        (pl.col("_fold") & (pl.col("_gap") > 1)).sum().alias("seq_resp_fold_delayed_n"),
        (pl.col("_call") & (pl.col("_gap") == 1)).sum().alias("seq_resp_call_direct_n"),
        (pl.col("_raise") & (pl.col("_gap") == 1)).sum().alias("seq_resp_raise_direct_n"),
        (pl.col("_fold") & (pl.col("players_active") == 2)).sum().alias("seq_resp_fold_hu_n"),
        (pl.col("_fold") & (pl.col("players_active") > 2)).sum().alias("seq_resp_fold_mw_n"),
        pl.col("_gap").mean().alias("seq_resp_gap_mean"),
        pl.col("_gap").max().alias("seq_resp_gap_max"),
        pl.when(pl.col("_fold")).then(pl.col("_gap")).otherwise(None).mean().alias("seq_resp_fold_gap_mean"),
        pl.when(pl.col("_fold")).then(pl.col("to_call_bb")).otherwise(None).mean().alias("seq_resp_fold_to_call_mean"),
        pl.when(pl.col("_fold")).then(pl.col("to_call_bb")).otherwise(None).max().alias("seq_resp_fold_to_call_max"),
        (pl.col("_fold") & pl.col("_forward")).sum().alias("seq_resp_fold_forward_n"),
        (pl.col("_fold") & ~pl.col("_forward")).sum().alias("seq_resp_fold_reverse_n"),
    ]
    for street in ("preflop", "flop", "turn", "river"):
        tag = "pre" if street == "preflop" else street
        agg.append(
            (pl.col("_fold") & (pl.col("street") == street)).sum().alias(f"seq_resp_fold_{tag}_n")
        )
    return frame.group_by(["phase", "pair_id", "hand_id"]).agg(agg)


def _adjacency_hand_stats(state_path: Path, pairs: pl.LazyFrame) -> pl.LazyFrame:
    frame = (
        pl.scan_parquet(state_path)
        .filter(pl.col("_prev_player").is_not_null() & (pl.col("_prev_player") != pl.col("player_id")))
        .with_columns(
            *_canonical_pair_exprs("_prev_player", "player_id"),
            (pl.col("street") == pl.col("_prev_street")).alias("_same_street"),
            ((pl.col("street") == pl.col("_prev_street")) & (pl.col("street") != "preflop")).alias("_postflop"),
        )
        .join(pairs, on=["phase", "p_low", "p_high"], how="inner")
        .with_columns(
            (pl.col("_postflop") & (pl.col("_prev_action") == "check") & (pl.col("action") == "check")).alias("_check_check"),
            (pl.col("_postflop") & pl.col("_prev_action").is_in(["check", "call"]) & pl.col("action").is_in(["check", "call"])).alias("_passive_chain"),
            (pl.col("_same_street") & pl.col("_prev_is_aggressive") & (pl.col("action") == "fold")).alias("_agg_fold"),
            (pl.col("_same_street") & pl.col("_prev_is_aggressive") & pl.col("action").is_in(["call", "all_in"]) & ~pl.col("is_aggressive")).alias("_agg_call"),
            (pl.col("_same_street") & pl.col("_prev_is_aggressive") & pl.col("is_aggressive")).alias("_agg_reagg"),
            ((pl.col("street") != pl.col("_prev_street")) & pl.col("_prev_action").is_in(["check", "call"]) & (pl.col("action") == "check")).alias("_cross_street_passive"),
            (pl.col("_prev_player") == pl.col("p_low")).alias("_forward"),
        )
    )
    return frame.group_by(["phase", "pair_id", "hand_id"]).agg(
        pl.len().alias("seq_adj_n"),
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
    )


def _sandwich_hand_stats(state_path: Path, pairs: pl.LazyFrame) -> pl.LazyFrame:
    frame = (
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
            (pl.col("action").is_in(["call", "all_in"]) & ~pl.col("is_aggressive")).alias("_outsider_call"),
            pl.col("is_aggressive").alias("_outsider_raise"),
            pl.col("_next_action").is_in(["check", "call"]).alias("_partner_passive"),
            (pl.col("_next_action") == "fold").alias("_partner_fold"),
            pl.col("_next_is_aggressive").alias("_partner_raise"),
        )
        .join(pairs, on=["phase", "p_low", "p_high"], how="inner")
        .with_columns((pl.col("last_aggressor") == pl.col("p_low")).alias("_forward"))
    )
    return frame.group_by(["phase", "pair_id", "hand_id"]).agg(
        pl.len().alias("seq_sand_n"),
        pl.col("_outsider_fold").sum().alias("seq_sand_outsider_fold_n"),
        pl.col("_outsider_call").sum().alias("seq_sand_outsider_call_n"),
        pl.col("_outsider_raise").sum().alias("seq_sand_outsider_raise_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_passive")).sum().alias("seq_sand_fold_then_partner_passive_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_raise")).sum().alias("seq_sand_fold_then_partner_raise_n"),
        (pl.col("_outsider_call") & pl.col("_partner_passive")).sum().alias("seq_sand_call_then_partner_passive_n"),
        (pl.col("_outsider_call") & pl.col("_partner_fold")).sum().alias("seq_sand_call_then_partner_fold_n"),
        (pl.col("_outsider_call") & pl.col("_partner_raise")).sum().alias("seq_sand_call_then_partner_raise_n"),
        (pl.col("_outsider_raise") & pl.col("_partner_passive")).sum().alias("seq_sand_raise_then_partner_passive_n"),
        (pl.col("_outsider_raise") & pl.col("_partner_fold")).sum().alias("seq_sand_raise_then_partner_fold_n"),
        (pl.col("_outsider_fold") & pl.col("_partner_passive") & pl.col("street").is_in(["flop", "turn", "river"])).sum().alias("seq_sand_iso_yield_post_n"),
        (pl.col("_outsider_raise") & (pl.col("_partner_passive") | pl.col("_partner_fold"))).sum().alias("seq_sand_squeeze_surrender_n"),
        (pl.col("_outsider_fold") & pl.col("_forward")).sum().alias("seq_sand_outsider_fold_forward_n"),
        (pl.col("_outsider_fold") & ~pl.col("_forward")).sum().alias("seq_sand_outsider_fold_reverse_n"),
        pl.when(pl.col("_outsider_fold")).then(pl.col("to_call_bb")).otherwise(None).max().alias("seq_sand_outsider_fold_to_call_max"),
    )


def _add_hand_sequence_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.with_columns(
            ((pl.col("seq_resp_fold_forward_n") - pl.col("seq_resp_fold_reverse_n")).abs() / (pl.col("seq_resp_fold_n") + EPS)).cast(pl.Float32).alias("seq_resp_fold_direction_imbalance"),
            ((pl.col("seq_adj_check_check_forward_n") - pl.col("seq_adj_check_check_reverse_n")).abs() / (pl.col("seq_adj_check_check_n") + EPS)).cast(pl.Float32).alias("seq_adj_check_direction_imbalance"),
            ((pl.col("seq_sand_outsider_fold_forward_n") - pl.col("seq_sand_outsider_fold_reverse_n")).abs() / (pl.col("seq_sand_outsider_fold_n") + EPS)).cast(pl.Float32).alias("seq_sand_fold_direction_imbalance"),
        )
        .with_columns(
            (2.0 * pl.col("seq_resp_fold_delayed_n") + pl.col("seq_resp_fold_to_call_max") + pl.col("seq_adj_agg_fold_n")).log1p().cast(pl.Float32).alias("seq_hand_directed_signal"),
            (3.0 * pl.col("seq_adj_check_check_late_n") + 2.0 * pl.col("seq_adj_cross_street_passive_n") + pl.col("seq_adj_passive_chain_n")).log1p().cast(pl.Float32).alias("seq_hand_soft_signal"),
            (3.0 * pl.col("seq_sand_iso_yield_post_n") + 2.0 * pl.col("seq_sand_squeeze_surrender_n") + pl.col("seq_sand_fold_then_partner_passive_n")).log1p().cast(pl.Float32).alias("seq_hand_isolation_signal"),
        )
    )


def _build_sparse_sequence_features(
    state_path: Path,
    dev_pairs_path: Path,
    eval_pairs_path: Path,
    dev_out: Path,
    eval_out: Path,
) -> Dict[str, object]:
    if dev_out.is_file() and eval_out.is_file():
        return {
            "built": False,
            "dev_rows": pl.scan_parquet(dev_out).select(pl.len()).collect().item(),
            "eval_rows": pl.scan_parquet(eval_out).select(pl.len()).collect().item(),
        }

    started = time.time()
    pairs = _pair_map(dev_pairs_path, eval_pairs_path)
    response = _response_hand_stats(state_path, pairs)
    adjacency = _adjacency_hand_stats(state_path, pairs)
    sandwich = _sandwich_hand_stats(state_path, pairs)
    keys = ["phase", "pair_id", "hand_id"]
    sparse = response.join(adjacency, on=keys, how="full", coalesce=True).join(
        sandwich, on=keys, how="full", coalesce=True
    )
    schema = sparse.collect_schema()
    numeric = [column for column, dtype in schema.items() if dtype.is_numeric()]
    sparse = _add_hand_sequence_features(
        sparse.with_columns([pl.col(column).fill_null(0) for column in numeric])
    )
    sparse.filter(pl.col("phase") == "development").drop("phase").sink_parquet(
        dev_out, compression="zstd"
    )
    sparse.filter(pl.col("phase") == "evaluation").drop("phase").sink_parquet(
        eval_out, compression="zstd"
    )
    dev_audit = pl.scan_parquet(dev_out).select(
        pl.len().alias("rows"), pl.struct(["pair_id", "hand_id"]).n_unique().alias("unique_keys")
    ).collect().row(0, named=True)
    eval_audit = pl.scan_parquet(eval_out).select(
        pl.len().alias("rows"), pl.struct(["pair_id", "hand_id"]).n_unique().alias("unique_keys")
    ).collect().row(0, named=True)
    if dev_audit["rows"] != dev_audit["unique_keys"] or eval_audit["rows"] != eval_audit["unique_keys"]:
        raise RuntimeError("Sparse sequence pair-hand keys are not unique")
    return {
        "built": True,
        "duration_s": time.time() - started,
        "dev_rows": dev_audit["rows"],
        "eval_rows": eval_audit["rows"],
    }


def _hand_feature_names(path: Path) -> List[str]:
    skip = {
        "pair_id", "hand_id", "player_1", "player_2", "p_low", "p_high",
        "table_id", "evidence_rank", "is_evidence", "hand_order", "pair_hand_no",
    }
    return [
        column
        for column, dtype in pl.scan_parquet(path).collect_schema().items()
        if dtype.is_numeric() and column not in skip and "label" not in column and "behavior_id" not in column
    ]


def _prepare_evidence_base(
    dev_hand_path: Path,
    dev_sequence_path: Path,
    labels_path: Path,
    evidence_path: Path,
) -> Tuple[pl.DataFrame, List[str], List[str], Dict[str, set]]:
    hand_features = _hand_feature_names(dev_hand_path)
    labels = pl.read_csv(labels_path)
    positive = (
        labels.filter(pl.col("label") == 1)
        .select(["pair_id", "behavior_family"])
        .with_columns(
            pl.when(pl.col("behavior_family") == "directed_transfer").then(1)
            .when(pl.col("behavior_family") == "soft_play").then(2)
            .otherwise(3).cast(pl.Int8).alias("behavior_id_true")
        )
    )
    evidence = pl.read_csv(evidence_path)
    evidence_keys = evidence.select(["pair_id", "hand_id"]).unique().with_columns(
        pl.lit(1, dtype=pl.Int8).alias("is_evidence")
    )
    sequence_schema = pl.scan_parquet(dev_sequence_path).collect_schema()
    sparse_numeric = [
        column for column, dtype in sequence_schema.items()
        if dtype.is_numeric() and column not in {"pair_id", "hand_id"}
    ]
    missing = sorted(set(SEQUENCE_MODEL_FEATURES) - set(sparse_numeric) - {
        "seq_ev_directed_transfer_combo", "seq_ev_directed_gap_combo",
        "seq_ev_soft_showdown_combo", "seq_ev_soft_pot_combo",
        "seq_ev_isolation_fold_combo", "seq_ev_isolation_priority_combo",
    })
    if missing:
        raise ValueError(f"Sparse sequence cache is missing preregistered features: {missing}")

    base = (
        pl.scan_parquet(dev_hand_path)
        .select(["pair_id", "hand_id"] + hand_features)
        .join(positive.lazy(), on="pair_id", how="inner")
        .join(pl.scan_parquet(dev_sequence_path), on=["pair_id", "hand_id"], how="left")
        .join(evidence_keys.lazy(), on=["pair_id", "hand_id"], how="left")
        .with_columns(
            [pl.col(column).fill_null(0) for column in sparse_numeric]
            + [pl.col("is_evidence").fill_null(0)]
        )
        .with_columns(
            ((pl.col("seq_resp_fold_n") + pl.col("seq_adj_agg_fold_n")) * (1.0 + pl.col("transfer_any_bb").clip(0, None))).log1p().cast(pl.Float32).alias("seq_ev_directed_transfer_combo"),
            (pl.col("seq_resp_fold_delayed_n") * (1.0 + pl.col("contrib_gap_bb").abs())).log1p().cast(pl.Float32).alias("seq_ev_directed_gap_combo"),
            (pl.col("seq_adj_check_check_late_n") * (1.0 + pl.col("both_showdown"))).log1p().cast(pl.Float32).alias("seq_ev_soft_showdown_combo"),
            ((pl.col("seq_adj_check_check_n") + pl.col("seq_adj_cross_street_passive_n")) * (1.0 + pl.col("pot_bb").clip(0, None))).log1p().cast(pl.Float32).alias("seq_ev_soft_pot_combo"),
            (pl.col("seq_sand_fold_then_partner_passive_n") * (1.0 + pl.col("outsider_folds_to_pair"))).log1p().cast(pl.Float32).alias("seq_ev_isolation_fold_combo"),
            ((pl.col("seq_sand_iso_yield_post_n") + pl.col("seq_sand_squeeze_surrender_n")) * (1.0 + pl.col("isolation_priority").clip(0, None))).log1p().cast(pl.Float32).alias("seq_ev_isolation_priority_combo"),
        )
        .sort(["pair_id", "hand_id"])
        .collect(engine="streaming")
    )

    sequence_features = list(SEQUENCE_MODEL_FEATURES)
    rank_sources = hand_features + sequence_features
    rank_exprs = [
        (pl.col(column).rank(method="average").over("pair_id") / pl.len().over("pair_id"))
        .cast(pl.Float32).alias(f"{column}_pair_pct")
        for column in rank_sources
    ] + [
        (pl.col(column) / (pl.col(column).max().over("pair_id") + EPS))
        .cast(pl.Float32).alias(f"{column}_to_max")
        for column in rank_sources
    ]
    base = base.with_columns(rank_exprs).with_row_index("_row")
    baseline_features = hand_features + [f"{c}_pair_pct" for c in hand_features] + [f"{c}_to_max" for c in hand_features]
    augmented_features = baseline_features + sequence_features + [f"{c}_pair_pct" for c in sequence_features] + [f"{c}_to_max" for c in sequence_features]
    truth = {
        (pair_key[0] if isinstance(pair_key, tuple) else pair_key): set(
            group["hand_id"].to_list()
        )
        for pair_key, group in evidence.group_by("pair_id", maintain_order=True)
    }
    return base, baseline_features, augmented_features, truth


def _pair_ap5(
    pair_ids: np.ndarray,
    hand_ids: np.ndarray,
    scores: np.ndarray,
    truth: Mapping[str, set],
) -> pd.DataFrame:
    frame = pd.DataFrame({"pair_id": pair_ids, "hand_id": hand_ids, "score": scores})
    rows = []
    for pair_id, group in frame.groupby("pair_id", sort=False):
        relevant = truth.get(pair_id, set())
        if not relevant:
            rows.append((pair_id, 0.0))
            continue
        predicted = group.sort_values(["score", "hand_id"], ascending=[False, True], kind="mergesort")["hand_id"].tolist()[:5]
        hits = 0
        total = 0.0
        for rank, hand_id in enumerate(predicted, start=1):
            if hand_id in relevant:
                hits += 1
                total += hits / rank
        rows.append((pair_id, total / min(len(relevant), 5)))
    return pd.DataFrame(rows, columns=["pair_id", "ap5"])


def _score_set(
    frame: pl.DataFrame,
    scores: np.ndarray,
    truth: Mapping[str, set],
) -> Dict[str, object]:
    pair_scores = _pair_ap5(
        frame["pair_id"].to_numpy(), frame["hand_id"].to_numpy(), scores, truth
    )
    family = frame.select(["pair_id", "behavior_id_true"]).unique().to_pandas()
    pair_scores = pair_scores.merge(family, on="pair_id", how="left", validate="one_to_one")
    by_family = {
        BEHAVIORS[family_id]: float(pair_scores.loc[pair_scores["behavior_id_true"] == family_id, "ap5"].mean())
        for family_id in (1, 2, 3)
    }
    return {
        "map5": float(pair_scores["ap5"].mean()),
        "by_family": by_family,
        "pair_scores": pair_scores,
    }


def _matrix(frame: pl.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    return (
        frame.select(feature_columns).to_pandas()
        .replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    )


def _run_dual_engine_oof(
    evidence_base: pl.DataFrame,
    feature_columns: Sequence[str],
    route_path: Path,
    model_seed: int,
    label: str,
) -> Tuple[Dict[str, np.ndarray], pl.DataFrame]:
    route = pd.read_parquet(route_path)
    required = {
        "pair_id", "outer_fold", "specialist_directed",
        "specialist_soft", "specialist_isolation",
    }
    if not required.issubset(route.columns):
        raise ValueError(f"{route_path} lacks columns {sorted(required - set(route.columns))}")
    route = route[[
        "pair_id", "outer_fold", "specialist_directed",
        "specialist_soft", "specialist_isolation",
    ]].copy()
    route["behavior_id_pred"] = route[[
        "specialist_directed", "specialist_soft", "specialist_isolation"
    ]].to_numpy().argmax(axis=1).astype(np.int8) + 1
    route_pl = pl.from_pandas(route[["pair_id", "outer_fold", "behavior_id_pred"]]).with_columns(
        pl.col("outer_fold").cast(pl.Int8), pl.col("behavior_id_pred").cast(pl.Int8)
    )
    frame = evidence_base.join(route_pl, on="pair_id", how="inner")
    if frame.height != evidence_base.height:
        raise RuntimeError(f"{label}: route join lost evidence rows")
    fold_values = set(frame["outer_fold"].unique().to_list())
    if fold_values != set(range(N_FOLDS)):
        raise ValueError(f"{label}: unexpected folds {sorted(fold_values)}")

    X = _matrix(frame, feature_columns)
    y = frame["is_evidence"].to_numpy().astype(np.int8)
    folds = frame["outer_fold"].to_numpy().astype(np.int8)
    true_family = frame["behavior_id_true"].to_numpy().astype(np.int8)
    pred_family = frame["behavior_id_pred"].to_numpy().astype(np.int8)
    global_score = np.zeros(frame.height, dtype=np.float32)
    specialist_score = np.zeros(frame.height, dtype=np.float32)

    print(f"{label}: {frame.height:,} hands, {len(feature_columns)} features", flush=True)
    for fold in range(N_FOLDS):
        tr_mask = folds != fold
        va_mask = folds == fold
        global_model = HistGradientBoostingClassifier(
            max_iter=240, learning_rate=0.045, max_leaf_nodes=40,
            min_samples_leaf=20, l2_regularization=2.0,
            random_state=model_seed + fold,
        )
        global_model.fit(X.loc[tr_mask], y[tr_mask])
        global_score[va_mask] = global_model.predict_proba(X.loc[va_mask])[:, 1]

        for family_id in (1, 2, 3):
            family_train = tr_mask & (true_family == family_id)
            family_valid = va_mask & (pred_family == family_id)
            if not family_valid.any():
                continue
            seed = model_seed + fold
            hgb = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.045, max_leaf_nodes=30,
                min_samples_leaf=15, l2_regularization=2.5,
                random_state=seed,
            )
            lgb = LGBMClassifier(
                n_estimators=180, learning_rate=0.045, num_leaves=31,
                min_child_samples=15, reg_lambda=2.5, random_state=seed,
                n_jobs=-1, verbose=-1,
            )
            xgb = XGBClassifier(
                n_estimators=180, learning_rate=0.045, max_depth=5,
                min_child_weight=2, reg_lambda=2.5, random_state=seed,
                tree_method="hist", n_jobs=-1,
            )
            hgb.fit(X.loc[family_train], y[family_train])
            lgb.fit(X.loc[family_train], y[family_train])
            xgb.fit(X.loc[family_train], y[family_train])
            tree_score = (
                hgb.predict_proba(X.loc[family_valid])[:, 1]
                + lgb.predict_proba(X.loc[family_valid])[:, 1]
                + xgb.predict_proba(X.loc[family_valid])[:, 1]
            ) / 3.0

            train_indices = np.flatnonzero(family_train)
            valid_indices = np.flatnonzero(family_valid)
            train_order = np.lexsort((frame["hand_id"].to_numpy()[train_indices], frame["pair_id"].to_numpy()[train_indices]))
            valid_order = np.lexsort((frame["hand_id"].to_numpy()[valid_indices], frame["pair_id"].to_numpy()[valid_indices]))
            sorted_train = train_indices[train_order]
            sorted_valid = valid_indices[valid_order]
            sorted_pairs = frame["pair_id"].to_numpy()[sorted_train]
            _, group_sizes = np.unique(sorted_pairs, return_counts=True)
            ranker = XGBRanker(
                n_estimators=200, learning_rate=0.035, max_depth=4,
                objective="rank:pairwise", eval_metric="map@5",
                tree_method="hist", random_state=seed, n_jobs=-1,
            )
            ranker.fit(X.iloc[sorted_train], y[sorted_train], group=group_sizes)
            rank_sorted = ranker.predict(X.iloc[sorted_valid])
            rank_score = np.empty(valid_indices.size, dtype=np.float32)
            rank_score[valid_order] = rank_sorted
            specialist_score[family_valid] = 0.90 * tree_score + 0.10 * rank_score
        print(f"{label}: fold {fold + 1}/{N_FOLDS} complete", flush=True)

    blend = 0.75 * specialist_score + 0.25 * global_score
    return {
        "global": global_score,
        "specialized": specialist_score,
        "blend": blend.astype(np.float32),
    }, frame


def _experiment_metrics(
    frame: pl.DataFrame,
    predictions: Mapping[str, np.ndarray],
    truth: Mapping[str, set],
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for engine, scores in predictions.items():
        result = _score_set(frame, scores, truth)
        output[engine] = {
            "map5": result["map5"],
            "by_family": result["by_family"],
        }
    return output


def _persist_oof(
    path: Path,
    frame: pl.DataFrame,
    baseline: Mapping[str, np.ndarray],
    augmented: Mapping[str, np.ndarray],
) -> None:
    frame.select([
        "pair_id", "hand_id", "is_evidence", "behavior_id_true",
        "behavior_id_pred", "outer_fold",
    ]).with_columns(
        pl.Series("baseline_global", baseline["global"]),
        pl.Series("baseline_specialized", baseline["specialized"]),
        pl.Series("baseline_blend", baseline["blend"]),
        pl.Series("augmented_global", augmented["global"]),
        pl.Series("augmented_specialized", augmented["specialized"]),
        pl.Series("augmented_blend", augmented["blend"]),
    ).write_parquet(path, compression="zstd")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    prepared = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    out_dir = root / "outputs" / "poker_collusion" / "lamhuy_sequence_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / "outputs" / "poker_collusion" / "lamhuy_sequence_confirmation" / "ordered_action_state.parquet"
    dev_sequence_path = out_dir / "dev_hand_sequence_features.parquet"
    eval_sequence_path = out_dir / "eval_hand_sequence_features.parquet"
    summary_path = out_dir / "_sequence_evidence_summary.json"

    required = [
        state_path,
        prepared / "dev_pairs.parquet",
        prepared / "eval_pairs.parquet",
        prepared / "dev_hand_features.parquet",
        root / "data" / "poker" / "development_labels.csv",
        root / "data" / "poker" / "development_evidence.csv",
        root / "outputs" / "poker_collusion" / "lamhuy_sequence_specialists" / "oof_rows.parquet",
        root / "outputs" / "poker_collusion" / "lamhuy_sequence_confirmation" / "oof_rows.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    started = time.time()
    sparse_audit = _build_sparse_sequence_features(
        state_path,
        prepared / "dev_pairs.parquet",
        prepared / "eval_pairs.parquet",
        dev_sequence_path,
        eval_sequence_path,
    )
    print(f"Sparse sequence features: {sparse_audit}", flush=True)

    evidence_base, baseline_features, augmented_features, truth = _prepare_evidence_base(
        prepared / "dev_hand_features.parquet",
        dev_sequence_path,
        root / "data" / "poker" / "development_labels.csv",
        root / "data" / "poker" / "development_evidence.csv",
    )
    evidence_audit = {
        "rows": evidence_base.height,
        "pairs": evidence_base["pair_id"].n_unique(),
        "positive_hands": int(evidence_base["is_evidence"].sum()),
        "baseline_features": len(baseline_features),
        "augmented_features": len(augmented_features),
        "new_features": len(augmented_features) - len(baseline_features),
        "sequence_sparse_coverage": float(
            evidence_base.select(
                (pl.sum_horizontal([pl.col(c) for c in SEQUENCE_MODEL_FEATURES]) > 0).mean()
            ).item()
        ),
    }
    if evidence_audit["pairs"] != 372 or evidence_audit["positive_hands"] != 1817:
        raise RuntimeError(f"Evidence training audit failed: {evidence_audit}")
    print(f"Evidence base audit: {evidence_audit}", flush=True)

    experiments = [
        (
            "discovery_seed42",
            root / "outputs" / "poker_collusion" / "lamhuy_sequence_specialists" / "oof_rows.parquet",
            42,
        ),
        (
            "confirmation_seed137",
            root / "outputs" / "poker_collusion" / "lamhuy_sequence_confirmation" / "oof_rows.parquet",
            137,
        ),
    ]
    results: Dict[str, object] = {}
    for name, route_path, seed in experiments:
        baseline_pred, baseline_frame = _run_dual_engine_oof(
            evidence_base, baseline_features, route_path, seed, f"{name}/baseline"
        )
        baseline_metrics = _experiment_metrics(baseline_frame, baseline_pred, truth)
        gc.collect()
        augmented_pred, augmented_frame = _run_dual_engine_oof(
            evidence_base, augmented_features, route_path, seed, f"{name}/augmented"
        )
        if baseline_frame.select(["pair_id", "hand_id"]).equals(
            augmented_frame.select(["pair_id", "hand_id"])
        ) is False:
            raise AssertionError(f"{name}: baseline and augmented rows are not aligned")
        augmented_metrics = _experiment_metrics(augmented_frame, augmented_pred, truth)
        delta = augmented_metrics["blend"]["map5"] - baseline_metrics["blend"]["map5"]
        family_delta = {
            family: augmented_metrics["blend"]["by_family"][family]
            - baseline_metrics["blend"]["by_family"][family]
            for family in BEHAVIORS.values()
        }
        oof_path = out_dir / f"{name}_oof_rows.parquet"
        _persist_oof(oof_path, augmented_frame, baseline_pred, augmented_pred)
        results[name] = {
            "route_path": str(route_path),
            "route_sha256": _sha256(route_path),
            "baseline": baseline_metrics,
            "augmented": augmented_metrics,
            "blend_delta": delta,
            "family_delta": family_delta,
            "oof_path": str(oof_path),
            "oof_sha256": _sha256(oof_path),
        }
        print(
            f"{name}: baseline={baseline_metrics['blend']['map5']:.6f} "
            f"augmented={augmented_metrics['blend']['map5']:.6f} delta={delta:+.6f}",
            flush=True,
        )
        del baseline_pred, augmented_pred, baseline_frame, augmented_frame
        gc.collect()

    discovery = results["discovery_seed42"]
    confirmation = results["confirmation_seed137"]
    all_family_deltas = list(discovery["family_delta"].values()) + list(
        confirmation["family_delta"].values()
    )
    gates = {
        "discovery_material_delta": discovery["blend_delta"] >= MIN_DISCOVERY_DELTA,
        "confirmation_nonregression": confirmation["blend_delta"] >= MIN_CONFIRMATION_DELTA,
        "family_regression_floor": min(all_family_deltas) >= -MAX_FAMILY_REGRESSION,
        "baseline_reproduction_plausible": min(
            discovery["baseline"]["blend"]["map5"],
            confirmation["baseline"]["blend"]["map5"],
        ) >= V30_EVIDENCE_MAP5 - 0.05,
    }
    summary = {
        "recipe_id": "lamhuy_v30_hand_sequence_evidence_v1",
        "note": "BUILD + MEASURE ONLY; no Kaggle submission was made.",
        "v30_reference_evidence_map5": V30_EVIDENCE_MAP5,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "sparse_features": {
            **sparse_audit,
            "dev_path": str(dev_sequence_path),
            "dev_sha256": _sha256(dev_sequence_path),
            "eval_path": str(eval_sequence_path),
            "eval_sha256": _sha256(eval_sequence_path),
        },
        "evidence_audit": evidence_audit,
        "sequence_model_features": SEQUENCE_MODEL_FEATURES,
        "experiments": results,
        "duration_s": time.time() - started,
        "next_step": (
            "Train full augmented production evidence models and rerank top-risk eval pairs."
            if all(gates.values())
            else "Reject this sequence evidence block; preserve current V30 evidence."
        ),
    }
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary), indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
