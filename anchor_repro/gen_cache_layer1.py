"""Layer-1 permanent cache for generator recovery.

ONE pass over the raw action stream (18.6M rows) + seats (12M) + hands (2M), producing
reusable, label-free, pair-independent aggregates that NEVER need rebuilding when we change
pair-level features:

  * per (hand_id, player_id): action structure (aggression, folds, checks, calls, all-ins,
    per-street counts, to-call severity, bet sizing) + seat outcome (contribution, net,
    showdown, hole strength).
  * per hand_id: table context (n seats, n folded, n showdown, total/max contribution,
    pot, blinds, players dealt/at showdown, board) — used to derive OUTSIDER behavior.

Layer 2 (a separate cheap module) joins a pair's two members to these caches and derives
pair-level features in seconds, restricted to the competition's 9.65M shared-hand universe.

Idempotent: skips work if the cache parquets already exist and are non-empty.

Run:  python -m anchor_repro.gen_cache_layer1
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
HAND_PLAYER = CACHE / "hand_player_agg.parquet"
HAND_CTX = CACHE / "hand_context.parquet"

_RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def _hole_strength(c1: str, c2: str) -> pl.Expr:
    r1 = pl.col(c1).str.slice(0, 1).replace_strict(_RANKS, default=0).cast(pl.Int8)
    r2 = pl.col(c2).str.slice(0, 1).replace_strict(_RANKS, default=0).cast(pl.Int8)
    hi = pl.max_horizontal(r1, r2)
    lo = pl.min_horizontal(r1, r2)
    suited = pl.col(c1).str.slice(1, 1) == pl.col(c2).str.slice(1, 1)
    pair = r1 == r2
    return (
        (hi.cast(pl.Float32) / 14.0) * 0.5
        + (lo.cast(pl.Float32) / 14.0) * 0.2
        + pair.cast(pl.Float32) * 0.25
        + suited.cast(pl.Float32) * 0.05
    )


def build_hand_player_agg() -> None:
    """Per (hand_id, player_id): action structure joined to seat outcome + hole strength."""
    acts = (
        pl.scan_parquet(D / "actions.parquet")
        .with_columns(
            pl.col("action").is_in(["bet", "raise"]).alias("_agg"),
            (pl.col("action") == "all_in").alias("_allin"),
            (pl.col("action") == "fold").alias("_fold"),
            (pl.col("action") == "check").alias("_check"),
            (pl.col("action") == "call").alias("_call"),
            (pl.col("to_call") / pl.col("pot_before").clip(1, None)).alias("_tc_ratio"),
            (pl.col("street") != "preflop").alias("_postflop"),
            pl.col("street").is_in(["turn", "river"]).alias("_late"),
        )
        .group_by(["hand_id", "player_id"])
        .agg(
            pl.len().alias("n_actions"),
            pl.col("_agg").sum().alias("n_agg"),
            pl.col("_allin").sum().alias("n_allin"),
            pl.col("_fold").sum().alias("n_fold"),
            pl.col("_check").sum().alias("n_check"),
            pl.col("_call").sum().alias("n_call"),
            (pl.col("_agg") & pl.col("_postflop")).sum().alias("n_postflop_agg"),
            (pl.col("_check") & pl.col("_late")).sum().alias("n_late_check"),
            (pl.col("_agg") & ~pl.col("_postflop")).sum().alias("n_preflop_agg"),
            pl.col("_tc_ratio").max().alias("max_tc_ratio"),
            pl.col("amount").max().alias("max_amount"),
            pl.col("amount").sum().alias("sum_amount"),
            pl.col("street").n_unique().alias("n_streets"),
            pl.col("players_active").min().alias("min_players_active"),
        )
    )
    seats = (
        pl.scan_parquet(D / "seats.parquet")
        .select(["hand_id", "player_id", "seat_no", "hole_card_1", "hole_card_2",
                 "starting_stack", "total_contribution", "net_chips", "folded",
                 "went_to_showdown", "won_share"])
        .with_columns(_hole_strength("hole_card_1", "hole_card_2").alias("hs"))
    )
    out = seats.join(acts, on=["hand_id", "player_id"], how="left").with_columns(
        [pl.col(c).fill_null(0) for c in
         ["n_actions", "n_agg", "n_allin", "n_fold", "n_check", "n_call",
          "n_postflop_agg", "n_late_check", "n_preflop_agg", "max_tc_ratio",
          "max_amount", "sum_amount", "n_streets"]]
    )
    out.sink_parquet(HAND_PLAYER, compression="zstd")


def build_hand_context() -> None:
    """Per hand_id: table-wide context (all players) + hand metadata."""
    seat_ctx = (
        pl.scan_parquet(D / "seats.parquet")
        .group_by("hand_id")
        .agg(
            pl.len().alias("t_n_seats"),
            pl.col("folded").sum().alias("t_n_folded"),
            pl.col("went_to_showdown").sum().alias("t_n_showdown"),
            pl.col("total_contribution").sum().alias("t_total_contrib"),
            pl.col("total_contribution").max().alias("t_max_contrib"),
            pl.col("net_chips").max().alias("t_max_net"),
            pl.col("net_chips").min().alias("t_min_net"),
        )
    )
    hand_meta = pl.scan_parquet(D / "hands.parquet").select(
        ["hand_id", "table_id", "big_blind", "small_blind", "final_pot",
         "players_dealt", "players_at_showdown"]
    )
    seat_ctx.join(hand_meta, on="hand_id", how="left").sink_parquet(HAND_CTX, compression="zstd")


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)

    if HAND_PLAYER.is_file() and pl.scan_parquet(HAND_PLAYER).select(pl.len()).collect().item() > 0:
        print(f"hand_player_agg cached: {HAND_PLAYER}")
    else:
        t = time.time()
        build_hand_player_agg()
        n = pl.scan_parquet(HAND_PLAYER).select(pl.len()).collect().item()
        print(f"hand_player_agg built: {n:,} rows ({time.time()-t:.1f}s) -> {HAND_PLAYER}")

    if HAND_CTX.is_file() and pl.scan_parquet(HAND_CTX).select(pl.len()).collect().item() > 0:
        print(f"hand_context cached: {HAND_CTX}")
    else:
        t = time.time()
        build_hand_context()
        n = pl.scan_parquet(HAND_CTX).select(pl.len()).collect().item()
        print(f"hand_context built: {n:,} rows ({time.time()-t:.1f}s) -> {HAND_CTX}")

    # quick audit
    hp = pl.scan_parquet(HAND_PLAYER)
    hc = pl.scan_parquet(HAND_CTX)
    print("hand_player cols:", hp.collect_schema().names())
    print("hand_context cols:", hc.collect_schema().names())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
