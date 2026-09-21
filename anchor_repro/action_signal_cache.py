"""Cache per-(hand, player, street) action signals ONCE — the expensive 18.6M-action groupby.

The street-template / sequence experiments kept re-grouping the full 18.6M-row action stream by
(hand, player, street) on every run (minutes each, CPU polars, no GPU engine available on Windows).
That groupby is PAIR-INDEPENDENT: per (hand, player, street) action counts don't depend on which
pair we're scoring. So we materialize it ONCE for ALL hands, then any pair-level template feature
becomes a cheap join + role-tag (seconds).

Output: `action_signals.parquet` — per (hand_id, player_id, street): agg/check/fold/call/allin
counts, first/last action ordinal, to_call severity. ~ (players x hands x streets) rows, cached.

Run:  python -m anchor_repro.action_signal_cache
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
OUT = CACHE / "action_signals.parquet"


def build():
    t = time.time()
    a = pl.scan_parquet(D / "actions.parquet")
    sig = a.group_by(["hand_id", "player_id", "street"]).agg(
        (pl.col("action").is_in(["bet", "raise", "all_in"])).sum().alias("n_agg"),
        (pl.col("action") == "check").sum().alias("n_check"),
        (pl.col("action") == "fold").sum().alias("n_fold"),
        (pl.col("action") == "call").sum().alias("n_call"),
        (pl.col("action") == "all_in").sum().alias("n_allin"),
        (pl.col("action") == "raise").sum().alias("n_raise"),
        (pl.col("action") == "bet").sum().alias("n_bet"),
        pl.len().alias("n_act"),
        pl.col("action_no").min().alias("first_no"),
        pl.col("action_no").max().alias("last_no"),
        (pl.col("to_call") / pl.col("pot_before").clip(1, None)).max().alias("max_tc_ratio"),
        (pl.col("amount") / pl.col("pot_before").clip(1, None)).max().alias("max_bet_pot"),
    )
    sig.sink_parquet(OUT, compression="zstd")
    n = pl.scan_parquet(OUT).select(pl.len()).collect().item()
    print(f"action_signals cached: {n:,} (hand,player,street) rows ({time.time()-t:.0f}s) -> {OUT}")


def main() -> int:
    if OUT.is_file() and pl.scan_parquet(OUT).select(pl.len()).collect().item() > 0:
        print(f"already cached: {OUT}")
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
