"""Cache per-(hand, player) seat outcome + TRUE hole strength ONCE (GPU eval), for ALL hands.

The template/sequence experiments recomputed hole-card strength on every run. This materializes it
ONCE for ALL seats (not just showdown — we compute strength for every seat's hole cards vs board;
non-showdown seats still have hole cards in seats.parquet). Combined with action_signals.parquet,
the full per-hand substrate is cached and any pair feature is a cheap join.

Output: `seat_full.parquet` — per (hand_id, player_id): seat_no, net_chips, total_contribution,
folded, went_to_showdown, won_share, true_strength, + hand context (board, pot, bb, button).

Run:  python -m anchor_repro.seat_strength_cache
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from anchor_repro.hand_eval_gpu import strength_for_seats

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
OUT = CACHE / "seat_full.parquet"
BATCH = 2_000_000


def build():
    t = time.time()
    seats = pl.read_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "seat_no", "net_chips", "total_contribution", "folded",
         "went_to_showdown", "won_share", "hole_card_1", "hole_card_2"])
    hands = pl.read_parquet(D / "hands.parquet").select(
        ["hand_id", "board_cards", "final_pot", "big_blind", "button_seat"])
    df = seats.join(hands, on="hand_id", how="left")
    print(f"seats: {df.height:,} rows; evaluating strength on GPU in batches...")
    strengths = np.empty(df.height, np.float32)
    h1 = df["hole_card_1"].to_list(); h2 = df["hole_card_2"].to_list(); bd = df["board_cards"].to_list()
    for start in range(0, df.height, BATCH):
        end = min(start + BATCH, df.height)
        cl = [[a, b] + (z.split() if z else []) for a, b, z in
              zip(h1[start:end], h2[start:end], bd[start:end])]
        strengths[start:end] = strength_for_seats(cl)
        print(f"  strength batch {start//BATCH+1}/{-(-df.height//BATCH)} ({time.time()-t:.0f}s)", flush=True)
    df = df.with_columns(pl.Series("true_strength", strengths)).drop(["hole_card_1", "hole_card_2"])
    df.write_parquet(OUT, compression="zstd")
    print(f"seat_full cached: {df.height:,} rows ({time.time()-t:.0f}s) -> {OUT}")


def main() -> int:
    if OUT.is_file() and pl.scan_parquet(OUT).select(pl.len()).collect().item() > 0:
        print(f"already cached: {OUT}")
        return 0
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
