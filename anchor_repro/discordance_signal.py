"""DISCORDANCE signal — the LITERAL collusion mechanism, never built before.

Collusion = acting AGAINST your own cards for the partner's benefit:
  * soft-play / chip-dump: STRONG true hand played PASSIVELY (check/call/fold) when partner is in pot
  * loose-commit: WEAK true hand played AGGRESSIVELY / committed because partner is driving

We now have TRUE hand strength (hand_eval) on the 659k showdown seats. This measures, per player,
how their action-vs-strength behaves WITH the partner in the pot vs WITHOUT (their field baseline).
A colluder shows a DIRECTED discordance: unusually passive-with-strength (or loose-with-weakness)
specifically when the partner benefits. Ordinary players show the same action-vs-strength curve to
everyone -> near-zero directed discordance.

Pipeline:
 1. Cache true_strength per showdown seat (hand_eval over seats+board).
 2. Per (seat, hand): aggression proxy from actions (n_agg, folded, went_to_showdown, contribution).
    discordance_passive = true_strength * (1 - aggression)   [strong but passive]
    discordance_loose   = (1 - true_strength) * committed     [weak but committed]
 3. Per player: baseline discordance over ALL their showdown hands (field).
 4. Per pair: mean discordance in shared showdown hands MINUS each member's field baseline
    = DIRECTED discordance (the tell). Plus who-benefits alignment (does the passive member's
    loss go to the partner?).

Then train pair risk on these directed-discordance features. Judge eval-honest vs 0.709.
Run:  python -m anchor_repro.discordance_signal   [--build-cache]
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import polars as pl

from anchor_repro.hand_eval_gpu import strength_for_seats
from anchor_repro.eval_honest_harness import CACHE, D

OUT = CACHE / "discordance"
OUT.mkdir(parents=True, exist_ok=True)
STRENGTH_CACHE = OUT / "seat_strength.parquet"


def build_strength_cache():
    """Evaluate true hand strength for every SHOWDOWN seat (cards revealed)."""
    seats = pl.read_parquet(D / "seats.parquet").filter(pl.col("went_to_showdown") == True)
    hands = pl.read_parquet(D / "hands.parquet").select(["hand_id", "board_cards"])
    df = seats.join(hands, on="hand_id", how="left").select(
        ["hand_id", "player_id", "hole_card_1", "hole_card_2", "board_cards",
         "net_chips", "total_contribution", "folded", "won_share"])
    print(f"evaluating {df.height:,} showdown seats (GPU)...")
    t = time.time()
    # build card lists: [hole1, hole2, *board]
    h1 = df["hole_card_1"].to_list(); h2 = df["hole_card_2"].to_list()
    boards = df["board_cards"].to_list()
    card_lists = [[a, b] + ((bd.split() if bd else [])) for a, b, bd in zip(h1, h2, boards)]
    strength = strength_for_seats(card_lists)
    df = df.with_columns(pl.Series("true_strength", strength))
    print(f"  done ({time.time()-t:.0f}s). strength dist: "
          f"p50={df['true_strength'].median():.3f} p90={df['true_strength'].quantile(0.9):.3f}")
    df.select(["hand_id", "player_id", "true_strength", "net_chips",
               "total_contribution", "won_share"]).write_parquet(STRENGTH_CACHE)
    print(f"  wrote {STRENGTH_CACHE}")


def main() -> int:
    if "--build-cache" in sys.argv or not STRENGTH_CACHE.exists():
        build_strength_cache()
    if "--cache-only" in sys.argv:
        return 0
    # (discordance feature build + model in the next step, after cache is verified)
    st = pl.read_parquet(STRENGTH_CACHE)
    print(f"strength cache: {st.height:,} showdown seats, cols {st.columns}")
    print(f"strong-hand rate (strength>0.5): {float((st['true_strength']>0.5).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
