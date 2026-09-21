"""WORK BACKWARDS from the 1,817 planted evidence hands — let them define their own signature.

Every forward-proposed tell (discordance, consistency, sequence, joint-advantage) came back
near-chance. Invert: take the KNOWN planted hands and ask what makes them different from the SAME
colluding pairs' OTHER (non-planted) shared hands. The organizers flagged THESE specific hands as
the evidence — so whatever separates them IS the tell, including signatures we'd never propose.

CRITICAL CONTROL: compare planted vs the same pairs' NON-planted shared hands (within-pair), not vs
random hands. This isolates the planting signature from the pair's baseline behavior.

Profiles EVERY raw dimension per (pair, shared hand):
  hand strength (true, GPU-eval both members), action (agg/fold/check/call per member), outcome
  (net, contribution, won_share, opposite-flow), board (n board cards / street reached, pot size),
  table context (players_active, players_dealt, n_folded), position (seat vs button), timing.
Ranks each by planted-vs-nonplanted separation (AUC + effect size) over the within-pair contrast.

Run:  python -m anchor_repro.backwards_planted
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from anchor_repro.hand_eval_gpu import strength_for_seats
from anchor_repro.eval_honest_harness import CACHE, D

OUT = CACHE / "backwards"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    t = time.time()
    labels = pl.read_csv(D / "development_labels.csv").filter(pl.col("label") == 1).select(
        ["pair_id", "player_1", "player_2"])
    evidence = pl.read_csv(D / "development_evidence.csv").select(["pair_id", "hand_id"]).unique()
    planted_keys = set((r["pair_id"], r["hand_id"]) for r in evidence.iter_rows(named=True))

    # all shared hands of the 372 colluding pairs (from the V30 universe = their shared hands)
    V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
    uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"]).unique()
    shared = uni.join(labels.lazy(), on="pair_id", how="inner").collect()
    print(f"colluding-pair shared hands: {shared.height:,} ({time.time()-t:.0f}s)")

    # --- join raw dimensions ---
    seats = pl.read_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "seat_no", "starting_stack", "total_contribution", "net_chips",
         "folded", "went_to_showdown", "won_share", "hole_card_1", "hole_card_2"])
    hands = pl.read_parquet(D / "hands.parquet").select(
        ["hand_id", "board_cards", "final_pot", "big_blind", "button_seat", "players_dealt",
         "players_at_showdown"])
    acts = pl.scan_parquet(D / "actions.parquet").group_by(["hand_id", "player_id"]).agg(
        (pl.col("action").is_in(["bet", "raise", "all_in"])).sum().alias("n_agg"),
        (pl.col("action") == "fold").sum().alias("n_fold"),
        (pl.col("action") == "check").sum().alias("n_check"),
        (pl.col("action") == "call").sum().alias("n_call"),
        pl.len().alias("n_act"),
    ).collect()

    def member_side(col_player):
        s = shared.join(labels.select(["pair_id", col_player]).rename({col_player: "player_id"}),
                        on="pair_id").join(seats, on=["hand_id", "player_id"], how="left").join(
                        acts, on=["hand_id", "player_id"], how="left").join(hands, on="hand_id", how="left")
        return s

    a = member_side("player_1"); b = member_side("player_2")
    # true strength for both members (GPU) — only where showdown cards present
    def strength(df):
        h1 = df["hole_card_1"].to_list(); h2 = df["hole_card_2"].to_list(); bd = df["board_cards"].to_list()
        cl = [[x, y] + (z.split() if z else []) for x, y, z in zip(h1, h2, bd)]
        return strength_for_seats(cl)
    a = a.with_columns(pl.Series("strength", strength(a)))
    b = b.with_columns(pl.Series("strength", strength(b)))

    # combine both members per (pair,hand)
    keep_a = ["pair_id", "hand_id", "seat_no", "total_contribution", "net_chips", "folded",
              "went_to_showdown", "won_share", "n_agg", "n_fold", "n_check", "n_call", "n_act",
              "strength", "board_cards", "final_pot", "big_blind", "button_seat", "players_dealt",
              "players_at_showdown"]
    A = a.select(keep_a)
    B = b.select(["pair_id", "hand_id", "seat_no", "total_contribution", "net_chips", "folded",
                  "went_to_showdown", "won_share", "n_agg", "n_fold", "n_check", "n_call", "n_act", "strength"])
    df = A.join(B, on=["pair_id", "hand_id"], suffix="_b")

    # planted label
    df = df.with_columns(
        pl.struct(["pair_id", "hand_id"]).map_elements(
            lambda s: 1 if (s["pair_id"], s["hand_id"]) in planted_keys else 0,
            return_dtype=pl.Int8).alias("planted")
    )
    print(f"contrast set: {df.height:,} hands, planted={int(df['planted'].sum())}")

    # --- derived raw dimensions ---
    bb = pl.col("big_blind")
    df = df.with_columns(
        (pl.col("final_pot") / bb).alias("pot_bb"),
        (pl.col("total_contribution") / bb).alias("a_contrib_bb"),
        (pl.col("total_contribution_b") / bb).alias("b_contrib_bb"),
        (pl.col("net_chips") / bb).alias("a_net_bb"),
        (pl.col("net_chips_b") / bb).alias("b_net_bb"),
        (pl.col("board_cards").str.split(" ").list.len()).alias("n_board"),
        (pl.col("n_agg") + pl.col("n_agg_b")).alias("pair_agg"),
        (pl.col("n_agg") - pl.col("n_agg_b")).abs().alias("agg_gap"),
        (pl.col("strength") - pl.col("strength_b")).abs().alias("strength_gap"),
        pl.max_horizontal("strength", "strength_b").alias("max_strength"),
        pl.min_horizontal("strength", "strength_b").alias("min_strength"),
        ((pl.col("net_chips") * pl.col("net_chips_b")) < 0).cast(pl.Int8).alias("opposite_flow"),
        (pl.col("net_chips").abs() + pl.col("net_chips_b").abs()).alias("abs_flow"),
        (pl.col("won_share") + pl.col("won_share_b")).alias("pair_won"),
        (pl.col("went_to_showdown").cast(pl.Int8) + pl.col("went_to_showdown_b").cast(pl.Int8)).alias("n_showdown"),
        (pl.col("folded").cast(pl.Int8) + pl.col("folded_b").cast(pl.Int8)).alias("n_folded"),
        # who wins gets how much of pot; loser contributes
        (pl.max_horizontal("net_chips", "net_chips_b") / (pl.col("final_pot") + 1)).alias("winner_pot_share"),
    )

    dims = ["pot_bb", "a_contrib_bb", "b_contrib_bb", "a_net_bb", "b_net_bb", "n_board",
            "pair_agg", "agg_gap", "strength_gap", "max_strength", "min_strength", "strength", "strength_b",
            "opposite_flow", "abs_flow", "pair_won", "n_showdown", "n_folded", "winner_pot_share",
            "n_act", "n_act_b", "n_fold", "n_fold_b", "n_check", "n_check_b", "players_dealt",
            "players_at_showdown", "final_pot", "seat_no", "seat_no_b"]

    # --- GLOBAL separation (planted vs all non-planted of colluding pairs) ---
    y = df["planted"].to_numpy()
    print(f"\n=== PLANTED vs NON-PLANTED separation (within colluding pairs) — ranked ===")
    print(f"{'dimension':<20}{'AUC':>8}{'planted_mean':>15}{'nonplanted_mean':>17}")
    rows = []
    for d in dims:
        v = df[d].to_numpy().astype(float)
        v = np.nan_to_num(v)
        try:
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)
        except Exception:
            continue
        pm = v[y == 1].mean(); npm = v[y == 0].mean()
        rows.append((d, auc, pm, npm))
    for d, auc, pm, npm in sorted(rows, key=lambda r: -r[1]):
        print(f"  {d:<20}{auc:>8.4f}{pm:>15.3f}{npm:>17.3f}")

    df.write_parquet(OUT / "planted_contrast.parquet")
    (OUT / "_separation.json").write_text(json.dumps(
        {d: {"auc": auc, "planted_mean": pm, "nonplanted_mean": npm} for d, auc, pm, npm in rows},
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/planted_contrast.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
