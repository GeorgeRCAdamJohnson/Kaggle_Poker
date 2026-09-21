"""Generator hunt: within positive pairs, what EXACTLY marks the planted evidence hands?

Builds per-(pair, hand) records for every SHARED hand of every positive pair (both
members dealt in), tags which are planted (in development_evidence.csv), and compares
planted vs non-planted on candidate generator predicates per family. A near-deterministic
planting rule should show up as an exact / near-exact separator on the evidence hands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

D = Path("data/poker")


def main() -> int:
    labels = pl.read_csv(D / "development_labels.csv")
    evidence = pl.read_csv(D / "development_evidence.csv")
    pos = labels.filter(pl.col("label") == 1).select(
        ["pair_id", "player_1", "player_2", "behavior_family"]
    )
    print(f"positive pairs={pos.height}, evidence rows={evidence.height}")

    seats = pl.scan_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "seat_no", "hole_card_1", "hole_card_2",
         "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"]
    )
    hands = pl.scan_parquet(D / "hands.parquet").select(
        ["hand_id", "table_id", "big_blind", "final_pot", "players_dealt", "players_at_showdown", "board_cards"]
    )

    # Shared hands for each positive pair: both members appear in seats for the hand.
    seats_p1 = seats.rename({c: f"p1_{c}" for c in seats.collect_schema().names() if c != "hand_id"})
    seats_p2 = seats.rename({c: f"p2_{c}" for c in seats.collect_schema().names() if c != "hand_id"})

    pos_lf = pos.lazy()
    s1 = seats.rename({"player_id": "player_1"}).join(pos_lf, on="player_1")
    # For each (pair, hand) where player_1 present, require player_2 also present in same hand.
    p1_rows = (
        seats.join(pos_lf.select(["pair_id", "player_1", "player_2", "behavior_family"]),
                   left_on="player_id", right_on="player_1", how="inner")
        .rename({c: f"a_{c}" for c in ["seat_no", "hole_card_1", "hole_card_2", "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"]})
        .rename({"player_id": "player_a"})
    )
    p2_rows = (
        seats.rename({c: f"b_{c}" for c in ["seat_no", "hole_card_1", "hole_card_2", "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"]})
        .rename({"player_id": "player_b"})
    )
    shared = (
        p1_rows.join(p2_rows, on="hand_id", how="inner")
        .filter(pl.col("player_b") == pl.col("player_2"))
        .join(hands, on="hand_id", how="left")
    )

    ev_keys = evidence.select(["pair_id", "hand_id", "evidence_rank"]).lazy()
    shared = shared.join(ev_keys, on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("evidence_rank").is_not_null().alias("is_planted")
    )

    df = shared.collect()
    print(f"shared (pair,hand) rows={df.height}, planted={int(df['is_planted'].sum())}")
    # sanity: planted count should be close to 1817 (some evidence hands may not have both seats?)
    ev_matched = df.filter(pl.col("is_planted")).height
    print(f"evidence rows matched into shared table: {ev_matched} / {evidence.height}")

    # Derived per-hand pair quantities
    df = df.with_columns(
        (pl.col("a_net_chips") / pl.col("big_blind")).alias("a_net_bb"),
        (pl.col("b_net_chips") / pl.col("big_blind")).alias("b_net_bb"),
        (pl.col("a_total_contribution") / pl.col("big_blind")).alias("a_contrib_bb"),
        (pl.col("b_total_contribution") / pl.col("big_blind")).alias("b_contrib_bb"),
        (pl.col("final_pot") / pl.col("big_blind")).alias("pot_bb"),
    ).with_columns(
        # directed transfer between the two: one loses to the other in the same hand
        (pl.col("a_net_bb") * pl.col("b_net_bb")).alias("net_product"),  # <0 => opposite signs
        (pl.col("a_net_bb").abs() + pl.col("b_net_bb").abs()).alias("abs_flow_bb"),
        (pl.col("a_folded").cast(pl.Int8) + pl.col("b_folded").cast(pl.Int8)).alias("n_folded"),
        (pl.col("a_went_to_showdown").cast(pl.Int8) + pl.col("b_went_to_showdown").cast(pl.Int8)).alias("n_showdown"),
        pl.min_horizontal("a_contrib_bb", "b_contrib_bb").alias("min_contrib_bb"),
    )

    print("\n=== PLANTED vs NON-PLANTED (within positive pairs), by family ===")
    feats = ["abs_flow_bb", "net_product", "pot_bb", "min_contrib_bb", "n_folded", "n_showdown",
             "a_contrib_bb", "b_contrib_bb"]
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        sub = df.filter(pl.col("behavior_family") == fam)
        pl_hands = sub.filter(pl.col("is_planted"))
        np_hands = sub.filter(~pl.col("is_planted"))
        print(f"\n--- {fam}: shared={sub.height} planted={pl_hands.height} nonplanted={np_hands.height} ---")
        for f in feats:
            pv = pl_hands[f].drop_nulls().to_numpy()
            nv = np_hands[f].drop_nulls().to_numpy()
            if len(pv) == 0 or len(nv) == 0:
                continue
            print(f"  {f:16s} planted[med={np.median(pv):8.3f} mean={pv.mean():8.3f}] "
                  f"nonplanted[med={np.median(nv):8.3f} mean={nv.mean():8.3f}]")

    # Save the labeled table for the predicate search step.
    out = Path("outputs/poker_collusion/generator_hunt")
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "positive_shared_hands.parquet")
    print(f"\nwrote {out / 'positive_shared_hands.parquet'} ({df.height} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
