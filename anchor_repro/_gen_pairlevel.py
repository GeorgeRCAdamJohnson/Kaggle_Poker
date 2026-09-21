"""Pair-level reconstruction: can a simple count of qualifying interaction hands
separate the 372 positive pairs from the 1,488 confirmed negatives?

This is the decisive test. The LB rewards pair-level PairAP (70%). Build, for ALL
labeled pairs (pos + neg), the count/rate of shared hands satisfying candidate
generator predicates, then measure separation (AP / AUC-like) vs the label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

D = Path("data/poker")
OUT = Path("outputs/poker_collusion/generator_hunt")


def build_all_labeled_shared() -> pl.DataFrame:
    labels = pl.read_csv(D / "development_labels.csv").select(
        ["pair_id", "player_1", "player_2", "label", "behavior_family"]
    )
    seats = pl.scan_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "total_contribution", "net_chips",
         "folded", "went_to_showdown"]
    )
    hands = pl.scan_parquet(D / "hands.parquet").select(
        ["hand_id", "big_blind", "final_pot", "players_dealt"]
    )
    lab = labels.lazy()
    a = (
        seats.join(lab.select(["pair_id", "player_1", "player_2", "label", "behavior_family"]),
                   left_on="player_id", right_on="player_1", how="inner")
        .rename({c: f"a_{c}" for c in ["total_contribution", "net_chips", "folded", "went_to_showdown"]})
        .rename({"player_id": "player_a"})
    )
    b = seats.rename({c: f"b_{c}" for c in ["total_contribution", "net_chips", "folded", "went_to_showdown"]}).rename({"player_id": "player_b"})
    shared = (
        a.join(b, on="hand_id", how="inner")
        .filter(pl.col("player_b") == pl.col("player_2"))
        .join(hands, on="hand_id", how="left")
        .with_columns(
            (pl.col("a_net_chips") / pl.col("big_blind")).alias("a_net_bb"),
            (pl.col("b_net_chips") / pl.col("big_blind")).alias("b_net_bb"),
            (pl.col("a_total_contribution") / pl.col("big_blind")).alias("a_contrib_bb"),
            (pl.col("b_total_contribution") / pl.col("big_blind")).alias("b_contrib_bb"),
            (pl.col("final_pot") / pl.col("big_blind")).alias("pot_bb"),
        )
        .with_columns(
            ((pl.col("a_contrib_bb") > 1.0) & (pl.col("b_contrib_bb") > 1.0)).alias("both_commit"),
            (pl.col("a_net_bb").abs() + pl.col("b_net_bb").abs()).alias("abs_flow_bb"),
            (pl.min_horizontal("a_net_bb", "b_net_bb") <= -3.0).alias("a_or_b_loses3"),
            ((pl.col("a_went_to_showdown")) & (pl.col("b_went_to_showdown"))).alias("both_sd"),
        )
    )
    return shared.collect()


def main() -> int:
    shared = build_all_labeled_shared()
    print(f"all-labeled shared (pair,hand) rows={shared.height}")

    # Per-pair aggregates.
    agg = shared.group_by("pair_id").agg(
        pl.first("label").alias("label"),
        pl.first("behavior_family").alias("behavior_family"),
        pl.len().alias("shared_hands"),
        pl.col("both_commit").sum().alias("n_both_commit"),
        (pl.col("both_commit") & (pl.col("abs_flow_bb") >= 5.0)).sum().alias("n_commit_flow5"),
        (pl.col("both_commit") & (pl.col("abs_flow_bb") >= 10.0)).sum().alias("n_commit_flow10"),
        pl.col("both_sd").sum().alias("n_both_sd"),
        pl.col("abs_flow_bb").sum().alias("tot_abs_flow"),
        pl.col("abs_flow_bb").max().alias("max_abs_flow"),
    ).with_columns(
        (pl.col("n_both_commit") / pl.col("shared_hands")).alias("rate_both_commit"),
        (pl.col("n_commit_flow5") / pl.col("shared_hands")).alias("rate_commit_flow5"),
    )

    # Also fold in ALL labeled pairs (some negatives may have zero shared hands -> absent).
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    full = labels.join(agg.drop("label"), on="pair_id", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in
         ["shared_hands", "n_both_commit", "n_commit_flow5", "n_commit_flow10",
          "n_both_sd", "tot_abs_flow", "max_abs_flow", "rate_both_commit", "rate_commit_flow5"]]
    )
    y = full["label"].to_numpy()
    print(f"pairs={full.height} positives={int(y.sum())} negatives={int((y==0).sum())}")
    print(f"negatives with 0 shared hands: {int(full.filter((pl.col('label')==0)&(pl.col('shared_hands')==0)).height)}")
    print(f"positives with 0 shared hands: {int(full.filter((pl.col('label')==1)&(pl.col('shared_hands')==0)).height)}")

    print("\n=== PAIR-LEVEL separation of positive vs negative (AP / AUC) ===")
    for feat in ["n_both_commit", "n_commit_flow5", "n_commit_flow10", "n_both_sd",
                 "tot_abs_flow", "max_abs_flow", "rate_both_commit", "rate_commit_flow5", "shared_hands"]:
        v = full[feat].to_numpy().astype(float)
        if np.unique(v).size < 2:
            continue
        ap = average_precision_score(y, v)
        auc = roc_auc_score(y, v)
        print(f"  {feat:20s} AP={ap:.4f} AUC={auc:.4f}")

    # Base rate for AP reference.
    print(f"\nbase positive rate = {y.mean():.4f} (AP of random)")

    # Exact-threshold probe: is there a count threshold that cleanly splits?
    print("\n=== threshold sweep on n_commit_flow5 ===")
    v = full["n_commit_flow5"].to_numpy()
    for t in [1, 2, 3, 5, 8, 10, 15, 20]:
        pred = (v >= t)
        tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / int(y.sum())
        print(f"  n_commit_flow5>={t:3d}: flagged={int(pred.sum()):5d} precision={prec:.3f} recall={rec:.3f}")

    full.write_parquet(OUT / "pair_level_generator.parquet")
    print(f"\nwrote {OUT / 'pair_level_generator.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
