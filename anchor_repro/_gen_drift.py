"""Is the generator-style pair signal TRANSPORT-CLEAN (dev==eval distribution)?

V30 hits 0.98 dev PairAP but ~0.79 eval (transfer collapse). The leaders at ~0.89 get
rich signal to TRANSFER. Hypothesis: simple, label-free interaction counts (both-commit /
flow / showdown rates) are far more transport-stable than V30's engineered features, so
even though they separate WEAKER on dev (AUC 0.84), they may hold their rank on eval.

Build the SAME pair-level generator features for dev AND eval via one shared function,
then run the adversarial drift gate (dev-vs-eval AUC; <0.65 = transport-clean).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from anchor_repro.own_submission_recipes import adversarial_drift_auc_matrix

D = Path("data/poker")
GEN_FEATURES = [
    "shared_hands", "n_both_commit", "n_commit_flow5", "n_commit_flow10",
    "n_both_sd", "tot_abs_flow", "max_abs_flow", "rate_both_commit", "rate_commit_flow5",
]


def build_pair_generator(pair_frame: pl.LazyFrame) -> pl.DataFrame:
    """Shared builder: pair_frame must have pair_id, player_1, player_2. Same code dev+eval.

    Memory-safe: build a (player, pair, role) long map, join seats to it ONCE to tag each
    seat row with the pairs+role it belongs to, then self-join the tagged rows on
    (hand_id, pair_id) so only genuine co-occurrences survive (no player x player blowup).
    """
    seats = pl.scan_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "total_contribution", "net_chips", "went_to_showdown"]
    )
    hands = pl.scan_parquet(D / "hands.parquet").select(["hand_id", "big_blind", "final_pot"])
    pf = pair_frame.select(["pair_id", "player_1", "player_2"])
    a_map = pf.select(["pair_id", pl.col("player_1").alias("player_id")])
    b_map = pf.select(["pair_id", pl.col("player_2").alias("player_id")])
    a = (
        seats.join(a_map, on="player_id", how="inner")
        .rename({c: f"a_{c}" for c in ["total_contribution", "net_chips", "went_to_showdown"]})
        .drop("player_id")
    )
    b = (
        seats.join(b_map, on="player_id", how="inner")
        .rename({c: f"b_{c}" for c in ["total_contribution", "net_chips", "went_to_showdown"]})
        .drop("player_id")
    )
    shared = (
        a.join(b, on=["hand_id", "pair_id"], how="inner")
        .join(hands, on="hand_id", how="left")
        .with_columns(
            (pl.col("a_net_chips") / pl.col("big_blind")).alias("a_net_bb"),
            (pl.col("b_net_chips") / pl.col("big_blind")).alias("b_net_bb"),
            (pl.col("a_total_contribution") / pl.col("big_blind")).alias("a_contrib_bb"),
            (pl.col("b_total_contribution") / pl.col("big_blind")).alias("b_contrib_bb"),
        )
        .with_columns(
            ((pl.col("a_contrib_bb") > 1.0) & (pl.col("b_contrib_bb") > 1.0)).alias("both_commit"),
            (pl.col("a_net_bb").abs() + pl.col("b_net_bb").abs()).alias("abs_flow_bb"),
            ((pl.col("a_went_to_showdown")) & (pl.col("b_went_to_showdown"))).alias("both_sd"),
        )
    )
    agg = shared.group_by("pair_id").agg(
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
    full = pair_frame.select(["pair_id"]).join(agg, on="pair_id", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in GEN_FEATURES]
    )
    return full.collect()


def main() -> int:
    dev_pairs = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2"]).lazy()
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").lazy()
    ecols = eval_pairs.collect_schema().names()
    print("eval_pairs cols:", ecols)

    dev = build_pair_generator(dev_pairs)
    evl = build_pair_generator(eval_pairs)
    print(f"dev pairs={dev.height} eval pairs={evl.height}")

    Xd = dev.select(GEN_FEATURES).to_numpy().astype(np.float32)
    Xe = evl.select(GEN_FEATURES).to_numpy().astype(np.float32)
    res = adversarial_drift_auc_matrix(Xd, Xe, block="pair_generator", seed=42)
    print(f"\nJOINT drift AUC = {res.auc:.4f} (threshold {res.threshold}; pass={res.passed})")

    # Per-feature drift: quantile grids dev vs eval.
    print("\nper-feature dev vs eval medians (transport check):")
    for c in GEN_FEATURES:
        dv = dev[c].to_numpy(); ev = evl[c].to_numpy()
        print(f"  {c:20s} dev[med={np.median(dv):8.3f} mean={dv.mean():8.3f}] eval[med={np.median(ev):8.3f} mean={ev.mean():8.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
