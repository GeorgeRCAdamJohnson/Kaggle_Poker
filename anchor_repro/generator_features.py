"""Shared dev/eval per-(pair,hand) feature builder for generator recovery.

ONE builder, called identically for development and evaluation (transport-fidelity rule,
dossier §92). For every SHARED hand of every labeled/eval pair it emits rich per-hand
features describing how the two members played THAT hand together: seat outcomes
(contribution, net, showdown, hole strength), directed value flow, and per-street action
structure (aggression, folds, checks, to-call severity) for each member and for outsiders.

The planted evidence hands (development_evidence.csv) are the supervised target for the
per-hand classifier (§3). This module only BUILDS features; it does not train.

Memory-safe: per-hand/player action aggregates are computed ONCE over the 18.6M-row action
stream, then joined to the pair-hand skeleton (built via the keyed (pair_id, player_id)
self-join, never a player x player cross-join).

Run:  python -m anchor_repro.generator_features
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

D = Path("data/poker")
OUT = Path("outputs/poker_collusion/generator_hunt")
# Authoritative (pair, hand) universe = the competition's shared-hand set, taken from the
# V30 prepared caches (verified: eval = exactly 9,651,820 rows). We restrict our per-hand
# features to this set so training/scoring match the real problem and evidence scoring is valid.
V30_DEV_HANDS = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
V30_EVAL_HANDS = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")

# Cheap 13-rank hole-card strength (high card -> ace); board texture ignored for now.
_RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def _hole_strength_expr(c1: str, c2: str) -> pl.Expr:
    """Crude preflop hole strength in [0,1]: high-pair/high-card proxy from rank chars."""
    r1 = pl.col(c1).str.slice(0, 1).replace(_RANKS, default=0).cast(pl.Int8)
    r2 = pl.col(c2).str.slice(0, 1).replace(_RANKS, default=0).cast(pl.Int8)
    hi = pl.max_horizontal(r1, r2)
    lo = pl.min_horizontal(r1, r2)
    suited = pl.col(c1).str.slice(1, 1) == pl.col(c2).str.slice(1, 1)
    pair = r1 == r2
    # normalized: pair bonus + high-card + suited bonus
    return (
        (hi.cast(pl.Float32) / 14.0) * 0.5
        + (lo.cast(pl.Float32) / 14.0) * 0.2
        + pair.cast(pl.Float32) * 0.25
        + suited.cast(pl.Float32) * 0.05
    )


def _per_hand_player_actions() -> pl.LazyFrame:
    """One pass over actions -> per (hand_id, player_id) action structure."""
    a = pl.scan_parquet(D / "actions.parquet").with_columns(
        pl.col("action").is_in(["bet", "raise"]).alias("_agg"),
        (pl.col("action") == "all_in").alias("_allin"),
        (pl.col("action") == "fold").alias("_fold"),
        (pl.col("action") == "check").alias("_check"),
        (pl.col("action") == "call").alias("_call"),
        (pl.col("to_call") / pl.col("pot_before").clip(1, None)).alias("_tc_ratio"),
    )
    return a.group_by(["hand_id", "player_id"]).agg(
        pl.len().alias("n_actions"),
        pl.col("_agg").sum().alias("n_agg"),
        pl.col("_allin").sum().alias("n_allin"),
        pl.col("_fold").sum().alias("n_fold"),
        pl.col("_check").sum().alias("n_check"),
        pl.col("_call").sum().alias("n_call"),
        (pl.col("_agg") & (pl.col("street") != "preflop")).sum().alias("n_postflop_agg"),
        (pl.col("_check") & (pl.col("street").is_in(["turn", "river"]))).sum().alias("n_late_check"),
        pl.col("_tc_ratio").max().alias("max_tc_ratio"),
        pl.col("amount").max().alias("max_amount"),
        pl.col("street").n_unique().alias("n_streets"),
    )


def _member_seats() -> pl.LazyFrame:
    return pl.scan_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "seat_no", "hole_card_1", "hole_card_2",
         "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"]
    ).with_columns(_hole_strength_expr("hole_card_1", "hole_card_2").alias("hs"))


def _per_hand_table_context() -> pl.LazyFrame:
    """One pass over seats -> per-hand table-wide aggregates (all players, incl. outsiders).

    Used to derive OUTSIDER behavior for the pair: how many non-pair players folded, how
    much they contributed, whether the pot was contested by anyone outside the pair. This
    captures coordinated-isolation, which is planted in the OUTSIDER/partner yielding, not
    in both pair members committing.
    """
    s = pl.scan_parquet(D / "seats.parquet").select(
        ["hand_id", "player_id", "total_contribution", "folded", "went_to_showdown", "won_share"]
    )
    return s.group_by("hand_id").agg(
        pl.len().alias("t_n_seats"),
        pl.col("folded").sum().alias("t_n_folded"),
        pl.col("went_to_showdown").sum().alias("t_n_showdown"),
        pl.col("total_contribution").sum().alias("t_total_contrib"),
        pl.col("total_contribution").max().alias("t_max_contrib"),
    )


def build_pair_hand_features(
    pair_frame: pl.LazyFrame, phase_name: str, universe: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    """Shared builder. pair_frame needs pair_id, player_1, player_2. Same code dev+eval.

    If ``universe`` (a LazyFrame with pair_id, hand_id) is given, the output is restricted to
    exactly those (pair, hand) rows = the competition's shared-hand set.
    """
    hands = pl.scan_parquet(D / "hands.parquet").select(
        ["hand_id", "big_blind", "final_pot", "players_dealt", "players_at_showdown"]
    )
    seats = _member_seats()
    acts = _per_hand_player_actions()
    table_ctx = _per_hand_table_context()

    # Attach per-player action structure to each member's seat row ONCE.
    seat_full = seats.join(acts, on=["hand_id", "player_id"], how="left")

    pf = pair_frame.select(["pair_id", "player_1", "player_2"])
    a_map = pf.select(["pair_id", pl.col("player_1").alias("player_id")])
    b_map = pf.select(["pair_id", pl.col("player_2").alias("player_id")])

    acols = ["seat_no", "hole_card_1", "hole_card_2", "total_contribution", "net_chips",
             "folded", "went_to_showdown", "won_share", "hs", "n_actions", "n_agg",
             "n_allin", "n_fold", "n_check", "n_call", "n_postflop_agg", "n_late_check",
             "max_tc_ratio", "max_amount", "n_streets"]
    a = seat_full.join(a_map, on="player_id", how="inner").rename({c: f"a_{c}" for c in acols}).drop("player_id")
    b = seat_full.join(b_map, on="player_id", how="inner").rename({c: f"b_{c}" for c in acols}).drop("player_id")

    shared = a.join(b, on=["hand_id", "pair_id"], how="inner")
    if universe is not None:
        shared = shared.join(universe.select(["pair_id", "hand_id"]), on=["pair_id", "hand_id"], how="inner")
    shared = (
        shared
        .join(hands, on="hand_id", how="left")
        .join(table_ctx, on="hand_id", how="left")
        .with_columns(
            (pl.col("a_net_chips") / pl.col("big_blind")).alias("a_net_bb"),
            (pl.col("b_net_chips") / pl.col("big_blind")).alias("b_net_bb"),
            (pl.col("a_total_contribution") / pl.col("big_blind")).alias("a_contrib_bb"),
            (pl.col("b_total_contribution") / pl.col("big_blind")).alias("b_contrib_bb"),
            (pl.col("final_pot") / pl.col("big_blind")).alias("pot_bb"),
        )
        .with_columns(
            # directed flow between the two
            (pl.col("a_net_bb").abs() + pl.col("b_net_bb").abs()).alias("abs_flow_bb"),
            (pl.col("a_net_bb") * pl.col("b_net_bb")).alias("net_product"),
            pl.min_horizontal("a_net_bb", "b_net_bb").alias("loser_net_bb"),
            pl.max_horizontal("a_net_bb", "b_net_bb").alias("winner_net_bb"),
            pl.min_horizontal("a_contrib_bb", "b_contrib_bb").alias("min_contrib_bb"),
            pl.max_horizontal("a_contrib_bb", "b_contrib_bb").alias("max_contrib_bb"),
            ((pl.col("a_contrib_bb") > 1.0) & (pl.col("b_contrib_bb") > 1.0)).cast(pl.Int8).alias("both_commit"),
            (pl.col("a_went_to_showdown") & pl.col("b_went_to_showdown")).cast(pl.Int8).alias("both_sd"),
            (pl.col("a_folded").cast(pl.Int8) + pl.col("b_folded").cast(pl.Int8)).alias("n_folded"),
            # aggression asymmetry: one drives, the other yields
            (pl.col("a_n_agg") + pl.col("b_n_agg")).alias("pair_agg"),
            (pl.col("a_n_agg") - pl.col("b_n_agg")).abs().alias("agg_asym"),
            (pl.col("a_hs") - pl.col("b_hs")).alias("hs_gap"),
            pl.min_horizontal("a_hs", "b_hs").alias("min_hs"),
            pl.max_horizontal("a_hs", "b_hs").alias("max_hs"),
            # outsiders in the hand (players_dealt minus the pair)
            (pl.col("players_dealt") - 2).clip(0, None).alias("n_outsiders"),
            # who commits more chips: strong-hand-surrender proxy (loser has better hole?)
            pl.when(pl.col("a_net_bb") < pl.col("b_net_bb"))
              .then(pl.col("a_hs")).otherwise(pl.col("b_hs")).alias("loser_hs"),
        )
        .with_columns(
            # strong-hand-surrendered-to-partner: loser had a good hand but lost to partner
            (pl.col("loser_hs") * (pl.col("winner_net_bb").clip(0, None))).alias("surrender_signal"),
            (pl.col("min_contrib_bb") * pl.col("both_commit")).alias("joint_commit_bb"),
            # --- OUTSIDER / ISOLATION features (the third-player + partner-yield structure) ---
            # outsiders = table seats minus the pair
            (pl.col("t_n_seats") - 2).clip(0, None).alias("o_n_outsiders"),
            # outsiders who folded (table folds minus any pair folds)
            (pl.col("t_n_folded") - pl.col("n_folded")).clip(0, None).alias("o_n_outsider_folds"),
            # one member commits while the OTHER yields (folds / min contribution) = isolation
            (
                ((pl.col("a_contrib_bb") > 2.0) & (pl.col("b_folded")))
                | ((pl.col("b_contrib_bb") > 2.0) & (pl.col("a_folded")))
            ).cast(pl.Int8).alias("iso_one_commits_other_folds"),
            # member wins uncontested by outsiders (partner + all outsiders fold to member)
            (
                (pl.col("winner_net_bb") > 0)
                & (pl.col("t_n_folded") >= (pl.col("t_n_seats") - 1))
            ).cast(pl.Int8).alias("iso_uncontested_take"),
            # fraction of the whole table that folded (isolation => most fold to the pair)
            (pl.col("t_n_folded") / pl.col("t_n_seats").clip(1, None)).alias("o_table_fold_rate"),
            # member's chips as a share of all chips committed (dominance of the pot)
            (pl.col("max_contrib_bb") / (pl.col("t_total_contrib") / pl.col("big_blind") + 1e-3)).alias("o_member_pot_share"),
            pl.lit(phase_name).alias("phase"),
        )
    )
    return shared


def _feature_columns(schema_names) -> list[str]:
    drop = {"pair_id", "hand_id", "phase", "a_hole_card_1", "a_hole_card_2",
            "b_hole_card_1", "b_hole_card_2", "a_seat_no", "b_seat_no"}
    return [c for c in schema_names if c not in drop]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2"]).lazy()
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"]).lazy()
    evidence = pl.read_csv(D / "development_evidence.csv").select(["pair_id", "hand_id", "evidence_rank"])

    dev_universe = pl.scan_parquet(V30_DEV_HANDS).select(["pair_id", "hand_id"])
    eval_universe = pl.scan_parquet(V30_EVAL_HANDS).select(["pair_id", "hand_id"])

    dev_path = OUT / "dev_pairhand_features.parquet"
    t = time.time()
    dev = build_pair_hand_features(labels, "development", universe=dev_universe).join(
        evidence.lazy(), on=["pair_id", "hand_id"], how="left"
    ).with_columns(pl.col("evidence_rank").is_not_null().cast(pl.Int8).alias("is_planted"))
    dev_df = dev.collect(engine="streaming")
    dev_df.write_parquet(dev_path, compression="zstd")
    planted = int(dev_df["is_planted"].sum())
    print(f"dev pair-hands={dev_df.height:,} planted={planted} ({time.time()-t:.1f}s) -> {dev_path}")
    # Some evidence hands may fall outside the V30 shared-hand universe; record how many.
    print(f"  planted retained in universe: {planted}/1817")

    # Eval is 9.65M pair-hands; build in pair batches and write incrementally to bound RAM.
    t = time.time()
    eval_pairs_df = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    eval_path = OUT / "eval_pairhand_features.parquet"
    batch = 10000
    n = eval_pairs_df.height
    total = 0
    writer = None
    import pyarrow.parquet as pq  # noqa: E402

    for start in range(0, n, batch):
        chunk_pairs = eval_pairs_df.slice(start, batch).lazy()
        chunk = build_pair_hand_features(
            chunk_pairs, "evaluation", universe=eval_universe
        ).collect(engine="streaming")
        total += chunk.height
        arrow = chunk.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(eval_path, arrow.schema, compression="zstd")
        writer.write_table(arrow)
        print(f"  eval batch {start//batch + 1}/{-(-n//batch)}: pairs[{start}:{min(start+batch,n)}] "
              f"+{chunk.height:,} rows (cum {total:,})", flush=True)
        del chunk, arrow
    if writer is not None:
        writer.close()
    print(f"eval pair-hands={total:,} ({time.time()-t:.1f}s) -> {eval_path}")

    feats = _feature_columns(dev_df.columns)
    (OUT / "generator_feature_columns.json").write_text(
        __import__("json").dumps([c for c in feats if c not in {"evidence_rank", "is_planted"}], indent=2),
        encoding="utf-8",
    )
    print(f"feature columns: {len([c for c in feats if c not in {'evidence_rank','is_planted'}])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
