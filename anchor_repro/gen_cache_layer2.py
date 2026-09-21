"""Layer-2 cheap pair-level feature derivation (generator recovery).

Joins each pair's two members to the Layer-1 caches (hand_player_agg + hand_context),
restricts to the competition's 9.65M shared-hand universe, and derives per-(pair,hand)
features. FAST (no action rescan): Layer-1 did the expensive work once.

Corrected family structure (from inspecting real planted hands):
  * directed_transfer : one member loses big, the OTHER wins it (opposite-sign net flow).
  * soft_play         : both reach showdown / check down passively; muted aggression.
  * coordinated_isolation : ONE member commits/raises, the PARTNER yields (folds or minimal
                            contribution) and outsiders fold -> member takes it (NOT both-commit).

Same shared builder for dev and eval (transport fidelity). Restricted to the V30 universe.

Run:  python -m anchor_repro.gen_cache_layer2
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
HAND_PLAYER = CACHE / "hand_player_agg.parquet"
HAND_CTX = CACHE / "hand_context.parquet"
V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")

# Member columns pulled from Layer-1 for each of the two players.
_MEMBER_COLS = [
    "seat_no", "hs", "starting_stack", "total_contribution", "net_chips", "folded",
    "went_to_showdown", "won_share", "n_actions", "n_agg", "n_allin", "n_fold",
    "n_check", "n_call", "n_postflop_agg", "n_late_check", "n_preflop_agg",
    "max_tc_ratio", "max_amount", "sum_amount", "n_streets", "min_players_active",
]


def build_layer2(pair_frame: pl.LazyFrame, universe: pl.LazyFrame) -> pl.LazyFrame:
    """Shared dev/eval builder. pair_frame: pair_id, player_1, player_2."""
    # Only pull the member columns we need (no hole-card strings).
    hp = pl.scan_parquet(HAND_PLAYER).select(["hand_id", "player_id", *_MEMBER_COLS])
    ctx = pl.scan_parquet(HAND_CTX)
    pf = pair_frame.select(["pair_id", "player_1", "player_2"])

    a = (
        hp.join(pf.select(["pair_id", pl.col("player_1").alias("player_id")]), on="player_id", how="inner")
        .rename({c: f"a_{c}" for c in _MEMBER_COLS}).drop("player_id")
    )
    b = (
        hp.join(pf.select(["pair_id", pl.col("player_2").alias("player_id")]), on="player_id", how="inner")
        .rename({c: f"b_{c}" for c in _MEMBER_COLS}).drop("player_id")
    )

    shared = (
        a.join(b, on=["hand_id", "pair_id"], how="inner")
        .join(universe.select(["pair_id", "hand_id"]), on=["pair_id", "hand_id"], how="inner")
        .join(ctx, on="hand_id", how="left")
        .with_columns(
            (pl.col("a_net_chips") / pl.col("big_blind")).alias("a_net_bb"),
            (pl.col("b_net_chips") / pl.col("big_blind")).alias("b_net_bb"),
            (pl.col("a_total_contribution") / pl.col("big_blind")).alias("a_contrib_bb"),
            (pl.col("b_total_contribution") / pl.col("big_blind")).alias("b_contrib_bb"),
            (pl.col("final_pot") / pl.col("big_blind")).alias("pot_bb"),
        )
        .with_columns(
            # --- shared pair structure ---
            (pl.col("a_net_bb").abs() + pl.col("b_net_bb").abs()).alias("abs_flow_bb"),
            (pl.col("a_net_bb") * pl.col("b_net_bb")).alias("net_product"),
            pl.min_horizontal("a_net_bb", "b_net_bb").alias("loser_net_bb"),
            pl.max_horizontal("a_net_bb", "b_net_bb").alias("winner_net_bb"),
            pl.min_horizontal("a_contrib_bb", "b_contrib_bb").alias("min_contrib_bb"),
            pl.max_horizontal("a_contrib_bb", "b_contrib_bb").alias("max_contrib_bb"),
            (pl.col("a_n_agg") + pl.col("b_n_agg")).alias("pair_agg"),
            (pl.col("a_n_agg") - pl.col("b_n_agg")).abs().alias("agg_asym"),
            (pl.col("a_hs") - pl.col("b_hs")).abs().alias("hs_gap"),
            pl.min_horizontal("a_hs", "b_hs").alias("min_hs"),
            pl.max_horizontal("a_hs", "b_hs").alias("max_hs"),
            (pl.col("a_folded").cast(pl.Int8) + pl.col("b_folded").cast(pl.Int8)).alias("n_pair_folded"),
        )
        .with_columns(
            # --- DIRECTED: opposite-sign flow (one loses to the other) ---
            ((pl.col("loser_net_bb") < 0) & (pl.col("winner_net_bb") > 0)
             & (pl.col("net_product") < 0)).cast(pl.Int8).alias("dir_opposite_flow"),
            (pl.col("winner_net_bb").clip(0, None) * (pl.col("loser_net_bb") < 0).cast(pl.Float32)).alias("dir_transfer_bb"),
            # --- SOFT: both to showdown / passive checkdown, muted aggression ---
            (pl.col("a_went_to_showdown") & pl.col("b_went_to_showdown")).cast(pl.Int8).alias("soft_both_sd"),
            ((pl.col("a_n_agg") + pl.col("b_n_agg")) == 0).cast(pl.Int8).alias("soft_no_agg"),
            (pl.col("a_n_late_check") + pl.col("b_n_late_check")).alias("soft_late_checks"),
            ((pl.col("a_contrib_bb") > 1.0) & (pl.col("b_contrib_bb") > 1.0)).cast(pl.Int8).alias("both_commit"),
            # --- ISOLATION (CORRECTED): one commits, partner yields, outsiders fold ---
            (
                ((pl.col("a_contrib_bb") > 2.0) & (pl.col("b_folded")))
                | ((pl.col("b_contrib_bb") > 2.0) & (pl.col("a_folded")))
            ).cast(pl.Int8).alias("iso_one_commits_partner_folds"),
            # partner minimally contributes while member drives
            (
                ((pl.col("a_contrib_bb") > 2.0) & (pl.col("b_contrib_bb") <= 1.0) & (~pl.col("b_folded")))
                | ((pl.col("b_contrib_bb") > 2.0) & (pl.col("a_contrib_bb") <= 1.0) & (~pl.col("a_folded")))
            ).cast(pl.Int8).alias("iso_partner_min_yield"),
            # outsiders = table minus pair; how many folded
            (pl.col("t_n_seats") - 2).clip(0, None).alias("n_outsiders"),
            (pl.col("t_n_folded") - (pl.col("a_folded").cast(pl.Int8) + pl.col("b_folded").cast(pl.Int8)))
                .clip(0, None).alias("n_outsider_folds"),
            # member takes it uncontested (>=n-1 of the whole table folded)
            ((pl.col("winner_net_bb") > 0) & (pl.col("t_n_folded") >= (pl.col("t_n_seats") - 1)))
                .cast(pl.Int8).alias("iso_uncontested_take"),
            (pl.col("t_n_folded") / pl.col("t_n_seats").clip(1, None)).alias("table_fold_rate"),
            (pl.col("max_contrib_bb") / ((pl.col("t_total_contrib") / pl.col("big_blind")) + 1e-3)).alias("member_pot_share"),
            # joint preflop aggression: BOTH members raise preflop (the real isolation squeeze tell)
            ((pl.col("a_n_preflop_agg") > 0) & (pl.col("b_n_preflop_agg") > 0)).cast(pl.Int8).alias("iso_both_preflop_agg"),
            (pl.col("a_n_preflop_agg") + pl.col("b_n_preflop_agg")).alias("iso_pair_preflop_agg"),
        )
        .with_columns(
            # sharpened isolation: joint preflop aggression x outsiders folding x pot dominance
            (
                pl.col("iso_pair_preflop_agg").cast(pl.Float32)
                * (pl.col("n_outsider_folds").cast(pl.Float32) + 1.0)
                * (pl.col("member_pot_share") + 0.1)
            ).alias("iso_squeeze_signal"),
            # directed sharpened: transfer magnitude x opposite-flow
            (pl.col("dir_transfer_bb") * pl.col("dir_opposite_flow").cast(pl.Float32)).alias("dir_signal"),
            # soft sharpened: both showdown / passive x low aggression x both committed
            (
                (pl.col("soft_both_sd") + pl.col("soft_late_checks").cast(pl.Float32))
                * pl.col("both_commit").cast(pl.Float32)
                / (pl.col("pair_agg").cast(pl.Float32) + 1.0)
            ).alias("soft_signal"),
        )
    )
    return shared


def _feature_columns(names) -> list[str]:
    drop = {"pair_id", "hand_id", "table_id", "a_seat_no", "b_seat_no"}
    return [c for c in names if c not in drop]


def main() -> int:
    import sys
    dev_only = "--dev-only" in sys.argv
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2"]).lazy()
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    evidence = pl.read_csv(D / "development_evidence.csv").select(["pair_id", "hand_id", "evidence_rank"])
    dev_uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"])
    eval_uni = pl.scan_parquet(V30_EVAL).select(["pair_id", "hand_id"])

    dev_path = CACHE / "dev_layer2.parquet"
    eval_path = CACHE / "eval_layer2.parquet"

    t = time.time()
    dev = build_layer2(labels, dev_uni).join(
        evidence.lazy(), on=["pair_id", "hand_id"], how="left"
    ).with_columns(pl.col("evidence_rank").is_not_null().cast(pl.Int8).alias("is_planted"))
    dev_df = dev.collect(engine="streaming")
    dev_df.write_parquet(dev_path, compression="zstd")
    planted = int(dev_df["is_planted"].sum())
    print(f"dev layer2={dev_df.height:,} planted={planted}/1817 ({time.time()-t:.1f}s) -> {dev_path}")
    assert planted == 1817, f"expected 1817 planted, got {planted}"

    feats = [c for c in _feature_columns(dev_df.columns) if c not in {"evidence_rank", "is_planted"}]
    (CACHE / "layer2_feature_columns.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
    print(f"layer2 feature columns: {len(feats)}")
    if dev_only:
        print("--dev-only: skipping eval rebuild")
        return 0

    # Eval in pair-batches to bound RAM (writer streaming).
    import pyarrow.parquet as pq
    t = time.time()
    batch = 10000
    n = eval_pairs.height
    total = 0
    writer = None
    for start in range(0, n, batch):
        chunk = build_layer2(eval_pairs.slice(start, batch).lazy(), eval_uni).collect(engine="streaming")
        total += chunk.height
        arrow = chunk.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(eval_path, arrow.schema, compression="zstd")
        writer.write_table(arrow)
        print(f"  eval batch {start//batch+1}/{-(-n//batch)}: +{chunk.height:,} (cum {total:,})", flush=True)
        del chunk, arrow
    if writer:
        writer.close()
    print(f"eval layer2={total:,} ({time.time()-t:.1f}s) -> {eval_path}")
    assert total == 9651820, f"expected 9,651,820 eval rows, got {total}"

    feats = [c for c in _feature_columns(dev_df.columns) if c not in {"evidence_rank", "is_planted"}]
    (CACHE / "layer2_feature_columns.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
    print(f"layer2 feature columns: {len(feats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
