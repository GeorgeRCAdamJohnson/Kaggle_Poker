"""OUR EDGE over hosen42: features they under-exploited, built on their cached substrate.

hosen42's pair model is dominated by loose_pf_sum = PREFLOP entry surprise, summed MARGINALLY over the
two members. Three things they left on the table:

 (A) POSTFLOP policy deviation. They only model preflop VPIP surprise. We build P(action | true-hand-
     rank bucket, street, facing-bet) over ALL 12M seats, then postflop_loose = -log P(action taken).
     Catches soft-play checkdowns / calling-with-junk / folding-the-winner POSTFLOP, invisible to them.

 (B) JOINT surprise (the base-rate-robust signal). loose_pf_sum is A+B marginal. Two players
     coincidentally entering junk factorizes; COLLUSION does not. Per pair we compute the EXCESS
     co-surprise over independent expectation: obs P(both loose) - P(A loose)*P(B loose), plus the
     within-hand correlation of the two members' surprise, over the pair's shared hands.

 (C) TIMING / sequence for coordinated_isolation (their weakest family): does the 2nd member act
     (enter/raise) right AFTER the 1st in the same hand, consistently, hand after hand.

Outputs an augmented pair-feature parquet (their 242 + our new cols) for dev and eval, plus a
STANDALONE judge of our features vs labels. Built identically dev/eval (parity by construction).
Uses only the cached substrate in outputs/poker_collusion/hosen42_step3/.
Run:  python -m anchor_repro.policy_edge_features
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

STEP3 = Path("outputs/poker_collusion/hosen42_step3")
PH = STEP3 / "player_hands_policy.parquet"
PS = STEP3 / "player_strength.parquet"
AC = STEP3 / "action_context.parquet"
OUT = STEP3 / "edge"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# (A) POSTFLOP policy: P(action | rank-bucket, street, facing-bet) over ALL seats
# ---------------------------------------------------------------------------
def build_postflop_policy() -> pl.DataFrame:
    """Per (hand,player): postflop_loose = sum over postflop actions of -log P(action | rk_bucket, street, facing).
    Also fold_winner_surprise (folded while holding a top-rank hand) and call_junk_surprise (called with bottom rank)."""
    strength = pl.scan_parquet(PS).select(["hand_id", "player_id", "rk_flop", "rk_turn", "rk_river"])
    # action stream postflop only, tag street + facing-bet + simplified action
    acts = (pl.scan_parquet(AC).filter(pl.col("street_no") >= 1)
            .with_columns(
                pl.when(pl.col("is_aggr")).then(pl.lit("aggr"))
                 .when(pl.col("action") == "fold").then(pl.lit("fold"))
                 .when(pl.col("action") == "call").then(pl.lit("call"))
                 .when(pl.col("action") == "check").then(pl.lit("check"))
                 .otherwise(pl.lit("other")).alias("act3"),
                (pl.col("to_call") > 0).alias("facing"),
            )
            .join(strength, on=["hand_id", "player_id"], how="left")
            # rank at the action's street: rk_flop for street 1, rk_turn 2, rk_river 3
            .with_columns(
                pl.when(pl.col("street_no") == 1).then(pl.col("rk_flop"))
                 .when(pl.col("street_no") == 2).then(pl.col("rk_turn"))
                 .otherwise(pl.col("rk_river")).alias("rk"))
            .with_columns(pl.col("rk").fill_null(6).clip(1, 6).cast(pl.Int8).alias("rk_bucket"))
            .select(["hand_id", "phase", "player_id", "street_no", "facing", "rk_bucket", "act3"]))

    # population policy P(act3 | rk_bucket, street, facing) over ALL actions (both phases; colluders negligible)
    ctx = acts.group_by(["rk_bucket", "street_no", "facing", "act3"]).agg(pl.len().alias("n"))
    tot = acts.group_by(["rk_bucket", "street_no", "facing"]).agg(pl.len().alias("N"))
    K = 20.0
    # class prior over act3 within (rk_bucket) for smoothing
    prior = acts.group_by(["act3"]).agg(pl.len().alias("np")).with_columns((pl.col("np") / pl.col("np").sum()).alias("p_prior"))
    policy = (ctx.join(tot, on=["rk_bucket", "street_no", "facing"]).join(prior, on="act3")
              .with_columns(((pl.col("n") + K * pl.col("p_prior")) / (pl.col("N") + K)).clip(1e-4, 1 - 1e-4).alias("p_act"))
              .select(["rk_bucket", "street_no", "facing", "act3", "p_act"]))
    scored = (acts.join(policy, on=["rk_bucket", "street_no", "facing", "act3"], how="left")
              .with_columns(pl.col("p_act").fill_null(0.1))
              .with_columns((-pl.col("p_act").log()).alias("act_surprise"),
                            # folded a top-1/2 hand
                            ((pl.col("act3") == "fold") & (pl.col("rk_bucket") <= 2)).cast(pl.Int8).alias("fold_strong"),
                            # called/checked (passive) with a bottom-rank hand facing nothing lost, or called with junk
                            ((pl.col("act3").is_in(["call", "check"])) & (pl.col("rk_bucket") >= 5)).cast(pl.Int8).alias("passive_junk")))
    perhand = scored.group_by(["hand_id", "player_id"]).agg(
        pl.col("act_surprise").sum().alias("postflop_loose"),
        pl.col("act_surprise").max().alias("postflop_loose_max"),
        pl.col("fold_strong").max().alias("fold_strong"),
        pl.col("passive_junk").sum().alias("passive_junk"),
    )
    return perhand.collect(engine="streaming")


def main() -> int:
    log("building postflop policy over all seats...")
    postflop = build_postflop_policy()
    postflop.write_parquet(OUT / "postflop_perhand.parquet")
    log(f"postflop per-hand: {postflop.height:,} rows; mean postflop_loose="
        f"{postflop['postflop_loose'].mean():.3f}")

    # attach preflop `loose` (from policy cache) so we have per-(hand,player) TOTAL surprise
    pre = pl.scan_parquet(PH).select(["hand_id", "player_id", "phase", "loose", "vpip_a", "tight"])
    per = pre.join(postflop.lazy(), on=["hand_id", "player_id"], how="left").with_columns(
        pl.col("postflop_loose").fill_null(0.0), pl.col("postflop_loose_max").fill_null(0.0),
        pl.col("fold_strong").fill_null(0), pl.col("passive_junk").fill_null(0),
        (pl.col("loose") + pl.col("postflop_loose").fill_null(0.0)).alias("total_loose"),
    )
    per.collect(engine="streaming").write_parquet(OUT / "per_hand_surprise.parquet")
    log("wrote per_hand_surprise.parquet (preflop loose + postflop loose per player-hand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
