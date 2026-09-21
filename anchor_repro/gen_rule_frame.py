"""§219 PHASE 1a — GENERATOR-RULE measurement frame (§109, finally). Build per-hand raw facts for the 1,817
PLANTED evidence hands vs the NON-planted sibling hands of the SAME 372 positive pairs. This is the
coverage/purity substrate: for each family, planted-set vs the pair's other shared hands, with the RAW
generator-relevant quantities (chip flow direction, showdown, aggression asymmetry, pot, net) so Phase 1b can
search for a near-deterministic RULE.

Uses seats.parquet (net_chips, folded, showdown, contribution) + hands (pot, bb) + a compact action summary.
Keyed (pair_id, hand_id, is_planted, behavior_family). CPU (branchy aggregation, not tensor).
Run: python -m anchor_repro.gen_rule_frame
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); OUT=STEP3/"edge"/"genrule"; OUT.mkdir(parents=True,exist_ok=True)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def build():
    labels=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2","behavior_family"])
    ev=pl.read_csv(D/"development_evidence.csv").select(["pair_id","hand_id"]).with_columns(pl.lit(1,dtype=pl.Int8).alias("is_planted"))
    # shared hands of positive pairs = dev_pair_hands filtered to positive pairs
    ph=pl.read_parquet(STEP3/"dev_pair_hands.parquet").select(["pair_id","hand_id","player_1","player_2"])
    ph=ph.join(labels.select("pair_id"),on="pair_id",how="inner")
    ph=ph.join(ev,on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_planted").fill_null(0))
    ph=ph.join(labels.select(["pair_id","behavior_family"]),on="pair_id",how="left")
    log(f"positive-pair shared hands {ph.height:,} | planted {int(ph['is_planted'].sum())}")
    hset=ph["hand_id"].unique().to_list()
    seats=pl.read_parquet(D/"seats.parquet").filter(pl.col("hand_id").is_in(hset)).select(
        ["hand_id","player_id","net_chips","total_contribution","folded","went_to_showdown"])
    hands=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(hset)).select(["hand_id","big_blind","final_pot","board_cards"])
    # action summary per (hand,player): n_aggr, n_call, n_check, n_fold, last street reached
    acts=pl.scan_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(hset)).select(["hand_id","player_id","action","street"]).collect()
    asum=acts.group_by(["hand_id","player_id"]).agg(
        (pl.col("action").is_in(["bet","raise","all_in"])).sum().alias("n_aggr"),
        (pl.col("action")=="call").sum().alias("n_call"),
        (pl.col("action")=="check").sum().alias("n_check"),
        (pl.col("action")=="fold").sum().alias("n_fold"),
        (pl.col("street")!="preflop").any().alias("saw_postflop"))
    seats=seats.join(asum,on=["hand_id","player_id"],how="left").fill_null(0)
    # join member A and B facts onto each pair-hand
    def side(tag,pcol):
        return seats.rename({c:f"{tag}_{c}" for c in ["player_id","net_chips","total_contribution","folded","went_to_showdown","n_aggr","n_call","n_check","n_fold","saw_postflop"]})
    A=ph.join(side("a","player_1"),left_on=["hand_id","player_1"],right_on=["hand_id","a_player_id"],how="left")
    A=A.join(side("b","player_2"),left_on=["hand_id","player_2"],right_on=["hand_id","b_player_id"],how="left")
    A=A.join(hands,on="hand_id",how="left")
    # generator-relevant raw derived quantities (per hand, in bb)
    A=A.with_columns([pl.col(c).cast(pl.Float64) for c in ["a_net_chips","b_net_chips","a_total_contribution","b_total_contribution","final_pot"]])
    A=A.with_columns(pl.col("big_blind").cast(pl.Float64).clip(1,None).alias("bb"))
    A=A.with_columns(
        (pl.col("a_net_chips")/pl.col("bb")).alias("a_net"),(pl.col("b_net_chips")/pl.col("bb")).alias("b_net"),
        (pl.col("final_pot")/pl.col("bb")).alias("pot_bb_g"),
    ).with_columns(
        (pl.col("a_net")*pl.col("b_net")<0).alias("opposite_sign_net"),      # one wins one loses
        (pl.max_horizontal("a_net","b_net")).alias("winner_net"),
        (pl.min_horizontal("a_net","b_net")).alias("loser_net"),
        ((pl.col("a_went_to_showdown")>0)&(pl.col("b_went_to_showdown")>0)).alias("both_showdown"),
        ((pl.col("a_n_aggr")==0)&(pl.col("b_n_aggr")==0)).alias("both_no_aggr"),
        (pl.col("a_n_aggr")+pl.col("b_n_aggr")).alias("pair_aggr"),
        ((pl.col("a_folded").cast(pl.Int8))+(pl.col("b_folded").cast(pl.Int8))).alias("n_folded"),
    )
    keep=["pair_id","hand_id","is_planted","behavior_family","a_net","b_net","pot_bb_g","winner_net","loser_net",
          "opposite_sign_net","both_showdown","both_no_aggr","pair_aggr","n_folded",
          "a_n_aggr","b_n_aggr","a_went_to_showdown","b_went_to_showdown","board_cards"]
    out=A.select(keep)
    out.write_parquet(OUT/"gen_rule_frame.parquet")
    log(f"wrote gen_rule_frame {out.height:,} rows, {int(out['is_planted'].sum())} planted -> genrule/gen_rule_frame.parquet")
    # quick per-family planted-vs-nonplanted means of the key raw quantities
    pos=out
    print("\n=== per-family: planted vs non-planted means of raw generator quantities ===")
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        f=pos.filter(pl.col("behavior_family")==fam)
        pl_=f.filter(pl.col("is_planted")==1); nn=f.filter(pl.col("is_planted")==0)
        def mn(c,fr): return fr[c].mean()
        print(f"  {fam}: n_planted {pl_.height} n_other {nn.height}")
        for c in ["pot_bb_g","winner_net","loser_net","pair_aggr","n_folded"]:
            print(f"     {c:14s} planted {mn(c,pl_):8.2f}  nonplanted {mn(c,nn):8.2f}")
        print(f"     opposite_sign_net planted {pl_['opposite_sign_net'].mean():.3f} nonpl {nn['opposite_sign_net'].mean():.3f} | both_showdown {pl_['both_showdown'].mean():.3f}/{nn['both_showdown'].mean():.3f} | both_no_aggr {pl_['both_no_aggr'].mean():.3f}/{nn['both_no_aggr'].mean():.3f}")
    return out

if __name__=="__main__":
    raise SystemExit(build())
