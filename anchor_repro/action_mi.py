"""§216 Task 1 — cross-member ACTION-COORDINATION features (Bonjour 2022 action-MI idea, adapted to per-hand).
The hope: an ACTION-DEPENDENCE signal orthogonal to the pot/outcome features that made every prior evidence
lever (equity §175, outsider-extraction §214) margin-redundant.

Per (pair, hand), from actions.parquet, capture how COORDINATED the two members' actions are IN THIS HAND
in a CHIP-INDEPENDENT way (this is the orthogonality bet):
  mi_both_passive  : both members took only passive actions (check/call/fold), no aggression -> mutual yielding
  mi_one_aggr_one_fold : exactly one aggressed and the other folded to a non-partner spot (hand-off pattern)
  mi_action_sync   : do their actions mirror across streets (both check, both fold, both call) beyond chance
  mi_no_conflict   : the two NEVER put money in against each other this hand (never both contest same pot round)
  mi_seq_align     : normalized alignment of their action TYPES over the hand (not amounts) -- pure action shape
  mi_turns_between : whether they act adjacently / one right after the other (coordination timing within hand)
All from action TYPES + order, NOT amounts/pot/net -> chip-independent by construction.
Emit per (pair_id, hand_id). Run: python -m anchor_repro.action_mi --phase development
"""
from __future__ import annotations
import time, argparse, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); OUT=STEP3/"edge"/"actionmi"; OUT.mkdir(parents=True,exist_ok=True)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)
MI_FEATS=["mi_both_passive","mi_one_aggr_one_fold","mi_both_aggr","mi_no_conflict","mi_action_sync","mi_seq_align"]
AGGR={"bet","raise","all_in"}; PASS={"check","call","fold"}

def build_action_mi(phase, pair_hands_path):
    pairs=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2"])
    hset=pairs["hand_id"].unique().to_list()
    acts=(pl.scan_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(hset))
          .select(["hand_id","action_no","player_id","action","street"]).collect())
    # AVOID many-to-many hand_id join (a hand belongs to many pairs). Instead: build a long
    # (pair_id, hand_id, member_player) key, then inner-join actions on BOTH hand_id AND player_id.
    keyA=pairs.select(["pair_id","hand_id",pl.col("player_1").alias("player_id")]).with_columns(pl.lit("A").alias("role"))
    keyB=pairs.select(["pair_id","hand_id",pl.col("player_2").alias("player_id")]).with_columns(pl.lit("B").alias("role"))
    key=pl.concat([keyA,keyB])  # 2 rows per pair-hand: the two members
    pm=acts.join(key,on=["hand_id","player_id"],how="inner")  # only member actions, correctly keyed per pair
    pm=pm.with_columns(pl.col("action").is_in(list(AGGR)).alias("is_aggr"))
    g=(pm.sort(["pair_id","hand_id","action_no"]).group_by(["pair_id","hand_id"]).agg(
        (pl.col("role")=="A").sum().alias("nA"),(pl.col("role")=="B").sum().alias("nB"),
        ((pl.col("role")=="A")&pl.col("is_aggr")).sum().alias("aA"),((pl.col("role")=="B")&pl.col("is_aggr")).sum().alias("aB"),
        (pl.col("action")=="fold").filter(pl.col("role")=="A").any().alias("fA"),
        (pl.col("action")=="fold").filter(pl.col("role")=="B").any().alias("fB"),
        pl.col("role").alias("roleseq"), pl.col("action").alias("actseq"), pl.col("street").alias("strseq")))
    g=g.with_columns(
        ((pl.col("aA")==0)&(pl.col("aB")==0)&(pl.col("nA")>0)&(pl.col("nB")>0)).cast(pl.Int8).alias("mi_both_passive"),
        (((pl.col("aA")>0)&(pl.col("fB"))) | ((pl.col("aB")>0)&(pl.col("fA")))).cast(pl.Int8).alias("mi_one_aggr_one_fold"),
        ((pl.col("aA")>0)&(pl.col("aB")>0)).cast(pl.Int8).alias("mi_both_aggr"),
    )
    # action-shape alignment + no-conflict + adjacency need the sequences; compute in python (small: member rows only)
    rows=[]
    for r in g.iter_rows(named=True):
        roles=r["roleseq"]; acts_=r["actseq"]; 
        # no_conflict: never both aggressive in the same street sequence window (approx: not both aggressed)
        no_conflict=int(not (r["aA"]>0 and r["aB"]>0))
        # action_sync: fraction of A-actions whose TYPE also appears as a B-action (shape overlap), symmetric
        aacts=[acts_[i] for i in range(len(roles)) if roles[i]=="A"]; bacts=[acts_[i] for i in range(len(roles)) if roles[i]=="B"]
        if aacts and bacts:
            from collections import Counter
            ca,cb=Counter(aacts),Counter(bacts); inter=sum((ca & cb).values()); tot=max(len(aacts),len(bacts))
            sync=inter/tot if tot else 0.0
        else: sync=0.0
        # seq_align: how often the two act adjacently (B right after A or vice versa) = coordination timing
        adj=sum(1 for i in range(1,len(roles)) if roles[i]!=roles[i-1]); align=adj/max(len(roles)-1,1)
        rows.append((r["pair_id"],r["hand_id"],no_conflict,round(sync,4),round(align,4)))
    extra=pl.DataFrame(rows,schema=["pair_id","hand_id","mi_no_conflict","mi_action_sync","mi_seq_align"],orient="row")
    out=g.select(["pair_id","hand_id","mi_both_passive","mi_one_aggr_one_fold","mi_both_aggr"]).join(extra,on=["pair_id","hand_id"],how="left")
    p=OUT/f"action_mi_{phase}.parquet"; out.write_parquet(p)
    log(f"{phase}: action-MI {out.height:,} pair-hands -> {p.name}")
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",default="development"); a=ap.parse_args()
    ph={"development":STEP3/"dev_pair_hands.parquet","evaluation":STEP3/"eval_pair_hands.parquet"}[a.phase]
    build_action_mi(a.phase,ph)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
