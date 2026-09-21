"""HAVE WE BEEN SOLVING THE WRONG PROBLEM? (user: is top-5 a re-ranking problem — cast to top-100, then
find the 5 — rather than a single-pass scoring problem?)

Decisive diagnostic. For each positive pair, order its shared hands by the base FAM ranker, then measure:
  RECALL@K : fraction of the pair's TRUE evidence hands that fall within the top-K of our ordering.
             If true hands are almost always in top-100, the ceiling is a RE-RANKING problem (winnable by a
             2-stage retrieve->rerank). If they fall outside, it's a base SCORING/feature problem (not).
  ORACLE MAP@5 @K : the MAP@5 we would get if we could PERFECTLY reorder within the top-K candidate set
             (true hands first). This is the hard ceiling a re-ranker on that candidate set could reach.
             - oracle@5   = our current MAP@5 (can't reorder beyond what's already top-5)
             - oracle@100 vs current: the headroom a re-ranker could unlock.
Also report shared-hands-per-pair (how big the candidate pool even is) and how many pairs have <100 hands
(where top-100 == all hands, so re-ranking == full reorder).
Run:  python -m anchor_repro.evidence_recall_oracle
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

def fam_ranker(dh, feats):
    dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    return dh["hand_score"].to_numpy()

def oracle_map5_at_k(pos, K):
    """If we keep our top-K by score, then perfectly reorder (true hands first), what MAP@5? = fraction of
    the (<=5) true hands that survive in top-K, scored as if placed at the front."""
    vals=[]
    for _,g in pos.groupby("pair_id",sort=False):
        g=g.sort_values(["hand_score","pot_bb","hand_id"],ascending=[False,False,True],kind="mergesort")
        rel=g["is_evidence"].to_numpy(); nr=int(rel.sum())
        if nr==0: continue
        kept=int(rel[:K].sum())  # true hands surviving in top-K
        m=min(nr,5)
        # perfect reorder: place all `kept` true hands first -> precision 1.0 for the first min(kept,5)
        got=min(kept,5)
        vals.append(float(got/m))  # AP@5 with all relevant at front = (#relevant retrieved in 5)/min(nr,5)
    return float(np.mean(vals))

def recall_at_k(pos, K):
    vals=[]
    for _,g in pos.groupby("pair_id",sort=False):
        g=g.sort_values(["hand_score","pot_bb","hand_id"],ascending=[False,False,True],kind="mergesort")
        rel=g["is_evidence"].to_numpy(); nr=int(rel.sum())
        if nr==0: continue
        vals.append(int(rel[:K].sum())/nr)
    return float(np.mean(vals))

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    FEATS=HAND_FEATS+CTXCOLS

    # candidate pool size
    hp=dh[dh["label"]==1].groupby("pair_id").size()
    log(f"positive pairs {len(hp)}; shared hands/pair: mean {hp.mean():.0f} median {int(hp.median())} min {hp.min()} max {hp.max()}")
    for K in [20,50,100]:
        log(f"  pairs with <= {K} shared hands: {(hp<=K).mean()*100:.0f}%")

    dh["hand_score"]=fam_ranker(dh,FEATS)
    pos=dh[dh["label"]==1].copy()
    cur=map5(pos)
    log(f"=== base FAM MAP@5 = {cur:.4f} ===")
    log("--- RECALL@K of true evidence hands (is the truth even in our candidate set?) ---")
    for K in [5,10,20,50,100,200]:
        log(f"  recall@{K:<4d} = {recall_at_k(pos,K):.4f}   oracle-MAP@5 if perfect-reorder within top-{K} = {oracle_map5_at_k(pos,K):.4f}")
    log("--- per family: recall@100 and oracle@100 (where is the truth escaping?) ---")
    for fam in TARGET_BEHAVIORS:
        p=pos[pos["behavior_family"]==fam]
        log(f"  {fam:22s} recall@100 {recall_at_k(p,100):.4f}  oracle@100 {oracle_map5_at_k(p,100):.4f}  cur {map5(p):.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
