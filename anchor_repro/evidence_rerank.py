"""STAGE-2 RE-RANKER over top-K candidates (§177 breakthrough). The true evidence hands are already in our
top-10/20 (recall@10=0.90, @20=0.98); our 0.5989 is a RE-RANKING failure. Fix = a listwise re-ranker
trained ONLY on the candidate hands to order the true 5 ahead of the near-miss 6th-10th.

Two stages, both OOF (table folds), within-pair (phase-immune, no gate, local==LB 1:1):
  STAGE 1 (retrieve): base family-conditional pointwise ranker (HAND_FEATS+ctx) -> per-pair top-K.
  STAGE 2 (rerank):   LightGBM lambdarank on the STAGE-1 top-K candidates ONLY, per-pair groups,
                      graded relevance = (6 - evidence_rank) for true hands else 0 (rank1 highest gain).
                      lambdarank directly optimizes NDCG-at-top = exactly what MAP@5 rewards.

We must avoid leakage: stage-1 scores used to select candidates are OOF; stage-2 is trained on OTHER folds'
candidates and applied to the held fold's candidates. Compare final MAP@5 (host formula) vs base 0.5989.
Sweep K in {10,15,20,30} and relevance schemes. Ship the best.
Run:  python -m anchor_repro.evidence_rerank
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; DATA=Path("data/poker")
SEED=42

def build_dh_ranked():
    dev_pairs=pl.read_parquet(STEP3/"dev_pairs.parquet")
    dev_ev=pl.read_csv(DATA/"development_evidence.csv")
    DEV_HF=sorted(STEP3.glob("dev_hand_features_*.parquet"))
    TABLES=sorted(dev_pairs["table_id"].unique().to_list()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
    lab=set(dev_pairs.filter(pl.col("is_labeled"))["pair_id"].to_list())
    d=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(lab))).collect() for f in DEV_HF])
    d=d.join(dev_pairs.select(["pair_id","label","behavior_family","is_labeled"]),on="pair_id")
    d=d.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold"))
    d=d.join(dev_ev.select(["pair_id","hand_id","evidence_rank"]),on=["pair_id","hand_id"],how="left")
    d=d.with_columns(pl.col("evidence_rank").fill_null(0).cast(pl.Int64).alias("evidence_rank"))
    d=d.with_columns((pl.col("evidence_rank")>0).cast(pl.Boolean).alias("is_evidence"))
    assert d["is_evidence"].sum()==1817, d["is_evidence"].sum()
    return d

def stage1_scores(dh, feats):
    """OOF pointwise family-conditional retrieval scores (the current base ranker)."""
    dh=dh.copy(); dh["s1"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"s1"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    return dh["s1"].to_numpy()

RANK_PARAMS=dict(objective="lambdarank",metric="ndcg",eval_at=[5],learning_rate=0.05,num_leaves=31,
                 min_data_in_leaf=20,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,
                 lambda_l2=5.0,verbose=-1,num_threads=-1,max_position=5)

def relevance(rank_arr, scheme):
    r=rank_arr.astype(float)
    if scheme=="binary":   return (r>0).astype(int)
    if scheme=="linear":   return np.where(r>0, 6-r, 0)          # rank1->5 ... rank5->1
    if scheme=="steep":    return np.where(r>0, (6-r)**2, 0)     # emphasize top ranks
    raise ValueError(scheme)

def rerank(dh, cand_mask_col, feats, scheme, K):
    """Train lambdarank on top-K candidates (per pair) from OTHER folds, apply to held fold's candidates.
    Non-candidate hands keep a -inf-ish score so they stay below all re-ranked candidates."""
    dh=dh.copy(); dh["s2"]=-1e9
    # candidate = within-pair stage1 rank < K
    dh["s1_rk"]=dh.groupby("pair_id")["s1"].rank(method="first",ascending=False)
    cand=(dh["s1_rk"]<=K).to_numpy() & (dh["label"]==1).to_numpy()
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        trm=(dh["fold"]!=f).to_numpy() & cand
        # build groups (per pair) on training candidates, sorted by pair
        tr=dh.loc[trm].sort_values("pair_id",kind="mergesort")
        grp=tr.groupby("pair_id",sort=False).size().to_numpy()
        yrel=relevance(tr["evidence_rank"].to_numpy(),scheme)
        ms=[]
        for sd in HAND_SEEDS:
            ds=lgb.Dataset(tr[feats],yrel,group=grp)
            ms.append(lgb.train({**RANK_PARAMS,"seed":sd},ds,num_boost_round=HAND_ROUNDS))
        tem=te & cand
        if tem.sum()==0: continue
        dh.loc[tem,"s2"]=np.mean([m.predict(dh.loc[tem,feats]) for m in ms],axis=0)
    return dh["s2"].to_numpy()

def map5_from(dh, score):
    d=dh.copy(); d["hand_score"]=score; return map5(d[d["label"]==1])

def main():
    dh=build_dh_ranked()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet"); dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    dh["evidence_rank"]=dh["evidence_rank"].fillna(0).astype(int)
    dh["is_evidence"]=(dh["evidence_rank"]>0)
    FEATS=HAND_FEATS+CTXCOLS
    dh["s1"]=stage1_scores(dh,FEATS)
    base=map5_from(dh,dh["s1"].to_numpy())
    log(f"=== STAGE-1 base MAP@5 = {base:.4f} (target: oracle@10=0.90, @20=0.98) ===")
    best=(base,"base",None,None)
    for K in [10,15,20,30]:
        for scheme in ["linear","steep","binary"]:
            s2=rerank(dh,"cand",FEATS,scheme,K)
            # final score: re-ranked candidates on top (s2), non-candidates below by s1
            final=np.where(s2>-1e8, s2, -1e9 + dh["s1"].to_numpy())
            m=map5_from(dh,final)
            fam={fm:map5_from(dh[dh["behavior_family"]==fm], final[(dh["behavior_family"]==fm).to_numpy()]) for fm in TARGET_BEHAVIORS}
            log(f"  K={K:<3d} rel={scheme:7s} MAP@5 {m:.4f}  d {m-base:+.4f}  fam={{ {', '.join(f'{k[:4]}:{v:.3f}' for k,v in fam.items())} }}")
            if m>best[0]: best=(m,scheme,K,None)
    log(f"=== BEST rerank: MAP@5 {best[0]:.4f} (scheme={best[1]}, K={best[2]})  vs base {base:.4f}  d {best[0]-base:+.4f} ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
