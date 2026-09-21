"""STAGE-2 RE-RANKER on GPU (§177 lever + §179 GPU offload). XGBoost rank:ndcg, device=cuda — tree-building
on the 12GB VRAM, off contended system RAM. Stage-1 retrieval stays LightGBM (preserves the banked 0.5989
baseline for candidate selection). Within-pair => phase-immune, no gate, local MAP@5 == LB 1:1.

Oracle (§177): true evidence hands are in top-10 (recall 0.90) / top-20 (0.98); current MAP@5 0.5989 is a
RE-RANKING failure. This trains a listwise re-ranker over the stage-1 top-K to promote the buried true 5.

  STAGE 1 (LightGBM, CPU): family-conditional pointwise retrieval -> per-pair top-K candidates (OOF).
  STAGE 2 (XGBoost rank:ndcg, GPU): trained on OTHER folds' top-K candidates (per-pair groups, graded
           relevance = 6-evidence_rank), applied to the held fold's candidates.
Sweep K in {10,15,20,30} x relevance {linear,steep}. Report host-exact MAP@5 + per family + seed stability.
Run:  python -m anchor_repro.evidence_rerank_gpu
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from anchor_repro.seq_evidence import HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.evidence_rerank import build_dh_ranked, relevance
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

XGB_RANK=dict(objective="rank:ndcg",eval_metric="ndcg@5",device="cuda",tree_method="hist",
              max_depth=6,eta=0.05,subsample=0.8,colsample_bytree=0.8,reg_lambda=5.0,
              min_child_weight=5,lambdarank_num_pair_per_sample=8,lambdarank_pair_method="topk")
XGB_ROUNDS=400

S1_CACHE=SEQ/"rerank_stage1_lgb.parquet"

def stage1_lgb(dh, feats):
    """OOF family-conditional pointwise retrieval (LightGBM, preserves banked baseline). Cached to parquet
    keyed (pair_id,hand_id) so the ~2min CPU cost is paid ONCE and every GPU re-ranker sweep reuses it."""
    if S1_CACHE.exists():
        c=pl.read_parquet(S1_CACHE)
        m=dh[["pair_id","hand_id"]].merge(c.to_pandas(),on=["pair_id","hand_id"],how="left")
        if not m["s1"].isna().any():
            log(f"stage-1 LGB scores loaded from cache ({S1_CACHE.name})"); return m["s1"].to_numpy()
        log("stage-1 cache stale -> recompute")
    dh=dh.copy(); s=np.zeros(len(dh)); folds=sorted(dh["fold"].unique())
    for f in folds:
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            s[tgt.nonzero()[0]]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
        log(f"  stage-1 LGB fold {f+1}/{len(folds)} done")
    pl.DataFrame({"pair_id":dh["pair_id"].to_numpy(),"hand_id":dh["hand_id"].to_numpy(),"s1":s}).write_parquet(S1_CACHE)
    log(f"stage-1 LGB scores cached -> {S1_CACHE.name}")
    return s

def stage2_xgb_gpu(dh, feats, s1, scheme, K, seeds=(42,49)):
    """XGBoost rank:ndcg on GPU over stage-1 top-K candidates, OOF."""
    dh=dh.copy(); dh["s1"]=s1; out=np.full(len(dh),-1e9)
    dh["rk"]=dh.groupby("pair_id")["s1"].rank(method="first",ascending=False)
    cand=(dh["rk"]<=K).to_numpy() & (dh["label"]==1).to_numpy()
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        trm=(dh["fold"]!=f).to_numpy() & cand
        tr=dh.loc[trm].sort_values("pair_id",kind="mergesort")
        grp=tr.groupby("pair_id",sort=False).size().to_numpy()
        yrel=relevance(tr["evidence_rank"].to_numpy(),scheme).astype(int)
        preds=[]
        tem=te & cand
        if tem.sum()==0: continue
        tecand=dh.loc[tem].sort_values("pair_id",kind="mergesort")
        te_idx=tecand.index.to_numpy()
        dtest=xgb.DMatrix(tecand[feats].to_numpy(dtype=np.float32))
        for sd in seeds:
            dtrain=xgb.DMatrix(tr[feats].to_numpy(dtype=np.float32),label=yrel); dtrain.set_group(grp)
            b=xgb.train({**XGB_RANK,"seed":sd},dtrain,num_boost_round=XGB_ROUNDS)
            preds.append(b.predict(dtest))
        out[te_idx]=np.mean(preds,axis=0)
    return np.where(out>-1e8, out, -1e9+s1)

def m5(dh, score):
    d=dh.copy(); d["hand_score"]=score; return map5(d[d["label"]==1])

def main():
    t=time.time()
    dh=build_dh_ranked()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet"); dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    dh["evidence_rank"]=dh["evidence_rank"].fillna(0).astype(int); dh["is_evidence"]=dh["evidence_rank"]>0
    FEATS=HAND_FEATS+CTXCOLS
    s1=stage1_lgb(dh,FEATS)
    base=m5(dh,s1)
    log(f"=== STAGE-1 LGB base MAP@5 = {base:.4f}  (oracle@10=0.90 @20=0.98) ===")
    best=(base,None,None)
    for K in [10,15,20,30]:
        for scheme in ["linear","steep"]:
            log(f"  ...training GPU re-ranker K={K} rel={scheme}")
            s2=stage2_xgb_gpu(dh,FEATS,s1,scheme,K)
            m=m5(dh,s2)
            fam={fm:m5(dh[dh["behavior_family"]==fm], s2[(dh["behavior_family"]==fm).to_numpy()]) for fm in TARGET_BEHAVIORS}
            log(f"  K={K:<3d} rel={scheme:6s} GPU-rerank MAP@5 {m:.4f}  d {m-base:+.4f}  fam={{ {', '.join(f'{k[:4]}:{v:.3f}' for k,v in fam.items())} }}")
            if m>best[0]: best=(m,K,scheme)
    log(f"=== BEST GPU-rerank MAP@5 {best[0]:.4f} (K={best[1]}, rel={best[2]})  vs base {base:.4f}  d {best[0]-base:+.4f}  [{time.time()-t:.0f}s] ===")
    if best[1] is not None:
        # seed stability of the winner
        log("--- seed stability of winner ---")
        ds=[]
        for seeds in [(42,49),(101,108),(202,209)]:
            s2=stage2_xgb_gpu(dh,FEATS,s1,best[2],best[1],seeds=seeds); ds.append(m5(dh,s2)-base)
        log(f"  winner deltas over seed-sets: {[round(x,4) for x in ds]}  mean {np.mean(ds):+.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
