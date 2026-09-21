"""EVIDENCE-SIDE consensus — the RIGHT entry point (§154/§160 consensus failed because they were RISK-side,
where drift/redundancy kill everything; evidence consensus is WITHIN-PAIR = phase-immune, local==LB 1:1,
and the oracle (§177) proved the truth is already in top-10/20 so voting is over a set that CONTAINS the
answer, not noise).

Each evidence VARIANT is a voter, scored OOF (table folds) on the 372 labelled dev pairs (known truth):
  V1 base    : HAND_FEATS pointwise            (family-conditional routing)
  V2 ctx     : HAND_FEATS + ctx_emb            (the §150 win)
  V3 equity  : HAND_FEATS + ctx_emb + equity   (transfer+soft only, §175)
  V4 rerank  : lambdarank re-ranker over top-K (§177)          [listwise, top-focused]
Tests, all host-exact MAP@5:
  A) per-voter MAP@5
  B) RANK-CONSENSUS: mean within-pair rank-percentile across a voter subset -> MAP@5 (does voting beat best single?)
  C) DISAGREEMENT diagnostic: among top-10 candidates where voters SPLIT on top-5 membership, each voter's
     hit-rate on the TRUE hand -> tells us which voter to trust in the disagreement zone (the re-ranker's job).
Run:  python -m anchor_repro.evidence_consensus
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from anchor_repro.seq_evidence import HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.evidence_equity import build_perhand, EQEV_FEATS
from anchor_repro.evidence_rerank import build_dh_ranked, RANK_PARAMS, relevance
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
TS=("directed_transfer","soft_play")

# GPU offload (§179): tree-building on the 12GB VRAM via XGBoost device=cuda (pip LightGBM has no GPU build).
XGB_PW=dict(objective="binary:logistic",device="cuda",tree_method="hist",max_depth=6,eta=0.05,
            subsample=0.8,colsample_bytree=0.8,reg_lambda=10.0,min_child_weight=10,eval_metric="logloss")
XGB_PW_ROUNDS=400

def pointwise(dh, feats_of, seeds=(42,49)):
    """OOF family-conditional pointwise scores on GPU (XGBoost device=cuda). feats_of maps family->features.
    NOTE: XGB scores differ from the banked LightGBM 0.5989 baseline — these are GPU voters for the CONSENSUS
    question (voting diversity), not a reproduction of the banked pipeline."""
    dh=dh.copy(); s=np.zeros(len(dh))
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            feats=feats_of[fam]
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            dtr=xgb.QuantileDMatrix(dh.loc[tr,feats].to_numpy(np.float32),label=dh.loc[tr,"is_evidence"].astype(int).to_numpy())
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dte=xgb.DMatrix(dh.loc[tgt,feats].to_numpy(np.float32))
            preds=[]
            for sd in seeds:
                b=xgb.train({**XGB_PW,"seed":sd},dtr,num_boost_round=XGB_PW_ROUNDS); preds.append(b.predict(dte))
            s[tgt.nonzero()[0]]=np.mean(preds,axis=0)
        log(f"  pointwise-GPU fold {f+1} done")
    return s

def rerank_scores(dh, feats, s1, scheme="linear", K=20):
    dh=dh.copy(); dh["s1"]=s1; out=np.full(len(dh),-1e9)
    dh["rk"]=dh.groupby("pair_id")["s1"].rank(method="first",ascending=False)
    cand=(dh["rk"]<=K).to_numpy() & (dh["label"]==1).to_numpy()
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        trm=(dh["fold"]!=f).to_numpy() & cand
        tr=dh.loc[trm].sort_values("pair_id",kind="mergesort")
        grp=tr.groupby("pair_id",sort=False).size().to_numpy()
        yrel=relevance(tr["evidence_rank"].to_numpy(),scheme)
        ms=[lgb.train({**RANK_PARAMS,"seed":sd},lgb.Dataset(tr[feats],yrel,group=grp),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
        tem=te & cand
        if tem.sum()==0: continue
        out[tem.nonzero()[0]]=np.mean([m.predict(dh.loc[tem,feats]) for m in ms],axis=0)
    # candidates keep rerank score on top; non-candidates fall below by s1
    return np.where(out>-1e8, out, -1e9+s1)

def m5(dh, score):
    d=dh.copy(); d["hand_score"]=score; return map5(d[d["label"]==1])

def within_pair_pct(dh, score):
    d=dh.copy(); d["v"]=score; return d.groupby("pair_id")["v"].rank(pct=True).to_numpy()

def main():
    dh=build_dh_ranked()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet"); dh=dh.join(emb,on=["pair_id","hand_idx"],how="left")
    per=build_perhand("development",STEP3/"edge"/"equity"/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=set(dh["pair_id"].to_list()))
    dh=dh.join(per,on=["pair_id","hand_id"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    for c in EQEV_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    dh["evidence_rank"]=dh["evidence_rank"].fillna(0).astype(int); dh["is_evidence"]=dh["evidence_rank"]>0

    BASE={fm:HAND_FEATS for fm in TARGET_BEHAVIORS}
    CTX ={fm:HAND_FEATS+CTXCOLS for fm in TARGET_BEHAVIORS}
    EQ  ={fm:(HAND_FEATS+CTXCOLS+EQEV_FEATS if fm in TS else HAND_FEATS+CTXCOLS) for fm in TARGET_BEHAVIORS}

    log("=== A) per-voter MAP@5 (V4 re-ranker dropped: proven negative §180) ===")
    V={}
    V["V1_base"]  = pointwise(dh,BASE);  log(f"  V1 base   {m5(dh,V['V1_base']):.4f}")
    V["V2_ctx"]   = pointwise(dh,CTX);   log(f"  V2 ctx    {m5(dh,V['V2_ctx']):.4f}")
    V["V3_equity"]= pointwise(dh,EQ);    log(f"  V3 equity {m5(dh,V['V3_equity']):.4f}")
    bestsingle=max(m5(dh,V[k]) for k in V)

    log("=== B) RANK-CONSENSUS (mean within-pair percentile across subsets) vs best single ===")
    pct={k:within_pair_pct(dh,v) for k,v in V.items()}
    subsets=[("ctx+equity",     ["V2_ctx","V3_equity"]),
             ("base+ctx",       ["V1_base","V2_ctx"]),
             ("base+ctx+equity",["V1_base","V2_ctx","V3_equity"])]
    for name,keys in subsets:
        cons=np.mean([pct[k] for k in keys],axis=0)
        log(f"  consensus[{name:16s}] MAP@5 {m5(dh,cons):.4f}  vs best-single {bestsingle:.4f}  d {m5(dh,cons)-bestsingle:+.4f}")
    # weighted: lean on the strongest voter (ctx), equity as tiebreak
    for w in [(0.7,0.3),(0.8,0.2),(0.9,0.1)]:
        cons=w[0]*pct["V2_ctx"]+w[1]*pct["V3_equity"]
        log(f"  consensus[ctx*{w[0]}+eq*{w[1]}] MAP@5 {m5(dh,cons):.4f}  d {m5(dh,cons)-bestsingle:+.4f}")

    log("=== C) DISAGREEMENT diagnostic: top-10 zone, who is right when voters split on top-5 ===")
    d=dh[dh["label"]==1].copy()
    # per pair, each voter's top-5 set membership
    for k in V: d[k+"_top5"]=0
    for _,idx in d.groupby("pair_id").groups.items():
        pass
    # simpler: compute per-voter within-pair rank, flag top5
    for k,v in V.items():
        d[k+"_rk"]=pd.Series(v,index=dh.index).loc[d.index]
    for k in V:
        d[k+"_r"]=d.groupby("pair_id")[k+"_rk"].rank(method="first",ascending=False)
    # a hand is in "disagreement zone" if some voter has it top5 and another doesn't, and it's a top-10 candidate by ctx
    d["ctx_r"]=d.groupby("pair_id")["V2_ctx_rk"].rank(method="first",ascending=False)
    zone=d[(d["ctx_r"]<=10)].copy()
    top5={k:(zone[k+"_r"]<=5) for k in V}
    agree_all=np.all([top5[k].to_numpy() for k in V],axis=0)
    disagree=(~agree_all) & np.any([top5[k].to_numpy() for k in V],axis=0)
    zd=zone[disagree]
    log(f"  top-10 candidate hands: {len(zone)}; in disagreement zone: {len(zd)}; true among them: {int(zd['is_evidence'].sum())}")
    for k in V:
        picks=zd[zd[k+"_r"]<=5]
        hit=picks["is_evidence"].mean() if len(picks) else float('nan')
        log(f"    when {k:10s} puts a disputed hand in top5: it's TRUE {hit:.3f} ({len(picks)} picks)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
