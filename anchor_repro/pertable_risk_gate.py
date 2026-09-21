"""PER-TABLE RELATIVE risk (structural reframe, not a tweak): make risk phase-immune by construction.

Thesis: our only clean LB transfers are phase-immune-by-construction (within-pair evidence) or low-drift
learned (seq rep). Global risk DRIFTS (dev->eval). Collusion is a WITHIN-TABLE phenomenon (disjoint
30-player pools) and PairAP rewards ORDERING. A pair's within-table RANK is scale-free across dev->eval
= phase-immune, same mechanism as evidence. tdir_p95_pertable had the LEAST drift of any risk feature.

CHEAP TEST: take the current-best dev OOF risk (base+PT+seq_risk), form global-rank and per-table-rank,
sweep a blend risk_w = (1-w)*global + w*pertable, measure at each w:
  - dev eval-mirrored PairAP (all pairs)
  - eval-weighted held-out PairAP (the §147 trustworthy gate)
If any w>0 improves/holds the eval-weighted gate, table-relative framing is the plateau lever -> build
the full per-table-trained model (task #2). Worst case: w=0 is best -> no ship, learned it's not the lever.
Run:  python -m anchor_repro.pertable_risk_gate
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    tbl=dev["table_id"].astype(str).to_numpy()
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    # distill seq_risk (dev OOF)
    def distill():
        Xd=dev[SEQF].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=500); o1[te]=m.predict(Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=500); o[te]=m.predict(Xd[te])
        return o
    dev["seq_risk"]=distill()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk"]
    FEATS=base+PT+["seq_risk"]

    # eval-importance weights (§147)
    shared=[c for c in base if c in evf.columns]; n=min(len(dev),len(evf),40000)
    evf2=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    for c in PT: evf2[c]=evf2[c].astype("float32").fillna(0)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf2.iloc[rng.choice(len(evf2),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()

    heldout=[3,4]; heldm=np.isin(folds,heldout)
    # full OOF risk (all folds) for the global risk
    def oof_full():
        X=dev[FEATS].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]=m.predict(X[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); o[te]+=m.predict(X[te])/3
        return o
    g=oof_full()
    log("global OOF risk built")

    # global rank (over all pairs) vs per-table rank
    gr=pd.Series(g).rank(pct=True).to_numpy()
    df=pd.DataFrame({"g":g,"tbl":tbl}); ptr=df.groupby("tbl")["g"].rank(pct=True).to_numpy()

    print("\n=== per-table blend sweep: risk = (1-w)*global_rank + w*pertable_rank ===")
    print(f"{'w':>5} | {'full PairAP':>11} | {'held EVAL-W':>11}")
    for w in [0.0,0.1,0.2,0.3,0.5,0.75,1.0]:
        r=(1-w)*gr + w*ptr
        full=average_precision_score(y,r)
        hw=average_precision_score(y[heldm],r[heldm],sample_weight=w_eval[heldm])
        print(f"{w:5.2f} | {full:11.4f} | {hw:11.4f}")
    print("\n  w=0 is the current global risk. If any w>0 raises held EVAL-W above w=0, table-relative helps.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
