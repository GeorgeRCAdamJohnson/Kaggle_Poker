"""BAND-CONDITIONAL re-test of §196 strategy-value CV (§198 Recommendation 1). The §196 gate measured a
GLOBAL eval-weighted delta (-0.015). But §127-128 showed a feature can be weak globally yet additive in a
precision band. Test: slice dev pairs by BASE-model rank; within each band, eval-weighted AP delta of
+CV vs base. If any band shows >+0.005 clean gain, a band-conditional re-rank is worth building.
Reuses the trusted gate machinery (table-fold two-stage PU OOF + adversarial eval-weights).
Run:  python -m anchor_repro.collusion_value_band_gate
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.collusion_value import build_cv_features, CV_FEATS
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
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    cd=build_cv_features("development").to_pandas(); cd["pair_id"]=cd["pair_id"].astype(str)
    dev=dev.merge(cd,on="pair_id",how="left")
    for c in CV_FEATS: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    def distill():
        Xd=dev[SEQF].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            mm=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=500); o1[te]=mm.predict(Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            mm=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=500); o[te]=mm.predict(Xd[te])
        return o
    dev["seq_risk"]=distill()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","nsh"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk" and c not in CV_FEATS]
    shared=[c for c in base if c in evf.columns]
    n=min(len(dev),len(evf),40000)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf.iloc[rng.choice(len(evf),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0); yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    def oof(feats):
        X=dev[feats].astype("float32"); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            mm=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]=mm.predict(X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                mm=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); o[te]+=mm.predict(X[te])/3
        return o
    log("computing base OOF..."); ob=oof(base+PT+["seq_risk"])
    log("computing +CV OOF...");  oe=oof(base+PT+["seq_risk"]+CV_FEATS)
    # global (reproduce §196)
    gb=average_precision_score(y[heldm],ob[heldm],sample_weight=w_eval[heldm]); ge=average_precision_score(y[heldm],oe[heldm],sample_weight=w_eval[heldm])
    log(f"GLOBAL eval-weighted: base {gb:.4f} | +CV {ge:.4f} | delta {ge-gb:+.4f}  (reproduces §196)")

    # BAND-CONDITIONAL: rank held-out pairs by base model, slice into bands, eval-weighted AP delta within each
    print("\n=== §198 Rec 1 — BAND-CONDITIONAL eval-weighted AP delta (base rank bands) ===")
    hy=y[heldm]; hw=w_eval[heldm]; hb=ob[heldm]; he=oe[heldm]
    order=np.argsort(-hb)  # rank held pairs by base risk (desc)
    N=len(hb)
    bands=[("top-100",0,100),("100-500",100,500),("500-2000",500,2000),("2000+",2000,N)]
    for name,lo,hi in bands:
        idx=order[lo:hi]
        if len(idx)<10 or hy[idx].sum()==0: log(f"  {name:10s} n={len(idx)} pos={int(hy[idx].sum())} -> skip (no positives)"); continue
        ab=average_precision_score(hy[idx],hb[idx],sample_weight=hw[idx])
        ae=average_precision_score(hy[idx],he[idx],sample_weight=hw[idx])
        log(f"  {name:10s} n={len(idx):4d} pos={int(hy[idx].sum()):3d} | base AP {ab:.4f} | +CV {ae:.4f} | delta {ae-ab:+.4f}")
    print("  SHIP a band-conditional re-rank only if a band shows >+0.005 clean (leak-checked).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
