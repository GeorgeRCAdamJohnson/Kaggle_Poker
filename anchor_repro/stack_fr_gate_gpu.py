"""GPU (XGBoost cuda) STACK gate — fast DELTA measurement for §223 Option 2.

Measures the DELTA of stacking two ground-up field-relative residuals (EQ conf-weighted + JA k=3000) on a
fully-consistent XGBoost-GPU base (seq_risk ALSO XGBoost-distilled, per §223's estimator-mismatch lesson).
A delta is estimator-consistent and valid on XGBoost alone; the absolute-score / bank-comparability question
is deferred to a LightGBM confirm run IF the stack clears +0.005 here.

base | +EQ | +JA | +EQ+JA(stack), 5 seeds, eval-weighted held[3,4].
Run:  python -m anchor_repro.stack_fr_gate_gpu
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, xgboost as xgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_field_relative import FEATS_FR, FEATS_FR_AUX, add_shrunk as add_eq
from anchor_repro.ja_field_relative import FEATS_JA, FEATS_JA_AUX, add_shrunk_ja
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"
N_FOLDS=5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
XP=dict(objective="binary:logistic",tree_method="hist",device="cuda",max_depth=6,eta=0.03,subsample=0.8,colsample_bytree=0.5,reg_lambda=10.0,min_child_weight=20,verbosity=0)
NR=500
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)
def _tr(X,yv,w,sd): return xgb.train({**XP,"seed":int(sd)},xgb.DMatrix(X,label=yv,weight=w),num_boost_round=NR)
def _pr(m,X): return m.predict(xgb.DMatrix(X))

def main():
    eq=add_eq(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas()); eq["pair_id"]=eq["pair_id"].astype(str)
    ja=add_shrunk_ja(pl.read_parquet(EQ/"ja_field_relative_development.parquet").to_pandas()); ja["pair_id"]=ja["pair_id"].astype(str)
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    dev=dev.merge(eq,on="pair_id",how="left").merge(ja[["pair_id"]+FEATS_JA+FEATS_JA_AUX],on="pair_id",how="left")
    for c in PT+SEQF+FEATS_FR+FEATS_FR_AUX+FEATS_JA+FEATS_JA_AUX: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    nopp=dev["fr_n_opp"].to_numpy(); conf=(nopp/(nopp+100.0)).astype("float32")
    EQC=[c+"_c" for c in FEATS_FR]
    for c in FEATS_FR: dev[c+"_c"]=(dev[c].to_numpy()*conf).astype("float32")
    TABLES=sorted(dev["table_id"].unique().tolist())
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","n_hands"}
    DROP=set(PT+SEQF+FEATS_FR+FEATS_FR_AUX+FEATS_JA+FEATS_JA_AUX+EQC)|{"seq_risk"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in DROP]
    evf=pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"]=evf["pair_id"].astype(str)
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    shared=[c for c in base if c in evf.columns]; nn=min(len(dev),len(evf),40000); evf=evf[["pair_id"]+shared].copy()

    def distill(folds,sd):
        Xd=dev[SEQF].to_numpy(np.float32); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1).astype(np.float32); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f; o1[te]=_pr(_tr(Xd[tr],y[tr],w1[tr],sd),Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))).astype(np.float32); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f; o[te]=_pr(_tr(Xd[tr],y2[tr],w2[tr],sd),Xd[te])
        return o
    def oof(feats,folds,heldm,sd):
        X=dev[feats].to_numpy(np.float32); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1).astype(np.float32); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f; o1[te]=_pr(_tr(X[tr],y[tr],w1[tr],sd),X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0).astype(np.float32)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for s2 in [sd,sd+11,sd+23]: o[te]+=_pr(_tr(X[tr],y2[tr],w2[tr],s2),X[te])/3
        return o
    def weval(sd):
        r=np.random.RandomState(sd); ds=dev.iloc[r.choice(len(dev),nn,replace=False)]; es=evf.iloc[r.choice(len(evf),nn,replace=False)]
        Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
        yc=np.concatenate([np.zeros(nn),np.ones(nn)]).astype(np.float32)
        dc=xgb.train({**XP,"eta":0.05},xgb.DMatrix(Xc,label=yc),num_boost_round=200)
        pe=dc.predict(xgb.DMatrix(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)))
        w=np.clip(pe/(1-pe+1e-6),0.05,20.0); return w/w.mean()

    R={"base":[],"eq":[],"ja":[],"stack":[]}
    for sd in [42,101,202,303,404]:
        r=np.random.RandomState(sd); TF={t:int(v) for t,v in zip(TABLES,r.permutation(len(TABLES))%N_FOLDS)}
        folds=dev["table_id"].map(TF).to_numpy(); heldm=np.isin(folds,[3,4])
        dev["seq_risk"]=distill(folds,sd); w=weval(sd)
        b=oof(base+PT+["seq_risk"],folds,heldm,sd)
        e=oof(base+PT+["seq_risk"]+EQC,folds,heldm,sd)
        j=oof(base+PT+["seq_risk"]+FEATS_JA,folds,heldm,sd)
        s=oof(base+PT+["seq_risk"]+EQC+FEATS_JA,folds,heldm,sd)
        bw=average_precision_score(y[heldm],b[heldm],sample_weight=w[heldm]); ew=average_precision_score(y[heldm],e[heldm],sample_weight=w[heldm])
        jw=average_precision_score(y[heldm],j[heldm],sample_weight=w[heldm]); sw=average_precision_score(y[heldm],s[heldm],sample_weight=w[heldm])
        R["base"].append(bw); R["eq"].append(ew); R["ja"].append(jw); R["stack"].append(sw)
        log(f"seed{sd}: base {bw:.4f} | +EQ {ew-bw:+.4f} | +JA {jw-bw:+.4f} | +STACK {sw-bw:+.4f}")
    b=np.array(R["base"])
    print("\n=== GPU-XGB STACK gate (DELTA), 5 seeds — delta over base+PT+seq_risk ===")
    for tag,k in [("+EQ (conf)","eq"),("+JA (k3000)","ja"),("+EQ+JA STACK","stack")]:
        d=np.array(R[k])-b; print(f"  {tag:16s} mean {d.mean():+.4f}  ({int((d>0).sum())}/5)  {np.round(d,4).tolist()}")
    d=np.array(R["stack"])-b; go=(d.mean()>0.005) and ((d>0).sum()>=4)
    print(f"  STACK -> {'CLEARS +0.005 (confirm on LightGBM before submit)' if go else 'below bar'} on GPU-XGB delta")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
