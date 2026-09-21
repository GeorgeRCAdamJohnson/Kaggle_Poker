"""PURE table-conditioning (§162 -> the drift-free version). Normalize each pair against its OWN TABLE's
DEV-PHASE baseline — valid across the dev->eval boundary because all 400 tables persist in both phases
(same 30 players). The table-relative feature then means EXACTLY the same thing in dev and eval.

Upgrades vs §162 approximation:
  1. Reference = the table's DEV-phase pair distribution (common yardstick for dev AND eval pairs),
     not the within-frame table mean. Drift-free by construction.
  2. Condition MORE strong signals (all of PT + the transfer/direction/seq_risk set).
  3. Add per-table RANK of the composite risk itself (built from a first-pass model).
Gate: base+PT+seq_risk vs +pure-table-relative. Ship if eval-weighted > +0.005 (clean).
Run:  python -m anchor_repro.table_conditioned_pure
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
STRONG=["transfer_dominant","transfer_rate","transfer_imbalance","direction_consistency","seq_risk",
        "post_loose_pf_sum","post_loose_pf_min","tight_pf_sum","tight_pf_max","both_post_min_mean","both_post_min_top3"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def table_baseline_features(dev, ref_stats, cols):
    """z-score + percentile of each pair's cols against the DEV-phase per-table baseline (ref_stats:
    table_id -> {col: (mean,std,sorted_values)}). Percentile via searchsorted on the dev reference."""
    out={}
    tbl=dev["table_id"].astype(str).to_numpy()
    for c in cols:
        if c not in dev.columns: continue
        v=dev[c].to_numpy(); z=np.zeros(len(v),np.float32); pct=np.full(len(v),0.5,np.float32)
        mu=np.array([ref_stats.get(t,{}).get(c,(0.0,1.0,None))[0] for t in tbl])
        sd=np.array([ref_stats.get(t,{}).get(c,(0.0,1.0,None))[1] for t in tbl]); sd=np.where(sd<=0,np.nan,sd)
        z=((v-mu)/sd)
        # percentile against the table's dev-phase sorted reference
        for i,t in enumerate(tbl):
            st=ref_stats.get(t,{}).get(c)
            if st is not None and st[2] is not None and len(st[2])>0:
                arr=st[2]; pct[i]=np.searchsorted(arr,v[i],side="right")/len(arr)
        out[f"{c}_tz"]=np.nan_to_num(z,nan=0.0).astype(np.float32)
        out[f"{c}_tp"]=pct.astype(np.float32)
    return out

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

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

    # DEV-phase per-table baseline (reference = ALL dev pairs at each table; the common yardstick).
    # NOTE: dev IS the development-phase pairs, so this is the correct reference set.
    ref_stats={}
    for t,grp in dev.groupby("table_id"):
        d={}
        for c in STRONG:
            if c in grp.columns:
                vals=np.sort(grp[c].to_numpy().astype(np.float64))
                d[c]=(float(vals.mean()),float(vals.std()),vals)
        ref_stats[str(t)]=d
    tr_dev=table_baseline_features(dev,ref_stats,STRONG)
    for k,v in tr_dev.items(): dev[k]=v
    TRELF=list(tr_dev.keys())
    log(f"pure dev-phase-baseline table-relative feats: {len(TRELF)}")

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk" and c not in TRELF]

    evf2=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    for c in PT: evf2[c]=evf2[c].astype("float32").fillna(0)
    shared=[c for c in base if c in evf2.columns]; n=min(len(dev),len(evf2),40000)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf2.iloc[rng.choice(len(evf2),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()

    heldout=[3,4]; heldm=np.isin(folds,heldout)
    def oof(feats):
        X=dev[feats].astype("float32"); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]=m.predict(X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); o[te]+=m.predict(X[te])/3
        return o
    def gate(feats,tag):
        o=oof(feats); hp=average_precision_score(y[heldm],o[heldm]); hw=average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm]); full=average_precision_score(y,o)
        log(f"[{tag}] full {full:.4f} | held PLAIN {hp:.4f} | held EVAL-W {hw:.4f}"); return hp,hw
    print("\n=== PURE table-conditioning (dev-phase baseline) gate ===")
    b_p,b_w=gate(base+PT+["seq_risk"],"base+PT+seq_risk")
    e_p,e_w=gate(base+PT+["seq_risk"]+TRELF,"+pure table-relative")
    print(f"\n  pure table-relative delta: plain {e_p-b_p:+.4f} | EVAL-WEIGHTED {e_w-b_w:+.4f}")
    print("  SHIP if EVAL-WEIGHTED > +0.005 (clean, above the §154 burn zone).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
