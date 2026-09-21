"""Push the CLEAN table-conditioning config (fold-blind reference, §164 = +0.0047) to its purest:
  1. rank each signal's individual eval-weighted contribution (add its _tz/_tp alone),
  2. greedily keep the winners + test an EXPANDED signal set,
  3. report the best clean config to ship.
Fold-blind z-score computed VECTORIZED (per fold: table mean/std from other folds via groupby, mapped).
Run:  python -m anchor_repro.table_prune
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
# expanded candidate signal set for table-relative conditioning
CAND=["transfer_dominant","transfer_rate","transfer_imbalance","direction_consistency","seq_risk",
      "post_loose_pf_sum","post_loose_pf_min","tight_pf_sum","tight_pf_max","both_post_min_mean","both_post_min_top3",
      "both_post_surp_ct","total_loose_pf_sum"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    evf=pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"]=evf["pair_id"].astype(str)
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    for c in PT: evf[c]=evf[c].astype("float32").fillna(0)
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

    # VECTORIZED fold-blind table-relative features: for each fold, table mean/std from OTHER folds
    tbl=dev["table_id"].astype(str)
    def foldblind_feats(cols):
        feats={}
        tbl_np=tbl.to_numpy()
        cvals={c:dev[c].to_numpy() for c in cols}
        for c in cols:
            tz=np.zeros(len(dev),np.float32)
            for f in range(N_FOLDS):
                te=(folds==f); ref=~te
                gm=dev.loc[ref].groupby("table_id")[c].mean().to_dict(); gs=dev.loc[ref].groupby("table_id")[c].std().to_dict()
                idx=np.where(te)[0]
                mu=np.array([gm.get(tbl_np[i],np.nan) for i in idx]); sd=np.array([gs.get(tbl_np[i],np.nan) for i in idx])
                sd=np.where((sd>0)&np.isfinite(sd),sd,np.nan)
                tz[idx]=np.nan_to_num((cvals[c][idx]-mu)/sd,nan=0.0)
            feats[f"{c}_tz"]=tz
            feats[f"{c}_tp"]=dev.groupby("table_id")[c].rank(pct=True).fillna(0.5).to_numpy().astype(np.float32)
        return feats

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk"]
    shared=[c for c in base if c in evf.columns]; n=min(len(dev),len(evf),40000)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf.iloc[rng.choice(len(evf),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    # precompute all fold-blind feats once
    allf=foldblind_feats(CAND)
    dev=pd.concat([dev, pd.DataFrame(allf, index=dev.index)], axis=1).copy()  # single insert (no fragmentation)
    log(f"precomputed {len(allf)} fold-blind feats for {len(CAND)} signals")

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
        return average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])
    BASE=base+PT+["seq_risk"]; bw=oof(BASE); log(f"baseline eval-W {bw:.4f}")

    # 1) per-signal individual contribution
    print("\n=== per-signal fold-blind contribution (add one signal's _tz+_tp alone) ===")
    contrib={}
    for c in CAND:
        d=oof(BASE+[f"{c}_tz",f"{c}_tp"])-bw; contrib[c]=d; log(f"  {c:22s} {d:+.4f}")
    winners=[c for c,d in sorted(contrib.items(),key=lambda x:-x[1]) if d>0]
    log(f"positive-contribution signals ({len(winners)}): {winners}")

    # 2) all winners together (pruned pure set) + all CAND (full)
    print("\n=== combined configs ===")
    allkeys=[f"{c}_{s}" for c in CAND for s in ("tz","tp")]
    winkeys=[f"{c}_{s}" for c in winners for s in ("tz","tp")]
    log(f"ALL {len(CAND)} signals   : {oof(BASE+allkeys)-bw:+.4f}")
    if winners: log(f"PRUNED winners only  : {oof(BASE+winkeys)-bw:+.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
