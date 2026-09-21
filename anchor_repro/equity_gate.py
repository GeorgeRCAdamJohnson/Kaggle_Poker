"""DECISIVE eval-weighted phase gate for equity-surrender features on FULL dev (task #3).

Full-dev equity now exists (27.3M rows, all 122k pairs). Build equity-surrender features for ALL dev
pairs, then run the §147 eval-weighted held-out gate: base+PT+seq_risk vs +equity. This is the real test
on the eval-mirrored population (positives vs all pairs, prevalence 0.2%, PairAP ~0.73 with headroom) —
unlike the saturated labelled pre-gate (§168). SHIP only if eval-weighted delta > +0.005 clean.
Run:  python -m anchor_repro.equity_gate
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_features import build_features, FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    # seq_risk distill (full two-step PU as in prior gates)
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
    dev["seq_risk"]=distill(); log("seq_risk distilled")

    # equity-surrender features on FULL dev (all pairs)
    agg=build_features("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=None).to_pandas()
    dev=dev.merge(agg,on="pair_id",how="left")
    for c in FEATS: dev[c]=dev[c].astype("float32").fillna(0)
    log(f"equity features merged; coverage {dev['eq_surrendered_mean'].gt(0).mean():.2%} pairs with surrender>0")

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","n_hands"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS]

    # eval-importance weights
    evf=pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"]=evf["pair_id"].astype(str)
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    shared=[c for c in base if c in evf.columns]; n=min(len(dev),len(evf),40000)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf.iloc[rng.choice(len(evf),n,replace=False)]
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
    print("\n=== EQUITY GATE (full-dev, eval-mirrored, the decisive test) ===")
    b_p,b_w=gate(base+PT+["seq_risk"],"base+PT+seq_risk")
    e_p,e_w=gate(base+PT+["seq_risk"]+FEATS,"+equity-surrender")
    print(f"\n  equity delta: plain {e_p-b_p:+.4f} | EVAL-WEIGHTED {e_w-b_w:+.4f}")
    print("  SHIP if EVAL-WEIGHTED > +0.005 clean (above the §154 burn zone).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
