"""PHASE-AWARE gate (§147 fix): eval-distribution-weighted held-out PairAP.

The held-out-TABLE gate can't see dev->eval phase drift (all dev tables are dev-phase) -> it
false-positived the scale block (LB-regressed) and we need it to trust only phase-robust gains.
Fix: train a dev-vs-eval classifier on the pair features, get per-dev-pair importance weight
w = P(eval|x)/P(dev|x) (density ratio), and weight the held-out eval-mirrored PairAP by w. A feature
that only helps within the dev phase gets down-weighted where dev!=eval -> the weighted gate rejects it.

VALIDATION: run the gate on (a) the per-street SCALE block (LB-regressed -0.0005/-0.0008) and (b) the
sequence-rep seq_risk (LB-gained +0.0092). The fixed gate should now RANK seq_risk positive and scale
NEGATIVE (or at least seq >> scale), matching the LB. If so, the gate is fixed and trustworthy.
Run:  python -m anchor_repro.phase_gate
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.scale_perstreet import pair_block, build_block
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    build_block()
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    # scale block + seq_risk (reuse cached distilled seq_risk if present via seq emb)
    dps=pair_block(STEP3/"dev_pair_hands.parquet","development").to_pandas(); dev=dev.merge(dps,on="pair_id",how="left")
    PS=[c for c in dps.columns if c!="pair_id"]
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+PS+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PS and c not in SEQF and not c.startswith("seq_")
          and c not in PT and c!="seq_risk"]

    # ---- dev-pair eval-importance weights: P(eval)/P(dev) from a dev-vs-eval classifier on shared feats ----
    shared=[c for c in base if c in evf.columns]
    n=min(len(dev),len(evf),40000)
    dsamp=dev.iloc[rng.choice(len(dev),n,replace=False)]; esamp=evf.iloc[rng.choice(len(evf),n,replace=False)]
    Xc=pd.concat([dsamp[shared],esamp[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])  # 1=eval
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),
                 lgb.Dataset(Xc,yc),num_boost_round=200)
    p_eval=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0))
    w_eval=np.clip(p_eval/(1-p_eval+1e-6),0.05,20.0)  # density ratio P(eval)/P(dev)
    w_eval=w_eval/ w_eval.mean()
    log(f"eval-importance weights: pos mean {w_eval[labm&(y==1)].mean():.3f}, neg mean {w_eval[labm&(y==0)].mean():.3f}")

    heldout=[3,4]; heldm=np.isin(folds,heldout)
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    def oof(feats):
        X=dev[feats].astype("float32"); pool=~heldm
        w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
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
        o=oof(feats)
        ho_plain=average_precision_score(y[heldm],o[heldm])
        ho_w=average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])
        log(f"[{tag}] held-out PLAIN {ho_plain:.4f} | held-out EVAL-WEIGHTED {ho_w:.4f}")
        return ho_plain,ho_w
    # distill seq to a column for the seq gate
    def emb_risk():
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
    dev["seq_risk"]=emb_risk()

    print("\n=== PHASE-AWARE GATE VALIDATION (plain held-out was the biased gate; eval-weighted is the fix) ===")
    b_p,b_w=gate(base+PT,"base+PT")
    s_p,s_w=gate(base+PT+["seq_risk"],"+seq_risk (LB +0.0092)")
    c_p,c_w=gate(base+PT+PS,"+scale block (LB -0.0005)")
    print(f"\n  seq   delta: plain {s_p-b_p:+.4f} | EVAL-WEIGHTED {s_w-b_w:+.4f}")
    print(f"  scale delta: plain {c_p-b_p:+.4f} | EVAL-WEIGHTED {c_w-b_w:+.4f}")
    print("  GATE FIXED if: eval-weighted ranks seq POSITIVE and scale <= seq (ideally scale negative), matching LB.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
