"""Stack the two orthogonal gate-positive signals: LEARNED sequence embedding + per-street SCALE block.

seq-rep (LB 0.82082) and the per-street scale block (§143, held-out +0.0047) are DIFFERENT signals
(learned action-sequence structure vs per-street engineered aggregates). Add BOTH on top of
base+postflop+tight, distill seq to a column, keep the scale block as features, double-gate, and build
the submission if it clears the current best. behavior+evidence identical to 0.81092.
Run:  python -m anchor_repro.seq_plus_scale
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
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
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    dps=pair_block(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    eps=pair_block(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas()
    dev=dev.merge(dps,on="pair_id",how="left"); evf=evf.merge(eps,on="pair_id",how="left")
    PS_FEATS=[c for c in dps.columns if c!="pair_id"]
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); ee=pd.read_parquet(SEQ/"seq_emb_v2_eval.parquet")
    SEQF=[c for c in de.columns if c.startswith("seq_")]
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str); de["pair_id"]=de["pair_id"].astype(str); ee["pair_id"]=ee["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); evf=evf.merge(ee,on="pair_id",how="left")
    for c in PT+PS_FEATS+SEQF:
        dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)

    # distill seq embedding -> seq_risk (OOF dev + eval)
    def emb_risk():
        Xd=dev[SEQF].astype("float32"); Xe=evf[SEQF].astype("float32")
        w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=600); oof1[te]=m.predict(Xd[te])
        pos=oof1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
        oof=np.zeros(len(dev)); evr=np.zeros(len(evf))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=600); oof[te]=m.predict(Xd[te])
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=600); evr+=m.predict(Xe)/3
        return oof,evr
    soof,sev=emb_risk(); dev["seq_risk"]=soof; evf["seq_risk"]=sev
    log(f"seq_risk distilled (eval-mirrored {average_precision_score(y,soof):.4f})")

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in PS_FEATS and c!="seq_risk" and not c.startswith("seq_")]
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    def gate(feats,tag,eval_too=False):
        Xd=dev[feats].astype("float32"); pool=~heldm
        w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=750); oof1[te]=m.predict(Xd[te])
        pos=oof1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp)&pool; ps=unl&(oof1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        oof=np.zeros(len(dev)); evr=np.zeros(len(evf))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(Xd[te])/3
        if eval_too:
            w2f=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2f>0],y2[w2f>0],weight=w2f[w2f>0]),num_boost_round=750); evr+=m.predict(evf[feats].astype("float32"))/3
        full=average_precision_score(y,oof); ho=average_precision_score(y[heldm],oof[heldm])
        log(f"[{tag}] full {full:.4f} | held-out {ho:.4f}"); return full,ho,evr
    b_f,b_h,_=gate(base+PT,"base+PT")
    q_f,q_h,_=gate(base+PT+["seq_risk"],"+seq only")
    x_f,x_h,x_ev=gate(base+PT+["seq_risk"]+PS_FEATS,"+seq+scale",eval_too=True)
    print(f"\n  seq only:  full {q_f-b_f:+.4f} | held-out {q_h-b_h:+.4f}")
    print(f"  seq+scale: full {x_f-b_f:+.4f} | held-out {x_h-b_h:+.4f}")
    stage = x_h>q_h+0.0005  # scale adds ON TOP of seq
    print(f"  -> scale-on-top-of-seq {'STACKS (build+submit)' if stage else 'no extra gain over seq alone'}")
    if stage:
        base_sub=pd.read_csv(STEP3/"submission.csv"); base_sub["pair_id"]=base_sub["pair_id"].astype(str)
        rmap=dict(zip(evf["pair_id"],x_ev.astype(np.float64))); r=base_sub["pair_id"].map(rmap).to_numpy(dtype=np.float64)
        r=(r-r.min())/(r.max()-r.min()+1e-12); order=pd.Series(r).rank(method="first").to_numpy(); r=np.clip(r*(1-1e-9)+1e-9*(order/len(r)),0,1)
        out=base_sub.copy(); out["risk_score"]=r; out.to_csv("outputs/poker_collusion/seq_scale_submission.csv",index=False)
        log(f"wrote seq_scale_submission.csv (Spearman vs base {spearmanr(r,base_sub['risk_score']).correlation:.3f})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
