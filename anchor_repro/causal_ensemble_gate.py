"""Does the CAUSAL 2nd encoder ADD to risk on top of the masked v2 encoder? (task #2 gate)

Distills BOTH learned embeddings to OOF risk (seq_risk from v2 masked, cau_risk from causal), measures:
  - standalone eval-mirrored PairAP of each,
  - Spearman(seq_risk, cau_risk)  -> decorrelation (low corr => genuinely diverse => ensembles),
  - eval-weighted PHASE gate: base+PT+seq_risk  vs  base+PT+seq_risk+cau_risk.
Ship the ensemble ONLY if the eval-weighted held-out delta is positive (the §147 trustworthy gate).
Also builds eval cau_risk so, if it clears, seq_evidence's sibling assembler can drop it into the risk.

Run:  python -m anchor_repro.causal_ensemble_gate
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
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
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); ee=pd.read_parquet(SEQ/"seq_emb_v2_eval.parquet")
    ca=pd.read_parquet(SEQ/"seq_emb_causal_dev.parquet"); cae=pd.read_parquet(SEQ/"seq_emb_causal_eval.parquet")
    for df in (de,ee,ca,cae): df["pair_id"]=df["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left").merge(ca,on="pair_id",how="left")
    evf=evf.merge(ee,on="pair_id",how="left").merge(cae,on="pair_id",how="left")
    SEQF=[c for c in de.columns if c.startswith("seq_")]; CAUF=[c for c in ca.columns if c.startswith("cau_")]
    for c in PT+SEQF+CAUF: dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF and c not in CAUF
          and not c.startswith("seq_") and not c.startswith("cau_") and c not in ("seq_risk","cau_risk")]

    def distill(FEATS):
        Xd=dev[FEATS].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=500); o1[te]=m.predict(Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
        o=np.zeros(len(dev)); evr=np.zeros(len(evf))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=500); o[te]=m.predict(Xd[te])
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=500); evr+=m.predict(evf[FEATS].astype("float32"))/3
        return o,evr
    seq_o,seq_e=distill(SEQF); cau_o,cau_e=distill(CAUF)
    dev["seq_risk"]=seq_o; dev["cau_risk"]=cau_o; evf["seq_risk"]=seq_e; evf["cau_risk"]=cau_e
    log(f"standalone eval-mirrored PairAP: v2 seq {average_precision_score(y,seq_o):.4f} | causal {average_precision_score(y,cau_o):.4f}")
    log(f"Spearman(seq_risk, cau_risk) = {spearmanr(seq_o,cau_o).correlation:.4f}  (low = diverse = ensembles)")

    # ---- eval-importance weights (density ratio) for the §147 phase gate ----
    shared=[c for c in base if c in evf.columns]
    n=min(len(dev),len(evf),40000)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf.iloc[rng.choice(len(evf),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    p_eval=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0))
    w_eval=np.clip(p_eval/(1-p_eval+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()

    heldout=[3,4]; heldm=np.isin(folds,heldout)
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
        o=oof(feats); hp=average_precision_score(y[heldm],o[heldm]); hw=average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])
        log(f"[{tag}] held-out PLAIN {hp:.4f} | EVAL-WEIGHTED {hw:.4f}"); return hp,hw
    print("\n=== CAUSAL ENSEMBLE GATE (does causal add on top of the current best seq_risk config?) ===")
    b_p,b_w=gate(base+PT+["seq_risk"],"base+PT+seq_risk (current best)")
    e_p,e_w=gate(base+PT+["seq_risk","cau_risk"],"+cau_risk (ENSEMBLE)")
    print(f"\n  causal-ensemble delta: plain {e_p-b_p:+.4f} | EVAL-WEIGHTED {e_w-b_w:+.4f}")
    print("  SHIP if EVAL-WEIGHTED delta > +0.001 (the §147 trustworthy gate).")
    if e_w-b_w>0.001:
        # persist eval risk columns for the assembler
        pd.DataFrame({"pair_id":evf["pair_id"],"seq_risk":seq_e,"cau_risk":cau_e}).to_parquet(SEQ/"ensemble_eval_risk.parquet")
        log("GATE CLEARED -> wrote ensemble_eval_risk.parquet for assembly")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
