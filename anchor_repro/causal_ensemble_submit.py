"""Assemble the ENSEMBLE-risk submission: base+PT+seq_risk+cau_risk (§ causal 2nd encoder cleared the
eval-weighted gate at +0.0039). Swaps ONLY risk_score into the current best 0.83185 submission
(seq_evidence_submission.csv); behavior + evidence stay BYTE-IDENTICAL. Risk-only change.

Reproduces the full-data ensemble risk on eval (two-step PU, 3-seed full fit) exactly like
seq_v2_assemble, but with BOTH distilled learned-embedding risks (v2 masked + causal) as features.
Verifies pair set/order, behavior+evidence identical, risk in [0,1] unique. Run:
  python -m anchor_repro.causal_ensemble_submit
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
BASE_SUB=Path("outputs/poker_collusion/seq_evidence_submission.csv")   # the 0.83185 best (evidence-upgraded)
OUT_SUB=Path("outputs/poker_collusion/causal_ensemble_submission.csv")
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

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF and c not in CAUF
          and not c.startswith("seq_") and not c.startswith("cau_") and c not in ("seq_risk","cau_risk")]
    FEATS=base+PT+["seq_risk","cau_risk"]

    # full-data two-step PU risk on eval (OOF for a sanity PairAP, full-fit 3-seed for eval)
    Xd=dev[FEATS].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w1>0); te=folds==f
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]+=m.predict(Xd[te])/3
    pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
    w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
    oof=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w2>0); te=folds==f
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(Xd[te])/3
    log(f"ensemble OOF eval-mirrored PairAP {average_precision_score(y,oof):.4f}")
    evr=np.zeros(len(evf))
    for sd in [SEED,SEED+11,SEED+23]:
        m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=750); evr+=m.predict(evf[FEATS].astype("float32"))/3

    # ---- swap ONLY risk into the 0.83185 base (behavior + evidence byte-identical) ----
    base_sub=pd.read_csv(BASE_SUB); base_sub["pair_id"]=base_sub["pair_id"].astype(str)
    rmap=dict(zip(evf["pair_id"],evr.astype(np.float64))); r=base_sub["pair_id"].map(rmap).to_numpy(dtype=np.float64)
    assert not np.isnan(r).any(), "some eval pair missing ensemble risk"
    r=(r-r.min())/(r.max()-r.min()+1e-12)
    order=pd.Series(r).rank(method="first").to_numpy(); r=np.clip(r*(1-1e-9)+1e-9*(order/len(r)),0,1)
    out=base_sub.copy(); out["risk_score"]=r
    EV=[f"evidence_hand_{i}" for i in range(1,6)]
    assert (out["pair_id"].values==base_sub["pair_id"].values).all()
    assert (out["predicted_behavior"].values==base_sub["predicted_behavior"].values).all(), "behavior changed!"
    assert (out[EV].values==base_sub[EV].values).all(), "evidence changed!"
    assert out["risk_score"].between(0,1).all() and out["risk_score"].is_unique
    out.to_csv(OUT_SUB,index=False)
    log(f"wrote {OUT_SUB.name}: {len(out):,} rows; behavior+evidence IDENTICAL; risk=ensemble; Spearman vs prev risk {spearmanr(r,base_sub['risk_score']).correlation:.3f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
