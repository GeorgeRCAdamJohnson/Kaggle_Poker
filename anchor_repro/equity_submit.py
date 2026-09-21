"""Assemble the EQUITY risk submission (the LB story datapoint - user wants it even though the gate said
-0.0135, to tell the equity story on the real leaderboard). Risk-only swap on the 0.83460 base
(seq_evidence_familycond_submission.csv -> behavior+evidence byte-identical); risk = base+PT+seq_risk+equity.

Needs: full-dev equity + eval equity (both from equity_build_gpu). Trains two-step PU risk with the equity
features on all dev, scores eval, swaps risk into the current best submission.
Run:  python -m anchor_repro.equity_submit
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_features import build_features, FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"
BASE_SUB=Path("outputs/poker_collusion/evidence_familycond_submission.csv")  # the 0.83460 best
OUT_SUB=Path("outputs/poker_collusion/equity_submission.csv")
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
    de["pair_id"]=de["pair_id"].astype(str); ee["pair_id"]=ee["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); evf=evf.merge(ee,on="pair_id",how="left")
    SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    def distill(df, phase_key):
        Xd=df[SEQF].astype("float32"); return None
    # distill seq_risk on dev (OOF) + eval (full fit)
    Xd=dev[SEQF].astype("float32"); Xe=evf[SEQF].astype("float32")
    w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w1>0); te=folds==f
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=500); o1[te]=m.predict(Xd[te])
    pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
    w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
    seq_o=np.zeros(len(dev)); seq_e=np.zeros(len(evf))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w2>0); te=folds==f
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=500); seq_o[te]=m.predict(Xd[te])
    for sd in [SEED,SEED+11,SEED+23]:
        m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=500); seq_e+=m.predict(Xe)/3
    dev["seq_risk"]=seq_o; evf["seq_risk"]=seq_e; log("seq_risk done")

    aggd=build_features("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=None).to_pandas()
    agge=build_features("evaluation",EQ/"pairhand_equity_eval.parquet","eval_hand_features.parquet",keep_pairs=None).to_pandas()
    dev=dev.merge(aggd,on="pair_id",how="left"); evf=evf.merge(agge,on="pair_id",how="left")
    for c in FEATS: dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    log("equity features merged dev+eval")

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","n_hands"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS]
    FEATSET=base+PT+["seq_risk"]+FEATS

    # full two-step PU risk on eval with equity
    Xd=dev[FEATSET].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w1>0); te=folds==f
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]+=m.predict(Xd[te])/3
    pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
    w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
    oof=np.zeros(len(dev)); evr=np.zeros(len(evf))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w2>0); te=folds==f
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(Xd[te])/3
    log(f"equity-risk OOF eval-mirrored PairAP {average_precision_score(y,oof):.4f}")
    for sd in [SEED,SEED+11,SEED+23]:
        m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=750); evr+=m.predict(evf[FEATSET].astype("float32"))/3

    base_sub=pd.read_csv(BASE_SUB); base_sub["pair_id"]=base_sub["pair_id"].astype(str)
    rmap=dict(zip(evf["pair_id"],evr.astype(np.float64))); r=base_sub["pair_id"].map(rmap).to_numpy(dtype=np.float64)
    assert not np.isnan(r).any()
    r=(r-r.min())/(r.max()-r.min()+1e-12); order=pd.Series(r).rank(method="first").to_numpy(); r=np.clip(r*(1-1e-9)+1e-9*(order/len(r)),0,1)
    out=base_sub.copy(); out["risk_score"]=r
    EV=[f"evidence_hand_{i}" for i in range(1,6)]
    assert (out["pair_id"].values==base_sub["pair_id"].values).all()
    assert (out["predicted_behavior"].values==base_sub["predicted_behavior"].values).all(), "behavior changed"
    assert (out[EV].values==base_sub[EV].values).all(), "evidence changed"
    assert out["risk_score"].between(0,1).all() and out["risk_score"].is_unique
    out.to_csv(OUT_SUB,index=False)
    log(f"wrote {OUT_SUB.name}; behavior+evidence IDENTICAL; risk=equity; Spearman vs base risk {spearmanr(r,base_sub['risk_score']).correlation:.3f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
