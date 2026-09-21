"""Does the seq embedding carry ANY independent collusion signal, or is it noise? (before killing it)

Adding 192 dims to a 230-feat tree HURT (-0.015). But that could be dilution (feature_fraction=0.5 rarely
samples useful dims) not absence of signal. Check:
  1. embedding-ONLY risk model (no engineered feats) -> full-CV + held-out PairAP. If >> base-rate, the
     representation has real signal; if ~chance, it's noise.
  2. a distilled version: train embedding-only OOF risk, then add just THAT 1 column to base+PT and gate.
     A single strong distilled column integrates better than 192 raw dims.
Run:  python -m anchor_repro.seq_signal_check
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_dev.parquet"); dev["pair_id"]=dev["pair_id"].astype(str); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left")
    SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)

    def oof_pred(feats):
        X=dev[feats].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=600); oof1[te]=m.predict(X[te])
        pos=oof1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
        oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=600); oof[te]=m.predict(X[te])
        return oof

    # 1. embedding-only
    oof_seq=oof_pred(SEQF)
    log(f"embedding-ONLY: eval-mirrored PairAP {average_precision_score(y,oof_seq):.4f} | AUC {roc_auc_score(y,oof_seq):.4f} (base-rate {y.mean():.4f})")
    # correlation with base risk (is it independent?)
    oof_base=oof_pred(SEQF[:0] or ["shared_hands"])  # placeholder; use precomputed base oof if present
    try:
        ob=np.load(STEP3/"edge"/"oof_base.npy"); from scipy.stats import spearmanr
        log(f"embedding-only vs base risk Spearman: {spearmanr(oof_seq, ob).correlation:.3f}")
    except Exception: pass
    # 2. distilled single column into base+PT
    dev["seq_risk"]=oof_seq
    log("distilled seq_risk column built; add to base+PT and gate separately via pvf_double_gate if promising")
    pd.DataFrame({"pair_id":dev["pair_id"],"seq_risk":oof_seq}).to_parquet(SEQ/"seq_risk_dev.parquet",index=False)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
