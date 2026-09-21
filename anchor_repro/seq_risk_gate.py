"""Gate the DISTILLED seq_risk (1 col, decorrelated 0.94-AUC learned representation) vs base+PT.

Embedding-only PairAP 0.2248 / AUC 0.9375, Spearman 0.207 vs base risk -> genuinely INDEPENDENT signal
from the raw action stream. Raw 192 dims diluted the tree (-0.015); a single distilled column integrates
cleanly. Test: (a) add seq_risk column to base+PT, double-gate; (b) rank-BLEND base-oof with seq_risk
(decorrelated blend, the classic add). Only a LARGE held-out gain (>0.005, §143) earns a submission.
Run:  python -m anchor_repro.seq_risk_gate
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    sr=pd.read_parquet(SEQ/"seq_risk_dev.parquet"); dev["pair_id"]=dev["pair_id"].astype(str); sr["pair_id"]=sr["pair_id"].astype(str)
    dev=dev.merge(sr,on="pair_id",how="left")
    for c in PT+["seq_risk"]: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c!="seq_risk" and not c.startswith("seq_")]
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    def gate(feats,tag,ret_oof=False):
        X=dev[feats].astype("float32"); pool=~heldm
        w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); oof1[te]=m.predict(X[te])
        pos=oof1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp)&pool; ps=unl&(oof1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(X[te])/3
        full=average_precision_score(y,oof); ho=average_precision_score(y[heldm],oof[heldm])
        log(f"[{tag}] full {full:.4f} | held-out {ho:.4f}")
        return (full,ho,oof) if ret_oof else (full,ho)
    b_f,b_h,b_oof=gate(base+PT,"base+PT",ret_oof=True)
    c_f,c_h=gate(base+PT+["seq_risk"],"+seq_risk col")
    # rank blend base_oof + seq_risk
    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    print(f"\n  +seq_risk col delta: full {c_f-b_f:+.4f} | held-out {c_h-b_h:+.4f}")
    print("  rank-blend base_oof + w*seq_risk (held-out only shown):")
    for w in [0.05,0.1,0.2,0.3]:
        bl=(1-w)*pr(b_oof)+w*pr(dev["seq_risk"].to_numpy())
        print(f"    w={w}: full {average_precision_score(y,bl):.4f} | held-out {average_precision_score(y[heldm],bl[heldm]):.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
