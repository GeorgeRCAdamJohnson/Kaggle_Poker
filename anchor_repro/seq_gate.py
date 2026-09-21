"""Decisive test: does the learned SEQUENCE EMBEDDING clear the double gate vs the 0.81092 base?

Adds the 192-dim self-supervised pair embedding (seq_emb) to the risk model on top of base+postflop+tight,
judged on full-CV AND held-out-table PairAP. Per §143, local breaks down <0.005, so we require a LARGE
held-out gain (>0.005) to consider it real. Also reports the embedding ALONE (base + seq only, no pvf)
and the standalone predictive AUC of the embedding, to see if the representation carries independent signal.
Run:  python -m anchor_repro.seq_gate
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
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c in dev.columns]
    log(f"base {len(base)}, PT {len(PT)}, seq {len(SEQF)}")
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    def gate(feats,tag):
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
        log(f"[{tag}] full {full:.4f} | held-out {ho:.4f}"); return full,ho
    b_f,b_h=gate(base+PT,"base+PT (0.81092)")
    s_f,s_h=gate(base+PT+SEQF,"+seq embedding")
    print(f"\n  SEQ delta vs base+PT: full {s_f-b_f:+.4f} | held-out {s_h-b_h:+.4f}")
    print(f"  ({'STRONG STAGE (>0.005 held-out)' if s_h>b_h+0.005 else 'STAGE' if (s_f>b_f+0.001 and s_h>b_h+0.001) else 'sub-noise / no'})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
