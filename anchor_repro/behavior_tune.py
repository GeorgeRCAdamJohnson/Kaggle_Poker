"""Behavior head tune (10% weight, currently 0.731) — the one component we never optimized ourselves.

Behavior MAP = mean over 3 families of OvR-AP where predicted_behavior==family uses risk else 0.
Levers: (1) the routing rate rho (fraction of top-risk pairs that get a family vs 'none'); (2) the
family-classifier prior weighting (predict_family divides by family prior). hosen42 picks rho on the
dev host-metric. We sweep rho AND a per-family prior-power on the dev population host metric to
maximize BehaviorMAP without hurting PairAP (risk unchanged). Reports best rho/power and the host
metric delta. Uses the reproduced pipeline's OOF risk + family OOF (rebuild the family heads).
Run:  python -m anchor_repro.behavior_tune
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"
SEED,N_FOLDS=42,5
TARGET=["directed_transfer","soft_play","coordinated_isolation"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    from anchor_repro.policy_edge_pvf import pvf_features, PVF_FEATS
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    de=pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    dev=dev.merge(de,on="pair_id",how="left")
    for c in PVF_FEATS: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    behavior_y=np.array([{"directed_transfer":1,"soft_play":2,"coordinated_isolation":3}.get(b,0) for b in dev["behavior_family"].fillna("none")])
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    FEATS=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and not c.startswith("evl_")]
    X=dev[FEATS].astype("float32")

    # OOF risk (the 0.81092 config) + per-family OOF probs
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w1>0); te=folds==f
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); oof1[te]=m.predict(X[te])
    pos=oof1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
    y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
    oof=np.zeros(len(dev)); oof_fam={f_:np.zeros(len(dev)) for f_ in range(3)}
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w2>0); te=folds==f
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(X[te])/3
        # family OvR heads (trained on labelled positives only)
        trf=tr & labm & (~((y2==1)&~labm))
        for k in range(3):
            yf=(behavior_y==k+1).astype(int)
            mf=lgb.train({**P,"num_leaves":7},lgb.Dataset(X[trf],yf[trf],weight=w2[trf]),num_boost_round=400)
            oof_fam[k][te]=mf.predict(X[te])
    log("OOF risk + family heads done")

    fam_prior={k:(behavior_y==k+1).sum() for k in range(3)}
    def behavior_map(active, power):
        fam_mat=np.column_stack([oof_fam[k]/(fam_prior[k]**power) for k in range(3)])
        fam_pred=fam_mat.argmax(1)
        scores=[]
        for k in range(3):
            bt=(behavior_y==k+1).astype(int)
            cls=np.where(active & (fam_pred==k), oof, 0.0)
            scores.append(average_precision_score(bt, cls))
        return float(np.mean(scores))

    rank_pct=pd.Series(oof).rank(ascending=False,method="first").to_numpy()/len(oof)
    print("\n=== behavior tune: rho x prior-power -> BehaviorMAP (baseline ~0.731 at rho~0.05,power=1) ===")
    best=(-1,None)
    for rho in [0.01,0.02,0.03,0.05,0.08,0.12,0.20,0.35,0.50]:
        active=rank_pct<=rho
        for power in [0.0,0.5,1.0]:
            bm=behavior_map(active,power)
            if bm>best[0]: best=(bm,(rho,power))
        # print at power=1 for the row
        print(f"  rho={rho:<5} power=1.0 BehaviorMAP={behavior_map(active,1.0):.4f}")
    print(f"\n  BEST BehaviorMAP={best[0]:.4f} at rho,power={best[1]}  (host weight 0.10 -> +{0.10*(best[0]-0.731):+.4f} if baseline 0.731)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
