"""Full submission: 0.81092 config (base+postflop+tight) + per-street SCALE block on risk.

The per-street block (84 cols) cleared BOTH double gates (held-out +0.0047). Build the eval risk with
it, keep behavior+evidence BYTE-IDENTICAL to the 0.81092 reproduced submission (only risk changes).
Run: python -m anchor_repro.build_scale_submission
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.policy_edge_pvf import pvf_features, PVF_FEATS
from anchor_repro.scale_perstreet import pair_block, build_block
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"
SEED,N_FOLDS=42,5
OUT=Path("outputs/poker_collusion/scale_submission.csv")
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    build_block()
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); ev=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    ev=ev.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    dps=pair_block(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    eps=pair_block(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas()
    dev=dev.merge(dps,on="pair_id",how="left"); ev=ev.merge(eps,on="pair_id",how="left")
    PS_FEATS=[c for c in dps.columns if c!="pair_id"]
    PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
    for c in PT+PS_FEATS:
        dev[c]=dev[c].astype("float32").fillna(0); ev[c]=ev[c].astype("float32").fillna(0)
    log(f"scale block {len(PS_FEATS)} feats merged; dev {dev.shape} eval {ev.shape}")
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PS_FEATS and c not in PT and c not in PVF_FEATS]
    FEATS=base+PT+PS_FEATS
    log(f"total feats {len(FEATS)}")
    Xd=dev[FEATS].astype("float32"); Xe=ev[FEATS].astype("float32")
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    SEEDS=[SEED,SEED+11,SEED+23]
    w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w1>0); te=folds==f
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=750); oof1[te]=m.predict(Xd[te])
    pos=oof1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
    y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
    ev_risk=np.zeros(len(ev)); oof=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w2>0); te=folds==f
        for sd in SEEDS:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(Xd[te])/3
    for sd in SEEDS:
        m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=750); ev_risk+=m.predict(Xe)/3
    from sklearn.metrics import average_precision_score
    log(f"dev eval-mirrored PairAP (scale) {average_precision_score(y,oof):.4f} (base 0.7268)")
    base_sub=pd.read_csv(STEP3/"submission.csv"); base_sub["pair_id"]=base_sub["pair_id"].astype(str); ev["pair_id"]=ev["pair_id"].astype(str)
    rmap=dict(zip(ev["pair_id"],ev_risk.astype(np.float64))); r=base_sub["pair_id"].map(rmap).to_numpy(dtype=np.float64)
    assert not np.isnan(r).any()
    r=(r-r.min())/(r.max()-r.min()+1e-12); order=pd.Series(r).rank(method="first").to_numpy(); r=np.clip(r*(1-1e-9)+1e-9*(order/len(r)),0,1)
    out=base_sub.copy(); out["risk_score"]=r
    assert out["risk_score"].between(0,1).all() and out["risk_score"].is_unique and (out["predicted_behavior"]==base_sub["predicted_behavior"]).all() and (out["evidence_hand_1"]==base_sub["evidence_hand_1"]).all()
    OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False)
    from scipy.stats import spearmanr
    log(f"wrote {OUT}; Spearman(scale risk, 0.81092 risk-proxy)={spearmanr(r, base_sub['risk_score']).correlation:.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
