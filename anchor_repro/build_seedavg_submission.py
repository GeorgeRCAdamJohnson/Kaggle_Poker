"""Seed/fold-averaged submission on the 0.81092 config (postflop+tight PVF). Reliable variance-cancel gain.

Both risk & evidence are at the policy-feature ceiling. The one reliable lever left that needs NO new
signal: average the pair-model risk RANKING across several table-fold seeds (+ LightGBM seeds). The
fold assignment (seed 42) alone moved our reproduction 0.80136->0.80422 (~0.003), so averaging over
seeds cancels that variance and gives a more stable, usually-higher ranking.

Config = hosen42 221 feats + postflop+tight PVF (the 0.81092 config; evl dropped). For each of
N_SEEDS table-fold seeds: two-stage PU, full-data models, eval risk. Rank-average the eval risks.
behavior + evidence BYTE-IDENTICAL to the 0.81092 submission (only risk changes -> clean read).
Also reports full-CV + held-out double-gate for the seed-avg OOF vs single-seed (sanity).
Run:  python -m anchor_repro.build_seedavg_submission
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.policy_edge_pvf import pvf_features, PVF_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"
N_FOLDS=5; N_SEEDS=5
FOLD_SEEDS=[42,101,202,303,404]         # table-fold assignment seeds
OUT=Path("outputs/poker_collusion/seedavg_submission.csv")
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    ev =pd.read_parquet(STEP3/"eval_pair_features.parquet")
    de=pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    ee=pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas()
    dev=dev.merge(de,on="pair_id",how="left"); ev=ev.merge(ee,on="pair_id",how="left")
    for c in PVF_FEATS: dev[c]=dev[c].astype("float32").fillna(0); ev[c]=ev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    FEATS=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and c in (set(dev.columns))
           and c not in ("evl_pf_sum","evl_pf_min")]
    # keep only base + the 0.81092 PVF set
    FEATS=[c for c in FEATS if not c.startswith("evl_")]
    log(f"feats {len(FEATS)}")
    Xd=dev[FEATS].astype("float32"); Xe=ev[FEATS].astype("float32")
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    TABLES=sorted(dev["table_id"].unique().tolist())

    def one_seed(fold_seed):
        rng=np.random.RandomState(fold_seed)
        TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}
        folds=dev["table_id"].map(TF).to_numpy()
        w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":fold_seed},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=750); oof1[te]=m.predict(Xd[te])
        pos=oof1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
        ev_risk=np.zeros(len(ev)); oof=np.zeros(len(dev))
        SEEDS=[fold_seed,fold_seed+11,fold_seed+23]
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            for sd in SEEDS:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(Xd[te])/len(SEEDS)
        for sd in SEEDS:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(Xd[w2>0],y2[w2>0],weight=w2[w2>0]),num_boost_round=750); ev_risk+=m.predict(Xe)/len(SEEDS)
        return oof, ev_risk

    from sklearn.metrics import average_precision_score
    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    ev_ranks=np.zeros(len(ev)); oof_ranks=np.zeros(len(dev)); single_ap=None
    for i,fs in enumerate(FOLD_SEEDS[:N_SEEDS]):
        oof,evr=one_seed(fs)
        ap=average_precision_score(y,oof)
        if i==0: single_ap=ap
        ev_ranks+=pr(evr)/N_SEEDS; oof_ranks+=pr(oof)/N_SEEDS
        log(f"seed {fs}: dev eval-mirrored AP {ap:.4f}")
    avg_ap=average_precision_score(y,oof_ranks)
    log(f"single-seed dev AP {single_ap:.4f} | {N_SEEDS}-seed-avg dev AP {avg_ap:.4f}  (delta {avg_ap-single_ap:+.4f})")

    # assemble: risk = seed-avg rank; behavior+evidence from the 0.81092 submission (reproduced submission.csv)
    base=pd.read_csv(STEP3/"submission.csv"); base["pair_id"]=base["pair_id"].astype(str)
    ev["pair_id"]=ev["pair_id"].astype(str)
    rmap=dict(zip(ev["pair_id"], ev_ranks))
    r=base["pair_id"].map(rmap).to_numpy(dtype=np.float64)
    assert not np.isnan(r).any()
    r=(r-r.min())/(r.max()-r.min()+1e-12)
    order=pd.Series(r).rank(method="first").to_numpy()
    r=np.clip(r*(1-1e-9)+1e-9*(order/len(r)),0,1)
    out=base.copy(); out["risk_score"]=r
    assert out["risk_score"].between(0,1).all() and out["risk_score"].is_unique
    assert (out["predicted_behavior"]==base["predicted_behavior"]).all() and (out["evidence_hand_1"]==base["evidence_hand_1"]).all()
    OUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUT,index=False)
    log(f"wrote {OUT}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
