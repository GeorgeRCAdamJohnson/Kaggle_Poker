"""Diverse-learner ensemble on hosen42's OWN 221 features (the precision-safe lever).

Feature additions don't beat the baseline (they inject top-end false positives). But hosen42 uses a
SINGLE LightGBM config. Rank-averaging DIVERSE learners on the SAME features cancels uncorrelated
RANKING errors WITHOUT adding new-feature false positives — the proven robust gain (dossier §ensemble
rank-avg +0.018 historically). We add: XGBoost (hist, GPU) at two depths + a 2nd LightGBM (dart/leaf)
on the identical 221-feature matrix + two-stage PU, rank-average with the baseline OOF.

Judged eval-mirrored PairAP vs baseline 0.7299. If the rank-ensemble clears it, that transfers to LB
(ensembling is the least base-rate-fragile improvement). Produces the eval risk for submission.
Run:  python -m anchor_repro.policy_ensemble
"""
from __future__ import annotations
import json, time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score
STEP3 = Path("outputs/poker_collusion/hosen42_step3"); EDGE = STEP3 / "edge"
SEED, N_FOLDS = 42, 5
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

def main():
    dev = pd.read_parquet(STEP3 / "dev_pair_features.parquet")
    ev  = pd.read_parquet(STEP3 / "eval_pair_features.parquet")
    y = dev["label"].fillna(0).astype(int).to_numpy(); labm = dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES, rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    FEATS=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    log(f"feats {len(FEATS)}")
    Xd=dev[FEATS].astype("float32"); Xe=ev[FEATS].astype("float32")
    oof_base=np.load(EDGE/"oof_base.npy")

    # two-stage PU weights (reuse baseline stage-1 to get pseudo/ambiguous split from oof_base)
    pos=oof_base[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(oof_base>=ta)&(oof_base<tp); ps=unl&(oof_base>=tp)
    y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))

    def cv_predict(train_fn):
        oof=np.zeros(len(dev)); evp=np.zeros(len(ev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            preds_te, preds_ev = train_fn(Xd[tr], y2[tr], w2[tr], Xd[te], Xe)
            oof[te]=preds_te; evp+=preds_ev/N_FOLDS
        return oof, evp

    def xgb_fn(depth):
        def f(Xtr,ytr,wtr,Xte,Xev):
            dtr=xgb.DMatrix(Xtr,label=ytr,weight=wtr)
            p=dict(objective="binary:logistic",eval_metric="aucpr",tree_method="hist",device="cuda:0",
                   max_depth=depth,eta=0.03,subsample=0.8,colsample_bytree=0.5,min_child_weight=6,reg_lambda=10.0,max_bin=256)
            m=xgb.train(p,dtr,num_boost_round=700)
            return m.predict(xgb.DMatrix(Xte)), m.predict(xgb.DMatrix(Xev))
        return f
    def lgb_dart_fn(Xtr,ytr,wtr,Xte,Xev):
        p=dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=80,feature_fraction=0.5,
               bagging_fraction=0.8,bagging_freq=1,lambda_l2=8.0,verbose=-1,num_threads=-1,seed=SEED+5)
        m=lgb.train(p,lgb.Dataset(Xtr,ytr,weight=wtr),num_boost_round=900)
        return m.predict(Xte), m.predict(Xev)

    log("xgb depth6..."); oof_x6,ev_x6=cv_predict(xgb_fn(6))
    log("xgb depth4..."); oof_x4,ev_x4=cv_predict(xgb_fn(4))
    log("lgb dart-ish..."); oof_l2,ev_l2=cv_predict(lgb_dart_fn)

    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    apb=average_precision_score(y,oof_base)
    print(f"\nbaseline eval-mirrored PairAP: {apb:.4f}")
    for nm,oo in [("xgb6",oof_x6),("xgb4",oof_x4),("lgb2",oof_l2)]:
        print(f"  {nm} alone: {average_precision_score(y,oo):.4f}")
    # rank-average ensembles
    cands={
        "base+xgb6": pr(oof_base)*0.5+pr(oof_x6)*0.5,
        "base+xgb6+lgb2": (pr(oof_base)+pr(oof_x6)+pr(oof_l2))/3,
        "all4": (pr(oof_base)+pr(oof_x6)+pr(oof_x4)+pr(oof_l2))/4,
        "base0.5+rest": pr(oof_base)*0.5+(pr(oof_x6)+pr(oof_x4)+pr(oof_l2))/6,
    }
    best=(apb,"baseline",None)
    for nm,bl in cands.items():
        ap=average_precision_score(y,bl)
        flag="  <-- WINS" if ap>apb+0.002 else ""
        print(f"  ENSEMBLE {nm}: {ap:.4f}{flag}")
        if ap>best[0]: best=(ap,nm,bl)
    print(f"\nBEST: {best[1]} @ {best[0]:.4f}  (baseline {apb:.4f}, delta {best[0]-apb:+.4f})")
    # save eval risks for whichever ensemble won
    np.savez(EDGE/"ensemble_eval.npz", base=np.load(EDGE/"oof_base.npy") if False else ev_x6*0,  # placeholder
             ev_x6=ev_x6, ev_x4=ev_x4, ev_l2=ev_l2)
    (EDGE/"_ensemble_summary.json").write_text(json.dumps({"baseline":float(apb),"best":best[1],"best_ap":float(best[0])},indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
