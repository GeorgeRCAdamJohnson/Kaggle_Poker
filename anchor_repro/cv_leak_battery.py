"""Task #4 leak/redundancy battery for the strategy-value collusion features (CV_FEATS). Before the full
gate, verify: (1) count-invariance (corr with shared_hands ~0), (2) dev/eval regime match (means close),
(3) REDUNDANCY vs existing risk — the §195 killer: the clean equity-rate feature was leak-free AND
drift-free but REGRESSED because redundant with base+PT+seq_risk. If cvd_rate correlates ~1 with the base
risk OOF, it will regress too. We want MODERATE correlation (adds orthogonal signal).
Run:  python -m anchor_repro.cv_leak_battery
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from anchor_repro.collusion_value import build_cv_features, CV_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3")
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","is_labeled","shared_hands","table_id"]).to_pandas()
    cd=build_cv_features("development").to_pandas(); ce=build_cv_features("evaluation").to_pandas()
    m=dp.merge(cd,on="pair_id",how="left")
    for c in CV_FEATS: m[c]=m[c].fillna(0.0)
    lab=m["is_labeled"].to_numpy().astype(bool); y=m["label"].to_numpy()

    log("=== (1) count-invariance: corr(feature, shared_hands) — want ~0 ===")
    for c in CV_FEATS:
        log(f"  {c:12s} corr(shared_hands) {np.corrcoef(m[c],m['shared_hands'].fillna(0))[0,1]:+.3f}")
    log("=== (2) dev/eval regime match: feature means (close = transports) ===")
    for c in CV_FEATS: log(f"  {c:12s} dev {m[c].mean():.4f}  eval {ce[c].mean():.4f}")
    log("=== (3) univariate AUC (labeled) ===")
    for c in CV_FEATS:
        a=roc_auc_score(y[lab],m[c].to_numpy()[lab]); log(f"  {c:12s} AUC {max(a,1-a):.4f}")

    # (4) REDUNDANCY vs a quick base risk OOF (the §195 killer). Build a fast base-risk proxy: table-fold
    # OOF LightGBM on the standard pair features, correlate with cvd_rate. Moderate corr = adds signal.
    log("=== (4) REDUNDANCY vs base risk (the §195 killer — want MODERATE corr, not ~1) ===")
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    m["pair_id"]=m["pair_id"].astype(str)
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    feats=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    for c in feats: dev[c]=dev[c].astype("float32").fillna(0)
    yb=dev["label"].fillna(0).astype(int).to_numpy(); labb=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(42)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}; folds=dev["table_id"].map(TF).to_numpy()
    P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    X=dev[feats]; w=np.where(labb,np.where(yb==1,5.0,1.0),0.1); oof=np.zeros(len(dev))
    for f in range(5):
        tr=(folds!=f)&(w>0); te=folds==f
        mm=lgb.train({**P,"seed":42},lgb.Dataset(X[tr],yb[tr],weight=w[tr]),num_boost_round=400); oof[te]=mm.predict(X[te])
    base_oof=pd.Series(oof,index=dev["pair_id"].values)
    mm=m.set_index("pair_id")
    for c in CV_FEATS:
        common=mm.index.intersection(base_oof.index)
        rho=spearmanr(mm.loc[common,c], base_oof.loc[common]).correlation
        log(f"  spearman({c:12s}, base_risk_OOF) = {rho:+.3f}  ({'REDUNDANT' if abs(rho)>0.85 else 'orthogonal-ish'})")
    log("  (base risk OOF PairAP: {:.4f})".format(roc_auc_score(yb,oof)))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
