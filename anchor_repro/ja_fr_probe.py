"""Cheap GATE-1 (drift) + GATE-2 (incremental) probe for field-relative joint-advantage residuals (§223 Opt 2).
Same discipline as the equity probe: only carry to the expensive 5-seed stack gate if drift<0.65 AND
incremental over made-value >+0.008. Also reports orthogonality vs the field-relative EQUITY residuals
(want LOW correlation, so stacking them is additive not redundant).
Run:  python -m anchor_repro.ja_fr_probe
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.ja_field_relative import FEATS_JA, add_shrunk_ja
from anchor_repro.equity_field_relative import FEATS_FR, add_shrunk as add_shrunk_eq
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=8)
MADE=["transfer_dominant","transfer_rate","hs_mean","hs_max","hs_top3"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    d=add_shrunk_ja(pl.read_parquet(EQ/"ja_field_relative_development.parquet").to_pandas()); d["pair_id"]=d["pair_id"].astype(str)
    e=add_shrunk_ja(pl.read_parquet(EQ/"ja_field_relative_evaluation.parquet").to_pandas()); e["pair_id"]=e["pair_id"].astype(str)
    rng=np.random.RandomState(42)
    print("\n=== GATE 1: drift (adversarial dev-vs-eval AUC on JA residuals; want <0.65) ===")
    n=min(len(d),len(e),40000); ds=d.iloc[rng.choice(len(d),n,replace=False)][FEATS_JA].replace([np.inf,-np.inf],np.nan).fillna(0)
    es=e.iloc[rng.choice(len(e),n,replace=False)][FEATS_JA].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xc=pd.concat([ds,es],ignore_index=True); yc=np.concatenate([np.zeros(n),np.ones(n)]); fo=rng.randint(0,5,size=len(yc)); od=np.zeros(len(yc))
    for f in range(5):
        tr=fo!=f; te=fo==f
        m=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=8),lgb.Dataset(Xc.iloc[tr],yc[tr]),num_boost_round=200); od[te]=m.predict(Xc.iloc[te])
    drift=roc_auc_score(yc,od); log(f"  JA-residual drift AUC = {drift:.4f}")
    for c in FEATS_JA:
        a=roc_auc_score(yc,pd.concat([ds[[c]],es[[c]]],ignore_index=True)[c].to_numpy()); print(f"      {c:16s} solo-drift {max(a,1-a):.3f}")

    # GATE 2: standalone label-AUC + incremental over made-value
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev=dev.merge(d,on="pair_id",how="left")
    for c in FEATS_JA: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool); ly=y[labm]
    print("\n=== GATE 2: standalone label-AUC + incremental over made-value ===")
    best=0.0
    for c in FEATS_JA:
        a=roc_auc_score(ly,dev.loc[labm,c].to_numpy()); best=max(best,max(a,1-a)); print(f"      {c:16s} label-AUC {max(a,1-a):.3f}")
    made=[c for c in MADE if c in dev.columns]
    def oof(feats):
        X=dev.loc[labm,feats].astype("float32").fillna(0).to_numpy(); o=np.zeros(labm.sum()); ff=rng.randint(0,5,size=labm.sum())
        for f in range(5):
            tr=ff!=f; te=ff==f
            m=lgb.train({**P,"seed":42},lgb.Dataset(X[tr],ly[tr]),num_boost_round=300); o[te]=m.predict(X[te])
        return roc_auc_score(ly,o)
    mv=oof(made); mve=oof(made+FEATS_JA); log(f"  made-value {mv:.4f} -> +JA {mve:.4f} (delta {mve-mv:+.4f})")

    # orthogonality vs field-relative EQUITY residuals (want LOW corr -> additive stack)
    eq=add_shrunk_eq(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas()); eq["pair_id"]=eq["pair_id"].astype(str)
    mrg=d.merge(eq[["pair_id"]+FEATS_FR],on="pair_id",how="inner")
    print("\n=== orthogonality: max|corr| of each JA feat vs equity FR feats (want LOW) ===")
    for c in FEATS_JA:
        cors=[abs(np.corrcoef(mrg[c].fillna(0),mrg[fc].fillna(0))[0,1]) for fc in FEATS_FR]
        print(f"      {c:16s} max|corr| {max(cors):.3f}")
    print(f"\n  VERDICT: carry to stack gate if drift {drift:.3f}<0.65 AND incremental {mve-mv:+.4f}>+0.008 AND orthogonal to equity.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
