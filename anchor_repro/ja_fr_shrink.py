"""JA field-relative: does heavier exposure-shrink collapse the joint drift (0.699), as it did for equity
(0.685->0.615)? Individual JA residuals are at chance drift (0.502) so the joint drift is exposure-variance.
Sweep k and report joint drift + retained incremental over made-value at each k.
Run:  python -m anchor_repro.ja_fr_shrink
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=8)
MADE=["transfer_dominant","transfer_rate","hs_mean","hs_max","hs_top3"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def shrink(df,k):
    n=df["jar_n_opp"].clip(lower=1); df=df.copy()
    df["s_net"]=df["jar_net_sum"]/(n+k); df["s_cofold"]=df["jar_cofold_sum"]/(n+k)
    df["s_won"]=df["jar_won_sum"]/(n+k); df["s_netz"]=df["jar_net_sum"]/np.sqrt(n+k)
    return df
SFE=["s_net","s_cofold","s_won","s_netz"]

def adv(d,e,feats,rng):
    n=min(len(d),len(e),40000); ds=d.iloc[rng.choice(len(d),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    es=e.iloc[rng.choice(len(e),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xc=pd.concat([ds,es],ignore_index=True); yc=np.concatenate([np.zeros(n),np.ones(n)]); fo=rng.randint(0,5,size=len(yc)); o=np.zeros(len(yc))
    for f in range(5):
        tr=fo!=f; te=fo==f
        m=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=8),lgb.Dataset(Xc.iloc[tr],yc[tr]),num_boost_round=200); o[te]=m.predict(Xc.iloc[te])
    return roc_auc_score(yc,o)

def main():
    d0=pl.read_parquet(EQ/"ja_field_relative_development.parquet").to_pandas(); d0["pair_id"]=d0["pair_id"].astype(str)
    e0=pl.read_parquet(EQ/"ja_field_relative_evaluation.parquet").to_pandas(); e0["pair_id"]=e0["pair_id"].astype(str)
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool); ly=y[labm]
    made=[c for c in MADE if c in dev.columns]; rng=np.random.RandomState(42)
    def incr(dd):
        m=dev.merge(dd[["pair_id"]+SFE],on="pair_id",how="left")
        for c in SFE: m[c]=m[c].astype("float32").fillna(0)
        def oof(feats):
            X=m.loc[labm,feats].astype("float32").fillna(0).to_numpy(); o=np.zeros(labm.sum()); ff=rng.randint(0,5,size=labm.sum())
            for f in range(5):
                tr=ff!=f; te=ff==f
                mm=lgb.train({**P,"seed":42},lgb.Dataset(X[tr],ly[tr]),num_boost_round=300); o[te]=mm.predict(X[te])
            return roc_auc_score(ly,o)
        return oof(made), oof(made+SFE)
    print("\n=== JA exposure-shrink sweep: joint drift + incremental over made-value ===")
    for k in [100,300,1000,3000]:
        d=shrink(d0,k); e=shrink(e0,k); dr=adv(d,e,SFE,rng); mv,mve=incr(d)
        log(f"  k={k:5d}: joint drift {dr:.4f} | made {mv:.4f}->+JA {mve:.4f} (incr {mve-mv:+.4f})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
