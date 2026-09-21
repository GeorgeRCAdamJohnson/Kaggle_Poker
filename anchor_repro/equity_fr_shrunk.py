"""Exposure-shrunk field-relative residual — the fix the drift diagnosis (§220) points to.

Diagnosis: residual DIRECTION is phase-invariant (dev/eval means match to 4 decimals) but residual
MAGNITUDE/VARIANCE scales with exposure (n_opp dev 15.9 vs eval 11.5), so sum-form residuals inherit the
exposure drift and even the mean-residual's variance shifts. Standard fix: empirical-Bayes shrink toward 0
by exposure -> z = resid_sum / sqrt(n_opp + k) and eb = resid_sum / (n_opp + k). These are variance-
stabilized, so a pair with few hands isn't a high-variance outlier. Test drift of the shrunk forms alone.
Run:  python -m anchor_repro.equity_fr_shrunk   [--k 20]
"""
from __future__ import annotations
import argparse, time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
EQ = Path("outputs/poker_collusion/hosen42_step3")/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def add_shrunk(df, k):
    n = df["fr_n_opp"].clip(lower=1)
    df = df.copy()
    df["frs_eb"]  = df["fr_surr_resid_sum"] / (n + k)          # empirical-Bayes shrunk mean residual
    df["frs_z"]   = df["fr_surr_resid_sum"] / np.sqrt(n + k)   # variance-stabilized residual score
    df["frs_ebmag"] = df["fr_surr_excess_mag"] / (n + k)       # magnitude EB
    df["frs_ebpos"] = df["fr_surr_resid_pos"] / (n + k)        # positive-only EB
    return df

SHRUNK = ["frs_eb","frs_z","frs_ebmag","frs_ebpos"]

def adv(dfd, dfe, feats, seed=42):
    rng=np.random.RandomState(seed); n=min(len(dfd),len(dfe),40000)
    ds=dfd.iloc[rng.choice(len(dfd),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    es=dfe.iloc[rng.choice(len(dfe),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xc=pd.concat([ds,es],ignore_index=True); yc=np.concatenate([np.zeros(n),np.ones(n)])
    fo=rng.randint(0,5,size=len(yc)); o=np.zeros(len(yc))
    for f in range(5):
        tr=fo!=f; te=fo==f
        m=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),
                    lgb.Dataset(Xc.iloc[tr],yc[tr]),num_boost_round=200); o[te]=m.predict(Xc.iloc[te])
    return roc_auc_score(yc,o)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--k",type=float,default=20.0); a=ap.parse_args()
    d=add_shrunk(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas(), a.k)
    e=add_shrunk(pl.read_parquet(EQ/"field_relative_evaluation.parquet").to_pandas(), a.k)
    print(f"\n=== exposure-shrunk residual drift (k={a.k}) — want <0.65, target <0.55 ===")
    for c in SHRUNK:
        print(f"  {c:10s} solo-drift {adv(d,e,[c]):.4f}  | dev mean {d[c].mean():+.5f} std {d[c].std():.5f} | eval mean {e[c].mean():+.5f} std {e[c].std():.5f}")
    log(f"  JOINT (all 4 shrunk) drift = {adv(d,e,SHRUNK):.4f}")
    log(f"  JOINT (frs_eb + frs_z) drift = {adv(d,e,['frs_eb','frs_z']):.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
