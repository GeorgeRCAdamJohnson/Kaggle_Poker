"""Diagnose WHY the field-relative residuals drift jointly (0.685) while each is at chance (~0.51).
Hypothesis: the null model was fit SEPARATELY per phase, so the residual DEFINITION differs subtly between
dev and eval -> a multivariate interaction separates them even though margins match. Test:
  (A) which single feature, added greedily, drives the joint drift up
  (B) does a SHARED null (fit on dev+eval pooled) collapse the joint drift?
Run:  python -m anchor_repro.equity_fr_drift_diag
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.equity_field_relative import FEATS_FR
EQ = Path("outputs/poker_collusion/hosen42_step3")/"edge"/"equity"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

def adv(dfd, dfe, feats, seed=42):
    rng = np.random.RandomState(seed); n = min(len(dfd), len(dfe), 40000)
    ds = dfd.iloc[rng.choice(len(dfd),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    es = dfe.iloc[rng.choice(len(dfe),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xc = pd.concat([ds,es],ignore_index=True); yc = np.concatenate([np.zeros(n),np.ones(n)])
    fo = rng.randint(0,5,size=len(yc)); o = np.zeros(len(yc))
    for f in range(5):
        tr=fo!=f; te=fo==f
        m = lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),
                      lgb.Dataset(Xc.iloc[tr],yc[tr]),num_boost_round=200); o[te]=m.predict(Xc.iloc[te])
    return roc_auc_score(yc,o)

def main():
    d = pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas()
    e = pl.read_parquet(EQ/"field_relative_evaluation.parquet").to_pandas()
    print("\n=== (A) greedy: which feature drives the joint drift ===")
    chosen=[]; remaining=list(FEATS_FR)
    while remaining:
        best=None
        for c in remaining:
            a=adv(d,e,chosen+[c]); 
            if best is None or a>best[1]: best=(c,a)
        chosen.append(best[0]); remaining.remove(best[0])
        log(f"  + {best[0]:22s} -> joint drift {best[1]:.4f}")
    print("\n=== (B) dev-vs-eval MEAN/STD of each residual feat (are the DISTRIBUTIONS themselves shifted?) ===")
    for c in FEATS_FR:
        print(f"  {c:22s} dev mean {d[c].mean():+.5f} std {d[c].std():.5f} | eval mean {e[c].mean():+.5f} std {e[c].std():.5f}")
    print("\n=== (C) the aux exp_rate (field null level) dev vs eval — is the NULL itself phase-shifted? ===")
    for c in ["fr_surr_obs_rate","fr_surr_exp_rate","fr_n_opp"]:
        if c in d.columns:
            print(f"  {c:22s} dev mean {d[c].mean():.5f} | eval mean {e[c].mean():.5f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
