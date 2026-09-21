"""Is the 1.0 dev/eval drift a BUG (a leaky phase-marker feature) or diffuse real phase structure?

Find which base features drive the perfect dev-vs-eval separability. If ONE/few features do it, they're
phase markers (potential bug / should be dropped or phase-normalized). If diffuse, it's real phase drift
everyone faces. Also: does the 0.804 model actually USE the high-drift features? (importance x drift).
Run:  python -m anchor_repro.drift_culprit
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3")

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    feats=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and c in evf.columns]
    # per-feature univariate dev-vs-eval AUC
    n=min(len(dev),len(evf),40000); rng=np.random.RandomState(0)
    d=dev.iloc[rng.choice(len(dev),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    e=evf.iloc[rng.choice(len(evf),n,replace=False)][feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    y=np.concatenate([np.ones(n),np.zeros(n)])
    rows=[]
    for c in feats:
        v=np.concatenate([d[c].to_numpy(),e[c].to_numpy()])
        try: a=roc_auc_score(y,v); rows.append((c,max(a,1-a)))
        except Exception: pass
    rows.sort(key=lambda x:-x[1])
    print("=== top per-feature dev-vs-eval univariate drift AUC ===")
    for c,a in rows[:15]: print(f"  {a:.4f}  {c}")
    n_high=sum(1 for _,a in rows if a>0.9); n_mid=sum(1 for _,a in rows if 0.7<a<=0.9)
    print(f"\n  features with drift>0.90: {n_high}/{len(rows)}  | 0.70-0.90: {n_mid}")
    print("  -> if a FEW feats >0.9 dominate = phase markers (bug-like, droppable); if MANY = diffuse real drift")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
