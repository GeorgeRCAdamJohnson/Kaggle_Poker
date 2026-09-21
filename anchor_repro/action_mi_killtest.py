"""§216 Task 2 — KILL-TEST action-MI for evidence. TWO checks:
(a) single-feature within-pair MAP@5 + ev-vs-nonev separation on the 1817 labels.
(b) THE DECISIVE ONE: orthogonality. Regress mi_* on HAND_FEATS (ridge+gbm R2). Every prior evidence lever
    (equity, outsider-extraction, card-rel) was RECONSTRUCTABLE from HAND_FEATS (R2 0.55-0.78) -> margin-
    redundant -> gated to noise. If action-MI is ALSO reconstructable (R2>0.6), it dies the same way -> STOP.
    If LOW R2 -> genuinely orthogonal -> the first evidence signal with a structural reason to add -> gate it.
Run:  python -m anchor_repro.action_mi_killtest
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from anchor_repro.action_mi import MI_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3")

def main():
    from anchor_repro.seq_evidence import build_dh, map5, TARGET_BEHAVIORS, HAND_FEATS
    dh=build_dh()
    mi=pl.read_parquet(STEP3/"edge"/"actionmi"/"action_mi_development.parquet")
    dh=dh.join(mi,on=["pair_id","hand_id"],how="left").to_pandas()
    for c in MI_FEATS: dh[c]=pd.to_numeric(dh[c],errors="coerce").fillna(0).astype("float32")
    dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    pos=dh[dh["label"]==1].copy(); ev=pos[pos["is_evidence"]]; nv=pos[~pos["is_evidence"]]
    print(f"labeled hands {len(dh):,} evidence {int(dh['is_evidence'].sum())}")
    print("\n=== (a) action-MI: ev vs nonev + single-feature within-pair MAP@5 ===")
    for c in MI_FEATS:
        m=map5(pos.assign(hand_score=pos[c]))
        print(f"  {c:22s} ev {ev[c].mean():.3f} nonev {nv[c].mean():.3f} sep {ev[c].mean()-nv[c].mean():+.3f} | MAP@5 {m:.4f}")
    print("\n=== per family (mi_one_aggr_one_fold / mi_both_passive ev vs nonev) ===")
    for fam in TARGET_BEHAVIORS:
        f=pos[pos["behavior_family"]==fam]; fe=f[f["is_evidence"]]; fn=f[~f["is_evidence"]]
        print(f"  {fam:22s} one_aggr_one_fold {fe['mi_one_aggr_one_fold'].mean():.3f}/{fn['mi_one_aggr_one_fold'].mean():.3f} | both_passive {fe['mi_both_passive'].mean():.3f}/{fn['mi_both_passive'].mean():.3f} | sync {fe['mi_action_sync'].mean():.3f}/{fn['mi_action_sync'].mean():.3f}")
    print("\n=== (b) DECISIVE ORTHOGONALITY: reconstruct each mi_* from HAND_FEATS (30k subsample) ===")
    ps=pos.sample(n=min(30000,len(pos)),random_state=42); X=ps[HAND_FEATS].fillna(0).to_numpy()
    worst=0.0
    for c in MI_FEATS:
        y=ps[c].to_numpy()
        if y.std()<1e-9: print(f"  {c:22s} constant, skip"); continue
        r2r=r2_score(y,cross_val_predict(Ridge(alpha=10),X,y,cv=3))
        r2g=r2_score(y,cross_val_predict(HistGradientBoostingRegressor(max_depth=4,max_iter=120),X,y,cv=3))
        best=max(r2r,r2g); worst=max(worst,best)
        print(f"  {c:22s} ridge R2 {r2r:+.3f} | gbm R2 {r2g:+.3f}  {'REDUNDANT' if best>0.6 else 'orthogonal-ish'}")
    print(f"\n=== VERDICT ===")
    print(f"  most-reconstructable mi feature R2 = {worst:.3f}")
    if worst>0.6: print("  >0.6: HAND_FEATS already encode action-MI -> same redundancy wall as equity/outsider -> likely dies. proceed to gate only cautiously.")
    else: print("  <0.6: action-MI is ORTHOGONAL to pot/outcome feats (UNLIKE all prior evidence levers). Structural reason to ADD. GATE IT.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
