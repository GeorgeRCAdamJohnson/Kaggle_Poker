"""Is there ANY evidence headroom? Oracle MAP@5 + what truly separates planted hands within a pair.

Two evidence adds failed (postflop +0.0007, policy -0.0098). Before more attempts, measure:
  1. ORACLE MAP@5 (rank planted hands perfectly within each pair) -> the ceiling. If baseline 0.5276 is
     already near a low oracle, evidence has little headroom and we pivot.
  2. Per-feature within-pair AUC (planted vs the SAME pair's non-planted hands) over ALL candidate
     features incl our policy surprise -> what ACTUALLY marks a planted hand within a positive pair.
Run:  python -m anchor_repro.evidence_headroom
"""
from __future__ import annotations
import numpy as np, pandas as pd, polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
DEV_HF=[STEP3/f"dev_hand_features_{k}.parquet" for k in range(4)]; PER=EDGE/"per_hand_surprise.parquet"

def map5(df,score_col):
    vals=[]
    d=df.sort_values(["pair_id",score_col,"pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    for _,g in d.groupby("pair_id",sort=False):
        rel=g["is_evidence"].to_numpy(); nr=int(rel.sum())
        if nr==0: continue
        top=rel[:5]; hits=np.cumsum(top)
        vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(nr,5)))
    return float(np.mean(vals)) if vals else 0.0

def main():
    dev_ev=pl.read_csv(DATA/"development_evidence.csv")
    labelled=pl.read_parquet(STEP3/"dev_pairs.parquet").filter(pl.col("is_labeled"))
    lab_ids=set(labelled["pair_id"].to_list())
    hf=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(lab_ids))).collect() for f in DEV_HF])
    hf=hf.join(labelled.select(["pair_id","label","behavior_family"]),on="pair_id")
    hf=hf.join(dev_ev.select(["pair_id","hand_id",pl.lit(True).alias("is_evidence")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_evidence").fill_null(False))
    S=["loose","tight","postflop_loose","total_loose"]
    sur=pl.scan_parquet(PER).filter(pl.col("phase")=="development").select(["hand_id","player_id"]+S)
    hf=(hf.join(sur.rename({c:f"p1_{c}" for c in S}).rename({"player_id":"player_1"}).collect(),on=["hand_id","player_1"],how="left")
          .join(sur.rename({c:f"p2_{c}" for c in S}).rename({"player_id":"player_2"}).collect(),on=["hand_id","player_2"],how="left"))
    for c in S:
        hf=hf.with_columns(pl.min_horizontal(pl.col(f"p1_{c}").fill_null(0),pl.col(f"p2_{c}").fill_null(0)).alias(f"jt_{c}_min"))
    dh=hf.to_pandas(); pos=(dh["label"]==1).to_numpy(); dpos=dh[pos].copy()

    # 1. ORACLE (perfect within-pair ranking) vs baseline-ish (transfer_any) vs random
    dpos["oracle"]=dpos["is_evidence"].astype(float)
    print(f"ORACLE MAP@5 (perfect within-pair) = {map5(dpos,'oracle'):.4f}  (this is the ceiling)")
    for c in ["transfer_any","both_vpip","jt_total_loose_min","jt_postflop_loose_min","pot_bb"]:
        if c in dpos.columns: print(f"  single-feature MAP@5 [{c}] = {map5(dpos,c):.4f}")

    # 2. within-pair planted-vs-nonplanted separation (AUC), pooled across pairs
    y=dpos["is_evidence"].astype(int).to_numpy()
    cands=[c for c in dpos.columns if pd.api.types.is_numeric_dtype(dpos[c]) and c not in
           ("pair_id","hand_id","table_id","hand_idx","player_1","player_2","label","is_evidence","oracle")]
    aucs={}
    for c in cands:
        v=pd.to_numeric(dpos[c],errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).any() or np.nanstd(v)==0: continue
        try:
            a=roc_auc_score(y,np.nan_to_num(v)); aucs[c]=max(a,1-a)
        except Exception: pass
    top=sorted(aucs.items(),key=lambda x:-x[1])[:20]
    print("\nTop within-pair planted-vs-nonplanted separators (AUC, pooled):")
    for c,a in top: print(f"  {a:.4f}  {c}")
    print("\npolicy-surprise feats rank:", {c:round(aucs.get(c,0),4) for c in ['jt_total_loose_min','jt_postflop_loose_min','jt_loose_min','jt_tight_min']})
    return 0

if __name__=="__main__":
    raise SystemExit(main())
