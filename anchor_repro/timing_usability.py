"""§205b — Is the PROBE-1 timing signal (planted hands spaced closer, §205) USABLE at eval, or a label-
dependent artifact like the rings (§199)? The gap-between-PLANTED-hands needs the labels to compute, which
are hidden at eval. Test whether an EVAL-COMPUTABLE timing feature (no label needed) separates the pair label.

Eval-computable timing candidates (per pair, from shared-hand timestamps ONLY — no planted flags):
  t_burstiness  : cv of gaps between the pair's CONSECUTIVE shared hands (do they play in tight bursts?)
  t_med_gap     : median gap between consecutive shared hands
  t_min_gap_frac: fraction of consecutive-shared-hand gaps that are very small (<=2 hand-index) = clustering
These use ALL shared hands, computable identically dev & eval. If they separate the label AND transfer
(dev/eval means match), timing is a real lever. If flat, the §205 planted-spacing signal is label-only = dead.
Run:  python -m anchor_repro.timing_usability
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
D=Path("data/poker"); CACHE=Path("outputs/poker_collusion/generator_cache"); STEP3=Path("outputs/poker_collusion/hosen42_step3")

def pair_timing(shared):
    sh=shared.sort(["pair_id","started_at"])
    # gap in seconds between consecutive shared hands per pair
    sh=sh.with_columns(pl.col("started_at").diff().over("pair_id").alias("gap"))
    g=sh.group_by("pair_id").agg(
        pl.col("gap").drop_nulls().alias("gaps"))
    def feats(gaps):
        a=np.array(gaps,float)
        if len(a)<3: return (np.nan,np.nan,np.nan)
        a=a[a>0]
        if len(a)<3: return (np.nan,np.nan,np.nan)
        return (a.std()/ (a.mean()+1e-9), np.median(a), np.mean(a<=np.percentile(a,10)))
    rows=[]
    for pid,gs in zip(g["pair_id"].to_list(),g["gaps"].to_list()):
        b,med,minf=feats(gs); rows.append((pid,b,med,minf))
    return pl.DataFrame(rows,schema=["pair_id","t_burstiness","t_med_gap","t_min_gap_frac"],orient="row")

def main():
    labels=pl.read_csv(D/"development_labels.csv").select(["pair_id","label"])
    # shared hands for ALL labeled pairs (pos+neg) so we can score separation honestly
    shared=pl.scan_parquet(CACHE/"dev_layer2.parquet").select(["pair_id","hand_id"]).collect().join(labels.select("pair_id"),on="pair_id",how="inner")
    hands=pl.scan_parquet(D/"hands.parquet").select(["hand_id","started_at"]).collect()
    # started_at may be datetime; cast to epoch seconds
    shared=shared.join(hands,on="hand_id",how="left")
    try: shared=shared.with_columns(pl.col("started_at").cast(pl.Int64).alias("started_at"))
    except Exception: shared=shared.with_columns(pl.col("started_at").dt.epoch("s").alias("started_at"))
    tf=pair_timing(shared)
    m=labels.join(tf,on="pair_id",how="left").to_pandas()
    y=m["label"].to_numpy()
    print(f"labeled pairs with timing feats: {m['t_burstiness'].notna().sum()}")
    print("=== eval-computable timing features: label-separation AUC (dev labeled pos vs neg) ===")
    for c in ["t_burstiness","t_med_gap","t_min_gap_frac"]:
        v=m[c].fillna(m[c].median()).to_numpy()
        a=roc_auc_score(y,v); print(f"  {c:16s} AUC {max(a,1-a):.4f} ({'hi=pos' if a>0.5 else 'lo=pos'})")
    print("\n  (AUC ~0.5 => the §205 planted-spacing signal is LABEL-ONLY (uncomputable at eval) = dead, like rings §199.")
    print("   AUC clearly >0.55 => a real eval-usable timing lever -> graduate to gated build.)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
