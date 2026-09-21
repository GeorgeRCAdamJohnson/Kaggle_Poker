"""Cache per-(hand,player) proxy EV-loss (decision-side), both phases, for the field-normalized channel.
Reuses the builder from evloss_decorr_check (decorrelation-gate passed: max|corr|=0.27 vs equity feats).
"""
from __future__ import annotations
import polars as pl
from pathlib import Path
from anchor_repro.evloss_decorr_check import build_evloss_perhand
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"
def main():
    el=build_evloss_perhand().with_columns((pl.col("evloss_call_weak")+pl.col("evloss_fold_strong")).alias("evloss"))
    # attach phase for dev/eval split
    ph=pl.read_parquet(Path("data/poker")/"hands.parquet",columns=["hand_id","phase"])
    el=el.join(ph.lazy().collect() if hasattr(ph,'lazy') else ph, on="hand_id", how="left")
    el.write_parquet(EDGE/"evloss_perhand.parquet")
    print("wrote evloss_perhand", el.height, "phases", el["phase"].value_counts().to_dict() if "phase" in el.columns else "?")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
