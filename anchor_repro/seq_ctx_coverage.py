"""Diagnose the ctx-emb fill-0 gap: are the EVIDENCE hands (the ones MAP@5 scores) covered, or is the
missing half concentrated in inactive/overflow hands that don't matter?  If evidence hands are already
covered, the +0.062 is not window-limited and widening won't help much. If evidence hands are being
dropped, widening the window is the lever."""
from __future__ import annotations
import polars as pl
from pathlib import Path
from anchor_repro.seq_evidence import build_dh
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
dh=build_dh()
emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet").select(["pair_id","hand_idx"]).with_columns(pl.lit(True).alias("has_ctx"))
d=dh.join(emb,on=["pair_id","hand_idx"],how="left").with_columns(pl.col("has_ctx").fill_null(False))
pos=d.filter(pl.col("label")==1)
print(f"positive-pair hands: {pos.height:,}  covered {pos['has_ctx'].sum():,} ({100*pos['has_ctx'].mean():.1f}%)")
ev=pos.filter(pl.col("is_evidence"))
print(f"EVIDENCE hands (what MAP@5 scores): {ev.height:,}  covered {ev['has_ctx'].sum():,} ({100*ev['has_ctx'].mean():.1f}%)")
# per positive pair: how many hands, how many covered
pp=(pos.group_by("pair_id").agg(pl.len().alias("n_hands"),pl.col("has_ctx").sum().alias("n_cov"),
                                pl.col("is_evidence").sum().alias("n_ev"),
                                (pl.col("is_evidence")&pl.col("has_ctx")).sum().alias("n_ev_cov")))
print(f"positive pairs: {pp.height}  median hands/pair {pp['n_hands'].median():.0f}  "
      f"pairs with ALL evidence covered {(pp['n_ev_cov']==pp['n_ev']).sum()}/{pp.height}")
print(f"evidence hands dropped by window: {ev.height-ev['has_ctx'].sum():,}")
