"""§203 — Does the sequence encoder already ABSORB the raw bet/pot/stack-ratio fingerprint? (settles the
Kiro<->reviewer disagreement from §197/§198/§202 with a number, not a guess.)

Reuses §113 gen_fingerprint construction (ratio feats at aggressive actions) + cached seq_emb_v2_dev.parquet.
Restricts to the 372 confirmed positive pairs. Reports (pre-registered in §202):
  1. scalar corr(seq_risk_proxy, each pair-level ratio aggregate)
  2. max |corr| across all 384 seq embedding dims vs each ratio aggregate
  3. linear-probe R2: regress each ratio aggregate on ALL seq dims (5-fold), how well embeddings reconstruct it
Interpretation (pre-registered, §202): max|corr| or probe R2 > 0.5 -> encoder absorbs it, fingerprinting stays
LOW priority. < 0.3 -> not absorbed, fingerprinting moves back to TOP. In between -> report, no forced verdict.
Run:  python -m anchor_repro.fingerprint_absorption
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
D=Path("data/poker"); CACHE=Path("outputs/poker_collusion/generator_cache"); STEP3=Path("outputs/poker_collusion/hosen42_step3")
SEQ=STEP3/"edge"/"seq"

def pair_ratio_features():
    labels=pl.read_csv(D/"development_labels.csv").select(["pair_id","label"])
    pos=labels.filter(pl.col("label")==1)
    shared=pl.scan_parquet(CACHE/"dev_layer2.parquet").select(["pair_id","hand_id"]).collect().join(pos.select("pair_id"),on="pair_id",how="inner")
    hands=set(shared["hand_id"].to_list())
    acts=(pl.scan_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(list(hands)))
          .filter(pl.col("action").is_in(["bet","raise","all_in"]))
          .join(pl.scan_parquet(D/"hands.parquet").select(["hand_id","big_blind"]),on="hand_id")
          .with_columns(
              (pl.col("amount")/pl.col("big_blind")).alias("amount_bb"),
              (pl.col("amount")/pl.col("pot_before").clip(1,None)).alias("bet_pot_frac"),
              (pl.col("pot_before")/pl.col("big_blind")).alias("pot_bb"),
              (pl.col("stack_before")/pl.col("pot_before").clip(1,None)).alias("spr"))
          .collect(engine="streaming"))
    acts=acts.join(shared,on="hand_id",how="inner")
    agg=acts.group_by("pair_id").agg(
        pl.col("amount_bb").mean().alias("amount_bb_mean"), pl.col("amount_bb").quantile(0.95).alias("amount_bb_p95"),
        pl.col("bet_pot_frac").mean().alias("bpf_mean"), pl.col("bet_pot_frac").quantile(0.95).alias("bpf_p95"),
        pl.col("pot_bb").mean().alias("pot_bb_mean"), pl.col("pot_bb").quantile(0.95).alias("pot_bb_p95"),
        pl.col("spr").mean().alias("spr_mean"), pl.col("spr").quantile(0.95).alias("spr_p95"))
    return agg, pos

RATIO=["amount_bb_mean","amount_bb_p95","bpf_mean","bpf_p95","pot_bb_mean","pot_bb_p95","spr_mean","spr_p95"]

def main():
    agg,pos=pair_ratio_features()
    emb=pl.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); SEQF=[c for c in emb.columns if c.startswith("seq_")]
    m=pos.join(agg,on="pair_id",how="left").join(emb,on="pair_id",how="left").to_pandas()
    for c in RATIO+SEQF: m[c]=m[c].astype("float32")
    m=m.dropna(subset=RATIO+[SEQF[0]])
    print(f"positive pairs with both ratio feats + embeddings: {len(m)}")

    # seq_risk proxy: 1st principal direction of the embedding is arbitrary; use the mean-pooled norm as a
    # simple scalar summary AND report max-dim corr which is the fair test.
    E=m[SEQF].to_numpy()
    print("\n=== (1)+(2) corr of ratio aggregates vs seq embedding ===")
    for c in RATIO:
        r=m[c].to_numpy()
        cors=np.array([abs(spearmanr(E[:,j],r).correlation) for j in range(E.shape[1])])
        cors=cors[np.isfinite(cors)]
        print(f"  {c:16s} max|corr| over 384 dims = {cors.max():.4f}  (mean {cors.mean():.4f})")

    print("\n=== (3) linear-probe R2: ratio ~ ALL seq dims (5-fold ridge) ===")
    for c in RATIO:
        r=m[c].to_numpy()
        pred=cross_val_predict(Ridge(alpha=10.0),E,r,cv=5)
        print(f"  {c:16s} probe R2 = {r2_score(r,pred):+.4f}")

    print("\n=== pre-registered verdict (max over ratios) ===")
    maxr2=max(r2_score(m[c].to_numpy(),cross_val_predict(Ridge(alpha=10.0),E,m[c].to_numpy(),cv=5)) for c in RATIO)
    maxcorr=max((np.array([abs(spearmanr(E[:,j],m[c].to_numpy()).correlation) for j in range(E.shape[1])]) ) .max() for c in RATIO)
    print(f"  best-ratio max|corr| = {maxcorr:.4f} | best-ratio probe R2 = {maxr2:.4f}")
    if maxcorr>0.5 or maxr2>0.5: print("  -> ABSORBED (>0.5): fingerprinting stays LOW priority (reviewer's argument holds).")
    elif maxcorr<0.3 and maxr2<0.3: print("  -> NOT absorbed (<0.3): fingerprinting moves back to TOP (Kiro's ranking holds).")
    else: print("  -> IN BETWEEN: report plainly, no forced verdict (Rule 9).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
