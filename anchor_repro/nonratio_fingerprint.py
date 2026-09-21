"""§205 — Two NON-RATIO fingerprint existence-probes (§204 spec). The last structurally-distinct candidates
the per-pair sequence encoder CANNOT see (§203 showed it absorbs all per-hand ratios). Cheap existence checks
with a PRE-COMMITTED stop condition (§204): null on BOTH -> bank 0.83460, fingerprinting closed in every form.

PROBE 1 — CROSS-HAND TIMING REGULARITY: for each positive pair, gaps between consecutive PLANTED hands
  (by hand order within the pair) vs a null of same-count random draws from that pair's shared hands.
  A per-hand encoder cannot represent "gap since last flagged hand". KS/quantile vs null.
PROBE 2 — RAW-AMOUNT LATTICE: §113 tested ratio-lock (null). This tests the RAW `amount` (chip count, not
  amount/pot) for clustering/lattice in planted vs non-planted decisive aggressive actions (same discreteness
  method as §113, pointed at raw amount).
Run:  python -m anchor_repro.nonratio_fingerprint
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from scipy.stats import ks_2samp
D=Path("data/poker"); CACHE=Path("outputs/poker_collusion/generator_cache")

def main():
    labels=pl.read_csv(D/"development_labels.csv").select(["pair_id","label"])
    pos=labels.filter(pl.col("label")==1)
    ev=pl.read_csv(D/"development_evidence.csv").select(["pair_id","hand_id"])
    shared=pl.scan_parquet(CACHE/"dev_layer2.parquet").select(["pair_id","hand_id"]).collect().join(pos.select("pair_id"),on="pair_id",how="inner")
    hands=pl.scan_parquet(D/"hands.parquet").select(["hand_id","started_at"]).collect()
    shared=shared.join(hands,on="hand_id",how="left")

    # ---- PROBE 1: cross-hand timing regularity ----
    print("=== PROBE 1: cross-hand TIMING regularity (planted-hand spacing vs random-draw null) ===")
    # order each pair's shared hands by started_at -> integer index; planted positions within that order
    sh=shared.sort(["pair_id","started_at"]).with_columns(pl.col("hand_id").cum_count().over("pair_id").alias("ord"))
    plset=set((ev["pair_id"]+"|"+ev["hand_id"]).to_list())
    sh=sh.with_columns((pl.col("pair_id")+"|"+pl.col("hand_id")).is_in(list(plset)).alias("planted"))
    obs_gaps=[]; null_gaps=[]; rng=np.random.RandomState(42)
    for pid,g in sh.group_by("pair_id"):
        ords=g.sort("ord")["ord"].to_numpy(); pl_ord=g.filter(pl.col("planted"))["ord"].to_numpy()
        n=len(ords); k=len(pl_ord)
        if k<2: continue
        pl_ord=np.sort(pl_ord); obs_gaps.extend(np.diff(pl_ord).tolist())
        # null: draw k random positions from the n, same-count
        idx=np.sort(rng.choice(n,size=k,replace=False)); null_gaps.extend(np.diff(idx).tolist())
    obs_gaps=np.array(obs_gaps,float); null_gaps=np.array(null_gaps,float)
    ks=ks_2samp(obs_gaps,null_gaps)
    print(f"  planted-hand consecutive gaps: obs n={len(obs_gaps)} mean {obs_gaps.mean():.2f} cv {obs_gaps.std()/obs_gaps.mean():.3f}")
    print(f"  random-draw null gaps:         n={len(null_gaps)} mean {null_gaps.mean():.2f} cv {null_gaps.std()/null_gaps.mean():.3f}")
    print(f"  KS(obs, null): stat {ks.statistic:.4f} p {ks.pvalue:.4g} -> {'DIFFERS from random (regularity!)' if ks.pvalue<0.01 else 'INDISTINGUISHABLE from random (null)'}")
    p1_signal = ks.pvalue<0.01 and ks.statistic>0.1

    # ---- PROBE 2: raw-amount lattice ----
    print("\n=== PROBE 2: RAW-AMOUNT lattice (planted vs non-planted decisive aggressive actions) ===")
    hset=set(shared["hand_id"].to_list()); planted_hands=set(ev["hand_id"].to_list())
    acts=(pl.scan_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(list(hset)))
          .filter(pl.col("action").is_in(["bet","raise","all_in"]))
          .select(["hand_id","amount"]).collect())
    acts=acts.with_columns(pl.col("hand_id").is_in(list(planted_hands)).alias("planted"))
    ap=acts.filter(pl.col("planted"))["amount"].to_numpy().astype(float)
    an=acts.filter(~pl.col("planted"))["amount"].to_numpy().astype(float)
    print(f"  planted aggressive actions n={len(ap):,} | nonplanted n={len(an):,}")
    # lattice/discreteness: fraction on round chip values, and concentration (top-value mass)
    def conc(a):
        vals,cnts=np.unique(a,return_counts=True); top=np.sort(cnts)[::-1][:10].sum()/len(a); return top,len(vals)
    tp,vp=conc(ap); tn,vn=conc(an)
    print(f"  top-10-value mass: planted {tp:.3f} ({vp} distinct) | nonplanted {tn:.3f} ({vn} distinct)")
    # round-number lattice (multiples of 5,10,25,50,100)
    for m in [5,10,25,50,100]:
        fp=np.mean(np.abs(ap-np.round(ap/m)*m)<1e-6); fn=np.mean(np.abs(an-np.round(an/m)*m)<1e-6)
        print(f"  multiple-of-{m:<3d}: planted {fp:.3f} nonpl {fn:.3f}  (excess {fp-fn:+.3f})")
    ks2=ks_2samp(ap,an)
    print(f"  KS(planted amt, nonplanted amt): stat {ks2.statistic:.4f} p {ks2.pvalue:.4g}")
    # signal only if planted has EXCESS lattice concentration beyond nonplanted
    p2_signal = (tp-tn>0.05) or any(abs(np.mean(np.abs(ap-np.round(ap/m)*m)<1e-6)-np.mean(np.abs(an-np.round(an/m)*m)<1e-6))>0.05 for m in [5,10,25,50,100])

    print("\n=== PRE-COMMITTED STOP CONDITION (§204) ===")
    if not p1_signal and not p2_signal:
        print("  BOTH NULL -> fingerprinting CLOSED in every form (ratio §203, timing, raw-amount). BANK 0.83460.")
    else:
        print(f"  SIGNAL FOUND (timing={p1_signal}, lattice={p2_signal}) -> graduate to a gated feature build.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
