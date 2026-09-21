"""IS CO-OCCURRENCE A REAL COLLUSION SIGNAL OR A DEV-SELECTION ARTIFACT? (user: a real path to .9 exists,
one of the co-occurrence methods may be onto something we're missing. §193 found solo-baseline
NORMALIZATION leaks — but that doesn't prove co-occurrence ITSELF is worthless. Test directly.)

The distinguishing question: does the shared-hand-count / co-occurrence distribution of dev-LABELED pairs
match the EVAL pairs we score? 
  - If eval pairs ALSO skew high co-occurrence -> it's a real generator property (AAAI colluders share
    tables by construction) -> use it DIRECTLY as a risk feature.
  - If eval is uniform while dev-positives spike -> pure selection artifact -> dead on LB.

Probes:
  P1 dev: shared_hands distribution for POS vs NEG labeled pairs (does it separate? AUC).
  P2 dev-vs-eval: shared_hands distribution of dev-labeled pairs vs the 112540 eval pairs (the SCORED
     population). If eval matches dev-POS -> real; if eval matches dev-NEG/uniform -> artifact.
  P3 the KEY control: among dev pairs at SIMILAR shared_hands, does the label still separate on other
     signal? (i.e., is co-occurrence confounded with, or independent of, the real collusion signal)
Run:  python -m anchor_repro.cooccur_probe
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3")
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    log(f"dev cols has shared_hands: {'shared_hands' in dev.columns}; eval: {'shared_hands' in evf.columns}")
    sh="shared_hands"
    y=dev["label"].fillna(0).astype(int).to_numpy(); lab=dev["is_labeled"].to_numpy().astype(bool)

    log("=== P1 dev: shared_hands separates POS vs NEG (labeled)? ===")
    v=dev[sh].to_numpy().astype(float)
    a=roc_auc_score(y[lab],v[lab]); log(f"  shared_hands AUC (labeled pos vs neg) = {max(a,1-a):.4f} ({'hi=pos' if a>0.5 else 'lo=pos'})")
    posv=dev.loc[lab&(y==1),sh]; negv=dev.loc[lab&(y==0),sh]
    log(f"  POS shared_hands: mean {posv.mean():.1f} median {posv.median():.0f} p10 {posv.quantile(.1):.0f} p90 {posv.quantile(.9):.0f}")
    log(f"  NEG shared_hands: mean {negv.mean():.1f} median {negv.median():.0f} p10 {negv.quantile(.1):.0f} p90 {negv.quantile(.9):.0f}")

    log("=== P2 THE KEY TEST: does EVAL shared_hands match dev-POS (real) or dev-NEG/uniform (artifact)? ===")
    ev=evf[sh].to_numpy().astype(float)
    log(f"  EVAL shared_hands: mean {ev.mean():.1f} median {np.median(ev):.0f} p10 {np.percentile(ev,10):.0f} p90 {np.percentile(ev,90):.0f} max {ev.max():.0f}")
    log(f"  dev-POS mean {posv.mean():.1f} | dev-NEG mean {negv.mean():.1f} | EVAL mean {ev.mean():.1f}")
    # what fraction of eval pairs exceed the dev-POS median co-occurrence?
    thr=posv.median()
    log(f"  frac EVAL pairs with shared_hands >= dev-POS median ({thr:.0f}): {(ev>=thr).mean():.3f}")
    log(f"  frac dev-NEG pairs with shared_hands >= dev-POS median: {(negv>=thr).mean():.3f}")
    log("  -> if EVAL frac ~ dev-NEG frac (low), co-occurrence is a dev-POS SELECTION artifact (eval pairs")
    log("     were NOT selected for high co-occurrence). if EVAL frac high, it's a real property.")

    log("=== P3 control: within similar shared_hands bands, does label still separate (co-occ independent of signal)? ===")
    dd=dev[lab].copy(); dd["y"]=y[lab]
    dd["band"]=pd.qcut(dd[sh],4,labels=False,duplicates="drop")
    for b in sorted(dd["band"].dropna().unique()):
        s=dd[dd["band"]==b]; log(f"  band {int(b)} shared_hands~[{s[sh].min():.0f},{s[sh].max():.0f}] pos_rate {s['y'].mean():.3f} (n={len(s)})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
