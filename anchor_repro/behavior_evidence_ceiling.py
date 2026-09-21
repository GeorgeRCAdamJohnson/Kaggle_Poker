"""§213 — MEASURE the actual Behavior and Evidence component scores (not assumptions) and their true
headroom, using the EXACT host metric on dev labels. We keep quoting '99.5% family accuracy' as if it's the
behavior MAP — those are different. And 'evidence floored 0.599' — measure what PERFECT would give.

BehaviorMAP (host §185) = mean over 3 families of AP(is_family, risk_score where predicted==family else 0).
Compute our CURRENT behavior MAP on dev, and the CEILING behavior MAP if predicted_behavior were PERFECT
(argmax = true family for every pair). The gap = recoverable behavior points.
EvidenceMAP: current 0.599 (measured). Ceiling = 1.0. But also compute a REALISTIC ceiling: if we perfectly
ranked WITHIN our current candidate set (oracle), what's the max? (recall@5 bound).
Run:  python -m anchor_repro.behavior_evidence_ceiling
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); OUT=Path("outputs/poker_collusion")
TARGET=("directed_transfer","soft_play","coordinated_isolation")

def ap(y_true,scores):
    y=np.asarray(y_true); s=np.asarray(scores)
    if y.sum()==0: return 0.0
    o=np.argsort(-s,kind="mergesort"); yr=y[o]; tp=np.cumsum(yr); ranks=np.arange(1,len(yr)+1)
    return float(np.sum((tp/ranks)*yr)/y.sum())

def main():
    # use the labeled dev pairs with a risk proxy: reproduce from the banked submission is eval-only, so
    # for dev we need dev risk. Use dev_pairs label + a dev OOF risk if available; else use behavior-only logic.
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").to_pandas()
    print("dev_pairs cols:",[c for c in dp.columns if c in ('label','behavior_family','is_labeled','pred_family','risk')] )
    lab=dp[dp["is_labeled"]==True].copy() if "is_labeled" in dp.columns else dp.copy()
    y=lab["label"].fillna(0).astype(int).to_numpy()
    fam=lab["behavior_family"].fillna("none").to_numpy()
    print(f"labeled dev pairs {len(lab)} | positives {int(y.sum())}")
    print("family counts (positives):", pd.Series(fam[y==1]).value_counts().to_dict())

    # We need a RISK score on dev. Try common columns.
    riskcol=None
    for c in ["risk","risk_score","oof","sus_score"]:
        if c in lab.columns: riskcol=c; break
    if riskcol is None:
        print("\nNO dev risk column in dev_pairs — computing behavior CEILING structure only (family-independent).")
        # Behavior MAP with PERFECT ranking (risk = label) and PERFECT family:
        perfect=y.astype(float)
        bmap_perfect_fam_perfect_rank=np.mean([ap((fam==f)&(y==1),np.where(True,perfect,0)) for f in TARGET])
        print(f"  ceiling BehaviorMAP (perfect risk-rank + perfect family) = {bmap_perfect_fam_perfect_rank:.4f}")
        # note: with perfect rank this is ~1.0 per family that has positives -> mean ~1.0
        return 0

    r=lab[riskcol].to_numpy()
    fampred=lab["pred_family"].fillna("none").to_numpy() if "pred_family" in lab.columns else fam
    print(f"\nusing dev risk col '{riskcol}'")
    # CURRENT behavior MAP: our predicted family, our risk
    cur=np.mean([ap((fam==f)&(y==1), np.where(fampred==f,r,0.0)) for f in TARGET])
    # CEILING A: perfect family (pred=true), our risk ranking
    ceilfam=np.mean([ap((fam==f)&(y==1), np.where(fam==f,r,0.0)) for f in TARGET])
    # CEILING B: perfect family + perfect rank
    ceilall=np.mean([ap((fam==f)&(y==1), np.where(fam==f,y.astype(float),0.0)) for f in TARGET])
    print(f"  CURRENT  BehaviorMAP (our family, our risk) = {cur:.4f}")
    print(f"  CEILING  perfect family + our risk          = {ceilfam:.4f}  (headroom from family fix {ceilfam-cur:+.4f})")
    print(f"  CEILING  perfect family + perfect rank       = {ceilall:.4f}")
    print(f"\n  behavior weight 0.10 -> recoverable from perfect FAMILY routing = {0.10*(ceilfam-cur):+.4f} LB")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
