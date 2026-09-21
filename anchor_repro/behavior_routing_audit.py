"""§213 — Behavior routing audit: does the banked submission's family split (among its top-risk pairs, the
ones that will be scored as positive) MIRROR the dev true-family prior? If our family routing is skewed vs
the true distribution, behavior MAP leaks points that better routing could recover (behavior is 10% weight,
never actually measured — we assumed '99.5% accuracy' = done).
"""
from __future__ import annotations
import pandas as pd, polars as pl
from pathlib import Path
D=Path("data/poker"); OUT=Path("outputs/poker_collusion")

def main():
    lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).to_pandas()
    dev=lab["behavior_family"].value_counts()
    tot=dev.sum()
    print("=== DEV positive family distribution (true prior) ===")
    for k,v in dev.items(): print(f"  {k:24s} {v:4d}  ({100*v/tot:.1f}%)")
    s=pd.read_csv(OUT/"evidence_familycond_submission.csv")
    top=s.sort_values("risk_score",ascending=False).head(int(tot))
    print(f"\n=== banked submission: family split among TOP-{tot} by risk (should mirror dev prior) ===")
    print(top["predicted_behavior"].value_counts().to_string())
    print(f"\n=== all non-none predictions ({(s['predicted_behavior']!='none').sum()}) ===")
    print(s[s["predicted_behavior"]!="none"]["predicted_behavior"].value_counts().to_string())
    # normalized comparison
    print("\n=== normalized ratios (dev prior vs top-risk submission) ===")
    devr={k:v/tot for k,v in dev.items()}
    tv=top["predicted_behavior"].value_counts(); ttot=tv.sum()
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        d=devr.get(fam,0); t=tv.get(fam,0)/ttot if ttot else 0
        print(f"  {fam:24s} dev {d*100:5.1f}%  vs  top-risk {t*100:5.1f}%  gap {(t-d)*100:+.1f}pp")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
