"""FORENSIC AUDIT of the banked 0.83460 submission — the one thing 15 component-level experiments never did:
look at the ASSEMBLED OUTPUT for structural defects invisible to component gates.

Checks:
  1. risk_score distribution: range, uniqueness, how many pairs get meaningfully-nonzero risk vs floored.
  2. predicted_behavior distribution: are we flagging plausible counts per family, or dumping everything to
     'none'? Compare to the ~0.2% expected positive rate.
  3. evidence completeness: do all pairs have 5 evidence hands, or are many NO_EVIDENCE / duplicated?
  4. risk vs behavior coherence: do high-risk pairs actually get a target behavior (else behavior MAP is
     capped by mis-routing)?
  5. THE KEY ONE: does the risk_score RANKING look sane at the top? Inspect the top-200 pairs — do they
     cluster in ways that suggest a defect (e.g., all from few tables, all one behavior)?
Run:  python -m anchor_repro.submission_forensics
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
OUT=Path("outputs/poker_collusion")

def main():
    s=pd.read_csv(OUT/"evidence_familycond_submission.csv")
    print(f"=== BANKED SUBMISSION 0.83460: {len(s)} rows, cols {list(s.columns)} ===")
    r=pd.to_numeric(s["risk_score"],errors="coerce")
    print(f"\n1. risk_score: min {r.min():.6f} max {r.max():.6f} mean {r.mean():.4f} unique {r.nunique()}/{len(r)}")
    for q in [0.5,0.9,0.99,0.999]: print(f"   q{q}: {r.quantile(q):.6f}")
    print(f"   pairs with risk>0.5: {(r>0.5).sum()} | >0.1: {(r>0.1).sum()} | ~0 (<1e-4): {(r<1e-4).sum()}")

    print(f"\n2. predicted_behavior distribution:")
    print(s["predicted_behavior"].value_counts().to_string())
    exp_pos=int(len(s)*0.002)  # ~0.2% expected positive rate
    print(f"   (expected ~{exp_pos} true positives at 0.2% rate; how many non-'none' behaviors do we emit?)")
    nonnone=(s["predicted_behavior"]!="none").sum()
    print(f"   non-'none' predicted: {nonnone}")

    EV=[f"evidence_hand_{i}" for i in range(1,6)]
    print(f"\n3. evidence completeness:")
    ne=(s[EV]=="NO_EVIDENCE").sum(axis=1)
    print(f"   pairs with all 5 evidence: {(ne==0).sum()} | with some NO_EVIDENCE: {(ne>0).sum()} | all NO_EVIDENCE: {(ne==5).sum()}")
    dup=s[EV].apply(lambda row: len(set(row))<len([x for x in row if x!='NO_EVIDENCE']),axis=1).sum()
    print(f"   pairs with duplicate evidence hands: {dup}")

    print(f"\n4. risk vs behavior coherence (do high-risk pairs get a target behavior?):")
    top=s.assign(r=r).sort_values("r",ascending=False)
    for k in [50,200,500]:
        tk=top.head(k); nn=(tk["predicted_behavior"]!="none").mean()
        print(f"   top-{k} by risk: {nn*100:.0f}% have a non-'none' behavior; families {dict(tk['predicted_behavior'].value_counts())}")

    print(f"\n5. top-200 concentration (defect check):")
    tk=top.head(200)
    # if pairs carry table_id or player info we could check clustering; submission likely has only pair_id
    print(f"   top-200 risk range [{tk['r'].min():.4f}, {tk['r'].max():.4f}]")
    print(f"   distinct evidence_hand_1 values in top-200: {tk['evidence_hand_1'].nunique()} (low => repeated/defective)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
