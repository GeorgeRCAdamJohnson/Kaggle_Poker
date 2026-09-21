"""§217 — quick probe of corrected pair-level action-MI (risk feature). Before a full gate: (1) label AUC,
(2) drift (dev vs eval means + adversarial), (3) is the field-normalization doing anything (pair vs field).
Build eval too. Decide whether a full eval-weighted gate is warranted.
Run: python -m anchor_repro.action_mi_v2_probe
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.action_mi_v2 import MIV2_FEATS, build
STEP3=Path("outputs/poker_collusion/hosen42_step3")

def main():
    dpath=STEP3/"edge"/"actionmi"/"action_mi_v2_development.parquet"
    epath=STEP3/"edge"/"actionmi"/"action_mi_v2_evaluation.parquet"
    d=pl.read_parquet(dpath).to_pandas()
    if not epath.exists():
        print("building eval..."); build("evaluation",STEP3/"eval_pair_hands.parquet")
    e=pl.read_parquet(epath).to_pandas()
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","is_labeled"]).to_pandas()
    m=dp.merge(d,on="pair_id",how="left")
    lab=m["is_labeled"].astype(bool).to_numpy() if m["is_labeled"].dtype!=bool else m["is_labeled"].to_numpy()
    y=m["label"].fillna(0).astype(int).to_numpy()
    print("=== (1) label AUC (labeled dev pos vs neg) ===")
    for c in MIV2_FEATS:
        v=m[c].fillna(0).to_numpy().astype(float)
        try: a=roc_auc_score(y[lab],v[lab]); print(f"  {c:16s} AUC {max(a,1-a):.4f} ({'hi=pos' if a>0.5 else 'lo=pos'})")
        except Exception as ex: print(f"  {c}: {ex}")
    print("\n=== (2) dev vs eval means (transfer check) + pos vs neg ===")
    pos=m[m.label==1]; neg=m[(m.label==0)&lab]
    for c in MIV2_FEATS:
        print(f"  {c:16s} dev {m[c].mean():.4f} eval {e[c].mean():.4f} | pos {pos[c].mean():.4f} neg {neg[c].mean():.4f}")
    print("\n=== interpretation ===")
    print("  ami_excess (pair MI minus field baseline) is the key: if pos>>neg AND AUC>0.6 AND dev~eval means,")
    print("  it's a real phase-robust RISK signal -> full gate. If flat, corrected action-MI is also dead.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
