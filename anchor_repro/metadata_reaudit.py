"""§212 — RE-AUDIT player metadata, NOT trusting the old "flat 0.3558 vs 0.3566" verdict (that was the broken
~§1540 era pipeline, tested as a RISK input, pre-eval-weighted-gate, pre-leak-battery). User: "already tested
doesn't mean we did it right." Re-examine with today's discipline, and for BOTH components:
  (A) RISK/label separation: do same_region/same_client/account_age_gap/same_stake/same_experience separate
      the 372 positive pairs from negatives? Univariate AUC on dev labels, HONEST. Then the KEY transfer
      question: do the metadata-MATCH RATES differ dev-labeled-pos vs eval population? (if eval pairs are
      mostly non-matching, a 'match=collusion' signal won't transfer -- the recurring §194 trap).
  (B) Was the old test even measuring the right thing? Report per-family and the actual match rates.
Metadata is per-pair CONSTANT so it cannot be a within-pair EVIDENCE signal directly -- but audit (A) first;
if it separates AND transfers, it is a RISK lever the old broken pipeline may have wrongly dismissed.
Run:  python -m anchor_repro.metadata_reaudit
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
D=Path("data/poker")

def main():
    lab=pl.read_csv(D/"development_labels.csv").to_pandas()
    ev=pl.read_csv(D/"evaluation_pairs.csv").to_pandas()
    pmeta=pl.read_parquet(D/"players.parquet").to_pandas().set_index("player_id")
    def meta_feats(df):
        rows=[]
        for a,b in zip(df["player_1"],df["player_2"]):
            pa=pmeta.loc[a] if a in pmeta.index else None; pb=pmeta.loc[b] if b in pmeta.index else None
            if pa is None or pb is None: rows.append((0,0,0,0,np.nan)); continue
            rows.append((int(pa.region_bucket==pb.region_bucket),int(pa.client_family==pb.client_family),
                         int(pa.preferred_stake==pb.preferred_stake),int(pa.experience_hands_bucket==pb.experience_hands_bucket),
                         abs(pa.account_age_days-pb.account_age_days)))
        return pd.DataFrame(rows,columns=["same_region","same_client","same_stake","same_exp","age_gap"])
    ml=meta_feats(lab); me=meta_feats(ev)
    for c in ml.columns: lab[c]=ml[c].values
    y=lab["label"].to_numpy()
    print("=== (A) metadata univariate label AUC on dev (HONEST, today) ===")
    for c in ["same_region","same_client","same_stake","same_exp","age_gap"]:
        v=lab[c].fillna(lab[c].median()).to_numpy().astype(float)
        a=roc_auc_score(y,v); print(f"  {c:12s} AUC {max(a,1-a):.4f} ({'hi=pos' if a>0.5 else 'lo=pos'})")
    print("\n=== match RATES: dev-pos vs dev-neg vs EVAL population (transfer check) ===")
    pos=lab[lab.label==1]; neg=lab[lab.label==0]
    for c in ["same_region","same_client","same_stake","same_exp"]:
        print(f"  {c:12s} dev-pos {pos[c].mean():.3f} | dev-neg {neg[c].mean():.3f} | EVAL {me[c].mean():.3f}")
    print(f"  age_gap      dev-pos {pos['age_gap'].mean():.0f} | dev-neg {neg['age_gap'].mean():.0f} | EVAL {me['age_gap'].mean():.0f}")
    print("\n=== combined metadata AUC (all 5, quick logistic) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    X=lab[["same_region","same_client","same_stake","same_exp","age_gap"]].fillna(0).to_numpy()
    Xs=(X-X.mean(0))/(X.std(0)+1e-9)
    p=cross_val_predict(LogisticRegression(max_iter=500),Xs,y,cv=5,method="predict_proba")[:,1]
    print(f"  combined metadata 5-fold AUC = {roc_auc_score(y,p):.4f}")
    print("  INTERPRETATION: if AUC>0.6 AND dev-pos match rate differs from EVAL -> old 'flat' verdict was")
    print("  wrong for the wrong reason, but signal may be dev-selection (won't transfer). If match rates")
    print("  MATCH dev-pos<->eval AND AUC>0.6 -> a real transferable RISK signal the old pipeline missed.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
