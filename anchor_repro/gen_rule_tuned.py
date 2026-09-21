"""§219b — TUNED rule search (the tuning step I skipped in §219). Instead of my hand-picked thresholds,
LET A TREE LEARN the best rule from the raw generator quantities per family, and report the MAX achievable
purity at the coverage the metric cares about (top hands per pair). This tests whether "no rule works" is
real or an artifact of my hand-tuning. If a learned shallow tree also caps ~0.30 purity -> §219 verdict solid.
If it finds 0.6+ -> a recoverable rule exists that I missed.

Also: the honest ranker comparison — train a tree on the raw generator quantities ONLY (no HAND_FEATS/seq),
OOF, within-pair MAP@5. If it approaches the 0.60 learned ranker, the generator quantities alone are enough;
if it caps ~0.2 like the hand rules, the extra 0.4 MAP@5 genuinely needs the richer features (or unmodelable info).
Run: python -m anchor_repro.gen_rule_tuned
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3")

RAWQ=["a_net","b_net","pot_bb_g","winner_net","loser_net","pair_aggr","n_folded",
      "a_n_aggr","b_n_aggr"]
BOOLQ=["opposite_sign_net","both_showdown","both_no_aggr"]

def map5(df,score):
    d=df[["pair_id","is_planted","pot_bb_g"]].copy(); d["s"]=score
    d=d.sort_values(["pair_id","s","pot_bb_g"],ascending=[True,False,False],kind="mergesort")
    vals=[]
    for _,g in d.groupby("pair_id",sort=False):
        rel=g["is_planted"].to_numpy(); nr=int(rel.sum())
        if nr==0: continue
        top=rel[:5]; hits=np.cumsum(top); vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(nr,5)))
    return float(np.mean(vals)) if vals else 0.0

def main():
    fr=pl.read_parquet(STEP3/"edge"/"genrule"/"gen_rule_frame.parquet").to_pandas()
    for c in BOOLQ: fr[c]=fr[c].astype(float)
    fr["net_gap_g"]=(fr["winner_net"]-fr["loser_net"]).abs()
    FEATS=RAWQ+BOOLQ+["net_gap_g"]
    print("=== TUNED rule: shallow tree LEARNS best rule per family (max purity a rule of this form achieves) ===")
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        f=fr[fr.behavior_family==fam].copy().reset_index(drop=True)
        base=f["is_planted"].mean()
        X=f[FEATS].fillna(0).to_numpy(); y=f["is_planted"].to_numpy()
        grp=f["pair_id"].to_numpy()
        # OOF tree scores (group by pair) at depths 3 and 6
        for depth in [3,6]:
            oof=np.zeros(len(f))
            for tr,va in GroupKFold(5).split(X,y,grp):
                m=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=30,class_weight="balanced",random_state=42).fit(X[tr],y[tr])
                oof[va]=m.predict_proba(X[va])[:,1]
            auc=roc_auc_score(y,oof)
            # purity at top-coverage: take the highest-scoring hands = 2x planted count, measure purity
            k=int(y.sum()*2); order=np.argsort(-oof); topk=order[:k]
            purity=y[topk].mean(); cov=y[topk].sum()/y.sum()
            m5=map5(f,oof)
            print(f"  {fam:22s} depth{depth}: AUC {auc:.4f} | top-2x purity {purity:.3f} (base {base:.3f}) cov {cov:.3f} | within-pair MAP@5 {m5:.4f}")
    print("\n=== VERDICT ===")
    print("  If learned-tree MAP@5 ~0.2 and purity ~0.3 (like hand rules) -> §219 'no rule' is SOLID, not a")
    print("  hand-tuning artifact. If MAP@5 ->0.6 or purity ->0.6 -> a recoverable rule was missed.")
    print("  (learned ranker with full HAND_FEATS+seq = 0.60; raw-generator-only tree isolates what the")
    print("   generator quantities ALONE can do.)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
