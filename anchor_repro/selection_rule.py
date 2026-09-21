"""HOW DID THE HOST CHOOSE WHICH HANDS ARE EVIDENCE? (§184/§185: our collusiveness ranking is AUC 0.98 but
the labeled hands aren't always the most collusive-looking -> the host uses a SELECTION rule we don't model.
Metric is set-based, so we only need to get the right 5 hands into the top-5.)

Test candidate selection rules as PURE within-pair rankers of the true evidence hands (host-exact MAP@5),
AND as separators of buried-true (our rank>5 true) vs promoted-decoy (our rank<=5 non-true) — the §184 gap.
Candidate rules (raw hand properties, no learned collusiveness):
  pot_bb              : biggest pots
  transfer_any        : largest directed chip flow
  net_gap             : biggest net swing between the pair
  max_amount_bb       : biggest single bet
  hand_idx  (+/-)     : chronology (earliest / latest occurrence)
  last_street         : hands that went deeper (more streets)
  transfer_pot_ratio  : transfer as fraction of pot
  winner_weak_won     : winner had weak hand
If a raw rule ranks true hands well (MAP@5) or separates buried-true>decoy where collusiveness fails, it's
the missing selection signal -> add it. Also test blends of the base collusiveness score with the best rule.
Run:  python -m anchor_repro.selection_rule
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.seq_evidence import HAND_FEATS, map5, TARGET_BEHAVIORS, log
from anchor_repro.evidence_rerank import build_dh_ranked
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
S1_CACHE=SEQ/"rerank_stage1_lgb.parquet"

RULES=["pot_bb","transfer_any","net_gap","max_amount_bb","hand_idx","last_street","transfer_pot_ratio","winner_weak_won","transfer_pot_ratio_prank","pot_bb_prank"]

def m5(pos, col, asc=False):
    d=pos.copy(); d["hand_score"]=(-d[col] if asc else d[col]).astype(float); return map5(d)

def main():
    dh=build_dh_ranked().to_pandas()
    dh["evidence_rank"]=dh["evidence_rank"].fillna(0).astype(int); dh["is_evidence"]=dh["evidence_rank"]>0
    for c in RULES:
        if c in dh.columns: dh[c]=dh[c].astype("float32")
    c=pl.read_parquet(S1_CACHE).to_pandas(); dh=dh.merge(c,on=["pair_id","hand_id"],how="left")
    pos=dh[dh["label"]==1].copy()
    base=map5(pos.rename(columns={"s1":"hand_score"}))
    log(f"base collusiveness MAP@5 = {base:.4f}")

    log("=== candidate SELECTION rules as pure within-pair rankers (host MAP@5) ===")
    res=[]
    for c in RULES:
        if c not in pos.columns: continue
        hi=m5(pos,c,asc=False); lo=m5(pos,c,asc=True)
        best=max(hi,lo); dirn="high" if hi>=lo else "low"
        res.append((c,best,dirn)); log(f"  {c:26s} MAP@5 {best:.4f} ({dirn})   [hi {hi:.4f} / lo {lo:.4f}]")
    res.sort(key=lambda x:-x[1])

    log("=== §184 gap test: does a raw rule separate BURIED-TRUE from PROMOTED-DECOY? ===")
    pos=pos.sort_values(["pair_id","s1","pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    pos["rk"]=pos.groupby("pair_id").cumcount()+1
    miss=pos[(pos["is_evidence"])&(pos["rk"]>5)]; decoy=pos[(~pos["is_evidence"])&(pos["rk"]<=5)]
    lab=np.r_[np.ones(len(miss)),np.zeros(len(decoy))]
    for c,_,_ in res:
        v=np.r_[miss[c].to_numpy(),decoy[c].to_numpy()]
        try:
            a=roc_auc_score(lab,v); log(f"  {c:26s} buried-vs-decoy AUC {max(a,1-a):.4f} ({'high=true' if a>0.5 else 'low=true'})")
        except Exception: pass

    log("=== blend base collusiveness (rank-pct) with top rule (rank-pct), within pair ===")
    pos["s1_pct"]=pos.groupby("pair_id")["s1"].rank(pct=True)
    top_rule=res[0][0]
    pos["rule_pct"]=pos.groupby("pair_id")[top_rule].rank(pct=True, ascending=(res[0][2]=="low"))
    for w in [0.0,0.1,0.2,0.3,0.5]:
        blend=(1-w)*pos["s1_pct"]+w*pos["rule_pct"]
        d=pos.copy(); d["hand_score"]=blend
        log(f"  base*(1-{w})+{top_rule}*{w}: MAP@5 {map5(d):.4f}  d {map5(d)-base:+.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
