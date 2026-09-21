"""§219 PHASE 1b — per-family RULE search with COVERAGE + PURITY (the §109 core, never done).
For each family, define candidate near-deterministic rules and measure:
  COVERAGE = planted hands matching rule / all planted hands (of that family)
  PURITY   = planted hands matching rule / ALL pair-hands matching rule (that family's shared hands)
  within-pair MAP@5 if the rule-score ranks (tie-break by magnitude)
Base rate = planted / all shared hands per family (purity floor). A rule with purity >> base rate AND high
coverage is a recovered generator signature. This reframes evidence from ranking-lift to rule-recovery.
Run: python -m anchor_repro.gen_rule_search
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3")

def cov_pur(df, mask):
    pl_=df["is_planted"].to_numpy(); m=mask
    planted=pl_.sum()
    match=m.sum()
    match_planted=(m & (pl_==1)).sum()
    coverage=match_planted/max(planted,1)
    purity=match_planted/max(match,1)
    return coverage,purity,int(match),int(match_planted)

def map5_score(df, score):
    import pandas as pd
    d=df.select(["pair_id","is_planted","pot_bb_g"]).to_pandas(); d["s"]=score
    d=d.sort_values(["pair_id","s","pot_bb_g"],ascending=[True,False,False],kind="mergesort")
    vals=[]
    for _,g in d.groupby("pair_id",sort=False):
        rel=g["is_planted"].to_numpy(); nr=int(rel.sum())
        if nr==0: continue
        top=rel[:5]; hits=np.cumsum(top); vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(nr,5)))
    return float(np.mean(vals)) if vals else 0.0

def main():
    fr=pl.read_parquet(STEP3/"edge"/"genrule"/"gen_rule_frame.parquet")
    print("=== per-family RULE coverage/purity (base rate = planted/all shared hands) ===")
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        f=fr.filter(pl.col("behavior_family")==fam)
        base=f["is_planted"].mean()
        print(f"\n--- {fam}  (n={f.height}, planted={int(f['is_planted'].sum())}, base rate {base:.3f}) ---")
        wnet=f["winner_net"].to_numpy(); lnet=f["loser_net"].to_numpy(); pot=f["pot_bb_g"].to_numpy()
        opp=f["opposite_sign_net"].to_numpy(); bsd=f["both_showdown"].to_numpy(); paggr=f["pair_aggr"].to_numpy()
        both_no=f["both_no_aggr"].to_numpy()
        rules={}
        if fam=="directed_transfer":
            rules["opp_sign & pot>=20"]=(opp) & (pot>=20)
            rules["opp_sign & pot>=40"]=(opp) & (pot>=40)
            rules["opp_sign & loser_net<=-15"]=(opp) & (lnet<=-15)
            rules["opp_sign & pot>=30 & pair_aggr>=1"]=(opp)&(pot>=30)&(paggr>=1)
        elif fam=="soft_play":
            rules["both_showdown & pot>=20"]=(bsd)&(pot>=20)
            rules["opp_sign & both_showdown"]=(opp)&(bsd)
            rules["both_showdown & pot>=15 & pair_aggr<=3"]=(bsd)&(pot>=15)&(paggr<=3)
            rules["opp_sign & pot>=25"]=(opp)&(pot>=25)
        else:
            rules["pair_aggr>=2 & pot>=30"]=(paggr>=2)&(pot>=30)
            rules["pair_aggr>=2 & pot>=40"]=(paggr>=2)&(pot>=40)
            rules["opp_sign & pair_aggr>=2"]=(opp)&(paggr>=2)
            rules["pot>=40"]=(pot>=40)
        for name,mask in rules.items():
            c,p,mm,mp=cov_pur(f,mask)
            lift=p/max(base,1e-9)
            print(f"  {name:34s} coverage {c:.3f} purity {p:.3f} (x{lift:.1f} base) match {mm} planted-in {mp}")
        # best rule as a ranker score = rule match (1/0) + pot tiebreak, within-pair MAP@5
        best=max(rules.items(),key=lambda kv: cov_pur(f,kv[1])[1]*cov_pur(f,kv[1])[0])
        m5=map5_score(f, best[1].astype(float))
        print(f"  -> best-purity*coverage rule '{best[0]}' within-pair MAP@5 {m5:.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
