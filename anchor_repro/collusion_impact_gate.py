"""Does per-hand COLLUSION-IMPACT (§186 paradigm) LIFT the family-conditional evidence ranker? Within-pair,
phase-immune (§185 set-metric) -> local MAP@5 == LB, no gate. Current best FAM = 0.5989.

Also the §184 GAP test: does ci_peak_eq_gap separate buried-true (rank>5) from promoted-decoy (rank<=5)?
§184 showed buried-true are the SUBTLE hands; ci_peak_eq_gap measures subtlety (how far ahead the loser was
by equity). If it separates where action-features failed, it's the missing selection signal.
Seed-robustness on the winner (5 seeds) — the same rigor that caught the §175 equity false positive.
Run:  python -m anchor_repro.collusion_impact_gate
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.collusion_impact import build_impact, CI_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"
HAND_SEEDS_ORIG=list(HAND_SEEDS)
TS=("directed_transfer","soft_play")

def fam_ranker(dh, base_feats, ci_families=(), ci_use=CI_FEATS):
    dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            feats=base_feats+list(ci_use) if fam in ci_families else base_feats
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    return dh["hand_score"].to_numpy()

def score(dh, hs, tag):
    d=dh.copy(); d["hand_score"]=hs; pos=d[d["label"]==1]
    o=map5(pos); fam={fm:map5(pos[pos["behavior_family"]==fm]) for fm in TARGET_BEHAVIORS}
    log(f"[{tag}] MAP@5 {o:.4f}  fam={{ {', '.join(f'{k[:4]}:{v:.3f}' for k,v in fam.items())} }}"); return o,fam

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet"); dh=dh.join(emb,on=["pair_id","hand_idx"],how="left")
    ci=build_impact("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=set(dh["pair_id"].to_list()))
    dh=dh.join(ci,on=["pair_id","hand_id"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    for c in CI_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    FEATS=HAND_FEATS+CTXCOLS

    log("=== §184 GAP: does ci separate buried-true (rank>5) vs promoted-decoy (rank<=5)? ===")
    s0=fam_ranker(dh,FEATS)
    d0=dh.copy(); d0["s"]=s0; pos=d0[d0["label"]==1].sort_values(["pair_id","s","pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    pos["rk"]=pos.groupby("pair_id").cumcount()+1
    miss=pos[(pos["is_evidence"])&(pos["rk"]>5)]; decoy=pos[(~pos["is_evidence"])&(pos["rk"]<=5)]
    lab=np.r_[np.ones(len(miss)),np.zeros(len(decoy))]
    for c in CI_FEATS:
        v=np.r_[miss[c].to_numpy(),decoy[c].to_numpy()]
        try: a=roc_auc_score(lab,v); log(f"  {c:18s} AUC {max(a,1-a):.4f} ({'high=true' if a>0.5 else 'low=true'})")
        except Exception: pass

    log("=== gate: FAM vs +collusion-impact (default seed) ===")
    o0,f0=score(dh, s0, "FAM base                ")
    o_all,_=score(dh, fam_ranker(dh,FEATS,ci_families=TARGET_BEHAVIORS),      "FAM+CI all              ")
    o_ts,_ =score(dh, fam_ranker(dh,FEATS,ci_families=TS),                    "FAM+CI txfr+soft        ")
    o_gap,_=score(dh, fam_ranker(dh,FEATS,ci_families=TS,ci_use=["ci_peak_eq_gap"]), "FAM+CI peak_eq_gap only ")
    o_gs,_ =score(dh, fam_ranker(dh,FEATS,ci_families=TS,ci_use=["ci_peak_eq_gap","ci_ev_surrender"]), "FAM+CI gap+surrender    ")
    log(f"  d all {o_all-o0:+.4f} | d txfr+soft {o_ts-o0:+.4f} | d gap-only {o_gap-o0:+.4f} | d gap+surr {o_gs-o0:+.4f}")

    # winner seed-robustness
    cand=max([("all",TARGET_BEHAVIORS,CI_FEATS,o_all),("ts",TS,CI_FEATS,o_ts),
              ("gap",TS,["ci_peak_eq_gap"],o_gap),("gs",TS,["ci_peak_eq_gap","ci_ev_surrender"],o_gs)],key=lambda x:x[3])
    log(f"--- seed robustness of winner: {cand[0]} ---")
    ds=[cand[3]-o0]
    for s in [101,202,303,404]:
        globals()["HAND_SEEDS"]=[s,s+7]
        a,_=score(dh, fam_ranker(dh,FEATS), f"  base seed{s}")
        b,_=score(dh, fam_ranker(dh,FEATS,ci_families=cand[1],ci_use=cand[2]), f"  +CI  seed{s}")
        ds.append(b-a); log(f"    d seed{s} {b-a:+.4f}")
    globals()["HAND_SEEDS"]=HAND_SEEDS_ORIG
    log(f"winner delta over 5 seeds: mean {np.mean(ds):+.4f} min {min(ds):+.4f} max {max(ds):+.4f} (>0 in {sum(x>0 for x in ds)}/5)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
