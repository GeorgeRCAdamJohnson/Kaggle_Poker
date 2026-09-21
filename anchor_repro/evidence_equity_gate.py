"""Does per-hand equity-evidence LIFT the family-conditional evidence ranker? (§174 plan #1)

Current best evidence = family-conditional (FAM) OOF MAP@5 0.5989 (§157, the shipped 0.83460 evidence).
This adds EQEV_FEATS (per-hand equity: surrender / checkdown / winner-ahead / loser-gave-edge) to the
per-family rankers and measures OOF MAP@5 under the EXACT host formula. Within-pair ranking is phase-immune
(no gate) -> local MAP@5 lift == LB evidence lift (proven 1:1). Ship if FAM+EQ > FAM by a real margin.

Reports:
  FAM        family-conditional, HAND_FEATS + ctx_emb                 [current best 0.5989]
  FAM+EQ     family-conditional, HAND_FEATS + ctx_emb + EQEV_FEATS
  per-family deltas (expect transfer/soft_play to gain; isolation ~flat).
Run:  python -m anchor_repro.evidence_equity_gate
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.evidence_equity import build_perhand, EQEV_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"
HAND_SEEDS_ORIG=list(HAND_SEEDS)

def fam_ranker(dh, base_feats, eq_families=()):
    """Per-family rankers routed by the pair's behavior_family. `eq_families` = families that ALSO get
    EQEV_FEATS (transfer/soft_play benefit; isolation is hurt by them so keep it on base_feats only)."""
    dh=dh.copy(); dh["hand_score"]=0.0; folds=sorted(dh["fold"].unique())
    for f in folds:
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            feats=base_feats+EQEV_FEATS if fam in eq_families else base_feats
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            models=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def score(dh, hs, tag):
    d=dh.copy(); d["hand_score"]=hs; pos=d[d["label"]==1]
    overall=map5(pos); fam={fm:map5(pos[pos["behavior_family"]==fm]) for fm in TARGET_BEHAVIORS}
    log(f"[{tag}] MAP@5 = {overall:.4f}  per family: {{ {', '.join(f'{k}: {v:.4f}' for k,v in fam.items())} }}")
    return overall, fam

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left")
    per=build_perhand("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=set(dh["pair_id"].to_list()))
    dh=dh.join(per,on=["pair_id","hand_id"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    for c in EQEV_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    log(f"equity coverage on labelled hands: {(dh['eq_max_pf']>0).mean():.3f}")

    FEATS=HAND_FEATS+CTXCOLS
    log("=== equity-evidence gate (current best FAM = 0.5989) ===")
    o0,f0=score(dh, fam_ranker(dh,FEATS),                                          "FAM              (HAND_FEATS+ctx)")
    o2,f2=score(dh, fam_ranker(dh,FEATS,eq_families=("directed_transfer","soft_play")), "FAM+EQ txfr+soft (eq on transfer+soft only)")
    log(f"delta txfr+soft= {o2-o0:+.4f}   ({ {fm:round(f2[fm]-f0[fm],4) for fm in TARGET_BEHAVIORS} })")

    # robustness: the whole delta rides on soft_play (a small gain). Re-run under 4 extra seed sets to
    # bound the noise. fam_ranker reads this module's global HAND_SEEDS, so patch it here.
    log("--- seed robustness for FAM+EQ txfr+soft ---")
    deltas=[o2-o0]
    for s in [101,202,303,404]:
        globals()["HAND_SEEDS"]=[s,s+7]
        oa,_=score(dh, fam_ranker(dh,FEATS),                                              f"  FAM seed{s}")
        ob,_=score(dh, fam_ranker(dh,FEATS,eq_families=("directed_transfer","soft_play")),f"  FAM+EQ seed{s}")
        deltas.append(ob-oa); log(f"    delta seed{s} = {ob-oa:+.4f}")
    globals()["HAND_SEEDS"]=HAND_SEEDS_ORIG
    log(f"delta over 5 seed-sets: mean {np.mean(deltas):+.4f}  min {min(deltas):+.4f}  max {max(deltas):+.4f}  (>0 in {sum(d>0 for d in deltas)}/5)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
