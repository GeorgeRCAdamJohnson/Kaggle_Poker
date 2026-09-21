"""§216 Lead 2 — RELATIVE (cross-candidate) reranker. §180's lambdarank reranker failed using ABSOLUTE
features. The AtMem/Jev finding: reranking helps when the model sees candidates RELATIVE to each other. And
§184 showed the host picks SUBTLE hands relative to the PAIR'S OWN distribution. So: augment the family-
conditional ranker with RELATIVE features = each hand's key signals normalized WITHIN its pair (value minus
pair-median, and within-pair percentile rank). This is chip-independent of absolute scale and is exactly the
"is this hand notable FOR THIS PAIR" signal a within-pair metric rewards. Guard vs §180: relative, not absolute.

Build per-pair-relative versions of the strongest raw signals (pot, winner_net, transfer, board_hit, the
ci/ox families) and gate on the family-conditional ranker, 5 seeds. Ship if mean >+0.003 AND >=4/5 positive.
Run:  python -m anchor_repro.relative_rerank_gate
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
HAND_SEEDS_ORIG=list(HAND_SEEDS)
# raw signals to make pair-RELATIVE (within-pair percentile + deviation-from-pair-median)
RAWS=["pot_bb","net_gap","transfer_pot_ratio","max_amount_bb","players_at_showdown","last_street"]
REL=[f"rel_{c}_pct" for c in RAWS]+[f"rel_{c}_dev" for c in RAWS]

def add_relative(dh):
    dh=dh.copy()
    for c in RAWS:
        if c not in dh.columns: dh[c]=0.0
        g=dh.groupby("pair_id")[c]
        dh[f"rel_{c}_pct"]=g.rank(pct=True).astype("float32")
        med=g.transform("median"); dh[f"rel_{c}_dev"]=(dh[c]-med).astype("float32")
    return dh

def fam_ranker(dh, feats, rel_families=()):
    dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            ff=feats+REL if fam in rel_families else feats
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,ff],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te&(dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,ff]) for m in ms],axis=0)
    return dh["hand_score"].to_numpy()

def m5(dh,hs):
    d=dh.copy(); d["hand_score"]=hs; return map5(d[d["label"]==1])

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    dh=add_relative(dh)
    for c in REL: dh[c]=dh[c].astype("float32").fillna(0)
    FEATS=HAND_FEATS+CTXCOLS; ALL3=TARGET_BEHAVIORS
    # quick orthogonality: are the RELATIVE feats reconstructable from HAND_FEATS? (they shouldn't be — they
    # encode within-pair rank which absolute per-hand features lack)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score
    pos=dh[dh["label"]==1]; ps=pos.sample(n=min(30000,len(pos)),random_state=42)
    X=ps[HAND_FEATS].fillna(0).to_numpy()
    log("orthogonality of relative feats vs HAND_FEATS (want LOW):")
    for c in ["rel_pot_bb_pct","rel_net_gap_pct","rel_pot_bb_dev"]:
        r2=r2_score(ps[c].to_numpy(),cross_val_predict(Ridge(alpha=10),X,ps[c].to_numpy(),cv=3))
        log(f"  {c:20s} R2 {r2:+.3f}")
    b=m5(dh,fam_ranker(dh,FEATS)); log(f"FAM base MAP@5 {b:.4f}")
    e=m5(dh,fam_ranker(dh,FEATS,rel_families=ALL3)); log(f"FAM+relative MAP@5 {e:.4f}  delta {e-b:+.4f}")
    log("--- seed robustness (5 seeds) ---")
    ds=[e-b]
    for s in [101,202,303,404]:
        globals()["HAND_SEEDS"]=[s,s+7]
        bb=m5(dh,fam_ranker(dh,FEATS)); ee=m5(dh,fam_ranker(dh,FEATS,rel_families=ALL3)); ds.append(ee-bb); log(f"  seed{s}: base {bb:.4f} +rel {ee:.4f} d {ee-bb:+.4f}")
    globals()["HAND_SEEDS"]=HAND_SEEDS_ORIG
    log(f"delta over 5 seeds: mean {np.mean(ds):+.4f} min {min(ds):+.4f} max {max(ds):+.4f} (>0 in {sum(x>0 for x in ds)}/5)")
    log("SHIP if mean >+0.003 AND >=4/5 positive.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
