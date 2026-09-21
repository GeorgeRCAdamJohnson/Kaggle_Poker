"""FAMILY-CONDITIONAL evidence ranking (build from what the dossier CONFIRMED about collusion shape).

The dossier §3.1 confirms the three families have MECHANICALLY DISTINCT per-hand signatures:
  directed_transfer  -> net directed chip flow A->B via a value-committing action (transfer_any)
  soft_play          -> declined partner-directed aggression (fold_to_partner / low raise_partner)
  coordinated_isolation -> JOINT bet/raise pressure on a THIRD player (raise_other by both)
The current evidence ranker (§150, HAND_FEATS+ctx_emb -> MAP@5 0.5710) pools ALL families into ONE
generic is_evidence model -> a hand with high transfer_any is evidence for a directed_transfer pair but
noise for an isolation pair, and the single ranker must AVERAGE over that. Since family classification is
99.5% accurate (§152), we can ROUTE: train 3 per-family rankers (each on that family's evidence hands)
and score each pair's hands with the ranker matching its family. Encodes the confirmed shape into target.

Compares (OOF MAP@5, exact host formula, over positive pairs):
  GEN  generic ranker  HAND_FEATS+ctx_emb           [current best, reproduce §150 0.5710]
  FAM  family-conditional: 3 rankers routed by behavior_family
Also FAM+GEN blend (rank-avg) as a safety. Ship the best if it clears 0.5710 (evidence is phase-immune).
Run:  python -m anchor_repro.evidence_familycond
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

def gen_ranker(dh, feats):
    dh=dh.copy(); dh["hand_score"]=0.0; folds=sorted(dh["fold"].unique())
    for f in folds:
        tr_pos=((dh["fold"]!=f)&(dh["label"]==1)).to_numpy(); te=(dh["fold"]==f).to_numpy()
        models=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr_pos,feats],dh.loc[tr_pos,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
        dh.loc[te,"hand_score"]=np.mean([m.predict(dh.loc[te,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def fam_ranker(dh, feats):
    """Per-family rankers, routed by the pair's behavior_family. For a held-out fold, each pair's hands
    are scored by the ranker trained (on the other folds) for THAT pair's family."""
    dh=dh.copy(); dh["hand_score"]=0.0; folds=sorted(dh["fold"].unique())
    for f in folds:
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            # train on this family's positive-pair hands from the OTHER folds
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            models=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            # score held-out pairs OF THIS FAMILY
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def score(dh, hs, tag):
    d=dh.copy(); d["hand_score"]=hs; pos=d[d["label"]==1]
    overall=map5(pos); fam={fm:map5(pos[pos["behavior_family"]==fm]) for fm in TARGET_BEHAVIORS}
    log(f"[{tag}] MAP@5 = {overall:.4f}  per family: {{ {', '.join(f'{k}: {v:.4f}' for k,v in fam.items())} }}")
    return overall

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    FEATS=HAND_FEATS+CTXCOLS
    log("=== family-conditional evidence (current best generic = 0.5710) ===")
    gen=gen_ranker(dh,FEATS); score(dh,gen,"GEN generic (HAND_FEATS+ctx)")
    fam=fam_ranker(dh,FEATS); score(dh,fam,"FAM family-conditional")
    # rank-avg blend (within pair) as a safety
    d=dh.copy(); d["g"]=gen; d["f"]=fam
    d["gr"]=d.groupby("pair_id")["g"].rank(pct=True); d["fr"]=d.groupby("pair_id")["f"].rank(pct=True)
    score(dh,(d["gr"]+d["fr"]).to_numpy(),"BLEND rank-avg(gen,fam)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
