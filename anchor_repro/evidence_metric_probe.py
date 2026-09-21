"""IS OUR LOCAL MAP@5 THE RIGHT OBJECTIVE? (user question: is MAP@5 a problem we've made for ourselves?)

The host ground truth (development_evidence.csv) has evidence_rank 1..5 — a RANKED truth, not just a set.
Our local map5 uses is_evidence as a BINARY set (order among true hands ignored). If the real metric is
RANK-AWARE, our proxy is slightly wrong, and features good at ORDERING (like equity by surrender severity)
could help the real metric while being flat on our set-based proxy — which would exactly explain the
'real signal, zero set-MAP lift' paradox (§176).

This probe computes, on the SAME OOF hand scores (base FAM vs FAM+equity), FOUR evidence metrics:
  SET   set-based AP@5 (our current map5; relevance = is_evidence, binary)               [what we optimize]
  RW    rank-weighted: reward putting HIGH-truth-rank hands EARLY (gain = (6-truth_rank), DCG-style)
  NDCG  normalized DCG@5 with gain=(6-truth_rank)                                        [rank-aware, bounded]
  KEND  Kendall-tau agreement between our order of the true hands and their evidence_rank [pure ordering]
If equity lifts RW/NDCG/KEND while flat on SET -> the metric IS the problem and we should optimize ordering.
If equity is flat on ALL -> genuinely exhausted (redundant even for ordering).
Run:  python -m anchor_repro.evidence_metric_probe
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.evidence_equity import build_perhand, EQEV_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"; DATA=Path("data/poker")
SEED=42

def build_dh_ranked():
    """Same as seq_evidence.build_dh but KEEP evidence_rank (0 = not evidence)."""
    dev_pairs=pl.read_parquet(STEP3/"dev_pairs.parquet")
    dev_ev=pl.read_csv(DATA/"development_evidence.csv")
    DEV_HF=sorted(STEP3.glob("dev_hand_features_*.parquet"))
    TABLES=sorted(dev_pairs["table_id"].unique().to_list()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
    lab=set(dev_pairs.filter(pl.col("is_labeled"))["pair_id"].to_list())
    d=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(lab))).collect() for f in DEV_HF])
    d=d.join(dev_pairs.select(["pair_id","label","behavior_family","is_labeled"]),on="pair_id")
    d=d.with_columns(pl.col("table_id").replace_strict(TF,return_dtype=pl.Int8).alias("fold"))
    d=d.join(dev_ev.select(["pair_id","hand_id","evidence_rank"]),on=["pair_id","hand_id"],how="left")
    d=d.with_columns(pl.col("evidence_rank").fill_null(0).alias("evidence_rank"),
                     (pl.col("evidence_rank").fill_null(0)>0).alias("is_evidence"))
    assert d["is_evidence"].sum()==1817
    return d

# ---- metrics (all @5, per positive pair, averaged) ----
def _order(g):  # host tie-break: score desc, pot_bb desc, hand_id asc
    return g.sort_values(["hand_score","pot_bb","hand_id"],ascending=[False,False,True],kind="mergesort")

def m_set(pos):
    vals=[]
    for _,g in pos.groupby("pair_id",sort=False):
        g=_order(g); rel=(g["evidence_rank"].to_numpy()>0).astype(int); nr=int(rel.sum())
        if nr==0: continue
        top=rel[:5]; hits=np.cumsum(top)
        vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(nr,5)))
    return float(np.mean(vals))

def m_ndcg(pos):
    vals=[]
    for _,g in pos.groupby("pair_id",sort=False):
        g=_order(g); tr=g["evidence_rank"].to_numpy()[:5]
        gain=np.where(tr>0,6-tr,0.0)  # rank1 -> 5, rank5 -> 1, non-evidence -> 0
        disc=1/np.log2(np.arange(2,2+len(gain)))
        dcg=float(np.sum(gain*disc))
        ideal=np.sort(np.where(g["evidence_rank"].to_numpy()>0,6-g["evidence_rank"].to_numpy(),0.0))[::-1][:5]
        idcg=float(np.sum(ideal*disc[:len(ideal)]))
        if idcg>0: vals.append(dcg/idcg)
    return float(np.mean(vals))

def m_rw(pos):
    # rank-weighted precision@5: each retrieved true hand contributes (6-truth_rank)/position, normalized
    vals=[]
    for _,g in pos.groupby("pair_id",sort=False):
        g=_order(g).head(5); tr=g["evidence_rank"].to_numpy()
        got=np.where(tr>0,(6-tr)/np.arange(1,len(tr)+1),0.0).sum()
        best=np.sort((6-np.arange(1,6)))/np.arange(1,6); best=best.sum()
        vals.append(float(got/best))
    return float(np.mean(vals))

def m_kendall(pos):
    from scipy.stats import kendalltau
    vals=[]
    for _,g in pos.groupby("pair_id",sort=False):
        gt=g[g["evidence_rank"]>0]
        if len(gt)<2: continue
        gt=gt.sort_values(["hand_score","pot_bb","hand_id"],ascending=[False,False,True],kind="mergesort")
        tau=kendalltau(np.arange(len(gt)), gt["evidence_rank"].to_numpy()).correlation
        if not np.isnan(tau): vals.append(-tau)  # our order asc idx vs truth rank; negate so higher=better
    return float(np.mean(vals))

def fam_ranker(dh, feats, seeds=None):
    seeds=seeds or HAND_SEEDS; dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],(dh.loc[tr,"evidence_rank"]>0).astype(int)),num_boost_round=HAND_ROUNDS) for sd in seeds]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    return dh["hand_score"].to_numpy()

def evalall(dh, score, tag):
    d=dh.copy(); d["hand_score"]=score; pos=d[d["label"]==1]
    s,n,r,k=m_set(pos),m_ndcg(pos),m_rw(pos),m_kendall(pos)
    log(f"[{tag:16s}] SET {s:.4f} | NDCG {n:.4f} | RW {r:.4f} | KEND {k:.4f}")
    return np.array([s,n,r,k])

def main():
    dh=build_dh_ranked()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet"); dh=dh.join(emb,on=["pair_id","hand_idx"],how="left")
    per=build_perhand("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=set(dh["pair_id"].to_list()))
    dh=dh.join(per,on=["pair_id","hand_id"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    for c in EQEV_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    FEATS=HAND_FEATS+CTXCOLS
    log("=== base vs +equity across 4 evidence metrics (transfer+soft eq) ===")
    TS=("directed_transfer","soft_play")
    b=evalall(dh, fam_ranker(dh,FEATS), "FAM base")
    # equity only on transfer+soft: emulate by scoring per-family
    dh2=dh.copy(); dh2["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            feats=FEATS+EQEV_FEATS if fam in TS else FEATS
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],(dh.loc[tr,"evidence_rank"]>0).astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            dh2.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    e=evalall(dh2, dh2["hand_score"].to_numpy(), "FAM+EQ txfr+soft")
    for i,nm in enumerate(["SET","NDCG","RW","KEND"]):
        log(f"   delta {nm:5s} = {e[i]-b[i]:+.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
