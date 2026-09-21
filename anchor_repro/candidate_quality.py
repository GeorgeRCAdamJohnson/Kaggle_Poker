"""ARE OUR TOP CANDIDATES ACTUALLY RIGHT? (user: consensus fails because our evidence isn't strong enough
to have the RIGHT candidates in the top — recall@10=0.90 may just be a broad net CONTAINING the truth, not
CONFIDENTLY identifying it). Diagnostic before any build.

Uses the cached stage-1 LGB scores (family-conditional pointwise, the banked 0.5989 ranker). All on the 372
labelled dev pairs (known truth), within-pair (phase-immune).

P1 CONFIDENCE: for each pair, the SCORE GAP between true evidence hands and the decoys around them. Strong
   candidate set = true hands score clearly above decoys. Weak = interleaved (scorer not separating, just
   not excluding). Report separation (AUC of score at distinguishing true vs decoy WITHIN each pair) and the
   rank distribution of true hands.
P2 WHERE: is weakness uniform or concentrated by family? (§181: isolation worst 0.5165). Per-family within-
   pair separation + true-hand rank distribution.
P3 WHAT'S DIFFERENT about the true hands we RANK LOW (rank>5) vs the DECOYS we rank ABOVE them (rank<=5,
   not evidence)? Compare raw HAND_FEATS means: if a systematic difference exists -> a missing feature; if
   indistinguishable -> label depends on info we don't have.
Run:  python -m anchor_repro.candidate_quality
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.seq_evidence import HAND_FEATS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.evidence_rerank import build_dh_ranked
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
S1_CACHE=SEQ/"rerank_stage1_lgb.parquet"

def within_pair_sep_auc(pos):
    """Mean over pairs of the AUC that the score separates true-evidence from decoy hands WITHIN that pair."""
    aucs=[]
    for _,g in pos.groupby("pair_id",sort=False):
        y=g["is_evidence"].to_numpy().astype(int)
        if y.sum()==0 or y.sum()==len(y): continue
        try: aucs.append(roc_auc_score(y,g["s1"].to_numpy()))
        except Exception: pass
    return float(np.mean(aucs)), len(aucs)

def true_rank_dist(pos):
    """Within-pair rank (1=best) of each TRUE evidence hand -> distribution."""
    ranks=[]
    for _,g in pos.groupby("pair_id",sort=False):
        g=g.sort_values(["s1","pot_bb","hand_id"],ascending=[False,False,True],kind="mergesort").reset_index(drop=True)
        r=np.where(g["is_evidence"].to_numpy())[0]+1
        ranks.extend(r.tolist())
    ranks=np.array(ranks)
    return ranks

def main():
    dh=build_dh_ranked()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet"); dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    dh["evidence_rank"]=dh["evidence_rank"].fillna(0).astype(int); dh["is_evidence"]=dh["evidence_rank"]>0
    c=pl.read_parquet(S1_CACHE).to_pandas()
    dh=dh.merge(c,on=["pair_id","hand_id"],how="left")
    assert not dh["s1"].isna().any(), "stage-1 cache missing pairs"
    pos=dh[dh["label"]==1].copy()
    log(f"MAP@5 (from cache) = {map5(pos.rename(columns={'s1':'hand_score'})):.4f}")

    log("=== P1 CONFIDENCE: within-pair separation of TRUE vs decoy ===")
    auc,n=within_pair_sep_auc(pos)
    log(f"  mean within-pair separation AUC = {auc:.4f}  (1.0=perfect, 0.5=random)  over {n} pairs")
    rk=true_rank_dist(pos)
    for k in [1,3,5,10,20]:
        log(f"  true evidence hands at rank<= {k:<3d}: {(rk<=k).mean()*100:5.1f}%")
    log(f"  true-hand rank: mean {rk.mean():.1f} median {int(np.median(rk))} p90 {int(np.percentile(rk,90))} max {rk.max()}")

    log("=== P2 WHERE: per-family separation + true-hand rank ===")
    for fam in TARGET_BEHAVIORS:
        p=pos[pos["behavior_family"]==fam]
        a,nn=within_pair_sep_auc(p); r=true_rank_dist(p)
        log(f"  {fam:22s} sepAUC {a:.4f}  true@rank<=5 {(r<=5).mean()*100:4.1f}%  meanrank {r.mean():.1f}  MAP@5 {map5(p.rename(columns={'s1':'hand_score'})):.4f}")

    log("=== P3 WHAT: true hands we rank LOW (rank>5) vs decoys we rank HIGH (rank<=5, not evidence) ===")
    pos=pos.sort_values(["pair_id","s1","pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    pos["rk"]=pos.groupby("pair_id").cumcount()+1
    miss=pos[(pos["is_evidence"])&(pos["rk"]>5)]          # true but buried
    decoy=pos[(~pos["is_evidence"])&(pos["rk"]<=5)]        # false but promoted
    log(f"  buried-true n={len(miss)}, promoted-decoy n={len(decoy)}")
    diffs=[]
    for c_ in HAND_FEATS:
        mu_m,mu_d=miss[c_].mean(),decoy[c_].mean()
        sd=pos[c_].std()+1e-9
        diffs.append((c_,(mu_m-mu_d)/sd,mu_m,mu_d))
    diffs.sort(key=lambda x:-abs(x[1]))
    log("  top 15 features by |standardized diff| (buried-true minus promoted-decoy):")
    for c_,z,mm,md in diffs[:15]:
        log(f"    {z:+.3f}  {c_:28s} buried={mm:.3f} decoy={md:.3f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
