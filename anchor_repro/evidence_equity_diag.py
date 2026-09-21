"""WHY is per-hand equity-evidence unstable? (user hunch: real signal, constructed badly). §175 follow-up.

Four probes on the DEV labelled per-hand frame (equity is phase-immune within-pair, so dev==LB for evidence):
  P1 MISS-ANALYSIS: on true-evidence hands the base FAM ranker puts OUTSIDE top-5 (the ones we miss),
     is equity systematically higher than on non-evidence hands of the same pairs? If yes -> real signal
     on the hands we currently fail, and the full-feature ranker is diluting it.
  P2 NO-SHOWDOWN slice: the original thesis = equity sees mid-hand folds that showdown-rank can't.
     Compare eq_surr_hand / eq_winner_was_ahead on evidence vs non-evidence, split by went-to-showdown.
  P3 FEATURE-COUNT: is instability from dumping 9 correlated eq feats into the tree? Test minimal sets
     (surr only; surr+winner_ahead) vs the full 9, on transfer+soft_play, default seed.
  P4 STABILITY of the best minimal set across 5 seeds (vs the full-set mean -0.0002 / 2-of-5).
Run:  python -m anchor_repro.evidence_equity_diag
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
from anchor_repro.evidence_equity import build_perhand, EQEV_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"

def fam_ranker(dh, base_feats, eq_feats=(), eq_families=(), seeds=None):
    seeds=seeds or HAND_SEEDS
    dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            feats=base_feats+list(eq_feats) if fam in eq_families else base_feats
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            models=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in seeds]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def rank_within_pair(dh, score):
    d=dh.copy(); d["hand_score"]=score
    d=d.sort_values(["pair_id","hand_score","pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    d["rk"]=d.groupby("pair_id",sort=False).cumcount()  # 0-based rank within pair
    return d

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left")
    per=build_perhand("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=set(dh["pair_id"].to_list()))
    dh=dh.join(per,on=["pair_id","hand_id"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    for c in EQEV_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    FEATS=HAND_FEATS+CTXCOLS
    TS=("directed_transfer","soft_play")

    # base ranker once (default seed) for the miss analysis
    base_score=fam_ranker(dh,FEATS)
    ranked=rank_within_pair(dh, base_score)
    pos=ranked[ranked["label"]==1]

    log("=== P1 MISS-ANALYSIS: equity on evidence hands the base ranker MISSES (rank>=5) vs non-evidence ===")
    # limit to transfer+soft_play pairs (where equity is meant to help)
    ts=pos[pos["behavior_family"].isin(TS)]
    ev_hit =ts[(ts["is_evidence"]) & (ts["rk"]<5)]
    ev_miss=ts[(ts["is_evidence"]) & (ts["rk"]>=5)]
    nonev  =ts[~ts["is_evidence"]]
    for c in ["eq_surr_hand","eq_winner_was_ahead","eq_loser_gave_edge","eq_checkdown_hi","eq_max_pf"]:
        log(f"  {c:20s} ev_hit={ev_hit[c].mean():.3f}  ev_MISS={ev_miss[c].mean():.3f}  nonev={nonev[c].mean():.3f}")
    log(f"  (n: ev_hit={len(ev_hit)}, ev_miss={len(ev_miss)}, nonev={len(nonev)})")

    log("=== P2 NO-SHOWDOWN slice (transfer+soft_play): equity ev vs non-ev, split by showdown ===")
    ts_all=dh[dh["behavior_family"].isin(TS) & (dh["label"]==1)].copy()
    ts_all["sd"]=((ts_all["p1_went_to_showdown"]>0)|(ts_all["p2_went_to_showdown"]>0)).astype(int) if "p1_went_to_showdown" in ts_all.columns else 0
    # p*_went_to_showdown not in dh; derive showdown from players_at_showdown in HAND_FEATS
    ts_all["sd"]=(ts_all["players_at_showdown"]>0).astype(int)
    for sd,lab in [(0,"NO-showdown"),(1,"showdown")]:
        sub=ts_all[ts_all["sd"]==sd]
        for c in ["eq_surr_hand","eq_winner_was_ahead"]:
            e=sub[sub["is_evidence"]][c].mean(); nn=sub[~sub["is_evidence"]][c].mean()
            log(f"  [{lab:11s}] {c:20s} ev={e:.3f} nonev={nn:.3f} sep={e-nn:+.3f}  (n_ev={int(sub['is_evidence'].sum())})")

    log("=== P3 FEATURE-COUNT sensitivity (default seed, eq on transfer+soft) ===")
    o0=map5(pos)  # base already computed
    log(f"  FAM base                         MAP@5 {o0:.4f}")
    for name,ef in [("surr only",["eq_surr_hand"]),
                    ("surr+winner_ahead",["eq_surr_hand","eq_winner_was_ahead"]),
                    ("surr+winner+loser",["eq_surr_hand","eq_winner_was_ahead","eq_loser_gave_edge"]),
                    ("full 9",EQEV_FEATS)]:
        s=fam_ranker(dh,FEATS,eq_feats=ef,eq_families=TS)
        m=map5(dh.assign(hand_score=s)[dh["label"]==1])
        log(f"  +{name:28s} MAP@5 {m:.4f}  delta {m-o0:+.4f}")

    log("=== P4 STABILITY of best-minimal set across 5 seeds ===")
    best=["eq_surr_hand","eq_winner_was_ahead"]  # updated below if P3 says otherwise (manual read)
    for label,ef in [("surr+winner_ahead",best),("surr only",["eq_surr_hand"])]:
        ds=[]
        for s in [42,101,202,303,404]:
            sd=[s,s+7]
            oa=map5(dh.assign(hand_score=fam_ranker(dh,FEATS,seeds=sd))[dh["label"]==1])
            ob=map5(dh.assign(hand_score=fam_ranker(dh,FEATS,eq_feats=ef,eq_families=TS,seeds=sd))[dh["label"]==1])
            ds.append(ob-oa)
        log(f"  [{label:18s}] deltas {[round(x,4) for x in ds]}  mean {np.mean(ds):+.4f}  (>0 in {sum(x>0 for x in ds)}/5)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
