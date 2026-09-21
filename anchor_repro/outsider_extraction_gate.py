"""§214 — GATE the outsider-extraction features (the eyeball breakthrough). ev-vs-nonev separations are the
biggest we've seen (ox_pot_to_winner +35, ox_winner_net +16, single-feat MAP@5 ~0.089 vs prior best 0.045).
Does it ADD to the family-conditional evidence ranker (best 0.5989)? Seed-robustness 5 seeds. Within-pair =
phase-immune, local==LB. This is the decisive test of the first real evidence breakthrough.
Run:  python -m anchor_repro.outsider_extraction_gate
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
HAND_SEEDS_ORIG=list(HAND_SEEDS)
OX=["ox_winner_net","ox_pot_to_winner","ox_joint_take","ox_from_outsider"]

def add_ox(dh):
    if "player_1" not in dh.columns:
        lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2"]).to_pandas()
        dh=dh.merge(lab,on="pair_id",how="left")
    hset=set(dh["hand_id"].unique().tolist())
    seats=pl.read_parquet(D/"seats.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","player_id","net_chips","folded"]).to_pandas()
    hands=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","final_pot","big_blind"]).to_pandas()
    net={(r.hand_id,r.player_id):(r.net_chips,r.folded) for r in seats.itertuples()}
    pot={r.hand_id:(r.final_pot,r.big_blind) for r in hands.itertuples()}
    wn=[];ptw=[];jt=[];fo=[]
    for t in dh.itertuples():
        a=net.get((t.hand_id,t.player_1)); b=net.get((t.hand_id,t.player_2)); pb=pot.get(t.hand_id,(0,1))
        if a is None or b is None: wn.append(0);ptw.append(0);jt.append(0);fo.append(0); continue
        an,_=a; bn,_=b; bb=max(pb[1],1); P=pb[0]/bb
        wn.append(max(an,bn)/bb); ptw.append(P if max(an,bn)>0 else 0.0); jt.append((an+bn)/bb)
        fo.append((max(an,bn)+min(an,bn))/bb if max(an,bn)>0 else 0.0)
    dh["ox_winner_net"]=np.array(wn,float); dh["ox_pot_to_winner"]=np.array(ptw,float)
    dh["ox_joint_take"]=np.array(jt,float); dh["ox_from_outsider"]=np.array(fo,float)
    return dh

def fam_ranker(dh, feats, ox_families=()):
    dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            ff=feats+OX if fam in ox_families else feats
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
    dh=add_ox(dh)
    for c in OX: dh[c]=dh[c].astype("float32")
    FEATS=HAND_FEATS+CTXCOLS; ALL3=TARGET_BEHAVIORS
    b=m5(dh,fam_ranker(dh,FEATS)); log(f"FAM base MAP@5 {b:.4f}")
    e=m5(dh,fam_ranker(dh,FEATS,ox_families=ALL3)); log(f"FAM+outsider-extraction (all 3) MAP@5 {e:.4f}  delta {e-b:+.4f}")
    log("--- seed robustness (5 seeds, all-3-families) ---")
    ds=[e-b]
    for s in [101,202,303,404]:
        globals()["HAND_SEEDS"]=[s,s+7]
        bb=m5(dh,fam_ranker(dh,FEATS)); ee=m5(dh,fam_ranker(dh,FEATS,ox_families=ALL3)); ds.append(ee-bb); log(f"  seed{s}: base {bb:.4f} +ox {ee:.4f} d {ee-bb:+.4f}")
    globals()["HAND_SEEDS"]=HAND_SEEDS_ORIG
    log(f"delta over 5 seeds: mean {np.mean(ds):+.4f} min {min(ds):+.4f} max {max(ds):+.4f} (>0 in {sum(x>0 for x in ds)}/5)")
    log("SHIP if mean clearly >0 across seeds (evidence phase-immune -> local==LB).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
