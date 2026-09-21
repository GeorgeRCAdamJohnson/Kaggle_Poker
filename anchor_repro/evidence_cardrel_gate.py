"""§212 — Gate cr_board_hit_gap (the one live card-relationship evidence signal, §probe: ev 0.53 vs nonev
0.31, +0.22 sep, concentrated in transfer/soft). Does it ADD to the family-conditional evidence ranker
(current best 0.5989)? board_hit_gap on transfer+soft rankers (near-zero for isolation, leave it out).
Seed-robustness (5 seeds) — the discipline that caught the §175 false positive. Within-pair = phase-immune,
local==LB. Ship if mean delta clearly >0 across seeds.
Run:  python -m anchor_repro.evidence_cardrel_gate
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
HAND_SEEDS_ORIG=list(HAND_SEEDS); TS=("directed_transfer","soft_play")
RANKS="23456789TJQKA"; RMAP={c:i for i,c in enumerate(RANKS)}
def prank(card): return RMAP.get(card[0],-1) if isinstance(card,str) and len(card)>=2 else -1
CR=["cr_board_hit_gap","cr_a_hit","cr_b_hit"]

def add_cardrel(dh):
    if "player_1" not in dh.columns:
        lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2"]).to_pandas()
        d=dh.merge(lab,on="pair_id",how="left")
    else:
        d=dh.copy()
    hset=set(dh["hand_id"].unique().tolist())
    seats=pl.read_parquet(D/"seats.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","player_id","hole_card_1","hole_card_2"]).to_pandas()
    board=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","board_cards"]).to_pandas()
    sidx={(r.hand_id,r.player_id):(r.hole_card_1,r.hole_card_2) for r in seats.itertuples()}
    bmap={r.hand_id:r.board_cards for r in board.itertuples()}
    ghg=[]; ah=[]; bh=[]
    for t in d.itertuples():
        ca=sidx.get((t.hand_id,t.player_1)); cb=sidx.get((t.hand_id,t.player_2))
        if ca is None or cb is None: ghg.append(0); ah.append(0); bh.append(0); continue
        ar={prank(ca[0]),prank(ca[1])}-{-1}; br={prank(cb[0]),prank(cb[1])}-{-1}
        bset={prank(x) for x in str(bmap.get(t.hand_id,"")).split() if x}-{-1}
        a=len(ar&bset); b=len(br&bset); ah.append(a); bh.append(b); ghg.append(abs(a-b))
    d["cr_board_hit_gap"]=np.array(ghg,float); d["cr_a_hit"]=np.array(ah,float); d["cr_b_hit"]=np.array(bh,float)
    return d

def fam_ranker(dh, feats, cr_families=()):
    dh=dh.copy(); dh["hand_score"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            ff=feats+CR if fam in cr_families else feats
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,ff],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
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
    dh=add_cardrel(dh)
    for c in CR: dh[c]=dh[c].astype("float32")
    FEATS=HAND_FEATS+CTXCOLS
    b=m5(dh,fam_ranker(dh,FEATS)); log(f"FAM base MAP@5 {b:.4f}")
    e=m5(dh,fam_ranker(dh,FEATS,cr_families=TS)); log(f"FAM+cardrel(transfer+soft) MAP@5 {e:.4f}  delta {e-b:+.4f}")
    log("--- seed robustness (5 seeds) ---")
    ds=[e-b]
    for s in [101,202,303,404]:
        globals()["HAND_SEEDS"]=[s,s+7]
        bb=m5(dh,fam_ranker(dh,FEATS)); ee=m5(dh,fam_ranker(dh,FEATS,cr_families=TS)); ds.append(ee-bb); log(f"  seed{s}: base {bb:.4f} +cr {ee:.4f} d {ee-bb:+.4f}")
    globals()["HAND_SEEDS"]=HAND_SEEDS_ORIG
    log(f"delta over 5 seeds: mean {np.mean(ds):+.4f} min {min(ds):+.4f} max {max(ds):+.4f} (>0 in {sum(x>0 for x in ds)}/5)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
