"""§214 — EYEBALL the actual evidence hands. Stop testing top-down hypotheses; look bottom-up at what the
host LABELED vs what our ranker PICKS. For pairs we miss, dump the raw record of the TRUE evidence hand we
ranked LOW next to the DECOY we ranked #1. What is literally different?

For each of a few 'miss' pairs (true evidence ranked >5 by base ranker), print for BOTH the missed-true hand
and our top decoy: pot, both members' actions (from actions.parquet), both hole cards, board, net_bb, who won,
street reached. Read them like a human reviewing a collusion report.
Run:  python -m anchor_repro.evidence_eyeball
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, TARGET_BEHAVIORS
from anchor_repro.seq_ctx_evidence import CTXCOLS
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

def base_scores(dh):
    dh=dh.copy(); dh["hs"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            feats=HAND_FEATS+CTXCOLS
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
            tgt=te&(dh["behavior_family"]==fam).to_numpy()
            dh.loc[tgt,"hs"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    return dh["hs"].to_numpy()

def dump_hand(hid, p1, p2, acts, seats, hands):
    a=acts[acts.hand_id==hid].sort_values("action_no")
    h=hands[hands.hand_id==hid]
    board=h["board_cards"].iloc[0] if len(h) else "?"
    pot=h["final_pot"].iloc[0] if len(h) else "?"
    def cards(pid):
        r=seats[(seats.hand_id==hid)&(seats.player_id==pid)]
        if len(r)==0: return "??"
        return f"{r.hole_card_1.iloc[0]} {r.hole_card_2.iloc[0]} net={r.net_chips.iloc[0]} {'SD' if r.went_to_showdown.iloc[0] else ('FOLD' if r.folded.iloc[0] else '-')}"
    print(f"    hand {hid}  board=[{board}] final_pot={pot}")
    print(f"      P1 {p1[:8]}: {cards(p1)}")
    print(f"      P2 {p2[:8]}: {cards(p2)}")
    seq=[]
    for r in a.itertuples():
        who="P1" if r.player_id==p1 else ("P2" if r.player_id==p2 else "..")
        seq.append(f"{r.street[:2]}:{who}:{r.action}{('' if r.amount==0 else '/'+str(r.amount))}")
    print(f"      actions: {' '.join(seq)[:300]}")

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    dh["hs"]=base_scores(dh)
    if "player_1" not in dh.columns:
        lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2"]).to_pandas()
        dh=dh.merge(lab,on="pair_id",how="left")
    pos=dh[dh["label"]==1].copy()
    pos=pos.sort_values(["pair_id","hs"],ascending=[True,False])
    pos["rk"]=pos.groupby("pair_id").cumcount()+1
    # miss pairs: true evidence ranked >5, and we have a clear decoy at rank 1 that's NOT evidence
    hset=set(pos["hand_id"].unique().tolist())
    acts=pl.read_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(list(hset))).to_pandas()
    seats=pl.read_parquet(D/"seats.parquet").filter(pl.col("hand_id").is_in(list(hset))).to_pandas()
    hands=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(list(hset))).to_pandas()
    shown=0
    for pid,g in pos.groupby("pair_id"):
        miss=g[(g["is_evidence"])&(g["rk"]>5)]; decoy=g[(~g["is_evidence"])&(g["rk"]==1)]
        if len(miss)==0 or len(decoy)==0: continue
        fam=g["behavior_family"].iloc[0]; p1=g["player_1"].iloc[0]; p2=g["player_2"].iloc[0]
        print(f"\n===== PAIR {pid[:10]} family={fam} =====")
        print(f"  >> MISSED true-evidence (we ranked #{int(miss['rk'].iloc[0])}):")
        dump_hand(miss["hand_id"].iloc[0],p1,p2,acts,seats,hands)
        print(f"  >> DECOY we ranked #1 (NOT evidence):")
        dump_hand(decoy["hand_id"].iloc[0],p1,p2,acts,seats,hands)
        shown+=1
        if shown>=6: break
    return 0

if __name__=="__main__":
    raise SystemExit(main())
