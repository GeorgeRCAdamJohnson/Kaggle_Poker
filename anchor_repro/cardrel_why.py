"""WHY doesn't cr_board_hit_gap add, despite +0.22 ev-vs-nonev separation? Test the actual mechanism instead
of asserting 'redundant'. Three concrete measurements:
 1. REDUNDANCY: is cr_board_hit_gap reconstructable from HAND_FEATS? (regress it on HAND_FEATS, R2). High R2
    => the ranker already has it => adding it is redundant (my claim). Low R2 => I was hand-waving.
 2. COVERAGE: cr needs BOTH hole cards (only ~20% of hands per §probe). On hands where cr is UNDEFINED
    (cards hidden), what value did it get (0) and does that 0 collide with real-0 hands -> destroys ranking?
 3. WITHIN-PAIR VARIANCE: does cr_board_hit_gap even vary within a pair's hands, or is it near-constant
    (can't reorder if it doesn't vary)? Report per-pair std.
Run:  python -m anchor_repro.cardrel_why
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from anchor_repro.seq_evidence import build_dh, HAND_FEATS
from anchor_repro.evidence_cardrel_gate import add_cardrel
D=Path("data/poker")

def main():
    dh=build_dh().to_pandas()
    dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    dh=add_cardrel(dh)
    pos=dh[dh["label"]==1].copy()
    # coverage
    both=(pos["cr_a_hit"]+pos["cr_b_hit"]+pos["cr_board_hit_gap"])
    # a hand has cards if we could compute a hit; approximate "cards present" via seats join done in add_cardrel
    seats=pl.read_parquet(D/"seats.parquet").select(["hand_id","player_id"]).to_pandas()
    present=set(zip(seats["hand_id"],seats["player_id"]))
    has=[( (t.hand_id,t.player_1) in present and (t.hand_id,t.player_2) in present) for t in pos.itertuples()]
    pos["cards_present"]=has
    print(f"=== 2. COVERAGE ===")
    print(f"  positive-pair hands: {len(pos)} | both hole cards present: {int(pos['cards_present'].sum())} ({pos['cards_present'].mean()*100:.1f}%)")
    print(f"  => {100-pos['cards_present'].mean()*100:.0f}% of hands get cr=0 by DEFAULT (cards hidden), not by real board-miss")
    ev=pos[pos["is_evidence"]]
    print(f"  evidence hands with cards present: {ev['cards_present'].mean()*100:.1f}%  (if high, cr only informative on a biased slice)")

    print(f"\n=== 3. WITHIN-PAIR VARIANCE (can it even reorder?) ===")
    g=pos.groupby("pair_id")["cr_board_hit_gap"].std()
    print(f"  per-pair std of cr_board_hit_gap: mean {g.mean():.3f} median {g.median():.3f} | pairs with ~0 variance: {(g<0.3).mean()*100:.0f}%")

    print(f"\n=== 1. REDUNDANCY: reconstruct cr_board_hit_gap from HAND_FEATS ===")
    ps=pos.sample(n=min(30000,len(pos)),random_state=42)
    X=ps[HAND_FEATS].fillna(0).to_numpy(); y=ps["cr_board_hit_gap"].to_numpy()
    r2r=r2_score(y,cross_val_predict(Ridge(alpha=10),X,y,cv=3))
    r2g=r2_score(y,cross_val_predict(HistGradientBoostingRegressor(max_depth=4,max_iter=120),X,y,cv=3))
    print(f"  (30k subsample) ridge R2 {r2r:+.4f} | gbm R2 {r2g:+.4f}")
    print(f"  => if HIGH (>0.5): HAND_FEATS already encode board-hit-gap = REDUNDANT (claim confirmed).")
    print(f"     if LOW (<0.3): NOT redundant; the non-add is COVERAGE/variance, not redundancy (I was wrong).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
