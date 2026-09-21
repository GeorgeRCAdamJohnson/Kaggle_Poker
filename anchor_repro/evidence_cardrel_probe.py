"""§212 — CARD-RELATIONSHIP evidence probe. The one raw signal never tested with today's discipline: the
relationship between the TWO colluders' actual hole cards within a hand (we only ever used derived per-player
strength, never the cross-player card relationship). This VARIES hand-to-hand so it's a legitimate EVIDENCE
candidate (evidence = the +0.08 lever, §211). Nothing in the §6447 triage table covers it.

For the 1,817 TRUE evidence hands vs non-evidence shared hands of the SAME 372 positive pairs:
  cr_share_rank : the two hold a matching rank (card-removal / one blocks the other)
  cr_share_suit : share a suit
  cr_dominated  : shared top rank, different second (AK vs AQ) -> soft-play/transfer setup
  cr_board_hit_gap : |A's ranks on board - B's ranks on board| (one hits, one whiffs = isolation)
  cr_max_gap    : |maxrank(A) - maxrank(B)| (strength asymmetry from raw cards)
Report ev-vs-nonev mean separation + single-feature within-pair MAP@5 (beat magnitude 0.18 / ci_peak_eq_gap
0.24 to be interesting) + per family. Sources: seats.parquet hole_card_1/2, hands.parquet board_cards.
Run:  python -m anchor_repro.evidence_cardrel_probe
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
D=Path("data/poker")
RANKS="23456789TJQKA"; RMAP={c:i for i,c in enumerate(RANKS)}

def parse(card):
    if not isinstance(card,str) or len(card)<2: return (-1,"")
    return (RMAP.get(card[0],-1), card[1])

def main():
    from anchor_repro.seq_evidence import build_dh, map5, TARGET_BEHAVIORS
    dh=build_dh().select(["pair_id","hand_id","is_evidence","pot_bb","label","behavior_family"]).to_pandas()
    lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2"]).to_pandas()
    dh=dh.merge(lab,on="pair_id",how="left")
    hset=set(dh["hand_id"].unique().tolist())
    seats=pl.read_parquet(D/"seats.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","player_id","hole_card_1","hole_card_2"]).to_pandas()
    board=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","board_cards"]).to_pandas()
    sidx={(r.hand_id,r.player_id):(r.hole_card_1,r.hole_card_2) for r in seats.itertuples()}
    bmap={r.hand_id:r.board_cards for r in board.itertuples()}
    rows=[]
    for t in dh.itertuples():
        ca=sidx.get((t.hand_id,t.player_1)); cb=sidx.get((t.hand_id,t.player_2))
        if ca is None or cb is None: rows.append((0,0,0,0,0)); continue
        a1,a2=parse(ca[0]),parse(ca[1]); b1,b2=parse(cb[0]),parse(cb[1])
        ar={a1[0],a2[0]}-{-1}; br={b1[0],b2[0]}-{-1}; asu={a1[1],a2[1]}-{""}; bsu={b1[1],b2[1]}-{""}
        share_rank=int(len(ar & br)>0); share_suit=int(len(asu & bsu)>0)
        amax=max([x for x in ar]+[-1]); bmax=max([x for x in br]+[-1])
        dominated=int(len(ar & br)>0 and amax!=bmax)
        bc=[parse(x)[0] for x in str(bmap.get(t.hand_id,"")).split() if x]; bset=set(bc)-{-1}
        board_hit_gap=abs(len(ar & bset)-len(br & bset))
        max_gap=abs(amax-bmax) if amax>=0 and bmax>=0 else 0
        rows.append((share_rank,share_suit,dominated,board_hit_gap,max_gap))
    F=pd.DataFrame(rows,columns=["cr_share_rank","cr_share_suit","cr_dominated","cr_board_hit_gap","cr_max_gap"])
    for c in F.columns: dh[c]=F[c].values.astype(float)
    matched=(dh[["cr_share_rank","cr_share_suit","cr_dominated","cr_board_hit_gap","cr_max_gap"]].abs().sum(axis=1)>=0).sum()
    have_cards=sum(1 for t in dh.itertuples() if (t.hand_id,t.player_1) in sidx and (t.hand_id,t.player_2) in sidx)
    print(f"labeled hands {len(dh):,}; evidence {int(dh['is_evidence'].sum())}; both-hole-cards present {have_cards:,}")
    pos=dh[dh["label"]==1].copy(); ev=pos[pos["is_evidence"]]; nv=pos[~pos["is_evidence"]]
    print("\n=== card-relationship: ev vs non-ev mean + single-feature within-pair MAP@5 (beat 0.24) ===")
    for c in F.columns:
        m=map5(pos.assign(hand_score=pos[c]))
        print(f"  {c:16s} ev {ev[c].mean():.3f} nonev {nv[c].mean():.3f} sep {ev[c].mean()-nv[c].mean():+.3f} | MAP@5 {m:.4f}")
    print("\n=== per family (share_rank / dominated / board_hit_gap: ev vs nonev) ===")
    for fam in TARGET_BEHAVIORS:
        f=pos[pos["behavior_family"]==fam]; fe=f[f["is_evidence"]]; fn=f[~f["is_evidence"]]
        print(f"  {fam:22s} share_rank {fe['cr_share_rank'].mean():.3f}/{fn['cr_share_rank'].mean():.3f} | dominated {fe['cr_dominated'].mean():.3f}/{fn['cr_dominated'].mean():.3f} | board_hit_gap {fe['cr_board_hit_gap'].mean():.2f}/{fn['cr_board_hit_gap'].mean():.2f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
