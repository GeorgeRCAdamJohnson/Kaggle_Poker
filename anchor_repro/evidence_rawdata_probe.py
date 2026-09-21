"""§212 — RAW-DATA evidence probe. We've recycled derived features (chen, hole_class, action aggregates) but
NEVER put these RAW columns in front of the evidence ranker: hole_card_1/2 (actual cards), board_cards,
seat_no (relative seating), starting_stack dynamics. For the 1,817 TRUE evidence hands vs the non-evidence
shared hands of the SAME positive pairs, which untouched raw signal separates them (within-pair MAP@5)?

Card-relationship signals for the PAIR in a hand (both colluders' hole cards + board):
  cr_share_rank    : the two colluders hold a matching rank (e.g., both have a Q) -> card-removal collusion
  cr_share_suit    : share a suit (flush blocker coordination)
  cr_connected     : their cards form connectors/overlap
  cr_dominated     : one player's hand dominates the other (AK vs AQ) -> soft-play setup
  cr_board_hit_gap : one hits board hard, other whiffs (isolation setup)
Seating: seat_dist (relative seat distance), one_on_button.
Stack: stack_ratio (max/min starting_stack of the pair).
All computed from seats.parquet + hands.parquet, joined to the labeled evidence frame. Report single-feature
within-pair MAP@5 (magnitude rules maxed 0.18, ci_peak_eq_gap 0.24 — beat those to be interesting).
Run:  python -m anchor_repro.evidence_rawdata_probe
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
D=Path("data/poker"); CACHE=Path("outputs/poker_collusion/generator_cache"); STEP3=Path("outputs/poker_collusion/hosen42_step3")

RANKS="23456789TJQKA"; RMAP={c:i for i,c in enumerate(RANKS)}
def parse(card):
    if not card or len(card)<2: return (-1,-1)
    return (RMAP.get(card[0],-1), ord(card[1]))

def main():
    from anchor_repro.seq_evidence import build_dh, map5, TARGET_BEHAVIORS
    dh=build_dh()  # labeled per-hand frame with is_evidence, pot_bb, label, behavior_family
    keep=set(dh["pair_id"].to_list())
    # pair members
    lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2"])
    seats=pl.read_parquet(D/"seats.parquet").select(["hand_id","player_id","seat_no","starting_stack","hole_card_1","hole_card_2"])
    hands=pl.read_parquet(D/"hands.parquet").select(["hand_id","board_cards","button_seat"])
    # restrict seats to hands in dh
    hset=set(dh["hand_id"].unique().to_list())
    seats=seats.filter(pl.col("hand_id").is_in(list(hset)))
    dhp=dh.select(["pair_id","hand_id","is_evidence","pot_bb","label","behavior_family"]).to_pandas()
    dhp=dhp.merge(lab.to_pandas(),on="pair_id",how="left")
    sd=seats.to_pandas(); hd=hands.to_pandas()
    # index seats by (hand,player)
    sidx={(r.hand_id,r.player_id):r for r in sd.itertuples()}
    board={r.hand_id:r.board_cards for r in hd.itertuples()}
    rows=[]
    for t in dhp.itertuples():
        a=sidx.get((t.hand_id,t.player_1)); b=sidx.get((t.hand_id,t.player_2))
        if a is None or b is None:
            rows.append((0,0,0,0,0,0,0,0)); continue
        a1,a2=parse(a.hole_card_1),parse(a.hole_card_2); b1,b2=parse(b.hole_card_1),parse(b.hole_card_2)
        aranks={a1[0],a2[0]}; branks={b1[0],b2[0]}; asuits={a1[1],a2[1]}; bsuits={b1[1],b2[1]}
        share_rank=len(aranks & branks & set(range(13)))>0
        share_suit=len((asuits & bsuits)-{-1})>0
        amax=max(a1[0],a2[0]); bmax=max(b1[0],b2[0])
        dominated=(amax>=0 and bmax>=0 and (a1[0]==b1[0] or a1[0]==b2[0] or a2[0]==b1[0] or a2[0]==b2[0]) and amax!=bmax)
        # board hit: count board ranks matching each player's ranks
        bc=[parse(x) for x in str(board.get(t.hand_id,"")).split() if x]
        bset={r for r,_ in bc}
        ahit=len(aranks & bset); bhit=len(branks & bset)
        board_hit_gap=abs(ahit-bhit)
        seat_dist=abs(a.seat_no-b.seat_no)
        stack_ratio=max(a.starting_stack,b.starting_stack)/(min(a.starting_stack,b.starting_stack)+1)
        rows.append((int(share_rank),int(share_suit),int(dominated),board_hit_gap,seat_dist,stack_ratio,ahit+bhit,int(amax==bmax)))
    import pandas as pd
    F=pd.DataFrame(rows,columns=["cr_share_rank","cr_share_suit","cr_dominated","cr_board_hit_gap","seat_dist","stack_ratio","board_hit_tot","cr_same_top"])
    for c in F.columns: dhp[c]=F[c].values.astype(float)
    pos=dhp[dhp["label"]==1].copy()
    print(f"labeled hands {len(dhp):,}; evidence {int(dhp['is_evidence'].sum())}; seat-matched {int((dhp['seat_dist']>=0).sum())}")
    # also base-rate: what fraction of evidence vs non-evidence have each flag
    print("\n=== raw signal: mean on EVIDENCE vs NON-evidence (within positive pairs) + single-feature within-pair MAP@5 ===")
    ev=pos[pos["is_evidence"]]; nv=pos[~pos["is_evidence"]]
    for c in F.columns:
        m=map5(pos.assign(hand_score=pos[c]))
        print(f"  {c:16s} ev {ev[c].mean():.3f} nonev {nv[c].mean():.3f} sep {ev[c].mean()-nv[c].mean():+.3f} | MAP@5 {m:.4f}")
    print("\n=== per family: cr_share_rank / cr_dominated means (transfer/soft/iso) ===")
    for fam in TARGET_BEHAVIORS:
        f=pos[pos["behavior_family"]==fam]
        fe=f[f["is_evidence"]]; fn=f[~f["is_evidence"]]
        print(f"  {fam:22s} share_rank ev {fe['cr_share_rank'].mean():.3f}/nv {fn['cr_share_rank'].mean():.3f} | dominated ev {fe['cr_dominated'].mean():.3f}/nv {fn['cr_dominated'].mean():.3f} | board_hit_gap ev {fe['cr_board_hit_gap'].mean():.2f}/nv {fn['cr_board_hit_gap'].mean():.2f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
