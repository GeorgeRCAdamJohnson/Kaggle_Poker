"""Dump the gameplay of specific (pair, evidence-hand) cases for the required 5 case reviews.
For each case: the two members' seats/hole cards, the full action sequence, net chips, showdown, board.
Run:  python -m anchor_repro.case_review_dump
"""
from __future__ import annotations
import polars as pl
from pathlib import Path
DATA=Path("data/poker")

CASES=[
    ("PD5956DB3CCF8","directed_transfer",["HA4449DCF8340CB","H86AD7E78C9C300","H8D15A034671CDA"]),
    ("PC178304F6F93","soft_play",["H22311A667D6951","H3965E8228C0661","H90FC271477B452"]),
    ("PF653436402F6","coordinated_isolation",["H339F3BB44B285F","HF0BFAD22FA23D0","H2D936F7870BEAE"]),
    ("P8F02C81D9B3B","directed_transfer",["H520F9D51A2348B","HDFE0DF189E11C3","H80C2BF0F055AD6"]),
    ("PE207D4A85DC0","soft_play",["HB20F602E8A5C70","H18F9B2CFE78EE6","HE82B6BE81A772E"]),
]
CASES=CASES[-1:]  # only the last (PE207) for this run

def main():
    ep=pl.read_csv(DATA/"evaluation_pairs.csv")
    seats=pl.scan_parquet(DATA/"seats.parquet")
    acts=pl.scan_parquet(DATA/"actions.parquet")
    hands=pl.scan_parquet(DATA/"hands.parquet")
    for pair_id,fam,hids in CASES:
        row=ep.filter(pl.col("pair_id")==pair_id)
        if row.height==0: print(f"{pair_id}: not in evaluation_pairs?"); continue
        p1,p2=row["player_1"][0],row["player_2"][0]
        print(f"\n################ {pair_id}  [{fam}]  members {p1} | {p2} ################")
        for hid in hids[:2]:  # 2 hands per case is enough to characterize
            h=hands.filter(pl.col("hand_id")==hid).collect()
            if h.height==0: print(f"  {hid}: not found"); continue
            bb=h["big_blind"][0]; board=h["board_cards"][0]; pot=h["final_pot"][0]
            print(f"\n  --- hand {hid}  BB={bb}  board=[{board}]  final_pot={pot} ---")
            sd=seats.filter(pl.col("hand_id")==hid).collect()
            for pl_id in (p1,p2):
                s=sd.filter(pl.col("player_id")==pl_id)
                if s.height==0: print(f"    {pl_id}: not seated"); continue
                cols=s.columns
                hc=[c for c in cols if "hole" in c.lower() or "card" in c.lower()]
                net=[c for c in cols if "net" in c.lower() or "won" in c.lower() or "chip" in c.lower()]
                fold=[c for c in cols if "fold" in c.lower()]
                info={c:s[c][0] for c in (hc+net+fold) if c in cols}
                print(f"    {pl_id}: {info}")
            aa=acts.filter(pl.col("hand_id")==hid).sort("action_no").collect()
            aa=aa.filter(pl.col("player_id").is_in([p1,p2]))
            print(f"    actions (pair members only):")
            for r in aa.iter_rows(named=True):
                print(f"      #{r['action_no']} {r['street']:>7} {r['player_id']}  {r['action']:>6} amt={r.get('amount')} to={r.get('amount_to')} pot_before={r.get('pot_before')} to_call={r.get('to_call')}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
