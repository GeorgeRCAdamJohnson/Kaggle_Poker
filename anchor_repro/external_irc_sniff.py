"""EXTERNAL VALIDATION SNIFF TEST — run our collusion-detector CONCEPT on real IRC poker data.

No labels exist (real data), so this is a SNIFF TEST not a scored benchmark: does our detector's core
signal light up on PLAUSIBLE collusion patterns on an INDEPENDENT dataset (real human Limit hold'em, IRC
1995-2001), or is it degenerate? This directly tests the §225c worry: do we detect COLLUSION, or did we
learn THIS competition's NL-generator signature?

IRC format (per month dir):
  hroster: <ts> <nplayers> <p1> <p2> ...            -> who was seated each hand
  pdb/pdb.<player>: <player> <ts> <nseats> <pos> <pf> <flop> <turn> <river> <bankroll> <totbet> <winnings> [cards]
Action codes: B=blind k=check c=call b=bet r=raise f=fold A=allin Q=quit K=kicked.

We build pair-level features our real detector keys on, over co-seated pairs with >= MIN_SHARED hands:
  net_flow_gap      : asymmetry of net chip flow between the two (directed-transfer tell)
  cofold_rate       : both fold same hand (coordinated non-contest)
  one_wins_one_dumps: hands where A wins pot AND B put chips in then folded (dump signature)
  mutual_softness   : low mutual aggression when heads-up together
Then rank pairs by a directed-transfer-style score and INSPECT the top flagged pairs for face validity.
Run:  python -m anchor_repro.external_irc_sniff
"""
from __future__ import annotations
import time, tarfile, io, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

IRC=Path("_external_val/IRCdata")
MONTHS=["holdem.199807.tgz","holdem.199808.tgz","holdem.199809.tgz"]  # a few months of ring-game hold'em
MIN_SHARED=30
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def parse_month(tgz_path):
    """Yield per-(hand,player) records: (ts, player, pos, nseats, folded, n_aggr, went_sd, winnings, totbet)."""
    recs=[]
    with tarfile.open(tgz_path,"r:gz") as tf:
        pdb_members=[m for m in tf.getmembers() if "/pdb/pdb." in m.name.replace("\\","/")]
        for m in pdb_members:
            f=tf.extractfile(m)
            if f is None: continue
            for raw in io.TextIOWrapper(f,encoding="latin-1"):
                parts=raw.split()
                if len(parts)<11: continue
                player=parts[0]
                try:
                    ts=int(parts[1]); nseats=int(parts[2]); pos=int(parts[3])
                except ValueError: continue
                pf,flop,turn,river=parts[4],parts[5],parts[6],parts[7]
                try:
                    totbet=int(parts[9]); winnings=int(parts[10])
                except ValueError: continue
                actions=(pf+flop+turn+river).replace("-","")
                folded="f" in actions.lower()
                n_aggr=sum(actions.count(c) for c in ("b","r","A"))
                went_sd=("c" in actions.lower() or "k" in actions.lower()) and not folded and (river not in ("-","") )
                recs.append((ts,player,pos,nseats,int(folded),n_aggr,int(bool(went_sd)),winnings,totbet))
    return recs

def main():
    all_recs=[]
    for mo in MONTHS:
        p=IRC/mo
        if not p.exists(): log(f"skip missing {mo}"); continue
        r=parse_month(p); all_recs+=r; log(f"{mo}: {len(r):,} player-hand records (cum {len(all_recs):,})")
    df=pd.DataFrame(all_recs,columns=["ts","player","pos","nseats","folded","n_aggr","went_sd","winnings","totbet"])
    log(f"total player-hand records: {len(df):,}; unique players {df.player.nunique():,}; unique hands {df.ts.nunique():,}")

    # co-seated pairs per hand -> accumulate pair features
    from itertools import combinations
    pair_stats=defaultdict(lambda: dict(n=0, flow_a=0, flow_b=0, cofold=0, dump_ab=0, dump_ba=0, both_sd=0, mutual_lowaggr=0))
    ghand=df.groupby("ts")
    nh=0
    for ts,g in ghand:
        nh+=1
        rows=g.to_dict("records")
        if len(rows)<2 or len(rows)>10: continue
        for x,y in combinations(rows,2):
            a,b=(x,y) if x["player"]<y["player"] else (y,x)
            key=(a["player"],b["player"]); s=pair_stats[key]; s["n"]+=1
            s["flow_a"]+=a["winnings"]; s["flow_b"]+=b["winnings"]
            if a["folded"] and b["folded"]: s["cofold"]+=1
            # dump: one wins chips while the other put money in (totbet>0) then folded
            if a["winnings"]>0 and b["folded"] and b["totbet"]>0: s["dump_ba"]+=1   # b dumped to a
            if b["winnings"]>0 and a["folded"] and a["totbet"]>0: s["dump_ab"]+=1   # a dumped to b
            if a["went_sd"] and b["went_sd"]: s["both_sd"]+=1
            if a["n_aggr"]==0 and b["n_aggr"]==0: s["mutual_lowaggr"]+=1
        if nh%20000==0: log(f"  processed {nh:,} hands, {len(pair_stats):,} pairs")
    log(f"done pairing: {len(pair_stats):,} co-seated pairs")

    rows=[]
    for (pa,pb),s in pair_stats.items():
        if s["n"]<MIN_SHARED: continue
        n=s["n"]
        rows.append(dict(player_a=pa,player_b=pb,n_shared=n,
            cofold_rate=s["cofold"]/n, both_sd_rate=s["both_sd"]/n, mutual_lowaggr_rate=s["mutual_lowaggr"]/n,
            dump_rate=(s["dump_ab"]+s["dump_ba"])/n, dump_dir_gap=abs(s["dump_ab"]-s["dump_ba"])/n,
            flow_gap_perhand=abs(s["flow_a"]-s["flow_b"])/n))
    P=pd.DataFrame(rows)
    log(f"pairs with >= {MIN_SHARED} shared hands: {len(P):,}")
    # directed-transfer-style suspicion score (our detector's core intuition): directed dumping + coordinated non-contest
    P["susp"]=( (P.dump_dir_gap - P.dump_dir_gap.median())/ (P.dump_dir_gap.std()+1e-9)
              + (P.cofold_rate - P.cofold_rate.median())/(P.cofold_rate.std()+1e-9)
              + (P.mutual_lowaggr_rate - P.mutual_lowaggr_rate.median())/(P.mutual_lowaggr_rate.std()+1e-9) )
    print("\n=== distribution sanity (is the signal degenerate?) ===")
    for c in ["cofold_rate","dump_rate","dump_dir_gap","mutual_lowaggr_rate","flow_gap_perhand","susp"]:
        print(f"  {c:20s} mean {P[c].mean():.4f} p50 {P[c].median():.4f} p95 {P[c].quantile(.95):.4f} max {P[c].max():.4f}")
    print("\n=== TOP 15 flagged pairs (face-validity inspection) ===")
    top=P.sort_values("susp",ascending=False).head(15)
    print(top[["player_a","player_b","n_shared","dump_dir_gap","cofold_rate","mutual_lowaggr_rate","flow_gap_perhand","susp"]].to_string(index=False))
    print("\n=== BOTTOM 5 (should look benign) ===")
    print(P.sort_values("susp").head(5)[["player_a","player_b","n_shared","dump_dir_gap","cofold_rate","susp"]].to_string(index=False))
    P.to_parquet("_external_val/irc_pair_features.parquet")
    log("wrote _external_val/irc_pair_features.parquet")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
