"""RING EXISTENCE CHECK (§198 Recommendation 2) — near-zero-cost, no modeling.
Does the labeled positive set contain MULTI-MEMBER ring structure (a player in 2+ positive pairs), or is it
essentially a random matching (each colluder appears once)? Everything in the pipeline is PAIR-scoped; if
rings exist, the whole model is structurally blind to them. This decides whether ring-features are a real
untried axis or refuted immediately.
Run:  python -m anchor_repro.ring_existence
"""
from __future__ import annotations
import numpy as np, polars as pl
from collections import Counter, defaultdict

def main():
    d=pl.read_csv("data/poker/development_labels.csv")
    pos=d.filter(pl.col("label")==1)
    neg=d.filter(pl.col("label")==0)
    print(f"positive pairs: {pos.height} | negative pairs: {neg.height}")

    # count how many positive pairs each player appears in
    players=pos["player_1"].to_list()+pos["player_2"].to_list()
    cnt=Counter(players)
    deg=np.array(list(cnt.values()))
    print(f"\n=== player degree in POSITIVE-pair graph (how many positive pairs each colluder is in) ===")
    print(f"  unique players in positive pairs: {len(cnt)}")
    print(f"  degree: mean {deg.mean():.3f} max {deg.max()} | players with deg>=2: {(deg>=2).sum()} ({(deg>=2).mean()*100:.1f}%)")
    dist=Counter(deg.tolist())
    for k in sorted(dist): print(f"    degree {k}: {dist[k]} players")

    # fraction of positive pairs that SHARE a member with another positive pair
    padj=defaultdict(set)
    rows=pos.select(["pair_id","player_1","player_2"]).rows()
    pl_to_pairs=defaultdict(list)
    for pid,a,b in rows:
        pl_to_pairs[a].append(pid); pl_to_pairs[b].append(pid)
    shared_pairs=set()
    for pid,a,b in rows:
        if len(pl_to_pairs[a])>1 or len(pl_to_pairs[b])>1: shared_pairs.add(pid)
    print(f"\n=== ring linkage ===")
    print(f"  positive pairs sharing a member with another positive pair: {len(shared_pairs)}/{pos.height} ({len(shared_pairs)/pos.height*100:.1f}%)")

    # connected components (rings) among positive pairs
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y): parent[find(x)]=find(y)
    for pid,a,b in rows: union(a,b)
    comp=defaultdict(list)
    for p in cnt: comp[find(p)].append(p)
    sizes=sorted([len(v) for v in comp.values()],reverse=True)
    print(f"  connected components (rings) among positive players: {len(sizes)}")
    print(f"  component sizes (players per ring), top 15: {sizes[:15]}")
    print(f"  components with >2 players (true rings): {sum(1 for s in sizes if s>2)}")

    # baseline: what would random matching look like? players appearing once => all deg=1, all comp size 2
    print(f"\n=== VERDICT ===")
    if (deg>=2).sum()==0:
        print("  NO ring structure: every colluder appears in exactly ONE positive pair (random matching).")
        print("  -> ring/multi-way features are REFUTED cheaply. Pair-scoped model loses nothing.")
    else:
        print(f"  RING STRUCTURE EXISTS: {(deg>=2).sum()} players in 2+ positive pairs, {sum(1 for s in sizes if s>2)} rings >2 players.")
        print("  -> pair-scoped model is BLIND to this. First genuinely untested structural axis. BUILD ring features.")
        # does ring membership correlate with behavior_family?
        fam=pos.select(["pair_id","behavior_family"]).to_dict(as_series=False)
        fam_of=dict(zip(fam["pair_id"],fam["behavior_family"]))
        ring_fams=Counter(fam_of[pid] for pid in shared_pairs)
        print(f"  behavior_family of ring-linked positive pairs: {dict(ring_fams)}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
