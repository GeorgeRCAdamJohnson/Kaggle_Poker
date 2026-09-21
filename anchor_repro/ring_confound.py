"""RING CONFOUND CHECK — before treating ring structure as a lever, rule out the two killers:
(A) Is 7.1% deg>=2 actually ABOVE chance? Permute the pairing (same players, random re-matching) and see the
    null distribution of deg>=2 rate. If observed ~ null, the "rings" are coincidence.
(B) THE DECISIVE ONE: a ring feature for scoring EVAL pairs can only use info available at eval time. The dev
    label is HIDDEN for eval. So "my partner is also a confirmed colluder" is UNUSABLE on eval. The only
    usable ring signal = graph structure over ALL pairs (co-play network), which §160/§193-194 showed leaks
    via degree/co-occurrence. Test: does degree in the ALL-PAIRS candidate graph (not the label graph)
    separate the label? If it just re-encodes co-occurrence/activity, it's the §193 leak again.
Run:  python -m anchor_repro.ring_confound
"""
from __future__ import annotations
import numpy as np, polars as pl
from collections import Counter

def deg_ge2_rate(p1,p2):
    c=Counter(list(p1)+list(p2)); d=np.array(list(c.values())); return (d>=2).mean(), (d>=2).sum()

def main():
    d=pl.read_csv("data/poker/development_labels.csv")
    pos=d.filter(pl.col("label")==1)
    p1=pos["player_1"].to_list(); p2=pos["player_2"].to_list()
    obs_rate,obs_n=deg_ge2_rate(p1,p2)
    print(f"observed deg>=2 rate among positive-pair players: {obs_rate*100:.1f}% ({obs_n} players)")

    # (A) null: keep the SAME multiset of players, randomly re-pair them, recompute deg>=2 rate
    allp=p1+p2; rng=np.random.RandomState(42); nulls=[]
    for _ in range(500):
        s=allp.copy(); rng.shuffle(s); h=len(s)//2
        r,_=deg_ge2_rate(s[:h],s[h:]); nulls.append(r)
    nulls=np.array(nulls)
    print(f"=== (A) permutation null (random re-matching of the same players) ===")
    print(f"  null deg>=2 rate: mean {nulls.mean()*100:.1f}% p95 {np.percentile(nulls,95)*100:.1f}%")
    print(f"  observed {obs_rate*100:.1f}% -> {'ABOVE chance (real clustering)' if obs_rate>np.percentile(nulls,95) else 'WITHIN chance (coincidence)'}")

    # (B) usability on eval: degree in ALL-PAIRS graph (the only eval-available ring signal)
    p1a=d["player_1"].to_list(); p2a=d["player_2"].to_list()
    call=Counter(p1a+p2a)
    y=d["label"].to_numpy()
    degfeat=np.array([max(call[a],call[b]) for a,b in zip(p1a,p2a)])
    from sklearn.metrics import roc_auc_score
    auc=roc_auc_score(y,degfeat)
    print(f"=== (B) ALL-PAIRS-graph max-degree as a label feature (eval-available) ===")
    print(f"  AUC = {max(auc,1-auc):.4f} ({'hi=pos' if auc>0.5 else 'lo=pos'})")
    print(f"  (if ~0.5 -> no usable eval signal; if high -> likely the co-occurrence/activity leak §193, verify vs shared_hands)")
    # how many candidate pairs even exist per player in dev_pair_features (the scored universe)?
    return 0

if __name__=="__main__":
    raise SystemExit(main())
