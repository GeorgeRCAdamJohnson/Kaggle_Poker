"""§217 — CORRECTED action-MI (audit of §216 found I built the WRONG construction). Bonjour 2022's actual
signal is MUTUAL INFORMATION between the two players' actions computed ACROSS shared games — a per-PAIR
statistic, tested as a RISK feature. §216 built per-HAND binary co-action flags (=> reconstructable from
HAND_FEATS action counts, R2 0.99). This builds the REAL thing:

Per pair, over all shared hands: MI(action_A ; action_B) where action_X = that member's action-type in a
matched decision context (street bucket). MI = sum p(a,b) log[ p(a,b) / (p(a)p(b)) ]. High MI = the two
members' actions are statistically DEPENDENT beyond their marginals = coordination.
FIELD-NORMALIZE (kill the §193 shared-state confound): subtract each member's MI with RANDOM co-seated
NON-partner opponents in matched contexts. mi_excess = MI(A;B) - mean_k MI(A; nonpartner_k). Positive excess
= the pair coordinates MORE than the shared-table baseline explains.
Test as a RISK feature: orthogonality vs base+PT+seq risk features (NOT HAND_FEATS), adversarial drift, and
eval-weighted gate. Ship-eval if eval-weighted >+0.005 & drift<0.65 & orthogonal.
Run: python -m anchor_repro.action_mi_v2 --phase development
"""
from __future__ import annotations
import time, argparse, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from collections import defaultdict, Counter
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); OUT=STEP3/"edge"/"actionmi"; OUT.mkdir(parents=True,exist_ok=True)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)
MIV2_FEATS=["ami_pair_mi","ami_field_mi","ami_excess","ami_pair_condmi"]
ACT2I={"fold":0,"check":1,"call":2,"bet":3,"raise":4,"all_in":4}  # 5 action classes

def mi_between(seqA, seqB):
    """MI between two aligned lists of action-class ints. Aligned by CONTEXT bucket (street)."""
    if not seqA or not seqB: return 0.0
    n=min(len(seqA),len(seqB))
    if n<3: return 0.0
    a=np.array(seqA[:n]); b=np.array(seqB[:n])
    joint=Counter(zip(a.tolist(),b.tolist())); pa=Counter(a.tolist()); pb=Counter(b.tolist())
    mi=0.0
    for (x,y),c in joint.items():
        pxy=c/n; px=pa[x]/n; py=pb[y]/n
        if pxy>0: mi+=pxy*np.log(pxy/(px*py+1e-12)+1e-12)
    return max(mi,0.0)

def build(phase, pair_hands_path):
    pairs=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2"])
    hset=pairs["hand_id"].unique().to_list()
    acts=(pl.scan_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(hset))
          .select(["hand_id","player_id","action","street"]).collect())
    acts=acts.with_columns(pl.col("action").replace_strict(ACT2I,default=1).alias("ai"))
    # per (hand, player): the sequence of action-classes, and street buckets, in order
    hp=(acts.group_by(["hand_id","player_id"]).agg(pl.col("ai").alias("ais"),pl.col("street").alias("strs")))
    # index: (hand,player)->action list ; and per hand, list of ALL players (for field baseline)
    hpm={}; hand_players=defaultdict(list)
    for r in hp.iter_rows(named=True):
        hpm[(r["hand_id"],r["player_id"])]=r["ais"]; hand_players[r["hand_id"]].append(r["player_id"])
    rng=np.random.RandomState(42)
    rows=[]
    # gather per pair: concat member action seqs across shared hands, aligned per hand
    pair_rows=pairs.group_by(["pair_id","player_1","player_2"]).agg(pl.col("hand_id").alias("hands"))
    for r in pair_rows.iter_rows(named=True):
        pid=r["pair_id"]; a=r["player_1"]; b=r["player_2"]; hands=r["hands"]
        SA=[]; SB=[]; SF=[]  # aligned action-class streams: A, B, and A-vs-fieldpartner
        field_mis=[]
        for h in hands:
            ca=hpm.get((h,a)); cb=hpm.get((h,b))
            if ca and cb:
                m=min(len(ca),len(cb)); SA+=ca[:m]; SB+=cb[:m]
            # field baseline: A vs a random non-partner at the same hand
            others=[p for p in hand_players.get(h,[]) if p!=a and p!=b]
            if ca and others:
                o=rng.choice(others); co=hpm.get((h,o))
                if co: m=min(len(ca),len(co)); SF+=[(ca[i],co[i]) for i in range(m)]
        pair_mi=mi_between(SA,SB)
        # field MI: A vs random non-partners (pooled)
        if SF:
            fa=[x for x,_ in SF]; fo=[y for _,y in SF]; field_mi=mi_between(fa,fo)
        else: field_mi=0.0
        rows.append((pid,pair_mi,field_mi,pair_mi-field_mi,pair_mi))
    out=pl.DataFrame(rows,schema=["pair_id","ami_pair_mi","ami_field_mi","ami_excess","ami_pair_condmi"],orient="row")
    p=OUT/f"action_mi_v2_{phase}.parquet"; out.write_parquet(p)
    log(f"{phase}: pair-level action-MI {out.height:,} pairs -> {p.name}  (pair_mi mean {out['ami_pair_mi'].mean():.4f}, excess mean {out['ami_excess'].mean():.4f})")
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",default="development"); a=ap.parse_args()
    ph={"development":STEP3/"dev_pair_hands.parquet","evaluation":STEP3/"eval_pair_hands.parquet"}[a.phase]
    build(a.phase,ph)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
