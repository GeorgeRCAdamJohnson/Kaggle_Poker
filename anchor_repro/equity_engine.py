"""EXACT hand-equity engine (§ the one new-information lever, §154-166 diagnosis).

We have 100% of hole cards, so for each shared-hand pair MEMBER at the street they LAST acted
(fold/check = the surrender moment), compute their TRUE P(win) vs the ACTUAL live opponents' cards by
enumerating remaining board runouts. This is NEW information beyond the pipeline's made-value v_flop/
turn/river and showdown-only rk_final: equity accounts for draws/future cards AND is defined even when
there is no showdown (72-85% of collusion hands).

Method per (hand, member, street):
  known = member 2 cards + each LIVE opponent's 2 cards + board_so_far (0/3/4/5 cards)
  need  = 5 - len(board_so_far) more board cards from the remaining deck
  equity = over runouts: fraction where member's best-7 strictly beats ALL live opponents (ties split)
  EXACT enumerate if C(remaining, need) small; else Monte-Carlo N=2000 samples (preflop need=5).
Reuses eval_best/eval5 (numba) from the pipeline via a local copy.
Cache: outputs/.../edge/equity/pairhand_equity_{dev,eval}.parquet (pair_id,hand_id,member(1/2),street,equity,n_live).
Run:  python -m anchor_repro.equity_engine --phase development   (and evaluation)
"""
from __future__ import annotations
import sys, time, itertools, numpy as np, polars as pl
from pathlib import Path
from numba import njit
STEP3=Path("outputs/poker_collusion/hosen42_step3"); DATA=Path("data/poker")
OUT=STEP3/"edge"/"equity"; OUT.mkdir(parents=True,exist_ok=True)
AC=STEP3/"action_context.parquet"
RANK_CHARS="23456789TJQKA"; SUIT_CHARS="cdhs"
def card_to_int(c): return RANK_CHARS.index(c[0])*4+SUIT_CHARS.index(c[1])
MC_N=2000; SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

# ---- evaluator (copied verbatim from policy_pipeline eval5/eval_best) ----
@njit(cache=True)
def eval5(c0,c1,c2,c3,c4):
    ranks=np.zeros(5,np.int64); ranks[0]=c0>>2; ranks[1]=c1>>2; ranks[2]=c2>>2; ranks[3]=c3>>2; ranks[4]=c4>>2
    flush=((c0&3)==(c1&3)) and ((c1&3)==(c2&3)) and ((c2&3)==(c3&3)) and ((c3&3)==(c4&3))
    cnt=np.zeros(13,np.int64)
    for i in range(5): cnt[ranks[i]]+=1
    mask=0
    for i in range(5): mask|=1<<ranks[i]
    straight_high=-1
    for hi in range(12,3,-1):
        if (mask>>(hi-4))&31==31: straight_high=hi; break
    if straight_high<0 and (mask&0b1000000001111)==0b1000000001111: straight_high=3
    four=-1; three=-1; pair_hi=-1; pair_lo=-1
    for r in range(12,-1,-1):
        if cnt[r]==4: four=r
        elif cnt[r]==3: three=r
        elif cnt[r]==2:
            if pair_hi<0: pair_hi=r
            else: pair_lo=r
    kick=np.full(5,-1,np.int64); ki=0
    for r in range(12,-1,-1):
        if cnt[r]==1: kick[ki]=r; ki+=1
    B=13
    if straight_high>=0 and flush: return 8*B**5+straight_high
    if four>=0:
        k=-1
        for r in range(12,-1,-1):
            if cnt[r]>=1 and r!=four: k=r; break
        return 7*B**5+four*B+k
    if three>=0 and pair_hi>=0: return 6*B**5+three*B+pair_hi
    if flush:
        v=5*B**5
        for i in range(5): v+=kick[i]*B**(4-i)
        return v
    if straight_high>=0: return 4*B**5+straight_high
    if three>=0: return 3*B**5+three*B**2+kick[0]*B+kick[1]
    if pair_hi>=0 and pair_lo>=0: return 2*B**5+pair_hi*B**2+pair_lo*B+kick[0]
    if pair_hi>=0: return 1*B**5+pair_hi*B**3+kick[0]*B**2+kick[1]*B+kick[2]
    v=0
    for i in range(5): v+=kick[i]*B**(4-i)
    return v

@njit(cache=True)
def eval7(cards):
    best=-1; n=len(cards)
    for a in range(n-4):
        for b in range(a+1,n-3):
            for c in range(b+1,n-2):
                for d in range(c+1,n-1):
                    for e in range(d+1,n):
                        v=eval5(cards[a],cards[b],cards[c],cards[d],cards[e])
                        if v>best: best=v
    return best

@njit(cache=True)
def _equity_runouts(hero, opps, board_fixed, deck, need, runidx):
    """hero: (2,), opps: (K,2), board_fixed: (b,), deck: remaining cards, runidx: (R,need) index rows.
    Returns hero equity (win=1, tie=1/ties) averaged over the R runouts."""
    K=opps.shape[0]; tot=0.0; R=runidx.shape[0]
    seven=np.empty(7,np.int64); oseven=np.empty(7,np.int64)
    for r in range(R):
        # build board = board_fixed + deck[runidx[r]]
        seven[0]=hero[0]; seven[1]=hero[1]
        for i in range(board_fixed.shape[0]): seven[2+i]=board_fixed[i]
        base=board_fixed.shape[0]
        for j in range(need): seven[2+base+j]=deck[runidx[r,j]]
        hv=eval7(seven[:2+base+need]); best_opp=-1; ntie=0
        for k in range(K):
            oseven[0]=opps[k,0]; oseven[1]=opps[k,1]
            for i in range(base): oseven[2+i]=seven[2+i]
            for j in range(need): oseven[2+base+j]=seven[2+base+j]
            ov=eval7(oseven[:2+base+need])
            if ov>best_opp: best_opp=ov
        if hv>best_opp: tot+=1.0
        elif hv==best_opp: tot+=0.5   # split (approx: half regardless of #tied)
    return tot/R

def _combos_index(m, need, rng, cap=1600):
    """Row-index matrix (R,need) of board runouts from m remaining cards. Exact if C(m,need)<=cap else MC."""
    from math import comb
    if need==0: return np.zeros((1,0),np.int64)
    total=comb(m,need)
    if total<=cap:
        return np.array(list(itertools.combinations(range(m),need)),np.int64)
    idx=np.empty((MC_N,need),np.int64)
    for r in range(MC_N): idx[r]=rng.choice(m,need,replace=False)
    return idx

# ---- PREFLOP equity lookup: 169 starting-hand classes vs N random opponents (precomputed once) ----
def _hole_class(c1,c2):
    """169-class index: rank pair (hi,lo) + suited flag. Returns (hi_rank, lo_rank, suited)."""
    r1,s1=c1>>2,c1&3; r2,s2=c2>>2,c2&3
    hi,lo=(r1,r2) if r1>=r2 else (r2,r1); suited=1 if s1==s2 else 0
    return hi,lo,suited
def _class_key(c1,c2):
    hi,lo,su=_hole_class(c1,c2); return hi*13*2+lo*2+su

def build_preflop_table(cache=OUT/"preflop_equity.npy", n_mc=4000, max_opp=5):
    if cache.exists(): return np.load(cache)
    log("building preflop equity table (169 classes x 1..5 opponents, MC)")
    rng=np.random.RandomState(SEED)
    # table[classkey, nopp] = equity vs nopp random opponents
    T=np.full((13*13*2, max_opp+1), np.nan, np.float32)
    # representative cards per class
    reps={}
    for c1 in range(52):
        for c2 in range(c1+1,52):
            k=_class_key(c1,c2)
            if k not in reps: reps[k]=(c1,c2)
    for k,(h1,h2) in reps.items():
        for nopp in range(1,max_opp+1):
            wins=0.0
            for _ in range(n_mc):
                deck=[c for c in range(52) if c!=h1 and c!=h2]
                pick=rng.choice(len(deck),2*nopp+5,replace=False)
                oc=[deck[i] for i in pick]
                opps=[(oc[2*j],oc[2*j+1]) for j in range(nopp)]; board=oc[2*nopp:2*nopp+5]
                hv=eval7(np.array([h1,h2]+board,np.int64))
                best=max(eval7(np.array([o[0],o[1]]+board,np.int64)) for o in opps)
                wins+= 1.0 if hv>best else (0.5 if hv==best else 0.0)
            T[k,nopp]=wins/n_mc
    np.save(cache,T); log("preflop table cached"); return T

def build(phase, pair_hands_path, out_path, keep_pairs=None, chunk=None, nchunks=None):
    if out_path.exists(): log(f"{phase} equity cached"); return
    rng=np.random.RandomState(SEED)
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2","table_id"] if "table_id" in pl.read_parquet(pair_hands_path).columns else ["pair_id","hand_id","player_1","player_2"])
    if keep_pairs is not None:
        ph=ph.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    if chunk is not None and nchunks is not None:
        # table-disjoint chunking for parallel shard runs (hash table_id -> chunk)
        if "table_id" in ph.columns:
            tbls=sorted(ph["table_id"].unique().to_list())
            assign={t:(i%nchunks) for i,t in enumerate(tbls)}
            ph=ph.filter(pl.col("table_id").replace_strict(assign,default=-1)==chunk)
        else:
            ph=ph.with_row_index("_ri").filter((pl.col("_ri")%nchunks)==chunk).drop("_ri")
        log(f"{phase} chunk {chunk}/{nchunks}: {ph['pair_id'].n_unique():,} pairs, {ph.height:,} pair-hands")
    elif keep_pairs is not None:
        log(f"{phase}: restricted to {ph['pair_id'].n_unique():,} pairs, {ph.height:,} pair-hands")
    hand_ids=ph["hand_id"].unique().to_list()
    # cards for all seats in those hands
    seats=(pl.scan_parquet(DATA/"seats.parquet").filter(pl.col("hand_id").is_in(hand_ids))
           .select(["hand_id","player_id","hole_card_1","hole_card_2","folded"]).collect())
    board=(pl.read_parquet(DATA/"hands.parquet",columns=["hand_id","board_cards"])
           .filter(pl.col("hand_id").is_in(hand_ids)))
    # last-acted street per (hand,player) from action_context
    la=(pl.scan_parquet(AC).filter(pl.col("phase")==phase)
        .group_by(["hand_id","player_id"]).agg(pl.col("street_no").max().alias("last_street")).collect())
    log(f"{phase}: {len(hand_ids):,} hands, {seats.height:,} seats")
    # encode cards
    cmap={f"{r}{s}":RANK_CHARS.index(r)*4+SUIT_CHARS.index(s) for r in RANK_CHARS for s in SUIT_CHARS}
    def enc(x): return cmap.get(x,-1)
    seats=seats.with_columns(pl.col("hole_card_1").replace_strict(cmap,default=-1).alias("h1"),
                             pl.col("hole_card_2").replace_strict(cmap,default=-1).alias("h2")).to_pandas()
    la=la.to_pandas(); board=board.to_pandas()
    # board -> up to 5 ints
    def benc(s):
        if not isinstance(s,str) or not s.strip(): return []
        return [cmap.get(t,-1) for t in s.split() if t in cmap]
    board["b"]=board["board_cards"].map(benc)
    bmap=dict(zip(board["hand_id"],board["b"]))
    # group seats by hand
    seats_by_hand={}
    for hid,g in seats.groupby("hand_id"): seats_by_hand[hid]=g
    la_map={(r.hand_id,r.player_id):int(r.last_street) for r in la.itertuples()}

    PFT=build_preflop_table()   # 169-class preflop equity lookup (instant per query)
    STREET_NB={0:0,1:3,2:4,3:5}  # board cards visible at street (pf=0, flop=3, turn=4, river=5)
    rows=[]
    php=ph.to_pandas(); n=len(php); t=time.time()
    for i,r in enumerate(php.itertuples()):
        hid=r.hand_id; g=seats_by_hand.get(hid)
        if g is None: continue
        full_board=bmap.get(hid,[])
        cards={row.player_id:(row.h1,row.h2,row.folded) for row in g.itertuples()}
        for mi,pid in [(1,r.player_1),(2,r.player_2)]:
            if pid not in cards: continue
            h1,h2,_=cards[pid]
            if h1<0 or h2<0: continue
            st=la_map.get((hid,pid),3)
            opps=[(c[0],c[1]) for q,c in cards.items() if q!=pid and c[0]>=0 and not c[2]]
            if len(opps)==0: opps=[(c[0],c[1]) for q,c in cards.items() if q!=pid and c[0]>=0][:1]
            if len(opps)==0: continue
            nopp=min(len(opps),5)
            if st==0:
                # PREFLOP: table lookup vs nopp random opponents (instant, no enumeration)
                eq=float(PFT[_class_key(h1,h2),nopp])
            else:
                nb=STREET_NB.get(st,5); bf=full_board[:nb]
                used=set([h1,h2]+list(bf)+[c for o in opps for c in o])
                deck=np.array([c for c in range(52) if c not in used],np.int64)
                need=5-len(bf)
                if need<0: need=0
                if need>len(deck): continue
                idx=_combos_index(len(deck),need,rng)
                eq=_equity_runouts(np.array([h1,h2],np.int64),np.array(opps,np.int64),np.array(bf,np.int64),deck,need,idx)
            rows.append((r.pair_id,hid,mi,st,float(eq),len(opps)))
        if i%40000==0:
            rate=(i+1)/max(time.time()-t,1e-9); log(f"  {phase} {i:,}/{n:,} pairs ({rate:.0f}/s, ETA {(n-i)/max(rate,1e-9):.0f}s)")
    df=pl.DataFrame(rows,schema=["pair_id","hand_id","member","street","equity","n_live"],orient="row")
    df.write_parquet(out_path)
    log(f"{phase}: wrote {df.height:,} equity rows")

def _arg(name,default=None):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default

def main():
    ph=_arg("--phase","development"); full="--full" in sys.argv
    chunk=_arg("--chunk"); nchunks=_arg("--nchunks")
    chunk=int(chunk) if chunk is not None else None; nchunks=int(nchunks) if nchunks is not None else None
    src=STEP3/("dev_pair_hands.parquet" if ph=="development" else "eval_pair_hands.parquet")
    tag="dev" if ph=="development" else "eval"
    if "--merge" in sys.argv:
        # concat all shards -> final
        shards=sorted(OUT.glob(f"_shard_{tag}_*.parquet"))
        df=pl.concat([pl.read_parquet(s) for s in shards])
        out=OUT/(f"pairhand_equity_{tag}.parquet")
        df.write_parquet(out); log(f"merged {len(shards)} shards -> {out.name} ({df.height:,} rows)")
        return 0
    keep=None
    if ph=="development" and not full and chunk is None:
        dp=pl.read_parquet(STEP3/"dev_pairs.parquet"); keep=set(dp.filter(pl.col("is_labeled"))["pair_id"].to_list())
        build("development",src,OUT/"pairhand_equity_devlab.parquet",keep_pairs=keep); return 0
    if chunk is not None:
        out=OUT/f"_shard_{tag}_{chunk}.parquet"
        build(ph,src,out,chunk=chunk,nchunks=nchunks); return 0
    build(ph,src,OUT/f"pairhand_equity_{tag}.parquet"); return 0

if __name__=="__main__":
    raise SystemExit(main())
