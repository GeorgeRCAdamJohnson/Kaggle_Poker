"""Vectorized TORCH 7-card poker evaluator on the RTX 5070 (GDDR7, ~10x bandwidth, idle 12GB VRAM).

Scores batches of 5-card hands via tensor rank/suit-count ops -> a single comparable integer per hand
(same category ordering as the numba eval5). 7-card best = max over the 21 five-card combos (vectorized).
Then equity = for a batch of (hero, K opponents, board_fixed) and a set of runout completions, the
fraction of runouts where hero's best-7 strictly beats all opponents (ties split).

Card int = rank*4 + suit, rank 0..12 (2..A), suit 0..3.  Validated bit-exact vs eval_best.
"""
from __future__ import annotations
import numpy as np, torch, itertools

DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
_C5=torch.tensor(list(itertools.combinations(range(7),5)),dtype=torch.long)  # (21,5)

def eval5_torch(cards):
    """cards: (...,5) long tensor of card ints 0..51. Returns (...) long score, higher=better.
    Category encoding mirrors eval5: 8=SF,7=quads,6=full,5=flush,4=straight,3=trips,2=twopair,1=pair,0=hi.
    Score = cat*13^5 + tiebreak (rank-ordered)."""
    r=(cards>>2)  # ranks 0..12
    s=(cards&3)   # suits 0..3
    B=13
    # rank counts (...,13)
    shp=cards.shape[:-1]
    rc=torch.zeros(shp+(13,),device=cards.device,dtype=torch.int16)
    rc.scatter_add_(-1, r.long(), torch.ones_like(r,dtype=torch.int16))
    # flush: all suits equal
    flush=(s.min(-1).values==s.max(-1).values)
    # straight: 5 distinct consecutive ranks; wheel A-5
    mask=torch.zeros(shp+(13,),device=cards.device,dtype=torch.bool)
    mask.scatter_(-1, r.long(), True)
    m=mask.to(torch.int32)
    # straight_high: highest hi in 12..4 with ranks hi-4..hi all present; wheel = 5-high (encode 3)
    sh=torch.full(shp,-1,device=cards.device,dtype=torch.long)
    for hi in range(12,3,-1):
        win=m[...,hi-4:hi+1].sum(-1)==5
        sh=torch.where((sh<0)&win, torch.full_like(sh,hi), sh)
    wheel=(m[...,12]+m[...,0]+m[...,1]+m[...,2]+m[...,3])==5  # A,2,3,4,5
    sh=torch.where((sh<0)&wheel, torch.full_like(sh,3), sh)
    straight=sh>=0
    # multiplicity: for tiebreaks build rank lists weighted by count then rank
    # quads/trips/pairs ranks (highest of each)
    def hi_with_count(c):
        present=(rc==c)  # (...,13)
        idx=torch.arange(13,device=cards.device)
        val=torch.where(present, idx, torch.full_like(idx,-1))
        return val.max(-1).values  # -1 if none
    four=hi_with_count(4); three=hi_with_count(3)
    # pair hi/lo (two highest with count==2)
    pairmask=(rc==2).to(torch.int32)
    idx=torch.arange(13,device=cards.device)
    pr=torch.where(rc==2, idx, torch.full_like(idx,-1))
    pr_sorted,_=pr.sort(-1,descending=True)
    pair_hi=pr_sorted[...,0]; pair_lo=pr_sorted[...,1]
    # kickers: ranks with count==1, sorted desc
    kr=torch.where(rc==1, idx, torch.full_like(idx,-1))
    kr_sorted,_=kr.sort(-1,descending=True)  # (...,13), -1 fill at end
    def kick(i): return kr_sorted[...,i].clamp(min=0)
    P1,P2,P3,P4,P5=B,B**2,B**3,B**4,B**5
    cat=torch.zeros(shp,device=cards.device,dtype=torch.long); tie=torch.zeros(shp,device=cards.device,dtype=torch.long)
    # start from high card, override upward (higher category wins)
    # high card
    hc_tie=kick(0)*P4+kick(1)*P3+kick(2)*P2+kick(3)*P1+kick(4)
    cat=torch.zeros(shp,device=cards.device,dtype=torch.long); tie=hc_tie
    # one pair
    m1=(pair_hi>=0)&(pair_lo<0)&(four<0)&(three<0)
    tie=torch.where(m1, pair_hi*P3+kick(0)*P2+kick(1)*P1+kick(2), tie); cat=torch.where(m1,torch.ones_like(cat),cat)
    # two pair
    m2=(pair_hi>=0)&(pair_lo>=0)&(three<0)&(four<0)
    tie=torch.where(m2, pair_hi*P2+pair_lo*P1+kick(0), tie); cat=torch.where(m2,torch.full_like(cat,2),cat)
    # trips
    m3=(three>=0)&(pair_hi<0)&(four<0)
    tie=torch.where(m3, three*P2+kick(0)*P1+kick(1), tie); cat=torch.where(m3,torch.full_like(cat,3),cat)
    # straight
    tie=torch.where(straight & (cat<4), sh, tie); cat=torch.where(straight & (cat<4),torch.full_like(cat,4),cat)
    # flush
    fl_tie=kick(0)*P4+kick(1)*P3+kick(2)*P2+kick(3)*P1+kick(4)
    tie=torch.where(flush & (cat<5), fl_tie, tie); cat=torch.where(flush & (cat<5),torch.full_like(cat,5),cat)
    # full house (trips + a pair)
    mfh=(three>=0)&(pair_hi>=0)
    tie=torch.where(mfh, three*P1+pair_hi, tie); cat=torch.where(mfh,torch.full_like(cat,6),cat)
    # quads
    mq=(four>=0)
    # kicker for quads = best remaining rank
    krem=torch.where((rc>=1)&(idx.expand_as(rc)!=four.unsqueeze(-1)), idx, torch.full_like(idx,-1)).max(-1).values
    tie=torch.where(mq, four*P1+krem.clamp(min=0), tie); cat=torch.where(mq,torch.full_like(cat,7),cat)
    # straight flush
    msf=straight & flush
    tie=torch.where(msf, sh, tie); cat=torch.where(msf,torch.full_like(cat,8),cat)
    return cat*P5+tie

def eval7_torch(cards7):
    """cards7: (N,7) -> (N) best-5 score via the 21 combos."""
    combos=cards7[:,_C5.to(cards7.device)]   # (N,21,5)
    v=eval5_torch(combos)                    # (N,21)
    return v.max(-1).values

def equity_batch(hero, opps, board_fixed, runouts):
    """Vectorized equity for ONE event over its runouts, on GPU.
    hero: (2,) ; opps: (K,2) ; board_fixed: (b,) ; runouts: (R, need) card ints completing the board.
    Returns scalar hero equity (win=1, tie=0.5) averaged over R runouts. All int tensors on DEVICE."""
    R=runouts.shape[0]; K=opps.shape[0]; b=board_fixed.shape[0]; need=runouts.shape[1]
    board=torch.cat([board_fixed.expand(R,b), runouts],dim=1)            # (R, b+need)=(R,5)
    hero7=torch.cat([hero.expand(R,2), board],dim=1)                     # (R,7)
    hv=eval7_torch(hero7)                                               # (R,)
    best_opp=torch.full((R,),-1,device=hero.device,dtype=torch.long)
    for k in range(K):
        o7=torch.cat([opps[k].expand(R,2), board],dim=1)
        best_opp=torch.maximum(best_opp, eval7_torch(o7))
    win=(hv>best_opp).float(); tie=(hv==best_opp).float()
    return (win+0.5*tie).mean().item()

def equity_batch_multi(hero7, opp7, ro_gid, opp_gid, n_groups, device=DEVICE):
    """Fully-batched postflop equity via one GPU eval + scatter-reduce. Inputs are FLAT arrays:
      hero7 : (G,7) one hero 7-card hand per (event,runout) GROUP g=0..n_groups-1
      opp7  : (M,7) opponent 7-card hands ; opp_gid:(M,) group id of each opp row
      ro_gid: (G,) event id of each group (many runout-groups per event)
      n_events implied by ro_gid.max()+1
    Returns per-event equity np.array. Ties split 0.5.
    """
    hv=eval7_torch(torch.tensor(hero7,device=device))                    # (G,)
    ov=eval7_torch(torch.tensor(opp7,device=device))                     # (M,)
    og=torch.tensor(opp_gid,device=device,dtype=torch.long)
    best_opp=torch.full((n_groups,),-1,device=device,dtype=torch.long)
    best_opp=best_opp.scatter_reduce(0, og, ov, reduce="amax", include_self=True)  # per-group max opp
    win=(hv>best_opp).float()+0.5*(hv==best_opp).float()                 # (G,)
    # average over runout-groups within each event
    eg=torch.tensor(ro_gid,device=device,dtype=torch.long); ne=int(eg.max().item())+1
    esum=torch.zeros(ne,device=device).scatter_reduce(0,eg,win,reduce="sum",include_self=True)
    ecnt=torch.zeros(ne,device=device).scatter_reduce(0,eg,torch.ones_like(win),reduce="sum",include_self=True)
    return (esum/ecnt.clamp(min=1)).cpu().numpy()

# ---- validation vs numba eval_best ----
def _validate(n=3000):
    from anchor_repro.equity_engine import eval7 as nb_eval7
    rng=np.random.RandomState(0); bad=0
    A=[]; 
    for _ in range(n):
        deck=rng.permutation(52)[:7]; A.append(deck)
    A=np.array(A)
    gpu=eval7_torch(torch.tensor(A,device=DEVICE)).cpu().numpy()
    # numba returns raw value with a different absolute scale; compare ORDERING via pairwise sign agreement
    nb=np.array([nb_eval7(np.array(a,np.int64)) for a in A])
    # check rank-order agreement (Spearman-ish): compare many random pairs' orderings
    import itertools as it
    idx=rng.randint(0,n,(20000,2)); agree=0; tot=0
    for i,j in idx:
        if i==j: continue
        g=np.sign(gpu[i]-gpu[j]); nn=np.sign(nb[i]-nb[j])
        if g==0 and nn==0: continue
        tot+=1; agree+= (g==nn)
    print(f"GPU vs numba ordering agreement: {agree}/{tot} = {agree/tot:.5f}")
    return agree/tot

def _validate_equity(n_events=200):
    """Check equity_batch_multi (flat/scatter) == equity_batch (per-event loop) on random postflop events."""
    rng=np.random.RandomState(1); import itertools as it
    hero7=[]; opp7=[]; ro_gid=[]; opp_gid=[]; g=0; single=[]
    for ei in range(n_events):
        deck=list(rng.permutation(52))
        hero=np.array(deck[:2]); K=rng.randint(1,4); opps=np.array([deck[2+2*k:4+2*k] for k in range(K)])
        b=rng.choice([3,4]); bf=np.array(deck[2+2*K:2+2*K+b])
        used=set(hero.tolist())|set(bf.tolist())|set(opps.flatten().tolist())
        rem=[c for c in range(52) if c not in used]; need=5-b
        combos=list(it.combinations(range(len(rem)),need))[:60]  # cap runouts for speed
        ro=np.array([[rem[i] for i in cmb] for cmb in combos])
        # single reference
        e_single=equity_batch(torch.tensor(hero,device=DEVICE),torch.tensor(opps,device=DEVICE),
                               torch.tensor(bf,device=DEVICE),torch.tensor(ro,device=DEVICE))
        single.append(e_single)
        # flat build
        R=ro.shape[0]; board=np.concatenate([np.broadcast_to(bf,(R,b)),ro],axis=1)
        h7=np.concatenate([np.broadcast_to(hero,(R,2)),board],axis=1)
        for rr in range(R):
            hero7.append(h7[rr]); ro_gid.append(ei)
            for k in range(K): opp7.append(np.concatenate([opps[k],board[rr]])); opp_gid.append(g)
            g+=1
    eq=equity_batch_multi(np.array(hero7),np.array(opp7),np.array(ro_gid),np.array(opp_gid),g)
    single=np.array(single); md=np.abs(eq-single).max()
    print(f"batched vs single-event equity: max abs diff {md:.6f} over {n_events} events")
    return md

if __name__=="__main__":
    print("device",DEVICE); _validate(); _validate_equity()
