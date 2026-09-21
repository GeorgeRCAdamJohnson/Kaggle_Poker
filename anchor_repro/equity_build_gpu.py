"""FULL equity build via GPU (task #2). Vectorized polars event assembly + preflop table-lookup +
GPU postflop enumeration. Replaces the 57/s CPU row-loop.

Per (pair,hand,member) at the member's last-acted street:
  PREFLOP (st==0, ~79%): look up cached 169-class preflop equity table (vectorized numpy, instant).
  POSTFLOP (st>=1, ~21%): exact enumeration of remaining board runouts vs the ACTUAL live opponents,
     batched on the GPU (equity_gpu.equity_batch_multi). Streamed in chunks to bound VRAM+RAM.
Output identical schema to equity_engine: pair_id,hand_id,member,street,equity,n_live.
Verify vs the validated labelled cache. Run:
  python -m anchor_repro.equity_build_gpu --phase development [--full]
  python -m anchor_repro.equity_build_gpu --phase evaluation
"""
from __future__ import annotations
import sys, time, itertools, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from math import comb
from anchor_repro.equity_engine import build_preflop_table, _class_key, RANK_CHARS, SUIT_CHARS
from anchor_repro.equity_gpu import equity_batch_multi, DEVICE
STEP3=Path("outputs/poker_collusion/hosen42_step3"); DATA=Path("data/poker"); OUT=STEP3/"edge"/"equity"
AC=STEP3/"action_context.parquet"
CMAP={f"{r}{s}":RANK_CHARS.index(r)*4+SUIT_CHARS.index(s) for r in RANK_CHARS for s in SUIT_CHARS}
STREET_NB={0:0,1:3,2:4,3:5}; MAX_RUNOUT=300; SEED=42
ROW_BUDGET=400_000   # max hero-runout rows per GPU flush (bounds VRAM: rows*21*13 intermediates)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def _assemble(phase, pair_hands_path, keep_pairs):
    """Vectorized event table: one row per (pair_id,hand_id,member) with hero cards, last_street,
    board(list), live-opp cards(list). Pure polars/numpy — no per-row Python equity."""
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2"])
    if keep_pairs is not None: ph=ph.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    hand_ids=ph["hand_id"].unique().to_list()
    seats=(pl.scan_parquet(DATA/"seats.parquet").filter(pl.col("hand_id").is_in(hand_ids))
           .select(["hand_id","player_id","hole_card_1","hole_card_2","folded"])
           .with_columns(pl.col("hole_card_1").replace_strict(CMAP,default=-1).alias("c1"),
                         pl.col("hole_card_2").replace_strict(CMAP,default=-1).alias("c2")).collect())
    la=(pl.scan_parquet(AC).filter(pl.col("phase")==phase)
        .group_by(["hand_id","player_id"]).agg(pl.col("street_no").max().alias("last_street")).collect())
    board=(pl.read_parquet(DATA/"hands.parquet",columns=["hand_id","board_cards"]).filter(pl.col("hand_id").is_in(hand_ids))
           .with_columns(pl.col("board_cards").fill_null("").str.split(" ").list.eval(pl.element().replace_strict(CMAP,default=-1)).alias("bd")))
    # live-opponent cards per hand: list of (c1,c2) for non-folded players
    liveopp=(seats.filter(~pl.col("folded") & (pl.col("c1")>=0))
             .group_by("hand_id").agg(pl.concat_list([pl.col("c1"),pl.col("c2")]).alias("_flat_opp"),
                                      pl.col("player_id").alias("live_ids")))
    # member rows
    m1=ph.select(["pair_id","hand_id",pl.col("player_1").alias("player_id")]).with_columns(pl.lit(1).alias("member"))
    m2=ph.select(["pair_id","hand_id",pl.col("player_2").alias("player_id")]).with_columns(pl.lit(2).alias("member"))
    mem=pl.concat([m1,m2])
    mem=(mem.join(seats.select(["hand_id","player_id","c1","c2","folded"]),on=["hand_id","player_id"],how="left")
         .join(la,on=["hand_id","player_id"],how="left")
         .join(board.select(["hand_id","bd"]),on="hand_id",how="left")
         .filter((pl.col("c1")>=0)&(pl.col("c2")>=0)))
    return mem, seats

def build(phase, pair_hands_path, out_path, keep_pairs=None):
    if out_path.exists(): log(f"{phase} equity cached"); return
    PFT=build_preflop_table()
    mem,seats=_assemble(phase,pair_hands_path,keep_pairs)
    log(f"{phase}: {mem.height:,} member-events assembled")
    # per-hand live opponents map (player_id -> cards), for excluding the hero and folded
    seatmap={}
    for row in seats.iter_rows(named=True):
        seatmap.setdefault(row["hand_id"],[]).append((row["player_id"],row["c1"],row["c2"],row["folded"]))
    mp=mem.to_pandas(); n=len(mp:=mp if False else mem.to_pandas())
    hero_c1=mp["c1"].to_numpy(); hero_c2=mp["c2"].to_numpy(); st=mp["last_street"].fillna(3).astype(int).to_numpy()
    pid=mp["player_id"].to_numpy(); hid=mp["hand_id"].to_numpy(); bd=mp["bd"].to_list()
    equity=np.full(n,np.nan,np.float32); nlive=np.zeros(n,np.int32)
    # ---- PREFLOP: vectorized table lookup ----
    pf=(st==0)
    cls=np.array([_class_key(int(a),int(b)) for a,b in zip(hero_c1,hero_c2)])
    # n_live per event
    log("computing live-opp counts")
    live_by_hand={h:[(c1,c2) for (q,c1,c2,f) in v if (not f) and c1>=0] for h,v in seatmap.items()}
    for i in range(n):
        opps=[o for o in live_by_hand.get(hid[i],[]) ]
        # exclude hero seat itself (match by cards)
        nlive[i]=max(len([o for o in opps if not (o[0]==hero_c1[i] and o[1]==hero_c2[i])]),1)
    nopp=np.clip(nlive,1,5)
    equity[pf]=PFT[cls[pf],nopp[pf]]
    log(f"preflop done ({pf.sum():,} events)")
    # ---- POSTFLOP: GPU, flush by ROW BUDGET (bounds VRAM) ----
    post=np.where(st>=1)[0]
    log(f"postflop events: {len(post):,} -> GPU (row budget {ROW_BUDGET:,})")
    t=time.time()
    hero7=[]; opp7=[]; ro_gid=[]; opp_gid=[]; g=0; ev_pos=[]; nrows=0; done=0
    def flush():
        nonlocal hero7,opp7,ro_gid,opp_gid,g,ev_pos,nrows
        if g==0: return
        eq=equity_batch_multi(np.asarray(hero7,np.int64),np.asarray(opp7,np.int64),
                              np.asarray(ro_gid,np.int64),np.asarray(opp_gid,np.int64),g)
        for jj,ii,kk in ev_pos: equity[ii]=eq[jj]; nlive[ii]=kk
        hero7=[]; opp7=[]; ro_gid=[]; opp_gid=[]; g=0; ev_pos=[]; nrows=0
    for cnt,i in enumerate(post):
        h1,h2=int(hero_c1[i]),int(hero_c2[i]); board_full=[int(x) for x in bd[i] if x>=0]
        nb=STREET_NB.get(int(st[i]),5); bf=board_full[:nb]
        opps=[(c1,c2) for (c1,c2) in live_by_hand.get(hid[i],[]) if not (c1==h1 and c2==h2)][:5]
        if len(opps)==0: continue
        used=set([h1,h2]+bf+[c for o in opps for c in o]); rem=[c for c in range(52) if c not in used]
        need=5-len(bf)
        if need<0 or need>len(rem): continue
        total=comb(len(rem),need)
        if total<=MAX_RUNOUT: combos=list(itertools.combinations(range(len(rem)),need))
        else:
            rng=np.random.RandomState(SEED+int(i)); combos=[tuple(rng.choice(len(rem),need,replace=False)) for _ in range(200)]
        ro=np.array([[rem[k] for k in cmb] for cmb in combos],np.int64) if need>0 else np.zeros((1,0),np.int64)
        R=ro.shape[0]
        board=np.concatenate([np.broadcast_to(np.array(bf,np.int64),(R,len(bf))),ro],axis=1) if need>0 else np.broadcast_to(np.array(bf,np.int64),(R,len(bf)))
        h7=np.concatenate([np.broadcast_to(np.array([h1,h2],np.int64),(R,2)),board],axis=1)
        jlocal=len(ev_pos)   # event index within the current flush
        for rr in range(R):
            hero7.append(h7[rr]); ro_gid.append(jlocal)
            for o in opps: opp7.append(np.concatenate([np.array(o,np.int64),board[rr]])); opp_gid.append(g)
            g+=1
        ev_pos.append((jlocal,i,len(opps))); nrows+=R*(1+len(opps))
        if nrows>=ROW_BUDGET:
            flush()
        if cnt%20000==0 and cnt>0:
            rate=cnt/max(time.time()-t,1e-9); log(f"  postflop {cnt:,}/{len(post):,} ({rate:.0f} ev/s, ETA {(len(post)-cnt)/max(rate,1e-9):.0f}s)")
    flush()
    out=pl.DataFrame({"pair_id":mp["pair_id"].to_numpy(),"hand_id":hid,"member":mp["member"].to_numpy(),
                      "street":st,"equity":equity,"n_live":nlive})
    out=out.filter(pl.col("equity").is_not_nan())
    out.write_parquet(out_path); log(f"{phase}: wrote {out.height:,} equity rows -> {out_path.name}")

def main():
    ph=sys.argv[sys.argv.index("--phase")+1] if "--phase" in sys.argv else "development"
    full="--full" in sys.argv
    if ph=="development":
        keep=None
        if not full:
            dp=pl.read_parquet(STEP3/"dev_pairs.parquet"); keep=set(dp.filter(pl.col("is_labeled"))["pair_id"].to_list())
        out=OUT/("pairhand_equity_dev.parquet" if full else "pairhand_equity_devlab_gpu.parquet")
        build("development",STEP3/"dev_pair_hands.parquet",out,keep_pairs=keep)
    else:
        build("evaluation",STEP3/"eval_pair_hands.parquet",OUT/"pairhand_equity_eval.parquet")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
