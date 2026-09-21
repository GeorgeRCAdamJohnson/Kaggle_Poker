"""CONTEXTUALIZED per-hand embedding for EVIDENCE (§149, the correct construction).

§148 refuted the naive version (embed each hand's isolated 4-token slice -> OOD -> hurts). The correct
construction: run the FULL pair document through the encoder ONCE (in-distribution, ~180 tokens) and
pool the per-TOKEN hidden states by their hand, giving each hand a representation informed by the whole
pair trajectory (which hand stands out AGAINST this pair's baseline = the evidence question).

Reuses: seq_tokenize's EXACT pair-doc ordering (so it matches what the encoder trained on), the trained
v2 encoder (_v2_encoder.pt, hidden()), and seq_evidence's MAP@5 harness (build_dh, run_ranker, map5).
The ONE new piece is a token->hand_idx alignment track parallel to the pair doc.

Run:  python -m anchor_repro.seq_ctx_evidence
"""
from __future__ import annotations
import json, time, numpy as np, polars as pl, torch
from pathlib import Path
from anchor_repro.seq_evidence import Encoder, build_dh, run_ranker, HAND_FEATS, map5, log, DEVICE, SEQ, STEP3, D_MODEL, MAXLEN, PAD
ACTIONS=["fold","check","call","bet","raise","all_in"]
AC=STEP3/"action_context.parquet"
MAX_HANDS=60; MAX_LEN=512
CTXCOLS=[f"hctx_{i}" for i in range(D_MODEL)]   # per-token hidden is D_MODEL (192), pooled per hand

def build_aligned(pair_hands_path, phase, out_path):
    """Replicate seq_tokenize.build EXACTLY but emit a parallel hand_idx track aligned to each token
    (BOS -> -1). Output: pair_id, tokens, hand_idxs (same length, after BOS-prepend + head(MAX_LEN))."""
    if out_path.exists(): log(f"{phase} aligned tokens cached"); return
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","hand_idx","player_1","player_2"])
    ac=(pl.scan_parquet(AC).filter(pl.col("phase")==phase)
        .select(["hand_id","action_no","street_no","player_id","action","is_aggr","last_aggr","to_call","amount_pot_ratio"]))
    m1=ph.select(["pair_id","hand_id","hand_idx",pl.col("player_1").alias("player_id"),pl.col("player_2").alias("partner")]).with_columns(pl.lit(0,dtype=pl.Int64).alias("role"))
    m2=ph.select(["pair_id","hand_id","hand_idx",pl.col("player_2").alias("player_id"),pl.col("player_1").alias("partner")]).with_columns(pl.lit(1,dtype=pl.Int64).alias("role"))
    members=pl.concat([m1,m2])
    j=(members.lazy().join(ac,on=["hand_id","player_id"],how="inner")
       .with_columns(
           pl.col("action").replace_strict({a:i for i,a in enumerate(ACTIONS)},default=1).alias("act"),
           pl.col("street_no").clip(0,3).alias("street"),
           pl.when(pl.col("to_call")<=0).then(2).when(pl.col("last_aggr")==pl.col("partner")).then(0).otherwise(1).alias("facing"),
           pl.col("amount_pot_ratio").fill_null(0).alias("apr"),
       )
       .with_columns(pl.when(pl.col("apr")<=0).then(0).when(pl.col("apr")<0.5).then(1).when(pl.col("apr")<1.0).then(2).otherwise(3).alias("size"))
       .with_columns((10+(((pl.col("role")*6+pl.col("act"))*4+pl.col("street"))*3+pl.col("facing"))*4+pl.col("size")).alias("token"))
       .select(["pair_id","hand_id","hand_idx","action_no","token"])
       .collect(engine="streaming"))
    j=j.sort(["pair_id","hand_idx","action_no"])
    # KEEP ALL HANDS (no MAX_HANDS cap) -> windowed embedding covers hands the single-doc window drops (§149 headroom)
    # per-hand token list AND matching hand_idx list, then flatten in hand_idx order (matches seq_tokenize)
    perhand=(j.group_by(["pair_id","hand_idx"]).agg(
                pl.col("token").sort_by("action_no").alias("_tok"),
                pl.col("token").len().alias("_n")).sort(["pair_id","hand_idx"]))
    perhand=perhand.with_columns(pl.col("hand_idx").repeat_by(pl.col("_n")).alias("_hidx"))  # hand_idx repeated per token
    # FULL untruncated token + aligned hand_idx lists per pair (windowing happens at embed time)
    doc=(perhand.group_by("pair_id",maintain_order=True).agg(
            pl.col("_tok").flatten().alias("tokens"),
            pl.col("_hidx").flatten().alias("hand_idxs")))
    docs=doc.select(["pair_id","tokens","hand_idxs"])
    docs.write_parquet(out_path)
    log(f"{phase}: {docs.height:,} aligned pair docs (full length, mean {docs['tokens'].list.len().mean():.0f} tok)")

WIN=MAXLEN-1   # tokens per window (leave room for BOS); MAXLEN=512

def _windows(tok, hidx):
    """Tile a pair's full token stream into overlapping <=WIN windows, aligned to HAND boundaries so a
    hand is never split. Returns list of (win_tokens_with_BOS, win_hidx_with_-1). 50% stride overlap so
    interior hands appear with full left+right context in some window."""
    n=len(tok)
    if n<=WIN:
        return [([1]+list(tok),[-1]+list(hidx))]
    # hand start offsets
    hidx=np.asarray(hidx); starts=[0]+[k for k in range(1,n) if hidx[k]!=hidx[k-1]]+[n]
    wins=[]; s=0
    while s<n:
        # extend to the last hand boundary that fits in WIN
        end=s
        for j in range(len(starts)-1):
            if starts[j]>=s and starts[j+1]-s<=WIN: end=starts[j+1]
        if end<=s: end=min(s+WIN,n)   # single monster hand > WIN: hard cut
        wins.append(([1]+list(tok[s:end]),[-1]+list(hidx[s:end])))
        if end>=n: break
        # stride ~50% of a full window, snapped to a hand boundary
        target=s+WIN//2; ns=next((st for st in starts if st>=target),end)
        s=ns if ns>s else end
    return wins

def embed_ctx(phase, aligned_path, out_path, keep_pairs=None):
    """Full pair stream -> OVERLAPPING WINDOWS (<=512 tok, hand-aligned) -> encoder.hidden() -> per-token
    states -> mean-pool by hand -> per-hand contextualized embedding. Guarantees every hand is covered
    (§149 headroom: single-doc window dropped 61% of evidence hands). Output: pair_id, hand_idx, hctx_*.

    PERF (why the earlier versions were CPU-bound): the naive per-hand Python loop and the per-batch
    torch.unique + 3x .cpu() syncs both stalled the GPU. Here EVERYTHING accumulates into TWO PERSISTENT
    GPU BUFFERS (sum[total_hands,D], count[total_hands]) via a single index_add_ per batch, with the
    token->global-hand-row map PRECOMPUTED ON THE HOST ONCE. No per-batch unique, no per-batch transfer;
    one .cpu() at the very end. A hand appearing in 2 overlapping windows simply contributes its tokens
    twice (equal-weight mean over all its token occurrences) -> no best-window bookkeeping needed and
    verified to reproduce the dev MAP@5 exactly."""
    if out_path.exists(): log(f"{phase} ctx emb cached"); return
    df=pl.read_parquet(aligned_path)
    if keep_pairs is not None:
        df=df.filter(pl.col("pair_id").is_in(list(keep_pairs)))
        log(f"{phase}: restricted to {df.height:,} pairs")
    model=Encoder().to(DEVICE); st=torch.load(SEQ/"_v2_encoder.pt",map_location=DEVICE)
    model.load_state_dict(st["model"]); model.eval()
    pids=df["pair_id"].to_list(); toks=df["tokens"].to_list(); hidxs=df["hand_idxs"].to_list()
    # ---- host-side prep (ONCE): assign every (pair,hand) a fixed global row id; build windows carrying
    #      per-token global-row ids so pooling is one flat scatter with NO unique needed ----
    row_of={}; row_pid=[]; row_hidx=[]
    def rid(pr,hv):
        k=(pr,hv); r=row_of.get(k)
        if r is None:
            r=len(row_pid); row_of[k]=r; row_pid.append(pids[pr]); row_hidx.append(hv)
        return r
    win_tok=[]; win_grp=[]   # each: np.array per window of token ids / global-row ids (-1 for BOS)
    for pr in range(len(toks)):
        for wt,wh in _windows(toks[pr],hidxs[pr]):
            g=[(-1 if hv<0 else rid(pr,int(hv))) for hv in wh]
            win_tok.append(np.asarray(wt,np.int64)); win_grp.append(np.asarray(g,np.int64))
    n_rows=len(row_pid); nwin=len(win_tok)
    wlen=np.asarray([t.shape[0] for t in win_tok],np.int64)
    # LENGTH-SORTED batching: each batch pads to its own max length (most windows are short, mean ~4 tok/
    # hand); padding every batch to the global max wasted ~50x compute on the attention. Accumulation is
    # by GLOBAL row id (order-independent) so we can process windows in any order. This is the real fix.
    order=np.argsort(wlen,kind="stable")
    log(f"{phase}: {len(toks):,} pairs -> {nwin:,} windows, {n_rows:,} (pair,hand) rows (ckpt ep{st['ep']}) [persistent-GPU, len-sorted]")
    D=D_MODEL
    sumb=torch.zeros(n_rows,D,device=DEVICE); cntb=torch.zeros(n_rows,device=DEVICE)
    B=256
    # RESUMABLE: checkpoint the accumulator every CKPT_EVERY batches. Window order is deterministic
    # (stable argsort) and accumulation is order-free, so resuming from a saved batch index is EXACT.
    ck=out_path.with_suffix(".accum.pt"); start_b=0
    if ck.exists():
        st2=torch.load(ck,map_location=DEVICE)
        if st2["nwin"]==nwin and st2["n_rows"]==n_rows:
            sumb=st2["sumb"].to(DEVICE); cntb=st2["cntb"].to(DEVICE); start_b=int(st2["next_b"])
            log(f"{phase}: RESUMED accumulator from batch {start_b:,}/{nwin:,}")
        else:
            log(f"{phase}: stale accum ckpt (shape mismatch) -> fresh start"); ck.unlink()
    CKPT_EVERY=400  # batches (~100k windows) between checkpoints
    t_loop=time.time()
    with torch.no_grad():
        for bi,b in enumerate(range(start_b,nwin,B)):
            oi=order[b:b+B]; nb=len(oi); L=int(min(MAXLEN,wlen[oi].max()))
            x=np.zeros((nb,L),np.int64); g=np.full((nb,L),-1,np.int64)
            for i,wi in enumerate(oi):
                t=win_tok[wi]; ln=min(L,t.shape[0]); x[i,:ln]=t[:ln]; g[i,:ln]=win_grp[wi][:ln]
            h,_=model.hidden(torch.from_numpy(x).to(DEVICE,non_blocking=True))    # (nb,L,D) GPU
            gt=torch.from_numpy(g).to(DEVICE,non_blocking=True).reshape(-1)
            hv=h.reshape(-1,D); m=gt>=0
            idx=gt[m]
            sumb.index_add_(0,idx,hv[m])
            cntb.index_add_(0,idx,torch.ones_like(idx,dtype=sumb.dtype))
            if bi>0 and bi%CKPT_EVERY==0:
                torch.save({"sumb":sumb,"cntb":cntb,"next_b":b+B,"nwin":nwin,"n_rows":n_rows},ck)
                log(f"  {phase} ckpt @ {b+nb:,}/{nwin:,}")
            if (b//B)%40==0:
                done=b+nb; rate=(done-start_b)/max(time.time()-t_loop,1e-9)
                log(f"  {phase} {done:,}/{nwin:,} windows ({rate:.0f} win/s, ETA {(nwin-done)/max(rate,1e-9):.0f}s)")
    emb=(sumb/cntb.clamp(min=1).unsqueeze(1)).cpu().numpy().astype(np.float16)   # ONE transfer; float16 (feeds LGB, precision fine) halves RAM
    del sumb,cntb,win_tok,win_grp,row_of
    import gc; gc.collect()
    if DEVICE.startswith("cuda"): torch.cuda.empty_cache()
    # CHUNKED parquet write (a 9.6M x 194 frame OOMs on 15GB RAM). Build+write row-chunks via pyarrow.
    import pyarrow as pa, pyarrow.parquet as papq
    pid_arr=np.asarray(row_pid); hidx_arr=np.asarray(row_hidx,np.int32)
    CH=1_000_000; writer=None
    try:
        for s in range(0,emb.shape[0],CH):
            e=min(s+CH,emb.shape[0])
            cols={"pair_id":pa.array(pid_arr[s:e]),"hand_idx":pa.array(hidx_arr[s:e])}
            for i in range(len(CTXCOLS)):
                cols[CTXCOLS[i]]=pa.array(emb[s:e,i])   # float16
            tbl=pa.table(cols)
            if writer is None: writer=papq.ParquetWriter(str(out_path),tbl.schema)
            writer.write_table(tbl)
    finally:
        if writer is not None: writer.close()
    if ck.exists(): ck.unlink()   # embed complete -> drop the resume checkpoint
    log(f"{phase}: wrote ctx per-hand emb {emb.shape} float16 ({len(set(row_pid)):,} pairs)")

def main():
    dh=build_dh()
    labelled=set(dh["pair_id"].to_list())
    build_aligned(STEP3/"dev_pair_hands.parquet","development",SEQ/"seq_aligned_dev.parquet")
    embed_ctx("development",SEQ/"seq_aligned_dev.parquet",SEQ/"seq_ctx_emb_dev.parquet",keep_pairs=labelled)
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    # join by (pair_id, hand_idx) — the aligned track is keyed by hand_idx, dh has hand_idx too
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left")
    dhp=dh.to_pandas()
    miss=dhp[CTXCOLS[0]].isna().sum()
    log(f"labelled hands {len(dhp):,}, missing ctx emb {miss:,} (fill 0)")
    dhp[CTXCOLS]=dhp[CTXCOLS].fillna(0.0).astype("float32")
    dhp[HAND_FEATS]=dhp[HAND_FEATS].astype("float32")
    log("=== CONTEXTUALIZED per-hand emb — OOF MAP@5 (baseline A=0.5118 in-harness, 0.5276 pipeline) ===")
    run_ranker(dhp,HAND_FEATS,"A HAND_FEATS only")
    run_ranker(dhp,HAND_FEATS+CTXCOLS,"B HAND_FEATS + ctx_emb")
    run_ranker(dhp,CTXCOLS,"C ctx_emb only")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
