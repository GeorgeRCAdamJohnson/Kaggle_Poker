"""EVIDENCE via the trained sequence encoder, per-hand (task #2).

The evidence sub-metric (20% of the final) is a WITHIN-PAIR AP@5: for each confirmed positive pair,
rank its shared hands and submit the top-5; score against the true evidence hands. Because it is a
within-pair ranking it is base-rate-free and phase-drift-immune -> no gate needed, just local MAP@5.

Baseline (hosen42 per-hand LightGBM on engineered HAND_FEATS) OOF MAP@5 = 0.5276.

This module:
  1. embeds every (pair,hand) token doc through the trained v2 encoder (pool = mean|max, 384-dim);
  2. rebuilds the labelled per-hand frame `dh` (is_evidence target, table folds) EXACTLY as the pipeline;
  3. joins the per-hand sequence embedding, then runs the SAME OOF LightGBM ranker under three feature
     sets: (A) HAND_FEATS only [baseline], (B) HAND_FEATS + seq emb, (C) seq emb only;
  4. reports overall + per-family MAP@5 with map5_within_pairs (host formula) so we can see the lift.

Run:  python -m anchor_repro.seq_evidence
"""
from __future__ import annotations
import json, time, os, numpy as np, polars as pl, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; DATA=Path("data/poker")
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
V=json.loads((SEQ/"seq_vocab.json").read_text()); VOCAB=V["vocab_size"]; PAD=0; MASK=VOCAB; MAXLEN=V["max_len"]
D_MODEL,N_LAYERS,N_HEADS,FFN=192,4,8,384
SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

# ---- encoder (must match seq_encoder_v2.Encoder exactly to load the checkpoint) ----
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB+1,D_MODEL,padding_idx=PAD); self.pos=nn.Embedding(MAXLEN,D_MODEL)
        layer=nn.TransformerEncoderLayer(D_MODEL,N_HEADS,FFN,dropout=0.1,batch_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(layer,N_LAYERS)
        self.mlm=nn.Linear(D_MODEL,VOCAB)
        self.clf=nn.Sequential(nn.Linear(2*D_MODEL,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,1))
    def hidden(self,x):
        pad=(x==PAD); pos=torch.arange(x.size(1),device=x.device).unsqueeze(0)
        h=self.enc(self.emb(x)+self.pos(pos),src_key_padding_mask=pad); return h,pad
    def pool(self,x):
        h,pad=self.hidden(x); h=h.masked_fill(pad.unsqueeze(-1),0.0)
        cnt=(~pad).sum(1,keepdim=True).clamp(min=1); mean=h.sum(1)/cnt
        mx=h.masked_fill(pad.unsqueeze(-1),-1e9).max(1).values
        return torch.cat([mean,mx],-1)

def pad_batch(seqs, maxlen=MAXLEN):
    L=min(maxlen,max(len(s) for s in seqs)); x=np.zeros((len(seqs),L),np.int64)
    for i,s in enumerate(seqs):
        s=s[:L]; x[i,:len(s)]=s
    return torch.from_numpy(x)

SEQCOLS=[f"hseq_{i}" for i in range(2*D_MODEL)]

def embed_perhand(phase, tok_path, out_path, keep_pairs=None):
    """Embed each (pair,hand) doc -> 384-dim. Cache pair_id,hand_id,hseq_*.
    keep_pairs: optional set of pair_ids to restrict to (dev measurement only needs labelled pairs)."""
    if out_path.exists(): log(f"{phase} per-hand emb cached"); return
    df=pl.read_parquet(tok_path)
    if keep_pairs is not None:
        df=df.filter(pl.col("pair_id").is_in(list(keep_pairs)))
        log(f"{phase}: restricted to {df.height:,} hands of {len(keep_pairs):,} pairs")
    toks=df["tokens"].to_list(); pids=df["pair_id"].to_list(); hids=df["hand_id"].to_list()
    model=Encoder().to(DEVICE); st=torch.load(SEQ/"_v2_encoder.pt",map_location=DEVICE)
    model.load_state_dict(st["model"]); model.eval()
    log(f"{phase}: embedding {len(toks):,} hands through encoder (ckpt ep{st['ep']} acc {st.get('acc',0):.3f})")
    out=np.zeros((len(toks),2*D_MODEL),np.float32); B=1024
    with torch.no_grad():
        for b in range(0,len(toks),B):
            out[b:b+B]=model.pool(pad_batch(toks[b:b+B]).to(DEVICE)).cpu().numpy()
            if (b//B)%20==0: log(f"  {phase} {b:,}/{len(toks):,}")
    pl.DataFrame({"pair_id":pids,"hand_id":hids,**{SEQCOLS[i]:out[:,i] for i in range(len(SEQCOLS))}}).write_parquet(out_path)
    log(f"{phase}: wrote per-hand emb {out.shape}")

# ---- host MAP@5 within pairs (exact, from policy_pipeline.map5_within_pairs) ----
def map5(df, score_col="hand_score", target_col="is_evidence", pair_col="pair_id"):
    vals=[]
    d=df.sort_values([pair_col,score_col,"pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    for _,g in d.groupby(pair_col,sort=False):
        rel=g[target_col].to_numpy(); n_rel=int(rel.sum())
        if n_rel==0: continue
        top=rel[:5]; hits=np.cumsum(top)
        vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(n_rel,5)))
    return float(np.mean(vals)) if vals else 0.0

HAND_RAW=["pot_bb","players_at_showdown","both_vpip","both_showdown","one_folded","partner_won_other_lost","net_gap","transfer_any","transfer_pot_ratio",
    "loser_chen","winner_chen","chen_gap","loser_stronger_preflop","loser_contrib_bb","pair_contrib_bb",
    "pair_aggr","pair_raise","pair_call","pair_check","pair_postflop_check","n_partners_aggr","max_amount_bb","last_street","pair_overbets",
    "fold_to_partner","call_partner","raise_partner","fold_to_other","call_other","raise_other",
    "hu_actions","hu_check","hu_call","hu_aggr","hu_late_check","hu_fold",
    "outsider_fold_to_pair","outsider_call_to_pair","outsider_raise_to_pair",
    "loser_cat_final","winner_cat_final","loser_rk_final","winner_rk_final","loser_rk_flop","winner_rk_flop","cat_gap_final","max_cat_final",
    "loser_beats_winner_flop","loser_beats_winner_turn","loser_beats_winner_final","loser_best_final","winner_best_final","fold_better_hand","both_made_no_aggr","winner_weak_won",
    "fold_to_partner_pf","fold_to_partner_post","call_partner_post","raise_partner_pf","raise_partner_post","outsider_fold_to_pair_pf",
    "both_vpip_a","n_vpip_a","n_pfr","loose_sum","loose_min","loose_max","tight_max","loose_pfr_sum","junk_vpip_sum","junk_vpip_min","junk_pfr_sum","vpip_resid_sum",
    "chen_max_pair","chen_min_pair","pf_faced_raise_any","n_in_blinds","second_entrant_loose","both_vpip_junk","both_vpip_one_junk","second_entrant_junk","both_surprising","junk_raise"]
HAND_FLAGS=["fold_stronger_to_partner","fold_better_to_partner","both_strong_no_raise","multi_call_partner","squeeze","squeeze_then_fold","pf_squeeze","squeeze_then_fold_pf",
    "dump","dump_better_hand","checkdown","hu_checkdown_strong","hu_check_two_pair_plus","transfer_x_call"]
HAND_TOMAX=["transfer_to_max","net_gap_to_max","pot_to_max","outsider_fold_to_max"]
PRANK_BASE=HAND_RAW+HAND_FLAGS
HAND_FEATS=HAND_RAW+HAND_FLAGS+HAND_TOMAX+[f"{c}_prank" for c in PRANK_BASE]
TARGET_BEHAVIORS=("directed_transfer","soft_play","coordinated_isolation")
HAND_PARAMS=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,
                 bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,seed=SEED,num_threads=os.cpu_count())
HAND_SEEDS=[SEED,SEED+7]; HAND_ROUNDS=400

def build_dh():
    """Rebuild the labelled per-hand frame EXACTLY as policy_pipeline does (dev_pairs cache + evidence)."""
    dev_pairs=pl.read_parquet(STEP3/"dev_pairs.parquet")
    dev_evidence=pl.read_csv(DATA/"development_evidence.csv")
    DEV_HF=sorted(STEP3.glob("dev_hand_features_*.parquet"))
    # reproduce the pipeline's table-fold assignment EXACTLY (rng=RandomState(SEED), permutation % 5)
    TABLES=sorted(dev_pairs["table_id"].unique().to_list())
    rng=np.random.RandomState(SEED)
    TABLE_FOLD={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}
    labelled_ids=set(dev_pairs.filter(pl.col("is_labeled"))["pair_id"].to_list())
    dev_hf_lab=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(labelled_ids))).collect() for f in DEV_HF])
    dev_hf_lab=dev_hf_lab.join(dev_pairs.select(["pair_id","label","behavior_family","is_labeled"]),on="pair_id")
    dev_hf_lab=dev_hf_lab.with_columns(pl.col("table_id").replace_strict(TABLE_FOLD,return_dtype=pl.Int8).alias("fold"))
    dev_hf_lab=dev_hf_lab.join(dev_evidence.select(["pair_id","hand_id",pl.lit(True).alias("is_evidence")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_evidence").fill_null(False))
    assert dev_hf_lab["is_evidence"].sum()==1817, f"evidence hands reconstructed = {dev_hf_lab['is_evidence'].sum()} != 1817"
    return dev_hf_lab

def run_ranker(dh, feats, tag):
    dh=dh.copy(); dh["hand_score"]=0.0
    pos_mask=(dh["label"]==1).to_numpy()
    folds=sorted(dh["fold"].unique())
    for f in folds:
        tr_all=(dh["fold"]!=f).to_numpy(); te=(dh["fold"]==f).to_numpy()
        tr_pos=tr_all & (dh["label"]==1).to_numpy()
        models=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr_pos,feats],dh.loc[tr_pos,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in HAND_SEEDS]
        dh.loc[te,"hand_score"]=np.mean([m.predict(dh.loc[te,feats]) for m in models],axis=0)
    pos_all=dh[pos_mask]
    overall=map5(pos_all)
    fam={fm:map5(pos_all[pos_all["behavior_family"]==fm]) for fm in TARGET_BEHAVIORS}
    log(f"[{tag}] MAP@5 = {overall:.4f}  per family: {{ {', '.join(f'{k}: {v:.4f}' for k,v in fam.items())} }}")
    return overall, fam

def main():
    dh=build_dh()
    labelled_pairs=set(dh["pair_id"].to_list())
    embed_perhand("development",SEQ/"seq_perhand_dev.parquet",SEQ/"seq_perhand_emb_dev.parquet",keep_pairs=labelled_pairs)
    emb=pl.read_parquet(SEQ/"seq_perhand_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_id"],how="left")
    dhp=dh.to_pandas()
    miss=dhp[SEQCOLS[0]].isna().sum()
    log(f"labelled hands {len(dhp):,}, missing per-hand emb {miss:,} (fill 0)")
    dhp[SEQCOLS]=dhp[SEQCOLS].fillna(0.0).astype("float32")
    dhp[HAND_FEATS]=dhp[HAND_FEATS].astype("float32")
    log("=== OOF MAP@5 comparison (baseline 0.5276) ===")
    run_ranker(dhp,HAND_FEATS,"A HAND_FEATS only")
    run_ranker(dhp,HAND_FEATS+SEQCOLS,"B HAND_FEATS + seq_emb")
    run_ranker(dhp,SEQCOLS,"C seq_emb only")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
