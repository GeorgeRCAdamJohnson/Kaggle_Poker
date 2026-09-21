"""DIVERSE 2nd sequence encoder: CAUSAL (next-action / autoregressive), for a risk ENSEMBLE (task #2).

The v2 encoder is MASKED-LM (bidirectional, BERT-style). For an ensemble to add, the 2nd encoder needs a
DIFFERENT inductive bias. Causal next-action prediction (GPT-style, left-to-right) captures DIRECTIONAL /
temporal structure (who-leads -> who-responds -> who-folds ordering) that a bidirectional masked model
smooths over. Same corpus (dev+eval POOLED -> phase-robust, §147), same tokenizer/vocab, same pooled
embedding interface, so it drops into seq_v2_assemble's distill+gate. Only the objective differs:
  - causal attention mask (position t sees <= t)
  - predict token[t+1] from token[<=t] (shift target), NO masking
  - then the SAME supervised fine-tune on 372 labels (shape rep to collusion)
Separate checkpoint _causal_encoder.pt; writes seq_emb_causal_dev/eval.parquet (mean|max pool, 384-dim).
GPU. Run:  python -m anchor_repro.seq_encoder_causal
"""
from __future__ import annotations
import json, time, numpy as np, polars as pl, torch, torch.nn as nn
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; DATA=Path("data/poker")
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
V=json.loads((SEQ/"seq_vocab.json").read_text()); VOCAB=V["vocab_size"]; PAD=0; MAXLEN=V["max_len"]
D_MODEL,N_LAYERS,N_HEADS,FFN=192,4,8,384
T0=time.time()
_LOG=open(SEQ/"_causal_log.txt","w",encoding="utf-8")
def log(m):
    s=f"[{time.time()-T0:6.0f}s] {m}"; print(s,flush=True); _LOG.write(s+"\n"); _LOG.flush()

def load(path):
    df=pl.read_parquet(path); return df["pair_id"].to_list(), df["tokens"].to_list()
def pad_batch(seqs, maxlen=MAXLEN):
    L=min(maxlen,max(len(s) for s in seqs)); x=np.zeros((len(seqs),L),np.int64)
    for i,s in enumerate(seqs):
        s=s[:L]; x[i,:len(s)]=s
    return torch.from_numpy(x)

class CausalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB+1,D_MODEL,padding_idx=PAD); self.pos=nn.Embedding(MAXLEN,D_MODEL)
        layer=nn.TransformerEncoderLayer(D_MODEL,N_HEADS,FFN,dropout=0.1,batch_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(layer,N_LAYERS)
        self.lm=nn.Linear(D_MODEL,VOCAB)
        self.clf=nn.Sequential(nn.Linear(2*D_MODEL,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,1))
    def hidden(self,x):
        pad=(x==PAD); pos=torch.arange(x.size(1),device=x.device).unsqueeze(0)
        L=x.size(1); cmask=torch.triu(torch.ones(L,L,device=x.device,dtype=torch.bool),diagonal=1)  # causal
        h=self.enc(self.emb(x)+self.pos(pos),mask=cmask,src_key_padding_mask=pad); return h,pad
    def lm_logits(self,x):
        h,_=self.hidden(x); return self.lm(h)
    def pool(self,x):
        h,pad=self.hidden(x); h=h.masked_fill(pad.unsqueeze(-1),0.0)
        cnt=(~pad).sum(1,keepdim=True).clamp(min=1); mean=h.sum(1)/cnt
        mx=h.masked_fill(pad.unsqueeze(-1),-1e9).max(1).values
        return torch.cat([mean,mx],-1)
    def clf_logit(self,x): return self.clf(self.pool(x)).squeeze(-1)

def main():
    torch.manual_seed(43); np.random.seed(43)   # different seed from v2 (diversity)
    dids,dtok=load(SEQ/"seq_tokens_dev.parquet"); eids,etok=load(SEQ/"seq_tokens_eval.parquet")
    all_tok=dtok+etok; log(f"corpus {len(all_tok):,} docs, device {DEVICE}")
    model=CausalEncoder().to(DEVICE); log(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
    lmf=nn.CrossEntropyLoss(ignore_index=-100)
    CKPT=SEQ/"_causal_encoder.pt"; start_ep=0; best=0.0; stall=0; st=None
    if CKPT.exists():
        st=torch.load(CKPT,map_location=DEVICE); model.load_state_dict(st["model"]); start_ep=st["ep"]+1
        best=st.get("acc",0.0); log(f"resumed ep {st['ep']} (next-tok acc {best:.3f})")
    # PERF: causal L×L mask + near-full VRAM made variable-length batches thrash (ep1 >2x ep0). Fix:
    # (1) BS 256->128 for headroom, (2) LENGTH-SORTED batches so each pads to its own (mostly short) L,
    # keeping the L×L causal mask small and per-batch memory stable. Order reshuffled among equal lengths.
    BS=128; PRETRAIN_EPOCHS=25; PATIENCE=3
    tlen=np.array([len(t) for t in all_tok])
    for ep in range(start_ep,PRETRAIN_EPOCHS):
        model.train(); tot=0.0; nb=0
        jitter=np.random.rand(len(all_tok))*4.0            # small noise so equal-length docs shuffle
        order=np.argsort(tlen+jitter,kind="stable")
        batch_starts=np.arange(0,len(order),BS); np.random.shuffle(batch_starts)  # shuffle batch order
        for bs in batch_starts:
            bi=order[bs:bs+BS]; x=pad_batch([all_tok[i] for i in bi]).to(DEVICE)
            if x.size(1)<2: continue
            xin=x[:,:-1]; tgt=x[:,1:].clone(); tgt[xin==PAD]=-100; tgt[tgt==PAD]=-100
            if (tgt!=-100).sum()==0: continue
            loss=lmf(model.lm_logits(xin).reshape(-1,VOCAB),tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item(); nb+=1
        model.eval()
        with torch.no_grad():
            probe=list(range(0,len(all_tok),max(1,len(all_tok)//512)))[:512]   # fixed spread-out probe
            xb=pad_batch([all_tok[i] for i in probe]).to(DEVICE)
            xin=xb[:,:-1]; tgt=xb[:,1:]; m=(xin!=PAD)&(tgt!=PAD)
            acc=((model.lm_logits(xin).argmax(-1)==tgt)&m).sum().item()/max(m.sum().item(),1)
        log(f"pretrain ep {ep}: CE {tot/max(nb,1):.3f} next-tok-acc {acc:.3f}")
        torch.save({"model":model.state_dict(),"ep":ep,"acc":acc},CKPT)
        if acc>best+0.003: best=acc; stall=0
        else: stall+=1
        if stall>=PATIENCE: log(f"converged ep {ep} (best {best:.3f})"); break

    # ---- supervised fine-tune on 372 vs 1488 labels (same as v2) ----
    labels=pl.read_csv(DATA/"development_labels.csv").select(["pair_id","player_1","player_2","label"]).to_pandas()
    pid2tok=dict(zip(dids,dtok)); labels=labels[labels["pair_id"].astype(str).isin(set(map(str,dids)))]
    ph=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id"]).to_pandas()  # not needed for split here
    ltok=[pid2tok[str(p)] for p in labels["pair_id"].astype(str)]; ylab=labels["label"].to_numpy().astype(np.float32)
    log(f"fine-tune on {len(ltok)} labelled ({int(ylab.sum())} pos)")
    optf=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.01)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ylab==0).sum()/max(ylab.sum(),1),device=DEVICE))
    li=np.arange(len(ltok))
    for ep in range(8):
        model.train(); np.random.shuffle(li); tot=0.0; nb=0
        for b in range(0,len(li),128):
            bi=li[b:b+128]; x=pad_batch([ltok[i] for i in bi]).to(DEVICE); yv=torch.tensor(ylab[bi],device=DEVICE)
            loss=bce(model.clf_logit(x),yv); optf.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optf.step()
            tot+=loss.item(); nb+=1
        log(f"finetune ep {ep}: BCE {tot/max(nb,1):.3f}")

    model.eval()
    def embed(toks):
        out=np.zeros((len(toks),2*D_MODEL),np.float32)
        with torch.no_grad():
            for b in range(0,len(toks),512): out[b:b+512]=model.pool(pad_batch(toks[b:b+512]).to(DEVICE)).cpu().numpy()
        return out
    de=embed(dtok); ee=embed(etok); cols=[f"cau_{i}" for i in range(2*D_MODEL)]
    pl.DataFrame({"pair_id":dids,**{cols[i]:de[:,i] for i in range(len(cols))}}).write_parquet(SEQ/"seq_emb_causal_dev.parquet")
    pl.DataFrame({"pair_id":eids,**{cols[i]:ee[:,i] for i in range(len(cols))}}).write_parquet(SEQ/"seq_emb_causal_eval.parquet")
    log(f"wrote causal embeddings dev {de.shape} eval {ee.shape}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
