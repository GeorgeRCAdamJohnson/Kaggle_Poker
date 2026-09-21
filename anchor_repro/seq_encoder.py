"""Self-supervised sequence encoder over pair action streams (GPU). Masked-action pretrain -> per-pair embedding.

372 labels are too few to train a transformer supervised, so we PRETRAIN on ALL 235k pair documents
(dev+eval pooled, label-free) via masked-action prediction. The encoder learns coordination structure
from the raw ordered stream. Then we extract a fixed per-pair embedding (mean+max pool of the final
hidden states) for the supervised head.

Small transformer: d_model=96, 3 layers, 4 heads, ~1M params. torch cuda:0. ~few min on RTX 5070.
Run:  python -m anchor_repro.seq_encoder
"""
from __future__ import annotations
import json, time, math, numpy as np, polars as pl, torch, torch.nn as nn
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
V=json.loads((SEQ/"seq_vocab.json").read_text()); VOCAB=V["vocab_size"]; PAD=0; MASK=VOCAB  # extra id for mask
MAXLEN=V["max_len"]
D_MODEL,N_LAYERS,N_HEADS,FFN=96,3,4,192
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def load(path):
    df=pl.read_parquet(path)
    toks=df["tokens"].to_list(); ids=df["pair_id"].to_list()
    return ids, toks

def pad_batch(seqs, maxlen=MAXLEN):
    B=len(seqs); L=min(maxlen, max(len(s) for s in seqs))
    x=np.zeros((B,L),np.int64)
    for i,s in enumerate(seqs):
        s=s[:L]; x[i,:len(s)]=s
    return torch.from_numpy(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB+1,D_MODEL,padding_idx=PAD)
        self.pos=nn.Embedding(MAXLEN,D_MODEL)
        layer=nn.TransformerEncoderLayer(D_MODEL,N_HEADS,FFN,dropout=0.1,batch_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(layer,N_LAYERS)
        self.head=nn.Linear(D_MODEL,VOCAB)   # masked-action prediction
    def forward(self,x,return_hidden=False):
        pad_mask=(x==PAD)
        pos=torch.arange(x.size(1),device=x.device).unsqueeze(0)
        h=self.emb(x)+self.pos(pos)
        h=self.enc(h,src_key_padding_mask=pad_mask)
        if return_hidden:
            return h,pad_mask
        return self.head(h)

def main():
    torch.manual_seed(42); np.random.seed(42)
    dids,dtok=load(SEQ/"seq_tokens_dev.parquet")
    eids,etok=load(SEQ/"seq_tokens_eval.parquet")
    all_tok=dtok+etok
    log(f"pretrain corpus: {len(all_tok):,} pair docs, device={DEVICE}")
    model=Encoder().to(DEVICE)
    n_params=sum(p.numel() for p in model.parameters()); log(f"model params {n_params/1e6:.2f}M")
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
    lossf=nn.CrossEntropyLoss(ignore_index=-100)
    BS=256; EPOCHS=6; idx=np.arange(len(all_tok))
    model.train()
    for ep in range(EPOCHS):
        np.random.shuffle(idx); tot=0.0; nb=0; correct=0; seen=0
        for b in range(0,len(idx),BS):
            bi=idx[b:b+BS]; x=pad_batch([all_tok[i] for i in bi]).to(DEVICE)
            # mask 15% of non-pad tokens
            probmat=torch.rand(x.shape,device=DEVICE); mask=(probmat<0.15)&(x!=PAD)
            if mask.sum()==0: continue
            target=x.clone(); target[~mask]=-100
            xin=x.clone(); xin[mask]=MASK
            logits=model(xin)
            loss=lossf(logits.reshape(-1,VOCAB),target.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item(); nb+=1
            with torch.no_grad():
                pred=logits.argmax(-1); correct+=((pred==x)&mask).sum().item(); seen+=mask.sum().item()
        log(f"epoch {ep}: masked-CE {tot/max(nb,1):.3f}  masked-acc {correct/max(seen,1):.3f}  (chance ~{1/VOCAB:.4f})")

    # extract per-pair embeddings (mean+max pool over non-pad hidden states)
    model.eval()
    def embed(ids,toks):
        embs=np.zeros((len(toks),2*D_MODEL),np.float32)
        with torch.no_grad():
            for b in range(0,len(toks),512):
                x=pad_batch(toks[b:b+512]).to(DEVICE)
                h,pad_mask=model(x,return_hidden=True)
                h=h.masked_fill(pad_mask.unsqueeze(-1),0.0)
                cnt=(~pad_mask).sum(1,keepdim=True).clamp(min=1)
                mean=h.sum(1)/cnt
                mx=h.masked_fill(pad_mask.unsqueeze(-1),-1e9).max(1).values
                embs[b:b+512]=torch.cat([mean,mx],-1).cpu().numpy()
        return embs
    log("extracting embeddings...")
    de=embed(dids,dtok); ee=embed(eids,etok)
    cols=[f"seq_{i}" for i in range(2*D_MODEL)]
    pl.DataFrame({"pair_id":dids,**{cols[i]:de[:,i] for i in range(len(cols))}}).write_parquet(SEQ/"seq_emb_dev.parquet")
    pl.DataFrame({"pair_id":eids,**{cols[i]:ee[:,i] for i in range(len(cols))}}).write_parquet(SEQ/"seq_emb_eval.parquet")
    log(f"wrote seq embeddings: dev {de.shape}, eval {ee.shape}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
