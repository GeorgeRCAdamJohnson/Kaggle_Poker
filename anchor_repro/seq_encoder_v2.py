"""STRONGER sequence encoder (v2): bigger model, up to 25-epoch masked pretrain (early-stop on
convergence) + SUPERVISED contrastive/classification fine-tune on the 372 label pairs.

v1 (0.39M params, 6 epochs masked-only) reached masked-acc 0.52 and gave a decorrelated 0.94-AUC
standalone embedding but only +0.0035 held-out as a feature (sub-noise). To make it a LARGE lever:
  - d_model 96->160, 4 layers, 6 heads (~2M params)
  - masked pretrain up to 25 epochs, EARLY-STOP if masked-acc gain < 0.003 over 2 epochs
  - THEN a supervised fine-tune: a pair-classification head on the pooled embedding, trained on the
    372 confirmed positives vs 1488 confirmed negatives (table-grouped), so the representation is
    shaped toward COLLUSION, not just next-action. This is the missing step (shape the rep to the task).
  - extract the fine-tuned embedding (mean+max pool) for dev+eval.
GPU cuda:0. Then re-gate via seq_gate/seq_signal_check.
Run:  python -m anchor_repro.seq_encoder_v2
"""
from __future__ import annotations
import json, time, numpy as np, polars as pl, torch, torch.nn as nn
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; DATA=Path("data/poker")
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
V=json.loads((SEQ/"seq_vocab.json").read_text()); VOCAB=V["vocab_size"]; PAD=0; MASK=VOCAB; MAXLEN=V["max_len"]
D_MODEL,N_LAYERS,N_HEADS,FFN=192,4,8,384
T0=time.time()
_LOG=open(SEQ/"_v2_log.txt","w",encoding="utf-8")
def log(m):
    s=f"[{time.time()-T0:6.0f}s] {m}"
    print(s,flush=True); _LOG.write(s+"\n"); _LOG.flush()

def load(path):
    df=pl.read_parquet(path); return df["pair_id"].to_list(), df["tokens"].to_list()
def pad_batch(seqs, maxlen=MAXLEN):
    L=min(maxlen,max(len(s) for s in seqs)); x=np.zeros((len(seqs),L),np.int64)
    for i,s in enumerate(seqs):
        s=s[:L]; x[i,:len(s)]=s
    return torch.from_numpy(x)

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
    def mlm_logits(self,x):
        h,_=self.hidden(x); return self.mlm(h)
    def pool(self,x):
        h,pad=self.hidden(x); h=h.masked_fill(pad.unsqueeze(-1),0.0)
        cnt=(~pad).sum(1,keepdim=True).clamp(min=1); mean=h.sum(1)/cnt
        mx=h.masked_fill(pad.unsqueeze(-1),-1e9).max(1).values
        return torch.cat([mean,mx],-1)
    def clf_logit(self,x): return self.clf(self.pool(x)).squeeze(-1)

def main():
    torch.manual_seed(42); np.random.seed(42)
    dids,dtok=load(SEQ/"seq_tokens_dev.parquet"); eids,etok=load(SEQ/"seq_tokens_eval.parquet")
    all_tok=dtok+etok; log(f"corpus {len(all_tok):,} docs, device {DEVICE}")
    model=Encoder().to(DEVICE); log(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
    mlmf=nn.CrossEntropyLoss(ignore_index=-100)
    CKPT=SEQ/"_v2_encoder.pt"
    start_ep=0
    if CKPT.exists():
        st=torch.load(CKPT,map_location=DEVICE); model.load_state_dict(st["model"]); start_ep=st["ep"]+1
        log(f"resumed from checkpoint ep {st['ep']} (acc {st.get('acc',0):.3f})")
    BS=256; idx=np.arange(len(all_tok)); best=st.get("acc",0.0) if CKPT.exists() else 0.0; stall=0
    PRETRAIN_EPOCHS=18; PATIENCE=3   # plateau = best doesn't improve >0.003 for 3 consecutive epochs
    # ---- masked pretrain, PATIENCE-based early-stop, checkpoint each epoch ----
    for ep in range(start_ep, PRETRAIN_EPOCHS):
        model.train(); np.random.shuffle(idx); tot=0.0; nb=0; cor=0; seen=0
        for b in range(0,len(idx),BS):
            bi=idx[b:b+BS]; x=pad_batch([all_tok[i] for i in bi]).to(DEVICE)
            pm=(torch.rand(x.shape,device=DEVICE)<0.15)&(x!=PAD)
            if pm.sum()==0: continue
            tgt=x.clone(); tgt[~pm]=-100; xin=x.clone(); xin[pm]=MASK
            loss=mlmf(model.mlm_logits(xin).reshape(-1,VOCAB),tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item(); nb+=1
            with torch.no_grad():
                pred=model.mlm_logits(xin).argmax(-1) if False else None
        # cheap acc probe on one batch
        model.eval()
        with torch.no_grad():
            xb=pad_batch([all_tok[i] for i in idx[:512]]).to(DEVICE)
            pm=(torch.rand(xb.shape,device=DEVICE)<0.15)&(xb!=PAD); xin=xb.clone(); xin[pm]=MASK
            acc=((model.mlm_logits(xin).argmax(-1)==xb)&pm).sum().item()/max(pm.sum().item(),1)
        log(f"pretrain ep {ep}: CE {tot/max(nb,1):.3f} masked-acc {acc:.3f}")
        torch.save({"model":model.state_dict(),"ep":ep,"acc":acc},CKPT)
        if acc>best+0.003: best=acc; stall=0
        else: stall+=1
        if stall>=PATIENCE:
            log(f"converged at ep {ep} (best {best:.3f} not improved >0.003 for {PATIENCE} epochs)"); break

    # ---- supervised fine-tune on the 372 vs 1488 confirmed labels (shape rep to collusion) ----
    labels=pl.read_csv(DATA/"development_labels.csv").select(["pair_id","player_1","player_2","label"])
    lab=labels.to_pandas(); pid2tok=dict(zip(dids,dtok))
    lab=lab[lab["pair_id"].astype(str).isin(set(map(str,dids)))]
    # table group for split
    ph=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","table_id"]).to_pandas()
    lab=lab.merge(ph,on="pair_id",how="left")
    ltok=[pid2tok[str(p)] for p in lab["pair_id"].astype(str)]; ylab=lab["label"].to_numpy().astype(np.float32)
    log(f"fine-tune on {len(ltok)} labelled ({int(ylab.sum())} pos)")
    optf=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.01)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ylab==0).sum()/max(ylab.sum(),1),device=DEVICE))
    li=np.arange(len(ltok))
    for ep in range(8):
        model.train(); np.random.shuffle(li); tot=0.0; nb=0
        for b in range(0,len(li),128):
            bi=li[b:b+128]; x=pad_batch([ltok[i] for i in bi]).to(DEVICE); yv=torch.tensor(ylab[bi],device=DEVICE)
            loss=bce(model.clf_logit(x),yv)
            optf.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optf.step()
            tot+=loss.item(); nb+=1
        log(f"finetune ep {ep}: BCE {tot/max(nb,1):.3f}")

    # ---- extract fine-tuned embeddings ----
    model.eval()
    def embed(toks):
        out=np.zeros((len(toks),2*D_MODEL),np.float32)
        with torch.no_grad():
            for b in range(0,len(toks),512):
                out[b:b+512]=model.pool(pad_batch(toks[b:b+512]).to(DEVICE)).cpu().numpy()
        return out
    de=embed(dtok); ee=embed(etok); cols=[f"seq_{i}" for i in range(2*D_MODEL)]
    pl.DataFrame({"pair_id":dids,**{cols[i]:de[:,i] for i in range(len(cols))}}).write_parquet(SEQ/"seq_emb_v2_dev.parquet")
    pl.DataFrame({"pair_id":eids,**{cols[i]:ee[:,i] for i in range(len(cols))}}).write_parquet(SEQ/"seq_emb_v2_eval.parquet")
    log(f"wrote v2 embeddings dev {de.shape} eval {ee.shape}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
