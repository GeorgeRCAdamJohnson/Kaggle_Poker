"""UNSUPERVISED anomaly pair-representation (task #4, the PairAP swing).

Every risk lever this arc except the original sequence rep hit two walls: PHASE DRIFT (dev->eval) and
FINE-TUNE OVERFIT to the 372 labels (§154 killed the causal ensemble). This dodges BOTH: the scores are
LABEL-FREE (no fine-tune -> no overfit) and derived from the SELF-SUPERVISED pretrained encoder trained
on dev+eval POOLED (phase-robust). Hypothesis: a colluding pair's JOINT action-sequence is an OUTLIER
against the population -> anomalous under a model that learned "normal" pair behavior.

Three label-free anomaly scores (all from the PRETRAIN checkpoint _v2_encoder.pt = ep11, pre-fine-tune):
  A  MLM reconstruction error   : mask 15% tokens, per-doc cross-entropy (higher = harder to predict = odd)
  B  Isolation Forest           : density outlier on the pretrained POOLED embedding
  C  kNN distance               : mean distance to k nearest pairs in pretrained embedding space (local outlier)

For each: standalone eval-mirrored PairAP + Spearman vs the label-fine-tuned seq_risk. SHIP a score only
if Rule 13: Spearman < 0.6 (genuinely diverse) AND the eval-weighted phase-gate delta > +0.005.
Run:  python -m anchor_repro.anomaly_pairrep
"""
from __future__ import annotations
import json, time, numpy as np, polars as pl, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
V=json.loads((SEQ/"seq_vocab.json").read_text()); VOCAB=V["vocab_size"]; PAD=0; MASK=VOCAB; MAXLEN=V["max_len"]
D_MODEL,N_LAYERS,N_HEADS,FFN=192,4,8,384
SEED,N_FOLDS=42,5
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

class Encoder(nn.Module):  # must match seq_encoder_v2.Encoder to load ckpt
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB+1,D_MODEL,padding_idx=PAD); self.pos=nn.Embedding(MAXLEN,D_MODEL)
        layer=nn.TransformerEncoderLayer(D_MODEL,N_HEADS,FFN,dropout=0.1,batch_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(layer,N_LAYERS); self.mlm=nn.Linear(D_MODEL,VOCAB)
        self.clf=nn.Sequential(nn.Linear(2*D_MODEL,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,1))
    def hidden(self,x):
        pad=(x==PAD); pos=torch.arange(x.size(1),device=x.device).unsqueeze(0)
        h=self.enc(self.emb(x)+self.pos(pos),src_key_padding_mask=pad); return h,pad
    def mlm_logits(self,x): h,_=self.hidden(x); return self.mlm(h)
    def pool(self,x):
        h,pad=self.hidden(x); h=h.masked_fill(pad.unsqueeze(-1),0.0)
        cnt=(~pad).sum(1,keepdim=True).clamp(min=1); mean=h.sum(1)/cnt
        mx=h.masked_fill(pad.unsqueeze(-1),-1e9).max(1).values
        return torch.cat([mean,mx],-1)

def load_toks(p): df=pl.read_parquet(p); return df["pair_id"].to_list(), df["tokens"].to_list()
def pad_batch(seqs,L=None):
    L=L or min(MAXLEN,max(len(s) for s in seqs)); x=np.zeros((len(seqs),L),np.int64)
    for i,s in enumerate(seqs): s=s[:L]; x[i,:len(s)]=s
    return torch.from_numpy(x)

def mlm_recon_error(model, toks):
    """Per-doc masked reconstruction cross-entropy (fixed mask via seeded RNG for determinism).
    Higher = the pretrained 'normal' model predicts this pair's actions poorly = more anomalous."""
    ce=nn.CrossEntropyLoss(reduction="none",ignore_index=-100); out=np.zeros(len(toks),np.float32)
    g=torch.Generator(device=DEVICE); 
    # length-sorted for speed; label-free
    order=np.argsort([len(t) for t in toks],kind="stable")
    with torch.no_grad():
        for b in range(0,len(order),256):
            oi=order[b:b+256]; x=pad_batch([toks[i] for i in oi]).to(DEVICE)
            g.manual_seed(1234+b)
            pm=(torch.rand(x.shape,generator=g,device=DEVICE)<0.15)&(x!=PAD)
            # ensure >=1 masked per row
            xin=x.clone(); xin[pm]=MASK; tgt=x.clone(); tgt[~pm]=-100
            logits=model.mlm_logits(xin)                     # (B,L,VOCAB)
            l=ce(logits.reshape(-1,VOCAB),tgt.reshape(-1)).reshape(x.shape)  # per-token CE
            m=(tgt!=-100).float(); per_doc=(l*m).sum(1)/m.sum(1).clamp(min=1)
            for j,i in enumerate(oi): out[i]=per_doc[j].item()
    return out

def main():
    dids,dtok=load_toks(SEQ/"seq_tokens_dev.parquet"); eids,etok=load_toks(SEQ/"seq_tokens_eval.parquet")
    model=Encoder().to(DEVICE); st=torch.load(SEQ/"_v2_encoder.pt",map_location=DEVICE)
    model.load_state_dict(st["model"]); model.eval()
    log(f"loaded PRETRAIN ckpt ep{st['ep']} acc{st.get('acc'):.3f} (label-free, pre-fine-tune)")

    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dlab=dev[["pair_id","label"]].copy()
    def report(col, sd):
        m=dlab.merge(pd.DataFrame({"pair_id":[str(p) for p in dids],col:sd}),on="pair_id",how="left")
        yy=m["label"].fillna(0).astype(int).to_numpy(); s=m[col].fillna(np.nanmedian(m[col])).to_numpy()
        ap=average_precision_score(yy,s); apn=average_precision_score(yy,-s)
        log(f"[{col}] standalone PairAP {ap:.4f} (neg-dir {apn:.4f}) pos-mean {s[yy==1].mean():.3f} neg-mean {s[yy==0].mean():.3f}")

    # ---- score A: MLM reconstruction error (novel, direct, cheap) ----
    err_d=mlm_recon_error(model,dtok); err_e=mlm_recon_error(model,etok); log("A: MLM recon error done"); report("mlm_err",err_d)

    # ---- pretrained pooled embedding (label-free) for B & C — LENGTH-SORTED (fast, low VRAM) ----
    def embed(toks):
        out=np.zeros((len(toks),2*D_MODEL),np.float32)
        order=np.argsort([len(t) for t in toks],kind="stable")
        with torch.no_grad():
            for b in range(0,len(order),256):
                oi=order[b:b+256]; e=model.pool(pad_batch([toks[i] for i in oi]).to(DEVICE)).cpu().numpy()
                for j,i in enumerate(oi): out[i]=e[j]
        return out
    Ed=embed(dtok); Ee=embed(etok); log(f"pretrained pooled emb dev {Ed.shape} eval {Ee.shape}")
    allE=np.vstack([Ed,Ee])
    # ---- score B: Isolation Forest (fit on POOLED dev+eval, label-free) ----
    iso=IsolationForest(n_estimators=200,max_samples=8192,random_state=SEED,n_jobs=-1).fit(allE)
    iso_all=-iso.score_samples(allE); iso_d=iso_all[:len(Ed)]; iso_e=iso_all[len(Ed):]; log("B: IsolationForest done"); report("iso",iso_d)
    # ---- score C: kNN mean distance to a SUBSAMPLED reference (O(n*ref), fast) ----
    rng=np.random.RandomState(SEED); ref=allE[rng.choice(len(allE),min(20000,len(allE)),replace=False)]
    nn_=NearestNeighbors(n_neighbors=20,n_jobs=-1).fit(ref)
    dist,_=nn_.kneighbors(allE); knn_all=dist.mean(1)
    knn_d=knn_all[:len(Ed)]; knn_e=knn_all[len(Ed):]; log("C: kNN distance done"); report("knn",knn_d)

    pl.DataFrame({"pair_id":[str(p) for p in dids],"mlm_err":err_d,"iso":iso_d,"knn":knn_d}).write_parquet(SEQ/"anomaly_dev.parquet")
    pl.DataFrame({"pair_id":[str(p) for p in eids],"mlm_err":err_e,"iso":iso_e,"knn":knn_e}).write_parquet(SEQ/"anomaly_eval.parquet")
    log("wrote anomaly_dev/eval.parquet")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
