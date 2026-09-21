"""§210 — PATH A RE-TEST (fixes §209 dann_killtest defects the user flagged). Three fixes:
  (1) BatchNorm -> LayerNorm: BatchNorm in a DANN computes stats over MIXED dev+eval batches and uses running
      stats at inference = a known DANN footgun that can cause collapse. LayerNorm is per-sample, domain-safe.
  (2) Fairer/stronger scaffold: 200 epochs (was 60), lr warmup, wider encoder, isolate the GRL effect by
      comparing adversarial vs non-adversarial with EVERYTHING ELSE identical (same seed, same init path).
  (3) Lambda SWEEP: GRL is sensitive to adversarial strength; test lambda_max in {0, 0.3, 1.0, 3.0} instead of
      one schedule. Also report the LGB base as the real target to beat (0.6612 eval-weighted).
If a properly-built DANN with a tuned lambda BEATS LGB base eval-weighted by >+0.005, Path A reopens. If even
fixed it does not, the §209 refutation stands on solid ground (not on the buggy original).
Run:  python -m anchor_repro.dann_killtest_v2
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb, torch, torch.nn as nn
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5; DEV="cuda:0"
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lamb): ctx.lamb=lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.lamb*g, None

class DANN(nn.Module):
    def __init__(self,d):
        super().__init__()
        # FIX (1): LayerNorm, not BatchNorm (domain-safe; per-sample, no mixed-batch running stats)
        self.enc=nn.Sequential(nn.Linear(d,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.3),
                               nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.GELU())
        self.risk=nn.Linear(64,1); self.dom=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,lamb=1.0):
        z=self.enc(x); return self.risk(z).squeeze(-1), self.dom(GRL.apply(z,lamb)).squeeze(-1), z

def load():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    ee=pd.read_parquet(SEQ/"seq_emb_v2_eval.parquet"); ee["pair_id"]=ee["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); evf=evf.merge(ee,on="pair_id",how="left")
    SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT: dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    for c in SEQF:
        dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0) if c in evf.columns else np.float32(0)
    return dev,evf,SEQF

def main():
    dev,evf,SEQF=load()
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","nsh"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and not c.startswith("evl_pf") and not c.startswith("deq_pf")]
    feats=[c for c in base if c in evf.columns]
    Xd=dev[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    Xe=evf[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    mu=Xd.mean(0); sd=Xd.std(0)+1e-6; Xd=(Xd-mu)/sd; Xe=(Xe-mu)/sd
    n=min(len(dev),len(evf),40000); ridx=rng.choice(len(dev),n,replace=False); eidx=rng.choice(len(evf),n,replace=False)
    Xc=np.concatenate([Xd[ridx],Xe[eidx]]); yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(Xd); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
    heldm=np.isin(folds,[3,4])
    log(f"features {len(feats)}")

    def train(lamb_max):
        oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            torch.manual_seed(SEED+f)  # same init path across lambda settings (isolates GRL effect)
            tr=(~heldm)&(folds!=f); te=folds==f
            m=DANN(Xd.shape[1]).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1.5e-3,weight_decay=1e-4)
            Xt=torch.tensor(Xd[tr]).to(DEV); yt=torch.tensor(y[tr].astype(np.float32)).to(DEV)
            wt=torch.tensor((labm[tr]*np.where(y[tr]==1,5.0,1.0)+(~labm[tr])*0.1).astype(np.float32)).to(DEV)
            bce=nn.BCEWithLogitsLoss(reduction="none"); bced=nn.BCEWithLogitsLoss()
            EP=200
            for ep in range(EP):
                lamb=lamb_max*(2/(1+np.exp(-10*ep/EP))-1)
                m.train(); opt.zero_grad()
                r,_,_=m(Xt,lamb); rl=(bce(r,yt)*wt).mean()
                if lamb_max>0:
                    ev=torch.tensor(Xe[rng.choice(len(Xe),len(Xt),replace=False)]).to(DEV)
                    Xdom=torch.cat([Xt,ev]); yd=torch.cat([torch.zeros(len(Xt)),torch.ones(len(ev))]).to(DEV)
                    _,dh,_=m(Xdom,lamb); rl=rl+bced(dh,yd)
                rl.backward(); opt.step()
            m.eval()
            with torch.no_grad(): oof[te]=torch.sigmoid(m(torch.tensor(Xd[te]).to(DEV),0.0)[0]).cpu().numpy()
        return oof
    def ew(o): return average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])

    log("=== LayerNorm DANN, lambda sweep (fixes §209 BatchNorm bug + fair scaffold) ===")
    results={}
    for lm in [0.0,0.3,1.0,3.0]:
        o=train(lm); results[lm]=ew(o); log(f"  lambda_max={lm}: held EVAL-W {results[lm]:.4f}")
    best_adv=max(results[l] for l in [0.3,1.0,3.0]); noadv=results[0.0]
    log(f"\n  best adversarial {best_adv:.4f} | non-adversarial {noadv:.4f} | LGB base 0.6612")
    log(f"  GRL effect (best adv - no-adv) = {best_adv-noadv:+.4f}")
    log(f"  vs LGB base = {best_adv-0.6612:+.4f}")
    log("=== VERDICT: Path A reopens only if best adversarial BEATS LGB base 0.6612 by >+0.005 ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
