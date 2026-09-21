"""§209 PATH A kill-test — DOMAIN-ADVERSARIAL (gradient-reversal) risk model. The dev->eval wall has only
been fought POST-HOC (eval-weighted gate). This bakes phase-invariance into the ENCODER via GRL. Decisive
question: does the DANN representation's eval-weighted held-out PairAP BEAT base+PT+seq_risk (0.6612 from
collusion_value_gate), or only MATCH the post-hoc correction? STOP unless it beats by >+0.005.

Uses the SAME shared feature set + SAME adversarial-eval-weight + SAME held folds [3,4] as the trusted gate,
so the comparison is apples-to-apples. Torch on GPU.
Run:  python -m anchor_repro.dann_killtest
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb, torch, torch.nn as nn
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
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
        self.enc=nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,64),nn.GELU())
        self.risk=nn.Linear(64,1); self.dom=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,lamb=1.0):
        z=self.enc(x); return self.risk(z).squeeze(-1), self.dom(GRL.apply(z,lamb)).squeeze(-1), z

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    ee_path=SEQ/"seq_emb_v2_eval.parquet"
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    have_eval_emb=ee_path.exists()
    if have_eval_emb:
        ee=pd.read_parquet(ee_path); ee["pair_id"]=ee["pair_id"].astype(str); evf=evf.merge(ee,on="pair_id",how="left")
    for c in PT: dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    for c in SEQF:
        dev[c]=dev[c].astype("float32").fillna(0)
        if c not in evf.columns: evf[c]=np.float32(0)
        evf[c]=evf[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","nsh"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and not c.startswith("evl_pf") and not c.startswith("deq_pf")]
    feats=[c for c in base if c in evf.columns]
    log(f"features {len(feats)} | eval seq emb available: {have_eval_emb}")

    # standardize on dev
    Xd=dev[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    Xe=evf[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    mu=Xd.mean(0); sd=Xd.std(0)+1e-6; Xd=(Xd-mu)/sd; Xe=(Xe-mu)/sd

    # adversarial eval-weights (same as trusted gate)
    n=min(len(dev),len(evf),40000); ridx=rng.choice(len(dev),n,replace=False); eidx=rng.choice(len(evf),n,replace=False)
    Xc=np.concatenate([Xd[ridx],Xe[eidx]]); yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(Xd); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
    heldm=np.isin(folds,[3,4])

    def train_dann(adversarial):
        torch.manual_seed(SEED)
        oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(~heldm)&(folds!=f); te=folds==f
            m=DANN(Xd.shape[1]).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
            Xt=torch.tensor(Xd[tr]).to(DEV); yt=torch.tensor(y[tr].astype(np.float32)).to(DEV); wt=torch.tensor((labm[tr]*np.where(y[tr]==1,5.0,1.0)+(~labm[tr])*0.1).astype(np.float32)).to(DEV)
            # domain batch: dev(tr) label 0 vs eval sample label 1
            Xev=torch.tensor(Xe[rng.choice(len(Xe),min(len(Xe),tr.sum()),replace=False)]).to(DEV)
            bce=nn.BCEWithLogitsLoss(reduction="none"); bced=nn.BCEWithLogitsLoss()
            m.train()
            for ep in range(60):
                lamb=(2/(1+np.exp(-10*ep/60))-1) if adversarial else 0.0
                opt.zero_grad()
                r,_,_=m(Xt,lamb); rl=(bce(r,yt)*wt).mean()
                # domain loss on combined dev+eval
                Xdom=torch.cat([Xt,Xev]); yd=torch.cat([torch.zeros(len(Xt)),torch.ones(len(Xev))]).to(DEV)
                _,dh,_=m(Xdom,lamb); dl=bced(dh,yd)
                (rl+(dl if adversarial else 0.0*dl)).backward(); opt.step()
            m.eval()
            with torch.no_grad(): oof[te]=torch.sigmoid(m(torch.tensor(Xd[te]).to(DEV),0.0)[0]).cpu().numpy()
        return oof

    log("training NON-adversarial (lambda=0) baseline DANN-arch..."); o0=train_dann(False)
    log("training ADVERSARIAL (GRL) DANN..."); o1=train_dann(True)
    def ew(o): return average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])
    def pl_(o): return average_precision_score(y[heldm],o[heldm])
    log(f"[no-adv DANN]  held PLAIN {pl_(o0):.4f} | EVAL-W {ew(o0):.4f}")
    log(f"[adversarial]  held PLAIN {pl_(o1):.4f} | EVAL-W {ew(o1):.4f}")
    log(f"  adversarial vs no-adv EVAL-W delta {ew(o1)-ew(o0):+.4f}")
    log(f"  reference: base+PT+seq_risk LGB held EVAL-W ~0.6612 (collusion_value_gate)")
    log(f"=== KILL-TEST: does adversarial BEAT the LGB base eval-weighted 0.6612 by >+0.005? adv EVAL-W {ew(o1):.4f} ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
