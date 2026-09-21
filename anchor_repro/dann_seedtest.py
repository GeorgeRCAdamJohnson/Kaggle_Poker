"""§210b — DANN lambda=1.0 MULTI-SEED hardening. The §210 sweep (0.67/0.59/0.71/0.56) ZIGZAGS with lambda,
which is the signature of high variance on the 372-positive held set, NOT a coherent GRL effect. The 0.7116
at lambda=1.0 is one seed and MUST be reseeded before any trust. Run lambda=1.0 AND lambda=0.0 across 5 seeds;
report mean +- std of each and the GRL delta. Verdict: real only if lambda=1.0 mean clearly beats lambda=0.0
mean AND beats LGB base 0.6612, with std small enough that the gain isn't a variance draw.
Run:  python -m anchor_repro.dann_seedtest
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb, torch, torch.nn as nn
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.dann_killtest_v2 import DANN, load, PT
STEP3=Path("outputs/poker_collusion/hosen42_step3")
N_FOLDS=5; DEV="cuda:0"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev,evf,SEQF=load()
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    base_rng=np.random.RandomState(42)
    TABLES=sorted(dev["table_id"].unique().tolist())
    TF={t:int(v) for t,v in zip(TABLES,base_rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family","nsh"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and not c.startswith("evl_pf") and not c.startswith("deq_pf")]
    feats=[c for c in base if c in evf.columns]
    Xd=dev[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    Xe=evf[feats].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    mu=Xd.mean(0); sd=Xd.std(0)+1e-6; Xd=(Xd-mu)/sd; Xe=(Xe-mu)/sd
    n=min(len(dev),len(evf),40000); ridx=base_rng.choice(len(dev),n,replace=False); eidx=base_rng.choice(len(evf),n,replace=False)
    Xc=np.concatenate([Xd[ridx],Xe[eidx]]); yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(Xd); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
    heldm=np.isin(folds,[3,4])

    def train(lamb_max, seed):
        oof=np.zeros(len(dev)); rng=np.random.RandomState(seed)
        for f in range(N_FOLDS):
            torch.manual_seed(seed*100+f)
            tr=(~heldm)&(folds!=f); te=folds==f
            m=DANN(Xd.shape[1]).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1.5e-3,weight_decay=1e-4)
            Xt=torch.tensor(Xd[tr]).to(DEV); yt=torch.tensor(y[tr].astype(np.float32)).to(DEV)
            wt=torch.tensor((labm[tr]*np.where(y[tr]==1,5.0,1.0)+(~labm[tr])*0.1).astype(np.float32)).to(DEV)
            bce=nn.BCEWithLogitsLoss(reduction="none"); bced=nn.BCEWithLogitsLoss(); EP=200
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
        return average_precision_score(y[heldm],oof[heldm],sample_weight=w_eval[heldm])

    SEEDS=[42,101,202,303,404]
    log(f"=== multi-seed hardening: lambda 1.0 vs 0.0 over {len(SEEDS)} seeds (LGB base 0.6612) ===")
    a1=[]; a0=[]
    for s in SEEDS:
        v1=train(1.0,s); v0=train(0.0,s); a1.append(v1); a0.append(v0)
        log(f"  seed {s}: lambda1.0 {v1:.4f} | lambda0.0 {v0:.4f} | GRL delta {v1-v0:+.4f}")
    a1=np.array(a1); a0=np.array(a0)
    log(f"\n  lambda1.0: mean {a1.mean():.4f} std {a1.std():.4f} min {a1.min():.4f} max {a1.max():.4f}")
    log(f"  lambda0.0: mean {a0.mean():.4f} std {a0.std():.4f}")
    log(f"  GRL effect: mean {(a1-a0).mean():+.4f} (>0 in {int((a1>a0).sum())}/5)")
    log(f"  lambda1.0 mean vs LGB base 0.6612 = {a1.mean()-0.6612:+.4f}")
    real = a1.mean()-0.6612>0.005 and a1.mean()-a0.mean()>0.005 and a1.std()<0.02
    log(f"=== VERDICT: {'REAL — graduate to full gate + leak battery' if real else 'VARIANCE ARTIFACT — the 0.71 single-seed was a lucky draw; Path A closes'} ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
