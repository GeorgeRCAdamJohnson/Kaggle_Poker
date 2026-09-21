"""BEHAVIOR upgrade: add the trained sequence embedding to the family-classification head (task #3/#1).

BehaviorMAP (10% weight) = mean over 3 target families of AP(true==fam, risk if pred==fam else 0). It
depends on (a) risk_score (FIXED = our 0.83185) and (b) the predicted family routing. The family head is
the lever: better family discrimination -> higher BehaviorMAP. The seq embedding lifted risk (§144) and
evidence (§149) — this tests whether it also lifts family classification.

Reproduces the hosen42 behavior construction EXACTLY (per-family LightGBM, prior-normalized argmax,
rho none-threshold, host_score), then compares family heads:
  A  PAIR_FEATS only          [baseline]
  B  PAIR_FEATS + seq_emb
Reports family accuracy on positives + BehaviorMAP at the tuned rho.

Run:  python -m anchor_repro.behavior_seq
"""
from __future__ import annotations
import time, os, json, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
TARGET_BEHAVIORS=("directed_transfer","soft_play","coordinated_isolation")
POS_WEIGHT,NEG_WEIGHT,PU_WEIGHT=5.0,1.0,0.1
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def _ap(y,s): 
    return average_precision_score(y,s) if y.sum()>0 else 0.0

def behavior_map(true_beh, pred_beh, risk):
    scores=[]
    for b in TARGET_BEHAVIORS:
        bt=(true_beh==b).astype(int)
        if bt.sum()==0: scores.append(0.0); continue
        scores.append(_ap(bt,np.where(pred_beh==b,risk,0.0)))
    return float(np.mean(scores)), scores

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    dev["pair_id"]=dev["pair_id"].astype(str); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left")
    for c in SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    PAIR_FEATS=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and not c.startswith("seq_")]
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    PP=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=os.cpu_count())
    PAIR_SEEDS=[SEED,SEED+11,SEED+23]; PAIR_ROUNDS=750

    # ---- reproduce OOF risk (stage-1 -> two-step PU -> stage-2), needed for the metric's risk_values ----
    def fit_risk(y_fit,w_fit,feats,seeds,rounds):
        X=dev[feats].astype("float32"); oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w_fit>0); te=folds==f
            for sd in seeds:
                m=lgb.train({**PP,"seed":sd},lgb.Dataset(X[tr],y_fit[tr],weight=w_fit[tr]),num_boost_round=rounds); oof[te]+=m.predict(X[te])/len(seeds)
        return oof
    w1=np.where(labm,np.where(y==1,POS_WEIGHT,NEG_WEIGHT),PU_WEIGHT)
    oof1=fit_risk(y,w1,PAIR_FEATS,PAIR_SEEDS[:1],PAIR_ROUNDS)
    pos_oof=oof1[labm&(y==1)]; ta,tp=np.quantile(pos_oof,0.05),np.quantile(pos_oof,0.50)
    unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
    y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,POS_WEIGHT,NEG_WEIGHT),np.where(ps,1.0,np.where(amb,0.0,PU_WEIGHT)))
    oof=fit_risk(y2,w2,PAIR_FEATS,PAIR_SEEDS,PAIR_ROUNDS)
    log(f"OOF eval-mirrored PairAP {average_precision_score(y,oof):.4f} (risk fixed for behavior test)")

    fam_prior={fam:int((dev["behavior_family"]==fam).sum()) for fam in TARGET_BEHAVIORS}
    def fam_heads(feats):
        X=dev[feats].astype("float32"); oof_f={fam:np.zeros(len(dev)) for fam in TARGET_BEHAVIORS}
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            tr_f=tr & ~((y2==1)&~labm)   # pseudo-positives excluded from family heads
            for fam in TARGET_BEHAVIORS:
                yf=(dev["behavior_family"]==fam).to_numpy().astype(int)
                mf=lgb.train({**PP,"num_leaves":7,"seed":SEED},lgb.Dataset(X[tr_f],yf[tr_f],weight=w2[tr_f]),num_boost_round=400)
                oof_f[fam][te]=mf.predict(X[te])
        return oof_f
    def predict_family(fam_probs):
        mat=np.column_stack([fam_probs[fam]/fam_prior[fam] for fam in TARGET_BEHAVIORS])
        return np.array(TARGET_BEHAVIORS)[mat.argmax(axis=1)]

    # host BehaviorMAP over the dev population at tuned rho (true beh: positives=family, else none)
    true_beh=np.where(y==1,dev["behavior_family"].to_numpy(),"none")
    rank_pct=pd.Series(oof).rank(ascending=False,method="first").to_numpy()/len(oof)
    NONE_GRID=[1.0,0.10,0.05,0.03,0.02,0.015,0.01,0.0075,0.005,0.004,0.003]
    pos=(y==1)
    def evaluate(feats,tag):
        of=fam_heads(feats); pf=predict_family(of)
        acc=(pf[pos]==dev["behavior_family"].to_numpy()[pos]).mean()
        best=(-1,None)
        for rho in NONE_GRID:
            pb=np.where(rank_pct<=rho,pf,"none")
            bm,_=behavior_map(true_beh,pb,oof)
            if bm>best[0]: best=(bm,rho)
        log(f"[{tag}] family-acc(pos) {acc:.3f} | best BehaviorMAP {best[0]:.4f} @ rho={best[1]}")
        return best[0]
    log("=== behavior family head: baseline vs +seq_emb ===")
    a=evaluate(PAIR_FEATS,"A PAIR_FEATS only")
    b=evaluate(PAIR_FEATS+SEQF,"B PAIR_FEATS + seq_emb")
    log(f"BehaviorMAP delta (B-A): {b-a:+.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
