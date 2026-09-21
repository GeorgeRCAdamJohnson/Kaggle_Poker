"""WHY did the flow-tail fail, and can it be made stronger? (real diagnosis, not hand-wave)

Facts to explain:
  - tdir_p95 standalone AUC 0.876 (strong) but adding the 15-feat tail block REGRESSED the gate -0.0161.
  - field-normalization LOWERED standalone AUC (tdir_p95 0.876 -> _fn 0.854; gap 0.749 -> 0.701).
Questions:
  1. Is tdir_p95 REDUNDANT with what the model already has (transfer_dominant, _asnd, transfer_rate)?
     -> correlation + does tdir_p95 add incremental AUC over the existing transfer feats alone?
  2. Is the regression from the 15-feat block (collinearity/noise) or from tdir_p95 itself?
     -> gate with ONLY tdir_p95 added, and with ONLY tdir_p95_fn.
  3. Can a BETTER normalization make it transferable? Try:
     (a) per-TABLE percentile of tdir_p95 (graph/pool-relative, the §literature angle),
     (b) log1p(tdir_p95) (tame the tail scale),
     (c) tdir_p95 / n_shared control (more shared hands -> higher p95 by chance).
Everything measured with dev-label AUC AND the eval-weighted phase gate.
Run:  python -m anchor_repro.flowtail_diagnose
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.flowtail_gate import flow_tail, FT, PT, P, SEED, N_FOLDS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    ftd=flow_tail(STEP3/"dev_pair_hands.parquet","development"); ftd["pair_id"]=ftd["pair_id"].astype(str)
    dev=dev.merge(ftd[["pair_id"]+FT],on="pair_id",how="left")
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    # existing transfer feats already in the model
    EXIST=[c for c in ["transfer_dominant","transfer_rate","transfer_imbalance","direction_consistency","transfer_any_mean","transfer_any_max","transfer_any_top3","net_gap_mean","net_gap_max","net_gap_top3"] if c in dev.columns]
    log(f"existing transfer feats present: {EXIST}")

    # ---- Q1: redundancy ----
    lab=labm
    for c in ["transfer_dominant","transfer_rate"]:
        if c in dev.columns:
            r=spearmanr(dev.loc[lab,"tdir_p95"],dev.loc[lab,c].fillna(0)).correlation
            log(f"Spearman(tdir_p95, {c}) = {r:.3f}")
    # incremental AUC: existing transfer feats alone vs +tdir_p95 (5-fold OOF, labelled)
    def oof_auc(feats):
        X=dev[feats].astype("float32").fillna(0); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&labm; te=(folds==f)&labm
            if te.sum()==0: continue
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr]),num_boost_round=300); o[te]=m.predict(X[te])
        return roc_auc_score(y[labm],o[labm])
    if EXIST:
        a=oof_auc(EXIST); b=oof_auc(EXIST+["tdir_p95"])
        log(f"Q1 incremental: existing-transfer OOF AUC {a:.4f} -> +tdir_p95 {b:.4f} (delta {b-a:+.4f})")

    # ---- Q3: better normalizations ----
    d=dev.copy()
    d["tdir_p95_log"]=np.log1p(d["tdir_p95"].fillna(0))
    d["tdir_p95_pertable"]=d.groupby("table_id")["tdir_p95"].rank(pct=True)   # graph/pool-relative
    d["tdir_p95_pernsh"]=d["tdir_p95"]/(d.get("shared_hands",pd.Series(1,index=d.index)).fillna(1)**0.5 + 1)
    for c in ["tdir_p95","tdir_p95_fn","tdir_p95_log","tdir_p95_pertable","tdir_p95_pernsh"]:
        s=d.loc[lab,c].fillna(d.loc[lab,c].median()).to_numpy()
        log(f"  {c:20s} dev-label AUC {roc_auc_score(y[lab],s):.4f}")

    # ---- Q2/Q3 GATE: does the phase-robust per-table-rank tail clear the eval-weighted gate on the FULL model? ----
    # add the new normalizations to dev + eval, build seq_risk, run the §147 gate with ONE clean feature.
    log("=== building eval + seq_risk for the gate ===")
    fte=flow_tail(STEP3/"eval_pair_hands.parquet","evaluation"); fte["pair_id"]=fte["pair_id"].astype(str)
    evf2=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left").merge(fte[["pair_id"]+FT],on="pair_id",how="left")
    d["tdir_p95_pertable"]=d.groupby("table_id")["tdir_p95"].rank(pct=True)
    evf2["tdir_p95_pertable"]=evf2.groupby("table_id")["tdir_p95"].rank(pct=True)
    d["tdir_p95_log"]=np.log1p(d["tdir_p95"].fillna(0)); evf2["tdir_p95_log"]=np.log1p(evf2["tdir_p95"].fillna(0))
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    seqf=[c for c in de.columns if c.startswith("seq_")]; d=d.merge(de,on="pair_id",how="left")
    for c in seqf: d[c]=d[c].astype("float32").fillna(0)
    for c in PT+["tdir_p95_pertable","tdir_p95_log","tdir_p95"]:
        d[c]=d[c].astype("float32").fillna(0); evf2[c]=evf2[c].astype("float32").fillna(0)
    def distill():
        X=d[seqf].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(d))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=500); o1[te]=m.predict(X[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); o=np.zeros(len(d))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=500); o[te]=m.predict(X[te])
        return o
    d["seq_risk"]=distill()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in d.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in FT
          and c not in seqf and not c.startswith("seq_") and c!="seq_risk"
          and c not in ("tdir_p95_pertable","tdir_p95_log")]
    shared=[c for c in base if c in evf2.columns]; n=min(len(d),len(evf2),40000)
    ds=d.iloc[rng.choice(len(d),n,replace=False)]; es=evf2.iloc[rng.choice(len(evf2),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(d[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    def oof(feats):
        X=d[feats].astype("float32"); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(d))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]=m.predict(X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        o=np.zeros(len(d))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); o[te]+=m.predict(X[te])/3
        return o
    def gate(feats,tag):
        o=oof(feats); hp=average_precision_score(y[heldm],o[heldm]); hw=average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])
        log(f"[{tag}] held-out PLAIN {hp:.4f} | EVAL-WEIGHTED {hw:.4f}"); return hp,hw
    print("\n=== GATE: phase-robust per-table-rank tail (one clean feat) ===")
    b_p,b_w=gate(base+PT+["seq_risk"],"base+PT+seq_risk")
    r_p,r_w=gate(base+PT+["seq_risk","tdir_p95_pertable"],"+tdir_p95_pertable")
    l_p,l_w=gate(base+PT+["seq_risk","tdir_p95_log"],"+tdir_p95_log")
    print(f"\n  per-table-rank delta: plain {r_p-b_p:+.4f} | EVAL-WEIGHTED {r_w-b_w:+.4f}")
    print(f"  log-tail       delta: plain {l_p-b_p:+.4f} | EVAL-WEIGHTED {l_w-b_w:+.4f}")
    print("  SHIP if EVAL-WEIGHTED > +0.005.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
