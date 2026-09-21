"""RESIDUAL-ERROR analysis (the user's "go negative / find the boundary" idea, done productively).

Every add-on §154-165 failed because it re-mixed the SAME basis and was too correlated to add (Rule 13).
Escape: find what our model gets SYSTEMATICALLY WRONG. The feature that separates our ERRORS from our
correct calls is ORTHOGONAL to our current signal by construction = the residual = the missing basis.

Steps:
  1. reproduce current-best dev OOF risk (base+PT+seq_risk).
  2. FALSE NEGATIVES = true positives (label==1) our risk ranks LOW (bottom of positives).
     FALSE POSITIVES = unlabeled pairs our risk ranks HIGH (top of the non-positives).
     CORRECT POS = true positives we rank high.
  3. For every raw feature, compare FN vs correct-POS (what do the missed colluders have that we ignore?)
     and FP vs true-NEG (what makes us wrongly flag?). Rank by separation (AUC / mean-gap).
  4. Also: train a model to predict "is this a residual error" from raw features -> its top features ARE
     the missing basis; report its held-out AUC (can we even predict our own errors? if yes, signal exists).
Run:  python -m anchor_repro.residual_analysis
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")  # silence pandas fragmentation PerformanceWarning spam
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev=dev.merge(de,on="pair_id",how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    for c in PT+SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    fam=dev["behavior_family"].fillna("none").to_numpy()
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    def distill():
        Xd=dev[SEQF].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=500); o1[te]=m.predict(Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=500); o[te]=m.predict(Xd[te])
        return o
    dev["seq_risk"]=distill()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
          and not c.startswith("seq_") and c!="seq_risk"]
    FEATS=base+PT+["seq_risk"]

    # full OOF risk (the model whose errors we analyze)
    X=dev[FEATS].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w1>0); te=folds==f
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]=m.predict(X[te])
    pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
    w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); risk=np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr=(folds!=f)&(w2>0); te=folds==f
        for sd in [SEED,SEED+11,SEED+23]:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); risk[te]+=m.predict(X[te])/3
    dev["risk"]=risk
    log(f"OOF risk built; positives {int((labm&(y==1)).sum())}")

    # rank within: positives' risk percentile, and non-positives' risk percentile
    r_pct=pd.Series(risk).rank(pct=True).to_numpy()
    posm=labm&(y==1); negm=labm&(y==0); unlm=~labm
    # among positives, bottom-third by risk = FALSE-NEG (we miss); top-third = CORRECT-POS (we catch)
    pos_idx=np.where(posm)[0]; pr=risk[pos_idx]; q33,q67=np.quantile(pr,[0.33,0.67])
    FN=pos_idx[pr<=q33]; CP=pos_idx[pr>=q67]
    log(f"FALSE-NEG positives (we rank low): {len(FN)} | CORRECT-POS (we rank high): {len(CP)}")
    # per-family breakdown of FN
    from collections import Counter
    log(f"  FN family mix: {dict(Counter(fam[FN]))}  | CP family mix: {dict(Counter(fam[CP]))}")

    # which RAW features separate FN from CP? (what the missed colluders have that we underuse)
    RAWALL=[c for c in dev.columns if c not in META and c not in ("risk","seq_risk") and dev[c].dtype.kind in "fiu" and not c.startswith("seq_")]
    yy=np.concatenate([np.ones(len(FN)),np.zeros(len(CP))]); idx=np.concatenate([FN,CP])
    seps=[]
    for c in RAWALL:
        v=dev[c].to_numpy()[idx]
        if np.all(~np.isfinite(v)) or np.nanstd(v)==0: continue
        v=np.nan_to_num(v,nan=np.nanmedian(v))
        try: a=roc_auc_score(yy,v); seps.append((c,max(a,1-a),a))
        except Exception: pass
    seps.sort(key=lambda x:-x[1])
    print("\n=== features separating FALSE-NEG (missed colluders) from CORRECT-POS (top 20) ===")
    print("  AUC   dir  feature")
    for c,a,raw in seps[:20]:
        print(f"  {a:.3f}  {'FN-hi' if raw>0.5 else 'FN-lo'}  {c}")

    # can we PREDICT our own errors from raw feats? (residual signal existence test)
    Xr=dev[RAWALL].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    # target: among positives, is this a FN (1) vs CP (0); grouped CV by table
    mask=np.zeros(len(dev),bool); mask[FN]=True; mask[CP]=True
    yt=np.zeros(len(dev)); yt[FN]=1
    gfolds=folds[mask]; Xm=Xr[mask]; ym=yt[mask]
    oof=np.zeros(mask.sum())
    gi=np.unique(gfolds)
    for f in gi:
        tr=gfolds!=f; te=gfolds==f
        if te.sum()==0 or ym[tr].sum()==0: continue
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(Xm[tr],ym[tr]),num_boost_round=300); oof[te]=m.predict(Xm[te])
    try:
        auc=roc_auc_score(ym,oof)
        log(f"RESIDUAL predictability: can we predict FN-vs-CP from raw feats? grouped-CV AUC {auc:.4f}")
        log("  (>0.65 => a learnable residual signal exists that our risk under-uses = the missing basis)")
    except Exception as e:
        log(f"residual AUC failed: {e}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
