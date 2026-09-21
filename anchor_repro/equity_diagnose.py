"""WHY did the equity-surrender features regress the gate (-0.0135)? Test the 4 hypotheses, don't guess.

H1 REDUNDANCY: are the equity feats correlated with the existing made-value surrender feats? If yes,
   they add variance not signal. (max |corr| of each equity feat vs fold_better_hand/dump/etc.)
H2 DRIFT: do the equity feats drift dev->eval? adversarial AUC (dev-vs-eval classifier on equity feats
   alone). >0.65 = drifts = the eval-weighted gate correctly punishes them.
H3 INCREMENTAL: does equity add AUC OVER the made-value feats alone (labelled OOF)? isolates whether
   there is ANY signal beyond made-value, separate from the full-model gate.
H4 CONSTRUCTION: sanity — do the surrender feats actually separate positives when computed simply
   (mean equity of FOLDED hands where partner won, per pair)? rules out a join/logic bug.
Run:  python -m anchor_repro.equity_diagnose
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.equity_features import build_features, FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
SEED,N_FOLDS=42,5
P=dict(objective="binary",learning_rate=0.05,num_leaves=15,min_data_in_leaf=30,verbose=-1,num_threads=-1)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)
MADEVAL=["fold_better_hand","dump","dump_better_hand","loser_beats_winner_final","winner_weak_won",
         "fold_stronger_to_partner","fold_better_to_partner","transfer_dominant","transfer_rate","hs_mean","hs_max","hs_top3"]

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    agg=build_features("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=None).to_pandas()
    dev=dev.merge(agg,on="pair_id",how="left")
    for c in FEATS: dev[c]=dev[c].astype("float32").fillna(0)
    madeval=[c for c in MADEVAL if c in dev.columns]
    log(f"made-value feats present in dev_pair_features: {madeval}")

    # H1 REDUNDANCY
    print("\n=== H1 redundancy: max|corr| of each equity feat vs made-value feats ===")
    if madeval:
        for c in FEATS:
            cors=[abs(np.corrcoef(dev[c].fillna(0),dev[mc].fillna(0))[0,1]) for mc in madeval]
            j=int(np.argmax(cors)); print(f"  {c:22s} max|corr| {cors[j]:.3f} vs {madeval[j]}")
    else:
        log("(made-value feats live in HAND_FEATS/aggregates, not dev_pair_features - see H3 instead)")

    # H2 DRIFT: adversarial dev-vs-eval on equity feats alone
    print("\n=== H2 drift: adversarial dev-vs-eval AUC on equity feats alone (>0.65 = drifts) ===")
    ev=pd.read_parquet(STEP3/"eval_pair_features.parquet"); ev["pair_id"]=ev["pair_id"].astype(str)
    agge=build_features("evaluation",EQ/"pairhand_equity_eval.parquet","eval_hand_features.parquet",keep_pairs=None).to_pandas() if (EQ/"pairhand_equity_eval.parquet").exists() else None
    if agge is not None:
        ev=ev.merge(agge,on="pair_id",how="left")
        for c in FEATS: ev[c]=ev[c].astype("float32").fillna(0)
        n=min(len(dev),len(ev),40000); ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=ev.iloc[rng.choice(len(ev),n,replace=False)]
        Xc=pd.concat([ds[FEATS],es[FEATS]],ignore_index=True).fillna(0); yc=np.concatenate([np.zeros(n),np.ones(n)])
        oof=np.zeros(2*n); idx=rng.permutation(2*n)
        m=lgb.train(P,lgb.Dataset(Xc.iloc[idx[:n]],yc[idx[:n]]),num_boost_round=150); oof=m.predict(Xc)
        log(f"  adversarial AUC (equity feats) = {roc_auc_score(yc,oof):.4f}")
    else:
        log("  eval equity not built yet - drift test skipped (need pairhand_equity_eval.parquet)")

    # H3 INCREMENTAL: made-value-only OOF AUC vs +equity (labelled)
    print("\n=== H3 incremental: does equity add AUC OVER made-value alone? (labelled OOF) ===")
    # proxy made-value set = the aggregate made-value cols present in dev_pair_features
    mv=[c for c in dev.columns if any(k in c for k in ['fold_better','dump','loser_beats','winner_weak','transfer_dom','transfer_rate','hs_'])]
    mv=[c for c in mv if dev[c].dtype.kind in 'fiu']
    def oof_auc(feats):
        X=dev[feats].astype('float32').fillna(0); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&labm; te=(folds==f)&labm
            if te.sum()==0: continue
            m=lgb.train(P,lgb.Dataset(X[tr],y[tr]),num_boost_round=250); o[te]=m.predict(X[te])
        return roc_auc_score(y[labm],o[labm])
    if mv:
        a=oof_auc(mv); b=oof_auc(mv+FEATS)
        log(f"  made-value-only OOF AUC {a:.4f} -> +equity {b:.4f} (delta {b-a:+.4f})")
    # H4 CONSTRUCTION sanity: simplest surrender = mean equity of folded hands, per pair, pos vs neg
    print("\n=== H4 construction sanity: eq_surrendered_mean pos vs neg ===")
    pos=dev[dev['label']==1]['eq_surrendered_mean']; neg=dev[(dev['label']==0)&labm]['eq_surrendered_mean']
    log(f"  eq_surrendered_mean: pos {pos.mean():.4f}  neg {neg.mean():.4f}  AUC {roc_auc_score(y[labm],dev.loc[labm,'eq_surrendered_mean'].fillna(0)):.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
