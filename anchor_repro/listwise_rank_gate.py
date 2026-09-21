"""§218 — LISTWISE rank:map@5 objective for evidence (from the 0.82412 notebook, the one concrete untested
lever). Our evidence ranker is LightGBM BINARY is_evidence classification. The notebook uses XGBoost
`rank:map` / eval_metric map@5 — a TRUE LISTWISE objective that optimizes MAP@5 DIRECTLY, per-pair groups.
We have NEVER trained on the actual metric as the objective. This is orthogonal to the feature-completeness
finding (§216): same features, DIFFERENT objective. Does directly optimizing MAP@5 beat binary classification?

Compare, family-conditional, OOF (table folds), host-exact MAP@5, 5 seeds:
  BIN  : current LightGBM binary is_evidence (base 0.5989)
  LW   : XGBoost rank:map, per-pair groups, same HAND_FEATS+ctx features, GPU (device=cuda)
Ship-eval if LW mean >+0.003 AND >=4/5 positive (evidence phase-immune, local==LB).
Run: python -m anchor_repro.listwise_rank_gate
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb, xgboost as xgb
warnings.simplefilter("ignore")
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, HAND_SEEDS, HAND_ROUNDS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

def bin_ranker(dh, feats, seeds):
    dh=dh.copy(); dh["hs"]=0.0
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            ms=[lgb.train({**HAND_PARAMS,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=HAND_ROUNDS) for sd in seeds]
            tgt=te&(dh["behavior_family"]==fam).to_numpy()
            dh.loc[tgt,"hs"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in ms],axis=0)
    return dh["hs"].to_numpy()

def lw_ranker(dh, feats, seeds):
    dh=dh.copy(); dh["hs"]=0.0
    XP=dict(objective="rank:map",eval_metric="map@5",eta=0.1,max_depth=5,subsample=0.9,colsample_bytree=0.8,
            min_child_weight=5,device="cuda",tree_method="hist")
    for f in sorted(dh["fold"].unique()):
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            trm=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if trm.sum()<50: continue
            tr=dh.loc[trm].sort_values("pair_id",kind="mergesort")
            grp=tr.groupby("pair_id",sort=False).size().to_numpy()
            tgt=te&(dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dtest=xgb.DMatrix(dh.loc[tgt,feats].to_numpy(np.float32))
            preds=[]
            for sd in seeds:
                d=xgb.DMatrix(tr[feats].to_numpy(np.float32),label=tr["is_evidence"].astype(int).to_numpy()); d.set_group(grp)
                b=xgb.train({**XP,"seed":sd},d,num_boost_round=180); preds.append(b.predict(dtest))
            dh.loc[tgt,"hs"]=np.mean(preds,axis=0)
    return dh["hs"].to_numpy()

def m5(dh,hs):
    d=dh.copy(); d["hand_score"]=hs; return map5(d[d["label"]==1])

def main():
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    FEATS=HAND_FEATS+CTXCOLS
    log("=== BIN (LightGBM binary) vs LW (XGBoost rank:map@5) — family-conditional, 5 seeds ===")
    ds=[]
    for i,(s1,s2) in enumerate([(42,49),(101,108),(202,209),(303,310),(404,411)]):
        b=m5(dh,bin_ranker(dh,FEATS,[s1,s2])); l=m5(dh,lw_ranker(dh,FEATS,[s1,s2])); ds.append(l-b)
        log(f"  seedset{i}: BIN {b:.4f} | LW {l:.4f} | LW-BIN {l-b:+.4f}")
    log(f"LW vs BIN over 5 seeds: mean {np.mean(ds):+.4f} min {min(ds):+.4f} max {max(ds):+.4f} (>0 in {sum(x>0 for x in ds)}/5)")
    log("SHIP-EVAL LW if mean >+0.003 AND >=4/5 positive.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
