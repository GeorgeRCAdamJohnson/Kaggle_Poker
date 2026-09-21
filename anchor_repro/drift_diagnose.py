"""Why does our local ranking disagree with the LB (scale block: local-up, LB-down, twice)?

Hypothesis: it's not noise, it's PHASE DRIFT. Our held-out gate holds out dev TABLES (same time phase
as train), so it can't see dev->eval PHASE overfitting. The scale block is per-street action COUNTS,
which may drift between the development and evaluation time phases; the sequence encoder was pretrained
on dev+eval POOLED so it's phase-robust. Test directly:
  For each feature family, train a classifier to distinguish DEV pairs from EVAL pairs (drift AUC).
  High drift AUC = the feature family differs by phase -> a dev-tuned model overfits phase, LB regresses.
Families: scale-block per-street feats, the sequence embedding, and the base policy pvf feats.
If scale-block drift AUC >> seq-embedding drift AUC, hypothesis CONFIRMED (and the gate needs a phase split).
Run:  python -m anchor_repro.drift_diagnose
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.scale_perstreet import pair_block, build_block
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

def drift_auc(devX, evX, name):
    n=min(len(devX),len(evX),40000)
    rng=np.random.RandomState(0)
    d=devX.iloc[rng.choice(len(devX),n,replace=False)]; e=evX.iloc[rng.choice(len(evX),n,replace=False)]
    X=pd.concat([d,e],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    y=np.concatenate([np.ones(len(d)),np.zeros(len(e))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    m=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.1,verbose=-1,num_threads=-1),
                lgb.Dataset(Xtr,ytr),num_boost_round=120)
    auc=roc_auc_score(yte,m.predict(Xte))
    print(f"  {name:28} dev-vs-eval drift AUC = {auc:.4f}  ({'HIGH DRIFT -> phase-overfit risk' if auc>0.75 else 'phase-robust' if auc<0.62 else 'moderate'})")
    return auc

def main():
    build_block()
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    # base policy pvf
    dpvf=pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(); epvf=pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas()
    PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
    # scale block
    dps=pair_block(STEP3/"dev_pair_hands.parquet","development").to_pandas(); eps=pair_block(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas()
    PS=[c for c in dps.columns if c!="pair_id"]
    # seq embedding
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); ee=pd.read_parquet(SEQ/"seq_emb_v2_eval.parquet")
    SEQF=[c for c in de.columns if c.startswith("seq_")]
    print("=== dev-vs-eval PHASE DRIFT by feature family (higher = more phase-specific = LB-regression risk) ===")
    # hosen42 base engineered feats (the ones already in the 0.804 model)
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    common=[c for c in base if c in evf.columns]
    drift_auc(dev[common], evf[common], "hosen42 base (in 0.804)")
    drift_auc(dpvf[PT], epvf[PT], "policy postflop+tight")
    drift_auc(dps[PS], eps[PS], "per-street SCALE block")
    drift_auc(de[SEQF], ee[SEQF], "learned sequence embedding")
    print("\nIf SCALE >> SEQ, the LB regression is PHASE DRIFT (gate holds out tables not phase) -> a real")
    print("systematic effect, not noise. The seq embedding is phase-robust (pretrained on dev+eval pooled).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
