"""Augment hosen42's EVIDENCE ranker with per-hand POSTFLOP-POLICY surprise; judge within-pair MAP@5.

Evidence is within-pair hand ranking over KNOWN positives -> base-rate-wall-free (the risk-side
false-positive problem does NOT apply). hosen42's per-hand features have preflop loose/junk but NO
postflop policy surprise. We join our postflop_loose / fold_strong / passive_junk (both members) onto
their dev hand-feature matrix and re-run their EXACT evidence-ranker 5-fold CV (rank:pairwise-style
lgb binary on is_evidence, seeds [42,49], 400 rounds), comparing MAP@5 vs their 0.5118, per family.
Run:  python -m anchor_repro.evidence_edge
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
SEED,N_FOLDS=42,5
DEV_HF=[STEP3/f"dev_hand_features_{k}.parquet" for k in range(4)]
PER=EDGE/"per_hand_surprise.parquet"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def map5_within_pairs(df, score_col):
    vals=[]
    d=df.sort_values(["pair_id",score_col,"pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
    for _,g in d.groupby("pair_id",sort=False):
        rel=g["is_evidence"].to_numpy(); nr=int(rel.sum())
        if nr==0: continue
        top=rel[:5]; hits=np.cumsum(top)
        vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(nr,5)))
    return float(np.mean(vals)) if vals else 0.0

def main():
    dev_labels=pl.read_csv(DATA/"development_labels.csv")
    dev_ev=pl.read_csv(DATA/"development_evidence.csv")
    lab=dev_labels.with_columns(pl.min_horizontal("player_1","player_2").alias("p_low"),pl.max_horizontal("player_1","player_2").alias("p_high"))
    labelled=pl.read_parquet(STEP3/"dev_pairs.parquet").filter(pl.col("is_labeled"))
    lab_ids=set(labelled["pair_id"].to_list())
    # their per-(pair,hand) features for labelled pairs
    hf=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(lab_ids))).collect() for f in DEV_HF])
    hf=hf.join(labelled.select(["pair_id","label","behavior_family"]),on="pair_id")
    hf=hf.join(dev_ev.select(["pair_id","hand_id",pl.lit(True).alias("is_evidence")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_evidence").fill_null(False))
    # OUR postflop per-hand surprise for both members
    sur=pl.scan_parquet(PER).filter(pl.col("phase")=="development").select(["hand_id","player_id","postflop_loose","postflop_loose_max","fold_strong","passive_junk"])
    hf=(hf.join(sur.rename({c:f"p1_{c}" for c in ["postflop_loose","postflop_loose_max","fold_strong","passive_junk"]}).rename({"player_id":"player_1"}).collect(),on=["hand_id","player_1"],how="left")
          .join(sur.rename({c:f"p2_{c}" for c in ["postflop_loose","postflop_loose_max","fold_strong","passive_junk"]}).rename({"player_id":"player_2"}).collect(),on=["hand_id","player_2"],how="left"))
    NEW=[]
    for c in ["postflop_loose","postflop_loose_max","fold_strong","passive_junk"]:
        hf=hf.with_columns((pl.col(f"p1_{c}").fill_null(0)+pl.col(f"p2_{c}").fill_null(0)).alias(f"{c}_sum"),
                           pl.min_horizontal(pl.col(f"p1_{c}").fill_null(0),pl.col(f"p2_{c}").fill_null(0)).alias(f"{c}_min"))
        NEW+= [f"{c}_sum",f"{c}_min"]
    dh=hf.to_pandas()
    log(f"labelled hand rows {len(dh):,}, evidence {int(dh['is_evidence'].sum())}")

    META={"pair_id","hand_id","table_id","hand_idx","player_1","player_2","label","behavior_family","is_evidence"}
    BASE=[c for c in dh.columns if c not in META and c not in NEW and dh[c].dtype!=object]
    OURS=BASE+NEW
    dh[BASE+NEW]=dh[BASE+NEW].astype("float32").fillna(0)

    TABLES=sorted(pl.read_parquet(STEP3/"dev_pairs.parquet")["table_id"].unique().to_list())
    rng=np.random.RandomState(SEED); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}
    dh["fold"]=dh["table_id"].map(TF)
    pos_mask=(dh["label"]==1).to_numpy()
    P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    def cv_map5(feats,tag):
        dh["hs"]=0.0
        for f in range(N_FOLDS):
            trp=pos_mask&(dh["fold"]!=f).to_numpy(); te=(dh["fold"]==f).to_numpy()
            ms=[lgb.train({**P,"seed":sd},lgb.Dataset(dh.loc[trp,feats],dh.loc[trp,"is_evidence"].astype(int)),num_boost_round=400) for sd in [SEED,SEED+7]]
            dh.loc[te,"hs"]=np.mean([m.predict(dh.loc[te,feats]) for m in ms],axis=0)
        overall=map5_within_pairs(dh[pos_mask],"hs")
        fams={fam:map5_within_pairs(dh[pos_mask & (dh["behavior_family"]==fam).to_numpy()],"hs") for fam in ["directed_transfer","soft_play","coordinated_isolation"]}
        print(f"  [{tag}] MAP@5={overall:.4f}  per-family {{ {', '.join(f'{k}:{v:.3f}' for k,v in fams.items())} }}")
        return overall
    log("baseline evidence ranker..."); mb=cv_map5(BASE,"baseline")
    log("ours (+postflop) ..."); mo=cv_map5(OURS,"+postflop")
    print(f"\n  MAP@5 delta {mo-mb:+.4f}  ({'WINS' if mo>mb+0.003 else 'no clear gain'})   (hosen42 baseline 0.5118)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
