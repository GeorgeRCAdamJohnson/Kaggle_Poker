"""Evidence rebuild v2: add per-hand POLICY-SURPRISE + WITHIN-PAIR percentile ranks to the ranker.

MAP@5 is within-pair ranking, so the signal that matters is "which of THIS pair's hands is the most
anomalous joint policy deviation relative to their other hands". hosen42's ranker _prank's their own
features but has NO postflop/tight policy surprise. We add, per (pair,hand):
  joint policy surprise = min/sum of the two members' (loose, tight, postflop_loose, total_loose)
  + the WITHIN-PAIR percentile rank of each (over the pair's shared hands) — the ranking-relevant form.
The planted evidence hands should be the top-percentile joint-deviation hands within each pair.
Judged within-pair MAP@5 per family (baseline 0.5118; iso weakest 0.40) vs the baseline ranker.
Run:  python -m anchor_repro.evidence_edge2
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

def map5(df,score_col):
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
    labelled=pl.read_parquet(STEP3/"dev_pairs.parquet").filter(pl.col("is_labeled"))
    lab_ids=set(labelled["pair_id"].to_list())
    hf=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(lab_ids))).collect() for f in DEV_HF])
    hf=hf.join(labelled.select(["pair_id","label","behavior_family"]),on="pair_id")
    hf=hf.join(dev_ev.select(["pair_id","hand_id",pl.lit(True).alias("is_evidence")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_evidence").fill_null(False))
    # per-hand policy surprise for both members
    S=["loose","tight","postflop_loose","total_loose"]
    sur=pl.scan_parquet(PER).filter(pl.col("phase")=="development").select(["hand_id","player_id"]+S)
    hf=(hf.join(sur.rename({c:f"p1_{c}" for c in S}).rename({"player_id":"player_1"}).collect(),on=["hand_id","player_1"],how="left")
          .join(sur.rename({c:f"p2_{c}" for c in S}).rename({"player_id":"player_2"}).collect(),on=["hand_id","player_2"],how="left"))
    NEW=[]
    for c in S:
        hf=hf.with_columns((pl.col(f"p1_{c}").fill_null(0)+pl.col(f"p2_{c}").fill_null(0)).alias(f"jt_{c}_sum"),
                           pl.min_horizontal(pl.col(f"p1_{c}").fill_null(0),pl.col(f"p2_{c}").fill_null(0)).alias(f"jt_{c}_min"))
        NEW += [f"jt_{c}_sum", f"jt_{c}_min"]
    # WITHIN-PAIR percentile ranks of the joint surprise (the ranking-relevant form)
    for c in NEW[:]:
        hf=hf.with_columns((pl.col(c).rank(method="average").over("pair_id")/pl.len().over("pair_id")).cast(pl.Float32).alias(f"{c}_prank"))
        NEW.append(f"{c}_prank")
    dh=hf.to_pandas()
    log(f"labelled hand rows {len(dh):,}, evidence {int(dh['is_evidence'].sum())}, NEW feats {len(NEW)}")

    META={"pair_id","hand_id","table_id","hand_idx","player_1","player_2","label","behavior_family","is_evidence"}
    BASE=[c for c in dh.columns if c not in META and c not in NEW and dh[c].dtype!=object]
    dh[BASE+NEW]=dh[BASE+NEW].astype("float32").fillna(0)
    TABLES=sorted(pl.read_parquet(STEP3/"dev_pairs.parquet")["table_id"].unique().to_list())
    rng=np.random.RandomState(SEED); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}
    dh["fold"]=dh["table_id"].map(TF); pos=(dh["label"]==1).to_numpy()
    P=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    def cv(feats,tag):
        dh["hs"]=0.0
        for f in range(N_FOLDS):
            trp=pos&(dh["fold"]!=f).to_numpy(); te=(dh["fold"]==f).to_numpy()
            ms=[lgb.train({**P,"seed":sd},lgb.Dataset(dh.loc[trp,feats],dh.loc[trp,"is_evidence"].astype(int)),num_boost_round=400) for sd in [SEED,SEED+7]]
            dh.loc[te,"hs"]=np.mean([m.predict(dh.loc[te,feats]) for m in ms],axis=0)
        o=map5(dh[pos],"hs")
        fams={fam:map5(dh[pos&(dh["behavior_family"]==fam).to_numpy()],"hs") for fam in ["directed_transfer","soft_play","coordinated_isolation"]}
        print(f"  [{tag}] MAP@5={o:.4f}  {{ {', '.join(f'{k}:{v:.3f}' for k,v in fams.items())} }}")
        return o
    log("baseline ranker..."); mb=cv(BASE,"baseline")
    log("+policy surprise+prank..."); mo=cv(BASE+NEW,"+policy")
    print(f"\n  MAP@5 delta {mo-mb:+.4f}  ({'WINS' if mo>mb+0.003 else 'no clear gain'})  (hosen42 0.5118, host-metric weight 0.20)")
    # feature importance of the NEW feats
    dh["hs"]=0.0
    m=lgb.train(P,lgb.Dataset(dh.loc[pos,BASE+NEW],dh.loc[pos,"is_evidence"].astype(int)),num_boost_round=400)
    imp=pd.Series(m.feature_importance("gain"),index=BASE+NEW).sort_values(ascending=False)
    print("  top NEW feats among all:", [f for f in imp.index[:25] if f in NEW])
    return 0

if __name__=="__main__":
    raise SystemExit(main())
