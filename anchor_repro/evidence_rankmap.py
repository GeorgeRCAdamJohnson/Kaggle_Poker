"""rank:map learning-to-rank evidence head — the one non-redundant technique from the LB 0.82412 approach.

hosen42 AND our evidence use a BINARY classifier on is_evidence. sharif485's 0.824 approach uses XGBoost
LEARNING-TO-RANK (objective='rank:map', eval_metric='map@5', group=per-pair), PER-FAMILY + generic
fallback. This optimizes MAP@5 DIRECTLY instead of a proxy binary loss -> may break the evidence ceiling.

We train it on hosen42's RICH per-hand features (198-feat dev_hand_features, incl their loose/junk/
transfer/strength) so it's their features + the rank:map objective. Table-grouped 5-fold OOF, compare
within-pair MAP@5 per family vs the binary baseline (0.5276). If it wins -> assemble + submit.
Run:  python -m anchor_repro.evidence_rankmap
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, polars as pl, xgboost as xgb
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); DATA=Path("data/poker")
SEED,N_FOLDS=42,5; MIN_LISTED=40
DEV_HF=[STEP3/f"dev_hand_features_{k}.parquet" for k in range(4)]
FAMILIES=["directed_transfer","soft_play","coordinated_isolation"]
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
    dev_ev=pl.read_csv(DATA/"development_evidence.csv")
    labelled=pl.read_parquet(STEP3/"dev_pairs.parquet").filter(pl.col("is_labeled"))
    lab_ids=set(labelled["pair_id"].to_list())
    hf=pl.concat([pl.scan_parquet(f).filter(pl.col("pair_id").is_in(list(lab_ids))).collect() for f in DEV_HF])
    hf=hf.join(labelled.select(["pair_id","label","behavior_family"]),on="pair_id")
    hf=hf.join(dev_ev.select(["pair_id","hand_id",pl.lit(True).alias("is_evidence")]),on=["pair_id","hand_id"],how="left").with_columns(pl.col("is_evidence").fill_null(False))
    dh=hf.to_pandas()
    pos=dh[dh["label"]==1].copy()
    META={"pair_id","hand_id","table_id","hand_idx","player_1","player_2","label","behavior_family","is_evidence"}
    HF=[c for c in dh.columns if c not in META and pd.api.types.is_numeric_dtype(dh[c])]
    dh[HF]=dh[HF].astype("float32")
    pos=dh[dh["label"]==1].copy()
    TABLES=sorted(pl.read_parquet(STEP3/"dev_pairs.parquet")["table_id"].unique().to_list())
    rng=np.random.RandomState(SEED); TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}
    pos["fold"]=pos["table_id"].map(TF)
    log(f"positive hands {len(pos):,}, listed {int(pos['is_evidence'].sum())}, feats {len(HF)}")

    def fit_ranker(frame,seed):
        frame=frame.sort_values("pair_id")
        groups=frame.groupby("pair_id",sort=False).size().to_numpy()
        d=xgb.DMatrix(frame[HF].to_numpy(dtype=float),label=frame["is_evidence"].astype(int).to_numpy(),group=groups,missing=np.nan)
        p=dict(objective="rank:map",eta=0.1,max_depth=5,subsample=0.9,colsample_bytree=0.8,min_child_weight=5,
               seed=seed,eval_metric="map@5",tree_method="hist")
        return xgb.train(p,d,num_boost_round=180)
    def score(model,frame):
        if not len(frame): return np.zeros(0)
        return model.predict(xgb.DMatrix(frame[HF].to_numpy(dtype=float),missing=np.nan))
    def fam_rankers(frame,seed):
        out={}
        for f in FAMILIES:
            b=frame[frame["behavior_family"]==f]
            if b["is_evidence"].sum()>=MIN_LISTED: out[f]=fit_ranker(b,seed)
        return out

    oof=np.zeros(len(pos))
    for k in range(N_FOLDS):
        tr=pos["fold"]!=k; va=pos["fold"]==k
        gen=fit_ranker(pos[tr],SEED+11+k); per=fam_rankers(pos[tr],SEED+21+k)
        blk=pos[va]; sc=score(gen,blk)
        for f,mdl in per.items():
            m=(blk["behavior_family"]==f).to_numpy()
            if m.any(): sc[m]=score(mdl,blk[m])
        oof[va.to_numpy()]=sc
    pos["hs"]=oof
    overall=map5(pos,"hs")
    fams={f:map5(pos[pos["behavior_family"]==f],"hs") for f in FAMILIES}
    print(f"\n=== rank:map evidence MAP@5 = {overall:.4f}  {{ {', '.join(f'{k}:{v:.3f}' for k,v in fams.items())} }}")
    print(f"    binary baseline (evidence_edge2) = 0.5276  {{ directed:0.567, soft:0.555, iso:0.424 }}")
    print(f"    delta {overall-0.5276:+.4f}  ({'WINS -> assemble+submit' if overall>0.5276+0.003 else 'no clear gain'})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
