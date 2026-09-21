"""§208 Task 1 — KILL-TEST for Teacher-Student LUPI on the §205 label-only timing signal.
Question: can EVAL-AVAILABLE features reconstruct the PRIVILEGED (label-conditioned) timing target?
If reconstruction is near-zero, the timing signal is genuinely label-only (like rings §199) and a Student
can never proxy it -> Teacher-Student REFUTED cheaply. If reconstruction is real, proceed to build Teacher.

Privileged target (per positive pair, needs labels): planted-hand spacing regularity = 1/(cv of gaps between
consecutive PLANTED hands). Also a coarser per-ALL-pair proxy: we can only build the target on the 372
positives (planted hands defined only there), so we test reconstruction on those 372.
Eval-available inputs: seq_emb_v2_dev (384 dims) + dev_pair_features (base) + all-shared-hand timing feats.
Pre-registered STOP: recon R2 < 0.10 => label-only, STOP. > 0.30 => proceed to Teacher. between => report.
Run:  python -m anchor_repro.lupi_killtest
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
D=Path("data/poker"); CACHE=Path("outputs/poker_collusion/generator_cache"); STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"

def privileged_timing_target():
    """Per positive pair: regularity of planted-hand spacing (needs labels). Lower cv of gaps = more regular."""
    pos=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select("pair_id")
    ev=pl.read_csv(D/"development_evidence.csv").select(["pair_id","hand_id"])
    shared=pl.scan_parquet(CACHE/"dev_layer2.parquet").select(["pair_id","hand_id"]).collect().join(pos,on="pair_id",how="inner")
    hands=pl.scan_parquet(D/"hands.parquet").select(["hand_id","started_at"]).collect()
    shared=shared.join(hands,on="hand_id",how="left")
    plset=set((ev["pair_id"]+"|"+ev["hand_id"]).to_list())
    sh=shared.sort(["pair_id","started_at"]).with_columns((pl.col("pair_id")+"|"+pl.col("hand_id")).is_in(list(plset)).alias("planted"))
    sh=sh.with_columns(pl.col("hand_id").cum_count().over("pair_id").alias("ord"))
    rows=[]
    for pid,g in sh.group_by("pair_id"):
        pid=pid[0] if isinstance(pid,tuple) else pid  # group key is a tuple
        pord=np.sort(g.filter(pl.col("planted"))["ord"].to_numpy())
        if len(pord)<3: rows.append((str(pid),np.nan)); continue
        gaps=np.diff(pord).astype(float)
        cv=gaps.std()/(gaps.mean()+1e-9)
        rows.append((str(pid), 1.0/(cv+0.5)))  # regularity score: higher = more evenly spaced
    return pl.DataFrame(rows,schema=["pair_id","priv_timing"],orient="row")

def eval_available_features():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    emb=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); emb["pair_id"]=emb["pair_id"].astype(str)
    SEQF=[c for c in emb.columns if c.startswith("seq_")]
    # all-shared-hand timing (label-free), reuse timing_usability construction
    shared=pl.scan_parquet(CACHE/"dev_layer2.parquet").select(["pair_id","hand_id"]).collect()
    hands=pl.scan_parquet(D/"hands.parquet").select(["hand_id","started_at"]).collect()
    shared=shared.join(hands,on="hand_id",how="left").sort(["pair_id","started_at"])
    shared=shared.with_columns(pl.col("started_at").cast(pl.Int64).alias("ts"))
    shared=shared.with_columns(pl.col("ts").diff().over("pair_id").alias("gap"))
    tf=shared.group_by("pair_id").agg(
        (pl.col("gap").drop_nulls().std()/(pl.col("gap").drop_nulls().mean()+1)).alias("t_burst"),
        pl.col("gap").drop_nulls().median().alias("t_medgap"),
        pl.col("gap").drop_nulls().min().alias("t_mingap")).to_pandas()
    tf["pair_id"]=tf["pair_id"].astype(str)
    m=dev.merge(emb,on="pair_id",how="left").merge(tf,on="pair_id",how="left")
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    basef=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"]
    feats=basef+SEQF+["t_burst","t_medgap","t_mingap"]
    for c in feats: m[c]=pd.to_numeric(m[c],errors="coerce").astype("float32").fillna(0)
    return m, feats

def main():
    tgt=privileged_timing_target().to_pandas(); tgt["pair_id"]=tgt["pair_id"].astype(str)
    m,feats=eval_available_features()
    d=m.merge(tgt,on="pair_id",how="inner").dropna(subset=["priv_timing"])
    print(f"positive pairs with privileged timing target + eval features: {len(d)}")
    X=d[feats].to_numpy(); y=d["priv_timing"].to_numpy()
    print(f"target priv_timing: mean {y.mean():.3f} std {y.std():.3f}")
    # reconstruct with ridge + gbm (nonlinear), 5-fold
    pr=cross_val_predict(Ridge(alpha=10.0),X,y,cv=5); r2r=r2_score(y,pr)
    pg=cross_val_predict(HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05),X,y,cv=5); r2g=r2_score(y,pg)
    print(f"\n=== reconstruction of privileged timing from EVAL-AVAILABLE features ===")
    print(f"  ridge  5-fold R2 = {r2r:+.4f}")
    print(f"  gbm    5-fold R2 = {r2g:+.4f}")
    best=max(r2r,r2g)
    print(f"\n=== PRE-REGISTERED STOP (Task 1) ===")
    if best<0.10: print(f"  R2 {best:.3f} < 0.10 -> timing signal is LABEL-ONLY (unreconstructable). Teacher-Student REFUTED. STOP.")
    elif best>0.30: print(f"  R2 {best:.3f} > 0.30 -> reconstructable. PROCEED to build Teacher (Task 2).")
    else: print(f"  R2 {best:.3f} in [0.10,0.30] -> weak/partial. Report; lean STOP unless Teacher uplift is large.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
