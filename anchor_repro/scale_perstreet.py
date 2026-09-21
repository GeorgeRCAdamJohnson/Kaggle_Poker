"""SCALE play: large per-street x per-context feature block (the 0.824 '700-field' direction).

We keep testing 2-col channels that wash out in a 221-field tree. The 0.824 edge is a 700-field model.
Build a LARGE simultaneous block: per-STREET (flop/turn/river) policy-surprise + action-response +
equity-mistake, both members, aggregated to pairs (partner-vs-field where applicable). Dozens of cols
at once so the tree can find per-street collusion structure (e.g. soft-play shows on turn/river checks;
isolation on flop). Gated on the double gate vs the postflop+tight base.

Per-street per-(hand,player) from action_context + player_strength:
  loose_street{1,2,3}, aggr/call/check/fold counts per street, equity-mistake per street.
Aggregate to pair: for each street, both-members sum/min + field-normalized where a baseline exists.
Run:  python -m anchor_repro.scale_perstreet   (builds block, then double-gates it)
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
AC=STEP3/"action_context.parquet"; PS=STEP3/"player_strength.parquet"
BLOCK=EDGE/"perstreet_pair.parquet"
SEED,N_FOLDS=42,5
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def build_block():
    if BLOCK.exists():
        log("perstreet block cached"); return
    strength=pl.scan_parquet(PS).select(["hand_id","player_id","rk_flop","rk_turn","rk_river"])
    ac=(pl.scan_parquet(AC).filter(pl.col("street_no")>=1).join(strength,on=["hand_id","player_id"],how="left")
        .with_columns(pl.when(pl.col("street_no")==1).then(pl.col("rk_flop"))
                      .when(pl.col("street_no")==2).then(pl.col("rk_turn")).otherwise(pl.col("rk_river")).alias("rk"),
                      pl.col("players_active").clip(2,6).cast(pl.Float64).alias("k"),
                      (pl.col("to_call").cast(pl.Float64)/(pl.col("to_call").cast(pl.Float64)+pl.max_horizontal(pl.lit(1.0),pl.col("to_call_bb").cast(pl.Float64))+1e-6)).alias("pot_odds"))
        .with_columns(pl.when(pl.col("rk").is_null()).then(0.5).otherwise((pl.col("k")-pl.col("rk").cast(pl.Float64))/(pl.col("k")-1.0+1e-6)).clip(0,1).alias("equity")))
    # per (hand,player,street) aggregates
    per=ac.group_by(["hand_id","player_id","street_no"]).agg(
        (pl.col("action")=="check").sum().alias("n_check"),
        (pl.col("action")=="call").sum().alias("n_call"),
        (pl.col("action")=="fold").sum().alias("n_fold"),
        pl.col("is_aggr").sum().alias("n_aggr"),
        ((pl.col("action")=="call")&(pl.col("equity")<pl.col("pot_odds"))).sum().alias("bad_call"),
        ((pl.col("action")=="fold")&(pl.col("equity")>pl.col("pot_odds")+0.05)).sum().alias("bad_fold"),
        (pl.col("equity")*(pl.col("action")=="check").cast(pl.Float64)).max().alias("check_equity_max"),
    ).collect(engine="streaming")
    # pivot street into columns
    wide=per.pivot(on="street_no",index=["hand_id","player_id"],
                   values=["n_check","n_call","n_fold","n_aggr","bad_call","bad_fold","check_equity_max"],
                   aggregate_function="first").fill_null(0)
    ph=pl.scan_parquet(DATA/"hands.parquet").select(["hand_id","phase"]).collect()
    wide=wide.join(ph,on="hand_id",how="left")
    wide.write_parquet(BLOCK)
    log(f"perstreet per-hand block {wide.height:,} rows, {len(wide.columns)} cols")

def pair_block(pair_hands_path, phase):
    ph=pl.scan_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2"])
    w=pl.scan_parquet(BLOCK).filter(pl.col("phase")==phase).drop("phase")
    valcols=[c for c in pl.scan_parquet(BLOCK).collect_schema().names() if c not in ("hand_id","player_id","phase")]
    a=ph.join(w,left_on=["hand_id","player_1"],right_on=["hand_id","player_id"],how="left")
    b=ph.join(w,left_on=["hand_id","player_2"],right_on=["hand_id","player_id"],how="left")
    # per (pair,hand): both-min and sum of each per-street col
    j=a.select(["pair_id","hand_id"]+[pl.col(c).alias(f"a_{c}") for c in valcols]).join(
        b.select(["pair_id","hand_id"]+[pl.col(c).alias(f"b_{c}") for c in valcols]),on=["pair_id","hand_id"])
    exprs=[]
    for c in valcols:
        exprs.append((pl.col(f"a_{c}").fill_null(0)+pl.col(f"b_{c}").fill_null(0)).alias(f"ps_{c}_sum"))
        exprs.append(pl.min_horizontal(pl.col(f"a_{c}").fill_null(0),pl.col(f"b_{c}").fill_null(0)).alias(f"ps_{c}_min"))
    j=j.with_columns(exprs)
    aggcols=[f"ps_{c}_sum" for c in valcols]+[f"ps_{c}_min" for c in valcols]
    agg=[pl.col(c).mean().alias(f"{c}_mean") for c in aggcols]+[pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in aggcols]
    return j.group_by("pair_id").agg(agg).collect(engine="streaming")

def main():
    build_block()
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    from anchor_repro.policy_edge_pvf import pvf_features, PVF_FEATS
    dpvf=pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    dev=dev.merge(dpvf,on="pair_id",how="left")
    dps=pair_block(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    dev=dev.merge(dps,on="pair_id",how="left")
    PS_FEATS=[c for c in dps.columns if c!="pair_id"]
    for c in PVF_FEATS+PS_FEATS:
        if c in dev.columns: dev[c]=dev[c].astype("float32").fillna(0)
    log(f"per-street block: {len(PS_FEATS)} new pair feats")
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PS_FEATS and c not in PT and c not in PVF_FEATS]
    heldout=[3,4]; heldm=np.isin(folds,heldout)
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    def gate(feats,tag):
        X=dev[feats].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev)); tp_pool=~heldm
        for f in range(N_FOLDS):
            tr=tp_pool&(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); oof1[te]=m.predict(X[te])
        pos=oof1[labm&(y==1)&tp_pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp)&tp_pool; ps=unl&(oof1>=tp)&tp_pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(tp_pool,w2,0.0)
        oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=tp_pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); oof[te]+=m.predict(X[te])/3
        full=average_precision_score(y,oof); ho=average_precision_score(y[heldm],oof[heldm])
        log(f"[{tag}] full {full:.4f} | held-out {ho:.4f}"); return full,ho
    pt_f,pt_h=gate(base+PT,"postflop+tight (base)")
    sc_f,sc_h=gate(base+PT+PS_FEATS,"+per-street scale block")
    print(f"\n  scale delta vs PT: full {sc_f-pt_f:+.4f} | held-out {sc_h-pt_h:+.4f}  ({'STAGE' if (sc_f>pt_f+0.001 and sc_h>pt_h+0.001) else 'no'})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
