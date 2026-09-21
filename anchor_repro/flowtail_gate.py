"""VALUE-FLOW TAIL risk features (dossier §2.4.1 inspiration: value_flow_abs_p95 AUC 0.886 = the single
strongest measured separator, yet the current risk model aggregates transfer as SUM/RATE, not TAIL).

Hypothesis: the collusion tell is in the TAIL of directed chip flow, not the sum. A colluding pair has a
few hands with huge directed transfers (p95 spikes); summing dilutes them into the mass of normal hands.
Discovery measured p95 0.886 >> mean 0.799; current model uses sum/rate. Build per-pair TAIL quantiles of
|directed flow| and net_gap, FIELD-NORMALIZED (pair tail minus the members' own baseline vs the field, so
a globally loose/tilted player doesn't trigger it -> phase-robust, dodges §154/§155 walls).

Dodges both walls: supervised+mechanistic (no learned-rep fine-tune overfit §154); directed-flow contrast
(not anomaly §155). Gate on the eval-weighted phase gate (§147). SHIP if eval-weighted delta > +0.005.
Run:  python -m anchor_repro.flowtail_gate
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED,N_FOLDS=42,5
PT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min","both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def flow_tail(pair_hands_path, phase):
    """Per-pair TAIL statistics of directed flow, field-normalized. Uses player_hands net_bb.
    For each shared hand: directed flow = |p1_net_bb - p2_net_bb| (net_gap) and the transfer magnitude.
    Aggregate per pair: p90/p95/max/std of net_gap; and the FIELD baseline = each member's typical
    |net_bb| spread vs ALL their hands (not just shared) -> normalize."""
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","hand_idx","player_1","player_2"])
    PHH=STEP3/"player_hands.parquet"
    phh=pl.scan_parquet(PHH).filter(pl.col("phase")==phase).select(["hand_id","player_id","net_bb"])
    # join both members' net_bb per shared hand
    j=(ph.lazy()
       .join(phh.rename({"player_id":"player_1","net_bb":"p1_net"}),on=["hand_id","player_1"],how="inner")
       .join(phh.rename({"player_id":"player_2","net_bb":"p2_net"}),on=["hand_id","player_2"],how="inner")
       .with_columns((pl.col("p1_net")-pl.col("p2_net")).abs().alias("gap"),
                     pl.min_horizontal((-pl.col("p1_net")).clip(lower_bound=0),pl.col("p2_net").clip(lower_bound=0)).alias("t12"),
                     pl.min_horizontal((-pl.col("p2_net")).clip(lower_bound=0),pl.col("p1_net").clip(lower_bound=0)).alias("t21"))
       .with_columns(pl.max_horizontal("t12","t21").alias("tdir")))
    agg=(j.group_by("pair_id").agg(
            pl.col("gap").quantile(0.90).alias("gap_p90"), pl.col("gap").quantile(0.95).alias("gap_p95"),
            pl.col("gap").max().alias("gap_max"), pl.col("gap").std().alias("gap_std"), pl.col("gap").mean().alias("gap_mean"),
            pl.col("tdir").quantile(0.90).alias("tdir_p90"), pl.col("tdir").quantile(0.95).alias("tdir_p95"),
            pl.col("tdir").max().alias("tdir_max"), pl.col("tdir").mean().alias("tdir_mean"),
            pl.len().alias("nsh"))
         .collect(engine="streaming"))
    # FIELD baseline per player: spread of |net_bb| over ALL their hands (this phase)
    field=(phh.group_by("player_id").agg(pl.col("net_bb").abs().quantile(0.95).alias("f_p95"),
                                         pl.col("net_bb").abs().mean().alias("f_mean")).collect(engine="streaming"))
    fmap95=dict(zip(field["player_id"],field["f_p95"])); # per-player field tail
    # normalize pair tail by the max of the two members' field tail (so a naturally swingy player doesn't trigger)
    m=ph.select(["pair_id","player_1","player_2"]).unique().to_pandas()
    m["f1"]=m["player_1"].map(fmap95); m["f2"]=m["player_2"].map(fmap95)
    m["field_p95"]=np.maximum(m["f1"].fillna(0),m["f2"].fillna(0))
    a=agg.to_pandas().merge(m[["pair_id","field_p95"]],on="pair_id",how="left")
    for c in ["gap_p95","gap_p90","gap_max","tdir_p95","tdir_p90","tdir_max"]:
        a[f"{c}_fn"]=a[c]/(a["field_p95"]+1.0)   # field-normalized tail
    return a

FT=["gap_p90","gap_p95","gap_max","gap_std","gap_mean","tdir_p90","tdir_p95","tdir_max","tdir_mean",
    "gap_p95_fn","gap_p90_fn","gap_max_fn","tdir_p95_fn","tdir_p90_fn","tdir_max_fn"]

def main():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    dev=dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(),on="pair_id",how="left")
    evf=evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(),on="pair_id",how="left")
    ft_d=flow_tail(STEP3/"dev_pair_hands.parquet","development"); ft_e=flow_tail(STEP3/"eval_pair_hands.parquet","evaluation")
    ft_d["pair_id"]=ft_d["pair_id"].astype(str); ft_e["pair_id"]=ft_e["pair_id"].astype(str)
    dev=dev.merge(ft_d[["pair_id"]+FT],on="pair_id",how="left"); evf=evf.merge(ft_e[["pair_id"]+FT],on="pair_id",how="left")
    for c in PT+FT: dev[c]=dev[c].astype("float32").fillna(0); evf[c]=evf[c].astype("float32").fillna(0)
    # seq_risk from cache (current best config includes it)
    de=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()

    # standalone AUC of the tail feats (reproduce the 0.886 claim on dev labels, labelled only)
    from sklearn.metrics import roc_auc_score
    lab=labm
    log("standalone dev-label AUC (labelled pairs, matches §2.4.1 confirmed-negative setting):")
    for c in ["gap_p95","gap_p95_fn","tdir_p95","tdir_p95_fn","gap_mean"]:
        s=dev.loc[lab,c].to_numpy(); yy=y[lab]
        log(f"  {c:14s} AUC {roc_auc_score(yy,s):.4f}")

    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    de_seqf=[c for c in de.columns if c.startswith("seq_")]
    dev=dev.merge(de,on="pair_id",how="left")
    for c in de_seqf: dev[c]=dev[c].astype("float32").fillna(0)
    # distill seq_risk (dev only, for the gate's current-best config)
    def distill():
        Xd=dev[de_seqf].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
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
    base=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
          and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in FT
          and c not in de_seqf and not c.startswith("seq_") and c!="seq_risk"]

    # eval-importance weights (§147 phase gate)
    shared=[c for c in base if c in evf.columns]; n=min(len(dev),len(evf),40000)
    ds=dev.iloc[rng.choice(len(dev),n,replace=False)]; es=evf.iloc[rng.choice(len(evf),n,replace=False)]
    Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    yc=np.concatenate([np.zeros(n),np.ones(n)])
    dc=lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=-1),lgb.Dataset(Xc,yc),num_boost_round=200)
    pe=dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()

    heldout=[3,4]; heldm=np.isin(folds,heldout)
    def oof(feats):
        X=dev[feats].astype("float32"); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750); o1[te]=m.predict(X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for sd in [SEED,SEED+11,SEED+23]:
                m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750); o[te]+=m.predict(X[te])/3
        return o
    def gate(feats,tag):
        o=oof(feats); hp=average_precision_score(y[heldm],o[heldm]); hw=average_precision_score(y[heldm],o[heldm],sample_weight=w_eval[heldm])
        log(f"[{tag}] held-out PLAIN {hp:.4f} | EVAL-WEIGHTED {hw:.4f}"); return hp,hw
    print("\n=== FLOW-TAIL gate (current best = base+PT+seq_risk) ===")
    b_p,b_w=gate(base+PT+["seq_risk"],"base+PT+seq_risk")
    e_p,e_w=gate(base+PT+["seq_risk"]+FT,"+flow_tail")
    print(f"\n  flow-tail delta: plain {e_p-b_p:+.4f} | EVAL-WEIGHTED {e_w-b_w:+.4f}")
    print("  SHIP if EVAL-WEIGHTED > +0.005 (Rule 13 threshold for risk).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
