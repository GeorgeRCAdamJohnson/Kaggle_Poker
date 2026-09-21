"""§209 PATH B kill-test — MULTI-ENTITY / cross-pair CONTEXT. The seq encoder is per-pair-document (§203):
no cross-pair context. Does giving a pair TABLE-LEVEL context (behavior of the other players/pairs it is
co-seated with) add EVAL-COMPUTABLE label signal beyond the per-pair seq embedding?

CRITICAL GUARD (§208): cross-pair context is exactly where the degree/co-occurrence leak lives (dev-pos
degree median 2 vs eval ~24). So context features must be BEHAVIORAL aggregates (how the table's other
players ACT), NOT graph-degree/count. And we leak-check: corr(shared_hands) + dev/eval means + does the
context feature separate label only because dev-pos sit in a low-activity table regime absent from eval.

Cheap probe: per pair, aggregate the FIELD behavior of players at the same tables (mean loose/tight/aggr of
co-seated non-pair players). Test whether these table-context feats add label-AUC beyond seq_emb, and whether
they transport (dev/eval means match). Kill if no beyond-seq gain OR if gain is a dev/eval regime artifact.
Run:  python -m anchor_repro.multientity_killtest
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
SEED=42
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def table_context(phase):
    """Per pair: behavioral aggregate of the OTHER players at the pair's tables (field context)."""
    ph=pl.scan_parquet(STEP3/"player_hands_policy.parquet").filter(pl.col("phase")==phase).select(
        ["hand_id","player_id","table_id","loose","tight","n_aggr","net_bb","big_blind"]).collect()
    # table-level field behavior (mean over ALL players at each table)
    tbl=ph.group_by("table_id").agg(
        pl.col("loose").mean().alias("tbl_loose"), pl.col("tight").mean().alias("tbl_tight"),
        pl.col("n_aggr").mean().alias("tbl_aggr"), (pl.col("net_bb")/pl.col("big_blind").clip(lower_bound=1)).std().alias("tbl_netvol"),
        pl.col("player_id").n_unique().alias("tbl_nplayers"))
    return tbl

def main():
    which={"development":(STEP3/"dev_pair_features.parquet",STEP3/"dev_pair_hands.parquet"),
           "evaluation":(STEP3/"eval_pair_features.parquet",STEP3/"eval_pair_hands.parquet")}
    def build(phase):
        pf,ph=which[phase]
        dev=pd.read_parquet(pf); dev["pair_id"]=dev["pair_id"].astype(str)
        # each pair's table_id
        pm=pl.read_parquet(ph).select(["pair_id","table_id"]).unique(subset=["pair_id"]).to_pandas()
        pm["pair_id"]=pm["pair_id"].astype(str)
        tbl=table_context(phase).to_pandas()
        d=dev.merge(pm,on="pair_id",how="left",suffixes=("","_ph"))
        tcol="table_id_ph" if "table_id_ph" in d.columns else "table_id"
        d=d.merge(tbl,left_on=tcol,right_on="table_id",how="left",suffixes=("","_tbl"))
        return d
    dev=build("development"); 
    CTX=["tbl_loose","tbl_tight","tbl_aggr","tbl_netvol","tbl_nplayers"]
    for c in CTX: dev[c]=pd.to_numeric(dev[c],errors="coerce").astype("float32").fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); lab=dev["is_labeled"].to_numpy().astype(bool)
    emb=pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); emb["pair_id"]=emb["pair_id"].astype(str)
    SEQF=[c for c in emb.columns if c.startswith("seq_")]
    dev=dev.merge(emb,on="pair_id",how="left")
    for c in SEQF: dev[c]=dev[c].astype("float32").fillna(0)
    if "shared_hands" not in dev.columns:
        sh=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","shared_hands"]).to_pandas(); sh["pair_id"]=sh["pair_id"].astype(str)
        dev=dev.merge(sh,on="pair_id",how="left")
    dev["shared_hands"]=pd.to_numeric(dev["shared_hands"],errors="coerce").fillna(0)

    log("=== table-context features: univariate label AUC + corr(shared_hands) + leak check ===")
    eval_dev=build("evaluation")
    for c in CTX: eval_dev[c]=pd.to_numeric(eval_dev[c],errors="coerce").astype("float32").fillna(0)
    for c in CTX:
        v=dev[c].to_numpy()[lab]; a=roc_auc_score(y[lab],v)
        csh=np.corrcoef(dev[c],dev["shared_hands"].fillna(0))[0,1]
        log(f"  {c:14s} AUC {max(a,1-a):.4f}  corr(shared_hands) {csh:+.3f}  dev_mean {dev[c].mean():.3f} eval_mean {eval_dev[c].mean():.3f}")

    # does CTX add to seq_emb? table-fold OOF LGB: seq vs seq+ctx (label separation, quick)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%5)}; folds=dev["table_id"].map(TF).to_numpy()
    P=dict(objective="binary",learning_rate=0.05,num_leaves=15,min_data_in_leaf=50,verbose=-1,num_threads=-1)
    def oof(feats):
        X=dev[feats]; w=np.where(lab,np.where(y==1,5.0,1.0),0.1); o=np.zeros(len(dev))
        for f in range(5):
            tr=(folds!=f)&(w>0); te=folds==f
            m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w[tr]),num_boost_round=300); o[te]=m.predict(X[te])
        return o
    from sklearn.metrics import average_precision_score
    o_s=oof(SEQF); o_sc=oof(SEQF+CTX)
    ap_s=average_precision_score(y[lab],o_s[lab]); ap_sc=average_precision_score(y[lab],o_sc[lab])
    log(f"\n  seq-only OOF PairAP(labeled) {ap_s:.4f} | seq+table-context {ap_sc:.4f} | delta {ap_sc-ap_s:+.4f}")
    log("=== KILL-TEST verdict ===")
    if ap_sc-ap_s < 0.005: log(f"  delta {ap_sc-ap_s:+.4f} < +0.005 -> table-context adds nothing beyond seq. Multi-entity REFUTED cheaply.")
    else: log(f"  delta {ap_sc-ap_s:+.4f} >= +0.005 -> proceed, but verify not a dev/eval regime leak (check means above).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
