"""DEEP evidence-ranker tuning (competition endgame). Loads data ONCE, sweeps every knob against the
0.5989 FAM baseline with the exact host map5. Evidence is phase-immune (OOF==LB) and evidence-only swaps
keep risk+behavior byte-identical, so any OOF improvement is a safe LB improvement.

Stages (each prints MAP@5 overall + per-family; baseline FAM=0.5989):
  0 baseline GEN / FAM / BLEND (re-confirm)
  1 LightGBM hyperparameter grid on FAM (num_leaves, min_data_in_leaf, lr, rounds, feat_frac, l2)
  2 seed-count / ensemble size; routing (GEN vs FAM vs blend weights)
  3 feature sets (HAND_FEATS only, +ctx, +per-hand seq emb, +field-relative residuals)
  4 tie-break secondary-sort sweep
  5 per-family targeted tuning (weakest family)
Run:  python -m anchor_repro.evidence_tune --stage 0   (etc; --stage all for everything)
"""
from __future__ import annotations
import argparse, time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, map5, TARGET_BEHAVIORS, log
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"; EQ=STEP3/"edge"/"equity"
SEED=42
BASE_PARAMS=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=8)
T0=time.time()

_DH=None; _FEATS=None
def load(extra_feats=None):
    """Load labelled per-hand frame + ctx emb once; optionally join per-pair field-relative residuals."""
    global _DH,_FEATS
    if _DH is not None and extra_feats is None: return _DH,_FEATS
    dh=build_dh()
    emb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(emb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    _DH=dh; _FEATS=HAND_FEATS+CTXCOLS
    return _DH,_FEATS

def fam_ranker(dh, feats, params, rounds, seeds):
    dh=dh.copy(); dh["hand_score"]=0.0; folds=sorted(dh["fold"].unique())
    for f in folds:
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            models=[lgb.train({**params,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=rounds) for sd in seeds]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def fam_ranker_perfam(dh, feats, fam_params, fam_rounds, seeds):
    """Per-family ranker where EACH family gets its own params/rounds (dicts keyed by family)."""
    dh=dh.copy(); dh["hand_score"]=0.0; folds=sorted(dh["fold"].unique())
    for f in folds:
        te=(dh["fold"]==f).to_numpy()
        for fam in TARGET_BEHAVIORS:
            tr=((dh["fold"]!=f)&(dh["label"]==1)&(dh["behavior_family"]==fam)).to_numpy()
            if tr.sum()<50: continue
            pr=fam_params[fam]; rd=fam_rounds[fam]
            models=[lgb.train({**pr,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=rd) for sd in seeds]
            tgt=te & (dh["behavior_family"]==fam).to_numpy()
            if tgt.sum()==0: continue
            dh.loc[tgt,"hand_score"]=np.mean([m.predict(dh.loc[tgt,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def gen_ranker(dh, feats, params, rounds, seeds):
    dh=dh.copy(); dh["hand_score"]=0.0; folds=sorted(dh["fold"].unique())
    for f in folds:
        tr=((dh["fold"]!=f)&(dh["label"]==1)).to_numpy(); te=(dh["fold"]==f).to_numpy()
        models=[lgb.train({**params,"seed":sd},lgb.Dataset(dh.loc[tr,feats],dh.loc[tr,"is_evidence"].astype(int)),num_boost_round=rounds) for sd in seeds]
        dh.loc[te,"hand_score"]=np.mean([m.predict(dh.loc[te,feats]) for m in models],axis=0)
    return dh["hand_score"].to_numpy()

def score(dh, hs, tag, sort2="pot_bb"):
    d=dh.copy(); d["hand_score"]=hs
    d=d.sort_values(["pair_id","hand_score",sort2,"hand_id"],ascending=[True,False,False,True],kind="mergesort")
    pos=d[d["label"]==1]
    def m5(sub):
        vals=[]
        for _,g in sub.groupby("pair_id",sort=False):
            rel=g["is_evidence"].to_numpy(); nr=int(rel.sum())
            if nr==0: continue
            top=rel[:5]; hits=np.cumsum(top); vals.append(float(np.sum((hits/np.arange(1,len(top)+1))*top)/min(nr,5)))
        return float(np.mean(vals)) if vals else 0.0
    ov=m5(pos); fam={fm:m5(pos[pos["behavior_family"]==fm]) for fm in TARGET_BEHAVIORS}
    log(f"[{tag}] MAP@5={ov:.4f}  DT {fam['directed_transfer']:.4f} SP {fam['soft_play']:.4f} CI {fam['coordinated_isolation']:.4f}")
    return ov

def stage0(dh,feats):
    log("=== STAGE 0: baseline (FAM target 0.5989) ===")
    g=gen_ranker(dh,feats,BASE_PARAMS,400,[SEED,SEED+7]); score(dh,g,"GEN base")
    f=fam_ranker(dh,feats,BASE_PARAMS,400,[SEED,SEED+7]); score(dh,f,"FAM base")
    d=dh.copy(); d["g"]=g; d["f"]=f
    d["gr"]=d.groupby("pair_id")["g"].rank(pct=True); d["fr"]=d.groupby("pair_id")["f"].rank(pct=True)
    score(dh,(d["gr"]+d["fr"]).to_numpy(),"BLEND 50/50")

def stage1(dh,feats):
    log("=== STAGE 1: LightGBM hyperparameter grid on FAM ===")
    for nl in [15,31,63,127]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,"num_leaves":nl},400,[SEED,SEED+7]); score(dh,f,f"num_leaves={nl}")
    for md in [20,50,100,200]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,"min_data_in_leaf":md},400,[SEED,SEED+7]); score(dh,f,f"min_data={md}")
    for lr,rd in [(0.03,700),(0.05,400),(0.02,1000),(0.1,200)]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,"learning_rate":lr},rd,[SEED,SEED+7]); score(dh,f,f"lr={lr} rounds={rd}")
    for ff in [0.5,0.7,0.8,1.0]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,"feature_fraction":ff},400,[SEED,SEED+7]); score(dh,f,f"feat_frac={ff}")
    for l2 in [1.0,5.0,10.0,30.0]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,"lambda_l2":l2},400,[SEED,SEED+7]); score(dh,f,f"lambda_l2={l2}")

def stage2(dh,feats):
    log("=== STAGE 2: ensemble size + routing/blend ===")
    for seeds in [[SEED],[SEED,SEED+7],[SEED,SEED+7,SEED+13,SEED+23],[SEED+i for i in range(8)]]:
        f=fam_ranker(dh,feats,BASE_PARAMS,400,seeds); score(dh,f,f"seeds={len(seeds)}")
    g=gen_ranker(dh,feats,BASE_PARAMS,400,[SEED,SEED+7]); f=fam_ranker(dh,feats,BASE_PARAMS,400,[SEED,SEED+7])
    d=dh.copy(); d["g"]=g; d["f"]=f
    d["gr"]=d.groupby("pair_id")["g"].rank(pct=True); d["fr"]=d.groupby("pair_id")["f"].rank(pct=True)
    for wf in [0.0,0.25,0.5,0.6,0.7,0.75,0.8,0.9,1.0]:
        score(dh,(wf*d["fr"]+(1-wf)*d["gr"]).to_numpy(),f"blend fam_w={wf}")

def stage3(dh,feats):
    log("=== STAGE 3: feature sets ===")
    from anchor_repro.seq_evidence import SEQCOLS
    f=fam_ranker(dh,HAND_FEATS,BASE_PARAMS,400,[SEED,SEED+7]); score(dh,f,"HAND_FEATS only")
    f=fam_ranker(dh,HAND_FEATS+CTXCOLS,BASE_PARAMS,400,[SEED,SEED+7]); score(dh,f,"HAND_FEATS+ctx (base)")
    # + field-relative residuals (per-pair, broadcast to hands)
    for name,path,cols in [("eqFR","field_relative_development.parquet",None),("jaFR","ja_field_relative_development.parquet",None)]:
        try:
            r=pl.read_parquet(EQ/path).to_pandas(); r["pair_id"]=r["pair_id"].astype(str)
            rc=[c for c in r.columns if c!="pair_id"]
            dh2=dh.merge(r,on="pair_id",how="left"); 
            for c in rc: dh2[c]=dh2[c].astype("float32").fillna(0)
            fr=fam_ranker(dh2,HAND_FEATS+CTXCOLS+rc,BASE_PARAMS,400,[SEED,SEED+7]); score(dh2,fr,f"+{name} ({len(rc)} cols)")
        except Exception as ex: log(f"  {name} skip: {ex}")

def stage4(dh,feats):
    log("=== STAGE 4: tie-break secondary sort ===")
    f=fam_ranker(dh,feats,BASE_PARAMS,400,[SEED,SEED+7])
    for s2 in ["pot_bb","net_gap","transfer_pot_ratio","pair_contrib_bb","max_amount_bb"]:
        if s2 in dh.columns: score(dh,f,f"sort2={s2}",sort2=s2)

def stage5(dh,feats):
    log("=== STAGE 5: low feat_frac + winner combinations (baseline 0.5989, best-so-far 0.6025 @ff0.5) ===")
    for ff in [0.25,0.30,0.35,0.40,0.45,0.50]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,"feature_fraction":ff},400,[SEED,SEED+7]); score(dh,f,f"ff={ff}")
    log("-- combine winners --")
    for tag,p in [
        ("ff0.5+md100",{"feature_fraction":0.5,"min_data_in_leaf":100}),
        ("ff0.5+l2_1",{"feature_fraction":0.5,"lambda_l2":1.0}),
        ("ff0.5+md100+l2_1",{"feature_fraction":0.5,"min_data_in_leaf":100,"lambda_l2":1.0}),
        ("ff0.35+md100",{"feature_fraction":0.35,"min_data_in_leaf":100}),
        ("ff0.4+l2_1",{"feature_fraction":0.4,"lambda_l2":1.0}),
        ("ff0.4+md100+l2_1",{"feature_fraction":0.4,"min_data_in_leaf":100,"lambda_l2":1.0}),
    ]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,**p},400,[SEED,SEED+7]); score(dh,f,tag)
    log("-- best combo with more seeds (variance reduction) --")
    for tag,p,sc in [("ff0.5 4seeds",{"feature_fraction":0.5},[SEED,SEED+7,SEED+13,SEED+23]),
                     ("ff0.5+md100 4seeds",{"feature_fraction":0.5,"min_data_in_leaf":100},[SEED,SEED+7,SEED+13,SEED+23])]:
        f=fam_ranker(dh,feats,{**BASE_PARAMS,**p},400,sc); score(dh,f,tag)

def stage6(dh,feats):
    log("=== STAGE 6: per-family params (DT/SP fixed at ff0.25 winner; sweep CI's own params) ===")
    DTSP={"feature_fraction":0.25}
    # first: confirm DT/SP@ff0.25 + CI@baseline reference
    fp={"directed_transfer":{**BASE_PARAMS,**DTSP},"soft_play":{**BASE_PARAMS,**DTSP},"coordinated_isolation":{**BASE_PARAMS}}
    fr={fm:400 for fm in TARGET_BEHAVIORS}
    hs=fam_ranker_perfam(dh,feats,fp,fr,[SEED,SEED+7]); score(dh,hs,"DTSP=ff0.25 CI=base")
    # sweep CI's own knobs
    ci_grid=[
        ("CI ff0.8 md20",{"feature_fraction":0.8,"min_data_in_leaf":20}),
        ("CI ff1.0 md20",{"feature_fraction":1.0,"min_data_in_leaf":20}),
        ("CI ff0.5",{"feature_fraction":0.5}),
        ("CI ff0.25",{"feature_fraction":0.25}),
        ("CI nl15 md20",{"num_leaves":15,"min_data_in_leaf":20}),
        ("CI nl63",{"num_leaves":63}),
        ("CI l2_1 md20",{"lambda_l2":1.0,"min_data_in_leaf":20}),
        ("CI lr0.03 r700",{"learning_rate":0.03}),
    ]
    for tag,cip in ci_grid:
        rd={"directed_transfer":400,"soft_play":400,"coordinated_isolation":700 if "r700" in tag else 400}
        fp={"directed_transfer":{**BASE_PARAMS,**DTSP},"soft_play":{**BASE_PARAMS,**DTSP},"coordinated_isolation":{**BASE_PARAMS,**cip}}
        hs=fam_ranker_perfam(dh,feats,fp,rd,[SEED,SEED+7]); score(dh,hs,tag)
    # best-guess combined: DT/SP ff0.25, CI best from grid + 4 seeds
    log("-- combined per-family, 4 seeds --")
    fp={"directed_transfer":{**BASE_PARAMS,"feature_fraction":0.25},"soft_play":{**BASE_PARAMS,"feature_fraction":0.25},"coordinated_isolation":{**BASE_PARAMS,"feature_fraction":0.8,"min_data_in_leaf":20}}
    hs=fam_ranker_perfam(dh,feats,fp,{fm:400 for fm in TARGET_BEHAVIORS},[SEED,SEED+7,SEED+13,SEED+23]); score(dh,hs,"DTSP=ff0.25 CI=ff0.8md20 4seeds")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",default="0"); a=ap.parse_args()
    dh,feats=load()
    log(f"loaded {len(dh):,} labelled hands, {dh['is_evidence'].sum():,} evidence")
    stages={"0":stage0,"1":stage1,"2":stage2,"3":stage3,"4":stage4,"5":stage5,"6":stage6}
    if a.stage=="all":
        for s in ["0","1","2","3","4","5"]: stages[s](dh,feats)
    else:
        stages[a.stage](dh,feats)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
