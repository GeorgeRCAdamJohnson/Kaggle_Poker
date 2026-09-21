"""TUNED family-conditional evidence submission (endgame tuning §evidence_tune).

Same pipeline as evidence_familycond_submit.py (swap ONLY evidence into the 0.83185 base; risk+behavior
byte-identical) but with the tuned per-family ranker params found by the OOF sweep:
  DT/SP: feature_fraction=0.25  (data-rich families like aggressive subsampling; OOF DT 0.6517 SP 0.6164)
  CI:    feature_fraction=0.8, min_data_in_leaf=20  (data-poor family likes gentler reg; OOF CI 0.5210)
  4 seeds (variance reduction; robust OOF ~0.6010 vs baseline 0.5989, +~0.002).
Writes a NEW file (does NOT touch the banked 0.83460). Verifies risk+behavior identical to base.
Run:  python -m anchor_repro.evidence_tuned_submit
"""
from __future__ import annotations
import time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
from pathlib import Path
from anchor_repro.seq_evidence import build_dh, HAND_FEATS, HAND_PARAMS, TARGET_BEHAVIORS
from anchor_repro.seq_ctx_evidence import CTXCOLS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); SEQ=STEP3/"edge"/"seq"
BASE_SUB=Path("outputs/poker_collusion/seq_evidence_submission.csv")   # 0.83185 base (risk+behavior source)
OUT_SUB=Path("outputs/poker_collusion/evidence_tuned_submission.csv")  # NEW file — banked 0.83460 untouched
SEED=42; SEEDS=[SEED,SEED+7,SEED+13,SEED+23]; ROUNDS=400
# tuned per-family params (rest inherit HAND_PARAMS)
FAM_PARAMS={
    "directed_transfer":   {**HAND_PARAMS,"feature_fraction":0.25},
    "soft_play":           {**HAND_PARAMS,"feature_fraction":0.25},
    "coordinated_isolation":{**HAND_PARAMS,"feature_fraction":0.8,"min_data_in_leaf":20},
}
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dh=build_dh()
    demb=pl.read_parquet(SEQ/"seq_ctx_emb_dev.parquet")
    dh=dh.join(demb,on=["pair_id","hand_idx"],how="left").to_pandas()
    dh[CTXCOLS]=dh[CTXCOLS].fillna(0.0).astype("float32"); dh[HAND_FEATS]=dh[HAND_FEATS].astype("float32")
    FEATS=HAND_FEATS+CTXCOLS
    pos=(dh["label"]==1).to_numpy()

    # ---- 1. per-eval-pair family argmax (IDENTICAL to evidence_familycond_submit) ----
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet"); evf=pd.read_parquet(STEP3/"eval_pair_features.parquet")
    dev["pair_id"]=dev["pair_id"].astype(str); evf["pair_id"]=evf["pair_id"].astype(str)
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    PF=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser" and c in evf.columns]
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    PP=dict(objective="binary",learning_rate=0.03,num_leaves=7,min_data_in_leaf=100,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    Xd=dev[PF].astype("float32"); Xe=evf[PF].astype("float32")
    fam_prior={fam:int((dev["behavior_family"]==fam).sum()) for fam in TARGET_BEHAVIORS}
    fam_ev=np.zeros((len(evf),len(TARGET_BEHAVIORS))); w=np.where(y==1,5.0,1.0)
    for j,fam in enumerate(TARGET_BEHAVIORS):
        yf=(dev["behavior_family"]==fam).to_numpy().astype(int)
        m=lgb.train(PP,lgb.Dataset(Xd[labm],yf[labm],weight=w[labm]),num_boost_round=400)
        fam_ev[:,j]=m.predict(Xe)/fam_prior[fam]
    eval_fam=np.array(TARGET_BEHAVIORS)[fam_ev.argmax(1)]
    evfam=dict(zip(evf["pair_id"],eval_fam))
    log(f"eval family argmax: {pd.Series(eval_fam).value_counts().to_dict()}")

    # ---- 2. train 3 per-family evidence rankers with TUNED per-family params, 4 seeds ----
    models={}
    for fam in TARGET_BEHAVIORS:
        m=pos & (dh["behavior_family"]==fam).to_numpy()
        pr=FAM_PARAMS[fam]
        models[fam]=[lgb.train({**pr,"seed":sd},lgb.Dataset(dh.loc[m,FEATS],dh.loc[m,"is_evidence"].astype(int)),num_boost_round=ROUNDS) for sd in SEEDS]
        log(f"trained {fam} ranker on {int(m.sum()):,} hands  (ff={pr['feature_fraction']} md={pr['min_data_in_leaf']}, {len(SEEDS)} seeds)")

    # ---- 3. score eval hands routed by family (streamed, IDENTICAL) ----
    eh=pl.read_parquet(STEP3/"eval_hand_features.parquet"); eemb=pl.read_parquet(SEQ/"seq_ctx_emb_eval.parquet")
    n=eh.height; sel=[c for c in ["pair_id","hand_id","pot_bb","hand_idx"] if c not in HAND_FEATS]
    log(f"streaming-score {n:,} eval hands, routed by family")
    parts=[]; CH=1_500_000; ehf=pl.scan_parquet(STEP3/"eval_hand_features.parquet")
    for s in range(0,n,CH):
        chunk=(ehf.slice(s,CH).select(sel+HAND_FEATS).collect().join(eemb,on=["pair_id","hand_idx"],how="left"))
        cp=chunk.to_pandas()
        for c in CTXCOLS:
            if c not in cp.columns: cp[c]=np.float32(0.0)
        cp[CTXCOLS]=cp[CTXCOLS].fillna(0.0).astype("float32"); cp[HAND_FEATS]=cp[HAND_FEATS].astype("float32")
        fam_of=cp["pair_id"].map(evfam).fillna("directed_transfer").to_numpy()
        hs=np.zeros(len(cp),np.float64)
        for fam in TARGET_BEHAVIORS:
            mrow=fam_of==fam
            if mrow.sum()==0: continue
            hs[mrow]=np.mean([mm.predict(cp.loc[mrow,FEATS]) for mm in models[fam]],axis=0)
        part=cp[["pair_id","hand_id","pot_bb"]].copy(); part["hand_score"]=hs; parts.append(part)
        del chunk,cp; log(f"  scored {min(s+CH,n):,}/{n:,}")
    del eemb
    ehp=pd.concat(parts,ignore_index=True); del parts
    top=(ehp.sort_values(["pair_id","hand_score","pot_bb","hand_id"],ascending=[True,False,False,True],kind="mergesort")
         .groupby("pair_id")["hand_id"].apply(lambda s:list(s.head(5))))
    ev=pd.DataFrame({"pair_id":top.index})
    for i in range(5): ev[f"evidence_hand_{i+1}"]=[(v[i] if i<len(v) else "NO_EVIDENCE") for v in top.values]

    # ---- 4. swap evidence into the 0.83185 base; risk+behavior MUST stay identical ----
    base=pd.read_csv(BASE_SUB); base["pair_id"]=base["pair_id"].astype(str); ev["pair_id"]=ev["pair_id"].astype(str)
    EV=[f"evidence_hand_{i}" for i in range(1,6)]
    out=base.drop(columns=EV).merge(ev,on="pair_id",how="left")
    miss=out["evidence_hand_1"].isna()
    if miss.any():
        log(f"WARN {miss.sum()} pairs missing new evidence -> fallback to base"); fb=base.set_index("pair_id")[EV]
        for c in EV: out.loc[miss,c]=out.loc[miss,"pair_id"].map(fb[c])
    out=out[list(base.columns)]
    assert (out["pair_id"].values==base["pair_id"].values).all()
    assert (out["risk_score"].values==base["risk_score"].values).all(), "risk changed!"
    assert (out["predicted_behavior"].values==base["predicted_behavior"].values).all(), "behavior changed!"
    changed=(out[EV].values!=base[EV].values).any(axis=1).sum()
    out.to_csv(OUT_SUB,index=False)
    log(f"wrote {OUT_SUB.name}: {len(out):,} rows; risk+behavior IDENTICAL; evidence changed on {changed:,} pairs")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
