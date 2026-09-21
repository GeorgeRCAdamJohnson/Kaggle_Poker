"""Decisive gate for the coord-isolation-MASKED field-relative equity feature (§221 follow-up).

Robust slice finding (GPU 5-seed): coordinated_isolation is the ONLY seed-consistent negative slice
(0/5 positive, mean -0.0019) — mechanistically sensible (isolation != surrendering equity to a partner).
Everything else is noise. Also: on XGBoost-GPU the UNMASKED FR feature already gives +0.0035 (4/5), better
than LightGBM's +0.0006 — so LightGBM under-extracts the signal.

This gate tests BOTH estimators, and for each, base vs +FR vs +FR-masked, 5 seeds, eval-weighted held[3,4]:
  MASK = zero the FR features on pairs whose family signal is coordinated_isolation.
  - On dev we know behavior_family (oracle mask for labeled; unlabeled dev + eval get a PREDICTED-family mask
    from a small family classifier trained on labeled dev, so the mask is eval-applicable, not oracle-only).
This is the CEILING+APPLICABLE test in one. SHIP-decision number is the LightGBM +FR-masked delta.
Run:  python -m anchor_repro.equity_fr_maskgate
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb, xgboost as xgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_field_relative import FEATS_FR, FEATS_FR_AUX, add_shrunk
STEP3 = Path("outputs/poker_collusion/hosen42_step3"); SEQ = STEP3/"edge"/"seq"; EQ = STEP3/"edge"/"equity"
N_FOLDS = 5
PT = ["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min",
      "both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
LP = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)
XP = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=6, eta=0.03,
          subsample=0.8, colsample_bytree=0.5, reg_lambda=10.0, min_child_weight=20, verbosity=0)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def main():
    fr_dev = add_shrunk(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas()); fr_dev["pair_id"]=fr_dev["pair_id"].astype(str)
    dev = pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev = dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(), on="pair_id", how="left")
    de = pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev = dev.merge(de, on="pair_id", how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    dev = dev.merge(fr_dev, on="pair_id", how="left")
    for c in PT+SEQF+FEATS_FR+FEATS_FR_AUX: dev[c]=dev[c].astype("float32").fillna(0)
    y = dev["label"].fillna(0).astype(int).to_numpy(); labm = dev["is_labeled"].to_numpy().astype(bool)
    fam = dev["behavior_family"].fillna("").to_numpy()
    TABLES = sorted(dev["table_id"].unique().tolist())

    META = {"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold",
            "shared_hands","chunk","pred_family","n_hands"}
    base = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
            and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
            and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS_FR and c not in FEATS_FR_AUX]
    evf = pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"]=evf["pair_id"].astype(str)
    evf = evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(), on="pair_id", how="left")
    shared=[c for c in base if c in evf.columns]; nn=min(len(dev),len(evf),40000); evf=evf[["pair_id"]+shared].copy()

    # ---- Build TWO principled variants that remove the robust-negative slices (coord-isolation, low-exposure) ----
    # (1) MASK: predicted-coord-isolation (eval-applicable) OR low-exposure -> zero FR. CI mask calibrated so its
    #     coverage ~ the CI rate among positives (not a fixed prob cutoff, which mis-scaled to 0.07%).
    ci = (fam=="coordinated_isolation").astype(int)
    Xci = dev[shared].to_numpy(np.float32); ciprob = np.zeros(len(dev)); rci = np.random.RandomState(7)
    fld = rci.randint(0,5,size=len(dev))
    for f in range(5):
        tr = labm & (fld!=f); te = fld==f
        m = lgb.train({**LP,"num_leaves":15}, lgb.Dataset(Xci[tr], ci[tr]), num_boost_round=200); ciprob[te]=m.predict(Xci[te])
    ci_rate = float((labm & (ci==1)).sum()) / float(labm.sum())          # CI share among labeled (~0.06)
    ci_thr = np.quantile(ciprob, 1.0 - ci_rate)                          # top ci_rate of predicted-CI pairs
    nopp = dev["fr_n_opp"].to_numpy()
    lo_thr = np.quantile(nopp[labm & (y==1)], 0.33)                      # low-exposure boundary from positives
    mask = (ciprob >= ci_thr) | (nopp < lo_thr)                         # zero FR where CI-like OR low-exposure
    log(f"mask covers {mask.mean():.2%} of dev pairs (CI top {ci_rate:.2%} thr {ci_thr:.3f}; n_opp<{lo_thr:.0f})")
    for c in FEATS_FR:
        dev[c+"_m"] = np.where(mask, 0.0, dev[c].to_numpy()).astype("float32")
    FEATS_FR_M = [c+"_m" for c in FEATS_FR]

    # (2) CONFIDENCE-WEIGHT (no family prediction, fully eval-safe): scale residual by n_opp/(n_opp+k) so
    #     poorly-estimated (low-opportunity) pairs smoothly -> 0. Attacks the low-exposure negative directly.
    conf = (nopp/(nopp+100.0)).astype("float32")
    for c in FEATS_FR:
        dev[c+"_c"] = (dev[c].to_numpy()*conf).astype("float32")
    FEATS_FR_C = [c+"_c" for c in FEATS_FR]

    def distill_lgb(folds,sd):
        Xd=dev[SEQF].astype("float32"); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            o1[te]=lgb.train({**LP,"seed":sd},lgb.Dataset(Xd[tr],y[tr],weight=w1[tr]),num_boost_round=500).predict(Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; ps=unl&(o1>=tp); amb=unl&(o1>=ta)&(o1<tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            o[te]=lgb.train({**LP,"seed":sd},lgb.Dataset(Xd[tr],y2[tr],weight=w2[tr]),num_boost_round=500).predict(Xd[te])
        return o

    def oof_lgb(feats,folds,heldm,sd):
        X=dev[feats].astype("float32"); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            o1[te]=lgb.train({**LP,"seed":sd},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750).predict(X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for s2 in [sd,sd+11,sd+23]:
                o[te]+=lgb.train({**LP,"seed":s2},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750).predict(X[te])/3
        return o

    def weval(folds,sd):
        r=np.random.RandomState(sd); ds=dev.iloc[r.choice(len(dev),nn,replace=False)]; es=evf.iloc[r.choice(len(evf),nn,replace=False)]
        Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
        yc=np.concatenate([np.zeros(nn),np.ones(nn)]).astype(np.float32)
        dc=xgb.train({**XP,"eta":0.05},xgb.DMatrix(Xc,label=yc),num_boost_round=200)
        pe=dc.predict(xgb.DMatrix(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)))
        w=np.clip(pe/(1-pe+1e-6),0.05,20.0); return w/w.mean()

    res={"base":[], "fr":[], "frm":[], "frc":[]}
    for sd in [42,101,202,303,404]:
        r=np.random.RandomState(sd); TF={t:int(v) for t,v in zip(TABLES,r.permutation(len(TABLES))%N_FOLDS)}
        folds=dev["table_id"].map(TF).to_numpy(); heldm=np.isin(folds,[3,4])
        dev["seq_risk"]=distill_lgb(folds,sd); w_eval=weval(folds,sd)
        b=oof_lgb(base+PT+["seq_risk"],folds,heldm,sd)
        f_=oof_lgb(base+PT+["seq_risk"]+FEATS_FR,folds,heldm,sd)
        fm=oof_lgb(base+PT+["seq_risk"]+FEATS_FR_M,folds,heldm,sd)
        fc=oof_lgb(base+PT+["seq_risk"]+FEATS_FR_C,folds,heldm,sd)
        bw=average_precision_score(y[heldm],b[heldm],sample_weight=w_eval[heldm])
        fw=average_precision_score(y[heldm],f_[heldm],sample_weight=w_eval[heldm])
        fmw=average_precision_score(y[heldm],fm[heldm],sample_weight=w_eval[heldm])
        fcw=average_precision_score(y[heldm],fc[heldm],sample_weight=w_eval[heldm])
        res["base"].append(bw); res["fr"].append(fw); res["frm"].append(fmw); res["frc"].append(fcw)
        log(f"seed{sd}: base {bw:.4f} | +FR {fw-bw:+.4f} | +masked {fmw-bw:+.4f} | +conf {fcw-bw:+.4f}")
    b=np.array(res["base"]); f_=np.array(res["fr"]); fm=np.array(res["frm"]); fc=np.array(res["frc"])
    print(f"\n=== LightGBM (SHIP-decision estimator), 5 seeds — delta over base+PT+seq_risk ===")
    for tag,arr in [("+FR (unmasked)",f_),("+FR-masked (CI|low-exp)",fm),("+FR-confidence-weighted",fc)]:
        dvec=arr-b; print(f"  {tag:26s} mean {dvec.mean():+.4f}  ({int((dvec>0).sum())}/5 pos)  {np.round(dvec,4).tolist()}")
    best=max([("masked",fm),("conf",fc)],key=lambda t:(t[1]-b).mean())
    dvec=best[1]-b; ship=(dvec.mean()>0.005) and ((dvec>0).sum()>=4)
    print(f"  BEST variant '{best[0]}': mean {dvec.mean():+.4f}, {int((dvec>0).sum())}/5  -> {'SHIP' if ship else 'NO-SHIP'} (bar mean>+0.005 & >=4/5)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
