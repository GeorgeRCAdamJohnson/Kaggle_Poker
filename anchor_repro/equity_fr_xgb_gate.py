"""OPTION 1 (§222): XGBoost-GPU-NATIVE risk gate for the confidence-weighted field-relative equity feature.

§222 found: XGBoost extracts ~4x more from this feature than LightGBM (+0.0035 vs +0.0022). This gate tests
the HONEST question: with a FULLY re-baselined XGBoost base+PT+seq_risk (seq_risk ALSO distilled with XGBoost,
so base and candidate use the SAME estimator), does the confidence-weighted FR clear +0.005 eval-weighted at
>=4/5 seeds? Everything runs on cuda:0. If YES -> justifies a full XGBoost risk-head swap (own leak battery,
then submit). If NO -> the signal is genuinely sub-threshold on both estimators; go to Option 2 (stacking).

Confidence-weight (eval-safe, no family prediction): FR_c = FR * n_opp/(n_opp+100).
Run:  python -m anchor_repro.equity_fr_xgb_gate
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, xgboost as xgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_field_relative import FEATS_FR, FEATS_FR_AUX, add_shrunk
STEP3 = Path("outputs/poker_collusion/hosen42_step3"); SEQ = STEP3/"edge"/"seq"; EQ = STEP3/"edge"/"equity"
N_FOLDS = 5
PT = ["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min",
      "both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
XP = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=6, eta=0.03,
          subsample=0.8, colsample_bytree=0.5, reg_lambda=10.0, min_child_weight=20, verbosity=0)
NROUND = 500
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
def _tr(X,y,w,sd): return xgb.train({**XP,"seed":int(sd)}, xgb.DMatrix(X,label=y,weight=w), num_boost_round=NROUND)
def _pr(m,X): return m.predict(xgb.DMatrix(X))


def main():
    fr = add_shrunk(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas()); fr["pair_id"]=fr["pair_id"].astype(str)
    dev = pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"]=dev["pair_id"].astype(str)
    dev = dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(), on="pair_id", how="left")
    de = pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"]=de["pair_id"].astype(str)
    dev = dev.merge(de, on="pair_id", how="left"); SEQF=[c for c in de.columns if c.startswith("seq_")]
    dev = dev.merge(fr, on="pair_id", how="left")
    for c in PT+SEQF+FEATS_FR+FEATS_FR_AUX: dev[c]=dev[c].astype("float32").fillna(0)
    y = dev["label"].fillna(0).astype(int).to_numpy(); labm = dev["is_labeled"].to_numpy().astype(bool)
    nopp = dev["fr_n_opp"].to_numpy(); conf=(nopp/(nopp+100.0)).astype("float32")
    for c in FEATS_FR: dev[c+"_c"]=(dev[c].to_numpy()*conf).astype("float32")
    FEATS_FR_C=[c+"_c" for c in FEATS_FR]
    TABLES = sorted(dev["table_id"].unique().tolist())

    META = {"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold",
            "shared_hands","chunk","pred_family","n_hands"}
    base = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
            and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
            and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS_FR and c not in FEATS_FR_AUX and c not in FEATS_FR_C]
    evf = pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"]=evf["pair_id"].astype(str)
    evf = evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(), on="pair_id", how="left")
    shared=[c for c in base if c in evf.columns]; nn=min(len(dev),len(evf),40000); evf=evf[["pair_id"]+shared].copy()

    def distill(folds,sd):  # seq_risk distilled with XGBoost (consistent estimator)
        Xd=dev[SEQF].to_numpy(np.float32); w1=np.where(labm,np.where(y==1,5.0,1.0),0.1).astype(np.float32); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f; o1[te]=_pr(_tr(Xd[tr],y[tr],w1[tr],sd),Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))).astype(np.float32); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f; o[te]=_pr(_tr(Xd[tr],y2[tr],w2[tr],sd),Xd[te])
        return o

    def oof(feats,folds,heldm,sd):
        X=dev[feats].to_numpy(np.float32); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1).astype(np.float32); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f; o1[te]=_pr(_tr(X[tr],y[tr],w1[tr],sd),X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0).astype(np.float32)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for s2 in [sd,sd+11,sd+23]: o[te]+=_pr(_tr(X[tr],y2[tr],w2[tr],s2),X[te])/3
        return o

    def weval(sd):
        r=np.random.RandomState(sd); ds=dev.iloc[r.choice(len(dev),nn,replace=False)]; es=evf.iloc[r.choice(len(evf),nn,replace=False)]
        Xc=pd.concat([ds[shared],es[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
        yc=np.concatenate([np.zeros(nn),np.ones(nn)]).astype(np.float32)
        dc=xgb.train({**XP,"eta":0.05},xgb.DMatrix(Xc,label=yc),num_boost_round=200)
        pe=dc.predict(xgb.DMatrix(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)))
        w=np.clip(pe/(1-pe+1e-6),0.05,20.0); return w/w.mean()

    bases=[]; cands=[]
    for sd in [42,101,202,303,404]:
        r=np.random.RandomState(sd); TF={t:int(v) for t,v in zip(TABLES,r.permutation(len(TABLES))%N_FOLDS)}
        folds=dev["table_id"].map(TF).to_numpy(); heldm=np.isin(folds,[3,4])
        dev["seq_risk"]=distill(folds,sd); w_eval=weval(sd)
        b=oof(base+PT+["seq_risk"],folds,heldm,sd); c=oof(base+PT+["seq_risk"]+FEATS_FR_C,folds,heldm,sd)
        bw=average_precision_score(y[heldm],b[heldm],sample_weight=w_eval[heldm]); cw=average_precision_score(y[heldm],c[heldm],sample_weight=w_eval[heldm])
        bases.append(bw); cands.append(cw); log(f"seed{sd}: XGB base {bw:.4f} | +FR-conf {cw:.4f} | delta {cw-bw:+.4f}")
    b=np.array(bases); c=np.array(cands); d=c-b
    print(f"\n=== XGBoost-GPU-native risk gate (base ALSO XGBoost), 5 seeds ===")
    print(f"  +FR-confidence delta: mean {d.mean():+.4f}  ({int((d>0).sum())}/5 pos)  {np.round(d,4).tolist()}")
    ship=(d.mean()>0.005) and ((d>0).sum()>=4)
    print(f"  -> {'SHIP-CANDIDATE (justifies full XGB risk-head swap + leak battery)' if ship else 'NO-SHIP (signal sub-threshold on both estimators -> go Option 2)'}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
