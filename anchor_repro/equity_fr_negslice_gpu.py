"""GPU (XGBoost cuda) version of the §221 negative-slice investigation. LightGBM has no GPU backend in the
pip wheel, so port the gradient-boosting to XGBoost device=cuda (verified working, ~2s/300 rounds on the 5070).

RELATIVE diagnostic only (where does the field-relative equity feature HELP vs HURT, by behavior family and
exposure bucket), so the estimator swap is fine — we are not producing the ship-decision number here (that
stays LightGBM for byte-comparability to 0.83460). Mirrors the CPU logic: table-fold map per seed, two-stage
PU distill for seq_risk, eval-weighted held-fold[3,4] AP, then per-slice score-shift of held positives,
aggregated across 5 seeds.
Run:  python -m anchor_repro.equity_fr_negslice_gpu
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
# XGBoost GPU params ~matched to the LightGBM gate (depth-ish via max_depth, same lr/regularization spirit)
XP = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=6, eta=0.03,
          subsample=0.8, colsample_bytree=0.5, reg_lambda=10.0, min_child_weight=20, verbosity=0)
NROUND = 400
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def _train(X, y, w, seed):
    d = xgb.DMatrix(X, label=y, weight=w)
    return xgb.train({**XP, "seed": int(seed)}, d, num_boost_round=NROUND)

def _pred(m, X):
    return m.predict(xgb.DMatrix(X))


def main():
    fr_dev = add_shrunk(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas())
    fr_dev["pair_id"] = fr_dev["pair_id"].astype(str)
    dev = pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"] = dev["pair_id"].astype(str)
    dev = dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(), on="pair_id", how="left")
    de = pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"] = de["pair_id"].astype(str)
    dev = dev.merge(de, on="pair_id", how="left"); SEQF = [c for c in de.columns if c.startswith("seq_")]
    dev = dev.merge(fr_dev, on="pair_id", how="left")
    for c in PT+SEQF+FEATS_FR+FEATS_FR_AUX: dev[c] = dev[c].astype("float32").fillna(0)
    y = dev["label"].fillna(0).astype(int).to_numpy(); labm = dev["is_labeled"].to_numpy().astype(bool)
    fam = dev["behavior_family"].fillna("neg").to_numpy(); nopp = dev["fr_n_opp"].to_numpy()
    TABLES = sorted(dev["table_id"].unique().tolist())

    META = {"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold",
            "shared_hands","chunk","pred_family","n_hands"}
    base = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
            and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
            and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS_FR and c not in FEATS_FR_AUX]
    evf = pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"] = evf["pair_id"].astype(str)
    evf = evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(), on="pair_id", how="left")
    shared = [c for c in base if c in evf.columns]; nn = min(len(dev),len(evf),40000)
    evf = evf[["pair_id"]+shared].copy()

    def distill(folds, sd):
        Xd = dev[SEQF].to_numpy(np.float32); w1 = np.where(labm, np.where(y==1,5.0,1.0), 0.1).astype(np.float32); o1 = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            o1[te]=_pred(_train(Xd[tr],y[tr],w1[tr],sd),Xd[te])
        pos=o1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp); ps=unl&(o1>=tp); y2=np.where(ps,1,y)
        w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))).astype(np.float32); o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            o[te]=_pred(_train(Xd[tr],y2[tr],w2[tr],sd),Xd[te])
        return o

    def oof(feats, folds, heldm, sd):
        X=dev[feats].to_numpy(np.float32); pool=~heldm; w1=np.where(labm,np.where(y==1,5.0,1.0),0.1).astype(np.float32); o1=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w1>0); te=folds==f
            o1[te]=_pred(_train(X[tr],y[tr],w1[tr],sd),X[te])
        pos=o1[labm&(y==1)&pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(o1>=ta)&(o1<tp)&pool; ps=unl&(o1>=tp)&pool
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2=np.where(pool,w2,0.0).astype(np.float32)
        o=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=pool&(folds!=f)&(w2>0); te=folds==f
            for s2 in [sd,sd+11,sd+23]:
                o[te]+=_pred(_train(X[tr],y2[tr],w2[tr],s2),X[te])/3
        return o

    fam_shift={f:[] for f in ["directed_transfer","soft_play","coordinated_isolation"]}
    exp_shift={b:[] for b in ["lo","mid","hi"]}
    gq=np.quantile(nopp[labm&(y==1)],[0,0.33,0.66,1.0])
    deltas=[]
    for sd in [42,101,202,303,404]:
        r=np.random.RandomState(sd)
        TF={t:int(v) for t,v in zip(TABLES,r.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
        dev["seq_risk"]=distill(folds,sd)
        ds2=dev.iloc[r.choice(len(dev),nn,replace=False)]; es2=evf.iloc[r.choice(len(evf),nn,replace=False)]
        Xc=pd.concat([ds2[shared],es2[shared]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
        yc=np.concatenate([np.zeros(nn),np.ones(nn)]).astype(np.float32)
        dc=xgb.train({**XP,"eta":0.05},xgb.DMatrix(Xc,label=yc),num_boost_round=200)
        pe=dc.predict(xgb.DMatrix(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)))
        w_eval=np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
        heldm=np.isin(folds,[3,4])
        ob=oof(base+PT+["seq_risk"],folds,heldm,sd); of=oof(base+PT+["seq_risk"]+FEATS_FR,folds,heldm,sd)
        bw=average_precision_score(y[heldm],ob[heldm],sample_weight=w_eval[heldm]); ew=average_precision_score(y[heldm],of[heldm],sample_weight=w_eval[heldm])
        deltas.append(ew-bw); log(f"seed{sd}: base {bw:.4f} +FR {ew:.4f} delta {ew-bw:+.4f}")
        hp=heldm&labm&(y==1)
        for f in ["directed_transfer","soft_play","coordinated_isolation"]:
            msk=hp&(fam==f)
            if msk.sum()==0: continue
            ds=(of[msk]-ob[msk]); fam_shift[f].append(ds.mean())
            print(f"     {f:24s} n={msk.sum():3d} shift {ds.mean():+.5f}")
        for i,bn in enumerate(["lo","mid","hi"]):
            msk=hp&(nopp>=gq[i])&(nopp<=gq[i+1] if i==2 else nopp<gq[i+1])
            if msk.sum()==0: continue
            ds=(of[msk]-ob[msk]); exp_shift[bn].append(ds.mean())
            print(f"     n_opp {bn:3s} n={msk.sum():3d} shift {ds.mean():+.5f}")

    print("\n=== AGGREGATE across 5 seeds (GPU-XGB; RELATIVE slice effect) ===")
    dd=np.array(deltas); print(f"  overall delta: mean {dd.mean():+.5f} per-seed {np.round(dd,4).tolist()} ({int((dd>0).sum())}/5 pos)")
    for f in ["directed_transfer","soft_play","coordinated_isolation"]:
        v=np.array(fam_shift[f]); print(f"  {f:24s} mean {v.mean():+.5f} per-seed {np.round(v,5).tolist()} ({int((v>0).sum())}/{len(v)} pos)")
    for bn in ["lo","mid","hi"]:
        v=np.array(exp_shift[bn]); print(f"  n_opp {bn:3s}                mean {v.mean():+.5f} per-seed {np.round(v,5).tolist()} ({int((v>0).sum())}/{len(v)} pos)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
