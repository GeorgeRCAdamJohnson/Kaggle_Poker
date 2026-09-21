"""STAGED gate for FIELD-RELATIVE equity-surrender residuals (§220).

Rebuilds nothing inherited: reuses base+PT+seq_risk purely as the reference to beat, but the CANDIDATE is
the field-relative residual basis from equity_field_relative.py. Runs three gates in order and short-circuits:

  GATE 1 (drift, decisive-early): adversarial dev-vs-eval AUC on the residual feats alone.
          Original absolute equity feats drifted at 0.716 -> KILLED. Residual must be < 0.65 (target < 0.55).
          If it still drifts, the idea is genuinely dead (not infrastructure) -> STOP, record.
  GATE 2 (signal retained): standalone label-AUC (>0.70 on best feat) AND incremental over made-value (>+0.004).
  GATE 3 (decisive): eval-weighted held-fold[3,4] delta over base+PT+seq_risk, 5 seeds.
          SHIP if mean > +0.005 AND >=4/5 positive AND drift < 0.65 (from GATE 1).

Run:  python -m anchor_repro.equity_field_relative_gate
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_field_relative import FEATS_FR, FEATS_FR_AUX, add_shrunk

STEP3 = Path("outputs/poker_collusion/hosen42_step3"); SEQ = STEP3/"edge"/"seq"; EQ = STEP3/"edge"/"equity"
SEED, N_FOLDS = 42, 5
PT = ["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min",
      "both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4)
NThr = 4  # cap LightGBM threads — 15GB RAM machine crashed with num_threads=-1 oversubscription
CKPT = None  # set in main(); per-seed checkpoint dir so a crash doesn't lose completed seeds
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

MADE = ["transfer_dominant","transfer_rate","hs_mean","hs_max","hs_top3"]  # made-value proxies (from _diag2)


def main():
    fr_dev = add_shrunk(pl.read_parquet(EQ/"field_relative_development.parquet").to_pandas())
    fr_dev["pair_id"] = fr_dev["pair_id"].astype(str)
    fr_eval = add_shrunk(pl.read_parquet(EQ/"field_relative_evaluation.parquet").to_pandas())
    fr_eval["pair_id"] = fr_eval["pair_id"].astype(str)

    dev = pd.read_parquet(STEP3/"dev_pair_features.parquet"); dev["pair_id"] = dev["pair_id"].astype(str)
    dev = dev.merge(pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas(), on="pair_id", how="left")
    de = pd.read_parquet(SEQ/"seq_emb_v2_dev.parquet"); de["pair_id"] = de["pair_id"].astype(str)
    dev = dev.merge(de, on="pair_id", how="left"); SEQF = [c for c in de.columns if c.startswith("seq_")]
    dev = dev.merge(fr_dev, on="pair_id", how="left")
    for c in PT+SEQF+FEATS_FR+FEATS_FR_AUX: dev[c] = dev[c].astype("float32").fillna(0)
    y = dev["label"].fillna(0).astype(int).to_numpy(); labm = dev["is_labeled"].to_numpy().astype(bool)
    rng = np.random.RandomState(SEED)

    # ---------- GATE 1: DRIFT on residual feats alone ----------
    print("\n=== GATE 1: adversarial dev-vs-eval AUC on FIELD-RELATIVE residual feats (want <0.65, target <0.55) ===")
    n = min(len(fr_dev), len(fr_eval), 40000)
    ds = fr_dev.iloc[rng.choice(len(fr_dev), n, replace=False)][FEATS_FR].replace([np.inf,-np.inf],np.nan).fillna(0)
    es = fr_eval.iloc[rng.choice(len(fr_eval), n, replace=False)][FEATS_FR].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xc = pd.concat([ds, es], ignore_index=True); yc = np.concatenate([np.zeros(n), np.ones(n)])
    fo = rng.randint(0,5,size=len(yc)); od = np.zeros(len(yc))
    for f in range(5):
        tr = fo!=f; te = fo==f
        m = lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=NThr),
                      lgb.Dataset(Xc.iloc[tr], yc[tr]), num_boost_round=200); od[te] = m.predict(Xc.iloc[te])
    drift = roc_auc_score(yc, od)
    log(f"  residual-feats adversarial dev-vs-eval AUC = {drift:.4f}  (original absolute-equity feats = 0.7159)")
    # per-feat drift breakdown
    for c in FEATS_FR:
        try:
            a = roc_auc_score(yc, pd.concat([ds[[c]],es[[c]]],ignore_index=True)[c].to_numpy())
            print(f"      {c:22s} solo-drift AUC {max(a,1-a):.3f}")
        except Exception: pass
    if drift >= 0.65:
        print(f"\n  GATE 1 FAILED: residual still drifts ({drift:.4f} >= 0.65). Field-relative form did NOT")
        print("  remove the phase drift -> idea is genuinely dead, not infrastructure. STOP.")
        return 0
    print(f"  GATE 1 PASSED: {drift:.4f} < 0.65. Drift removed by field-relative construction. Proceed.")

    # ---------- GATE 2: signal retained ----------
    print("\n=== GATE 2: standalone label-AUC + incremental over made-value ===")
    lm = labm & np.isin(dev["behavior_family"].fillna("").to_numpy(), ["directed_transfer","soft_play","coordinated_isolation"]) | (labm & (y==0))
    ly = y[labm]
    best_auc = 0.0
    for c in FEATS_FR:
        v = dev.loc[labm, c].to_numpy().astype(float)
        try:
            a = roc_auc_score(ly, v); best_auc = max(best_auc, max(a,1-a))
            print(f"      {c:22s} label-AUC {max(a,1-a):.3f} ({'hi' if a>0.5 else 'lo'}=sus)")
        except Exception: pass
    # incremental over made-value (labelled OOF AUC)
    made = [c for c in MADE if c in dev.columns]
    def oof_auc(feats):
        X = dev.loc[labm, feats].astype("float32").fillna(0).to_numpy(); o = np.zeros(labm.sum())
        ff = rng.randint(0,5,size=labm.sum())
        for f in range(5):
            tr = ff!=f; te = ff==f
            m = lgb.train({**P,"seed":SEED}, lgb.Dataset(X[tr], ly[tr]), num_boost_round=300); o[te] = m.predict(X[te])
        return roc_auc_score(ly, o)
    mv = oof_auc(made) if made else float("nan")
    mve = oof_auc(made+FEATS_FR) if made else float("nan")
    log(f"  made-value-only OOF AUC {mv:.4f} -> +field-relative {mve:.4f} (delta {mve-mv:+.4f})")
    print(f"  best standalone label-AUC {best_auc:.3f} (want >0.70); incremental delta {mve-mv:+.4f} (want >+0.004)")

    # ---------- GATE 3: eval-weighted held-fold decisive gate, 5 seeds ----------
    print("\n=== GATE 3: eval-weighted held-fold[3,4] delta over base+PT+seq_risk (5 seeds) ===")
    for c in PT+SEQF: dev[c] = dev[c].astype("float32").fillna(0)
    TABLES = sorted(dev["table_id"].unique().tolist())
    def distill(folds):
        Xd = dev[SEQF].astype("float32"); w1 = np.where(labm, np.where(y==1,5.0,1.0), 0.1); o1 = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = (folds!=f)&(w1>0); te = folds==f
            m = lgb.train({**P,"seed":SEED}, lgb.Dataset(Xd[tr], y[tr], weight=w1[tr]), num_boost_round=500); o1[te] = m.predict(Xd[te])
        pos = o1[labm&(y==1)]; ta,tp = np.quantile(pos,0.05), np.quantile(pos,0.50)
        unl = ~labm; amb = unl&(o1>=ta)&(o1<tp); ps = unl&(o1>=tp); y2 = np.where(ps,1,y)
        w2 = np.where(labm, np.where(y==1,5.0,1.0), np.where(ps,1.0,np.where(amb,0.0,0.1))); o = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = (folds!=f)&(w2>0); te = folds==f
            m = lgb.train({**P,"seed":SEED}, lgb.Dataset(Xd[tr], y2[tr], weight=w2[tr]), num_boost_round=500); o[te] = m.predict(Xd[te])
        return o

    META = {"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold",
            "shared_hands","chunk","pred_family","n_hands"}
    base = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
            and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
            and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS_FR and c not in FEATS_FR_AUX]
    evf = pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"] = evf["pair_id"].astype(str)
    evf = evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(), on="pair_id", how="left")
    shared = [c for c in base if c in evf.columns]; nn = min(len(dev),len(evf),40000)
    evf = evf[["pair_id"] + shared].copy()  # trim eval frame to only needed cols (15GB RAM headroom)
    import gc as _gc; _gc.collect()

    def run_seed(sd):
        r = np.random.RandomState(sd)
        TF = {t:int(v) for t,v in zip(TABLES, r.permutation(len(TABLES))%N_FOLDS)}; folds = dev["table_id"].map(TF).to_numpy()
        dev["seq_risk"] = distill(folds)
        ds2 = dev.iloc[r.choice(len(dev),nn,replace=False)]; es2 = evf.iloc[r.choice(len(evf),nn,replace=False)]
        Xc2 = pd.concat([ds2[shared], es2[shared]], ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
        yc2 = np.concatenate([np.zeros(nn), np.ones(nn)])
        dc = lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=NThr),
                       lgb.Dataset(Xc2, yc2), num_boost_round=200)
        pe = dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0))
        w_eval = np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval = w_eval/w_eval.mean()
        heldm = np.isin(folds,[3,4])
        def oof(feats):
            X = dev[feats].astype("float32"); pool = ~heldm; w1 = np.where(labm,np.where(y==1,5.0,1.0),0.1); o1 = np.zeros(len(dev))
            for f in range(N_FOLDS):
                tr = pool&(folds!=f)&(w1>0); te = folds==f
                m = lgb.train({**P,"seed":sd}, lgb.Dataset(X[tr],y[tr],weight=w1[tr]), num_boost_round=750); o1[te] = m.predict(X[te])
            pos = o1[labm&(y==1)&pool]; ta,tp = np.quantile(pos,0.05),np.quantile(pos,0.50)
            unl = ~labm; amb = unl&(o1>=ta)&(o1<tp)&pool; ps = unl&(o1>=tp)&pool
            y2 = np.where(ps,1,y); w2 = np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2 = np.where(pool,w2,0.0)
            o = np.zeros(len(dev))
            for f in range(N_FOLDS):
                tr = pool&(folds!=f)&(w2>0); te = folds==f
                for s2 in [sd,sd+11,sd+23]:
                    m = lgb.train({**P,"seed":s2}, lgb.Dataset(X[tr],y2[tr],weight=w2[tr]), num_boost_round=750); o[te] += m.predict(X[te])/3
            return o
        bw = average_precision_score(y[heldm], oof(base+PT+["seq_risk"])[heldm], sample_weight=w_eval[heldm])
        ew = average_precision_score(y[heldm], oof(base+PT+["seq_risk"]+FEATS_FR)[heldm], sample_weight=w_eval[heldm])
        return bw, ew

    import json, gc
    ck = EQ/"_fr_gate3_seeds.json"
    done = json.loads(ck.read_text()) if ck.exists() else {}  # {seed: [bw, ew]} survives a crash -> resume
    for sd in [42,101,202,303,404]:
        if str(sd) in done:
            bw, ew = done[str(sd)]; log(f"  seed{sd}: RESUMED base EVAL-W {bw:.4f} | +FR {ew:.4f} | delta {ew-bw:+.4f}")
            continue
        bw, ew = run_seed(sd)
        done[str(sd)] = [float(bw), float(ew)]; ck.write_text(json.dumps(done))  # checkpoint immediately
        log(f"  seed{sd}: base EVAL-W {bw:.4f} | +FR {ew:.4f} | delta {ew-bw:+.4f}")
        gc.collect()  # free per-seed frames on a 15GB machine before the next seed
    deltas = np.array([done[str(sd)][1]-done[str(sd)][0] for sd in [42,101,202,303,404]])
    print(f"\n  FR eval-weighted delta over 5 seeds: mean {deltas.mean():+.4f} min {deltas.min():+.4f} max {deltas.max():+.4f} (>0 in {int((deltas>0).sum())}/5)")
    print(f"  SHIP if mean > +0.005 AND >=4/5 positive AND drift {drift:.4f} < 0.65.")
    ship = (deltas.mean() > 0.005) and ((deltas>0).sum() >= 4) and (drift < 0.65)
    print(f"  VERDICT: {'SHIP' if ship else 'NO-SHIP'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
