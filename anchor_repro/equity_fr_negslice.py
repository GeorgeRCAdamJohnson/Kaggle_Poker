"""Investigate the two GATE-3 NEGATIVES (§221): are they fold-variance, or a specific slice the FR feature
HURTS? Fixes the two negative-seed fold maps (42, 202), rebuilds base vs +FR held-fold OOF, and decomposes
the eval-weighted held-fold delta by BEHAVIOR FAMILY and by EXPOSURE (n_opp) bucket among held positives.
Also reports per-seed (delta vs base-difficulty) correlation to test the fold-variance hypothesis directly.

If the negative concentrates in ONE family/slice -> a conditional FR feature could keep only the positive
slice (actionable). If it's uniform sign-flips across slices -> pure eval-transfer noise (dead, §221 stands).
Run:  python -m anchor_repro.equity_fr_negslice
"""
from __future__ import annotations
import time, warnings, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features
from anchor_repro.equity_field_relative import FEATS_FR, FEATS_FR_AUX, add_shrunk
STEP3 = Path("outputs/poker_collusion/hosen42_step3"); SEQ = STEP3/"edge"/"seq"; EQ = STEP3/"edge"/"equity"
N_FOLDS = 5; NThr = 8  # 8 physical cores; RAM is the constraint (not cores), so raise threads not parallel seeds
PT = ["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min",
      "both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=NThr)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


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
    fam = dev["behavior_family"].fillna("neg").to_numpy()
    TABLES = sorted(dev["table_id"].unique().tolist())

    META = {"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold",
            "shared_hands","chunk","pred_family","n_hands"}
    base = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
            and not c.startswith("evl_pf") and not c.startswith("deq_pf") and c not in PT and c not in SEQF
            and not c.startswith("seq_") and c!="seq_risk" and c not in FEATS_FR and c not in FEATS_FR_AUX]
    evf = pd.read_parquet(STEP3/"eval_pair_features.parquet"); evf["pair_id"] = evf["pair_id"].astype(str)
    evf = evf.merge(pvf_features(STEP3/"eval_pair_hands.parquet","evaluation").to_pandas(), on="pair_id", how="left")
    shared = [c for c in base if c in evf.columns]; nn = min(len(dev),len(evf),40000)

    def distill(folds, sd):
        Xd = dev[SEQF].astype("float32"); w1 = np.where(labm, np.where(y==1,5.0,1.0), 0.1); o1 = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = (folds!=f)&(w1>0); te = folds==f
            m = lgb.train({**P,"seed":sd}, lgb.Dataset(Xd[tr], y[tr], weight=w1[tr]), num_boost_round=500); o1[te]=m.predict(Xd[te])
        pos = o1[labm&(y==1)]; ta,tp = np.quantile(pos,0.05), np.quantile(pos,0.50)
        unl = ~labm; amb = unl&(o1>=ta)&(o1<tp); ps = unl&(o1>=tp); y2 = np.where(ps,1,y)
        w2 = np.where(labm, np.where(y==1,5.0,1.0), np.where(ps,1.0,np.where(amb,0.0,0.1))); o = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = (folds!=f)&(w2>0); te = folds==f
            m = lgb.train({**P,"seed":sd}, lgb.Dataset(Xd[tr], y2[tr], weight=w2[tr]), num_boost_round=500); o[te]=m.predict(Xd[te])
        return o

    def oof(feats, folds, heldm, sd):
        X = dev[feats].astype("float32"); pool = ~heldm; w1 = np.where(labm,np.where(y==1,5.0,1.0),0.1); o1 = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = pool&(folds!=f)&(w1>0); te = folds==f
            m = lgb.train({**P,"seed":sd}, lgb.Dataset(X[tr],y[tr],weight=w1[tr]), num_boost_round=750); o1[te]=m.predict(X[te])
        pos = o1[labm&(y==1)&pool]; ta,tp = np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl = ~labm; amb = unl&(o1>=ta)&(o1<tp)&pool; ps = unl&(o1>=tp)&pool
        y2 = np.where(ps,1,y); w2 = np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1))); w2 = np.where(pool,w2,0.0)
        o = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = pool&(folds!=f)&(w2>0); te = folds==f
            for s2 in [sd,sd+11,sd+23]:
                m = lgb.train({**P,"seed":s2}, lgb.Dataset(X[tr],y2[tr],weight=w2[tr]), num_boost_round=750); o[te]+=m.predict(X[te])/3
        return o

    # decompose ALL 5 seeds so we see whether the +seeds are ALSO dragged down by the same off-mechanism
    # slices (family / low-exposure). Aggregate the score-shifts across seeds for a stable picture.
    fam_shift = {f: [] for f in ["directed_transfer","soft_play","coordinated_isolation"]}
    exp_shift = {b: [] for b in ["lo","mid","hi"]}
    SEEDS = [42, 101, 202, 303, 404]
    for sd in SEEDS:
        r = np.random.RandomState(sd)
        TF = {t:int(v) for t,v in zip(TABLES, r.permutation(len(TABLES))%N_FOLDS)}; folds = dev["table_id"].map(TF).to_numpy()
        dev["seq_risk"] = distill(folds, sd)
        ds2 = dev.iloc[r.choice(len(dev),nn,replace=False)]; es2 = evf.iloc[r.choice(len(evf),nn,replace=False)]
        Xc2 = pd.concat([ds2[shared], es2[shared]], ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0)
        yc2 = np.concatenate([np.zeros(nn), np.ones(nn)])
        dc = lgb.train(dict(objective="binary",num_leaves=31,learning_rate=0.05,verbose=-1,num_threads=NThr), lgb.Dataset(Xc2,yc2), num_boost_round=200)
        pe = dc.predict(dev[shared].replace([np.inf,-np.inf],np.nan).fillna(0)); w_eval = np.clip(pe/(1-pe+1e-6),0.05,20.0); w_eval=w_eval/w_eval.mean()
        heldm = np.isin(folds,[3,4])
        ob = oof(base+PT+["seq_risk"], folds, heldm, sd)
        of = oof(base+PT+["seq_risk"]+FEATS_FR, folds, heldm, sd)
        bw = average_precision_score(y[heldm], ob[heldm], sample_weight=w_eval[heldm])
        ew = average_precision_score(y[heldm], of[heldm], sample_weight=w_eval[heldm])
        log(f"seed{sd}: base {bw:.4f} +FR {ew:.4f} delta {ew-bw:+.4f}")
        # per-family held-positive rank change (higher score = better; positive mean = FR helped that family)
        hp = heldm & labm & (y==1)
        print(f"  --- seed{sd} held labeled-positives: {hp.sum()}, by family ---")
        for f in ["directed_transfer","soft_play","coordinated_isolation"]:
            msk = hp & (fam==f)
            if msk.sum()==0: continue
            dscore = (of[msk]-ob[msk])
            fam_shift[f].append(dscore.mean())
            print(f"     {f:24s} n={msk.sum():3d}  mean score-shift {dscore.mean():+.5f}  (FR {'helps' if dscore.mean()>0 else 'hurts'})")
        # by exposure bucket among held positives (fixed global n_opp thirds so buckets are comparable across seeds)
        nopp = dev["fr_n_opp"].to_numpy()
        gq = np.quantile(nopp[labm&(y==1)], [0,0.33,0.66,1.0])
        print(f"  --- seed{sd} held positives by exposure (n_opp global thirds {gq.round(0).tolist()}) ---")
        for i,bn in enumerate(["lo","mid","hi"]):
            msk = hp & (nopp>=gq[i]) & (nopp<=gq[i+1] if i==2 else nopp<gq[i+1])
            if msk.sum()==0: continue
            dscore = (of[msk]-ob[msk]); exp_shift[bn].append(dscore.mean())
            print(f"     n_opp {bn} [{gq[i]:.0f},{gq[i+1]:.0f}] n={msk.sum():3d}  mean score-shift {dscore.mean():+.5f}")
    print("\n=== AGGREGATE across 5 seeds (mean score-shift; consistent sign = real slice effect) ===")
    for f in ["directed_transfer","soft_play","coordinated_isolation"]:
        v = np.array(fam_shift[f]); print(f"  {f:24s} mean {v.mean():+.5f}  per-seed {np.round(v,5).tolist()}  ({int((v>0).sum())}/{len(v)} pos)")
    for bn in ["lo","mid","hi"]:
        v = np.array(exp_shift[bn]); print(f"  n_opp {bn:3s}               mean {v.mean():+.5f}  per-seed {np.round(v,5).tolist()}  ({int((v>0).sum())}/{len(v)} pos)")

    # fold-variance hypothesis: correlation of delta with base difficulty across the 5 recorded seeds
    import json
    ck = EQ/"_fr_gate3_seeds.json"
    if ck.exists():
        d = json.loads(ck.read_text()); bws = np.array([d[k][0] for k in d]); dels = np.array([d[k][1]-d[k][0] for k in d])
        print(f"\n=== fold-variance test (5 recorded seeds) ===")
        print(f"  base-difficulty (AP): {np.round(bws,4).tolist()}")
        print(f"  deltas:               {np.round(dels,4).tolist()}")
        if len(bws)>2: print(f"  corr(delta, base-AP) = {np.corrcoef(bws,dels)[0,1]:+.3f}  (near 0 => sign is fold-noise, not difficulty-driven)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
