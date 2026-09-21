"""FIELD-NORMALIZED postflop surprise — the fix the precision diagnosis pointed to.

Our raw postflop/joint surprise had 0.90 AUC but injected top-end false positives (ordinary loose
players). hosen42's preflop loose_pf_sum is precision-robust ONLY because it is PARTNER-VS-FIELD
normalized (player's surprise WITH partner minus WITHOUT, shrunk) — cancelling "this player is just
loose". hosen42 has NO postflop baseline at all, so a field-normalized POSTFLOP surprise is genuinely
non-redundant.

Build per player x phase: base_postflop_loose = mean postflop_loose over ALL that player's hands.
Per pair: p{i}_postloose_pf = (mean postflop_loose of member i on the pair's SHARED hands  -  member
i's field postflop_loose on hands WITHOUT the partner), * shrink(n_shared). Then pf_sum/pf_min. Same
for total_loose (pre+post) and for the JOINT excess. Append ONLY these pvf feats to baseline, re-fit,
judge eval-mirrored vs 0.7299.
Run:  python -m anchor_repro.policy_edge_pvf
"""
from __future__ import annotations
import json, time, numpy as np, pandas as pd, polars as pl, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

STEP3 = Path("outputs/poker_collusion/hosen42_step3"); EDGE = STEP3 / "edge"
PER = EDGE / "per_hand_surprise.parquet"   # hand_id, player_id, phase, loose, postflop_loose, total_loose, ...
SEED, N_FOLDS, SHRINK = 42, 5, 25.0
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def pvf_features(pair_hands_path: Path, phase: str) -> pl.DataFrame:
    sur = pl.scan_parquet(PER).filter(pl.col("phase") == phase).select(
        ["hand_id", "player_id", "postflop_loose", "total_loose", "tight"])
    # join decision-side proxy EV-loss (decorrelation-gate passed, §evloss) — kept for compat, unused in 0.81092 cfg
    evl_path = EDGE / "evloss_perhand.parquet"
    if evl_path.exists():
        evl = pl.scan_parquet(evl_path).filter(pl.col("phase") == phase).select(["hand_id", "player_id", "evloss"])
        sur = sur.join(evl, on=["hand_id", "player_id"], how="left").with_columns(pl.col("evloss").fill_null(0.0))
    else:
        sur = sur.with_columns(pl.lit(0.0).alias("evloss"))
    # join CONTINUOUS decision-equity (chips of EV given up per decision, exact-equity based, §decision_equity)
    deq_path = EDGE / "decision_equity.parquet"
    if deq_path.exists():
        deq = pl.scan_parquet(deq_path).filter(pl.col("phase") == phase).select(["hand_id", "player_id", "ev_mistake_total"])
        sur = sur.join(deq, on=["hand_id", "player_id"], how="left").with_columns(pl.col("ev_mistake_total").fill_null(0.0).alias("deq"))
    else:
        sur = sur.with_columns(pl.lit(0.0).alias("deq"))
    # each player's field baseline over ALL their hands in this phase
    base = sur.group_by("player_id").agg(
        pl.len().alias("base_hands"),
        pl.col("postflop_loose").mean().alias("base_post"),
        pl.col("total_loose").mean().alias("base_tot"),
        pl.col("tight").mean().alias("base_tight"),
        pl.col("evloss").mean().alias("base_evl"),
        pl.col("deq").mean().alias("base_deq"))
    ph = pl.scan_parquet(pair_hands_path).select(["pair_id", "hand_id", "player_1", "player_2"])
    # member i's mean surprise on the pair's shared hands
    def side(pcol, key):
        return (ph.join(sur, left_on=["hand_id", pcol], right_on=["hand_id", "player_id"], how="left")
                .group_by("pair_id").agg(
                    pl.len().alias("n_shared"),
                    pl.col("postflop_loose").mean().alias(f"{key}_post_shared"),
                    pl.col("total_loose").mean().alias(f"{key}_tot_shared"),
                    pl.col("tight").mean().alias(f"{key}_tight_shared"),
                    pl.col("evloss").mean().alias(f"{key}_evl_shared"),
                    pl.col("deq").mean().alias(f"{key}_deq_shared"),
                    pl.first(pcol).alias(f"{key}_pid")))
    a = side("player_1", "a"); b = side("player_2", "b").drop("n_shared")
    # per (pair,hand) BOTH-postflop-surprised aggregation (pure postflop, strong form)
    both_post = (ph.join(sur.select(["hand_id", "player_id", "postflop_loose"]).rename({"postflop_loose": "a_pl"}),
                         left_on=["hand_id", "player_1"], right_on=["hand_id", "player_id"], how="left")
                 .join(sur.select(["hand_id", "player_id", "postflop_loose"]).rename({"postflop_loose": "b_pl"}),
                       left_on=["hand_id", "player_2"], right_on=["hand_id", "player_id"], how="left")
                 .with_columns(pl.min_horizontal(pl.col("a_pl").fill_null(0), pl.col("b_pl").fill_null(0)).alias("both_post_min"))
                 .group_by("pair_id").agg(
                     pl.col("both_post_min").top_k(3).mean().alias("both_post_min_top3"),
                     pl.col("both_post_min").mean().alias("both_post_min_mean"),
                     ((pl.col("both_post_min") >= 1.0).cast(pl.Int8)).sum().alias("both_post_surp_ct")))
    j = a.join(b, on="pair_id").join(both_post, on="pair_id", how="left")
    j = (j.join(base.rename({"player_id": "a_pid", "base_hands": "a_bh", "base_post": "a_bpost", "base_tot": "a_btot", "base_tight": "a_btight", "base_evl": "a_bevl", "base_deq": "a_bdeq"}), on="a_pid", how="left")
          .join(base.rename({"player_id": "b_pid", "base_hands": "b_bh", "base_post": "b_bpost", "base_tot": "b_btot", "base_tight": "b_btight", "base_evl": "b_bevl", "base_deq": "b_bdeq"}), on="b_pid", how="left"))
    def pvf(shared, base_mean, base_n, ncol):
        # field = mean over hands WITHOUT the partner; shrink by shared exposure
        field = (base_mean * base_n - shared * pl.col(ncol)) / (base_n - pl.col(ncol)).clip(lower_bound=1)
        return ((shared - field) * pl.col(ncol) / (pl.col(ncol) + SHRINK))
    j = j.with_columns(
        pvf(pl.col("a_post_shared"), pl.col("a_bpost"), pl.col("a_bh"), "n_shared").alias("a_post_pf"),
        pvf(pl.col("b_post_shared"), pl.col("b_bpost"), pl.col("b_bh"), "n_shared").alias("b_post_pf"),
        pvf(pl.col("a_tot_shared"), pl.col("a_btot"), pl.col("a_bh"), "n_shared").alias("a_tot_pf"),
        pvf(pl.col("b_tot_shared"), pl.col("b_btot"), pl.col("b_bh"), "n_shared").alias("b_tot_pf"),
        pvf(pl.col("a_tight_shared"), pl.col("a_btight"), pl.col("a_bh"), "n_shared").alias("a_tight_pf"),
        pvf(pl.col("b_tight_shared"), pl.col("b_btight"), pl.col("b_bh"), "n_shared").alias("b_tight_pf"),
        pvf(pl.col("a_evl_shared"), pl.col("a_bevl"), pl.col("a_bh"), "n_shared").alias("a_evl_pf"),
        pvf(pl.col("b_evl_shared"), pl.col("b_bevl"), pl.col("b_bh"), "n_shared").alias("b_evl_pf"),
        pvf(pl.col("a_deq_shared"), pl.col("a_bdeq"), pl.col("a_bh"), "n_shared").alias("a_deq_pf"),
        pvf(pl.col("b_deq_shared"), pl.col("b_bdeq"), pl.col("b_bh"), "n_shared").alias("b_deq_pf"),
    ).with_columns(
        (pl.col("a_post_pf") + pl.col("b_post_pf")).alias("post_loose_pf_sum"),
        pl.min_horizontal("a_post_pf", "b_post_pf").alias("post_loose_pf_min"),
        (pl.col("a_tot_pf") + pl.col("b_tot_pf")).alias("total_loose_pf_sum"),
        pl.min_horizontal("a_tot_pf", "b_tot_pf").alias("total_loose_pf_min"),
        (pl.col("a_tight_pf") + pl.col("b_tight_pf")).alias("tight_pf_sum"),
        pl.max_horizontal("a_tight_pf", "b_tight_pf").alias("tight_pf_max"),
        (pl.col("a_evl_pf") + pl.col("b_evl_pf")).alias("evl_pf_sum"),
        pl.min_horizontal("a_evl_pf", "b_evl_pf").alias("evl_pf_min"),
        (pl.col("a_deq_pf") + pl.col("b_deq_pf")).alias("deq_pf_sum"),
        pl.min_horizontal("a_deq_pf", "b_deq_pf").alias("deq_pf_min"),
    ).select(["pair_id", "post_loose_pf_sum", "post_loose_pf_min", "total_loose_pf_sum", "total_loose_pf_min",
              "both_post_min_top3", "both_post_min_mean", "both_post_surp_ct", "tight_pf_sum", "tight_pf_max",
              "evl_pf_sum", "evl_pf_min", "deq_pf_sum", "deq_pf_min"])
    return j.collect(engine="streaming")


# 0.81092 config = postflop + tight channels ONLY. evl_pf_* are computed but EXCLUDED (§141: EV-loss
# failed both gates -> dead). Do NOT re-add evl_pf_sum/evl_pf_min.
PVF_FEATS = ["post_loose_pf_sum", "post_loose_pf_min", "total_loose_pf_sum", "total_loose_pf_min",
             "both_post_min_top3", "both_post_min_mean", "both_post_surp_ct", "tight_pf_sum", "tight_pf_max"]


def main():
    dev = pd.read_parquet(STEP3 / "dev_pair_features.parquet")
    de = pvf_features(STEP3 / "dev_pair_hands.parquet", "development").to_pandas()
    dev = dev.merge(de, on="pair_id", how="left")
    for c in PVF_FEATS: dev[c] = dev[c].astype("float32").fillna(0)
    y = dev["label"].fillna(0).astype(int).to_numpy(); labm = dev["is_labeled"].to_numpy().astype(bool)
    log(f"pvf feats built dev {de.shape}")

    print("\n=== field-normalized postflop pvf feats: univariate AUC (labelled | eval-mirrored) + precision@400 ===")
    for c in PVF_FEATS:
        v = dev[c].to_numpy()
        a_lab = roc_auc_score(y[labm], v[labm]); a_all = roc_auc_score(y, v)
        order = np.argsort(-v); p400 = int(y[order[:400]].sum())
        print(f"  {c:22} AUC {max(a_lab,1-a_lab):.4f} | {max(a_all,1-a_all):.4f}   true-pos@top400={p400}")

    TABLES = sorted(dev["table_id"].unique().tolist())
    rng = np.random.RandomState(SEED)
    TF = {t: int(v) for t, v in zip(TABLES, rng.permutation(len(TABLES)) % N_FOLDS)}
    folds = dev["table_id"].map(TF).to_numpy()
    META = {"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base_feats = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c != "top5s_same_loser" and c not in PVF_FEATS]
    P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=-1)
    SEEDS = [SEED, SEED+11, SEED+23]

    def two_stage(feats):
        X = dev[feats].astype("float32")
        w1 = np.where(labm, np.where(y==1,5.0,1.0), 0.1)
        oof1 = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w1>0); te=folds==f
            m=lgb.train({**P,"seed":SEED}, lgb.Dataset(X[tr], y[tr], weight=w1[tr]), num_boost_round=750); oof1[te]=m.predict(X[te])
        pos=oof1[labm&(y==1)]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
        unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp); ps=unl&(oof1>=tp)
        y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
        oof=np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr=(folds!=f)&(w2>0); te=folds==f
            for sd in SEEDS:
                m=lgb.train({**P,"seed":sd}, lgb.Dataset(X[tr], y2[tr], weight=w2[tr]), num_boost_round=750); oof[te]+=m.predict(X[te])/len(SEEDS)
        return oof
    log("fit baseline..."); ob=two_stage(base_feats)
    log("fit +pvf...");    op=two_stage(base_feats+PVF_FEATS)
    apb,app=average_precision_score(y,ob),average_precision_score(y,op)
    print(f"\n=== eval-mirrored PairAP: baseline {apb:.4f} | +field-norm-postflop {app:.4f}  delta {app-apb:+.4f} ({'WINS' if app>apb+0.002 else 'no gain'})")
    np.save(EDGE/"oof_pvf.npy", op)
    (EDGE/"_pvf_summary.json").write_text(json.dumps({"base":float(apb),"pvf":float(app),"delta":float(app-apb)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
