"""Augment hosen42's pair features with OUR edge (postflop + joint + timing), re-fit, judge vs baseline.

Builds pair-level features from per_hand_surprise (postflop policy) + the pair-hand maps:
  POSTFLOP:  mean/max/top3 of postflop_loose, total_loose over the pair's shared hands; fold_strong,
             passive_junk sums (soft-play / dump POSTFLOP, invisible to hosen42's preflop-only model).
  JOINT:     per pair, EXCESS co-surprise = obs P(both loose) - P(A loose)*P(B loose) [collusion doesn't
             factorize]; within-hand correlation of the two members' loose; both-postflop-passive rate.
  TIMING:    second-entrant-follows-first rate (already partially in baseline; we add postflop follow).

Appends these to dev_pair_features / eval_pair_features (242 cols), re-fits the EXACT two-stage PU pair
model + family heads (copied from the baseline recipe), and reports host_score on the dev population for
BASELINE (their 221 feats) vs OURS (+ edge feats). Also standalone univariate AUC of each edge feature.
Run:  python -m anchor_repro.policy_edge_train
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

STEP3 = Path("outputs/poker_collusion/hosen42_step3")
EDGE = STEP3 / "edge"
DATA = Path("data/poker")
SEED = 42
N_FOLDS = 5
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

PER_SURPRISE = EDGE / "per_hand_surprise.parquet"   # hand_id, player_id, phase, loose, postflop_loose, total_loose, fold_strong, passive_junk, ...


def _pair_edge_features(pair_hands_path: Path, phase: str) -> pl.DataFrame:
    """One row per pair: postflop + joint + passive-junk aggregates over the pair's shared hands."""
    ph = pl.scan_parquet(pair_hands_path).select(["pair_id", "hand_id", "player_1", "player_2"])
    sur = pl.scan_parquet(PER_SURPRISE).filter(pl.col("phase") == phase).select(
        ["hand_id", "player_id", "loose", "postflop_loose", "postflop_loose_max", "total_loose", "fold_strong", "passive_junk", "vpip_a"])
    a = ph.join(sur, left_on=["hand_id", "player_1"], right_on=["hand_id", "player_id"], how="left")
    b = ph.join(sur, left_on=["hand_id", "player_2"], right_on=["hand_id", "player_id"], how="left")
    # per (pair,hand) member surprises
    j = (a.select(["pair_id", "hand_id",
                   pl.col("loose").alias("a_loose"), pl.col("postflop_loose").alias("a_post"),
                   pl.col("total_loose").alias("a_tot"), pl.col("fold_strong").alias("a_foldstrong"),
                   pl.col("passive_junk").alias("a_passjunk"), pl.col("vpip_a").alias("a_vpip")])
         .join(b.select(["pair_id", "hand_id",
                         pl.col("loose").alias("b_loose"), pl.col("postflop_loose").alias("b_post"),
                         pl.col("total_loose").alias("b_tot"), pl.col("fold_strong").alias("b_foldstrong"),
                         pl.col("passive_junk").alias("b_passjunk"), pl.col("vpip_a").alias("b_vpip")]),
               on=["pair_id", "hand_id"]))
    LOOSE_THR = 1.0   # "surprised" = -log P >= 1  (P(action) <= 0.37)
    j = j.with_columns(
        (pl.col("a_tot") + pl.col("b_tot")).alias("h_total_loose_sum"),
        pl.min_horizontal("a_tot", "b_tot").alias("h_total_loose_min"),
        (pl.col("a_post") + pl.col("b_post")).alias("h_post_loose_sum"),
        pl.min_horizontal("a_post", "b_post").alias("h_post_loose_min"),
        ((pl.col("a_tot") >= LOOSE_THR)).cast(pl.Int8).alias("a_surp"),
        ((pl.col("b_tot") >= LOOSE_THR)).cast(pl.Int8).alias("b_surp"),
        (pl.col("a_foldstrong") + pl.col("b_foldstrong")).alias("h_foldstrong"),
        (pl.col("a_passjunk") + pl.col("b_passjunk")).alias("h_passjunk"),
        ((pl.col("a_passjunk") > 0) & (pl.col("b_passjunk") > 0)).cast(pl.Int8).alias("h_both_passjunk"),
    ).with_columns((pl.col("a_surp") & pl.col("b_surp")).cast(pl.Int8).alias("h_both_surp"))

    agg = j.group_by("pair_id").agg(
        pl.len().alias("_n"),
        # POSTFLOP aggregates
        pl.col("h_post_loose_sum").mean().alias("post_loose_mean"),
        pl.col("h_post_loose_sum").top_k(3).mean().alias("post_loose_top3"),
        pl.col("h_post_loose_min").mean().alias("post_loose_min_mean"),
        pl.col("h_total_loose_sum").mean().alias("total_loose_mean"),
        pl.col("h_total_loose_sum").top_k(3).mean().alias("total_loose_top3"),
        pl.col("h_total_loose_min").top_k(3).mean().alias("total_loose_min_top3"),
        pl.col("h_foldstrong").sum().alias("foldstrong_sum"),
        pl.col("h_passjunk").sum().alias("passjunk_sum"),
        pl.col("h_both_passjunk").sum().alias("both_passjunk_sum"),
        pl.col("h_both_passjunk").mean().alias("both_passjunk_rate"),
        # JOINT surprise pieces
        pl.col("a_surp").mean().alias("_pa"),
        pl.col("b_surp").mean().alias("_pb"),
        pl.col("h_both_surp").mean().alias("_pab"),
        pl.col("h_both_surp").sum().alias("both_surp_sum"),
        # within-hand surprise correlation over shared hands
        pl.corr("a_tot", "b_tot").alias("loose_corr"),
    ).with_columns(
        # EXCESS co-surprise over independent expectation (the collusion-specific signal)
        (pl.col("_pab") - pl.col("_pa") * pl.col("_pb")).alias("joint_surp_excess"),
        # lift: how much more often both are surprised than independent play predicts
        (pl.col("_pab") / (pl.col("_pa") * pl.col("_pb") + 1e-4)).clip(0, 50).alias("joint_surp_lift"),
        pl.col("loose_corr").fill_null(0.0),
    ).drop(["_pa", "_pb", "_pab", "_n"])
    return agg.collect(engine="streaming")


EDGE_FEATS = ["post_loose_mean", "post_loose_top3", "post_loose_min_mean", "total_loose_mean",
              "total_loose_top3", "total_loose_min_top3", "foldstrong_sum", "passjunk_sum",
              "both_passjunk_sum", "both_passjunk_rate", "both_surp_sum", "loose_corr",
              "joint_surp_excess", "joint_surp_lift"]


def main() -> int:
    # baseline pair features (already carry label/fold/behavior_family/is_labeled)
    dev = pd.read_parquet(STEP3 / "dev_pair_features.parquet")
    ev = pd.read_parquet(STEP3 / "eval_pair_features.parquet")
    log(f"baseline dev {dev.shape}, eval {ev.shape}")

    dev_edge = _pair_edge_features(STEP3 / "dev_pair_hands.parquet", "development").to_pandas()
    ev_edge = _pair_edge_features(STEP3 / "eval_pair_hands.parquet", "evaluation").to_pandas()
    log(f"edge feats dev {dev_edge.shape}, eval {ev_edge.shape}")
    dev = dev.merge(dev_edge, on="pair_id", how="left")
    ev = ev.merge(ev_edge, on="pair_id", how="left")
    for c in EDGE_FEATS:
        dev[c] = dev[c].astype("float32").fillna(0); ev[c] = ev[c].astype("float32").fillna(0)

    # recover label/fold/is_labeled (already in dev)
    y = dev["label"].fillna(0).astype(int).to_numpy()
    labm = dev["is_labeled"].to_numpy().astype(bool)
    # rebuild the SAME table-fold as baseline (TABLE_FOLD by permutation seed 42)
    TABLES = sorted(dev["table_id"].unique().tolist())
    rng = np.random.RandomState(SEED)
    TABLE_FOLD = {t: int(v) for t, v in zip(TABLES, rng.permutation(len(TABLES)) % N_FOLDS)}
    folds = dev["table_id"].map(TABLE_FOLD).to_numpy()

    # standalone univariate AUC of edge feats (positives vs confirmed negatives AND vs all eligible)
    print("\n=== edge feature univariate AUC (labelled pos vs confirmed neg | pos vs ALL eligible) ===")
    lab_pos = labm & (y == 1); lab_neg = labm & (y == 0)
    for c in EDGE_FEATS:
        v = dev[c].to_numpy()
        try:
            a_lab = roc_auc_score(y[labm], v[labm]); a_lab = max(a_lab, 1 - a_lab)
            a_all = roc_auc_score(y, v); a_all = max(a_all, 1 - a_all)
            print(f"  {a_lab:.4f} | {a_all:.4f}  {c}")
        except Exception: pass

    META = {"pair_id", "table_id", "player_1", "player_2", "label", "behavior_family", "is_labeled", "fold", "shared_hands", "chunk", "pred_family"}
    base_feats = [c for c in dev.columns if c not in META and not c.startswith("sus_") and c != "top5s_same_loser" and c not in EDGE_FEATS]
    our_feats = base_feats + EDGE_FEATS
    log(f"baseline feats {len(base_feats)}, ours {len(our_feats)} (+{len(EDGE_FEATS)})")

    PAIR_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                       feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                       verbose=-1, seed=SEED, num_threads=-1)
    PAIR_SEEDS = [SEED, SEED + 11, SEED + 23]
    POS_W, NEG_W, PU_W = 5.0, 1.0, 0.1

    def two_stage_oof(feats):
        X = dev[feats].astype("float32")
        w1 = np.where(labm, np.where(y == 1, POS_W, NEG_W), PU_W)
        # stage 1
        oof1 = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = (folds != f) & (w1 > 0); te = folds == f
            m = lgb.train({**PAIR_PARAMS, "seed": SEED}, lgb.Dataset(X[tr], y[tr], weight=w1[tr]), num_boost_round=750)
            oof1[te] = m.predict(X[te])
        pos_oof = oof1[labm & (y == 1)]
        thr_a, thr_p = np.quantile(pos_oof, 0.05), np.quantile(pos_oof, 0.50)
        unl = ~labm
        ambig = unl & (oof1 >= thr_a) & (oof1 < thr_p); pseudo = unl & (oof1 >= thr_p)
        y2 = np.where(pseudo, 1, y)
        w2 = np.where(labm, np.where(y == 1, POS_W, NEG_W), np.where(pseudo, 1.0, np.where(ambig, 0.0, PU_W)))
        oof = np.zeros(len(dev))
        for f in range(N_FOLDS):
            tr = (folds != f) & (w2 > 0); te = folds == f
            for sd in PAIR_SEEDS:
                m = lgb.train({**PAIR_PARAMS, "seed": sd}, lgb.Dataset(X[tr], y2[tr], weight=w2[tr]), num_boost_round=750)
                oof[te] += m.predict(X[te]) / len(PAIR_SEEDS)
        return oof

    log("fitting BASELINE (their feats)...")
    oof_base = two_stage_oof(base_feats)
    log("fitting OURS (their feats + edge)...")
    oof_ours = two_stage_oof(our_feats)

    def report(oof, tag):
        lab_ap = average_precision_score(y[labm], oof[labm])
        pu_ap = average_precision_score(y, oof)   # eval-mirrored: positives vs ALL eligible
        print(f"  [{tag}] labelled PairAP={lab_ap:.4f} | eval-mirrored PairAP (pos vs ALL {len(y):,})={pu_ap:.4f}")
        return pu_ap
    print("\n=== PAIR MODEL: baseline vs ours (eval-mirrored PairAP is the honest judge) ===")
    pu_base = report(oof_base, "baseline 221")
    pu_ours = report(oof_ours, "ours +edge")
    print(f"\n  DELTA eval-mirrored PairAP: {pu_ours - pu_base:+.4f}  ({'OURS WINS' if pu_ours > pu_base + 0.002 else 'no clear gain' })")

    np.save(EDGE / "oof_base.npy", oof_base); np.save(EDGE / "oof_ours.npy", oof_ours)
    (EDGE / "_train_summary.json").write_text(json.dumps(
        {"pu_ap_base": float(pu_base), "pu_ap_ours": float(pu_ours), "delta": float(pu_ours - pu_base)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
