"""AAAI joint-unit-advantage feature + eval-honest test (§111).

Collusion (AAAI collusion-table concept) = the pair, as ONE unit, gains MORE than a
matched INDEPENDENT-play baseline predicts. This is a RELATIVE-TO-BASELINE quantity, so it
should be far more transport-robust than the raw consistency rates that rode the dev-selection
bias (§108/§110).

Construction (all from the Layer-1 per-(hand,player) cache, over the 9.65M shared-hand universe):
  * per-player baseline: mean net_bb per hand across ALL that player's hands (their general
    expectation vs the field), computed on the FULL population (label-free, dev+eval pooled).
  * per shared hand of a pair: observed joint net (a_net_bb + b_net_bb) minus expected
    independent (a_baseline + b_baseline). Positive => the unit did better together than
    independent play predicts.
  * per pair: mean / sum / positive-rate / concentration of the per-hand advantage, plus the
    "who-benefits" asymmetry (one member consistently gains from the other).

Then evaluate on the eval-honest harness (importance-weighted AP + drift) vs:
  * the ~0.61 eval-honest consistency ceiling (§110),
  * V30's ~0.79 eval PairAP.

Run:  python -m anchor_repro.gen_joint_advantage
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.own_submission_recipes import adversarial_drift_auc_matrix

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
HAND_PLAYER = CACHE / "hand_player_agg.parquet"
HAND_CTX = CACHE / "hand_context.parquet"
SEED = 42


def _player_baseline() -> pl.DataFrame:
    """Per-player mean net_bb per hand across ALL their hands (label-free, full population)."""
    hp = pl.scan_parquet(HAND_PLAYER).select(["hand_id", "player_id", "net_chips"])
    bb = pl.scan_parquet(HAND_CTX).select(["hand_id", "big_blind"])
    return (
        hp.join(bb, on="hand_id")
        .with_columns((pl.col("net_chips") / pl.col("big_blind")).alias("net_bb"))
        .group_by("player_id")
        .agg(
            pl.col("net_bb").mean().alias("base_net_bb"),
            pl.col("net_bb").count().alias("base_n"),
        )
        .collect(engine="streaming")
    )


def build_pair_advantage(pair_frame: pl.LazyFrame, universe: pl.LazyFrame, baseline: pl.DataFrame) -> pl.DataFrame:
    """Per-pair joint-unit-advantage features over the pair's shared hands."""
    hp = pl.scan_parquet(HAND_PLAYER).select(["hand_id", "player_id", "net_chips"])
    bb = pl.scan_parquet(HAND_CTX).select(["hand_id", "big_blind"])
    base = baseline.lazy()
    pf = pair_frame.select(["pair_id", "player_1", "player_2"])

    def side(col):
        return (
            hp.join(pf.select(["pair_id", pl.col(col).alias("player_id")]), on="player_id", how="inner")
            .join(bb, on="hand_id")
            .join(base, on="player_id", how="left")
            .with_columns((pl.col("net_chips") / pl.col("big_blind")).alias("net_bb"))
            .select(["pair_id", "hand_id", "net_bb", "base_net_bb"])
        )

    a = side("player_1").rename({"net_bb": "a_net_bb", "base_net_bb": "a_base"})
    b = side("player_2").rename({"net_bb": "b_net_bb", "base_net_bb": "b_base"})
    shared = (
        a.join(b, on=["pair_id", "hand_id"], how="inner")
        .join(universe.select(["pair_id", "hand_id"]), on=["pair_id", "hand_id"], how="inner")
        .with_columns(
            (pl.col("a_net_bb") + pl.col("b_net_bb")).alias("joint_net"),
            (pl.col("a_base") + pl.col("b_base")).alias("joint_base"),
        )
        .with_columns(
            (pl.col("joint_net") - pl.col("joint_base")).alias("advantage"),
        )
    )
    agg = shared.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        pl.col("advantage").mean().alias("adv_mean"),
        pl.col("advantage").sum().alias("adv_sum"),
        pl.col("advantage").median().alias("adv_median"),
        (pl.col("advantage") > 0).mean().alias("adv_pos_rate"),
        pl.col("advantage").top_k(5).mean().alias("adv_top5"),
        pl.col("advantage").quantile(0.95).alias("adv_p95"),
        pl.col("joint_net").mean().alias("joint_net_mean"),
        pl.col("joint_net").sum().alias("joint_net_sum"),
        # who-benefits asymmetry: does one member gain while the other funds it?
        (pl.col("a_net_bb") - pl.col("b_net_bb")).abs().mean().alias("member_net_gap_mean"),
        ((pl.col("a_net_bb") > 0) & (pl.col("b_net_bb") < 0)).mean().alias("a_gains_b_funds_rate"),
        ((pl.col("b_net_bb") > 0) & (pl.col("a_net_bb") < 0)).mean().alias("b_gains_a_funds_rate"),
    ).with_columns(
        # advantage per hand normalized (concentration)
        (pl.col("adv_sum") / pl.col("n_hands").clip(1, None)).alias("adv_per_hand"),
        pl.max_horizontal("a_gains_b_funds_rate", "b_gains_a_funds_rate").alias("directional_funding_rate"),
    )
    return agg.collect(engine="streaming")


ADV_FEATS = ["adv_mean", "adv_sum", "adv_median", "adv_pos_rate", "adv_top5", "adv_p95",
             "joint_net_mean", "joint_net_sum", "member_net_gap_mean", "a_gains_b_funds_rate",
             "b_gains_a_funds_rate", "adv_per_hand", "directional_funding_rate", "n_hands"]


def _weighted_ap(y, scores, w):
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]; w = w[order]
    tp = np.cumsum(y * w); fp = np.cumsum((1 - y) * w)
    precision = tp / np.clip(tp + fp, 1e-9, None)
    total_pos = np.sum(y * w)
    return float(np.sum(precision * y * w) / total_pos) if total_pos > 0 else 0.0


def main() -> int:
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label", "behavior_family"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    dev_uni = pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
    eval_uni = pl.scan_parquet(CACHE / "eval_layer2.parquet").select(["pair_id", "hand_id"])

    print("building player baselines (full population)...", flush=True)
    baseline = _player_baseline()
    print(f"  {baseline.height:,} players, mean base_net_bb={baseline['base_net_bb'].mean():.4f} "
          f"(should be ~0 zero-sum)")

    dev = build_pair_advantage(labels.select(["pair_id", "player_1", "player_2"]).lazy(), dev_uni, baseline)
    dev = dev.join(labels.select(["pair_id", "label", "behavior_family"]), on="pair_id", how="left")
    y = dev["label"].to_numpy().astype(np.int8)
    print(f"dev pairs={dev.height} pos={int(y.sum())} device={xgb_device()}")

    # Univariate eval-honest requires eval features + weights; first do quick dev-CV univariate.
    print("\n=== univariate confirmed PairAP (dev) of joint-advantage features ===")
    for c in ADV_FEATS:
        v = dev[c].fill_null(0).to_numpy().astype(float)
        if np.unique(v).size < 2:
            continue
        ap = average_precision_score(y, v)
        print(f"  {c:26s} AP={ap:.4f}")

    # Per-family (advantage should be strongest for directed_transfer).
    print("\n=== adv_mean by family (planted signature) ===")
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation", "none"]:
        m = (dev["behavior_family"].fill_null("none") == fam).to_numpy()
        if m.sum() == 0:
            continue
        print(f"  {fam:24s} adv_mean median={np.median(dev['adv_mean'].to_numpy()[m]):.3f} "
              f"adv_pos_rate median={np.median(dev['adv_pos_rate'].to_numpy()[m]):.3f} n={int(m.sum())}")

    # Build eval features for the eval-honest harness.
    print("\nbuilding eval joint-advantage (this is the expensive join)...", flush=True)
    evf = build_pair_advantage(eval_pairs.lazy(), eval_uni, baseline)
    print(f"  eval pairs={evf.height}")

    Xd = dev.select(ADV_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    Xe = evf.select(ADV_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # Importance weights (density ratio) from an OOF dev-vs-eval classifier.
    dev_prob = np.zeros(len(Xd))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(Xd, y):
        X_tr = np.vstack([Xd.to_numpy()[tr], Xe.to_numpy()])
        y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(Xe))])
        clf = xgb.XGBClassifier(**xgb_params(objective="binary:logistic", eval_metric="auc",
                                             max_depth=4, n_estimators=300, learning_rate=0.05,
                                             subsample=0.85, colsample_bytree=0.8, reg_lambda=5))
        clf.fit(X_tr, y_tr)
        dev_prob[va] = clf.predict_proba(Xd.to_numpy()[va])[:, 1]
    dev_prob = np.clip(dev_prob, 1e-4, 1 - 1e-4)
    w = dev_prob / (1 - dev_prob)
    w = np.clip(w, np.quantile(w, 0.01), np.quantile(w, 0.99)); w = w / w.mean()

    # Table-grouped OOF detector.
    pair_tbl = (
        pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
        .join(pl.scan_parquet(HAND_CTX).select(["hand_id", "table_id"]), on="hand_id")
        .group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
        .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
        .select(["pair_id", "table_id"]).collect()
    )
    groups = dev.join(pair_tbl, on="pair_id", how="left")["table_id"].to_numpy()
    oof = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(Xd, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.04,
                                 subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0,
                                 scale_pos_weight=float(spw)),
                      xgb.DMatrix(Xd.iloc[tr], label=y[tr]), num_boost_round=350)
        oof[va] = m.predict(xgb.DMatrix(Xd.iloc[va]))
    naive = average_precision_score(y, oof)
    wtd = _weighted_ap(y, oof, w)
    res = adversarial_drift_auc_matrix(Xd.to_numpy(), Xe.to_numpy(), block="joint_adv", seed=SEED)
    print("\n=== JOINT-ADVANTAGE detector (eval-honest) ===")
    print(f"  naive dev-CV AP = {naive:.4f}")
    print(f"  EVAL-WEIGHTED AP = {wtd:.4f}   [consistency ceiling ~0.61 | V30 eval ~0.79]")
    print(f"  AUC = {roc_auc_score(y, oof):.4f}")
    print(f"  drift AUC = {res.auc:.4f} (thr {res.threshold}; pass={res.passed})")
    print(f"  pos mean w={w[y==1].mean():.3f} neg mean w={w[y==0].mean():.3f}")

    summary = {"naive_ap": float(naive), "eval_weighted_ap": float(wtd),
               "auc": float(roc_auc_score(y, oof)), "drift_auc": float(res.auc),
               "drift_pass": bool(res.passed), "consistency_ceiling": 0.61, "v30_eval": 0.79}
    (CACHE / "_joint_advantage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote _joint_advantage_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
