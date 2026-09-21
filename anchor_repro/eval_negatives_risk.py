"""THE negative-space experiment: train pair risk with EVAL pairs as the unlabeled-negative class,
using the STRONG V30 feature basis. Never shipped (cand_13 tried it ONCE with WEAK features -> 0.172).

Root cause of every prior failure (dossier §108/§110): the dev labeled set is a CURATED sample
(dev-NEG vs eval AUC 0.712; positives 3.5x denser than eval). Models trained to beat the dev negatives
learn "dev-selected pair", not "collusion" -> collapse on eval. The named-but-unshipped fix: draw the
decision boundary against the ACTUAL scored population.

Construction:
  positives  = 372 confirmed dev colluding pairs (label 1)
  negatives  = a large sample of the 112,540 EVAL pairs (label 0, the real population)
  features   = V30 per-hand basis (65 cols) aggregated to pair level (mean/max/p95/top5), IDENTICAL
               dev/eval (parity confirmed). Phase-agnostic pair aggregates, so the model cannot trivially
               learn dev-vs-eval phase.
  guard      = we ALSO train dev-positives vs dev-negatives(confirmed) and check the eval-negative model
               isn't just phase detection (feature importances / a dev-vs-eval AUC sanity).

Then score ALL 112,540 eval pairs, judge eval-honest vs V30 0.70904.
Run:  python -m anchor_repro.eval_negatives_risk
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from anchor_repro.gpu_config import xgb_params
from anchor_repro.eval_honest_harness import Harness, D, CACHE

DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
EVAL_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
OUT = CACHE / "eval_negatives"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# the V30 per-hand signal columns (exclude ids/keys). Aggregated to pair level below.
SKIP = {"pair_id", "hand_id", "table_id"}


def _pair_aggregate(path: Path, feats: list[str]) -> pl.DataFrame:
    """Aggregate per-hand features to pair level: mean/max/p95/top5 (V30-style)."""
    lf = pl.scan_parquet(path)
    agg = [pl.len().alias("shared_hands")]
    agg += [pl.col(c).mean().alias(f"{c}_mean") for c in feats]
    agg += [pl.col(c).max().alias(f"{c}_max") for c in feats]
    agg += [pl.col(c).quantile(0.95, interpolation="nearest").alias(f"{c}_p95") for c in feats]
    agg += [pl.col(c).top_k(5).mean().alias(f"{c}_top5") for c in feats]
    return lf.group_by("pair_id").agg(agg).collect(engine="streaming")


def main() -> int:
    H = Harness()
    t = time.time()
    feats = [c for c in pl.scan_parquet(DEV_HF).collect_schema().names() if c not in SKIP]
    print(f"per-hand signal cols: {len(feats)}")

    dev_pair = _pair_aggregate(DEV_HF, feats)
    eval_pair = _pair_aggregate(EVAL_HF, feats)
    print(f"aggregated: dev {dev_pair.height:,} pairs, eval {eval_pair.height:,} pairs ({time.time()-t:.0f}s)")

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev_pair = dev_pair.join(labels, on="pair_id", how="left")
    pos = dev_pair.filter(pl.col("label") == 1)
    print(f"confirmed positives with features: {pos.height}")

    FEAT_COLS = [c for c in dev_pair.columns if c not in {"pair_id", "label", "shared_hands"}]
    Xpos = pos.select(FEAT_COLS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # ---- negatives = large sample of EVAL pairs (the ACTUAL scored population) ----
    rng = np.random.default_rng(SEED)
    n_neg = 40000
    neg_idx = rng.choice(eval_pair.height, n_neg, replace=False)
    eval_neg = eval_pair[neg_idx]
    Xneg = eval_neg.select(FEAT_COLS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    print(f"negatives = {n_neg:,} EVAL pairs (the real population)")

    # ---- GUARD: is dev-pos-vs-eval-neg just phase detection? check dev-neg vs eval-neg AUC ----
    dev_neg = dev_pair.filter(pl.col("label") == 0)
    Xdevneg = dev_neg.select(FEAT_COLS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    Xg = pd.concat([Xdevneg, Xneg], ignore_index=True)
    yg = np.concatenate([np.ones(len(Xdevneg)), np.zeros(len(Xneg))])
    gm = xgb.train(xgb_params(objective="binary:logistic", eval_metric="auc", max_depth=4, eta=0.1,
                              subsample=0.8, colsample_bytree=0.8),
                   xgb.DMatrix(Xg, label=yg), num_boost_round=120)
    from sklearn.model_selection import train_test_split
    ph_auc = roc_auc_score(yg, gm.predict(xgb.DMatrix(Xg)))
    print(f"GUARD dev-NEG vs eval-NEG separability AUC (in-sample) = {ph_auc:.4f} "
          f"(high => features carry dev/eval phase structure; model may exploit it)")

    # ---- train PU risk: pos (dev confirmed) vs eval-population negatives ----
    Xtr = pd.concat([Xpos, Xneg], ignore_index=True)
    ytr = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    # bag over eval-negative draws for stability
    n_bags = 6
    eval_all = eval_pair.select(FEAT_COLS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    dev_all = dev_pair.select(FEAT_COLS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    eval_score = np.zeros(eval_pair.height, np.float64)
    dev_score = np.zeros(dev_pair.height, np.float64)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.03,
                    subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=6.0)
    for bag in range(n_bags):
        bidx = rng.choice(eval_pair.height, n_neg, replace=False)
        Xb = eval_pair[bidx].select(FEAT_COLS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        Xt = pd.concat([Xpos, Xb], ignore_index=True)
        yt = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xb))])
        spw = len(Xb) / len(Xpos)
        m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(Xt, label=yt), num_boost_round=350)
        eval_score += m.predict(xgb.DMatrix(eval_all)) / n_bags
        dev_score += m.predict(xgb.DMatrix(dev_all)) / n_bags
    print(f"trained {n_bags}-bag eval-negative PU risk ({time.time()-t:.0f}s)")

    # ---- dev OOF-style check: how well does it separate confirmed pos from confirmed NEG? ----
    dev_pair = dev_pair.with_columns(pl.Series("risk", dev_score))
    conf = dev_pair.filter(pl.col("label").is_in([0, 1]))
    yc = conf["label"].to_numpy(); rc = conf["risk"].to_numpy()
    print(f"confirmed pos-vs-neg AUC = {roc_auc_score(yc, rc):.4f}  AP = {average_precision_score(yc, rc):.4f}")

    # ---- eval-honest judge vs V30 ----
    # dev oof for the harness: use the dev_score on confirmed pairs
    dev_oof = conf["risk"].to_numpy()
    dev_ids = conf["pair_id"].to_list()
    eval_ids = eval_pair["pair_id"].to_list()
    escore = (eval_score - eval_score.min()) / (eval_score.max() - eval_score.min() + 1e-12)
    r = H.judge("eval_negatives_risk", dev_oof, dev_ids, escore, eval_ids)
    print(f"\n=== EVAL-NEGATIVES RISK (eval-honest) ===")
    print(f"  {r}")
    print(f"  bar = V30 real LB 0.70904 / eval-honest V30 ~0.79")

    # save eval risk for possible submission
    pl.DataFrame({"pair_id": eval_ids, "risk_score": escore.astype(np.float32)}).write_parquet(OUT / "eval_negatives_risk.parquet")
    (OUT / "_summary.json").write_text(json.dumps({
        "phase_guard_auc": float(ph_auc), "confirmed_auc": float(roc_auc_score(yc, rc)),
        "eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
