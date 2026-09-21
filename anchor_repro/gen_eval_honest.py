"""Eval-honest transfer test of the consistency pair signal (§109 highest-EV action).

Three things at once:
 1. IMPORTANCE WEIGHTS: train a dev-vs-eval classifier on pair features -> density ratio
    w(pair) = P(eval)/P(dev) for each dev pair. This reweights the biased dev sample (§108)
    to look like the eval population.
 2. SELECTION-ROBUST SUBSET: keep only features whose dev_NEG-vs-eval solo drift < 0.55
    (features where a dev NEGATIVE already looks like an eval pair -> not selection-tainted).
 3. EVAL-HONEST SEPARATION: re-measure pair positive/negative separation with the eval
    importance weights applied to the metric (weighted AP), on the robust subset, and report
    the joint drift of that subset. Compare to the naive dev-CV number (0.909).

Verdict:
 - If weighted separation stays high AND subset joint drift < 0.65 -> TRANSFERABLE signal.
 - If weighted separation collapses -> the 0.909 was dev-selection artifact (proves §108).

Run:  python -m anchor_repro.gen_eval_honest
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

from anchor_repro.gen_pair_detector import pair_features
from anchor_repro.gpu_config import xgb_params, xgb_device

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
SEED = 42


def _weighted_ap(y, scores, w):
    """Importance-weighted average precision (weights on the ranking population)."""
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]; w = w[order]
    tp = np.cumsum(y * w)
    fp = np.cumsum((1 - y) * w)
    precision = tp / np.clip(tp + fp, 1e-9, None)
    total_pos = np.sum(y * w)
    if total_pos <= 0:
        return 0.0
    # AP = sum over positives of precision-at-that-rank * weight, normalized
    return float(np.sum(precision * y * w) / total_pos)


def _solo_drift(dv, ev, rng, n=80000):
    m = min(len(dv), len(ev), n)
    di = rng.choice(len(dv), min(len(dv), m), replace=False)
    ei = rng.choice(len(ev), m, replace=False)
    x = np.concatenate([dv[di], ev[ei]]); y = np.concatenate([np.zeros(len(di)), np.ones(m)])
    return max(roc_auc_score(y, x), roc_auc_score(y, -x))


def main() -> int:
    dev = pair_features(CACHE / "dev_layer2.parquet")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(labels, on="pair_id", how="left")
    all_feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")
                 and not c.endswith("_sum")]
    y = dev["label"].to_numpy().astype(np.int8)
    rng = np.random.default_rng(SEED)
    print(f"dev pairs={dev.height} pos={int(y.sum())} eval={evf.height} feats={len(all_feats)} dev={xgb_device()}")

    Xd = dev.select(all_feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    Xe = evf.select(all_feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # ---- 1. Importance weights: density ratio P(eval)/P(dev) per dev pair ----
    Xall = np.vstack([Xd.to_numpy(), Xe.to_numpy()])
    yall = np.concatenate([np.zeros(len(Xd)), np.ones(len(Xe))])
    dev_prob = np.zeros(len(Xd))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    # OOF eval-probability for dev rows: train on (dev_fold_train + all eval), predict dev_fold_val.
    dev_idx = np.arange(len(Xd))
    for tr, va in skf.split(Xd, y):  # stratify by label to keep folds balanced
        X_tr = np.vstack([Xd.to_numpy()[tr], Xe.to_numpy()])
        y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(Xe))])
        clf = xgb.XGBClassifier(**xgb_params(objective="binary:logistic", eval_metric="auc",
                                             max_depth=4, n_estimators=300, learning_rate=0.05,
                                             subsample=0.85, colsample_bytree=0.8, reg_lambda=5))
        clf.fit(X_tr, y_tr)
        dev_prob[va] = clf.predict_proba(Xd.to_numpy()[va])[:, 1]
    # density ratio w = p/(1-p), clipped; normalize mean to 1.
    dev_prob = np.clip(dev_prob, 1e-4, 1 - 1e-4)
    w = dev_prob / (1 - dev_prob)
    w = np.clip(w, np.quantile(w, 0.01), np.quantile(w, 0.99))
    w = w / w.mean()
    print(f"importance weights: mean={w.mean():.3f} p10={np.quantile(w,0.1):.3f} "
          f"p90={np.quantile(w,0.9):.3f} max={w.max():.3f}")
    print(f"  pos mean w={w[y==1].mean():.3f}  neg mean w={w[y==0].mean():.3f}  "
          f"(pos<neg => positives are OVER-represented in dev vs eval)")

    # ---- 2. Selection-robust subset: dev_NEG vs eval solo drift < 0.55 ----
    neg_mask = y == 0
    robust = []
    drift_rows = []
    for c in all_feats:
        dvn = Xd[c].to_numpy()[neg_mask]
        ev = Xe[c].to_numpy()
        d = _solo_drift(dvn, ev, rng)
        drift_rows.append((c, d))
        if d < 0.55:
            robust.append(c)
    drift_rows.sort(key=lambda r: r[1])
    print(f"\nselection-robust features (dev_NEG vs eval solo drift < 0.55): {len(robust)}/{len(all_feats)}")
    print("  cleanest:", [f"{c}={d:.3f}" for c, d in drift_rows[:6]])
    print("  dirtiest:", [f"{c}={d:.3f}" for c, d in drift_rows[-6:]])

    # ---- 3. Eval-honest detector on the robust subset, table-grouped OOF ----
    pair_tbl = (
        pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
        .join(pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"]), on="hand_id")
        .group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
        .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
        .select(["pair_id", "table_id"]).collect()
    )
    dev2 = dev.join(pair_tbl, on="pair_id", how="left")
    groups = dev2["table_id"].to_numpy()

    def run(cols, weights=None, tag=""):
        X = dev2.select(cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        oof = np.zeros(len(y))
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        for tr, va in sgkf.split(X, y, groups):
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            dtr = xgb.DMatrix(X.iloc[tr], label=y[tr], weight=(weights[tr] if weights is not None else None))
            m = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5,
                                     eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                                     reg_lambda=6.0, scale_pos_weight=float(spw)),
                          dtr, num_boost_round=350)
            oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
        naive = average_precision_score(y, oof)
        wtd = _weighted_ap(y, oof, w)  # eval-weighted AP = eval-honest metric
        print(f"  [{tag}] confirmed_AP={naive:.4f}  EVAL-WEIGHTED_AP={wtd:.4f}  AUC={roc_auc_score(y,oof):.4f}")
        return naive, wtd, oof

    print("\n=== detector variants (naive dev-CV vs eval-honest weighted) ===")
    print("baseline positive rate:", round(float(y.mean()), 4))
    a_all_unw = run(all_feats, None, "all feats, unweighted")
    a_all_w = run(all_feats, w, "all feats, EVAL-WEIGHTED")
    a_rob_unw = run(robust, None, "robust subset, unweighted") if robust else None
    a_rob_w = run(robust, w, "robust subset, EVAL-WEIGHTED") if robust else None

    # joint drift of the robust subset
    from anchor_repro.own_submission_recipes import adversarial_drift_auc_matrix
    if robust:
        res = adversarial_drift_auc_matrix(Xd[robust].to_numpy(), Xe[robust].to_numpy(), block="robust", seed=SEED)
        print(f"\nrobust-subset joint drift AUC={res.auc:.4f} (thr {res.threshold}; pass={res.passed})")

    summary = {
        "n_robust": len(robust), "robust_features": robust,
        "all_unweighted_confirmed": a_all_unw[0], "all_unweighted_evalwtd": a_all_unw[1],
        "all_weighted_confirmed": a_all_w[0], "all_weighted_evalwtd": a_all_w[1],
        "robust_weighted_evalwtd": (a_rob_w[1] if a_rob_w else None),
        "robust_joint_drift": (float(res.auc) if robust else None),
        "pos_mean_weight": float(w[y == 1].mean()), "neg_mean_weight": float(w[y == 0].mean()),
    }
    (CACHE / "_eval_honest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote _eval_honest_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
