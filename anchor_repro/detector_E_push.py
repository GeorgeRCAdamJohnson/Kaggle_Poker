"""Push detector E — the train-against-eval PU-learner (dossier §115, eval-honest 0.639).

E was the ONLY fresh detector to beat the consistency ceiling, and it's crude: one negative
draw, depth 4, Layer-2 stable basis only. This module strengthens it and sweeps the knobs,
judged ONLY on the eval-honest harness gate.

Improvements:
  * BAGGED negative sampling: average many independent draws of unlabeled eval pairs as
    negatives (cuts the single-draw variance that likely capped E at 0.639).
  * RICHER feature basis: union of the stable Layer-2 pair aggregates + the directed-transfer
    view (A) + the soft-play view (B) — one wide matrix, still transport-diagnosed.
  * Knob sweep: negative pool size, #bags, depth, rounds.

Leak-free eval-honest OOF (same structure as E):
  positives get a HELD-OUT score (only bags whose training positives excluded them); dev
  NEGATIVES are never training rows (negatives come from eval) so they're scored by the full
  bag ensemble. eval scored by the full bag ensemble.

Run:  python -m anchor_repro.detector_E_push
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D, SEED
from anchor_repro.gen_pair_detector import pair_features
from anchor_repro.detector_bakeoff import _agg_from_layer2, _A_COLS, _B_COLS


def _rich_features():
    """Union feature matrix: stable Layer-2 basis + directed view + soft-play view."""
    lab = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    # stable basis (drops _sum)
    dev_s = pair_features(CACHE / "dev_layer2.parquet")
    ev_s = pair_features(CACHE / "eval_layer2.parquet")
    stable = [c for c in dev_s.columns if c not in ("pair_id", "label", "table_id") and not c.endswith("_sum")]
    # directed + soft views (rename to avoid collisions; they use different agg cols)
    dev_a = _agg_from_layer2(CACHE / "dev_layer2.parquet", _A_COLS)
    ev_a = _agg_from_layer2(CACHE / "eval_layer2.parquet", _A_COLS)
    dev_b = _agg_from_layer2(CACHE / "dev_layer2.parquet", _B_COLS)
    ev_b = _agg_from_layer2(CACHE / "eval_layer2.parquet", _B_COLS)

    def merge(s, a, b):
        a = a.drop("n_hands"); b = b.drop("n_hands")
        return s.join(a, on="pair_id", how="left").join(b, on="pair_id", how="left")

    dev = merge(dev_s.select(["pair_id", *stable]), dev_a, dev_b).join(lab, on="pair_id", how="left")
    ev = merge(ev_s.select(["pair_id", *stable]), ev_a, ev_b)
    feats = [c for c in dev.columns if c not in ("pair_id", "label")]
    return dev, ev, feats


def _bagged_pu(Xd: np.ndarray, y: np.ndarray, Xe: np.ndarray, feats: list,
               n_bags: int, n_neg: int, depth: int, rounds: int, seed: int = SEED):
    """Bagged PU-learner. Returns (dev_oof leak-free, eval_scores)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_dev_idx = np.where(y == 0)[0]
    n = len(y)

    oof_accum = np.zeros(n, dtype=np.float64)
    oof_pos_count = np.zeros(n, dtype=np.float64)   # how many bags contributed a HELD-OUT score to each positive
    neg_accum = np.zeros(len(neg_dev_idx), dtype=np.float64)
    eval_accum = np.zeros(len(Xe), dtype=np.float64)

    params = lambda spw: xgb_params(objective="binary:logistic", eval_metric="aucpr",
                                    max_depth=depth, eta=0.04, subsample=0.85,
                                    colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0,
                                    scale_pos_weight=float(spw))

    # For each bag: split positives into 5 folds, train (train-fold pos + fresh eval negs),
    # predict the held-out positives; also accumulate negative + eval predictions.
    for bag in range(n_bags):
        kf = KFold(5, shuffle=True, random_state=seed + bag)
        neg_sample = rng.choice(len(Xe), n_neg, replace=False)
        Xe_neg = Xe[neg_sample]
        for tr_pos, va_pos in kf.split(pos_idx):
            train_pos = pos_idx[tr_pos]
            held_pos = pos_idx[va_pos]
            X_tr = np.vstack([Xd[train_pos], Xe_neg])
            y_tr = np.concatenate([np.ones(len(train_pos)), np.zeros(n_neg)])
            spw = n_neg / max(len(train_pos), 1)
            m = xgb.train(params(spw), xgb.DMatrix(X_tr, label=y_tr), num_boost_round=rounds)
            oof_accum[held_pos] += m.predict(xgb.DMatrix(Xd[held_pos]))
            oof_pos_count[held_pos] += 1
            neg_accum += m.predict(xgb.DMatrix(Xd[neg_dev_idx])) / 5.0
            eval_accum += m.predict(xgb.DMatrix(Xe)) / 5.0

    oof = np.zeros(n, dtype=np.float32)
    oof[pos_idx] = (oof_accum[pos_idx] / np.clip(oof_pos_count[pos_idx], 1, None)).astype(np.float32)
    oof[neg_dev_idx] = (neg_accum / n_bags).astype(np.float32)
    eval_scores = (eval_accum / n_bags).astype(np.float32)
    return oof, eval_scores


def main() -> int:
    import sys
    H = Harness()
    dev, ev, feats = _rich_features()
    # align to harness dev order
    dev = dev.join(pl.DataFrame({"pair_id": H.dev_pair_ids, "_ord": np.arange(len(H.dev_pair_ids))}),
                   on="pair_id", how="left").sort("_ord")
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    y = dev["label"].to_numpy().astype(np.int8)
    dev_ids = dev["pair_id"].to_list()
    Xe = ev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    eval_ids = ev["pair_id"].to_list()
    print(f"rich basis: {len(feats)} feats  dev={len(Xd)} pos={int(y.sum())} eval={len(Xe)} device={xgb_device()}")

    # baseline recap
    print(f"\nGATE reference: consistency 0.61 | E(orig) 0.639 | V30 lineage ~0.79\n")

    # --- sweep ---
    configs = [
        # (label, n_bags, n_neg, depth, rounds)
        ("E2_base_rich",      8, 20000, 4, 350),   # richer feats + bagging vs E
        ("E2_more_bags",     16, 20000, 4, 350),
        ("E2_big_negpool",    8, 40000, 4, 350),
        ("E2_small_negpool",  8, 10000, 4, 350),
        ("E2_depth5",         8, 20000, 5, 350),
        ("E2_depth3",         8, 20000, 3, 350),
        ("E2_rounds600",      8, 20000, 4, 600),
    ]
    if "--quick" in sys.argv:
        configs = configs[:2]

    results = []
    best = None
    for label, nb, nn, dp, rd in configs:
        t = time.time()
        oof, es = _bagged_pu(Xd, y, Xe, feats, n_bags=nb, n_neg=nn, depth=dp, rounds=rd)
        conf = average_precision_score(y, oof)
        eh = weighted_ap(y, oof, H.w)
        auc = roc_auc_score(y, oof)
        results.append({"label": label, "n_bags": nb, "n_neg": nn, "depth": dp, "rounds": rd,
                        "eval_honest": float(eh), "confirmed": float(conf), "auc": float(auc)})
        print(f"  {label:<18} bags={nb} neg={nn} d={dp} r={rd}  "
              f"eval_honest={eh:.4f}  confirmed={conf:.4f}  AUC={auc:.4f}  ({time.time()-t:.1f}s)", flush=True)
        if best is None or eh > best["eval_honest"]:
            best = results[-1]
            best_scores = (oof.copy(), es.copy(), dev_ids, eval_ids)

    print(f"\nBEST: {best['label']}  eval_honest={best['eval_honest']:.4f} "
          f"(E orig 0.639, consistency 0.61, V30 ~0.79)")

    # save best eval scores for a possible submission stage
    oof, es, dids, eids = best_scores
    pl.DataFrame({"pair_id": eids, "E2_score": es}).write_parquet(CACHE / "_E2_best_eval_scores.parquet")
    pl.DataFrame({"pair_id": dids, "E2_oof": oof}).write_parquet(CACHE / "_E2_best_dev_oof.parquet")
    (CACHE / "_E_push_summary.json").write_text(
        json.dumps({"best": best, "all": results}, indent=2), encoding="utf-8")
    print("wrote _E_push_summary.json + _E2_best_*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
