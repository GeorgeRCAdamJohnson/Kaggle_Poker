"""GPU per-hand planted-evidence classifier (generator recovery, task 3).

Trains an xgboost (device=cuda) classifier of ``is_planted`` on the per-(pair,hand)
generator features, with TABLE-GROUPED 5-fold CV (a pair's table never spans train/val)
to avoid leakage. Measures how well the model recovers the exact planted evidence hands:
 - per-hand planted-AP (ranking planted vs non-planted within positive pairs),
 - EvidenceMAP@5 proxy (top-5 planted hands per positive pair vs the truth),
 - hand-level recall@k.
Also runs the adversarial drift gate on the feature block (dev vs eval) so we know the
representation transports.

BUILD + MEASURE ONLY. No submission. GPU via anchor_repro.gpu_config.

Run:  python -m anchor_repro.generator_perhand
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.own_submission_recipes import adversarial_drift_auc_matrix

D = Path("data/poker")
OUT = Path("outputs/poker_collusion/generator_hunt")
SEED = 42
N_SPLITS = 5


def _load_dev() -> tuple[pd.DataFrame, list[str]]:
    cols = json.loads((OUT / "generator_feature_columns.json").read_text())
    dev = pl.read_parquet(OUT / "dev_pairhand_features.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label", "behavior_family"])
    hands = pl.scan_parquet(D / "hands.parquet").select(["hand_id", "table_id"]).collect()
    dev = dev.join(labels, on="pair_id", how="left").join(hands, on="hand_id", how="left")
    # Restrict training candidates to positive pairs (planted hands only live there).
    dev = dev.filter(pl.col("label") == 1)
    return dev.to_pandas(), cols


def _evidence_map5(pair_ids: np.ndarray, is_planted: np.ndarray, scores: np.ndarray) -> float:
    """MAP@5 over positive pairs: rank hands by score desc, denom = min(#planted,5)."""
    frame = pd.DataFrame({"pair_id": pair_ids, "planted": is_planted, "score": scores})
    aps = []
    for _, g in frame.groupby("pair_id", sort=False):
        rel = int(g["planted"].sum())
        if rel == 0:
            continue
        top = g.sort_values("score", ascending=False, kind="mergesort").head(5)["planted"].to_numpy()
        hits = 0
        psum = 0.0
        for r, h in enumerate(top, start=1):
            if h:
                hits += 1
                psum += hits / r
        aps.append(psum / min(rel, 5))
    return float(np.mean(aps)) if aps else 0.0


def _recall_at_k(pair_ids, is_planted, scores, k=5) -> float:
    frame = pd.DataFrame({"pair_id": pair_ids, "planted": is_planted, "score": scores})
    recs = []
    for _, g in frame.groupby("pair_id", sort=False):
        rel = int(g["planted"].sum())
        if rel == 0:
            continue
        top = g.sort_values("score", ascending=False, kind="mergesort").head(k)["planted"].sum()
        recs.append(top / min(rel, k))
    return float(np.mean(recs)) if recs else 0.0


def main() -> int:
    from sklearn.metrics import average_precision_score

    dev, feat_cols = _load_dev()
    print(f"dev positive-pair hands={len(dev):,} planted={int(dev['is_planted'].sum())} "
          f"pairs={dev['pair_id'].nunique()} device={xgb_device()}")

    X = dev[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["is_planted"].to_numpy().astype(np.int8)
    groups = dev["table_id"].to_numpy()
    pair_ids = dev["pair_id"].to_numpy()
    fam = dev["behavior_family"].to_numpy()

    # Table-grouped, planted-stratified CV.
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(dev), dtype=np.float32)
    params = xgb_params(
        objective="binary:logistic", eval_metric="aucpr",
        max_depth=6, eta=0.05, subsample=0.85, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=5.0,
    )
    t = time.time()
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        assert not (set(groups[tr]) & set(groups[va])), f"fold {fold} table leak"
        # class imbalance handling
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        dtr = xgb.DMatrix(X.iloc[tr], label=y[tr])
        dva = xgb.DMatrix(X.iloc[va], label=y[va])
        model = xgb.train(
            {**params, "scale_pos_weight": float(spw)}, dtr,
            num_boost_round=400, evals=[(dva, "va")],
            early_stopping_rounds=40, verbose_eval=False,
        )
        oof[va] = model.predict(dva)
        print(f"  fold {fold+1}/{N_SPLITS}: planted-AP={average_precision_score(y[va], oof[va]):.4f} "
              f"best_iter={model.best_iteration}", flush=True)
    print(f"trained in {time.time()-t:.1f}s")

    overall_ap = average_precision_score(y, oof)
    emap5 = _evidence_map5(pair_ids, y, oof)
    rec5 = _recall_at_k(pair_ids, y, oof, 5)
    print(f"\n=== PER-HAND OOF (within positive pairs) ===")
    print(f"planted-AP (global)     = {overall_ap:.4f}  (base rate {y.mean():.4f})")
    print(f"EvidenceMAP@5 (proxy)   = {emap5:.4f}   [V30 evidence ~0.5067]")
    print(f"recall@5                = {rec5:.4f}")

    # Per-family EvidenceMAP@5.
    print("\nper-family EvidenceMAP@5:")
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        m = fam == f
        e = _evidence_map5(pair_ids[m], y[m], oof[m])
        print(f"  {f:24s} {e:.4f}  (pairs={dev.loc[m,'pair_id'].nunique()})")

    # Drift gate on the feature block (dev positive hands vs eval hands sample).
    print("\n=== DRIFT GATE (dev vs eval feature block) ===")
    ev = pl.read_parquet(OUT / "eval_pairhand_features.parquet", columns=feat_cols)
    ev_s = ev.sample(n=min(200000, ev.height), seed=SEED).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    dv_s = X.sample(n=min(200000, len(X)), random_state=SEED)
    res = adversarial_drift_auc_matrix(dv_s.to_numpy(), ev_s.to_numpy(), block="generator_perhand", seed=SEED)
    print(f"drift AUC = {res.auc:.4f} (threshold {res.threshold}; pass={res.passed})")

    # Persist OOF for the pair-level step.
    outdf = dev[["pair_id", "hand_id", "is_planted", "behavior_family"]].copy()
    outdf["planted_score"] = oof
    outdf.to_parquet(OUT / "dev_perhand_oof.parquet", index=False)
    summary = {
        "planted_ap": float(overall_ap), "evidence_map5": float(emap5), "recall5": float(rec5),
        "drift_auc": float(res.auc), "drift_pass": bool(res.passed),
        "device": xgb_device(), "n_features": len(feat_cols),
        "v30_evidence_map5_reference": 0.5067,
    }
    (OUT / "_perhand_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'dev_perhand_oof.parquet'} and _perhand_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
