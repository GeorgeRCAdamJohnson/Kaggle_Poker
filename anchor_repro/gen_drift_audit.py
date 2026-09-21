"""Audit the pair-level drift measurement: is 0.767 real drift or an artifact?

Suspected bug: the drift gate compares 1,860 dev pairs vs 112,540 eval pairs (60:1
imbalance) AND the dev pairs are a CURATED labeled sample, not a random draw. Both can
inflate the dev-vs-eval AUC without reflecting a distribution shift that would hurt transfer.

Tests:
 1. Balanced drift: subsample eval to 1,860 (match dev n), repeat over seeds.
 2. Proper control: dev NEGATIVES only vs eval (the honest question is whether a NEGATIVE
    dev pair looks like an eval pair; positives SHOULD differ — that's the signal, not drift).
 3. eval-vs-eval null: split eval in half, measure AUC (should be ~0.5; calibrates the metric).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from anchor_repro.gen_pair_detector import pair_features
from anchor_repro.gpu_config import xgb_device

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")


def drift_auc(Xa, Xb, seed=42):
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85,
                            colsample_bytree=0.8, reg_lambda=5, objective="binary:logistic",
                            eval_metric="auc", tree_method="hist", device=xgb_device(), n_jobs=-1,
                            random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, oof))


def main() -> int:
    dev = pair_features(CACHE / "dev_layer2.parquet")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(labels, on="pair_id", how="left")
    feat_cols = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")
                 and not c.endswith("_sum")]

    def mat(frame, mask=None):
        f = frame if mask is None else frame.filter(mask)
        return f.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()

    Xd = mat(dev)
    Xd_neg = mat(dev, pl.col("label") == 0)
    Xd_pos = mat(dev, pl.col("label") == 1)
    Xe = evf.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    rng = np.random.default_rng(42)

    print(f"n dev={len(Xd)} dev_neg={len(Xd_neg)} dev_pos={len(Xd_pos)} eval={len(Xe)}")

    # 1. Original imbalanced (reproduce the 0.767)
    print(f"\n[imbalanced] dev(1860) vs eval(112540)         AUC={drift_auc(Xd, Xe):.4f}")

    # 2. Balanced: subsample eval to match dev n, avg over seeds
    aucs = []
    for s in range(5):
        ei = rng.choice(len(Xe), len(Xd), replace=False)
        aucs.append(drift_auc(Xd, Xe[ei], seed=100 + s))
    print(f"[balanced ] dev(1860) vs eval(1860) x5          AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # 3. dev NEGATIVES vs eval (positives are supposed to differ; negatives are the honest test)
    aucs = []
    for s in range(5):
        ei = rng.choice(len(Xe), len(Xd_neg), replace=False)
        aucs.append(drift_auc(Xd_neg, Xe[ei], seed=200 + s))
    print(f"[neg-only ] dev_NEG(1488) vs eval(1488) x5       AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # 4. eval-vs-eval null (calibration: should be ~0.5)
    aucs = []
    for s in range(5):
        idx = rng.permutation(len(Xe))
        half = len(Xd)
        aucs.append(drift_auc(Xe[idx[:half]], Xe[idx[half:2*half]], seed=300 + s))
    print(f"[eval-null] eval(1860) vs eval(1860) x5          AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # 5. dev POS vs dev NEG (this is the SIGNAL — should be high, that's good, not drift)
    print(f"\n[signal   ] dev_POS vs dev_NEG                  AUC={drift_auc(Xd_pos, Xd_neg):.4f}  (high = good separation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
