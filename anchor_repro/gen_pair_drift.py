"""Per-column drift diagnosis of the pair-level consistency features (dev vs eval).

The joint block drifts (AUC 0.76). Find WHICH columns drift so we can keep a transport-clean
subset and re-measure whether the surviving signal still separates positives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from anchor_repro.gen_pair_detector import pair_features, SIG_COLS

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")


def solo_drift(dv, ev):
    """Single-feature dev-vs-eval AUC (subsample eval to match)."""
    n = min(len(dv), len(ev), 100000)
    rng = np.random.default_rng(42)
    di = rng.choice(len(dv), min(len(dv), n), replace=False)
    ei = rng.choice(len(ev), n, replace=False)
    x = np.concatenate([dv[di], ev[ei]])
    y = np.concatenate([np.zeros(len(di)), np.ones(len(ei))])
    # simple threshold AUC via rank
    return max(roc_auc_score(y, x), roc_auc_score(y, -x))


def main() -> int:
    dev = pair_features(CACHE / "dev_layer2.parquet")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(labels, on="pair_id", how="left")
    y = dev["label"].to_numpy().astype(np.int8)
    feat_cols = [c for c in dev.columns if c not in ("pair_id", "label", "table_id", "n_hands")]

    rows = []
    for c in feat_cols:
        dv = dev[c].fill_null(0).to_numpy().astype(float)
        ev = evf[c].fill_null(0).to_numpy().astype(float)
        d = solo_drift(dv, ev)
        ap = average_precision_score(y, dv) if np.unique(dv).size > 1 else 0.0
        rows.append((c, d, ap))
    rows.sort(key=lambda r: r[1])
    print(f"{'feature':28s} {'driftAUC':>9} {'posAP':>8}  (clean if drift<0.65)")
    clean = []
    for c, d, ap in rows:
        flag = "" if d < 0.65 else "  DRIFT"
        if d < 0.65:
            clean.append(c)
        print(f"  {c:26s} {d:9.4f} {ap:8.4f}{flag}")
    print(f"\nclean features: {len(clean)}/{len(feat_cols)}")

    # Re-measure separation using ONLY clean features (5-fold logistic, quick).
    if clean:
        X = dev.select(clean).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        Xn = (X - X.mean()) / (X.std() + 1e-6)
        oof = np.zeros(len(y))
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for tr, va in skf.split(Xn, y):
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(Xn.iloc[tr], y[tr])
            oof[va] = lr.predict_proba(Xn.iloc[va])[:, 1]
        print(f"\nCLEAN-only pair detector: confirmed PairAP={average_precision_score(y, oof):.4f} "
              f"AUC={roc_auc_score(y, oof):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
