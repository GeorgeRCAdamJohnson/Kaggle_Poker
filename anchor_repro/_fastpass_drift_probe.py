"""Diagnose WHICH v3 features drive the dev-vs-eval drift (contract Rule 11: debug, not retreat).

The fast-pass drift gate DISQUALIFIED v3 (AUC 0.761) and v2 (0.762). Before recording
the null, Rule 11 requires understanding the EXECUTION cause: is the separation a
scale/count artifact of a few unnormalized aggregate columns, or a pervasive basis
shift? This probe reports the drift classifier's top feature importances and the
per-feature dev-vs-eval mean gap, so §50 can state the mechanism honestly.

Read-only; prints a small report. No ladder mutation, no submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anchor_repro.own_submission_recipes import _NON_FEATURE_COLUMNS, default_own_cache_locations

POKER_ROOT = Path(__file__).resolve().parents[1]


def probe(ver: str) -> None:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    caches = default_own_cache_locations(POKER_ROOT, ver)
    dev = pd.read_parquet(caches.dev_feats)
    ev = pd.read_parquet(caches.eval_feats)
    feats = [c for c in dev.columns if c in ev.columns and c not in _NON_FEATURE_COLUMNS]

    Xd = dev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xe = ev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X = np.vstack([Xd, Xe])
    y = np.concatenate([np.zeros(len(Xd)), np.ones(len(Xe))]).astype(int)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    clf = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85,
        colsample_bytree=0.8, reg_lambda=5, objective="binary:logistic",
        eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=7,
    )
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
    imp = clf.feature_importances_

    print(f"\n=== {ver} drift probe (holdout AUC={auc:.4f}) ===")
    order = np.argsort(imp)[::-1][:12]
    print(f"{'feature':28s} {'imp':>7s} {'dev_mean':>12s} {'eval_mean':>12s} {'ratio':>8s}")
    for i in order:
        dm = float(np.nanmean(dev[feats[i]].replace([np.inf, -np.inf], np.nan)))
        em = float(np.nanmean(ev[feats[i]].replace([np.inf, -np.inf], np.nan)))
        ratio = (em / dm) if dm not in (0.0,) else float("nan")
        print(f"{feats[i]:28s} {imp[i]:7.4f} {dm:12.4f} {em:12.4f} {ratio:8.3f}")

    # AUC using ONLY the single most-drifting feature, to quantify how much one scale
    # column carries.
    top = int(order[0])
    single = roc_auc_score(
        yva, XGBClassifier(
            n_estimators=100, max_depth=2, tree_method="hist", n_jobs=-1, random_state=7
        ).fit(Xtr[:, [top]], ytr).predict_proba(Xva[:, [top]])[:, 1]
    )
    print(f"single-feature AUC ({feats[top]}): {single:.4f}")


if __name__ == "__main__":
    probe("v3")
    probe("v2")


def probe_drop_counts(ver: str, drop: list) -> None:
    """Diagnostic (contract Rule 11): does the block still drift AFTER dropping the
    count/scale features that dominate the separation? Reports OOF AUC on the reduced
    block. This is a DIAGNOSTIC of the mechanism, NOT a re-engineered ladder point —
    the recipe as-submitted fits on the full block, so this cannot revive the point;
    it only tells us whether the drift is the count artifact alone or pervasive.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    caches = default_own_cache_locations(POKER_ROOT, ver)
    dev = pd.read_parquet(caches.dev_feats)
    ev = pd.read_parquet(caches.eval_feats)
    feats = [
        c for c in dev.columns
        if c in ev.columns and c not in _NON_FEATURE_COLUMNS and c not in drop
    ]
    Xd = dev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xe = ev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X = np.vstack([Xd, Xe]); y = np.concatenate([np.zeros(len(Xd)), np.ones(len(Xe))]).astype(int)
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
        oof[va] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85,
            colsample_bytree=0.8, reg_lambda=5, objective="binary:logistic",
            eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=7,
        ).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    print(f"[drop-counts] {ver} minus {drop}: OOF AUC={roc_auc_score(y, oof):.4f} "
          f"({len(feats)} feats)")


if __name__ != "__main__":
    pass
