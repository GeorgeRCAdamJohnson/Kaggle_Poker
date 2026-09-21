"""ALGO 1 — ANOMALY DETECTION: risk = 'distance from ordinary', not pos-vs-neg classification.

Every prior model is supervised pos-vs-neg (or pos-vs-eval). The unified per-hand model cratered
(0.16) because per-hand coordination looks like ordinary flashy hands. Anomaly detection attacks
the ACTUAL problem directly: a colluding pair is ABNORMAL relative to the ordinary population. Train
a density/outlier model on the ordinary eval-population pair features (treated as 'normal'), then
score every pair by how anomalous it is. No labels needed for the model -> can't overfit dev labels.

Three anomaly scorers on the stable Layer-2 pair basis:
  * IsolationForest (fit on eval-population sample)
  * Mahalanobis distance from the eval-population mean/cov (elliptic envelope proxy)
  * Autoencoder reconstruction error (torch, fit on eval population)

Judged on the eval-honest harness (importance-weighted PairAP over labeled pairs). This tests
whether 'unusualness vs ordinary' separates colluders — the frame the project never tried.

Run:  python -m anchor_repro.algo_anomaly
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE
from anchor_repro.gen_pair_detector import pair_features

OUT = CACHE / "algos"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def main() -> int:
    H = Harness()
    dev = pair_features(CACHE / "dev_layer2.parquet")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id") and not c.endswith("_sum")]
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    dev_ids = dev["pair_id"].to_list()
    eval_ids = evf["pair_id"].to_list()

    scaler = StandardScaler().fit(Xe)  # normalize on ordinary population
    Xds, Xes = scaler.transform(Xd), scaler.transform(Xe)
    rng = np.random.default_rng(SEED)

    results = {}

    def judge(name, dscore, escore):
        r = H.judge(name, dscore.astype(np.float32), dev_ids, escore.astype(np.float32), eval_ids)
        results[name] = r.eval_honest_ap
        print(f"  {name:<22} eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f} AUC={r.auc:.4f}")

    print("=== ALGO 1: anomaly detection (risk = distance from ordinary) ===\n")

    # --- IsolationForest fit on ordinary eval population ---
    t = time.time()
    iso = IsolationForest(n_estimators=300, max_samples=50000, contamination="auto",
                          random_state=SEED, n_jobs=-1)
    bg = Xes[rng.choice(len(Xes), 80000, replace=False)]
    iso.fit(bg)
    # higher score_samples = more normal; risk = -score (more anomalous = higher risk)
    judge("isoforest", -iso.score_samples(Xds), -iso.score_samples(Xes))
    print(f"    (isoforest {time.time()-t:.0f}s)")

    # --- Mahalanobis distance from ordinary population ---
    mu = Xes.mean(0)
    cov = np.cov(Xes.T) + np.eye(Xes.shape[1]) * 1e-3
    inv = np.linalg.pinv(cov)
    def maha(X):
        d = X - mu
        return np.einsum("ij,jk,ik->i", d, inv, d)
    judge("mahalanobis", maha(Xds), maha(Xes))

    # --- Autoencoder reconstruction error (torch, fit on ordinary population) ---
    t = time.time()
    import torch
    import torch.nn as nn
    dev_t = "cuda" if torch.cuda.is_available() else "cpu"
    D = Xes.shape[1]
    ae = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(),
                       nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, D)).to(dev_t)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr = torch.tensor(Xes[rng.choice(len(Xes), 80000, replace=False)], device=dev_t)
    ae.train()
    for epoch in range(60):
        opt.zero_grad()
        out = ae(Xtr)
        loss = ((out - Xtr) ** 2).mean()
        loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        rd = ((ae(torch.tensor(Xds, device=dev_t)) - torch.tensor(Xds, device=dev_t)) ** 2).mean(1).cpu().numpy()
        re = ((ae(torch.tensor(Xes, device=dev_t)) - torch.tensor(Xes, device=dev_t)) ** 2).mean(1).cpu().numpy()
    judge("autoencoder", rd, re)
    print(f"    (autoencoder {time.time()-t:.0f}s, final recon loss {loss.item():.4f})")

    print(f"\nreference: V30 rank_sum_w50 = 0.9203 eval-honest")
    best = max(results, key=results.get)
    print(f"BEST anomaly scorer: {best} = {results[best]:.4f} (delta vs V30 {results[best]-0.9203:+.4f})")
    (OUT / "_anomaly_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
