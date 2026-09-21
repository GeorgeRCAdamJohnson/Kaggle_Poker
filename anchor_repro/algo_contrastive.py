"""ALGO 2 — METRIC LEARNING / CONTRASTIVE embedding.

Learn an embedding where colluding pairs cluster together and AWAY from ordinary pairs, then
risk = how close a pair sits to the positive cluster (or far from the ordinary centroid). This
builds the 'distance from ordinary' signal directly, as a learned metric rather than raw features.

Setup: supervised contrastive on confirmed positives (anchor class) vs a large sample of ordinary
eval-population pairs (negative class). Encoder = small MLP -> L2-normalized embedding. Loss pulls
positives together, pushes them from ordinary. Risk = cosine similarity to the positive centroid
(computed OOF/held-out to avoid leakage).

Judged eval-honest. Confirmed-anchored -> harness trustworthy.
Run:  python -m anchor_repro.algo_contrastive
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.eval_honest_harness import Harness, CACHE, D
from anchor_repro.gen_pair_detector import pair_features

OUT = CACHE / "algos"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)


class Encoder(nn.Module):
    def __init__(self, d_in, d_emb=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, d_emb))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def _train_encoder(Xpos, Xbg, dev_t, epochs=200):
    enc = Encoder(Xpos.shape[1]).to(dev_t)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    Xp = torch.tensor(Xpos, device=dev_t); Xb = torch.tensor(Xbg, device=dev_t)
    for ep in range(epochs):
        opt.zero_grad()
        ep_pos = enc(Xp); ep_bg = enc(Xb[torch.randint(len(Xb), (min(4000, len(Xb)),), device=dev_t)])
        # supervised contrastive: positives cohere, separate from background
        pos_centroid = F.normalize(ep_pos.mean(0, keepdim=True), dim=1)
        pull = (1 - (ep_pos @ pos_centroid.T)).mean()          # positives -> their centroid
        push = F.relu((ep_bg @ pos_centroid.T) - 0.0).mean()   # background away from pos centroid
        loss = pull + push
        loss.backward(); opt.step()
    return enc, loss.item()


def main() -> int:
    H = Harness()
    dev = pair_features(CACHE / "dev_layer2.parquet")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id") and not c.endswith("_sum")]
    lab = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(lab, on="pair_id", how="left")
    # table for grouped OOF
    from anchor_repro.gen_pair_detector import CACHE as GC
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    ptbl = (pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
            .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
            .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
            .select(["pair_id", "table_id"]).collect())
    dev = dev.join(ptbl, on="pair_id", how="left")

    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    y = dev["label"].to_numpy().astype(np.int8)
    groups = dev["table_id"].to_numpy()
    dev_ids = dev["pair_id"].to_list(); eval_ids = evf["pair_id"].to_list()
    dev_t = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler().fit(Xe)
    Xds, Xes = scaler.transform(Xd).astype(np.float32), scaler.transform(Xe).astype(np.float32)
    rng = np.random.default_rng(SEED)

    print("=== ALGO 2: contrastive embedding (risk = similarity to positive cluster) ===\n")
    # OOF on dev: train encoder on train-fold positives + ordinary bg, score held-out dev by
    # cosine-sim to the train positive centroid.
    oof = np.zeros(len(y), np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    t = time.time()
    for tr, va in sgkf.split(Xds, y, groups):
        pos_tr = Xds[tr][y[tr] == 1]
        bg = Xes[rng.choice(len(Xes), 20000, replace=False)]
        enc, _ = _train_encoder(pos_tr, bg, dev_t)
        enc.eval()
        with torch.no_grad():
            emb_pos = enc(torch.tensor(pos_tr, device=dev_t))
            cen = F.normalize(emb_pos.mean(0, keepdim=True), dim=1)
            emb_va = enc(torch.tensor(Xds[va], device=dev_t))
            oof[va] = (emb_va @ cen.T).squeeze(1).cpu().numpy()
    # full model for eval
    pos_all = Xds[y == 1]
    bg = Xes[rng.choice(len(Xes), 20000, replace=False)]
    enc, final = _train_encoder(pos_all, bg, dev_t)
    enc.eval()
    with torch.no_grad():
        cen = F.normalize(enc(torch.tensor(pos_all, device=dev_t)).mean(0, keepdim=True), dim=1)
        escore = (enc(torch.tensor(Xes, device=dev_t)) @ cen.T).squeeze(1).cpu().numpy()
    r = H.judge("contrastive", oof, dev_ids, escore, eval_ids)
    print(f"  contrastive  eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f} AUC={r.auc:.4f} ({time.time()-t:.0f}s)")
    print(f"  vs V30 0.9203 -> delta {r.eval_honest_ap-0.9203:+.4f}")
    (OUT / "_contrastive_summary.json").write_text(json.dumps(
        {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap}, indent=2), encoding="utf-8")
    pl.DataFrame({"pair_id": eval_ids, "risk_score": ((escore - escore.min())/(escore.max()-escore.min())).astype(np.float32)}).write_parquet(OUT / "contrastive_eval_risk.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
