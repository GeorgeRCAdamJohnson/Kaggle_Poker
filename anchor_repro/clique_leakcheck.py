"""Leak-check the clique signal before submitting: is eval-honest 0.83 REAL or leaking V30 risk?

The clique feature = mean/max V30-risk of the OTHER pairs a player is in. Risk: it may trivially
recover V30's own risk (a pair's neighbors are correlated with itself) OR leak dev label structure
(693 colluders in overlapping positive pairs). Two checks:

 1. CORRELATION with V30 risk: if the clique eval score is ~identical to V30 (corr>0.95), it adds
    nothing (a blend is a no-op) — not a breakthrough, just V30 relabeled.
 2. LEAVE-ONE-OUT purity: recompute clique neighbor features EXCLUDING each pair's own risk (a pair
    contributes 0 to its own members' neighbor stats). If the signal survives LOO, it's structural;
    if it collapses, it was self-leak.

Also report how DECORRELATED the clique eval score is from V30 (a real add must be decorrelated AND
high; a leak is high-but-redundant).
Run:  python -m anchor_repro.clique_leakcheck
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D


def _loo_clique(pairs: pl.DataFrame, risk: dict):
    """Leave-one-out neighbor risk: for pair p with members u,v, neighbor stats EXCLUDE p itself."""
    # player -> list of (pair_id, risk)
    from collections import defaultdict
    pmap = defaultdict(list)
    for r in pairs.iter_rows(named=True):
        pmap[r["player_1"]].append((r["pair_id"], risk.get(r["pair_id"], 0.0)))
        pmap[r["player_2"]].append((r["pair_id"], risk.get(r["pair_id"], 0.0)))
    out = {}
    for r in pairs.iter_rows(named=True):
        pid, u, v = r["pair_id"], r["player_1"], r["player_2"]
        # neighbor risks of u and v EXCLUDING this pair
        nu = [rk for (p, rk) in pmap[u] if p != pid]
        nv = [rk for (p, rk) in pmap[v] if p != pid]
        neigh = nu + nv
        out[pid] = np.mean(neigh) if neigh else 0.0
    return out


def main() -> int:
    H = Harness()
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    seq = pl.read_parquet("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
    dev_risk = dict(zip(seq["pair_id"].to_list(), seq["rank_sum_w50"].to_numpy()))
    v30csv = pl.read_csv("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
    eval_risk = dict(zip(v30csv["pair_id"].to_list(), v30csv["risk_score"].to_numpy()))

    # LOO clique neighbor-mean
    dev_loo = _loo_clique(labels, dev_risk)
    eval_loo = _loo_clique(eval_pairs, eval_risk)
    y = labels["label"].to_numpy()
    dloo = np.array([dev_loo[p] for p in labels["pair_id"].to_list()])
    print(f"LOO clique_neighbor_mean univariate AUC (dev): {max(roc_auc_score(y,dloo),1-roc_auc_score(y,dloo)):.4f}")

    # eval-honest of the LOO clique feature ALONE
    wmap = dict(zip(H.dev_pair_ids, H.w))
    w = np.array([wmap[p] for p in labels["pair_id"].to_list()])
    eh_loo = weighted_ap(y, dloo, w)
    print(f"LOO clique feature eval-honest (dev-weighted): {eh_loo:.4f}")

    # correlation of eval clique score with V30 eval risk (redundancy check)
    ecl = np.array([eval_loo[p] for p in eval_pairs["pair_id"].to_list()])
    ev30 = np.array([eval_risk.get(p, 0.0) for p in eval_pairs["pair_id"].to_list()])
    corr = np.corrcoef(pd.Series(ecl).rank(), pd.Series(ev30).rank())[0, 1]
    print(f"eval clique-LOO vs V30 rank correlation: {corr:.4f}")
    print("  (>0.9 = redundant with V30, blend is no-op; <0.7 = decorrelated, could genuinely add)")

    # does V30 + LOO-clique blend beat V30 on the dev eval-honest?
    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    v30d = np.array([dev_risk.get(p, 0.0) for p in labels["pair_id"].to_list()])
    v30_eh = weighted_ap(y, v30d, w)
    print(f"\nV30 dev eval-honest: {v30_eh:.4f}")
    for bw in [0.1, 0.2, 0.3, 0.5]:
        blend = (1-bw)*pr(v30d) + bw*pr(dloo)
        print(f"  V30 + {bw:.1f}*clique-LOO: eval-honest={weighted_ap(y, blend, w):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
