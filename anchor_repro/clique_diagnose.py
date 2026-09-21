"""Was the clique collapse a REAL self-leak, or a BUG in the leave-one-out? Diagnose directly.

Three hypotheses for why non-LOO clique = 0.83 but LOO = 0.55:
  (1) real self-leak: a pair's own risk leaked into its neighbor features; LOO removes it correctly.
  (2) LOO BUG: the LOO emptied most neighbor lists (degenerate 0.0), artificially killing signal.
  (3) genuine noise.

Checks:
  * How many pairs have a NON-EMPTY leave-one-out neighbor set? (if most are empty -> LOO is
    degenerate because most players are in only 1 pair -> not a fair test, and the 0.83 was
    structural-degeneracy, not signal).
  * Player degree distribution: how many pairs does each player belong to? If ~1, the whole clique
    idea is moot on eval (no cliques to propagate).
  * Non-LOO vs LOO feature correlation and the fraction of the non-LOO value that IS the self term.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from anchor_repro.eval_honest_harness import D

def main() -> int:
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    seq = pl.read_parquet("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
    dev_risk = dict(zip(seq["pair_id"].to_list(), seq["rank_sum_w50"].to_numpy()))
    v30csv = pl.read_csv("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
    eval_risk = dict(zip(v30csv["pair_id"].to_list(), v30csv["risk_score"].to_numpy()))

    def analyze(pairs, risk, tag):
        pmap = defaultdict(list)
        for r in pairs.iter_rows(named=True):
            pmap[r["player_1"]].append((r["pair_id"], risk.get(r["pair_id"], 0.0)))
            pmap[r["player_2"]].append((r["pair_id"], risk.get(r["pair_id"], 0.0)))
        degs = [len(v) for v in pmap.values()]
        print(f"\n[{tag}] players={len(pmap)}  pairs={pairs.height}")
        print(f"  player degree (pairs/player): mean={np.mean(degs):.2f} median={np.median(degs):.0f} "
              f"max={max(degs)} frac_deg1={np.mean([d==1 for d in degs]):.3f}")
        # for each pair, size of LOO neighbor set
        empty = 0; nonempty = 0; selfshare = []
        loo_vals = {}; nonloo_vals = {}
        for r in pairs.iter_rows(named=True):
            pid, u, v = r["pair_id"], r["player_1"], r["player_2"]
            nu = [rk for (p, rk) in pmap[u] if p != pid]
            nv = [rk for (p, rk) in pmap[v] if p != pid]
            neigh = nu + nv
            alln = [rk for (p, rk) in pmap[u]] + [rk for (p, rk) in pmap[v]]
            loo_vals[pid] = np.mean(neigh) if neigh else 0.0
            nonloo_vals[pid] = np.mean(alln) if alln else 0.0
            if neigh: nonempty += 1
            else: empty += 1
        print(f"  pairs with NON-EMPTY leave-one-out neighbor set: {nonempty}/{pairs.height} "
              f"({100*nonempty/pairs.height:.1f}%)  empty: {empty}")
        return loo_vals, nonloo_vals

    dloo, dnon = analyze(labels, dev_risk, "DEV")
    eloo, enon = analyze(eval_pairs, eval_risk, "EVAL")

    y = labels["label"].to_numpy()
    dl = np.array([dloo[p] for p in labels["pair_id"].to_list()])
    dn = np.array([dnon[p] for p in labels["pair_id"].to_list()])
    print(f"\nDEV non-LOO clique AUC = {max(roc_auc_score(y,dn),1-roc_auc_score(y,dn)):.4f}")
    print(f"DEV LOO    clique AUC = {max(roc_auc_score(y,dl),1-roc_auc_score(y,dl)):.4f}")
    print(f"corr(non-LOO, LOO) = {np.corrcoef(dn, dl)[0,1]:.4f}")
    # among NON-empty-neighbor pairs only (fair LOO test)
    mask = np.array([len([1 for (p,_) in [] ]) for _ in labels["pair_id"].to_list()])  # placeholder
    # recompute mask properly
    pmap = defaultdict(list)
    for r in labels.iter_rows(named=True):
        pmap[r["player_1"]].append(r["pair_id"]); pmap[r["player_2"]].append(r["pair_id"])
    has_neigh = np.array([ (len([1 for p in pmap[r['player_1']] if p!=r['pair_id']]) + len([1 for p in pmap[r['player_2']] if p!=r['pair_id']])) > 0
                           for r in labels.iter_rows(named=True)])
    print(f"\nDEV pairs with a real LOO neighbor: {has_neigh.sum()}/{len(has_neigh)}")
    if has_neigh.sum() > 20 and y[has_neigh].sum() > 3:
        print(f"  LOO clique AUC on THOSE pairs only: {max(roc_auc_score(y[has_neigh], dl[has_neigh]), 1-roc_auc_score(y[has_neigh], dl[has_neigh])):.4f}")
        print(f"  positive rate among them: {y[has_neigh].mean():.3f} vs overall {y.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
