"""Why did +edge not help? Redundancy vs complementarity diagnosis (data, not opinion).

Naive concat gave -0.0022. Either the edge signal is redundant with their preflop loose, or the model
isn't using it. Check:
  1. rank-corr of joint_surp_excess / total_loose_min_top3 with their loose_pf_sum (redundancy).
  2. Among pairs the BASELINE ranks WRONG (true positives it puts low), do the edge features rank them
     high? i.e. residual analysis: does edge separate positives WITHIN the baseline's error set.
  3. A blend: baseline OOF rank + edge-feature rank -> does a rank blend beat baseline (complementary
     even if concat didn't, because trees drowned the 14 edge cols among 221)?
Run:  python -m anchor_repro.policy_edge_diagnose
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

STEP3 = Path("outputs/poker_collusion/hosen42_step3"); EDGE = STEP3 / "edge"

def main():
    dev = pd.read_parquet(STEP3 / "dev_pair_features.parquet")
    ede = pd.read_parquet(EDGE.parent / "edge" / "per_hand_surprise.parquet") if False else None
    # reload edge pair feats by recomputing? they were merged in train; instead reload oof + rebuild edge quickly
    from anchor_repro.policy_edge_train import _pair_edge_features, EDGE_FEATS
    dedge = _pair_edge_features(STEP3 / "dev_pair_hands.parquet", "development").to_pandas()
    dev = dev.merge(dedge, on="pair_id", how="left")
    for c in EDGE_FEATS: dev[c] = dev[c].astype("float32").fillna(0)
    y = dev["label"].fillna(0).astype(int).to_numpy()
    labm = dev["is_labeled"].to_numpy().astype(bool)

    oof_base = np.load(EDGE / "oof_base.npy")

    # 1. redundancy
    print("=== rank-corr of edge feats with baseline OOF risk and with loose_pf_sum ===")
    lps = dev["loose_pf_sum"].to_numpy() if "loose_pf_sum" in dev.columns else None
    for c in ["joint_surp_excess", "total_loose_min_top3", "loose_corr", "post_loose_min_mean"]:
        v = dev[c].to_numpy()
        r_oof = spearmanr(v, oof_base).correlation
        r_lps = spearmanr(v, lps).correlation if lps is not None else float("nan")
        print(f"  {c:24} corr(base_oof)={r_oof:.3f}  corr(loose_pf_sum)={r_lps:.3f}")

    # 2. rank blend on eval-mirrored population
    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    print("\n=== rank blend: baseline OOF + edge feature (eval-mirrored PairAP, pos vs ALL) ===")
    base_ap = average_precision_score(y, oof_base)
    print(f"  baseline OOF: {base_ap:.4f}")
    for c in ["joint_surp_excess", "total_loose_min_top3"]:
        for w in [0.1, 0.2, 0.3, 0.5]:
            blend = (1 - w) * pr(oof_base) + w * pr(dev[c].to_numpy())
            print(f"  base + {w:.1f}*{c}: {average_precision_score(y, blend):.4f}")

    # 3. residual: among true positives the baseline ranks BELOW its median positive, does edge rank them high?
    pos = labm & (y == 1)
    base_pos_rank = pd.Series(oof_base).rank(pct=True).to_numpy()
    missed = pos & (base_pos_rank < np.quantile(base_pos_rank[pos], 0.5))  # positives the baseline underranks
    print(f"\n=== residual: {int(missed.sum())} underranked positives — does edge rank them above negatives? ===")
    neg = y == 0
    for c in ["joint_surp_excess", "total_loose_min_top3", "loose_corr"]:
        v = dev[c].to_numpy()
        sel = missed | neg
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(np.concatenate([np.ones(missed.sum()), np.zeros(neg.sum())]),
                          np.concatenate([v[missed], v[neg]]))
        print(f"  {c:24} AUC(underranked-pos vs neg)={max(a,1-a):.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
