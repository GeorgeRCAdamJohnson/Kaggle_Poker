"""The crux: why do 0.90-AUC, low-correlation edge feats HURT? Check precision-at-top-k, not AUC.

If edge feats have high AUC but promote false positives at the TOP of the ranking (where AP lives),
that explains concat/blend/stack all losing. Compare, on the eval-mirrored dev population, the
precision@k of: baseline OOF vs joint_surp_excess vs total_loose_min_top3, for k = 200,400,800.
Also: of the baseline's top-400, how many are true pos; of the edge feat's top-400, how many; and the
OVERLAP. If edge's top-400 is mostly different pairs that are NEGATIVE, it's a false-positive machine
at the top despite high AUC.
Run:  python -m anchor_repro.policy_edge_precision
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
STEP3 = Path("outputs/poker_collusion/hosen42_step3"); EDGE = STEP3 / "edge"

def main():
    from anchor_repro.policy_edge_train import _pair_edge_features
    dev = pd.read_parquet(STEP3 / "dev_pair_features.parquet")
    de = _pair_edge_features(STEP3 / "dev_pair_hands.parquet", "development").to_pandas()
    dev = dev.merge(de, on="pair_id", how="left")
    y = dev["label"].fillna(0).astype(int).to_numpy()
    oof_base = np.load(EDGE / "oof_base.npy")
    npos = int(y.sum())
    print(f"total pos among {len(y):,} eligible pairs = {npos}")

    def prec_at_k(score, ks=(200, 400, 800, npos)):
        order = np.argsort(-score)
        return {k: int(y[order[:k]].sum()) for k in ks}

    for name, s in [("baseline_oof", oof_base), ("joint_surp_excess", dev["joint_surp_excess"].fillna(0).to_numpy()),
                    ("total_loose_min_top3", dev["total_loose_min_top3"].fillna(0).to_numpy())]:
        pk = prec_at_k(s)
        print(f"  {name:22} true-pos in top-k: " + "  ".join(f"@{k}={v}" for k, v in pk.items()))

    # overlap of top-400 sets
    ob = set(np.argsort(-oof_base)[:400])
    oe = set(np.argsort(-dev["joint_surp_excess"].fillna(0).to_numpy())[:400])
    print(f"\n  baseline top-400 true-pos: {y[list(ob)].sum()}")
    print(f"  joint_surp top-400 true-pos: {y[list(oe)].sum()}")
    print(f"  overlap of the two top-400 sets: {len(ob & oe)}")
    # of joint_surp's top-400, how many are NEG that baseline correctly ranks low
    oe_only = np.array(list(oe - ob))
    print(f"  joint_surp top-400 NOT in baseline top-400: {len(oe_only)}; of those true-pos: {y[oe_only].sum()} (rest are false positives it would inject)")
    # do any of joint_surp's top-400 rescue positives baseline MISSES from its top-400?
    base_top_npos = set(np.argsort(-oof_base)[:npos])
    missed_pos = [i for i in range(len(y)) if y[i]==1 and i not in base_top_npos]
    rescued = [i for i in missed_pos if i in oe]
    print(f"  positives baseline misses from top-{npos}: {len(missed_pos)}; of those in joint_surp top-400: {len(rescued)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
