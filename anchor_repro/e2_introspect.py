"""Interrogate E2's OWN score structure — what refinements does E2 ITSELF ask for?

Not "does V30's structure help E2" (§118, answered). Instead: look INSIDE E2's scores.
Where is E2 strong/weak, what does it miss, is 0.65 uniform mediocrity or bimodal, is it a
ranking or calibration problem, and where does it DISAGREE with V30 (complementary signal)?

Uses cached E2 best dev OOF + eval scores, joined to labels/family/V30. Reports diagnostics
that POINT AT a refinement, rather than assuming one.

Run:  python -m anchor_repro.e2_introspect
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE
from anchor_repro.gen_pair_detector import pair_features

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
D = Path("data/poker")


def _prank(v):
    return pd.Series(np.asarray(v, np.float64)).rank(pct=True).to_numpy()


def main() -> int:
    H = Harness()
    e2 = pl.read_parquet(CACHE / "_E2_best_dev_oof.parquet")
    oof = pl.read_parquet(SEQ / "oof_rows.parquet").select(
        ["pair_id", "known", "y", "behavior_y", "baseline_risk", "rank_sum_w50"])
    df = oof.join(e2, on="pair_id", how="left").filter(pl.col("known"))
    y = df["y"].to_numpy().astype(int)
    fam = df["behavior_y"].to_numpy().astype(int)
    e2s = df["E2_oof"].to_numpy()
    v30 = df["rank_sum_w50"].to_numpy()
    w = H.w  # eval-honest weights aligned to H.dev_pair_ids; realign
    wmap = dict(zip(H.dev_pair_ids, H.w))
    wv = np.array([wmap[p] for p in df["pair_id"].to_list()])

    print("=== E2 INTROSPECTION (confirmed labeled set, n=%d, pos=%d) ===" % (len(y), y.sum()))
    print(f"E2 confirmed AP={average_precision_score(y,e2s):.4f}  AUC={roc_auc_score(y,e2s):.4f}  "
          f"eval-honest={weighted_ap(y,e2s,wv):.4f}")
    print(f"V30 confirmed AP={average_precision_score(y,v30):.4f}  AUC={roc_auc_score(y,v30):.4f}  "
          f"eval-honest={weighted_ap(y,v30,wv):.4f}")

    # 1. Per-FAMILY AP: is E2 uniformly weak or blind to a specific family?
    print("\n--- per-family AP (does E2 miss a whole family?) ---")
    for fid, fname in [(1, "directed"), (2, "soft"), (3, "isolation")]:
        yf = (fam == fid).astype(int)
        if yf.sum() == 0:
            continue
        # AP of THIS family's positives vs all negatives (fam==0), ignoring other families
        keep = (fam == 0) | (fam == fid)
        e2_ap = average_precision_score(yf[keep], e2s[keep])
        v30_ap = average_precision_score(yf[keep], v30[keep])
        print(f"  {fname:<10} (n={int(yf.sum())}): E2 AP={e2_ap:.4f}  V30 AP={v30_ap:.4f}  "
              f"gap={e2_ap-v30_ap:+.4f}")

    # 2. Is 0.65 uniform or BIMODAL? Look at positive-score distribution.
    pos_scores = e2s[y == 1]
    neg_scores = e2s[y == 0]
    print("\n--- positive score distribution (uniform-weak vs bimodal blind-spot) ---")
    print(f"  positives: p10={np.quantile(pos_scores,0.1):.3f} p50={np.median(pos_scores):.3f} "
          f"p90={np.quantile(pos_scores,0.9):.3f}")
    print(f"  negatives: p50={np.median(neg_scores):.3f} p90={np.quantile(neg_scores,0.9):.3f} "
          f"p99={np.quantile(neg_scores,0.99):.3f}")
    # fraction of positives scored BELOW the negative median = E2's blind positives
    blind = float((pos_scores < np.median(neg_scores)).mean())
    print(f"  positives below neg-median (E2 BLIND SPOT fraction): {blind:.3f}")

    # 3. WHERE E2 and V30 DISAGREE on positives — complementary signal?
    print("\n--- E2 vs V30 complementarity on positives ---")
    e2r, v30r = _prank(e2s), _prank(v30)
    pos = y == 1
    # positives E2 ranks well but V30 ranks poorly (E2's unique catches)
    e2_catches = pos & (e2r > 0.8) & (v30r < 0.5)
    v30_catches = pos & (v30r > 0.8) & (e2r < 0.5)
    both_miss = pos & (e2r < 0.5) & (v30r < 0.5)
    print(f"  positives E2 catches but V30 misses: {int(e2_catches.sum())}")
    print(f"  positives V30 catches but E2 misses: {int(v30_catches.sum())}")
    print(f"  positives BOTH miss (hard core):     {int(both_miss.sum())}")
    if e2_catches.sum() > 0:
        print(f"    -> E2 has UNIQUE signal on {int(e2_catches.sum())} positives V30 can't see "
              f"(families: {np.bincount(fam[e2_catches], minlength=4)[1:]})")
    spearman = np.corrcoef(e2r[pos], v30r[pos])[0, 1]
    print(f"  E2-V30 rank corr on positives: {spearman:.3f} "
          f"({'decorrelated -> blend may add' if spearman < 0.6 else 'correlated -> blend redundant'})")

    # 4. Does a RANK blend of E2+V30 beat V30 alone eval-honestly? (E2 as CORRECTION, not replacement)
    print("\n--- E2 as a CORRECTION layer on V30 (eval-honest) ---")
    v30_eh = weighted_ap(y, v30, wv)
    for bw in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = (1 - bw) * v30r + bw * e2r
        eh = weighted_ap(y, blend, wv)
        print(f"  V30 + {bw:.2f}*E2: eval-honest={eh:.4f} ({eh-v30_eh:+.4f} vs V30 {v30_eh:.4f})")

    # 5. Calibration: is E2's signal compressed (ranking ok, scores flat)?
    print("\n--- E2 calibration (ranking vs score spread) ---")
    print(f"  E2 score range on positives: [{pos_scores.min():.3f}, {pos_scores.max():.3f}]")
    print(f"  E2 unique scores: {len(np.unique(np.round(e2s,4)))} of {len(e2s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
