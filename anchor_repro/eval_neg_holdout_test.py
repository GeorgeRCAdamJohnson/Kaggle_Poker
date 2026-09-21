"""DECISIVE test: is the eval-negatives model reading COLLUSION or PHASE? Data, not opinion.

The prior runs reported confirmed AUC 1.0000 / eval-honest 1.0000, but the dev_score there came from
bags that TRAINED ON those positives (contaminated) and the harness leans on the eval population the
model trained against. This test removes both confounds with a clean held-out split:

  1. Split the 372 positives: TRAIN on 300, HOLD OUT 72 (never seen in training).
  2. Train positives(300) vs eval-negatives(40k), V30 pair-aggregate features.
  3. Score, on the SAME model: (a) the 72 HELD-OUT positives, (b) the 1,488 dev confirmed-NEGATIVES.

Interpretation (the whole point):
  * REAL COLLUSION signal  -> held-out positives rank ABOVE dev negatives (AUC >> 0.5).
  * PHASE leak             -> held-out positives ~ dev negatives (both are dev-phase; AUC ~ 0.5),
                             while BOTH score far above eval pairs (phase, not collusion).

We also report: score of held-out-POS vs a fresh eval-neg sample (should be high if phase),
and dev-NEG vs eval-neg (should be high if phase). The gap tells us everything.
Run:  python -m anchor_repro.eval_neg_holdout_test
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from anchor_repro.gpu_config import xgb_params
from anchor_repro.eval_honest_harness import D, CACHE

DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
EVAL_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
SKIP = {"pair_id", "hand_id", "table_id"}
SEED = 42


def _agg(path, feats):
    lf = pl.scan_parquet(path)
    agg = ([pl.len().alias("shared_hands")]
           + [pl.col(c).mean().alias(f"{c}_mean") for c in feats]
           + [pl.col(c).max().alias(f"{c}_max") for c in feats]
           + [pl.col(c).quantile(0.95, interpolation="nearest").alias(f"{c}_p95") for c in feats]
           + [pl.col(c).top_k(5).mean().alias(f"{c}_top5") for c in feats])
    return lf.group_by("pair_id").agg(agg).collect(engine="streaming")


def main() -> int:
    t = time.time()
    feats = [c for c in pl.scan_parquet(DEV_HF).collect_schema().names() if c not in SKIP]
    dev = _agg(DEV_HF, feats)
    ev = _agg(EVAL_HF, feats)
    lab = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(lab, on="pair_id", how="left")
    FEATS = [c for c in dev.columns if c not in {"pair_id", "label", "shared_hands"}]
    print(f"aggregated ({time.time()-t:.0f}s); feats={len(FEATS)}")

    rng = np.random.default_rng(SEED)
    # attach table_id per pair (most-common table over its hands) for a TABLE-DISJOINT split
    hp = pl.scan_parquet(DEV_HF).select(["pair_id", "table_id"]).group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
    ptab = (hp.sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
            .select(["pair_id", "table_id"]).collect())
    dev = dev.join(ptab, on="pair_id", how="left")
    pos = dev.filter(pl.col("label") == 1).to_pandas()
    neg = dev.filter(pl.col("label") == 0).to_pandas()   # 1488 dev confirmed negatives
    # TABLE-DISJOINT split of positives: holdout positives share NO table with train positives
    tabs = pos["table_id"].fillna("NA").to_numpy()
    uniq = rng.permutation(np.unique(tabs))
    ho_tabs = set(uniq[: max(1, len(uniq) // 5)])       # ~20% of positive-tables held out
    ho_mask = np.array([t in ho_tabs for t in tabs])
    tr_pos = pos.iloc[~ho_mask]; ho_pos = pos.iloc[ho_mask]
    print(f"positives: train={len(tr_pos)} holdout={len(ho_pos)} (TABLE-DISJOINT); dev negatives={len(neg)}")

    def M(df): return df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # negatives = eval-population sample
    neg_ev = ev[rng.choice(ev.height, 40000, replace=False)].to_pandas()
    Xtr = pd.concat([M(tr_pos), M(neg_ev)], ignore_index=True)
    ytr = np.concatenate([np.ones(len(tr_pos)), np.zeros(len(neg_ev))])
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.03,
                    subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=6.0,
                    scale_pos_weight=float(len(neg_ev)/len(tr_pos)))
    m = xgb.train(pp, xgb.DMatrix(Xtr, label=ytr), num_boost_round=350)
    print(f"trained on 300 pos vs 40k eval-neg ({time.time()-t:.0f}s)")

    s_ho_pos = m.predict(xgb.DMatrix(M(ho_pos)))     # held-out positives (never trained)
    s_dev_neg = m.predict(xgb.DMatrix(M(neg)))       # dev confirmed negatives
    fresh_ev = ev[rng.choice(ev.height, 20000, replace=False)].to_pandas()
    s_ev = m.predict(xgb.DMatrix(M(fresh_ev)))       # fresh eval pairs

    print("\n=== score distributions (mean, median) ===")
    for name, s in [("HELD-OUT positives (72)", s_ho_pos), ("dev NEGATIVES (1488)", s_dev_neg),
                    ("fresh EVAL pairs (20k)", s_ev)]:
        print(f"  {name:28} mean={s.mean():.4f} median={np.median(s):.4f}")

    # THE decisive AUCs
    y1 = np.concatenate([np.ones(len(s_ho_pos)), np.zeros(len(s_dev_neg))])
    auc_pos_vs_devneg = roc_auc_score(y1, np.concatenate([s_ho_pos, s_dev_neg]))
    y2 = np.concatenate([np.ones(len(s_dev_neg)), np.zeros(len(s_ev))])
    auc_devneg_vs_ev = roc_auc_score(y2, np.concatenate([s_dev_neg, s_ev]))
    y3 = np.concatenate([np.ones(len(s_ho_pos)), np.zeros(len(s_ev))])
    auc_pos_vs_ev = roc_auc_score(y3, np.concatenate([s_ho_pos, s_ev]))

    print("\n=== DECISIVE AUCs ===")
    print(f"  held-out POS vs dev NEG : {auc_pos_vs_devneg:.4f}   <- REAL collusion signal if >>0.5; PHASE leak if ~0.5")
    print(f"  dev NEG    vs eval pairs: {auc_devneg_vs_ev:.4f}   <- PHASE signal (dev-neg look dev-phase) if >>0.5")
    print(f"  held-out POS vs eval    : {auc_pos_vs_ev:.4f}   <- mixes collusion + phase")
    print()
    if auc_pos_vs_devneg < 0.65:
        print("  VERDICT: held-out POS ~ dev NEG => the model does NOT separate real positives from")
        print("           dev negatives. The 1.0000 was PHASE (dev vs eval), not collusion. LEAK CONFIRMED.")
    else:
        print(f"  VERDICT: held-out POS rank above dev NEG at AUC {auc_pos_vs_devneg:.3f} => REAL signal beyond phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
