"""Assemble a submission from the eval-negatives risk (held-out-validated AUC 0.95, table-disjoint).

Controlled test: risk_score = eval-negatives PU risk (trained: 372 confirmed positives vs a bagged
sample of the 112,540 EVAL pairs as the unlabeled-negative class, V30 pair-aggregate features).
behavior + evidence = BYTE-IDENTICAL to the 0.70904 shipped submission (so ONLY risk changes -> a
clean read of whether training against the eval population lifts PairAP on the real LB).

The held-out table-disjoint test showed held-out-POS vs dev-NEG AUC 0.95 (real, not phase). The open
question the LB answers: does 0.95 AUC hold PRECISION on the 0.2%-positive eval population, or dilute
like the fingerprint (AUC 0.88 -> LB 0.16)? LB is the only honest judge (harness leans on eval-neg).

Run:  python -m anchor_repro.assemble_evalneg_submission
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from anchor_repro.gpu_config import xgb_params
from anchor_repro.eval_honest_harness import D

DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
EVAL_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
SHIPPED = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
OUT = Path("outputs/poker_collusion/evalneg_submission.csv")
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

    pos = dev.filter(pl.col("label") == 1).to_pandas()
    def M(df): return df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    Xpos = M(pos)
    eval_all = ev.select(FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    rng = np.random.default_rng(SEED)
    n_neg, n_bags = 40000, 8
    eval_score = np.zeros(ev.height, np.float64)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.03,
                    subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=6.0)
    for _ in range(n_bags):
        Xb = ev[rng.choice(ev.height, n_neg, replace=False)].select(FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        Xt = pd.concat([Xpos, Xb], ignore_index=True); yt = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xb))])
        m = xgb.train({**pp, "scale_pos_weight": float(len(Xb)/len(Xpos))}, xgb.DMatrix(Xt, label=yt), num_boost_round=350)
        eval_score += m.predict(xgb.DMatrix(eval_all)) / n_bags
    escore = (eval_score - eval_score.min()) / (eval_score.max() - eval_score.min() + 1e-12)
    print(f"trained {n_bags}-bag full eval-neg risk ({time.time()-t:.0f}s)")

    risk = dict(zip(ev["pair_id"].to_list(), escore))
    sub = pd.read_csv(SHIPPED); sub["pair_id"] = sub["pair_id"].astype(str)
    new_risk = sub["pair_id"].map(risk)
    print(f"eval pairs matched: {new_risk.notna().sum()}/{len(sub)}; unmatched filled with 0")
    sub["risk_score"] = new_risk.fillna(0.0).clip(0, 1).astype(np.float32)
    # behavior + evidence BYTE-IDENTICAL to shipped (already in `sub`)
    assert sub["risk_score"].between(0, 1).all() and not sub["risk_score"].isna().any()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(sub)} rows); risk=eval-neg, behavior+evidence=shipped 0.70904")
    # correlation vs shipped risk (how different is the ranking?)
    from scipy.stats import spearmanr
    old = pd.read_csv(SHIPPED)["risk_score"].to_numpy()
    print(f"Spearman(eval-neg risk, V30 risk) = {spearmanr(sub['risk_score'], old).correlation:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
