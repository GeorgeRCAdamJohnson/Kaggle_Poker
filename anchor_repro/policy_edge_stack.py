"""Complementarity test: STACK the baseline OOF risk + our edge feats in a small model (force interaction).

Diagnosis showed: edge feats are NON-redundant (corr 0.04-0.14 w/ loose_pf_sum, 0.21-0.26 w/ base OOF)
and contain the baseline's MISSED positives (residual AUC 0.865), but a naive rank-BLEND craters
(0.73->0.44) because the edge feature ranks some ordinary loose players sky-high (base-rate). And naive
CONCAT into 221 feats gave the edge cols ~0 weight (-0.0022).

Fix: a SMALL stacker where the tree must LEARN to gate the edge signal on the baseline risk — i.e.
"trust high joint-surprise ONLY when baseline risk is also elevated". Features = base_oof (their full
model's risk) + the ~6 strong edge feats + a few explicit interactions (edge * base_oof). Table-grouped
OOF, judged eval-mirrored vs baseline 0.7299.
Run:  python -m anchor_repro.policy_edge_stack
"""
from __future__ import annotations
import json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score

STEP3 = Path("outputs/poker_collusion/hosen42_step3"); EDGE = STEP3 / "edge"
SEED, N_FOLDS = 42, 5

def main():
    from anchor_repro.policy_edge_train import _pair_edge_features
    dev = pd.read_parquet(STEP3 / "dev_pair_features.parquet")
    de = _pair_edge_features(STEP3 / "dev_pair_hands.parquet", "development").to_pandas()
    dev = dev.merge(de, on="pair_id", how="left")
    oof_base = np.load(EDGE / "oof_base.npy")
    dev["base_oof"] = oof_base
    y = dev["label"].fillna(0).astype(int).to_numpy()
    labm = dev["is_labeled"].to_numpy().astype(bool)
    TABLES = sorted(dev["table_id"].unique().tolist())
    rng = np.random.RandomState(SEED)
    TABLE_FOLD = {t: int(v) for t, v in zip(TABLES, rng.permutation(len(TABLES)) % N_FOLDS)}
    folds = dev["table_id"].map(TABLE_FOLD).to_numpy()

    EDGEF = ["joint_surp_excess", "total_loose_min_top3", "loose_corr", "post_loose_min_mean", "both_surp_sum", "total_loose_top3"]
    for c in EDGEF: dev[c] = dev[c].astype("float32").fillna(0)
    # explicit interactions: edge only "counts" when base risk is elevated
    br = pd.Series(oof_base).rank(pct=True).to_numpy()
    for c in EDGEF:
        er = pd.Series(dev[c].to_numpy()).rank(pct=True).to_numpy()
        dev[f"{c}_x_base"] = (er * br).astype("float32")
    STACK_FEATS = ["base_oof"] + EDGEF + [f"{c}_x_base" for c in EDGEF]

    X = dev[STACK_FEATS].astype("float32")
    w = np.where(labm, np.where(y == 1, 5.0, 1.0), 0.1)
    P = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=80, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=8.0, verbose=-1, num_threads=-1)
    oof = np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr = (folds != f) & (w > 0); te = folds == f
        for sd in [SEED, SEED+11, SEED+23]:
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr], weight=w[tr]), num_boost_round=500)
            oof[te] += m.predict(X[te]) / 3
    base_ap = average_precision_score(y, oof_base)
    stack_ap = average_precision_score(y, oof)
    print(f"\n=== STACK (base_oof + edge + interactions) eval-mirrored PairAP ===")
    print(f"  baseline: {base_ap:.4f}")
    print(f"  stack   : {stack_ap:.4f}   delta {stack_ap-base_ap:+.4f}  ({'WINS' if stack_ap>base_ap+0.002 else 'no gain'})")
    # also try stacker WITHOUT interactions (edge raw only) for ablation
    X2 = dev[["base_oof"] + EDGEF].astype("float32")
    oof2 = np.zeros(len(dev))
    for f in range(N_FOLDS):
        tr = (folds != f) & (w > 0); te = folds == f
        for sd in [SEED, SEED+11, SEED+23]:
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(X2[tr], y[tr], weight=w[tr]), num_boost_round=500)
            oof2[te] += m.predict(X2[te]) / 3
    print(f"  stack (no interactions): {average_precision_score(y, oof2):.4f}")
    np.save(EDGE / "oof_stack.npy", oof)
    (EDGE / "_stack_summary.json").write_text(json.dumps({"base": float(base_ap), "stack": float(stack_ap)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
