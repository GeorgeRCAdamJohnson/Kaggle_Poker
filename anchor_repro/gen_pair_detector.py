"""Task 4: pair-level detector on multi-hand CONSISTENCY features (generator recovery).

Builds pair-level features (per-hand aggregates + CONSISTENCY ratios) for dev AND eval from
the Layer-2 caches, drift-gates the block (dev vs eval), trains a GPU pair detector with
table-grouped OOF, and measures:
  * confirmed PairAP + PU-stress PairAP vs the frozen 0.70904 baseline_risk,
  * whether a rank-blend of the detector onto rank_sum_w50 improves PU-stress.

The consistency angle (fraction of a pair's hands showing the family pattern) separated
positives at AUC ~0.90 univariately — this tests if it TRANSPORTS and ADDS.

Run:  python -m anchor_repro.gen_pair_detector
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.own_submission_recipes import adversarial_drift_auc_matrix

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
CONF_OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
SEED = 42
PU_STRESS_WEIGHT = 112540 / 24000

SIG_COLS = ["dir_signal", "soft_signal", "iso_squeeze_signal", "abs_flow_bb", "both_commit",
            "iso_pair_preflop_agg", "soft_both_sd", "dir_opposite_flow", "member_pot_share",
            "n_outsider_folds", "pair_agg", "min_contrib_bb"]


def pair_features(layer2_path: Path) -> pl.DataFrame:
    """Pair-level aggregates + consistency ratios from a Layer-2 per-hand cache."""
    df = pl.scan_parquet(layer2_path)
    agg = df.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        *[pl.col(c).max().alias(f"{c}_max") for c in SIG_COLS],
        *[pl.col(c).mean().alias(f"{c}_mean") for c in SIG_COLS],
        *[pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in SIG_COLS],
        *[(pl.col(c) > 0).mean().alias(f"{c}_rate") for c in SIG_COLS],
        *[pl.col(c).sum().alias(f"{c}_sum") for c in SIG_COLS],
    ).collect(engine="streaming")
    return agg


def _score_cv(X, y, groups, params, n_rounds=400):
    oof = np.zeros(len(y), dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)},
                      xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=n_rounds)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    return oof


def main() -> int:
    dev = pair_features(CACHE / "dev_layer2.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label", "player_1", "player_2"])
    # table_id per pair (dominant table)
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    pair_tbl = (
        pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
        .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
        .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
        .select(["pair_id", "table_id"]).collect()
    )
    dev = dev.join(labels.select(["pair_id", "label"]), on="pair_id", how="left").join(pair_tbl, on="pair_id", how="left")

    # Drop the drift-prone raw-count (_sum) features; rate/mean/max/top3 are transport-clean
    # (per-column drift diagnosis §gen_pair_drift: all 7 drifters are _sum count features).
    feat_cols = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")
                 and not c.endswith("_sum")]
    X = dev.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    groups = dev["table_id"].to_numpy()
    print(f"dev pairs={dev.height} positives={int(y.sum())} feats={len(feat_cols)} device={xgb_device()}")

    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.04,
                        subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    oof = _score_cv(X, y, groups, params)
    conf_ap = average_precision_score(y, oof)
    conf_auc = roc_auc_score(y, oof)
    print(f"\n=== generator pair-detector (table-grouped OOF) ===")
    print(f"confirmed PairAP={conf_ap:.4f}  AUC={conf_auc:.4f}  (base {y.mean():.4f})")

    # --- Drift gate: build eval pair features, compare dev vs eval on the SAME cols ---
    print("\n=== drift gate (dev vs eval pair features) ===")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    Xe = evf.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    res = adversarial_drift_auc_matrix(X.to_numpy(), Xe.to_numpy(), block="gen_pair", seed=SEED)
    print(f"drift AUC={res.auc:.4f} (thr {res.threshold}; pass={res.passed})")

    # --- Compare / combine with frozen 0.70904 baseline on confirmed + PU-stress ---
    base = pl.read_parquet(CONF_OOF).select(["pair_id", "known", "y", "baseline_risk", "rank_sum_w50"])
    det = dev.select(["pair_id"]).with_columns(pl.Series("gen_score", oof))
    merged = base.join(det, on="pair_id", how="left").with_columns(pl.col("gen_score").fill_null(0.0))
    yb = merged["y"].to_numpy().astype(np.int8)
    known = merged["known"].to_numpy().astype(bool)
    sw = np.where(known, 1.0, PU_STRESS_WEIGHT)
    rsw = merged["rank_sum_w50"].to_numpy().astype(float)
    gen = merged["gen_score"].to_numpy().astype(float)

    # NOTE: gen_score is only defined for labeled pairs (dev has 1860 labeled; the 24k PU
    # pairs get 0). So PU-stress here is a LOWER bound for a gen-only arm. We still report it.
    def rankn(a): return pd.Series(a).rank(pct=True).to_numpy()
    base_conf = average_precision_score(yb[known], rsw[known])
    base_stress = average_precision_score(yb, rsw, sample_weight=sw)
    print(f"\nbaseline rank_sum_w50: confirmed={base_conf:.5f} pu_stress={base_stress:.5f}")
    print(f"generator detector on labeled pairs only: confirmed_AP={average_precision_score(yb[known], gen[known]):.5f}")

    br = rankn(rsw)
    gr = rankn(gen)
    for w in [0.1, 0.2, 0.3]:
        blend = (1 - w) * br + w * gr
        cc = average_precision_score(yb[known], blend[known])
        print(f"  blend w={w:.1f}: confirmed={cc:.5f} ({cc-base_conf:+.5f})")

    summary = {
        "confirmed_pair_ap": float(conf_ap), "confirmed_auc": float(conf_auc),
        "drift_auc": float(res.auc), "drift_pass": bool(res.passed),
        "baseline_confirmed": float(base_conf), "device": xgb_device(),
        "note": "consistency-based generator pair detector; PU-stress vs baseline is lower-bounded (gen defined on labeled pairs only)",
    }
    (CACHE / "_pair_detector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # top feature importances
    m = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=200)
    imp = pd.Series(m.get_score(importance_type="gain"))
    print("\ntop-12 gain features:")
    for k, v in imp.sort_values(ascending=False).head(12).items():
        print(f"  {v:10.1f}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
