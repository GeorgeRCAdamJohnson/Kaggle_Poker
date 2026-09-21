"""FORWARD x BACKWARD -> attack the PRECISION problem (dossier §127 next step).

§127: the backward fingerprint separates planted hands (AUC 0.92) and confirmed pairs (0.88) but
eval-honest PairAP is 0.22 — because ordinary pairs also throw off random big-aggressive-opposite-
flow hands, so the fingerprint fires on them too and swamps the ~0.2% true positives.

The precision fix combines the two halves:
  FORWARD (pair-level rate)  x  BACKWARD (per-hand fingerprint)  x  PRECISION levers:
   (a) EXCESS over table expectation: fingerprint-hit-rate MINUS the mean rate of all pairs at that
       table with similar shared-hand count. Ordinary flashy pairs sit ~at expectation; colluders
       far above. This cancels the base-rate false-fire.
   (b) DIRECTIONAL CONSISTENCY: over the pair's high-fingerprint hands, does the SAME member fund
       the other repeatedly? |sum(flow_sign)| / count. Ordinary flashy hands transfer in RANDOM
       direction (~0); colluders are consistent (~1). Ordinary pairs cannot fake this conjunction.
   (c) HIGH-PRECISION REGIME: count of ultra-high-fingerprint hands (fp>0.7, >0.9) — the regime
       where the per-hand model is near-pure; trades recall for precision.

Build per-hand fp + flow_sign for dev+eval, aggregate these precision features per pair, train,
judge eval-honest vs 0.22 (raw fingerprint) and vs 0.709 (V30). Blend with V30 too.
Run:  python -m anchor_repro.precision_combine
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, CACHE, D
from anchor_repro.backwards_fingerprint import FP_FEATS, _build_dev_perhand_dims, _build_eval_perhand_dims

OUT = CACHE / "backwards"
CONTRAST = OUT / "planted_contrast.parquet"
SEED = 42


def _fit_fingerprint():
    """Train the per-hand fingerprint model on the planted-vs-nonplanted contrast."""
    con = pl.read_parquet(CONTRAST)
    y = con["planted"].to_numpy()
    X = con.select(FP_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.05,
                        subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=4.0)
    return xgb.train({**params, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=300)


def _precision_features(dims: pl.DataFrame, fp_model) -> pl.DataFrame:
    """Per pair: fingerprint-rate excess-over-table-expectation + directional consistency + hi-regime."""
    X = dims.select(FP_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    fp = fp_model.predict(xgb.DMatrix(X))
    # per-hand flow sign: +1 if member A funds B (a<0,b>0), -1 if B funds A, 0 else
    d = dims.with_columns(
        pl.Series("fp", fp),
        pl.when((pl.col("net_chips") < 0) & (pl.col("net_chips_b") > 0)).then(1)
          .when((pl.col("net_chips_b") < 0) & (pl.col("net_chips") > 0)).then(-1)
          .otherwise(0).alias("flow_sign"),
    )
    # table id per pair-hand for the expectation baseline
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    d = d.join(hp.collect(), on="hand_id", how="left")

    # pair aggregates
    per_pair = d.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        pl.first("table_id").alias("table_id"),
        pl.col("fp").mean().alias("fp_mean"),
        (pl.col("fp") > 0.5).mean().alias("fp_rate50"),
        (pl.col("fp") > 0.7).sum().alias("fp_cnt70"),
        (pl.col("fp") > 0.9).sum().alias("fp_cnt90"),
        pl.col("fp").top_k(5).mean().alias("fp_top5"),
        # directional consistency over HIGH-fingerprint hands only
        (pl.col("flow_sign").filter(pl.col("fp") > 0.5).sum()).alias("hi_sign_sum"),
        (pl.col("flow_sign").filter(pl.col("fp") > 0.5).abs().sum()).alias("hi_sign_absn"),
        # directional consistency over ALL hands
        pl.col("flow_sign").sum().alias("all_sign_sum"),
        (pl.col("flow_sign") != 0).sum().alias("all_sign_nonzero"),
    ).with_columns(
        # directional consistency: |net direction| / count of directed hi-fp hands (1=one-way)
        (pl.col("hi_sign_sum").abs() / (pl.col("hi_sign_absn") + 1e-6)).alias("hi_dir_consistency"),
        (pl.col("all_sign_sum").abs() / (pl.col("all_sign_nonzero") + 1e-6)).alias("all_dir_consistency"),
    )
    # table expectation: mean fp_rate50 over pairs at the same table -> excess
    tbl = per_pair.group_by("table_id").agg(
        pl.col("fp_rate50").mean().alias("tbl_fp_rate50"),
        pl.col("fp_mean").mean().alias("tbl_fp_mean"),
        pl.col("fp_cnt70").mean().alias("tbl_cnt70"),
    )
    per_pair = per_pair.join(tbl, on="table_id", how="left").with_columns(
        (pl.col("fp_rate50") - pl.col("tbl_fp_rate50")).alias("fp_rate_excess"),
        (pl.col("fp_mean") - pl.col("tbl_fp_mean")).alias("fp_mean_excess"),
        (pl.col("fp_cnt70") - pl.col("tbl_cnt70")).alias("fp_cnt70_excess"),
        # the KEY conjunction: excess fingerprint rate WEIGHTED by directional consistency
        ((pl.col("fp_rate50") - pl.col("tbl_fp_rate50")) * pl.col("hi_dir_consistency")).alias("excess_x_consistency"),
        (pl.col("fp_cnt70").cast(pl.Float64) * pl.col("hi_dir_consistency")).alias("cnt70_x_consistency"),
    )
    return per_pair


PFEATS = ["fp_mean", "fp_rate50", "fp_cnt70", "fp_cnt90", "fp_top5", "hi_dir_consistency",
          "all_dir_consistency", "fp_rate_excess", "fp_mean_excess", "fp_cnt70_excess",
          "excess_x_consistency", "cnt70_x_consistency", "n_hands"]


def main() -> int:
    H = Harness()
    fp_model = _fit_fingerprint()
    print("fingerprint model fit; building per-hand dims + precision features...")

    t = time.time()
    dev_dims = _build_dev_perhand_dims()
    dev_pp = _precision_features(dev_dims, fp_model)
    print(f"dev precision features ({time.time()-t:.0f}s)")
    t = time.time()
    eval_dims = _build_eval_perhand_dims()
    eval_pp = _precision_features(eval_dims, fp_model)
    print(f"eval precision features ({time.time()-t:.0f}s)")

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    ptbl = dev_pp.select(["pair_id", "table_id"])
    dev = labels.join(dev_pp, on="pair_id", how="left")
    X = dev.select(PFEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].fill_null("?").to_numpy()

    print("\n=== precision features univariate AUC (vs labels) ===")
    for c in PFEATS:
        v = X[c].to_numpy()
        try:
            a = roc_auc_score(y, v); print(f"  {max(a,1-a):.4f}  {c}")
        except Exception: pass

    oof = np.zeros(len(y), np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    for tr, va in sgkf.split(X, y, g):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=300)
    evf = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id"]).join(eval_pp, on="pair_id", how="left")
    Xe = evf.select(PFEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    escore = full.predict(xgb.DMatrix(Xe))
    r = H.judge("precision_combine", oof, dev["pair_id"].to_list(), escore, evf["pair_id"].to_list())
    print(f"\n=== PRECISION-COMBINE risk (eval-honest) ===")
    print(f"  {r}")
    print(f"  vs raw fingerprint 0.2249  |  vs V30 real LB 0.70904")

    # blend with V30
    def prank(a): import pandas as pd; return pd.Series(a).rank(pct=True).to_numpy()
    seq = pl.read_parquet(Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet"))
    v30_dev = dict(zip(seq["pair_id"].to_list(), seq["rank_sum_w50"].to_numpy()))
    ebase = pl.read_csv("outputs/poker_collusion/repro_lamhuy/submission.csv").select(["pair_id","risk_score"])
    espec = pl.read_parquet("outputs/poker_collusion/lamhuy_sequence_confirmation/eval_specialist_scores.parquet")
    ej = ebase.join(espec, on="pair_id", how="inner")
    import pandas as pd
    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    v30_e = np.clip(0.5*pr(ej["risk_score"].to_numpy()) + 0.5*pr(np.clip(np.column_stack([ej["specialist_directed"],ej["specialist_soft"],ej["specialist_isolation"]]).sum(1),0,1)),0,1)
    v30_eids = ej["pair_id"].to_list()
    dv = np.array([v30_dev.get(p, 0.0) for p in dev["pair_id"].to_list()])
    e3map = dict(zip(evf["pair_id"].to_list(), escore)); e3e = np.array([e3map.get(p, np.median(escore)) for p in v30_eids])
    print("\n  V30 + precision blend (eval-honest):")
    for bw in [0.1, 0.2, 0.3]:
        db = (1-bw)*pr(dv) + bw*pr(oof); eb = (1-bw)*pr(v30_e) + bw*pr(e3e)
        rb = H.judge(f"v30+prec_w{int(bw*100)}", db, dev["pair_id"].to_list(), eb, v30_eids)
        print(f"    w={bw}: eval_honest={rb.eval_honest_ap:.4f}")

    # persist eval risk IMMEDIATELY (before the optional blend section) so it is always saved
    pl.DataFrame({"pair_id": evf["pair_id"].to_list(), "risk_score": np.clip(escore,0,1).astype(np.float32)}).write_parquet(OUT / "precision_eval_risk.parquet")
    (OUT / "_precision_summary.json").write_text(json.dumps(
        {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc,
         "vs_raw_fingerprint": r.eval_honest_ap - 0.2249}, indent=2), encoding="utf-8")
    print("wrote precision_eval_risk.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
