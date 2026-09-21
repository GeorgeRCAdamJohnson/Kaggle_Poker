"""NEW PAIR-LEVEL risk: relationship CONSISTENCY/recurrence, not per-hand magnitude.

Why the whole lineage caps at ~0.71 and the unified per-hand model cratered (0.16): an ORDINARY
pair occasionally has a big directed transfer / checkdown / isolation (these happen naturally). A
COLLUDING pair does it REPEATEDLY and in a CONSISTENT DIRECTION. Magnitude-based features (all of
V30, E-line, unified) can't separate them because the per-hand mark is a look-alike (§30/§31).

This builds CONSISTENCY features that ordinary pairs should NOT exhibit:
  * directional PERSISTENCE: does the SAME member consistently fund the other? (sign consistency of
    net flow across the pair's shared hands, magnitude-independent)
  * RECURRENCE RATE of directed/soft/iso events, NORMALIZED vs the pair's own TABLE baseline (idea #1:
    subtract what a typical co-seated pair at that table shows -> cancels the look-alike base rate)
  * CONCENTRATION: are coordinated hands clustered (episodic relationship) vs spread (coincidence)
  * ASYMMETRY of who-wins-when-both-committed (a colluder yields to partner systematically)

Trained on CONFIRMED labels (pos vs neg), table-grouped OOF -> the harness is TRUSTWORTHY here
(confirmed base, unlike the PU E-line). Judged eval-honest PairAP vs V30 0.92.

Run:  python -m anchor_repro.consistency_risk
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D

OUT = CACHE / "consistency"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _pair_consistency(layer2_path: Path) -> pl.DataFrame:
    """Per-pair CONSISTENCY features (recurrence, direction persistence, concentration)."""
    df = pl.scan_parquet(layer2_path).with_columns(
        # per-hand directed-transfer event: opposite-sign flow, both committed
        (pl.col("dir_opposite_flow") == 1).cast(pl.Int8).alias("ev_directed"),
        # per-hand soft event: both showdown, no aggression between
        ((pl.col("a_went_to_showdown") == 1) & (pl.col("b_went_to_showdown") == 1)
         & (pl.col("pair_agg") == 0)).cast(pl.Int8).alias("ev_soft"),
        # per-hand iso event
        (pl.col("iso_one_commits_partner_folds") == 1).cast(pl.Int8).alias("ev_iso"),
        # directional sign: +1 if a funds b (a loses, b wins), -1 if b funds a, 0 else
        pl.when((pl.col("a_net_bb") < 0) & (pl.col("b_net_bb") > 0)).then(1)
          .when((pl.col("b_net_bb") < 0) & (pl.col("a_net_bb") > 0)).then(-1)
          .otherwise(0).alias("flow_sign"),
        pl.col("abs_flow_bb").alias("flow_mag"),
    )
    agg = df.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        pl.first("table_id").alias("table_id"),
        # RECURRENCE rates
        pl.col("ev_directed").mean().alias("directed_rate"),
        pl.col("ev_soft").mean().alias("soft_rate"),
        pl.col("ev_iso").mean().alias("iso_rate"),
        pl.col("ev_directed").sum().alias("directed_n"),
        # DIRECTIONAL PERSISTENCE: among directed hands, how consistently same direction?
        # net sum of signs / count of nonzero signs -> |.| near 1 = one member always funds
        pl.col("flow_sign").sum().alias("flow_sign_sum"),
        (pl.col("flow_sign") != 0).sum().alias("flow_nonzero_n"),
        # net directional chip flow (who funds whom, cumulative) normalized by hands
        pl.col("a_net_bb").sum().alias("a_cum_net"),
        pl.col("b_net_bb").sum().alias("b_cum_net"),
        # CONCENTRATION: variance of event positions proxy -> use event rate vs uniform
        pl.col("flow_mag").filter(pl.col("ev_directed") == 1).mean().alias("directed_flow_mag_mean"),
        # win-when-committed asymmetry
        ((pl.col("a_net_bb") > pl.col("b_net_bb")).cast(pl.Int8)).mean().alias("a_wins_rate"),
    ).with_columns(
        # persistence: |sum of signs| / nonzero count (1 = perfectly one-directional)
        (pl.col("flow_sign_sum").abs() / (pl.col("flow_nonzero_n") + 1e-6)).alias("dir_persistence"),
        # net imbalance of cumulative flow (colluder pair: strongly one-sided)
        ((pl.col("a_cum_net") - pl.col("b_cum_net")).abs()
         / (pl.col("a_cum_net").abs() + pl.col("b_cum_net").abs() + 1e-6)).alias("net_imbalance"),
        # win asymmetry away from 0.5
        (pl.col("a_wins_rate") - 0.5).abs().alias("win_asymmetry"),
    )
    return agg.collect(engine="streaming")


def _table_normalize(pairs: pl.DataFrame) -> pl.DataFrame:
    """Idea #1: subtract each pair's TABLE baseline from its recurrence rates.

    For each table, the mean rate over all co-seated pairs at that table = the look-alike base rate.
    A colluding pair's EXCESS over its table = the signal that cancels natural coordination.
    """
    rate_cols = ["directed_rate", "soft_rate", "iso_rate", "dir_persistence", "net_imbalance",
                 "win_asymmetry"]
    tbl = pairs.group_by("table_id").agg(
        [pl.col(c).mean().alias(f"tbl_{c}") for c in rate_cols])
    out = pairs.join(tbl, on="table_id", how="left")
    for c in rate_cols:
        out = out.with_columns((pl.col(c) - pl.col(f"tbl_{c}")).alias(f"{c}_excess"))
    return out


def main() -> int:
    H = Harness()
    t = time.time()
    dev = _table_normalize(_pair_consistency(CACHE / "dev_layer2.parquet"))
    evf = _table_normalize(_pair_consistency(CACHE / "eval_layer2.parquet"))
    print(f"consistency features built dev={dev.height} eval={evf.height} ({time.time()-t:.0f}s)")

    lab = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(lab, on="pair_id", how="left")
    feat_cols = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")
                 and not c.startswith("tbl_")]
    X = dev.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    groups = dev["table_id"].to_numpy()
    print(f"pairs={dev.height} pos={int(y.sum())} feats={len(feat_cols)} device={xgb_device()}")

    # univariate AUC of the consistency features (are they real separators?)
    print("\n=== univariate AUC (consistency features vs confirmed labels) ===")
    rows = []
    for c in feat_cols:
        v = X[c].to_numpy()
        auc = max(roc_auc_score(y, v), roc_auc_score(y, -v))
        rows.append((c, auc))
    for c, a in sorted(rows, key=lambda r: -r[1])[:12]:
        print(f"  {a:.4f}  {c}")

    # table-grouped OOF risk model
    oof = np.zeros(len(y), np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                        subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    for tr, va in sgkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)},
                      xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=350)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    # full model -> eval
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**params, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=350)
    Xe = evf.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    escore = full.predict(xgb.DMatrix(Xe))

    r = H.judge("consistency_risk", oof, dev["pair_id"].to_list(), escore, evf["pair_id"].to_list())
    print(f"\n=== CONSISTENCY risk (eval-honest; confirmed base -> TRUSTWORTHY) ===")
    print(f"  {r}")
    print(f"  vs V30 rank_sum_w50 = 0.9203 -> delta {r.eval_honest_ap - 0.9203:+.4f}")

    # persist eval risk for possible assembly
    pl.DataFrame({"pair_id": evf["pair_id"].to_list(),
                  "risk_score": np.clip(escore, 0, 1).astype(np.float32)}).write_parquet(OUT / "eval_risk.parquet")
    (OUT / "_consistency_summary.json").write_text(json.dumps(
        {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc,
         "vs_v30": r.eval_honest_ap - 0.9203, "top_univariate": sorted(rows, key=lambda x: -x[1])[:12]},
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/eval_risk.parquet, _consistency_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
