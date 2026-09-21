"""Round 2: Detector D (victim counterfactual) + HONEST ensemble of all detectors.

Detector D — outsider/victim counterfactual:
  For each outsider seated at a pair's shared hand, measure the outsider's outcome DEVIATION
  vs that outsider's own field baseline (mean net_bb over ALL their hands). If a pair is
  colluding, outsiders at their table should systematically lose more / fold more than their
  own baseline predicts. This is a per-outsider matched counterfactual (§114 route 2).

Ensemble — honest, eval-honest-harness-only:
  Rank-average the dev OOF scores from round 1 (A, B, E) + D across labeled dev pairs and
  judge via the SAME importance-weighted AP (no PU-blend universe, no fill, no leak). Also
  compare to V30 baseline by building importance weights on the SAME labeled dev set.

Run:  python -m anchor_repro.detector_bakeoff_round2
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, KFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D, SEED
from anchor_repro.gen_pair_detector import pair_features

HAND_PLAYER = CACHE / "hand_player_agg.parquet"
HAND_CTX = CACHE / "hand_context.parquet"
V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")


# ========================================================================================
# DETECTOR D: VICTIM / OUTSIDER COUNTERFACTUAL
# ========================================================================================

def _player_field_baseline() -> pl.DataFrame:
    """Per player: their own field baseline (mean net_bb, fold rate, agg rate over ALL hands)."""
    hp = pl.scan_parquet(HAND_PLAYER)
    ctx = pl.scan_parquet(HAND_CTX).select(["hand_id", "big_blind"])
    return (
        hp.join(ctx, on="hand_id", how="left")
        .with_columns(
            (pl.col("net_chips") / pl.col("big_blind")).alias("net_bb"),
            pl.col("folded").cast(pl.Int8).alias("is_fold"),
            (pl.col("n_agg") > 0).cast(pl.Int8).alias("is_agg"),
        )
        .group_by("player_id").agg(
            pl.len().alias("bl_n"),
            pl.col("net_bb").mean().alias("bl_net_mean"),
            pl.col("is_fold").mean().alias("bl_fold_rate"),
            pl.col("is_agg").mean().alias("bl_agg_rate"),
            pl.col("net_bb").std().alias("bl_net_std"),
        )
        .collect(engine="streaming")
    )


def _outsider_deviations(pair_frame: pl.LazyFrame, universe: pl.LazyFrame,
                         baseline: pl.DataFrame) -> pl.DataFrame:
    """Per (pair_id, hand_id): outsider outcome deviation from their own baseline.

    An 'outsider' is any player seated at the hand who is NOT a member of the pair.
    For each outsider, compute deviation = observed - baseline. Then aggregate per hand:
    mean outsider loss, mean outsider fold surplus, etc. Then aggregate per pair.
    """
    pf = pair_frame.select(["pair_id", "player_1", "player_2"])
    uni = universe.select(["pair_id", "hand_id"])

    # all seats in the shared-hand universe
    seats = (
        pl.scan_parquet(HAND_PLAYER).select(["hand_id", "player_id", "net_chips", "folded", "n_agg"])
        .join(pl.scan_parquet(HAND_CTX).select(["hand_id", "big_blind"]), on="hand_id", how="left")
        .with_columns(
            (pl.col("net_chips") / pl.col("big_blind")).alias("net_bb"),
            pl.col("folded").cast(pl.Int8).alias("is_fold"),
            (pl.col("n_agg") > 0).cast(pl.Int8).alias("is_agg"),
        )
    )
    # join to pair universe: for each (pair_id, hand_id), get all seated players
    paired = (
        uni.join(pf, on="pair_id", how="inner")
        .join(seats, on="hand_id", how="inner")
    )
    # filter to OUTSIDERS only (not player_1 or player_2)
    outsiders = paired.filter(
        (pl.col("player_id") != pl.col("player_1")) &
        (pl.col("player_id") != pl.col("player_2"))
    )
    # join baseline
    bl = baseline.lazy().select(["player_id", "bl_net_mean", "bl_fold_rate", "bl_agg_rate", "bl_net_std"])
    outsiders = outsiders.join(bl, on="player_id", how="left")

    # per-outsider deviation
    outsiders = outsiders.with_columns(
        (pl.col("net_bb") - pl.col("bl_net_mean")).alias("net_dev"),
        (pl.col("is_fold") - pl.col("bl_fold_rate")).alias("fold_dev"),
        (pl.col("is_agg").cast(pl.Float64) - pl.col("bl_agg_rate")).alias("agg_dev"),
    )

    # aggregate per (pair_id, hand_id): mean outsider deviation
    per_hand = outsiders.group_by(["pair_id", "hand_id"]).agg(
        pl.col("net_dev").mean().alias("out_net_dev_mean"),
        pl.col("fold_dev").mean().alias("out_fold_dev_mean"),
        pl.col("agg_dev").mean().alias("out_agg_dev_mean"),
        pl.col("net_dev").min().alias("out_net_dev_min"),  # worst outsider loss
        pl.len().alias("n_outsiders"),
    )

    # aggregate per pair
    _OUT_COLS = ["out_net_dev_mean", "out_fold_dev_mean", "out_agg_dev_mean", "out_net_dev_min"]
    per_pair = per_hand.group_by("pair_id").agg(
        pl.len().alias("out_n_hands"),
        *[pl.col(c).mean().alias(f"{c}_pair_mean") for c in _OUT_COLS],
        *[pl.col(c).top_k(3).mean().alias(f"{c}_pair_top3") for c in _OUT_COLS],
        *[(pl.col(c) < 0).mean().alias(f"{c}_pair_neg_rate") for c in _OUT_COLS],
    ).collect(engine="streaming")
    return per_pair


def build_detector_D(H: Harness):
    """D: outsider/victim counterfactual."""
    t = time.time()
    baseline = _player_field_baseline()
    print(f"  player baselines: {baseline.height:,} players ({time.time()-t:.1f}s)")

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    dev_uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"])

    t = time.time()
    dev_out = _outsider_deviations(labels.select(["pair_id", "player_1", "player_2"]).lazy(), dev_uni, baseline)
    print(f"  dev outsider features: {dev_out.height:,} pairs ({time.time()-t:.1f}s)")

    # table map for grouped OOF
    hp = pl.scan_parquet(HAND_CTX).select(["hand_id", "table_id"])
    pair_tbl = (
        dev_uni.join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
        .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
        .select(["pair_id", "table_id"]).collect()
    )
    dev = dev_out.join(labels.select(["pair_id", "label"]), on="pair_id", how="left").join(pair_tbl, on="pair_id", how="left")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")]
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].to_numpy()
    print(f"  feats={len(feats)} pairs={dev.height} pos={int(y.sum())}")

    # OOF
    oof = np.zeros(len(y), dtype=np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4,
                        eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=6.0)
    for tr, va in sgkf.split(Xd, y, g):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)},
                      xgb.DMatrix(Xd.iloc[tr], label=y[tr]), num_boost_round=350)
        oof[va] = m.predict(xgb.DMatrix(Xd.iloc[va]))
    # full model for eval
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**params, "scale_pos_weight": float(spw)},
                     xgb.DMatrix(Xd, label=y), num_boost_round=350)

    # eval features (batched)
    t = time.time()
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    eval_uni = pl.scan_parquet(V30_EVAL).select(["pair_id", "hand_id"])
    parts = []
    batch = 20000
    for start in range(0, eval_pairs.height, batch):
        pf = eval_pairs.slice(start, batch).lazy()
        parts.append(_outsider_deviations(pf, eval_uni, baseline))
    evf = pl.concat(parts, how="vertical_relaxed")
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    es = full.predict(xgb.DMatrix(Xe))
    print(f"  eval outsider features: {evf.height:,} pairs ({time.time()-t:.1f}s)")
    return oof, dev["pair_id"].to_list(), es, evf["pair_id"].to_list()


# ========================================================================================
# HONEST ENSEMBLE: rank-average dev OOF scores, judge via eval-honest harness
# ========================================================================================

def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    return pd.Series(arr).rank(pct=True).to_numpy()


def run_ensembles(H: Harness, detector_scores: dict):
    """Test diverse combinations by rank-averaging dev OOF, judged eval-honest.

    detector_scores: {name: (dev_oof_aligned_to_H.dev_pair_ids, eval_scores_aligned_to_H.eval_pair_ids)}
    """
    y = H.y
    w = H.w

    # align all dev oofs to harness pair order
    aligned = {}
    for name, (doof, dids, es, eids) in detector_scores.items():
        dmap = dict(zip(dids, doof))
        arr = np.array([dmap.get(pid, np.nan) for pid in H.dev_pair_ids], dtype=float)
        if np.isnan(arr).any():
            arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
        aligned[name] = arr

    print("\n================ ENSEMBLE TEST (gate = eval_honest_AP) ================")
    combos = [
        ("E_alone", ["E_pu_learn"]),
        ("E+B", ["E_pu_learn", "B_softplay"]),
        ("E+A", ["E_pu_learn", "A_directed"]),
        ("E+B+A", ["E_pu_learn", "B_softplay", "A_directed"]),
        ("E+B+A+D", ["E_pu_learn", "B_softplay", "A_directed", "D_victim"]),
        ("E+D", ["E_pu_learn", "D_victim"]),
        ("B+A+D", ["B_softplay", "A_directed", "D_victim"]),
        ("all5", ["E_pu_learn", "B_softplay", "A_directed", "C_policy_cf", "D_victim"]),
    ]
    results = []
    for label, members in combos:
        present = [m for m in members if m in aligned]
        if not present:
            continue
        ranks = np.mean([_rank_normalize(aligned[m]) for m in present], axis=0)
        conf = average_precision_score(y, ranks)
        eh = weighted_ap(y, ranks, w)
        auc = roc_auc_score(y, ranks)
        results.append((label, len(present), conf, eh, auc))
        print(f"  {label:<14}  members={len(present):d}  eval_honest={eh:.4f}  "
              f"confirmed={conf:.4f}  AUC={auc:.4f}")

    print(f"\n  {'CONSISTENCY':<14}  eval_honest=0.6100  (prior §110 ceiling)")
    print(f"  {'V30 lineage':<14}  eval_honest=~0.79   (= 0.70904 LB)")
    return results


def main() -> int:
    H = Harness()

    # --- build D ---
    print("\n----- building D_victim -----", flush=True)
    d_oof, d_dids, d_es, d_eids = build_detector_D(H)
    d_res = H.judge("D_victim", d_oof, d_dids, d_es, d_eids)
    print(d_res)

    # --- reload round-1 detectors by re-running them (fast: caches + GPU) ---
    from anchor_repro.detector_bakeoff import build_detector_A, build_detector_B, build_detector_E, build_detector_C
    detectors = {}
    for name, fn in [("A_directed", build_detector_A), ("B_softplay", build_detector_B),
                     ("E_pu_learn", build_detector_E), ("C_policy_cf", build_detector_C)]:
        print(f"\n----- rebuilding {name} -----", flush=True)
        doof, dids, es, eids = fn(H)
        res = H.judge(name, doof, dids, es, eids)
        print(res)
        detectors[name] = (doof, dids, es, eids)
    detectors["D_victim"] = (d_oof, d_dids, d_es, d_eids)

    # --- honest ensemble test ---
    ens_results = run_ensembles(H, detectors)

    # save
    out = {"D_victim": {"eval_honest_ap": d_res.eval_honest_ap, "confirmed_ap": d_res.confirmed_ap,
                        "auc": d_res.auc},
           "ensembles": [{"combo": r[0], "n": r[1], "eval_honest": r[2], "confirmed": r[3],
                         "auc": r[4]} for r in ens_results]}
    (CACHE / "_bakeoff_round2_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote _bakeoff_round2_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
