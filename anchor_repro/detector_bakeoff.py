"""Fresh-slate PU detector BAKE-OFF, judged on the shared eval-honest harness (no V30 anchor).

Builds several INDEPENDENT, first-principles pair detectors from the Layer-1/Layer-2 caches
and scores them ALL through anchor_repro.eval_honest_harness.Harness so results are
apples-to-apples. The gate is eval_honest_ap (importance-weighted). Confirmed dev AP is
diagnostic only (it historically over-reads).

Detectors:
  A  directed-transfer / value-flow   — who wins/loses chips, directionally, per pair.
  B  passivity / soft-play            — muted mutual aggression + showdown passivity.
  C  policy-counterfactual (Seam 1)   — per-player self-deviation (reuses gen_policy_counterfactual).
  E  pure PU-learning                 — positives vs UNLABELED EVAL pairs as negatives (the only
                                        detector trained against the real eval distribution).

Each detector = a builder returning (dev_oof, dev_pair_ids, eval_scores, eval_pair_ids).
The harness does all scoring/blending.

Run:  python -m anchor_repro.detector_bakeoff
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, CACHE, D, SEED

HAND_CTX = CACHE / "hand_context.parquet"


# ----------------------------------------------------------------------------------------
# shared bookkeeping
# ----------------------------------------------------------------------------------------

def _labels() -> pl.DataFrame:
    return pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])


def _pair_table_map(layer2_path: Path) -> pl.DataFrame:
    hp = pl.scan_parquet(HAND_CTX).select(["hand_id", "table_id"])
    return (
        pl.scan_parquet(layer2_path).select(["pair_id", "hand_id"])
        .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
        .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
        .select(["pair_id", "table_id"]).collect()
    )


def _grouped_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                 depth: int = 4, rounds: int = 350) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=depth,
                        eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=6.0)
    for tr, va in sgkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)},
                      xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=rounds)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    # full-data model for eval scoring
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    full = xgb.train({**params, "scale_pos_weight": float(spw)},
                     xgb.DMatrix(X, label=y), num_boost_round=rounds)
    return oof, full


# ----------------------------------------------------------------------------------------
# Layer-2 pair-aggregate feature builders (family-specific views, no V30 features)
# ----------------------------------------------------------------------------------------

def _agg_from_layer2(layer2_path: Path, cols: list[str]) -> pl.DataFrame:
    df = pl.scan_parquet(layer2_path)
    return df.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        *[pl.col(c).mean().alias(f"{c}_mean") for c in cols],
        *[pl.col(c).max().alias(f"{c}_max") for c in cols],
        *[pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in cols],
        *[(pl.col(c) > 0).mean().alias(f"{c}_rate") for c in cols],
    ).collect(engine="streaming")


# Directed-transfer view: directional net flow between members.
_A_COLS = ["dir_signal", "dir_opposite_flow", "dir_transfer_bb", "abs_flow_bb",
           "net_product", "loser_net_bb", "winner_net_bb"]
# Soft-play view: muted mutual aggression, showdown passivity.
_B_COLS = ["soft_signal", "soft_both_sd", "soft_no_agg", "soft_late_checks",
           "both_commit", "pair_agg", "agg_asym"]


def build_detector_A(H: Harness):
    """A: raw directed-transfer / value-flow."""
    lab = _labels()
    dev = _agg_from_layer2(CACHE / "dev_layer2.parquet", _A_COLS)
    tbl = _pair_table_map(CACHE / "dev_layer2.parquet")
    dev = dev.join(lab.select(["pair_id", "label"]), on="pair_id", how="left").join(tbl, on="pair_id", how="left")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")]
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].to_numpy()
    oof, full = _grouped_oof(Xd, y, g)
    evf = _agg_from_layer2(CACHE / "eval_layer2.parquet", _A_COLS)
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    es = full.predict(xgb.DMatrix(Xe))
    return oof, dev["pair_id"].to_list(), es, evf["pair_id"].to_list()


def build_detector_B(H: Harness):
    """B: passivity / soft-play."""
    lab = _labels()
    dev = _agg_from_layer2(CACHE / "dev_layer2.parquet", _B_COLS)
    tbl = _pair_table_map(CACHE / "dev_layer2.parquet")
    dev = dev.join(lab.select(["pair_id", "label"]), on="pair_id", how="left").join(tbl, on="pair_id", how="left")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")]
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].to_numpy()
    oof, full = _grouped_oof(Xd, y, g)
    evf = _agg_from_layer2(CACHE / "eval_layer2.parquet", _B_COLS)
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    es = full.predict(xgb.DMatrix(Xe))
    return oof, dev["pair_id"].to_list(), es, evf["pair_id"].to_list()


def build_detector_C(H: Harness):
    """C: policy-counterfactual (Seam 1). Reuses gen_policy_counterfactual feature builders."""
    from anchor_repro.gen_policy_counterfactual import (
        _player_hand_states, _player_state_policy, _shared_residuals, _pair_cf_features)
    V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
    V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
    states = _player_hand_states()
    policy = _player_state_policy(states).collect(engine="streaming").lazy()
    lab = _labels()
    dev_uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"])
    dev_cf = _pair_cf_features(_shared_residuals(lab.select(["pair_id", "player_1", "player_2"]).lazy(),
                                                 dev_uni, states, policy))
    tbl = _pair_table_map(CACHE / "dev_layer2.parquet")
    dev = dev_cf.join(lab.select(["pair_id", "label"]), on="pair_id", how="left").join(tbl, on="pair_id", how="left")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")]
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].to_numpy()
    oof, full = _grouped_oof(Xd, y, g)
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    eval_uni = pl.scan_parquet(V30_EVAL).select(["pair_id", "hand_id"])
    parts = []
    for start in range(0, eval_pairs.height, 20000):
        pf = eval_pairs.slice(start, 20000).lazy()
        parts.append(_pair_cf_features(_shared_residuals(pf, eval_uni, states, policy)))
    evf = pl.concat(parts, how="vertical_relaxed")
    Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    es = full.predict(xgb.DMatrix(Xe))
    return oof, dev["pair_id"].to_list(), es, evf["pair_id"].to_list()


def build_detector_E(H: Harness):
    """E: pure PU-learning — positives vs sampled UNLABELED EVAL pairs as negatives.

    The only detector trained against the ACTUAL eval distribution. Uses the same stable
    Layer-2 basis as the harness weights (49 feats). Confirmed positives (label==1) are the
    positive class; a random sample of eval pairs are treated as negatives (PU assumption:
    eval is ~99.8% negative). Scored OOF on dev by held-out folds of positives + fresh eval
    negatives; eval scored by the full model.
    """
    from anchor_repro.gen_pair_detector import pair_features
    lab = _labels()
    dev = pair_features(CACHE / "dev_layer2.parquet").join(lab.select(["pair_id", "label"]), on="pair_id", how="left")
    evf = pair_features(CACHE / "eval_layer2.parquet")
    feats = [c for c in dev.columns if c not in ("pair_id", "label", "table_id") and not c.endswith("_sum")]

    Xd_all = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    Xe_all = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    pos_idx = np.where(y == 1)[0]
    rng = np.random.default_rng(SEED)

    # dev OOF (leak-free): positives get a HELD-OUT score (predicted only by folds that did
    # NOT train on them). Dev NEGATIVES are never training rows (negatives always come from
    # eval), so they are scored by the full ensemble of fold models -> also leak-free.
    from sklearn.model_selection import KFold
    oof = np.zeros(len(y), dtype=np.float32)
    neg_dev_idx = np.where(y == 0)[0]
    neg_accum = np.zeros(len(neg_dev_idx), dtype=np.float32)
    kf = KFold(5, shuffle=True, random_state=SEED)
    n_neg = 20000
    for tr_pos, va_pos in kf.split(pos_idx):
        train_pos = pos_idx[tr_pos]
        held_pos = pos_idx[va_pos]
        neg_sample = rng.choice(len(Xe_all), n_neg, replace=False)
        X_tr = np.vstack([Xd_all.to_numpy()[train_pos], Xe_all.to_numpy()[neg_sample]])
        y_tr = np.concatenate([np.ones(len(train_pos)), np.zeros(n_neg)])
        spw = n_neg / max(len(train_pos), 1)
        m = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4,
                                 eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                                 reg_lambda=6.0, scale_pos_weight=float(spw)),
                      xgb.DMatrix(X_tr, label=y_tr), num_boost_round=350)
        oof[held_pos] = m.predict(xgb.DMatrix(Xd_all.to_numpy()[held_pos]))   # held-out positives
        neg_accum += m.predict(xgb.DMatrix(Xd_all.to_numpy()[neg_dev_idx])) / 5.0  # avg for negatives
    oof[neg_dev_idx] = neg_accum
    # full model for eval
    neg_sample = rng.choice(len(Xe_all), n_neg, replace=False)
    X_tr = np.vstack([Xd_all.to_numpy()[pos_idx], Xe_all.to_numpy()[neg_sample]])
    y_tr = np.concatenate([np.ones(len(pos_idx)), np.zeros(n_neg)])
    spw = n_neg / max(len(pos_idx), 1)
    full = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4,
                                eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                                reg_lambda=6.0, scale_pos_weight=float(spw)),
                     xgb.DMatrix(X_tr, label=y_tr), num_boost_round=350)
    es = full.predict(xgb.DMatrix(Xe_all.to_numpy()))
    return oof, dev["pair_id"].to_list(), es, evf["pair_id"].to_list()


BUILDERS = {"A_directed": build_detector_A, "B_softplay": build_detector_B,
            "C_policy_cf": build_detector_C, "E_pu_learn": build_detector_E}


def main() -> int:
    H = Harness()
    results = {}
    dev_oofs, eval_scores = {}, {}
    for name, fn in BUILDERS.items():
        print(f"\n----- building {name} -----", flush=True)
        doof, dids, es, eids = fn(H)
        res = H.judge(name, doof, dids, es, eids)
        print(res, flush=True)
        results[name] = res
        # keep rank-normalized scores for ensembling (aligned to eval order)
        dev_oofs[name] = dict(zip(dids, doof))
        eval_scores[name] = dict(zip(eids, es))

    print("\n================ BAKE-OFF SUMMARY (gate = eval_honest_AP) ================")
    print(f"{'detector':<14} {'eval_honest':>12} {'confirmed':>11} {'AUC':>7} "
          f"{'blend_w':>8} {'pu_stress':>10} {'vs_base':>9}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].eval_honest_ap):
        print(f"{name:<14} {r.eval_honest_ap:>12.4f} {r.confirmed_ap:>11.4f} {r.auc:>7.4f} "
              f"{r.best_blend_w:>8.2f} {r.best_blend_pu_stress:>10.5f} "
              f"{r.best_blend_pu_stress-r.baseline_pu_stress:>+9.5f}")
    print(f"{'CONSISTENCY':<14} {0.61:>12.4f}  (prior §110 ceiling)")
    print(f"{'V30 lineage':<14} {0.79:>12.4f}  (prior eval-honest estimate, = 0.70904 LB)")

    out = {name: {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap,
                  "auc": r.auc, "best_blend_w": r.best_blend_w,
                  "best_blend_pu_stress": r.best_blend_pu_stress,
                  "baseline_pu_stress": r.baseline_pu_stress, "blend_table": r.blend_table}
           for name, r in results.items()}
    (CACHE / "_bakeoff_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote _bakeoff_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
