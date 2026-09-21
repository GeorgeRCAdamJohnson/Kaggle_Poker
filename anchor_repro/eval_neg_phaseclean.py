"""Option 2: the CLEAN version of the eval-negatives experiment — PHASE-INVARIANT features only.

The naive version leaked: positives are all dev-phase, negatives all eval-phase, and features carry
phase structure (dev-NEG vs eval-NEG AUC 0.91) -> model learned PHASE, scored a fake 1.0000.

Fix (the only construction that tests the real idea without the bug): keep ONLY features where
dev-NEG and eval-NEG are INDISTINGUISHABLE (per-feature AUC in [0.5-tol, 0.5+tol]). On those, phase
CANNOT be the discriminator. Then train positives-vs-eval-negatives. If positives STILL separate from
the eval population, that separation is real collusion signal drawn against the true scored population
(the §108/§110 fix), not phase.

Honest prior: §110 found the drift-clean subset did NOT lift eval-honest above V30. This tests it
cleanly with the strong V30 basis + eval negatives (never done). LB is the arbiter if it clears.
Run:  python -m anchor_repro.eval_neg_phaseclean
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from anchor_repro.gpu_config import xgb_params
from anchor_repro.eval_honest_harness import Harness, D, CACHE

DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
EVAL_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
OUT = CACHE / "eval_negatives"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
SKIP = {"pair_id", "hand_id", "table_id"}
PHASE_TOL = 0.05   # keep features with dev-NEG-vs-eval-NEG AUC in [0.45, 0.55]


def _pair_aggregate(path: Path, feats: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(path)
    agg = [pl.len().alias("shared_hands")]
    agg += [pl.col(c).mean().alias(f"{c}_mean") for c in feats]
    agg += [pl.col(c).max().alias(f"{c}_max") for c in feats]
    agg += [pl.col(c).quantile(0.95, interpolation="nearest").alias(f"{c}_p95") for c in feats]
    agg += [pl.col(c).top_k(5).mean().alias(f"{c}_top5") for c in feats]
    return lf.group_by("pair_id").agg(agg).collect(engine="streaming")


def main() -> int:
    H = Harness()
    t = time.time()
    feats = [c for c in pl.scan_parquet(DEV_HF).collect_schema().names() if c not in SKIP]
    dev_pair = _pair_aggregate(DEV_HF, feats)
    eval_pair = _pair_aggregate(EVAL_HF, feats)
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev_pair = dev_pair.join(labels, on="pair_id", how="left")
    print(f"aggregated dev {dev_pair.height:,} / eval {eval_pair.height:,} ({time.time()-t:.0f}s)")

    ALL_FEATS = [c for c in dev_pair.columns if c not in {"pair_id", "label", "shared_hands"}]
    rng = np.random.default_rng(SEED)

    # ---- per-feature dev-NEG vs eval-NEG AUC -> keep PHASE-INVARIANT features ----
    dev_neg = dev_pair.filter(pl.col("label") == 0)
    n_probe = min(40000, eval_pair.height)
    eval_probe = eval_pair[rng.choice(eval_pair.height, n_probe, replace=False)]
    Dn = dev_neg.select(ALL_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0)
    En = eval_probe.select(ALL_FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0)
    yphase = np.concatenate([np.ones(len(Dn)), np.zeros(len(En))])
    clean = []
    aucs = {}
    for c in ALL_FEATS:
        v = np.concatenate([Dn[c].to_numpy(), En[c].to_numpy()])
        try:
            a = roc_auc_score(yphase, v); a = max(a, 1 - a)
        except Exception:
            a = 1.0
        aucs[c] = a
        if a <= 0.5 + PHASE_TOL:
            clean.append(c)
    print(f"phase-invariant features (dev-NEG~eval-NEG AUC<= {0.5+PHASE_TOL}): {len(clean)}/{len(ALL_FEATS)}")
    if len(clean) < 5:
        # relax if too few
        clean = [c for c in ALL_FEATS if aucs[c] <= 0.60]
        print(f"  too few; relaxed to AUC<=0.60 -> {len(clean)}")

    # ---- confirm the phase signal is GONE on the clean subset ----
    Xg = pd.concat([Dn[clean], En[clean]], ignore_index=True)
    gm = xgb.train(xgb_params(objective="binary:logistic", eval_metric="auc", max_depth=4, eta=0.1,
                              subsample=0.8, colsample_bytree=0.8),
                   xgb.DMatrix(Xg, label=yphase), num_boost_round=120)
    ph_auc = roc_auc_score(yphase, gm.predict(xgb.DMatrix(Xg)))
    print(f"phase separability on CLEAN subset AUC = {ph_auc:.4f} (want ~0.5-0.6; was 0.91 on all feats)")

    # ---- train pos vs eval-neg on CLEAN features only ----
    pos = dev_pair.filter(pl.col("label") == 1)
    Xpos = pos.select(clean).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    eval_all = eval_pair.select(clean).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    dev_all = dev_pair.select(clean).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    n_neg = 40000; n_bags = 6
    eval_score = np.zeros(eval_pair.height, np.float64); dev_score = np.zeros(dev_pair.height, np.float64)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.03,
                    subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=6.0)
    for bag in range(n_bags):
        Xb = eval_pair[rng.choice(eval_pair.height, n_neg, replace=False)].select(clean).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        Xt = pd.concat([Xpos, Xb], ignore_index=True); yt = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xb))])
        m = xgb.train({**pp, "scale_pos_weight": float(len(Xb)/len(Xpos))}, xgb.DMatrix(Xt, label=yt), num_boost_round=350)
        eval_score += m.predict(xgb.DMatrix(eval_all)) / n_bags
        dev_score += m.predict(xgb.DMatrix(dev_all)) / n_bags
    print(f"trained clean eval-negative PU risk ({time.time()-t:.0f}s)")

    dev_pair = dev_pair.with_columns(pl.Series("risk", dev_score))
    conf = dev_pair.filter(pl.col("label").is_in([0, 1]))
    yc = conf["label"].to_numpy(); rc = conf["risk"].to_numpy()
    cauc = roc_auc_score(yc, rc)
    print(f"confirmed pos-vs-neg AUC = {cauc:.4f}  AP = {average_precision_score(yc, rc):.4f}  "
          f"(1.0 = still leaking; <1.0 & meaningful = real)")

    escore = (eval_score - eval_score.min()) / (eval_score.max() - eval_score.min() + 1e-12)
    r = H.judge("eval_neg_phaseclean", conf["risk"].to_numpy(), conf["pair_id"].to_list(),
                escore, eval_pair["pair_id"].to_list())
    print(f"\n=== PHASE-CLEAN EVAL-NEGATIVES (eval-honest) ===")
    print(f"  {r}")
    print(f"  bar = V30 real LB 0.70904 / eval-honest V30 ~0.79")
    pl.DataFrame({"pair_id": eval_pair["pair_id"].to_list(), "risk_score": escore.astype(np.float32)}).write_parquet(OUT / "eval_neg_phaseclean_risk.parquet")
    (OUT / "_phaseclean_summary.json").write_text(json.dumps({
        "n_clean_feats": len(clean), "phase_auc_clean": float(ph_auc),
        "confirmed_auc": float(cauc), "eval_honest_ap": r.eval_honest_ap,
        "clean_feats": clean}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
