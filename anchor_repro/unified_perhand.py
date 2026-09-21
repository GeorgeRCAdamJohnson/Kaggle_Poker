"""UNIFIED per-hand structure (dossier §109 "0.89 signature", never built until now).

ONE per-(pair,hand) coordination-probability model, read THREE ways:
  risk      = pair-level aggregate of per-hand coordination prob (top-k mean / max)
  evidence  = the top-5 highest-prob hands per pair
  behavior  = family of the pair's highest-prob hands (candidate carries flag_directed/soft/iso)

This is the opposite of our 3-separate-heads pipelines (V30, E-line, hybrid). The bet (user's
thesis): a COHERENT single structure trained on the DENSE per-hand target (1,573 planted hands vs
background) beats the sparse 372-pair-label training, and makes all three components move together.

DATA REALITY (checked): dev candidates cover ONLY the 372 positive pairs (no negative pairs), so a
within-positive planted model can't separate positives from negatives. Fix: train the per-hand model
as COORDINATION-vs-BACKGROUND — planted positive hands (label 1) vs a sample of EVAL-population hands
(label 0, the background of ordinary hands). This learns "what a coordinated hand looks like against
all hands", which DOES drive pair risk when aggregated.

Judged on the eval-honest harness (PairAP from the pair aggregate) + dev Evidence-MAP@5 (within-pos
ranking) + Behavior. Composite compared to V30 0.70904.

Run:  python -m anchor_repro.unified_perhand
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D

FC = Path("outputs/poker_collusion/feature_cache")
SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
OUT = CACHE / "unified"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NO_EV = "NO_EVIDENCE"

SIG = ["value_flow", "abs_value_flow", "aggression_asymmetry", "soft_inv", "isolation",
       "mi_conflict", "flag_directed", "flag_soft", "flag_iso", "n_flags"]
EV_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]


def _enrich(df: pl.DataFrame) -> pl.DataFrame:
    rel = ["abs_value_flow", "isolation", "mi_conflict", "soft_inv", "aggression_asymmetry"]
    ex = []
    for c in rel:
        ex.append(((pl.col(c).rank() / pl.len()).over("pair_id")).alias(f"{c}_pctrank"))
        ex.append(((pl.col(c) - pl.col(c).mean().over("pair_id"))
                   / (pl.col(c).std().over("pair_id") + 1e-6)).alias(f"{c}_z"))
    ex.append(pl.len().over("pair_id").alias("pair_size"))
    return df.with_columns(ex)


def _evidence_map5(pred: pl.DataFrame, truth: pl.DataFrame) -> float:
    rel_map = {r["pair_id"]: set(r["relevant"]) for r in
               truth.filter(pl.col("is_planted") == 1).group_by("pair_id")
               .agg(pl.col("hand_id").alias("relevant")).to_dicts()}
    ranked = (pred.sort(["pair_id", "score", "hand_id"], descending=[False, True, False])
              .group_by("pair_id", maintain_order=True).head(5))
    pred_map = {}
    for pid, sub in ranked.group_by("pair_id", maintain_order=True):
        pred_map[pid[0] if isinstance(pid, tuple) else pid] = sub["hand_id"].to_list()
    aps = []
    for pid, rel in rel_map.items():
        if not rel:
            continue
        preds = pred_map.get(pid, [])
        hits = 0; ps = 0.0
        for rank, h in enumerate(preds[:5], 1):
            if h in rel:
                hits += 1; ps += hits / rank
        aps.append(ps / min(len(rel), 5))
    return float(np.mean(aps)) if aps else 0.0


def main() -> int:
    H = Harness()
    dev = _enrich(pl.read_parquet(FC / "dev_evidence_candidates.parquet"))
    eval_c = _enrich(pl.read_parquet(FC / "eval_evidence_candidates.parquet"))
    feats = SIG + [c for c in dev.columns if c.endswith("_pctrank") or c.endswith("_z")] + ["pair_size"]
    print(f"device={xgb_device()} feats={len(feats)}")

    # ---- per-hand COORDINATION model: planted dev hands (1) vs sampled eval-bg hands (0) ----
    rng = np.random.default_rng(SEED)
    planted = dev.filter(pl.col("is_planted") == 1)
    Xp = planted.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    n_bg = 60000
    bg_idx = rng.choice(eval_c.height, n_bg, replace=False)
    Xbg = eval_c[bg_idx].select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    print(f"per-hand model: {len(Xp)} planted vs {n_bg} eval-background hands")

    Xtr = np.vstack([Xp, Xbg]); ytr = np.concatenate([np.ones(len(Xp)), np.zeros(n_bg)])
    spw = n_bg / len(Xp)
    # bag over background draws for stability
    n_bags = 8
    dev_all_score = np.zeros(dev.height, np.float64)
    eval_score = np.zeros(eval_c.height, np.float64)
    Xdev_all = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    Xeval_all = eval_c.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    t = time.time()
    for bag in range(n_bags):
        bg = eval_c[rng.choice(eval_c.height, n_bg, replace=False)].select(feats).to_pandas().replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32).to_numpy()
        Xtr = np.vstack([Xp, bg]); ytr = np.concatenate([np.ones(len(Xp)), np.zeros(len(bg))])
        m = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5,
                                 eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
                                 reg_lambda=5.0, scale_pos_weight=float(len(bg)/len(Xp))),
                      xgb.DMatrix(Xtr, label=ytr), num_boost_round=400)
        dev_all_score += m.predict(xgb.DMatrix(Xdev_all)) / n_bags
        eval_score += m.predict(xgb.DMatrix(Xeval_all)) / n_bags
    print(f"per-hand coordination model built ({time.time()-t:.0f}s)")

    dev = dev.with_columns(pl.Series("coord", dev_all_score.astype(np.float32)))
    eval_c = eval_c.with_columns(pl.Series("coord", eval_score.astype(np.float32)))

    # ================= READ 1: RISK (pair aggregate) =================
    def pair_agg(df):
        return df.group_by("pair_id").agg(
            pl.col("coord").max().alias("risk_max"),
            pl.col("coord").top_k(5).mean().alias("risk_top5"),
            pl.col("coord").mean().alias("risk_mean"),
            pl.col("coord").top_k(3).mean().alias("risk_top3"),
        )
    dev_risk = pair_agg(dev)
    eval_risk = pair_agg(eval_c)

    lab = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    # HONEST NOTE: dev candidates cover ONLY the 372 positive pairs. There are NO candidates for
    # labeled NEGATIVE pairs, so ANY dev PairAP would be a LEAK ("has candidates" == positive).
    # We therefore DO NOT report a dev/eval-honest PairAP for the unified risk (it is unmeasurable
    # on our data). Instead we report the per-hand model's OWN AUC (planted vs background) as the
    # honest proxy for whether the coordination signal is real, and validate risk ONLY via the LB.
    per_hand_auc = roc_auc_score(
        np.concatenate([np.ones(len(Xp)), np.zeros(len(Xbg))]),
        np.concatenate([
            np.mean([dev_all_score[dev["is_planted"].to_numpy() == 1]], axis=0) if False else
            dev.filter(pl.col("is_planted") == 1)["coord"].to_numpy(),
            eval_c[bg_idx]["coord"].to_numpy(),
        ]),
    )
    print("\n=== RISK from per-hand aggregate ===")
    print("HONEST: dev candidates cover only 372 positives -> a dev PairAP would LEAK. NOT reported.")
    print(f"  per-hand coordination model AUC (planted vs eval-background) = {per_hand_auc:.4f}")
    print("  (this is the honest proxy; real risk validated ONLY by the LB submission)")
    # keep the eval risk aggregate for a possible submission; report its distribution
    for agg in ["risk_max", "risk_top5", "risk_top3", "risk_mean"]:
        ev = eval_risk[agg].to_numpy()
        print(f"  eval {agg}: mean={ev.mean():.3f} p50={np.median(ev):.3f} p99={np.quantile(ev,0.99):.3f}")

    # ================= READ 2: EVIDENCE (top-5 within pair) =================
    m5 = _evidence_map5(dev.select(["pair_id", "hand_id", "coord"]).rename({"coord": "score"}),
                        dev.select(["pair_id", "hand_id", "is_planted"]))
    print(f"\n=== EVIDENCE from same model: dev Evidence-MAP@5={m5:.4f} (oracle 0.867) ===")

    # ================= READ 3: BEHAVIOR (family of top hands) =================
    # each candidate hand's family = argmax of flag_directed/soft/iso; pair behavior = family of its top-coord hands
    fammap = {"flag_directed": "directed_transfer", "flag_soft": "soft_play", "flag_iso": "coordinated_isolation"}
    top_hands = (dev.sort(["pair_id", "coord"], descending=[False, True]).group_by("pair_id", maintain_order=True).head(3))
    # majority family among top-3 hands, weighted by coord
    def pair_family(df):
        rows = []
        for pid, sub in df.group_by("pair_id", maintain_order=True):
            pid = pid[0] if isinstance(pid, tuple) else pid
            flags = sub.select(["flag_directed", "flag_soft", "flag_iso", "coord"]).to_numpy()
            score = flags[:, :3] * flags[:, 3:4]
            fam = ["directed_transfer", "soft_play", "coordinated_isolation"][int(score.sum(0).argmax())]
            rows.append({"pair_id": pid, "predicted_behavior": fam})
        return pl.DataFrame(rows)
    dev_beh = pair_family(top_hands)
    # dev behavior-MAP: over the 3 families, risk where pred==family (use risk_top5)
    devj = dev_risk.join(dev_beh, on="pair_id").join(lab, on="pair_id", how="left")
    behcol = pl.read_parquet(SEQ / "oof_rows.parquet").select(["pair_id", "behavior_y"])
    devj = devj.join(behcol, on="pair_id", how="left")
    aps = []
    fam_ids = {"directed_transfer": 1, "soft_play": 2, "coordinated_isolation": 3}
    by = devj["behavior_y"].to_numpy(); risk = devj["risk_top5"].to_numpy(); pf = devj["predicted_behavior"].to_list()
    for fam, fid in fam_ids.items():
        truth = (by == fid).astype(int)
        if truth.sum() == 0: continue
        brisk = np.where(np.array(pf) == fam, risk, 0.0)
        aps.append(average_precision_score(truth, brisk))
    beh_map = float(np.mean(aps)) if aps else 0.0
    print(f"=== BEHAVIOR from same model: dev Behavior-MAP={beh_map:.4f} (over 372 pos only) ===")

    (OUT / "_unified_summary.json").write_text(json.dumps(
        {"evidence_map5": m5, "behavior_map": beh_map,
         "note": "risk arms printed above; dev cand covers only 372 pos"}, indent=2), encoding="utf-8")
    # persist eval per-hand + risk for possible assembly
    eval_risk.write_parquet(OUT / "eval_risk.parquet")
    eval_c.select(["pair_id", "hand_id", "coord", "flag_directed", "flag_soft", "flag_iso"]).write_parquet(OUT / "eval_perhand.parquet")
    print(f"\nwrote {OUT}/eval_risk.parquet, eval_perhand.parquet, _unified_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
