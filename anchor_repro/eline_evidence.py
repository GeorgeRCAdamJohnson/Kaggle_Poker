"""E-LINE EVIDENCE column — learned per-hand evidence ranker (the one genuinely-new column).

Ranks the 5 evidence hands per pair. Built on the cached per-(pair,hand) evidence candidates
(no actions rescan). Follows the dossier's documented DID-TRANSFER recipe (§6.4): learned ranker
at MODEST depth (d2-d3), per-pair-relative features (within-pair percentile-rank + z-score of the
signals), grouped leave-one-pool-out CV. The dossier notes the oracle retriever ceiling is 0.867,
so RANKING is the bottleneck, not the candidate gate.

Trains on is_planted (dev), scores every eval candidate, emits top-5 hand_ids per pair with
NO_EVIDENCE padding and an ascending-hand-id tie-break (Req 8.5/8.8). Measures dev Evidence-MAP@5.

Run:  python -m anchor_repro.eline_evidence
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import CACHE, D

FC = Path("outputs/poker_collusion/feature_cache")
OUT = CACHE / "eline"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NO_EV = "NO_EVIDENCE"

BASE_SIG = ["value_flow", "abs_value_flow", "aggression_asymmetry", "soft_inv", "isolation",
            "mi_conflict", "flag_directed", "flag_soft", "flag_iso", "n_flags"]


def _enrich(df: pl.DataFrame) -> pl.DataFrame:
    """Per-pair-relative features (dossier §6.4 DID-TRANSFER): within-pair pct-rank + z-score."""
    rel = ["abs_value_flow", "isolation", "mi_conflict", "soft_inv", "aggression_asymmetry"]
    exprs = []
    for c in rel:
        exprs.append(((pl.col(c).rank() / pl.len()).over("pair_id")).alias(f"{c}_pctrank"))
        exprs.append(((pl.col(c) - pl.col(c).mean().over("pair_id"))
                      / (pl.col(c).std().over("pair_id") + 1e-6)).alias(f"{c}_z"))
    exprs.append(pl.len().over("pair_id").alias("pair_size"))
    return df.with_columns(exprs)


def _evidence_map5(pred_frame: pl.DataFrame, truth: pl.DataFrame) -> float:
    """Mean over true-positive pairs of AP@5; denom = min(#planted,5). Matches metric §1.2."""
    truth_by_pair = truth.filter(pl.col("is_planted") == 1).group_by("pair_id").agg(
        pl.col("hand_id").alias("relevant"))
    rel_map = {r["pair_id"]: set(r["relevant"]) for r in truth_by_pair.to_dicts()}
    # top-5 predicted hands per pair by score desc, hand_id asc tie-break
    ranked = (pred_frame.sort(["pair_id", "score", "hand_id"], descending=[False, True, False])
              .group_by("pair_id", maintain_order=True).head(5))
    pred_map = {}
    for pid, sub in ranked.group_by("pair_id", maintain_order=True):
        pred_map[pid[0] if isinstance(pid, tuple) else pid] = sub["hand_id"].to_list()
    aps = []
    for pid, rel in rel_map.items():
        if not rel:
            continue
        preds = pred_map.get(pid, [])
        hits = 0; psum = 0.0
        for rank, h in enumerate(preds[:5], start=1):
            if h in rel:
                hits += 1; psum += hits / rank
        aps.append(psum / min(len(rel), 5))
    return float(np.mean(aps)) if aps else 0.0


def main() -> int:
    print(f"device={xgb_device()}")
    dev = _enrich(pl.read_parquet(FC / "dev_evidence_candidates.parquet"))
    feats = BASE_SIG + [c for c in dev.columns if c.endswith("_pctrank") or c.endswith("_z")] + ["pair_size"]
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    yd = dev["is_planted"].to_numpy().astype(np.int8)
    groups = dev["pool"].to_numpy()
    print(f"dev candidates={dev.height} planted={int(yd.sum())} feats={len(feats)}")

    # leave-one-pool-out OOF ranker (modest depth per §6.4)
    best_map5, best_cfg, best_oof = -1, None, None
    for depth, rounds in [(2, 300), (3, 300), (3, 400)]:
        oof = np.zeros(len(yd), np.float32)
        gkf = GroupKFold(5)
        for tr, va in gkf.split(Xd, yd, groups):
            spw = (yd[tr] == 0).sum() / max((yd[tr] == 1).sum(), 1)
            m = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=depth,
                                     eta=0.05, subsample=0.85, colsample_bytree=0.8, min_child_weight=2,
                                     reg_lambda=2.5, scale_pos_weight=float(spw)),
                          xgb.DMatrix(Xd.iloc[tr], label=yd[tr]), num_boost_round=rounds)
            oof[va] = m.predict(xgb.DMatrix(Xd.iloc[va]))
        pf = dev.select(["pair_id", "hand_id"]).with_columns(pl.Series("score", oof))
        m5 = _evidence_map5(pf, dev.select(["pair_id", "hand_id", "is_planted"]))
        print(f"  ranker d{depth} it{rounds}: dev Evidence-MAP@5={m5:.4f}")
        if m5 > best_map5:
            best_map5, best_cfg, best_oof = m5, (depth, rounds), oof
    print(f"BEST evidence ranker: d{best_cfg[0]} it{best_cfg[1]} MAP@5={best_map5:.4f} (oracle ceiling 0.867)")

    # full model, score eval candidates
    depth, rounds = best_cfg
    spw = (yd == 0).sum() / max((yd == 1).sum(), 1)
    full = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=depth,
                                eta=0.05, subsample=0.85, colsample_bytree=0.8, min_child_weight=2,
                                reg_lambda=2.5, scale_pos_weight=float(spw)),
                     xgb.DMatrix(Xd, label=yd), num_boost_round=rounds)
    t = time.time()
    ev = _enrich(pl.read_parquet(FC / "eval_evidence_candidates.parquet"))
    Xe = ev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    ev = ev.select(["pair_id", "hand_id"]).with_columns(pl.Series("score", full.predict(xgb.DMatrix(Xe))))
    print(f"scored {ev.height:,} eval candidates ({time.time()-t:.0f}s)")

    # top-5 per pair -> wide evidence columns with NO_EVIDENCE padding
    ranked = (ev.sort(["pair_id", "score", "hand_id"], descending=[False, True, False])
              .group_by("pair_id", maintain_order=True).head(5))
    rows = []
    for pid, sub in ranked.group_by("pair_id", maintain_order=True):
        pid = pid[0] if isinstance(pid, tuple) else pid
        hands = sub["hand_id"].to_list()[:5]
        hands += [NO_EV] * (5 - len(hands))
        rows.append({"pair_id": pid, **{f"evidence_hand_{i+1}": hands[i] for i in range(5)}})
    ev_wide = pl.DataFrame(rows)
    ev_wide.write_parquet(OUT / "evidence_eval.parquet")
    (OUT / "_evidence_summary.json").write_text(json.dumps(
        {"best_cfg": {"depth": best_cfg[0], "rounds": best_cfg[1]}, "dev_map5": best_map5,
         "oracle_ceiling": 0.867}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}/evidence_eval.parquet ({ev_wide.height:,} pairs), _evidence_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
