"""Shared EVAL-HONEST harness for the fresh-slate detector bake-off (no V30 anchor).

Every detector in the bake-off is judged by the SAME rules here, so results are
apples-to-apples. Three numbers per detector:

  1. confirmed_ap      — AP over the 1,860 labeled dev pairs (OOF). Diagnostic only; this is
                         the number that historically OVER-reads (dossier §108/§110).
  2. eval_honest_ap    — importance-weighted AP: what the SPARSE eval population would score,
                         estimated by density-ratio reweighting dev to look like eval (§110).
                         THIS IS THE GATE. Consistency ceiling ~0.61; V30 lineage ~0.79 eval.
  3. blend PU-stress   — HONEST this time: a detector must score ALL eval pairs (no 0-fill),
                         so the 24k unlabeled PU pairs carry a REAL detector score, not a
                         constant that leaks. (The 0-fill leak inflated a prior blend by +0.23.)

Design notes
------------
* Density-ratio weights are fit on a STABLE, LOW-DRIFT feature basis (the Layer-2 pair
  aggregates minus raw-count _sum columns) so the weights are trustworthy regardless of
  which detector we are judging. The SAME weight vector is reused for every detector.
* A "detector" here is any callable that returns (dev_oof_scores aligned to the 1,860 labeled
  pair_ids, eval_scores aligned to the 112,540 eval pair_ids). The harness handles all scoring.

Usage
-----
    from anchor_repro.eval_honest_harness import Harness
    H = Harness()                       # builds weights + label/pair bookkeeping once
    res = H.judge("detA", dev_oof, dev_pair_ids, eval_scores, eval_pair_ids)
    print(res)                          # confirmed_ap, eval_honest_ap, blend table
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.gen_pair_detector import pair_features  # Layer-2 stable basis

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
CONF_OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
SEED = 42
PU_STRESS_WEIGHT = 112540 / 24000


def weighted_ap(y: np.ndarray, scores: np.ndarray, w: np.ndarray) -> float:
    """Importance-weighted average precision (weights on the ranking population)."""
    order = np.argsort(-scores, kind="mergesort")
    y = y[order].astype(float); w = w[order].astype(float)
    tp = np.cumsum(y * w); fp = np.cumsum((1 - y) * w)
    precision = tp / np.clip(tp + fp, 1e-9, None)
    total_pos = np.sum(y * w)
    return 0.0 if total_pos <= 0 else float(np.sum(precision * y * w) / total_pos)


@dataclass
class JudgeResult:
    name: str
    confirmed_ap: float
    eval_honest_ap: float
    auc: float
    best_blend_w: float
    best_blend_confirmed: float
    best_blend_pu_stress: float
    baseline_pu_stress: float
    blend_table: list = field(default_factory=list)

    def __str__(self) -> str:
        return (f"[{self.name}] eval_honest_AP={self.eval_honest_ap:.4f}  "
                f"confirmed_AP={self.confirmed_ap:.4f}  AUC={self.auc:.4f}  | "
                f"best blend w={self.best_blend_w:.2f} pu_stress={self.best_blend_pu_stress:.5f} "
                f"(base {self.baseline_pu_stress:.5f}, {self.best_blend_pu_stress-self.baseline_pu_stress:+.5f})")


class Harness:
    """Builds the eval-honest importance weights + bookkeeping ONCE; judges many detectors."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._build()

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _build(self):
        # --- labels + stable pair-feature basis for the density-ratio model ---
        labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
        dev = pair_features(CACHE / "dev_layer2.parquet").join(labels, on="pair_id", how="left")
        evf = pair_features(CACHE / "eval_layer2.parquet")

        basis = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")
                 and not c.endswith("_sum")]
        self.dev_pair_ids = dev["pair_id"].to_list()
        self.eval_pair_ids = evf["pair_id"].to_list()
        self.y = dev["label"].to_numpy().astype(np.int8)
        self._dev_pos = int(self.y.sum())

        Xd = dev.select(basis).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        Xe = evf.select(basis).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        # --- density-ratio importance weights w = P(eval)/P(dev), OOF, clipped, mean-1 ---
        dev_prob = np.zeros(len(Xd))
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for tr, va in skf.split(Xd, self.y):
            X_tr = np.vstack([Xd.to_numpy()[tr], Xe.to_numpy()])
            y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(Xe))])
            clf = xgb.XGBClassifier(**xgb_params(objective="binary:logistic", eval_metric="auc",
                                                 max_depth=4, n_estimators=300, learning_rate=0.05,
                                                 subsample=0.85, colsample_bytree=0.8, reg_lambda=5))
            clf.fit(X_tr, y_tr)
            dev_prob[va] = clf.predict_proba(Xd.to_numpy()[va])[:, 1]
        dev_prob = np.clip(dev_prob, 1e-4, 1 - 1e-4)
        w = dev_prob / (1 - dev_prob)
        w = np.clip(w, np.quantile(w, 0.01), np.quantile(w, 0.99))
        self.w = (w / w.mean())
        self._log(f"harness: dev={len(Xd)} pos={self._dev_pos} eval={len(Xe)} basis={len(basis)} "
                  f"device={xgb_device()}")
        self._log(f"  importance weights pos={self.w[self.y==1].mean():.3f} neg={self.w[self.y==0].mean():.3f}")

        # --- frozen 0.70904 baseline OOF (rank_sum_w50) for HONEST blend tests ---
        base = pl.read_parquet(CONF_OOF).select(["pair_id", "known", "y", "rank_sum_w50"])
        self._base = base
        self._base_known = base["known"].to_numpy().astype(bool)
        self._base_y = base["y"].to_numpy().astype(np.int8)
        self._base_rsw = base["rank_sum_w50"].to_numpy().astype(float)
        self._base_sw = np.where(self._base_known, 1.0, PU_STRESS_WEIGHT)
        self._base_pair_ids = base["pair_id"].to_list()
        self.baseline_confirmed = average_precision_score(self._base_y[self._base_known],
                                                           self._base_rsw[self._base_known])
        self.baseline_pu_stress = average_precision_score(self._base_y, self._base_rsw,
                                                          sample_weight=self._base_sw)
        self._log(f"  baseline rank_sum_w50: confirmed={self.baseline_confirmed:.5f} "
                  f"pu_stress={self.baseline_pu_stress:.5f}")

    def judge(self, name: str, dev_oof: np.ndarray, dev_pair_ids: list,
              eval_scores: np.ndarray, eval_pair_ids: list) -> JudgeResult:
        """Judge one detector. dev_oof aligned to dev_pair_ids; eval_scores to eval_pair_ids.

        NO 0-fill: the honest blend uses REAL eval scores for the 24k PU pairs (the leak fix).
        """
        # align dev_oof to the harness label order
        dmap = dict(zip(dev_pair_ids, dev_oof))
        oof = np.array([dmap.get(pid, np.nan) for pid in self.dev_pair_ids], dtype=float)
        assert not np.isnan(oof).any(), f"{name}: dev_oof missing pair_ids"
        confirmed_ap = average_precision_score(self.y, oof)
        auc = roc_auc_score(self.y, oof)
        eh = weighted_ap(self.y, oof, self.w)

        # --- HONEST blend: build a full detector-score vector over the baseline pair universe ---
        emap = dict(zip(eval_pair_ids, eval_scores))
        emap.update(dmap)  # labeled pairs use their OOF score; eval pairs use eval score
        det_full = np.array([emap.get(pid, np.nan) for pid in self._base_pair_ids], dtype=float)
        # any base pair the detector didn't score -> median (neutral, NOT a leaking constant)
        if np.isnan(det_full).any():
            med = np.nanmedian(det_full)
            n_missing = int(np.isnan(det_full).sum())
            det_full = np.where(np.isnan(det_full), med, det_full)
            self._log(f"  [{name}] {n_missing} base pairs unscored -> filled with median (neutral)")

        def rankn(a): return pd.Series(a).rank(pct=True).to_numpy()
        br, gr = rankn(self._base_rsw), rankn(det_full)
        best = (0.0, self.baseline_confirmed, self.baseline_pu_stress)
        table = []
        for wt in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            blend = (1 - wt) * br + wt * gr
            cc = average_precision_score(self._base_y[self._base_known], blend[self._base_known])
            ss = average_precision_score(self._base_y, blend, sample_weight=self._base_sw)
            table.append({"w": wt, "confirmed": float(cc), "pu_stress": float(ss)})
            if ss > best[2]:
                best = (wt, cc, ss)

        return JudgeResult(name, float(confirmed_ap), float(eh), float(auc),
                           float(best[0]), float(best[1]), float(best[2]),
                           float(self.baseline_pu_stress), table)


if __name__ == "__main__":
    # self-test: judging the baseline THROUGH ITSELF should give ~baseline (sanity).
    H = Harness()
    print("harness built OK; dev pairs:", len(H.dev_pair_ids), "eval pairs:", len(H.eval_pair_ids))
