"""Decide if conditional top-band reranking can work, BEFORE spending a submission.

Conditional reranking (top_band_rerank.py) only helps if, INSIDE V30's top band, the eval-neg score
ranks the TRUE POSITIVES higher than V30 does. If eval-neg is no better than V30 within the band, the
rerank just injects sparse-population noise there (the blend failure in miniature).

We measure this on the ONLY ground truth we have: the dev confirmed labels (372 pos / 1488 neg).
Using each model's DEV scores:
  1. Rank all confirmed dev pairs by V30 dev risk; take the top band (same fractions as the probe).
  2. Within that band, compute AP of V30 vs eval-neg vs the blend at each weight.
  3. If blend AP > V30 AP within-band (beyond noise), conditional rerank should help -> submit the
     best band/weight. If not, it's a no-go and we save the slot.

CAVEAT: dev is dense (~20% pos) vs eval (0.2%), so within-band AP here is optimistic in absolute
terms, but the RELATIVE comparison (does eval-neg reorder band positives better than V30?) is the
transferable question. Reports the full grid so we pick a justified candidate, not a blind one.
Run:  python -m anchor_repro.rerank_diagnose
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from anchor_repro.eval_honest_harness import D, CACHE

OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
EVAL_NEG_DEV = CACHE / "eval_negatives" / "eval_neg_dev_scores.parquet"  # may not exist; fallback below


def _rank(a): return pd.Series(a).rank(pct=True).to_numpy()


def main() -> int:
    labels = pd.read_csv(D / "development_labels.csv")[["pair_id", "label"]]
    labels["pair_id"] = labels["pair_id"].astype(str)

    oof = pd.read_parquet(OOF).drop_duplicates("pair_id")
    oof["pair_id"] = oof["pair_id"].astype(str)
    v30 = oof[["pair_id", "rank_sum_w50"]].rename(columns={"rank_sum_w50": "v30"})

    # eval-neg DEV scores: rebuild if not cached (train pos vs eval-neg, score dev). Reuse the
    # eval_neg_holdout_test path would retrain; instead recompute dev scores here quickly.
    en_dev = _dev_eval_neg_scores()
    df = labels.merge(v30, on="pair_id", how="left").merge(en_dev, on="pair_id", how="left")
    df = df.dropna(subset=["v30", "en"])
    y = df["label"].to_numpy()
    print(f"confirmed dev pairs with both scores: {len(df)}  positives={int(y.sum())}")

    vr = _rank(df["v30"].to_numpy())
    er = _rank(df["en"].to_numpy())

    print("\n=== WITHIN-BAND AP: V30 vs eval-neg vs blend (dev confirmed labels) ===")
    print(f"{'band':>7} {'n':>5} {'pos':>4} {'V30_AP':>8} {'EN_AP':>8}  best-blend(w): AP")
    results = []
    for frac in (0.05, 0.10, 0.20, 0.35, 0.50):  # dev is dense; use larger bands than eval probe
        thr = np.quantile(vr, 1 - frac)
        inb = vr >= thr
        yb = y[inb]
        if yb.sum() < 3:
            continue
        lv = _rank(df.loc[inb, "v30"].to_numpy())
        le = _rank(df.loc[inb, "en"].to_numpy())
        ap_v = average_precision_score(yb, lv)
        ap_e = average_precision_score(yb, le)
        best = (ap_v, 0.0)
        for w in (0.10, 0.20, 0.35, 0.50, 0.75):
            apb = average_precision_score(yb, (1 - w) * lv + w * le)
            if apb > best[0]:
                best = (apb, w)
        flag = "  <-- blend beats V30" if best[0] > ap_v + 1e-4 else ""
        print(f"{frac:>7.2f} {int(inb.sum()):>5} {int(yb.sum()):>4} {ap_v:>8.4f} {ap_e:>8.4f}  w={best[1]:.2f}: {best[0]:.4f}{flag}")
        results.append({"band": frac, "n": int(inb.sum()), "pos": int(yb.sum()),
                        "v30_ap": ap_v, "en_ap": ap_e, "best_blend_ap": best[0], "best_w": best[1]})

    # overall verdict
    any_gain = any(r["best_blend_ap"] > r["v30_ap"] + 0.005 for r in results)
    print(f"\nVERDICT: {'blend improves within-band AP somewhere -> conditional rerank MAY help' if any_gain else 'NO within-band AP gain -> conditional rerank will NOT help; do not submit'}")
    (CACHE / "eval_negatives" / "_rerank_diag.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def _dev_eval_neg_scores() -> pd.DataFrame:
    """Train pos vs eval-neg on V30 pair-aggregate features; return dev-pair scores."""
    import polars as pl, xgboost as xgb
    from anchor_repro.gpu_config import xgb_params
    DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
    EVAL_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
    SKIP = {"pair_id", "hand_id", "table_id"}
    feats = [c for c in pl.scan_parquet(DEV_HF).collect_schema().names() if c not in SKIP]

    def agg(path):
        lf = pl.scan_parquet(path)
        a = ([pl.len().alias("shared_hands")]
             + [pl.col(c).mean().alias(f"{c}_mean") for c in feats]
             + [pl.col(c).max().alias(f"{c}_max") for c in feats]
             + [pl.col(c).quantile(0.95, interpolation="nearest").alias(f"{c}_p95") for c in feats]
             + [pl.col(c).top_k(5).mean().alias(f"{c}_top5") for c in feats])
        return lf.group_by("pair_id").agg(a).collect(engine="streaming")

    dev = agg(DEV_HF); ev = agg(EVAL_HF)
    lab = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label"])
    dev = dev.join(lab, on="pair_id", how="left")
    FEATS = [c for c in dev.columns if c not in {"pair_id", "label", "shared_hands"}]
    pos = dev.filter(pl.col("label") == 1).to_pandas()
    def M(d): return d[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    Xpos = M(pos)
    dev_all = dev.select(FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    rng = np.random.default_rng(42)
    dev_score = np.zeros(dev.height)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=5, eta=0.03,
                    subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=6.0)
    for _ in range(6):
        Xb = ev[rng.choice(ev.height, 40000, replace=False)].select(FEATS).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        Xt = pd.concat([Xpos, Xb], ignore_index=True); yt = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xb))])
        m = xgb.train({**pp, "scale_pos_weight": float(len(Xb)/len(Xpos))}, xgb.DMatrix(Xt, label=yt), num_boost_round=350)
        dev_score += m.predict(xgb.DMatrix(dev_all)) / 6
    return pd.DataFrame({"pair_id": dev["pair_id"].to_list(), "en": dev_score})


if __name__ == "__main__":
    raise SystemExit(main())
