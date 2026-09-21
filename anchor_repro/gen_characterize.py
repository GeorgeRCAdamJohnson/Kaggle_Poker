"""Task 3: characterize the generator per family on the Layer-2 dev cache.

For each family, measure how well the (corrected) features separate the PLANTED evidence
hands from the pair's other shared hands:
  * univariate planted-AP per feature (which features carry the planting signal),
  * best-predicate coverage/purity,
  * per-family EvidenceMAP@5 with a quick GPU model (vs V30 evidence ~0.5067).

The key check: did the corrected isolation features (iso_*) lift coordinated_isolation
from the diagnosed 0.07 dead spot?

Run:  python -m anchor_repro.gen_characterize
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
FAMILIES = ["directed_transfer", "soft_play", "coordinated_isolation"]
SEED = 42


def _evidence_map5(pair_ids, planted, scores) -> float:
    frame = pd.DataFrame({"p": pair_ids, "y": planted, "s": scores})
    aps = []
    for _, g in frame.groupby("p", sort=False):
        rel = int(g["y"].sum())
        if rel == 0:
            continue
        top = g.sort_values("s", ascending=False, kind="mergesort").head(5)["y"].to_numpy()
        hits = psum = 0.0
        for r, h in enumerate(top, start=1):
            if h:
                hits += 1
                psum += hits / r
        aps.append(psum / min(rel, 5))
    return float(np.mean(aps)) if aps else 0.0


def main() -> int:
    feats = json.loads((CACHE / "layer2_feature_columns.json").read_text())
    dev = pl.read_parquet(CACHE / "dev_layer2.parquet")
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "label", "behavior_family"])
    ctx = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"]).collect()
    dev = dev.join(labels, on="pair_id", how="left").join(ctx, on="hand_id", how="left")
    dev = dev.filter(pl.col("label") == 1)
    print(f"dev positive-pair hands={dev.height:,} planted={int(dev['is_planted'].sum())} device={xgb_device()}")

    iso_feats = [c for c in feats if c.startswith(("iso_", "n_outsider", "table_fold", "member_pot"))]
    print(f"\ncorrected isolation feature set ({len(iso_feats)}): {iso_feats}")

    # 1. Univariate planted-AP per family (focus on isolation recovery).
    print("\n=== UNIVARIATE planted-AP by family (top 8) ===")
    for fam in FAMILIES:
        sub = dev.filter(pl.col("behavior_family") == fam)
        y = sub["is_planted"].to_numpy()
        rows = []
        for c in feats:
            v = sub[c].fill_null(0).to_numpy().astype(float)
            if np.unique(v).size < 2:
                continue
            ap = max(average_precision_score(y, v), average_precision_score(y, -v))
            rows.append((c, ap))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{fam} (base {y.mean():.4f}):")
        for c, a in rows[:8]:
            tag = " <-ISO" if c in iso_feats else ""
            print(f"   {c:32s} {a:.4f}{tag}")

    # 2. Per-family GPU model EvidenceMAP@5, table-grouped OOF.
    print("\n=== PER-FAMILY EvidenceMAP@5 (GPU, table-grouped OOF) vs V30 0.5067 ===")
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=6,
                        eta=0.05, subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0)
    overall = {}
    for fam in FAMILIES:
        sub = dev.filter(pl.col("behavior_family") == fam)
        X = sub.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        y = sub["is_planted"].to_numpy().astype(np.int8)
        groups = sub["table_id"].to_numpy()
        pair_ids = sub["pair_id"].to_numpy()
        oof = np.zeros(len(sub), dtype=np.float32)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr, va in sgkf.split(X, y, groups):
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            m = xgb.train({**params, "scale_pos_weight": float(spw)},
                          xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
            oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
        e = _evidence_map5(pair_ids, y, oof)
        ap = average_precision_score(y, oof)
        overall[fam] = e
        print(f"   {fam:24s} EvidenceMAP@5={e:.4f}  planted-AP={ap:.4f}  pairs={sub['pair_id'].n_unique()}")

    # 3. Combined all-family EvidenceMAP@5 (single model, family-agnostic).
    X = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["is_planted"].to_numpy().astype(np.int8)
    groups = dev["table_id"].to_numpy()
    pair_ids = dev["pair_id"].to_numpy()
    oof = np.zeros(len(dev), dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)},
                      xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    e_all = _evidence_map5(pair_ids, y, oof)
    print(f"\n   GLOBAL (family-agnostic) EvidenceMAP@5 = {e_all:.4f}  [V30 ~0.5067]")

    summary = {
        "per_family_evidence_map5": overall, "global_evidence_map5": float(e_all),
        "v30_reference": 0.5067, "prev_isolation_dead": 0.0702, "device": xgb_device(),
    }
    (CACHE / "_characterize_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nprev isolation was 0.0702; now {overall.get('coordinated_isolation'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
