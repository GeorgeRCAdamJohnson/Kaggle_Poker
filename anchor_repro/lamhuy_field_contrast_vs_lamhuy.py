"""Field-contrast eval risk vs lamhuy's ACTUAL persisted risk (real, not the base approximation).

BUILD + MEASURE ONLY. No submission is written or overwritten. This is the
honest limit of what can be measured on eval: there are no eval labels, so no
eval PairAP can be computed here (only the confirmed-label blend sweep in
``lamhuy_field_contrast_blend.py`` supports the +0.0191 lift claim). This
script instead:

* fits the field-contrast model on ALL 1,860 confirmed dev pairs (a
  production-style fit, not an OOF fold) and scores the real 112,540 eval pairs;
* compares that eval-side ranking against lamhuy's ACTUAL persisted risk_score
  (not the base approximation) via Spearman + top-K overlap;
* surfaces the disagreement region for manual inspection, exactly as the
  earlier lamhuy-vs-prior-models disagreement analysis did.

A high overlap with lamhuy would mean the field-contrast signal is likely
redundant with lamhuy's own (lamhuy already computes partner-vs-field
contrasts internally, per the notebook source). A genuinely different eval
ranking is a necessary (not sufficient) condition for the confirmed-label
lift to matter on the real leaderboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from xgboost import XGBClassifier

from anchor_repro.lamhuy_field_contrast import FEATURE_COLUMNS


def _fit_full_model(X: pd.DataFrame, y: np.ndarray, seed: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def build_report(root: Path, top_k: int = 500, region_size: int = 500) -> dict:
    prepared = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    report_dir = root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_field = pd.read_parquet(report_dir / "dev_field_contrast.parquet")
    eval_field = pd.read_parquet(report_dir / "eval_field_contrast.parquet")

    labeled = dev_pairs[dev_pairs["is_labeled"].astype(bool)].merge(
        dev_field, on="pair_id", how="inner", validate="one_to_one"
    )
    y = labeled["label"].astype(int).to_numpy()
    X = labeled[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = _fit_full_model(X, y, seed=5051)

    eval_X = eval_field[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    field_risk = model.predict_proba(eval_X)[:, 1]
    field_frame = pd.DataFrame({"pair_id": eval_field["pair_id"], "field_risk": field_risk})

    lamhuy = pd.read_csv(
        root / "outputs" / "poker_collusion" / "repro_lamhuy" / "submission.csv",
        usecols=["pair_id", "risk_score"],
    ).rename(columns={"risk_score": "lamhuy_risk"})
    merged = field_frame.merge(lamhuy, on="pair_id", how="inner", validate="one_to_one")

    spearman = float(spearmanr(merged["field_risk"], merged["lamhuy_risk"]).statistic)
    field_top = set(merged.nlargest(top_k, "field_risk")["pair_id"])
    lamhuy_top = set(merged.nlargest(top_k, "lamhuy_risk")["pair_id"])
    overlap = len(field_top & lamhuy_top)

    merged["field_pct"] = rankdata(merged["field_risk"], method="average") / len(merged)
    merged["lamhuy_pct"] = rankdata(merged["lamhuy_risk"], method="average") / len(merged)
    merged["field_advantage"] = merged["field_pct"] - merged["lamhuy_pct"]

    field_high_lamhuy_low = merged.nlargest(region_size, "field_advantage")
    lamhuy_high_field_low = merged.nsmallest(region_size, "field_advantage")

    field_high_lamhuy_low.to_csv(report_dir / "field_high_lamhuy_low.csv", index=False)
    lamhuy_high_field_low.to_csv(report_dir / "lamhuy_high_field_low.csv", index=False)

    return {
        "note": (
            "No eval labels exist, so no eval PairAP is computed here. This measures ranking "
            "behavior only: how much the field-contrast eval risk agrees/disagrees with "
            "lamhuy's ACTUAL persisted risk_score. The confirmed-label +0.0191 blend lift "
            "(lamhuy_field_contrast_blend.py) is the only quantitative lift claim; this is a "
            "necessary-but-not-sufficient follow-up check."
        ),
        "n_eval_pairs": int(len(merged)),
        "spearman_field_vs_lamhuy": spearman,
        "top_k": int(top_k),
        "top_k_overlap": overlap,
        "top_k_overlap_fraction": overlap / top_k,
        "field_high_lamhuy_low_mean_field_pct": float(field_high_lamhuy_low["field_pct"].mean()),
        "field_high_lamhuy_low_mean_lamhuy_pct": float(field_high_lamhuy_low["lamhuy_pct"].mean()),
        "lamhuy_high_field_low_mean_field_pct": float(lamhuy_high_field_low["field_pct"].mean()),
        "lamhuy_high_field_low_mean_lamhuy_pct": float(lamhuy_high_field_low["lamhuy_pct"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--region-size", type=int, default=500)
    args = parser.parse_args()
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root, top_k=args.top_k, region_size=args.region_size)
    (report_dir / "_field_contrast_vs_lamhuy_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
