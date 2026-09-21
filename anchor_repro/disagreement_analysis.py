"""Analyze the winning lamhuy ranking against the earlier detector ensemble.

This is a BUILD + MEASURE report only. It creates no submission and does not
choose a blend. The purpose is to identify complementary ranking regions before
any new model or leaderboard experiment is attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


MODEL_FILES = {
    "lamhuy": "outputs/poker_collusion/repro_lamhuy/submission.csv",
    "honghanh": "outputs/poker_collusion/repro_064469/submission.csv",
    "nomannic": "outputs/poker_collusion/repro_nomannic/submission.csv",
    "floor": "outputs/poker_collusion/submission_best_044519.csv",
}


def _load_rankings(root: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model, relative_path in MODEL_FILES.items():
        frame = pd.read_csv(root / relative_path, usecols=["pair_id", "risk_score"])
        frame = frame.rename(columns={"risk_score": model})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="pair_id", how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError("no comparable submission rankings were loaded")
    for model in MODEL_FILES:
        merged[f"{model}_pct"] = rankdata(merged[model], method="average") / len(merged)
    merged["baseline_pct"] = merged[["honghanh_pct", "nomannic_pct", "floor_pct"]].mean(axis=1)
    merged["lamhuy_advantage"] = merged["lamhuy_pct"] - merged["baseline_pct"]
    return merged


def build_report(root: Path, top_k: int = 500, region_size: int = 500) -> dict:
    frame = _load_rankings(root)
    models = list(MODEL_FILES)
    correlations = {
        left: {
            right: float(spearmanr(frame[left], frame[right]).statistic)
            for right in models
            if right != left
        }
        for left in models
    }
    top_overlap = {}
    lamhuy_top = set(frame.nlargest(top_k, "lamhuy")["pair_id"])
    for model in models:
        top = set(frame.nlargest(top_k, model)["pair_id"])
        top_overlap[model] = {
            "intersection": len(lamhuy_top & top),
            "fraction_of_lamhuy_top": len(lamhuy_top & top) / top_k,
        }

    lamhuy_high_baseline_low = frame.nlargest(
        region_size, ["lamhuy_advantage", "lamhuy"]
    )
    baseline_high_lamhuy_low = frame.nsmallest(
        region_size, ["lamhuy_advantage", "lamhuy"]
    )

    def records(region: pd.DataFrame) -> list[dict]:
        columns = [
            "pair_id",
            "lamhuy",
            "honghanh",
            "nomannic",
            "floor",
            "lamhuy_pct",
            "baseline_pct",
            "lamhuy_advantage",
        ]
        return region[columns].to_dict(orient="records")

    return {
        "note": "Ranking disagreement report only; no submission or blend was created.",
        "n_pairs": int(len(frame)),
        "models": MODEL_FILES,
        "spearman": correlations,
        "lamhuy_top_k": int(top_k),
        "top_k_overlap_with_lamhuy": top_overlap,
        "regions": {
            "lamhuy_high_baseline_low": records(lamhuy_high_baseline_low),
            "baseline_high_lamhuy_low": records(baseline_high_lamhuy_low),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--region-size", type=int, default=500)
    args = parser.parse_args()

    if args.top_k <= 0 or args.region_size <= 0:
        raise ValueError("--top-k and --region-size must be positive")
    output_dir = args.poker_root / "outputs" / "poker_collusion" / "disagreement_lamhuy"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.poker_root, top_k=args.top_k, region_size=args.region_size)
    output_path = output_dir / "_disagreement_summary.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    frame = _load_rankings(args.poker_root)
    frame.sort_values("lamhuy_advantage", ascending=False).head(args.region_size).to_csv(
        output_dir / "lamhuy_high_baseline_low.csv", index=False
    )
    frame.sort_values("lamhuy_advantage", ascending=True).head(args.region_size).to_csv(
        output_dir / "baseline_high_lamhuy_low.csv", index=False
    )
    print(json.dumps({"summary": str(output_path), "n_pairs": report["n_pairs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())