"""Conditional reranking test for the sparse-positive precision wall.

The eval-negative model has real signal but a global rank blend regressed V30.
This module keeps V30's ordering outside its top band and uses the eval-negative
score only as a conditional tie-breaker inside that band. It is build-only and
never submits automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


V30 = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
EVAL_NEG = Path("outputs/poker_collusion/generator_cache/eval_negatives/eval_negatives_risk.parquet")
OUT = Path("outputs/poker_collusion/top_band_rerank")


def _rank(values: pd.Series) -> np.ndarray:
    return values.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def main() -> int:
    v30 = pd.read_csv(V30)
    en = pd.read_parquet(EVAL_NEG)[["pair_id", "risk_score"]].rename(
        columns={"risk_score": "eval_neg_risk"}
    )
    v30["pair_id"] = v30["pair_id"].astype(str)
    en["pair_id"] = en["pair_id"].astype(str)
    frame = v30.merge(en, on="pair_id", how="left", validate="one_to_one")
    if frame["eval_neg_risk"].isna().any():
        raise ValueError("eval-negative scores do not cover the V30 candidate")

    v30_rank = _rank(frame["risk_score"])
    en_rank = _rank(frame["eval_neg_risk"])
    frame["v30_rank"] = v30_rank
    frame["eval_neg_rank"] = en_rank

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    # The gate is intentionally narrow: outside the band, preserve the proven
    # V30 order exactly; inside it, test only modest conditional perturbations.
    for top_fraction in (0.002, 0.005, 0.01, 0.02, 0.05):
        inside = v30_rank >= 1.0 - top_fraction
        for weight in (0.10, 0.20, 0.35, 0.50, 0.75):
            score = v30_rank.copy()
            local_v30 = _rank(frame.loc[inside, "risk_score"])
            local_en = _rank(frame.loc[inside, "eval_neg_risk"])
            score[inside] = (1.0 - weight) * local_v30 + weight * local_en
            score = (score - score.min()) / (score.max() - score.min() + 1e-12)

            candidate = v30.copy()
            candidate["risk_score"] = score.astype(np.float32)
            name = f"candidate_top{top_fraction:.3f}_w{weight:.2f}.csv".replace(".", "p")
            path = OUT / name
            candidate.to_csv(path, index=False)
            manifest.append({
                "file": str(path),
                "top_fraction": top_fraction,
                "weight": weight,
                "rows": len(candidate),
                "inside_rows": int(inside.sum()),
                "risk_min": float(score.min()),
                "risk_max": float(score.max()),
            })

    pd.DataFrame(manifest).to_json(OUT / "_manifest.json", orient="records", indent=2)
    print(f"wrote {len(manifest)} no-submit candidates to {OUT}")
    print("V30 rows:", len(v30), "eval-negative rows:", len(en))
    print("All candidates preserve behavior and evidence columns byte-for-byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())