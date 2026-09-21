"""Conditional top-band reranking, FIXED so it truly stays within the band (v1 scrambled the top).

v1 bug: inside the band it wrote fresh local 0-1 percentiles into the global score array, so band
pairs (which should sit at global rank ~0.998-1.0) got values like 0.3 and fell BELOW thousands of
outside pairs -> top500 churn ~224 from a 226-pair band, Spearman(top2000) went NEGATIVE. That is
rank-scramble, not conditional rerank.

FIX: outside the band, keep V30's global percentile EXACTLY. Inside the band, reorder ONLY among the
band pairs and map the result back into the band's OWN global-rank window [lo, hi] = the percentile
span V30 assigned to those pairs. So band pairs stay above every outside pair and merely permute among
themselves by the (1-w)*V30 + w*eval-neg blend. This is the mechanism as intended: preserve V30
everywhere, rerank only inside its top band, without ejecting band members from the top.

Emits a small, sane candidate grid + a blast-radius report (should show LOW top-500 churn now, since
band pairs never leave the top). Build-only; never auto-submits.
Run:  python -m anchor_repro.top_band_rerank_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V30 = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
EVAL_NEG = Path("outputs/poker_collusion/generator_cache/eval_negatives/eval_negatives_risk.parquet")
OUT = Path("outputs/poker_collusion/top_band_rerank_v2")


def _pct(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(pct=True).to_numpy()


def main() -> int:
    v30 = pd.read_csv(V30); v30["pair_id"] = v30["pair_id"].astype(str)
    en = pd.read_parquet(EVAL_NEG)[["pair_id", "risk_score"]].rename(columns={"risk_score": "en"})
    en["pair_id"] = en["pair_id"].astype(str)
    f = v30.merge(en, on="pair_id", how="left", validate="one_to_one")
    assert f["en"].notna().all()

    vpct = _pct(f["risk_score"].to_numpy())        # V30 global percentile (the base ordering)
    epct = _pct(f["en"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)

    v30_rank_desc = pd.Series(f["risk_score"].to_numpy()).rank(ascending=False)
    top500_v30 = set(f.loc[v30_rank_desc <= 500, "pair_id"])
    top2000 = (v30_rank_desc <= 2000).to_numpy()

    manifest = []
    for frac in (0.005, 0.01, 0.02, 0.05):
        inb = vpct >= 1.0 - frac
        lo, hi = vpct[inb].min(), vpct[inb].max()   # the band's OWN global-percentile window
        for w in (0.20, 0.35, 0.50):
            score = vpct.copy()
            # blend inside band, then RESCALE the blended order back into [lo, hi] so band pairs
            # keep their global position (above all outside pairs) and only permute among themselves.
            blend_in = (1 - w) * _pct(f.loc[inb, "risk_score"].to_numpy()) + w * _pct(f.loc[inb, "en"].to_numpy())
            r = pd.Series(blend_in).rank(pct=True).to_numpy()   # order within band, 0..1
            score[inb] = lo + r * (hi - lo)
            score = np.clip(score, 0, 1)

            cand = v30.copy()
            cand["risk_score"] = score.astype(np.float32)
            name = f"cand_top{int(frac*1000):03d}permil_w{int(w*100):02d}.csv"
            cand.to_csv(OUT / name, index=False)

            crank = pd.Series(score).rank(ascending=False)
            churn = len(top500_v30.symmetric_difference(set(f.loc[crank <= 500, "pair_id"]))) // 2
            sp = spearmanr(f.loc[top2000, "risk_score"], score[top2000]).correlation
            manifest.append({"file": name, "band": frac, "w": w, "inside": int(inb.sum()),
                             "top500_churn": churn, "spearman_top2000": round(float(sp), 4)})

    dfm = pd.DataFrame(manifest)
    (OUT / "_manifest.json").write_text(dfm.to_json(orient="records", indent=2), encoding="utf-8")
    print(dfm.to_string(index=False))
    print("\nExpect LOW churn now (band pairs never leave the top). Behavior+evidence byte-identical to V30.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
