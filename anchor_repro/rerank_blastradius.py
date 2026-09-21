"""Pick the conditional-rerank candidate by BLAST RADIUS vs V30 (local, honest — no degenerate proxy).

The dev within-band AP proxy is degenerate (dev top band is ~all positive). We cannot locally rank
the 25 candidates. But we CAN measure, for each, how much it perturbs V30's ranking among the TOP
pairs (where PairAP is decided): how many of V30's top-K pairs get reordered, and the rank churn.
This bounds downside: the minimal-perturbation candidate that still reranks a meaningful band is the
safest single LB probe of the mechanism.

Reports, per candidate: band size, #pairs whose top-500 membership changes vs V30, Spearman on the
top 2000, so we choose a justified minimal probe (narrow band, modest weight) rather than blind.
Run:  python -m anchor_repro.rerank_blastradius
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V30 = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
BAND_DIR = Path("outputs/poker_collusion/top_band_rerank")


def main() -> int:
    v30 = pd.read_csv(V30)[["pair_id", "risk_score"]].rename(columns={"risk_score": "v30"})
    v30["pair_id"] = v30["pair_id"].astype(str)
    v30_rank = v30["v30"].rank(ascending=False)
    top500_v30 = set(v30.loc[v30_rank <= 500, "pair_id"])
    top2000_idx = v30_rank <= 2000

    man = json.loads((BAND_DIR / "_manifest.json").read_text())
    rows = []
    for entry in man:
        # actual filenames have every '.' replaced by 'p' (incl the extension -> 'pcsv')
        cand_path = BAND_DIR / f"candidate_top{entry['top_fraction']:.3f}_w{entry['weight']:.2f}.csv".replace(".", "p")
        if not cand_path.is_file():
            continue
        c = pd.read_csv(cand_path)[["pair_id", "risk_score"]].rename(columns={"risk_score": "c"})
        c["pair_id"] = c["pair_id"].astype(str)
        m = v30.merge(c, on="pair_id")
        c_rank = m["c"].rank(ascending=False)
        top500_c = set(m.loc[c_rank <= 500, "pair_id"])
        churn500 = len(top500_v30.symmetric_difference(top500_c)) // 2
        sp_top = spearmanr(m.loc[top2000_idx.values, "v30"], m.loc[top2000_idx.values, "c"]).correlation
        rows.append({"band": entry["top_fraction"], "w": entry["weight"], "inside": entry["inside_rows"],
                     "top500_churn": churn500, "spearman_top2000": round(float(sp_top), 4)})

    df = pd.DataFrame(rows).sort_values(["band", "w"])
    print(df.to_string(index=False))
    # justified pick: smallest band, w that moves SOME pairs but keeps top500 churn modest (<~60)
    safe = df[(df.top500_churn > 5) & (df.top500_churn <= 60)].sort_values(["band", "w"])
    print("\nJustified minimal probes (moves some pairs, low churn):")
    print(safe.head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
