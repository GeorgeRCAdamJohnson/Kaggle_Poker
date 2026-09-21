"""Last submission: rank-blend of V30 (0.70904, proven anchor) + eval-neg risk (0.52735, decorrelated).

Rationale: eval-neg is a REAL signal (held-out AUC 0.95) but base-rate-diluted to 0.527, and it is
DECORRELATED from V30 (Spearman 0.49). Standard ensemble play: if the two catch DIFFERENT true
positives, a rank-blend exceeds either alone. Most likely lands between 0.53 and 0.71; small chance of
complementarity gain. V30-dominant weights (V30 is the proven anchor; eval-neg is the risky add).

risk = (1-w)*rank(V30) + w*rank(eval-neg), swept; behavior+evidence BYTE-IDENTICAL to 0.70904.
We pick a V30-dominant w (0.15-0.25) — NOT the harness argmax (harness leans on eval-neg population,
would over-pick it). Writes the chosen blend for submission.

Run:  python -m anchor_repro.assemble_blend_submission
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

SHIPPED = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
EVALNEG = Path("outputs/poker_collusion/evalneg_submission.csv")
OUT = Path("outputs/poker_collusion/blend_submission.csv")
W = 0.20  # V30-dominant: 80% V30 rank + 20% eval-neg rank


def main() -> int:
    v30 = pd.read_csv(SHIPPED); v30["pair_id"] = v30["pair_id"].astype(str)
    en = pd.read_csv(EVALNEG)[["pair_id", "risk_score"]]; en["pair_id"] = en["pair_id"].astype(str)
    en = en.rename(columns={"risk_score": "en_risk"})
    m = v30.merge(en, on="pair_id", how="left")
    assert m["en_risk"].notna().all(), "eval-neg risk missing for some pairs"

    rv = pd.Series(m["risk_score"].to_numpy()).rank(pct=True).to_numpy()
    re = pd.Series(m["en_risk"].to_numpy()).rank(pct=True).to_numpy()
    blend = (1 - W) * rv + W * re
    # normalize to [0,1]
    blend = (blend - blend.min()) / (blend.max() - blend.min() + 1e-12)

    out = v30.copy()
    out["risk_score"] = blend.astype(np.float32)   # ONLY risk changes; behavior+evidence identical
    print(f"blend w={W}: 80% V30 rank + 20% eval-neg rank")
    print(f"Spearman(blend, V30) = {spearmanr(blend, m['risk_score']).correlation:.4f}")
    print(f"Spearman(blend, eval-neg) = {spearmanr(blend, m['en_risk']).correlation:.4f}")
    print(f"risk range [{blend.min():.3f},{blend.max():.3f}] mean {blend.mean():.4f}")
    assert out["risk_score"].between(0, 1).all() and not out["risk_score"].isna().any()
    assert (out["predicted_behavior"] == v30["predicted_behavior"]).all()
    assert (out["evidence_hand_1"] == v30["evidence_hand_1"]).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows); behavior+evidence identical to 0.70904")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
