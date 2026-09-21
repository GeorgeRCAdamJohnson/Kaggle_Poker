"""Assemble + emit the two in-lineage confirmed-base candidates for submission.

Both keep V30's behavior + evidence BYTE-IDENTICAL to the 0.70904 shipped pipeline, changing
ONLY risk_score — so any LB delta is attributable to the risk change alone (controlled).

  cand1  hybrid_both_max : 0.5*rank(V30 base) + 0.5*max(rank(V30 union), rank(Eline union))
          (eval-honest 0.9165 ~ V30; confirmed base -> trustworthy estimate; tests if E-line
           specialists help on the PRIVATE population even though they don't on confirmed dev)
  cand2  rank_max_w70    : V30's own arm at higher specialist weight + max union (§117 sweep best,
          eval-honest 0.9213, +0.001 vs shipped; pure in-lineage arm tweak)

Writes both CSVs. Submission is done by the caller after confirmation.
Run:  python -m anchor_repro.submit_inlineage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

CACHE = Path("outputs/poker_collusion/generator_cache")
SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
V30_EVAL_BASE = Path("outputs/poker_collusion/repro_lamhuy/submission.csv")
SHIPPED = SEQ / "candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv"  # 0.70904, source of beh+ev
OUT = CACHE / "hybrid"
EV_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior", *EV_COLS]


def _prank(v):
    return pd.Series(np.asarray(v, np.float64)).rank(method="average", pct=True).to_numpy(np.float32)


def _assemble(risk_df: pl.DataFrame, path: Path):
    """risk_df: pair_id, risk_score. Take behavior+evidence from the shipped 0.70904 CSV."""
    shipped = pl.read_csv(SHIPPED).select(["pair_id", "predicted_behavior", *EV_COLS])
    sub = risk_df.join(shipped, on="pair_id", how="inner").select(SUB_COLS)
    sub = sub.with_columns(pl.col("risk_score").clip(0.0, 1.0))
    assert sub.height == 112540, f"row count {sub.height}"
    assert list(sub.columns) == SUB_COLS
    assert sub["risk_score"].min() >= 0 and sub["risk_score"].max() <= 1
    assert int(sub.null_count().sum_horizontal().item()) == 0
    sub.write_csv(path)
    print(f"  wrote {path} ({sub.height} rows)")


def main() -> int:
    # eval components
    ebase = pl.read_csv(V30_EVAL_BASE).select(["pair_id", "risk_score"])
    espec = pl.read_parquet(SEQ / "eval_specialist_scores.parquet")
    ej = ebase.join(espec, on="pair_id", how="inner")
    eval_ids = ej["pair_id"].to_list()
    base = ej["risk_score"].to_numpy().astype(np.float32)
    v30spec = np.column_stack([ej["specialist_directed"].to_numpy(), ej["specialist_soft"].to_numpy(),
                               ej["specialist_isolation"].to_numpy()]).astype(np.float32)
    ee = pl.read_parquet(CACHE / "eline" / "risk_eval.parquet")
    ee_map = {r["pair_id"]: (r["spec_directed"], r["spec_soft"], r["spec_isolation"]) for r in ee.to_dicts()}
    elspec = np.array([ee_map.get(p, (0., 0., 0.)) for p in eval_ids], np.float32)

    br = _prank(base)
    v30u = _prank(np.clip(v30spec.sum(1), 0, 1))
    elu = _prank(np.clip(elspec.sum(1), 0, 1))
    v30umax = _prank(v30spec.max(1))

    # cand1: hybrid both_max
    cand1 = np.clip(0.5 * br + 0.5 * np.maximum(v30u, elu), 0, 1)
    _assemble(pl.DataFrame({"pair_id": eval_ids, "risk_score": cand1.astype(np.float32)}),
              OUT / "candidate_hybrid_both_max.csv")

    # cand2: V30 rank_max_w70 (base + max-union at w=0.70)
    cand2 = np.clip(0.3 * br + 0.7 * v30umax, 0, 1)
    _assemble(pl.DataFrame({"pair_id": eval_ids, "risk_score": cand2.astype(np.float32)}),
              OUT / "candidate_v30_rank_max_w70.csv")

    print("both candidates assembled (behavior+evidence byte-identical to 0.70904 shipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
