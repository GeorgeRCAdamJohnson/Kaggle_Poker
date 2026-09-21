"""HYBRID risk: E-line's strong sequence specialists onto V30's CONFIRMED base via arm machinery.

The E-line specialists are individually strong (directed 0.93, soft 0.94, iso 0.76) and
transport-diagnosed, but the pure E-line's PU BASE is weak (0.65) and its harness estimate ran
~0.23 HOT on the LB (§120). This grafts the E-line specialists onto V30's CONFIRMED-trained base
(the strong base that gives V30 its tight, trustworthy local->LB mapping). Because the base is
confirmed-trained, the harness estimate here should be TRUSTWORTHY (unlike the pure E-line).

Arms (all rank-normalized, V30's percentile-rank + blend machinery):
  - V30 shipped:            0.5*rank(base) + 0.5*rank(V30_spec_union)        [= 0.9203 reference]
  - hybrid_eline:           (1-w)*rank(base) + w*rank(Eline_spec_union)
  - hybrid_both:            base + mean of V30 and E-line specialist unions
  - hybrid_max3:            base + max over {V30 union, Eline union}

Judged on the shared eval-honest harness. Persists the best hybrid eval risk for assembly.
Run:  python -m anchor_repro.hybrid_risk
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from anchor_repro.eval_honest_harness import Harness, CACHE

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
V30_EVAL_BASE = Path("outputs/poker_collusion/repro_lamhuy/submission.csv")
ELINE = CACHE / "eline"
OUT = CACHE / "hybrid"
OUT.mkdir(parents=True, exist_ok=True)


def _prank(v):
    return pd.Series(np.asarray(v, np.float64)).rank(method="average", pct=True).to_numpy(np.float32)


def main() -> int:
    H = Harness()

    # ---------- DEV components ----------
    oof = pl.read_parquet(SEQ / "oof_rows.parquet")
    dev_ids = oof["pair_id"].to_list()
    v30_base_d = oof["baseline_risk"].to_numpy().astype(np.float32)
    v30_spec_d = np.column_stack([oof["specialist_directed"].to_numpy(),
                                  oof["specialist_soft"].to_numpy(),
                                  oof["specialist_isolation"].to_numpy()]).astype(np.float32)
    v30_shipped_d = oof["rank_sum_w50"].to_numpy().astype(np.float32)

    # E-line specialists on dev (aligned by pair_id)
    er = pl.read_parquet(ELINE / "risk_dev.parquet")
    er_map = {r["pair_id"]: (r["spec_directed"], r["spec_soft"], r["spec_isolation"])
              for r in er.to_dicts()}
    eline_spec_d = np.array([er_map.get(p, (0.0, 0.0, 0.0)) for p in dev_ids], np.float32)

    # ---------- EVAL components ----------
    ebase = pl.read_csv(V30_EVAL_BASE).select(["pair_id", "risk_score"])
    espec = pl.read_parquet(SEQ / "eval_specialist_scores.parquet")
    ej = ebase.join(espec, on="pair_id", how="inner")
    eval_ids = ej["pair_id"].to_list()
    v30_base_e = ej["risk_score"].to_numpy().astype(np.float32)
    v30_spec_e = np.column_stack([ej["specialist_directed"].to_numpy(),
                                  ej["specialist_soft"].to_numpy(),
                                  ej["specialist_isolation"].to_numpy()]).astype(np.float32)
    ee = pl.read_parquet(ELINE / "risk_eval.parquet")
    ee_map = {r["pair_id"]: (r["spec_directed"], r["spec_soft"], r["spec_isolation"])
              for r in ee.to_dicts()}
    eline_spec_e = np.array([ee_map.get(p, (0.0, 0.0, 0.0)) for p in eval_ids], np.float32)

    def union(spec):  # rank-of-sum union like V30
        return np.clip(spec.sum(1), 0, 1)

    # rank-normalized building blocks (dev + eval)
    dbr, ebr = _prank(v30_base_d), _prank(v30_base_e)
    d_v30u, e_v30u = _prank(union(v30_spec_d)), _prank(union(v30_spec_e))
    d_elu, e_elu = _prank(union(eline_spec_d)), _prank(union(eline_spec_e))

    print("=== HYBRID risk arms (eval-honest; base is CONFIRMED-trained -> trustworthy) ===")
    print(f"reference: V30 rank_sum_w50 = 0.9203 eval-honest = 0.70904 LB\n")

    arms_dev, arms_eval, scores = {}, {}, {}

    def add(name, dv, ev):
        r = H.judge(name, dv.astype(np.float32), dev_ids, ev.astype(np.float32), eval_ids)
        scores[name] = r.eval_honest_ap
        arms_dev[name], arms_eval[name] = dv, ev
        print(f"  {name:<26} eval_honest={r.eval_honest_ap:.4f}  confirmed={r.confirmed_ap:.4f}")

    # sanity: reconstruct V30 shipped from base+V30union (should ~match 0.9203)
    add("V30_repro_base+v30spec_w50", 0.5 * dbr + 0.5 * d_v30u, 0.5 * ebr + 0.5 * e_v30u)
    # hybrid: base + E-line specialists at several weights
    for w in [0.3, 0.5, 0.7]:
        add(f"hybrid_eline_w{int(w*100)}", (1 - w) * dbr + w * d_elu, (1 - w) * ebr + w * e_elu)
    # hybrid: base + BOTH unions (mean of V30 and E-line specialist ranks)
    add("hybrid_both_mean_w50", 0.5 * dbr + 0.5 * (0.5 * d_v30u + 0.5 * d_elu),
        0.5 * ebr + 0.5 * (0.5 * e_v30u + 0.5 * e_elu))
    add("hybrid_both_mean_w60", 0.4 * dbr + 0.6 * (0.5 * d_v30u + 0.5 * d_elu),
        0.4 * ebr + 0.6 * (0.5 * e_v30u + 0.5 * e_elu))
    # hybrid: base + max of the two unions (take whichever specialist family fires strongest)
    add("hybrid_both_max_w50", 0.5 * dbr + 0.5 * np.maximum(d_v30u, d_elu),
        0.5 * ebr + 0.5 * np.maximum(e_v30u, e_elu))

    best = max(scores, key=scores.get)
    v30_ref = scores.get("V30_repro_base+v30spec_w50", 0.9203)
    print(f"\nBEST hybrid arm: {best} = {scores[best]:.4f} eval-honest")
    print(f"  vs V30 repro {v30_ref:.4f} -> delta {scores[best]-v30_ref:+.4f}")
    print(f"  vs V30 shipped 0.9203 -> delta {scores[best]-0.9203:+.4f}")
    print(f"  (harness noise band ~0.01; needs > +0.01 to be a real gain)")

    # persist best hybrid eval risk for assembly
    pl.DataFrame({"pair_id": eval_ids,
                  "risk_score": np.clip(arms_eval[best], 0, 1).astype(np.float32)}).write_parquet(
        OUT / "hybrid_risk_eval.parquet")
    (OUT / "_hybrid_summary.json").write_text(json.dumps(
        {"best_arm": best, "scores": scores, "v30_repro_ref": v30_ref,
         "delta_vs_shipped": scores[best] - 0.9203}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/hybrid_risk_eval.parquet, _hybrid_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
