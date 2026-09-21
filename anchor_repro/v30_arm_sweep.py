"""Sweep V30's OWN risk-blend arms through the EVAL-HONEST harness (no retrain).

The user's point (correct): V30 is not one baseline — it's a stack of tunable knobs, and the
shipped 0.70904 uses ONE frozen arm (rank_sum_w50) selected on PU-STRESS, not on the eval-honest
importance-weighted metric we now trust (§115). This asks: under the HONEST gate, is w50 actually
the best blend of {V30 base risk, family specialists}, or did we freeze the wrong knob?

No retraining: reconstruct every arm from CACHED components —
  dev  : oof_rows.parquet (baseline_risk + specialist_{directed,soft,isolation})
  eval : repro_lamhuy/submission.csv (recovered-original V30 base risk) + eval_specialist_scores
Arms rebuilt EXACTLY as lamhuy_sequence_specialists._build_arms, plus a FINER weight grid.

Judged on the shared Harness: confirmed AP + eval-honest importance-weighted AP (THE GATE).
The winning arm's eval scores are aligned to the eval pair order so a candidate could be emitted
if (and only if) an arm beats w50 on the honest gate beyond noise.

Run:  python -m anchor_repro.v30_arm_sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
V30_EVAL_BASE = Path("outputs/poker_collusion/repro_lamhuy/submission.csv")


def _prank(v: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).to_numpy(np.float32)


def _arms(base: np.ndarray, fam: np.ndarray, weights) -> dict:
    """Rebuild V30 arms + finer grid. fam shape (n,3)."""
    base = np.asarray(base, np.float32)
    usum = np.clip(fam.sum(axis=1), 0, 1).astype(np.float32)
    umax = fam.max(axis=1).astype(np.float32)
    br, sr, mr = _prank(base), _prank(usum), _prank(umax)
    arms = {"base_only": base, "specialist_sum": usum, "specialist_max": umax}
    for w in weights:
        arms[f"rank_sum_w{int(w*100):02d}"] = np.clip((1 - w) * br + w * sr, 0, 1).astype(np.float32)
        arms[f"rank_max_w{int(w*100):02d}"] = np.clip((1 - w) * br + w * mr, 0, 1).astype(np.float32)
        arms[f"raw_sum_w{int(w*100):02d}"] = np.clip((1 - w) * base + w * usum, 0, 1).astype(np.float32)
    return arms


def main() -> int:
    H = Harness()

    # --- dev components ---
    dev = pl.read_parquet(SEQ / "oof_rows.parquet")
    dbase = dev["baseline_risk"].to_numpy()
    dfam = np.column_stack([dev["specialist_directed"].to_numpy(),
                            dev["specialist_soft"].to_numpy(),
                            dev["specialist_isolation"].to_numpy()]).astype(np.float32)
    dev_ids = dev["pair_id"].to_list()
    # restrict to the 1,860 labeled harness pairs (dev oof has 25,860 incl PU)
    known = dev["known"].to_numpy().astype(bool)

    # --- eval components ---
    ebase_df = pl.read_csv(V30_EVAL_BASE).select(["pair_id", "risk_score"])
    efam_df = pl.read_parquet(SEQ / "eval_specialist_scores.parquet")
    e = ebase_df.join(efam_df, on="pair_id", how="inner")
    ebase = e["risk_score"].to_numpy().astype(np.float32)
    efam = np.column_stack([e["specialist_directed"].to_numpy(),
                            e["specialist_soft"].to_numpy(),
                            e["specialist_isolation"].to_numpy()]).astype(np.float32)
    eval_ids = e["pair_id"].to_list()

    weights = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.85]
    dev_arms = _arms(dbase, dfam, weights)
    eval_arms = _arms(ebase, efam, weights)

    print(f"\n=== V30 ARM SWEEP through eval-honest harness ===")
    print(f"{'arm':<18} {'eval_honest':>12} {'confirmed':>11} {'AUC':>7}")
    # harness judges via dev_oof aligned to labeled pairs; use judge() which aligns by pair_id
    results = {}
    for name, dscore in dev_arms.items():
        escore = eval_arms[name]
        res = H.judge(name, dscore, dev_ids, escore, eval_ids)
        results[name] = res
    # print sorted by eval-honest
    w50 = results.get("rank_sum_w50")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].eval_honest_ap):
        star = "  <== shipped 0.70904" if name == "rank_sum_w50" else ""
        delta = f"  ({r.eval_honest_ap - w50.eval_honest_ap:+.4f} vs w50)" if w50 and name != "rank_sum_w50" else ""
        print(f"{name:<18} {r.eval_honest_ap:>12.4f} {r.confirmed_ap:>11.4f} {r.auc:>7.4f}{star}{delta}")

    best = max(results.values(), key=lambda r: r.eval_honest_ap)
    print(f"\nBEST arm by eval-honest: {best.name}  eval_honest={best.eval_honest_ap:.4f}  "
          f"(shipped w50={w50.eval_honest_ap:.4f}, delta={best.eval_honest_ap-w50.eval_honest_ap:+.4f})")
    print(f"NOTE: consistency 0.61 | E2 push 0.65 | V30 lineage ~0.79 reference")

    out = {name: {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc}
           for name, r in results.items()}
    out["_best"] = best.name
    out["_shipped_w50_eval_honest"] = w50.eval_honest_ap
    (CACHE / "_v30_arm_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote _v30_arm_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
