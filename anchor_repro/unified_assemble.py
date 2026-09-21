"""Assemble the COMPLETE unified-per-hand submission — all 3 columns from ONE model.

risk_score        = pair-level top-3 mean of per-hand coordination prob
predicted_behavior= family (directed/soft/iso) of the pair's top-coord hands (coord-weighted flags)
evidence_hand_1..5= the 5 highest per-hand coordination-prob hands (dev Evidence-MAP@5 0.388)

All three read from the SAME per-hand coordination model (dossier §123). Contract-validated.
Run:  python -m anchor_repro.unified_assemble
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("outputs/poker_collusion/generator_cache")
UNI = CACHE / "unified"
D = Path("data/poker")
CAND = UNI / "candidate_unified_perhand.csv"
EV_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior", *EV_COLS]
NO_EV = "NO_EVIDENCE"
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]


def main() -> int:
    ph = pl.read_parquet(UNI / "eval_perhand.parquet")  # pair_id, hand_id, coord, flag_directed/soft/iso
    risk = pl.read_parquet(UNI / "eval_risk.parquet")   # pair_id, risk_max/top5/top3/mean

    # RISK = top3 (locked choice), min-max scaled to [0,1]
    r = risk.select(["pair_id", "risk_top3"]).rename({"risk_top3": "risk_score"})
    lo, hi = r["risk_score"].min(), r["risk_score"].max()
    r = r.with_columns(((pl.col("risk_score") - lo) / (hi - lo)).clip(0, 1).alias("risk_score"))

    # EVIDENCE = top-5 coord hands per pair (coord desc, hand_id asc tie-break)
    ranked = (ph.sort(["pair_id", "coord", "hand_id"], descending=[False, True, False])
              .group_by("pair_id", maintain_order=True).head(5))
    ev_rows = []
    for pid, sub in ranked.group_by("pair_id", maintain_order=True):
        pid = pid[0] if isinstance(pid, tuple) else pid
        hands = sub["hand_id"].to_list()[:5]
        hands += [NO_EV] * (5 - len(hands))
        ev_rows.append({"pair_id": pid, **{f"evidence_hand_{i+1}": hands[i] for i in range(5)}})
    ev_wide = pl.DataFrame(ev_rows)

    # BEHAVIOR = coord-weighted family of top-3 coord hands
    top3 = (ph.sort(["pair_id", "coord"], descending=[False, True]).group_by("pair_id", maintain_order=True).head(3))
    beh_rows = []
    for pid, sub in top3.group_by("pair_id", maintain_order=True):
        pid = pid[0] if isinstance(pid, tuple) else pid
        f = sub.select(["flag_directed", "flag_soft", "flag_iso", "coord"]).to_numpy()
        score = (f[:, :3] * f[:, 3:4]).sum(0)
        fam = FAMS[int(score.argmax())] if score.sum() > 0 else "none"
        beh_rows.append({"pair_id": pid, "predicted_behavior": fam})
    beh = pl.DataFrame(beh_rows)

    sub = (r.join(beh, on="pair_id", how="inner").join(ev_wide, on="pair_id", how="inner").select(SUB_COLS))

    # validate contract
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id"])
    checks = {
        "rows_112540": sub.height == 112540,
        "schema": list(sub.columns) == SUB_COLS,
        "pair_set": set(sub["pair_id"].to_list()) == set(eval_pairs["pair_id"].to_list()),
        "risk_in_01": bool(sub["risk_score"].min() >= 0 and sub["risk_score"].max() <= 1),
        "no_null": int(sub.null_count().sum_horizontal().item()) == 0,
    }
    dup = 0
    for row in sub.select(EV_COLS).to_numpy():
        real = [h for h in row if h != NO_EV]
        if len(real) != len(set(real)):
            dup += 1
    checks["no_dup_evidence"] = dup == 0
    print("=== contract validation ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("behavior dist:", sub["predicted_behavior"].value_counts().sort("count", descending=True).to_dicts())
    print(f"risk: mean={sub['risk_score'].mean():.3f} p50={sub['risk_score'].median():.3f} p99={sub['risk_score'].quantile(0.99):.3f}")

    if all(checks.values()):
        sub.write_csv(CAND)
        print(f"\nwrote {CAND}")
    else:
        print("\nVALIDATION FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
