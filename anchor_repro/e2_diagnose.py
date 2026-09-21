"""Diagnose WHY E2 catches less than V30 — test 3 hypotheses, don't guess.

H_sets   : not enough negative-sample sets / bags (variance) -> more bags should lift E2.
H_metric : E2 is missing a FEATURE/signal V30 has (the isolation SEQUENCE tell, dossier §6.3).
H_bug    : a setup bug (leakage direction, label misalignment, target definition) caps E2.

Concrete tests:
 1. FEATURE BASIS COMPARISON: what does E2 train on vs what V30's specialists train on?
    E2 = Layer-2 generic pair aggregates (49); V30 specialists = 128 sequence features incl the
    isolation SEQUENCE columns (seq_sand_*, seq_iso_*). The dossier §6.3 says isolation is a
    SEQUENCE not a count -> E2's count-basis structurally CANNOT see isolation. Verify by checking
    whether the isolation-sequence columns even exist in E2's basis.
 2. TRAINING-TARGET CHECK (bug hunt): E2 trains "confirmed positives vs UNLABELED EVAL negatives".
    V30 trains "confirmed pos vs confirmed NEG + PU-sampled unknowns". Does E2's use of eval pairs
    as negatives POISON isolation (if eval isolation pairs look like dev isolation positives, they
    get pushed as negatives -> E2 taught to IGNORE the isolation signal)? Measure isolation-pair
    score suppression.
 3. VARIANCE TEST: is the per-family gap stable across seeds, or noise from 92 isolation positives?

Run:  python -m anchor_repro.e2_diagnose
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D
from anchor_repro.gen_pair_detector import pair_features, SIG_COLS

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")


def main() -> int:
    print("=== H_metric: does E2's feature basis even CONTAIN the isolation-sequence signal? ===")
    # E2 basis = Layer-2 pair_features columns (SIG_COLS-derived aggregates)
    e2_basis = [c for c in pair_features(CACHE / "dev_layer2.parquet").columns
                if c not in ("pair_id", "label", "table_id") and not c.endswith("_sum")]
    print(f"E2 basis ({len(e2_basis)} feats), SIG_COLS source: {SIG_COLS}")
    iso_in_e2 = [c for c in e2_basis if "iso" in c.lower() or "squeeze" in c.lower()]
    print(f"  E2 isolation-related feats: {iso_in_e2}")

    seq_cols = pl.read_parquet(SEQ / "dev_sequence_features.parquet").columns
    iso_seq = [c for c in seq_cols if "iso" in c.lower() or "squeeze" in c.lower() or "sand" in c.lower()]
    print(f"\nV30 sequence basis has {len(iso_seq)} isolation/squeeze/sandwich SEQUENCE feats:")
    for c in iso_seq:
        print(f"    {c}")
    print("\n-> If E2 has only COUNT-based iso feats and V30 has SEQUENCE iso feats, that is the")
    print("   STRUCTURAL reason E2 is blind to isolation (dossier §6.3: 'isolation is a SEQUENCE').")

    # --- H_bug: are eval isolation-like pairs poisoning E2's isolation as negatives? ---
    print("\n=== H_bug: does training on UNLABELED EVAL negatives suppress the isolation signal? ===")
    # E2's own iso-signal feature on dev positives by family vs eval pairs
    dev = pair_features(CACHE / "dev_layer2.parquet")
    oof = pl.read_parquet(SEQ / "oof_rows.parquet").select(["pair_id", "known", "behavior_y"])
    dj = dev.join(oof, on="pair_id", how="inner").filter(pl.col("known"))
    ev = pair_features(CACHE / "eval_layer2.parquet")
    iso_feat = "iso_squeeze_signal_mean"
    if iso_feat in dj.columns:
        fam = dj["behavior_y"].to_numpy()
        vals = dj[iso_feat].to_numpy()
        ev_vals = ev[iso_feat].to_numpy() if iso_feat in ev.columns else None
        print(f"  {iso_feat} by dev family:")
        for fid, fn in [(1, "directed"), (2, "soft"), (3, "isolation"), (0, "neg")]:
            m = fam == fid
            if m.sum():
                print(f"    {fn:<10}: mean={vals[m].mean():.4f} p90={np.quantile(vals[m],0.9):.4f} (n={int(m.sum())})")
        if ev_vals is not None:
            # what fraction of EVAL pairs look like dev ISOLATION positives on this feature?
            iso_thresh = np.quantile(vals[fam == 3], 0.5) if (fam == 3).sum() else 0
            ev_isolike = float((ev_vals >= iso_thresh) > 0) if np.isscalar(ev_vals) else float((ev_vals >= iso_thresh).mean())
            print(f"  eval pairs above dev-isolation-median on {iso_feat}: {ev_isolike:.3f}")
            print("  -> if a NONtrivial fraction of eval pairs look isolation-like, E2 (which treats")
            print("     ALL eval pairs as negatives) is TAUGHT to suppress the isolation signal = poisoning.")

    # --- H_sets: is the isolation gap seed-stable or variance from 92 positives? ---
    print("\n=== H_sets: is E2's isolation weakness stable, or noise from only 92 positives? ===")
    print("  (isolation has 92 confirmed positives; directed 148, soft 132 — smallest family)")
    print("  E2 per-family AP gaps vs V30 (from introspection): directed -0.08, soft -0.10, iso -0.23")
    print("  The gap ORDER tracks BOTH (a) family rarity AND (b) sequence-dependence.")
    print("  Decisive check: does V30 ALSO find isolation hardest? If yes, it's family difficulty +")
    print("  V30's sequence features rescue it; if E2 alone finds iso hardest, E2's basis is the cause.")

    # V30 per-family AP recomputed here for the head-to-head
    oof2 = pl.read_parquet(SEQ / "oof_rows.parquet").select(
        ["pair_id", "known", "y", "behavior_y", "rank_sum_w50"]).filter(pl.col("known"))
    e2o = pl.read_parquet(CACHE / "_E2_best_dev_oof.parquet")
    m = oof2.join(e2o, on="pair_id", how="left")
    fam = m["behavior_y"].to_numpy(); y = m["y"].to_numpy().astype(int)
    v30 = m["rank_sum_w50"].to_numpy(); e2 = m["E2_oof"].to_numpy()
    print("\n  per-family AP (this family's pos vs all negatives):")
    print(f"  {'family':<10} {'E2':>8} {'V30':>8} {'E2 rank':>9} {'V30 rank':>9}")
    aps = []
    for fid, fn in [(1, "directed"), (2, "soft"), (3, "isolation")]:
        keep = (fam == 0) | (fam == fid)
        yf = (fam[keep] == fid).astype(int)
        e2ap = average_precision_score(yf, e2[keep]); v30ap = average_precision_score(yf, v30[keep])
        aps.append((fn, e2ap, v30ap))
    e2_rank = sorted(aps, key=lambda r: r[1])
    v30_rank = sorted(aps, key=lambda r: r[2])
    for fn, e2ap, v30ap in aps:
        er = [i for i, r in enumerate(e2_rank) if r[0] == fn][0] + 1
        vr = [i for i, r in enumerate(v30_rank) if r[0] == fn][0] + 1
        print(f"  {fn:<10} {e2ap:>8.4f} {v30ap:>8.4f} {er:>9} {vr:>9}")
    print("  (rank 1 = hardest family for that model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
