"""§67 — Negative-space block composed onto the honghanh 0.64 base (BUILD + HOLDOUT-TUNE).

Pre-registration: RESEARCH_DOSSIER §67 (append-only; bars FIXED before any number was
computed). Governed by the workspace ``reverse-engineering-accountability`` contract
(Rules 1, 2, 3, 4, 6, 7, 9, 11, 12, 15, 16, 17, 18, 19).

WHAT THIS DOES (and ONLY this)
------------------------------
Composes the §62 negative-space feature block onto the honghanh dev-holdout OOF risk
head (real LB 0.64262) as a FLOOR-GUARANTEED rank blend (contract Rule 15: ``w=0`` is
in the grid, so honghanh is a HARD FLOOR that a corrupting layer collapses back to and
can never regress below). The blend weight is tuned ONLY on the honghanh dev-holdout OOF
confirmed pairs (an EXTERNAL-to-this-layer split — honghanh's own held-out predictions).
The adversarial drift AUC on the ns composite is reported FOR INFORMATION ONLY (Rule 17:
it is a gatekeeper for the LB read, not a veto on the build/tune).

PRE-REGISTERED BAR (FIXED before running — Rules 2, 12)
-------------------------------------------------------
* k = 50.0                       (the §65 A/B k that transferred on the floor; LOCKED)
* w_grid = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)  (w=0 present — floor guarantee)
* bar = 0.005
  A candidate "clears the bar" IFF best_w > 0 AND (best_pair_ap - w0_pair_ap) >= bar.
  A candidate that does NOT clear the bar is an HONEST NULL (Rule 3): no CSV is built,
  the null is reported explicitly, honghanh stays the floor.

This module BUILDS + HOLDOUT-TUNES ONLY. It NEVER submits to Kaggle, NEVER overwrites the
live ``submission.csv`` / ``submission_best_*.csv``. Any eval candidate is written to a
NEW path under ``outputs/poker_collusion/ns_on_honghanh/`` (created if absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from anchor_repro.combo_screen import (
    build_combo_eval_submission,
    sweep_blend,
    validate_eval_submission,
)
from anchor_repro.negative_space import (
    NEGATIVE_SPACE_RANK_COLUMNS,
    _dev_pairs_ab,
    _dev_positive_players,
    _eval_pairs_ab,
    build_negative_space,
    default_negative_space_cache_locations,
    negative_space_drift_gate,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from poker_collusion.config import PipelineConfig

# --------------------------------------------------------------------------- #
# LOCKED §67 parameters (do not change).                                       #
# --------------------------------------------------------------------------- #
K: float = 50.0
W_GRID: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
BAR: float = 0.005
DRIFT_INFO_THRESHOLD: float = 0.65  # information-only PASS/FAIL line (NOT a veto).

#: honghanh dev-holdout OOF + eval submission (the FLOOR this layer composes onto).
_HH_OOF_REL = "outputs/poker_collusion/repro_064469/dev_holdout_oof_064469.parquet"
_HH_EVAL_SUB_REL = "outputs/poker_collusion/repro_064469/submission.csv"

#: NEW output dir (never touches the live submission files).
_OUT_DIR_REL = "outputs/poker_collusion/ns_on_honghanh"


def _candidate_encodings(block: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return the two ns candidate lever encodings indexed by str(pair_id).

    * ``ns_composite``   = the ns_anomaly_composite column (the fitted §63 feature).
    * ``ns_3rank_mean``  = mean of the three NEGATIVE_SPACE_RANK_COLUMNS diagnostics.
    """
    b = block.copy()
    b["pair_id"] = b["pair_id"].astype(str)
    b = b.set_index("pair_id")
    three = b[list(NEGATIVE_SPACE_RANK_COLUMNS)].mean(axis=1)
    return {
        "ns_composite": b["ns_anomaly_composite"].astype(float),
        "ns_3rank_mean": three.astype(float),
    }


def run(poker_root: Path) -> Dict[str, object]:
    """Build the ns block, tune the floor-guaranteed blend on honghanh's OOF, report.

    Returns the JSON-serialisable summary (also written to disk).
    """
    root = Path(poker_root)

    # 1. Resolve caches; refuse (SystemExit) if anything is missing.
    caches = default_negative_space_cache_locations(root)
    missing = caches.missing()
    if missing:
        raise SystemExit(
            "negative_space_on_honghanh: missing required caches:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    data_dir = root / "data" / "poker"

    # 2. Dev labels + optional evidence -> CanonicalScorer.
    labels = pd.read_csv(data_dir / "development_labels.csv")
    evidence_path = data_dir / "development_evidence.csv"
    evidence = pd.read_csv(evidence_path) if evidence_path.is_file() else None
    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=data_dir),
        labels=labels,
        evidence=evidence,
    )

    # 3. Build the DEV and EVAL negative-space blocks through the SAME builder (§21).
    print("[§67] building DEV negative-space block (k=%.1f) ..." % K)
    dev_pairs = _dev_pairs_ab(caches)
    dev_block = build_negative_space(
        dev_pairs, "development", caches,
        k=K, positive_players=_dev_positive_players(caches),
    )
    print("[§67] building EVAL negative-space block (k=%.1f) ..." % K)
    eval_pairs = _eval_pairs_ab(caches)
    eval_block = build_negative_space(
        eval_pairs, "evaluation", caches,
        k=K, positive_players=set(),
    )

    dev_encodings = _candidate_encodings(dev_block)
    eval_encodings = _candidate_encodings(eval_block)

    # 4. honghanh dev-holdout OOF (risk = oof_risk). pair_id -> str everywhere.
    oof = pd.read_parquet(root / _HH_OOF_REL)
    oof["pair_id"] = oof["pair_id"].astype(str)
    hh_risk_all = oof.set_index("pair_id")["oof_risk"].astype(float)
    hh_confirmed = set(oof.loc[oof["is_labeled"] == True, "pair_id"])  # noqa: E712

    # 6. Common confirmed dev pairs = honghanh confirmed ∩ ns dev block pair_ids.
    ns_dev_ids = set(dev_encodings["ns_composite"].index)
    common = sorted(hh_confirmed & ns_dev_ids)
    n_common = len(common)
    print(
        "[§67] honghanh confirmed=%d, ns-dev pairs=%d, common confirmed=%d"
        % (len(hh_confirmed), len(ns_dev_ids), n_common)
    )
    if n_common == 0:
        raise SystemExit("[§67] no common confirmed dev pairs — cannot tune (refusing).")

    honghanh_risk = np.array([hh_risk_all[p] for p in common], dtype=float)

    # 8. Drift gate — INFORMATION ONLY (Rule 17 — not a veto).
    print("[§67] running negative-space drift gate (k=%.1f) — INFORMATION ONLY ..." % K)
    drift = negative_space_drift_gate(caches, k=K)
    drift_auc = float(drift.auc)
    drift_pass = drift_auc < DRIFT_INFO_THRESHOLD

    # 7. Sweep the floor-guaranteed blend per candidate on the common confirmed pairs.
    sweeps: Dict[str, Dict[str, object]] = {}
    for name in ("ns_composite", "ns_3rank_mean"):
        lever_series = dev_encodings[name]
        ns_lever = np.array([float(lever_series[p]) for p in common], dtype=float)
        sw = sweep_blend(scorer, name, common, honghanh_risk, ns_lever, W_GRID)
        w0 = float(sw.floor_alone_pair_ap)  # PairAP at w=0 == honghanh floor alone.
        delta = float(sw.best_pair_ap) - w0
        clears = bool(sw.best_w > 0.0 and delta >= BAR)
        sweeps[name] = {
            "w_grid": list(sw.w_grid),
            "pair_ap": [float(x) for x in sw.pair_ap],
            "best_w": float(sw.best_w),
            "best_pair_ap": float(sw.best_pair_ap),
            "w0_pair_ap": w0,
            "delta": delta,
            "clears_bar": clears,
        }

    # 9. Full auditable report.
    print("\n" + "=" * 72)
    print("§67 NEGATIVE-SPACE ON HONGHANH 0.64 — FLOOR-GUARANTEED BLEND (HOLDOUT TUNE)")
    print("=" * 72)
    print("k                 = %.1f  (LOCKED)" % K)
    print("w_grid            = %s" % (", ".join("%.2f" % w for w in W_GRID)))
    print("bar               = %.4f  (best_w>0 AND best-w0 >= bar)" % BAR)
    print("n_common_confirmed= %d" % n_common)
    print(
        "drift(ns composite) AUC = %.4f  -> %s (<%.2f)  [INFORMATION ONLY, NOT a veto]"
        % (drift_auc, "PASS" if drift_pass else "FAIL", DRIFT_INFO_THRESHOLD)
    )
    # w0 must be identical across candidates (honghanh alone at w=0).
    w0_values = {name: sweeps[name]["w0_pair_ap"] for name in sweeps}
    print("w0 (honghanh floor-alone) PairAP per candidate:")
    for name, w0 in w0_values.items():
        print("    %-14s w0 PairAP = %.6f" % (name, w0))

    for name in ("ns_composite", "ns_3rank_mean"):
        s = sweeps[name]
        print("\n--- candidate: %s ---" % name)
        print("    %-6s %-12s %-12s" % ("w", "PairAP", "delta_vs_w0"))
        for w, ap in zip(s["w_grid"], s["pair_ap"]):
            print("    %-6.2f %-12.6f %-+12.6f" % (w, ap, ap - s["w0_pair_ap"]))
        print(
            "    best_w=%.2f  best_PairAP=%.6f  w0=%.6f  delta=%+.6f  clears_bar=%s"
            % (s["best_w"], s["best_pair_ap"], s["w0_pair_ap"], s["delta"], s["clears_bar"])
        )

    # 10/11. Build eval CSV per candidate that clears the bar; explicit NULL otherwise.
    out_dir = root / _OUT_DIR_REL
    hh_eval_sub = pd.read_csv(root / _HH_EVAL_SUB_REL, dtype={"pair_id": str})
    eval_pairs_csv = pd.read_csv(caches.eval_pairs_csv, dtype={"pair_id": str})

    built: List[Dict[str, object]] = []
    print("\n" + "-" * 72)
    print("§67 EVAL-CANDIDATE BUILD (only for bar-clearing candidates; NEW path only)")
    print("-" * 72)
    for name in ("ns_composite", "ns_3rank_mean"):
        s = sweeps[name]
        if not s["clears_bar"]:
            print(
                "HONEST NULL: candidate %r did NOT clear the bar "
                "(best_w=%.2f, delta=%+.6f < %.4f) — no CSV built; honghanh stays floor."
                % (name, s["best_w"], s["delta"], BAR)
            )
            continue

        best_w = float(s["best_w"])
        lever_col = eval_encodings[name]
        lever_eval = pd.DataFrame(
            {
                "pair_id": [str(p) for p in lever_col.index],
                "risk_score": lever_col.to_numpy(dtype=float),
            }
        )
        out_path = out_dir / ("ns_on_honghanh_%s_w%s.csv" % (name, ("%.2f" % best_w)))
        out = build_combo_eval_submission(hh_eval_sub, lever_eval, best_w, out_path)

        checks = validate_eval_submission(out, eval_pairs_csv)
        all_ok = bool(
            checks["n_rows_ok"]
            and checks["schema_ok"]
            and checks["risk_in_unit"]
            and checks["pairs_match_eval"]
        )
        assert all_ok, "validate_eval_submission failed for %s: %r" % (name, checks)

        # Spearman of new risk_score vs honghanh base risk_score (joined by pair_id).
        merged = out[["pair_id", "risk_score"]].merge(
            hh_eval_sub[["pair_id", "risk_score"]],
            on="pair_id", suffixes=("_new", "_hh"),
        )
        rho, _p = spearmanr(merged["risk_score_new"], merged["risk_score_hh"])
        rho = float(rho)

        print(
            "BUILT: %s  best_w=%.2f  -> %s"
            % (name, best_w, out_path)
        )
        print("       validation checks = %r  (all_ok=%s)" % (checks, all_ok))
        print("       Spearman(new risk vs honghanh base) = %.6f" % rho)
        built.append(
            {
                "candidate": name,
                "best_w": best_w,
                "out_path": str(out_path),
                "validation": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                               for k, v in checks.items()},
                "all_checks_ok": all_ok,
                "spearman_vs_honghanh": rho,
            }
        )

    if not built:
        print("\n[§67] NO eval CSVs built (all candidates were honest nulls).")

    # 12. JSON summary to the NEW dir.
    summary: Dict[str, object] = {
        "section": "§67",
        "k": K,
        "w_grid": list(W_GRID),
        "bar": BAR,
        "n_common_confirmed": n_common,
        "drift_composite_auc": drift_auc,
        "drift_pass_info_only": drift_pass,
        "candidates": sweeps,
        "built_eval_candidates": built,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "_ns_on_honghanh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[§67] wrote summary -> %s" % summary_path)
    return summary


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[1]
    poker_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root
    run(poker_root)
