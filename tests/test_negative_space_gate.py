"""§62 NEGATIVE-SPACE integration gate: GATE A (drift) FIRST, then GATE B (k-sweep).

Pre-registration: RESEARCH_DOSSIER §62 (bars FIXED before any feature was computed).
Governed by the workspace ``reverse-engineering-accountability`` contract
(Rules 1,2,3,4,5,9,11,13,15,16,17,18,19).

Run with (from ``.../kAGGLE/poker``):
    $env:PYTHONUTF8="1"; python -m pytest tests/test_negative_space_gate.py -m integration -q -s

What this test does (exactly the pre-registered sequence):
  1. GATE A (RUN FIRST, Rules 5, 17): adversarial dev-vs-eval AUC on the negative-space
     block via the SAME machinery §51/§33 used. PRE-REGISTERED BAR: AUC < 0.65 => PASS.
  2. GATE B (only meaningful if GATE A passes, but ALWAYS computed + printed so the
     measured numbers are on the record — Rules 3, 9): foundation-alone canonical PairAP
     vs foundation+block PairAP on the seed-7 60/40 table-disjoint HELD-OUT tables, over
     a FULL k-sweep (Rule 13). PRE-REGISTERED BAR: best swept-k delta >= +0.010.
  3. The motivating diagnostic (§62): does the shrunk RESIDUAL concentrate on known
     positives MORE than the PRESENCE-only counterpart? MEASURED AUCs printed.

Honesty (Rules 3, 9, 12): this test asserts the gates BEHAVE (finite measured numbers,
foundation floor is a real number, the sweep is computed) and PRINTS every measured
value VERBATIM. It does NOT assert a lift. The MEASURED result — PASS/FAIL of each bar —
is the deliverable, whether it is a candidate or an honest null. If the real caches /
data are absent the test ``skip``s with a clear reason; it never fabricates data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig

from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.own_submission_recipes import DRIFT_AUC_THRESHOLD
from anchor_repro.negative_space import (
    default_negative_space_cache_locations,
    negative_space_drift_gate,
    run_gate_b_ksweep,
    presence_vs_residual_concentration,
)

pytestmark = pytest.mark.integration

_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_DEV_LABELS = _DATA_DIR / "development_labels.csv"
_DEV_EVIDENCE = _DATA_DIR / "development_evidence.csv"

# The pre-registered shrinkage k grid (Rule 13 — swept, not guessed). Spans "almost no
# shrink" (k=5) through "heavy shrink" (k=400), bracketing the eval co-hand median (~76).
_K_GRID = (5.0, 25.0, 50.0, 100.0, 200.0, 400.0)

# Field-dyad sample: full DEFAULT (20k) would be slow across the run; a stable field
# estimate needs only a few thousand co-seated dyads. Fixed for reproducibility.
_N_FIELD_DYADS = 8_000


def _dev_evidence() -> Optional[pd.DataFrame]:
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


def _require_caches():
    if not _DEV_LABELS.is_file():
        pytest.skip(f"Real dev labels absent ({_DEV_LABELS}); skipping.")
    caches = default_negative_space_cache_locations(_POKER_ROOT)
    missing = caches.missing()
    if missing:
        pytest.skip(
            "Negative-space inputs absent; missing: "
            + ", ".join(str(p) for p in missing)
        )
    return caches


@pytest.mark.integration
def test_negative_space_gate_a_then_gate_b_ksweep() -> None:
    """GATE A drift FIRST, then GATE B k-sweep + concentration diagnostic (§62)."""
    caches = _require_caches()
    labels = pd.read_csv(_DEV_LABELS)
    evidence = _dev_evidence()

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )

    # ------------------------------------------------------------------ #
    # GATE A — adversarial drift (RUN FIRST, Rules 5, 17). BAR: AUC<0.65. #
    # The gated block is the SINGLE pool-relative composite anomaly score #
    # (§63 debugged execution). Computed at a mid-grid k=50; the composite #
    # drift is stable across k (measured 0.52-0.53 for k in 25..200).      #
    # ------------------------------------------------------------------ #
    dg = negative_space_drift_gate(
        caches, k=50.0, n_field_dyads=_N_FIELD_DYADS
    )
    print("\n================ §62 NEGATIVE-SPACE GATES (MEASURED) ================")
    print(
        f"[GATE A] adversarial dev-vs-eval AUC = {dg.auc:.4f} "
        f"(bar < {DRIFT_AUC_THRESHOLD}) -> {dg.verdict} "
        f"| n_features={dg.n_features} n_dev={dg.n_dev} n_eval={dg.n_eval}"
    )
    assert math.isfinite(dg.auc) and 0.0 <= dg.auc <= 1.0
    assert dg.threshold == DRIFT_AUC_THRESHOLD

    gate_a_passed = dg.passed

    # ------------------------------------------------------------------ #
    # Motivating diagnostic (§62): residual vs presence concentration.    #
    # ------------------------------------------------------------------ #
    conc = presence_vs_residual_concentration(
        caches, k=50.0, n_field_dyads=_N_FIELD_DYADS
    )
    print(
        f"[DIAGNOSTIC] confirmed={int(conc['n_confirmed'])} "
        f"positives={int(conc['n_positive'])} "
        f"(in-sample separation AUC, label==1 vs label==0):"
    )
    for name in ("clash", "contest", "foldyield"):
        print(
            f"    {name:9s}: residual AUC={conc[f'residual_auc_{name}']:.4f} "
            f"| presence AUC={conc[f'presence_auc_{name}']:.4f}"
        )
    print(
        f"    combined : residual AUC={conc['residual_auc_combined']:.4f} "
        f"| presence AUC={conc['presence_auc_combined']:.4f}"
    )
    for key in ("residual_auc_combined", "presence_auc_combined"):
        assert math.isfinite(conc[key])

    # ------------------------------------------------------------------ #
    # GATE B — foundation vs foundation+block, seed-7 60/40 HELD-OUT      #
    # tables, FULL k-sweep (Rule 13). BAR: best delta >= +0.010.          #
    # Always computed + printed (Rules 3, 9), even if GATE A failed, so   #
    # the measured numbers are on the record.                             #
    # ------------------------------------------------------------------ #
    sweep = run_gate_b_ksweep(
        caches,
        labels,
        scorer,
        k_values=_K_GRID,
        n_field_dyads=_N_FIELD_DYADS,
        evidence=evidence,
    )
    print(
        f"[GATE B] foundation-alone (v5+MFg+DIRc) PairAP = "
        f"{sweep.foundation_pair_ap:.6f} "
        f"over {sweep.n_holdout_confirmed} confirmed held-out pairs"
    )
    print(f"[GATE B] k-sweep (foundation + negative-space block), bar delta >= +{sweep.bar}:")
    for k, ap, d in zip(sweep.k_values, sweep.block_pair_ap, sweep.deltas):
        flag = "  <== CLEARS BAR" if d >= sweep.bar else ""
        print(f"    k={k:7.1f}  PairAP={ap:.6f}  delta={d:+.6f}{flag}")
    print(
        f"[GATE B] best k={sweep.best_k:.1f}  best delta={sweep.best_delta:+.6f}  "
        f"-> {'CANDIDATE (>=+0.010)' if sweep.clears_bar else 'HONEST NULL (<+0.010)'}"
    )
    print(
        f"[VERDICT] GATE A: {'PASS' if gate_a_passed else 'FAIL'} (AUC {dg.auc:.4f}) | "
        f"GATE B: {'CANDIDATE' if sweep.clears_bar else 'NULL'} "
        f"(best delta {sweep.best_delta:+.6f})"
    )
    print("=====================================================================\n")

    # Sanity assertions only (Rules 3, 9 — no lift is asserted).
    assert math.isfinite(sweep.foundation_pair_ap)
    assert len(sweep.block_pair_ap) == len(_K_GRID)
    for ap in sweep.block_pair_ap:
        assert math.isfinite(ap)
    assert sweep.n_holdout_confirmed > 0
