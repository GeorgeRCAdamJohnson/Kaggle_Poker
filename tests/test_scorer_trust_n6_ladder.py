"""§56/§57 integration test: n=6 rank-tracking ladder — is the local scorer a
TRUSTWORTHY JUDGE at scale?

Pre-registered bar: RESEARCH_DOSSIER §56 (BINDING). Governed by the workspace
``reverse-engineering-accountability`` contract (Rules 2, 3, 6, 9, 12).

CONTEXT (§55). Contract Rule 17 was FALSIFIED for the ladder's RANKING purpose: three
drift-FAILING bases (floor, nomannic, honghanh) ordered three distinct LB levels
perfectly (Spearman 1.0). Drift AUC is now a REPORTED DIAGNOSTIC, not a hard admission
gate. That UNBLOCKED the three points §50/§51 disqualified purely on drift (v2 0.32580,
v3 0.37063, exp3c 0.41289), which are now ADMISSIBLE.

THE 6-POINT LADDER (all real LB EXTERNALLY VERIFIED, ascending):
* cand_14   real LB 0.32580  (v2 cache, single XGB PU risk)        [drift ~0.7618 diag]
* cand_17   real LB 0.37063  (v3 cache, rank-avg XGB d4+d6 PU risk) [drift ~0.7611 diag]
* cand_exp3c real LB 0.41289 (v5+MFg cache, one XGB)                [drift 0.7620 diag]
* floor     real LB 0.44519  (floor stack v5+MFg+DIRc)              [drift ~0.7621 diag]
* nomannic  real LB 0.56202  (self-earned, own 129-col pair matrix) [drift ~0.7594 diag]
* honghanh  real LB 0.64262  (self-earned competitor 064469 recipe) [shared-corpus diag]

The three NEW points (v2/v3/exp3c) are computed here for the first time (§50/§51 built
the recipes + drift but SKIPPED the dev-holdout PairAP because the then-hard gate blocked
it). The three others REUSE the §55 PairAPs by re-running the same recipes.

§56 STATISTIC + BARS (FIXED — do NOT move the 0.7 Spearman bar, Rule 12):
* PRIMARY = spearman_corr(local_pair_ap, real_lb); SECONDARY = combined (reported only).
* Trustworthy-judge bar: spearman >= 0.7 AND the ~95% BOOTSTRAP CI EXCLUDES 0 at n>=5.
* Verdict tiers: TRUSTED (>=0.7 AND bootstrap-CI-lower>0 at n>=5); WEAK (0<sp<0.7, OR
  >=0.7 but CI includes 0); NULL (sp<=0).
* If r=1 exactly the t-CI is degenerate; rely on the BOOTSTRAP CI (§56).

Skips cleanly if any real cache is absent — never fabricates data (Rules 1, 9).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from poker_collusion.config import PipelineConfig
from poker_collusion.validation.lb_cv import spearman_corr

from anchor_repro.competitor_recipe import (
    default_competitor_paths,
    rerun_competitor_recipe,
)
from anchor_repro.ladder import (
    OPERATIONAL_BAR,
    SCORABLE,
    LadderPoint,
    LadderValidator,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.nomannic_recipe import (
    default_nomannic_cache_locations,
    rerun_nomannic_recipe,
)
from anchor_repro.own_submission_recipes import (
    V2_DIGITS,
    V3_DIGITS,
    default_exp3c_cache_locations,
    default_own_cache_locations,
    rerun_v2_single_stack,
    rerun_v3_ens_stack,
    rerun_v5_mfg_stack,
)
from anchor_repro.recipe_registry import default_cache_locations, rerun_floor_stack
from anchor_repro.scorer import CanonicalScorer

pytestmark = pytest.mark.integration

_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_DEV_LABELS = _DATA_DIR / "development_labels.csv"
_DEV_EVIDENCE = _DATA_DIR / "development_evidence.csv"

# §56 externally-verified LB coordinates (ascending). Read as EXTERNAL-JUDGE ground
# truth, never fabricated (contract Rule 1).
_CAND14_LB = 0.32580
_CAND17_LB = 0.37063
_EXP3C_LB = 0.41289
_FLOOR_LB = 0.44519
_NOMANNIC_LB = 0.56202
_HONGHANH_LB = 0.64262

# Drift AUCs carried forward as REPORTED DIAGNOSTIC context (§50/§51/§52; not gates).
_DRIFT = {
    "cand_14": 0.7618,
    "cand_17": 0.7611,
    "cand_exp3c": 0.7620,
    "floor": 0.7621,
    "nomannic": 0.7594,
    "honghanh": float("nan"),  # shared-corpus basis (FAIL-class, §55)
}


def _spearman_ci_t(r: float, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Approx 95% CI for Spearman via t-approx SE=sqrt((1-r^2)/(n-2)).

    Degenerate at |r|>=1 (SE=0) — returns (r, r, 0.0) and the caller relies on the
    bootstrap CI instead (§56). nan for n<3.
    """
    if n < 3:
        return (float("nan"), float("nan"), float("nan"))
    if abs(r) >= 1.0:
        return (r, r, 0.0)
    se = math.sqrt((1.0 - r * r) / (n - 2))
    return (max(-1.0, r - z * se), min(1.0, r + z * se), se)


def _spearman_bootstrap_ci(
    local: List[float],
    real_lb: List[float],
    *,
    n_draws: int = 10000,
    seed: int = 7,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap 95% CI for Spearman: resample the (local, real_lb) points with
    replacement ``n_draws`` times, recompute Spearman each draw, take 2.5/97.5 pct.

    A resample that draws <2 distinct points, or is constant in either axis, yields an
    undefined Spearman (spearman_corr returns 0.0 for size<2; a constant axis gives a
    0/0 Pearson). Those draws are recorded honestly as the value spearman_corr returns
    (no silent dropping) so the reported CI reflects real resampling behavior (Rule 9).

    Returns ``(lo, hi, draws)`` — the 2.5/97.5 percentiles and the full draw array.
    """
    x = np.asarray(local, dtype=float)
    y = np.asarray(real_lb, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        idx = rng.integers(0, n, size=n)
        draws[i] = spearman_corr(x[idx].tolist(), y[idx].tolist())
    lo = float(np.percentile(draws, 2.5))
    hi = float(np.percentile(draws, 97.5))
    return lo, hi, draws


def _dev_evidence() -> Optional[pd.DataFrame]:
    if _DEV_EVIDENCE.is_file():
        return pd.read_csv(_DEV_EVIDENCE)
    return None


def _require_inputs():
    """Skip unless every cache the 6-point ladder needs exists."""
    if not _DEV_LABELS.is_file():
        pytest.skip(f"Real dev labels absent ({_DEV_LABELS}); skipping.")

    floor_caches = default_cache_locations(_POKER_ROOT)
    if floor_caches.missing():
        pytest.skip("Floor caches absent; missing: "
                    + ", ".join(str(p) for p in floor_caches.missing()))

    competitor_paths = default_competitor_paths(_POKER_ROOT)
    if competitor_paths.missing():
        pytest.skip("Competitor OOF cache absent; missing: "
                    + ", ".join(str(p) for p in competitor_paths.missing()))

    nomannic_caches = default_nomannic_cache_locations(_POKER_ROOT)
    if nomannic_caches.missing():
        pytest.skip("nomannic caches absent; missing: "
                    + ", ".join(str(p) for p in nomannic_caches.missing()))

    v2_caches = default_own_cache_locations(_POKER_ROOT, "v2")
    if v2_caches.missing():
        pytest.skip("v2 caches absent; missing: "
                    + ", ".join(str(p) for p in v2_caches.missing()))

    v3_caches = default_own_cache_locations(_POKER_ROOT, "v3")
    if v3_caches.missing():
        pytest.skip("v3 caches absent; missing: "
                    + ", ".join(str(p) for p in v3_caches.missing()))

    exp3c_caches = default_exp3c_cache_locations(_POKER_ROOT)
    if exp3c_caches.missing():
        pytest.skip("exp3c (v5+MFg) caches absent; missing: "
                    + ", ".join(str(p) for p in exp3c_caches.missing()))

    return (floor_caches, competitor_paths, nomannic_caches,
            v2_caches, v3_caches, exp3c_caches)


@pytest.mark.integration
def test_scorer_trust_n6_ladder() -> None:
    """Compute the 3 NEW dev-holdout PairAPs (v2/v3/exp3c), reuse the 3 §55 PairAPs
    (floor/nomannic/honghanh), assemble the n=6 ladder under ONE canonical recipe id,
    run rank_tracking, compute PRIMARY/SECONDARY Spearman + t-CI + bootstrap CI, and
    apply the §56 verdict tiers EXACTLY. The MEASURED result is the deliverable — the
    test never asserts a particular verdict direction (Rule 3)."""
    (floor_caches, competitor_paths, nomannic_caches,
     v2_caches, v3_caches, exp3c_caches) = _require_inputs()
    labels = pd.read_csv(_DEV_LABELS)
    evidence = _dev_evidence()

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=_DATA_DIR),
        labels=labels,
        evidence=evidence,
    )
    recipe_id = scorer.recipe.recipe_id

    # --- (1) THREE NEW dev-holdout PairAPs (v2/v3/exp3c) — computed for the first time.
    v2 = rerun_v2_single_stack(v2_caches, labels, evidence=evidence)
    v2_score = scorer.score_dev_predictions(v2.dev_predictions)

    v3 = rerun_v3_ens_stack(v3_caches, labels, evidence=evidence)
    v3_score = scorer.score_dev_predictions(v3.dev_predictions)

    exp3c = rerun_v5_mfg_stack(exp3c_caches, labels, evidence=evidence)
    exp3c_score = scorer.score_dev_predictions(exp3c.dev_predictions)

    # --- (1b) THREE REUSED §55 PairAPs (floor / nomannic / honghanh).
    floor = rerun_floor_stack(floor_caches, labels, evidence=evidence)
    floor_score = scorer.score_dev_predictions(floor.dev_predictions)

    nomannic = rerun_nomannic_recipe(nomannic_caches, labels, evidence=evidence)
    nomannic_score = scorer.score_dev_predictions(nomannic.dev_predictions)

    honghanh = rerun_competitor_recipe(competitor_paths, labels, evidence=evidence)
    honghanh_score = scorer.score_dev_predictions(honghanh.dev_predictions)

    all_scores = [v2_score, v3_score, exp3c_score,
                  floor_score, nomannic_score, honghanh_score]
    for s in all_scores:
        assert s.pair_ap is not None and math.isfinite(s.pair_ap)
        assert 0.0 <= s.pair_ap <= 1.0

    # --- (2) Assemble the n=6 ladder under ONE canonical recipe id (Req 2.6). Built
    #     directly because the self-earned/own LB coordinates have no on-disk artifact
    #     filename; the LB is EXTERNAL ground truth read here, never faked (Rule 1).
    points: List[LadderPoint] = [
        LadderPoint(
            config_id="ladder_032580_cand14",
            real_lb=_CAND14_LB,
            local_pair_ap=float(v2_score.pair_ap),
            local_combined=float(v2_score.combined),
            recipe_id=recipe_id,
            source="cand_14 v2 single-XGB PU risk; drift ~0.7618 (DIAGNOSTIC, §56)",
        ),
        LadderPoint(
            config_id="ladder_037063_cand17",
            real_lb=_CAND17_LB,
            local_pair_ap=float(v3_score.pair_ap),
            local_combined=float(v3_score.combined),
            recipe_id=recipe_id,
            source="cand_17 v3 rank-avg(XGB d4+d6) PU risk; drift ~0.7611 (DIAGNOSTIC, §56)",
        ),
        LadderPoint(
            config_id="ladder_041289_exp3c",
            real_lb=_EXP3C_LB,
            local_pair_ap=float(exp3c_score.pair_ap),
            local_combined=float(exp3c_score.combined),
            recipe_id=recipe_id,
            source="cand_exp3c v5+MFg one XGB; drift 0.7620 (DIAGNOSTIC, §56)",
        ),
        LadderPoint(
            config_id="ladder_044519_floor",
            real_lb=_FLOOR_LB,
            local_pair_ap=float(floor_score.pair_ap),
            local_combined=float(floor_score.combined),
            recipe_id=recipe_id,
            source="floor stack (v5+MFg+DIRc); drift ~0.7621 (DIAGNOSTIC, §56); reuse §55",
        ),
        LadderPoint(
            config_id="ladder_056202_nomannic",
            real_lb=_NOMANNIC_LB,
            local_pair_ap=float(nomannic_score.pair_ap),
            local_combined=float(nomannic_score.combined),
            recipe_id=recipe_id,
            source="nomannic own 129-col pair matrix; drift ~0.7594 (DIAGNOSTIC, §56); reuse §55",
        ),
        LadderPoint(
            config_id="ladder_064262_honghanh",
            real_lb=_HONGHANH_LB,
            local_pair_ap=float(honghanh_score.pair_ap),
            local_combined=float(honghanh_score.combined),
            recipe_id=recipe_id,
            source="honghanh competitor 064469 recipe (self-earned LB 0.64262); reuse §55",
        ),
    ]

    validator = LadderValidator(scorer)
    verdict = validator.rank_tracking(points)

    scorable = [p for p in points if p.status == SCORABLE]
    assert len(scorable) == 6
    assert verdict.n_points == 6
    assert verdict.primary.metric_name == "pair_ap"
    assert verdict.secondary.metric_name == "combined"
    assert verdict.verdict in {"TRUSTED", "WEAK", "NULL"}

    # --- (3) PRIMARY / SECONDARY Spearman + t-CI + BOOTSTRAP CI.
    real_lb = [p.real_lb for p in points]
    pair_ap = [float(p.local_pair_ap) for p in points]
    combined = [float(p.local_combined) for p in points]
    r_primary = spearman_corr(pair_ap, real_lb)
    r_secondary = spearman_corr(combined, real_lb)

    t_lo, t_hi, t_se = _spearman_ci_t(r_primary, len(points))
    b_lo, b_hi, b_draws = _spearman_bootstrap_ci(pair_ap, real_lb, n_draws=10000, seed=7)
    frac_below_bar = float(np.mean(b_draws < OPERATIONAL_BAR))

    # ACTUAL local ordering (auditable, §56 guardrail): LB order is fixed ascending; is
    # PairAP monotone increasing in that order?
    lb_order = [_CAND14_LB, _CAND17_LB, _EXP3C_LB, _FLOOR_LB, _NOMANNIC_LB, _HONGHANH_LB]
    pair_ap_in_lb_order = [ap for _lb, ap in sorted(zip(lb_order, pair_ap))]
    orders_correctly = pair_ap_in_lb_order == sorted(pair_ap_in_lb_order)

    # --- (4) §56 PRE-REGISTERED verdict tiers, applied EXACTLY (bar NOT moved, Rule 12).
    if r_primary <= 0.0:
        s56_verdict = "NULL"
    elif r_primary >= OPERATIONAL_BAR and b_lo > 0.0 and len(points) >= 5:
        s56_verdict = "TRUSTED"
    else:  # 0<sp<0.7, OR >=0.7 but bootstrap CI includes 0
        s56_verdict = "WEAK"

    labels_by_lb = ["cand_14", "cand_17", "cand_exp3c", "floor", "nomannic", "honghanh"]
    aps_by_lb = pair_ap  # points already built ascending by LB
    print(
        "\n[§56 n=6 scorer-trust ladder] MEASURED dev-holdout PairAPs "
        "(drift AUC = REPORTED DIAGNOSTIC, not a gate — §55):"
    )
    confirmed = [v2.n_holdout_confirmed, v3.n_holdout_confirmed,
                 exp3c.n_holdout_confirmed, floor.n_holdout_confirmed,
                 nomannic.n_confirmed, honghanh.n_confirmed]
    for name, lb, ap, nconf in zip(labels_by_lb, lb_order, aps_by_lb, confirmed):
        print(f"  {name:11s} real_lb={lb:.5f}  PairAP={ap:.4f}  "
              f"confirmed_n={nconf}  drift_auc={_DRIFT[name]:.4f}(diag)")
    print(f"  local PairAP order (by ascending LB) = {[round(a, 4) for a in aps_by_lb]}"
          f"; orders_correctly={orders_correctly}")
    print(f"  PRIMARY(PairAP) spearman = {r_primary:.4f}")
    print(f"    t-approx 95% CI ~ [{t_lo:.4f}, {t_hi:.4f}] (SE={t_se})"
          + ("  <- DEGENERATE (r=1); rely on bootstrap" if abs(r_primary) >= 1.0 else ""))
    print(f"    BOOTSTRAP 95% CI = [{b_lo:.4f}, {b_hi:.4f}] over 10000 draws; "
          f"frac(resample r<0.7) = {frac_below_bar:.4f}")
    print(f"  SECONDARY(combined) spearman = {r_secondary:.4f}")
    print(f"  §56 VERDICT = {s56_verdict}  (bar {OPERATIONAL_BAR}, NOT moved — Rule 12; "
          f"TRUSTED needs sp>=0.7 AND bootstrap-CI-lower>0 at n>=5)")
    if s56_verdict == "TRUSTED":
        print("  SCOPE (Rule 4): trustworthy for RELATIVE RANKING across ~0.33-0.64 LB "
              "only; does NOT license absolute-LB extrapolation nor trust outside [0.33,0.64].")

    # Assert the experiment RAN and produced finite statistics + a valid tier — never a
    # direction (Rule 3: the measured verdict is the deliverable).
    assert s56_verdict in {"TRUSTED", "WEAK", "NULL"}
    assert math.isfinite(r_primary) and math.isfinite(r_secondary)
    assert math.isfinite(b_lo) and math.isfinite(b_hi)
    assert b_lo <= b_hi
