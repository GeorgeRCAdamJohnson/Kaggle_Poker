"""Fast-pass ladder measurement driver (§49 -> §50). NOT a test; produces MEASURED numbers.

Runs, in the pre-registered SEQUENTIAL order (contract Rules 5, 17):
  1. adversarial drift gate on each new feature block (v3, v2) -> PASS/DISQUALIFIED
  2. re-run only drift-PASSING own points + floor + competitor on the dev holdout
  3. score the ladder + rank_tracking, compute PairAP Spearman + ~95% CI (t-approx)
     and combined Spearman, apply the §49 verdict tiers.

Emits a JSON blob to stdout the caller records verbatim into dossier §50. Makes NO
LB claim: dev-holdout PairAP is a MEASURED LOCAL number; real LB is EXTERNALLY VERIFIED
from the artifact filename.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from poker_collusion.config import PipelineConfig
from poker_collusion.validation.lb_cv import spearman_corr

from anchor_repro.competitor_recipe import (
    COMPETITOR_DIGITS,
    build_competitor_recipe_runner,
    default_competitor_paths,
)
from anchor_repro.ladder import SCORABLE, LadderValidator
from anchor_repro.models import ScoringRecipe
from anchor_repro.recipe_registry import FLOOR_DIGITS, default_cache_locations
from anchor_repro.own_submission_recipes import (
    V2_DIGITS,
    V3_DIGITS,
    adversarial_drift_auc,
    build_multi_point_ladder_runner,
    build_own_submission_runner,
    default_own_cache_locations,
)
from anchor_repro.scorer import CanonicalScorer

POKER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = POKER_ROOT / "data" / "poker"
ARTIFACT_DIR = POKER_ROOT / "outputs" / "poker_collusion"


def spearman_ci_t(r: float, n: int, z: float = 1.96) -> tuple:
    """Approx 95% CI for a Spearman r via the t-approx SE = sqrt((1-r^2)/(n-2)).

    Returns (low, high, se). Clamped to [-1, 1]. Undefined for n < 3 (SE needs n-2>0
    and a finite denominator) -> returns (nan, nan, nan) so the caller reports it as
    not-computable rather than fabricating an interval (contract Rules 1, 9).
    """
    if n < 3 or abs(r) >= 1.0:
        # r == +/-1 gives SE 0 (degenerate); n<3 has no dof. Honest non-CI.
        if abs(r) >= 1.0 and n >= 3:
            return (r, r, 0.0)
        return (float("nan"), float("nan"), float("nan"))
    se = math.sqrt((1.0 - r * r) / (n - 2))
    return (max(-1.0, r - z * se), min(1.0, r + z * se), se)


def main() -> None:
    labels = pd.read_csv(DATA_DIR / "development_labels.csv")
    ev_path = DATA_DIR / "development_evidence.csv"
    evidence = pd.read_csv(ev_path) if ev_path.is_file() else None

    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=DATA_DIR),
        labels=labels,
        evidence=evidence,
    )
    recipe_id = scorer.recipe.recipe_id

    out: dict = {"recipe_id": recipe_id, "drift": {}, "points": [], "verdict": {}}

    # --- 1. SEQUENTIAL drift gate (Rules 5, 17): PASS first, then trust PairAP. ---
    own_runners = []
    drift_pass_digits = set()
    for digits, ver in ((V3_DIGITS, "v3"), (V2_DIGITS, "v2")):
        caches = default_own_cache_locations(POKER_ROOT, ver)
        dg = adversarial_drift_auc(caches.dev_feats, caches.eval_feats, block=ver)
        out["drift"][ver] = {
            "digits": digits,
            "auc": dg.auc,
            "threshold": dg.threshold,
            "verdict": dg.verdict,
            "n_features": dg.n_features,
            "n_dev": dg.n_dev,
            "n_eval": dg.n_eval,
        }
        print(f"[drift] {ver} ({digits}): AUC={dg.auc:.4f} -> {dg.verdict}", flush=True)
        if dg.passed:
            drift_pass_digits.add(digits)
            own_runners.append(
                build_own_submission_runner(
                    digits, caches, labels, canonical_recipe_id=recipe_id, evidence=evidence
                )
            )

    # --- 2. Build the multi-point ladder runner (floor + competitor + drift-PASS own). ---
    floor_caches = default_cache_locations(POKER_ROOT)
    competitor_paths = default_competitor_paths(POKER_ROOT)
    competitor_runner = build_competitor_recipe_runner(
        competitor_paths, labels, canonical_recipe_id=recipe_id, evidence=evidence
    )
    runner = build_multi_point_ladder_runner(
        floor_caches,
        labels,
        canonical_recipe_id=recipe_id,
        competitor_runner=competitor_runner,
        own_runners=own_runners,
        evidence=evidence,
    )

    # Artifacts: floor + competitor + each drift-PASSING own point.
    artifacts = [
        ARTIFACT_DIR / f"submission_best_{FLOOR_DIGITS}.csv",
        ARTIFACT_DIR / f"submission_best_{COMPETITOR_DIGITS}.csv",
    ]
    for digits in (V3_DIGITS, V2_DIGITS):
        if digits in drift_pass_digits:
            artifacts.append(ARTIFACT_DIR / f"submission_best_{digits}.csv")

    validator = LadderValidator(scorer, recipe_runner=runner)
    points = validator.score_ladder(artifacts)
    verdict = validator.rank_tracking(points)

    for p in points:
        out["points"].append(
            {
                "config_id": p.config_id,
                "real_lb": p.real_lb,
                "local_pair_ap": p.local_pair_ap,
                "local_combined": p.local_combined,
                "status": p.status,
                "blocked_reason": p.blocked_reason,
            }
        )
        tag = (
            f"PairAP={p.local_pair_ap:.4f}" if p.local_pair_ap is not None else "BLOCKED"
        )
        print(f"[point] {p.config_id} realLB={p.real_lb:.5f} {p.status} {tag}", flush=True)

    scorable = [p for p in points if p.status == SCORABLE]
    n = len(scorable)
    real_lb = [p.real_lb for p in scorable]
    pair_ap = [float(p.local_pair_ap) for p in scorable]
    combined = [float(p.local_combined) for p in scorable]

    r_primary = spearman_corr(pair_ap, real_lb) if n >= 2 else float("nan")
    r_secondary = spearman_corr(combined, real_lb) if n >= 2 else float("nan")
    lo, hi, se = spearman_ci_t(r_primary, n)

    # §49 verdict tiers: TRUSTED requires spearman>=0.7 AND CI excludes 0 at n>=5.
    ci_excludes_zero = (not math.isnan(lo)) and (lo > 0.0 or hi < 0.0)
    if r_primary <= 0.0:
        tier = "NULL"
    elif r_primary >= 0.7 and n >= 5 and ci_excludes_zero:
        tier = "TRUSTED"
    else:
        tier = "WEAK"

    out["verdict"] = {
        "n_points": n,
        "primary_metric": "pair_ap",
        "primary_spearman": r_primary,
        "primary_ci_low": lo,
        "primary_ci_high": hi,
        "primary_ci_se": se,
        "primary_ci_excludes_zero": ci_excludes_zero,
        "secondary_metric": "combined",
        "secondary_spearman": r_secondary,
        "validator_verdict": verdict.verdict,
        "validator_primary_spearman": verdict.primary.spearman,
        "validator_secondary_spearman": verdict.secondary.spearman,
        "fastpass_tier": tier,
        "n_target": 5,
        "n_min_informative": 4,
    }

    print(
        f"\n[VERDICT] n={n} PRIMARY(PairAP) spearman={r_primary:.4f} "
        f"95%CI~[{lo:.4f},{hi:.4f}] excl0={ci_excludes_zero} | "
        f"SECONDARY(combined) spearman={r_secondary:.4f} | TIER={tier}",
        flush=True,
    )
    print("\n===JSON_BEGIN===")
    print(json.dumps(out, indent=2))
    print("===JSON_END===")


if __name__ == "__main__":
    main()
