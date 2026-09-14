"""TIERED / ORDINAL consensus builder + pre-registered evidence gate (§80).

BUILD + MEASURE ONLY. No Kaggle submit. Does NOT touch ``submission.csv`` or any
``submission_best_*.csv``; writes only under ``outputs/poker_collusion/tiered_consensus/``.

Why tiered, not a weighted mean
-------------------------------
§78 blended the three independent detectors into a FLAT weighted mean of percentiles.
That nulled: mixing a strong ranker (honghanh) with a weaker one (floor) on a linear
scale drags the strong pairs down toward the weak model's disagreement. §80 instead
uses AGREEMENT DEPTH as an ORDINAL tier: a pair that all three detectors flag as high
is strictly above a pair only two competitors flag, which is strictly above a pair only
one competitor flags, which is strictly above everything else. Ranking is tier-major,
within-tier-minor, so tiers can never cross — the strong-model ranking is preserved
inside each tier while agreement depth sets the coarse order.

The three independent detectors (eval risk CSVs, 112,540 pairs, identical pair sets)
-----------------------------------------------------------------------------------
* floor      — LB 0.44519 — ``submission_best_044519.csv``
* nomannic   — LB 0.56202 — ``repro_nomannic/submission.csv``
* honghanh   — LB 0.64262 — ``repro_064469/submission.csv``

NEW-BEST evidence stack
-----------------------
A submission just scored a NEW BEST: ``candidate_honghanh_pure_ranker_evidence.csv``
= LB 0.65242 (honghanh risk+behavior + the pure-ranker evidence, +0.0098 over honghanh
alone). So the tiered CSV REUSES that new-best file's ``predicted_behavior`` +
``evidence_hand_1..5`` columns VERBATIM (join by pair_id). The tiering experiment thus
changes ONLY the risk ranking and stacks on the confirmed evidence gain — it does NOT
reuse honghanh's older ``submission.csv`` evidence.

Pre-registered evidence gate (§80) — decides SUBMIT vs NO-SUBMIT
----------------------------------------------------------------
Two tests on the COMMON confirmed dev pairs (intersection of the 3 dev prediction sets,
same construction as §78). Everything reuses persisted dev caches — no new fit.

* TEST 1 (dev PairAP): build the SAME tiered ranking on the common DEV pairs (tiers
  from the 3 models' dev percentiles, Q=0.95), score canonical dev PairAP (behavior
  "none", NO_EVIDENCE — PairAP only), compare to honghanh-alone dev PairAP on the same
  common set. Clears iff ``tiered_dev_PairAP >= honghanh_dev_PairAP + 0.005``.
* TEST 2 (Tier-1 dev precision — the falsifiable test): among common DEV pairs, take
  the tiered ranking's TIER 1 set (all-three-high on dev) and compute
  ``precision = fraction with label==1``. Compare to honghanh-alone's TOP-N dev pairs
  (N = |Tier1|) by oof_risk, same precision. Clears iff
  ``Tier1_precision >= honghanh_topN_precision + 0.05``.

VERDICT (locked §80): SUBMIT-WORTHY iff (Test 1 clears) OR (Test 2 clears); else
NO-SUBMIT (honest null). We report the verdict; we DO NOT submit.

Contract compliance
-------------------
* Rule 1/6 (external judge, reuse): dev rankings reuse ``rerun_floor_stack`` /
  ``rerun_nomannic_recipe`` / honghanh's persisted dev OOF; dev PairAP is scored via
  the canonical verbatim official metric (:class:`anchor_repro.scorer.CanonicalScorer`).
* Rule 2 (pre-register the bar): thresholds (+0.005 PairAP, +0.05 precision), the null
  (honghanh alone), and the exact held-out set (the common dev intersection) are fixed
  HERE, before running, and judged on that set. No post-hoc narrative.
* Rule 3 (honest nulls): a NO-SUBMIT verdict is a valid, reported result.
* Rule 4/16 (no extrapolation, shrink small samples): the tier gate is a rank threshold
  reused identically on dev; Tier-1 precision is reported alongside |Tier1| and the raw
  true-positive counts so a tiny-Tier1 result is not oversold.
* Rule 9 (state uncertainty): dev PairAP / precision are MEASURED LOCAL numbers, NOT LB
  claims. The eval tier counts describe the candidate; they are not an LB prediction.

Run: ``python -m anchor_repro.tiered_consensus`` (from the poker package root).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from poker_collusion.config import PipelineConfig
from poker_collusion.submission.writer import SUBMISSION_COLUMNS

from anchor_repro.models import ScoringRecipe
from anchor_repro.nomannic_recipe import (
    default_nomannic_cache_locations,
    rerun_nomannic_recipe,
)
from anchor_repro.recipe_registry import default_cache_locations, rerun_floor_stack
from anchor_repro.scorer import CanonicalScorer

__all__ = [
    "MODELS",
    "MODEL_ORDER",
    "Q_GRID",
    "PRIMARY_Q",
    "TEST1_MARGIN",
    "TEST2_MARGIN",
    "run_tiered_consensus",
]

# --------------------------------------------------------------------------- #
# Constants (verbatim from §80 / the dossier).                                 #
# --------------------------------------------------------------------------- #

#: Model id -> (eval submission CSV relative to poker root, real LB).
MODELS: Dict[str, Tuple[str, float]] = {
    "floor": ("outputs/poker_collusion/submission_best_044519.csv", 0.44519),
    "nomannic": ("outputs/poker_collusion/repro_nomannic/submission.csv", 0.56202),
    "honghanh": ("outputs/poker_collusion/repro_064469/submission.csv", 0.64262),
}

#: The order the three percentile columns are built + reported in.
MODEL_ORDER: Tuple[str, str, str] = ("floor", "nomannic", "honghanh")

#: The two competitor models (tiers 2/3 are defined by competitor agreement only).
COMPETITORS: Tuple[str, str] = ("nomannic", "honghanh")

#: NEW-BEST evidence stack — its behavior + evidence columns are reused VERBATIM.
NEW_BEST_EVIDENCE: str = (
    "outputs/poker_collusion/evidence_ranker/candidate_honghanh_pure_ranker_evidence.csv"
)

#: honghanh's persisted dev-holdout OOF (oof_risk + label over the confirmed pairs).
HONGHANH_DEV_OOF: str = (
    "outputs/poker_collusion/repro_064469/dev_holdout_oof_064469.parquet"
)

#: The two "high" quantile thresholds to report; Q=0.95 drives the primary CSV.
Q_GRID: Tuple[float, float] = (0.90, 0.95)
PRIMARY_Q: float = 0.95

#: Pre-registered bars (§80). Test 1 = dev PairAP margin; Test 2 = Tier-1 precision margin.
TEST1_MARGIN: float = 0.005
TEST2_MARGIN: float = 0.05

#: The eval row count the candidate submission must have.
EVAL_ROW_COUNT: int = 112_540

#: The five evidence-hand columns.
EVIDENCE_COLS: Tuple[str, ...] = (
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
)


# --------------------------------------------------------------------------- #
# Core tiered-ranking arithmetic (pure rank arithmetic — no fit).              #
# --------------------------------------------------------------------------- #


def _percentile(values: Sequence[float]) -> np.ndarray:
    """Rank-normalise ``values`` to a percentile in [0,1] (average ties / N).

    Uses ``scipy.stats.rankdata`` average-tie ranks divided by N — a strictly monotone
    transform of the ranking (PairAP-invariant) mapped into the unit interval. Highest
    value -> percentile ~1.0.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    return rankdata(arr, method="average") / float(n)


def _assign_tiers(
    percentiles: Dict[str, np.ndarray], q: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Assign each pair its HIGHEST qualifying tier from the 3 models' percentiles.

    ``high[m] = percentile[m] >= q``. Then, per §80:

    * TIER 1: high in ALL THREE.
    * TIER 2: high in BOTH COMPETITORS (nomannic AND honghanh) but not all three.
    * TIER 3: high in EXACTLY ONE competitor (nomannic XOR honghanh), not tier 1/2.
    * TIER 4: everything else.

    Returns the integer tier array (1..4) and the per-model boolean ``high`` masks.
    """
    high = {m: (percentiles[m] >= q) for m in MODEL_ORDER}
    all_three = high["floor"] & high["nomannic"] & high["honghanh"]
    both_comp = high["nomannic"] & high["honghanh"]
    exactly_one_comp = high["nomannic"] ^ high["honghanh"]  # XOR

    n = len(percentiles[MODEL_ORDER[0]])
    tier = np.full(n, 4, dtype=int)
    # Assign lowest (best) tier last so the HIGHEST qualifying tier wins.
    tier[exactly_one_comp] = 3
    tier[both_comp] = 2
    tier[all_three] = 1
    return tier, high


def _tiered_score(tier: np.ndarray, percentiles: Dict[str, np.ndarray]) -> np.ndarray:
    """Tier-major, within-tier-minor score, min-maxed to [0,1]; tier 1 = highest.

    Within a tier, order by the competitor mean percentile
    ``(nomannic_pct + honghanh_pct) / 2``. Construction (§80): tiers never cross —
    ``raw = (5 - tier) + 0.999 * within_tier_meanpct`` gives disjoint per-tier bands
    ([4,5) for tier1 down to [1,2) for tier4 since within ∈ [0, 0.999]); then min-max to
    [0,1]. Tier 1 therefore maps to the top band and tier 4 to the bottom.
    """
    within = (percentiles["nomannic"] + percentiles["honghanh"]) / 2.0
    raw = (5.0 - tier).astype(float) + 0.999 * within
    lo, hi = float(raw.min()), float(raw.max())
    if hi <= lo:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)


def _tier_counts(tier: np.ndarray) -> Dict[str, int]:
    """Count of pairs in each tier (as a JSON-friendly dict)."""
    return {f"tier{t}": int((tier == t).sum()) for t in (1, 2, 3, 4)}


# --------------------------------------------------------------------------- #
# CanonicalScorer-ready neutral frame (PairAP depends only on risk_score).     #
# --------------------------------------------------------------------------- #


def _neutral_frame(pair_ids: Sequence[str], risk: Sequence[float]) -> pd.DataFrame:
    """A CanonicalScorer-ready frame: behavior "none", NO_EVIDENCE (PairAP only)."""
    frame = pd.DataFrame(
        {
            "pair_id": [str(p) for p in pair_ids],
            "risk_score": np.asarray(risk, dtype=float).clip(0.0, 1.0),
            "predicted_behavior": "none",
        }
    )
    for col in EVIDENCE_COLS:
        frame[col] = "NO_EVIDENCE"
    return frame[list(SUBMISSION_COLUMNS)]


# --------------------------------------------------------------------------- #
# PART 1 — Build the tiered EVAL CSV.                                          #
# --------------------------------------------------------------------------- #


@dataclass
class TieredCandidate:
    """The built tiered eval candidate + its validation + tier counts."""

    path: str
    n_rows: int
    schema_ok: bool
    pair_set_matches_eval: bool
    risk_in_unit_interval: bool
    no_dup_evidence_within_pair: bool
    tier_counts: Dict[str, Dict[str, int]]  # "Q0.90"/"Q0.95" -> tier -> count
    spearman_vs_honghanh: float
    spearman_vs_new_best: float


def _load_eval_percentiles(poker_root: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Merge the 3 eval CSVs on pair_id (inner), return merged frame + per-model pcts."""
    merged: Optional[pd.DataFrame] = None
    for m in MODEL_ORDER:
        path = poker_root / MODELS[m][0]
        df = pd.read_csv(path, usecols=["pair_id", "risk_score"], dtype={"pair_id": str})
        df = df.rename(columns={"risk_score": f"{m}_risk"})
        merged = df if merged is None else merged.merge(df, on="pair_id", how="inner")
    assert merged is not None
    percentiles: Dict[str, np.ndarray] = {}
    for m in MODEL_ORDER:
        percentiles[m] = _percentile(merged[f"{m}_risk"].to_numpy())
        merged[f"{m}_pct"] = percentiles[m]
    return merged, percentiles


def build_tiered_candidate(poker_root: Path, out_dir: Path) -> TieredCandidate:
    """PART 1: build + validate the tiered eval candidate CSV (Q=0.95 primary)."""
    merged, percentiles = _load_eval_percentiles(poker_root)

    # Tier counts for BOTH quantiles (report both; use PRIMARY_Q for the CSV).
    tier_counts: Dict[str, Dict[str, int]] = {}
    tiers_by_q: Dict[float, np.ndarray] = {}
    for q in Q_GRID:
        tier_q, _ = _assign_tiers(percentiles, q)
        tiers_by_q[q] = tier_q
        tier_counts[f"Q{q:.2f}"] = _tier_counts(tier_q)

    tier = tiers_by_q[PRIMARY_Q]
    tiered_risk = _tiered_score(tier, percentiles)

    # NEW-BEST evidence stack: reuse predicted_behavior + evidence columns VERBATIM.
    new_best = pd.read_csv(poker_root / NEW_BEST_EVIDENCE, dtype={"pair_id": str})

    cand = pd.DataFrame({"pair_id": merged["pair_id"].astype(str)})
    cand["risk_score"] = tiered_risk.clip(0.0, 1.0)
    reuse_cols = ["pair_id", "predicted_behavior", *EVIDENCE_COLS]
    cand = cand.merge(new_best[reuse_cols], on="pair_id", how="left")
    cand = cand[list(SUBMISSION_COLUMNS)]

    # ---- Validate ----
    n_rows = len(cand)
    schema_ok = tuple(cand.columns) == SUBMISSION_COLUMNS
    eval_pairs = pd.read_csv(
        poker_root / "data" / "poker" / "evaluation_pairs.csv", dtype={"pair_id": str}
    )
    pair_set_matches_eval = set(cand["pair_id"]) == set(eval_pairs["pair_id"].astype(str))
    risk = pd.to_numeric(cand["risk_score"], errors="coerce")
    risk_in_unit_interval = bool(risk.notna().all() and risk.between(0.0, 1.0).all())

    def _dup(row) -> bool:
        hands = [h for h in row if isinstance(h, str) and h != "NO_EVIDENCE"]
        return len(hands) != len(set(hands))

    no_dup = not cand[list(EVIDENCE_COLS)].apply(_dup, axis=1).any()

    # Spearman of tiered risk vs honghanh risk AND vs the new-best file's risk.
    hh_risk = (
        merged.set_index("pair_id")["honghanh_risk"].reindex(cand["pair_id"]).to_numpy()
    )
    rho_hh, _ = spearmanr(cand["risk_score"].to_numpy(), hh_risk)
    nb_risk = (
        new_best.set_index("pair_id")["risk_score"].reindex(cand["pair_id"]).to_numpy()
    )
    rho_nb, _ = spearmanr(cand["risk_score"].to_numpy(), nb_risk)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidate_tiered_consensus_Q95.csv"
    cand.to_csv(out_path, index=False)

    return TieredCandidate(
        path=str(out_path),
        n_rows=n_rows,
        schema_ok=schema_ok,
        pair_set_matches_eval=pair_set_matches_eval,
        risk_in_unit_interval=risk_in_unit_interval,
        no_dup_evidence_within_pair=bool(no_dup),
        tier_counts=tier_counts,
        spearman_vs_honghanh=float(rho_hh),
        spearman_vs_new_best=float(rho_nb),
    )


# --------------------------------------------------------------------------- #
# PART 2 — Pre-registered evidence gate (dev caches; no new fit).              #
# --------------------------------------------------------------------------- #


def _floor_dev_risk(
    poker_root: Path, labels: pd.DataFrame, evidence: pd.DataFrame
) -> Dict[str, float]:
    """{pair_id: risk} for the floor dev-holdout confirmed pairs (reused runner)."""
    result = rerun_floor_stack(default_cache_locations(poker_root), labels, evidence=evidence)
    df = result.dev_predictions
    return dict(zip(df["pair_id"].astype(str), df["risk_score"].astype(float)))


def _nomannic_dev_risk(
    poker_root: Path, labels: pd.DataFrame, evidence: pd.DataFrame
) -> Dict[str, float]:
    """{pair_id: risk} for the nomannic dev-holdout confirmed pairs (reused runner)."""
    result = rerun_nomannic_recipe(
        default_nomannic_cache_locations(poker_root), labels, evidence=evidence
    )
    df = result.dev_predictions
    return dict(zip(df["pair_id"].astype(str), df["risk_score"].astype(float)))


def _honghanh_dev(poker_root: Path) -> pd.DataFrame:
    """honghanh's confirmed (is_labeled) dev pairs: pair_id, oof_risk, label (reused OOF)."""
    oof = pd.read_parquet(poker_root / HONGHANH_DEV_OOF)
    conf = oof[oof["is_labeled"].astype(bool)].copy()
    conf["pair_id"] = conf["pair_id"].astype(str)
    return conf[["pair_id", "oof_risk", "label"]]


@dataclass
class EvidenceGate:
    """The two pre-registered tests + the SUBMIT / NO-SUBMIT verdict (§80)."""

    n_common: int
    # Test 1 (dev PairAP).
    tiered_dev_pair_ap: float
    honghanh_dev_pair_ap: float
    test1_delta: float
    test1_clears: bool
    # Test 2 (Tier-1 dev precision).
    tier1_size: int
    tier1_precision: float
    tier1_true_positives: int
    honghanh_topn_precision: float
    honghanh_topn_true_positives: int
    test2_delta: float
    test2_clears: bool
    # Verdict.
    submit_worthy: bool
    verdict: str
    cleared_by: List[str]
    notes: List[str] = field(default_factory=list)


def build_evidence_gate(
    poker_root: Path,
    scorer: CanonicalScorer,
    labels: pd.DataFrame,
    evidence: pd.DataFrame,
) -> EvidenceGate:
    """PART 2: run Test 1 + Test 2 on the COMMON confirmed dev set; lock the verdict."""
    notes: List[str] = []

    floor_risk = _floor_dev_risk(poker_root, labels, evidence)
    nomannic_risk = _nomannic_dev_risk(poker_root, labels, evidence)
    honghanh_dev = _honghanh_dev(poker_root)
    honghanh_risk = dict(
        zip(honghanh_dev["pair_id"], honghanh_dev["oof_risk"].astype(float))
    )
    honghanh_label = dict(zip(honghanh_dev["pair_id"], honghanh_dev["label"].astype(int)))

    notes.append(
        "dev prediction set sizes (confirmed): "
        f"floor={len(floor_risk)}, nomannic={len(nomannic_risk)}, "
        f"honghanh={len(honghanh_risk)}"
    )

    # Ground-truth label (label==1 = positive) from the dev labels, as a fallback / cross
    # check; honghanh's OOF carries the same label for its confirmed pairs.
    label_map = dict(zip(labels["pair_id"].astype(str), labels["label"].astype(int)))

    # Common intersection so all 3 detectors are comparable (same construction as §78).
    common = set(floor_risk) & set(nomannic_risk) & set(honghanh_risk)
    common_ids = sorted(common)
    n_common = len(common_ids)
    notes.append(f"n_common (intersection of all 3 dev prediction sets) = {n_common}")

    # Per-model dev risk aligned to the common set, then percentiles ON the common set.
    aligned = {
        "floor": np.array([floor_risk[p] for p in common_ids], dtype=float),
        "nomannic": np.array([nomannic_risk[p] for p in common_ids], dtype=float),
        "honghanh": np.array([honghanh_risk[p] for p in common_ids], dtype=float),
    }
    dev_pct = {m: _percentile(aligned[m]) for m in MODEL_ORDER}

    # The SAME tiered ranking on the common dev pairs, Q=PRIMARY_Q.
    tier, _ = _assign_tiers(dev_pct, PRIMARY_Q)
    tiered_dev_risk = _tiered_score(tier, dev_pct)

    # ---- TEST 1: dev PairAP, tiered vs honghanh-alone, on the common set ----
    tiered_frame = _neutral_frame(common_ids, tiered_dev_risk)
    honghanh_frame = _neutral_frame(common_ids, dev_pct["honghanh"])
    tiered_ap = float(scorer.score_dev_predictions(tiered_frame).pair_ap)
    honghanh_ap = float(scorer.score_dev_predictions(honghanh_frame).pair_ap)
    test1_delta = tiered_ap - honghanh_ap
    test1_clears = tiered_ap >= honghanh_ap + TEST1_MARGIN

    # ---- TEST 2: Tier-1 dev precision vs honghanh top-N precision ----
    labels_common = np.array(
        [honghanh_label.get(p, label_map.get(p, 0)) for p in common_ids], dtype=int
    )
    tier1_mask = tier == 1
    tier1_size = int(tier1_mask.sum())
    if tier1_size > 0:
        tier1_tp = int(labels_common[tier1_mask].sum())
        tier1_precision = tier1_tp / tier1_size
    else:
        tier1_tp = 0
        tier1_precision = 0.0

    # honghanh-alone top-N by oof_risk on the common set, N = |Tier1|.
    if tier1_size > 0:
        order = np.argsort(-aligned["honghanh"], kind="stable")  # descending risk
        topn_idx = order[:tier1_size]
        hh_topn_tp = int(labels_common[topn_idx].sum())
        hh_topn_precision = hh_topn_tp / tier1_size
    else:
        hh_topn_tp = 0
        hh_topn_precision = 0.0

    test2_delta = tier1_precision - hh_topn_precision
    test2_clears = (tier1_size > 0) and (tier1_precision >= hh_topn_precision + TEST2_MARGIN)

    if tier1_size == 0:
        notes.append(
            "Tier-1 is EMPTY on the common dev set (no pair is all-three-high at "
            f"Q={PRIMARY_Q}); Test 2 cannot clear (reported precision 0.0)."
        )

    # ---- Verdict (locked §80): SUBMIT iff (Test 1 clears) OR (Test 2 clears) ----
    cleared_by: List[str] = []
    if test1_clears:
        cleared_by.append("Test 1 (dev PairAP)")
    if test2_clears:
        cleared_by.append("Test 2 (Tier-1 dev precision)")
    submit_worthy = bool(cleared_by)
    verdict = "SUBMIT-WORTHY" if submit_worthy else "NO-SUBMIT"

    return EvidenceGate(
        n_common=n_common,
        tiered_dev_pair_ap=tiered_ap,
        honghanh_dev_pair_ap=honghanh_ap,
        test1_delta=test1_delta,
        test1_clears=test1_clears,
        tier1_size=tier1_size,
        tier1_precision=tier1_precision,
        tier1_true_positives=tier1_tp,
        honghanh_topn_precision=hh_topn_precision,
        honghanh_topn_true_positives=hh_topn_tp,
        test2_delta=test2_delta,
        test2_clears=test2_clears,
        submit_worthy=submit_worthy,
        verdict=verdict,
        cleared_by=cleared_by,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #


def run_tiered_consensus(poker_root: Path) -> Dict[str, object]:
    """Build the tiered eval CSV, run the §80 evidence gate, write the JSON summary."""
    poker_root = Path(poker_root)
    config = PipelineConfig()
    data_dir = Path(config.input_dir)
    labels = pd.read_csv(data_dir / "development_labels.csv")
    evidence = pd.read_csv(data_dir / "development_evidence.csv")

    out_dir = poker_root / "outputs" / "poker_collusion" / "tiered_consensus"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("§80 TIERED / ORDINAL CONSENSUS + EVIDENCE GATE — BUILD + MEASURE ONLY (no submit)")
    print("=" * 88)

    # ---- PART 1: build the tiered eval CSV ----
    print("PART 1: build the tiered eval CSV (Q=0.95 primary; Q=0.90 reported) ...")
    cand = build_tiered_candidate(poker_root, out_dir)
    print(f"  tier counts Q=0.90: {cand.tier_counts['Q0.90']}")
    print(f"  tier counts Q=0.95: {cand.tier_counts['Q0.95']}")
    print(f"  wrote {cand.path}")
    print(
        f"  validation: rows={cand.n_rows}, schema_ok={cand.schema_ok}, "
        f"pair_set==eval={cand.pair_set_matches_eval}, "
        f"risk_in_[0,1]={cand.risk_in_unit_interval}, "
        f"no_dup_evidence={cand.no_dup_evidence_within_pair}"
    )
    print(f"  Spearman(tiered risk, honghanh risk) = {cand.spearman_vs_honghanh:.4f}")
    print(f"  Spearman(tiered risk, new-best risk) = {cand.spearman_vs_new_best:.4f}")

    # ---- PART 2: pre-registered evidence gate ----
    print("PART 2: pre-registered evidence gate (dev caches; no new fit) ...")
    scorer = CanonicalScorer(
        ScoringRecipe(recipe_id="tiered_consensus_v1"),
        config,
        labels=labels,
        evidence=evidence,
    )
    gate = build_evidence_gate(poker_root, scorer, labels, evidence)
    print(f"  n_common = {gate.n_common}")
    print(
        f"  TEST 1 (dev PairAP): tiered={gate.tiered_dev_pair_ap:.4f}, "
        f"honghanh={gate.honghanh_dev_pair_ap:.4f}, delta={gate.test1_delta:+.4f}, "
        f"bar=+{TEST1_MARGIN} -> clears={gate.test1_clears}"
    )
    print(
        f"  TEST 2 (Tier-1 precision): |Tier1|={gate.tier1_size}, "
        f"tier1_prec={gate.tier1_precision:.4f} (TP={gate.tier1_true_positives}), "
        f"honghanh_topN_prec={gate.honghanh_topn_precision:.4f} "
        f"(TP={gate.honghanh_topn_true_positives}), delta={gate.test2_delta:+.4f}, "
        f"bar=+{TEST2_MARGIN} -> clears={gate.test2_clears}"
    )
    print(
        f"  VERDICT: {gate.verdict}"
        + (f" (cleared by {', '.join(gate.cleared_by)})" if gate.cleared_by else " (honest null)")
    )

    # ---- JSON summary ----
    summary: Dict[str, object] = {
        "note": (
            "BUILD + MEASURE ONLY (§80). TIERED/ORDINAL consensus of 3 INDEPENDENT "
            "collusion detectors; agreement DEPTH sets the tier (not a flat weighted "
            "mean, which was §78 and nulled). Behavior + evidence reuse the NEW-BEST "
            "candidate_honghanh_pure_ranker_evidence.csv (LB 0.65242) VERBATIM. Dev "
            "PairAP / precision are MEASURED LOCAL numbers on the common confirmed set, "
            "NOT LB claims (contract Rules 1, 9). No submit; submission.csv untouched."
        ),
        "quantiles": {"grid": list(Q_GRID), "primary": PRIMARY_Q},
        "tier_definition": {
            "tier1": "high in ALL THREE (floor & nomannic & honghanh)",
            "tier2": "high in BOTH competitors (nomannic & honghanh), not all three",
            "tier3": "high in EXACTLY ONE competitor (nomannic XOR honghanh)",
            "tier4": "everything else",
            "high": "model percentile >= Q",
            "within_tier_order": "(nomannic_pct + honghanh_pct)/2",
            "score": "min-max of (5 - tier) + 0.999*within_tier_meanpct; tiers never cross",
        },
        "part1_tiered_candidate": {
            "path": cand.path,
            "n_rows": cand.n_rows,
            "schema_ok": cand.schema_ok,
            "pair_set_matches_eval": cand.pair_set_matches_eval,
            "risk_in_unit_interval": cand.risk_in_unit_interval,
            "no_dup_evidence_within_pair": cand.no_dup_evidence_within_pair,
            "tier_counts": cand.tier_counts,
            "spearman_vs_honghanh": cand.spearman_vs_honghanh,
            "spearman_vs_new_best": cand.spearman_vs_new_best,
            "evidence_source": NEW_BEST_EVIDENCE,
        },
        "part2_evidence_gate": {
            "n_common": gate.n_common,
            "test1_dev_pair_ap": {
                "tiered_dev_pair_ap": gate.tiered_dev_pair_ap,
                "honghanh_dev_pair_ap": gate.honghanh_dev_pair_ap,
                "delta": gate.test1_delta,
                "margin": TEST1_MARGIN,
                "clears": gate.test1_clears,
            },
            "test2_tier1_precision": {
                "tier1_size": gate.tier1_size,
                "tier1_precision": gate.tier1_precision,
                "tier1_true_positives": gate.tier1_true_positives,
                "honghanh_topN_precision": gate.honghanh_topn_precision,
                "honghanh_topN_true_positives": gate.honghanh_topn_true_positives,
                "delta": gate.test2_delta,
                "margin": TEST2_MARGIN,
                "clears": gate.test2_clears,
            },
            "notes": gate.notes,
        },
        "verdict": {
            "submit_worthy": gate.submit_worthy,
            "verdict": gate.verdict,
            "cleared_by": gate.cleared_by,
            "rule": "SUBMIT-WORTHY iff (Test 1 clears) OR (Test 2 clears); else NO-SUBMIT",
        },
        "candidate_csv_path": cand.path,
    }
    summary_path = out_dir / "_tiered_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print("=" * 88)
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="§80 tiered/ordinal consensus + evidence gate (BUILD + MEASURE ONLY)."
    )
    parser.add_argument("--poker-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    run_tiered_consensus(Path(args.poker_root))


if __name__ == "__main__":
    _main()
