"""Consensus / triangulation over the three independent collusion detectors (§77).

BUILD + MEASURE ONLY. No Kaggle submit. Does NOT touch ``submission.csv`` or any
``submission_best_*.csv``; only writes under ``outputs/poker_collusion/consensus/``.

Intent (triangulation, NOT model selection)
--------------------------------------------
The three models are three INDEPENDENT detectors of the same latent truth (collusion),
built from different feature bases / recipes (pairwise Spearman only 0.50-0.71, dossier
§77):

* floor        — LB 0.44519 — ``submission_best_044519.csv``
* nomannic     — LB 0.56202 — ``repro_nomannic/submission.csv``
* honghanh     — LB 0.64262 — ``repro_064469/submission.csv``

Agreement across INDEPENDENT detectors = high-confidence detection. We rank-normalise
each model to a percentile, blend into a CONSENSUS ranking under three weight schemes,
and surface the agreement structure (the pairs all three agree are suspicious).

Contract compliance
-------------------
* Rule 6 (reuse, don't rebuild): eval rankings are the three persisted submission CSVs
  (risk_score columns); dev-holdout rankings reuse ``rerun_floor_stack`` /
  ``rerun_nomannic_recipe`` / honghanh's persisted dev OOF; dev PairAP is scored via the
  canonical :class:`anchor_repro.scorer.CanonicalScorer` (verbatim official metric).
* Rule 2 (pre-register the bar): a consensus scheme "wins" iff its dev PairAP on the
  COMMON confirmed set >= best-single-model-on-common + 0.005. Written here, judged on
  the held-out common set, no post-hoc narrative.
* Rule 3 (honest nulls): if no scheme clears +0.005 that is reported as a dev-PairAP
  NULL. The agreement structure (Part 1) is the deliverable regardless.
* Rule 9 (state uncertainty): dev PairAP is a MEASURED LOCAL number, NOT an LB claim.
  honghanh confirmed-set saturation (§71) is an EXPECTED reason a dev-PairAP null does
  not falsify the eval-agreement deliverable.

The three parts
---------------
PART 1 — EVAL consensus + agreement structure (the deliverable): merge the 3 eval CSVs
on ``pair_id`` (112,540 rows), rank-normalise each risk_score to a percentile in [0,1],
build 3 weighted-mean consensus scores + an AGREEMENT (1 - std of the 3 percentiles)
column, and report the triple-agreed / 2-of-3 set sizes + the top-20 highest-consensus
and top-20 highest-disagreement pairs. Full table saved to parquet.

PART 2 — DEV-holdout validation: get each model's dev-holdout risk on the COMMON
confirmed pairs (intersection of the 3 dev prediction sets — floor is a 760-pair seed-7
split, honghanh/nomannic are 1860; the intersection makes all 3 comparable), rank-
normalise, blend under each scheme, and score canonical dev PairAP vs each single model
on the SAME common set (the floor-guarantee baselines).

PART 3 — cache a consensus eval candidate (build only, no submit): for the best scheme
by dev PairAP (or LB_PROPORTIONAL as the sensible default if all null), write a full
112,540-row eval submission — consensus percentile as risk_score, behavior + evidence
columns REUSED verbatim from honghanh's eval submission (strongest single model, real
heads). Validated and written to ``candidate_consensus_<scheme>.csv``.

Run: ``python -m anchor_repro.consensus_triangulation`` (from the poker package root).
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
    "WEIGHT_SCHEMES",
    "BAR_MARGIN",
    "run_consensus_triangulation",
]

# --------------------------------------------------------------------------- #
# Constants (verbatim from §77 / the dossier).                                 #
# --------------------------------------------------------------------------- #

#: Model id -> (eval submission CSV relative to poker root, real LB).
MODELS: Dict[str, Tuple[str, float]] = {
    "floor": ("outputs/poker_collusion/submission_best_044519.csv", 0.44519),
    "nomannic": ("outputs/poker_collusion/repro_nomannic/submission.csv", 0.56202),
    "honghanh": ("outputs/poker_collusion/repro_064469/submission.csv", 0.64262),
}

#: The order the three percentile columns are built + reported in.
MODEL_ORDER: Tuple[str, str, str] = ("floor", "nomannic", "honghanh")

#: honghanh's persisted dev-holdout OOF (oof_risk over the confirmed pairs).
HONGHANH_DEV_OOF: str = "outputs/poker_collusion/repro_064469/dev_holdout_oof_064469.parquet"

#: The pre-registered win margin (§77 / contract Rule 2): a consensus scheme wins iff
#: its dev PairAP on the common set >= best-single-model-on-common + this margin.
BAR_MARGIN: float = 0.005

#: The eval row count the candidate submission must have.
EVAL_ROW_COUNT: int = 112_540


def _weight_schemes() -> Dict[str, Dict[str, float]]:
    """Return the three pre-registered weight schemes, normalised to sum to 1.

    (a) EQUAL             — 1/3 each.
    (b) LB_PROPORTIONAL   — weights proportional to each model's real LB, normalised.
    (c) HONGHANH_ANCHORED — honghanh 0.6, floor 0.2, nomannic 0.2.
    """
    equal = {m: 1.0 / 3.0 for m in MODEL_ORDER}

    lbs = {m: MODELS[m][1] for m in MODEL_ORDER}
    total = sum(lbs.values())
    lb_prop = {m: lbs[m] / total for m in MODEL_ORDER}

    anchored = {"floor": 0.2, "nomannic": 0.2, "honghanh": 0.6}

    return {"EQUAL": equal, "LB_PROPORTIONAL": lb_prop, "HONGHANH_ANCHORED": anchored}


#: The three weight schemes (built once).
WEIGHT_SCHEMES: Dict[str, Dict[str, float]] = _weight_schemes()


# --------------------------------------------------------------------------- #
# Helpers.                                                                      #
# --------------------------------------------------------------------------- #


def _percentile(values: Sequence[float]) -> np.ndarray:
    """Rank-normalise ``values`` to a percentile in [0,1] (average ties / N).

    Uses ``scipy.stats.rankdata`` average-tie ranks divided by N, so the result is a
    strictly monotone transform of the ranking (does not change PairAP) mapped into the
    unit interval. Highest value -> percentile ~1.0.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    return rankdata(arr, method="average") / float(n)


def _consensus(percentiles: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted mean of the per-model percentile columns."""
    total_w = sum(weights[m] for m in MODEL_ORDER)
    acc = np.zeros_like(percentiles[MODEL_ORDER[0]], dtype=float)
    for m in MODEL_ORDER:
        acc += weights[m] * percentiles[m]
    return acc / total_w


def _neutral_frame(pair_ids: Sequence[str], risk: Sequence[float]) -> pd.DataFrame:
    """A CanonicalScorer-ready frame (PairAP depends only on risk_score)."""
    frame = pd.DataFrame(
        {
            "pair_id": [str(p) for p in pair_ids],
            "risk_score": np.asarray(risk, dtype=float).clip(0.0, 1.0),
            "predicted_behavior": "none",
        }
    )
    for col in ("evidence_hand_1", "evidence_hand_2", "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"):
        frame[col] = "NO_EVIDENCE"
    return frame[list(SUBMISSION_COLUMNS)]


# --------------------------------------------------------------------------- #
# PART 1 — EVAL consensus + agreement structure.                               #
# --------------------------------------------------------------------------- #


@dataclass
class EvalConsensus:
    """The merged eval consensus table + agreement structure."""

    table: pd.DataFrame  # pair_id, <model>_pct x3, consensus_<scheme> x3, agreement, disagreement
    n_rows: int
    triple_agreed: Dict[str, int]  # "top1"/"top5"/"top10" -> count in top-k% of ALL three
    two_of_three: Dict[str, int]  # "top5" -> count in top-5% of exactly 2 of 3
    top20_consensus: List[Dict[str, object]]
    top20_disagreement: List[Dict[str, object]]


def _load_eval_percentiles(poker_root: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Merge the 3 eval CSVs on pair_id, return the merged frame + per-model percentiles."""
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


def _agreement_structure(
    merged: pd.DataFrame, percentiles: Dict[str, np.ndarray]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Triple-agreed (top-k% of ALL 3) and 2-of-3 (top-5% of exactly 2) set sizes."""
    n = len(merged)
    triple: Dict[str, int] = {}
    for label, frac in (("top1", 0.01), ("top5", 0.05), ("top10", 0.10)):
        # percentile threshold for the top frac (percentile > 1 - frac).
        thresh = 1.0 - frac
        in_top = [percentiles[m] > thresh for m in MODEL_ORDER]
        all_three = in_top[0] & in_top[1] & in_top[2]
        triple[label] = int(all_three.sum())

    # 2-of-3 at top-5%: in top-5% of exactly two models (not all three).
    thresh5 = 1.0 - 0.05
    flags = np.vstack([(percentiles[m] > thresh5).astype(int) for m in MODEL_ORDER])
    counts = flags.sum(axis=0)
    two_of_three = {"top5": int((counts == 2).sum())}
    return triple, two_of_three


def _top20(merged: pd.DataFrame, sort_col: str, ascending: bool) -> List[Dict[str, object]]:
    """Top-20 rows by ``sort_col``; report pair_id + each model's percentile."""
    sub = merged.sort_values(sort_col, ascending=ascending).head(20)
    out: List[Dict[str, object]] = []
    for _, row in sub.iterrows():
        out.append(
            {
                "pair_id": str(row["pair_id"]),
                "floor_pct": round(float(row["floor_pct"]), 4),
                "nomannic_pct": round(float(row["nomannic_pct"]), 4),
                "honghanh_pct": round(float(row["honghanh_pct"]), 4),
                sort_col: round(float(row[sort_col]), 4),
            }
        )
    return out


def build_eval_consensus(poker_root: Path) -> EvalConsensus:
    """PART 1: eval consensus scores + agreement structure over all 112,540 pairs."""
    merged, percentiles = _load_eval_percentiles(poker_root)
    n = len(merged)

    # Consensus scores under the three schemes.
    for scheme, weights in WEIGHT_SCHEMES.items():
        merged[f"consensus_{scheme}"] = _consensus(percentiles, weights)

    # Agreement = 1 - std of the 3 percentiles; disagreement = std.
    pct_stack = np.vstack([percentiles[m] for m in MODEL_ORDER]).T  # (N, 3)
    std = pct_stack.std(axis=1)
    merged["disagreement"] = std
    merged["agreement"] = 1.0 - std

    triple, two_of_three = _agreement_structure(merged, percentiles)

    # Top-20 highest consensus (use EQUAL as the neutral consensus for eyeballing) and
    # top-20 highest disagreement.
    top20_consensus = _top20(merged, "consensus_EQUAL", ascending=False)
    top20_disagreement = _top20(merged, "disagreement", ascending=False)

    return EvalConsensus(
        table=merged,
        n_rows=n,
        triple_agreed=triple,
        two_of_three=two_of_three,
        top20_consensus=top20_consensus,
        top20_disagreement=top20_disagreement,
    )


# --------------------------------------------------------------------------- #
# PART 2 — DEV-holdout validation of consensus.                                #
# --------------------------------------------------------------------------- #


@dataclass
class DevValidation:
    """Dev-holdout PairAP for each single model + each consensus scheme on the common set."""

    n_common: int
    single_pair_ap: Dict[str, float]  # model -> dev PairAP on common set
    best_single_model: str
    best_single_pair_ap: float
    scheme_pair_ap: Dict[str, float]  # scheme -> dev PairAP on common set
    winners: Dict[str, bool]  # scheme -> cleared bar (>= best_single + BAR_MARGIN)
    bar: float
    notes: List[str] = field(default_factory=list)


def _floor_dev_risk(poker_root: Path, labels: pd.DataFrame, evidence: pd.DataFrame) -> Dict[str, float]:
    """{pair_id: risk} for the floor dev-holdout confirmed pairs (reused runner)."""
    result = rerun_floor_stack(default_cache_locations(poker_root), labels, evidence=evidence)
    df = result.dev_predictions
    return dict(zip(df["pair_id"].astype(str), df["risk_score"].astype(float)))


def _nomannic_dev_risk(poker_root: Path, labels: pd.DataFrame, evidence: pd.DataFrame) -> Dict[str, float]:
    """{pair_id: risk} for the nomannic dev-holdout confirmed pairs (reused runner)."""
    result = rerun_nomannic_recipe(
        default_nomannic_cache_locations(poker_root), labels, evidence=evidence
    )
    df = result.dev_predictions
    return dict(zip(df["pair_id"].astype(str), df["risk_score"].astype(float)))


def _honghanh_dev_risk(poker_root: Path) -> Dict[str, float]:
    """{pair_id: oof_risk} for honghanh's confirmed (is_labeled) dev pairs (reused OOF)."""
    oof = pd.read_parquet(poker_root / HONGHANH_DEV_OOF)
    conf = oof[oof["is_labeled"].astype(bool)]
    return dict(zip(conf["pair_id"].astype(str), conf["oof_risk"].astype(float)))


def build_dev_validation(
    poker_root: Path, scorer: CanonicalScorer, labels: pd.DataFrame, evidence: pd.DataFrame
) -> DevValidation:
    """PART 2: score each single model + each consensus scheme on the COMMON dev set."""
    notes: List[str] = []

    floor_risk = _floor_dev_risk(poker_root, labels, evidence)
    nomannic_risk = _nomannic_dev_risk(poker_root, labels, evidence)
    honghanh_risk = _honghanh_dev_risk(poker_root)

    dev_risk = {"floor": floor_risk, "nomannic": nomannic_risk, "honghanh": honghanh_risk}
    notes.append(
        "dev prediction set sizes (confirmed): "
        + ", ".join(f"{m}={len(dev_risk[m])}" for m in MODEL_ORDER)
    )

    # Common intersection so all 3 are comparable (floor is a 760-pair seed-7 split).
    common = set(floor_risk) & set(nomannic_risk) & set(honghanh_risk)
    common_ids = sorted(common)
    n_common = len(common_ids)
    notes.append(f"n_common (intersection of all 3 dev prediction sets) = {n_common}")

    # Per-model risk aligned to the common set, then percentiles on the common set.
    aligned = {m: np.array([dev_risk[m][pid] for pid in common_ids], dtype=float) for m in MODEL_ORDER}
    dev_pct = {m: _percentile(aligned[m]) for m in MODEL_ORDER}

    # Single-model dev PairAP on the common set (the floor-guarantee baselines).
    single_pair_ap: Dict[str, float] = {}
    for m in MODEL_ORDER:
        frame = _neutral_frame(common_ids, dev_pct[m])
        single_pair_ap[m] = float(scorer.score_dev_predictions(frame).pair_ap)

    best_single_model = max(single_pair_ap, key=single_pair_ap.get)
    best_single_pair_ap = single_pair_ap[best_single_model]

    # Consensus dev PairAP on the common set, per scheme.
    scheme_pair_ap: Dict[str, float] = {}
    winners: Dict[str, bool] = {}
    bar = best_single_pair_ap + BAR_MARGIN
    for scheme, weights in WEIGHT_SCHEMES.items():
        cons = _consensus(dev_pct, weights)
        frame = _neutral_frame(common_ids, cons)
        ap = float(scorer.score_dev_predictions(frame).pair_ap)
        scheme_pair_ap[scheme] = ap
        winners[scheme] = ap >= bar

    return DevValidation(
        n_common=n_common,
        single_pair_ap=single_pair_ap,
        best_single_model=best_single_model,
        best_single_pair_ap=best_single_pair_ap,
        scheme_pair_ap=scheme_pair_ap,
        winners=winners,
        bar=bar,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# PART 3 — cache a consensus eval candidate (build only, no submit).           #
# --------------------------------------------------------------------------- #


@dataclass
class CandidateResult:
    """The built consensus eval candidate + its validation."""

    scheme: str
    path: str
    n_rows: int
    schema_ok: bool
    pair_set_matches_eval: bool
    risk_in_unit_interval: bool
    no_dup_evidence_within_pair: bool
    spearman_vs_honghanh: float
    selection_reason: str


def _choose_scheme(dev: DevValidation) -> Tuple[str, str]:
    """Pick the best scheme by dev PairAP; if none clears the bar, LB_PROPORTIONAL default."""
    any_winner = any(dev.winners.values())
    if any_winner:
        # best scheme by dev PairAP among winners.
        winners = {s: dev.scheme_pair_ap[s] for s, w in dev.winners.items() if w}
        scheme = max(winners, key=winners.get)
        reason = (
            f"scheme {scheme} cleared the pre-registered bar (dev PairAP "
            f"{dev.scheme_pair_ap[scheme]:.4f} >= {dev.bar:.4f} = best single "
            f"{dev.best_single_model} {dev.best_single_pair_ap:.4f} + {BAR_MARGIN})."
        )
    else:
        scheme = "LB_PROPORTIONAL"
        reason = (
            "no scheme cleared the pre-registered +0.005 bar on the common dev set "
            "(HONEST NULL for dev PairAP); LB_PROPORTIONAL chosen as the sensible "
            "default for the build-only eval candidate (contract Rules 2, 3)."
        )
    return scheme, reason


def build_candidate(
    poker_root: Path, eval_consensus: EvalConsensus, scheme: str, selection_reason: str, out_dir: Path
) -> CandidateResult:
    """PART 3: write a full 112,540-row consensus eval candidate (no submit)."""
    # honghanh eval submission provides behavior + evidence columns verbatim.
    honghanh_path = poker_root / MODELS["honghanh"][0]
    honghanh = pd.read_csv(honghanh_path, dtype={"pair_id": str})

    table = eval_consensus.table
    cons_col = f"consensus_{scheme}"

    # Consensus risk normalised to [0,1] (percentile mean is already in [0,1]; clip for safety).
    cand = pd.DataFrame({"pair_id": table["pair_id"].astype(str)})
    cand["risk_score"] = table[cons_col].to_numpy().clip(0.0, 1.0)

    # Reuse honghanh's behavior + evidence columns verbatim (join on pair_id).
    hh_cols = ["pair_id", "predicted_behavior", "evidence_hand_1", "evidence_hand_2",
               "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"]
    cand = cand.merge(honghanh[hh_cols], on="pair_id", how="left")
    cand = cand[list(SUBMISSION_COLUMNS)]

    # ---- Validate ----
    n_rows = len(cand)
    schema_ok = tuple(cand.columns) == SUBMISSION_COLUMNS
    eval_pairs = pd.read_csv(poker_root / "data" / "poker" / "evaluation_pairs.csv", dtype={"pair_id": str})
    pair_set_matches_eval = set(cand["pair_id"]) == set(eval_pairs["pair_id"].astype(str))
    risk = pd.to_numeric(cand["risk_score"], errors="coerce")
    risk_in_unit_interval = bool(risk.notna().all() and risk.between(0.0, 1.0).all())

    # No duplicate evidence hand within a pair (ignoring NO_EVIDENCE placeholders).
    ev_cols = ["evidence_hand_1", "evidence_hand_2", "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"]
    def _dup(row) -> bool:
        hands = [h for h in row if isinstance(h, str) and h != "NO_EVIDENCE"]
        return len(hands) != len(set(hands))
    no_dup = not cand[ev_cols].apply(_dup, axis=1).any()

    # Spearman of consensus risk vs honghanh risk (how different is it).
    hh_risk = honghanh.set_index("pair_id")["risk_score"].reindex(cand["pair_id"]).to_numpy()
    rho, _ = spearmanr(cand["risk_score"].to_numpy(), hh_risk)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"candidate_consensus_{scheme}.csv"
    cand.to_csv(out_path, index=False)

    return CandidateResult(
        scheme=scheme,
        path=str(out_path),
        n_rows=n_rows,
        schema_ok=schema_ok,
        pair_set_matches_eval=pair_set_matches_eval,
        risk_in_unit_interval=risk_in_unit_interval,
        no_dup_evidence_within_pair=bool(no_dup),
        spearman_vs_honghanh=float(rho),
        selection_reason=selection_reason,
    )


# --------------------------------------------------------------------------- #
# Orchestration.                                                               #
# --------------------------------------------------------------------------- #


def run_consensus_triangulation(poker_root: Path) -> Dict[str, object]:
    """Run all three parts, write the JSON summary + the consensus table + candidate CSV."""
    poker_root = Path(poker_root)
    config = PipelineConfig()
    data_dir = Path(config.input_dir)
    labels = pd.read_csv(data_dir / "development_labels.csv")
    evidence = pd.read_csv(data_dir / "development_evidence.csv")

    out_dir = poker_root / "outputs" / "poker_collusion" / "consensus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- PART 1 ----
    print("=" * 88)
    print("§77 CONSENSUS / TRIANGULATION — BUILD + MEASURE ONLY (no submit)")
    print("=" * 88)
    print("PART 1: EVAL consensus + agreement structure ...")
    eval_consensus = build_eval_consensus(poker_root)
    table_path = out_dir / "consensus_eval_table.parquet"
    eval_consensus.table.to_parquet(table_path, index=False)
    print(f"  merged {eval_consensus.n_rows} eval pairs; wrote {table_path}")
    print(f"  triple-agreed sets (top-k% of ALL three): {eval_consensus.triple_agreed}")
    print(f"  2-of-3 (top-5% of exactly two): {eval_consensus.two_of_three}")

    # ---- PART 2 ----
    print("PART 2: DEV-holdout validation of consensus ...")
    scorer = CanonicalScorer(
        ScoringRecipe(recipe_id="consensus_triangulation_v1"),
        config,
        labels=labels,
        evidence=evidence,
    )
    dev = build_dev_validation(poker_root, scorer, labels, evidence)
    print(f"  n_common = {dev.n_common}")
    print(f"  single-model dev PairAP: {({m: round(v, 4) for m, v in dev.single_pair_ap.items()})}")
    print(f"  best single = {dev.best_single_model} ({dev.best_single_pair_ap:.4f}); bar = {dev.bar:.4f}")
    print(f"  scheme dev PairAP: {({s: round(v, 4) for s, v in dev.scheme_pair_ap.items()})}")
    print(f"  winners (>= bar): {dev.winners}")

    # ---- PART 3 ----
    print("PART 3: cache a consensus eval candidate (build only) ...")
    scheme, selection_reason = _choose_scheme(dev)
    candidate = build_candidate(poker_root, eval_consensus, scheme, selection_reason, out_dir)
    print(f"  chose scheme {scheme}: {selection_reason}")
    print(f"  wrote {candidate.path}")
    print(
        f"  validation: rows={candidate.n_rows}, schema_ok={candidate.schema_ok}, "
        f"pair_set==eval={candidate.pair_set_matches_eval}, "
        f"risk_in_[0,1]={candidate.risk_in_unit_interval}, "
        f"no_dup_evidence={candidate.no_dup_evidence_within_pair}"
    )
    print(f"  Spearman(consensus risk, honghanh risk) = {candidate.spearman_vs_honghanh:.4f}")

    # ---- JSON summary ----
    summary: Dict[str, object] = {
        "note": (
            "BUILD + MEASURE ONLY (§77). Consensus/triangulation of 3 INDEPENDENT collusion "
            "detectors. Dev PairAP is a MEASURED LOCAL number on the common confirmed set, "
            "NOT an LB claim (contract Rules 1, 9). No submit; submission.csv untouched."
        ),
        "weight_schemes": WEIGHT_SCHEMES,
        "part1_agreement_structure": {
            "n_eval_pairs": eval_consensus.n_rows,
            "triple_agreed_top_k_pct_of_all_three": eval_consensus.triple_agreed,
            "two_of_three_top5_pct": eval_consensus.two_of_three,
            "top20_consensus": eval_consensus.top20_consensus,
            "top20_disagreement": eval_consensus.top20_disagreement,
            "consensus_table_path": str(table_path),
        },
        "part2_dev_validation": {
            "n_common": dev.n_common,
            "single_model_dev_pair_ap": dev.single_pair_ap,
            "best_single_model": dev.best_single_model,
            "best_single_pair_ap": dev.best_single_pair_ap,
            "scheme_dev_pair_ap": dev.scheme_pair_ap,
            "bar": dev.bar,
            "bar_margin": BAR_MARGIN,
            "winners": dev.winners,
            "any_winner": any(dev.winners.values()),
            "notes": dev.notes,
        },
        "part3_candidate": {
            "scheme": candidate.scheme,
            "path": candidate.path,
            "n_rows": candidate.n_rows,
            "schema_ok": candidate.schema_ok,
            "pair_set_matches_eval": candidate.pair_set_matches_eval,
            "risk_in_unit_interval": candidate.risk_in_unit_interval,
            "no_dup_evidence_within_pair": candidate.no_dup_evidence_within_pair,
            "spearman_vs_honghanh": candidate.spearman_vs_honghanh,
            "selection_reason": candidate.selection_reason,
        },
    }
    summary_path = out_dir / "_consensus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print("=" * 88)
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="§77 consensus/triangulation (BUILD + MEASURE ONLY).")
    parser.add_argument("--poker-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    run_consensus_triangulation(Path(args.poker_root))


if __name__ == "__main__":
    _main()
