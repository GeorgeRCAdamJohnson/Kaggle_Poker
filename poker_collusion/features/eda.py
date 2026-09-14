"""EDA & Label Audit + versioned threshold artifact (task 8.1, Requirement 2).

This module is the concrete, runnable realisation of Requirement 2 (Exploratory Data
Analysis and Label Audit). It computes reproducible summary statistics over the REAL
competition data via :class:`poker_collusion.io.DataLoader` (column-projected reads;
``actions.parquet`` is NEVER loaded whole), audits the Positive-Unlabelled label
structure and the disclosed-family distribution, confirms the checkable part of
evidence validity, re-confirms the ``evaluation_pairs.csv`` leakage exclusions, and
persists a **versioned threshold artifact** (``eda_artifact.json``).

Grounding in confirmed Discovery facts (``RESEARCH_DOSSIER.md`` §2/§3)
----------------------------------------------------------------------
* **Real schema, hex string IDs.** ``player_id``/``hand_id``/``table_id``/``pair_id`` are
  opaque hex strings. ``development_labels.csv`` carries ``player_1``/``player_2`` columns
  (the pair is NOT an integer-parseable ``"<a>_<b>"`` id), so leakage checks read the
  player columns directly rather than parsing the pair id.
* **PU structure.** ``development_labels.csv`` lists ``confirmed_target`` (trusted
  positive) and ``confirmed_non_target`` (confirmed negative) pairs; every other pair —
  and every evaluation pair — is UNKNOWN (unlabelled), never negative (Req 2.2, 6.1).
* **No ``other_coordination`` in public labels.** All public positives are one of the
  three disclosed families (confirmed in dossier §2.3/§3.1); the family distribution is
  reported over exactly those three, and the artifact records ``other_coordination``
  prior as absent-in-public.
* **``development_evidence.csv`` has no public behavior-action flag.** Its columns are
  ``pair_id, evidence_rank, hand_id, behavior_family`` (dossier §2.0/§3.2). The
  "behavior-specific action present" clause of evidence validity is therefore NOT
  publicly checkable; this module documents that and checks only the checkable part —
  each evidence hand exists and seats BOTH players of the pair (a Shared_Hand).
* **Shared hands from seats, phase from ``hands.phase``.** Co-seated hands are derived
  from ``seats`` and split dev/eval via ``hands.phase`` (Req 1.3/1.4), exactly as
  :meth:`DataLoader.shared_hands` does.

The threshold-artifact contract (Requirement 2.6 / 5.4)
-------------------------------------------------------
Feature engineering (Phase 1+) MUST read its calibrated constants ONLY from the
persisted ``eda_artifact.json`` — never recompute them ad hoc. The artifact carries a
``threshold_version`` (from :class:`~poker_collusion.config.PipelineConfig`) and is
deterministic given the seeds and the input data. :func:`load_eda_artifact` round-trips
it back into an :class:`EdaArtifact`.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6. Design: Phase 1 EDA & Label Audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from poker_collusion.config import (
    DISCLOSED_FAMILY_NAMES,
    PipelineConfig,
    get_config,
)
from poker_collusion.discovery.label_evidence_audit import (
    AuditResult,
    PENDING,
    evaluation_pairs_exclusions,
)
from poker_collusion.io import DataLoader

__all__ = [
    "EDA_ARTIFACT_SCHEMA",
    "EdaArtifact",
    "SharedHandCountStats",
    "compute_shared_hand_count_stats",
    "compute_pu_counts",
    "compute_family_distribution",
    "audit_evidence_validity",
    "audit_evaluation_leakage",
    "run_eda",
    "write_eda_artifact",
    "load_eda_artifact",
]

#: Artifact schema tag; bump when the artifact structure changes.
EDA_ARTIFACT_SCHEMA: str = "eda_artifact_v1"

#: Total number of within-pool candidate pairs (400 pools x C(30,2)); dossier §2.3.
#: Used only as the denominator for the reported pool-wide positive prevalence.
POOL_CANDIDATE_PAIR_COUNT: int = 174_000

# Label-status tokens (mirrors DataLoader; kept local so this module is self-describing).
_LABEL_STATUS_POSITIVE = "confirmed_target"
_LABEL_STATUS_NEGATIVE = "confirmed_non_target"

# Percentiles reported for the shared-hand-count distribution and used as
# feature-normalization anchors in the persisted artifact.
_PERCENTILES: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95, 99)


# --------------------------------------------------------------------------- #
# Result carriers
# --------------------------------------------------------------------------- #
@dataclass
class SharedHandCountStats:
    """Summary statistics of shared-hand counts across a set of pairs (Req 2.1).

    ``percentiles`` maps an integer percentile (e.g. ``50``) to its value. All
    values are plain Python floats/ints so the struct is JSON-serialisable.
    """

    n_pairs: int
    min: float
    max: float
    mean: float
    median: float
    percentiles: Dict[int, float] = field(default_factory=dict)


@dataclass
class EdaArtifact:
    """The versioned EDA / threshold artifact persisted to ``eda_artifact.json``.

    Feature engineering consumes ONLY the calibrated constants in ``thresholds``
    (Req 2.6). ``report`` carries the human-readable EDA numbers for the dossier;
    it does not feed feature generation.

    Attributes:
        schema: Artifact schema tag (:data:`EDA_ARTIFACT_SCHEMA`).
        threshold_version: The ``config.threshold_version`` these constants belong to.
        master_seed: The master seed the run was made under (determinism marker).
        thresholds: The calibrated constants feature engineering may read (min shared
            hands, shared-hand-count normalization anchors, pool-prior positive
            prevalence, per-family priors, ...).
        report: The full EDA report (PU counts, family distribution, evidence-validity
            audit, leakage checks, shared-hand-count stats).
    """

    schema: str
    threshold_version: str
    master_seed: int
    thresholds: Dict[str, object] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _percentile_map(series: pd.Series, percentiles: Sequence[int]) -> Dict[int, float]:
    """Return ``{p: value}`` for each requested percentile of a numeric series."""
    if len(series) == 0:
        return {int(p): 0.0 for p in percentiles}
    qs = [p / 100.0 for p in percentiles]
    vals = series.quantile(qs).tolist()
    return {int(p): float(v) for p, v in zip(percentiles, vals)}


def _canonical_pair_key(player_1: str, player_2: str) -> tuple[str, str]:
    """Return the two player ids in canonical (sorted) order for an unordered pair."""
    a, b = str(player_1), str(player_2)
    return (a, b) if a <= b else (b, a)


# --------------------------------------------------------------------------- #
# 2.1 : shared-hand-count distribution
# --------------------------------------------------------------------------- #
def compute_shared_hand_count_stats(
    counts: Sequence[int],
    percentiles: Sequence[int] = _PERCENTILES,
) -> SharedHandCountStats:
    """Summarise a collection of per-pair shared-hand counts (Req 2.1).

    Args:
        counts: One shared-hand count per pair.
        percentiles: Integer percentiles to report.

    Returns:
        A :class:`SharedHandCountStats` with min/median/percentiles/max + mean.
    """
    series = pd.Series(list(counts), dtype="float64")
    if len(series) == 0:
        return SharedHandCountStats(
            n_pairs=0, min=0.0, max=0.0, mean=0.0, median=0.0,
            percentiles={int(p): 0.0 for p in percentiles},
        )
    return SharedHandCountStats(
        n_pairs=int(len(series)),
        min=float(series.min()),
        max=float(series.max()),
        mean=float(series.mean()),
        median=float(series.median()),
        percentiles=_percentile_map(series, percentiles),
    )


def _shared_counts_from_seats(
    seats: pd.DataFrame,
    hands: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    player_1_col: str = "player_1",
    player_2_col: str = "player_2",
) -> Dict[str, Dict[str, int]]:
    """Compute per-pair shared-hand counts partitioned dev/eval, from seats+hands.

    Uses the same seats-derived, phase-split definition as
    :meth:`DataLoader.shared_hands` but vectorised over many pairs so a large
    ``evaluation_pairs.csv`` is affordable. Returns a mapping
    ``pair_id -> {"development": n, "evaluation": m, "total": n+m}``.
    """
    # Map each player to the set of (hand_id) they are seated in, split by phase.
    phase_by_hand = dict(zip(hands["hand_id"].tolist(), hands["phase"].tolist()))
    seats = seats[["hand_id", "player_id"]].copy()
    seats["phase"] = seats["hand_id"].map(phase_by_hand)

    # player -> {phase -> set(hand_id)}
    dev_hands: Dict[str, set] = {}
    eval_hands: Dict[str, set] = {}
    for hand_id, player_id, phase in zip(
        seats["hand_id"].tolist(), seats["player_id"].tolist(), seats["phase"].tolist()
    ):
        target = dev_hands if phase == "development" else eval_hands if phase == "evaluation" else None
        if target is None:
            continue
        target.setdefault(str(player_id), set()).add(hand_id)

    out: Dict[str, Dict[str, int]] = {}
    for row in pairs.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        p1 = str(getattr(row, player_1_col))
        p2 = str(getattr(row, player_2_col))
        dev = len(dev_hands.get(p1, set()) & dev_hands.get(p2, set()))
        ev = len(eval_hands.get(p1, set()) & eval_hands.get(p2, set()))
        out[pair_id] = {"development": dev, "evaluation": ev, "total": dev + ev}
    return out


# --------------------------------------------------------------------------- #
# 2.2 : PU counts
# --------------------------------------------------------------------------- #
def compute_pu_counts(
    labels: pd.DataFrame,
    evaluation_pairs: Optional[pd.DataFrame] = None,
    *,
    pool_candidate_pairs: int = POOL_CANDIDATE_PAIR_COUNT,
) -> Dict[str, object]:
    """Report Trusted_Positive / Confirmed_Negative / Unknown counts (Req 2.2).

    Trusted positives are ``label_status == 'confirmed_target'``; confirmed negatives
    are ``label_status == 'confirmed_non_target'``. Everything else is UNKNOWN
    (unlabelled) and is never treated as negative. Evaluation pairs are all unknown by
    construction (they carry no public label), so their count is added to the unknown
    tally and reported separately.

    Returns a JSON-serialisable dict of counts and prevalences.
    """
    status = labels["label_status"].astype(str) if "label_status" in labels.columns else pd.Series([], dtype=str)
    n_positive = int((status == _LABEL_STATUS_POSITIVE).sum())
    n_negative = int((status == _LABEL_STATUS_NEGATIVE).sum())
    n_labelled = int(len(labels))
    n_labelled_unknown = n_labelled - n_positive - n_negative  # e.g. any other status token
    n_eval = int(len(evaluation_pairs)) if evaluation_pairs is not None else 0
    # Unknown = labelled rows with neither status + all evaluation pairs (never negative).
    n_unknown = n_labelled_unknown + n_eval

    labelled_positive_prevalence = (n_positive / n_labelled) if n_labelled else 0.0
    pool_wide_positive_prevalence = (
        (n_positive / pool_candidate_pairs) if pool_candidate_pairs else 0.0
    )
    return {
        "trusted_positive": n_positive,
        "confirmed_negative": n_negative,
        "unknown": n_unknown,
        "unknown_from_evaluation_pairs": n_eval,
        "unknown_from_unlabelled_status": n_labelled_unknown,
        "labelled_rows": n_labelled,
        "labelled_positive_prevalence": labelled_positive_prevalence,
        "pool_candidate_pairs": pool_candidate_pairs,
        "pool_wide_positive_prevalence": pool_wide_positive_prevalence,
    }


# --------------------------------------------------------------------------- #
# 2.3 : disclosed-family label distribution among positives
# --------------------------------------------------------------------------- #
def compute_family_distribution(
    labels: pd.DataFrame,
    *,
    families: Sequence[str] = DISCLOSED_FAMILY_NAMES,
) -> Dict[str, object]:
    """Distribution of ``behavior_family`` among Trusted_Positive pairs (Req 2.3).

    Reports a count per disclosed family plus a residual ``other`` bucket for any
    positive whose family is outside the three disclosed families. Public labels
    contain NO ``other_coordination`` (dossier §2.3/§3.1); the audit records whether
    that holds on the data it is given.
    """
    if "label_status" in labels.columns:
        positives = labels[labels["label_status"].astype(str) == _LABEL_STATUS_POSITIVE]
    else:
        positives = labels
    fam = positives["behavior_family"].astype(str) if "behavior_family" in positives.columns else pd.Series([], dtype=str)
    counts = {f: int((fam == f).sum()) for f in families}
    n_positive = int(len(positives))
    n_disclosed = sum(counts.values())
    n_other = n_positive - n_disclosed
    return {
        "n_positive": n_positive,
        "by_family": counts,
        "other_or_undisclosed": int(n_other),
        "other_coordination_absent_in_public": bool(n_other == 0),
    }


# --------------------------------------------------------------------------- #
# 2.4 : evidence-validity confirmation (checkable part only)
# --------------------------------------------------------------------------- #
def audit_evidence_validity(
    evidence: pd.DataFrame,
    labels: pd.DataFrame,
    seats: pd.DataFrame,
    *,
    pair_col: str = "pair_id",
    hand_col: str = "hand_id",
) -> Dict[str, object]:
    """Confirm each dev-evidence hand is a Shared_Hand of its pair (Req 2.4).

    For every ``development_evidence.csv`` row, resolve the pair's two players from
    ``development_labels.csv`` (joined on ``pair_id``) and confirm the evidence
    ``hand_id`` exists and seats BOTH players. Reports valid vs invalid counts and the
    first few offending rows.

    The "hand contains a Behavior_Specific_Action in the public action log" clause of
    Req 2.4 is NOT publicly checkable — ``development_evidence.csv`` has no public
    behavior-action flag column (dossier §3.2), so it is documented as un-computable
    here and deferred to the Phase-0/1 action-log characterisation (the validity gate,
    task 12.1). Only the both-players-seated + hand-exists clause is evaluated.
    """
    result: Dict[str, object] = {
        "behavior_action_clause": "un-computable from public data (no flag column); "
        "deferred to the evidence validity gate (task 12.1)",
    }
    if evidence is None or len(evidence) == 0:
        result.update({"status": PENDING, "data_present": False,
                       "note": "development_evidence.csv absent/empty"})
        return result

    # pair_id -> (player_1, player_2) from labels.
    pair_players: Dict[str, tuple] = {}
    if {pair_col, "player_1", "player_2"} <= set(labels.columns):
        for row in labels.itertuples(index=False):
            pair_players[str(getattr(row, pair_col))] = (
                str(getattr(row, "player_1")), str(getattr(row, "player_2"))
            )

    # hand_id -> set(player_id) seated.
    seated: Dict[str, set] = {}
    if {"hand_id", "player_id"} <= set(seats.columns):
        for hand_id, player_id in zip(seats["hand_id"].tolist(), seats["player_id"].tolist()):
            seated.setdefault(str(hand_id), set()).add(str(player_id))

    can_check = bool(pair_players) and bool(seated)
    n_rows = int(len(evidence))
    valid = 0
    violations: List[Dict[str, object]] = []
    for row in evidence.itertuples(index=False):
        pid = str(getattr(row, pair_col))
        hid = str(getattr(row, hand_col))
        members = pair_players.get(pid)
        seated_here = seated.get(hid, set())
        both_present = members is not None and set(members) <= seated_here
        if both_present:
            valid += 1
        else:
            if len(violations) < 20:
                violations.append({
                    "pair_id": pid, "hand_id": hid,
                    "hand_exists": hid in seated,
                    "both_players_seated": bool(both_present),
                })

    result.update({
        "data_present": True,
        "checkable": can_check,
        "rows_checked": n_rows if can_check else 0,
        "valid_shared_hand": valid if can_check else 0,
        "invalid": (n_rows - valid) if can_check else 0,
        "violations": violations,
        "n_violations": len(violations),
        "status": (PENDING if not can_check else ("CONFIRMED" if not violations else "REFUTED")),
    })
    if not can_check:
        result["note"] = "labels player columns and/or seats absent; cannot resolve pair members"
    return result


# --------------------------------------------------------------------------- #
# 2.5 : evaluation-pair leakage checks (EXC-1 pair id; EXC-2 positive player)
# --------------------------------------------------------------------------- #
def audit_evaluation_leakage(
    evaluation_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    families: Sequence[str] = DISCLOSED_FAMILY_NAMES,
) -> Dict[str, object]:
    """Re-confirm EXC-1 (no eval pair id in labels) and EXC-2 (no eval pair reuses a
    labelled-positive player) and report any violations (Req 2.5).

    EXC-1 is delegated to :func:`label_evidence_audit.evaluation_pairs_exclusions`
    (its pair-id set-overlap is schema-agnostic). EXC-2 is computed here directly from
    the real ``player_1``/``player_2`` columns (the real labels carry no
    integer-parseable pair id), matching the dossier §3.3 method.
    """
    out: Dict[str, object] = {}

    # EXC-1: reuse the audit module's schema-agnostic pair-id overlap check.
    exc: AuditResult = evaluation_pairs_exclusions(evaluation_pairs, labels)
    out["EXC-1"] = exc.checks.get("EXC-1", {"status": PENDING})

    # EXC-2 (real schema): positive players = union of members of labelled-positive pairs.
    exc2: Dict[str, object] = {
        "check": "evaluation_pairs excludes pairs containing a publicly labelled positive player",
        "status": PENDING,
    }
    have_players = (
        evaluation_pairs is not None
        and labels is not None
        and {"player_1", "player_2"} <= set(getattr(evaluation_pairs, "columns", []))
        and {"player_1", "player_2"} <= set(getattr(labels, "columns", []))
    )
    if have_players and len(evaluation_pairs) and len(labels):
        if "label_status" in labels.columns:
            pos = labels[labels["label_status"].astype(str) == _LABEL_STATUS_POSITIVE]
        elif "behavior_family" in labels.columns:
            pos = labels[labels["behavior_family"].astype(str).isin(list(families))]
        else:
            pos = labels
        positive_players = set(pos["player_1"].astype(str)) | set(pos["player_2"].astype(str))
        violations: List[Dict[str, object]] = []
        for row in evaluation_pairs.itertuples(index=False):
            members = {str(getattr(row, "player_1")), str(getattr(row, "player_2"))}
            offending = sorted(members & positive_players)
            if offending:
                if len(violations) < 50:
                    violations.append({
                        "pair_id": str(getattr(row, "pair_id")),
                        "offending_players": offending,
                    })
        exc2.update({
            "positive_players": len(positive_players),
            "eval_pairs": int(len(evaluation_pairs)),
            "violations": violations,
            "n_violations": len(violations),
            "status": "CONFIRMED" if not violations else "REFUTED",
        })
    else:
        exc2["note"] = "player columns absent on evaluation_pairs/labels; EXC-2 left PENDING"
    out["EXC-2"] = exc2
    return out


# --------------------------------------------------------------------------- #
# Threshold derivation
# --------------------------------------------------------------------------- #
def _derive_thresholds(
    shared_stats_labelled: SharedHandCountStats,
    shared_stats_eval: SharedHandCountStats,
    pu_counts: Dict[str, object],
    family_distribution: Dict[str, object],
) -> Dict[str, object]:
    """Derive the calibrated constants feature engineering may read (Req 2.6).

    All constants are functions of the reported EDA statistics, so the artifact is
    deterministic given seeds + data. Constants:

    * ``min_shared_hands`` — a floor below which a pair shares too few hands for its
      episodic signal to be trustworthy; anchored to the 5th percentile of the
      evaluation pairs' shared-hand counts (>= 1).
    * ``shared_hand_count_anchors`` — the evaluation-pool shared-hand-count percentiles
      used as feature-normalization anchors (so dev/eval features share one scale).
    * ``pool_prior_positive_prevalence`` — the pool-wide positive prevalence used as the
      empirical-Bayes shrinkage prior for count-robust normalization (Req 4.4).
    * ``family_priors`` — per-disclosed-family prevalence among positives, plus an
      explicit ``other_coordination`` prior (0.0 in public data).
    """
    anchors = shared_stats_eval.percentiles or shared_stats_labelled.percentiles
    p5 = float(anchors.get(5, 1.0)) if anchors else 1.0
    min_shared_hands = max(1, int(round(p5)))

    by_family = dict(family_distribution.get("by_family", {}))
    n_positive = int(family_distribution.get("n_positive", 0)) or 0
    family_priors = {
        f: (by_family.get(f, 0) / n_positive if n_positive else 0.0) for f in by_family
    }
    family_priors["other_coordination"] = 0.0  # confirmed absent in public labels

    return {
        "min_shared_hands": min_shared_hands,
        "shared_hand_count_anchors": {str(k): v for k, v in anchors.items()},
        "pool_prior_positive_prevalence": float(pu_counts.get("pool_wide_positive_prevalence", 0.0)),
        "labelled_positive_prevalence": float(pu_counts.get("labelled_positive_prevalence", 0.0)),
        "family_priors": family_priors,
        "feature_normalization_contract": (
            "Feature engineering reads ONLY these persisted constants; it must not "
            "recompute thresholds ad hoc (Req 2.6). Anchors are the evaluation-pool "
            "shared-hand-count percentiles; the prior is the empirical-Bayes shrinkage "
            "target for count-robust normalization (Req 4.4)."
        ),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(
    loader: Optional[DataLoader] = None,
    config: Optional[PipelineConfig] = None,
) -> EdaArtifact:
    """Run the full EDA / label audit and assemble the versioned threshold artifact.

    Reads the small metadata + CSV tables via :class:`DataLoader` (column-projected;
    ``actions.parquet`` is never touched here), computes every Req-2 report, derives
    the calibrated thresholds, and returns an :class:`EdaArtifact`. Deterministic given
    the config seeds and the input data.
    """
    cfg = config or get_config()
    dl = loader or DataLoader(config=cfg)

    labels = dl.load_labels_frame()
    evaluation_pairs = dl.load_evaluation_pairs()
    seats = dl.load_seats()
    hands = dl.load_hands()
    try:
        evidence = dl.load_development_evidence()  # type: ignore[attr-defined]
    except AttributeError:
        evidence = _load_evidence_fallback(dl)

    # 2.1 shared-hand-count distributions (labelled pairs + evaluation pairs).
    labelled_counts_map = _shared_counts_from_seats(seats, hands, labels)
    eval_counts_map = _shared_counts_from_seats(seats, hands, evaluation_pairs)
    labelled_totals = [v["total"] for v in labelled_counts_map.values()]
    eval_totals = [v["total"] for v in eval_counts_map.values()]
    stats_labelled = compute_shared_hand_count_stats(labelled_totals)
    stats_eval = compute_shared_hand_count_stats(eval_totals)

    # 2.2 / 2.3 / 2.4 / 2.5
    pu = compute_pu_counts(labels, evaluation_pairs)
    fam = compute_family_distribution(labels)
    evidence_validity = audit_evidence_validity(evidence, labels, seats)
    leakage = audit_evaluation_leakage(evaluation_pairs, labels)

    thresholds = _derive_thresholds(stats_labelled, stats_eval, pu, fam)

    report: Dict[str, object] = {
        "shared_hand_counts": {
            "labelled_pairs": asdict(stats_labelled),
            "evaluation_pairs": asdict(stats_eval),
        },
        "pu_counts": pu,
        "family_distribution": fam,
        "evidence_validity": evidence_validity,
        "evaluation_leakage": leakage,
    }
    return EdaArtifact(
        schema=EDA_ARTIFACT_SCHEMA,
        threshold_version=cfg.threshold_version,
        master_seed=int(cfg.seeds.master),
        thresholds=thresholds,
        report=report,
    )


def _load_evidence_fallback(dl: DataLoader) -> pd.DataFrame:
    """Load ``development_evidence.csv`` directly if DataLoader has no dedicated method.

    Uses the loader's path resolution so both the local and Kaggle layouts work; falls
    back to an empty frame if the file is absent (so EDA degrades gracefully).
    """
    try:
        path = dl._resolve_path("development_evidence", "development_evidence.csv")  # type: ignore[attr-defined]
    except Exception:
        return pd.DataFrame(columns=["pair_id", "evidence_rank", "hand_id", "behavior_family"])
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Persistence (Requirement 2.6)
# --------------------------------------------------------------------------- #
def write_eda_artifact(
    artifact: EdaArtifact,
    path: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
) -> Path:
    """Persist the artifact to ``eda_artifact.json`` (Req 2.6).

    Writes deterministically (sorted keys) to ``config.eda_artifact_path`` unless an
    explicit ``path`` is given. Creates the parent directory if needed. Uses an atomic
    temp-then-rename so a partially written file is never observed.
    """
    cfg = config or get_config()
    target = Path(path) if path is not None else Path(cfg.eda_artifact_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(artifact)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(target)
    return target


def load_eda_artifact(
    path: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
) -> EdaArtifact:
    """Read back a persisted ``eda_artifact.json`` into an :class:`EdaArtifact`.

    Round-trips :func:`write_eda_artifact`. Raises ``FileNotFoundError`` if absent.
    """
    cfg = config or get_config()
    source = Path(path) if path is not None else Path(cfg.eda_artifact_path)
    with open(source, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return EdaArtifact(
        schema=payload.get("schema", EDA_ARTIFACT_SCHEMA),
        threshold_version=payload.get("threshold_version", ""),
        master_seed=int(payload.get("master_seed", 0)),
        thresholds=payload.get("thresholds", {}),
        report=payload.get("report", {}),
    )
