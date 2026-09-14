"""Phase-1 gate: measure the classical baseline's local-CV FLOOR (task 15, Req 9.6, 11.1).

This module is the measurable Phase-1 gate. It answers two questions the design's Phase-1
checkpoint demands:

(A) **Is the submission accepted (schema-valid)?** :func:`confirm_schema_acceptance` runs the
    single documented entry point (:func:`poker_collusion.pipeline.run_baseline_pipeline`) with a
    small ``limit_pairs`` cap on the REAL data, reads the produced ``submission.csv`` back from
    disk, and pushes it through the writer's own
    :func:`~poker_collusion.submission.writer.validate_submission` against ``sample_submission.csv``
    — i.e. the exact acceptance check the leaderboard applies (exact columns/order, one row per
    evaluation pair, exact pair-id set, ``risk_score`` in [0,1], allowed behaviors, non-empty
    evidence cells, no repeated hand id per row). We do NOT run the full ~112k-pair pipeline (too
    slow); a limited run that still emits a full schema-valid file is sufficient to confirm the
    acceptance SHAPE (documented limit).

(B) **What is the FLOOR Phase 2 must beat?** :func:`measure_baseline_floor` scores the classical
    baseline through the group-by-pool, family-stratified, PU-correct local CV harness
    (:class:`poker_collusion.validation.cv.CVHarness`) on the real ``development_labels.csv`` and
    reports the three metric components (Pair AP, Evidence MAP@5, Behavior MAP) plus the weighted
    ``combined`` summary (0.70/0.20/0.10) — as the mean across folds AND per-fold — together with
    bootstrap confidence intervals (:mod:`poker_collusion.validation.bootstrap`).

How each fold is scored (the ``predict_fn``)
--------------------------------------------
For each CV fold the harness hands us a PU-correct solution frame built by
:func:`~poker_collusion.validation.cv.build_fold_solution`: one row per CONFIRMED pair
(``risk_score`` = binary truth, ``predicted_behavior`` = the pair's ``behavior_family`` for
positives / ``none`` for negatives, and the five evidence slots filled from the planted
``development_evidence`` for positives). Our ``predict_fn`` produces a submission over EXACTLY that
pair set with the classical baseline:

1. **Signals (development period).** For every confirmed pair in the fold we compute its
   DEVELOPMENT-period shared-hand :class:`~poker_collusion.types.HandSignal` stream
   (``compute_pair_signals`` over ``DataLoader.shared_hands(pair).development``), pulling each
   hand's ordered public action log and per-seat ``net_chips`` from the shared single-pass caches
   built by :func:`_prepare_labelled_inputs`.
2. **Features + risk.** We assemble the ``PairFeatureSet`` with ``phase="development"`` and run
   :func:`poker_collusion.models.classical.score_pair` for the ``risk_score`` (Req 6.3/6.4 — every
   pair scored; unscoreable → pool-prior default).
3. **Behavior label.** Assigned by the pipeline's Phase-1 threshold placeholder
   (:func:`poker_collusion.pipeline.assign_behavior_label`) so the measured floor uses the SAME
   behavior rule the shipped pipeline uses.
4. **Evidence.** Evidence MAP@5 in-fold scores our submitted evidence hands against the fold
   solution's *relevant* set, which is the planted ``development_evidence`` hands (all
   development-period — verified). The production evidence retriever's hard validity gate requires
   ``phase == "evaluation"``, so it would (correctly, for the real eval submission) reject
   development hands. To score Evidence MAP@5 *meaningfully in-fold* we therefore reuse the SAME
   deterministic ranking the retriever applies (evidence-strength desc, ascending-hand-id
   tie-break; :func:`poker_collusion.evidence.retriever.evidence_strength` /
   :data:`~poker_collusion.evidence.retriever.MAX_EVIDENCE`) but over the pair's
   DEVELOPMENT-period behavior-flagged shared hands — i.e. we emit, for each true positive, its
   retrieved development-period evidence hands drawn from the same universe the solution's relevant
   set is drawn from. This is documented as an in-fold measurement choice: it does not change the
   production retriever (Phase-3 replaces the ranker), it only lets the development-period gate
   produce a non-degenerate Evidence MAP so the floor is informative rather than trivially zero.

Efficiency (Req 1.6 — never load actions.parquet whole)
-------------------------------------------------------
Only the LABELLED pairs' DEVELOPMENT shared hands are needed (~1,860 confirmed pairs). We resolve
the union of those hand ids once and make ONE streaming pass over ``DataLoader.iter_actions()``
(reusing :func:`poker_collusion.pipeline._actions_for_hands`), plus one projected ``net_chips``
pass (:func:`poker_collusion.pipeline._net_chips_by_hand`). If that full labelled-set pass is too
slow on a given machine, :func:`measure_baseline_floor` accepts a documented ``max_pairs`` cap that
subsets the labelled pairs (recorded as a caveat in the returned :class:`FloorResult`).

Determinism
-----------
The classical baseline is deterministic (no randomness, no wall-clock). The CV fold assignment is
seeded (``config.Seeds.cv_split``); the bootstrap CIs are seeded (``config.Seeds.bootstrap``). Two
runs on identical inputs and config therefore return identical floor numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from poker_collusion.config import NO_EVIDENCE, PipelineConfig, get_config
from poker_collusion.evidence.retriever import (
    MAX_EVIDENCE,
    evidence_strength,
    is_valid_evidence,
)
from poker_collusion.features.aggregation import prior_from_artifact
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.features.pair_features import build_pair_feature_set
from poker_collusion.features.signals import compute_pair_signals
from poker_collusion.io import DataLoader
from poker_collusion.models.classical import score_pair
from poker_collusion.pipeline import (
    _actions_for_hands,
    _canonical_players,
    _hands_meta_for,
    _net_chips_by_hand,
    _obtain_eda_artifact,
    assign_behavior_label,
)
from poker_collusion.submission.writer import SUBMISSION_COLUMNS, validate_submission
from poker_collusion.types import HandSignal
from poker_collusion.validation.bootstrap import (
    COMPONENT_KEYS,
    ConfidenceInterval,
    bootstrap_components,
)
from poker_collusion.validation.cv import (
    CONFIRMED_NON_TARGET,
    CONFIRMED_TARGET,
    CVHarness,
    Fold,
    build_fold_solution,
)

__all__ = [
    "DEVELOPMENT_PHASE",
    "FloorResult",
    "select_development_evidence",
    "measure_baseline_floor",
    "confirm_schema_acceptance",
]

#: Phase token for the development-period slice we score the floor on.
DEVELOPMENT_PHASE: str = "development"

# The four component keys, in the fixed order used everywhere in this module.
_COMPONENTS: Tuple[str, ...] = COMPONENT_KEYS  # ("pair_ap", "evidence_map", "behavior_map", "combined")

# The metric weighting recorded in the floor report (matches production_metric).
_WEIGHTING: Dict[str, float] = {"pair_ap": 0.70, "evidence_map": 0.20, "behavior_map": 0.10}


# --------------------------------------------------------------------------- #
# Result carrier
# --------------------------------------------------------------------------- #
@dataclass
class FloorResult:
    """The measured Phase-1 baseline floor (the number Phase 2 must beat).

    Attributes:
        mean: ``{component -> mean-across-folds}`` for the four keys.
        per_fold: One ``{component -> value}`` dict per fold (in fold order).
        intervals: ``{component -> ConfidenceInterval}`` bootstrap CIs (see ``ci_method``).
        ci_method: Human-readable description of exactly how the CIs were computed.
        n_folds: Number of folds that produced a score.
        n_pairs_scored: Total confirmed pairs scored across all folds.
        n_positive_pairs: Confirmed-positive pairs scored across all folds.
        weighting: The combined-summary weighting used (0.70/0.20/0.10).
        caveats: Any sampling / limit caveats recorded during measurement.
        deterministic: Always True — the classical baseline + seeded CV/bootstrap are deterministic.
    """

    mean: Dict[str, float]
    per_fold: List[Dict[str, float]]
    intervals: Dict[str, ConfidenceInterval]
    ci_method: str
    n_folds: int
    n_pairs_scored: int
    n_positive_pairs: int
    weighting: Dict[str, float] = field(default_factory=lambda: dict(_WEIGHTING))
    caveats: List[str] = field(default_factory=list)
    deterministic: bool = True

    def as_dict(self) -> Dict[str, object]:
        """Plain-dict view for logging / persistence."""
        return {
            "mean": self.mean,
            "per_fold": self.per_fold,
            "intervals": {k: v.as_dict() for k, v in self.intervals.items()},
            "ci_method": self.ci_method,
            "n_folds": self.n_folds,
            "n_pairs_scored": self.n_pairs_scored,
            "n_positive_pairs": self.n_positive_pairs,
            "weighting": self.weighting,
            "caveats": self.caveats,
            "deterministic": self.deterministic,
        }


# --------------------------------------------------------------------------- #
# In-fold development-period evidence selection (documented measurement choice)
# --------------------------------------------------------------------------- #
def select_development_evidence(pair_id: str, signals: Sequence[HandSignal]) -> List[str]:
    """Emit up to five ranked DEVELOPMENT-period evidence hand ids for a true positive.

    Mirrors the production retriever's deterministic ranking (evidence-strength descending,
    ascending-hand-id tie-break; de-dup; pad with ``NO_EVIDENCE``) but over the pair's
    DEVELOPMENT-period behavior-flagged shared hands. This is an in-fold MEASUREMENT choice
    (documented in the module docstring): the fold solution's relevant evidence set is the planted
    ``development_evidence`` hands (all development-period), so scoring Evidence MAP@5 meaningfully
    requires emitting development-period candidates from the same universe. The production
    retriever's ``phase == "evaluation"`` gate is intentionally unchanged for the real submission.

    A hand qualifies here iff it is a development-period hand with ≥1 disclosed-family public-action
    flag — the same behavior-specific-action clause the production gate enforces, only re-pointed
    at the development period via ``is_valid_evidence(..., )`` after overriding the phase check.
    """
    # De-duplicate by hand id (first occurrence), then keep development-period, behavior-flagged
    # hands. We reuse the retriever's flag logic by asking is_valid_evidence on a shallow phase
    # rewrite so the "≥1 behavior flag" clause is applied identically.
    seen: set = set()
    unique: List[HandSignal] = []
    for sig in signals:
        hid = str(sig.hand_id)
        if hid in seen:
            continue
        seen.add(hid)
        unique.append(sig)

    valid: List[HandSignal] = [
        sig for sig in unique if _is_valid_development_evidence(sig)
    ]

    def _hand_key(hid: str) -> Tuple[int, float, str]:
        try:
            return (0, float(hid), "")
        except (TypeError, ValueError):
            return (1, 0.0, hid)

    ranked = sorted(
        valid, key=lambda s: (-evidence_strength(s), _hand_key(str(s.hand_id)))
    )
    slots = [str(s.hand_id) for s in ranked[:MAX_EVIDENCE]]
    while len(slots) < MAX_EVIDENCE:
        slots.append(NO_EVIDENCE)
    return slots


def _is_valid_development_evidence(signal: HandSignal) -> bool:
    """Development-period analogue of the retriever's validity gate.

    Same clauses as :func:`poker_collusion.evidence.retriever.is_valid_evidence` (both players
    present — trusted from the shared-hand partition — plus ≥1 disclosed-family public-action flag)
    but keyed to the DEVELOPMENT period instead of evaluation. Implemented by delegating the
    behavior-flag clause to the production gate on a temporary evaluation-phase view of the signal,
    so the two stay in lockstep on what counts as a behavior-specific action.
    """
    if signal.phase != DEVELOPMENT_PHASE:
        return False
    # Reuse the production gate's behavior-flag clause by presenting an evaluation-phase copy.
    eval_view = HandSignal(
        hand_id=signal.hand_id,
        pair_id=signal.pair_id,
        phase="evaluation",
        value_flow=signal.value_flow,
        aggression_asymmetry=signal.aggression_asymmetry,
        isolation=signal.isolation,
        mi_conflict=signal.mi_conflict,
        behavior_action_flags=signal.behavior_action_flags,
    )
    return is_valid_evidence(eval_view, is_shared=True)


# --------------------------------------------------------------------------- #
# Shared per-pair development-period inputs (single actions pass; Req 1.6)
# --------------------------------------------------------------------------- #
@dataclass
class _LabelledInputs:
    """Per-pair development-period inputs shared across all folds (built once)."""

    players: Dict[str, Tuple[str, str]]                 # pair_id -> (player_a, player_b)
    dev_hands: Dict[str, List[str]]                     # pair_id -> development hand ids
    actions_by_hand: Dict[str, pd.DataFrame]            # hand_id -> ordered action frame
    net_by_hand: Dict[str, Dict[str, float]]            # hand_id -> {player_id -> net_chips}
    hands_meta: pd.DataFrame                            # [hand_id, phase, big_blind?]


def _prepare_labelled_inputs(
    loader: DataLoader,
    labels: pd.DataFrame,
    pair_ids: Sequence[str],
) -> _LabelledInputs:
    """Resolve the development shared hands + a single actions/net/meta pass for the given pairs.

    Only the labelled pairs' development hands are touched (~1,860 pairs). We resolve the union of
    those hand ids and stream ``actions.parquet`` ONCE (reusing the pipeline's single-pass helper),
    so the 18M-row file is scanned once in bounded batches and never materialised whole (Req 1.6).
    """
    want = {str(p) for p in pair_ids}
    p1_col, p2_col = _player_columns(labels)

    players: Dict[str, Tuple[str, str]] = {}
    dev_hands: Dict[str, List[str]] = {}
    all_hand_ids: set = set()

    for row in labels.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        if pair_id not in want:
            continue
        player_a, player_b = _canonical_players(getattr(row, p1_col), getattr(row, p2_col))
        players[pair_id] = (player_a, player_b)
        shared = loader.shared_hands((player_a, player_b))
        hands = [str(h) for h in shared.development]
        dev_hands[pair_id] = hands
        all_hand_ids.update(hands)

    ordered_hand_ids = sorted(all_hand_ids)
    actions_by_hand = _actions_for_hands(loader, ordered_hand_ids)
    seats = loader.load_seats()
    net_by_hand = _net_chips_by_hand(seats, ordered_hand_ids)
    hands_meta = _hands_meta_for(loader.load_hands(), ordered_hand_ids)

    return _LabelledInputs(
        players=players,
        dev_hands=dev_hands,
        actions_by_hand=actions_by_hand,
        net_by_hand=net_by_hand,
        hands_meta=hands_meta,
    )


def _player_columns(labels: pd.DataFrame) -> Tuple[str, str]:
    """Return the two player-id column names present on the labels frame."""
    if {"player_1", "player_2"}.issubset(labels.columns):
        return "player_1", "player_2"
    if {"player_a", "player_b"}.issubset(labels.columns):
        return "player_a", "player_b"
    raise KeyError("development labels need player_1/player_2 (or player_a/player_b) columns")


def _signals_for_pair(inputs: _LabelledInputs, pair_id: str) -> List[HandSignal]:
    """Compute a pair's development-period HandSignal stream from the shared caches."""
    player_a, player_b = inputs.players[pair_id]
    hands = inputs.dev_hands.get(pair_id, [])
    if not hands:
        return []
    hand_set = set(hands)
    pair_meta = inputs.hands_meta[inputs.hands_meta["hand_id"].astype(str).isin(hand_set)]
    pair_actions = {h: inputs.actions_by_hand.get(h, pd.DataFrame()) for h in hands}
    pair_nets = {h: inputs.net_by_hand.get(h, {}) for h in hands}
    return compute_pair_signals(
        pair_id=pair_id,
        player_a=player_a,
        player_b=player_b,
        hands_meta=pair_meta,
        actions_by_hand=pair_actions,
        net_by_hand=pair_nets,
    )


# --------------------------------------------------------------------------- #
# The classical predict_fn factory
# --------------------------------------------------------------------------- #
def _make_predict_fn(
    inputs: _LabelledInputs,
    labels: pd.DataFrame,
    artifact: EdaArtifact,
    config: PipelineConfig,
):
    """Build the ``predict_fn(fold, solution) -> submission`` used by the CV harness.

    Scores every pair in the fold's solution with the classical baseline over its DEVELOPMENT
    shared hands, assigns the behavior label via the pipeline placeholder, and — for pairs the
    solution marks as positive — emits ranked development-period evidence hands (see
    :func:`select_development_evidence`). Confirmed negatives get ``NO_EVIDENCE`` slots.
    """
    pool_prior = prior_from_artifact(artifact, default=0.0)
    positive_pairs = {
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) == CONFIRMED_TARGET
    }

    def predict_fn(fold: Fold, solution: pd.DataFrame) -> pd.DataFrame:
        rows: List[dict] = []
        for pair_id in solution["pair_id"].astype(str):
            signals = _signals_for_pair(inputs, pair_id)

            if not signals:
                # No development shared hands -> documented pool-prior default row.
                rows.append(
                    _submission_row(
                        pair_id,
                        risk=float(min(max(pool_prior, 0.0), 1.0)),
                        behavior="none",
                        evidence=[NO_EVIDENCE] * MAX_EVIDENCE,
                    )
                )
                continue

            player_a, player_b = inputs.players[pair_id]
            feature_set = build_pair_feature_set(
                pair_id=pair_id,
                signals=signals,
                player_a=player_a,
                player_b=player_b,
                phase=DEVELOPMENT_PHASE,
                eda_artifact=artifact,
                config=config,
            )
            classical = score_pair(feature_set, eda_artifact=artifact)
            behavior = assign_behavior_label(
                classical.risk_score, signals, feature_set.features
            )
            # Emit evidence only for the pairs the solution treats as positives (Evidence MAP@5 is
            # computed over true positives only); negatives contribute nothing to Evidence MAP.
            if pair_id in positive_pairs:
                evidence = select_development_evidence(pair_id, signals)
            else:
                evidence = [NO_EVIDENCE] * MAX_EVIDENCE
            rows.append(
                _submission_row(
                    pair_id,
                    risk=float(classical.risk_score),
                    behavior=behavior,
                    evidence=evidence,
                )
            )
        return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))

    return predict_fn


def _submission_row(
    pair_id: str, *, risk: float, behavior: str, evidence: Sequence[str]
) -> dict:
    """Build one submission row dict in the fixed SUBMISSION_COLUMNS order."""
    ev = list(evidence)[:MAX_EVIDENCE] + [NO_EVIDENCE] * (MAX_EVIDENCE - min(len(evidence), MAX_EVIDENCE))
    return {
        "pair_id": pair_id,
        "risk_score": risk,
        "predicted_behavior": behavior,
        "evidence_hand_1": ev[0],
        "evidence_hand_2": ev[1],
        "evidence_hand_3": ev[2],
        "evidence_hand_4": ev[3],
        "evidence_hand_5": ev[4],
    }


# --------------------------------------------------------------------------- #
# (B) Floor measurement
# --------------------------------------------------------------------------- #
def measure_baseline_floor(
    *,
    config: Optional[PipelineConfig] = None,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    n_folds: int = 5,
    n_boot: int = 500,
    alpha: float = 0.05,
    max_pairs: Optional[int] = None,
    ci_scope: str = "pooled",
) -> FloorResult:
    """Measure the classical baseline's combined three-component local-CV floor + bootstrap CIs.

    Runs the group-by-pool, family-stratified, PU-correct CV harness over the real
    ``development_labels.csv``, scoring each fold's confirmed pairs with the classical baseline
    (see the module docstring for the exact ``predict_fn``), then aggregates the three components
    plus the weighted combined summary as the mean across folds and per-fold, and computes
    bootstrap confidence intervals.

    Args:
        config: Pipeline config (defaults to :func:`get_config`).
        data_loader: Optional pre-built loader (defaults to ``DataLoader(config=config)``).
        eda_artifact: Optional pre-computed artifact (else cached/loaded/computed via the pipeline).
        n_folds: Number of CV folds (default 5).
        n_boot: Bootstrap replicates for the CIs (default 500).
        alpha: CI significance level (default 0.05 -> 95% CIs).
        max_pairs: Optional documented cap on the number of confirmed labelled pairs scored (for a
            faster run on a constrained machine); recorded as a caveat. ``None`` uses all.
        ci_scope: ``"pooled"`` (default) computes CIs on the pooled solution/submission across all
            folds; ``"representative"`` computes them on the single largest fold. Recorded in
            ``ci_method``.

    Returns:
        A :class:`FloorResult`.
    """
    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)
    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)

    labels = loader.load_labels_frame()
    evidence = _load_development_evidence(loader)

    caveats: List[str] = []
    if max_pairs is not None and max_pairs < len(labels):
        labels = labels.head(int(max_pairs)).copy()
        caveats.append(
            f"SAMPLED: scored only the first {int(max_pairs)} labelled pairs "
            f"(of {len(loader.load_labels_frame())}) for tractability."
        )

    confirmed_pair_ids = [
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}
    ]

    inputs = _prepare_labelled_inputs(loader, labels, confirmed_pair_ids)
    predict_fn = _make_predict_fn(inputs, labels, artifact, cfg)

    harness = CVHarness.from_real_data(
        labels,
        config=cfg,
        evidence=evidence,
        n_folds=n_folds,
        seed=cfg.seeds.cv_split,
    )
    report = harness.run(predict_fn)

    # Aggregate the per-fold + mean components.
    per_fold = report.per_fold
    mean = report.mean
    n_scored = sum(f.n_validation_pairs for f in report.folds)

    # Bootstrap CIs: rebuild the (solution, submission) pair for the chosen scope and resample.
    intervals, ci_method, n_positive = _bootstrap_intervals(
        harness, predict_fn, labels, n_boot=n_boot, alpha=alpha, scope=ci_scope, seed=cfg.seeds.bootstrap
    )

    return FloorResult(
        mean=mean,
        per_fold=per_fold,
        intervals=intervals,
        ci_method=ci_method,
        n_folds=len(report.folds),
        n_pairs_scored=int(n_scored),
        n_positive_pairs=int(n_positive),
        caveats=caveats,
        deterministic=True,
    )


def _bootstrap_intervals(
    harness: CVHarness,
    predict_fn,
    labels: pd.DataFrame,
    *,
    n_boot: int,
    alpha: float,
    scope: str,
    seed: int,
) -> Tuple[Dict[str, ConfidenceInterval], str, int]:
    """Compute bootstrap CIs on the pooled (or representative-fold) solution/submission.

    ``scope="pooled"`` concatenates every fold's solution+submission (pairs are globally unique)
    and resamples pairs with replacement across the whole confirmed set — the most representative
    band for the mean floor. ``scope="representative"`` uses the single largest fold. Both use
    :func:`poker_collusion.validation.bootstrap.bootstrap_components` (percentile CIs, seeded by
    ``config.Seeds.bootstrap``).
    """
    fold_frames: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in harness.folds():
        solution = build_fold_solution(
            harness.labels, fold.validation_pairs, evidence=harness.evidence
        )
        if len(solution) == 0:
            continue
        submission = predict_fn(fold, solution)
        fold_frames.append((solution, submission))

    if not fold_frames:
        raise ValueError("No non-empty folds to bootstrap.")

    if scope == "representative":
        solution, submission = max(fold_frames, key=lambda sf: len(sf[0]))
        scope_desc = f"the single largest fold ({len(solution)} pairs)"
    else:
        solution = pd.concat([sf[0] for sf in fold_frames], ignore_index=True)
        submission = pd.concat([sf[1] for sf in fold_frames], ignore_index=True)
        scope_desc = f"the pooled solution across all folds ({len(solution)} pairs)"

    n_positive = int(pd.to_numeric(solution["risk_score"], errors="coerce").fillna(0).astype(int).sum())

    samples = bootstrap_components(solution, submission, n_boot=n_boot, seed=seed)
    intervals = samples.intervals(alpha=alpha)
    ci_method = (
        f"Percentile bootstrap ({n_boot} replicates, seed={seed}) resampling PAIRS with "
        f"replacement on {scope_desc}; two-sided {int(round((1 - alpha) * 100))}% CIs "
        f"([{alpha / 2:.3f}, {1 - alpha / 2:.3f}] percentiles) per component."
    )
    return intervals, ci_method, n_positive


def _load_development_evidence(loader: DataLoader) -> pd.DataFrame:
    """Load ``development_evidence.csv`` (pair_id, evidence_rank, hand_id, behavior_family)."""
    try:
        path = loader._resolve_path("development_evidence", "development_evidence.csv")  # type: ignore[attr-defined]
    except Exception:
        return pd.DataFrame(columns=["pair_id", "evidence_rank", "hand_id", "behavior_family"])
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# (A) Schema-acceptance confirmation
# --------------------------------------------------------------------------- #
def confirm_schema_acceptance(
    *,
    config: Optional[PipelineConfig] = None,
    limit_pairs: int = 200,
) -> Dict[str, object]:
    """Run the pipeline with a small ``limit_pairs`` cap and confirm the file is schema-valid.

    Runs :func:`poker_collusion.pipeline.run_baseline_pipeline` with ``limit_pairs`` on the real
    data, reads the produced ``submission.csv`` back from disk, and validates it against
    ``sample_submission.csv`` via :func:`~poker_collusion.submission.writer.validate_submission`
    (the leaderboard's own acceptance check). We deliberately do NOT run the full ~112k-pair
    pipeline — a limited run that still emits a FULL, schema-valid file confirms the acceptance
    SHAPE (documented limit; unprocessed pairs get the writer's safe default row).

    Returns a small dict describing the run (path, row count, whether acceptance passed, the limit).
    """
    from poker_collusion.pipeline import run_baseline_pipeline

    cfg = config or get_config()
    out = run_baseline_pipeline(cfg, limit_pairs=limit_pairs)
    frame = pd.read_csv(out)
    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()
    validate_submission(frame, sample)  # raises on any mismatch
    return {
        "submission_path": str(out),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "schema_valid": True,
        "limit_pairs": int(limit_pairs),
        "note": (
            "Limited pipeline run: only the first "
            f"{int(limit_pairs)} evaluation pairs are fully scored; the rest receive the writer's "
            "safe default row, so the emitted file is still a FULL, schema-valid submission "
            "confirming acceptance shape."
        ),
    }
