"""Phase-2 gate: does the PU-ranker + behavior-classifier pipeline beat the Phase-1 floor?

Task 19 (Req 12.2, 12.4; Design "submission-selection rule"). This module answers the Phase-2
checkpoint question honestly:

    Does the **Phase-2** pipeline (PU-aware risk model + behavior classifier) beat the **Phase-1**
    classical floor **beyond the bootstrap noise band** on the combined summary **with no component
    regression**, on the SAME bounded real dev subset the floor was measured on?

Verdict rule (identical acceptance rule every phase gate uses)
--------------------------------------------------------------
On the pooled-across-folds (solution, phase1_submission, phase2_submission) triple we compute the
**paired bootstrap of the difference** ``phase2 - phase1`` (``paired_bootstrap_diff`` /
``is_improvement_beyond_band``, seed :data:`config.Seeds.bootstrap`) for the ``combined`` summary
and each of the three components. Then:

* ``BEATS-FLOOR`` iff the ``combined`` paired-diff CI lies strictly **above zero** AND no component
  paired-diff CI lies strictly **below zero** (no regression).
* ``DOES-NOT-BEAT`` otherwise (recorded with the deltas + CIs either way).

This is an HONEST measurement. If Phase 2 does not beat the floor, that is a valid, informative gate
outcome — the shipped submission stays the best-CV config (the Phase-1 floor) per the selection rule.

Apples-to-apples with the floor (the ONLY changes are risk + behavior)
----------------------------------------------------------------------
Both the Phase-1 and Phase-2 per-fold submissions are produced over EXACTLY the fold's PU-correct
solution pair set (:func:`~poker_collusion.validation.cv.build_fold_solution`), on the SAME bounded
real dev subset (``max_pairs=400``, the same subset as the Phase-1 floor and task 18), with the
per-pair development-period signals/features computed **once** via the floor's cached-inputs
pattern (:func:`~poker_collusion.validation.phase1_baseline_cv._prepare_labelled_inputs`). The two
submissions differ in only two fields:

* **risk_score.** Phase-1 uses the classical baseline
  (:func:`poker_collusion.models.classical.score_pair`). Phase-2 trains a
  :class:`~poker_collusion.models.pu_ranker.PURanker` on the fold's TRAIN confirmed pairs (trusted
  positives = positive, confirmed negatives = reliable negatives, unknowns = unlabelled) and scores
  the fold's VALIDATION pairs' risk with it.
* **predicted_behavior.** Phase-1 uses the pipeline threshold placeholder
  (:func:`poker_collusion.pipeline.assign_behavior_label`). Phase-2 trains a
  :class:`~poker_collusion.models.behavior_classifier.BehaviorClassifier` on the fold's TRAIN
  positives and assigns the validation labels using the PU risk for routing.

The **evidence** slots are IDENTICAL between the two (the floor's development-period
:func:`~poker_collusion.validation.phase1_baseline_cv.select_development_evidence` for positives,
``NO_EVIDENCE`` for negatives), so Evidence MAP@5 is comparable and cannot drift the verdict.

No leakage: proper per-fold train/val
-------------------------------------
Training uses only the fold's ``train_pairs`` (every confirmed pair in the OTHER pools); scoring
uses only the fold's ``validation_pairs``. The harness guarantees group-by-pool folds, so a pool
never spans train and validation — the PU model and behavior head never see a validation pool's
pairs during training.

Runtime discipline (a prior real run timed out at 60 min)
---------------------------------------------------------
All real-data measurement is BOUNDED: the signal/feature pass over ``actions.parquet`` runs ONCE
over the capped subset (``max_pairs=400``), exactly like the floor. Per fold we only *fit two cheap
logistic models* on already-cached features — no extra actions pass. The heavy real run is driven
by :func:`run_phase2_gate` (called from a plain script, NOT pytest); the unit test in
``tests/test_phase2_gate.py`` uses TINY synthetic data only.

Determinism
-----------
The classical baseline is deterministic; the PU/behavior estimators are seeded
(:data:`Seeds.pu_ranker`, :data:`Seeds.behavior_classifier`); the CV split is seeded
(:data:`Seeds.cv_split`); the bootstrap is seeded (:data:`Seeds.bootstrap`). Two runs on identical
inputs/config return identical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from poker_collusion.config import NO_EVIDENCE, PipelineConfig, get_config
from poker_collusion.evidence.retriever import MAX_EVIDENCE
from poker_collusion.features.aggregation import prior_from_artifact
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.features.pair_features import build_pair_feature_set
from poker_collusion.io import DataLoader
from poker_collusion.models.behavior_classifier import (
    DEFAULT_COORDINATION_THRESHOLD,
    DEFAULT_OTHER_COORDINATION_MARGIN,
    BehaviorClassifier,
)
from poker_collusion.models.classical import score_pair
from poker_collusion.models.pu_ranker import PURanker
from poker_collusion.pipeline import _obtain_eda_artifact, assign_behavior_label
from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.types import PairFeatureSet
from poker_collusion.validation.bootstrap import (
    COMPONENT_KEYS,
    ConfidenceInterval,
    confidence_interval,
    paired_bootstrap_diff,
)
from poker_collusion.validation.cv import (
    CONFIRMED_NON_TARGET,
    CONFIRMED_TARGET,
    CVHarness,
    Fold,
    build_fold_solution,
)
from poker_collusion.validation.phase1_baseline_cv import (
    DEVELOPMENT_PHASE,
    _LabelledInputs,
    _load_development_evidence,
    _prepare_labelled_inputs,
    _signals_for_pair,
    _submission_row,
    select_development_evidence,
)

__all__ = [
    "Phase2GateResult",
    "run_phase2_gate",
]

# The three components checked for regression (combined is handled separately as the headline gate).
_COMPONENTS: Tuple[str, ...] = ("pair_ap", "evidence_map", "behavior_map")


# --------------------------------------------------------------------------- #
# Result carrier
# --------------------------------------------------------------------------- #
@dataclass
class Phase2GateResult:
    """The Phase-2 gate outcome (task 19).

    Attributes:
        phase1: ``{component -> pooled score}`` for the Phase-1 classical floor submission.
        phase2: ``{component -> pooled score}`` for the Phase-2 (PU + behavior) submission.
        deltas: ``{component -> mean paired-bootstrap difference (phase2 - phase1)}``.
        cis: ``{component -> ConfidenceInterval}`` paired-bootstrap-of-the-difference CIs.
        combined_improves: True iff the ``combined`` paired-diff CI lies strictly above zero.
        regressed_components: Components whose paired-diff CI lies strictly below zero.
        beats_floor: The verdict — True iff ``combined_improves`` AND no regressed component.
        verdict: ``"BEATS-FLOOR"`` or ``"DOES-NOT-BEAT"``.
        n_pairs: Number of pooled pairs the verdict was measured on.
        n_positive_pairs: Confirmed positives among them.
        n_folds: Folds that produced a scored submission.
        n_boot: Bootstrap replicates used.
        seed: Bootstrap seed used.
        caveats: Sampling / limit caveats (e.g. the 400/1860 subset).
        deterministic: Always True (seeded models + seeded CV/bootstrap).
    """

    phase1: Dict[str, float]
    phase2: Dict[str, float]
    deltas: Dict[str, float]
    cis: Dict[str, ConfidenceInterval]
    combined_improves: bool
    regressed_components: List[str]
    beats_floor: bool
    verdict: str
    n_pairs: int
    n_positive_pairs: int
    n_folds: int
    n_boot: int
    seed: int
    caveats: List[str] = field(default_factory=list)
    deterministic: bool = True

    def as_dict(self) -> Dict[str, object]:
        """Plain-dict view for logging / persistence."""
        return {
            "phase1": self.phase1,
            "phase2": self.phase2,
            "deltas": self.deltas,
            "cis": {k: v.as_dict() for k, v in self.cis.items()},
            "combined_improves": self.combined_improves,
            "regressed_components": self.regressed_components,
            "beats_floor": self.beats_floor,
            "verdict": self.verdict,
            "n_pairs": self.n_pairs,
            "n_positive_pairs": self.n_positive_pairs,
            "n_folds": self.n_folds,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "caveats": self.caveats,
            "deterministic": self.deterministic,
        }


# --------------------------------------------------------------------------- #
# Per-fold feature-set builder (shared by both pipelines; one cache)
# --------------------------------------------------------------------------- #
def _feature_set_for(
    inputs: _LabelledInputs,
    pair_id: str,
    artifact: EdaArtifact,
    config: PipelineConfig,
) -> Optional[PairFeatureSet]:
    """Build a pair's development-period :class:`PairFeatureSet` from the shared caches.

    Returns ``None`` when the pair has no development shared hands (no signal to build on) — the
    caller then emits the pool-prior default row, exactly like the floor.
    """
    signals = _signals_for_pair(inputs, pair_id)
    if not signals:
        return None
    player_a, player_b = inputs.players[pair_id]
    return build_pair_feature_set(
        pair_id=pair_id,
        signals=signals,
        player_a=player_a,
        player_b=player_b,
        phase=DEVELOPMENT_PHASE,
        eda_artifact=artifact,
        config=config,
    )


# --------------------------------------------------------------------------- #
# The two per-fold predict_fns (Phase-1 classical, Phase-2 PU + behavior)
# --------------------------------------------------------------------------- #
def _make_phase1_predict_fn(
    inputs: _LabelledInputs,
    labels: pd.DataFrame,
    artifact: EdaArtifact,
    config: PipelineConfig,
):
    """Phase-1 classical submission per fold (identical to the floor's ``predict_fn``)."""
    pool_prior = prior_from_artifact(artifact, default=0.0)
    positive_pairs = _positive_pairs(labels)

    def predict_fn(fold: Fold, solution: pd.DataFrame) -> pd.DataFrame:
        rows: List[dict] = []
        for pair_id in solution["pair_id"].astype(str):
            fs = _feature_set_for(inputs, pair_id, artifact, config)
            if fs is None:
                rows.append(
                    _submission_row(
                        pair_id,
                        risk=float(min(max(pool_prior, 0.0), 1.0)),
                        behavior="none",
                        evidence=[NO_EVIDENCE] * MAX_EVIDENCE,
                    )
                )
                continue
            classical = score_pair(fs, eda_artifact=artifact)
            signals = _signals_for_pair(inputs, pair_id)
            behavior = assign_behavior_label(classical.risk_score, signals, fs.features)
            evidence = (
                select_development_evidence(pair_id, signals)
                if pair_id in positive_pairs
                else [NO_EVIDENCE] * MAX_EVIDENCE
            )
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


def _make_phase2_predict_fn(
    inputs: _LabelledInputs,
    labels: pd.DataFrame,
    artifact: EdaArtifact,
    config: PipelineConfig,
):
    """Phase-2 submission per fold: PU-ranker risk + behavior-classifier labels, SAME evidence.

    Within each fold we TRAIN on the fold's ``train_pairs`` (confirmed pairs in the other pools):

    * a :class:`PURanker` on trusted positives (+ reliable negatives + unknowns are the unlabelled
      mix implied by the PU status map) — then score the fold's VALIDATION pairs' risk with it;
    * a :class:`BehaviorClassifier` on the train positives — then assign validation labels using
      the PU risk for routing.

    Evidence is the SAME development-period selection the floor uses, so the only changes vs Phase-1
    are risk_score and predicted_behavior. If a fold's train set is degenerate for a model
    (e.g. no trusted positive, or <2 disclosed families), we fall back to the Phase-1 rule for that
    field so the fold still produces a valid, comparable submission (recorded implicitly — the fold
    simply contributes a Phase-1-equivalent field there).
    """
    pool_prior = prior_from_artifact(artifact, default=0.0)
    positive_pairs = _positive_pairs(labels)
    status_map = _status_map(labels)

    def predict_fn(fold: Fold, solution: pd.DataFrame) -> pd.DataFrame:
        # --- Build feature sets once for this fold's train + validation pairs. ---
        val_ids = [str(p) for p in solution["pair_id"].astype(str)]
        train_ids = [str(p) for p in fold.train_pairs]

        train_fs = [
            fs
            for fs in (_feature_set_for(inputs, pid, artifact, config) for pid in train_ids)
            if fs is not None
        ]
        val_fs_by_id: Dict[str, Optional[PairFeatureSet]] = {
            pid: _feature_set_for(inputs, pid, artifact, config) for pid in val_ids
        }
        scoreable_val_fs = [fs for fs in val_fs_by_id.values() if fs is not None]

        # --- Train the PU ranker on the fold's train confirmed pairs (HALT-safe fallback). ---
        risk_by_pair: Dict[str, float] = {}
        pu_ok = False
        train_status = {pid: status_map.get(pid, "") for pid in train_ids}
        has_train_pos = any(v == CONFIRMED_TARGET for v in train_status.values())
        if train_fs and has_train_pos and scoreable_val_fs:
            try:
                ranker = PURanker(config=config)
                ranker.fit(train_fs, train_status, eda_artifact=artifact)
                risk_by_pair = ranker.score_pairs(scoreable_val_fs, eda_artifact=artifact)
                pu_ok = True
            except Exception:
                pu_ok = False

        # --- Train the behavior classifier on the fold's train positives (HALT-safe fallback). ---
        clf: Optional[BehaviorClassifier] = None
        train_family_map = {
            pid: status_map.get(pid, "")  # only positives carry a disclosed family below
            for pid in train_ids
        }
        pos_family = {
            pid: _behavior_family(labels, pid)
            for pid, st in train_status.items()
            if st == CONFIRMED_TARGET
        }
        if train_fs and pos_family:
            try:
                clf = BehaviorClassifier(config=config)
                clf.fit(train_fs, pos_family, eda_artifact=artifact)
            except Exception:
                clf = None

        # Behavior labels for the scoreable validation pairs (needs a risk map for routing).
        label_by_pair: Dict[str, str] = {}
        if clf is not None and scoreable_val_fs:
            # Use the PU risk when available, else the classical risk (so routing has a value).
            routing_risk: Dict[str, float] = {}
            for fs in scoreable_val_fs:
                pid = str(fs.pair_id)
                if pid in risk_by_pair:
                    routing_risk[pid] = float(risk_by_pair[pid])
                else:
                    routing_risk[pid] = float(score_pair(fs, eda_artifact=artifact).risk_score)
            try:
                predicted = clf.predict_labels(
                    scoreable_val_fs,
                    routing_risk,
                    coordination_threshold=DEFAULT_COORDINATION_THRESHOLD,
                    other_coordination_margin=DEFAULT_OTHER_COORDINATION_MARGIN,
                    eda_artifact=artifact,
                )
                label_by_pair = {pid: str(lbl) for pid, lbl in predicted.items()}
            except Exception:
                label_by_pair = {}

        # --- Assemble the submission over EXACTLY the solution pair set. ---
        rows: List[dict] = []
        for pair_id in val_ids:
            fs = val_fs_by_id.get(pair_id)
            if fs is None:
                rows.append(
                    _submission_row(
                        pair_id,
                        risk=float(min(max(pool_prior, 0.0), 1.0)),
                        behavior="none",
                        evidence=[NO_EVIDENCE] * MAX_EVIDENCE,
                    )
                )
                continue

            signals = _signals_for_pair(inputs, pair_id)
            classical = score_pair(fs, eda_artifact=artifact)

            # Risk: PU when the fold could train it, else the classical floor (comparable fallback).
            if pu_ok and pair_id in risk_by_pair:
                risk = float(risk_by_pair[pair_id])
            else:
                risk = float(classical.risk_score)

            # Behavior: classifier label when available, else the Phase-1 placeholder rule.
            if pair_id in label_by_pair:
                behavior = label_by_pair[pair_id]
            else:
                behavior = assign_behavior_label(classical.risk_score, signals, fs.features)

            # Evidence: identical to the floor (development-period selection for positives).
            evidence = (
                select_development_evidence(pair_id, signals)
                if pair_id in positive_pairs
                else [NO_EVIDENCE] * MAX_EVIDENCE
            )
            rows.append(
                _submission_row(
                    pair_id,
                    risk=float(min(max(risk, 0.0), 1.0)),
                    behavior=behavior,
                    evidence=evidence,
                )
            )
        return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))

    return predict_fn


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #
def _positive_pairs(labels: pd.DataFrame) -> set:
    """Confirmed-target pair ids."""
    return {
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) == CONFIRMED_TARGET
    }


def _status_map(labels: pd.DataFrame) -> Dict[str, str]:
    """``{pair_id -> label_status}`` for confirmed pairs (unknowns absent -> unlabelled)."""
    out: Dict[str, str] = {}
    for r in labels.itertuples(index=False):
        out[str(getattr(r, "pair_id"))] = str(getattr(r, "label_status", ""))
    return out


def _behavior_family(labels: pd.DataFrame, pair_id: str) -> str:
    """Return the ``behavior_family`` for a pair id (empty string when absent)."""
    for r in labels.itertuples(index=False):
        if str(getattr(r, "pair_id")) == str(pair_id):
            return str(getattr(r, "behavior_family", ""))
    return ""


# --------------------------------------------------------------------------- #
# Pooled fold submissions
# --------------------------------------------------------------------------- #
def _pool_fold_submissions(
    harness: CVHarness, predict_fn
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return per-fold ``(solution, submission)`` frames (non-empty folds only)."""
    frames: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in harness.folds():
        solution = build_fold_solution(
            harness.labels, fold.validation_pairs, evidence=harness.evidence
        )
        if len(solution) == 0:
            continue
        submission = predict_fn(fold, solution)
        frames.append((solution, submission))
    return frames


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def run_phase2_gate(
    *,
    config: Optional[PipelineConfig] = None,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    n_folds: int = 5,
    n_boot: int = 500,
    alpha: float = 0.05,
    max_pairs: Optional[int] = None,
) -> Phase2GateResult:
    """Measure the Phase-2-minus-Phase-1 improvement and return the gate verdict (task 19).

    Runs the group-by-pool, family-stratified, PU-correct CV harness on the real
    ``development_labels.csv`` (capped to ``max_pairs`` for the bounded run), produces BOTH the
    Phase-1 classical submission and the Phase-2 (PU + behavior) submission per fold with the
    per-pair signals/features computed ONCE (the floor's cached-inputs pattern), pools both across
    folds, and computes the paired bootstrap of the difference ``phase2 - phase1`` on the combined
    summary and each component.

    Verdict: ``BEATS-FLOOR`` iff the combined paired-diff CI lies strictly above zero AND no
    component paired-diff CI lies strictly below zero; else ``DOES-NOT-BEAT`` (deltas + CIs recorded
    either way — an honest gate).

    Args:
        config: Pipeline config (defaults to :func:`get_config`).
        data_loader: Optional pre-built loader (defaults to ``DataLoader(config=config)``).
        eda_artifact: Optional pre-computed artifact (else cached/loaded/computed via the pipeline).
        n_folds: Number of CV folds (default 5).
        n_boot: Bootstrap replicates for the paired-diff CIs (default 500).
        alpha: CI significance level (default 0.05 -> 95% CIs).
        max_pairs: Optional documented cap on the number of confirmed labelled pairs (bounded run);
            recorded as a caveat. ``None`` uses all.

    Returns:
        A :class:`Phase2GateResult`.
    """
    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)
    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)

    labels = loader.load_labels_frame()
    evidence = _load_development_evidence(loader)

    caveats: List[str] = []
    total_labelled = len(labels)
    if max_pairs is not None and max_pairs < total_labelled:
        labels = labels.head(int(max_pairs)).copy()
        caveats.append(
            f"SAMPLED: scored only the first {int(max_pairs)} labelled pairs "
            f"(of {total_labelled}) for tractability — the SAME subset as the Phase-1 floor and "
            f"task 18 (apples-to-apples)."
        )

    confirmed_pair_ids = [
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}
    ]

    # Single cached actions/net/meta pass over the capped subset (Req 1.6 — bounded run).
    inputs = _prepare_labelled_inputs(loader, labels, confirmed_pair_ids)

    phase1_fn = _make_phase1_predict_fn(inputs, labels, artifact, cfg)
    phase2_fn = _make_phase2_predict_fn(inputs, labels, artifact, cfg)

    harness = CVHarness.from_real_data(
        labels,
        config=cfg,
        evidence=evidence,
        n_folds=n_folds,
        seed=cfg.seeds.cv_split,
    )

    phase1_frames = _pool_fold_submissions(harness, phase1_fn)
    phase2_frames = _pool_fold_submissions(harness, phase2_fn)
    if not phase1_frames or not phase2_frames:
        raise ValueError("No non-empty folds to measure the Phase-2 gate on.")

    # Pool across folds (pairs are globally unique across folds).
    solution = pd.concat([sf[0] for sf in phase1_frames], ignore_index=True)
    phase1_sub = pd.concat([sf[1] for sf in phase1_frames], ignore_index=True)
    phase2_sub = pd.concat([sf[1] for sf in phase2_frames], ignore_index=True)

    return _score_gate(
        solution,
        phase1_sub,
        phase2_sub,
        n_folds=len(phase1_frames),
        n_boot=n_boot,
        alpha=alpha,
        seed=cfg.seeds.bootstrap,
        caveats=caveats,
    )


def _score_gate(
    solution: pd.DataFrame,
    phase1_sub: pd.DataFrame,
    phase2_sub: pd.DataFrame,
    *,
    n_folds: int,
    n_boot: int,
    alpha: float,
    seed: int,
    caveats: List[str],
) -> Phase2GateResult:
    """Score the pooled triple and build the verdict (shared by the real run and the unit test)."""
    from poker_collusion.metric.production_metric import score_components

    phase1_scores = {k: float(v) for k, v in score_components(solution, phase1_sub).as_dict().items()}
    phase2_scores = {k: float(v) for k, v in score_components(solution, phase2_sub).as_dict().items()}

    deltas: Dict[str, float] = {}
    cis: Dict[str, ConfidenceInterval] = {}
    regressed: List[str] = []
    for key in COMPONENT_KEYS:
        diffs = paired_bootstrap_diff(
            solution, phase1_sub, phase2_sub, key=key, n_boot=n_boot, seed=seed
        )
        ci = confidence_interval(diffs, alpha=alpha)
        deltas[key] = float(ci.mean)
        cis[key] = ci
        if key in _COMPONENTS and ci.high < 0.0:
            regressed.append(key)

    combined_improves = cis["combined"].low > 0.0
    beats = combined_improves and not regressed
    verdict = "BEATS-FLOOR" if beats else "DOES-NOT-BEAT"

    n_positive = int(
        pd.to_numeric(solution["risk_score"], errors="coerce").fillna(0).astype(int).sum()
    )

    return Phase2GateResult(
        phase1=phase1_scores,
        phase2=phase2_scores,
        deltas=deltas,
        cis=cis,
        combined_improves=combined_improves,
        regressed_components=regressed,
        beats_floor=beats,
        verdict=verdict,
        n_pairs=int(len(solution)),
        n_positive_pairs=n_positive,
        n_folds=int(n_folds),
        n_boot=int(n_boot),
        seed=int(seed),
        caveats=caveats,
        deterministic=True,
    )
