"""Vectorized FULL Phase-2 evaluation submission generator (train-once-on-dev design).

:func:`run_phase2_pipeline_fast` produces a complete Kaggle ``submission.csv`` over ALL
evaluation pairs (112,540 on the real data) using the Phase-2 pipeline — a PU-aware risk model
(:class:`~poker_collusion.models.pu_ranker.PURanker`) plus a behavior classifier
(:class:`~poker_collusion.models.behavior_classifier.BehaviorClassifier`) — layered on top of the
proven Phase-1 vectorized generator.

What this module REUSES (DRY — nothing here is reinvented)
----------------------------------------------------------
* Phase-1 vectorized eval-hand precompute + single actions pass, verbatim, from
  :mod:`poker_collusion.pipeline_fast`:
    - :func:`~poker_collusion.pipeline_fast.build_eval_hands_by_player` (player -> eval hand set),
    - :func:`~poker_collusion.pipeline_fast.build_rows_by_hand` (one actions scan -> pre-parsed
      per-hand rows the signal path consumes),
    - the exact per-pair signal -> feature flow: :func:`compute_pair_signals` with
      ``actions_by_hand={}`` + shared ``rows_by_hand`` + ``net_by_hand`` + a tiny per-pair meta
      frame, then :func:`build_pair_feature_set` (``phase="evaluation"``),
    - :func:`~poker_collusion.evidence.retriever.select_evidence` (evidence is IDENTICAL to
      Phase-1 — only risk + behavior change),
    - :func:`~poker_collusion.submission.writer.write_submission` (same schema/order,
      self-validating, atomic).
* The PROVEN Phase-2 model wiring from :mod:`poker_collusion.validation.phase2_gate`: how the
  PURanker is trained on ``{pair_id -> label_status}`` and the BehaviorClassifier on
  ``{pair_id -> behavior_family}`` for the confirmed-target positives, and how
  ``predict_labels`` routes on a risk map — including the HALT-safe fallbacks (fall back to the
  classical risk / :func:`assign_behavior_label` so generation always completes).

The train-once-on-dev design (the ONLY behavioral difference from the CV gate)
------------------------------------------------------------------------------
The Phase-2 gate trains a fresh PU/behavior model PER CV FOLD (leakage-safe measurement). This
generator instead trains ONCE on ALL confirmed development labels, then scores every real
evaluation pair with that single fitted pair of models — the correct shape for producing the
actual competition submission (there are no folds at inference time; every confirmed dev label is
legitimate training signal for the eval set).

Two bounded actions passes (Req 1.6 — the 18.6M-row file is never materialised whole)
-------------------------------------------------------------------------------------
1. TRAIN pass: over the union of DEVELOPMENT-period shared hands of the confirmed dev pairs only
   (~1,860 pairs), built the same way as :func:`build_eval_hands_by_player` but filtered to the
   development phase and intersected per confirmed pair.
2. SCORE pass: the Phase-1 eval pass, verbatim, over the union of every eval pair's shared
   evaluation hands.

Evidence is bit-identical to Phase-1
------------------------------------
Only ``risk_score`` and ``predicted_behavior`` differ from the Phase-1 submission. The evidence
slots come from the same :func:`select_evidence` call over the same eval-period signals, so
Evidence MAP@5 is unchanged relative to the Phase-1 floor.

Determinism
-----------
The classical baseline is deterministic; the PU/behavior estimators are seeded
(:class:`~poker_collusion.config.Seeds`); the vectorized precompute is order-independent; the
writer emits rows in the fixed ``sample_submission`` order. Two runs on identical inputs/config
produce byte-identical submissions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

import pandas as pd

from poker_collusion.config import NO_EVIDENCE, PipelineConfig, get_config
from poker_collusion.evidence.retriever import MAX_EVIDENCE, select_evidence
from poker_collusion.features.aggregation import prior_from_artifact
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.features.pair_features import build_pair_feature_set
from poker_collusion.features.signals import compute_pair_signals
from poker_collusion.io import DataLoader
from poker_collusion.io.data_loader import PHASE_DEVELOPMENT
from poker_collusion.models.behavior_classifier import (
    DEFAULT_COORDINATION_THRESHOLD,
    DEFAULT_OTHER_COORDINATION_MARGIN,
    BehaviorClassifier,
)
from poker_collusion.models.classical import score_pair
from poker_collusion.models.pu_ranker import PURanker
from poker_collusion.parallel_scoring import (
    score_pairs_parallel,
    score_pairs_sequential,
)
from poker_collusion.pipeline import (
    _canonical_players,
    _hands_meta_for,
    _net_chips_by_hand,
    _obtain_eda_artifact,
    assign_behavior_label,
)
# Reuse the Phase-1 vectorized helpers verbatim (DRY: do NOT copy them).
from poker_collusion.pipeline_fast import (
    _EMPTY_ACTIONS,
    build_eval_hands_by_player,
    build_rows_by_hand,
)
from poker_collusion.submission.writer import write_submission
from poker_collusion.types import PairFeatureSet, PairPrediction

# Real label-status / behavior-family tokens (same source the gate uses).
from poker_collusion.validation.cv import CONFIRMED_NON_TARGET, CONFIRMED_TARGET

__all__ = [
    "run_phase2_pipeline_fast",
    "build_dev_hands_by_player",
    "main",
]

#: Phase token for the development slice the models are trained on.
_DEVELOPMENT_PHASE = "development"


def build_dev_hands_by_player(
    seats: pd.DataFrame,
    hands: pd.DataFrame,
) -> Dict[str, FrozenSet[str]]:
    """Build ``player_id -> frozenset(development-phase hand_id)`` in ONE pass.

    The development-phase analogue of
    :func:`poker_collusion.pipeline_fast.build_eval_hands_by_player`: it resolves the
    development-phase hand ids from ``hands`` (``phase == "development"``), filters ``seats`` to
    those hands in a single ``isin`` pass, and groups the surviving ``(player_id -> hand_id)`` rows
    into per-player frozensets. For any confirmed pair ``(p1, p2)`` the intersection of their
    frozensets is exactly the development-period shared-hand set
    ``DataLoader.shared_hands((p1, p2)).development`` produces.
    """
    if hands is None or len(hands) == 0 or seats is None or len(seats) == 0:
        return {}

    dev_hand_ids: Set[str] = set(
        hands.loc[hands["phase"] == PHASE_DEVELOPMENT, "hand_id"].astype(str).tolist()
    )
    if not dev_hand_ids:
        return {}

    seat_hand = seats["hand_id"].astype(str)
    mask = seat_hand.isin(dev_hand_ids)
    if not mask.any():
        return {}

    sub = pd.DataFrame(
        {
            "player_id": seats["player_id"].astype(str)[mask].to_numpy(),
            "hand_id": seat_hand[mask].to_numpy(),
        }
    )

    out: Dict[str, FrozenSet[str]] = {}
    for player_id, grp in sub.groupby("player_id", sort=False):
        out[str(player_id)] = frozenset(grp["hand_id"].tolist())
    return out


def _build_pair_feature_set(
    *,
    pair_id: str,
    player_a: str,
    player_b: str,
    hand_ids: Sequence[str],
    phase: str,
    rows_by_hand: Dict[str, List[dict]],
    net_by_hand: Dict[str, Dict[str, float]],
    meta_record_by_hand: Dict[str, dict],
    meta_cols: Sequence[str],
    artifact: EdaArtifact,
    config: PipelineConfig,
):
    """Signals -> :class:`PairFeatureSet` for one pair, reusing the Phase-1 fast per-pair flow.

    Returns ``(feature_set, signals)`` or ``(None, [])`` when the pair has no shared hands in the
    period (the caller then emits the pool-prior default row). ``rows_by_hand``/``net_by_hand``
    cover the whole union for the phase, so only this pair's own hand ids are read from them
    (results are keyed by hand id, so this is identical to a per-pair scan).
    """
    if not hand_ids:
        return None, []
    pair_meta = pd.DataFrame(
        [meta_record_by_hand[h] for h in hand_ids if h in meta_record_by_hand],
        columns=list(meta_cols),
    )
    signals = compute_pair_signals(
        pair_id=pair_id,
        player_a=player_a,
        player_b=player_b,
        hands_meta=pair_meta,
        actions_by_hand=_EMPTY_ACTIONS,
        net_by_hand=net_by_hand,
        rows_by_hand=rows_by_hand,
    )
    feature_set = build_pair_feature_set(
        pair_id=pair_id,
        signals=signals,
        player_a=player_a,
        player_b=player_b,
        phase=phase,
        eda_artifact=artifact,
        config=config,
    )
    return feature_set, signals


def _meta_record_map(hands_meta: pd.DataFrame):
    """Return ``(meta_cols, {hand_id -> record dict})`` for tiny per-pair meta assembly."""
    meta_cols = list(hands_meta.columns)
    record_by_hand: Dict[str, dict] = {
        str(getattr(r, "hand_id")): {c: getattr(r, c) for c in meta_cols}
        for r in hands_meta.itertuples(index=False)
    }
    return meta_cols, record_by_hand


# --------------------------------------------------------------------------- #
# TRAIN STAGE: fit PURanker + BehaviorClassifier once on ALL confirmed dev labels
# --------------------------------------------------------------------------- #
def _train_models(
    loader: DataLoader,
    labels: pd.DataFrame,
    seats: pd.DataFrame,
    hands: pd.DataFrame,
    artifact: EdaArtifact,
    config: PipelineConfig,
):
    """Train the PU ranker + behavior classifier once on the confirmed development labels.

    Mirrors the per-fold training in :mod:`poker_collusion.validation.phase2_gate` exactly, but
    trains ONCE on ALL confirmed dev pairs. Returns ``(ranker_or_None, classifier_or_None,
    status_map)``. HALT-safe: if a model cannot train (degenerate labels / solver failure) it is
    returned as ``None`` and the score stage falls back to the classical risk / the Phase-1
    behavior rule so generation still completes.
    """
    # {pair_id -> label_status} for confirmed pairs (the PU status map).
    status_map: Dict[str, str] = {}
    # {pair_id -> behavior_family} for confirmed-target positives (the behavior head labels).
    family_map: Dict[str, str] = {}
    # {pair_id -> (player_a, player_b)} for every confirmed pair.
    players: Dict[str, tuple] = {}

    for row in labels.itertuples(index=False):
        status = str(getattr(row, "label_status", ""))
        if status not in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}:
            continue
        pair_id = str(getattr(row, "pair_id"))
        players[pair_id] = _canonical_players(
            getattr(row, "player_1"), getattr(row, "player_2")
        )
        status_map[pair_id] = status
        if status == CONFIRMED_TARGET:
            family_map[pair_id] = str(getattr(row, "behavior_family", ""))

    if not players:
        return None, None, status_map

    # Development shared hands per confirmed pair (vectorized: player->dev-hand-set + intersect).
    dev_hands_by_player = build_dev_hands_by_player(seats, hands)
    dev_hands_by_pair: Dict[str, List[str]] = {}
    union_dev_hands: Set[str] = set()
    for pair_id, (player_a, player_b) in players.items():
        set_a = dev_hands_by_player.get(player_a, frozenset())
        set_b = dev_hands_by_player.get(player_b, frozenset())
        shared = sorted(set_a & set_b)
        dev_hands_by_pair[pair_id] = shared
        union_dev_hands.update(shared)

    # ONE bounded actions pass + net/meta over the confirmed dev hands only (~1,860 pairs).
    ordered = sorted(union_dev_hands)
    rows_by_hand = build_rows_by_hand(loader, ordered)
    net_by_hand = _net_chips_by_hand(seats, ordered)
    dev_meta = _hands_meta_for(hands, ordered)
    meta_cols, meta_record_by_hand = _meta_record_map(dev_meta)

    # One development-phase PairFeatureSet per confirmed pair (skip those with no dev hands).
    train_fs: List[PairFeatureSet] = []
    for pair_id, (player_a, player_b) in players.items():
        fs, _ = _build_pair_feature_set(
            pair_id=pair_id,
            player_a=player_a,
            player_b=player_b,
            hand_ids=dev_hands_by_pair.get(pair_id, []),
            phase=_DEVELOPMENT_PHASE,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=config,
        )
        if fs is not None:
            train_fs.append(fs)

    # --- Train the PU ranker (HALT-safe fallback). ---
    ranker: Optional[PURanker] = None
    has_pos = any(status_map.get(str(fs.pair_id)) == CONFIRMED_TARGET for fs in train_fs)
    if train_fs and has_pos:
        try:
            r = PURanker(config=config)
            r.fit(train_fs, status_map, eda_artifact=artifact)
            ranker = r
        except Exception:
            ranker = None

    # --- Train the behavior classifier (HALT-safe fallback). ---
    classifier: Optional[BehaviorClassifier] = None
    pos_family = {
        str(fs.pair_id): family_map[str(fs.pair_id)]
        for fs in train_fs
        if str(fs.pair_id) in family_map
    }
    if train_fs and pos_family:
        try:
            c = BehaviorClassifier(config=config)
            c.fit(train_fs, pos_family, eda_artifact=artifact)
            classifier = c
        except Exception:
            classifier = None

    return ranker, classifier, status_map


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_phase2_pipeline_fast(
    config: Optional[PipelineConfig] = None,
    *,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    limit_pairs: Optional[int] = None,
    parallel: bool = False,
    n_workers: Optional[int] = None,
) -> Path:
    """Generate the FULL Phase-2 evaluation submission (train-once-on-dev, vectorized scoring).

    Args:
        config: Pipeline config (defaults to :func:`get_config`).
        data_loader: Optional pre-built loader (defaults to ``DataLoader(config=config)``).
        eda_artifact: Optional pre-computed artifact (else cached/loaded/computed via the pipeline).
        limit_pairs: Smoke run — score only the first N evaluation pairs (the rest get the writer's
            safe default row so the emitted file is still a FULL, schema-valid submission).
        parallel: Route the independent per-pair eval feature/score/evidence build through the
            deterministic multiprocess driver (Job-1 parallel path). The PU risk + classifier still
            run as batch steps in the parent, so the emitted submission is identical to the
            sequential build. Default ``False`` keeps the sequential behavior.
        n_workers: Worker count when ``parallel=True`` (``None`` -> ``min(cpu_count, 8)``).

    Returns:
        The :class:`~pathlib.Path` the ``submission.csv`` was written to.
    """
    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)

    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)
    pool_prior = prior_from_artifact(artifact, default=0.0)

    labels = loader.load_labels_frame()
    evaluation_pairs = loader.load_evaluation_pairs()
    sample_submission = loader.load_sample_submission()
    seats = loader.load_seats()
    hands = loader.load_hands()

    if limit_pairs is not None:
        evaluation_pairs = evaluation_pairs.head(int(limit_pairs)).copy()

    # ---- (1) TRAIN STAGE: fit PU ranker + behavior classifier ONCE on confirmed dev labels. -- #
    ranker, classifier, _status_map = _train_models(
        loader, labels, seats, hands, artifact, cfg
    )

    # ---- (2) SCORE STAGE precompute: eval-hand sets + one actions pass (Phase-1 verbatim). --- #
    eval_hands_by_player = build_eval_hands_by_player(seats, hands)

    pair_rows: List[dict] = []
    all_eval_hand_ids: Set[str] = set()
    for row in evaluation_pairs.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        player_a, player_b = _canonical_players(
            getattr(row, "player_1"), getattr(row, "player_2")
        )
        set_a = eval_hands_by_player.get(player_a, frozenset())
        set_b = eval_hands_by_player.get(player_b, frozenset())
        eval_hands = sorted(set_a & set_b)
        all_eval_hand_ids.update(eval_hands)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "player_a": player_a,
                "player_b": player_b,
                "eval_hands": eval_hands,
            }
        )

    ordered_eval_hand_ids = sorted(all_eval_hand_ids)
    rows_by_hand = build_rows_by_hand(loader, ordered_eval_hand_ids)
    net_by_hand = _net_chips_by_hand(seats, ordered_eval_hand_ids)
    eval_meta = _hands_meta_for(hands, ordered_eval_hand_ids)
    meta_cols, meta_record_by_hand = _meta_record_map(eval_meta)

    # ---- (3) Build every scoreable eval pair's PairFeatureSet (same flow as Phase-1). -------- #
    # Route the model-free per-pair build through the shared driver. ``parallel=True`` uses the
    # Job-1 multiprocess path (byte-identical, contiguous chunks preserve order); the PU risk and
    # the behavior classifier remain batch steps in the parent below.
    if parallel:
        scored = score_pairs_parallel(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=cfg,
            n_workers=n_workers,
        )
    else:
        scored = score_pairs_sequential(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=cfg,
        )

    feature_by_pair: Dict[str, PairFeatureSet] = {}
    signals_by_pair: Dict[str, list] = {}
    evidence_by_pair: Dict[str, list] = {}
    phase1_behavior_by_pair: Dict[str, str] = {}
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        res = scored.get(pair_id)
        if res is None:
            continue
        feature_by_pair[pair_id] = res.feature_set
        signals_by_pair[pair_id] = res.signals
        evidence_by_pair[pair_id] = res.evidence_slots
        phase1_behavior_by_pair[pair_id] = res.phase1_behavior

    scoreable_fs = list(feature_by_pair.values())

    # ---- (4) Risk: batch PU score over all scoreable pairs (fallback to classical per pair). - #
    pu_risk: Dict[str, float] = {}
    if ranker is not None and scoreable_fs:
        try:
            pu_risk = ranker.score_pairs(scoreable_fs, eda_artifact=artifact)
        except Exception:
            pu_risk = {}

    risk_by_pair: Dict[str, float] = {}
    classical_risk: Dict[str, float] = {}
    for pair_id in feature_by_pair:
        # The driver already computed the deterministic classical risk (fallback source).
        classical = float(scored[pair_id].classical_risk)
        classical_risk[pair_id] = classical
        risk = pu_risk.get(pair_id, classical)
        risk_by_pair[pair_id] = float(min(max(risk, 0.0), 1.0))

    # ---- (5) Behavior: classifier label routed on the risk map (fallback to Phase-1 rule). --- #
    label_by_pair: Dict[str, str] = {}
    if classifier is not None and scoreable_fs:
        # Route on the PU risk when available, else the classical risk (so routing has a value).
        routing_risk = {
            str(fs.pair_id): float(risk_by_pair[str(fs.pair_id)]) for fs in scoreable_fs
        }
        try:
            predicted = classifier.predict_labels(
                scoreable_fs,
                routing_risk,
                coordination_threshold=DEFAULT_COORDINATION_THRESHOLD,
                other_coordination_margin=DEFAULT_OTHER_COORDINATION_MARGIN,
                eda_artifact=artifact,
            )
            label_by_pair = {pid: str(lbl) for pid, lbl in predicted.items()}
        except Exception:
            label_by_pair = {}

    # ---- (6) Assemble predictions (evidence IDENTICAL to Phase-1). --------------------------- #
    predictions: List[PairPrediction] = []
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        fs = feature_by_pair.get(pair_id)

        if fs is None:
            # No evaluation shared hands -> documented pool-prior default row (like Phase-1).
            predictions.append(
                PairPrediction(
                    pair_id=pair_id,
                    risk_score=float(min(max(pool_prior, 0.0), 1.0)),
                    predicted_behavior="none",
                    evidence=[NO_EVIDENCE] * MAX_EVIDENCE,
                    reasons=["no evaluation-period shared hands -> pool-prior default row"],
                )
            )
            continue

        risk = risk_by_pair[pair_id]

        # Behavior: classifier label when available, else the Phase-1 placeholder rule (from the
        # driver, which computed it against classical_risk[pair_id] exactly as before).
        if pair_id in label_by_pair:
            behavior = label_by_pair[pair_id]
        else:
            behavior = phase1_behavior_by_pair[pair_id]

        # Evidence: SAME selection as Phase-1 (only risk + behavior change).
        evidence_slots = evidence_by_pair[pair_id]

        risk_source = "pu_ranker" if pair_id in pu_risk else "classical_fallback"
        predictions.append(
            PairPrediction(
                pair_id=pair_id,
                risk_score=float(risk),
                predicted_behavior=behavior,
                evidence=list(evidence_slots),
                reasons=[f"phase2 risk={risk_source}"],
            )
        )

    # ---- (7) Route into the writer (self-validating, atomic, deterministic row order). ------- #
    return write_submission(
        predictions,
        sample_submission=sample_submission,
        config=cfg,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m poker_collusion.pipeline_phase2_fast [--limit-pairs N]``.

    Mirrors :func:`poker_collusion.pipeline_fast.main` but runs the Phase-2 generator.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m poker_collusion.pipeline_phase2_fast",
        description="Generate the FULL Phase-2 poker-collusion evaluation submission.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Smoke run: process only the first N evaluation pairs (rest get the default row).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Route the per-pair eval build through the deterministic multiprocess driver.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker count when --parallel is set (default: min(cpu_count, 8)).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = run_phase2_pipeline_fast(
        limit_pairs=args.limit_pairs,
        parallel=args.parallel,
        n_workers=args.n_workers,
    )
    print(f"submission written to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
