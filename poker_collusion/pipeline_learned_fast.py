"""Vectorized FULL evaluation submission generator with the LEARNED risk combiner.

:func:`run_learned_pipeline_fast` mirrors
:func:`poker_collusion.pipeline_phase2_fast.run_phase2_pipeline_fast` EXACTLY, with ONE behavioral
difference: the ``risk_score`` comes from the supervised
:class:`~poker_collusion.models.learned_risk.LearnedRiskModel` (a logistic over the SAME feature
records that reached group-by-pool dev CV AUC ~0.85 in the signal-separation measurement) instead of
the PU ranker. Everything else is bit-identical to Phase-2:

* behavior = :class:`~poker_collusion.models.behavior_classifier.BehaviorClassifier` (unchanged);
* evidence = :func:`~poker_collusion.evidence.retriever.select_evidence` (unchanged, so Evidence
  MAP@5 is identical to Phase-1/Phase-2);
* pairs with no evaluation shared hands -> the documented pool-prior default row (Phase-1/2);
* the writer (:func:`~poker_collusion.submission.writer.write_submission`) emits the same schema and
  row order, self-validating and atomic.

So ONLY the risk head changes versus Phase-2. This is the "replace the saturated classical risk with
the learned combiner" build — a single model swap, no score-chasing loop.

Reuse (DRY)
-----------
The Phase-2 fast helpers are reused verbatim: :func:`build_dev_hands_by_player`,
:func:`_build_pair_feature_set`, :func:`_meta_record_map`, and the Phase-1 vectorized precompute
(:func:`build_eval_hands_by_player`, :func:`build_rows_by_hand`). Only the risk MODEL differs, so the
train stage here trains the LearnedRiskModel (risk) plus the same BehaviorClassifier (behavior).

Two bounded actions passes (Req 1.6 — the 18.6M-row file is never materialised whole)
-------------------------------------------------------------------------------------
1. TRAIN pass over the confirmed dev pairs' development shared hands (~1,860 pairs).
2. SCORE pass over the union of every eval pair's shared evaluation hands.

Determinism
-----------
The classical fallback is deterministic; the learned/behavior estimators are seeded
(:class:`~poker_collusion.config.Seeds`); the vectorized precompute is order-independent; the writer
emits the fixed ``sample_submission`` row order. Two runs on identical inputs/config produce
byte-identical submissions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from poker_collusion.config import NO_EVIDENCE, PipelineConfig, get_config
from poker_collusion.evidence.retriever import MAX_EVIDENCE, select_evidence
from poker_collusion.features.aggregation import prior_from_artifact
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.io import DataLoader
from poker_collusion.models.behavior_classifier import (
    DEFAULT_COORDINATION_THRESHOLD,
    DEFAULT_OTHER_COORDINATION_MARGIN,
    BehaviorClassifier,
)
from poker_collusion.models.classical import score_pair
from poker_collusion.models.learned_risk import LearnedRiskModel
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
from poker_collusion.pipeline_fast import build_eval_hands_by_player, build_rows_by_hand
# Reuse the Phase-2 fast helpers verbatim (DRY: do NOT copy them).
from poker_collusion.pipeline_phase2_fast import (
    _DEVELOPMENT_PHASE,
    _build_pair_feature_set,
    _meta_record_map,
    build_dev_hands_by_player,
)
from poker_collusion.submission.writer import write_submission
from poker_collusion.types import PairFeatureSet, PairPrediction
from poker_collusion.validation.cv import CONFIRMED_NON_TARGET, CONFIRMED_TARGET

__all__ = [
    "run_learned_pipeline_fast",
    "main",
]


# --------------------------------------------------------------------------- #
# TRAIN STAGE: fit LearnedRiskModel (risk) + BehaviorClassifier once on confirmed dev labels
# --------------------------------------------------------------------------- #
def _train_models(
    loader: DataLoader,
    labels: pd.DataFrame,
    seats: pd.DataFrame,
    hands: pd.DataFrame,
    artifact: EdaArtifact,
    config: PipelineConfig,
):
    """Train the LEARNED risk model + behavior classifier once on the confirmed development labels.

    Mirrors :func:`poker_collusion.pipeline_phase2_fast._train_models` exactly but swaps the risk
    model: it trains a :class:`LearnedRiskModel` (risk) instead of a ``PURanker``. Returns
    ``(risk_model_or_None, classifier_or_None, status_map)``. HALT-safe: a model that cannot train
    (degenerate labels / solver failure) is returned as ``None`` and the score stage falls back to
    the classical risk / the Phase-1 behavior rule so generation still completes.
    """
    status_map: Dict[str, str] = {}   # {pair_id -> label_status} for confirmed pairs
    family_map: Dict[str, str] = {}   # {pair_id -> behavior_family} for confirmed targets
    players: Dict[str, tuple] = {}    # {pair_id -> (player_a, player_b)} for confirmed pairs

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

    # --- Train the LEARNED risk model (HALT-safe fallback). ---
    risk_model: Optional[LearnedRiskModel] = None
    has_pos = any(status_map.get(str(fs.pair_id)) == CONFIRMED_TARGET for fs in train_fs)
    has_neg = any(status_map.get(str(fs.pair_id)) == CONFIRMED_NON_TARGET for fs in train_fs)
    if train_fs and has_pos and has_neg:
        try:
            m = LearnedRiskModel(config=config)
            m.fit(train_fs, status_map, eda_artifact=artifact)
            risk_model = m
        except Exception:
            risk_model = None

    # --- Train the behavior classifier (HALT-safe fallback; identical to Phase-2). ---
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

    return risk_model, classifier, status_map


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_learned_pipeline_fast(
    config: Optional[PipelineConfig] = None,
    *,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    limit_pairs: Optional[int] = None,
    parallel: bool = False,
    n_workers: Optional[int] = None,
) -> Path:
    """Generate the FULL evaluation submission with the LEARNED risk combiner (vectorized scoring).

    Identical to :func:`poker_collusion.pipeline_phase2_fast.run_phase2_pipeline_fast` except the
    ``risk_score`` is produced by :class:`LearnedRiskModel` (fallback: the classical risk per pair).

    Args:
        config: Pipeline config (defaults to :func:`get_config`).
        data_loader: Optional pre-built loader.
        eda_artifact: Optional pre-computed artifact.
        limit_pairs: Smoke run — score only the first N evaluation pairs (the rest get the writer's
            safe default row so the emitted file is still a FULL, schema-valid submission).
        parallel: Route the independent per-pair eval feature/score/evidence build through the
            deterministic multiprocess driver (Job-1 parallel path). The learned risk + classifier
            still run as batch steps in the parent, so the emitted submission is identical to the
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

    # ---- (1) TRAIN STAGE: fit learned risk model + behavior classifier ONCE on confirmed dev. -- #
    risk_model, classifier, _status_map = _train_models(
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
    # The model-free per-pair build (signals -> feature set -> evidence -> classical risk/behavior)
    # is the bottleneck; route it through the shared driver. ``parallel=True`` uses the Job-1
    # multiprocess path (byte-identical, contiguous chunks preserve order); the risk MODEL and the
    # behavior CLASSIFIER remain batch steps in the parent below.
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
    # Iterate pair_rows to preserve the deterministic pair order.
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

    # ---- (4) Risk: batch LEARNED score over all scoreable pairs (fallback to classical). ----- #
    learned_risk: Dict[str, float] = {}
    if risk_model is not None and scoreable_fs:
        try:
            learned_risk = risk_model.score_pairs(scoreable_fs, eda_artifact=artifact)
        except Exception:
            learned_risk = {}

    risk_by_pair: Dict[str, float] = {}
    classical_risk: Dict[str, float] = {}
    for pair_id in feature_by_pair:
        # The driver already computed the deterministic classical risk (fallback source).
        classical = float(scored[pair_id].classical_risk)
        classical_risk[pair_id] = classical
        risk = learned_risk.get(pair_id, classical)
        risk_by_pair[pair_id] = float(min(max(risk, 0.0), 1.0))

    # ---- (5) Behavior: classifier label routed on the risk map (fallback to Phase-1 rule). --- #
    label_by_pair: Dict[str, str] = {}
    if classifier is not None and scoreable_fs:
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

    # ---- (6) Assemble predictions (evidence IDENTICAL to Phase-1/Phase-2). ------------------- #
    predictions: List[PairPrediction] = []
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        fs = feature_by_pair.get(pair_id)

        if fs is None:
            # No evaluation shared hands -> documented pool-prior default row (like Phase-1/2).
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

        # Evidence: SAME selection as Phase-1/Phase-2 (only risk + behavior change).
        evidence_slots = evidence_by_pair[pair_id]

        risk_source = "learned_risk" if pair_id in learned_risk else "classical_fallback"
        predictions.append(
            PairPrediction(
                pair_id=pair_id,
                risk_score=float(risk),
                predicted_behavior=behavior,
                evidence=list(evidence_slots),
                reasons=[f"learned risk={risk_source}"],
            )
        )

    # ---- (7) Route into the writer (self-validating, atomic, deterministic row order). ------- #
    return write_submission(
        predictions,
        sample_submission=sample_submission,
        config=cfg,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m poker_collusion.pipeline_learned_fast [--limit-pairs N]``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m poker_collusion.pipeline_learned_fast",
        description="Generate the FULL poker-collusion submission with the LEARNED risk combiner.",
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

    path = run_learned_pipeline_fast(
        limit_pairs=args.limit_pairs,
        parallel=args.parallel,
        n_workers=args.n_workers,
    )
    print(f"submission written to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
