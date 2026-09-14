"""Phase-1 baseline pipeline: the single, documented end-to-end entry point (task 14.1, Req 11.1).

This module wires the whole Phase-1 classical baseline into ONE entry function,
:func:`run_baseline_pipeline`, runnable identically on Windows PowerShell
(``python -m poker_collusion.pipeline``) and inside the Kaggle notebook (paths auto-detected by
:class:`~poker_collusion.config.PipelineConfig` via
:func:`~poker_collusion.config.is_kaggle_environment`). It composes, in order:

    Data_Loader                              (poker_collusion.io.data_loader.DataLoader)
      -> EDA / label audit                   (poker_collusion.features.eda.run_eda, or a cached
                                              eda_artifact.json loaded via load_eda_artifact)
      -> per evaluation pair:
           gather evaluation-period shared-hand signals
             (DataLoader.shared_hands(pair).evaluation  ->  compute_pair_signals, pulling each
              hand's ordered public action log via the loader and net_chips from seats)
           -> episodic aggregation + Pair_Feature_Set
             (poker_collusion.features.pair_features.build_pair_feature_set, phase="evaluation")
           -> classical detectors -> risk_score
             (poker_collusion.models.classical.score_pair -> ClassicalScore)
           -> evidence retrieval
             (poker_collusion.evidence.retriever.select_evidence over the eval-period signals)
           -> assemble a PairPrediction (risk_score, predicted_behavior, evidence, reasons)
      -> Submission_Writer                   (poker_collusion.submission.writer.write_submission)

The classical ``risk_score`` and the retrieved evidence are routed straight into the writer.

Behavior-label placeholder rule (Phase 1 only; the full classifier arrives in Phase 2)
--------------------------------------------------------------------------------------
Phase 1 has no trained behavior classifier, so ``predicted_behavior`` is assigned by a SIMPLE,
DOCUMENTED THRESHOLD PLACEHOLDER:

    * If ``risk_score < BEHAVIOR_RISK_THRESHOLD`` -> ``"none"`` (the pair is not flagged as
      coordinated, so no coordination family is claimed).
    * Otherwise, pick the disclosed family whose per-family detector/confounder signal is
      STRONGEST for the pair, using the pair's evaluation-period behavior-action-flag counts as
      the primary vote and the confounder-separating features as the tie-break:
          directed_transfer     <- flag_directed_transfer count / conf_directedness_contrast
          soft_play             <- flag_soft_play count       / conf_avoidance_gap
          coordinated_isolation <- flag_coordinated_isolation / conf_joint_isolation_rate
      When a flagged pair shows NO disclosed-family signal at all (all family signals zero) it
      routes to the documented default family :data:`DEFAULT_FLAGGED_BEHAVIOR`
      (``"other_coordination"`` — the behavior-agnostic catch-all backbone, free under Behavior
      MAP which excludes it). Ties among disclosed families break by the fixed disclosed-family
      order for determinism.

This is explicitly a PLACEHOLDER. It never claims calibration quality; Phase 2's
``behavior_classifier`` replaces it.

Keeping ``actions.parquet`` from being loaded whole (Req 1.6)
-------------------------------------------------------------
``actions.parquet`` is 18,609,028 rows and is NEVER materialised in full. The pipeline:

    1. computes the UNION of evaluation-period shared hand ids across the (optionally limited)
       set of evaluation pairs — a small set;
    2. makes ONE streaming pass over ``DataLoader.iter_actions()`` (parquet-row-group-sized,
       column-projected batches) filtering each batch to that hand-id set and grouping the kept
       rows by ``hand_id``;
    3. hands each pair only its own hands' action frames from that in-memory group.

So the file is read once, in bounded batches, and only the needed hands' rows are retained. A
per-hand fallback (:meth:`DataLoader.iter_hand_actions`) is available but the single-pass grouping
is used for the end-to-end run so cost is O(file scan) once, not O(pairs x file scan).

Determinism (Req 11.2)
----------------------
Given identical inputs and config/seeds the pipeline writes a byte-identical ``submission.csv``:
signals, aggregation, detectors, and evidence selection are all deterministic; the submission row
order follows ``sample_submission.csv`` unchanged; the writer sorts evidence deterministically.
A ``limit_pairs`` smoke argument processes only the first N evaluation pairs (in the file's order)
while still emitting a full, schema-valid submission (unprocessed pairs get the writer's safe
default row — pool-prior-style ``none`` / ``NO_EVIDENCE``).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from poker_collusion.config import (
    DISCLOSED_FAMILY_NAMES,
    NO_EVIDENCE,
    PipelineConfig,
    get_config,
)
from poker_collusion.evidence.retriever import MAX_EVIDENCE, select_evidence
from poker_collusion.features.aggregation import prior_from_artifact
from poker_collusion.features.eda import EdaArtifact, load_eda_artifact, run_eda
from poker_collusion.features.pair_features import build_pair_feature_set
from poker_collusion.features.signals import compute_pair_signals
from poker_collusion.io import DataLoader
from poker_collusion.models.classical import score_pair
from poker_collusion.submission.writer import write_submission
from poker_collusion.types import HandSignal, PairPrediction

__all__ = [
    "BEHAVIOR_RISK_THRESHOLD",
    "DEFAULT_FLAGGED_BEHAVIOR",
    "assign_behavior_label",
    "run_baseline_pipeline",
    "main",
]

#: Risk threshold below which the behavior-label placeholder assigns ``"none"`` (Phase-1 only).
#: A documented, fixed constant; Phase 2's classifier replaces the whole rule.
BEHAVIOR_RISK_THRESHOLD: float = 0.5

#: Family a flagged-but-untemplated pair routes to (documented default; excluded from Behavior MAP).
DEFAULT_FLAGGED_BEHAVIOR: str = "other_coordination"

# Per-family confounder feature that acts as the disclosed-family tie-break signal.
_FAMILY_CONF_FEATURE: Dict[str, str] = {
    "directed_transfer": "conf_directedness_contrast",
    "soft_play": "conf_avoidance_gap",
    "coordinated_isolation": "conf_joint_isolation_rate",
}


# --------------------------------------------------------------------------- #
# Behavior-label placeholder (Phase 1)
# --------------------------------------------------------------------------- #
def assign_behavior_label(
    risk_score: float,
    signals: Sequence[HandSignal],
    features: Mapping[str, float],
    *,
    threshold: float = BEHAVIOR_RISK_THRESHOLD,
) -> str:
    """Assign ``predicted_behavior`` via the Phase-1 threshold placeholder (documented above).

    Below ``threshold`` -> ``"none"``. Otherwise vote the disclosed family with the most
    evaluation-period behavior-action flags, tie-broken by that family's confounder-separating
    feature; a flagged pair with no disclosed-family signal routes to
    :data:`DEFAULT_FLAGGED_BEHAVIOR`. Deterministic (fixed family order breaks any remaining tie).

    Args:
        risk_score: The classical baseline risk in [0, 1].
        signals: The pair's evaluation-period :class:`HandSignal` list (source of the flag votes).
        features: The pair's assembled feature map (source of the confounder tie-break values).
        threshold: Risk threshold for the ``none`` cut (defaults to :data:`BEHAVIOR_RISK_THRESHOLD`).

    Returns:
        One of ``{"none", "directed_transfer", "soft_play", "coordinated_isolation",
        "other_coordination"}``.
    """
    if risk_score < threshold:
        return "none"

    # Primary vote: per-family behavior-action-flag counts over the pair's eval-period hands.
    flag_counts: Dict[str, int] = {fam: 0 for fam in DISCLOSED_FAMILY_NAMES}
    for sig in signals:
        flags = sig.behavior_action_flags or {}
        for fam in DISCLOSED_FAMILY_NAMES:
            if bool(flags.get(fam, False)):
                flag_counts[fam] += 1

    # Tie-break signal: the family's confounder-separating feature magnitude.
    def _family_key(fam: str) -> Tuple[int, float]:
        conf_feat = _FAMILY_CONF_FEATURE.get(fam, "")
        conf_val = float(features.get(conf_feat, 0.0)) if conf_feat else 0.0
        return (flag_counts[fam], conf_val)

    best_family = max(DISCLOSED_FAMILY_NAMES, key=_family_key)
    best_count, best_conf = _family_key(best_family)

    # A flagged pair with no disclosed-family signal at all -> documented catch-all default.
    if best_count <= 0 and best_conf <= 0.0:
        return DEFAULT_FLAGGED_BEHAVIOR
    return best_family


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _canonical_players(player_1: object, player_2: object) -> Tuple[str, str]:
    """Return the two player ids in canonical (ascending) order: (min, max)."""
    a, b = str(player_1), str(player_2)
    return (a, b) if a <= b else (b, a)


def _net_chips_by_hand(
    seats: pd.DataFrame,
    hand_ids: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Build ``hand_id -> {player_id -> net_chips}`` restricted to ``hand_ids`` (small, projected).

    Reads only the already-loaded seats frame; never touches actions. Restricting to the needed
    evaluation hands keeps the map tiny even on the real data.
    """
    wanted = set(str(h) for h in hand_ids)
    if not wanted or seats is None or len(seats) == 0:
        return {}
    sub = seats[seats["hand_id"].astype(str).isin(wanted)]
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    has_net = "net_chips" in sub.columns
    for row in sub.itertuples(index=False):
        hid = str(getattr(row, "hand_id"))
        pid = str(getattr(row, "player_id"))
        net = float(getattr(row, "net_chips")) if has_net else 0.0
        out[hid][pid] = net
    return dict(out)


def _actions_for_hands(
    loader: DataLoader,
    hand_ids: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Return ``hand_id -> ordered action frame`` for exactly ``hand_ids`` in ONE streaming pass.

    Makes a single pass over ``DataLoader.iter_actions()`` (column-projected, row-group-batched)
    filtering each batch to ``hand_ids`` and concatenating the kept rows per hand. ``actions.parquet``
    is therefore scanned once in bounded batches and NEVER materialised whole (Req 1.6). Each hand's
    frame is sorted by ``action_no`` so consumers see decision order.
    """
    wanted = set(str(h) for h in hand_ids)
    if not wanted:
        return {}
    collected: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    for batch in loader.iter_actions():
        if "hand_id" not in batch.columns or batch.empty:
            continue
        keep = batch[batch["hand_id"].astype(str).isin(wanted)]
        if keep.empty:
            continue
        for hid, grp in keep.groupby(keep["hand_id"].astype(str), sort=False):
            collected[hid].append(grp)
    out: Dict[str, pd.DataFrame] = {}
    for hid, frames in collected.items():
        frame = pd.concat(frames, ignore_index=True)
        if "action_no" in frame.columns:
            frame = frame.sort_values("action_no", kind="stable").reset_index(drop=True)
        out[hid] = frame
    return out


def _hands_meta_for(
    hands: pd.DataFrame,
    hand_ids: Sequence[str],
) -> pd.DataFrame:
    """Return a ``[hand_id, phase, big_blind]`` frame for the given evaluation hand ids.

    ``big_blind`` is included when present so ``compute_pair_signals`` normalises value flow to
    big blinds; absent, value flow falls back to raw chips (documented in signals).
    """
    wanted = set(str(h) for h in hand_ids)
    cols = ["hand_id", "phase"] + (["big_blind"] if "big_blind" in hands.columns else [])
    sub = hands[hands["hand_id"].astype(str).isin(wanted)][cols].copy()
    return sub.reset_index(drop=True)


def _obtain_eda_artifact(
    loader: DataLoader,
    config: PipelineConfig,
    eda_artifact: Optional[EdaArtifact],
) -> EdaArtifact:
    """Resolve the EDA/threshold artifact: explicit -> cached file -> freshly computed.

    Feature engineering and the classical scorer read their calibrated constants (pool prior,
    min-shared-hands) ONLY from this artifact (Req 2.6). If an explicit artifact is passed it wins;
    else a cached ``eda_artifact.json`` at ``config.eda_artifact_path`` is loaded; else it is
    computed on the fly via :func:`run_eda` (column-projected; never touches actions).
    """
    if eda_artifact is not None:
        return eda_artifact
    try:
        return load_eda_artifact(config=config)
    except (FileNotFoundError, OSError, ValueError):
        return run_eda(loader=loader, config=config)


# --------------------------------------------------------------------------- #
# Single entry point
# --------------------------------------------------------------------------- #
def run_baseline_pipeline(
    config: Optional[PipelineConfig] = None,
    *,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    limit_pairs: Optional[int] = None,
) -> Path:
    """Run the Phase-1 classical baseline end-to-end and write ``submission.csv``.

    Wires Data_Loader -> EDA/audit -> per-pair signals -> aggregation/Pair_Feature_Set ->
    classical detectors -> evidence retriever -> Submission_Writer (see the module docstring for
    the full flow and the behavior-label placeholder rule).

    Args:
        config: Pipeline configuration (paths/seeds/versions). Defaults to :func:`get_config`,
            which auto-detects the Kaggle vs local environment.
        data_loader: Optional pre-built :class:`DataLoader` (mainly for tests / custom inputs);
            defaults to ``DataLoader(config=config)``.
        eda_artifact: Optional pre-computed :class:`EdaArtifact`; when omitted a cached
            ``eda_artifact.json`` is loaded, else one is computed on the fly.
        limit_pairs: Optional smoke cap — process only the first N evaluation pairs (in the
            ``evaluation_pairs.csv`` order). Unprocessed pairs still appear in the submission with
            the writer's safe default row, so the file remains schema-valid.

    Returns:
        The :class:`~pathlib.Path` the ``submission.csv`` was written to.
    """
    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)

    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)
    pool_prior = prior_from_artifact(artifact, default=0.0)

    evaluation_pairs = loader.load_evaluation_pairs()
    sample_submission = loader.load_sample_submission()
    seats = loader.load_seats()
    hands = loader.load_hands()

    if limit_pairs is not None:
        evaluation_pairs = evaluation_pairs.head(int(limit_pairs)).copy()

    # ---- resolve each pair's evaluation-period shared hands (column-projected; no actions) ---- #
    pair_rows: List[dict] = []
    all_eval_hand_ids: set = set()
    for row in evaluation_pairs.itertuples(index=False):
        pair_id = str(getattr(row, "pair_id"))
        player_a, player_b = _canonical_players(
            getattr(row, "player_1"), getattr(row, "player_2")
        )
        shared = loader.shared_hands((player_a, player_b))
        eval_hands = [str(h) for h in shared.evaluation]
        all_eval_hand_ids.update(eval_hands)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "player_a": player_a,
                "player_b": player_b,
                "eval_hands": eval_hands,
            }
        )

    # ---- ONE streaming pass over actions for exactly the needed eval hands (Req 1.6) --------- #
    actions_by_hand = _actions_for_hands(loader, sorted(all_eval_hand_ids))
    net_by_hand = _net_chips_by_hand(seats, sorted(all_eval_hand_ids))
    eval_hands_meta = _hands_meta_for(hands, sorted(all_eval_hand_ids))
    meta_by_hand = {
        str(r.hand_id): r for r in eval_hands_meta.itertuples(index=False)
    }

    # ---- per-pair: signals -> features -> risk -> evidence -> PairPrediction ----------------- #
    predictions: List[PairPrediction] = []
    for pr in pair_rows:
        pair_id = pr["pair_id"]
        player_a = pr["player_a"]
        player_b = pr["player_b"]
        eval_hands = pr["eval_hands"]

        if not eval_hands:
            # No evaluation shared hands -> documented default row (Req 6.4 default risk = pool
            # prior, behavior "none", all evidence NO_EVIDENCE).
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

        # Assemble this pair's per-hand inputs from the shared single-pass caches.
        pair_meta = eval_hands_meta[eval_hands_meta["hand_id"].astype(str).isin(set(eval_hands))]
        pair_actions = {h: actions_by_hand.get(h, pd.DataFrame()) for h in eval_hands}
        pair_nets = {h: net_by_hand.get(h, {}) for h in eval_hands}

        signals = compute_pair_signals(
            pair_id=pair_id,
            player_a=player_a,
            player_b=player_b,
            hands_meta=pair_meta,
            actions_by_hand=pair_actions,
            net_by_hand=pair_nets,
        )

        feature_set = build_pair_feature_set(
            pair_id=pair_id,
            signals=signals,
            player_a=player_a,
            player_b=player_b,
            phase="evaluation",
            eda_artifact=artifact,
            config=cfg,
        )

        classical = score_pair(feature_set, eda_artifact=artifact)

        evidence = select_evidence(pair_id, signals)

        behavior = assign_behavior_label(
            classical.risk_score, signals, feature_set.features
        )

        predictions.append(
            PairPrediction(
                pair_id=pair_id,
                risk_score=float(classical.risk_score),
                predicted_behavior=behavior,
                evidence=list(evidence.slots),
                reasons=list(classical.reasons),
            )
        )

    # ---- route risk_score + evidence into the Submission_Writer (self-validating, atomic) ---- #
    return write_submission(
        predictions,
        sample_submission=sample_submission,
        config=cfg,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m poker_collusion.pipeline`` (Windows PowerShell / Kaggle).

    Parses an optional ``--limit-pairs N`` smoke flag, runs :func:`run_baseline_pipeline` with the
    environment-detected default config, and prints the written submission path. Returns a process
    exit code (0 on success).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m poker_collusion.pipeline",
        description="Run the Phase-1 classical baseline poker-collusion pipeline end-to-end.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Smoke run: process only the first N evaluation pairs (rest get the default row).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = run_baseline_pipeline(limit_pairs=args.limit_pairs)
    print(f"submission written to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
