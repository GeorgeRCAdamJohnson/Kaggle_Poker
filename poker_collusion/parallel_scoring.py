"""Deterministic parallel per-pair scoring for the fast submission generators.

The fast generators (:mod:`poker_collusion.pipeline_fast`,
:mod:`poker_collusion.pipeline_phase2_fast`, :mod:`poker_collusion.pipeline_learned_fast`) share
one bottleneck: after the shared per-hand caches are built ONCE (a single actions pass +
``net_by_hand`` + ``meta_record_by_hand``), a per-pair loop over ~112,540 evaluation pairs does the
model-free work

    compute_pair_signals -> build_pair_feature_set -> score_pair (classical)
        -> select_evidence -> assign_behavior_label (Phase-1 rule)

That per-pair work is INDEPENDENT across pairs and dominates wall-clock. This module runs it across
processes while guaranteeing the output is BYTE-IDENTICAL to the sequential path.

Determinism contract (protects reproducibility Property 16)
-----------------------------------------------------------
* The math is NOT reimplemented — the workers import and call the exact same functions the
  sequential generators use (``compute_pair_signals`` / ``build_pair_feature_set`` /
  ``score_pair`` / ``select_evidence`` / ``assign_behavior_label``).
* Ordering is trivially preserved: ``pair_rows`` is partitioned into N *contiguous* slices, each
  worker returns its slice's results IN INPUT ORDER, and the driver concatenates the slices back in
  chunk order. The concatenation is therefore the same sequence the sequential loop produces.
* No per-worker RNG: the classical score and the Phase-1 behavior rule are deterministic; workers
  receive read-only copies of the caches. Two runs on identical inputs produce identical results.

Windows / spawn safety
----------------------
The worker (:func:`_score_chunk`) is MODULE-LEVEL (picklable). Every argument passed to it is
picklable (plain dicts / lists / dataclasses / a small pandas frame is never sent — only per-hand
record dicts). The driver is called from inside the generators which are themselves guarded by
``if __name__ == "__main__"`` at their CLI entry points, so spawn re-import is safe.

Cache-sharing tradeoff (pickle size)
------------------------------------
Pickling the giant shared ``rows_by_hand`` / ``net_by_hand`` / ``meta_record_by_hand`` maps to every
worker would dominate the cost the parallelism is meant to save. Because each chunk is a contiguous
slice of ``pair_rows``, a worker only needs the cache entries for the hand ids ITS pairs reference.
The driver therefore builds a per-chunk SUB-MAP in the parent (subset the three maps to just the
union of that chunk's ``eval_hands``) and ships only those small maps. This keeps pickle payloads
bounded by the chunk's own hands, not the whole evaluation set, while keeping each worker fully
independent and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from poker_collusion.config import PipelineConfig
from poker_collusion.evidence.retriever import select_evidence
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.features.pair_features import build_pair_feature_set
from poker_collusion.features.signals import compute_pair_signals
from poker_collusion.models.classical import score_pair
from poker_collusion.pipeline import assign_behavior_label
from poker_collusion.types import HandSignal, PairFeatureSet

__all__ = [
    "PairScoreResult",
    "score_pairs_parallel",
    "score_pairs_sequential",
    "resolve_worker_count",
]

#: The per-pair signal path always uses pre-parsed ``rows_by_hand``; the action FRAME map is empty.
_EMPTY_ACTIONS: Dict[str, "pd.DataFrame"] = {}


@dataclass
class PairScoreResult:
    """The model-free per-pair intermediate every fast generator consumes.

    Fields:
        pair_id: The evaluation pair id.
        feature_set: The pair's :class:`PairFeatureSet` (``phase="evaluation"``).
        signals: The pair's evaluation-period :class:`HandSignal` list (for evidence + behavior).
        evidence_slots: The 5 evidence slot strings (identical to the sequential ``select_evidence``).
        classical_risk: The deterministic classical ``score_pair`` risk in ``[0, 1]``.
        classical_reasons: The classical scorer's per-detector reason strings (Phase-1 ``reasons``).
        phase1_behavior: The Phase-1 ``assign_behavior_label`` string (used directly by Phase-1;
            used as the fallback behavior by Phase-2 / learned when the classifier is unavailable).

    The learned / PU generators overwrite risk (batch model) and behavior (classifier) in the
    parent; Phase-1 uses ``classical_risk`` + ``phase1_behavior`` verbatim. Evidence is identical
    across all three.
    """

    pair_id: str
    feature_set: PairFeatureSet
    signals: List[HandSignal]
    evidence_slots: List[str]
    classical_risk: float
    classical_reasons: List[str]
    phase1_behavior: str


def _score_one_pair(
    *,
    pair_id: str,
    player_a: str,
    player_b: str,
    eval_hands: Sequence[str],
    rows_by_hand: Dict[str, List[dict]],
    net_by_hand: Dict[str, Dict[str, float]],
    meta_record_by_hand: Dict[str, dict],
    meta_cols: Sequence[str],
    artifact: Optional[EdaArtifact],
    config: PipelineConfig,
) -> PairScoreResult:
    """Run the exact sequential per-pair flow for ONE pair (shared by both code paths).

    This is the single source of truth for the per-pair math so the parallel and sequential paths
    call IDENTICAL code. ``eval_hands`` is assumed non-empty (callers handle the pool-prior default
    row for pairs with no shared eval hands, exactly as the sequential generators do).
    """
    pair_meta = pd.DataFrame(
        [meta_record_by_hand[h] for h in eval_hands if h in meta_record_by_hand],
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
        phase="evaluation",
        eda_artifact=artifact,
        config=config,
    )
    classical = score_pair(feature_set, eda_artifact=artifact)
    evidence = select_evidence(pair_id, signals)
    behavior = assign_behavior_label(classical.risk_score, signals, feature_set.features)
    return PairScoreResult(
        pair_id=pair_id,
        feature_set=feature_set,
        signals=list(signals),
        evidence_slots=list(evidence.slots),
        classical_risk=float(classical.risk_score),
        classical_reasons=list(classical.reasons),
        phase1_behavior=behavior,
    )


def _score_chunk(
    chunk: List[dict],
    rows_by_hand: Dict[str, List[dict]],
    net_by_hand: Dict[str, Dict[str, float]],
    meta_record_by_hand: Dict[str, dict],
    meta_cols: List[str],
    artifact: Optional[EdaArtifact],
    config: PipelineConfig,
) -> List[PairScoreResult]:
    """MODULE-LEVEL worker: score a contiguous chunk of pair rows IN INPUT ORDER (spawn-safe).

    Each ``chunk`` element is a pair-row dict ``{pair_id, player_a, player_b, eval_hands}``. The
    three cache maps are the per-chunk SUB-MAPS built by the driver (only this chunk's hands). Pairs
    with no ``eval_hands`` are skipped here — the parent emits their pool-prior default row — so the
    returned list contains one :class:`PairScoreResult` per SCOREABLE pair in the chunk, in order.
    """
    out: List[PairScoreResult] = []
    for pr in chunk:
        eval_hands = pr["eval_hands"]
        if not eval_hands:
            continue
        out.append(
            _score_one_pair(
                pair_id=pr["pair_id"],
                player_a=pr["player_a"],
                player_b=pr["player_b"],
                eval_hands=eval_hands,
                rows_by_hand=rows_by_hand,
                net_by_hand=net_by_hand,
                meta_record_by_hand=meta_record_by_hand,
                meta_cols=meta_cols,
                artifact=artifact,
                config=config,
            )
        )
    return out


def score_pairs_sequential(
    pair_rows: Sequence[dict],
    *,
    rows_by_hand: Dict[str, List[dict]],
    net_by_hand: Dict[str, Dict[str, float]],
    meta_record_by_hand: Dict[str, dict],
    meta_cols: Sequence[str],
    artifact: Optional[EdaArtifact],
    config: PipelineConfig,
) -> Dict[str, PairScoreResult]:
    """Score every SCOREABLE pair sequentially -> ``{pair_id -> PairScoreResult}`` (reference path).

    Pairs with no shared eval hands are omitted (the caller emits their pool-prior default row).
    Used both as the ``parallel=False`` path and by the tests as the byte-identical baseline.
    """
    results: Dict[str, PairScoreResult] = {}
    for pr in pair_rows:
        if not pr["eval_hands"]:
            continue
        res = _score_one_pair(
            pair_id=pr["pair_id"],
            player_a=pr["player_a"],
            player_b=pr["player_b"],
            eval_hands=pr["eval_hands"],
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=config,
        )
        results[res.pair_id] = res
    return results


def resolve_worker_count(n_workers: Optional[int]) -> int:
    """Resolve the worker count: explicit ``n_workers`` (>=1) or ``min(cpu_count, 8)``."""
    import os

    if n_workers is not None:
        return max(int(n_workers), 1)
    cpu = os.cpu_count() or 1
    return max(1, min(int(cpu), 8))


def _chunk_bounds(n_items: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Contiguous ``[start, end)`` slices splitting ``n_items`` into (up to) ``n_chunks`` blocks.

    Blocks are as even as possible and always contiguous, so concatenating their per-block results
    reproduces the original order exactly.
    """
    if n_items <= 0:
        return []
    k = max(1, min(n_chunks, n_items))
    base, extra = divmod(n_items, k)
    bounds: List[Tuple[int, int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _subset_maps_for_chunk(
    chunk: Sequence[dict],
    rows_by_hand: Dict[str, List[dict]],
    net_by_hand: Dict[str, Dict[str, float]],
    meta_record_by_hand: Dict[str, dict],
) -> Tuple[Dict[str, List[dict]], Dict[str, Dict[str, float]], Dict[str, dict]]:
    """Build per-chunk SUB-MAPS restricted to the hand ids this chunk's pairs reference.

    Minimises the pickle payload shipped to each worker (bounded by the chunk's own hands, not the
    whole evaluation set) while keeping the worker fully independent. Keying is by ``str(hand_id)``
    exactly as the parent maps are keyed, so lookups inside the worker are identical.
    """
    wanted: set = set()
    for pr in chunk:
        for h in pr["eval_hands"]:
            wanted.add(h)
    sub_rows = {h: rows_by_hand[h] for h in wanted if h in rows_by_hand}
    sub_net = {h: net_by_hand[h] for h in wanted if h in net_by_hand}
    sub_meta = {h: meta_record_by_hand[h] for h in wanted if h in meta_record_by_hand}
    return sub_rows, sub_net, sub_meta


def score_pairs_parallel(
    pair_rows: Sequence[dict],
    *,
    rows_by_hand: Dict[str, List[dict]],
    net_by_hand: Dict[str, Dict[str, float]],
    meta_record_by_hand: Dict[str, dict],
    meta_cols: Sequence[str],
    artifact: Optional[EdaArtifact],
    config: PipelineConfig,
    n_workers: Optional[int] = None,
) -> Dict[str, PairScoreResult]:
    """Score every SCOREABLE pair across processes -> ``{pair_id -> PairScoreResult}``.

    Deterministic and byte-identical to :func:`score_pairs_sequential`: ``pair_rows`` is split into
    contiguous chunks, each worker runs the SAME per-pair code on a per-chunk sub-map, and the
    per-chunk results are concatenated in chunk order (so the merged sequence is the sequential
    order). Falls back to the sequential path when only one worker is resolved or on very small
    inputs (spawning would cost more than it saves).

    Args:
        pair_rows: Ordered pair-row dicts ``{pair_id, player_a, player_b, eval_hands}``.
        rows_by_hand / net_by_hand / meta_record_by_hand: The shared per-hand caches (whole eval
            set); the driver subsets them per chunk before dispatch.
        meta_cols: The fixed meta column order for the tiny per-pair meta frame.
        artifact / config: Injected read-only artifact + config (picklable dataclasses).
        n_workers: Worker count; ``None`` -> ``min(cpu_count, 8)``.

    Returns:
        ``{pair_id -> PairScoreResult}`` for every scoreable pair (pairs with no shared eval hands
        are omitted; the caller emits their pool-prior default row).
    """
    workers = resolve_worker_count(n_workers)
    n_pairs = len(pair_rows)

    # Small inputs / single worker: the sequential path is both faster and trivially identical.
    if workers <= 1 or n_pairs < 2:
        return score_pairs_sequential(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=config,
        )

    from concurrent.futures import ProcessPoolExecutor

    bounds = _chunk_bounds(n_pairs, workers)
    chunks: List[List[dict]] = [list(pair_rows[s:e]) for (s, e) in bounds]
    meta_cols_list = list(meta_cols)

    # Build each chunk's sub-maps IN THE PARENT (bounded pickle payload per worker).
    payloads = []
    for chunk in chunks:
        sub_rows, sub_net, sub_meta = _subset_maps_for_chunk(
            chunk, rows_by_hand, net_by_hand, meta_record_by_hand
        )
        payloads.append((chunk, sub_rows, sub_net, sub_meta, meta_cols_list, artifact, config))

    # Dispatch; collect per-chunk result lists IN CHUNK ORDER (executor.map preserves order).
    results: Dict[str, PairScoreResult] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk_results in pool.map(_score_chunk_star, payloads):
                for res in chunk_results:  # already in input order within the chunk
                    results[res.pair_id] = res
    except Exception:
        # HALT-safe: any pool/pickle failure falls back to the deterministic sequential path so
        # generation always completes with a byte-identical result.
        return score_pairs_sequential(
            pair_rows,
            rows_by_hand=rows_by_hand,
            net_by_hand=net_by_hand,
            meta_record_by_hand=meta_record_by_hand,
            meta_cols=meta_cols,
            artifact=artifact,
            config=config,
        )
    return results


def _score_chunk_star(args: tuple) -> List[PairScoreResult]:
    """MODULE-LEVEL unpacker so ``ProcessPoolExecutor.map`` can pass one picklable tuple per task."""
    chunk, sub_rows, sub_net, sub_meta, meta_cols_list, artifact, config = args
    return _score_chunk(
        chunk, sub_rows, sub_net, sub_meta, meta_cols_list, artifact, config
    )
