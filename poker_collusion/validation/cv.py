"""Group-by-pool, family-stratified, PU-correct local cross-validation harness.

This module builds the local CV harness that mirrors the private leaderboard as
closely as possible so every downstream decision is gated on local CV rather than
the noisy public leaderboard (Design Principle 1; Requirements 10.1, 10.4, 10.5,
12.4).

Three properties drive the design, in priority order:

1. **Group by pool (hard constraint).** A pool is a ``hands.table_id`` — 400
   tables, each with exactly 30 disjoint players and 5,000 hands. A player (and
   therefore a labelled pair) belongs to exactly one pool. Folds split at the pool
   level so no pool ever appears in both train and validation; that is what
   prevents pair/player leakage across folds. This constraint is never violated.

2. **Family stratification (approximate, subordinate to the group constraint).**
   Positive pairs carry a ``behavior_family`` (``directed_transfer``,
   ``soft_play``, ``coordinated_isolation``). Because whole pools move together,
   perfect per-fold family balance is generally impossible; the splitter greedily
   assigns each pool to the fold that currently needs its positives most, so each
   family is spread across folds as evenly as the grouping permits. The
   tie-breaking is documented on :func:`make_pool_folds`.

3. **PU-correct scoring.** Development labels are Positive-Unlabelled. Only
   ``confirmed_target`` (positive) and ``confirmed_non_target`` (negative) rows are
   trustworthy; every pair absent from the label table is UNKNOWN, *not* negative.
   Each validation fold builds its solution frame from confirmed labels only —
   unknown pairs are never materialised as negatives in the frame the fold metric
   scores. See :func:`build_fold_solution`.

The harness never loads ``actions.parquet``. Pool membership is derived by joining
labelled pairs' players to ``seats``/``hands`` with column projection only; callers
may also inject a precomputed pool map (used by the fast synthetic tests).

Public API
----------
* :func:`derive_pool_map` — map each labelled ``pair_id`` to its pool from the real
  ``seats``/``hands`` files (column-projected), or from injected frames.
* :func:`make_pool_folds` — deterministic group-by-pool, family-stratified fold
  assignment.
* :func:`build_fold_solution` — PU-correct per-fold solution frame shaped for
  :func:`poker_collusion.metric.production_metric.score_components`.
* :func:`CVHarness` — bundles fold assignments and runs a scoring callback per
  fold, aggregating the three metric components separately plus the combined
  summary (Req 10.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from poker_collusion.config import (
    DISCLOSED_FAMILY_NAMES,
    NO_EVIDENCE,
    PipelineConfig,
    get_config,
)
from poker_collusion.exceptions import SchemaError
from poker_collusion.metric.production_metric import (
    ScoreComponents,
    score_components,
)
from poker_collusion.metric.reference_public_metric import EVIDENCE_COLUMNS

__all__ = [
    "FoldResult",
    "CVReport",
    "Fold",
    "CVHarness",
    "derive_pool_map",
    "make_pool_folds",
    "build_fold_solution",
]

# Label-status tokens as they appear verbatim in development_labels.csv.
CONFIRMED_TARGET = "confirmed_target"
CONFIRMED_NON_TARGET = "confirmed_non_target"

# The behavior written for confirmed negatives in the fold solution frame.
NEGATIVE_BEHAVIOR = "none"


# --------------------------------------------------------------------------- #
# Pool derivation                                                             #
# --------------------------------------------------------------------------- #
def derive_pool_map(
    labels: pd.DataFrame,
    *,
    seats: Optional[pd.DataFrame] = None,
    hands: Optional[pd.DataFrame] = None,
    config: Optional[PipelineConfig] = None,
    player_pool: Optional[Mapping] = None,
) -> Dict[str, object]:
    """Map each labelled ``pair_id`` to the pool its players belong to.

    A pair only ever exists within one pool (both players share exactly one
    ``table_id``), so the pool is well defined. Three input modes are supported,
    in precedence order:

    1. ``player_pool`` — an explicit ``{player_id: pool}`` mapping (used by the
       synthetic tests; no I/O).
    2. ``seats`` + ``hands`` frames — derive ``player -> table_id`` by joining
       ``seats[hand_id, player_id]`` to ``hands[hand_id, table_id]`` (both
       column-projected). ``actions.parquet`` is never touched.
    3. neither — read the real ``seats.parquet`` / ``hands.parquet`` from
       ``config.input_dir`` with only the two/one needed columns each.

    Args:
        labels: Development-label frame. Must expose ``pair_id`` plus the two
            player columns (``player_1``/``player_2``); a pre-split ``player_a``/
            ``player_b`` pair is also accepted.
        seats: Optional projected seats frame (``hand_id``, ``player_id``).
        hands: Optional projected hands frame (``hand_id``, ``table_id``).
        config: Pipeline config for locating the real files (mode 3).
        player_pool: Optional explicit ``{player_id: pool}`` mapping (mode 1).

    Returns:
        ``{pair_id: pool}`` for every labelled pair whose players resolve to a
        pool.

    Raises:
        SchemaError: if ``labels`` lacks the pair/player columns, or a required
            key is missing from ``seats``/``hands``.
    """
    if "pair_id" not in labels.columns:
        raise SchemaError("development_labels", "pair_id")

    p1, p2 = _player_columns(labels)

    if player_pool is None:
        player_pool = _player_pool_from_seats(seats, hands, config)

    pool_map: Dict[str, object] = {}
    for pair_id, a, b in zip(labels["pair_id"], labels[p1], labels[p2]):
        pool_a = player_pool.get(a, player_pool.get(str(a)))
        pool_b = player_pool.get(b, player_pool.get(str(b)))
        pool = pool_a if pool_a is not None else pool_b
        if pool is not None:
            pool_map[str(pair_id)] = pool
    return pool_map


def _player_columns(labels: pd.DataFrame) -> tuple[str, str]:
    """Return the two player-id column names present on ``labels``."""
    if {"player_1", "player_2"}.issubset(labels.columns):
        return "player_1", "player_2"
    if {"player_a", "player_b"}.issubset(labels.columns):
        return "player_a", "player_b"
    raise SchemaError("development_labels", "player_1/player_2")


def _player_pool_from_seats(
    seats: Optional[pd.DataFrame],
    hands: Optional[pd.DataFrame],
    config: Optional[PipelineConfig],
) -> Dict[object, object]:
    """Build ``{player_id: pool}`` from seats+hands (projected reads only)."""
    if seats is None or hands is None:
        config = config or get_config()
        d = Path(config.input_dir)
        if seats is None:
            seats = pd.read_parquet(d / "seats.parquet", columns=["hand_id", "player_id"])
        if hands is None:
            hands = pd.read_parquet(d / "hands.parquet", columns=["hand_id", "table_id"])

    for col in ("hand_id", "player_id"):
        if col not in seats.columns:
            raise SchemaError("seats", col)
    for col in ("hand_id", "table_id"):
        if col not in hands.columns:
            raise SchemaError("hands", col)

    merged = seats[["hand_id", "player_id"]].merge(
        hands[["hand_id", "table_id"]], on="hand_id", how="left"
    )
    merged = merged.dropna(subset=["table_id"]).drop_duplicates(subset=["player_id"])
    return dict(zip(merged["player_id"], merged["table_id"]))


# --------------------------------------------------------------------------- #
# Fold assignment                                                             #
# --------------------------------------------------------------------------- #
def make_pool_folds(
    labels: pd.DataFrame,
    pool_map: Mapping[str, object],
    *,
    n_folds: int = 5,
    seed: Optional[int] = None,
    families: Sequence[str] = DISCLOSED_FAMILY_NAMES,
) -> Dict[object, int]:
    """Assign every pool to one of ``n_folds`` folds, group-safe + family-balanced.

    Algorithm (deterministic given ``seed``):

    1. For each pool, count its confirmed-positive pairs by behavior family.
    2. Shuffle the pool order with a seeded RNG (stable tie randomisation), then
       sort pools by descending total positive count so the "heaviest" pools are
       placed first (classic greedy multiway partition — largest first minimises
       imbalance).
    3. Place each pool into the fold that, after placement, keeps the family
       distribution most even. Concretely we pick the fold minimising a cost that
       sums, over families, the resulting per-fold positive count for that family;
       this pushes each family's positives to spread across folds. Ties break
       toward the fold with the fewest total pools, then the lowest fold index.

    The group constraint is absolute: a pool lands in exactly one fold, so no pool
    can leak across the train/validation boundary. Family balance is best-effort
    and yields to the grouping — with whole pools moving together, perfect balance
    is usually unattainable, which is expected and documented here.

    Args:
        labels: Development-label frame (needs ``pair_id``, ``label_status``,
            ``behavior_family``).
        pool_map: ``{pair_id: pool}`` from :func:`derive_pool_map`.
        n_folds: Number of folds (default 5).
        seed: RNG seed; defaults to ``config.Seeds.cv_split`` for determinism.
        families: Behavior families used for stratification.

    Returns:
        ``{pool: fold_index}`` for every pool that owns at least one labelled pair.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if seed is None:
        seed = get_config().seeds.cv_split

    fam_index = {fam: i for i, fam in enumerate(families)}
    n_fam = len(families)

    # Per-pool positive-by-family counts and total pair counts.
    pool_pos = {}  # pool -> np.array over families
    pool_size = {}  # pool -> total labelled pairs
    status = labels.get("label_status")
    fam_col = labels.get("behavior_family")
    for i, pair_id in enumerate(labels["pair_id"].astype(str)):
        pool = pool_map.get(pair_id)
        if pool is None:
            continue
        pool_size[pool] = pool_size.get(pool, 0) + 1
        if pool not in pool_pos:
            pool_pos[pool] = np.zeros(n_fam, dtype=float)
        is_positive = status is None or status.iloc[i] == CONFIRMED_TARGET
        fam = None if fam_col is None else fam_col.iloc[i]
        if is_positive and fam in fam_index:
            pool_pos[pool][fam_index[fam]] += 1.0

    pools = list(pool_size.keys())
    rng = np.random.default_rng(seed)
    # Seeded shuffle then stable largest-first sort for a deterministic order.
    perm = rng.permutation(len(pools))
    pools = [pools[i] for i in perm]
    pools.sort(key=lambda p: float(pool_pos[p].sum()), reverse=True)

    fold_fam = np.zeros((n_folds, n_fam), dtype=float)
    fold_count = np.zeros(n_folds, dtype=int)
    assignment: Dict[object, int] = {}

    for pool in pools:
        pos = pool_pos[pool]
        best_fold = -1
        best_key: tuple | None = None
        for f in range(n_folds):
            # Cost after placing: sum of squared family loads keeps each family
            # even across folds; squared load penalises piling one family up.
            projected = fold_fam[f] + pos
            cost = float(np.sum(projected * projected))
            key = (cost, fold_count[f], f)
            if best_key is None or key < best_key:
                best_key = key
                best_fold = f
        assignment[pool] = best_fold
        fold_fam[best_fold] += pos
        fold_count[best_fold] += 1

    return assignment


# --------------------------------------------------------------------------- #
# PU-correct fold solution                                                    #
# --------------------------------------------------------------------------- #
def build_fold_solution(
    labels: pd.DataFrame,
    pair_ids: Sequence[str],
    *,
    evidence: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a PU-correct solution frame for the given confirmed pairs.

    The frame is shaped exactly for
    :func:`poker_collusion.metric.production_metric.score_components`: columns
    ``pair_id``, ``risk_score`` (binary ground-truth label), ``predicted_behavior``
    (= ``behavior_family`` for positives, ``none`` for negatives), and the five
    ``evidence_hand_*`` columns.

    **PU handling.** Only confirmed labels populate the frame:

    * ``confirmed_target`` -> ``risk_score = 1``, ``predicted_behavior`` = the
      pair's ``behavior_family``, evidence hands filled from
      ``development_evidence`` (ranked; padded with ``NO_EVIDENCE``).
    * ``confirmed_non_target`` -> ``risk_score = 0``, ``predicted_behavior`` =
      ``none``, all evidence ``NO_EVIDENCE``.

    Pairs absent from the confirmed labels are UNKNOWN and are simply not included
    — the fold never fabricates a negative row for an unknown pair. ``pair_ids``
    is therefore expected to be the confirmed pairs of the fold; any id in
    ``pair_ids`` without a confirmed label is skipped.

    Args:
        labels: Development-label frame (``pair_id``, ``label_status``,
            ``behavior_family``).
        pair_ids: The confirmed pair ids to place in this fold's solution.
        evidence: Optional long-form evidence (``pair_id``, ``hand_id`` and
            optionally ``evidence_rank``); used to fill positive evidence slots.

    Returns:
        A solution DataFrame with one row per confirmed pair.
    """
    for col in ("pair_id", "label_status"):
        if col not in labels.columns:
            raise SchemaError("development_labels", col)

    wanted = {str(pid) for pid in pair_ids}
    label_by_pair = {
        str(row.pair_id): row
        for row in labels.itertuples(index=False)
        if str(row.pair_id) in wanted
    }
    evidence_by_pair = _evidence_index(evidence)

    rows: List[dict] = []
    for pid in pair_ids:
        pid = str(pid)
        row = label_by_pair.get(pid)
        if row is None:
            continue  # unknown pair: never scored as a negative
        status = getattr(row, "label_status")
        if status == CONFIRMED_TARGET:
            behavior = getattr(row, "behavior_family", NEGATIVE_BEHAVIOR)
            hands = evidence_by_pair.get(pid, [])
            rows.append(
                {
                    "pair_id": pid,
                    "risk_score": 1,
                    "predicted_behavior": str(behavior),
                    **_evidence_slots(hands),
                }
            )
        elif status == CONFIRMED_NON_TARGET:
            rows.append(
                {
                    "pair_id": pid,
                    "risk_score": 0,
                    "predicted_behavior": NEGATIVE_BEHAVIOR,
                    **_evidence_slots([]),
                }
            )
        # any other status is treated as unknown and skipped.

    columns = ["pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def _evidence_index(evidence: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """Group evidence hand ids per pair, ordered by ``evidence_rank`` if present."""
    if evidence is None or len(evidence) == 0:
        return {}
    if "pair_id" not in evidence.columns or "hand_id" not in evidence.columns:
        raise SchemaError("development_evidence", "pair_id/hand_id")
    df = evidence
    if "evidence_rank" in df.columns:
        df = df.sort_values(["pair_id", "evidence_rank"])
    index: Dict[str, List[str]] = {}
    for pid, hand in zip(df["pair_id"].astype(str), df["hand_id"].astype(str)):
        index.setdefault(pid, []).append(hand)
    return index


def _evidence_slots(hands: Sequence[str]) -> Dict[str, str]:
    """Fill the five evidence columns from a hand list, padding NO_EVIDENCE."""
    padded = list(hands)[:5] + [NO_EVIDENCE] * (5 - min(len(hands), 5))
    return {col: padded[i] for i, col in enumerate(EVIDENCE_COLUMNS)}


# --------------------------------------------------------------------------- #
# Harness                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fold:
    """One CV fold: the pools and confirmed pairs held out for validation.

    Attributes:
        index: Zero-based fold index.
        validation_pools: Pools whose pairs are the held-out validation set.
        validation_pairs: Confirmed pair ids in the validation pools.
        train_pairs: Confirmed pair ids in every other pool (the train set).
    """

    index: int
    validation_pools: frozenset
    validation_pairs: tuple
    train_pairs: tuple


@dataclass(frozen=True)
class FoldResult:
    """Per-fold scoring outcome."""

    index: int
    components: ScoreComponents
    n_validation_pairs: int


@dataclass
class CVReport:
    """Aggregated CV report: per-fold results plus mean components (Req 10.5)."""

    folds: List[FoldResult] = field(default_factory=list)

    @property
    def per_fold(self) -> List[Dict[str, float]]:
        """Per-fold component dicts (pair_ap, evidence_map, behavior_map, combined)."""
        return [f.components.as_dict() for f in self.folds]

    @property
    def mean(self) -> Dict[str, float]:
        """Mean of each component and the combined summary across folds."""
        if not self.folds:
            return {"pair_ap": 0.0, "evidence_map": 0.0, "behavior_map": 0.0, "combined": 0.0}
        keys = ("pair_ap", "evidence_map", "behavior_map", "combined")
        stacked = {k: [f.components.as_dict()[k] for f in self.folds] for k in keys}
        return {k: float(np.mean(v)) for k, v in stacked.items()}

    def as_dict(self) -> Dict[str, object]:
        """Full report as a plain dict for logging / persistence."""
        return {"mean": self.mean, "per_fold": self.per_fold}


class CVHarness:
    """Group-by-pool, family-stratified, PU-correct CV harness.

    Construct with the development labels and a pool map (or let it derive the pool
    map from real ``seats``/``hands``). :meth:`folds` exposes the deterministic
    fold assignment; :meth:`run` scores a prediction callback per fold with the
    production three-part metric and aggregates the components.
    """

    def __init__(
        self,
        labels: pd.DataFrame,
        pool_map: Mapping[str, object],
        *,
        n_folds: int = 5,
        seed: Optional[int] = None,
        evidence: Optional[pd.DataFrame] = None,
        families: Sequence[str] = DISCLOSED_FAMILY_NAMES,
    ) -> None:
        self.labels = labels
        self.pool_map = dict(pool_map)
        self.n_folds = n_folds
        self.seed = seed if seed is not None else get_config().seeds.cv_split
        self.evidence = evidence
        self.families = tuple(families)
        self._pool_folds = make_pool_folds(
            labels, self.pool_map, n_folds=n_folds, seed=self.seed, families=families
        )
        self._folds = self._build_folds()

    @classmethod
    def from_real_data(
        cls,
        labels: pd.DataFrame,
        *,
        seats: Optional[pd.DataFrame] = None,
        hands: Optional[pd.DataFrame] = None,
        config: Optional[PipelineConfig] = None,
        evidence: Optional[pd.DataFrame] = None,
        n_folds: int = 5,
        seed: Optional[int] = None,
    ) -> "CVHarness":
        """Build a harness deriving the pool map from real seats/hands files."""
        pool_map = derive_pool_map(labels, seats=seats, hands=hands, config=config)
        return cls(
            labels,
            pool_map,
            n_folds=n_folds,
            seed=seed,
            evidence=evidence,
        )

    @property
    def pool_folds(self) -> Dict[object, int]:
        """The ``{pool: fold_index}`` assignment."""
        return dict(self._pool_folds)

    def folds(self) -> List[Fold]:
        """Return the list of :class:`Fold` objects."""
        return list(self._folds)

    def _build_folds(self) -> List[Fold]:
        """Materialise per-fold pool/pair partitions from the pool assignment."""
        # confirmed pair ids only (PU: unknown pairs are not part of any fold set).
        status = self.labels.get("label_status")
        confirmed_mask = (
            status.isin({CONFIRMED_TARGET, CONFIRMED_NON_TARGET})
            if status is not None
            else pd.Series(True, index=self.labels.index)
        )
        pairs_by_fold: Dict[int, List[str]] = {f: [] for f in range(self.n_folds)}
        pools_by_fold: Dict[int, set] = {f: set() for f in range(self.n_folds)}
        for pid, keep in zip(self.labels["pair_id"].astype(str), confirmed_mask):
            if not keep:
                continue
            pool = self.pool_map.get(pid)
            if pool is None:
                continue
            fold = self._pool_folds.get(pool)
            if fold is None:
                continue
            pairs_by_fold[fold].append(pid)
            pools_by_fold[fold].add(pool)

        all_pairs = [p for f in range(self.n_folds) for p in pairs_by_fold[f]]
        folds: List[Fold] = []
        for f in range(self.n_folds):
            val = pairs_by_fold[f]
            val_set = set(val)
            train = [p for p in all_pairs if p not in val_set]
            folds.append(
                Fold(
                    index=f,
                    validation_pools=frozenset(pools_by_fold[f]),
                    validation_pairs=tuple(val),
                    train_pairs=tuple(train),
                )
            )
        return folds

    def run(
        self,
        predict_fn: Callable[[Fold, pd.DataFrame], pd.DataFrame],
    ) -> CVReport:
        """Score ``predict_fn`` on each fold and aggregate the components.

        ``predict_fn(fold, solution)`` receives the fold and its PU-correct
        solution frame and must return a submission frame with the same
        ``pair_id`` set and the production-metric schema (``risk_score``,
        ``predicted_behavior``, five evidence columns). Each fold is scored with
        :func:`score_components`; folds with no confirmed positives are still
        scored (Pair AP is defined as 0 there by the metric).

        Args:
            predict_fn: Callback producing a submission for a fold's solution.

        Returns:
            A :class:`CVReport` with per-fold components and their means (Req 10.5).
        """
        report = CVReport()
        for fold in self._folds:
            solution = build_fold_solution(
                self.labels, fold.validation_pairs, evidence=self.evidence
            )
            if len(solution) == 0:
                continue
            submission = predict_fn(fold, solution)
            components = score_components(solution, submission)
            report.folds.append(
                FoldResult(
                    index=fold.index,
                    components=components,
                    n_validation_pairs=len(solution),
                )
            )
        return report
