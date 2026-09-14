"""Core data models for the poker_collusion pipeline.

These mirror the "Data Models" section of the design document exactly. The
dataclasses that represent immutable, hashable identities (``Pair``) are frozen;
the mutable feature/signal/prediction carriers follow the design's declarations.

A ``NOT_APPLICABLE`` sentinel is defined for hand-level signals whose denominator
is zero (Requirement 3.5), and a ``BehaviorLabel`` enum enumerates the target
coordination behaviors.

Requirements:
- 1.5 (SchemaError lives in ``exceptions``; referenced here for completeness)
- 5.4: PairFeatureSet carries a schema version and threshold version for
  reproducible feature generation.
- 11.2: Deterministic, seeded structures support reproducing the submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union

__all__ = [
    "NOT_APPLICABLE",
    "BehaviorLabel",
    "HandSignal",
    "LabelTable",
    "NotApplicable",
    "Pair",
    "PairFeatureSet",
    "PairPrediction",
    "Sentinel",
]


class NotApplicable:
    """Sentinel type indicating a signal has no defined value.

    Used when a hand-level ratio signal (aggression-asymmetry, isolation) has a
    zero denominator: no qualifying bet-or-raise opportunities exist for the
    relevant player or pair in the hand (Requirement 3.5). The single shared
    instance is exported as ``NOT_APPLICABLE``.
    """

    _instance: "NotApplicable | None" = None

    def __new__(cls) -> "NotApplicable":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "NOT_APPLICABLE"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self):  # keep the singleton identity across pickling
        return (_get_not_applicable, ())


def _get_not_applicable() -> "NotApplicable":
    return NOT_APPLICABLE


#: Shared sentinel instance for "not applicable" signal values (Requirement 3.5).
NOT_APPLICABLE: NotApplicable = NotApplicable()

#: Alias matching the design's type annotations (``float | Sentinel``).
Sentinel = NotApplicable


class BehaviorLabel(str, Enum):
    """The target coordination behavior families.

    Members are the exact string tokens written to ``predicted_behavior`` in the
    submission (Requirements 7.1, 9.4). Inheriting from ``str`` makes each member
    compare and serialize as its wire value.
    """

    NONE = "none"
    DIRECTED_TRANSFER = "directed_transfer"
    SOFT_PLAY = "soft_play"
    COORDINATED_ISOLATION = "coordinated_isolation"
    OTHER_COORDINATION = "other_coordination"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: The three publicly described coordination behaviors scored by Behavior MAP.
DISCLOSED_FAMILIES: tuple[BehaviorLabel, ...] = (
    BehaviorLabel.DIRECTED_TRANSFER,
    BehaviorLabel.SOFT_PLAY,
    BehaviorLabel.COORDINATED_ISOLATION,
)


@dataclass(frozen=True)
class Pair:
    """An unordered relationship between two players evaluated for coordination.

    ``player_a`` and ``player_b`` are stored in canonical order
    (``player_a = min(id)``, ``player_b = max(id)``) so a pair has one identity.
    Frozen so it is hashable and usable as a dictionary key.
    """

    pair_id: str
    player_a: int  # canonical: min(id)
    player_b: int  # canonical: max(id)
    pool_id: int


@dataclass
class HandSignal:
    """Per-hand coordination signals for one co-seated pair.

    ``aggression_asymmetry`` and ``isolation`` may hold the ``NOT_APPLICABLE``
    sentinel when their denominator is zero (Requirement 3.5).
    """

    hand_id: int
    pair_id: str
    phase: str  # "development" | "evaluation"
    value_flow: float
    aggression_asymmetry: Union[float, Sentinel]  # NOT_APPLICABLE allowed
    isolation: Union[float, Sentinel]
    mi_conflict: float
    behavior_action_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PairFeatureSet:
    """The aggregated feature vector describing a Pair.

    Carries the ``schema_version`` and ``threshold_version`` so feature
    generation is reproducible and consumers know which calibrated thresholds
    produced the values (Requirement 5.4).
    """

    pair_id: str
    schema_version: str
    features: Dict[str, float] = field(default_factory=dict)
    threshold_version: str = ""


@dataclass
class LabelTable:
    """Development label structure for Positive-Unlabelled learning.

    ``trusted_positive`` maps a confirmed-coordinated ``pair_id`` to its target
    behavior; ``confirmed_negative`` is the set of confirmed non-target pairs.
    Every other development pair is UNKNOWN (unlabelled), not negative.
    """

    trusted_positive: Dict[str, str] = field(default_factory=dict)
    confirmed_negative: set[str] = field(default_factory=set)
    # everything else in development = unknown (PU)


@dataclass
class PairPrediction:
    """The full prediction emitted for one evaluation pair.

    ``evidence`` is always length 5, each entry a hand id or the literal
    ``"NO_EVIDENCE"`` (Requirements 8.6, 8.7). ``reasons`` carries explainability
    strings used for the winner-verification case reviews.
    """

    pair_id: str
    risk_score: float  # [0, 1]
    predicted_behavior: str
    evidence: List[str] = field(default_factory=list)  # length 5, hand ids or "NO_EVIDENCE"
    reasons: List[str] = field(default_factory=list)  # explainability for case review
