"""poker_collusion: pipeline for detecting suspicious value transfers in poker.

Top-level package for the Kaggle competition "Detect Suspicious Value Transfers
in Poker". Exposes the core data models, configuration, and shared exception
types used across the ingestion, feature, model, evidence, metric, submission,
validation, and discovery sub-packages.
"""

from poker_collusion.exceptions import SchemaError
from poker_collusion.types import (
    NOT_APPLICABLE,
    BehaviorLabel,
    HandSignal,
    LabelTable,
    NotApplicable,
    Pair,
    PairFeatureSet,
    PairPrediction,
)

__all__ = [
    "NOT_APPLICABLE",
    "BehaviorLabel",
    "HandSignal",
    "LabelTable",
    "NotApplicable",
    "Pair",
    "PairFeatureSet",
    "PairPrediction",
    "SchemaError",
]

__version__ = "0.1.0"
