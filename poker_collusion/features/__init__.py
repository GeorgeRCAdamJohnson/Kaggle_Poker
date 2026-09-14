"""Feature layer: EDA, hand-level signals, episodic aggregation, pair features."""

from poker_collusion.features.eda import (
    EdaArtifact,
    SharedHandCountStats,
    audit_evaluation_leakage,
    audit_evidence_validity,
    compute_family_distribution,
    compute_pu_counts,
    compute_shared_hand_count_stats,
    load_eda_artifact,
    run_eda,
    write_eda_artifact,
)
from poker_collusion.features.pair_features import (
    CONFOUNDER_FEATURE_NAMES,
    FIELD_BASELINE_KEYS,
    build_pair_feature_set,
    confounder_separating_features,
    pair_feature_set_to_dict,
    pair_feature_set_to_frame,
)

__all__ = [
    "EdaArtifact",
    "SharedHandCountStats",
    "audit_evaluation_leakage",
    "audit_evidence_validity",
    "compute_family_distribution",
    "compute_pu_counts",
    "compute_shared_hand_count_stats",
    "load_eda_artifact",
    "run_eda",
    "write_eda_artifact",
    "CONFOUNDER_FEATURE_NAMES",
    "FIELD_BASELINE_KEYS",
    "build_pair_feature_set",
    "confounder_separating_features",
    "pair_feature_set_to_dict",
    "pair_feature_set_to_frame",
]
