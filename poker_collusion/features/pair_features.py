"""Deterministic, schema-versioned Pair_Feature_Set assembly (task 10.3, Requirement 5).

This module folds a pair's per-hand :class:`~poker_collusion.types.HandSignal` stream into the
single :class:`~poker_collusion.types.PairFeatureSet` that every downstream detector, PU ranker,
and behavior classifier consumes. It is the *single source of truth* for a pair's representation
and — critically — it is computed with the **identical definition** for the development and the
evaluation periods (Req 5.3): the only thing that differs dev-vs-eval is *which shared hands feed
it*, never the feature formulas.

What the feature set contains
-----------------------------
* **Core signal features (Req 5.1).** Directed value-flow (max positive / max abs / signed mean /
  tails), aggression-asymmetry, isolation, mi_conflict, and the full **episodic** feature block —
  all delegated to :func:`poker_collusion.features.aggregation.aggregate_pair` so peak-preserving
  stats, temporal-concentration/burst features, and the empirical-Bayes count-robust
  normalization (Req 4.4 feeding 5.x) are computed **once**, in one place, and never reinvented.

* **Confounder-separating features (Req 5.2, RESEARCH_DOSSIER §2.4 H14–H17).** Features engineered
  specifically to discriminate a planted positive from each *disclosed benign confounder*. Each
  needs per-player-vs-field context (a player's baseline against the *whole field*, not just the
  partner); that context is **injected** as a ``FieldBaseline`` mapping so the function stays
  hermetic, testable, and deterministic. The exact formulas are documented on
  :func:`confounder_separating_features` below. Where a needed input is not available at this
  layer (e.g. a change-point on a player's field baseline over time), we compute what we can and
  expose the rest as a **clearly-named documented stub feature** — never fabricated.

Determinism (Req 5.4 / 5.5)
---------------------------
No randomness, no wall-clock, no dict-iteration-order dependence. Signals are aggregated through
the deterministic :func:`aggregate_pair`; the confounder block is pure arithmetic over the same
sorted signals and the injected baseline; the assembled ``features`` dict is materialised in a
**fixed, sorted key order**. Reordering the input signals therefore yields a byte-identical
:class:`PairFeatureSet`. (The exhaustive Hypothesis determinism test is Property 5 / task 10.4 and
is intentionally NOT written here.)

Schema + threshold versioning (Req 5.4)
---------------------------------------
The assembled :class:`PairFeatureSet` carries ``schema_version`` (from
:class:`~poker_collusion.config.PipelineConfig`) and ``threshold_version`` (from the injected EDA
artifact when present, else the config) so feature generation is reproducible and consumers know
exactly which calibrated thresholds produced the values. :func:`build_pair_feature_set` reads the
pool prior for shrinkage ONLY from the injected EDA artifact (Req 2.6) — it never recomputes a
threshold ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from poker_collusion.config import (
    DISCLOSED_FAMILY_NAMES,
    PipelineConfig,
    get_config,
)
from poker_collusion.features.aggregation import (
    DEFAULT_PRIOR_STRENGTH,
    EpisodicFeatures,
    aggregate_pair,
    prior_from_artifact,
)
from poker_collusion.types import NOT_APPLICABLE, HandSignal, PairFeatureSet

__all__ = [
    "FIELD_BASELINE_KEYS",
    "CONFOUNDER_FEATURE_NAMES",
    "FieldBaseline",
    "confounder_separating_features",
    "build_pair_feature_set",
    "pair_feature_set_to_dict",
    "pair_feature_set_to_frame",
]

#: The per-player field-baseline inputs the confounder contrasts consume. Each maps a *player id*
#: (as a string) to that player's baseline statistic computed against the WHOLE field (every other
#: opponent), NOT just the partner. These are produced upstream (EDA / a field-baseline pass) and
#: injected so this layer stays hermetic and deterministic. Absent players default to a neutral 0.0.
FIELD_BASELINE_KEYS: tuple[str, ...] = (
    "mean_value_flow_vs_field",       # signed mean directed value flow of the player vs all opponents
    "mean_aggression_vs_field",       # player's mean bet/raise-toward-opponent ratio vs the field
    "coseat_hands",                   # number of hands the pair was merely co-seated (context for H16)
)

#: A ``FieldBaseline`` is ``{player_id -> {baseline_key -> float}}`` (see FIELD_BASELINE_KEYS).
FieldBaseline = Mapping[str, Mapping[str, float]]

#: Prefix the assembled feature dict uses for every confounder-separating feature.
_CONF_PREFIX = "conf_"

#: The exact confounder-separating features this module emits (documented contract; consumed by
#: the classical detectors and the behavior classifier). Names are stable and sorted at emit time.
CONFOUNDER_FEATURE_NAMES: tuple[str, ...] = (
    f"{_CONF_PREFIX}directedness_contrast",       # H14 vs weak play / tilt
    f"{_CONF_PREFIX}avoidance_gap",               # H15 vs similar strategies
    f"{_CONF_PREFIX}joint_isolation_rate",        # H16 vs repeated opponent selection
    f"{_CONF_PREFIX}temporal_concentration",      # H17 vs streaks
    f"{_CONF_PREFIX}field_baseline_change_stub",  # H17 vs strategy changes (documented stub)
)


def _num(value: object, default: float = 0.0) -> float:
    """Coerce a value to float, returning ``default`` for None/NaN/uncoercible/sentinel."""
    if value is None or value is NOT_APPLICABLE:
        return default
    if isinstance(value, bool):
        return default
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _baseline_for(field_baseline: Optional[FieldBaseline], player: str, key: str) -> float:
    """Look up one field-baseline statistic for a player, defaulting to 0.0 (neutral)."""
    if not field_baseline:
        return 0.0
    row = field_baseline.get(str(player))
    if not isinstance(row, Mapping):
        return 0.0
    return _num(row.get(key, 0.0))


# --------------------------------------------------------------------------- #
# Confounder-separating features (Req 5.2, Dossier §2.4 H14–H17)
# --------------------------------------------------------------------------- #
def confounder_separating_features(
    signals: Sequence[HandSignal],
    player_a: str,
    player_b: str,
    episodic: EpisodicFeatures,
    field_baseline: Optional[FieldBaseline] = None,
) -> Dict[str, float]:
    """Compute the confounder-separating features that discriminate a planted positive from each
    disclosed benign confounder (Req 5.2; RESEARCH_DOSSIER §2.4).

    Every feature is a **contrast** against the same player's *field* baseline (injected via
    ``field_baseline``) so that a confounder which moves the player's *whole-field* behaviour
    cancels out (contrast ~0) while a *pair-specific* effect survives (contrast large). This is
    exactly the dossier's discriminator logic.

    Formulas (all pure arithmetic over the pair's signals + the injected baseline)
    ------------------------------------------------------------------------------
    Let the pair's shared-hand signals give:

      * ``pair_mean_flow``  = signed mean of ``value_flow`` over applicable shared hands
        (positive => chips move A->B). This equals ``episodic.features['value_flow_mean']``.
      * ``pair_mean_aggr``  = mean of the pair-directed ``aggression_asymmetry`` over applicable
        hands (fraction of a member's bet/raises aimed at the partner; low => avoidance).

    and the injected field baseline gives, per player p:

      * ``flow_field(p)``   = ``mean_value_flow_vs_field`` — p's signed mean flow vs *all* opponents.
      * ``aggr_field(p)``   = ``mean_aggression_vs_field``  — p's mean bet/raise-toward ratio vs field.
      * ``coseat(p)``       = ``coseat_hands``               — hands the pair was merely co-seated.

    1. ``conf_directedness_contrast`` (H14 — vs weak play / tilt)::

           = pair_mean_flow - mean(flow_field(A), flow_field(B))

       The pair-directed value flow MINUS the players' mean flow against the whole field. Weak
       play / tilt leaks chips to *everyone*, moving the field baseline just as much as the pair
       flow, so the contrast is ~0. A planted ``directed_transfer`` moves chips *selectively* to
       the partner, so the contrast is large. (=> ~0 when pair flow equals the field baseline;
       large when pair flow exceeds it — the hand-checkable property the unit test asserts.)

    2. ``conf_avoidance_gap`` (H15 — vs similar strategies)::

           = mean(aggr_field(A), aggr_field(B)) - pair_mean_aggr

       The partner-vs-field aggression-avoidance GAP: how much *less* the pair aggresses toward
       each other than toward the field. Two players with *similar* (tight/passive) strategies
       are passive toward *everyone*, so their field baseline drops too and the gap collapses to
       ~0; colluders in ``soft_play`` open the gap (positive => avoids the partner specifically).

    3. ``conf_joint_isolation_rate`` (H16 — vs repeated opponent selection)::

           = episodic joint-isolation mass  /  max(coseat_hands, n_shared_hands, 1)

       Joint-isolation rate CONDITIONED on co-seating. Repeated opponent selection makes the pair
       merely *co-present* (large ``coseat_hands``) without joint targeting, so dividing the
       joint-isolation mass by the co-seating exposure drives the rate toward chance; genuine
       ``coordinated_isolation`` sustains a high rate per shared hand. The numerator reuses the
       episodic isolation peak mass (``isolation_mean`` * flagged exposure) already computed by
       aggregation — never reinvented here.

    4. ``conf_temporal_concentration`` (H17 — vs streaks)::

           = episodic.features['concentration_ratio']   (max_burst / flagged_count, in [0,1])

       The fraction of the pair's flagged (behavior-specific-action) mass sitting in its single
       densest temporal episode, straight from aggregation. A *streak* is a run of outcomes with
       NO accompanying coordinated action pattern, so its flagged mass is ~0 and this stays low;
       a planted relationship's coordination hands cluster, driving it high.

    5. ``conf_field_baseline_change_stub`` (H17 — vs strategy changes) — **documented stub**::

           = 0.0   (always, at this layer)

       Separating a *strategy change* (a persistent, field-wide step shift in one player's
       baseline over time) requires a **change-point on the player's field baseline across the
       timeline**, which needs the ordered per-player-vs-field signal series over time — an input
       NOT available at this pair-aggregation layer. Rather than fabricate a value, we emit a
       clearly-named stub fixed at 0.0 and document the contract: a later field-baseline-change
       pass (with the temporal field series) will populate it. Its presence keeps the feature
       schema stable dev-vs-eval so no feature is "available at training but absent at scoring".

    Returns:
        A ``{feature_name: float}`` dict over exactly :data:`CONFOUNDER_FEATURE_NAMES`.
    """
    feats = episodic.features

    # --- pair-side statistics (from the deterministic aggregation, never reinvented) --------- #
    pair_mean_flow = _num(feats.get("value_flow_mean", 0.0))
    pair_mean_aggr = _num(feats.get("aggression_asymmetry_mean", 0.0))
    isolation_mean = _num(feats.get("isolation_mean", 0.0))
    flagged_count = _num(feats.get("flagged_count", 0.0))
    concentration_ratio = _num(feats.get("concentration_ratio", 0.0))
    n_shared = float(episodic.n_hands)

    # --- injected field baselines, averaged over the two members ----------------------------- #
    flow_field = 0.5 * (
        _baseline_for(field_baseline, player_a, "mean_value_flow_vs_field")
        + _baseline_for(field_baseline, player_b, "mean_value_flow_vs_field")
    )
    aggr_field = 0.5 * (
        _baseline_for(field_baseline, player_a, "mean_aggression_vs_field")
        + _baseline_for(field_baseline, player_b, "mean_aggression_vs_field")
    )
    coseat = max(
        _baseline_for(field_baseline, player_a, "coseat_hands"),
        _baseline_for(field_baseline, player_b, "coseat_hands"),
    )

    # 1. H14 directedness contrast: pair-directed flow minus the field baseline (~0 for tilt).
    directedness_contrast = pair_mean_flow - flow_field

    # 2. H15 avoidance gap: field aggression minus pair-directed aggression (collapses for
    #    similar-strategy pairs whose field baseline drops too).
    avoidance_gap = aggr_field - pair_mean_aggr

    # 3. H16 joint-isolation rate conditioned on co-seating exposure.
    exposure = max(coseat, n_shared, 1.0)
    joint_isolation_mass = isolation_mean * flagged_count
    joint_isolation_rate = joint_isolation_mass / exposure

    # 4. H17 temporal concentration (streaks) — straight from aggregation.
    temporal_concentration = concentration_ratio

    # 5. H17 strategy-changes — documented stub (needs the temporal field-baseline series).
    field_baseline_change_stub = 0.0

    return {
        f"{_CONF_PREFIX}directedness_contrast": float(directedness_contrast),
        f"{_CONF_PREFIX}avoidance_gap": float(avoidance_gap),
        f"{_CONF_PREFIX}joint_isolation_rate": float(joint_isolation_rate),
        f"{_CONF_PREFIX}temporal_concentration": float(temporal_concentration),
        f"{_CONF_PREFIX}field_baseline_change_stub": float(field_baseline_change_stub),
    }


# --------------------------------------------------------------------------- #
# Top-level assembly
# --------------------------------------------------------------------------- #
def build_pair_feature_set(
    pair_id: str,
    signals: Sequence[HandSignal],
    player_a: str,
    player_b: str,
    *,
    phase: Optional[str] = None,
    field_baseline: Optional[FieldBaseline] = None,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    config: Optional[PipelineConfig] = None,
    prior: Optional[float] = None,
    alpha: float = DEFAULT_PRIOR_STRENGTH,
) -> PairFeatureSet:
    """Assemble the deterministic, schema-versioned :class:`PairFeatureSet` for one pair.

    The feature *definitions* are identical for the development and evaluation periods (Req 5.3):
    pass ``phase="development"`` or ``phase="evaluation"`` to choose *which* shared hands feed the
    aggregation, but every formula (core + confounder) is the same. Passing ``phase=None`` (or
    ``"all"``) aggregates the whole stream.

    Args:
        pair_id: The pair identifier (copied onto the result).
        signals: The pair's per-hand :class:`HandSignal` stream (any order; sorted internally).
        player_a: Pair's first player id (canonical: min id) — orients ``value_flow`` sign.
        player_b: Pair's second player id (canonical: max id).
        phase: Which slice to aggregate (``"development"`` / ``"evaluation"`` / ``None`` for all).
            This is the ONLY dev-vs-eval difference (Req 5.3).
        field_baseline: Injected per-player-vs-field baselines for the confounder contrasts
            (:data:`FIELD_BASELINE_KEYS`). Absent -> neutral 0.0 baselines (contrasts reduce to
            the raw pair statistic), keeping the call well-defined and deterministic.
        eda_artifact: Injected EDA/threshold artifact supplying the pool prior for shrinkage
            (Req 2.6/4.4) and the ``threshold_version`` stamped on the result.
        config: Pipeline config supplying ``schema_version`` (and the fallback
            ``threshold_version`` when no artifact is given). Defaults to :func:`get_config`.
        prior: Explicit pool-prior override for shrinkage (takes precedence over the artifact).
        alpha: Empirical-Bayes prior strength (pseudo-count) for count-robust shrinkage (Req 4.4).

    Returns:
        A :class:`PairFeatureSet` with a flat, sorted ``features`` map (core + episodic +
        confounder), ``schema_version``, and ``threshold_version``. Deterministic in the inputs.
    """
    cfg = config or get_config()

    # Core + episodic features, computed once by the deterministic aggregation (Req 5.1, 4.x).
    episodic = aggregate_pair(
        pair_id,
        signals,
        phase=phase,
        eda_artifact=eda_artifact,
        prior=prior,
        alpha=alpha,
    )

    # Confounder-separating features (Req 5.2). Aggregate over the SAME phase slice the episodic
    # features used, so dev/eval remain identical definitions over their respective hands.
    subset = (
        list(signals)
        if phase is None or phase == "all"
        else [s for s in signals if s.phase == phase]
    )
    conf = confounder_separating_features(
        subset, str(player_a), str(player_b), episodic, field_baseline
    )

    # Merge: core/episodic first, then confounder features; then materialise in a fixed sorted
    # key order so the dict is byte-deterministic regardless of insertion order (Req 5.4/5.5).
    merged: Dict[str, float] = {}
    merged.update({k: float(v) for k, v in episodic.features.items()})
    merged.update({k: float(v) for k, v in conf.items()})
    # Provenance markers useful downstream (phase label + shared-hand count for this slice).
    merged["n_shared_hands"] = float(episodic.n_hands)
    merged["phase_is_development"] = 1.0 if (phase == "development") else 0.0
    features = {k: merged[k] for k in sorted(merged)}

    threshold_version = _resolve_threshold_version(eda_artifact, cfg)

    return PairFeatureSet(
        pair_id=str(pair_id),
        schema_version=cfg.schema_version,
        features=features,
        threshold_version=threshold_version,
    )


def _resolve_threshold_version(
    eda_artifact: Optional[Union[Mapping, object]],
    config: PipelineConfig,
) -> str:
    """Prefer the injected artifact's ``threshold_version``; fall back to the config's."""
    if eda_artifact is not None:
        tv = getattr(eda_artifact, "threshold_version", None)
        if tv is None and isinstance(eda_artifact, Mapping):
            tv = eda_artifact.get("threshold_version")
        if tv:
            return str(tv)
    return config.threshold_version


# --------------------------------------------------------------------------- #
# Stable serialization (Req 5.4: persist with a stable/ordered path)
# --------------------------------------------------------------------------- #
def pair_feature_set_to_dict(fs: PairFeatureSet) -> Dict[str, object]:
    """Serialise a :class:`PairFeatureSet` into a stable, ordered plain dict.

    Keys are emitted in a fixed order (identity fields first, then the sorted feature names) so
    the serialization is byte-deterministic and safe to persist / checksum.
    """
    return {
        "pair_id": fs.pair_id,
        "schema_version": fs.schema_version,
        "threshold_version": fs.threshold_version,
        "features": {k: float(v) for k, v in sorted(fs.features.items())},
    }


def pair_feature_set_to_frame(feature_sets: Sequence[PairFeatureSet]) -> pd.DataFrame:
    """Flatten one or more :class:`PairFeatureSet` into a deterministic wide DataFrame.

    One row per pair; columns are ``pair_id, schema_version, threshold_version`` followed by every
    feature name in sorted order (the union across the input, so a stable column layout). Missing
    features (should not occur for a fixed schema) are filled with 0.0. Rows are ordered by
    ``pair_id`` for determinism.
    """
    feature_names: set[str] = set()
    for fs in feature_sets:
        feature_names.update(fs.features.keys())
    ordered_cols = sorted(feature_names)

    records = []
    for fs in feature_sets:
        rec: Dict[str, object] = {
            "pair_id": fs.pair_id,
            "schema_version": fs.schema_version,
            "threshold_version": fs.threshold_version,
        }
        for name in ordered_cols:
            rec[name] = float(fs.features.get(name, 0.0))
        records.append(rec)

    frame = pd.DataFrame.from_records(
        records, columns=["pair_id", "schema_version", "threshold_version", *ordered_cols]
    )
    if len(frame) > 0:
        frame = frame.sort_values("pair_id", kind="stable").reset_index(drop=True)
    return frame
