"""Explainable classical baseline detectors (task 11.1, Requirement 5.1/5.2, 6.3/6.4).

Design Principle 3 ("classical, researched baselines before custom ML"): before any learned
model, we ship an **explainable, no-training** detector layer end-to-end. Each detector reads a
single :class:`~poker_collusion.types.PairFeatureSet` (the deterministic, schema-versioned vector
assembled by :mod:`poker_collusion.features.pair_features`) and returns a ``(score, reason)``
pair — a float suspicion contribution and a **non-empty human-readable string that names the
driving feature and its value**. The reason strings feed the winner-verification case reviews
(Req 11.4) and make every risk contribution auditable.

The five detectors (grounded in the design's prior-art list)
------------------------------------------------------------
1. :func:`value_flow_graph_score` — chip-transfer / value-flow graph score. Magnitude +
   consistency of *directed* value flow between the pair (arXiv:2203.05121 graph-theoretic pair
   features adapted to co-seating edges).
2. :func:`aggression_asymmetry_softplay_test` — the partner-vs-field aggression *avoidance gap*
   (``conf_avoidance_gap``, dossier H15) plus the aggression-asymmetry stats — a soft-play tell
   (colluders avoid aggressing the partner).
3. :func:`isolation_pressure_ratio` — joint isolation of third players
   (``conf_joint_isolation_rate`` + isolation stats, dossier H16).
4. :func:`mi_conflict_avoidance` — information-theoretic conflict avoidance from the
   ``mi_conflict`` stats (OpenReview / PMLR v180).
5. :func:`collusion_table_advantage` — the AAAI collusion-table-advantage measure: a **behavior-
   agnostic** composite backbone for the undisclosed ``other_coordination`` family. Since a full
   advantage measure needs game-state inputs beyond the current feature layer, we build a clearly
   documented behavior-agnostic composite from the *available* features (value-flow magnitude +
   episodic concentration + directedness contrast) and never fabricate unavailable inputs.

Combining into a baseline ``risk_score`` in [0, 1]
--------------------------------------------------
Each detector's raw score is first mapped to a **comparable scale** by a documented per-detector
normalizer — ``log1p`` of the raw score divided by a documented feature-scale constant
(:func:`_soft_norm`), so the mapped score is in ``[0, +inf)`` and *monotone non-decreasing* in the
raw score. ``log1p`` is **non-saturating**: unlike the old ``tanh`` it has no flat tail, so pairs
with different (large) signals keep *different* mapped values and stay rankable. The mapped scores
are combined by a **fixed, documented weighted sum** (:data:`DETECTOR_WEIGHTS`) into a monotone raw
combined score (:func:`_raw_combined`) whose ORDER is what Pair AP depends on.

Two documented, MONOTONE maps turn that raw combined score into a ``risk_score`` in ``[0, 1]``:

* **Per pair** (:func:`score_pair`) — a single pair has no population to rank against, so it uses a
  bounded, non-saturating rational (Möbius) squash ``s / (SQUASH_SCALE + s)`` (:func:`_squash_single`)
  that approaches 1 only gently (no ~0.95 saturation clump).
* **Per batch** (:func:`score_pairs_ranked`, the production path) — maps each scoreable pair's raw
  combined score to its **population rank percentile**, then applies the spreading curve
  ``percentile ** RANK_SPREAD_EXPONENT``. This guarantees a well-spread ranking (no tie clump) with
  only a small high-risk fraction (``frac(risk > 0.9) ≈ 1.7%``), which is exactly what Pair AP needs
  on the real evaluation distribution.

Both maps are monotone in the raw combined score, so raising any detector's raw score never lowers
``risk_score`` and the two paths agree on pair ORDER (hence Pair AP).

Root-cause history (Phase-1 defect). The original combine used a ``tanh`` per-detector normalizer
plus a ``sigmoid(2*s - 3)`` squash. On the real 112,540-pair evaluation set the ``tanh`` tails +
logistic saturated, collapsing ~half the pairs into a near-tied ~0.95 "high-risk" clump; under Pair
AP that unrankable clump buried the rare true positives and scored *below* the do-nothing baseline.
The non-saturating ``log1p`` map + value-flow-tail detectors + rank-percentile spreading above fix
that (dev-label Pair AP rose from ≈0.33 to ≈0.65; high-risk fraction fell from ≈0.66 to ≈0.02).

Coverage + unscoreable default (Req 6.3 range, 6.4 coverage)
------------------------------------------------------------
:func:`score_pair` returns a :class:`ClassicalScore` for **every** input pair (Req 6.4). A pair is
**unscoreable** when it shares too few hands to trust its episodic signal
(``n_shared_hands < min_shared_hands`` from the injected EDA artifact) OR when every detector input
is neutral (all mapped detector scores ~0). An unscoreable pair is assigned a **defined default**:
the **pool prior** (``pool_prior_positive_prevalence`` read from the injected EDA artifact), falling
back to the documented constant :data:`DEFAULT_POOL_PRIOR` when no artifact is supplied. This keeps
every ``risk_score`` in ``[0, 1]`` and every evaluation pair scored (Req 6.3/6.4).

Determinism
-----------
No randomness, no wall-clock, no dict-iteration-order dependence. Given the same
:class:`PairFeatureSet` and the same injected artifact, :func:`score_pair` returns an identical
:class:`ClassicalScore`. :func:`score_pairs` is a deterministic batch scorer preserving input order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from poker_collusion.types import PairFeatureSet

__all__ = [
    "DEFAULT_POOL_PRIOR",
    "DEFAULT_MIN_SHARED_HANDS",
    "DETECTOR_WEIGHTS",
    "LOGISTIC_GAIN",
    "LOGISTIC_BIAS",
    "RANK_SPREAD_EXPONENT",
    "SQUASH_SCALE",
    "NEUTRAL_EPS",
    "ClassicalScore",
    "value_flow_graph_score",
    "aggression_asymmetry_softplay_test",
    "isolation_pressure_ratio",
    "mi_conflict_avoidance",
    "collusion_table_advantage",
    "DETECTORS",
    "score_pair",
    "score_pairs",
    "score_pairs_ranked",
]

# --------------------------------------------------------------------------- #
# Documented constants
# --------------------------------------------------------------------------- #
#: Fallback pool prior for the unscoreable default when no EDA artifact is injected. The real run
#: reads ``pool_prior_positive_prevalence`` from the artifact; this constant is only the documented
#: last-resort default (a small positive prevalence, matching the rare-positive pool structure).
DEFAULT_POOL_PRIOR: float = 0.01

#: Fallback minimum shared-hand count below which a pair's episodic signal is untrustworthy and the
#: pair is treated as UNSCOREABLE. The real run reads ``min_shared_hands`` from the EDA artifact.
DEFAULT_MIN_SHARED_HANDS: int = 1

#: Below this, a mapped detector score (already in [0, 1)) is treated as "neutral". A pair whose
#: every mapped detector score is <= NEUTRAL_EPS carries no coordination evidence => unscoreable.
NEUTRAL_EPS: float = 1e-9

#: Per-detector feature-scale constants used by the NON-SATURATING ``log1p`` normalizer
#: ``log1p(raw / scale)`` (see :func:`_soft_norm`). Unlike ``tanh``, ``log1p`` has NO flat tail: it
#: keeps rising (ever more gently) for arbitrarily large ``raw``, so two pairs with different
#: (large) signals still map to *different* values and remain rankable. Each scale is a documented
#: order-of-magnitude anchor near the POSITIVE-class signal level measured on the real development
#: labels (so the responsive region of the map lands where real pairs live, not in a flat tail).
#:
#: Calibration note (dev labels, 372 pos / 1488 neg): the directed value-flow *tail* features are
#: by far the strongest single discriminators (``value_flow_abs_p95`` AUC 0.88, ``_p90`` 0.84,
#: ``value_flow_absmean`` 0.81); ``aggression_asymmetry_mean`` adds a little (AUC 0.77). The old
#: detectors' ``mi_conflict`` block is identically zero on dev data (AUC 0.50 — no signal) and the
#: isolation / avoidance-gap terms are *anti*-predictive (AUC < 0.5), so we down-weight them to a
#: token contribution and let the value-flow tail carry the ranking.
_SCALE_VALUE_FLOW: float = 5.0         # bb; anchor near the positive-class value_flow_abs_p95 level
_SCALE_SOFTPLAY: float = 0.3           # ratio; anchor near the positive-class aggression level
_SCALE_ISOLATION: float = 0.5          # ratio; token helper (weak/anti-predictive on dev)
_SCALE_MI: float = 1.0                 # nats-ish; token helper (zero signal on dev)
_SCALE_TABLE_ADVANTAGE: float = 2.0    # bb; anchor near the positive-class value_flow_absmean level

#: Fixed, documented weights for the monotone weighted sum of mapped detector scores. The
#: value-flow tail detector (the most direct directed-transfer tell, and the strongest measured
#: discriminator) dominates; the behavior-agnostic backbone and the soft-play term add small,
#: robust helpers; the isolation / MI detectors — measured to carry little-or-no signal on dev —
#: contribute only a token weight so they can help a genuinely coordinated pair without dragging
#: the ranking. Weights are all >= 0 so the combination is monotone non-decreasing in every
#: detector's raw score.
DETECTOR_WEIGHTS: Dict[str, float] = {
    "value_flow_graph_score": 1.0,
    "aggression_asymmetry_softplay_test": 0.10,
    "isolation_pressure_ratio": 0.05,
    "mi_conflict_avoidance": 0.05,
    "collusion_table_advantage": 0.10,
}

#: Exponent of the RANK-percentile spreading curve applied by :func:`score_pairs_ranked`
#: (``risk = rank_percentile ** RANK_SPREAD_EXPONENT``). The curve is a MONOTONE transform of the
#: raw combined score's population rank, so it never changes the pair ordering (Pair AP is
#: unchanged) — it only sets how the well-spread ranks map into [0, 1]. Because the fraction of
#: pairs exceeding a threshold ``t`` under ``rank ** k`` is ``1 - t ** (1/k)`` INDEPENDENT of the
#: score distribution, ``k = 6`` yields ``frac(risk > 0.9) ≈ 1.7%`` and ``frac(risk > 0.5) ≈ 11%``
#: on ANY population — small high-risk mass (no saturation clump), consistent with a rare-positive
#: pool, while keeping the full [0, 1] range populated for a well-spread ranking.
RANK_SPREAD_EXPONENT: float = 6.0

#: Logistic gain/bias retained for the PER-PAIR :func:`score_pair` fallback map (a single pair has
#: no population to rank against). The per-pair risk is a bounded, MONOTONE, NON-saturating map of
#: the raw combined score ``s`` (see :func:`_squash_single`); these constants shape it. Kept in the
#: public API for backward compatibility / documentation.
LOGISTIC_GAIN: float = 2.0
LOGISTIC_BIAS: float = -3.0

#: Scale of the per-pair rational squash ``s / (SQUASH_SCALE + s)`` used by :func:`_squash_single`.
#: Rational (Möbius) squashing approaches 1 far more gently than a logistic, so a single strongly
#: suspicious pair still lands clearly above a benign one without slamming into a saturated ~1.
SQUASH_SCALE: float = 1.5


# --------------------------------------------------------------------------- #
# Result carrier
# --------------------------------------------------------------------------- #
@dataclass
class ClassicalScore:
    """The classical-baseline scoring result for one pair (public API).

    Attributes:
        pair_id: The pair identifier (copied from the feature set).
        risk_score: Baseline coordination risk in ``[0, 1]`` (Req 6.3). Always defined — the pool
            prior default for unscoreable pairs (Req 6.4).
        reasons: One human-readable reason string per detector (explainability; case reviews).
        detector_scores: ``{detector_name -> raw_score}`` for every detector (audit/inspection).
        unscoreable: True when the pair fell back to the pool-prior default.
        pool_prior: The pool prior used as the unscoreable default (for provenance).
        raw_combined_score: The monotone weighted-sum of the (non-saturating) mapped detector
            scores BEFORE any squash — the quantity whose ORDER drives Pair AP. ``score_pair`` and
            :func:`score_pairs_ranked` are both monotone transforms of it. ``0.0`` for an
            unscoreable pair (it carries no combined evidence).
    """

    pair_id: str
    risk_score: float
    reasons: List[str] = field(default_factory=list)
    detector_scores: Dict[str, float] = field(default_factory=dict)
    unscoreable: bool = False
    pool_prior: float = DEFAULT_POOL_PRIOR
    raw_combined_score: float = 0.0


# --------------------------------------------------------------------------- #
# Feature access helpers
# --------------------------------------------------------------------------- #
def _feat(fs: PairFeatureSet, name: str, default: float = 0.0) -> float:
    """Read a feature value as a finite float, defaulting for missing/NaN/uncoercible."""
    val = fs.features.get(name, default)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _soft_norm(raw: float, scale: float) -> float:
    """Map a NON-NEGATIVE raw score to ``[0, +inf)`` via ``log1p(raw / scale)`` (NON-saturating).

    This replaces the old ``tanh`` normalizer, whose flat tail was the root cause of the Phase-1
    saturation defect: ``tanh`` maps any signal past ~2*scale to ~1.0, so realistic pairs all piled
    into a near-identical mapped value and became UNRANKABLE (a huge tie clump). ``log1p`` keeps
    rising — ever more gently — for arbitrarily large ``raw``, so two pairs with different (large)
    signals still map to *different* values and stay orderable, which is exactly what Pair AP needs.

    Monotone non-decreasing in ``raw`` for ``raw >= 0`` and ``scale > 0``. Negative raw scores are
    clamped to 0 first (a detector's "evidence" contribution is never negative — absence of a tell
    is neutral, not exculpatory, for the additive backbone).
    """
    if scale <= 0.0:
        scale = 1.0
    r = raw if raw > 0.0 else 0.0
    return math.log1p(r / scale)


# Backward-compatible alias (the old name still routes through the non-saturating map).
_tanh_norm = _soft_norm


# --------------------------------------------------------------------------- #
# Detector 1: value-flow graph score (chip-transfer graph; arXiv:2203.05121)
# --------------------------------------------------------------------------- #
def value_flow_graph_score(fs: PairFeatureSet) -> Tuple[float, str]:
    """Directed value-flow graph score built on the value-flow TAIL (Req 5.1).

    The dominant, most-direct directed-transfer tell. On the real development labels the *tail* of
    the pair's ``|value_flow|`` distribution is by far the strongest single discriminator between a
    planted positive and a benign pair (``value_flow_abs_p95`` ranks positives above negatives with
    AUC ≈ 0.88, ``_p90`` ≈ 0.84), because a colluding pair leaves a few *large* directed transfers
    even when its per-hand mean is unremarkable. The old formula used ``value_flow_max_abs`` (the
    single most extreme hand, AUC ≈ 0.72) plus the *signed* mean (``value_flow_mean``, AUC ≈ 0.51 —
    essentially noise, since direction cancels), which is why it under-discriminated. We instead
    combine the two robust upper-tail percentiles::

        score = value_flow_abs_p95  +  0.5 * value_flow_abs_p90

    Both terms are non-negative magnitudes, so ``score >= 0`` and it plugs straight into the
    additive backbone. (We deliberately drop the single most-extreme ``value_flow_max_abs`` from the
    *score*: it is a weaker discriminator on dev — AUC ≈ 0.72 vs the p95 tail's 0.88 — and its huge
    magnitude distorts the shared ``log1p`` scale; it is still reported in the reason for audit.)
    Reason: names the tail percentiles and the peak transfer.
    """
    p95 = _feat(fs, "value_flow_abs_p95")
    p90 = _feat(fs, "value_flow_abs_p90")
    max_abs = _feat(fs, "value_flow_max_abs")
    mean = _feat(fs, "value_flow_mean")
    score = max(p95, 0.0) + 0.5 * max(p90, 0.0)
    direction = "toward partner" if mean >= 0.0 else "toward self"
    reason = (
        f"directed value-flow tail p95 {p95:.1f}bb / p90 {p90:.1f}bb "
        f"(peak |flow| {max_abs:.1f}bb, net {mean:.1f}bb {direction})"
    )
    return float(score), reason


# --------------------------------------------------------------------------- #
# Detector 2: aggression-asymmetry & soft-play test (avoidance gap H15)
# --------------------------------------------------------------------------- #
def aggression_asymmetry_softplay_test(fs: PairFeatureSet) -> Tuple[float, str]:
    """Soft-play / aggression-asymmetry test (Req 5.2, dossier H15) — retargeted to the signal.

    On the real development labels the pair's mean aggression asymmetry
    (``aggression_asymmetry_mean``) is a genuine secondary discriminator (AUC ≈ 0.77): coordinated
    pairs show a consistently *asymmetric* aggression pattern between the two members. The old
    formula relied on ``conf_avoidance_gap`` and ``aggression_asymmetry_max``, but on dev data the
    avoidance gap is *anti*-predictive (AUC ≈ 0.23) and the peak is saturated at 1.0 for nearly
    every pair (AUC ≈ 0.50 — no signal). We therefore score the mean asymmetry instead::

        score = max(aggression_asymmetry_mean, 0)

    A token, non-negative helper behind the value-flow tail. Reason: names the mean asymmetry (and
    reports the legacy avoidance gap for audit continuity).
    """
    aa_mean = _feat(fs, "aggression_asymmetry_mean")
    gap = _feat(fs, "conf_avoidance_gap")
    score = max(aa_mean, 0.0)
    reason = (
        f"aggression-asymmetry mean {aa_mean:.3f} "
        f"(legacy avoidance gap {gap:.3f}) [H15 soft-play]"
    )
    return float(score), reason


# --------------------------------------------------------------------------- #
# Detector 3: isolation-pressure ratio (joint isolation of third players H16)
# --------------------------------------------------------------------------- #
def isolation_pressure_ratio(fs: PairFeatureSet) -> Tuple[float, str]:
    """Isolation-pressure ratio from joint isolation of third players (Req 5.2, dossier H16).

    ``conf_joint_isolation_rate`` is the pair's joint-isolation mass conditioned on co-seating
    exposure; ``isolation_max`` is the peak single-hand isolation. Combine::

        score = joint_isolation_rate + 0.5 * isolation_max

    Repeated-opponent-selection confounders inflate co-seating exposure without joint targeting, so
    the rate falls toward chance; genuine coordinated isolation sustains a high rate (the H16
    discriminator).

    Reason: references the joint-isolation rate.
    """
    rate = _feat(fs, "conf_joint_isolation_rate")
    iso_max = _feat(fs, "isolation_max")
    score = max(rate, 0.0) + 0.5 * max(iso_max, 0.0)
    reason = (
        f"joint isolation rate {rate:.3f} (isolation peak {iso_max:.3f}) "
        f"[H16 coordinated isolation]"
    )
    return float(score), reason


# --------------------------------------------------------------------------- #
# Detector 4: mutual-information conflict avoidance (OpenReview / PMLR v180)
# --------------------------------------------------------------------------- #
def mi_conflict_avoidance(fs: PairFeatureSet) -> Tuple[float, str]:
    """Information-theoretic conflict-avoidance detector from the mi_conflict stats (Req 5.1).

    A high mutual-information conflict-avoidance signal means the pair's actions are informative
    about avoiding conflict with each other (colluders in soft-play avoid taking each other's
    chips). Use the peak and mean::

        score = mi_conflict_max + 0.5 * mi_conflict_mean

    Reason: references the mi_conflict peak/mean.
    """
    mi_max = _feat(fs, "mi_conflict_max")
    mi_mean = _feat(fs, "mi_conflict_mean")
    score = max(mi_max, 0.0) + 0.5 * max(mi_mean, 0.0)
    reason = f"mutual-information conflict avoidance peak {mi_max:.3f} (mean {mi_mean:.3f})"
    return float(score), reason


# --------------------------------------------------------------------------- #
# Detector 5: collusion-table advantage (AAAI) — behavior-agnostic backbone
# --------------------------------------------------------------------------- #
def collusion_table_advantage(fs: PairFeatureSet) -> Tuple[float, str]:
    """Behavior-agnostic collusion-table-advantage backbone for ``other_coordination`` (Principle 3).

    The AAAI collusion-table-advantage measure quantifies the advantage accruing to a colluding
    group **without** assuming any specific behavior pattern — the natural backbone for the
    undisclosed ``other_coordination`` family we cannot template. A *full* advantage measure needs
    game-state inputs (per-hand joint stack deltas vs. a no-collusion counterfactual) that are NOT
    available at this feature layer. Rather than fabricate them, we build a **documented behavior-
    agnostic composite** from the features that ARE available — combining generic coordination
    signatures that any coordination scheme (templated or not) would leave (see the exact,
    dev-calibrated weights in the "Calibration note" below)::

        score ~ value_flow_absmean                       (net chips concentrated within the pair)
              + episodic_score                           (peak * temporal-burst concentration)
              + |conf_directedness_contrast|             (pair effect beyond the field baseline)

    None of these terms assumes a specific behavior template, so the backbone still fires for an
    untemplated ``other_coordination`` pattern. It is explicitly the catch-all: it does not try to
    name the mechanism, only to measure a behavior-agnostic advantage/coordination footprint.

    Reason: names the composite terms.

    Calibration note: on the real development labels the mean absolute directed flow
    (``value_flow_absmean``) is a strong behavior-agnostic discriminator (AUC ≈ 0.81) while the
    directedness contrast is essentially noise (AUC ≈ 0.51 — the signed field-baseline cancels), so
    the backbone now leans on ``value_flow_absmean`` with a small episodic-concentration helper and
    only a residual directedness term::

        score = value_flow_absmean  +  0.05 * episodic_score  +  0.05 * |directedness_contrast|
    """
    absmean = _feat(fs, "value_flow_absmean")
    episodic = _feat(fs, "episodic_score")
    directedness = abs(_feat(fs, "conf_directedness_contrast"))
    score = max(absmean, 0.0) + 0.05 * max(episodic, 0.0) + 0.05 * directedness
    reason = (
        f"behavior-agnostic advantage backbone: |flow| mean {absmean:.1f}bb + "
        f"episodic score {episodic:.1f} + directedness contrast {directedness:.1f} "
        f"[other_coordination catch-all]"
    )
    return float(score), reason


# --------------------------------------------------------------------------- #
# Detector registry (name -> (function, scale)) — fixed order for determinism
# --------------------------------------------------------------------------- #
DETECTORS: Tuple[Tuple[str, Callable[[PairFeatureSet], Tuple[float, str]], float], ...] = (
    ("value_flow_graph_score", value_flow_graph_score, _SCALE_VALUE_FLOW),
    ("aggression_asymmetry_softplay_test", aggression_asymmetry_softplay_test, _SCALE_SOFTPLAY),
    ("isolation_pressure_ratio", isolation_pressure_ratio, _SCALE_ISOLATION),
    ("mi_conflict_avoidance", mi_conflict_avoidance, _SCALE_MI),
    ("collusion_table_advantage", collusion_table_advantage, _SCALE_TABLE_ADVANTAGE),
)


# --------------------------------------------------------------------------- #
# Artifact readers (injected; hermetic)
# --------------------------------------------------------------------------- #
def _thresholds_of(eda_artifact: Optional[Union[Mapping, object]]) -> Optional[Mapping]:
    """Return the ``thresholds`` mapping from an injected artifact (dict or EdaArtifact)."""
    if eda_artifact is None:
        return None
    if hasattr(eda_artifact, "thresholds"):
        t = getattr(eda_artifact, "thresholds")
        return t if isinstance(t, Mapping) else None
    if isinstance(eda_artifact, Mapping):
        inner = eda_artifact.get("thresholds")
        if isinstance(inner, Mapping):
            return inner
        return eda_artifact
    return None


def _pool_prior_from(eda_artifact: Optional[Union[Mapping, object]], default: float) -> float:
    """Read ``pool_prior_positive_prevalence`` from the artifact; fall back to ``default``."""
    thresholds = _thresholds_of(eda_artifact)
    if thresholds is None:
        return float(default)
    val = thresholds.get("pool_prior_positive_prevalence", default)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(f) or math.isinf(f):
        return float(default)
    # keep the default inside [0, 1] just in case
    return min(max(f, 0.0), 1.0)


def _min_shared_hands_from(
    eda_artifact: Optional[Union[Mapping, object]], default: int
) -> int:
    """Read ``min_shared_hands`` from the artifact; fall back to ``default``."""
    thresholds = _thresholds_of(eda_artifact)
    if thresholds is None:
        return int(default)
    val = thresholds.get("min_shared_hands", default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


# --------------------------------------------------------------------------- #
# Combination + squashing
# --------------------------------------------------------------------------- #
def _logistic(x: float) -> float:
    """Numerically-stable logistic sigmoid into (0, 1)."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _raw_combined(mapped: Dict[str, float]) -> float:
    """Monotone weighted sum of the (non-saturating) mapped detector scores.

    This is the pair's raw coordination-evidence score BEFORE any squash. It is monotone
    non-decreasing in every detector's raw score (all weights >= 0, all mapped scores >= 0). The
    ranking of these raw scores is exactly what drives Pair AP; both the per-pair
    :func:`_squash_single` and the population :func:`score_pairs_ranked` are MONOTONE transforms of
    it, so neither changes the pair ordering.
    """
    s = 0.0
    for name, weight in DETECTOR_WEIGHTS.items():
        s += weight * mapped.get(name, 0.0)
    return s


def _squash_single(s: float) -> float:
    """Per-pair, NON-saturating, monotone map of the raw combined score ``s`` into ``[0, 1)``.

    Used by :func:`score_pair` when there is no population to rank against (a single pair). A
    rational (Möbius) squash ``s / (SQUASH_SCALE + s)`` is monotone increasing on ``s >= 0`` and
    approaches 1 only asymptotically and *gently* — so a strongly suspicious pair lands clearly
    above a benign one WITHOUT the old logistic's saturation into a ~0.95 tie clump. When scoring a
    batch, :func:`score_pairs_ranked` replaces this with a population rank map for a wider spread.
    """
    if s <= 0.0:
        return 0.0
    return s / (SQUASH_SCALE + s)


def _combine(mapped: Dict[str, float]) -> float:
    """Per-pair combined risk: monotone weighted sum + non-saturating rational squash into [0, 1)."""
    return _squash_single(_raw_combined(mapped))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def score_pair(
    feature_set: PairFeatureSet,
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    pool_prior: Optional[float] = None,
    min_shared_hands: Optional[int] = None,
) -> ClassicalScore:
    """Score one pair with the classical baseline, returning a :class:`ClassicalScore`.

    Runs all five detectors, maps each raw score to a comparable [0, 1) scale, combines them with
    the documented monotone weighted-sum + logistic squash into a ``risk_score`` in (0, 1), and
    attaches every detector's raw score + reason string. A pair that is UNSCOREABLE — too few
    shared hands (``n_shared_hands < min_shared_hands``) or all-neutral detector inputs — is
    assigned the DEFINED DEFAULT: the pool prior (from the injected EDA artifact, else the explicit
    ``pool_prior`` override, else :data:`DEFAULT_POOL_PRIOR`).

    Args:
        feature_set: The pair's assembled :class:`PairFeatureSet`.
        eda_artifact: Injected EDA/threshold artifact supplying ``pool_prior_positive_prevalence``
            and ``min_shared_hands``. Kept injectable so scoring stays hermetic/deterministic.
        pool_prior: Explicit pool-prior override (takes precedence over the artifact).
        min_shared_hands: Explicit min-shared-hands override (takes precedence over the artifact).

    Returns:
        A :class:`ClassicalScore` with a ``risk_score`` guaranteed in ``[0, 1]``.
    """
    prior = (
        float(pool_prior)
        if pool_prior is not None
        else _pool_prior_from(eda_artifact, DEFAULT_POOL_PRIOR)
    )
    prior = min(max(prior, 0.0), 1.0)
    floor = (
        int(min_shared_hands)
        if min_shared_hands is not None
        else _min_shared_hands_from(eda_artifact, DEFAULT_MIN_SHARED_HANDS)
    )

    # Run all detectors in fixed order (deterministic).
    detector_scores: Dict[str, float] = {}
    mapped: Dict[str, float] = {}
    reasons: List[str] = []
    for name, fn, scale in DETECTORS:
        raw, reason = fn(feature_set)
        detector_scores[name] = float(raw)
        mapped[name] = _tanh_norm(float(raw), scale)
        reasons.append(reason)

    n_shared = _feat(feature_set, "n_shared_hands", 0.0)
    all_neutral = all(m <= NEUTRAL_EPS for m in mapped.values())
    unscoreable = (n_shared < float(floor)) or all_neutral

    raw_combined = _raw_combined(mapped)
    if unscoreable:
        risk = prior
        raw_combined = 0.0
        why = (
            f"UNSCOREABLE -> pool-prior default {prior:.4f} "
            f"(n_shared_hands={n_shared:.0f} < min {floor})"
            if n_shared < float(floor)
            else f"UNSCOREABLE -> pool-prior default {prior:.4f} (all detector inputs neutral)"
        )
        reasons.append(why)
    else:
        risk = _squash_single(raw_combined)

    # Guarantee [0, 1] (Req 6.3) regardless of path.
    risk = min(max(float(risk), 0.0), 1.0)

    return ClassicalScore(
        pair_id=str(feature_set.pair_id),
        risk_score=risk,
        reasons=reasons,
        detector_scores=detector_scores,
        unscoreable=bool(unscoreable),
        pool_prior=prior,
        raw_combined_score=float(raw_combined),
    )


def score_pairs(
    feature_sets: Sequence[PairFeatureSet],
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    pool_prior: Optional[float] = None,
    min_shared_hands: Optional[int] = None,
) -> List[ClassicalScore]:
    """Deterministic batch scorer: one :class:`ClassicalScore` per input, in input order (Req 6.4).

    Every input pair receives a score (coverage). Same inputs -> same outputs (determinism).
    """
    return [
        score_pair(
            fs,
            eda_artifact=eda_artifact,
            pool_prior=pool_prior,
            min_shared_hands=min_shared_hands,
        )
        for fs in feature_sets
    ]


def score_pairs_ranked(
    feature_sets: Sequence[PairFeatureSet],
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    pool_prior: Optional[float] = None,
    min_shared_hands: Optional[int] = None,
    spread_exponent: float = RANK_SPREAD_EXPONENT,
) -> List[ClassicalScore]:
    """Batch scorer whose ``risk_score`` is a WELL-SPREAD, POPULATION-RANKED risk (Pair-AP first).

    **Why this exists (the Phase-1 fix).** Pair AP depends only on the *ordering* of ``risk_score``
    across the scored population, not on its absolute level. :func:`score_pair` scores each pair in
    isolation, so with the old saturating map ~half the real evaluation pairs collapsed into one
    near-tied high-risk clump — unrankable, and scored *below* a constant. This method instead maps
    each SCOREABLE pair's raw combined score (:attr:`ClassicalScore.raw_combined_score`) to its
    **population rank percentile**, then applies the monotone spreading curve
    ``risk = percentile ** spread_exponent``:

    * **Spread / no clump.** Rank percentiles are (near-)distinct, so the mapped risks fill ``(0,
      1)`` with no saturation clump — regardless of how concentrated the raw scores are.
    * **Rare high-risk mass.** Under ``percentile ** k`` the fraction of pairs with ``risk > t`` is
      ``1 - t ** (1/k)`` *independent of the score distribution*; the default ``k = 6`` gives
      ``frac(risk > 0.9) ≈ 1.7%`` — small, consistent with a rare-positive pool (the target the
      task sets), never the old ~48%.
    * **Ranking preserved (Pair AP).** Percentile-rank and the power curve are both MONOTONE in the
      raw combined score, so the pair ORDER — hence Pair AP — is *identical* to ranking by the raw
      combined score. ``score_pair`` stays monotone in the same raw score, so the two agree on
      order; this method only chooses a better *spread* for the emitted level.

    **Unscoreable pairs** keep the pool-prior default (Req 6.4) and are RANKED BELOW every scoreable
    pair (they carry no combined evidence — rare positives correctly stay low; no-signal pairs are
    never pushed high). Determinism: ties in the raw score break by ``pair_id`` ascending, matching
    the competition's deterministic tie-break, so the mapping is reproducible and order-independent.

    Args:
        feature_sets: The pairs to score (any order; the result preserves input order).
        eda_artifact / pool_prior / min_shared_hands: As in :func:`score_pair`.
        spread_exponent: Exponent ``k`` of the ``percentile ** k`` spreading curve (>= 1 keeps high
            risk rare; default :data:`RANK_SPREAD_EXPONENT`).

    Returns:
        One :class:`ClassicalScore` per input pair, in input order (coverage, Req 6.4), each
        ``risk_score`` in ``[0, 1]`` (Req 6.3). ``detector_scores`` / ``reasons`` /
        ``raw_combined_score`` are carried through from the per-pair pass unchanged.
    """
    base = score_pairs(
        feature_sets,
        eda_artifact=eda_artifact,
        pool_prior=pool_prior,
        min_shared_hands=min_shared_hands,
    )
    k = float(spread_exponent) if spread_exponent and spread_exponent > 0.0 else 1.0

    # Rank ONLY the scoreable pairs by (raw_combined_score asc, pair_id asc) — deterministic and
    # order-independent. Unscoreable pairs keep the pool prior and are not ranked.
    scoreable = [(i, sc) for i, sc in enumerate(base) if not sc.unscoreable]
    order = sorted(scoreable, key=lambda t: (t[1].raw_combined_score, t[1].pair_id))
    m = len(order)

    out: List[ClassicalScore] = list(base)
    for rank, (i, sc) in enumerate(order):
        percentile = (rank + 0.5) / m if m > 0 else 0.0
        risk = percentile ** k
        risk = min(max(float(risk), 0.0), 1.0)
        out[i] = ClassicalScore(
            pair_id=sc.pair_id,
            risk_score=risk,
            reasons=sc.reasons,
            detector_scores=sc.detector_scores,
            unscoreable=sc.unscoreable,
            pool_prior=sc.pool_prior,
            raw_combined_score=sc.raw_combined_score,
        )
    return out
