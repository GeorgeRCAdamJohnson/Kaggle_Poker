"""Episodic Aggregation (task 10.1, Requirement 4, Design "Episodic Aggregation").

Coordination is **episodic** (Req 4, Glossary "Episodic_Activity"): a colluding pair mixes a
handful of manipulated hands into a long stream of ordinary play. Averaging therefore *dilutes*
the very signal we are hunting. This module turns a pair's per-hand :class:`HandSignal` stream
into pair-level features that (a) **preserve peaks**, (b) reward **temporal concentration** of
the high-signal hands, and (c) are **count-robust** so a 12-hand pair and a 4,000-hand pair are
comparable.

Everything here is deterministic, ``NOT_APPLICABLE``-aware (the zero-denominator sentinel from
:mod:`poker_collusion.types` is *excluded* from a signal's statistics, never coerced to ``0``),
and reads its calibrated prior only from the persisted EDA threshold artifact (injected for
tests; the real ``eda_artifact.json`` is never required at import).

--------------------------------------------------------------------------------------------
Formulas
--------------------------------------------------------------------------------------------

Peak-preserving statistics (Req 4.1)
====================================
For a scalar signal ``x`` over a pair's shared hands, dropping ``NOT_APPLICABLE`` hands, we
report ``mean``, ``max``, ``p90`` and ``p95`` (linear-interpolation percentiles, matching
``numpy.percentile``'s default). Peaks (``max``/``p95``) matter more than the mean because a
pair that manipulated *some* hands shows up in the tail, not the average.

* ``value_flow`` is *directed/signed* (``value_flow(A,B) > 0`` means chips moved A->B). We
  therefore capture three complementary views: ``value_flow_max_pos`` (largest positive directed
  transfer), ``value_flow_max_abs`` (largest transfer in *either* direction), and
  ``value_flow_mean`` (signed mean, which reveals a persistent directional bias). We also keep
  ``value_flow_absmean`` and the tail (``p90``/``p95``) of ``|value_flow|``.
* ``aggression_asymmetry`` and ``isolation`` may be ``NOT_APPLICABLE`` in a hand; those hands
  are excluded from that signal's stats. If *every* hand is ``NOT_APPLICABLE`` the stats are the
  neutral fills documented on :func:`_stat_block` (never a raised error, never a coerced 0).

Temporal concentration / burst features (Req 4.2, 4.3)
======================================================
Order the pair's shared hands chronologically (by ``started_at`` when present, else by
``hand_id``; ties broken by ``hand_id`` — deterministic). Mark each hand "flagged" iff any of
its :attr:`HandSignal.behavior_action_flags` is True. Let ``n`` be the number of shared hands
and ``k`` the number of flagged hands. We compute:

* ``flagged_count`` = ``k`` and ``flagged_rate`` = ``k / n``.
* ``max_burst`` = the length of the **longest maximal run of consecutive flagged hands**. This
  is the core temporal-concentration measure and the one that carries the Property-4 guarantee
  (see below). Spreading ``k`` flags apart yields ``max_burst = 1``; packing them together
  yields ``max_burst = k``.
* ``burst_count`` = the number of maximal flagged runs (fewer, longer runs = more concentrated).
* ``gap_gini`` = the Gini coefficient of the inter-flag gaps (positions between successive
  flagged hands). ``0`` = perfectly even spacing, ``-> 1`` = one tight cluster. Reported as an
  auxiliary concentration view; it is **not** used in the monotone episodic score (Gini is not
  monotone under arbitrary "concentration" moves), so it never endangers Property 4.
* ``concentration_ratio`` = ``max_burst / k`` in ``(0, 1]`` (0 when ``k == 0``): the fraction of
  the flagged mass sitting in the single densest episode.

Episodic coordination score (Req 4.1 + 4.2, carries Property 4)
===============================================================
The single feature the design's Property 4 constrains ("episodic concentration never lowers the
coordination signal"):

    peak            = max over flagged hands of the hand's signal magnitude
                      (max |value_flow|, or the peak of any configured signal); 0 if k == 0
    episodic_score  = peak * (1 + BURST_GAIN * (max_burst - 1))

``max_burst >= 1`` whenever ``k >= 1``, so the multiplier is ``>= 1`` and *increases* with
``max_burst``. ``peak`` depends only on *which* hands are flagged and their magnitudes — not on
their arrangement in time. Concentrating the same flagged hands in time can only *raise*
``max_burst`` (see the monotonicity proof), so ``episodic_score`` can only rise or stay equal.
``BURST_GAIN >= 0`` is a fixed, documented constant.

--------------------------------------------------------------------------------------------
Monotonicity guarantee for Property 4
--------------------------------------------------------------------------------------------
Claim: holding the *set* of flagged hands (hence ``peak`` and ``k``) fixed, any rearrangement
that moves flagged hands closer together (a "concentration" move) yields ``max_burst_new >=
max_burst_old``, hence ``episodic_score_new >= episodic_score_old``.

Why ``max_burst`` is monotone under concentration: ``max_burst`` is the length of the longest
run of consecutive flagged positions. The most-spread arrangement (flags separated by >=1
non-flagged hand) has ``max_burst = 1``; the fully-concentrated arrangement (all ``k`` flags
adjacent) has ``max_burst = k`` — the maximum attainable given ``k`` flags. Any move that
merges two adjacent runs into one, or extends a run, never shortens the longest run, so
``max_burst`` is non-decreasing. Because ``episodic_score`` is a non-decreasing (affine, positive
slope) function of ``max_burst`` with a peak term that is invariant to arrangement, the episodic
score is non-decreasing under concentration. This is exactly Property 4, and the ``>=`` (not
strict ``>``) matches the property's wording. Task 10.2's Hypothesis test exercises it across
random equal-mass timelines.

Count-robust normalization via empirical-Bayes shrinkage (Req 4.4)
==================================================================
A per-pair *rate* estimate (e.g. ``flagged_rate``) from a pair with 6 shared hands is far noisier
than one from a pair with 4,000. We shrink each rate toward the **pool prior**
``p0 = pool_prior_positive_prevalence`` (read from the EDA artifact) using a James-Stein-style
Beta-Binomial posterior mean:

    shrunk_rate = (k + ALPHA * p0) / (n + ALPHA)

where ``ALPHA > 0`` is the prior strength (pseudo-count). As ``n -> 0`` the estimate -> ``p0``
(few-hand pairs are pulled to the prior); as ``n -> inf`` it -> the empirical rate ``k / n``
(many-hand pairs keep their own signal). The *shrinkage weight toward the prior*,
``ALPHA / (n + ALPHA)``, therefore **decreases monotonically as ``n`` grows**, exactly as Req 4.4
requires. ``ALPHA`` defaults to ``DEFAULT_PRIOR_STRENGTH`` and may be overridden per call.

--------------------------------------------------------------------------------------------
Determinism
--------------------------------------------------------------------------------------------
No randomness, no wall-clock, no dict-iteration-order dependence: hands are sorted by an explicit
(started_at, hand_id) key and all reductions are order-invariant or applied to the sorted stream.
Given identical inputs, :func:`aggregate_pair` returns identical values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from poker_collusion.config import DISCLOSED_FAMILY_NAMES
from poker_collusion.types import NOT_APPLICABLE, HandSignal, Sentinel

__all__ = [
    "BURST_GAIN",
    "DEFAULT_PRIOR_STRENGTH",
    "EpisodicFeatures",
    "aggregate_pair",
    "empirical_bayes_shrink",
    "gap_gini",
    "longest_flagged_burst",
    "peak_preserving_stats",
    "prior_from_artifact",
]

#: Fixed slope of the burst multiplier in the episodic score. ``>= 0`` guarantees the multiplier
#: ``1 + BURST_GAIN * (max_burst - 1)`` is non-decreasing in ``max_burst`` (Property 4).
BURST_GAIN: float = 0.5

#: Default empirical-Bayes prior strength (pseudo-count) ``ALPHA`` for :func:`empirical_bayes_shrink`.
#: Larger -> more shrinkage toward the prior for a given ``n``.
DEFAULT_PRIOR_STRENGTH: float = 5.0

#: Neutral fill for a signal whose every hand was NOT_APPLICABLE (documented; never a coerced 0).
_NEUTRAL_FILL: float = 0.0


# --------------------------------------------------------------------------- #
# Result carrier
# --------------------------------------------------------------------------- #
@dataclass
class EpisodicFeatures:
    """Pair-level episodic aggregation of a :class:`HandSignal` stream.

    ``features`` is a flat, JSON-serialisable ``{name: float}`` map (the values later folded into
    :class:`~poker_collusion.types.PairFeatureSet`). The named fields expose the headline
    quantities used by tests and downstream detectors. ``phase`` records which slice
    (``development``/``evaluation``/``all``) was aggregated.
    """

    pair_id: str
    phase: str
    n_hands: int
    flagged_count: int
    max_burst: int
    burst_count: int
    episodic_score: float
    features: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Percentiles / peak-preserving statistics (Req 4.1)
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence (numpy default method).

    ``q`` is in ``[0, 100]``. Matches ``numpy.percentile(vals, q)`` without importing numpy so the
    module stays dependency-light and byte-deterministic.
    """
    n = len(sorted_vals)
    if n == 0:
        return _NEUTRAL_FILL
    if n == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _applicable_values(
    signals: Sequence[HandSignal],
    attr: str,
) -> List[float]:
    """Collect the finite float values of ``attr`` across signals, EXCLUDING NOT_APPLICABLE.

    The zero-denominator sentinel is dropped from the statistic (Req 3.5 / 4.1 documentation) —
    it is never coerced to 0, which would fabricate a spurious low value.
    """
    out: List[float] = []
    for s in signals:
        v = getattr(s, attr)
        if v is NOT_APPLICABLE:
            continue
        if isinstance(v, bool):  # guard: bools are ints in Python; not a signal value here
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def peak_preserving_stats(values: Sequence[float]) -> Dict[str, float]:
    """Return ``{mean, max, p90, p95}`` for a list of applicable (non-sentinel) values (Req 4.1).

    An empty input (e.g. every hand was NOT_APPLICABLE) yields the neutral fill for every stat —
    documented, never an error. Peaks are the tail statistics that survive episodic dilution.
    """
    if not values:
        return {"mean": _NEUTRAL_FILL, "max": _NEUTRAL_FILL, "p90": _NEUTRAL_FILL, "p95": _NEUTRAL_FILL}
    ordered = sorted(float(v) for v in values)
    return {
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "p90": _percentile(ordered, 90.0),
        "p95": _percentile(ordered, 95.0),
    }


# --------------------------------------------------------------------------- #
# Chronological ordering & flag stream (Req 4.2, 4.3)
# --------------------------------------------------------------------------- #
def _chrono_key(s: HandSignal) -> Tuple[float, str]:
    """Deterministic chronological sort key: (started_at, hand_id).

    ``started_at`` is used when the signal (or its optional attribute) carries it; otherwise the
    hand_id alone orders the stream. hand_id is coerced to a zero-padded string is unnecessary —
    a numeric hand_id sorts numerically via the tuple's second slot only as a tie-break, so we
    order primarily by started_at then by hand_id's natural order. To keep a single comparable
    type we map started_at to float and hand_id to str.
    """
    started = getattr(s, "started_at", None)
    try:
        started_f = float(started) if started is not None else 0.0
    except (TypeError, ValueError):
        started_f = 0.0
    return (started_f, str(s.hand_id))


def _is_flagged(s: HandSignal) -> bool:
    """A hand is 'flagged' iff any disclosed-family behavior-action flag is True."""
    flags = s.behavior_action_flags or {}
    return any(bool(flags.get(fam, False)) for fam in DISCLOSED_FAMILY_NAMES) or any(
        bool(v) for v in flags.values()
    )


def _flag_stream(signals: Sequence[HandSignal]) -> List[bool]:
    """Chronologically ordered boolean flag stream for the pair (deterministic)."""
    ordered = sorted(signals, key=_chrono_key)
    return [_is_flagged(s) for s in ordered]


def longest_flagged_burst(flags: Sequence[bool]) -> Tuple[int, int]:
    """Return ``(max_burst, burst_count)`` for a boolean flag stream.

    ``max_burst`` is the length of the longest maximal run of consecutive True values;
    ``burst_count`` is the number of maximal True runs. Both are 0 on an all-False / empty stream.

    Monotonicity (Property 4): for a fixed number ``k`` of True values, the minimum possible
    ``max_burst`` is 1 (all isolated) and the maximum is ``k`` (all adjacent). Merging/extending
    runs never shortens the longest run, so ``max_burst`` is non-decreasing under concentration.
    """
    max_burst = 0
    burst_count = 0
    current = 0
    for f in flags:
        if f:
            if current == 0:
                burst_count += 1
            current += 1
            if current > max_burst:
                max_burst = current
        else:
            current = 0
    return max_burst, burst_count


def gap_gini(flags: Sequence[bool]) -> float:
    """Gini coefficient of the gaps between successive flagged positions (auxiliary view).

    Gaps are the differences between the indices of consecutive True values. Even spacing -> 0;
    a single tight cluster (all gaps 1 except one large tail gap) -> toward 1. Returns 0.0 when
    there are fewer than two flagged hands (no gap distribution to speak of). This measure is
    reported for diagnostics only and deliberately kept OUT of :attr:`episodic_score` because the
    Gini of gaps is not monotone under every concentration move; keeping it out protects Property
    4.
    """
    idx = [i for i, f in enumerate(flags) if f]
    if len(idx) < 2:
        return 0.0
    gaps = [float(idx[i + 1] - idx[i]) for i in range(len(idx) - 1)]
    total = sum(gaps)
    if total <= 0.0:
        return 0.0
    gaps.sort()
    n = len(gaps)
    # Gini = (2 * sum(i * g_i) / (n * sum g)) - (n + 1) / n, with i 1-based over sorted gaps.
    weighted = sum((i + 1) * g for i, g in enumerate(gaps))
    gini = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    # numerical clamp into [0, 1]
    if gini < 0.0:
        return 0.0
    if gini > 1.0:
        return 1.0
    return gini


# --------------------------------------------------------------------------- #
# Count-robust normalization (Req 4.4)
# --------------------------------------------------------------------------- #
def empirical_bayes_shrink(
    k: float,
    n: float,
    prior: float,
    alpha: float = DEFAULT_PRIOR_STRENGTH,
) -> float:
    """Beta-Binomial posterior-mean shrinkage of a rate ``k/n`` toward ``prior`` (Req 4.4).

        shrunk = (k + alpha * prior) / (n + alpha)

    The weight placed on the prior is ``alpha / (n + alpha)``, which **decreases as ``n`` grows**
    (few-hand pairs are pulled toward the prior; many-hand pairs keep their empirical rate). With
    ``n == 0`` the result is exactly ``prior``. ``alpha`` must be positive; a non-positive
    ``alpha`` disables shrinkage (returns ``k/n`` or ``prior`` when ``n == 0``).
    """
    p0 = float(prior)
    a = float(alpha)
    nn = float(n)
    if a <= 0.0:
        return (float(k) / nn) if nn > 0.0 else p0
    return (float(k) + a * p0) / (nn + a)


def prior_from_artifact(
    eda_artifact: Optional[Union[Mapping, object]],
    *,
    default: float = 0.0,
) -> float:
    """Read ``pool_prior_positive_prevalence`` from an injected EDA artifact.

    Accepts either the raw JSON dict (``{"thresholds": {...}}`` or a bare ``thresholds`` dict), an
    :class:`~poker_collusion.features.eda.EdaArtifact` (via its ``thresholds`` attribute), or
    ``None`` (returns ``default``). Never imports/reads the real ``eda_artifact.json`` — the
    artifact is injected so tests stay hermetic (Req 4.4 note).
    """
    if eda_artifact is None:
        return float(default)
    thresholds: Optional[Mapping] = None
    if hasattr(eda_artifact, "thresholds"):
        thresholds = getattr(eda_artifact, "thresholds")  # EdaArtifact
    elif isinstance(eda_artifact, Mapping):
        if "thresholds" in eda_artifact and isinstance(eda_artifact["thresholds"], Mapping):
            thresholds = eda_artifact["thresholds"]
        else:
            thresholds = eda_artifact
    if not isinstance(thresholds, Mapping):
        return float(default)
    val = thresholds.get("pool_prior_positive_prevalence", default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


# --------------------------------------------------------------------------- #
# Top-level aggregation
# --------------------------------------------------------------------------- #
def _phase_filter(signals: Sequence[HandSignal], phase: Optional[str]) -> List[HandSignal]:
    """Return the signals belonging to ``phase`` (``None`` / ``"all"`` keeps everything)."""
    if phase is None or phase == "all":
        return list(signals)
    return [s for s in signals if s.phase == phase]


def aggregate_pair(
    pair_id: str,
    signals: Sequence[HandSignal],
    *,
    phase: Optional[str] = None,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    prior: Optional[float] = None,
    alpha: float = DEFAULT_PRIOR_STRENGTH,
) -> EpisodicFeatures:
    """Aggregate a pair's :class:`HandSignal` stream into episodic, peak-preserving features.

    Args:
        pair_id: The pair identifier (copied onto the result).
        signals: The pair's per-hand signals (any order; sorted internally for determinism).
        phase: Optional slice to aggregate (``"development"``/``"evaluation"``/``None`` for all).
        eda_artifact: Injected EDA artifact (dict or :class:`EdaArtifact`) supplying the pool
            prior for shrinkage. Ignored when ``prior`` is given explicitly.
        prior: Explicit pool prior override (takes precedence over ``eda_artifact``). Defaults to
            0.0 when neither is supplied.
        alpha: Empirical-Bayes prior strength (pseudo-count) for shrinkage (Req 4.4).

    Returns:
        An :class:`EpisodicFeatures` with the flat ``features`` map plus headline fields. Empty
        input yields an all-neutral, zero-count result (never an error).
    """
    subset = _phase_filter(signals, phase)
    phase_label = phase or "all"
    n = len(subset)

    # ---- peak-preserving stats per signal (Req 4.1), NOT_APPLICABLE-excluded ---------------- #
    # value_flow is signed/directed: derive positive-only and absolute views.
    vf_signed = _applicable_values(subset, "value_flow")
    vf_pos = [v for v in vf_signed if v > 0.0]
    vf_abs = [abs(v) for v in vf_signed]
    aa_vals = _applicable_values(subset, "aggression_asymmetry")
    iso_vals = _applicable_values(subset, "isolation")
    mi_vals = _applicable_values(subset, "mi_conflict")

    features: Dict[str, float] = {}

    def _emit(name: str, stats: Dict[str, float]) -> None:
        for stat_name, val in stats.items():
            features[f"{name}_{stat_name}"] = float(val)

    _emit("value_flow_abs", peak_preserving_stats(vf_abs))
    _emit("aggression_asymmetry", peak_preserving_stats(aa_vals))
    _emit("isolation", peak_preserving_stats(iso_vals))
    _emit("mi_conflict", peak_preserving_stats(mi_vals))

    # directed value-flow specific headline stats (Req 4.1: max positive, max |flow|, mean).
    features["value_flow_max_pos"] = max(vf_pos) if vf_pos else _NEUTRAL_FILL
    features["value_flow_max_abs"] = max(vf_abs) if vf_abs else _NEUTRAL_FILL
    features["value_flow_mean"] = (sum(vf_signed) / len(vf_signed)) if vf_signed else _NEUTRAL_FILL
    features["value_flow_absmean"] = (sum(vf_abs) / len(vf_abs)) if vf_abs else _NEUTRAL_FILL

    # ---- temporal concentration / burst features (Req 4.2, 4.3) ----------------------------- #
    flags = _flag_stream(subset)
    k = sum(1 for f in flags if f)
    max_burst, burst_count = longest_flagged_burst(flags)
    concentration_ratio = (max_burst / k) if k > 0 else 0.0
    gini = gap_gini(flags)

    features["flagged_count"] = float(k)
    features["flagged_rate"] = (k / n) if n > 0 else 0.0
    features["max_burst"] = float(max_burst)
    features["burst_count"] = float(burst_count)
    features["concentration_ratio"] = float(concentration_ratio)
    features["gap_gini"] = float(gini)

    # ---- episodic coordination score (carries Property 4) ----------------------------------- #
    # peak = magnitude of the strongest FLAGGED hand's |value_flow|; falls back to overall max
    # |value_flow| only when no hand is flagged (score then 0 via the multiplier being anchored to
    # max_burst == 0). We anchor the peak to flagged hands so the score reflects the episode.
    ordered = sorted(subset, key=_chrono_key)
    flagged_abs_vf = [
        abs(float(s.value_flow))
        for s in ordered
        if _is_flagged(s) and not isinstance(s.value_flow, bool) and isinstance(s.value_flow, (int, float))
    ]
    peak = max(flagged_abs_vf) if flagged_abs_vf else 0.0
    multiplier = 1.0 + BURST_GAIN * float(max(max_burst - 1, 0))
    episodic_score = peak * multiplier
    features["episodic_peak"] = float(peak)
    features["episodic_multiplier"] = float(multiplier)
    features["episodic_score"] = float(episodic_score)

    # ---- count-robust shrinkage of the flagged rate toward the pool prior (Req 4.4) --------- #
    pool_prior = float(prior) if prior is not None else prior_from_artifact(eda_artifact, default=0.0)
    shrunk = empirical_bayes_shrink(k=float(k), n=float(n), prior=pool_prior, alpha=alpha)
    features["flagged_rate_shrunk"] = float(shrunk)
    features["shrinkage_weight_prior"] = float(alpha / (n + alpha)) if alpha > 0.0 else 0.0
    features["pool_prior_used"] = float(pool_prior)

    return EpisodicFeatures(
        pair_id=pair_id,
        phase=phase_label,
        n_hands=n,
        flagged_count=k,
        max_burst=max_burst,
        burst_count=burst_count,
        episodic_score=float(episodic_score),
        features=features,
    )
