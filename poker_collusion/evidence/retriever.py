"""Evidence hand retrieval and ranking (Requirement 8, Design ``Evidence_Retriever``).

For one evaluation :class:`~poker_collusion.types.Pair` this module selects the up-to-five
evaluation-period shared hands that will be submitted as supporting evidence, ranks them, and
pads the five submission slots. It is deliberately **decoupled from I/O**: the core selector
:func:`select_evidence` operates on an in-memory list of :class:`~poker_collusion.types.HandSignal`
objects (the pair's evaluation-period shared hands with their computed signals), so it is fully
unit-testable on synthetic signals. A thin convenience helper :func:`retrieve_evidence_for_pair`
pulls those signals via the :class:`DataLoader` + signal computation for real runs, but no core
selection logic depends on it.

Hard validity gate (Req 8.1-8.4)
--------------------------------
A candidate :class:`HandSignal` qualifies as evidence **only if all** of the following hold; a
hand failing any clause is NEVER selected:

1. **Both players present** (Req 8.1, 8.2) - the hand is a *Shared_Hand* containing both members
   of the pair. Every :class:`HandSignal` is computed for a co-seated pair, so this is guaranteed
   by construction upstream; the gate still enforces it via the caller-supplied
   ``is_shared`` predicate (default: True, i.e. trust the upstream shared-hand partition) so a
   caller that passes non-shared hands cannot leak them into evidence.
2. **Evaluation period** (Req 8.3) - ``signal.phase == "evaluation"``. Any development-period hand
   is rejected.
3. **At least one behavior-specific public action** (Req 8.4) - at least one of the disclosed
   family flags in ``signal.behavior_action_flags`` (``directed_transfer`` / ``soft_play`` /
   ``coordinated_isolation``, the per-family public-action signatures reproduced from Discovery
   workstream C, :data:`FAMILY_SIGNATURES`) is ``True``. A hand whose collusion indication would
   derive solely from latent scenario activation (no public behavior-specific action) has all
   flags ``False`` and is rejected (Req 8.4, dossier INV-1).

Evidence-strength ranking (Req 8.5)
-----------------------------------
Qualifying hands are ordered by a documented **evidence-strength score** descending, with ties
broken by **ascending hand id** so ordering is deterministic and repeatable. The strength score is
a combined function of the hand's signals, taking the maximum family-specific contribution over
the families whose public-action flag fired for that hand:

    strength(hand) = max over flagged families f of  contribution_f(hand)

    contribution_directed_transfer     = |value_flow|
        (magnitude of the directed net chip transfer, Req 3.1 - a bigger dump is stronger)
    contribution_coordinated_isolation = isolation                  (0.0 if NOT_APPLICABLE)
        (how lopsidedly the pair ganged up on third players, Req 3.3 - higher is stronger)
    contribution_soft_play             = 1.0 - aggression_asymmetry  (0.0 if NOT_APPLICABLE)
        (how much partner-directed aggression was declined, Req 3.2 - a lower asymmetry, i.e.
         more avoidance, is stronger, so we invert it into [0, 1])

Only families whose flag is ``True`` for the hand contribute, so the strength always reflects an
*observed* public tell. A hand with no flag never reaches ranking (it fails the gate). The score is
a pure function of the hand's signals - no randomness, no wall-clock, no row-order dependence - so
running twice on the same signals yields the same ranking (determinism, Req 8.5).

Slot filling, de-duplication, completeness (Req 8.6-8.8)
--------------------------------------------------------
The top ``MAX_EVIDENCE`` (=5) qualifying hands' ids fill positions 1..5; any remaining positions are
filled with the literal :data:`~poker_collusion.config.NO_EVIDENCE`. Duplicate hand ids are removed
before ranking (keeping the first occurrence), so no hand id repeats within a pair (Req 8.8). The
returned list is always exactly length 5 with every cell non-empty - a hand id or ``NO_EVIDENCE``
(Req 8.6, 8.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from poker_collusion.config import DISCLOSED_FAMILY_NAMES, NO_EVIDENCE
from poker_collusion.types import HandSignal, Sentinel

__all__ = [
    "MAX_EVIDENCE",
    "EVALUATION_PHASE",
    "EvidenceSelection",
    "evidence_strength",
    "is_valid_evidence",
    "select_evidence",
    "retrieve_evidence_for_pair",
]

#: Number of evidence positions per pair in the submission (Req 8.1, 8.6, 8.7).
MAX_EVIDENCE: int = 5

#: The phase token marking an Evaluation_Period hand (matches ``DataLoader``/``HandSignal.phase``).
EVALUATION_PHASE: str = "evaluation"


@dataclass(frozen=True)
class EvidenceSelection:
    """Result of selecting evidence for one pair.

    Attributes:
        pair_id: The pair the evidence belongs to.
        slots: Exactly :data:`MAX_EVIDENCE` entries, each a hand id (as ``str``) in ranked order
            or the literal :data:`~poker_collusion.config.NO_EVIDENCE`. Every cell is non-empty
            (Req 8.6, 8.7) and no hand id repeats (Req 8.8).
        strengths: Per-slot evidence-strength score aligned with ``slots``; ``None`` for a
            ``NO_EVIDENCE`` slot. Provided for downstream ranking/inspection.
    """

    pair_id: str
    slots: List[str]
    strengths: List[Optional[float]]


def _as_float_or_zero(value: object) -> float:
    """Coerce a signal value to float; the ``NOT_APPLICABLE`` sentinel (and None) become 0.0.

    The ratio signals (``aggression_asymmetry``, ``isolation``) may carry the ``NOT_APPLICABLE``
    sentinel on a zero denominator (Req 3.5). A family whose signal is not applicable contributes
    nothing to that family's strength, so it maps to 0.0 here.
    """
    if value is None or isinstance(value, Sentinel):
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _flagged_families(signal: HandSignal) -> List[str]:
    """Disclosed families whose public-action flag fired for this hand (order-stable)."""
    flags = signal.behavior_action_flags or {}
    return [fam for fam in DISCLOSED_FAMILY_NAMES if bool(flags.get(fam, False))]


def evidence_strength(signal: HandSignal) -> float:
    """Documented evidence-strength score for a hand (Req 8.5).

    ``strength = max`` over the families whose public-action flag fired of that family's
    contribution:

      * ``directed_transfer``     -> ``|value_flow|``
      * ``coordinated_isolation`` -> ``isolation``            (0.0 when NOT_APPLICABLE)
      * ``soft_play``             -> ``1.0 - aggression_asymmetry`` (0.0 when NOT_APPLICABLE)

    A hand with no fired flag has no contributing family and scores ``0.0`` (it would already have
    failed the validity gate, so this only matters if strength is requested directly). The score is
    a pure, deterministic function of the hand's signals.
    """
    contributions: List[float] = []
    for fam in _flagged_families(signal):
        if fam == "directed_transfer":
            contributions.append(abs(_as_float_or_zero(signal.value_flow)))
        elif fam == "coordinated_isolation":
            contributions.append(_as_float_or_zero(signal.isolation))
        elif fam == "soft_play":
            contributions.append(1.0 - _as_float_or_zero(signal.aggression_asymmetry))
    if not contributions:
        return 0.0
    return max(contributions)


def is_valid_evidence(
    signal: HandSignal,
    *,
    is_shared: bool = True,
) -> bool:
    """Hard validity gate (Req 8.1-8.4): does this hand qualify as evidence?

    Returns True only when ALL clauses hold:
      1. ``is_shared`` (both players present, Req 8.1/8.2) - defaults True (trust upstream
         shared-hand partition); pass False for a hand that is not a shared hand of the pair.
      2. ``signal.phase == "evaluation"`` (Req 8.3) - development-period hands are rejected.
      3. at least one disclosed-family public-action flag is True (Req 8.4) - a latent-only hand
         with all flags False is rejected.
    """
    if not is_shared:
        return False
    if signal.phase != EVALUATION_PHASE:
        return False
    return len(_flagged_families(signal)) > 0


def select_evidence(
    pair_id: str,
    candidates: Sequence[HandSignal],
    *,
    is_shared: Optional[Callable[[HandSignal], bool]] = None,
    strength_fn: Optional[Callable[[HandSignal], float]] = None,
) -> EvidenceSelection:
    """Select and rank the 5 evidence slots for a pair from candidate hand signals.

    This is the I/O-decoupled core (Design ``Evidence_Retriever``). ``candidates`` is the pair's
    evaluation-period shared hands with computed signals (a caller may pass a superset - e.g.
    development hands or non-shared hands - and the hard validity gate will filter them out).

    Steps:
      1. **De-duplicate** by hand id (Req 8.8), keeping the first occurrence in input order.
      2. **Validity gate** (Req 8.1-8.4): keep only hands passing :func:`is_valid_evidence`.
      3. **Rank** by the strength score (``strength_fn``) descending, ties broken by ascending
         hand id (Req 8.5) - deterministic and independent of input order.
      4. **Fill** the top :data:`MAX_EVIDENCE` ids into slots 1..5; pad remaining slots with
         :data:`~poker_collusion.config.NO_EVIDENCE` (Req 8.6); every cell non-empty (Req 8.7).

    Args:
        pair_id: The pair identifier.
        candidates: Candidate :class:`HandSignal` list for the pair.
        is_shared: Optional predicate ``signal -> bool`` overriding the both-players-present clause
            per hand. When ``None`` (default) every candidate is trusted as a shared hand (the
            upstream ``shared_hands`` partition guarantees it).
        strength_fn: Optional per-hand strength scorer ``signal -> float`` used ONLY for the
            ranking ORDER among already-valid hands (step 3) and the reported per-slot strengths.
            When ``None`` (default) the heuristic :func:`evidence_strength` is used, so all existing
            behavior is unchanged. The hard validity gate (step 2), de-duplication (step 1),
            slot-fill (step 4), and the deterministic ascending-hand-id tie-break are INDEPENDENT of
            ``strength_fn`` and remain identical regardless of which scorer is supplied. A learned
            scorer (e.g. :meth:`poker_collusion.evidence.learned_ranker.LearnedEvidenceRanker.strength`)
            plugs in here without duplicating any selection logic.

    Returns:
        An :class:`EvidenceSelection` with exactly 5 slots and aligned per-slot strengths.
    """
    score = evidence_strength if strength_fn is None else strength_fn
    # (1) De-duplicate by hand id, preserving first occurrence (Req 8.8).
    seen: set = set()
    unique: List[HandSignal] = []
    for sig in candidates:
        hid = str(sig.hand_id)
        if hid in seen:
            continue
        seen.add(hid)
        unique.append(sig)

    # (2) Hard validity gate (Req 8.1-8.4).
    valid: List[HandSignal] = [
        sig
        for sig in unique
        if is_valid_evidence(
            sig,
            is_shared=(True if is_shared is None else bool(is_shared(sig))),
        )
    ]

    # (3) Rank: strength desc, hand id asc (Req 8.5). Sort key negates strength for descending
    # while keeping ascending hand id as the deterministic tie-break. Hand ids are compared by a
    # (numeric-if-possible, string) key so "10" sorts after "2" like the integer ids they are.
    def _hand_key(hid: str) -> Tuple[int, float, str]:
        try:
            return (0, float(hid), "")
        except (TypeError, ValueError):
            return (1, 0.0, hid)

    ranked = sorted(
        valid,
        key=lambda s: (-float(score(s)), _hand_key(str(s.hand_id))),
    )

    # (4) Fill slots and pad with NO_EVIDENCE (Req 8.6, 8.7).
    slots: List[str] = []
    strengths: List[Optional[float]] = []
    for sig in ranked[:MAX_EVIDENCE]:
        slots.append(str(sig.hand_id))
        strengths.append(float(score(sig)))
    while len(slots) < MAX_EVIDENCE:
        slots.append(NO_EVIDENCE)
        strengths.append(None)

    return EvidenceSelection(pair_id=pair_id, slots=slots, strengths=strengths)


def retrieve_evidence_for_pair(
    pair_signals: Sequence[HandSignal],
    pair_id: Optional[str] = None,
) -> EvidenceSelection:
    """Thin, I/O-free convenience wrapper: pick evidence from a pair's computed signals.

    Callers that already have the pair's evaluation-period shared-hand :class:`HandSignal` list
    (e.g. from ``compute_pair_signals`` over the ``DataLoader.shared_hands(pair).evaluation``
    partition) can call this directly. It simply forwards to :func:`select_evidence`; the
    validity gate rejects any development-period or latent-only hand, so passing a superset is
    safe.

    Args:
        pair_signals: The pair's candidate :class:`HandSignal` objects.
        pair_id: Optional explicit pair id; when omitted it is taken from the first signal.

    Returns:
        The :class:`EvidenceSelection` for the pair.
    """
    resolved_pair_id = pair_id
    if resolved_pair_id is None:
        resolved_pair_id = str(pair_signals[0].pair_id) if pair_signals else ""
    return select_evidence(resolved_pair_id, pair_signals)
