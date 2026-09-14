"""Phase -1 label & evidence structure audit (task 2.4, Workstream C).

This module is the concrete, runnable realisation of the checks described in
``RESEARCH_DOSSIER.md`` section 3 (Label / Evidence Signatures). It reverse-
engineers what makes a ``development_evidence.csv`` hand "planted" and verifies
the structural invariants and ``evaluation_pairs.csv`` exclusions that the
evidence validity gate (design ``Evidence_Retriever``, task 12.1) must reproduce.

It answers three questions, mirroring the dossier subsections exactly:

- :func:`family_action_signatures`  -> section 3.1  (per-family public action
  signature the validity gate must reproduce)
- :func:`structural_invariants`     -> section 3.2  (latent activation is never
  evidence; each evidence hand has BOTH players AND a public behavior-specific
  action)
- :func:`evaluation_pairs_exclusions` -> section 3.3 (excludes publicly labelled
  pair ids and pairs containing a publicly labelled positive player)

Design notes / guarantees (identical in spirit to ``datagen_probes``):

* **No competition data is required to import or unit-test this module.** Every
  audit accepts already-loaded ``pandas`` DataFrames (tiny synthetic fixtures in
  the tests). With empty/absent input an audit degrades to ``data_present=False``
  rather than raising, so the module is safe wherever the real files are missing.
* Every value returned is a plain Python scalar / list / dict so results are
  JSON-serialisable and paste-ready into the dossier.
* The per-family *signatures* are declarative predicate specifications
  (:data:`FAMILY_SIGNATURES`) derived from the requirements/design; the runtime
  auditors evaluate them against whatever evidence/action data is present and
  return the observed match/violation counts. When no data is present the
  auditors return the predicate spec plus ``status="OPEN - PENDING DATA"`` so the
  caller knows the invariant was specified but not yet empirically confirmed.

Nothing here *decides* collusion; these are *forensic* structural checks whose
observed values flip the dossier's section-3 invariants from
``OPEN - PENDING DATA`` to CONFIRMED/REFUTED and whose signatures seed the
evidence validity gate.

Requirements: 2.4, 2.5, 8.4. Design: Phase -1 Workstream C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = [
    "AuditResult",
    "DISCLOSED_FAMILIES",
    "FAMILY_SIGNATURES",
    "PENDING",
    "family_action_signatures",
    "structural_invariants",
    "evaluation_pairs_exclusions",
    "run_all_audits",
]


#: Status string used when a check is fully specified but has no data to run on.
PENDING = "OPEN - PENDING DATA"

#: The three publicly described coordination behaviors (Behavior MAP classes).
DISCLOSED_FAMILIES: tuple[str, ...] = (
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
)


#: Declarative, decision-time-observable public action signature per disclosed
#: family, derived from Requirements 3.1-3.3, 3.8 and 8.4 and the design's
#: Workstream C / Evidence_Retriever sections. Each entry names:
#:  * ``signal``           - the hand-level signal (Req 3) that carries the tell,
#:  * ``public_action``    - the concrete, public action-log pattern a *planted*
#:                           evidence hand must exhibit for the family,
#:  * ``predicate``        - the machine-checkable condition on hand-level fields,
#:  * ``gate_requirement`` - what the validity gate (task 12.1) must reproduce.
#: These are the signatures the gate reproduces; they are DERIVED-SPEC until
#: confirmed against ``development_evidence.csv`` (Q-DATA below).
FAMILY_SIGNATURES: Dict[str, Dict[str, str]] = {
    "directed_transfer": {
        "signal": "value_flow (Req 3.1): net signed chips A<->B within the hand",
        "public_action": (
            "One player commits chips (bet/raise/call, or all_in classified as "
            "aggressive per Req 3.4) into a pot the partner wins, producing a net "
            "directed chip transfer between the two specific players in that hand."
        ),
        "predicate": (
            "hand is a Shared_Hand containing both players AND the ordered public "
            "action log shows a value-committing action by the losing member "
            "resolved to the other member (|value_flow| materially > 0, directed)."
        ),
        "gate_requirement": (
            "reproduce: hand qualifies only if a public value-transfer action is "
            "present between the two players (not merely an outcome/latent flag)."
        ),
    },
    "soft_play": {
        "signal": "aggression_asymmetry (Req 3.2): bet/raise vs partner / vs field",
        "public_action": (
            "The two players decline to bet/raise into each other where their "
            "against-the-field behavior would predict aggression (checked-back "
            "strong hand, no raise vs the partner) - a public mutual "
            "aggression-avoidance action in the ordered action log."
        ),
        "predicate": (
            "hand is a Shared_Hand containing both players AND the public action "
            "log shows a qualifying bet/raise opportunity between the two that was "
            "declined (low partner-directed aggression relative to field)."
        ),
        "gate_requirement": (
            "reproduce: hand qualifies only when a public passive/checked action "
            "between the two players is observable, not from a scenario flag alone."
        ),
    },
    "coordinated_isolation": {
        "signal": "isolation (Req 3.3): pair's bet/raise at others / at each other",
        "public_action": (
            "Both players jointly apply bet/raise pressure to a third seated "
            "player (ganging up to isolate) within the hand - a public coordinated "
            "pressure action visible in the ordered action log."
        ),
        "predicate": (
            "hand is a Shared_Hand containing both players AND the public action "
            "log shows both members directing bet/raise actions at other seated "
            "players in the same hand (joint isolation, high isolation ratio)."
        ),
        "gate_requirement": (
            "reproduce: hand qualifies only when the joint pressure action on a "
            "third player is present in the public log."
        ),
    },
}


@dataclass
class AuditResult:
    """Structured output of one audit group.

    Attributes:
        checks: Maps a check id to a dict of the exact statistics / predicate
            spec for it. Values are plain scalars so the result is
            JSON-serialisable and paste-ready into the dossier.
        data_present: True when the audit received non-empty input to run on.
        status: ``"CONFIRMED"`` / ``"REFUTED"`` when data let the check run,
            otherwise :data:`PENDING`.
        notes: Human-readable remarks (e.g. why a check was skipped).
    """

    checks: Dict[str, Dict[str, object]] = field(default_factory=dict)
    data_present: bool = False
    status: str = PENDING
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _members_from_pair_id(pair_id: str) -> Optional[tuple]:
    """Parse a ``"<a>_<b>"`` pair id into a sorted (int, int) tuple, else None."""
    parts = str(pair_id).split("_")
    if len(parts) == 2:
        try:
            return tuple(sorted((int(parts[0]), int(parts[1]))))
        except ValueError:
            return None
    return None


def _seated_players(seats, hand_id) -> set:
    """Set of player ids seated in ``hand_id`` per the ``seats`` frame."""
    if seats is None or "hand_id" not in getattr(seats, "columns", []):
        return set()
    grp = seats[seats["hand_id"] == hand_id]
    if "player_id" not in grp.columns:
        return set()
    return set(int(p) for p in grp["player_id"].tolist())


def _has_behavior_action(row, flag_col: str = "behavior_action") -> Optional[bool]:
    """Read a boolean "public behavior-specific action present" flag from a row.

    Returns None when no such column exists (so the caller can mark PENDING
    rather than silently assuming absence).
    """
    for col in (flag_col, "has_behavior_action", "behavior_specific_action"):
        if col in getattr(row, "index", []):
            return bool(row[col])
    return None


# --------------------------------------------------------------------------- #
# 3.1 : Per-family action signatures
# --------------------------------------------------------------------------- #
def family_action_signatures(
    evidence=None,
    *,
    family_col: str = "target_behavior",
    families: Sequence[str] = DISCLOSED_FAMILIES,
) -> AuditResult:
    """Return the per-family public action signature the validity gate must reproduce.

    The signatures themselves (:data:`FAMILY_SIGNATURES`) are DERIVED-SPEC and are
    always returned so the gate has a concrete target. When ``evidence`` is
    supplied and carries a family column, the audit also reports the per-family
    count of planted evidence hands so the empirical signature can later be
    characterised against ``development_evidence.csv`` (Req 2.4).
    """
    result = AuditResult()
    for fam in families:
        result.checks[fam] = dict(FAMILY_SIGNATURES.get(fam, {}))
        result.checks[fam]["status"] = PENDING

    if evidence is None or len(evidence) == 0:
        result.notes.append(
            "development_evidence.csv absent; signatures are DERIVED-SPEC only "
            "(the validity gate target). Empirical per-family characterisation "
            "pending data."
        )
        return result

    result.data_present = True
    if family_col in getattr(evidence, "columns", []):
        vc = evidence[family_col].value_counts(dropna=False)
        counts = {str(k): int(v) for k, v in vc.items()}
        for fam in families:
            result.checks[fam]["observed_evidence_hand_count"] = counts.get(fam, 0)
        result.notes.append(
            "Observed per-family evidence-hand counts recorded; signature "
            "predicates still require action-log characterisation to CONFIRM."
        )
    else:
        result.notes.append(
            f"evidence present but no '{family_col}' column; cannot bucket by family."
        )
    return result


# --------------------------------------------------------------------------- #
# 3.2 : Structural invariants
# --------------------------------------------------------------------------- #
def structural_invariants(
    evidence=None,
    seats=None,
    *,
    pair_col: str = "pair_id",
    hand_col: str = "hand_id",
    behavior_action_col: str = "behavior_action",
) -> AuditResult:
    """Confirm/refute the two structural invariants of planted evidence hands.

    * **INV-1 (latent activation is never evidence).** Every listed evidence hand
      must carry a *public* behavior-specific action; a hand whose only collusion
      indication is latent scenario activation (no public action) is a violation.
      Predicate: for each evidence row, ``behavior_action`` flag is True.
    * **INV-2 (both players AND a public action).** Every listed evidence hand
      must be a Shared_Hand whose seated set includes BOTH members of the pair AND
      must contain a public behavior-specific action.
      Predicate: ``members(pair_id) subset seated(hand_id)`` AND action flag True.

    Both are the exact conditions the validity gate (task 12.1, Req 8.4)
    enforces. With no data the predicates are returned with status PENDING.
    """
    result = AuditResult()

    inv1_spec = {
        "invariant": "latent scenario activation alone is never evidence",
        "predicate": (
            "for every row in development_evidence.csv, the public action log for "
            "(pair, hand) contains at least one behavior-specific action "
            "(behavior_action == True); a latent-only hand is a violation."
        ),
        "gate_requirement": (
            "Evidence_Retriever validity gate (task 12.1) rejects any hand whose "
            "collusion indication is latent-only (Req 8.4)."
        ),
        "status": PENDING,
    }
    inv2_spec = {
        "invariant": "each planted evidence hand contains BOTH players AND a public action",
        "predicate": (
            "for every row, members(pair_id) is a subset of seated(hand_id) AND "
            "behavior_action == True."
        ),
        "gate_requirement": (
            "validity gate requires both-players-present (Req 8.1/8.2) AND a "
            "public behavior-specific action (Req 8.4) for a hand to be valid."
        ),
        "status": PENDING,
    }

    if evidence is None or len(evidence) == 0:
        result.checks["INV-1"] = inv1_spec
        result.checks["INV-2"] = inv2_spec
        result.notes.append(
            "development_evidence.csv absent; invariants specified but not "
            "confirmed. Gate (task 12.1) must enforce both predicates."
        )
        return result

    result.data_present = True
    cols = set(evidence.columns)
    action_col = None
    for c in (behavior_action_col, "has_behavior_action", "behavior_specific_action"):
        if c in cols:
            action_col = c
            break

    inv1_violations: List[Dict[str, object]] = []
    inv2_violations: List[Dict[str, object]] = []
    n_rows = 0
    can_check_action = action_col is not None
    can_check_members = (
        pair_col in cols and hand_col in cols and seats is not None
    )

    for _i, row in evidence.iterrows():
        n_rows += 1
        hand_id = row[hand_col] if hand_col in cols else None

        # INV-1: public behavior-specific action present.
        if can_check_action:
            if not bool(row[action_col]):
                inv1_violations.append(
                    {"pair_id": str(row.get(pair_col, "?")), "hand_id": _to_native(hand_id)}
                )

        # INV-2: both members seated AND action present.
        if can_check_members:
            members = _members_from_pair_id(str(row[pair_col]))
            seated = _seated_players(seats, hand_id)
            both_present = members is not None and set(members) <= seated
            action_ok = bool(row[action_col]) if can_check_action else False
            if not (both_present and action_ok):
                inv2_violations.append(
                    {
                        "pair_id": str(row[pair_col]),
                        "hand_id": _to_native(hand_id),
                        "both_players_seated": bool(both_present),
                        "public_action": (bool(row[action_col]) if can_check_action else None),
                    }
                )

    inv1_spec.update(
        {
            "rows_checked": n_rows if can_check_action else 0,
            "violations": inv1_violations,
            "n_violations": len(inv1_violations),
            "status": _status_from(can_check_action, inv1_violations),
        }
    )
    inv2_spec.update(
        {
            "rows_checked": n_rows if can_check_members and can_check_action else 0,
            "violations": inv2_violations,
            "n_violations": len(inv2_violations),
            "status": _status_from(can_check_members and can_check_action, inv2_violations),
        }
    )
    result.checks["INV-1"] = inv1_spec
    result.checks["INV-2"] = inv2_spec

    if not can_check_action:
        result.notes.append(
            f"no behavior-action flag column ({behavior_action_col}/...); INV-1/INV-2 "
            "action-presence part left PENDING."
        )
    if not can_check_members:
        result.notes.append(
            "seats and/or pair_id/hand_id columns absent; INV-2 both-players part "
            "left PENDING."
        )
    result.status = _combine_status([result.checks["INV-1"]["status"], result.checks["INV-2"]["status"]])
    return result


# --------------------------------------------------------------------------- #
# 3.3 : evaluation_pairs.csv exclusion checks
# --------------------------------------------------------------------------- #
def evaluation_pairs_exclusions(
    evaluation_pairs=None,
    labels=None,
    *,
    pair_col: str = "pair_id",
    label_family_col: str = "target_behavior",
    positive_families: Sequence[str] = DISCLOSED_FAMILIES,
) -> AuditResult:
    """Verify ``evaluation_pairs.csv`` exclusions (Req 2.5).

    * **EXC-1 (excludes publicly labelled pair ids).** No ``pair_id`` in
      ``evaluation_pairs.csv`` also appears in ``development_labels.csv``.
      Predicate: ``set(eval.pair_id) & set(labels.pair_id) == empty``.
    * **EXC-2 (excludes pairs containing a publicly labelled positive player).**
      No evaluation pair contains a player who is a member of any publicly
      labelled *positive* pair.
      Predicate: for every eval pair, ``members(pair) & positive_players == empty``
      where ``positive_players`` = union of members of labelled positive pairs.

    Any violation is reported (Req 2.5 requires reporting violations found). With
    no data the predicates are returned with status PENDING.
    """
    result = AuditResult()

    exc1_spec = {
        "check": "evaluation_pairs.csv excludes publicly labelled pair IDs",
        "predicate": "set(evaluation_pairs.pair_id) & set(development_labels.pair_id) == {}",
        "status": PENDING,
    }
    exc2_spec = {
        "check": "evaluation_pairs.csv excludes pairs containing a publicly labelled positive player",
        "predicate": (
            "for every eval pair p: members(p) & (union of members of labelled "
            "positive pairs) == {}"
        ),
        "status": PENDING,
    }

    if (
        evaluation_pairs is None
        or labels is None
        or len(evaluation_pairs) == 0
        or len(labels) == 0
    ):
        result.checks["EXC-1"] = exc1_spec
        result.checks["EXC-2"] = exc2_spec
        result.notes.append(
            "evaluation_pairs.csv and/or development_labels.csv absent; exclusion "
            "checks specified but not run. Report any violations once data lands."
        )
        return result

    result.data_present = True
    eval_ids = [str(v) for v in evaluation_pairs[pair_col].tolist()] if pair_col in evaluation_pairs.columns else []
    label_ids = set(str(v) for v in labels[pair_col].tolist()) if pair_col in labels.columns else set()

    # EXC-1: pair-id overlap.
    exc1_violations = sorted(set(eval_ids) & label_ids)
    exc1_spec.update(
        {
            "eval_pairs": len(eval_ids),
            "labelled_pairs": len(label_ids),
            "violations": exc1_violations,
            "n_violations": len(exc1_violations),
            "status": _status_from(True, exc1_violations),
        }
    )

    # EXC-2: positive-player overlap.
    positive_players: set = set()
    if label_family_col in getattr(labels, "columns", []):
        pos_rows = labels[labels[label_family_col].isin(list(positive_families))]
    else:
        # Without a family column, treat every labelled pair as positive
        # (conservative: development_labels lists trusted positives).
        pos_rows = labels
        result.notes.append(
            f"no '{label_family_col}' column on labels; treating all labelled pairs "
            "as positive for EXC-2 (conservative)."
        )
    for pid in pos_rows[pair_col].tolist():
        members = _members_from_pair_id(str(pid))
        if members:
            positive_players.update(members)

    exc2_violations: List[Dict[str, object]] = []
    for pid in eval_ids:
        members = _members_from_pair_id(pid)
        if members and (set(members) & positive_players):
            exc2_violations.append(
                {"pair_id": pid, "offending_players": sorted(set(members) & positive_players)}
            )
    exc2_spec.update(
        {
            "positive_players": len(positive_players),
            "violations": exc2_violations,
            "n_violations": len(exc2_violations),
            "status": _status_from(True, exc2_violations),
        }
    )

    result.checks["EXC-1"] = exc1_spec
    result.checks["EXC-2"] = exc2_spec
    result.status = _combine_status([exc1_spec["status"], exc2_spec["status"]])
    return result


# --------------------------------------------------------------------------- #
# Status helpers
# --------------------------------------------------------------------------- #
def _status_from(could_run: bool, violations: Sequence) -> str:
    if not could_run:
        return PENDING
    return "CONFIRMED" if not violations else "REFUTED"


def _combine_status(statuses: Sequence[str]) -> str:
    if any(s == "REFUTED" for s in statuses):
        return "REFUTED"
    if all(s == "CONFIRMED" for s in statuses):
        return "CONFIRMED"
    return PENDING


def _to_native(v):
    """Best-effort conversion of a pandas/NumPy scalar to a native Python type."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)


# --------------------------------------------------------------------------- #
# Convenience runner
# --------------------------------------------------------------------------- #
def run_all_audits(tables: Mapping[str, object]) -> Dict[str, AuditResult]:
    """Run every audit group over a mapping of loaded tables.

    ``tables`` maps names to DataFrames, e.g.::

        {"development_evidence": df_evidence, "seats": df_seats,
         "evaluation_pairs": df_eval, "development_labels": df_labels}

    Missing tables yield an AuditResult with ``data_present=False`` and
    ``status=PENDING`` for that group, so this runs gracefully whether or not the
    competition data is present. Returns a dict keyed by audit-group name.
    """
    evidence = tables.get("development_evidence")
    seats = tables.get("seats")
    evaluation_pairs = tables.get("evaluation_pairs")
    labels = tables.get("development_labels")

    return {
        "family_action_signatures": family_action_signatures(evidence),
        "structural_invariants": structural_invariants(evidence, seats),
        "evaluation_pairs_exclusions": evaluation_pairs_exclusions(evaluation_pairs, labels),
    }
