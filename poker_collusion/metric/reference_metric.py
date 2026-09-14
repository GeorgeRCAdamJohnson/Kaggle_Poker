"""Independent, deliberately-simple reference implementation of the metric.

This module is an INDEPENDENT re-derivation of the competition metric contract
(RESEARCH_DOSSIER.md section 1, status CONFIRMED-CODE). It is written from the
*contract*, not by copying ``reference_public_metric.py`` (which is the verbatim
official code). The whole point of having two implementations is that they were
written independently: when this naive path and the production metric agree on
the same numbers, that agreement is a meaningful cross-check (Property 15, task
4.3) rather than a tautology.

Design priorities here are, in order:

1. **Obvious correctness.** Explicit Python loops, no clever vectorization, no
   shared helpers with the production code. Each function is small enough to
   audit by eye and to compute by hand on a tiny example.
2. **Faithfulness to the contract.** Every clause of the Metric Contract is
   reflected directly, with the relevant contract sub-section noted in comments.
3. **Speed comes last.** This is a test oracle, not a hot path.

Metric contract (confirmed), for reference while reading this file::

    final = 0.70 * PairAP + 0.20 * EvidenceMAP@5 + 0.10 * BehaviorMAP

* **PairAP** (contract 1.1): average precision of the binary labels ranked by
  ``risk_score`` descending, ties broken by ``pair_id`` ascending; normalized by
  the number of positives; ``0.0`` when there are no positives.
* **EvidenceMAP@5** (contract 1.2): mean over the TRUE positive pairs only of a
  per-pair ``AP@5``; per pair ``AP@5 = (sum over ranks 1..5 of hits/rank) /
  min(#planted, 5)``; a true positive with no correct evidence contributes
  ``0.0`` but is still counted; ``NO_EVIDENCE`` / blank / NaN entries are cleaned
  out; hand ids are compared as strings.
* **BehaviorMAP** (contract 1.3): mean over exactly the three disclosed families
  (``directed_transfer``, ``soft_play``, ``coordinated_isolation``); per family
  the one-vs-rest AP of ``true_behavior == family`` ranked by
  ``risk_score where predicted_behavior == family else 0``; a family with zero
  true positives contributes ``0.0`` and stays in the 3-way denominator;
  ``other_coordination`` and ``none`` are excluded.

Do NOT import ``reference_public_metric`` here, and do NOT let this file share
helpers with the production metric.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

# --- Constants re-stated from the contract (independent restatement) ----------

# The three disclosed behavior families that Behavior MAP averages over. Note:
# ``other_coordination`` and ``none`` are deliberately absent (contract 1.3).
TARGET_BEHAVIORS: tuple[str, str, str] = (
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
)

# The five evidence columns, in rank order 1..5.
EVIDENCE_COLUMNS: tuple[str, ...] = tuple(f"evidence_hand_{r}" for r in range(1, 6))

# The exact literal sentinel meaning "no evidence supplied for this slot".
NO_EVIDENCE = "NO_EVIDENCE"

# Combined-score weights (contract combined summary).
WEIGHT_PAIR = 0.70
WEIGHT_EVIDENCE = 0.20
WEIGHT_BEHAVIOR = 0.10


# --- Small, hand-auditable primitives -----------------------------------------


def clean_evidence(values: Iterable[object]) -> list[str]:
    """Turn a row of raw evidence cells into a clean, ordered list of hand ids.

    Rules (contract 1.2):

    * drop ``NaN`` / ``None``;
    * ``str()`` the value and strip surrounding whitespace;
    * drop empty strings and the exact literal ``"NO_EVIDENCE"``;
    * keep everything else, preserving order.

    Kept intentionally naive: one explicit loop, no set/comprehension tricks, so
    the behavior is obvious.
    """
    cleaned: list[str] = []
    for value in values:
        # ``pd.isna`` handles float NaN and None uniformly.
        if value is None or (isinstance(value, float) and math.isnan(value)) or _is_nan(value):
            continue
        text = str(value).strip()
        if text == "":
            continue
        if text == NO_EVIDENCE:
            continue
        cleaned.append(text)
    return cleaned


def _is_nan(value: object) -> bool:
    """True when ``value`` is a pandas/NumPy missing value. Defensive helper."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def average_precision(labels: Sequence[int], risk: Sequence[float], pair_ids: Sequence[str]) -> float:
    """Average precision of binary ``labels`` ranked by ``risk`` descending.

    Independent, textbook AP with the contract's tie-break (1.1):

    * sort candidates by ``risk`` descending;
    * ties on ``risk`` are broken by ``pair_id`` ascending (lexicographic string);
    * walking the ranked list, whenever we hit a positive, add the running
      precision (positives-so-far / rank) to a sum;
    * AP = that sum / (#positives); ``0.0`` if there are no positives.

    This is computed with plain Python sorting and a single loop so it can be
    reproduced by hand on a small example.
    """
    n = len(labels)
    assert len(risk) == n and len(pair_ids) == n, "labels/risk/pair_ids length mismatch"

    total_positives = sum(1 for y in labels if y == 1)
    if total_positives == 0:
        return 0.0

    # Build an index list and sort it. Primary key: risk descending. Secondary
    # key (tie-break): pair_id ascending. Python's sort is stable, so we sort by
    # the secondary key first, then by the primary key.
    order = list(range(n))
    order.sort(key=lambda i: str(pair_ids[i]))          # pair_id ascending
    order.sort(key=lambda i: risk[i], reverse=True)      # risk descending (stable)

    positives_seen = 0
    precision_sum = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i] == 1:
            positives_seen += 1
            precision_sum += positives_seen / rank
    return precision_sum / total_positives


def evidence_ap_at_5(relevant: Sequence[str], submitted: Sequence[str]) -> float:
    """AP@5 for one true-positive pair (contract 1.2).

    ``relevant``  — the planted (correct) evidence hand ids for this pair.
    ``submitted`` — our predicted evidence hand ids, in rank order.

    * If there are no relevant hands, the pair scores ``0.0``.
    * Otherwise walk the first five submitted hands; each time a submitted hand
      is in the relevant set, count a hit and add ``hits / rank``.
    * Divide by ``min(#relevant, 5)``.

    Naive: an explicit membership test against a set built once, one loop over
    at most five ranks.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, hand_id in enumerate(submitted[:5], start=1):
        if hand_id in relevant_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_set), 5)


# --- Component scores ----------------------------------------------------------


def pair_ap(labels: Sequence[int], risk: Sequence[float], pair_ids: Sequence[str]) -> float:
    """Pair AP component (contract 1.1). Thin wrapper over :func:`average_precision`."""
    return average_precision(labels, risk, pair_ids)


def evidence_map5(
    labels: Sequence[int],
    relevant_per_pair: Sequence[Sequence[str]],
    submitted_per_pair: Sequence[Sequence[str]],
) -> float:
    """Evidence MAP@5 component (contract 1.2).

    Averages :func:`evidence_ap_at_5` over exactly the pairs with ``label == 1``.
    A true positive with no correct evidence still contributes ``0.0`` and is
    counted in the mean. Non-positive pairs are never scored. Returns ``0.0``
    when there are no true positives.
    """
    per_pair_scores: list[float] = []
    for i, label in enumerate(labels):
        if label != 1:
            continue  # non-positive pairs never scored (contract 1.2)
        per_pair_scores.append(
            evidence_ap_at_5(relevant_per_pair[i], submitted_per_pair[i])
        )
    if not per_pair_scores:
        return 0.0
    return sum(per_pair_scores) / len(per_pair_scores)


def behavior_map(
    true_behavior: Sequence[str],
    predicted_behavior: Sequence[str],
    risk: Sequence[float],
    pair_ids: Sequence[str],
) -> float:
    """Behavior MAP component (contract 1.3).

    Mean over exactly the three ``TARGET_BEHAVIORS`` of a one-vs-rest AP. For a
    family ``f``:

    * the binary label is ``true_behavior == f``;
    * the ranking score is ``risk`` where ``predicted_behavior == f``, else 0;
    * a family with zero true positives contributes ``0.0`` but STAYS in the
      3-way denominator.

    ``other_coordination`` and ``none`` never form a family here, so they are
    excluded from the average.
    """
    family_scores: list[float] = []
    for family in TARGET_BEHAVIORS:
        family_labels = [1 if tb == family else 0 for tb in true_behavior]
        if sum(family_labels) == 0:
            # Zero true positives for this family -> contributes 0.0 but the
            # family is still one of the three we divide by (contract 1.3).
            family_scores.append(0.0)
            continue
        family_risk = [
            (risk[i] if predicted_behavior[i] == family else 0.0)
            for i in range(len(true_behavior))
        ]
        family_scores.append(average_precision(family_labels, family_risk, pair_ids))
    # Always divide by exactly three (len(TARGET_BEHAVIORS)).
    return sum(family_scores) / len(TARGET_BEHAVIORS)


# --- Combined score over solution / submission frames --------------------------


def score(solution_df: pd.DataFrame, submission_df: pd.DataFrame) -> dict[str, float]:
    """Compute all three components plus the combined summary (contract combined).

    Parameters
    ----------
    solution_df:
        The ground-truth frame. Must be indexable by ``pair_id`` and carry the
        binary label in ``risk_score`` (0/1), the ground-truth family in
        ``predicted_behavior``, and the five ``evidence_hand_*`` columns holding
        the planted evidence.
    submission_df:
        Our prediction frame, same columns; ``risk_score`` is a float in [0, 1],
        ``predicted_behavior`` the predicted family, and the evidence columns our
        ranked evidence hands.

    Returns
    -------
    dict with keys ``pair_ap``, ``evidence_map5``, ``behavior_map`` (the three
    components reported separately, Req 10.5) and ``combined`` (the
    ``0.70/0.20/0.10`` weighted sum).

    This function does only the minimal, obvious alignment (index by ``pair_id``,
    reindex the submission to the solution order sorted by ``pair_id`` ascending)
    needed to feed the naive primitives above. It intentionally does NOT
    re-implement the official metric's participant-facing validation — that is
    the production metric's job; this oracle assumes well-formed inputs.
    """
    # Align both frames on pair_id, solution order = pair_id ascending
    # (contract 1.1's sort_index()).
    truth = solution_df.set_index("pair_id").sort_index()
    preds = submission_df.set_index("pair_id").loc[truth.index]

    pair_ids = [str(pid) for pid in truth.index]

    # Ground-truth binary labels come from the solution's risk_score column.
    labels = [int(v) for v in truth["risk_score"].tolist()]

    # Predicted risk scores, as floats.
    risk = [float(v) for v in preds["risk_score"].tolist()]

    # --- Pair AP -------------------------------------------------------------
    pair_ap_score = pair_ap(labels, risk, pair_ids)

    # --- Behavior MAP --------------------------------------------------------
    true_behavior = [str(v) for v in truth["predicted_behavior"].tolist()]
    predicted_behavior = [str(v) for v in preds["predicted_behavior"].tolist()]
    behavior_map_score = behavior_map(true_behavior, predicted_behavior, risk, pair_ids)

    # --- Evidence MAP@5 ------------------------------------------------------
    relevant_per_pair: list[list[str]] = []
    submitted_per_pair: list[list[str]] = []
    for pid in truth.index:
        relevant_per_pair.append(
            clean_evidence([truth.loc[pid, col] for col in EVIDENCE_COLUMNS])
        )
        submitted_per_pair.append(
            clean_evidence([preds.loc[pid, col] for col in EVIDENCE_COLUMNS])
        )
    evidence_map5_score = evidence_map5(labels, relevant_per_pair, submitted_per_pair)

    combined = (
        WEIGHT_PAIR * pair_ap_score
        + WEIGHT_EVIDENCE * evidence_map5_score
        + WEIGHT_BEHAVIOR * behavior_map_score
    )

    return {
        "pair_ap": pair_ap_score,
        "evidence_map5": evidence_map5_score,
        "behavior_map": behavior_map_score,
        "combined": combined,
    }
