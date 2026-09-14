"""Property test for the impossibility + coverage-mismatch guard (task 2.5).

Feature: poker-anchor-reproduction, Property 5: For any eval artifact and dev-label
set whose pair-id sets are NOT equal (empty intersection, or any missing/extra pair),
the scorer SHALL NOT return an absolute LB-score reproduction: it SHALL either
pre-check the pair-id intersection and refuse (the BELT path), or surface the official
metric's own ``ParticipantVisibleError`` coverage-mismatch raise (the SUSPENDERS path)
— never a silent zero or a fabricated number — and in BOTH paths the diagnostic SHALL
name the disjoint-set sizes (how many pair-ids are missing and how many are extra).

This is the load-bearing honest constraint of the whole spec (design.md, contract
Rules 1 & 9): the eval artifacts' pair sets are DISJOINT from the dev labels, so no
local metric can reproduce an absolute LB number. The guard must make that impossible
to hide behind a silent zero or a fabricated value.

The property draws random dev-vs-prediction pair sets that are guaranteed NOT equal
(they are disjoint, or the predictions add extras / drop confirmed pairs), scores
them, and asserts:
  * the scorer raises ``CoverageMismatchError`` (a ``ParticipantVisibleError`` subclass
    — no absolute-match / numeric return path exists for a non-equal pair set);
  * the raised diagnostic names BOTH the missing and the extra counts (never a silent
    zero), and the counts match the true set difference;
  * the ``describe()`` message text literally contains "N missing" and "M extra".
Both the BELT (pre-check) and SUSPENDERS (metric-raise, via monkeypatch) paths are
exercised under the same property so neither path can silently fabricate a number.

Validates: Requirements 2.4, 4.1
"""

from __future__ import annotations

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from poker_collusion.config import PipelineConfig
from poker_collusion.metric.reference_public_metric import (
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    ParticipantVisibleError,
)
from poker_collusion.submission.writer import SUBMISSION_COLUMNS

from anchor_repro.models import DisjointSetDiagnostic, ScoringRecipe
from anchor_repro.scorer import CanonicalScorer, CoverageMismatchError


# --------------------------------------------------------------------------- #
# Builders (reused from test_anchor_repro_integrity.py conventions)           #
# --------------------------------------------------------------------------- #
def _evidence_slots(hands=()):
    padded = list(hands)[:5] + [NO_EVIDENCE] * (5 - min(len(hands), 5))
    return {col: padded[i] for i, col in enumerate(EVIDENCE_COLUMNS)}


def _dev_labels(confirmed_ids) -> pd.DataFrame:
    """Confirmed DEV pairs (``D*``). Alternating target / non-target so
    ``build_fold_solution`` yields a non-empty solution when any are predicted."""
    rows = []
    for i, pid in enumerate(confirmed_ids):
        rows.append(
            {
                "pair_id": pid,
                "label_status": "confirmed_target" if i % 2 == 0 else "confirmed_non_target",
                "behavior_family": "directed_transfer" if i % 2 == 0 else "none",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["pair_id", "label_status", "behavior_family"],
    )


def _predictions(pairs) -> pd.DataFrame:
    rows = []
    for pid in pairs:
        rows.append(
            {
                "pair_id": pid,
                "risk_score": 0.5,
                "predicted_behavior": "none",
                **_evidence_slots([]),
            }
        )
    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))


def _make_scorer(confirmed_ids) -> CanonicalScorer:
    return CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(),
        labels=_dev_labels(confirmed_ids),
        evidence=None,
        sample_submission=None,
    )


# --------------------------------------------------------------------------- #
# Strategies: dev-label + prediction pair sets guaranteed NOT to be equal     #
# --------------------------------------------------------------------------- #
# Distinct id "pools" whose namespaces never collide, so we can independently
# control which confirmed pairs the predictions keep / drop and which extras
# (unknown-to-dev, e.g. eval) pairs they add.
_confirmed = st.lists(
    st.integers(min_value=0, max_value=40).map(lambda i: f"D{i}"),
    min_size=1,
    max_size=8,
    unique=True,
)
# "Extra" ids the dev labels have never heard of — these stand in for eval pairs
# (E*) or dropped-then-unknown ids that must NEVER be scored as a silent zero.
_extras = st.lists(
    st.integers(min_value=0, max_value=60).map(lambda i: f"E{i}"),
    min_size=0,
    max_size=8,
    unique=True,
)


def _assert_names_disjoint_sizes(err: CoverageMismatchError) -> None:
    """Common assertions: it is a ParticipantVisibleError, carries a diagnostic that
    names BOTH counts, and the message text literally states them (never silent)."""
    # No absolute-match / numeric return path exists: a mismatch is an EXPLICIT raise,
    # and it is a ParticipantVisibleError so existing metric-catching callers see it.
    assert isinstance(err, ParticipantVisibleError)
    diag = err.diagnostic
    assert isinstance(diag, DisjointSetDiagnostic)
    # The diagnostic NAMES the disjoint-set sizes (missing AND extra) — Property 5.
    assert diag.n_missing >= 0 and diag.n_extra >= 0
    assert not diag.is_equal  # a raise only happens for a NON-equal pair set
    # And the human-readable message literally contains both counts (never a silent
    # zero or a fabricated number).
    msg = str(err)
    assert f"{diag.n_missing} missing" in msg
    assert f"{diag.n_extra} extra" in msg


# --------------------------------------------------------------------------- #
# BELT path — the pre-check refuses before the metric ever runs               #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 5
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(confirmed=_confirmed, extras=_extras, keep=st.integers(min_value=0, max_value=8))
def test_belt_precheck_refuses_and_names_disjoint_sizes(confirmed, extras, keep):
    """Validates: Requirements 2.4, 4.1

    For a prediction frame whose pair-id set is NOT equal to the built dev-holdout
    solution, the BELT pre-check raises ``CoverageMismatchError`` naming BOTH the
    missing and the extra counts — never a silent zero, never a fabricated absolute
    score.

    Note on the honest mechanics (verified against ``build_fold_solution``): the
    solution is derived FROM the predicted pair-ids, keeping only those that are
    confirmed dev labels. So the built solution set == ``predicted ∩ confirmed`` and
    is always a SUBSET of the predicted set. Consequently the belt's ``n_missing`` is
    structurally 0 (nothing in the solution is absent from the predictions) and its
    ``n_extra`` equals the number of predicted pairs that are UNKNOWN to the dev
    labels (e.g. eval pairs). The disjoint-set sizes are still ALL named in the
    diagnostic (reference/predicted/intersection/missing/extra) — the guard never
    hides a count behind a silent zero. The fully-disjoint eval-vs-dev case
    (``confirmed`` all dropped, predictions all ``E*``) is exercised when ``keep``
    drops every confirmed pair."""
    # Keep only a prefix of the confirmed pairs, then add unknown-to-dev extras.
    kept = confirmed[: min(keep, len(confirmed))]
    predicted_pairs = list(dict.fromkeys(kept + extras))  # de-dup, preserve order

    # Guarantee the pair sets are NOT equal: ensure at least one predicted pair is
    # unknown to the dev labels (an "extra"). If the prediction is exactly a subset
    # of confirmed labels (no extras), the solution would equal it and no mismatch
    # fires — so force one guaranteed-extra id in that case.
    predicted_ids = set(predicted_pairs)
    confirmed_ids = set(confirmed)
    if not predicted_ids or predicted_ids <= confirmed_ids:
        predicted_pairs.append("E_FORCED_EXTRA")
        predicted_ids = set(predicted_pairs)

    scorer = _make_scorer(confirmed)
    preds = _predictions(predicted_pairs)

    with pytest.raises(CoverageMismatchError) as exc_info:
        scorer.score_dev_predictions(preds)

    err = exc_info.value
    _assert_names_disjoint_sizes(err)

    # The built solution set is ``predicted ∩ confirmed`` (unknowns are dropped, never
    # scored as negatives). The belt compares that solution against the predictions.
    solution_ids = predicted_ids & confirmed_ids
    # Solution ⊆ predicted ⇒ nothing missing; extras are the unknown-to-dev predictions.
    assert err.diagnostic.n_missing == len(solution_ids - predicted_ids) == 0
    assert err.diagnostic.n_extra == len(predicted_ids - solution_ids)
    assert err.diagnostic.n_extra == len(predicted_ids - confirmed_ids)
    assert err.diagnostic.n_extra >= 1  # a mismatch always names >=1 extra here


# --------------------------------------------------------------------------- #
# SUSPENDERS path — the metric's own coverage raise is re-surfaced explicitly #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 5
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    confirmed=_confirmed,
    extras=st.lists(
        st.integers(min_value=0, max_value=60).map(lambda i: f"E{i}"),
        min_size=1,
        max_size=8,
        unique=True,
    ),
)
def test_suspenders_metric_raise_is_resurfaced_naming_disjoint_sizes(confirmed, extras):
    """Validates: Requirements 2.4, 4.1

    If the belt is bypassed (simulated by forcing ``is_equal`` True) but the official
    metric itself raises its coverage-mismatch ``ParticipantVisibleError``, the scorer
    re-surfaces it as the SAME explicit ``CoverageMismatchError`` naming the disjoint
    sizes it recomputes from the frames — never swallowing it into a silent zero.

    ``monkeypatch`` is applied via ``MonkeyPatch.context()`` (not the function-scoped
    fixture) so each Hypothesis example gets a fresh, correctly-undone patch."""
    import anchor_repro.scorer as scorer_module

    scorer = _make_scorer(confirmed)
    # Predict all confirmed pairs plus unknown-to-dev extras. The recomputed solution
    # is ``predicted ∩ confirmed == set(confirmed)``, so the extras are the disjoint
    # "extra" ids the re-surfaced diagnostic must name.
    predicted_pairs = list(dict.fromkeys(list(confirmed) + list(extras)))
    preds = _predictions(predicted_pairs)

    with MonkeyPatch.context() as mp:
        # Bypass the belt: force the pre-check to see the sets as equal so control
        # reaches the metric call, then make the metric raise its own coverage error.
        mp.setattr(
            scorer_module.DisjointSetDiagnostic, "is_equal", property(lambda self: True)
        )

        def _raise_coverage(*_a, **_k):
            # The exact message shape the official metric emits; the numbers here are
            # deliberately different from the true set diff so we prove the scorer
            # recomputes its OWN measured counts rather than parroting the metric text.
            raise ParticipantVisibleError(
                "pair_id coverage mismatch: 7 missing and 9 extra."
            )

        mp.setattr(scorer_module, "score_components", _raise_coverage)

        with pytest.raises(CoverageMismatchError) as exc_info:
            scorer.score_dev_predictions(preds)

    err = exc_info.value
    _assert_names_disjoint_sizes(err)

    # The re-surfaced diagnostic is our OWN recomputed measurement, NOT the metric's
    # raw "7/9" text. The solution is ``predicted ∩ confirmed == set(confirmed)`` and
    # is a subset of the predictions, so nothing is missing and the extras are exactly
    # the unknown-to-dev predicted pairs.
    confirmed_ids = set(confirmed)
    predicted_ids = set(predicted_pairs)
    solution_ids = predicted_ids & confirmed_ids
    assert err.diagnostic.n_missing == len(solution_ids - predicted_ids) == 0
    assert err.diagnostic.n_extra == len(predicted_ids - solution_ids)
    assert err.diagnostic.n_extra == len(extras)  # every extra is unknown to dev
    # It is NOT the metric's fabricated 7/9 text — our measured number governs.
    assert "7 missing" not in str(err)


# --------------------------------------------------------------------------- #
# SUSPENDERS path — a genuine non-coverage metric error is NOT misclassified  #
# --------------------------------------------------------------------------- #
# Feature: poker-anchor-reproduction, Property 5
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(confirmed=_confirmed)
def test_suspenders_non_coverage_error_reraised_untouched(confirmed):
    """Validates: Requirements 2.4, 4.1

    A genuine, non-coverage ``ParticipantVisibleError`` from the metric (e.g. a bad
    risk_score) is re-raised untouched — it is NOT misclassified as a coverage
    mismatch, so the guard never fabricates disjoint-set numbers for an unrelated
    submission-contract violation."""
    import anchor_repro.scorer as scorer_module

    scorer = _make_scorer(confirmed)
    preds = _predictions(list(confirmed))

    with MonkeyPatch.context() as mp:
        mp.setattr(
            scorer_module.DisjointSetDiagnostic, "is_equal", property(lambda self: True)
        )

        def _raise_other(*_a, **_k):
            raise ParticipantVisibleError(
                "risk_score must be numeric and between 0 and 1."
            )

        mp.setattr(scorer_module, "score_components", _raise_other)

        with pytest.raises(ParticipantVisibleError) as exc_info:
            scorer.score_dev_predictions(preds)
    # Re-raised untouched: NOT wrapped as a CoverageMismatchError.
    assert not isinstance(exc_info.value, CoverageMismatchError)
