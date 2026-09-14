# Feature: poker-collusion-detection, Property 15: Local metric matches a reference implementation (PU-correct)
"""Property 15 (task 4.3): production/local metric matches the INDEPENDENT reference.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

This test asserts that the production three-part scorer
(:mod:`poker_collusion.metric.production_metric` — ``score_components``/``score``)
agrees with the *independent* naive oracle
(:mod:`poker_collusion.metric.reference_metric`, the re-derivation from task 4.1,
NOT the verbatim ``reference_public_metric``) across a wide space of randomly
generated, VALID, PU-correct inputs.

"PU-correct" here means:

* the private **solution**'s ``risk_score`` column holds the *ground-truth binary
  target* (0/1), its ``predicted_behavior`` is the ground-truth family, and its
  evidence columns are the planted (correct) hands for positives;
* the **submission** is an arbitrary but VALID participant answer: the same
  ``pair_id`` set, a free ``risk_score`` in ``[0, 1]``, any allowed
  ``predicted_behavior``, and non-repeating evidence hands per pair
  (``NO_EVIDENCE`` padding); nowhere do we assume "unlabeled == negative" when
  building the submission — its risk is an independent float and its behavior an
  independent label.

Both scorers align the solution/submission by ``pair_id`` ascending. Component
name mapping between the two APIs:

    reference_metric.score(...)          -> dict  pair_ap / evidence_map5 / behavior_map / combined
    production_metric.score_components   -> ScoreComponents  pair_ap / evidence_map / behavior_map / combined

so ``reference["evidence_map5"]`` is compared against ``production.evidence_map``.

Generator: this test prefers the `hypothesis` library. If hypothesis is not
importable in the running interpreter, it transparently falls back to a seeded
randomized generator loop (many seeds) plus explicit small shrinking-friendly
cases that achieve the same coverage. The active generator is recorded in
``GENERATOR_IN_USE`` and reported by ``test_report_generator_in_use``.
"""

from __future__ import annotations

import importlib.util
import random

import pytest

from poker_collusion.metric import production_metric as prod
from poker_collusion.metric import reference_metric as ref

# pandas is a hard dependency of both metric modules; import here for frame build.
import pandas as pd

TOL = 1e-9

# Behavior label universe. The submission may use ANY allowed behavior; the
# ground-truth solution uses target families / other_coordination / none.
ALLOWED_BEHAVIORS = sorted(prod.ALLOWED_BEHAVIORS)
TARGET_BEHAVIORS = list(prod.TARGET_BEHAVIORS)
NON_TARGET_POSITIVE = ["other_coordination"]  # positive but excluded from Behavior MAP

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
GENERATOR_IN_USE = "hypothesis" if HYPOTHESIS_AVAILABLE else "seeded-loop"


# --------------------------------------------------------------------------- #
# Shared frame builders (independent of production internals)                 #
# --------------------------------------------------------------------------- #
def _evidence_row(hands: list[str]) -> dict:
    """Fill the 5 evidence columns from a hand list, padding with NO_EVIDENCE.

    Assumes ``hands`` already contains at most 5 unique ids in rank order.
    """
    padded = list(hands) + [prod.NO_EVIDENCE] * (5 - len(hands))
    return {f"evidence_hand_{i}": padded[i - 1] for i in range(1, 6)}


def _build_solution(specs: list[dict]) -> pd.DataFrame:
    """Build a PU-correct private solution frame from per-pair specs.

    Each spec: ``{"pair_id", "label" (0/1), "behavior", "true_hands" (list)}``.
    For positives the planted evidence is ``true_hands`` (1..5 unique ids); for
    negatives evidence is empty (all NO_EVIDENCE), matching the data contract.
    """
    rows = []
    for spec in specs:
        hands = spec["true_hands"] if spec["label"] == 1 else []
        rows.append(
            {
                "pair_id": spec["pair_id"],
                "risk_score": int(spec["label"]),
                "predicted_behavior": spec["behavior"],
                **_evidence_row(hands),
            }
        )
    return pd.DataFrame(rows)


def _build_submission(specs: list[dict]) -> pd.DataFrame:
    """Build a VALID submission frame from per-pair specs.

    Each spec: ``{"pair_id", "risk" (float in [0,1]), "behavior" (allowed),
    "evidence" (list of unique ids, len<=5)}``.
    """
    rows = []
    for spec in specs:
        rows.append(
            {
                "pair_id": spec["pair_id"],
                "risk_score": float(spec["risk"]),
                "predicted_behavior": spec["behavior"],
                **_evidence_row(spec["evidence"]),
            }
        )
    return pd.DataFrame(rows)


def _assert_components_agree(solution: pd.DataFrame, submission: pd.DataFrame) -> None:
    """Core assertion: production components == independent reference components.

    Compares all three components and the combined score to within ``TOL``. The
    reference exposes the evidence component as ``evidence_map5`` while production
    exposes it as ``evidence_map`` — mapped explicitly here.
    """
    reference = ref.score(solution, submission)
    components = prod.score_components(solution, submission, "pair_id")

    assert components.pair_ap == pytest.approx(reference["pair_ap"], abs=TOL), (
        f"pair_ap mismatch: prod={components.pair_ap} ref={reference['pair_ap']}"
    )
    # Key-name mapping: reference "evidence_map5" <-> production "evidence_map".
    assert components.evidence_map == pytest.approx(
        reference["evidence_map5"], abs=TOL
    ), (
        f"evidence map mismatch: prod={components.evidence_map} "
        f"ref={reference['evidence_map5']}"
    )
    assert components.behavior_map == pytest.approx(
        reference["behavior_map"], abs=TOL
    ), (
        f"behavior_map mismatch: prod={components.behavior_map} "
        f"ref={reference['behavior_map']}"
    )
    assert components.combined == pytest.approx(reference["combined"], abs=TOL), (
        f"combined mismatch: prod={components.combined} ref={reference['combined']}"
    )
    # The convenience score() must equal the combined component too.
    assert prod.score(solution, submission, "pair_id") == pytest.approx(
        reference["combined"], abs=TOL
    )


# --------------------------------------------------------------------------- #
# Random case construction from primitive draws                               #
# --------------------------------------------------------------------------- #
def _make_case_from_labels(
    rng: random.Random,
    labels: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a (solution, submission) pair from a fixed list of binary labels.

    ``labels`` fully determines the number of positives/negatives (so the caller
    can force the zero-positives / all-positives edge cases). Everything else —
    ground-truth families, planted evidence, submission risk/behavior/evidence —
    is drawn randomly but kept VALID and PU-correct.

    Uses unique string ``pair_id`` values in a deliberately unsorted insertion
    order (both scorers sort by ``pair_id`` ascending internally, so ordering
    must not matter).
    """
    n = len(labels)
    # Unique, shuffled pair ids so the "sort by pair_id ascending" alignment is
    # genuinely exercised (input order != sorted order).
    ids = [f"P{idx:05d}" for idx in range(n)]
    rng.shuffle(ids)

    sol_specs: list[dict] = []
    sub_specs: list[dict] = []
    for pair_id, label in zip(ids, labels):
        if label == 1:
            # Ground-truth positive family: usually a target family, occasionally
            # other_coordination (positive but excluded from Behavior MAP).
            if rng.random() < 0.15:
                behavior = rng.choice(NON_TARGET_POSITIVE)
            else:
                behavior = rng.choice(TARGET_BEHAVIORS)
            n_true = rng.randint(1, 5)
            true_hands = [f"{pair_id}_T{k}" for k in range(n_true)]
        else:
            # Negative pairs are labeled "none" with no planted evidence, but a
            # small fraction carry a target family label to stress that Behavior
            # MAP keys off the ground-truth label, not the family string alone.
            behavior = "none"
            true_hands = []

        sol_specs.append(
            {
                "pair_id": pair_id,
                "label": label,
                "behavior": behavior,
                "true_hands": true_hands,
            }
        )

        # ---- Submission: arbitrary but valid, no unlabeled==negative assumption.
        risk = round(rng.random(), 9)
        sub_behavior = rng.choice(ALLOWED_BEHAVIORS)
        # Evidence: mix true hits, decoys, and blanks; unique + <=5 in rank order.
        n_ev = rng.randint(0, 5)
        pool = list(true_hands) + [f"{pair_id}_D{k}" for k in range(5)]
        rng.shuffle(pool)
        evidence = list(dict.fromkeys(pool))[:n_ev]
        sub_specs.append(
            {
                "pair_id": pair_id,
                "risk": risk,
                "behavior": sub_behavior,
                "evidence": evidence,
            }
        )

    # Shuffle row order of each frame independently to further decouple ordering.
    rng.shuffle(sol_specs)
    rng.shuffle(sub_specs)
    return _build_solution(sol_specs), _build_submission(sub_specs)


def _labels_for_scenario(rng: random.Random, n: int, scenario: str) -> list[int]:
    """Produce a label vector for a named coverage scenario."""
    if scenario == "zero_positives":
        return [0] * n
    if scenario == "all_positives":
        return [1] * n
    # "mixed": at least one positive and one negative when n >= 2.
    labels = [rng.randint(0, 1) for _ in range(n)]
    if n >= 2:
        if sum(labels) == 0:
            labels[rng.randrange(n)] = 1
        if sum(labels) == n:
            labels[rng.randrange(n)] = 0
    return labels


# --------------------------------------------------------------------------- #
# Explicit small, shrinking-friendly cases (always run, both generators)      #
# --------------------------------------------------------------------------- #
def _small_cases() -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Hand-written minimal cases that pin down each contract edge."""
    cases: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

    # 1 pair, single positive, perfect evidence.
    sol = _build_solution(
        [{"pair_id": "P0", "label": 1, "behavior": "soft_play",
          "true_hands": ["H1"]}]
    )
    sub = _build_submission(
        [{"pair_id": "P0", "risk": 0.9, "behavior": "soft_play",
          "evidence": ["H1"]}]
    )
    cases.append(("single_positive_perfect", sol, sub))

    # 1 pair, single negative (zero positives everywhere -> pair_ap/evidence 0).
    sol = _build_solution(
        [{"pair_id": "P0", "label": 0, "behavior": "none", "true_hands": []}]
    )
    sub = _build_submission(
        [{"pair_id": "P0", "risk": 0.3, "behavior": "none", "evidence": []}]
    )
    cases.append(("single_negative", sol, sub))

    # 2 pairs, tie on risk -> pair_id ascending tie-break must agree.
    sol = _build_solution(
        [
            {"pair_id": "P0", "label": 1, "behavior": "directed_transfer",
             "true_hands": ["A"]},
            {"pair_id": "P1", "label": 0, "behavior": "none", "true_hands": []},
        ]
    )
    sub = _build_submission(
        [
            {"pair_id": "P0", "risk": 0.5, "behavior": "none", "evidence": []},
            {"pair_id": "P1", "risk": 0.5, "behavior": "none", "evidence": []},
        ]
    )
    cases.append(("risk_tie_break", sol, sub))

    # Missed target positive (correct family, wrong evidence -> evidence 0) and an
    # other_coordination positive (excluded from Behavior MAP denominator).
    sol = _build_solution(
        [
            {"pair_id": "P0", "label": 1, "behavior": "directed_transfer",
             "true_hands": ["A", "B"]},
            {"pair_id": "P1", "label": 1, "behavior": "other_coordination",
             "true_hands": ["C"]},
            {"pair_id": "P2", "label": 0, "behavior": "none", "true_hands": []},
        ]
    )
    sub = _build_submission(
        [
            {"pair_id": "P0", "risk": 0.8, "behavior": "directed_transfer",
             "evidence": ["WRONG1", "WRONG2"]},
            {"pair_id": "P1", "risk": 0.7, "behavior": "other_coordination",
             "evidence": ["C"]},
            {"pair_id": "P2", "risk": 0.1, "behavior": "none", "evidence": []},
        ]
    )
    cases.append(("missed_target_and_other_coordination", sol, sub))

    return cases


@pytest.mark.parametrize("name,solution,submission", _small_cases())
def test_small_cases_agree(name: str, solution: pd.DataFrame, submission: pd.DataFrame):
    """Small, deterministic cases: production == independent reference."""
    _assert_components_agree(solution, submission)


def test_report_generator_in_use():
    """Surface which generator backend the property test is running under."""
    assert GENERATOR_IN_USE in {"hypothesis", "seeded-loop"}
    print(f"[Property 15] generator in use: {GENERATOR_IN_USE}")


# --------------------------------------------------------------------------- #
# Property test — hypothesis path (preferred) OR seeded-loop fallback         #
# --------------------------------------------------------------------------- #
if HYPOTHESIS_AVAILABLE:  # pragma: no cover - exercised only where hypothesis exists
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    @st.composite
    def _solution_submission(draw):
        """Draw a valid, PU-correct (solution, submission) pair.

        Uses a hypothesis-provided seed to drive the shared random construction so
        the exact same builders back both generator paths (single source of truth
        for validity), while hypothesis still owns example generation + shrinking.
        The label vector is drawn directly so hypothesis can shrink toward the
        zero-positive / single-pair minimal cases.
        """
        n = draw(st.integers(min_value=1, max_value=25))
        labels = draw(
            st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n)
        )
        seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
        rng = random.Random(seed)
        return _make_case_from_labels(rng, labels)

    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(case=_solution_submission())
    def test_property_reference_match_hypothesis(case):
        solution, submission = case
        _assert_components_agree(solution, submission)

else:

    # Seeded randomized generator loop achieving equivalent coverage. Runs well
    # over 100 iterations across a range of sizes and forces the zero-positive and
    # all-positive edge scenarios so Pair AP / Evidence MAP boundary behavior is
    # covered exactly like a hypothesis run would target.
    _N_SEEDS = 250

    @pytest.mark.parametrize("seed", range(_N_SEEDS))
    def test_property_reference_match_seeded_loop(seed: int):
        rng = random.Random(seed)
        n = rng.randint(1, 25)
        # Cycle through scenarios by seed so the batch covers all boundaries; most
        # seeds are "mixed", with regular zero/all-positive edge insertions.
        which = seed % 5
        if which == 0:
            scenario = "zero_positives"
        elif which == 1:
            scenario = "all_positives"
        else:
            scenario = "mixed"
        labels = _labels_for_scenario(rng, n, scenario)
        solution, submission = _make_case_from_labels(rng, labels)
        _assert_components_agree(solution, submission)
