"""End-to-end INTEGRATION / acceptance-shape test for the Phase-1 baseline pipeline (task 14.2).

Requirements 9.6, 11.1.

This complements the hermetic ``test_pipeline.py`` by exercising the SAME single entry point
(:func:`poker_collusion.pipeline.run_baseline_pipeline`) on a slightly RICHER but still tiny
synthetic "pool", and by treating the submission's *acceptance shape* as the thing under test:
the produced ``submission.csv`` is read back and pushed through the writer's own
:func:`~poker_collusion.submission.writer.validate_submission` against the sample submission, so
the test literally IS the acceptance check the leaderboard applies (exact columns/order, one row
per evaluation pair, exact pair-id set, ``risk_score`` in [0, 1], allowed behaviors, every
evidence cell non-empty, no repeated hand id per row).

It intentionally does NOT duplicate ``test_pipeline.py``'s assertions (byte-determinism, the
no-eval-hands default row mechanics, the ``limit_pairs`` smoke path). Instead it focuses on:
  * running the full wired pipeline on a richer pool (8 players, several eval pairs, a mix of
    development and evaluation hands, two coordinated pairs and two benign pairs, plus
    development labels/evidence that reflect the real CSV schema);
  * acceptance-shape validity via ``validate_submission`` directly;
  * the exact per-eval-pair row coverage and pair-id set/order matching the sample submission;
  * a coordinated pair's risk >= a benign pair's risk (ranking sanity);
  * at least one pair receiving real (non-``NO_EVIDENCE``) evidence hand ids.

Fixture shape (real schema, hex-ish string ids), ONE pool ``T7``
================================================================
Players (8): UA, UB, UC, UD, UE, UF, UG, UH.

Coordinated pairs (should score HIGH):
  * PAB (UA, UB) -- directed transfer: UA repeatedly dumps ~10bb -> UB in evaluation hands with a
    public bet/call value commitment; a third player folds early (joint isolation flavour too).
  * PEF (UE, UF) -- coordinated isolation: UE and UF BOTH bet/raise at an outsider (UG/UH) while
    both pair members stay active and never contest each other, forcing the outsider to fold,
    across several evaluation hands (the joint-isolation tell).

Benign pairs (should score LOW):
  * PCD (UC, UD) -- symmetric small pots where UC and UD contest each other (mutual aggression,
    no directed flow).
  * PGH (UG, UH) -- symmetric small pots, alternating tiny wins, no directed flow.

Hands: big_blind = 2 throughout, so net_chips 20 == 10bb. A handful of development hands provide
warm-up context that the eval-only pipeline must ignore.

CSV companions (real schema):
  * evaluation_pairs.csv covers PAB, PEF, PCD, PGH.
  * sample_submission.csv covers the same four pair ids (default zero rows).
  * development_labels.csv carries a confirmed_target (PXY over dev players) and a
    confirmed_non_target, exercising both PU label statuses.
  * development_evidence.csv carries one disclosed-family dev-evidence row.

Everything is tiny and hermetic; the real ``data/poker`` files (esp. the 18M-row
``actions.parquet``) are NEVER touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from poker_collusion.config import NO_EVIDENCE, PipelineConfig
from poker_collusion.io import DataLoader
from poker_collusion.metric.reference_public_metric import ALLOWED_BEHAVIORS
from poker_collusion.pipeline import run_baseline_pipeline
from poker_collusion.submission.writer import SUBMISSION_COLUMNS, validate_submission


# --------------------------------------------------------------------------- #
# Synthetic "pool" builders (real schema, hex-ish string ids).
# --------------------------------------------------------------------------- #
_PLAYER_IDS = ["UA", "UB", "UC", "UD", "UE", "UF", "UG", "UH"]

# Hands: development warm-ups + evaluation hands used for scoring/evidence.
#   D01, D02 development (ignored by the eval-only pipeline)
#   E01..E03 evaluation : PAB directed dump (UA -> UB), UC folds early
#   E04..E06 evaluation : PEF joint isolation/soft play (UE, UF pressure UG/UH)
#   E07, E08 evaluation : PCD benign symmetric contest (UC vs UD)
#   E09, E10 evaluation : PGH benign symmetric contest (UG vs UH)
_HAND_IDS = [
    "D01",
    "D02",
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08",
    "E09",
    "E10",
]
_HAND_PHASE = {
    "D01": "development",
    "D02": "development",
    "E01": "evaluation",
    "E02": "evaluation",
    "E03": "evaluation",
    "E04": "evaluation",
    "E05": "evaluation",
    "E06": "evaluation",
    "E07": "evaluation",
    "E08": "evaluation",
    "E09": "evaluation",
    "E10": "evaluation",
}

_HANDS = pd.DataFrame(
    {
        "hand_id": _HAND_IDS,
        "table_id": ["T7"] * len(_HAND_IDS),
        "phase": [_HAND_PHASE[h] for h in _HAND_IDS],
        "big_blind": [2] * len(_HAND_IDS),
        "small_blind": [1] * len(_HAND_IDS),
        "started_at": list(range(1, len(_HAND_IDS) + 1)),
        "button_seat": [0] * len(_HAND_IDS),
    }
)

_PLAYERS = pd.DataFrame(
    {
        "player_id": _PLAYER_IDS,
        "account_age_days": [10 * (i + 1) for i in range(len(_PLAYER_IDS))],
        "experience_hands_bucket": list("abcdefgh"),
        "preferred_stake": ["s"] * len(_PLAYER_IDS),
        "region_bucket": ["r"] * len(_PLAYER_IDS),
        "client_family": ["cf"] * len(_PLAYER_IDS),
    }
)


def _seat(hand_id: str, player_id: str, seat_no: int, net_chips: float, folded: bool = False):
    return {
        "hand_id": hand_id,
        "player_id": player_id,
        "seat_no": seat_no,
        "starting_stack": 200,
        "total_contribution": abs(net_chips) if net_chips < 0 else 0,
        "net_chips": net_chips,
        "folded": folded,
        "went_to_showdown": not folded,
        "won_share": 1.0 if net_chips > 0 else 0.0,
    }


_SEATS = pd.DataFrame(
    [
        # --- development warm-ups (ignored) ---
        _seat("D01", "UA", 0, -2),
        _seat("D01", "UB", 1, 2),
        _seat("D01", "UC", 2, 0),
        _seat("D02", "UE", 0, -2),
        _seat("D02", "UF", 1, 2),
        _seat("D02", "UG", 2, 0),
        # --- PAB directed dump: UA -> UB ~10bb, UC folds ---
        _seat("E01", "UA", 0, -20),
        _seat("E01", "UB", 1, 20),
        _seat("E01", "UC", 2, 0, folded=True),
        _seat("E02", "UA", 0, -20),
        _seat("E02", "UB", 1, 20),
        _seat("E02", "UC", 2, 0, folded=True),
        _seat("E03", "UA", 0, -18),
        _seat("E03", "UB", 1, 18),
        _seat("E03", "UC", 2, 0, folded=True),
        # --- PEF coordinated transfer + isolation: UE repeatedly dumps chips -> UF (a directed
        # value-flow tail between the pair, the discriminator the real data shows) WHILE the pair
        # also gangs up on / isolates the outsider (UG/UH) who is forced out. Neither pair member
        # contests the other. ---
        _seat("E04", "UE", 0, -14),
        _seat("E04", "UF", 1, 14),
        _seat("E04", "UG", 2, 0, folded=True),
        _seat("E05", "UE", 0, -14),
        _seat("E05", "UF", 1, 14),
        _seat("E05", "UH", 2, 0, folded=True),
        _seat("E06", "UE", 0, -12),
        _seat("E06", "UF", 1, 12),
        _seat("E06", "UG", 2, 0, folded=True),
        # --- PCD benign symmetric contest (UC vs UD alternate tiny wins) ---
        _seat("E07", "UC", 0, 2),
        _seat("E07", "UD", 1, -2),
        _seat("E07", "UA", 2, 0, folded=True),
        _seat("E08", "UC", 0, -2),
        _seat("E08", "UD", 1, 2),
        _seat("E08", "UB", 2, 0, folded=True),
        # --- PGH benign symmetric contest (UG vs UH alternate tiny wins) ---
        _seat("E09", "UG", 0, 2),
        _seat("E09", "UH", 1, -2),
        _seat("E09", "UE", 2, 0, folded=True),
        _seat("E10", "UG", 0, -2),
        _seat("E10", "UH", 1, 2),
        _seat("E10", "UF", 2, 0, folded=True),
    ]
)


def _act(hand_id, action_no, player_id, action, amount, to_call, players_active):
    return {
        "hand_id": hand_id,
        "action_no": action_no,
        "street": "pre",
        "player_id": player_id,
        "action": action,
        "amount": amount,
        "amount_to": amount,
        "to_call": to_call,
        "pot_before": 0,
        "stack_before": 200,
        "players_active": players_active,
    }


_ACTIONS = pd.DataFrame(
    [
        # PAB directed dump: UC folds, UA bets big, UB calls (public value commitment).
        _act("E01", 0, "UC", "fold", 0, 2, 3),
        _act("E01", 1, "UA", "bet", 20, 0, 2),
        _act("E01", 2, "UB", "call", 20, 20, 2),
        _act("E02", 0, "UC", "fold", 0, 2, 3),
        _act("E02", 1, "UA", "bet", 20, 0, 2),
        _act("E02", 2, "UB", "call", 20, 20, 2),
        _act("E03", 0, "UC", "fold", 0, 2, 3),
        _act("E03", 1, "UA", "bet", 18, 0, 2),
        _act("E03", 2, "UB", "call", 18, 18, 2),
        # PEF coordinated transfer + isolation: the outsider folds early (pressured/isolated),
        # then UE dumps chips to UF via a public bet/call value commitment (UE bets big, UF calls),
        # so chips flow UE -> UF between the pair; neither pair member contests the other.
        _act("E04", 0, "UG", "fold", 0, 2, 3),
        _act("E04", 1, "UE", "bet", 14, 0, 2),
        _act("E04", 2, "UF", "call", 14, 14, 2),
        _act("E05", 0, "UH", "fold", 0, 2, 3),
        _act("E05", 1, "UE", "bet", 14, 0, 2),
        _act("E05", 2, "UF", "call", 14, 14, 2),
        _act("E06", 0, "UG", "fold", 0, 2, 3),
        _act("E06", 1, "UE", "bet", 12, 0, 2),
        _act("E06", 2, "UF", "call", 12, 12, 2),
        # PCD benign: UA folds, UC and UD contest each other (mutual aggression).
        _act("E07", 0, "UA", "fold", 0, 2, 3),
        _act("E07", 1, "UC", "bet", 2, 0, 2),
        _act("E07", 2, "UD", "raise", 4, 2, 2),
        _act("E07", 3, "UC", "call", 4, 4, 2),
        _act("E08", 0, "UB", "fold", 0, 2, 3),
        _act("E08", 1, "UD", "bet", 2, 0, 2),
        _act("E08", 2, "UC", "raise", 4, 2, 2),
        _act("E08", 3, "UD", "call", 4, 4, 2),
        # PGH benign: UE folds, UG and UH contest each other.
        _act("E09", 0, "UE", "fold", 0, 2, 3),
        _act("E09", 1, "UG", "bet", 2, 0, 2),
        _act("E09", 2, "UH", "raise", 4, 2, 2),
        _act("E09", 3, "UG", "call", 4, 4, 2),
        _act("E10", 0, "UF", "fold", 0, 2, 3),
        _act("E10", 1, "UH", "bet", 2, 0, 2),
        _act("E10", 2, "UG", "raise", 4, 2, 2),
        _act("E10", 3, "UH", "call", 4, 4, 2),
        # development warm-up actions (must be ignored by the eval-only pipeline).
        _act("D01", 0, "UA", "bet", 2, 0, 3),
        _act("D01", 1, "UB", "call", 2, 2, 3),
        _act("D02", 0, "UE", "bet", 2, 0, 3),
        _act("D02", 1, "UF", "call", 2, 2, 3),
    ]
)

# Development labels exercise BOTH PU statuses: a confirmed_target and a confirmed_non_target.
_DEV_LABELS = pd.DataFrame(
    {
        "pair_id": ["PXY", "PZW"],
        "player_1": ["UA", "UC"],
        "player_2": ["UB", "UD"],
        "label": [1, 0],
        "label_status": ["confirmed_target", "confirmed_non_target"],
        "behavior_family": ["directed_transfer", "none"],
    }
)

# Development evidence for the confirmed_target, real 4-column schema.
_DEV_EVIDENCE = pd.DataFrame(
    {
        "pair_id": ["PXY"],
        "evidence_rank": [1],
        "hand_id": ["D01"],
        "behavior_family": ["directed_transfer"],
    }
)

# Evaluation pairs: two coordinated (PAB, PEF), two benign (PCD, PGH).
_EVAL_PAIRS = pd.DataFrame(
    {
        "pair_id": ["PAB", "PEF", "PCD", "PGH"],
        "player_1": ["UA", "UE", "UC", "UG"],
        "player_2": ["UB", "UF", "UD", "UH"],
        "shared_hands": [3, 3, 2, 2],
    }
)

_SAMPLE_SUBMISSION = pd.DataFrame(
    {
        "pair_id": ["PAB", "PEF", "PCD", "PGH"],
        "risk_score": [0.0, 0.0, 0.0, 0.0],
        "predicted_behavior": ["none"] * 4,
        "evidence_hand_1": [NO_EVIDENCE] * 4,
        "evidence_hand_2": [NO_EVIDENCE] * 4,
        "evidence_hand_3": [NO_EVIDENCE] * 4,
        "evidence_hand_4": [NO_EVIDENCE] * 4,
        "evidence_hand_5": [NO_EVIDENCE] * 4,
    }
)

_EVAL_PAIR_IDS = ["PAB", "PEF", "PCD", "PGH"]
_COORDINATED_PAIRS = {"PAB", "PEF"}
_BENIGN_PAIRS = {"PCD", "PGH"}


def _write_fixture(root: Path) -> None:
    _HANDS.to_parquet(root / "hands.parquet")
    _PLAYERS.to_parquet(root / "players.parquet")
    _SEATS.to_parquet(root / "seats.parquet")
    _ACTIONS.to_parquet(root / "actions.parquet")
    _DEV_LABELS.to_csv(root / "development_labels.csv", index=False)
    _DEV_EVIDENCE.to_csv(root / "development_evidence.csv", index=False)
    _EVAL_PAIRS.to_csv(root / "evaluation_pairs.csv", index=False)
    _SAMPLE_SUBMISSION.to_csv(root / "sample_submission.csv", index=False)


def _config_for(root: Path) -> PipelineConfig:
    cfg = PipelineConfig(input_dir=root, output_dir=root)
    cfg.row_group_size = 5  # force multi-batch streaming of the actions parquet
    return cfg


@pytest.fixture()
def pool(tmp_path: Path) -> Path:
    """Materialize the tiny synthetic pool under tmp_path and return its directory."""
    _write_fixture(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# Acceptance-shape integration tests (Req 9.6, 11.1)
# --------------------------------------------------------------------------- #
def test_pipeline_produces_acceptance_valid_submission(pool: Path) -> None:
    """Req 11.1 + 9.6: one documented entry point runs end-to-end and the read-back
    submission passes the leaderboard's own acceptance validation against the sample."""
    cfg = _config_for(pool)

    out = run_baseline_pipeline(cfg)

    # The returned path is the configured submission path and the file exists (Req 11.1).
    assert out == cfg.submission_path
    assert out.exists()

    # Read the file BACK from disk (acceptance is about the emitted artifact, not an in-memory
    # frame) and run the writer's own contract validator against the sample submission. This is
    # the acceptance check the leaderboard applies (Req 9.6): exact columns/order, identical row
    # count, exact pair-id set, risk in [0, 1], allowed behaviors, non-empty evidence cells, no
    # repeated hand id per row. validate_submission raises on any violation.
    frame = pd.read_csv(out)
    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()
    validate_submission(frame, sample)  # raises SubmissionValidationError on any mismatch


def test_submission_has_exactly_one_row_per_eval_pair_matching_sample(pool: Path) -> None:
    """Req 9.6: exactly one row per evaluation pair; pair-id set AND order match the sample."""
    cfg = _config_for(pool)
    out = run_baseline_pipeline(cfg)
    frame = pd.read_csv(out)

    loader = DataLoader(config=cfg)
    sample = loader.load_sample_submission()

    # Exact column headers and order.
    assert tuple(frame.columns) == SUBMISSION_COLUMNS

    # One row per evaluation pair, no duplicates, exact set equality.
    assert len(frame) == len(_EVAL_PAIR_IDS)
    assert not frame["pair_id"].duplicated().any()
    assert set(frame["pair_id"].astype(str)) == set(_EVAL_PAIR_IDS)

    # Row ORDER follows the sample submission unchanged.
    assert frame["pair_id"].astype(str).tolist() == sample["pair_id"].astype(str).tolist()


def test_field_domains_hold_across_richer_pool(pool: Path) -> None:
    """Req 9.6: every risk_score in [0, 1] and every predicted_behavior allowed, across the
    richer pool (a domain check spanning coordinated + benign pairs)."""
    cfg = _config_for(pool)
    out = run_baseline_pipeline(cfg)
    frame = pd.read_csv(out).set_index("pair_id")

    for pair_id in _EVAL_PAIR_IDS:
        risk = float(frame.loc[pair_id, "risk_score"])
        assert 0.0 <= risk <= 1.0
        assert str(frame.loc[pair_id, "predicted_behavior"]) in ALLOWED_BEHAVIORS


def test_coordinated_pair_risk_at_least_benign_pair_risk(pool: Path) -> None:
    """Ranking sanity: each coordinated pair's risk >= each benign pair's risk (Req 11.1 wiring)."""
    cfg = _config_for(pool)
    out = run_baseline_pipeline(cfg)
    frame = pd.read_csv(out).set_index("pair_id")

    coordinated_risks = {p: float(frame.loc[p, "risk_score"]) for p in _COORDINATED_PAIRS}
    benign_risks = {p: float(frame.loc[p, "risk_score"]) for p in _BENIGN_PAIRS}

    min_coordinated = min(coordinated_risks.values())
    max_benign = max(benign_risks.values())

    # A coordinated pair must not be ranked below a benign one. Using >= keeps the assertion
    # robust to score ties on such a tiny fixture while still catching an inverted wiring.
    assert min_coordinated >= max_benign, (
        f"coordinated risks {coordinated_risks} should dominate benign risks {benign_risks}"
    )


def test_at_least_one_pair_receives_real_evidence(pool: Path) -> None:
    """Sanity (Req 9.6): at least one evaluation pair gets a real (non-NO_EVIDENCE) evidence hand
    id, confirming the evidence gate/retriever are wired into the pipeline on this pool."""
    cfg = _config_for(pool)
    out = run_baseline_pipeline(cfg)
    frame = pd.read_csv(out)

    evidence_cols = [
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    ]
    real_hits = (
        frame[evidence_cols].astype(str).apply(lambda col: col != NO_EVIDENCE).to_numpy().sum()
    )
    assert real_hits >= 1, (
        "expected at least one non-NO_EVIDENCE evidence hand id across the submission; "
        "the coordinated pairs' eval hands carry behavior-specific public actions and should "
        "pass the evidence validity gate"
    )
