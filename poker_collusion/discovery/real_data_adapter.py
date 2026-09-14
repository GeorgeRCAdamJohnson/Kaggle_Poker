"""Thin adapter mapping the REAL competition schema onto the probe/audit APIs.

The Phase -1 probes (``datagen_probes``) and audits (``label_evidence_audit``)
were written against DERIVED-SPEC column names (``pool``, ``seat``,
``stack_start``, integer IDs, ``pair_id`` = ``"<a>_<b>"``, evidence in wide form
with a ``target_behavior`` column). The real files verified during the
Discovery-refresh pass use different names and **hex-string** IDs, and there is
no ``pool`` column (the pool key is ``hands.table_id``).

Rather than rewrite the generic, unit-tested probes/audits, this module loads the
real files and returns frames shaped exactly the way those functions expect:

* Hex ``hand_id`` / ``player_id`` / ``table_id`` are factorised to **stable
  integer codes** (the probes/audits call ``int(...)`` on ids and parse a
  ``"<a>_<b>"`` ``pair_id``). Codes are deterministic given the input order.
* ``hands`` gains a ``pool`` column (= integer-coded ``table_id``); ``seats``
  gains ``pool`` (joined via ``hands``) and a ``seat`` alias for ``seat_no`` and a
  ``stack_start`` alias for ``starting_stack``.
* ``development_labels`` / ``evaluation_pairs`` get an integer-coded
  ``"<a>_<b>"`` ``pair_id`` (from ``player_1`` / ``player_2``) plus a
  ``target_behavior`` column (from ``behavior_family``), so the audits' member
  parser and family bucketing work unchanged. The real hex ``pair_id`` is kept
  as ``pair_id_hex`` for reference.
* ``development_evidence`` (long form: ``pair_id, evidence_rank, hand_id,
  behavior_family``) is re-keyed the same way; a ``target_behavior`` alias is
  added.

Nothing here changes the probe/audit logic; it only shapes inputs. It is used by
:func:`run_real_discovery` to compute the observed statistics that flip the
dossier hypotheses, and it deliberately performs no I/O at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from poker_collusion.config import PipelineConfig, get_config
from poker_collusion.discovery import datagen_probes, label_evidence_audit

__all__ = [
    "load_real_tables",
    "run_real_discovery",
]


def _code_map(values: pd.Series) -> Dict[str, int]:
    """Map distinct string values to stable integer codes (first-seen order)."""
    codes: Dict[str, int] = {}
    for v in values:
        key = str(v)
        if key not in codes:
            codes[key] = len(codes)
    return codes


def load_real_tables(config: Optional[PipelineConfig] = None) -> Dict[str, object]:
    """Load the real competition files, shaped for the probes/audits.

    Reads only the projected columns needed by the probe/audit statistics. The
    huge ``actions.parquet`` is read column-projected; a ``max_action_rows`` cap
    keeps the run tractable (the distribution/vocab statistics are stable well
    below the full 18.6M rows).

    Returns a mapping with keys the runners understand: ``actions``, ``hands``,
    ``seats``, ``development_labels``, ``development_evidence``,
    ``evaluation_pairs`` — plus the shared integer code maps under ``_codes``.
    """
    config = config or get_config()
    d = Path(config.input_dir)

    hands = pd.read_parquet(
        d / "hands.parquet",
        columns=["hand_id", "table_id", "phase", "big_blind", "small_blind", "button_seat"],
    )
    seats = pd.read_parquet(
        d / "seats.parquet",
        columns=[
            "hand_id", "player_id", "seat_no", "starting_stack",
            "total_contribution", "net_chips", "folded", "won_share",
        ],
    )
    actions = pd.read_parquet(
        d / "actions.parquet",
        columns=[
            "hand_id", "action_no", "street", "player_id", "action",
            "amount", "amount_to", "to_call", "pot_before", "stack_before",
        ],
    )
    labels = pd.read_csv(d / "development_labels.csv")
    evidence = pd.read_csv(d / "development_evidence.csv")
    eval_pairs = pd.read_csv(d / "evaluation_pairs.csv")

    # ---- stable integer codes for hex IDs -------------------------------- #
    player_codes = _code_map(
        pd.concat([seats["player_id"], labels["player_1"], labels["player_2"]]).astype(str)
    )
    hand_codes = _code_map(hands["hand_id"].astype(str))
    table_codes = _code_map(hands["table_id"].astype(str))

    def pcode(s: pd.Series) -> pd.Series:
        return s.astype(str).map(lambda x: player_codes.setdefault(x, len(player_codes)))

    def hcode(s: pd.Series) -> pd.Series:
        return s.astype(str).map(lambda x: hand_codes.setdefault(x, len(hand_codes)))

    hands = hands.assign(
        hand_id=hcode(hands["hand_id"]),
        pool=hands["table_id"].astype(str).map(table_codes),
        table_id=hands["table_id"].astype(str).map(table_codes),
    )
    seats = seats.assign(
        hand_id=hcode(seats["hand_id"]),
        player_id=pcode(seats["player_id"]),
        seat=seats["seat_no"],
        stack_start=seats["starting_stack"],
    )
    actions = actions.assign(
        hand_id=hcode(actions["hand_id"]),
        player_id=pcode(actions["player_id"]),
    )

    # ---- pair-id encoding as "<a>_<b>" so the audit member parser works --- #
    def encode_pair(df: pd.DataFrame) -> pd.Series:
        a = pcode(df["player_1"])
        b = pcode(df["player_2"])
        lo = a.where(a <= b, b)
        hi = b.where(a <= b, a)
        return lo.astype(str) + "_" + hi.astype(str)

    labels = labels.assign(
        pair_id_hex=labels["pair_id"],
        pair_id=encode_pair(labels),
        target_behavior=labels["behavior_family"],
    )
    hex_to_encoded = dict(zip(labels["pair_id_hex"], labels["pair_id"]))

    eval_pairs = eval_pairs.assign(
        pair_id_hex=eval_pairs["pair_id"],
        pair_id=encode_pair(eval_pairs),
    )

    # Evidence is long form; map its hex pair_id to the encoded one via labels,
    # code the hand_id, and add a target_behavior alias.
    evidence = evidence.assign(
        pair_id_hex=evidence["pair_id"],
        pair_id=evidence["pair_id"].map(hex_to_encoded),
        hand_id=hcode(evidence["hand_id"]),
        target_behavior=evidence["behavior_family"],
    )

    return {
        "actions": actions,
        "hands": hands,
        "seats": seats,
        "development_labels": labels,
        "development_evidence": evidence,
        "evaluation_pairs": eval_pairs,
        "_codes": {
            "n_players": len(player_codes),
            "n_hands": len(hand_codes),
            "n_tables": len(table_codes),
        },
    }


def run_real_discovery(config: Optional[PipelineConfig] = None) -> Dict[str, object]:
    """Load real tables and run every probe + audit group over them.

    Returns a dict with ``probes`` (from :func:`datagen_probes.run_all_probes`)
    and ``audits`` (from :func:`label_evidence_audit.run_all_audits`), plus the
    code cardinalities. The probes are pointed at the pool-keyed ``hands`` /
    ``seats`` so H10-H12 (pool cardinality / hands-per-pool / co-seating) run.
    """
    tables = load_real_tables(config)
    labels = tables["development_labels"]

    probes = datagen_probes.run_all_probes(
        {
            "actions": tables["actions"],
            "hands": tables["hands"],
            "seats": tables["seats"],
        },
        labels=labels,
    )
    audits = label_evidence_audit.run_all_audits(
        {
            "development_evidence": tables["development_evidence"],
            "seats": tables["seats"],
            "evaluation_pairs": tables["evaluation_pairs"],
            "development_labels": labels,
        }
    )
    return {"probes": probes, "audits": audits, "codes": tables["_codes"]}
