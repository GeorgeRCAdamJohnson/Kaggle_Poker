"""Phase -1 data-generation reverse-engineering probes (task 2.3, Workstream B).

The competition dataset is fully **synthetic**, so its generator leaves
fingerprints. Using **only public data**, this module computes the exact
statistics named in ``RESEARCH_DOSSIER.md`` section 2 (hypotheses H1-H19) so each
hypothesis can be confirmed or refuted once the data is present. Every probe here
is the concrete, runnable realisation of the "*Statistic/query*" line of a
dossier hypothesis.

The probes are grouped exactly as the dossier subsections:

- :func:`distribution_artifacts`     -> H1-H5   (section 2.1)
- :func:`id_ordering_regularities`   -> H6-H9   (section 2.2)
- :func:`pool_construction`          -> H10-H13 (section 2.3)
- :func:`confounder_tells`           -> H14-H17 (section 2.4)
- :func:`other_coordination_signature` -> H18-H19 (section 2.5)

Design notes / guarantees:

* **No competition data is required to import or unit-test this module.** Every
  probe accepts already-loaded ``pandas`` DataFrames (tiny synthetic fixtures in
  the tests) so the statistics can be validated deterministically. A thin
  :func:`run_all_probes` convenience wraps a :class:`~poker_collusion.io`-style
  loader mapping when the real data is present.
* Probes that need labels (H13, H18, H19 and the confounder contrasts) accept an
  optional ``labels`` frame; when it is absent they return the label-independent
  parts and mark the label-dependent results as ``None`` rather than raising.
* Every numeric returned is a plain Python scalar / list / dict so results are
  JSON-serialisable and can be pasted straight into the dossier.

Nothing here decides collusion; these are *forensic* statistics whose observed
values flip the dossier hypotheses from ``OPEN - PENDING DATA`` to
CONFIRMED/REFUTED. The discriminators-vs-confounders logic is documented per
probe so the caller records the right comparison.

Requirements: 2.1, 2.2, 2.3. Design: Phase -1 Workstream B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "ProbeResult",
    "CANDIDATE_CHIP_UNITS",
    "distribution_artifacts",
    "id_ordering_regularities",
    "pool_construction",
    "confounder_tells",
    "other_coordination_signature",
    "run_all_probes",
]


# Candidate chip lattice units tested by H1 (multiples of a big blind / chip unit).
CANDIDATE_CHIP_UNITS: tuple[int, ...] = (1, 2, 5, 10, 20, 25, 50, 100)


@dataclass
class ProbeResult:
    """Structured output of one probe group.

    Attributes:
        hypotheses: Maps a hypothesis id (e.g. ``"H1"``) to a dict of the exact
            statistics named for it in the dossier. Values are plain scalars so
            the result is JSON-serialisable and paste-ready.
        data_present: True when the probe received non-empty input to compute on.
        labels_present: True when label-dependent statistics could be computed.
        notes: Human-readable remarks (e.g. why a statistic was skipped).
    """

    hypotheses: Dict[str, Dict[str, object]] = field(default_factory=dict)
    data_present: bool = False
    labels_present: bool = False
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Small numeric helpers (no third-party deps beyond pandas at call sites)
# --------------------------------------------------------------------------- #
def _gcd_of_ints(values: Iterable[int]) -> int:
    """GCD of a stream of non-negative ints (0 contributes nothing)."""
    g = 0
    for v in values:
        iv = int(v)
        if iv:
            g = gcd(g, abs(iv))
    return g


def _gini(values: Sequence[float]) -> Optional[float]:
    """Gini concentration coefficient of non-negative values in [0, 1].

    Used for temporal concentration of flagged hands (H13, H17). Returns None
    for an empty input or all-zero mass (undefined concentration).
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return None
    total = sum(xs)
    if total <= 0:
        return None
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    # Standard Gini for a discrete sample.
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _is_monotonic_nondecreasing(seq: Sequence) -> bool:
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation without SciPy. Returns None if degenerate."""
    n = len(a)
    if n != len(b) or n < 2:
        return None

    def _ranks(xs: Sequence[float]) -> List[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # average rank (1-based) for ties
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0 or vb <= 0:
        return None
    return cov / (va ** 0.5 * vb ** 0.5)


# --------------------------------------------------------------------------- #
# H1-H5 : Distribution artifacts (dossier section 2.1)
# --------------------------------------------------------------------------- #
def distribution_artifacts(
    actions,
    seats=None,
    *,
    chip_columns: Sequence[str] = ("amount", "amount_to", "pot_before", "stack_before", "to_call"),
    candidate_units: Sequence[int] = CANDIDATE_CHIP_UNITS,
    hands=None,
) -> ProbeResult:
    """Compute H1-H5 statistics from the ``actions`` (and optionally ``seats``) tables.

    * **H1 chip_unit** - GCD of nonzero chip columns and the fraction of values
      divisible by each candidate unit. If one unit captures >= 99% of values
      the lattice hypothesis is supported.
    * **H2 starting_stack** - distinct-value count and min/max/percentiles of the
      per-seat starting stack (``seats.stack_start`` if available, else
      ``actions.stack_before`` at each hand's first action).
    * **H3 blind_per_pool** - number of distinct inferred big-blind values; here we
      report the distinct smallest-positive ``to_call`` values globally (per-pool
      grouping is applied by :func:`pool_construction` when a pool key is present).
    * **H4 action_vocab** - value counts of ``action`` (and ``hands.phase`` when
      ``hands`` is supplied): confirms a small closed vocabulary.
    * **H5 chip_conservation** - per-hand residual of contributions vs. the pot,
      surfacing any rake term. Computed only when the needed columns exist.

    All statistics are label-independent (these are generator-global artifacts).
    """
    result = ProbeResult()
    if actions is None or len(actions) == 0:
        result.notes.append("actions table empty/absent; distribution probes skipped")
        return result
    result.data_present = True

    # ---- H1: chip lattice / unit -----------------------------------------
    present_cols = [c for c in chip_columns if c in actions.columns]
    all_vals: List[int] = []
    for c in present_cols:
        col = actions[c].dropna()
        # Only integer-like chip quantities participate in the lattice test.
        for v in col.tolist():
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                continue
            all_vals.append(iv)
    nonzero = [v for v in all_vals if v != 0]
    divisible_fraction: Dict[int, float] = {}
    for u in candidate_units:
        if nonzero:
            divisible_fraction[u] = sum(1 for v in nonzero if v % u == 0) / len(nonzero)
        else:
            divisible_fraction[u] = 0.0
    dominant_unit = None
    if nonzero:
        # Largest unit whose divisible fraction clears the 99% lattice threshold.
        clearing = [u for u in candidate_units if divisible_fraction[u] >= 0.99]
        dominant_unit = max(clearing) if clearing else None
    result.hypotheses["H1"] = {
        "columns_used": present_cols,
        "gcd_nonzero": _gcd_of_ints(nonzero),
        "divisible_fraction": divisible_fraction,
        "dominant_unit_ge_99pct": dominant_unit,
        "n_values": len(nonzero),
    }

    # ---- H2: starting-stack quantization ---------------------------------
    stacks: List[float] = []
    if seats is not None and "stack_start" in getattr(seats, "columns", []):
        stacks = [float(v) for v in seats["stack_start"].dropna().tolist()]
        stack_source = "seats.stack_start"
    elif "stack_before" in actions.columns and {"hand_id", "action_no"} <= set(actions.columns):
        first = actions.sort_values(["hand_id", "action_no"]).groupby("hand_id").head(1)
        stacks = [float(v) for v in first["stack_before"].dropna().tolist()]
        stack_source = "actions.stack_before@first-action"
    else:
        stack_source = None
    if stacks:
        srt = sorted(stacks)

        def _pct(p: float) -> float:
            idx = min(len(srt) - 1, max(0, int(round(p * (len(srt) - 1)))))
            return srt[idx]

        result.hypotheses["H2"] = {
            "source": stack_source,
            "distinct_count": len(set(stacks)),
            "min": srt[0],
            "max": srt[-1],
            "p50": _pct(0.5),
            "p95": _pct(0.95),
        }
    else:
        result.hypotheses["H2"] = {"source": stack_source, "distinct_count": 0}

    # ---- H3: blind homogeneity (global here; per-pool in pool_construction)
    blinds: List[float] = []
    if "to_call" in actions.columns:
        pos = [float(v) for v in actions["to_call"].dropna().tolist() if float(v) > 0]
        blinds = pos
    result.hypotheses["H3"] = {
        "distinct_positive_to_call": len(set(blinds)),
        "min_positive_to_call": min(blinds) if blinds else None,
    }

    # ---- H4: closed action/phase vocabulary ------------------------------
    action_vocab = {}
    if "action" in actions.columns:
        vc = actions["action"].value_counts(dropna=False)
        action_vocab = {str(k): int(v) for k, v in vc.items()}
    phase_vocab = {}
    if hands is not None and "phase" in getattr(hands, "columns", []):
        vc = hands["phase"].value_counts(dropna=False)
        phase_vocab = {str(k): int(v) for k, v in vc.items()}
    result.hypotheses["H4"] = {
        "action_vocab": action_vocab,
        "action_vocab_size": len(action_vocab),
        "phase_vocab": phase_vocab,
    }

    # ---- H5: per-hand chip conservation ----------------------------------
    if {"hand_id", "amount"} <= set(actions.columns):
        residuals: List[float] = []
        for _hid, grp in actions.groupby("hand_id"):
            contributed = float(grp["amount"].fillna(0).sum())
            residuals.append(contributed)  # awards unknown here; residual == contributions
        # We report contributed-chip totals per hand; when an awards column is
        # available the caller subtracts it. Variance near zero (after award
        # subtraction) => conservation; a constant offset => fixed rake.
        n = len(residuals)
        mean = sum(residuals) / n if n else None
        var = sum((r - mean) ** 2 for r in residuals) / n if n else None
        result.hypotheses["H5"] = {
            "note": "residual = per-hand contributed chips; subtract awards when available",
            "n_hands": n,
            "contributed_mean": mean,
            "contributed_var": var,
        }
    else:
        result.hypotheses["H5"] = {"note": "hand_id/amount missing; conservation not computed"}

    return result


# --------------------------------------------------------------------------- #
# H6-H9 : ID / timestamp / ordering regularities (dossier section 2.2)
# --------------------------------------------------------------------------- #
def id_ordering_regularities(
    hands,
    actions=None,
    seats=None,
    *,
    pool_col: str = "pool",
    timestamp_col: str = "timestamp",
) -> ProbeResult:
    """Compute H6-H9 statistics from the ``hands`` (+ optional ``actions``/``seats``) tables.

    * **H6 hand_id_monotonicity / pool_blocking** - is ``hand_id`` strictly
      increasing overall; per pool, are ``hand_id`` ranges contiguous and
      non-overlapping (so chronology == hand_id order within a pool).
    * **H7 timestamp_agreement** - Spearman(timestamp, hand_id) within pool and the
      spread of successive-hand time deltas (only when a timestamp column exists).
    * **H8 action_ordinal** - per hand, is ``action_no`` a contiguous range from a
      common base and is ``phase`` non-decreasing along ``action_no``.
    * **H9 seat_structure** - distinct seat ids per table and, per player, the
      button/seat step between consecutive hands (deterministic rotation vs.
      random reshuffle). Directly informs the ``repeated opponent selection``
      discriminator: deterministic rotation makes co-seating structural.
    """
    result = ProbeResult()
    if hands is None or len(hands) == 0:
        result.notes.append("hands table empty/absent; id/ordering probes skipped")
        return result
    result.data_present = True

    hand_ids = [int(v) for v in hands["hand_id"].tolist()] if "hand_id" in hands.columns else []

    # ---- H6: hand_id monotonicity + pool blocking ------------------------
    strictly_increasing = all(hand_ids[i] < hand_ids[i + 1] for i in range(len(hand_ids) - 1))
    pool_blocking = None
    overlaps = None
    if pool_col in hands.columns and hand_ids:
        ranges: Dict[object, tuple] = {}
        contiguous_flags: List[bool] = []
        for pool, grp in hands.groupby(pool_col):
            ids = sorted(int(v) for v in grp["hand_id"].tolist())
            lo, hi, cnt = ids[0], ids[-1], len(set(ids))
            ranges[pool] = (lo, hi)
            contiguous_flags.append((hi - lo + 1) == cnt)
        # Non-overlapping check across pools.
        sorted_ranges = sorted(ranges.values())
        overlaps = any(
            sorted_ranges[i][1] >= sorted_ranges[i + 1][0]
            for i in range(len(sorted_ranges) - 1)
        )
        pool_blocking = all(contiguous_flags)
    result.hypotheses["H6"] = {
        "hand_id_strictly_increasing": bool(strictly_increasing),
        "pools_contiguous": pool_blocking,
        "pool_ranges_overlap": overlaps,
    }

    # ---- H7: timestamp <-> hand_id agreement -----------------------------
    if timestamp_col in hands.columns and hand_ids:
        ts = [float(v) for v in hands[timestamp_col].tolist()]
        rho = _spearman(ts, [float(h) for h in hand_ids])
        deltas = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        n = len(deltas)
        mean = sum(deltas) / n if n else None
        var = sum((d - mean) ** 2 for d in deltas) / n if n else None
        result.hypotheses["H7"] = {
            "spearman_ts_vs_hand_id": rho,
            "delta_mean": mean,
            "delta_var": var,
            "distinct_deltas": len({round(d, 9) for d in deltas}) if deltas else 0,
        }
    else:
        result.hypotheses["H7"] = {"note": "no timestamp column; hand_id used as chronology proxy"}

    # ---- H8: action_no is a per-hand ordinal -----------------------------
    if actions is not None and {"hand_id", "action_no"} <= set(actions.columns):
        contiguous_all = True
        base_values: set = set()
        phase_monotone_all = True
        has_phase = "phase" in actions.columns
        for _hid, grp in actions.sort_values(["hand_id", "action_no"]).groupby("hand_id"):
            nos = [int(v) for v in grp["action_no"].tolist()]
            base_values.add(nos[0])
            expected = list(range(nos[0], nos[0] + len(nos)))
            if nos != expected:
                contiguous_all = False
            if has_phase:
                phases = list(grp["phase"])
                # Map ordered phase tokens to ranks if categorical; here we only
                # check that identical consecutive tokens or a known order hold.
                if not _phase_nondecreasing(phases):
                    phase_monotone_all = False
        result.hypotheses["H8"] = {
            "action_no_contiguous": contiguous_all,
            "distinct_base": sorted(base_values),
            "phase_nondecreasing": phase_monotone_all if has_phase else None,
        }
    else:
        result.hypotheses["H8"] = {"note": "actions hand_id/action_no missing; ordinal not checked"}

    # ---- H9: seat assignment structure -----------------------------------
    if seats is not None and {"hand_id", "player_id", "seat"} <= set(seats.columns):
        table_col = "table_id" if "table_id" in seats.columns else None
        distinct_seats_per_table: Dict[object, int] = {}
        if table_col:
            for tbl, grp in seats.groupby(table_col):
                distinct_seats_per_table[tbl] = int(grp["seat"].nunique())
        else:
            distinct_seats_per_table["<all>"] = int(seats["seat"].nunique())
        # Per-player seat step across consecutive hands.
        steps: List[int] = []
        for _pid, grp in seats.sort_values(["player_id", "hand_id"]).groupby("player_id"):
            seq = [int(s) for s in grp["seat"].tolist()]
            steps.extend(seq[i + 1] - seq[i] for i in range(len(seq) - 1))
        step_counts: Dict[int, int] = {}
        for s in steps:
            step_counts[s] = step_counts.get(s, 0) + 1
        result.hypotheses["H9"] = {
            "distinct_seats_per_table": {str(k): v for k, v in distinct_seats_per_table.items()},
            "seat_step_counts": step_counts,
            "deterministic_rotation_hint": (len(step_counts) <= 2) if step_counts else None,
        }
    else:
        result.hypotheses["H9"] = {"note": "seats hand_id/player_id/seat missing; structure not checked"}

    return result


_PHASE_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}


def _phase_nondecreasing(phases: Sequence) -> bool:
    """True if a phase token sequence never moves to an earlier betting round."""
    ranks = []
    for p in phases:
        key = str(p).lower()
        if key not in _PHASE_ORDER:
            # Unknown token: treat as compatible (cannot judge ordering).
            return True
        ranks.append(_PHASE_ORDER[key])
    return _is_monotonic_nondecreasing(ranks)


# --------------------------------------------------------------------------- #
# H10-H13 : Pool / table construction (dossier section 2.3)
# --------------------------------------------------------------------------- #
def pool_construction(
    hands,
    seats,
    *,
    pool_col: str = "pool",
    labels=None,
    expected_pools: int = 400,
    expected_players_per_pool: int = 30,
    expected_hands_per_pool: int = 5000,
) -> ProbeResult:
    """Compute H10-H13 statistics.

    * **H10 cardinality / disjointness** - distinct pool count; per-pool distinct
      players (expect 30) and tables (expect 1); cross-pool player-set overlaps
      (expect none). Disjoint pools justify group-by-pool CV.
    * **H11 hands_per_pool** - per-pool hand counts and mean/std/min/max (expect a
      tight cluster near ~5,000).
    * **H12 coseating_matrix** - per pool, the fraction of within-pool player pairs
      that ever share a hand and the shared-hand-count distribution. Broad,
      rotation-driven co-seating means co-seating frequency alone is *not* a
      collusion signal (discriminator vs. ``repeated opponent selection``).
    * **H13 positive_prevalence / episode_concentration** *(needs labels)* - the
      fraction of within-pool pairs that are Trusted_Positive, plus a placeholder
      for episode concentration (full flagged-hand concentration is computed by
      :func:`confounder_tells` where per-hand signals are available).
    """
    result = ProbeResult()
    if hands is None or seats is None or len(hands) == 0 or len(seats) == 0:
        result.notes.append("hands/seats empty/absent; pool probes skipped")
        return result
    result.data_present = True

    # Attach pool to seats via hands if seats lacks a pool column.
    seats_pool = seats
    if pool_col not in seats.columns and pool_col in hands.columns:
        seats_pool = seats.merge(hands[["hand_id", pool_col]], on="hand_id", how="left")

    have_pool = pool_col in seats_pool.columns

    # ---- H10: cardinality & disjointness ---------------------------------
    if have_pool:
        players_per_pool: Dict[object, set] = {}
        tables_per_pool: Dict[object, set] = {}
        for pool, grp in seats_pool.groupby(pool_col):
            players_per_pool[pool] = set(int(p) for p in grp["player_id"].tolist())
            if "table_id" in grp.columns:
                tables_per_pool[pool] = set(grp["table_id"].tolist())
        pools = list(players_per_pool)
        # Pairwise player-set intersections.
        overlap_total = 0
        for i in range(len(pools)):
            for j in range(i + 1, len(pools)):
                overlap_total += len(players_per_pool[pools[i]] & players_per_pool[pools[j]])
        result.hypotheses["H10"] = {
            "distinct_pools": len(pools),
            "matches_expected_pools": len(pools) == expected_pools,
            "players_per_pool": {str(k): len(v) for k, v in players_per_pool.items()},
            "all_pools_have_expected_players": all(
                len(v) == expected_players_per_pool for v in players_per_pool.values()
            ),
            "tables_per_pool": {str(k): len(v) for k, v in tables_per_pool.items()},
            "cross_pool_player_overlap_total": overlap_total,
            "pools_disjoint": overlap_total == 0,
        }
    else:
        result.hypotheses["H10"] = {"note": f"no '{pool_col}' column; cardinality not computed"}

    # ---- H11: hands per pool ---------------------------------------------
    if pool_col in hands.columns:
        counts = [int(len(grp)) for _p, grp in hands.groupby(pool_col)]
        n = len(counts)
        mean = sum(counts) / n if n else None
        var = sum((c - mean) ** 2 for c in counts) / n if n else None
        result.hypotheses["H11"] = {
            "per_pool_counts": counts,
            "mean": mean,
            "std": (var ** 0.5) if var is not None else None,
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "near_expected": (
                mean is not None and abs(mean - expected_hands_per_pool) <= 0.25 * expected_hands_per_pool
            ),
        }
    else:
        result.hypotheses["H11"] = {"note": f"no '{pool_col}' column on hands; counts not computed"}

    # ---- H12: co-seating opportunity -------------------------------------
    if have_pool:
        pair_shared: Dict[tuple, int] = {}
        pools_players: Dict[object, set] = {}
        for _hid, grp in seats_pool.groupby("hand_id"):
            pool_vals = grp[pool_col].tolist()
            pool = pool_vals[0] if pool_vals else None
            players = sorted(int(p) for p in grp["player_id"].tolist())
            pools_players.setdefault(pool, set()).update(players)
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    key = (pool, players[i], players[j])
                    pair_shared[key] = pair_shared.get(key, 0) + 1
        # Fraction of possible within-pool pairs that ever share a hand.
        possible_pairs = 0
        for pool, players in pools_players.items():
            k = len(players)
            possible_pairs += k * (k - 1) // 2
        realized_pairs = len(pair_shared)
        shared_counts = sorted(pair_shared.values())
        result.hypotheses["H12"] = {
            "possible_within_pool_pairs": possible_pairs,
            "pairs_with_shared_hand": realized_pairs,
            "fraction_pairs_coseated": (realized_pairs / possible_pairs) if possible_pairs else None,
            "shared_count_min": shared_counts[0] if shared_counts else None,
            "shared_count_max": shared_counts[-1] if shared_counts else None,
            "shared_count_mean": (sum(shared_counts) / len(shared_counts)) if shared_counts else None,
        }
    else:
        result.hypotheses["H12"] = {"note": f"no '{pool_col}' column; co-seating not computed"}

    # ---- H13: positive prevalence & episode structure (needs labels) -----
    if labels is not None and "pair_id" in getattr(labels, "columns", []):
        result.labels_present = True
        n_positive = int(len(labels))
        total_pairs = None
        if have_pool:
            total_pairs = result.hypotheses.get("H12", {}).get("possible_within_pool_pairs")
        result.hypotheses["H13"] = {
            "n_trusted_positive": n_positive,
            "possible_pairs": total_pairs,
            "positive_prevalence": (n_positive / total_pairs) if total_pairs else None,
            "episode_concentration": "see confounder_tells.temporal_concentration",
        }
    else:
        result.hypotheses["H13"] = {"note": "labels absent; positive prevalence not computed"}

    return result


# --------------------------------------------------------------------------- #
# H14-H17 : Statistical tells vs. confounders (dossier section 2.4)
# --------------------------------------------------------------------------- #
def confounder_tells(
    hand_signals,
    *,
    pair_col: str = "pair_id",
    labels=None,
) -> ProbeResult:
    """Compute the H14-H17 confounder-discriminating contrasts.

    ``hand_signals`` is a per-(pair, hand) frame of the decision-time signals the
    pipeline already extracts (design signals module), with at least:
    ``pair_id, player, opponent, hand_id, value_flow`` and, where available,
    ``aggression_toward, isolation_joint, is_flagged``. The key idea shared by
    H14-H17 is the **pair-directed-minus-field-baseline contrast**: a benign
    confounder (tilt, weak play, similar strategies) moves a player's baseline
    *against everyone*, so the pair-specific quantity minus the same player's
    all-opponents baseline is ~0; a planted positive opens a gap.

    Returned per hypothesis (when the needed columns exist):

    * **H14 directedness_contrast** - per pair, signed ``value_flow`` between the
      two players minus each player's mean signed flow against all other
      opponents.
    * **H15 avoidance_gap** - partner-directed aggression/avoidance minus
      field-directed, per player (needs ``aggression_toward``).
    * **H16 joint_isolation_rate** - the pair's joint isolation rate per shared
      hand (needs ``isolation_joint``), to be read *conditional on* co-seating.
    * **H17 temporal_concentration** - Gini/burst concentration of the pair's
      flagged hands over its shared-hand timeline (needs ``is_flagged``).
    """
    result = ProbeResult()
    if hand_signals is None or len(hand_signals) == 0:
        result.notes.append("hand_signals empty/absent; confounder probes skipped")
        return result
    result.data_present = True
    cols = set(hand_signals.columns)

    # ---- H14: directedness contrast --------------------------------------
    if {pair_col, "player", "opponent", "value_flow"} <= cols:
        # Mean signed flow player->opponent for each ordered (player, opponent).
        directed: Dict[tuple, List[float]] = {}
        for _i, row in hand_signals.iterrows():
            directed.setdefault((int(row["player"]), int(row["opponent"])), []).append(
                float(row["value_flow"])
            )
        directed_mean = {k: sum(v) / len(v) for k, v in directed.items()}
        # Per player, mean over all opponents (field baseline).
        by_player: Dict[int, List[float]] = {}
        for (p, _o), m in directed_mean.items():
            by_player.setdefault(p, []).append(m)
        field_baseline = {p: sum(v) / len(v) for p, v in by_player.items()}
        contrasts: Dict[str, float] = {}
        for pid, grp in hand_signals.groupby(pair_col):
            members = _pair_members(str(pid), grp)
            if len(members) >= 2:
                a, b = members[0], members[1]
                # Prefer the directed edge that is observed; fall back to reverse.
                pair_flow = directed_mean.get((a, b))
                anchor = a
                if pair_flow is None and (b, a) in directed_mean:
                    pair_flow = directed_mean.get((b, a))
                    anchor = b
                if pair_flow is not None and anchor in field_baseline:
                    contrasts[str(pid)] = pair_flow - field_baseline[anchor]
        result.hypotheses["H14"] = {
            "directedness_contrast_by_pair": contrasts,
            "note": "contrast ~0 for tilt/weak-play (baseline moves); large for planted positive",
        }
    else:
        result.hypotheses["H14"] = {"note": "need pair_id/player/opponent/value_flow columns"}

    # ---- H15: avoidance gap ----------------------------------------------
    if {pair_col, "player", "aggression_toward", "is_partner"} <= cols:
        gaps: Dict[str, float] = {}
        for pid, grp in hand_signals.groupby(pair_col):
            partner = grp[grp["is_partner"] == True]["aggression_toward"]  # noqa: E712
            field = grp[grp["is_partner"] == False]["aggression_toward"]  # noqa: E712
            if len(partner) and len(field):
                gaps[str(pid)] = float(field.mean()) - float(partner.mean())
        result.hypotheses["H15"] = {
            "avoidance_gap_by_pair": gaps,
            "note": "gap ~0 for similar-strategy pairs; large for soft_play colluders",
        }
    else:
        result.hypotheses["H15"] = {"note": "need pair_id/player/aggression_toward/is_partner columns"}

    # ---- H16: joint isolation rate ---------------------------------------
    if {pair_col, "isolation_joint"} <= cols:
        rates: Dict[str, float] = {}
        for pid, grp in hand_signals.groupby(pair_col):
            vals = [float(v) for v in grp["isolation_joint"].dropna().tolist()]
            if vals:
                rates[str(pid)] = sum(vals) / len(vals)
        result.hypotheses["H16"] = {
            "joint_isolation_rate_by_pair": rates,
            "note": "read conditional on co-seating; repeated-selection is at chance",
        }
    else:
        result.hypotheses["H16"] = {"note": "need pair_id/isolation_joint columns"}

    # ---- H17: temporal concentration -------------------------------------
    if {pair_col, "hand_id", "is_flagged"} <= cols:
        concentration: Dict[str, object] = {}
        for pid, grp in hand_signals.sort_values("hand_id").groupby(pair_col):
            flags = [1.0 if bool(f) else 0.0 for f in grp["is_flagged"].tolist()]
            n_shared = len(flags)
            n_flag = int(sum(flags))
            # Concentration of flagged positions: Gini over inter-flag gaps proxy.
            positions = [i for i, f in enumerate(flags) if f > 0]
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            concentration[str(pid)] = {
                "n_shared": n_shared,
                "n_flagged": n_flag,
                "flag_fraction": (n_flag / n_shared) if n_shared else None,
                "gap_gini": _gini(gaps) if gaps else None,
                "burst_count": _count_bursts(flags),
            }
        result.hypotheses["H17"] = {
            "temporal_concentration_by_pair": concentration,
            "note": "concentrated pair-directed bursts for positives; field-wide shift for strategy-change",
        }
    else:
        result.hypotheses["H17"] = {"note": "need pair_id/hand_id/is_flagged columns"}

    if labels is not None and "pair_id" in getattr(labels, "columns", []):
        result.labels_present = True

    return result


def _pair_members(pair_id: str, grp) -> List[int]:
    """Recover the two player ids of a pair.

    A pair_id is conventionally ``"<a>_<b>"`` with integer members; parse that
    first. If it is not parseable, fall back to the union of ``player`` and
    ``opponent`` values observed in the group's rows. Returns a sorted list.
    """
    parts = str(pair_id).split("_")
    if len(parts) == 2:
        try:
            return sorted((int(parts[0]), int(parts[1])))
        except ValueError:
            pass
    members: set = set()
    if "player" in grp.columns:
        members.update(int(x) for x in grp["player"].tolist())
    if "opponent" in grp.columns:
        members.update(int(x) for x in grp["opponent"].tolist())
    return sorted(members)


def _count_bursts(flags: Sequence[float]) -> int:
    """Number of maximal runs of consecutive flagged hands."""
    bursts = 0
    prev = 0.0
    for f in flags:
        if f > 0 and prev <= 0:
            bursts += 1
        prev = f
    return bursts


# --------------------------------------------------------------------------- #
# H18-H19 : Signature of hidden other_coordination (dossier section 2.5)
# --------------------------------------------------------------------------- #
def other_coordination_signature(
    pair_features,
    *,
    pair_col: str = "pair_id",
    advantage_col: str = "table_advantage",
    template_cols: Sequence[str] = (
        "directed_transfer_score",
        "soft_play_score",
        "coordinated_isolation_score",
    ),
    labels=None,
    disclosed_family_col: str = "target_behavior",
) -> ProbeResult:
    """Compute H18-H19 statistics for the undisclosed ``other_coordination`` family.

    * **H18 table_advantage** - distribution of a behavior-agnostic joint-unit
      advantage measure; when labels distinguish disclosed families, test whether
      known positives *outside* the three disclosed families still sit in the
      upper tail of the advantage measure.
    * **H19 residual_cluster** - among trusted positives, identify those whose
      disclosed-family template scores are all low but advantage is high (the
      residual, candidate ``other_coordination`` cluster) and report its size and
      centroid vs. the templated positives.

    Because ``other_coordination`` is *excluded* from Behavior MAP, this signal
    informs Pair-AP / evidence routing, not a behavior label.
    """
    result = ProbeResult()
    if pair_features is None or len(pair_features) == 0:
        result.notes.append("pair_features empty/absent; other_coordination probes skipped")
        return result
    result.data_present = True
    cols = set(pair_features.columns)

    # ---- H18: table-advantage distribution -------------------------------
    if advantage_col in cols:
        adv = [float(v) for v in pair_features[advantage_col].dropna().tolist()]
        srt = sorted(adv)

        def _q(p: float):
            if not srt:
                return None
            idx = min(len(srt) - 1, max(0, int(round(p * (len(srt) - 1)))))
            return srt[idx]

        h18: Dict[str, object] = {
            "n": len(adv),
            "min": srt[0] if srt else None,
            "p50": _q(0.5),
            "p90": _q(0.9),
            "max": srt[-1] if srt else None,
        }
        # Label-aware upper-tail test for non-disclosed positives.
        if (
            labels is not None
            and pair_col in getattr(labels, "columns", [])
            and disclosed_family_col in getattr(labels, "columns", [])
        ):
            result.labels_present = True
            disclosed = {"directed_transfer", "soft_play", "coordinated_isolation"}
            lab = labels.set_index(pair_col)[disclosed_family_col].to_dict()
            p90 = _q(0.9)
            other_ids = [pid for pid, fam in lab.items() if str(fam) not in disclosed]
            feat = pair_features.set_index(pair_col)[advantage_col].to_dict()
            others_in_tail = [
                pid for pid in other_ids
                if pid in feat and p90 is not None and float(feat[pid]) >= p90
            ]
            h18["n_other_coordination_labeled"] = len(other_ids)
            h18["n_other_in_upper_decile"] = len(others_in_tail)
        result.hypotheses["H18"] = h18
    else:
        result.hypotheses["H18"] = {"note": f"no '{advantage_col}' column; advantage not computed"}

    # ---- H19: residual cluster -------------------------------------------
    present_templates = [c for c in template_cols if c in cols]
    if advantage_col in cols and present_templates and labels is not None:
        result.labels_present = True
        positive_ids = set(labels[pair_col].tolist()) if pair_col in labels.columns else set()
        feat_idx = pair_features.set_index(pair_col)
        # High advantage but low on every disclosed template.
        adv_vals = [float(v) for v in pair_features[advantage_col].dropna().tolist()]
        adv_hi = sorted(adv_vals)[int(0.5 * (len(adv_vals) - 1))] if adv_vals else 0.0
        residual = []
        for pid in positive_ids:
            if pid not in feat_idx.index:
                continue
            row = feat_idx.loc[pid]
            adv_v = float(row[advantage_col])
            templates_low = all(float(row[c]) <= adv_hi for c in present_templates)
            if adv_v >= adv_hi and templates_low:
                residual.append(pid)
        result.hypotheses["H19"] = {
            "residual_cluster_size": len(residual),
            "templates_considered": present_templates,
            "advantage_threshold_used": adv_hi,
            "note": "high advantage + low on all disclosed templates => other_coordination candidate",
        }
    else:
        result.hypotheses["H19"] = {
            "note": "need advantage_col + disclosed-template score columns + labels"
        }

    return result


# --------------------------------------------------------------------------- #
# Convenience runner
# --------------------------------------------------------------------------- #
def run_all_probes(tables: Mapping[str, object], *, labels=None) -> Dict[str, ProbeResult]:
    """Run every probe group over a mapping of loaded tables.

    ``tables`` maps names to DataFrames, e.g.::

        {"actions": df_actions, "hands": df_hands, "seats": df_seats,
         "hand_signals": df_signals, "pair_features": df_features}

    Missing tables simply yield a ProbeResult with ``data_present=False`` for that
    group, so this runs gracefully whether or not the competition data is present.
    Returns a dict keyed by probe-group name.
    """
    actions = tables.get("actions")
    hands = tables.get("hands")
    seats = tables.get("seats")
    hand_signals = tables.get("hand_signals")
    pair_features = tables.get("pair_features")

    return {
        "distribution_artifacts": distribution_artifacts(actions, seats, hands=hands),
        "id_ordering_regularities": id_ordering_regularities(hands, actions, seats),
        "pool_construction": pool_construction(hands, seats, labels=labels)
        if (hands is not None and seats is not None)
        else ProbeResult(notes=["hands/seats absent"]),
        "confounder_tells": confounder_tells(hand_signals, labels=labels)
        if hand_signals is not None
        else ProbeResult(notes=["hand_signals absent"]),
        "other_coordination_signature": other_coordination_signature(pair_features, labels=labels)
        if pair_features is not None
        else ProbeResult(notes=["pair_features absent"]),
    }
