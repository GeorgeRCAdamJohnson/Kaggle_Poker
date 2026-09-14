"""§62 NEGATIVE-SPACE pair-interaction feature block (the HOLE, not the mark).

Pre-registration: RESEARCH_DOSSIER §62 (append-only, bars FIXED before any feature was
computed). Governed by the workspace ``reverse-engineering-accountability`` contract
(Rules 1,2,3,4,5,9,11,15,16,17,18,19).

The idea (Rule 19 — model the HOLE)
-----------------------------------
Presence-only detectors read the MARK (a transfer/checkdown/isolation that HAPPENED).
§30/§31 proved the per-hand mark is an indistinguishable subset of look-alikes. The
negative-space reframe: for two players who share MANY hands, the ABSENCE of normal
adversarial interaction is the tell. Normal opponents DO clash; colluders systematically
do NOT. Every feature here is an EXPECTED-minus-OBSERVED residual normalized by
OPPORTUNITY (co-hands) and SAMPLE-SIZE SHRUNK (Rule 16), NEVER a raw count (raw counts
drifted at AUC 0.865 in §33 exp4b; rates fixed it to 0.639 — same discipline here,
pre-committed).

The three interactions (all derivable from the confirmed actions/seats/hands schema)
-------------------------------------------------------------------------------------
Over the CO-HANDS of a pair (hands where BOTH members were dealt in — the opportunity
denominator):
  (a) clash_3bet_residual: expected minus OBSERVED rate of one member raising/betting
      when the other is already the aggressor in the same hand (action_no ordering +
      action in {raise, bet}). Colluders clash LESS => expected-minus-observed > 0 is
      the tell.
  (b) contest_showdown_residual: expected minus OBSERVED rate of both going to showdown
      AGAINST each other in a large pot (both went_to_showdown, net_chips opposite
      signs, final_pot above the pool-high threshold). Colluders contest LESS => > 0.
  (c) fold_yield_residual: OBSERVED minus expected rate of one folding after the other
      raises/bets in the same hand (yielding the pot without contest). Colluders yield
      MORE than the field => OBSERVED-minus-expected > 0 is the positive tell (note the
      sign is flipped vs (a)/(b) exactly as §62 specifies).

The "expected" baselines (Rule: leakage-safe, pool-local)
---------------------------------------------------------
Each "expected" per-opportunity rate is computed from NON-labeled co-seated player
dyads in the SAME pool (development hands for dev pairs, evaluation hands for eval
pairs). It uses ONLY co-play structure, never labels. Concretely: sample a large set of
random co-seated dyads within the pool, compute the same per-co-hand interaction rate,
and take the opportunity-weighted field mean. That single field rate is the "expected"
for every target pair in that pool.

Sample-size shrinkage (Rule 16)
-------------------------------
Each residual is shrunk toward 0 by ``n / (n + k)`` where ``n`` is the pair's co-hand
count and ``k`` is a HYPERPARAMETER SWEPT on the holdout (never guessed once — Rule 13).
A "zero-clash" pair with 5 co-hands is thereby NOT ranked over a 300-co-hand pair.

The cardinal parity rule (§21)
------------------------------
Dev and eval features are produced by the SAME function :func:`build_negative_space`
called with different pool arguments. Separate code paths caused the 882M bug in §21;
there is exactly one path here.

Composition (Rule 15, mirrors exp3c/exp4c precisely)
----------------------------------------------------
The block enters as ADDITIONAL columns to the SAME single-XGB foundation fit
(foundation v5 + MFg + DIRc, the floor, PairAP 0.8934 §57) exactly how MFg/DIRc entered
in §32/§33. The seed-7 60/40 table-disjoint holdout PairAP judges foundation+block
against foundation-alone. A block that only hurts fails the +0.010 bar and is a NULL,
never submitted (this is NOT a fixed-weight blend — it is the verified exp3c path).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from anchor_repro.recipe_registry import (
    _FLOOR_XGB_PARAMS,
    _load_pool_map,
    _PU_WEIGHT_EVAL_ROWS,
    _SPLIT_CUT_FRAC,
    _SPLIT_SEED,
    _V5_DROP_COLUMNS,
    MissingCacheError,
)
from anchor_repro.own_submission_recipes import (
    DRIFT_AUC_THRESHOLD,
    DriftGateResult,
    adversarial_drift_auc_matrix,
)

__all__ = [
    "NEGATIVE_SPACE_COLUMNS",
    "NEGATIVE_SPACE_RANK_COLUMNS",
    "PRESENCE_COLUMNS",
    "DEFAULT_HIGH_POT_QUANTILE",
    "DEFAULT_FIELD_DYAD_SAMPLE",
    "NegativeSpaceCacheLocations",
    "default_negative_space_cache_locations",
    "PairStats",
    "compute_pair_stats",
    "block_from_stats",
    "build_negative_space",
    "negative_space_drift_gate",
    "run_gate_b_ksweep",
    "GateBSweepResult",
    "presence_vs_residual_concentration",
]

# The GATED/FITTED feature columns emitted PER PAIR.
#
# DRIFT-DEBUGGING NOTE (§63, Rule 11 — a GATE-A fail is a debugging task, not a verdict):
# the FIRST execution of this block emitted seven columns — three expected-minus-observed
# residuals shrunk by n/(n+k), the three raw observed rates, AND ``ns_co_hands_log``.
# GATE A measured AUC 0.9992 (near-perfect dev/eval separation). Diagnostics (§63) showed
# NO single column separated the pools (max 1D-AUC 0.73, all pool-percentile marginals
# = 0.50); the separation lived ENTIRELY in the MULTIVARIATE interaction + the discrete
# tie-structure of the SEPARATE residual columns (the eval pool's lower co-hand count,
# median ~76 vs dev ~110, gives each per-co-hand rate a COARSER, differently-tied grid).
# ``ns_co_hands_log`` is also a count transform (it drifted 1D at 0.728) and violates
# §62's own "EVERY feature is a RATE or residual, NEVER a raw count" rule.
#
# THE FIX (debugged execution, still the SAME idea): (1) use an empirical-Bayes POSTERIOR
# rate ``(count + k*field)/(n + k)`` — a bounded [0,1] shrunk rate, so ``n`` enters ONLY
# as the shrink trust, never as a raw scale (Rule 16); (2) turn each residual into its
# POOL-RELATIVE percentile rank (location/scale free — removes any pool-level base-rate
# shift, leakage-safe since the rank is WITHIN the pool); (3) COLLAPSE the three ranks
# into ONE composite anomaly score (their mean), which removes the multivariate
# interaction + tie-structure XGB was exploiting. The single composite is the gated/fit
# feature and measures GATE-A AUC ~0.52-0.53 across all k (PASS). This is the negative-
# space idea EXECUTED until it transferred, not a retreat to the herd (Rules 10, 11, 18).
NEGATIVE_SPACE_COLUMNS: Tuple[str, ...] = (
    "ns_anomaly_composite",
)

#: The three per-interaction pool-relative percentile-rank residuals that the composite
#: is the mean of. Retained as NAMED diagnostic columns (NOT fitted directly — fitting
#: them separately is exactly what drifted at 0.78) so the per-interaction contribution
#: is auditable.
NEGATIVE_SPACE_RANK_COLUMNS: Tuple[str, ...] = (
    "ns_clash_rank",
    "ns_contest_rank",
    "ns_foldyield_rank",
)

#: Presence-only counterpart columns (the "mark" the residual is compared against in the
#: concentration diagnostic). These are the OBSERVED per-co-hand rates WITHOUT the
#: expected-minus-observed reframe and WITHOUT shrinkage — i.e. exactly what a
#: presence-only detector would read.
PRESENCE_COLUMNS: Tuple[str, ...] = (
    "pres_clash_rate",
    "pres_contest_rate",
    "pres_foldyield_rate",
)

#: Pool-high pot threshold for the contest-to-showdown interaction (quantile of
#: ``hands.final_pot`` WITHIN the pool — pool-local, Rule 4). 0.75 => "large pot".
DEFAULT_HIGH_POT_QUANTILE: float = 0.75

#: Number of random co-seated NON-target dyads sampled to estimate the pool field
#: baseline rate for each interaction (opportunity-weighted). Large enough to be a
#: stable field estimate; capped so the build stays CPU-cheap.
DEFAULT_FIELD_DYAD_SAMPLE: int = 20_000

#: RNG seed for the field-dyad sampling (fixed for reproducibility / parity).
_FIELD_SAMPLE_SEED: int = 7


@dataclass(frozen=True)
class NegativeSpaceCacheLocations:
    """Resolved on-disk paths the negative-space builder + gates read verbatim.

    Attributes:
        dev_v5: ``feature_cache/v5/dev_v5.parquet`` — its ``pair_id`` order is the
            authoritative dev row order the block is aligned to (so it hstacks onto the
            floor foundation with no reindex, exactly like MFg/DIRc).
        eval_v5: ``feature_cache/v5/eval_v5.parquet`` — authoritative eval row order.
        dev_pairs_v2: ``feature_cache/v2/dev_pairs_v2.parquet`` (pair_id, p_low, p_high,
            label — the dev pair→player map + PU label + pool key source).
        eval_pairs_csv: ``data/poker/evaluation_pairs.csv`` (pair_id, player_1,
            player_2 — the eval pair→player map).
        mfg_dev / mfg_eval: ``feature_cache/v7/_MFg_{dev,eval}.npy`` (floor MFg block).
        dirc_dev / dirc_eval: ``feature_cache/v7/_DIRc_{dev,eval}.npy`` (floor DIRc).
        actions / seats / hands: the raw ``data/poker`` parquets (interactions + pool).
    """

    dev_v5: Path
    eval_v5: Path
    dev_pairs_v2: Path
    eval_pairs_csv: Path
    mfg_dev: Path
    mfg_eval: Path
    dirc_dev: Path
    dirc_eval: Path
    actions: Path
    seats: Path
    hands: Path

    def missing(self) -> List[Path]:
        """Return the subset of required paths absent from disk."""
        return [
            p
            for p in (
                self.dev_v5,
                self.eval_v5,
                self.dev_pairs_v2,
                self.eval_pairs_csv,
                self.mfg_dev,
                self.mfg_eval,
                self.dirc_dev,
                self.dirc_eval,
                self.actions,
                self.seats,
                self.hands,
            )
            if not Path(p).is_file()
        ]


def default_negative_space_cache_locations(poker_root: Path) -> NegativeSpaceCacheLocations:
    """Resolve the standard negative-space cache locations under a ``poker/`` tree."""
    root = Path(poker_root)
    fc = root / "outputs" / "poker_collusion" / "feature_cache"
    data = root / "data" / "poker"
    return NegativeSpaceCacheLocations(
        dev_v5=fc / "v5" / "dev_v5.parquet",
        eval_v5=fc / "v5" / "eval_v5.parquet",
        dev_pairs_v2=fc / "v2" / "dev_pairs_v2.parquet",
        eval_pairs_csv=data / "evaluation_pairs.csv",
        mfg_dev=fc / "v7" / "_MFg_dev.npy",
        mfg_eval=fc / "v7" / "_MFg_eval.npy",
        dirc_dev=fc / "v7" / "_DIRc_dev.npy",
        dirc_eval=fc / "v7" / "_DIRc_eval.npy",
        actions=data / "actions.parquet",
        seats=data / "seats.parquet",
        hands=data / "hands.parquet",
    )


# --------------------------------------------------------------------------- #
# The SHARED dev/eval interaction extractor (the ONE code path — §21 parity).  #
# --------------------------------------------------------------------------- #


def _hand_interaction_tables(
    actions_path: Path,
    seats_path: Path,
    hands_path: Path,
    pool: str,
    *,
    high_pot_quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build the per-(hand,player) interaction primitives for ONE pool.

    Returns ``(seat_pool, hand_aggr, high_pot_threshold)`` where:
      * ``seat_pool``: rows (hand_id, player_id, went_to_showdown, net_chips) for hands
        in ``pool`` — the DEALT-IN membership + showdown/net primitives.
      * ``hand_aggr``: per (hand_id, player_id) action-derived primitives:
        ``first_aggr_no`` (first action_no where the player raised/bet, NaN if never),
        ``folded`` (did the player fold in the hand), ``final_pot``, ``is_high_pot``.
      * ``high_pot_threshold``: the pool-local ``final_pot`` quantile (Rule 4 scope).

    Pool-local by construction: only hands whose ``hands.phase == pool`` are read.
    """
    hands = pd.read_parquet(hands_path, columns=["hand_id", "phase", "final_pot"])
    hands = hands[hands["phase"] == pool]
    if hands.empty:
        raise MissingCacheError(
            f"no hands found for pool {pool!r}; cannot build pool-local negative space"
        )
    hand_ids = hands.set_index("hand_id")
    high_pot_threshold = float(hands["final_pot"].quantile(high_pot_quantile))

    seats = pd.read_parquet(
        seats_path,
        columns=["hand_id", "player_id", "went_to_showdown", "net_chips"],
    )
    seat_pool = seats[seats["hand_id"].isin(hand_ids.index)].copy()

    # Actions: only raise/bet (aggression) and fold matter for the three interactions.
    actions = pd.read_parquet(
        actions_path, columns=["hand_id", "action_no", "player_id", "action"]
    )
    actions = actions[actions["hand_id"].isin(hand_ids.index)]

    aggr = actions[actions["action"].isin(["raise", "bet"])]
    first_aggr = (
        aggr.groupby(["hand_id", "player_id"])["action_no"].min().rename("first_aggr_no")
    )
    folded = (
        actions.assign(_f=(actions["action"] == "fold"))
        .groupby(["hand_id", "player_id"])["_f"]
        .max()
        .rename("folded_action")
    )
    hand_aggr = pd.concat([first_aggr, folded], axis=1).reset_index()
    hand_aggr = hand_aggr.merge(
        hands[["hand_id", "final_pot"]], on="hand_id", how="left"
    )
    hand_aggr["is_high_pot"] = hand_aggr["final_pot"] >= high_pot_threshold
    return seat_pool, hand_aggr, high_pot_threshold


def _pair_interaction_counts(
    pairs: pd.DataFrame,
    seat_pool: pd.DataFrame,
    hand_aggr: pd.DataFrame,
) -> pd.DataFrame:
    """Co-hands + OBSERVED clash/contest/foldyield counts per dyad (auditable path).

    See :func:`_pair_interaction_counts` docstring for the definitions. This is the
    implementation actually used; it builds ONE per-(dyad, co-hand) long frame carrying
    BOTH members' per-hand primitives, then reduces to per-dyad counts.
    """
    members = set(pd.unique(pd.concat([pairs["a"], pairs["b"]], ignore_index=True)))

    # Per-(hand,player) primitives for the relevant members only.
    ha = hand_aggr[hand_aggr["player_id"].isin(members)].copy()
    ha["first_aggr_no"] = ha["first_aggr_no"].astype(float)
    ha["folded_action"] = ha["folded_action"].fillna(False).astype(bool)
    sp = seat_pool[seat_pool["player_id"].isin(members)][
        ["hand_id", "player_id", "went_to_showdown", "net_chips"]
    ].copy()
    # Dealt-in primitive frame: seats is the DEALT-IN truth; left-join action primitives.
    prim = sp.merge(ha, on=["hand_id", "player_id"], how="left")
    prim["first_aggr_no"] = prim["first_aggr_no"].astype(float)
    prim["folded_action"] = prim["folded_action"].fillna(False).astype(bool)
    prim["is_high_pot"] = prim["is_high_pot"].fillna(False).astype(bool)
    prim["went_to_showdown"] = prim["went_to_showdown"].fillna(False).astype(bool)
    prim["net_chips"] = prim["net_chips"].astype(float)

    # Join dyads to member-a's primitives, then to member-b's on the SAME hand_id.
    a_side = prim.rename(
        columns={
            "player_id": "a",
            "first_aggr_no": "a_aggr_no",
            "folded_action": "a_folded",
            "went_to_showdown": "a_sd",
            "net_chips": "a_net",
            "is_high_pot": "a_hp",
        }
    )
    b_side = prim.rename(
        columns={
            "player_id": "b",
            "first_aggr_no": "b_aggr_no",
            "folded_action": "b_folded",
            "went_to_showdown": "b_sd",
            "net_chips": "b_net",
        }
    )[["hand_id", "b", "b_aggr_no", "b_folded", "b_sd", "b_net"]]

    co = pairs.merge(a_side, on="a", how="inner").merge(
        b_side, on=["hand_id", "b"], how="inner"
    )
    if co.empty:
        return pd.DataFrame(
            {
                "pair_key": pairs["pair_key"].to_numpy(),
                "co_hands": 0,
                "clash": 0,
                "contest": 0,
                "foldyield": 0,
            }
        )

    a_agg = co["a_aggr_no"].to_numpy()
    b_agg = co["b_aggr_no"].to_numpy()
    a_is_aggr = ~np.isnan(a_agg)
    b_is_aggr = ~np.isnan(b_agg)

    # (a) clash: BOTH members raised/bet in the same hand (they contested aggression).
    co["_clash"] = (a_is_aggr & b_is_aggr).astype(np.int64)

    # (b) contest to showdown in a big pot: both saw showdown, opposite net signs,
    #     high pot.
    a_net = co["a_net"].to_numpy()
    b_net = co["b_net"].to_numpy()
    opp_sign = (np.sign(a_net) * np.sign(b_net)) < 0
    both_sd = co["a_sd"].to_numpy() & co["b_sd"].to_numpy()
    co["_contest"] = (both_sd & opp_sign & co["a_hp"].to_numpy()).astype(np.int64)

    # (c) fold-yield: exactly one folded AND the OTHER was an aggressor (one yielded to
    #     the other's aggression). Symmetric over who folded.
    a_yield = co["a_folded"].to_numpy() & b_is_aggr & (~co["b_folded"].to_numpy())
    b_yield = co["b_folded"].to_numpy() & a_is_aggr & (~co["a_folded"].to_numpy())
    co["_foldyield"] = (a_yield | b_yield).astype(np.int64)

    grp = co.groupby("pair_key").agg(
        co_hands=("hand_id", "size"),
        clash=("_clash", "sum"),
        contest=("_contest", "sum"),
        foldyield=("_foldyield", "sum"),
    )
    # Pairs with zero co-hands (never both dealt in) get zeros.
    out = (
        pairs[["pair_key"]]
        .drop_duplicates()
        .merge(grp, on="pair_key", how="left")
        .fillna(0)
    )
    for c in ("co_hands", "clash", "contest", "foldyield"):
        out[c] = out[c].astype(np.int64)
    return out


def _field_baseline_rates(
    seat_pool: pd.DataFrame,
    hand_aggr: pd.DataFrame,
    exclude_players: set,
    *,
    n_dyads: int,
    seed: int,
) -> Dict[str, float]:
    """Estimate the pool field per-co-hand rates from random NON-target co-seated dyads.

    Leakage-safe (Rule): samples dyads that are NOT any target pair member (colluders
    excluded) purely from co-play structure, computes the same three interaction rates,
    and returns the opportunity-weighted field mean rate for each interaction. This is
    the "expected" rate every target pair's residual is measured against.

    Args:
        exclude_players: players that appear in ANY target pair (kept OUT of the field
            sample so the baseline is a clean NON-pair reference).
        n_dyads: number of random co-seated dyads to sample.
        seed: RNG seed (fixed).
    """
    rng = np.random.default_rng(seed)
    # Build hand -> list of dealt-in players, restricted to non-excluded players.
    sp = seat_pool[~seat_pool["player_id"].isin(exclude_players)]
    by_hand = sp.groupby("hand_id")["player_id"].apply(list)
    by_hand = by_hand[by_hand.map(len) >= 2]
    if by_hand.empty:
        return {"clash": 0.0, "contest": 0.0, "foldyield": 0.0}

    hand_index = by_hand.index.to_numpy()
    # Sample random co-seated dyads: pick a random hand, then two distinct players in it.
    sampled: Dict[frozenset, None] = {}
    attempts = 0
    max_attempts = n_dyads * 6
    dyad_a: List = []
    dyad_b: List = []
    while len(dyad_a) < n_dyads and attempts < max_attempts:
        attempts += 1
        h = hand_index[rng.integers(len(hand_index))]
        players = by_hand.loc[h]
        if len(players) < 2:
            continue
        i, j = rng.choice(len(players), size=2, replace=False)
        pa, pb = players[i], players[j]
        key = frozenset((pa, pb))
        if key in sampled:
            continue
        sampled[key] = None
        dyad_a.append(pa)
        dyad_b.append(pb)

    if not dyad_a:
        return {"clash": 0.0, "contest": 0.0, "foldyield": 0.0}

    field_pairs = pd.DataFrame(
        {
            "pair_key": [f"F{i}" for i in range(len(dyad_a))],
            "a": dyad_a,
            "b": dyad_b,
        }
    )
    counts = _pair_interaction_counts(field_pairs, seat_pool, hand_aggr)
    counts = counts[counts["co_hands"] > 0]
    if counts.empty:
        return {"clash": 0.0, "contest": 0.0, "foldyield": 0.0}
    denom = float(counts["co_hands"].sum())
    return {
        "clash": float(counts["clash"].sum()) / denom,
        "contest": float(counts["contest"].sum()) / denom,
        "foldyield": float(counts["foldyield"].sum()) / denom,
    }


def _rank01(x: np.ndarray) -> np.ndarray:
    """Average-rank of ``x`` mapped to (0,1] — the pool-relative percentile rank.

    Location/scale free (removes any pool-level base-rate shift), and computed WITHIN
    the pool it is called on (leakage-safe: eval ranks use only eval structure).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    s = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return (ranks + 1.0) / n


def _residual_block(
    counts: pd.DataFrame,
    field: Dict[str, float],
    k: float,
) -> pd.DataFrame:
    """Per-pair EB-posterior residuals -> pool-percentile ranks -> composite (§63 fix).

    Rule 16 shrinkage via an empirical-Bayes POSTERIOR RATE ``(count + k*field)/(n + k)``
    (bounded [0,1]; ``n`` enters only as shrink trust, NEVER as a raw scale — the raw
    ``n``-scaled residual is what drove the 0.9992 drift). Sign convention per §62:
      * clash / contest residual = expected - posterior  (colluders clash/contest LESS).
      * foldyield residual       = posterior - expected   (colluders yield MORE).
    Each residual is then turned into its POOL-RELATIVE percentile rank and the three
    ranks are averaged into ONE composite anomaly score (the fitted feature). The three
    ranks are also returned as named diagnostics. Presence-only OBSERVED rates (the
    "mark") are returned for the §62 concentration diagnostic.
    """
    n = counts["co_hands"].to_numpy().astype(float)

    def _posterior(cnt: np.ndarray, f: float) -> np.ndarray:
        # EB posterior rate: pulls the observed rate toward the field rate f with
        # strength k; for n=0 it returns exactly f (neutral — no anomaly).
        return (cnt + float(k) * f) / (n + float(k))

    clash_cnt = counts["clash"].to_numpy().astype(float)
    contest_cnt = counts["contest"].to_numpy().astype(float)
    fold_cnt = counts["foldyield"].to_numpy().astype(float)

    clash_post = _posterior(clash_cnt, field["clash"])
    contest_post = _posterior(contest_cnt, field["contest"])
    fold_post = _posterior(fold_cnt, field["foldyield"])

    clash_res = field["clash"] - clash_post
    contest_res = field["contest"] - contest_post
    fold_res = fold_post - field["foldyield"]

    clash_rank = _rank01(clash_res)
    contest_rank = _rank01(contest_res)
    fold_rank = _rank01(fold_res)
    composite = (clash_rank + contest_rank + fold_rank) / 3.0

    # presence-only OBSERVED rates (the "mark"): raw per-co-hand rate, no reframe/shrink.
    clash_obs = np.where(n > 0, clash_cnt / np.maximum(n, 1), 0.0)
    contest_obs = np.where(n > 0, contest_cnt / np.maximum(n, 1), 0.0)
    fold_obs = np.where(n > 0, fold_cnt / np.maximum(n, 1), 0.0)

    return pd.DataFrame(
        {
            "pair_key": counts["pair_key"].to_numpy(),
            "ns_anomaly_composite": composite,
            "ns_clash_rank": clash_rank,
            "ns_contest_rank": contest_rank,
            "ns_foldyield_rank": fold_rank,
            "pres_clash_rate": clash_obs,
            "pres_contest_rate": contest_obs,
            "pres_foldyield_rate": fold_obs,
        }
    )


@dataclass(frozen=True)
class PairStats:
    """k-INDEPENDENT negative-space statistics for one pool (the expensive part).

    Split out from :func:`build_negative_space` so the k-sweep (Rule 13) computes the
    pool read + per-pair OBSERVED counts + pool field baseline ONCE, then re-derives the
    shrunk residuals cheaply for each k. Everything here is fixed BEFORE k is chosen, so
    sweeping k cannot change the observed counts or the field baseline — only the
    ``n/(n+k)`` shrink applied to them.

    Attributes:
        pair_order: ``pair_id`` + ``pair_key`` in the input (dev_v5 / eval) row order.
        counts: per-pair OBSERVED co_hands + clash/contest/foldyield counts.
        field: pool field per-co-hand rates (the "expected" baseline).
        pool: which pool ('development' / 'evaluation') — recorded for the audit.
    """

    pair_order: pd.DataFrame
    counts: pd.DataFrame
    field: Dict[str, float]
    pool: str


def compute_pair_stats(
    pairs: pd.DataFrame,
    pool: str,
    caches: NegativeSpaceCacheLocations,
    *,
    high_pot_quantile: float = DEFAULT_HIGH_POT_QUANTILE,
    n_field_dyads: int = DEFAULT_FIELD_DYAD_SAMPLE,
    positive_players: Optional[set] = None,
) -> PairStats:
    """Compute the k-INDEPENDENT pool statistics (expensive read + counts + field).

    THIS is the single dev/eval code path (§21 parity): the same function reads the
    ``development`` pool for dev pairs and the ``evaluation`` pool for eval pairs.
    """
    seat_pool, hand_aggr, _thr = _hand_interaction_tables(
        caches.actions,
        caches.seats,
        caches.hands,
        pool,
        high_pot_quantile=high_pot_quantile,
    )
    work = pairs.copy()
    work["pair_key"] = work["pair_id"].astype(str)
    counts = _pair_interaction_counts(work[["pair_key", "a", "b"]], seat_pool, hand_aggr)
    exclude = set(pd.unique(pd.concat([work["a"], work["b"]], ignore_index=True)))
    if positive_players:
        exclude |= set(positive_players)
    field = _field_baseline_rates(
        seat_pool, hand_aggr, exclude, n_dyads=n_field_dyads, seed=_FIELD_SAMPLE_SEED
    )
    return PairStats(
        pair_order=work[["pair_id", "pair_key"]].copy(),
        counts=counts,
        field=field,
        pool=pool,
    )


def block_from_stats(stats: PairStats, k: float) -> pd.DataFrame:
    """Cheap per-k residual assembly from pre-computed :class:`PairStats` (Rule 13).

    Applies ONLY the ``n/(n+k)`` shrink; the observed counts + field baseline are fixed.
    Re-aligned to the input pair order (parity: dev_v5 / eval order is authoritative).
    """
    block = _residual_block(stats.counts, stats.field, k)
    block = stats.pair_order.merge(block, on="pair_key", how="left")
    fill_cols = (
        list(NEGATIVE_SPACE_COLUMNS)
        + list(NEGATIVE_SPACE_RANK_COLUMNS)
        + list(PRESENCE_COLUMNS)
    )
    block[fill_cols] = block[fill_cols].fillna(0.0)
    return block[["pair_id"] + fill_cols]


def build_negative_space(
    pairs: pd.DataFrame,
    pool: str,
    caches: NegativeSpaceCacheLocations,
    *,
    k: float,
    high_pot_quantile: float = DEFAULT_HIGH_POT_QUANTILE,
    n_field_dyads: int = DEFAULT_FIELD_DYAD_SAMPLE,
    positive_players: Optional[set] = None,
) -> pd.DataFrame:
    """Build the negative-space block for a set of pairs in ONE pool (the shared path).

    Convenience wrapper: :func:`compute_pair_stats` then :func:`block_from_stats` for a
    single k. The k-sweep calls the two steps directly so the expensive pool read runs
    once (Rule 13). Called with the dev pairs + ``pool='development'`` for dev, and the
    eval pairs + ``pool='evaluation'`` for eval.

    Args:
        pairs: frame with columns ``pair_id, a, b`` (a/b = the two player ids). Row
            order is preserved in the output.
        pool: ``'development'`` or ``'evaluation'`` — selects the pool-local hands
            (leakage-safe: dev features never read eval hands and vice versa).
        caches: resolved paths.
        k: shrinkage hyperparameter (SWEPT on the holdout — Rule 13).
        high_pot_quantile: pool-local ``final_pot`` quantile for the big-pot contest.
        n_field_dyads: field-baseline dyad sample size.
        positive_players: players to EXCLUDE from the field baseline sample (known
            colluders). For dev pass the confirmed-positive players; for eval pass the
            empty set (no labels — the field is all co-seated dyads). Target-pair
            members are always excluded regardless.

    Returns:
        A frame with ``pair_id`` (in the input order) + :data:`NEGATIVE_SPACE_COLUMNS`
        + :data:`PRESENCE_COLUMNS`.
    """
    stats = compute_pair_stats(
        pairs, pool, caches,
        high_pot_quantile=high_pot_quantile,
        n_field_dyads=n_field_dyads,
        positive_players=positive_players,
    )
    return block_from_stats(stats, k)


# --------------------------------------------------------------------------- #
# Pair→player maps for dev and eval (built through the SAME helper).           #
# --------------------------------------------------------------------------- #


def _dev_pairs_ab(caches: NegativeSpaceCacheLocations) -> pd.DataFrame:
    """Return dev pairs in ``dev_v5`` row order with columns ``pair_id, a, b``."""
    dev_v5 = pd.read_parquet(caches.dev_v5, columns=["pair_id"])
    dp = pd.read_parquet(caches.dev_pairs_v2)[["pair_id", "p_low", "p_high"]]
    m = dev_v5.merge(dp, on="pair_id", how="left")
    return pd.DataFrame(
        {"pair_id": m["pair_id"].to_numpy(), "a": m["p_low"].to_numpy(), "b": m["p_high"].to_numpy()}
    )


def _eval_pairs_ab(caches: NegativeSpaceCacheLocations) -> pd.DataFrame:
    """Return eval pairs in ``eval_v5`` row order with columns ``pair_id, a, b``."""
    eval_v5 = pd.read_parquet(caches.eval_v5, columns=["pair_id"])
    ep = pd.read_csv(caches.eval_pairs_csv)[["pair_id", "player_1", "player_2"]]
    m = eval_v5.merge(ep, on="pair_id", how="left")
    return pd.DataFrame(
        {
            "pair_id": m["pair_id"].to_numpy(),
            "a": m["player_1"].to_numpy(),
            "b": m["player_2"].to_numpy(),
        }
    )


def _dev_positive_players(caches: NegativeSpaceCacheLocations) -> set:
    """Confirmed-positive players (label==1) to exclude from the dev field baseline."""
    dp = pd.read_parquet(caches.dev_pairs_v2)[["p_low", "p_high", "label"]]
    pos = dp[dp["label"] == 1]
    return set(pos["p_low"]).union(set(pos["p_high"]))


def _load_foundation_matrices(
    caches: NegativeSpaceCacheLocations,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble the FLOOR foundation matrices (v5 + MFg + DIRc) for dev and eval.

    Mirrors ``recipe_registry.rerun_floor_stack`` / ``_load_exp3c_feature_matrices``:
    ``feats5`` = dev_v5 columns present in BOTH dev+eval minus the bookkeeping, then
    hstack MFg then DIRc in saved order. Returns ``(Xd_found, Xe_found, feats5)``.
    """
    dev_v5 = pd.read_parquet(caches.dev_v5)
    eval_v5 = pd.read_parquet(caches.eval_v5)
    feats5 = [c for c in dev_v5.columns if c in eval_v5.columns and c not in _V5_DROP_COLUMNS]
    Xd5 = dev_v5[feats5].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xe5 = eval_v5[feats5].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Mgd = np.load(caches.mfg_dev)
    Mge = np.load(caches.mfg_eval)
    DIRd = np.load(caches.dirc_dev)
    DIRe = np.load(caches.dirc_eval)
    if not (len(Xd5) == len(Mgd) == len(DIRd)):
        raise MissingCacheError(
            f"dev foundation row mismatch (v5={len(Xd5)}, MFg={len(Mgd)}, DIRc={len(DIRd)})"
        )
    if not (len(Xe5) == len(Mge) == len(DIRe)):
        raise MissingCacheError(
            f"eval foundation row mismatch (v5={len(Xe5)}, MFg={len(Mge)}, DIRc={len(DIRe)})"
        )
    Xd = np.hstack([Xd5, Mgd, DIRd]).astype(np.float32)
    Xe = np.hstack([Xe5, Mge, DIRe]).astype(np.float32)
    return Xd, Xe, feats5


# --------------------------------------------------------------------------- #
# GATE A — adversarial drift (RUN FIRST, Rules 5, 17). Pre-registered <0.65.   #
# --------------------------------------------------------------------------- #


def negative_space_drift_gate(
    caches: NegativeSpaceCacheLocations,
    *,
    k: float,
    high_pot_quantile: float = DEFAULT_HIGH_POT_QUANTILE,
    n_field_dyads: int = DEFAULT_FIELD_DYAD_SAMPLE,
    seed: int = 7,
) -> DriftGateResult:
    """Run the adversarial dev-vs-eval drift gate on the negative-space block ONLY.

    Builds the dev block (dev pairs, ``development`` pool) and eval block (eval pairs,
    ``evaluation`` pool) through the SAME :func:`build_negative_space`, then classifies
    dev-vs-eval on the residual columns via
    :func:`own_submission_recipes.adversarial_drift_auc_matrix` (identical machinery to
    §51/§33). PASS iff AUC < :data:`own_submission_recipes.DRIFT_AUC_THRESHOLD`.

    Only the :data:`NEGATIVE_SPACE_COLUMNS` are gated (the features that actually enter
    the fit); the presence-only columns are diagnostic-only and excluded.
    """
    missing = caches.missing()
    if missing:
        raise MissingCacheError(
            "negative-space drift gate cannot run; missing: "
            + ", ".join(str(p) for p in missing)
        )
    dev_pairs = _dev_pairs_ab(caches)
    eval_pairs = _eval_pairs_ab(caches)
    pos_players = _dev_positive_players(caches)

    dev_block = build_negative_space(
        dev_pairs, "development", caches,
        k=k, high_pot_quantile=high_pot_quantile,
        n_field_dyads=n_field_dyads, positive_players=pos_players,
    )
    eval_block = build_negative_space(
        eval_pairs, "evaluation", caches,
        k=k, high_pot_quantile=high_pot_quantile,
        n_field_dyads=n_field_dyads, positive_players=set(),
    )
    cols = list(NEGATIVE_SPACE_COLUMNS)
    Xd = dev_block[cols].to_numpy(np.float32)
    Xe = eval_block[cols].to_numpy(np.float32)
    return adversarial_drift_auc_matrix(Xd, Xe, block="negative_space", seed=seed)


# --------------------------------------------------------------------------- #
# GATE B — foundation vs foundation+block on the seed-7 60/40 table holdout.    #
# k-sweep (Rule 13) + presence-vs-residual concentration diagnostic (§62).      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateBSweepResult:
    """Measured GATE B outcome (foundation-alone + the full k-sweep). Rules 2, 3, 9."""

    foundation_pair_ap: float
    k_values: Tuple[float, ...]
    block_pair_ap: Tuple[float, ...]
    deltas: Tuple[float, ...]
    n_holdout_confirmed: int
    bar: float = 0.010

    @property
    def best_k(self) -> float:
        i = int(np.argmax(self.block_pair_ap))
        return self.k_values[i]

    @property
    def best_delta(self) -> float:
        return float(max(self.deltas))

    @property
    def clears_bar(self) -> bool:
        return self.best_delta >= self.bar


def _seed7_split(dev: pd.DataFrame, caches: NegativeSpaceCacheLocations) -> Tuple[np.ndarray, np.ndarray]:
    """The seed-7 60/40 table-disjoint split masks (verbatim from the floor)."""
    player_pool = _load_pool_map(caches.seats, caches.hands)
    dev_table = dev["p_low"].map(player_pool)
    tabs = sorted(set(str(t) for t in dev_table.fillna("NA")))
    rng = np.random.default_rng(_SPLIT_SEED)
    rng.shuffle(tabs)
    cut = int(len(tabs) * _SPLIT_CUT_FRAC)
    train_tabs = set(tabs[:cut])
    tr = dev_table.astype(str).isin(train_tabs).to_numpy()
    return tr, ~tr


def run_gate_b_ksweep(
    caches: NegativeSpaceCacheLocations,
    labels: pd.DataFrame,
    scorer,
    *,
    k_values: Tuple[float, ...],
    high_pot_quantile: float = DEFAULT_HIGH_POT_QUANTILE,
    n_field_dyads: int = DEFAULT_FIELD_DYAD_SAMPLE,
    evidence: Optional[pd.DataFrame] = None,
) -> GateBSweepResult:
    """GATE B: canonical PairAP of foundation-alone vs foundation+block over swept k.

    Fits ONE XGB (the floor ``_FLOOR_XGB_PARAMS``) on the 60% train tables and scores
    canonical confirmed-only PairAP on the 40% held-out tables (the fit never touched)
    via ``scorer.score_dev_predictions`` — exactly the exp3c/exp4c path (Rule 15: the
    block enters as EXTRA columns to the same fit, so foundation-alone is the floor).

    Because the negative-space k only changes the block columns, foundation-alone is
    computed ONCE and every swept-k model re-uses the SAME train/holdout split, PU
    weights, and foundation matrix.

    Args:
        k_values: the shrinkage k grid to sweep (Rule 13 — the FULL sweep is reported).

    Returns:
        A :class:`GateBSweepResult` with foundation-alone PairAP and per-k block PairAP.
    """
    from xgboost import XGBClassifier

    missing = caches.missing()
    if missing:
        raise MissingCacheError(
            "GATE B cannot run; missing: " + ", ".join(str(p) for p in missing)
        )

    # dev frame in dev_v5 order + PU label/pool keys (exactly the floor's merge).
    dev_v5 = pd.read_parquet(caches.dev_v5)
    dp = pd.read_parquet(caches.dev_pairs_v2)[["pair_id", "p_low", "p_high", "label"]]
    dev = dev_v5.merge(dp, on="pair_id", how="left")

    Xd_found, _Xe_found, _feats5 = _load_foundation_matrices(caches)
    if len(Xd_found) != len(dev):
        raise MissingCacheError(
            f"dev foundation rows ({len(Xd_found)}) != dev rows ({len(dev)})"
        )

    lab = dev["label"].to_numpy()
    y = (lab == 1).astype(int)
    known = lab >= 0
    n_pu = int((lab == -1).sum())
    fw = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35))

    tr, ho = _seed7_split(dev, caches)

    dev_pairs = _dev_pairs_ab(caches)
    pos_players = _dev_positive_players(caches)

    def _fit_score(Xfull: np.ndarray) -> Tuple[float, int]:
        model = XGBClassifier(**_FLOOR_XGB_PARAMS)
        model.fit(Xfull[tr], y[tr], sample_weight=fw[tr])
        ho_proba = model.predict_proba(Xfull[ho])[:, 1]
        preds = _emit_confirmed(dev, ho, ho_proba, labels, evidence)
        return float(scorer.score_dev_predictions(preds).pair_ap), int(len(preds))

    foundation_ap, n_conf = _fit_score(Xd_found)

    # k-INDEPENDENT stats computed ONCE (Rule 13): sweeping k only re-applies the
    # n/(n+k) shrink, never re-reads the pool or recomputes observed counts/baseline.
    stats = compute_pair_stats(
        dev_pairs, "development", caches,
        high_pot_quantile=high_pot_quantile,
        n_field_dyads=n_field_dyads,
        positive_players=pos_players,
    )
    block_aps: List[float] = []
    for k in k_values:
        block = block_from_stats(stats, k)  # already dev_v5 order
        Xblk = block[list(NEGATIVE_SPACE_COLUMNS)].to_numpy(np.float32)
        Xfull = np.hstack([Xd_found, Xblk]).astype(np.float32)
        ap, _ = _fit_score(Xfull)
        block_aps.append(ap)

    deltas = tuple(a - foundation_ap for a in block_aps)
    return GateBSweepResult(
        foundation_pair_ap=foundation_ap,
        k_values=tuple(float(k) for k in k_values),
        block_pair_ap=tuple(block_aps),
        deltas=deltas,
        n_holdout_confirmed=n_conf,
    )


def _emit_confirmed(dev, ho, ho_proba, labels, evidence):
    """CanonicalScorer-ready confirmed-holdout frame (same construction as the floor)."""
    from poker_collusion.submission.writer import SUBMISSION_COLUMNS
    from poker_collusion.validation.cv import build_fold_solution

    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    ho_pairs = dev["pair_id"].astype(str).to_numpy()[ho]
    keep = np.array([pid in confirmed_ids for pid in ho_pairs])
    pred_pairs = ho_pairs[keep]
    pred_scores = ho_proba[keep]
    solution = build_fold_solution(labels, list(pred_pairs), evidence=evidence)
    score_by_pair = dict(zip((str(p) for p in pred_pairs), (float(s) for s in pred_scores)))
    kept = [pid for pid in solution["pair_id"].astype(str) if pid in score_by_pair]
    preds = pd.DataFrame({"pair_id": kept})
    preds["risk_score"] = [score_by_pair[pid] for pid in kept]
    preds["risk_score"] = preds["risk_score"].astype(float).clip(0.0, 1.0)
    preds["predicted_behavior"] = "none"
    for col in SUBMISSION_COLUMNS:
        if col not in preds.columns:
            preds[col] = "NO_EVIDENCE"
    return preds[list(SUBMISSION_COLUMNS)]


# --------------------------------------------------------------------------- #
# The motivating diagnostic (§62): does the RESIDUAL concentrate on known      #
# positives MORE than the PRESENCE-only counterpart does?                      #
# --------------------------------------------------------------------------- #


def presence_vs_residual_concentration(
    caches: NegativeSpaceCacheLocations,
    *,
    k: float,
    high_pot_quantile: float = DEFAULT_HIGH_POT_QUANTILE,
    n_field_dyads: int = DEFAULT_FIELD_DYAD_SAMPLE,
) -> Dict[str, float]:
    """Compare separation power of the residual block vs its presence-only counterpart.

    Builds the dev block, merges the confirmed labels, and computes for BOTH the
    shrunk-residual features and the presence-only rates a single-feature-family
    ranking AUC (label==1 vs label==0, PU unknowns excluded). The §62 mechanism claim
    is that the RESIDUAL concentrates on known positives MORE than presence does; this
    returns the MEASURED AUCs so the claim is checked, not asserted (Rules 3, 12).

    Returns a dict of measured AUCs (residual vs presence) per interaction + combined.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    dev_pairs = _dev_pairs_ab(caches)
    pos_players = _dev_positive_players(caches)
    block = build_negative_space(
        dev_pairs, "development", caches,
        k=k, high_pot_quantile=high_pot_quantile,
        n_field_dyads=n_field_dyads, positive_players=pos_players,
    )
    dp = pd.read_parquet(caches.dev_pairs_v2)[["pair_id", "label"]]
    m = block.merge(dp, on="pair_id", how="left")
    conf = m[m["label"].isin([0, 1])].copy()
    y = (conf["label"] == 1).astype(int).to_numpy()

    def _family_auc(cols: List[str]) -> float:
        X = conf[cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64)
        if len(np.unique(y)) < 2:
            return float("nan")
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X, y)
        return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    res_cols = list(NEGATIVE_SPACE_RANK_COLUMNS)  # the pool-relative rank residuals
    pres_cols = list(PRESENCE_COLUMNS)
    out: Dict[str, float] = {}
    for name, r, p in (
        ("clash", "ns_clash_rank", "pres_clash_rate"),
        ("contest", "ns_contest_rank", "pres_contest_rate"),
        ("foldyield", "ns_foldyield_rank", "pres_foldyield_rate"),
    ):
        out[f"residual_auc_{name}"] = _family_auc([r])
        out[f"presence_auc_{name}"] = _family_auc([p])
    # combined residual = the SINGLE fitted composite; combined presence = the 3 raw rates.
    out["residual_auc_combined"] = _family_auc(list(NEGATIVE_SPACE_COLUMNS))
    out["presence_auc_combined"] = _family_auc(pres_cols)
    out["n_confirmed"] = float(len(conf))
    out["n_positive"] = float(int(y.sum()))
    return out
