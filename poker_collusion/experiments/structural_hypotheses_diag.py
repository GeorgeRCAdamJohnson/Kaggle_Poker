"""§58b structural-hypotheses diagnostic (MEASUREMENT-ONLY).

Pre-registered in RESEARCH_DOSSIER.md §58b. This module MEASURES which of three
structural hypotheses carry REAL label signal on a false-positive-defense split,
BEFORE any detector is built. It does NOT build a detector, train a submission
model, or touch submission.csv.

The three hypotheses (each with its required control):
  1. RINGS      -- multi-listed players / cliques; confirm-split lift of "ring
                   membership" over base rate, CONTROLLING for co-hand count.
  2. POSITION   -- beneficiary systematically acts LATER (more info) than partner;
                   confirm-split lift over the FIELD's own winner-acts-later rate.
  3. TABLE-LEVEL-- discover-only per-table positive rate predicts the CONFIRM label
                   (leakage-safe: table rates from DISCOVER pairs only).

Decision rule (§58b, FIXED -- Rule 12): a hypothesis CONFIRMS iff its confirm-split
lift is clearly above the base rate AND survives its control; else NULL.

Governed by the reverse-engineering-accountability contract: Rules 1, 2, 3, 8, 9,
12, 16.  All lifts measured on the CONFIRM split only.  Fixed seed, reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Pre-registered base rate the eval detector faces (§58b / §58): 0.214%.
# This is the EVAL-population base rate (~372 positives among ~112,540/... pairs);
# it is the bar the §58b decision rule names. We also report the WITHIN-CONFIRM
# base rate for context, but the pre-registered bar is 0.214%.
PREREG_BASE_RATE = 0.00214

# Fixed split seed (reported in §59).
SPLIT_SEED = 58


# --------------------------------------------------------------------------- #
# Data location
# --------------------------------------------------------------------------- #
def _find_data_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate data/poker.  Returns None if not found (test skips cleanly)."""
    if explicit:
        p = Path(explicit)
        return p if (p / "development_labels.csv").exists() else None
    here = Path(__file__).resolve()
    cands = [here.parents[2] / "data" / "poker", Path.cwd() / "data" / "poker"]
    for c in cands:
        if (c / "development_labels.csv").exists():
            return c
    return None


def _find_prepared_v2(data_root: Path) -> Optional[Path]:
    cand = (
        data_root.parents[1]
        / "outputs"
        / "poker_collusion"
        / "repro_nomannic"
        / "prepared_v2"
    )
    return cand if (cand / "dev_pairs.parquet").exists() else None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank AUC via Mann-Whitney U (ties handled). NaN if degenerate."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    s_sorted = score[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(score) + 1, dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[y_true == 1].sum()
    u = sum_ranks_pos - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _precision_at_flag(y_true: np.ndarray, flag: np.ndarray) -> tuple[float, int, int]:
    """Positive rate among flagged rows.  Returns (precision, n_flagged, n_pos_flagged)."""
    flag = np.asarray(flag).astype(bool)
    y_true = np.asarray(y_true).astype(int)
    nf = int(flag.sum())
    if nf == 0:
        return float("nan"), 0, 0
    npos = int(y_true[flag].sum())
    return npos / nf, nf, npos


# --------------------------------------------------------------------------- #
# Split (Rule 8: false-positive-defense)
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    discover_pairs: pd.Index
    confirm_pairs: pd.Index
    seed: int
    n_discover: int
    n_confirm: int
    n_confirm_pos: int
    n_confirm_neg: int
    player_disjoint: bool
    n_shared_players_dropped: int


def build_split(labels: pd.DataFrame, seed: int = SPLIT_SEED) -> Split:
    """Player-disjoint DISCOVER(60%)/CONFIRM(40%) split (Rule 8).

    We partition the *players* into two groups so that a ring discovered in
    DISCOVER cannot trivially reappear in CONFIRM.  A pair is assigned to a split
    only if BOTH its members fall in that split's player group; cross-group pairs
    are dropped from the confirm-split measurement (reported).  This is the
    strongest form of the false-positive defense for the ring test.
    """
    rng = np.random.default_rng(seed)
    players = pd.unique(
        pd.concat([labels["player_1"], labels["player_2"]], ignore_index=True)
    )
    players = np.sort(players)
    perm = rng.permutation(len(players))
    n_disc_players = int(round(0.6 * len(players)))
    disc_players = set(players[perm[:n_disc_players]])
    conf_players = set(players[perm[n_disc_players:]])

    def side(row):
        p1, p2 = row["player_1"], row["player_2"]
        if p1 in disc_players and p2 in disc_players:
            return "discover"
        if p1 in conf_players and p2 in conf_players:
            return "confirm"
        return "cross"

    sides = labels.apply(side, axis=1)
    disc = labels.index[sides == "discover"]
    conf = labels.index[sides == "confirm"]
    cross = int((sides == "cross").sum())
    conf_lab = labels.loc[conf, "label"]
    return Split(
        discover_pairs=disc,
        confirm_pairs=conf,
        seed=seed,
        n_discover=len(disc),
        n_confirm=len(conf),
        n_confirm_pos=int((conf_lab == 1).sum()),
        n_confirm_neg=int((conf_lab == 0).sum()),
        player_disjoint=True,
        n_shared_players_dropped=cross,
    )


# --------------------------------------------------------------------------- #
# Hypothesis result container
# --------------------------------------------------------------------------- #
@dataclass
class HypothesisResult:
    name: str
    confirm_auc: float
    lift_before_control: float  # confirm-split flag precision / base rate
    lift_after_control: float
    base_rate_prereg: float
    base_rate_confirm: float
    verdict: str  # "CONFIRM" or "NULL"
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Hypothesis 1: RINGS
# --------------------------------------------------------------------------- #
def measure_rings(labels: pd.DataFrame, pair_meta: pd.DataFrame, split: Split) -> HypothesisResult:
    """Ring membership predictiveness on the CONFIRM split, co-hand controlled.

    Ring membership is defined ONLY from DISCOVER positives (Rule 8 / leakage):
    a player is a "ring player" if they appear in >=2 DISCOVER-positive pairs.
    A confirm pair is "ring-embedded" if either member is a discover-ring player.
    Because the split is player-disjoint, discover-ring players never appear in
    confirm pairs -- so this cross-split test cannot be a within-clique label
    proxy (the §18 0.94 trap).  We ALSO report the within-confirm ring signal for
    context (that one IS vulnerable to the trap and is labeled as such).
    """
    lab = labels
    # ---- descriptive: multi-listed players over ALL positives ---------------
    pos_all = lab[lab.label == 1]
    plong = pd.concat([pos_all["player_1"], pos_all["player_2"]], ignore_index=True)
    counts = plong.value_counts()
    multi_listed = counts[counts >= 2]
    n_multi = int(len(multi_listed))
    n_pos_players = int(counts.shape[0])

    # table co-occurrence for multi-listed players' pairs (real ring vs scatter)
    meta_pos = pair_meta.loc[pos_all.index]
    ring_player_set = set(multi_listed.index)
    same_table_frac = float("nan")
    if n_multi > 0:
        recs = []
        for pl in ring_player_set:
            mask = (pos_all["player_1"] == pl) | (pos_all["player_2"] == pl)
            tbls = meta_pos.loc[pos_all.index[mask.values], "table_id"].dropna()
            if len(tbls) >= 2:
                top = tbls.value_counts(normalize=True).iloc[0]
                recs.append(top)
        if recs:
            same_table_frac = float(np.mean(recs))  # mean share of a player's pairs at their modal table

    # ---- CONFIRM-split predictive test (leakage-safe, player-disjoint) -------
    disc_pos = lab.loc[split.discover_pairs]
    disc_pos = disc_pos[disc_pos.label == 1]
    dpl = pd.concat([disc_pos["player_1"], disc_pos["player_2"]], ignore_index=True)
    dcounts = dpl.value_counts()
    disc_ring_players = set(dcounts[dcounts >= 2].index)

    conf = lab.loc[split.confirm_pairs].copy()
    conf_meta = pair_meta.loc[conf.index]
    ring_flag = (
        conf["player_1"].isin(disc_ring_players)
        | conf["player_2"].isin(disc_ring_players)
    ).to_numpy()
    y = conf["label"].to_numpy().astype(int)

    base_conf = float(y.mean()) if len(y) else float("nan")
    auc = _auc(y, ring_flag.astype(float))
    prec, nf, npf = _precision_at_flag(y, ring_flag)
    # Lift is measured against the WITHIN-CONFIRM base rate (the honest baseline for
    # this labeled split); the pre-registered 0.214% eval base rate is reported
    # separately in the result.  (Comparing a within-labeled-set precision to the
    # eval-population 0.214% inflates every lift ~100x and is not meaningful.)
    lift_before = (prec / base_conf) if (nf > 0 and not math.isnan(prec) and base_conf > 0) else float("nan")

    # ---- PAIR-DISJOINT diagnostic variant (shares NO pairs, may share players) --
    # §58b asks for player-disjoint "where feasible"; here player-disjointness
    # empties the cross-split ring flag (a discover-ring player cannot appear in a
    # player-disjoint confirm pair).  We therefore ALSO report a pair-disjoint
    # ring test as a DIAGNOSTIC, explicitly labeled as vulnerable to the §18/0.94
    # clique-proxy trap (a confirm pair can share a player with a discover positive).
    rng_pd = np.random.default_rng(split.seed + 7)
    idx_all = lab.index.to_numpy()
    perm = rng_pd.permutation(len(idx_all))
    n_disc = int(round(0.6 * len(idx_all)))
    pd_disc = set(idx_all[perm[:n_disc]])
    pd_conf_idx = idx_all[perm[n_disc:]]
    pd_disc_df = lab.loc[list(pd_disc)]
    pd_disc_pos = pd_disc_df[pd_disc_df.label == 1]
    pdpl = pd.concat([pd_disc_pos["player_1"], pd_disc_pos["player_2"]], ignore_index=True)
    pdc = pdpl.value_counts()
    pd_ring_players = set(pdc[pdc >= 2].index)
    pd_conf = lab.loc[pd_conf_idx]
    pd_flag = (
        pd_conf["player_1"].isin(pd_ring_players) | pd_conf["player_2"].isin(pd_ring_players)
    ).to_numpy()
    pd_y = pd_conf["label"].to_numpy().astype(int)
    pd_base = float(pd_y.mean()) if len(pd_y) else float("nan")
    pd_prec, pd_nf, pd_npf = _precision_at_flag(pd_y, pd_flag)
    pd_lift = (pd_prec / pd_base) if (pd_nf > 0 and not math.isnan(pd_prec) and pd_base > 0) else float("nan")
    pd_auc = _auc(pd_y, pd_flag.astype(float))

    # ---- co-hand control (Rule 16): matched non-ring pairs at similar co-hand -
    sh = conf_meta["shared_hands"].to_numpy(dtype=float)
    lift_after = float("nan")
    control_detail = {}
    if nf > 0:
        ring_idx = np.where(ring_flag)[0]
        nonring_idx = np.where(~ring_flag)[0]
        ring_sh = sh[ring_idx]
        # For each ring pair, match a non-ring pair at nearest shared_hands (with
        # replacement), then compare positive rates on the MATCHED non-ring set.
        rng = np.random.default_rng(split.seed + 1)
        if len(nonring_idx) > 0:
            nonring_sh = sh[nonring_idx]
            matched = []
            for v in ring_sh:
                d = np.abs(nonring_sh - v)
                mind = d.min()
                ties = nonring_idx[np.where(d == mind)[0]]
                matched.append(rng.choice(ties))
            matched = np.array(matched)
            ring_pos_rate = float(y[ring_idx].mean())
            matched_pos_rate = float(y[matched].mean())
            # control-adjusted lift: ring precision over the co-hand-matched
            # non-ring precision, re-expressed against the pre-registered base rate.
            # If ring signal is purely co-hand opportunity, matched_pos_rate ~= ring_pos_rate
            # and the *excess* over the matched control collapses toward the base line.
            lift_after = (
                ring_pos_rate / matched_pos_rate
                if matched_pos_rate > 0
                else float("inf") if ring_pos_rate > 0 else float("nan")
            )
            control_detail = {
                "ring_pos_rate": round(ring_pos_rate, 4),
                "cohand_matched_nonring_pos_rate": round(matched_pos_rate, 4),
                "ring_median_cohand": float(np.median(ring_sh)),
                "nonring_median_cohand": float(np.median(sh[nonring_idx])),
            }

    # Decision (§58b): CONFIRM iff lift clearly above base rate AND survives control.
    # "Survives control" = ring positive rate stays clearly above the co-hand-matched
    # non-ring positive rate (excess is not just opportunity).
    survives = (
        (not math.isnan(lift_before))
        and lift_before > 1.0
        and (not math.isnan(lift_after))
        and lift_after > 1.15  # ring excess >15% over co-hand-matched control
    )
    verdict = "CONFIRM" if survives else "NULL"

    return HypothesisResult(
        name="RINGS",
        confirm_auc=auc,
        lift_before_control=lift_before,
        lift_after_control=lift_after,
        base_rate_prereg=PREREG_BASE_RATE,
        base_rate_confirm=base_conf,
        verdict=verdict,
        detail={
            "n_multi_listed_players_all_pos": n_multi,
            "n_distinct_players_in_positives": n_pos_players,
            "multi_listed_mean_modal_table_share": same_table_frac,
            "n_discover_ring_players_playerdisjoint": len(disc_ring_players),
            "confirm_n": int(len(y)),
            "confirm_n_ring_flagged_playerdisjoint": nf,
            "confirm_n_pos_ring_flagged_playerdisjoint": npf,
            "confirm_ring_precision_playerdisjoint": None if math.isnan(prec) else round(prec, 4),
            "PLAYERDISJOINT_NOTE": "player-disjoint split empties the cross-split ring flag by construction",
            "PAIRDISJOINT_diag_auc": None if math.isnan(pd_auc) else round(pd_auc, 4),
            "PAIRDISJOINT_diag_lift_vs_base": None if math.isnan(pd_lift) else round(pd_lift, 3),
            "PAIRDISJOINT_diag_flagged": pd_nf,
            "PAIRDISJOINT_diag_pos_flagged": pd_npf,
            "PAIRDISJOINT_diag_precision": None if math.isnan(pd_prec) else round(pd_prec, 4),
            "PAIRDISJOINT_diag_base": None if math.isnan(pd_base) else round(pd_base, 4),
            "PAIRDISJOINT_NOTE": "shares players with discover -> vulnerable to §18/0.94 clique-proxy trap; DIAGNOSTIC only",
            **control_detail,
        },
    )


# --------------------------------------------------------------------------- #
# Hypothesis 2: POSITION
# --------------------------------------------------------------------------- #
def _post_flop_order(seat_no: np.ndarray, button_seat: np.ndarray, n_seats: int = 6) -> np.ndarray:
    """Post-flop acting order: small blind (button+1) acts first (0), button acts
    last (n_seats-1).  Higher value = later action = more information."""
    return (seat_no - button_seat - 1) % n_seats


def measure_position(
    labels: pd.DataFrame,
    split: Split,
    data_root: Path,
    prepared_v2: Optional[Path],
    max_hands_field: int = 400_000,
) -> HypothesisResult:
    """Beneficiary-acts-later asymmetry vs the FIELD's own winner-acts-later rate.

    For each labeled pair's shared hands, on hands where the two members have a
    strict net_chips ordering, we ask: did the BENEFICIARY (higher net_chips) act
    LATER (higher post-flop order) than the partner?  The pair statistic is the
    fraction of such hands.  The FIELD baseline is the same winner-acts-later rate
    computed over ALL co-seated player-pairs within hands (position confers a real
    edge, so the field rate is naturally > 0.5; that is the correct null -- H9).

    Confirm-split lift = confirm-positive mean pair-statistic / field baseline.
    """
    conf = labels.loc[split.confirm_pairs].copy()

    # Which pairs' shared hands do we need?  Use the dev_pair_hands cache if
    # present (fast); else fall back to reconstructing from seats.
    seats = pd.read_parquet(
        data_root / "seats.parquet",
        columns=["hand_id", "player_id", "seat_no", "net_chips", "won_share"],
    )
    hands = pd.read_parquet(
        data_root / "hands.parquet", columns=["hand_id", "button_seat"]
    )
    seats = seats.merge(hands, on="hand_id", how="inner")
    seats["order"] = _post_flop_order(
        seats["seat_no"].to_numpy(), seats["button_seat"].to_numpy()
    )

    # ---- FIELD baseline: winner-acts-later over co-seated pairs -------------
    # VECTORIZED: for each hand build all C(6,2) ordered comparisons via a self
    # merge on hand_id with i<j (by seat_no), then compare net_chips vs order.
    rng = np.random.default_rng(split.seed + 2)
    all_hands = seats["hand_id"].drop_duplicates().to_numpy()
    if len(all_hands) > max_hands_field:
        samp = rng.choice(all_hands, size=max_hands_field, replace=False)
        field_seats = seats[seats["hand_id"].isin(samp)]
    else:
        field_seats = seats
    fs = field_seats[["hand_id", "seat_no", "net_chips", "order"]].copy()
    fm = fs.merge(fs, on="hand_id", suffixes=("_a", "_b"))
    fm = fm[fm["seat_no_a"] < fm["seat_no_b"]]  # each unordered pair once
    nca = fm["net_chips_a"].to_numpy(float)
    ncb = fm["net_chips_b"].to_numpy(float)
    oa = fm["order_a"].to_numpy(float)
    ob = fm["order_b"].to_numpy(float)
    valid = (nca != ncb) & (oa != ob)
    nca, ncb, oa, ob = nca[valid], ncb[valid], oa[valid], ob[valid]
    winner_a = nca > ncb
    win_order = np.where(winner_a, oa, ob)
    los_order = np.where(winner_a, ob, oa)
    fw_total = int(valid.sum())
    fw_later = int((win_order > los_order).sum())
    field_rate = fw_later / fw_total if fw_total else float("nan")

    # ---- pair statistic for confirm pairs (VECTORIZED) ----------------------
    conf_players = set(conf["player_1"]).union(set(conf["player_2"]))
    sub = seats[seats["player_id"].isin(conf_players)][
        ["hand_id", "player_id", "net_chips", "order"]
    ].copy()
    # Map each confirm pair to an integer id and gather both members' rows via a
    # keyed merge.  Build long frame: for member role 1 and 2 separately.
    conf = conf.reset_index().rename(columns={"index": "_orig_idx"})
    conf["_pid"] = np.arange(len(conf))
    p1map = conf[["_pid", "player_1"]].rename(columns={"player_1": "player_id"})
    p2map = conf[["_pid", "player_2"]].rename(columns={"player_2": "player_id"})
    # rows for player_1 across its hands, tagged by every pair that uses it as p1
    r1 = sub.merge(p1map, on="player_id")[["_pid", "hand_id", "net_chips", "order"]]
    r2 = sub.merge(p2map, on="player_id")[["_pid", "hand_id", "net_chips", "order"]]
    m = r1.merge(r2, on=["_pid", "hand_id"], suffixes=("_1", "_2"))
    nc1 = m["net_chips_1"].to_numpy(float)
    nc2 = m["net_chips_2"].to_numpy(float)
    o1 = m["order_1"].to_numpy(float)
    o2 = m["order_2"].to_numpy(float)
    vld = (nc1 != nc2) & (o1 != o2)
    m = m.loc[vld].copy()
    nc1, nc2, o1, o2 = nc1[vld], nc2[vld], o1[vld], o2[vld]
    winner_is_1 = nc1 > nc2
    win_ord = np.where(winner_is_1, o1, o2)
    los_ord = np.where(winner_is_1, o2, o1)
    m["_later"] = (win_ord > los_ord).astype(float)
    agg = m.groupby("_pid")["_later"].agg(["mean", "count"])
    stat_map = agg["mean"].to_dict()
    n_map = agg["count"].to_dict()
    stats = [stat_map.get(pid, np.nan) for pid in conf["_pid"]]
    ns = [int(n_map.get(pid, 0)) for pid in conf["_pid"]]
    conf = conf.assign(pos_stat=stats, pos_n=ns)
    valid_mask = conf["pos_stat"].notna().to_numpy()
    y = conf["label"].to_numpy().astype(int)
    stat = conf["pos_stat"].to_numpy(dtype=float)

    auc = _auc(y[valid_mask], stat[valid_mask])
    pos_mean = float(np.nanmean(stat[(y == 1) & valid_mask])) if ((y == 1) & valid_mask).any() else float("nan")
    neg_mean = float(np.nanmean(stat[(y == 0) & valid_mask])) if ((y == 0) & valid_mask).any() else float("nan")

    # Lift over FIELD baseline (the pre-registered control for POSITION, H9).
    # lift_before = positive-pair asymmetry relative to the field's own act-later
    # rate (>1 means positives are MORE positionally asymmetric than the field).
    lift_before = (pos_mean / field_rate) if (field_rate and not math.isnan(field_rate)) else float("nan")
    # "After control" here = positive asymmetry relative to the NEGATIVE pairs'
    # asymmetry (the within-labeled control).  >1 means positives exceed negatives.
    lift_after = (pos_mean / neg_mean) if (not math.isnan(neg_mean) and neg_mean > 0) else float("nan")
    pos_excess_over_field = (pos_mean - field_rate) if (not math.isnan(pos_mean) and not math.isnan(field_rate)) else float("nan")

    # Decision (§58b): CONFIRM iff positive asymmetry clearly exceeds the field
    # baseline AND exceeds the negative pairs' asymmetry (survives the field/negative
    # control). AUC materially above 0.5 corroborates.
    survives = (
        (not math.isnan(pos_mean))
        and (not math.isnan(field_rate))
        and (pos_mean - field_rate) > 0.02
        and (not math.isnan(neg_mean))
        and (pos_mean - neg_mean) > 0.02
        and (not math.isnan(auc))
        and auc > 0.55
    )
    verdict = "CONFIRM" if survives else "NULL"

    return HypothesisResult(
        name="POSITION",
        confirm_auc=auc,
        lift_before_control=lift_before,
        lift_after_control=lift_after,
        base_rate_prereg=PREREG_BASE_RATE,
        base_rate_confirm=float(y.mean()) if len(y) else float("nan"),
        verdict=verdict,
        detail={
            "field_winner_acts_later_rate": None if math.isnan(field_rate) else round(field_rate, 4),
            "field_pairs_counted": fw_total,
            "confirm_pos_mean_stat": None if math.isnan(pos_mean) else round(pos_mean, 4),
            "confirm_neg_mean_stat": None if math.isnan(neg_mean) else round(neg_mean, 4),
            "confirm_pos_excess_over_field": None if math.isnan(pos_excess_over_field) else round(pos_excess_over_field, 4),
            "confirm_n_valid_pairs": int(valid_mask.sum()),
            "confirm_median_pair_hands": float(np.median([n for n in ns if n > 0])) if any(n > 0 for n in ns) else 0.0,
        },
    )


# --------------------------------------------------------------------------- #
# Hypothesis 3: TABLE-LEVEL
# --------------------------------------------------------------------------- #
def measure_table_level(
    labels: pd.DataFrame, pair_meta: pd.DataFrame, split: Split
) -> HypothesisResult:
    """Discover-only per-table positive rate predicts the CONFIRM label.

    Leakage-safe (Rule 9): table positive rates are computed on DISCOVER pairs
    ONLY, then applied to CONFIRM pairs at the same tables.  Confirm pairs at
    tables unseen in discover get the discover global positive rate (smoothed).
    We ALSO report the circular within-confirm version, explicitly labeled as
    leakage, to show the gap.
    """
    disc = labels.loc[split.discover_pairs].copy()
    conf = labels.loc[split.confirm_pairs].copy()
    disc_meta = pair_meta.loc[disc.index]
    conf_meta = pair_meta.loc[conf.index]
    disc = disc.assign(table_id=disc_meta["table_id"].values)
    conf = conf.assign(table_id=conf_meta["table_id"].values)

    # Discover-only table positive rate (empirical-Bayes shrunk toward discover
    # global rate by pseudo-count k -- Rule 16, small per-table n).
    global_disc_rate = float(disc["label"].mean())
    k = 5.0
    grp = disc.groupby("table_id")["label"].agg(["sum", "count"])
    grp["rate"] = (grp["sum"] + k * global_disc_rate) / (grp["count"] + k)
    table_rate = grp["rate"].to_dict()

    conf_score = conf["table_id"].map(lambda t: table_rate.get(t, global_disc_rate)).to_numpy(dtype=float)
    y = conf["label"].to_numpy().astype(int)
    auc = _auc(y, conf_score)

    # Lift: positive rate in the TOP tercile of discover-only table score vs base rate.
    base_conf = float(y.mean()) if len(y) else float("nan")
    thr = np.quantile(conf_score, 2.0 / 3.0)
    top_flag = conf_score >= thr
    prec, nf, npf = _precision_at_flag(y, top_flag)
    # Lift vs the within-confirm base rate (honest baseline); prereg 0.214% reported separately.
    lift_before = (prec / base_conf) if (nf > 0 and not math.isnan(prec) and base_conf > 0) else float("nan")

    # Leakage (circular) version, explicitly labeled -- for the gap report.
    conf_grp = conf.groupby("table_id")["label"].transform("mean").to_numpy(dtype=float)
    auc_leak = _auc(y, conf_grp)

    # "After control" for TABLE-LEVEL = the discover-only (leakage-safe) result
    # itself; the control is that it must NOT rely on confirm labels.  We express
    # lift_after as the discover-only top-tercile lift (same as before here because
    # the whole test is already leakage-safe); the leakage AUC is reported as the
    # contrast.  If discover-only AUC ~ 0.5 while leakage AUC is high, that is the
    # §58b leakage-NULL signature.
    lift_after = lift_before

    survives = (
        (not math.isnan(auc))
        and auc > 0.55
        and (not math.isnan(lift_before))
        and lift_before > 1.0
        and npf > 0
    )
    verdict = "CONFIRM" if survives else "NULL"

    return HypothesisResult(
        name="TABLE-LEVEL",
        confirm_auc=auc,
        lift_before_control=lift_before,
        lift_after_control=lift_after,
        base_rate_prereg=PREREG_BASE_RATE,
        base_rate_confirm=float(y.mean()) if len(y) else float("nan"),
        verdict=verdict,
        detail={
            "discover_global_pos_rate": round(global_disc_rate, 4),
            "n_discover_tables": int(len(grp)),
            "n_confirm_tables_seen_in_discover": int(conf["table_id"].isin(set(grp.index)).sum()),
            "confirm_discover_only_auc": None if math.isnan(auc) else round(auc, 4),
            "confirm_LEAKAGE_within_confirm_auc": None if math.isnan(auc_leak) else round(auc_leak, 4),
            "top_tercile_n_flagged": nf,
            "top_tercile_n_pos": npf,
            "top_tercile_precision": None if math.isnan(prec) else round(prec, 4),
        },
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class DiagnosticReport:
    seed: int
    split: Split
    results: list[HypothesisResult]

    def confirmers_ranked(self) -> list[HypothesisResult]:
        c = [r for r in self.results if r.verdict == "CONFIRM"]
        return sorted(
            c,
            key=lambda r: (r.lift_after_control if not math.isnan(r.lift_after_control) else -1),
            reverse=True,
        )


def run_diagnostic(
    data_root_override: Optional[str] = None, seed: int = SPLIT_SEED
) -> DiagnosticReport:
    data_root = _find_data_root(data_root_override)
    if data_root is None:
        raise FileNotFoundError("data/poker not found")
    prepared_v2 = _find_prepared_v2(data_root)

    labels = pd.read_csv(data_root / "development_labels.csv")

    # pair_meta: shared_hands + table_id per labeled pair (from prepared_v2 cache if
    # present; else reconstruct from seats/hands).
    if prepared_v2 is not None:
        dp = pd.read_parquet(prepared_v2 / "dev_pairs.parquet")
        dp = dp[dp["is_labeled"]][["pair_id", "shared_hands", "table_id"]]
        pair_meta = labels.merge(dp, on="pair_id", how="left").set_index(labels.index)
    else:
        pair_meta = labels.copy()
        pair_meta["shared_hands"] = np.nan
        pair_meta["table_id"] = np.nan
    pair_meta = pair_meta[["shared_hands", "table_id"]]

    split = build_split(labels, seed=seed)

    r1 = measure_rings(labels, pair_meta, split)
    r2 = measure_position(labels, split, data_root, prepared_v2)
    r3 = measure_table_level(labels, pair_meta, split)

    return DiagnosticReport(seed=seed, split=split, results=[r1, r2, r3])


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x)):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def print_report(rep: DiagnosticReport) -> None:
    s = rep.split
    print("=" * 78)
    print("§58b STRUCTURAL-HYPOTHESES DIAGNOSTIC (MEASUREMENT-ONLY)")
    print("=" * 78)
    print(f"split seed={rep.seed} (player-disjoint)  base-rate(prereg)={PREREG_BASE_RATE:.5f}")
    print(
        f"DISCOVER n={s.n_discover}  CONFIRM n={s.n_confirm} "
        f"(pos={s.n_confirm_pos}, neg={s.n_confirm_neg})  "
        f"cross-split pairs dropped={s.n_shared_players_dropped}"
    )
    print(f"CONFIRM within-split base rate = {_fmt(rep.results[0].base_rate_confirm)}")
    print("-" * 78)
    for r in rep.results:
        print(f"[{r.name}]  VERDICT = {r.verdict}")
        print(f"    confirm-split AUC        = {_fmt(r.confirm_auc)}")
        print(f"    lift (before control)    = {_fmt(r.lift_before_control)} x CONFIRM base rate ({_fmt(r.base_rate_confirm)})")
        print(f"    lift (after control)     = {_fmt(r.lift_after_control)}")
        for k, v in r.detail.items():
            print(f"      - {k}: {_fmt(v) if isinstance(v,(int,float)) else v}")
        print("-" * 78)
    conf = rep.confirmers_ranked()
    if conf:
        print("CONFIRMERS ranked by after-control lift (build largest first):")
        for i, r in enumerate(conf, 1):
            print(f"    {i}. {r.name}  (after-control lift {_fmt(r.lift_after_control)})")
    else:
        print("ALL THREE HYPOTHESES ARE NULL on the confirm split (Rule 3 honest null).")
    print("=" * 78)


if __name__ == "__main__":
    print_report(run_diagnostic())
