"""Clean graph RETEST for the co-play topology lever (dossier §93). BUILD + MEASURE.

Governed by the workspace ``reverse-engineering-accountability`` contract. This module
NEVER submits to Kaggle and NEVER touches ``submission.csv`` / ``submission_best_*.csv``
in the repo root. Everything it writes lands under
``outputs/poker_collusion/graph_retest/``.

Why this module exists (§89-§93).
---------------------------------
The graph lever (cand_24) scored 0.340 on the LB — below the floor — and was filed as a
possible dead end (§89). §91/§92 traced the failure to a likely dev/eval PARITY BREAK:
the eval graph was built by a SEPARATE, unpersisted code path from the dev graph, so any
per-phase normalization / neighbor bookkeeping silently diverged dev->eval and the eval
columns did not mean the same thing as the dev columns the model fit on. That is a
feature-transport BUG (contract Rule 11: a gate/LB failure on a novel idea is a debugging
task, not a verdict on the idea), not proof the topology is signal-free.

This retest resolves the ambiguity the RIGHT way:

1. ONE shared builder :func:`build_graph_block` is called IDENTICALLY for
   ``phase="development"`` and ``phase="evaluation"``. There is exactly one code path;
   dev and eval can no longer diverge. All within-phase normalization is computed
   WITHIN each phase's own tables (never carrying a dev constant to eval — the §92 bug).
   The features are DEGREE-INVARIANT (no raw degree — the §19 PU-degree artifact):
   structural ratios + a LABEL-FREE neighbor statistic (never seeded by any dev OOF
   risk, which has no honest eval analogue — §92).

2. EVERY graph column is drift-gated with the EXISTING adversarial gate
   (:func:`own_submission_recipes.adversarial_drift_auc_matrix`, threshold 0.65 — Rule 5,
   Rule 17). Columns that separate dev from eval too easily are DISQUALIFIED.
   * GATE 1: if NO column passes, that is the honest DEAD-END proof (intrinsic drift) —
     report and STOP (Rule 3: an honest null is a result).

3. Only the drift-CLEAN columns are grafted onto the V30 base (lamhuy_topological,
   LB 0.678). V30's 274-feature matrix is rebuilt from its PERSISTED prepared cache with
   the notebook's OWN ``aggregate_pair_features`` logic (Rule 6 — reuse, do not rebuild
   the foundation), the drift-clean graph columns are appended by ``pair_id``, and V30's
   EXACT triple-GBDT risk head (XGB 0.40 + LGBM 0.35 + CatBoost 0.25) is retrained under
   V30's 5-fold StratifiedGroupKFold-by-table_id + PU-stress weighting to produce OOF.
   Confirmed-pair PairAP is scored the §69/§73 way (the REUSED
   :class:`anchor_repro.scorer.CanonicalScorer`) for BOTH V30-alone (graph cols dropped)
   and V30+graph.
   * GATE 2 bar (PRE-REGISTERED, Rule 2): augmented confirmed PairAP >= V30-alone
     confirmed PairAP + 0.003.

4. Three-way verdict:
   * GATE 1 fails (no clean cols)         -> "DEAD-END PROVEN (intrinsic drift)".
   * clean cols but GATE 2 delta < +0.003 -> "TRANSPORT-CLEAN BUT NULL ON V30".
   * clean cols and GATE 2 delta >= +0.003 -> "LIVE ORTHOGONAL LEVER" (+ eval candidate).

GPU (Rule / §79). The drift gate stays CPU (it is tiny: 5-fold, 300 trees, few columns).
The V30 retrain uses XGBoost ``device="cuda"`` on the RTX 5070 Ti; LightGBM/CatBoost stay
CPU. If ``device="cuda"`` errors we fall back to CPU ``tree_method="hist"`` and record it
(never fail the whole run over GPU).

Run:  ``python -m anchor_repro.graph_retest``  (from ``poker/``).
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from poker_collusion.config import PipelineConfig

from anchor_repro.competitor_recipe import _emit_dev_predictions
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.own_submission_recipes import (
    DRIFT_AUC_THRESHOLD,
    adversarial_drift_auc_matrix,
)

__all__ = [
    "GraphRetestPaths",
    "default_paths",
    "build_graph_block",
    "GRAPH_COLUMNS",
    "main",
]

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

#: Pre-registered GATE 2 bar (Rule 2, BINDING, FIXED before any number was produced).
GATE2_BAR: float = 0.003

#: XGB triple-GBDT ensemble weights — VERBATIM from the lamhuy notebook (Rule 6).
_W_XGB, _W_LGB, _W_CB = 0.40, 0.35, 0.25

#: V30's fixed seed (notebook ``SEED = 42``).
SEED: int = 42

#: The degree-invariant, label-free graph columns produced by the shared builder.
#: NO raw degree (the §19 PU-degree artifact); no dev-OOF-seeded suspicion (§92).
GRAPH_COLUMNS: List[str] = [
    "g_triangle_strength",
    "g_clustering_coefficient",
    "g_frac_hi_neighbors",
    "g_mean_neighbor_cointensity",
    "g_neighbor_transfer_imbalance",
]

#: The official eval-submission contract (8 columns, exact order).
_SUBMISSION_COLUMNS: List[str] = [
    "pair_id",
    "risk_score",
    "predicted_behavior",
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
]
_EVIDENCE_COLS: List[str] = [f"evidence_hand_{i}" for i in range(1, 6)]
_EXPECTED_EVAL_ROWS: int = 112_540


# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphRetestPaths:
    """Resolved on-disk paths for the graph retest (all outputs stay under out_dir)."""

    poker_root: Path
    data_dir: Path
    prepared_dir: Path  # V30's persisted prepared cache (prepared_v13)
    v30_submission: Path  # V30 eval submission (LB 0.678) — behavior/evidence source
    out_dir: Path

    # --- persisted V30 prepared-cache artifacts (reused verbatim, Rule 6) ---
    @property
    def dev_pairs(self) -> Path:
        return self.prepared_dir / "dev_pairs.parquet"

    @property
    def eval_pairs(self) -> Path:
        return self.prepared_dir / "eval_pairs.parquet"

    @property
    def dev_hand_features(self) -> Path:
        return self.prepared_dir / "dev_hand_features.parquet"

    @property
    def eval_hand_features(self) -> Path:
        return self.prepared_dir / "eval_hand_features.parquet"

    @property
    def player_baselines(self) -> Path:
        return self.prepared_dir / "player_baselines.parquet"

    # --- raw data (read-only) ---
    @property
    def players(self) -> Path:
        return self.data_dir / "players.parquet"

    @property
    def hands(self) -> Path:
        return self.data_dir / "hands.parquet"

    @property
    def seats(self) -> Path:
        return self.data_dir / "seats.parquet"

    @property
    def dev_labels_csv(self) -> Path:
        return self.data_dir / "development_labels.csv"

    @property
    def dev_evidence_csv(self) -> Path:
        return self.data_dir / "development_evidence.csv"

    @property
    def eval_pairs_csv(self) -> Path:
        return self.data_dir / "evaluation_pairs.csv"

    def missing(self) -> List[Path]:
        needed = [
            self.dev_pairs,
            self.eval_pairs,
            self.dev_hand_features,
            self.eval_hand_features,
            self.player_baselines,
            self.players,
            self.hands,
            self.seats,
            self.dev_labels_csv,
            self.eval_pairs_csv,
            self.v30_submission,
        ]
        return [p for p in needed if not Path(p).is_file()]


def default_paths(poker_root: Path) -> GraphRetestPaths:
    """Resolve the standard graph-retest paths under a ``poker/`` tree."""
    root = Path(poker_root)
    repro = root / "outputs" / "poker_collusion" / "repro_lamhuy"
    return GraphRetestPaths(
        poker_root=root,
        data_dir=(root / "data" / "poker"),
        prepared_dir=repro / "prepared_v13",
        v30_submission=repro / "submission.csv",
        out_dir=root / "outputs" / "poker_collusion" / "graph_retest",
    )


# --------------------------------------------------------------------------- #
# STEP 1 — the ONE shared graph builder (the parity fix, §92)                   #
# --------------------------------------------------------------------------- #


def _phase_coplay_edges(
    seats_lf: pl.LazyFrame, hands_lf: pl.LazyFrame, phase: str
) -> pl.DataFrame:
    """All co-seated player pairs (undirected) within one phase, with co-play weight.

    Nodes = players; an edge (u<v) exists if u and v sat at the SAME hand within the
    phase's tables. The edge weight is the number of shared hands (co-play intensity).
    A label-free per-edge ``mean_transfer_imbalance`` proxy is also aggregated so the
    neighbor statistic can be computed identically both phases (no dev OOF risk, §92).

    Returns a DataFrame with columns ``u, v, cohands, edge_transfer_imbalance`` where
    ``u < v`` (canonical undirected orientation).
    """
    # hand_id -> phase (restrict to this phase's hands only, so dev and eval graphs are
    # each built purely from their own phase — no cross-phase leakage).
    phase_hands = hands_lf.filter(pl.col("phase") == phase).select("hand_id")

    seats_phase = seats_lf.select(["hand_id", "player_id", "net_chips"]).join(
        phase_hands, on="hand_id", how="inner"
    )

    # Self-join on hand_id to enumerate all co-seated player pairs per hand, keep u<v.
    a = seats_phase.rename({"player_id": "u", "net_chips": "u_net"})
    b = seats_phase.rename({"player_id": "v", "net_chips": "v_net"})
    pairs = (
        a.join(b, on="hand_id", how="inner")
        .filter(pl.col("u") < pl.col("v"))
        # per co-seated hand, a label-free "who-moved-chips-toward-whom" magnitude:
        # |net_u - net_v| normalized by the total chips in motion (bounded [0,1]).
        # This is a structural transfer-imbalance proxy computed IDENTICALLY both
        # phases; it never touches labels or any dev OOF risk (§92, Rule).
        .with_columns(
            (
                (pl.col("u_net") - pl.col("v_net")).abs()
                / (pl.col("u_net").abs() + pl.col("v_net").abs() + 1.0)
            ).alias("_ti")
        )
        .group_by(["u", "v"])
        .agg(
            pl.len().alias("cohands"),
            pl.col("_ti").mean().alias("edge_transfer_imbalance"),
        )
    )
    return pairs.collect(engine="streaming")


def build_graph_block(
    phase: str,
    pairs_df: pl.DataFrame,
    *,
    seats_lf: pl.LazyFrame,
    hands_lf: pl.LazyFrame,
) -> pd.DataFrame:
    """Build the degree-invariant co-play graph block for the target pairs of ONE phase.

    THIS IS THE SINGLE SHARED CODE PATH (the §92 parity fix): it is called identically
    for ``phase="development"`` and ``phase="evaluation"``. Every within-phase
    normalization (the hi-intensity percentile) is computed WITHIN this phase's own
    edges — never a dev constant carried to eval.

    Per target pair (a, b) we compute DEGREE-INVARIANT structural features from the
    phase's co-play graph (NO raw degree — the §19 PU artifact):

    * ``g_triangle_strength``  — Jaccard overlap of a's and b's neighbor sets
      (shared-neighbor cohesion / how much the two players' social circles coincide).
    * ``g_clustering_coefficient`` — fraction of the pair's shared neighbors that are
      themselves connected to each other (local density of the pair's common circle).
    * ``g_frac_hi_neighbors`` — fraction of shared neighbors whose co-play intensity
      with the pair exceeds the phase's own hi-intensity percentile (a ratio, not a
      count, so degree cancels).
    * ``g_mean_neighbor_cointensity`` — mean co-play intensity (edge weight) between the
      pair and its shared neighbors, log-normalized (LABEL-FREE neighbor statistic).
    * ``g_neighbor_transfer_imbalance`` — mean of the label-free per-edge
      transfer-imbalance proxy over the pair's shared-neighbor edges (identical both
      phases; the "expected-vs-observed" negative-space flavour of §19, never seeded by
      any dev OOF risk).

    Returns a pandas DataFrame ``[pair_id] + GRAPH_COLUMNS`` (one row per target pair;
    pairs with no shared neighbors get 0.0 fills so the block is dense).
    """
    edges = _phase_coplay_edges(seats_lf, hands_lf, phase)

    # Adjacency as python dicts (players are integers; ~tens of thousands of nodes).
    # neigh[p] -> {q: (cohands, edge_ti)}
    # Player ids are opaque strings — keep them as strings (hashable dict keys).
    e_u = edges["u"].to_list()
    e_v = edges["v"].to_list()
    e_c = edges["cohands"].to_numpy().astype(np.float64)
    e_ti = edges["edge_transfer_imbalance"].to_numpy().astype(np.float64)

    # Phase-own hi-intensity percentile (the within-phase normalization constant).
    hi_thresh = float(np.quantile(e_c, 0.90)) if len(e_c) else 0.0

    neigh: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for u, v, c, ti in zip(e_u, e_v, e_c, e_ti):
        neigh.setdefault(u, {})[v] = (c, ti)
        neigh.setdefault(v, {})[u] = (c, ti)

    pids = pairs_df["pair_id"].to_list()
    p1 = pairs_df["player_1"].to_list()
    p2 = pairs_df["player_2"].to_list()

    n = len(pids)
    tri = np.zeros(n, dtype=np.float64)
    clu = np.zeros(n, dtype=np.float64)
    frac_hi = np.zeros(n, dtype=np.float64)
    coint = np.zeros(n, dtype=np.float64)
    nti = np.zeros(n, dtype=np.float64)

    for i in range(n):
        a, b = p1[i], p2[i]
        na = neigh.get(a, {})
        nb = neigh.get(b, {})
        set_a = set(na.keys())
        set_b = set(nb.keys())
        # Exclude the pair members themselves from each other's neighbor set.
        set_a.discard(b)
        set_b.discard(a)

        union = set_a | set_b
        inter = set_a & set_b  # shared neighbors of BOTH a and b
        # Jaccard shared-neighbor overlap (degree-invariant cohesion).
        tri[i] = (len(inter) / len(union)) if union else 0.0

        if inter:
            # Co-play intensity between the pair and each shared neighbor: the sum of
            # cohands from a->w and b->w, log-normalized (bounded, degree-cancelling).
            cointens = []
            tis = []
            hi_count = 0
            shared = list(inter)
            for w in shared:
                ca, tia = na[w]
                cb, tib = nb[w]
                cw = ca + cb
                cointens.append(np.log1p(cw))
                tis.append(0.5 * (tia + tib))
                if cw >= (2.0 * hi_thresh):
                    hi_count += 1
            coint[i] = float(np.mean(cointens))
            nti[i] = float(np.mean(tis))
            frac_hi[i] = hi_count / len(shared)

            # Local clustering: fraction of shared-neighbor PAIRS that are themselves
            # connected (density of the pair's common circle). Cap the shared-neighbor
            # count considered so an occasional hub pair does not blow up the O(k^2).
            if len(shared) >= 2:
                cap = shared[:64]
                links = 0
                total = 0
                for x_i in range(len(cap)):
                    wx = cap[x_i]
                    nwx = neigh.get(wx, {})
                    for y_i in range(x_i + 1, len(cap)):
                        total += 1
                        if cap[y_i] in nwx:
                            links += 1
                clu[i] = (links / total) if total else 0.0

    out = pd.DataFrame(
        {
            "pair_id": [str(p) for p in pids],
            "g_triangle_strength": tri,
            "g_clustering_coefficient": clu,
            "g_frac_hi_neighbors": frac_hi,
            "g_mean_neighbor_cointensity": coint,
            "g_neighbor_transfer_imbalance": nti,
        }
    )
    return out[["pair_id"] + GRAPH_COLUMNS]


# --------------------------------------------------------------------------- #
# STEP 3 helpers — rebuild V30's 274-feature matrix (Rule 6: reuse notebook)    #
# --------------------------------------------------------------------------- #


def _v30_aggregate_pair_features(
    hand_features_path: Path,
    pairs: pl.DataFrame,
    *,
    players_lf: pl.LazyFrame,
    baseline_path: Path,
) -> pl.DataFrame:
    """VERBATIM port of the lamhuy notebook ``aggregate_pair_features`` (Rule 6).

    Builds V30's pair-level feature matrix from the PERSISTED hand features + player
    baselines + raw players, using the notebook's OWN aggregation logic (standard
    moments + tail bursts + the four invariant manifolds + player-meta contrasts).
    This is NOT a reinvention — it is the notebook's exact expression list, so the
    grafted-onto base is byte-faithful to V30.
    """
    lf = pl.scan_parquet(hand_features_path)
    hand_schema = lf.collect_schema()
    skip_cols = {
        "pair_id", "hand_id", "player_1", "player_2", "p_low", "p_high",
        "table_id", "evidence_rank", "is_evidence", "hand_order", "pair_hand_no",
    }
    HAND_FEATURES = [
        c for c, dtype in hand_schema.items()
        if dtype.is_numeric() and c not in skip_cols and "label" not in c and "behavior_id" not in c
    ]
    SCORE_FEATURES = [
        c for c in HAND_FEATURES
        if c.endswith("_signal") or c.endswith("_score") or c.endswith("_priority")
    ]
    tail_tokens = (
        "signal", "priority", "score", "pot_bb", "contribution", "net_gap",
        "amount", "raise", "heads_up", "partner", "pressure", "to_max",
    )
    TAIL_FEATURES = list(
        dict.fromkeys(SCORE_FEATURES + [c for c in HAND_FEATURES if any(x in c for x in tail_tokens)])
    )[:40]
    score_q95 = (
        lf.select([pl.col(c).quantile(0.95).alias(c) for c in SCORE_FEATURES]).collect().row(0, named=True)
        if SCORE_FEATURES else {}
    )

    agg = [pl.len().alias("shared_hands_calc")]
    agg += [pl.col(c).mean().alias(f"{c}_mean") for c in HAND_FEATURES]
    agg += [pl.col(c).max().alias(f"{c}_max") for c in HAND_FEATURES]
    agg += [pl.col(c).quantile(0.95, interpolation="nearest").alias(f"{c}_p95") for c in HAND_FEATURES]
    agg += [pl.col(c).top_k(5).mean().alias(f"{c}_top5") for c in TAIL_FEATURES]
    agg += [(pl.col(c) >= score_q95[c]).mean().alias(f"{c}_rate95") for c in SCORE_FEATURES]
    agg += [
        pl.col("transfer_1_to_2_bb").sum().alias("_cum_t12"),
        pl.col("transfer_2_to_1_bb").sum().alias("_cum_t21"),
        pl.col("net_gap_bb").sum().alias("_cum_net_gap"),
        pl.col("pot_bb").sum().alias("_cum_pot"),
        pl.col("true_hu_checks").sum().alias("_cum_hu_checks"),
        pl.col("true_hu_actions").sum().alias("_cum_hu_actions"),
        pl.col("both_showdown").sum().alias("_cum_showdown"),
        pl.col("outsider_folds_to_pair").sum().alias("_cum_outsider_folds"),
        pl.col("partner_folds_to_pair").sum().alias("_cum_partner_folds"),
        pl.col("pair_raises").sum().alias("_cum_raises"),
        pl.col("pair_contribution_bb").sum().alias("_cum_pair_contrib"),
        pl.col("contrib_gap_bb").sum().alias("_cum_contrib_gap"),
        pl.col("hu_flop_folds").sum().alias("_cum_flop_folds"),
        pl.col("hu_late_checks").sum().alias("_cum_late_checks"),
        pl.col("is_pure_dump").sum().alias("_cum_pure_dump"),
        pl.col("is_soft_checkdown").sum().alias("_cum_soft_checkdown"),
        pl.col("is_squeeze_isolation").sum().alias("_cum_squeeze_isolation"),
    ]

    pair_stats = (
        lf.group_by("pair_id").agg(agg)
        .with_columns(
            ((pl.col("_cum_t12") - pl.col("_cum_t21")).abs() / (pl.col("_cum_t12") + pl.col("_cum_t21") + 1e-3)).alias("transfer_imbalance"),
            ((pl.col("_cum_t12") + pl.col("_cum_t21")) / (pl.col("shared_hands_calc") + 1e-3)).alias("transfer_per_hand"),
            (pl.col("_cum_net_gap") / (pl.col("_cum_pot") + 1e-3)).alias("net_flow_ratio"),
            (pl.col("_cum_hu_checks") / (pl.col("_cum_hu_actions") + 1e-3)).alias("showdown_check_ratio"),
            (pl.col("_cum_showdown") / (pl.col("shared_hands_calc") + 1e-3)).alias("both_showdown_rate"),
            (pl.col("_cum_outsider_folds") / (pl.col("shared_hands_calc") + 1e-3)).alias("outsider_fold_rate"),
            ((pl.col("_cum_outsider_folds") - pl.col("_cum_partner_folds")) / (pl.col("_cum_outsider_folds") + pl.col("_cum_partner_folds") + 1e-3)).alias("isolation_imbalance"),
            (pl.col("_cum_contrib_gap") / (pl.col("_cum_pair_contrib") + 1e-3)).alias("contrib_asymmetry_ratio"),
            (pl.col("_cum_flop_folds") / (pl.col("shared_hands_calc") + 1e-3)).alias("donor_flop_fold_rate"),
            (pl.col("_cum_late_checks") / (pl.col("shared_hands_calc") + 1e-3)).alias("late_checkdown_rate"),
            (pl.col("_cum_pure_dump") / (pl.col("shared_hands_calc") + 1e-3)).alias("pure_dump_rate"),
            (pl.col("_cum_soft_checkdown") / (pl.col("shared_hands_calc") + 1e-3)).alias("soft_checkdown_rate"),
            (pl.col("_cum_squeeze_isolation") / (pl.col("shared_hands_calc") + 1e-3)).alias("squeeze_isolation_rate"),
            (pl.col("transfer_any_bb_max") / (pl.col("shared_hands_calc").log1p() + 1e-3)).alias("transfer_max_log_norm"),
            (pl.col("net_gap_bb_max") / (pl.col("shared_hands_calc").log1p() + 1e-3)).alias("net_gap_max_log_norm"),
            (pl.col("pot_bb_max") / (pl.col("shared_hands_calc").log1p() + 1e-3)).alias("pot_max_log_norm"),
            (pl.col("contrib_gap_bb_max") / (pl.col("shared_hands_calc").log1p() + 1e-3)).alias("contrib_gap_max_log_norm"),
            (pl.col("pair_contribution_bb_max") / (pl.col("shared_hands_calc").log1p() + 1e-3)).alias("pair_contrib_max_log_norm"),
        )
        .drop([
            "_cum_t12", "_cum_t21", "_cum_net_gap", "_cum_pot", "_cum_hu_checks",
            "_cum_hu_actions", "_cum_showdown", "_cum_outsider_folds", "_cum_partner_folds",
            "_cum_raises", "_cum_pair_contrib", "_cum_contrib_gap", "_cum_flop_folds",
            "_cum_late_checks", "_cum_pure_dump", "_cum_soft_checkdown", "_cum_squeeze_isolation",
        ])
    )

    meta_cols = ["player_id", "account_age_days", "experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]
    meta = players_lf.select(meta_cols)
    m1 = meta.rename({c: ("player_1" if c == "player_id" else f"{c}_1") for c in meta_cols})
    m2 = meta.rename({c: ("player_2" if c == "player_id" else f"{c}_2") for c in meta_cols})

    base_cols = ["player_id", "hands", "base_aggression", "base_showdown", "base_calls", "base_checks"]
    player_base = (
        pl.scan_parquet(baseline_path).select(base_cols).group_by("player_id").agg([
            pl.col("hands").sum().alias("total_hands"),
            pl.col("base_aggression").mean().alias("base_aggression"),
            pl.col("base_showdown").mean().alias("base_showdown"),
            pl.col("base_calls").mean().alias("base_calls"),
            pl.col("base_checks").mean().alias("base_checks"),
        ])
    )
    pb1 = player_base.rename({c: ("player_1" if c == "player_id" else f"{c}_1") for c in ["player_id", "total_hands", "base_aggression", "base_showdown", "base_calls", "base_checks"]})
    pb2 = player_base.rename({c: ("player_2" if c == "player_id" else f"{c}_2") for c in ["player_id", "total_hands", "base_aggression", "base_showdown", "base_calls", "base_checks"]})

    pair_meta = (
        pairs.lazy().select(["pair_id", "player_1", "player_2"])
        .join(m1, on="player_1", how="left")
        .join(m2, on="player_2", how="left")
        .join(pb1, on="player_1", how="left")
        .join(pb2, on="player_2", how="left")
        .select(
            "pair_id",
            (pl.col("account_age_days_1").cast(pl.Float32) - pl.col("account_age_days_2").cast(pl.Float32)).abs().alias("account_age_gap"),
            (pl.col("experience_hands_bucket_1") == pl.col("experience_hands_bucket_2")).cast(pl.Int8).alias("same_experience"),
            (pl.col("preferred_stake_1") == pl.col("preferred_stake_2")).cast(pl.Int8).alias("same_stake"),
            (pl.col("region_bucket_1") == pl.col("region_bucket_2")).cast(pl.Int8).alias("same_region"),
            (pl.col("client_family_1") == pl.col("client_family_2")).cast(pl.Int8).alias("same_client"),
            (pl.col("total_hands_1").fill_null(0)).alias("player_hands_1"),
            (pl.col("total_hands_2").fill_null(0)).alias("player_hands_2"),
            (pl.col("base_aggression_1").fill_null(0.0)).alias("base_aggression_1"),
            (pl.col("base_aggression_2").fill_null(0.0)).alias("base_aggression_2"),
            (pl.col("base_showdown_1").fill_null(0.0)).alias("base_showdown_1"),
            (pl.col("base_showdown_2").fill_null(0.0)).alias("base_showdown_2"),
        )
    )

    base_cols_pairs = [c for c in ["pair_id", "player_1", "player_2", "shared_hands"] if c in pairs.columns]
    return (
        pairs.lazy().select(base_cols_pairs)
        .join(pair_stats, on="pair_id", how="left")
        .join(pair_meta, on="pair_id", how="left")
        .with_columns(
            (pl.col("shared_hands_calc") / (pl.col("player_hands_1") + 1e-3)).clip(0, 1).alias("overlap_share_1"),
            (pl.col("shared_hands_calc") / (pl.col("player_hands_2") + 1e-3)).clip(0, 1).alias("overlap_share_2"),
            (pl.col("both_showdown_rate") / (pl.col("base_showdown_1") * pl.col("base_showdown_2") + 1e-4)).alias("showdown_excess_ratio"),
            (pl.col("transfer_per_hand") / ((pl.col("base_aggression_1") + pl.col("base_aggression_2")) / 2.0 + 1e-3)).alias("transfer_vs_aggression"),
            (pl.col("shared_hands_calc") / (pl.col("player_hands_1") * pl.col("player_hands_2") + 1e-3).sqrt()).clip(0, 1).alias("co_presence_ratio"),
            (pl.col("shared_hands_calc") / ((pl.col("player_hands_1") + pl.col("player_hands_2")) / 2.0 + 1e-3)).clip(0, 1).alias("geometric_overlap"),
        )
        .with_columns(
            pl.max_horizontal("overlap_share_1", "overlap_share_2").alias("max_overlap_share"),
            pl.min_horizontal("overlap_share_1", "overlap_share_2").alias("min_overlap_share"),
        )
        .drop(["base_aggression_1", "base_aggression_2", "base_showdown_1", "base_showdown_2"])
        .collect(engine="streaming")
    )


# --------------------------------------------------------------------------- #
# STEP 3 — the V30 triple-GBDT retrain (VERBATIM params, Rule 6)                #
# --------------------------------------------------------------------------- #


def _xgb_risk_params(use_cuda: bool) -> dict:
    """V30's XGB risk params (verbatim), with device switched to cuda when available."""
    params = dict(
        n_estimators=1600, learning_rate=0.022, max_depth=6, min_child_weight=6,
        subsample=0.85, colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=8.0,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        early_stopping_rounds=120, n_jobs=-1,
    )
    if use_cuda:
        params["device"] = "cuda"
    return params


def _triple_gbdt_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    behavior_y: np.ndarray,
    groups: np.ndarray,
    known: np.ndarray,
    n_eval_pairs: int,
    *,
    use_cuda: bool,
    fit_final_on_all: bool = False,
    X_eval: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """V30's triple-GBDT risk head under 5-fold SGKF + PU-stress (VERBATIM, Rule 6).

    Returns a dict with ``oof_risk`` (the 0.40/0.35/0.25 blend) and, when
    ``fit_final_on_all`` and ``X_eval`` are given, ``eval_risk`` (mean over the 5 fold
    models applied to the eval matrix — exactly the notebook's eval inference).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    lgb_risk_params = dict(
        n_estimators=1200, learning_rate=0.025, num_leaves=45, min_child_samples=18,
        subsample=0.85, colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=8.0,
        objective="binary", n_jobs=-1, verbose=-1,
    )
    cb_risk_params = dict(
        iterations=1100, learning_rate=0.030, depth=6, l2_leaf_reg=6.0,
        loss_function="Logloss", eval_metric="Logloss", verbose=0, thread_count=-1,
    )

    pu_expansion = max(1.0, n_eval_pairs / max((~known).sum(), 1))
    stress_weight = np.where(known, 1.0, pu_expansion)
    fit_weight = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_xgb = np.zeros(len(X), dtype=np.float32)
    oof_lgb = np.zeros(len(X), dtype=np.float32)
    oof_cb = np.zeros(len(X), dtype=np.float32)
    oof_risk = np.zeros(len(X), dtype=np.float32)

    eval_xgb, eval_lgb, eval_cb = [], [], []

    fold_scores = []
    for fold, (tr, va) in enumerate(cv.split(X, behavior_y, groups)):
        m_xgb = XGBClassifier(**_xgb_risk_params(use_cuda), random_state=SEED + fold)
        m_xgb.fit(
            X.iloc[tr], y[tr], sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])], sample_weight_eval_set=[stress_weight[va]],
            verbose=False,
        )
        oof_xgb[va] = m_xgb.predict_proba(X.iloc[va])[:, 1]

        m_lgb = LGBMClassifier(**lgb_risk_params, random_state=SEED + fold)
        m_lgb.fit(
            X.iloc[tr], y[tr], sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])], eval_sample_weight=[stress_weight[va]],
        )
        oof_lgb[va] = m_lgb.predict_proba(X.iloc[va])[:, 1]

        m_cb = CatBoostClassifier(**cb_risk_params, random_seed=SEED + fold)
        m_cb.fit(
            X.iloc[tr], y[tr], sample_weight=fit_weight[tr],
            eval_set=[(X.iloc[va], y[va])], early_stopping_rounds=80, verbose=0,
        )
        oof_cb[va] = m_cb.predict_proba(X.iloc[va])[:, 1]

        oof_risk[va] = _W_XGB * oof_xgb[va] + _W_LGB * oof_lgb[va] + _W_CB * oof_cb[va]

        if fit_final_on_all and X_eval is not None:
            eval_xgb.append(m_xgb.predict_proba(X_eval)[:, 1])
            eval_lgb.append(m_lgb.predict_proba(X_eval)[:, 1])
            eval_cb.append(m_cb.predict_proba(X_eval)[:, 1])

        labeled_va = va[known[va]]
        if labeled_va.size and len(np.unique(y[labeled_va])) > 1:
            from sklearn.metrics import average_precision_score
            fold_scores.append(float(average_precision_score(y[labeled_va], oof_risk[labeled_va])))

    result: Dict[str, object] = {"oof_risk": oof_risk, "fold_labeled_ap": fold_scores}
    if fit_final_on_all and X_eval is not None:
        eval_risk = (
            _W_XGB * np.mean(eval_xgb, axis=0)
            + _W_LGB * np.mean(eval_lgb, axis=0)
            + _W_CB * np.mean(eval_cb, axis=0)
        )
        result["eval_risk"] = eval_risk.astype(np.float32)
    return result


# --------------------------------------------------------------------------- #
# Scoring helpers (reused §69/§73 path)                                         #
# --------------------------------------------------------------------------- #


def _confirmed_pair_ap(
    oof_df: pd.DataFrame,
    scorer: CanonicalScorer,
    labels: pd.DataFrame,
    evidence: Optional[pd.DataFrame],
) -> float:
    """Canonical confirmed-only PairAP of an OOF frame (§69/§73 construction, Rule 6)."""
    preds = _emit_dev_predictions(oof_df, labels, evidence=evidence)
    return float(scorer.score_dev_predictions(preds).pair_ap)


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def _colstats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {
        c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in cols
    }


def main() -> int:
    poker_root = Path(__file__).resolve().parents[1]
    paths = default_paths(poker_root)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        log_lines.append(str(msg))

    summary: Dict[str, object] = {
        "recipe_id": "graph_retest_v93",
        "gate2_bar": GATE2_BAR,
        "drift_threshold": DRIFT_AUC_THRESHOLD,
    }

    t0 = time.time()
    missing = paths.missing()
    if missing:
        summary["status"] = "BLOCKED"
        summary["failure_detail"] = "missing inputs: " + ", ".join(str(p) for p in missing)
        _write_summary(paths, summary, log_lines)
        log("BLOCKED: missing inputs:\n  " + "\n  ".join(str(p) for p in missing))
        return 1

    try:
        # ---- Load pairs + graph inputs (lazy) ----
        dev_pairs = pl.read_parquet(paths.dev_pairs)
        eval_pairs = pl.read_parquet(paths.eval_pairs)
        seats_lf = pl.scan_parquet(paths.seats)
        hands_lf = pl.scan_parquet(paths.hands)
        players_lf = pl.scan_parquet(paths.players)

        # ============ STEP 1 — SHARED GRAPH BUILDER (parity fix) ============
        log("=" * 70)
        log("STEP 1 — shared graph builder (dev + eval via ONE code path)")
        g_dev = build_graph_block("development", dev_pairs, seats_lf=seats_lf, hands_lf=hands_lf)
        g_eval = build_graph_block("evaluation", eval_pairs, seats_lf=seats_lf, hands_lf=hands_lf)
        g_dev.to_parquet(paths.out_dir / "_G_dev.parquet", index=False)
        g_eval.to_parquet(paths.out_dir / "_G_eval.parquet", index=False)
        log(f"  dev graph rows:  {len(g_dev):,}")
        log(f"  eval graph rows: {len(g_eval):,}")
        dev_stats = _colstats(g_dev, GRAPH_COLUMNS)
        eval_stats = _colstats(g_eval, GRAPH_COLUMNS)
        for c in GRAPH_COLUMNS:
            log(f"    {c:32s} dev mean={dev_stats[c]['mean']:.5f} std={dev_stats[c]['std']:.5f} "
                f"| eval mean={eval_stats[c]['mean']:.5f} std={eval_stats[c]['std']:.5f}")
        summary["dev_graph_rows"] = int(len(g_dev))
        summary["eval_graph_rows"] = int(len(g_eval))
        summary["dev_col_stats"] = dev_stats
        summary["eval_col_stats"] = eval_stats

        # ============ STEP 2 — DRIFT GATE (reuse) ============
        log("=" * 70)
        log("STEP 2 — adversarial drift gate (reused matrix gate, threshold 0.65)")
        drift_rows = []
        clean_cols: List[str] = []
        for c in GRAPH_COLUMNS:
            Xd = g_dev[[c]].to_numpy(np.float32)
            Xe = g_eval[[c]].to_numpy(np.float32)
            dg = adversarial_drift_auc_matrix(Xd, Xe, block=f"graph::{c}", seed=7)
            passed = bool(dg.passed)
            if passed:
                clean_cols.append(c)
            drift_rows.append({"column": c, "drift_auc": float(dg.auc), "passed": passed})
            log(f"    {c:32s} drift AUC = {dg.auc:.4f}  -> {'PASS' if passed else 'FAIL'}")
        # joint
        Xd_all = g_dev[GRAPH_COLUMNS].to_numpy(np.float32)
        Xe_all = g_eval[GRAPH_COLUMNS].to_numpy(np.float32)
        dg_joint = adversarial_drift_auc_matrix(Xd_all, Xe_all, block="graph::JOINT", seed=7)
        log(f"    {'ALL COLUMNS (joint)':32s} drift AUC = {dg_joint.auc:.4f}  -> "
            f"{'PASS' if dg_joint.passed else 'FAIL'}")
        summary["per_column_drift"] = drift_rows
        summary["joint_drift_auc"] = float(dg_joint.auc)
        summary["joint_drift_passed"] = bool(dg_joint.passed)
        summary["clean_columns"] = clean_cols
        summary["n_clean_columns"] = len(clean_cols)

        gate1_pass = len(clean_cols) >= 1
        summary["gate1_pass"] = gate1_pass

        if not gate1_pass:
            # ---- GATE 1 FAIL = honest DEAD-END proof (Rule 3) ----
            summary["gate2_pass"] = False
            summary["verdict"] = "DEAD-END PROVEN (intrinsic drift)"
            summary["v30_alone_pair_ap"] = None
            summary["augmented_pair_ap"] = None
            summary["delta"] = None
            summary["gpu_used"] = None
            summary["status"] = "COMPLETED"
            summary["duration_s"] = round(time.time() - t0, 1)
            log("=" * 70)
            log("GATE 1 FAILED: no graph column has drift AUC < 0.65.")
            log("VERDICT: DEAD-END PROVEN (intrinsic drift) — graph is dev/eval-divergent here.")
            log("Not grafting (Rule 3: honest null is a result).")
            _write_summary(paths, summary, log_lines)
            return 0

        log(f"  {len(clean_cols)} drift-clean column(s): {clean_cols}")

        # ============ STEP 3 — V30 GRAFT ============
        log("=" * 70)
        log("STEP 3 — rebuild V30 274-feat matrix + graft drift-clean graph cols")

        dev_feats = _v30_aggregate_pair_features(
            paths.dev_hand_features, dev_pairs, players_lf=players_lf, baseline_path=paths.player_baselines
        )
        eval_feats = _v30_aggregate_pair_features(
            paths.eval_hand_features, eval_pairs, players_lf=players_lf, baseline_path=paths.player_baselines
        )
        log(f"  V30 dev matrix:  {dev_feats.shape}")
        log(f"  V30 eval matrix: {eval_feats.shape}")

        # cv_group = table_id (V30 groups by the modal table of each pair's hands; the
        # persisted dev_pairs already carries table_id which is that modal table).
        pair_groups = dev_pairs.select("pair_id", pl.col("table_id").alias("cv_group"))

        # Build the training frame exactly as V30 does.
        labels_pl = dev_pairs.select(["pair_id", "behavior_family", "is_labeled"])
        train_df = (
            dev_feats.join(labels_pl, on="pair_id", how="left")
            .join(pair_groups, on="pair_id", how="left")
            .sort("pair_id")
        )
        behavior_to_id = {"none": 0, "directed_transfer": 1, "soft_play": 2, "coordinated_isolation": 3}
        known = train_df["is_labeled"].fill_null(False).to_numpy().astype(bool)
        behavior_y = np.array(
            [behavior_to_id.get(x, 0) for x in train_df["behavior_family"].fill_null("none").to_list()],
            dtype=np.int8,
        )
        y = (behavior_y > 0).astype(np.int8)
        groups = train_df["cv_group"].to_numpy()

        # Exactly the notebook's exclude set (Rule 6). ``is_labeled``/``behavior_family``
        # are the join markers here (the notebook's ``is_known``), so add them; keep
        # every genuine feature (incl. shared_hands) so the base is byte-faithful to V30.
        exclude = {"pair_id", "player_1", "player_2", "cv_group", "is_known", "is_labeled", "behavior_family"}
        feature_cols = [
            c for c, dtype in train_df.schema.items() if dtype.is_numeric() and c not in exclude
        ]
        X_base = train_df.select(feature_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        log(f"  V30-alone feature count: {len(feature_cols)}")

        # Eval base matrix (same feature_cols, same order).
        X_eval_base = (
            eval_feats.select(feature_cols).to_pandas()
            .replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        )

        # Graph cols aligned to train_df pair order.
        g_dev_idx = g_dev.set_index("pair_id")
        train_pids = train_df["pair_id"].to_list()
        g_dev_aligned = g_dev_idx.reindex([str(p) for p in train_pids])[clean_cols].fillna(0.0)
        X_aug = X_base.copy()
        for c in clean_cols:
            X_aug[c] = g_dev_aligned[c].to_numpy(np.float32)

        # Detect GPU availability once.
        use_cuda = _probe_cuda(log)
        summary["gpu_used"] = bool(use_cuda)

        # ---- V30-alone OOF ----
        log("  training V30-ALONE triple-GBDT (5-fold SGKF + PU-stress)...")
        n_eval = len(eval_pairs)
        base_res = _triple_gbdt_oof(
            X_base, y, behavior_y, groups, known, n_eval, use_cuda=use_cuda
        )
        oof_base = np.asarray(base_res["oof_risk"])

        # ---- V30 + graph OOF ----
        log("  training V30+GRAPH triple-GBDT (5-fold SGKF + PU-stress)...")
        aug_res = _triple_gbdt_oof(
            X_aug, y, behavior_y, groups, known, n_eval, use_cuda=use_cuda
        )
        oof_aug = np.asarray(aug_res["oof_risk"])

        # ---- Score confirmed PairAP both ways (§69/§73 path) ----
        labels_csv = pd.read_csv(paths.dev_labels_csv)
        evidence_csv = pd.read_csv(paths.dev_evidence_csv) if paths.dev_evidence_csv.is_file() else None
        scorer = CanonicalScorer(
            ScoringRecipe(),
            PipelineConfig(input_dir=paths.data_dir),
            labels=labels_csv,
            evidence=evidence_csv,
        )
        oof_base_df = pd.DataFrame({"pair_id": [str(p) for p in train_pids], "is_labeled": known, "oof_risk": oof_base})
        oof_aug_df = pd.DataFrame({"pair_id": [str(p) for p in train_pids], "is_labeled": known, "oof_risk": oof_aug})

        v30_alone_ap = _confirmed_pair_ap(oof_base_df, scorer, labels_csv, evidence_csv)
        augmented_ap = _confirmed_pair_ap(oof_aug_df, scorer, labels_csv, evidence_csv)
        delta = augmented_ap - v30_alone_ap
        gate2_pass = bool(delta >= GATE2_BAR)

        summary["v30_alone_pair_ap"] = v30_alone_ap
        summary["augmented_pair_ap"] = augmented_ap
        summary["delta"] = delta
        summary["gate2_pass"] = gate2_pass
        summary["v30_fold_labeled_ap"] = base_res.get("fold_labeled_ap")
        summary["aug_fold_labeled_ap"] = aug_res.get("fold_labeled_ap")

        log("=" * 70)
        log(f"  V30-alone confirmed PairAP : {v30_alone_ap:.6f}")
        log(f"  V30+graph confirmed PairAP : {augmented_ap:.6f}")
        log(f"  delta                      : {delta:+.6f}  (bar +{GATE2_BAR})")
        log(f"  GATE 2 (delta >= +{GATE2_BAR}): {'PASS' if gate2_pass else 'FAIL'}")

        # ============ STEP 4 — VERDICT + optional eval candidate ============
        if not gate2_pass:
            summary["verdict"] = "TRANSPORT-CLEAN BUT NULL ON V30 (V30 already captures it)"
            summary["eval_candidate_path"] = None
            log("VERDICT: TRANSPORT-CLEAN BUT NULL ON V30 (V30 already captures it).")
            log("Not building an eval candidate (Rule 3).")
        else:
            summary["verdict"] = "LIVE ORTHOGONAL LEVER"
            log("VERDICT: LIVE ORTHOGONAL LEVER — building eval candidate (no submit).")
            cand_path, spearman = _build_eval_candidate(
                paths, X_aug, y, behavior_y, groups, known, n_eval, feature_cols,
                clean_cols, X_eval_base, g_eval, eval_pairs, use_cuda, log,
            )
            summary["eval_candidate_path"] = str(cand_path) if cand_path else None
            summary["eval_candidate_spearman_vs_v30"] = spearman

        summary["status"] = "COMPLETED"
        summary["duration_s"] = round(time.time() - t0, 1)
        _write_summary(paths, summary, log_lines)
        _cleanup(paths)
        return 0

    except Exception as exc:  # noqa: BLE001 — record honestly, never fabricate.
        tb = traceback.format_exc()
        summary["status"] = "ERROR"
        summary["error"] = repr(exc)
        summary["traceback_tail"] = tb[-4000:]
        summary["duration_s"] = round(time.time() - t0, 1)
        _write_summary(paths, summary, log_lines)
        log("=" * 70)
        log("ERROR — traceback tail:")
        log(tb[-4000:])
        return 1


def _probe_cuda(log) -> bool:
    """Return True if an XGB fit on device='cuda' succeeds; else fall back to CPU."""
    try:
        import numpy as _np
        from xgboost import XGBClassifier
        _X = _np.random.rand(256, 4).astype("float32")
        _y = (_X[:, 0] > 0.5).astype(int)
        XGBClassifier(n_estimators=10, max_depth=3, device="cuda", tree_method="hist").fit(_X, _y)
        log("  GPU: xgboost device='cuda' OK (V30 XGB uses CUDA).")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"  GPU: device='cuda' unavailable ({exc!r}); falling back to CPU tree_method='hist'.")
        return False


def _build_eval_candidate(
    paths: GraphRetestPaths,
    X_aug: pd.DataFrame,
    y: np.ndarray,
    behavior_y: np.ndarray,
    groups: np.ndarray,
    known: np.ndarray,
    n_eval: int,
    feature_cols: List[str],
    clean_cols: List[str],
    X_eval_base: pd.DataFrame,
    g_eval: pd.DataFrame,
    eval_pairs: pl.DataFrame,
    use_cuda: bool,
    log,
) -> Tuple[Optional[Path], Optional[float]]:
    """Build the eval candidate CSV (V30 behavior+evidence verbatim + augmented risk)."""
    # Assemble the eval augmented matrix (base cols + drift-clean graph cols).
    eval_pids = [str(p) for p in eval_pairs["pair_id"].to_list()]
    g_eval_idx = g_eval.set_index("pair_id")
    g_eval_aligned = g_eval_idx.reindex(eval_pids)[clean_cols].fillna(0.0)
    X_eval_aug = X_eval_base.copy()
    for c in clean_cols:
        X_eval_aug[c] = g_eval_aligned[c].to_numpy(np.float32)

    log("  retraining augmented triple-GBDT on ALL dev, predicting eval...")
    res = _triple_gbdt_oof(
        X_aug, y, behavior_y, groups, known, n_eval,
        use_cuda=use_cuda, fit_final_on_all=True, X_eval=X_eval_aug,
    )
    eval_risk = np.asarray(res["eval_risk"], dtype=np.float64)
    # Normalize risk to [0,1].
    lo, hi = float(eval_risk.min()), float(eval_risk.max())
    risk_norm = (eval_risk - lo) / (hi - lo) if hi > lo else np.zeros_like(eval_risk)

    # V30's submission provides behavior + evidence verbatim.
    v30 = pd.read_csv(paths.v30_submission, dtype={"pair_id": str})
    cand = pd.DataFrame({"pair_id": eval_pids, "risk_score": risk_norm})
    reuse_cols = ["pair_id", "predicted_behavior", *_EVIDENCE_COLS]
    cand = cand.merge(v30[reuse_cols], on="pair_id", how="left")
    cand = cand[_SUBMISSION_COLUMNS]

    # Validate.
    eval_ref = pd.read_csv(paths.eval_pairs_csv, dtype={"pair_id": str})
    checks = {
        "row_count_112540": bool(len(cand) == _EXPECTED_EVAL_ROWS),
        "schema_8col_exact": bool(list(cand.columns) == _SUBMISSION_COLUMNS),
        "pair_set_matches_eval": bool(set(cand["pair_id"]) == set(eval_ref["pair_id"].astype(str))),
        "risk_in_0_1": bool(cand["risk_score"].between(0.0, 1.0).all() and cand["risk_score"].notna().all()),
        "no_nulls": bool(int(cand.isnull().sum().sum()) == 0),
    }
    # duplicate evidence within a pair (NO_EVIDENCE repeats allowed)
    dup = 0
    ev = cand[_EVIDENCE_COLS].astype(str)
    for row in ev.itertuples(index=False, name=None):
        real = [h for h in row if h != "NO_EVIDENCE"]
        if len(real) != len(set(real)):
            dup += 1
    checks["no_duplicate_evidence"] = bool(dup == 0)
    all_ok = all(checks.values())
    log(f"  eval candidate validation: {checks} -> all_passed={all_ok}")
    if not all_ok:
        log("  eval candidate FAILED validation; not writing (Rule 1/9).")
        return None, None

    # Spearman vs V30 risk.
    merged = cand[["pair_id", "risk_score"]].merge(
        v30[["pair_id", "risk_score"]], on="pair_id", suffixes=("_new", "_v30")
    )
    spearman = float(
        np.corrcoef(merged["risk_score_new"].rank().values, merged["risk_score_v30"].rank().values)[0, 1]
    )
    cand_path = paths.out_dir / "candidate_v30_plus_graph.csv"
    cand.to_csv(cand_path, index=False)
    log(f"  wrote eval candidate -> {cand_path}  (Spearman vs V30 risk = {spearman:.4f})")
    return cand_path, spearman


def _write_summary(paths: GraphRetestPaths, summary: Dict[str, object], log_lines: List[str]) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    (paths.out_dir / "_graph_retest_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (paths.out_dir / "_graph_retest.log").write_text("\n".join(log_lines), encoding="utf-8")


def _cleanup(paths: GraphRetestPaths) -> None:
    """Keep _G_*.parquet, summary, log, candidate CSV. (No scratch scripts to remove.)"""
    return None


if __name__ == "__main__":
    raise SystemExit(main())
