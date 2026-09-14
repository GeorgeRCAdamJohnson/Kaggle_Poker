"""Co-hand-INVARIANT feature basis for the pervasive dev-vs-eval drift gate.

Pre-registered bar: RESEARCH_DOSSIER §53 (BINDING). Prior results §50/§51/§52 established
that dev<->eval drift is a PERVASIVE pool-sampling property: dev pairs share ~121 hands,
eval ~86 (ratio 0.71), and EVERY visible feature basis tested (v2, v3, exp3c, full floor,
nomannic) separates dev from eval at drift AUC ~0.76 on the adversarial gate — including
nomannic's bb-normalized rate aggregates (still 0.7441 after dropping raw counts). The
distribution of behavior differs between pools, not just the count.

This module engineers a feature basis INTENDED to be invariant to shared-hand count AND
to the pool-composition distribution shift, using three construction families that are
ALL count-cancelling by design (§53, contract Rule 19 — normalize by opportunity, do NOT
feature it):

1. WITHIN-PAIR RELATIVE RANKS: a signal expressed as its mean within-pair percentile, and
   the fraction of a pair's OWN hands in the pair's OWN top decile (unitless, count-free).
2. EXCESS-OVER-EXPECTED (partner-vs-outsider CONTRAST): partner_* MINUS outsider_* action
   rates, averaged per pair. The outsider baseline absorbs the pool's action distribution
   so only the ANOMALOUS partner behavior survives (Rule 19's expected-minus-observed).
3. SHUFFLE-NULL-RELATIVE: an observed within-pair statistic MINUS a within-pair
   permutation-null mean (per-pair label-free sign/label shuffle). Cancels opportunity by
   construction.

The assembly mirrors ``nomannic_recipe.build_nomannic_pair_matrices`` (SAME cache, SAME
pair grouping) but replaces the count/mean/quantile aggregates with the invariant features
above. The whole-basis adversarial drift gate is the SAME 5-fold OOF gate §50/§51/§52 used
(:func:`own_submission_recipes.adversarial_drift_auc_matrix`, threshold 0.65).

Nothing here is scored until the whole basis PASSES the gate (contract Rule 17, sequential).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from anchor_repro.own_submission_recipes import (
    DRIFT_AUC_THRESHOLD,
    DriftGateResult,
    adversarial_drift_auc_matrix,
)

__all__ = [
    "InvariantCacheLocations",
    "default_invariant_cache_locations",
    "build_invariant_pair_matrices",
    "PerFeatureDrift",
    "per_feature_drift_audit",
    "invariant_basis_drift_gate",
    "InvariantBasisResult",
    "run_invariant_basis",
]

# --------------------------------------------------------------------------- #
# Raw per-hand signals (verified on disk, dev/eval_hand_features.parquet).      #
# --------------------------------------------------------------------------- #

#: Continuous per-hand transfer / pot signals whose WITHIN-PAIR relative rank is
#: count-invariant. These are the signals whose magnitude differs by pool distribution;
#: the rank cancels both the count and the scale.
_RANK_SIGNALS: Tuple[str, ...] = (
    "transfer_any_bb",
    "transfer_1_to_2_bb",
    "transfer_2_to_1_bb",
    "net_gap_bb",
    "pair_pot_share",
    "directed_signal",
    "soft_signal",
    "isolation_signal",
)

#: Partner-vs-outsider action columns (raw per-hand counts of the action toward the pair).
#: Family 2 forms the within-hand contrast partner_X - outsider_X so the outsider baseline
#: absorbs the pool's action distribution.
_CONTRAST_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("partner_folds_to_pair", "outsider_folds_to_pair", "folds"),
    ("partner_calls_to_pair", "outsider_calls_to_pair", "calls"),
    ("partner_raises_to_pair", "outsider_raises_to_pair", "raises"),
)

#: Signals used for the Family-3 shuffle-null-relative feature. For each we compare the
#: observed within-pair mean to a per-pair permutation-null mean of the SAME within-hand
#: contrast (partner-vs-outsider label shuffled within the pair's hands).
_SHUFFLE_TOP_DECILE_SIGNALS: Tuple[str, ...] = (
    "transfer_any_bb",
    "directed_signal",
    "soft_signal",
    "isolation_signal",
)

#: Number of within-pair shuffle permutations for the Family-3 null (fixed seed).
_N_SHUFFLE: int = 24
_SHUFFLE_SEED: int = 53

#: A feature whose dev AND eval means are both within this of 0 is "near-zero-mean": its
#: mean ratio is ill-defined (ratio of two tiny numbers), so invariance is judged by the
#: single-feature AUC alone, not the ratio (contract Rule 9). The shuffle-null residuals
#: are centred at 0 by construction and fall here.
_NEAR_ZERO_MEAN: float = 5e-3


@dataclass(frozen=True)
class InvariantCacheLocations:
    """Resolved on-disk paths for the invariant-basis build (dev + eval hand features)."""

    prepared_dir: Path

    @property
    def dev_hf(self) -> Path:
        return self.prepared_dir / "dev_hand_features.parquet"

    @property
    def eval_hf(self) -> Path:
        return self.prepared_dir / "eval_hand_features.parquet"

    def missing(self) -> List[Path]:
        return [p for p in (self.dev_hf, self.eval_hf) if not Path(p).is_file()]


def default_invariant_cache_locations(poker_root: Path) -> InvariantCacheLocations:
    """Resolve the standard invariant-basis cache locations under a ``poker/`` tree root."""
    root = Path(poker_root)
    return InvariantCacheLocations(
        prepared_dir=root / "outputs" / "poker_collusion" / "repro_nomannic" / "prepared_v2"
    )


# --------------------------------------------------------------------------- #
# Feature construction (all count-cancelling by design).                        #
# --------------------------------------------------------------------------- #


def _family1_within_pair_ranks(hf: pl.DataFrame) -> pl.DataFrame:
    """FAMILY 1 — within-pair relative ranks (unitless, count-invariant).

    For each rank signal and each pair, compute:
      * ``<sig>_pctl``: the mean within-pair percentile of the signal (rank over the
        pair's OWN hands mapped to [0,1]); a pair whose big-transfer hands sit at the top
        of ITS OWN distribution scores high regardless of how many hands it played.
      * ``<sig>_topdecile``: the fraction of the pair's OWN hands whose signal is in the
        pair's OWN top decile (>= within-pair 0.9 quantile). Count-free by construction.
    """
    # NOTE (contract Rule 9 — honest degeneracy): the MEAN within-pair percentile of a
    # signal is identically 0.5 for every pair (averaging a within-group rank over the
    # whole group is always its midpoint), so a ``<sig>__pctl`` mean carries ZERO
    # information and is a constant. We therefore do NOT emit the mean-percentile feature;
    # the informative count-invariant rank statistic is the top-decile CONCENTRATION
    # (``<sig>__topdec``: the fraction of a pair's OWN hands at/above the pair's OWN 0.9
    # quantile) plus the DISPERSION of within-pair percentiles (``<sig>__pctl_std``: the
    # spread of the pair's own ranks, which is NOT constant and captures whether a pair's
    # signal mass is concentrated in a few hands).
    exprs: List[pl.Expr] = []
    for sig in _RANK_SIGNALS:
        n = pl.len().over("pair_id")
        rank = pl.col(sig).rank(method="average").over("pair_id")
        pctl = pl.when(n > 1).then((rank - 1.0) / (n - 1.0)).otherwise(0.5)
        exprs.append(pctl.alias(f"{sig}__pctlraw"))
        q90 = pl.col(sig).quantile(0.9, interpolation="nearest").over("pair_id")
        exprs.append((pl.col(sig) >= q90).cast(pl.Float64).alias(f"{sig}__topdec"))
    ranked = hf.select(["pair_id", *exprs])
    agg: List[pl.Expr] = []
    for sig in _RANK_SIGNALS:
        # top-decile concentration (informative, count-invariant)
        agg.append(pl.col(f"{sig}__topdec").mean().alias(f"{sig}__topdec"))
        # within-pair percentile dispersion (NOT the degenerate mean; std is informative)
        agg.append(pl.col(f"{sig}__pctlraw").std().fill_null(0.0).alias(f"{sig}__pctlstd"))
    return ranked.group_by("pair_id").agg(agg)


def _family2_excess_over_expected(hf: pl.DataFrame) -> pl.DataFrame:
    """FAMILY 2 — excess-over-expected (partner-vs-outsider contrast).

    Per hand, form ``partner_X - outsider_X`` (folds / calls / raises), then average per
    pair. The outsider baseline is the same-hand non-partner action rate, so it absorbs
    the pool's action distribution; only anomalous partner behavior survives. Also emit a
    normalized contrast dividing by (partner+outsider+1) so a pool with more total action
    does not scale the excess (extra count-cancellation).
    """
    exprs: List[pl.Expr] = []
    for partner, outsider, tag in _CONTRAST_PAIRS:
        p = pl.col(partner).cast(pl.Float64)
        o = pl.col(outsider).cast(pl.Float64)
        exprs.append((p - o).alias(f"excess_{tag}"))
        exprs.append(((p - o) / (p + o + 1.0)).alias(f"excess_{tag}_norm"))
    contrast = hf.select(["pair_id", *exprs])
    agg = [pl.col(c).mean().alias(c) for c in contrast.columns if c != "pair_id"]
    return contrast.group_by("pair_id").agg(agg)


def _family3_shuffle_null_relative(hf: pl.DataFrame) -> pl.DataFrame:
    """FAMILY 3 — shuffle-null-relative (observed minus within-pair permutation null).

    For each signal we compute the observed within-pair top-decile fraction (same as the
    Family-1 ``topdec`` statistic), then subtract a per-pair permutation-null expectation
    of that statistic obtained by shuffling the signal values WITHIN the pair's own hands
    ``_N_SHUFFLE`` times. Because the top-decile fraction of a within-pair shuffle has an
    expectation fixed by the pair's own distribution (not its count), the observed-minus-
    null residual cancels opportunity by construction and centres a null pair near 0.

    Implemented in numpy per pair (polars group_map is awkward for permutation nulls).
    """
    rng = np.random.default_rng(_SHUFFLE_SEED)
    pdf = hf.select(["pair_id", *_SHUFFLE_TOP_DECILE_SIGNALS]).to_pandas()
    out: Dict[str, List[float]] = {"pair_id": []}
    for sig in _SHUFFLE_TOP_DECILE_SIGNALS:
        out[f"{sig}__nullrel"] = []

    for pid, grp in pdf.groupby("pair_id", sort=False):
        out["pair_id"].append(pid)
        n = len(grp)
        for sig in _SHUFFLE_TOP_DECILE_SIGNALS:
            v = grp[sig].to_numpy(dtype=float)
            if n < 2:
                out[f"{sig}__nullrel"].append(0.0)
                continue
            thresh = np.quantile(v, 0.9, method="nearest")
            observed = float(np.mean(v >= thresh))
            # Within-pair permutation null of the top-decile fraction. The top-decile
            # count is invariant to permutation for a fixed value multiset (the SET of
            # values above the threshold is unchanged by reordering), so the informative
            # null is a resampling of the threshold under a symmetric label shuffle:
            # draw bootstrap resamples of the pair's own values and recompute the fraction
            # above the pair's own 0.9 quantile. Centres a homogeneous pair at ~0.1.
            null_fracs = np.empty(_N_SHUFFLE, dtype=float)
            for k in range(_N_SHUFFLE):
                resample = v[rng.integers(0, n, size=n)]
                null_fracs[k] = np.mean(resample >= thresh)
            out[f"{sig}__nullrel"].append(observed - float(np.mean(null_fracs)))
    return pl.from_pandas(pd.DataFrame(out))


def build_invariant_pair_matrices(
    caches: InvariantCacheLocations,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """Assemble the dev + eval INVARIANT pair matrices from the per-hand caches.

    Mirrors the nomannic assembly (same cache, same ``group_by('pair_id')``) but replaces
    the count/mean/quantile aggregates with the three invariant families. Every feature is
    count-cancelling by design; the per-feature drift audit downstream verifies which
    actually are (§53 drop rule).

    Returns ``(Xd, Xe, feature_cols, dev_pd, eval_pd)`` — aligned float32 matrices, the
    shared feature column names, and the per-pair pandas frames (indexed by pair_id) for
    the audit / downstream scoring.

    Raises:
        FileNotFoundError: If any required cache is absent (names the missing paths).
    """
    missing = caches.missing()
    if missing:
        raise FileNotFoundError(
            "invariant basis cannot be built; missing cache file(s): "
            + ", ".join(str(p) for p in missing)
        )

    def _build_one(hf_path: Path) -> pd.DataFrame:
        hf = pl.read_parquet(hf_path)
        f1 = _family1_within_pair_ranks(hf)
        f2 = _family2_excess_over_expected(hf)
        f3 = _family3_shuffle_null_relative(hf)
        merged = f1.join(f2, on="pair_id", how="left").join(f3, on="pair_id", how="left")
        return merged.to_pandas().set_index("pair_id")

    dev_pd = _build_one(caches.dev_hf)
    eval_pd = _build_one(caches.eval_hf)

    feature_cols = [c for c in dev_pd.columns if c in eval_pd.columns]
    dev_pd = dev_pd[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eval_pd = eval_pd[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Xd = dev_pd[feature_cols].to_numpy(np.float32)
    Xe = eval_pd[feature_cols].to_numpy(np.float32)
    return Xd, Xe, feature_cols, dev_pd, eval_pd


# --------------------------------------------------------------------------- #
# Per-feature drift audit (contract Rules 5, 9; §53 drop rule).                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PerFeatureDrift:
    """Single-feature dev-vs-eval drift audit row.

    Attributes:
        feature: The feature column name.
        dev_mean: Dev mean of the feature.
        eval_mean: Eval mean of the feature.
        ratio: ``dev_mean / eval_mean`` (nan-safe; a value far from 1.0 => not invariant).
        auc: Single-feature dev-vs-eval separation AUC (0.5 = invariant, >=0.65 = drifts).
        kept: ``True`` if the feature survives the §53 drop rule.
        reason: Why it was kept/dropped.
    """

    feature: str
    dev_mean: float
    eval_mean: float
    ratio: float
    auc: float
    kept: bool
    reason: str


def _single_feature_auc(dev_col: np.ndarray, eval_col: np.ndarray) -> float:
    """Direction-agnostic single-feature dev-vs-eval AUC (max(auc, 1-auc)).

    A single feature separates dev from eval to the extent its value distributions differ;
    ``roc_auc_score`` with the raw value is direction-sensitive, so we report the
    magnitude of separation (>=0.5). A constant feature yields exactly 0.5.
    """
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(dev_col)), np.ones(len(eval_col))])
    x = np.concatenate([dev_col, eval_col]).astype(float)
    if np.allclose(x.max(), x.min()):
        return 0.5
    auc = roc_auc_score(y, x)
    return float(max(auc, 1.0 - auc))


def per_feature_drift_audit(
    dev_pd: pd.DataFrame,
    eval_pd: pd.DataFrame,
    feature_cols: List[str],
    *,
    ratio_lo: float = 0.85,
    ratio_hi: float = 1.15,
    auc_max: float = DRIFT_AUC_THRESHOLD,
) -> List[PerFeatureDrift]:
    """Audit each engineered feature for drift; apply the §53 drop rule.

    §53 drop rule: a feature is KEPT only if its dev/eval mean ratio is ~1.0 AND its
    single-feature drift AUC < 0.65. Any feature still carrying the count ratio (mean
    ratio far from 1.0) OR separating dev/eval on its own (AUC >= 0.65) is DROPPED BEFORE
    the whole-basis gate — no passing a gate by diluting a drifter among clean features.

    The ratio band is centred on 1.0. Excess-over-expected features can legitimately have
    means near 0 (making a raw ratio explosive/ill-defined); for those the AUC criterion
    is the binding invariance test and the ratio is reported for transparency but a near-
    zero-mean feature is judged on AUC alone (a mean-0 feature with AUC<0.65 is invariant).

    Returns the audit rows in the input feature order.
    """
    rows: List[PerFeatureDrift] = []
    for c in feature_cols:
        dv = dev_pd[c].to_numpy(dtype=float)
        ev = eval_pd[c].to_numpy(dtype=float)
        dm = float(np.mean(dv))
        em = float(np.mean(ev))
        auc = _single_feature_auc(dv, ev)
        # Near-zero-mean features (e.g. the shuffle-null residuals, centred at 0 by
        # construction) have an ill-defined / explosive mean ratio; the ratio test is
        # meaningless there and invariance is judged by AUC alone (contract Rule 9). A
        # feature is "near-zero-mean" when BOTH sides sit within _NEAR_ZERO of 0.
        near_zero_mean = abs(dm) < _NEAR_ZERO_MEAN and abs(em) < _NEAR_ZERO_MEAN
        if abs(em) < 1e-9 or near_zero_mean:
            ratio = dm / em if abs(em) >= 1e-9 else float("nan")
        else:
            ratio = dm / em

        auc_ok = auc < auc_max
        if near_zero_mean or np.isnan(ratio):
            # mean ~0 on both sides: ratio uninformative; invariance judged by AUC only.
            ratio_ok = True
            ratio_note = "mean~0 (ratio n/a; judged on AUC)"
        else:
            ratio_ok = ratio_lo <= ratio <= ratio_hi
            ratio_note = f"ratio {ratio:.3f}"

        kept = auc_ok and ratio_ok
        if kept:
            reason = f"KEEP: AUC {auc:.4f}<{auc_max}, {ratio_note}"
        elif not auc_ok and not ratio_ok:
            reason = f"DROP: AUC {auc:.4f}>={auc_max} AND {ratio_note} out of [{ratio_lo},{ratio_hi}]"
        elif not auc_ok:
            reason = f"DROP: single-feature AUC {auc:.4f}>={auc_max}"
        else:
            reason = f"DROP: {ratio_note} out of [{ratio_lo},{ratio_hi}]"

        rows.append(PerFeatureDrift(c, dm, em, ratio, auc, kept, reason))
    return rows


# --------------------------------------------------------------------------- #
# Whole-basis drift gate + orchestration.                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InvariantBasisResult:
    """Full outcome of the §53 invariant-basis experiment.

    Attributes:
        audit: Per-feature drift audit rows (all engineered features).
        kept_features: Feature columns surviving the §53 drop rule.
        dropped_features: Feature columns dropped and why.
        gate: The whole-basis drift gate result on the SURVIVING features.
        pair_ap: Dev-holdout confirmed PairAP (only computed if the gate PASSED, else None).
        pair_ap_chance: The chance-level PairAP baseline (positive prevalence) for context.
    """

    audit: List[PerFeatureDrift]
    kept_features: List[str]
    dropped_features: List[Tuple[str, str]]
    gate: DriftGateResult
    pair_ap: float | None
    pair_ap_chance: float | None


def invariant_basis_drift_gate(
    caches: InvariantCacheLocations, *, seed: int = 7
) -> Tuple[DriftGateResult, List[PerFeatureDrift], List[str], pd.DataFrame, pd.DataFrame]:
    """Build the basis, audit + drop per §53, run the whole-basis gate on survivors.

    Returns ``(gate, audit, kept_features, dev_pd_kept, eval_pd_kept)``.
    """
    Xd, Xe, feature_cols, dev_pd, eval_pd = build_invariant_pair_matrices(caches)
    audit = per_feature_drift_audit(dev_pd, eval_pd, feature_cols)
    kept = [r.feature for r in audit if r.kept]
    if not kept:
        # No feature survived; the gate is trivially undefined. Record a degenerate
        # DISQUALIFIED result honestly rather than fabricate a pass.
        gate = DriftGateResult(
            block="invariant_kept0",
            auc=float("nan"),
            threshold=DRIFT_AUC_THRESHOLD,
            passed=False,
            n_features=0,
            n_dev=int(len(dev_pd)),
            n_eval=int(len(eval_pd)),
        )
        return gate, audit, kept, dev_pd, eval_pd

    Xd_k = dev_pd[kept].to_numpy(np.float32)
    Xe_k = eval_pd[kept].to_numpy(np.float32)
    gate = adversarial_drift_auc_matrix(Xd_k, Xe_k, block="invariant_kept", seed=seed)
    return gate, audit, kept, dev_pd, eval_pd


# --------------------------------------------------------------------------- #
# Dev-holdout PairAP (ONLY if the gate PASSES — contract Rule 17, sequential).  #
# --------------------------------------------------------------------------- #

#: PU sample-weight for eval-count rescale (matches the floor / own_submission recipes).
_PU_WEIGHT_EVAL_ROWS: float = 112540.0


def _dev_labels_and_tables(caches: InvariantCacheLocations) -> pd.DataFrame:
    """Load dev pair labels + table_id + PU flag, indexed by pair_id.

    Uses ``dev_pairs.parquet`` from the SAME prepared cache the hand features come from.
    label in {1 positive, 0 known-negative, null PU}; ``table_id`` is the group key for
    the grouped-by-table OOF (contract: 5-fold grouped-by-table like the other recipes).
    """
    dp = pl.read_parquet(caches.prepared_dir / "dev_pairs.parquet")
    cols = ["pair_id", "label", "table_id"]
    return dp.select(cols).to_pandas().set_index("pair_id")


def _dev_holdout_pair_ap(
    dev_pd_kept: pd.DataFrame,
    labels: pd.DataFrame,
    kept: List[str],
    *,
    seed: int = 7,
) -> Tuple[float, float]:
    """Grouped-by-table 5-fold OOF confirmed PairAP of a simple XGB PU risk model.

    Matches the other recipes' PU setup: sample weights pos 2.0 / known-neg 1.0 / PU 0.35
    (fit weights), and ``_PU_WEIGHT_EVAL_ROWS/n_pu`` unknown rescale for the AP weighting.
    The OOF folds are GroupKFold over ``table_id`` (table-disjoint). The PairAP is the
    ``average_precision_score`` over the CONFIRMED dev pairs (label in {0,1}) using their
    OOF risk scores. Returns ``(pair_ap, chance)`` where chance = positive prevalence.

    Only called when the whole-basis drift gate PASSED (Rule 17).
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    df = dev_pd_kept.join(labels, how="left")
    lab = df["label"].to_numpy()
    # PU: null -> unknown (-1), else the {0,1} confirmed label.
    lab_int = np.where(pd.isna(lab), -1, lab).astype(int)
    y = (lab_int == 1).astype(int)
    known = lab_int >= 0
    n_pu = int((lab_int == -1).sum())
    fw = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35))

    groups = df["table_id"].astype(str).fillna("NA").to_numpy()
    X = df[kept].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)

    oof = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=5)
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=6,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X[tr_idx], y[tr_idx], sample_weight=fw[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]

    # Confirmed PairAP: over label in {0,1} only (PU unknowns never fabricated as negs).
    conf = known
    y_conf = y[conf]
    s_conf = oof[conf]
    if y_conf.sum() == 0 or y_conf.sum() == len(y_conf):
        return float("nan"), float(y_conf.mean())
    pair_ap = float(average_precision_score(y_conf, s_conf))
    chance = float(y_conf.mean())
    return pair_ap, chance


def run_invariant_basis(
    caches: InvariantCacheLocations, *, seed: int = 7
) -> InvariantBasisResult:
    """End-to-end §53 experiment: build -> audit/drop -> gate -> (PairAP iff PASS).

    Sequential per contract Rule 17: the dev-holdout PairAP is computed ONLY if the
    whole-basis drift gate PASSES (AUC < 0.65). If it FAILS, the PairAP is left None
    (a drifting basis makes the local PairAP an inverted, untrustworthy predictor).
    """
    gate, audit, kept, dev_pd, eval_pd = invariant_basis_drift_gate(caches, seed=seed)
    dropped = [(r.feature, r.reason) for r in audit if not r.kept]

    pair_ap: float | None = None
    chance: float | None = None
    if gate.passed and kept:
        labels = _dev_labels_and_tables(caches)
        pair_ap, chance = _dev_holdout_pair_ap(dev_pd[kept], labels, kept, seed=seed)

    return InvariantBasisResult(
        audit=audit,
        kept_features=kept,
        dropped_features=dropped,
        gate=gate,
        pair_ap=pair_ap,
        pair_ap_chance=chance,
    )
