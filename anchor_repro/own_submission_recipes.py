"""Own-submission dev-holdout recipe runners for the fast-pass ladder (§49 → §50).

design.md "Components and Interfaces" §3 (LadderValidator / the injected
``recipe_runner``); Requirements 2.1, 2.5, 2.6. Governed by the workspace
``reverse-engineering-accountability`` contract. Pre-registered bar: RESEARCH_DOSSIER
§49 (fast-pass ladder expansion to n>=5 using OUR OWN externally-verified submissions).

Why this module exists (contract Rules 2, 6, 11).
-------------------------------------------------
§47 reached ladder n=2 (floor 044519 + competitor 064469). §49 pre-registered a FAST
PASS to n>=4/5 using OUR OWN Kaggle submissions, each carrying a real externally-
verified public-LB score, re-run on the dev holdout via the EXISTING
:mod:`recipe_registry` machinery. This module closes the execution gap by adding two
own-submission dev-holdout recipe runners:

* ``037063`` (cand_17): rank-average of two XGBs (depth 4 + depth 6) PU risk on the
  cached v3 features (``feature_cache/v3/dev_v3.parquet``).
* ``032580`` (cand_14): a single XGB PU risk on the cached v2 features
  (``feature_cache/v2/dev_v2.parquet``).

Both mirror :func:`recipe_registry.rerun_floor_stack` EXACTLY for the seed-7 60/40
table-disjoint split, the PU sample weights (pos 2.0 / confirmed 1.0 / PU 0.35, with
the ``112540/n_pu`` unknown rescale), and the canonical confirmed-only PairAP scoring.
Only the feature block and the model head differ. No feature is recomputed — the
persisted v3/v2 caches are loaded verbatim (contract Rule 6).

Two AP definitions, stated honestly (contract Rule 9).
------------------------------------------------------
As in :mod:`recipe_registry`, each runner records BOTH the loopv-style weighted-all AP
(``average_precision_score`` over ALL holdout pairs with PU sample weights) AND the
canonical confirmed-only PairAP (the official metric over the CONFIRMED holdout pairs).
These are two different AP definitions on two different row sets and are never
conflated. Only the canonical PairAP feeds the rank-tracking gate.

The drift gate BEFORE trust (contract Rules 5, 17; §49).
--------------------------------------------------------
A new point's dev-holdout PairAP is a trustworthy LB predictor ONLY if its feature
basis does not drift dev->eval. :func:`adversarial_drift_auc` trains a classifier to
distinguish dev rows from eval rows on a feature block; AUC >= 0.65 => the block DRIFTS
and the point is DISQUALIFIED (recorded, not silently dropped). This gate is
SEQUENTIAL: drift PASS first, THEN the holdout PairAP is trusted (Rule 17).

The load-bearing honest constraint (contract Rules 1, 9).
---------------------------------------------------------
The dev-holdout PairAP is a MEASURED LOCAL number feeding only the RELATIVE
rank-tracking gate. It is NOT and cannot be an LB claim (the eval pairs are disjoint
from the dev labels). The real LB coordinate of each point is EXTERNALLY VERIFIED from
our own Kaggle submission history; it is read from the artifact filename, never faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.validation.cv import build_fold_solution

from anchor_repro.artifact_names import ArtifactLB
from anchor_repro.ladder import RecipeRerun, RecipeRunner
from anchor_repro.recipe_registry import (
    FLOOR_DIGITS,
    CacheLocations,
    FloorRerunResult,
    MissingCacheError,
    _FLOOR_XGB_PARAMS,
    _load_pool_map,
    _PU_WEIGHT_EVAL_ROWS,
    _SPLIT_CUT_FRAC,
    _SPLIT_SEED,
    _V5_DROP_COLUMNS,
    build_floor_recipe_runner,
)

__all__ = [
    "V3_DIGITS",
    "V2_DIGITS",
    "V3_RECIPE_ID",
    "V2_RECIPE_ID",
    "EXP3C_DIGITS",
    "EXP3C_RECIPE_ID",
    "EXP3C_REAL_LB",
    "DRIFT_AUC_THRESHOLD",
    "OwnSubmissionRerunResult",
    "Exp3cCacheLocations",
    "Exp3cRerunResult",
    "DriftGateResult",
    "OwnCacheLocations",
    "default_own_cache_locations",
    "default_exp3c_cache_locations",
    "rerun_v3_ens_stack",
    "rerun_v2_single_stack",
    "rerun_v5_mfg_stack",
    "adversarial_drift_auc",
    "adversarial_drift_auc_matrix",
    "exp3c_drift_gate",
    "build_own_submission_runner",
    "build_exp3c_runner",
    "build_multi_point_ladder_runner",
]

# --------------------------------------------------------------------------- #
# The two own-submission ladder points (§49 candidate list).                   #
# --------------------------------------------------------------------------- #

#: cand_17 — V3 ensemble rank-avg(XGB d4 + d6) PU risk on cached v3 feats. Real LB
#: 0.37063 (EXTERNALLY VERIFIED from our own Kaggle submission history).
V3_DIGITS: str = "037063"
V3_RECIPE_ID: str = "own_cand17_v3_ens_037063"

#: cand_14 — V2 richer feats + PU (single XGB on cached v2 feats). Real LB 0.32580
#: (EXTERNALLY VERIFIED from our own Kaggle submission history).
V2_DIGITS: str = "032580"
V2_RECIPE_ID: str = "own_cand14_v2_pu_032580"

#: cand_exp3c — the floor stack MINUS the DIRc directional block: foundation v5 + MFg
#: graded board-equity, ONE XGB (dossier §49 candidate list; §51). Real LB 0.41289
#: (EXTERNALLY VERIFIED from our own Kaggle submission history). Its feature basis is a
#: STRICT SUBSET of the already-drift-clean floor basis (v5 + MFg + DIRc), so it is the
#: highest-probability clean expansion point: the floor passed its drift gates
#: historically, and dropping DIRc only REMOVES features from that clean basis.
EXP3C_DIGITS: str = "041289"
EXP3C_RECIPE_ID: str = "own_cand_exp3c_v5_mfg_041289"

#: The externally-verified real LB of cand_exp3c (Kaggle submission history). Recorded
#: as a named constant for the audit trail; the ladder still parses ``real_lb`` from
#: the artifact filename (never from this constant), so this is documentation only.
EXP3C_REAL_LB: float = 0.41289

#: Adversarial drift gate threshold (§49, contract Rules 5, 17). AUC >= this => the
#: feature block separates dev from eval too easily => it DRIFTS => the point is
#: DISQUALIFIED (recorded, never silently dropped).
DRIFT_AUC_THRESHOLD: float = 0.65

#: v3 XGB depths for the rank-average ensemble (cand_17). Everything else in the two
#: heads matches the floor's ``hap`` shape (verbatim reuse; only depth differs so the
#: ensemble is a genuine d4+d6 rank-average).
_V3_ENSEMBLE_DEPTHS: Tuple[int, int] = (4, 6)

#: Bookkeeping columns that are NOT features in the v3/v2 parquets. Only ``pair_id`` is
#: present in those caches (labels/pool keys come from ``dev_pairs_v2``), but we guard
#: the full set defensively so a re-versioned cache cannot leak a label into X.
_NON_FEATURE_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "p_low",
    "p_high",
    "label",
    "behavior_family",
)

#: The XGB head shape shared by both own-submission recipes, VERBATIM from the floor's
#: ``_FLOOR_XGB_PARAMS`` (recipe_registry) EXCEPT ``max_depth`` (set per head). Reused,
#: not re-tuned — the PU weighting + split are identical to the floor so the only
#: recipe difference from the floor is the feature block and the head count/depth.
_OWN_XGB_PARAMS_BASE: Dict[str, object] = dict(
    n_estimators=700,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=6,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


@dataclass(frozen=True)
class OwnCacheLocations:
    """Resolved on-disk paths for an own-submission recipe re-run (v3 or v2).

    Every path is a persisted artifact the recipe was built from — this adapter reads
    them verbatim and never recomputes a feature (contract Rule 6).

    Attributes:
        dev_feats: The dev feature parquet (``v3/dev_v3.parquet`` or
            ``v2/dev_v2.parquet``) — has ``pair_id`` + the feature columns.
        eval_feats: The eval feature parquet (``v3/eval_v3.parquet`` /
            ``v2/eval_v2.parquet``) — used ONLY by the adversarial drift gate.
        dev_pairs_v2: ``feature_cache/v2/dev_pairs_v2.parquet`` (pair_id, p_low,
            p_high, label — the PU label + pool-key source, exactly as the floor uses).
        seats: ``data/poker/seats.parquet`` (for the pool map).
        hands: ``data/poker/hands.parquet`` (for the pool map).
    """

    dev_feats: Path
    eval_feats: Path
    dev_pairs_v2: Path
    seats: Path
    hands: Path

    def missing(self) -> List[Path]:
        """Return the subset of required cache paths absent from disk."""
        return [
            p
            for p in (self.dev_feats, self.eval_feats, self.dev_pairs_v2, self.seats, self.hands)
            if not Path(p).is_file()
        ]


def default_own_cache_locations(poker_root: Path, version: str) -> OwnCacheLocations:
    """Resolve the standard own-submission cache locations for a feature version.

    Args:
        poker_root: The ``.../kAGGLE/poker`` directory.
        version: ``"v3"`` or ``"v2"``.

    Returns:
        An :class:`OwnCacheLocations` with verbatim on-disk paths.
    """
    root = Path(poker_root)
    fc = root / "outputs" / "poker_collusion" / "feature_cache"
    data = root / "data" / "poker"
    return OwnCacheLocations(
        dev_feats=fc / version / f"dev_{version}.parquet",
        eval_feats=fc / version / f"eval_{version}.parquet",
        dev_pairs_v2=fc / "v2" / "dev_pairs_v2.parquet",
        seats=data / "seats.parquet",
        hands=data / "hands.parquet",
    )


@dataclass(frozen=True)
class OwnSubmissionRerunResult:
    """Measured outcome of re-running an own-submission recipe on the dev holdout.

    Attributes:
        dev_predictions: The dev-holdout prediction frame (production-metric schema)
            restricted to the CONFIRMED holdout pairs, ready for the CanonicalScorer.
        n_holdout_all: Number of ALL dev pairs in the holdout split (incl. PU unknowns)
            — the row set the loopv-style weighted AP is computed over.
        n_holdout_confirmed: Number of CONFIRMED holdout pairs (the canonical PairAP
            row set).
        loopv_weighted_ap: The loopv-style ``average_precision_score`` over ALL holdout
            pairs with PU sample weights — a MEASURED local re-run figure, NOT an LB
            claim (contract Rules 1, 9).
        n_features: Number of feature columns used (audit).
        n_heads: Number of XGB heads rank-averaged (2 for v3, 1 for v2).
        recipe_id: Human-readable id of the own-submission recipe.
    """

    dev_predictions: pd.DataFrame
    n_holdout_all: int
    n_holdout_confirmed: int
    loopv_weighted_ap: float
    n_features: int
    n_heads: int
    recipe_id: str


@dataclass(frozen=True)
class DriftGateResult:
    """Outcome of the adversarial dev-vs-eval drift gate on a feature block.

    Attributes:
        block: Which block was gated (``"v3"`` / ``"v2"``).
        auc: The measured dev-vs-eval classifier AUC (MEASURED; contract Rule 9).
        threshold: The disqualification threshold (:data:`DRIFT_AUC_THRESHOLD`).
        passed: ``auc < threshold`` — ``True`` means the block does NOT drift and the
            point may enter the ladder; ``False`` means DISQUALIFIED.
        n_features: Number of feature columns the classifier saw (audit).
        n_dev: Dev rows. n_eval: Eval rows.
    """

    block: str
    auc: float
    threshold: float
    passed: bool
    n_features: int
    n_dev: int
    n_eval: int

    @property
    def verdict(self) -> str:
        """``"PASS"`` (does not drift) or ``"DISQUALIFIED"`` (drifts)."""
        return "PASS" if self.passed else "DISQUALIFIED"


def _rank01(scores: np.ndarray) -> np.ndarray:
    """Return average ranks of ``scores`` mapped to [0,1] (for rank-averaging heads).

    Ties share the mean rank (matches ``scipy.rankdata`` "average"), then the ranks are
    normalised to [0,1]. A single-element or constant array maps to all-0.5 (neutral).
    """
    arr = np.asarray(scores, dtype=float)
    n = arr.size
    if n < 2:
        return np.full(n, 0.5, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        avg = (i + j) / 2.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks / (n - 1)


def _load_split_and_weights(
    caches: OwnCacheLocations,
) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the dev feature block + PU labels + seed-7 60/40 table-disjoint split.

    Mirrors :func:`recipe_registry.rerun_floor_stack` steps 1-5 EXACTLY (verbatim
    reuse of the split constants, the pool map, and the PU sample weights), differing
    only in which feature parquet is the block. Returns everything the model head needs.

    Returns:
        ``(dev, feats, X, y, fw, tr, ho)`` where ``dev`` is the merged dev frame,
        ``feats`` the feature column names, ``X`` the float32 feature matrix, ``y`` the
        binary PU target, ``fw`` the fit sample weights, ``tr``/``ho`` the train/holdout
        boolean masks. (``sw`` PU-weight-for-AP is derived by the caller from ``y`` +
        the known mask, mirroring the floor.)
    """
    missing = caches.missing()
    if missing:
        raise MissingCacheError(
            "own-submission recipe cannot be reconstructed; missing cache file(s): "
            + ", ".join(str(p) for p in missing)
        )

    dev_feats = pd.read_parquet(caches.dev_feats)
    dp = pd.read_parquet(caches.dev_pairs_v2)[["pair_id", "p_low", "p_high", "label"]]
    dev = dev_feats.merge(dp, on="pair_id", how="left")

    feats = [c for c in dev.columns if c not in _NON_FEATURE_COLUMNS]
    X = dev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)

    # --- PU labels + sample weights (verbatim from loopv.py / the floor). ---
    lab = dev["label"].to_numpy()
    y = (lab == 1).astype(int)
    known = lab >= 0
    n_pu = int((lab == -1).sum())
    sw = np.where(known, 1.0, _PU_WEIGHT_EVAL_ROWS / max(n_pu, 1))
    fw = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35))

    # --- seed-7 60/40 table-disjoint split (verbatim from loopv.py / the floor). ---
    player_pool = _load_pool_map(caches.seats, caches.hands)
    dev_table = dev["p_low"].map(player_pool)
    tabs = sorted(set(str(t) for t in dev_table.fillna("NA")))
    rng = np.random.default_rng(_SPLIT_SEED)
    rng.shuffle(tabs)
    cut = int(len(tabs) * _SPLIT_CUT_FRAC)
    train_tabs = set(tabs[:cut])
    tr = dev_table.astype(str).isin(train_tabs).to_numpy()
    ho = ~tr
    return dev, feats, X, y, fw, sw, tr, ho


def _emit_confirmed_frame(
    dev: pd.DataFrame,
    ho: np.ndarray,
    ho_proba: np.ndarray,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Emit the CanonicalScorer-ready confirmed-holdout-pairs prediction frame.

    Byte-for-byte the same construction as :func:`recipe_registry.rerun_floor_stack`:
    restrict to CONFIRMED holdout pairs (the ones :func:`build_fold_solution`
    materialises — PU-correct), fill the neutral behavior/evidence heads (PairAP
    depends only on ``risk_score``), and align to the solution's pair set.
    """
    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    ho_pairs = dev["pair_id"].astype(str).to_numpy()[ho]
    keep = np.array([pid in confirmed_ids for pid in ho_pairs])
    pred_pairs = ho_pairs[keep]
    pred_scores = ho_proba[keep]

    solution = build_fold_solution(labels, list(pred_pairs), evidence=evidence)
    score_by_pair = dict(zip((str(p) for p in pred_pairs), (float(s) for s in pred_scores)))
    kept_pairs = [pid for pid in solution["pair_id"].astype(str) if pid in score_by_pair]

    preds = pd.DataFrame({"pair_id": kept_pairs})
    preds["risk_score"] = [score_by_pair[pid] for pid in kept_pairs]
    preds["risk_score"] = preds["risk_score"].astype(float).clip(0.0, 1.0)
    preds["predicted_behavior"] = "none"
    for col in SUBMISSION_COLUMNS:
        if col not in preds.columns:
            preds[col] = "NO_EVIDENCE"
    return preds[list(SUBMISSION_COLUMNS)]


def rerun_v3_ens_stack(
    caches: OwnCacheLocations,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame] = None,
) -> OwnSubmissionRerunResult:
    """Re-run cand_17 (v3 ensemble rank-avg XGB d4+d6 PU risk) on the dev holdout.

    Mirrors :func:`recipe_registry.rerun_floor_stack` EXACTLY for the split, PU sample
    weights, and canonical confirmed-only PairAP scoring. The ONLY differences: the
    feature block is the v3 cache, and the model head is the rank-AVERAGE of two XGBs
    (depth 4 + depth 6) rather than one XGB (this is what made cand_17 real LB 0.37063).

    Returns:
        An :class:`OwnSubmissionRerunResult`.

    Raises:
        MissingCacheError: If a required cache is absent (names the missing paths).
        ImportError: If ``xgboost`` is unavailable.
    """
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    dev, feats, X, y, fw, sw, tr, ho = _load_split_and_weights(caches)

    # Two XGB heads (depth 4 + depth 6), rank-averaged on the holdout (cand_17).
    ho_ranks = []
    for depth in _V3_ENSEMBLE_DEPTHS:
        params = dict(_OWN_XGB_PARAMS_BASE, max_depth=depth)
        model = XGBClassifier(**params)
        model.fit(X[tr], y[tr], sample_weight=fw[tr])
        ho_ranks.append(_rank01(model.predict_proba(X[ho])[:, 1]))
    ho_proba = np.mean(np.vstack(ho_ranks), axis=0)

    loopv_weighted_ap = float(
        average_precision_score(y[ho], ho_proba, sample_weight=sw[ho])
    )
    preds = _emit_confirmed_frame(dev, ho, ho_proba, labels, evidence=evidence)

    return OwnSubmissionRerunResult(
        dev_predictions=preds,
        n_holdout_all=int(ho.sum()),
        n_holdout_confirmed=int(len(preds)),
        loopv_weighted_ap=loopv_weighted_ap,
        n_features=int(X.shape[1]),
        n_heads=len(_V3_ENSEMBLE_DEPTHS),
        recipe_id=V3_RECIPE_ID,
    )


def rerun_v2_single_stack(
    caches: OwnCacheLocations,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame] = None,
) -> OwnSubmissionRerunResult:
    """Re-run cand_14 (v2 single-XGB PU risk) on the dev holdout.

    Mirrors :func:`recipe_registry.rerun_floor_stack` EXACTLY for the split, PU sample
    weights, and canonical confirmed-only PairAP scoring. The ONLY differences: the
    feature block is the v2 cache, and the model head is a SINGLE XGB (depth 4) — this
    is what cand_14 (real LB 0.32580) was.

    Returns:
        An :class:`OwnSubmissionRerunResult`.

    Raises:
        MissingCacheError: If a required cache is absent (names the missing paths).
        ImportError: If ``xgboost`` is unavailable.
    """
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    dev, feats, X, y, fw, sw, tr, ho = _load_split_and_weights(caches)

    model = XGBClassifier(**dict(_OWN_XGB_PARAMS_BASE, max_depth=4))
    model.fit(X[tr], y[tr], sample_weight=fw[tr])
    ho_proba = model.predict_proba(X[ho])[:, 1]

    loopv_weighted_ap = float(
        average_precision_score(y[ho], ho_proba, sample_weight=sw[ho])
    )
    preds = _emit_confirmed_frame(dev, ho, ho_proba, labels, evidence=evidence)

    return OwnSubmissionRerunResult(
        dev_predictions=preds,
        n_holdout_all=int(ho.sum()),
        n_holdout_confirmed=int(len(preds)),
        loopv_weighted_ap=loopv_weighted_ap,
        n_features=int(X.shape[1]),
        n_heads=1,
        recipe_id=V2_RECIPE_ID,
    )


# --------------------------------------------------------------------------- #
# Adversarial drift gate (contract Rules 5, 17; §49).                          #
# --------------------------------------------------------------------------- #


def adversarial_drift_auc(
    dev_feats_path: Path,
    eval_feats_path: Path,
    *,
    block: str,
    seed: int = 7,
) -> DriftGateResult:
    """Train a dev-vs-eval classifier on a feature block; report the separation AUC.

    Contract Rules 5 & 17: BEFORE a new point's dev-holdout PairAP is trusted as an LB
    predictor, its feature basis must not drift dev->eval. We label dev rows 0 / eval
    rows 1 and fit an XGB with grouped-free stratified CV, reporting the out-of-fold
    AUC. AUC >= :data:`DRIFT_AUC_THRESHOLD` => the block separates dev from eval too
    easily => it DRIFTS => the point is DISQUALIFIED (recorded, never silently dropped).

    Only the feature columns common to both frames (minus ``pair_id`` and label/pool
    bookkeeping) are used — the same columns the recipe fits on.

    Args:
        dev_feats_path: The dev feature parquet.
        eval_feats_path: The eval feature parquet.
        block: A label for the block (``"v3"`` / ``"v2"``), recorded on the result.
        seed: RNG seed for the stratified CV (fixed for reproducibility).

    Returns:
        A :class:`DriftGateResult` carrying the MEASURED AUC and the PASS/DISQUALIFIED
        verdict.

    Raises:
        MissingCacheError: If either parquet is absent.
    """
    for p in (dev_feats_path, eval_feats_path):
        if not Path(p).is_file():
            raise MissingCacheError(f"drift gate cache absent: {p}")

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    dev = pd.read_parquet(dev_feats_path)
    ev = pd.read_parquet(eval_feats_path)
    feats = [
        c
        for c in dev.columns
        if c in ev.columns and c not in _NON_FEATURE_COLUMNS
    ]

    Xd = dev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xe = ev[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X = np.vstack([Xd, Xe])
    y = np.concatenate([np.zeros(len(Xd), dtype=int), np.ones(len(Xe), dtype=int)])

    # Out-of-fold AUC (a single train/test AUC can overstate separation on imbalanced
    # sizes; OOF over 5 stratified folds is the honest, standard adversarial-validation
    # estimate).
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]

    auc = float(roc_auc_score(y, oof))
    return DriftGateResult(
        block=block,
        auc=auc,
        threshold=DRIFT_AUC_THRESHOLD,
        passed=auc < DRIFT_AUC_THRESHOLD,
        n_features=len(feats),
        n_dev=int(len(Xd)),
        n_eval=int(len(Xe)),
    )


# --------------------------------------------------------------------------- #
# cand_exp3c — the floor stack MINUS DIRc (foundation v5 + MFg, one XGB).       #
# §49 candidate list / §51. Its basis is a STRICT SUBSET of the drift-clean     #
# floor basis, so it is the highest-probability clean expansion point.          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exp3cCacheLocations:
    """Resolved on-disk paths for the cand_exp3c (v5 + MFg) dev-holdout re-run.

    This is the floor's :class:`recipe_registry.CacheLocations` MINUS the DIRc block:
    exp3c is exactly the floor stack with the directional block omitted. Every path is
    a persisted artifact ``loopv.py`` / the LB-verified stack was built from — this
    adapter reads them verbatim and never recomputes a feature (contract Rule 6).

    Attributes:
        dev_v5: ``feature_cache/v5/dev_v5.parquet`` (foundation ``feats5`` + pair_id).
        eval_v5: ``feature_cache/v5/eval_v5.parquet`` — used ONLY by the drift gate.
        dev_pairs_v2: ``feature_cache/v2/dev_pairs_v2.parquet`` (pair_id, p_low,
            p_high, label — the PU label + pool-key source, exactly as the floor uses).
        mfg_dev: ``feature_cache/v7/_MFg_dev.npy`` (graded board-equity, 10 cols).
        mfg_eval: ``feature_cache/v7/_MFg_eval.npy`` — used ONLY by the drift gate.
        seats: ``data/poker/seats.parquet`` (for the pool map).
        hands: ``data/poker/hands.parquet`` (for the pool map).
    """

    dev_v5: Path
    eval_v5: Path
    dev_pairs_v2: Path
    mfg_dev: Path
    mfg_eval: Path
    seats: Path
    hands: Path

    def missing(self) -> List[Path]:
        """Return the subset of required cache paths absent from disk."""
        return [
            p
            for p in (
                self.dev_v5,
                self.eval_v5,
                self.dev_pairs_v2,
                self.mfg_dev,
                self.mfg_eval,
                self.seats,
                self.hands,
            )
            if not Path(p).is_file()
        ]


def default_exp3c_cache_locations(poker_root: Path) -> Exp3cCacheLocations:
    """Resolve the standard cand_exp3c (v5 + MFg) cache locations under a ``poker/`` tree.

    Args:
        poker_root: The ``.../kAGGLE/poker`` directory.

    Returns:
        An :class:`Exp3cCacheLocations` with verbatim on-disk paths (the floor's caches
        minus DIRc, plus the eval-side v5/MFg the drift gate needs).
    """
    root = Path(poker_root)
    fc = root / "outputs" / "poker_collusion" / "feature_cache"
    data = root / "data" / "poker"
    return Exp3cCacheLocations(
        dev_v5=fc / "v5" / "dev_v5.parquet",
        eval_v5=fc / "v5" / "eval_v5.parquet",
        dev_pairs_v2=fc / "v2" / "dev_pairs_v2.parquet",
        mfg_dev=fc / "v7" / "_MFg_dev.npy",
        mfg_eval=fc / "v7" / "_MFg_eval.npy",
        seats=data / "seats.parquet",
        hands=data / "hands.parquet",
    )


@dataclass(frozen=True)
class Exp3cRerunResult:
    """Measured outcome of re-running cand_exp3c (v5 + MFg, one XGB) on the dev holdout.

    Identical in shape to :class:`OwnSubmissionRerunResult` (kept distinct so the
    exp3c-specific ``n_features`` audit — 59 v5 + 10 MFg = 69 — is unambiguous).

    Attributes:
        dev_predictions: The dev-holdout prediction frame (production-metric schema)
            restricted to the CONFIRMED holdout pairs, ready for the CanonicalScorer.
        n_holdout_all: Number of ALL dev pairs in the holdout split (incl. PU unknowns)
            — the row set the loopv-style weighted AP is computed over.
        n_holdout_confirmed: Number of CONFIRMED holdout pairs (the canonical PairAP
            row set).
        loopv_weighted_ap: The loopv-style ``average_precision_score`` over ALL holdout
            pairs with PU sample weights — a MEASURED local re-run figure, NOT an LB
            claim (contract Rules 1, 9).
        n_features: Number of feature columns used (59 v5 foundation + 10 MFg = 69).
        recipe_id: Human-readable id of the exp3c recipe.
    """

    dev_predictions: pd.DataFrame
    n_holdout_all: int
    n_holdout_confirmed: int
    loopv_weighted_ap: float
    n_features: int
    recipe_id: str


def adversarial_drift_auc_matrix(
    Xd: np.ndarray,
    Xe: np.ndarray,
    *,
    block: str,
    seed: int = 7,
) -> DriftGateResult:
    """Train a dev-vs-eval classifier on ALREADY-ASSEMBLED matrices; report the AUC.

    The matrix twin of :func:`adversarial_drift_auc`. The parquet-reading gate cannot
    express the exp3c basis (v5 parquet + MFg .npy hstacked), so the caller assembles
    ``Xd`` (dev) / ``Xe`` (eval) exactly as :func:`rerun_v5_mfg_stack` assembles its
    training matrix, and passes them here. The classifier + CV + threshold are
    IDENTICAL to :func:`adversarial_drift_auc` (5-fold stratified OOF AUC, threshold
    :data:`DRIFT_AUC_THRESHOLD`) so the two gates are the same test on different inputs
    (contract Rules 5, 17).

    Args:
        Xd: Dev feature matrix (rows = dev pairs, cols = the exp3c basis).
        Xe: Eval feature matrix (rows = eval pairs, SAME cols in the SAME order).
        block: A label for the block (e.g. ``"v5+MFg"``), recorded on the result.
        seed: RNG seed for the stratified CV (fixed for reproducibility).

    Returns:
        A :class:`DriftGateResult` with the MEASURED AUC and PASS/DISQUALIFIED verdict.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    Xd = np.asarray(Xd, dtype=np.float32)
    Xe = np.asarray(Xe, dtype=np.float32)
    if Xd.shape[1] != Xe.shape[1]:
        raise ValueError(
            f"drift gate matrices must share column count; got dev {Xd.shape[1]} vs "
            f"eval {Xe.shape[1]} (refusing to fabricate an alignment; Rules 1, 9)"
        )
    X = np.vstack([Xd, Xe])
    y = np.concatenate([np.zeros(len(Xd), dtype=int), np.ones(len(Xe), dtype=int)])

    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]

    auc = float(roc_auc_score(y, oof))
    return DriftGateResult(
        block=block,
        auc=auc,
        threshold=DRIFT_AUC_THRESHOLD,
        passed=auc < DRIFT_AUC_THRESHOLD,
        n_features=int(Xd.shape[1]),
        n_dev=int(len(Xd)),
        n_eval=int(len(Xe)),
    )


def _load_exp3c_feature_matrices(
    caches: Exp3cCacheLocations,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble the dev + eval exp3c feature matrices (v5 foundation hstack MFg).

    Reused by both the drift gate and the re-run so the gated basis is byte-identical
    to the fitted basis (contract Rules 5, 6, 17). The v5 foundation columns are the
    dev_v5 columns minus the id/pool/label bookkeeping (:data:`_V5_DROP_COLUMNS`,
    verbatim from the floor); only columns present in BOTH dev and eval are used so the
    two matrices share their column set exactly. MFg is appended in its saved order.

    Returns:
        ``(Xd, Xe, feats5)`` — dev matrix, eval matrix (aligned columns), and the v5
        foundation feature names (MFg columns are unnamed .npy channels appended after).
    """
    dev_v5 = pd.read_parquet(caches.dev_v5)
    eval_v5 = pd.read_parquet(caches.eval_v5)
    feats5 = [
        c
        for c in dev_v5.columns
        if c in eval_v5.columns and c not in _V5_DROP_COLUMNS
    ]
    Xd5 = dev_v5[feats5].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xe5 = eval_v5[feats5].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Mgd = np.load(caches.mfg_dev)
    Mge = np.load(caches.mfg_eval)
    if len(Xd5) != len(Mgd) or len(Xe5) != len(Mge):
        raise MissingCacheError(
            "exp3c cache row-count mismatch (dev_v5="
            f"{len(Xd5)}/MFg_dev={len(Mgd)}, eval_v5={len(Xe5)}/MFg_eval={len(Mge)}); "
            "the caches must be the aligned matrices the stack hstacks — refusing to "
            "fabricate an alignment (contract Rules 1, 9)."
        )
    if Mgd.shape[1] != Mge.shape[1]:
        raise MissingCacheError(
            f"exp3c MFg column mismatch (dev {Mgd.shape[1]} vs eval {Mge.shape[1]})"
        )
    Xd = np.hstack([Xd5, Mgd]).astype(np.float32)
    Xe = np.hstack([Xe5, Mge]).astype(np.float32)
    return Xd, Xe, feats5


def exp3c_drift_gate(caches: Exp3cCacheLocations, *, seed: int = 7) -> DriftGateResult:
    """Run the adversarial dev-vs-eval drift gate on the exp3c (v5 + MFg) basis.

    Contract Rules 5 & 17: BEFORE cand_exp3c's dev-holdout PairAP is trusted as an LB
    predictor, its feature basis must not drift dev->eval. This assembles the exact
    v5+MFg hstack the re-run fits on (:func:`_load_exp3c_feature_matrices`) and runs the
    matrix drift gate (:func:`adversarial_drift_auc_matrix`). AUC >=
    :data:`DRIFT_AUC_THRESHOLD` => DISQUALIFIED (recorded, never silently dropped).

    Raises:
        MissingCacheError: If any required cache is absent (names the missing paths).
    """
    missing = caches.missing()
    if missing:
        raise MissingCacheError(
            "exp3c drift gate cannot run; missing cache file(s): "
            + ", ".join(str(p) for p in missing)
        )
    Xd, Xe, feats5 = _load_exp3c_feature_matrices(caches)
    return adversarial_drift_auc_matrix(Xd, Xe, block="v5+MFg", seed=seed)


def rerun_v5_mfg_stack(
    caches: Exp3cCacheLocations,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame] = None,
) -> Exp3cRerunResult:
    """Re-run cand_exp3c (foundation v5 + MFg, ONE XGB) on the dev holdout.

    This is :func:`recipe_registry.rerun_floor_stack` with the DIRc directional block
    OMITTED — nothing else changes. Verbatim reuse of the floor's:

    * ``feats5`` foundation block (dev_v5 columns minus :data:`_V5_DROP_COLUMNS`),
    * MFg graded board-equity block (``_MFg_dev.npy``),
    * ``Xfound = hstack([Xd5, Mgd])`` — ONE XGB (the floor's ``_FLOOR_XGB_PARAMS``),
    * seed-7 60/40 table-disjoint split, PU sample weights (pos 2.0 / confirmed 1.0 /
      PU 0.35, ``sw`` unknowns rescaled ``112540/n_pu``),
    * canonical confirmed-only PairAP scoring + the loopv-style weighted-all AP.

    Because the ONLY change from the floor is dropping DIRc, exp3c's feature basis is a
    strict subset of the floor's already-drift-clean basis (contract Rule 4 — no
    extrapolation: we stay inside the regime the floor was validated on).

    Returns:
        An :class:`Exp3cRerunResult`.

    Raises:
        MissingCacheError: If a required cache is absent (names the missing paths).
        ImportError: If ``xgboost`` is unavailable.
    """
    missing = caches.missing()
    if missing:
        raise MissingCacheError(
            "exp3c recipe cannot be reconstructed; missing cache file(s): "
            + ", ".join(str(p) for p in missing)
        )

    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    # --- load the persisted stack verbatim (dev_v5 order authoritative; MFg saved in
    # that order — exactly loopv's hstack with no reindex). Merge PU keys on pair_id. ---
    dev_v5 = pd.read_parquet(caches.dev_v5)
    dp = pd.read_parquet(caches.dev_pairs_v2)[["pair_id", "p_low", "p_high", "label"]]
    dev = dev_v5.merge(dp, on="pair_id", how="left")

    feats5 = [c for c in dev.columns if c not in _V5_DROP_COLUMNS]
    Xd5 = dev[feats5].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Mgd = np.load(caches.mfg_dev)
    if not (len(Xd5) == len(Mgd) == len(dev)):
        raise MissingCacheError(
            "exp3c cache row-count mismatch (dev_v5="
            f"{len(Xd5)}, MFg={len(Mgd)}); the caches must be the aligned dev-row "
            "matrices loopv.py hstacks — refusing to fabricate an alignment "
            "(contract Rules 1, 9)."
        )
    Xfound = np.hstack([Xd5, Mgd]).astype(np.float32)  # v5 + MFg, DIRc OMITTED

    # --- PU labels + sample weights (verbatim from loopv.py / the floor). ---
    lab = dev["label"].to_numpy()
    y = (lab == 1).astype(int)
    known = lab >= 0
    n_pu = int((lab == -1).sum())
    sw = np.where(known, 1.0, _PU_WEIGHT_EVAL_ROWS / max(n_pu, 1))
    fw = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35))

    # --- seed-7 60/40 table-disjoint split (verbatim from loopv.py / the floor). ---
    player_pool = _load_pool_map(caches.seats, caches.hands)
    dev_table = dev["p_low"].map(player_pool)
    tabs = sorted(set(str(t) for t in dev_table.fillna("NA")))
    rng = np.random.default_rng(_SPLIT_SEED)
    rng.shuffle(tabs)
    cut = int(len(tabs) * _SPLIT_CUT_FRAC)
    train_tabs = set(tabs[:cut])
    tr = dev_table.astype(str).isin(train_tabs).to_numpy()
    ho = ~tr

    # --- fit ONE XGB on the train split, predict the holdout (the floor's hap). ---
    model = XGBClassifier(**_FLOOR_XGB_PARAMS)
    model.fit(Xfound[tr], y[tr], sample_weight=fw[tr])
    ho_proba = model.predict_proba(Xfound[ho])[:, 1]

    loopv_weighted_ap = float(
        average_precision_score(y[ho], ho_proba, sample_weight=sw[ho])
    )
    preds = _emit_confirmed_frame(dev, ho, ho_proba, labels, evidence=evidence)

    return Exp3cRerunResult(
        dev_predictions=preds,
        n_holdout_all=int(ho.sum()),
        n_holdout_confirmed=int(len(preds)),
        loopv_weighted_ap=loopv_weighted_ap,
        n_features=int(Xfound.shape[1]),
        recipe_id=EXP3C_RECIPE_ID,
    )


def build_exp3c_runner(
    caches: Exp3cCacheLocations,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    evidence: Optional[pd.DataFrame] = None,
    on_result: Optional[Callable[[Exp3cRerunResult], None]] = None,
) -> RecipeRunner:
    """Build a :data:`anchor_repro.ladder.RecipeRunner` for the cand_exp3c point.

    The returned callback re-runs the exp3c (v5 + MFg, one XGB) recipe on the dev
    holdout for ``submission_best_041289.csv`` and returns a :class:`RecipeRerun` under
    the canonical recipe id; for every OTHER artifact it returns ``None`` (=> honest
    BLOCKED). The re-run is cached so scoring the ladder repeatedly does not refit.

    NOTE (contract Rules 5, 17): this factory does NOT itself run the drift gate — the
    caller runs :func:`exp3c_drift_gate` FIRST and only builds this runner (and passes
    it to :func:`build_multi_point_ladder_runner`) when the gate PASSES. A DISQUALIFIED
    exp3c is simply never wired onto the ladder.

    Args:
        caches: Resolved exp3c cache paths.
        labels: Confirmed dev-label frame (for the PU-correct scoring target).
        canonical_recipe_id: The scorer's ``recipe.recipe_id`` — recorded on the
            RecipeRerun so the ladder's cross-regime guard passes (Req 2.6).
        evidence: Optional dev-evidence frame.
        on_result: Optional one-shot callback receiving the Exp3cRerunResult.
    """
    cache: Dict[str, RecipeRerun] = {}

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if artifact.digits != EXP3C_DIGITS:
            return None
        if EXP3C_DIGITS not in cache:
            result = rerun_v5_mfg_stack(caches, labels, evidence=evidence)
            if on_result is not None:
                on_result(result)
            detail = (
                f"re-ran {result.recipe_id} (floor stack MINUS DIRc: foundation v5 + "
                f"MFg board-equity, {result.n_features} feats, one XGB, seed-7 60/40 "
                f"table-disjoint) on the dev holdout; canonical PairAP over "
                f"{result.n_holdout_confirmed} confirmed holdout pairs; loopv-style "
                f"weighted-all AP over {result.n_holdout_all} pairs = "
                f"{result.loopv_weighted_ap:.4f} (MEASURED local re-run figure, NOT an "
                "LB claim — contract Rules 1, 9)"
            )
            cache[EXP3C_DIGITS] = RecipeRerun(
                dev_predictions=result.dev_predictions,
                recipe_id=canonical_recipe_id,
                detail=detail,
            )
        return cache[EXP3C_DIGITS]

    return runner


# --------------------------------------------------------------------------- #
# The own-submission RecipeRunner factory + multi-point ladder runner.         #
# --------------------------------------------------------------------------- #


def build_own_submission_runner(
    digits: str,
    caches: OwnCacheLocations,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    evidence: Optional[pd.DataFrame] = None,
    on_result: Optional[Callable[[OwnSubmissionRerunResult], None]] = None,
) -> RecipeRunner:
    """Build a :data:`anchor_repro.ladder.RecipeRunner` for ONE own-submission point.

    The returned callback re-runs the own-submission recipe on the dev holdout for the
    matching artifact digits (``037063`` => v3 ensemble; ``032580`` => v2 single) and
    returns a :class:`RecipeRerun` under the canonical recipe id; for every OTHER
    artifact it returns ``None`` (=> honest BLOCKED). The re-run is cached so scoring
    the ladder repeatedly does not refit.

    Args:
        digits: :data:`V3_DIGITS` or :data:`V2_DIGITS`.
        caches: Resolved own-submission cache paths for that version.
        labels: Confirmed dev-label frame (for the PU-correct scoring target).
        canonical_recipe_id: The scorer's ``recipe.recipe_id`` — recorded on the
            RecipeRerun so the ladder's cross-regime guard passes (Req 2.6).
        evidence: Optional dev-evidence frame.
        on_result: Optional one-shot callback receiving the OwnSubmissionRerunResult.
    """
    if digits == V3_DIGITS:
        rerun_fn = rerun_v3_ens_stack
        head_desc = "rank-avg(XGB d4 + XGB d6) PU risk on v3 feats"
    elif digits == V2_DIGITS:
        rerun_fn = rerun_v2_single_stack
        head_desc = "single XGB PU risk on v2 feats"
    else:
        raise ValueError(f"unknown own-submission digits {digits!r}")

    cache: Dict[str, RecipeRerun] = {}

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if artifact.digits != digits:
            return None
        if digits not in cache:
            result = rerun_fn(caches, labels, evidence=evidence)
            if on_result is not None:
                on_result(result)
            detail = (
                f"re-ran {result.recipe_id} ({head_desc}, {result.n_features} feats, "
                f"{result.n_heads} head(s), seed-7 60/40 table-disjoint) on the dev "
                f"holdout; canonical PairAP over {result.n_holdout_confirmed} confirmed "
                f"holdout pairs; loopv-style weighted-all AP over {result.n_holdout_all} "
                f"pairs = {result.loopv_weighted_ap:.4f} (MEASURED local re-run figure, "
                "NOT an LB claim — contract Rules 1, 9)"
            )
            cache[digits] = RecipeRerun(
                dev_predictions=result.dev_predictions,
                recipe_id=canonical_recipe_id,
                detail=detail,
            )
        return cache[digits]

    return runner


def build_multi_point_ladder_runner(
    floor_caches: CacheLocations,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    competitor_runner: Optional[RecipeRunner] = None,
    own_runners: Optional[List[RecipeRunner]] = None,
    evidence: Optional[pd.DataFrame] = None,
    on_floor_result: Optional[Callable[[FloorRerunResult], None]] = None,
) -> RecipeRunner:
    """Compose floor + competitor + each drift-PASSING own point into ONE runner.

    Every point is scored under the SAME ``canonical_recipe_id`` so they are mutually
    comparable and the ladder's cross-regime guard passes (Req 2.6). The floor runner
    is built here (reusing :func:`recipe_registry.build_floor_recipe_runner`); the
    competitor runner and the own-submission runners are passed in already-built (the
    caller decides which own points cleared the drift gate — a DISQUALIFIED point is
    simply not passed in, so it never enters the ladder; §49, contract Rules 5, 17).

    Args:
        floor_caches: The floor-recipe feature caches.
        labels: Confirmed dev-label frame.
        canonical_recipe_id: The scorer's ``recipe.recipe_id``.
        competitor_runner: The already-built competitor runner (or ``None`` to omit).
        own_runners: The already-built own-submission runners that cleared the drift
            gate (or ``None``/empty to omit all).
        evidence: Optional dev-evidence frame.
        on_floor_result: Optional one-shot callback for the FloorRerunResult.

    Returns:
        A ``RecipeRunner`` that dispatches each artifact to the first sub-runner that
        answers (returns a RecipeRerun), else ``None`` (=> honest BLOCKED).
    """
    floor_runner = build_floor_recipe_runner(
        floor_caches,
        labels,
        canonical_recipe_id=canonical_recipe_id,
        evidence=evidence,
        on_result=on_floor_result,
    )

    sub_runners: List[RecipeRunner] = [floor_runner]
    if competitor_runner is not None:
        sub_runners.append(competitor_runner)
    if own_runners:
        sub_runners.extend(own_runners)

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        for sub in sub_runners:
            rerun = sub(path, artifact)
            if rerun is not None:
                return rerun
        return None  # honest BLOCKED for any artifact no sub-runner reconstructs

    return runner
