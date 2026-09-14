"""Nomannic 3rd-ladder-point candidate: drift gate FIRST (contract Rule 17).

Pre-registered bar: RESEARCH_DOSSIER §49 (adversarial dev-vs-eval AUC < 0.65 REQUIRED
before a point's dev-holdout PairAP is trusted as an LB predictor). Prior results: §50
(v2/v3 drift), §51 (exp3c drift; the pervasive dev-vs-eval separation is RAW CO-HAND-
COUNT scale in the visible v5 foundation — `shared_hands_calc`, dev pairs share ~120
hands vs eval ~86). §51 NEXT named nomannic (real LB 0.56202, self-earned) as a
candidate that MIGHT sidestep the shared foundation because it is a full end-to-end
reproduction with its OWN feature build.

This module reproduces the nomannic notebook's OWN ``aggregate_pair_features`` (cell 34,
verbatim logic — contract Rule 6) on the warm ``repro_nomannic/prepared_v2`` cache to
assemble the exact pair feature matrix the recipe fits on, for BOTH dev and eval, aligns
to the recipe's numeric ``feature_cols`` selection, and runs the SAME adversarial drift
gate §50/§51 used (:func:`own_submission_recipes.adversarial_drift_auc_matrix`, 5-fold
stratified OOF AUC, threshold 0.65). Drift PASS is a PREREQUISITE (Rule 17); only then
is the dev-holdout PairAP a trustworthy LB predictor.

The load-bearing honesty (Rules 1, 9): the nomannic ``submission.csv`` is an EVAL
submission (112,540 eval pairs disjoint from the dev labels); its real LB 0.56202 is
EXTERNALLY VERIFIED (self-earned on Kaggle). The dev-holdout PairAP (only computed if
the gate PASSES) would be a MEASURED LOCAL number, never an LB claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from anchor_repro.own_submission_recipes import DriftGateResult, adversarial_drift_auc_matrix
from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.validation.cv import build_fold_solution

__all__ = [
    "NOMANNIC_DIGITS",
    "NOMANNIC_RECIPE_ID",
    "NOMANNIC_REAL_LB",
    "NomannicCacheLocations",
    "default_nomannic_cache_locations",
    "build_nomannic_pair_matrices",
    "nomannic_drift_gate",
    "NomannicRerunResult",
    "rerun_nomannic_recipe",
    "build_nomannic_recipe_runner",
]

#: The externally-verified real LB of the nomannic reproduction (self-earned, this
#: session). Recorded for the audit trail; the ladder still parses ``real_lb`` from the
#: artifact filename, so this is documentation only.
NOMANNIC_DIGITS: str = "056202"
NOMANNIC_RECIPE_ID: str = "nomannic_pu_evidence_ranker_056202"
NOMANNIC_REAL_LB: float = 0.56202

#: Key/meta columns that are NEVER model features (verbatim from the notebook cell 34).
_SKIP_COLS = {
    "pair_id", "hand_id", "player_1", "player_2", "p_low", "p_high", "table_id",
    "evidence_rank", "is_evidence", "hand_order", "pair_hand_no",
}

#: Feature-selection excludes at the pair level (verbatim from the notebook cell 35).
_PAIR_EXCLUDE = {"pair_id", "player_1", "player_2", "cv_group", "is_known"}


class NomannicCacheLocations:
    """Resolved on-disk paths for the nomannic dev-holdout re-run / drift gate.

    Every path is a persisted artifact the reproduction wrote — this adapter reads them
    verbatim and rebuilds only the pair-aggregation the notebook's own code defines
    (contract Rule 6).
    """

    def __init__(self, prepared_dir: Path, players: Path) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.players = Path(players)
        self.dev_hf = self.prepared_dir / "dev_hand_features.parquet"
        self.eval_hf = self.prepared_dir / "eval_hand_features.parquet"
        self.dev_pairs = self.prepared_dir / "dev_pairs.parquet"
        self.eval_pairs = self.prepared_dir / "eval_pairs.parquet"

    def missing(self) -> List[Path]:
        return [
            p for p in (self.dev_hf, self.eval_hf, self.dev_pairs, self.eval_pairs, self.players)
            if not Path(p).is_file()
        ]


def default_nomannic_cache_locations(poker_root: Path) -> NomannicCacheLocations:
    """Resolve the standard nomannic cache locations under a ``poker/`` tree root."""
    root = Path(poker_root)
    return NomannicCacheLocations(
        prepared_dir=root / "outputs" / "poker_collusion" / "repro_nomannic" / "prepared_v2",
        players=root / "data" / "poker" / "players.parquet",
    )


def _hand_features(hf_schema) -> list:
    return [
        c for c, dtype in hf_schema.items()
        if dtype.is_numeric() and c not in _SKIP_COLS
        and "label" not in c and "behavior_id" not in c
    ]


def _aggregate_pair_features(
    hf_path: Path, pairs: pl.DataFrame, players_path: Path,
    hand_features, tail_features, score_features, score_q95,
) -> pl.DataFrame:
    """Verbatim reproduction of the nomannic ``aggregate_pair_features`` (cell 34)."""
    lf = pl.scan_parquet(hf_path)
    agg = [pl.len().alias("shared_hands_calc")]
    agg += [pl.col(c).mean().alias(f"{c}_mean") for c in hand_features]
    agg += [pl.col(c).max().alias(f"{c}_max") for c in hand_features]
    agg += [pl.col(c).quantile(.95, interpolation="nearest").alias(f"{c}_p95") for c in hand_features]
    agg += [pl.col(c).top_k(5).mean().alias(f"{c}_top5") for c in tail_features]
    agg += [(pl.col(c) >= score_q95[c]).mean().alias(f"{c}_rate95") for c in score_features]
    pair_stats = lf.group_by("pair_id").agg(agg)

    players_lf = pl.scan_parquet(players_path)
    meta_cols = ["player_id", "account_age_days", "experience_hands_bucket",
                 "preferred_stake", "region_bucket", "client_family"]
    meta = players_lf.select(meta_cols)
    m1 = meta.rename({c: ("player_1" if c == "player_id" else f"{c}_1") for c in meta_cols})
    m2 = meta.rename({c: ("player_2" if c == "player_id" else f"{c}_2") for c in meta_cols})
    pair_meta = (
        pairs.lazy().select(["pair_id", "player_1", "player_2"])
        .join(m1, on="player_1", how="left").join(m2, on="player_2", how="left")
        .select(
            "pair_id",
            (pl.col("account_age_days_1").cast(pl.Float32) - pl.col("account_age_days_2").cast(pl.Float32)).abs().alias("account_age_gap"),
            (pl.col("experience_hands_bucket_1") == pl.col("experience_hands_bucket_2")).cast(pl.Int8).alias("same_experience"),
            (pl.col("preferred_stake_1") == pl.col("preferred_stake_2")).cast(pl.Int8).alias("same_stake"),
            (pl.col("region_bucket_1") == pl.col("region_bucket_2")).cast(pl.Int8).alias("same_region"),
            (pl.col("client_family_1") == pl.col("client_family_2")).cast(pl.Int8).alias("same_client"),
        )
    )
    base_cols = [c for c in ["pair_id", "player_1", "player_2", "shared_hands"] if c in pairs.columns]
    return (
        pairs.lazy().select(base_cols)
        .join(pair_stats, on="pair_id", how="left")
        .join(pair_meta, on="pair_id", how="left")
        .collect(engine="streaming")
    )


def build_nomannic_pair_matrices(
    caches: NomannicCacheLocations,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """Assemble the nomannic dev + eval pair feature matrices (the recipe's basis).

    Returns ``(Xd, Xe, feature_cols, dev_pd, eval_pd)`` — the aligned dev/eval float32
    matrices, the shared numeric feature columns (exactly the recipe's ``feature_cols``
    selection), and the pandas frames (for the mechanism probe).

    Raises:
        FileNotFoundError: If any required cache is absent (names the missing paths).
    """
    missing = caches.missing()
    if missing:
        raise FileNotFoundError(
            "nomannic drift gate cannot run; missing cache file(s): "
            + ", ".join(str(p) for p in missing)
        )

    hf_schema = pl.scan_parquet(caches.dev_hf).collect_schema()
    hand_features = _hand_features(hf_schema)
    score_features = [c for c in hand_features if c.endswith("_score")]
    tail_tokens = ("score", "pot_bb", "contribution", "net_gap", "amount", "raise",
                   "heads_up", "partner", "pressure")
    tail_features = list(dict.fromkeys(
        score_features + [c for c in hand_features if any(x in c for x in tail_tokens)]
    ))[:28]
    score_q95 = (
        pl.scan_parquet(caches.dev_hf).select([pl.col(c).quantile(.95).alias(c) for c in score_features])
        .collect().row(0, named=True) if score_features else {}
    )

    dev_pairs = pl.read_parquet(caches.dev_pairs)
    eval_pairs = pl.read_parquet(caches.eval_pairs)

    dev_feat = _aggregate_pair_features(
        caches.dev_hf, dev_pairs, caches.players, hand_features, tail_features, score_features, score_q95)
    eval_feat = _aggregate_pair_features(
        caches.eval_hf, eval_pairs, caches.players, hand_features, tail_features, score_features, score_q95)

    dev_pd = dev_feat.to_pandas()
    eval_pd = eval_feat.to_pandas()
    dev_num = [c for c in dev_pd.columns if c not in _PAIR_EXCLUDE and pd.api.types.is_numeric_dtype(dev_pd[c])]
    feature_cols = [c for c in dev_num if c in eval_pd.columns]

    Xd = dev_pd[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xe = eval_pd[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    return Xd, Xe, feature_cols, dev_pd, eval_pd


def nomannic_drift_gate(caches: NomannicCacheLocations, *, seed: int = 7) -> DriftGateResult:
    """Run the adversarial dev-vs-eval drift gate on the nomannic pair basis.

    Contract Rules 5 & 17: BEFORE nomannic's dev-holdout PairAP is trusted as an LB
    predictor, its feature basis must not drift dev->eval. Assembles the exact pair
    matrix the recipe fits on and runs the shared matrix drift gate. AUC >= 0.65 =>
    DISQUALIFIED (recorded, never silently dropped).
    """
    Xd, Xe, _feats, _dev, _ev = build_nomannic_pair_matrices(caches)
    return adversarial_drift_auc_matrix(Xd, Xe, block="nomannic_pair", seed=seed)


# --------------------------------------------------------------------------- #
# §54/§55 — nomannic dev-holdout OOF PairAP (the DRIFT-FAILING basis scored     #
# ANYWAY to test contract Rule 17). Rule 17 blocks this in the normal §52 path; #
# §54 EXPLICITLY authorizes computing it here to measure whether a drift-failing #
# basis still rank-tracks the LB. This is the pre-registered experiment.        #
# --------------------------------------------------------------------------- #

from dataclasses import dataclass  # noqa: E402  (grouped with the §55 additions)
from typing import Callable, Dict, Optional  # noqa: E402

from anchor_repro.artifact_names import ArtifactLB  # noqa: E402
from anchor_repro.ladder import RecipeRerun, RecipeRunner  # noqa: E402

#: The notebook's EXACT risk-head hyperparameters (cell 35, verbatim — contract
#: Rule 6). Reused, not re-tuned; ``early_stopping_rounds`` is honoured via the
#: per-fold ``eval_set`` exactly as the notebook does.
_NOMANNIC_RISK_PARAMS: Dict[str, object] = dict(
    n_estimators=1400,
    learning_rate=0.025,
    max_depth=5,
    min_child_weight=8,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.15,
    reg_lambda=6,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    early_stopping_rounds=100,
    n_jobs=-1,
)

#: The notebook's CV seed (cell 34/35: ``SEED = 42``) and fold count.
_NOMANNIC_SEED: int = 42
_NOMANNIC_N_SPLITS: int = 5

_NOMANNIC_OOF_FILENAME: str = "dev_holdout_oof_056202.parquet"


@dataclass(frozen=True)
class NomannicRerunResult:
    """Measured outcome of re-running the nomannic risk head on the dev holdout.

    Attributes:
        dev_predictions: CanonicalScorer-ready confirmed-holdout-pairs frame.
        n_confirmed: Number of confirmed dev pairs carried into the scored frame.
        n_dev_all: Number of ALL dev pairs the OOF was produced over (incl. PU).
        labeled_ap: The notebook-style labeled OOF AP over the confirmed pairs
            (``average_precision_score(y[known], oof_risk[known])``) — a MEASURED local
            re-run consistency figure, NOT an LB claim (contract Rules 1, 9).
        n_features: Number of pair-level features in the reconstructed matrix (audit).
        reused_cache: ``True`` if a prior on-disk OOF cache was reused (no refit).
    """

    dev_predictions: pd.DataFrame
    n_confirmed: int
    n_dev_all: int
    labeled_ap: float
    n_features: int
    reused_cache: bool


def rerun_nomannic_recipe(
    caches: NomannicCacheLocations,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> NomannicRerunResult:
    """Re-run the nomannic risk head on the dev holdout → a scorable prediction frame.

    §54-authorized (overrides the §52 Rule-17 block for THIS pre-registered test):
    reproduces the notebook's OWN 5-fold ``StratifiedGroupKFold`` (grouped by the
    pair's most-frequent ``table_id``, exactly cell 35's ``cv_group``) OOF risk over
    the 25,860 dev pairs, keeps the CONFIRMED pairs' OOF risk, and emits a
    CanonicalScorer-ready frame (contract Rule 6 — reuses
    :func:`build_nomannic_pair_matrices` for the feature assembly and the notebook's
    verbatim ``risk_params`` / PU fit-weights). CPU only, ``tree_method='hist'``,
    ``n_jobs=-1``.

    The OOF cache is written to the prepared dir and REUSED on a subsequent call
    (unless ``force=True``), so scoring the ladder repeatedly does not refit.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    Xd, _Xe, feature_cols, dev_pd, _eval_pd = build_nomannic_pair_matrices(caches)

    # PU labels + fit weights (cell 35, verbatim). dev_pairs.parquet carries the PU
    # bookkeeping (label / is_labeled / behavior_id) aligned by pair_id to dev_pd rows.
    dev_pairs = pl.read_parquet(caches.dev_pairs).to_pandas()
    order = dev_pd["pair_id"].astype(str).tolist()
    dp_by_id = dev_pairs.set_index(dev_pairs["pair_id"].astype(str))
    dp_aligned = dp_by_id.reindex(order)

    behavior_id = dp_aligned["behavior_id"].fillna(0).to_numpy().astype(np.int8)
    y = (behavior_id > 0).astype(np.int8)
    known = dp_aligned["is_labeled"].fillna(False).to_numpy().astype(bool)
    fit_weight = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35)).astype(np.float32)

    # cv_group = the pair's most-frequent table_id (cell 34's pair_groups). dev_pairs
    # carries table_id per pair; reuse it directly (verbatim group key).
    groups = dp_aligned["table_id"].fillna("NA").astype(str).to_numpy()

    oof_path = caches.prepared_dir / _NOMANNIC_OOF_FILENAME
    reused = False
    if oof_path.is_file() and not force:
        oof_df = pd.read_parquet(oof_path)
        oof_by_id = dict(zip(oof_df["pair_id"].astype(str), oof_df["oof_risk"].astype(float)))
        oof_risk = np.array([oof_by_id.get(pid, 0.0) for pid in order], dtype=np.float32)
        reused = True
    else:
        cv = StratifiedGroupKFold(
            n_splits=_NOMANNIC_N_SPLITS, shuffle=True, random_state=_NOMANNIC_SEED
        )
        oof_risk = np.zeros(len(Xd), dtype=np.float32)
        Xdf = pd.DataFrame(Xd, columns=feature_cols)
        for fold, (tr, va) in enumerate(cv.split(Xdf, behavior_id, groups)):
            model = XGBClassifier(**_NOMANNIC_RISK_PARAMS, random_state=_NOMANNIC_SEED + fold)
            model.fit(
                Xdf.iloc[tr], y[tr], sample_weight=fit_weight[tr],
                eval_set=[(Xdf.iloc[va], y[va])], verbose=False,
            )
            oof_risk[va] = model.predict_proba(Xdf.iloc[va])[:, 1]
        pd.DataFrame({
            "pair_id": order,
            "is_labeled": known,
            "label": y,
            "oof_risk": oof_risk,
        }).to_parquet(oof_path, index=False)

    labeled_ap = (
        float(average_precision_score(y[known], oof_risk[known]))
        if known.sum() > 0 and y[known].sum() > 0
        else float("nan")
    )

    # Emit the CanonicalScorer-ready confirmed-only frame (same construction as the
    # floor / competitor runners: PairAP depends only on risk_score).
    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    score_by_pair = {
        pid: float(r) for pid, r, k in zip(order, oof_risk, known)
        if bool(k) and pid in confirmed_ids
    }
    solution = build_fold_solution(labels, list(score_by_pair.keys()), evidence=evidence)
    kept = [pid for pid in solution["pair_id"].astype(str) if pid in score_by_pair]

    preds = pd.DataFrame({"pair_id": kept})
    preds["risk_score"] = [score_by_pair[pid] for pid in kept]
    preds["risk_score"] = preds["risk_score"].astype(float).clip(0.0, 1.0)
    preds["predicted_behavior"] = "none"
    for col in SUBMISSION_COLUMNS:
        if col not in preds.columns:
            preds[col] = "NO_EVIDENCE"
    preds = preds[list(SUBMISSION_COLUMNS)]

    return NomannicRerunResult(
        dev_predictions=preds,
        n_confirmed=int(len(preds)),
        n_dev_all=int(len(Xd)),
        labeled_ap=labeled_ap,
        n_features=int(Xd.shape[1]),
        reused_cache=reused,
    )


def build_nomannic_recipe_runner(
    caches: NomannicCacheLocations,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    real_lb_digits: str,
    evidence: Optional[pd.DataFrame] = None,
    force: bool = False,
    on_result: Optional[Callable[[NomannicRerunResult], None]] = None,
) -> RecipeRunner:
    """Build a :data:`RecipeRunner` for the nomannic dev-holdout point (§54 test).

    Answers for the artifact whose digits equal ``real_lb_digits`` (the nomannic
    self-earned LB 0.56202 coordinate) with a :class:`RecipeRerun` under the canonical
    recipe id; returns ``None`` for every other artifact (honest BLOCKED).
    """
    cache: Dict[str, RecipeRerun] = {}

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if artifact.digits != real_lb_digits:
            return None
        if real_lb_digits not in cache:
            result = rerun_nomannic_recipe(caches, labels, evidence=evidence, force=force)
            if on_result is not None:
                on_result(result)
            detail = (
                f"re-ran {NOMANNIC_RECIPE_ID} risk head (5-fold StratifiedGroupKFold by "
                f"table_id, {result.n_features} pair feats) on the dev holdout; OOF risk "
                f"over {result.n_dev_all} dev pairs, canonical PairAP over "
                f"{result.n_confirmed} confirmed pairs; labeled OOF AP="
                f"{result.labeled_ap:.4f} (MEASURED local re-run figure, NOT an LB claim; "
                "DRIFT-FAILING basis scored ANYWAY per §54 to test contract Rule 17)"
                + ("; reused OOF cache" if result.reused_cache else "")
            )
            cache[real_lb_digits] = RecipeRerun(
                dev_predictions=result.dev_predictions,
                recipe_id=canonical_recipe_id,
                detail=detail,
            )
        return cache[real_lb_digits]

    return runner
