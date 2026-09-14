"""LEARNED risk combiner: a supervised logistic over the SAME features that reached CV AUC ~0.85.

Why this model exists
---------------------
The hand-tuned classical risk (:mod:`poker_collusion.models.classical`) saturated: its group-by-pool
dev CV AUC was only ~0.667. The Phase -1 signal-separation measurement
(:mod:`poker_collusion.discovery.signal_separation`) showed that ONE standardized logistic over ALL
the numeric features we already compute — the *multivariate upper bound* — reached group-by-pool dev
CV AUC ~0.850 on the confirmed dev pairs. That is a large, measured separation gain sitting unused
because the shipped risk head is the saturated classical score, not the learned combiner.

:class:`LearnedRiskModel` IS that learned combiner, packaged as a deterministic, persistable,
schema-fixed sklearn model with the same interface shape as
:class:`~poker_collusion.models.pu_ranker.PURanker` so the fast generator can swap the risk source
with a one-line change (:mod:`poker_collusion.pipeline_learned_fast`). It does NOT touch the
classical scorer (that explainable baseline stays intact) — the classical risk is consumed here only
as ONE input column, exactly as the separation module consumed it.

The feature contract (IDENTICAL to signal_separation.build_confirmed_feature_matrix)
------------------------------------------------------------------------------------
For every pair the per-row record is, verbatim, the record the separation module built and measured
the 0.85 on:

* every numeric feature in the pair's :class:`~poker_collusion.types.PairFeatureSet` ``features``
  dict (all PairFeatureSet features), each read as a finite float (NaN / +-inf -> 0.0 neutral);
* PLUS the classical risk pseudo-feature :data:`CLASSICAL_RISK_KEY` (deterministic
  :func:`poker_collusion.models.classical.score_pair` on the same injected EDA artifact);
* PLUS the cheap derived candidates
  (:func:`poker_collusion.discovery.signal_separation._cheap_derived_candidates`).

The trained column schema is the SORTED union of those record keys across the training pairs
(``sorted(set(...))``), so column order is deterministic and row-order independent. At predict time
a feature the model never trained on is filled as neutral 0.0 and any *extra* unseen column is
ignored (the schema is fixed by training) — we never invent or silently reorder columns.

The estimator (same recipe as the measured multivariate upper bound)
--------------------------------------------------------------------
``make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))`` — the exact pipeline
:func:`signal_separation._multivariate_cv_auc` used to score 0.85. It is fit on the CONFIRMED pairs
only (``confirmed_target -> 1``, ``confirmed_non_target -> 0``) — the same population the AUC was
measured on. ``predict_risk`` returns ``predict_proba[:, 1]`` clamped to ``[0, 1]``.

Unscoreable pairs & the pool-prior clamp (OFF by default for the LEARNED head)
------------------------------------------------------------------------------
PURanker and the classical head clamp an *unscoreable* pair (too few shared hands
``n_shared_hands < min_shared_hands``, or all-neutral features — the exact
:func:`poker_collusion.models.pu_ranker._is_unscoreable` rule) to the pool prior
(``pool_prior_positive_prevalence`` from the artifact, else
:data:`~poker_collusion.models.classical.DEFAULT_POOL_PRIOR`). That clamp is *counterproductive*
for the LEARNED head: it forces every low-shared-hands pair to the SAME constant, and because ~30%
of the confirmed positives share fewer than ``min_shared_hands`` hands, the constant collapses the
ranking (measured: pooled AUC 0.857 -> group-by-pool CV AUC 0.666). The calibrated logistic already
assigns low-signal pairs a low probability from their features, so the constant clamp only destroys
signal here.

Therefore the clamp is controlled by an explicit, documented flag
:attr:`LearnedRiskModel.apply_unscoreable_prior` (**default False** for the learned head). With the
flag OFF (default) EVERY pair is scored from its calibrated features
(``predict_proba[:, 1]`` clamped to ``[0, 1]``). With the flag ON the old PURanker-style
unscoreable->pool-prior clamp is applied (kept for parity / explicit coverage). This does NOT touch
:mod:`poker_collusion.models.classical` or :mod:`poker_collusion.models.pu_ranker`, whose
unscoreable->pool-prior semantics stay exactly as spec-defined for those heads.

HALT-safety & determinism
-------------------------
:meth:`fit` HALTS (raises, no partial state) when it cannot learn a genuine separation: no training
pairs, an empty feature matrix, or fewer than one example of EACH class (a degenerate single-class
fit cannot reproduce the separation). The generator catches this and falls back to the classical
risk, so generation always completes. Randomness is seeded from
:data:`poker_collusion.config.Seeds.learned_risk`; scoring is per-row so input order never changes a
pair's score. Persistence (:meth:`save` / :meth:`load`) is a secondary joblib round-trip mirroring
PURanker; determinism is the primary guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from poker_collusion.config import PipelineConfig, get_config
from poker_collusion.discovery.signal_separation import (
    CLASSICAL_RISK_KEY,
    _cheap_derived_candidates,
)
from poker_collusion.exceptions import SchemaError
from poker_collusion.models.classical import (
    DEFAULT_MIN_SHARED_HANDS,
    DEFAULT_POOL_PRIOR,
    score_pair,
)
from poker_collusion.models.pu_ranker import _is_unscoreable  # exact unscoreable rule (DRY)
from poker_collusion.types import LabelTable, PairFeatureSet

__all__ = [
    "CLASSICAL_RISK_KEY",
    "LearnedRiskModel",
    "LearnedRiskCVGate",
    "train_learned_risk",
    "learned_risk_cv_auc",
    "pair_record",
]

# Real label-status tokens (same source signal_separation / cv use).
CONFIRMED_TARGET = "confirmed_target"
CONFIRMED_NON_TARGET = "confirmed_non_target"


# --------------------------------------------------------------------------- #
# Per-pair record (IDENTICAL to signal_separation.build_confirmed_feature_matrix rows)
# --------------------------------------------------------------------------- #
def _finite(value: object, default: float = 0.0) -> float:
    """Coerce to a finite float; NaN / +-inf / uncoercible -> ``default`` (neutral)."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def pair_record(
    fs: PairFeatureSet,
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
) -> Dict[str, float]:
    """Build ONE pair's feature record exactly like ``signal_separation.build_confirmed_feature_matrix``.

    Every numeric ``PairFeatureSet`` feature (finite float; NaN/inf -> 0.0) PLUS the classical risk
    pseudo-feature (:data:`CLASSICAL_RISK_KEY`) PLUS the cheap derived candidates. This is the
    single source of truth for the record shape so the model trains and scores on the SAME columns
    the 0.85 CV AUC was measured over.
    """
    rec: Dict[str, float] = {k: _finite(v) for k, v in fs.features.items()}
    rec[CLASSICAL_RISK_KEY] = _finite(score_pair(fs, eda_artifact=eda_artifact).risk_score)
    rec.update({k: _finite(v) for k, v in _cheap_derived_candidates(fs.features).items()})
    return rec


def _matrix(
    feature_sets: Sequence[PairFeatureSet],
    columns: Sequence[str],
    *,
    eda_artifact: Optional[Union[Mapping, object]],
) -> np.ndarray:
    """Dense ``(n_pairs, n_columns)`` float matrix over the FIXED ``columns`` order.

    Missing columns read as neutral 0.0; extra keys not in ``columns`` are ignored (the schema is
    fixed by training). Order follows ``columns`` exactly, so the matrix is deterministic and
    row-order independent.
    """
    col_index = {name: j for j, name in enumerate(columns)}
    out = np.zeros((len(feature_sets), len(columns)), dtype=np.float64)
    for i, fs in enumerate(feature_sets):
        for name, val in pair_record(fs, eda_artifact=eda_artifact).items():
            j = col_index.get(name)
            if j is not None:
                out[i, j] = val
    return out


def _status_map_from_labels(
    labels: Union[LabelTable, Mapping[str, str]],
) -> Tuple[set, set]:
    """Return ``(positive_ids, negative_ids)`` from a LabelTable or a ``{pair_id -> status}`` map."""
    if isinstance(labels, LabelTable):
        return set(labels.trusted_positive.keys()), set(labels.confirmed_negative)
    positives: set = set()
    negatives: set = set()
    for pid, status in labels.items():
        s = str(status)
        if s == CONFIRMED_TARGET:
            positives.add(str(pid))
        elif s == CONFIRMED_NON_TARGET:
            negatives.add(str(pid))
    return positives, negatives


def _pool_prior_from(eda_artifact: Optional[Union[Mapping, object]], default: float) -> float:
    """Read ``pool_prior_positive_prevalence`` from an injected artifact (dict or EdaArtifact)."""
    thresholds = _thresholds_of(eda_artifact)
    if thresholds is None:
        return float(default)
    f = _finite(thresholds.get("pool_prior_positive_prevalence", default), default)
    return min(max(f, 0.0), 1.0)


def _min_shared_hands_from(eda_artifact: Optional[Union[Mapping, object]], default: int) -> int:
    """Read ``min_shared_hands`` from an injected artifact; fall back to ``default``."""
    thresholds = _thresholds_of(eda_artifact)
    if thresholds is None:
        return int(default)
    try:
        return int(thresholds.get("min_shared_hands", default))
    except (TypeError, ValueError):
        return int(default)


def _thresholds_of(eda_artifact: Optional[Union[Mapping, object]]) -> Optional[Mapping]:
    """Return the ``thresholds`` mapping from an injected artifact (dict or EdaArtifact)."""
    if eda_artifact is None:
        return None
    if hasattr(eda_artifact, "thresholds"):
        t = getattr(eda_artifact, "thresholds")
        return t if isinstance(t, Mapping) else None
    if isinstance(eda_artifact, Mapping):
        inner = eda_artifact.get("thresholds")
        if isinstance(inner, Mapping):
            return inner
        return eda_artifact
    return None


def _resolve_threshold_version(
    eda_artifact: Optional[Union[Mapping, object]],
    config: PipelineConfig,
) -> str:
    """Prefer the injected artifact's ``threshold_version``; fall back to the config's."""
    if eda_artifact is not None:
        tv = getattr(eda_artifact, "threshold_version", None)
        if tv is None and isinstance(eda_artifact, Mapping):
            tv = eda_artifact.get("threshold_version")
        if tv:
            return str(tv)
    return config.threshold_version


# --------------------------------------------------------------------------- #
# Persisted parameter carrier
# --------------------------------------------------------------------------- #
@dataclass
class LearnedRiskParams:
    """Everything needed to reproduce scoring exactly (persisted via joblib).

    Attributes:
        columns: Fixed feature-column order (the trained schema).
        pool_prior: Defined default risk for unscoreable pairs.
        min_shared_hands: Floor below which a pair is unscoreable.
        seed: Seed the estimator was fit under (:class:`Seeds.learned_risk`).
        schema_version / threshold_version: Feature schema identifiers (provenance).
        estimator: The fitted ``make_pipeline(StandardScaler, LogisticRegression)`` (pickled).
    """

    columns: List[str]
    pool_prior: float
    min_shared_hands: int
    seed: int
    schema_version: str
    threshold_version: str
    estimator: object = None


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class LearnedRiskModel:
    """Supervised logistic risk combiner over the confirmed-pair features (CV AUC ~0.85 recipe).

    See the module docstring for the feature contract, estimator, unscoreable default, HALT-safety,
    determinism and persistence.
    """

    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        seed: Optional[int] = None,
        apply_unscoreable_prior: bool = False,
    ) -> None:
        """Construct the learned risk combiner.

        Args:
            config: Pipeline config (defaults to the process config).
            seed: Estimator seed (defaults to :class:`Seeds.learned_risk`).
            apply_unscoreable_prior: When ``True`` an unscoreable pair (PURanker rule) is clamped to
                the pool prior — the old PURanker-inherited behavior. **Default ``False``**: the
                learned head scores EVERY pair from its calibrated features, because the constant
                clamp collapses the learned ranking (see module docstring). Keep ``False`` for the
                shipped learned head; ``True`` exists for parity / explicit coverage.
        """
        self._config = config or get_config()
        self._seed = int(seed if seed is not None else self._config.seeds.learned_risk)
        self._apply_unscoreable_prior = bool(apply_unscoreable_prior)
        self._params: Optional[LearnedRiskParams] = None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._params is not None and self._params.estimator is not None

    @property
    def columns(self) -> List[str]:
        self._require_fitted()
        return list(self._params.columns)  # type: ignore[union-attr]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise SchemaError("LearnedRiskModel", "fitted_estimator (call fit() or load() first)")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_feature_sets: Sequence[PairFeatureSet],
        labels: Union[LabelTable, Mapping[str, str]],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> "LearnedRiskModel":
        """Fit the logistic on the CONFIRMED pairs only (target=1, non_target=0).

        HALTS (raises, no partial state) when it cannot learn a genuine separation: no training
        pairs, an empty feature matrix, no confirmed pair, or fewer than one example of EACH class.

        Args:
            train_feature_sets: Development :class:`PairFeatureSet`\\ s (order-independent). Only the
                pairs that are confirmed_target / confirmed_non_target are used for training — the
                same population the 0.85 CV AUC was measured on.
            labels: A :class:`LabelTable` or ``{pair_id -> label_status}`` map.
            eda_artifact: Injected EDA artifact for the pool-prior default, the min-shared-hands
                floor, and the deterministic classical-risk column.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if train_feature_sets is None or len(train_feature_sets) == 0:
            raise SchemaError(
                "learned_risk.fit(train_feature_sets)", "non-empty training feature sets"
            )

        positive_ids, negative_ids = _status_map_from_labels(labels)

        # Keep confirmed pairs only, with their binary labels (target=1, non_target=0).
        confirmed_fs: List[PairFeatureSet] = []
        y: List[int] = []
        for fs in train_feature_sets:
            pid = str(fs.pair_id)
            if pid in positive_ids:
                confirmed_fs.append(fs)
                y.append(1)
            elif pid in negative_ids:
                confirmed_fs.append(fs)
                y.append(0)
        if not confirmed_fs:
            raise SchemaError(
                "learned_risk.fit(labels)", "at least one confirmed_target/confirmed_non_target pair"
            )
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise SchemaError(
                "learned_risk.fit(labels)",
                "both a confirmed_target (1) and a confirmed_non_target (0) example are required "
                "to learn a separation",
            )

        # Fixed, sorted feature-column schema over the confirmed-pair records (row-order safe).
        feature_names: set = set()
        for fs in confirmed_fs:
            feature_names.update(pair_record(fs, eda_artifact=eda_artifact).keys())
        if not feature_names:
            raise SchemaError("learned_risk.fit(features)", "at least one feature column")
        columns = sorted(feature_names)

        X = _matrix(confirmed_fs, columns, eda_artifact=eda_artifact)

        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=self._seed),
        )
        estimator.fit(X, y_arr)

        self._params = LearnedRiskParams(
            columns=list(columns),
            pool_prior=_pool_prior_from(eda_artifact, DEFAULT_POOL_PRIOR),
            min_shared_hands=_min_shared_hands_from(eda_artifact, DEFAULT_MIN_SHARED_HANDS),
            seed=self._seed,
            schema_version=self._config.schema_version,
            threshold_version=_resolve_threshold_version(eda_artifact, self._config),
            estimator=estimator,
        )
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def predict_risk(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> List[Tuple[str, float]]:
        """Score every eval pair -> ``[(pair_id, risk in [0,1]), ...]`` in input order.

        Coverage: exactly one score per input pair. By default
        (``apply_unscoreable_prior=False``) EVERY pair — including low-shared-hands pairs — is scored
        from its calibrated features (``predict_proba[:, 1]``), because the constant pool-prior clamp
        collapses the learned ranking. When ``apply_unscoreable_prior=True`` an unscoreable pair (too
        few shared hands / all neutral, the PURanker rule) is instead clamped to the pool-prior
        default. Scores are per-row, so input order never changes any pair's score. Risk is always
        clamped to ``[0, 1]``. HALTS on empty input.
        """
        self._require_fitted()
        params = self._params
        assert params is not None

        if eval_feature_sets is None or len(eval_feature_sets) == 0:
            raise SchemaError(
                "learned_risk.predict_risk(eval_feature_sets)", "non-empty evaluation pairs"
            )

        X = _matrix(eval_feature_sets, params.columns, eda_artifact=eda_artifact)
        proba = params.estimator.predict_proba(X)[:, 1]  # type: ignore[union-attr]

        out: List[Tuple[str, float]] = []
        for i, fs in enumerate(eval_feature_sets):
            if self._apply_unscoreable_prior and _is_unscoreable(fs, params.min_shared_hands):
                # Legacy PURanker-style clamp (opt-in): unscoreable -> constant pool prior.
                risk = params.pool_prior
            else:
                # Default learned head: score from calibrated features (no constant clamp).
                risk = float(proba[i])
            out.append((str(fs.pair_id), float(min(max(risk, 0.0), 1.0))))
        return out

    def score_pairs(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> Dict[str, float]:
        """Convenience: same as :meth:`predict_risk` but returns a ``{pair_id -> risk}`` dict."""
        return {
            pid: risk
            for pid, risk in self.predict_risk(eval_feature_sets, eda_artifact=eda_artifact)
        }

    # ------------------------------------------------------------------ #
    # Persistence (secondary; determinism is primary)
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> Path:
        """Persist the fitted model (estimator + schema + seed + versions) via joblib (atomic)."""
        self._require_fitted()
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        joblib.dump(self._params, tmp)
        tmp.replace(target)
        return target

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        config: Optional[PipelineConfig] = None,
        apply_unscoreable_prior: bool = False,
    ) -> "LearnedRiskModel":
        """Load a model persisted by :meth:`save`; reproduces identical scores.

        ``apply_unscoreable_prior`` (default ``False``, the shipped learned-head behavior) is a
        scoring-time choice, not a persisted parameter, so it is set here on load.
        """
        import joblib

        source = Path(path)
        if not source.exists():
            raise SchemaError(str(source), "existing persisted LearnedRiskModel file")
        params: LearnedRiskParams = joblib.load(source)
        obj = cls(config=config, seed=params.seed, apply_unscoreable_prior=apply_unscoreable_prior)
        obj._params = params
        return obj


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #
def train_learned_risk(
    train_feature_sets: Sequence[PairFeatureSet],
    labels: Union[LabelTable, Mapping[str, str]],
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    config: Optional[PipelineConfig] = None,
    seed: Optional[int] = None,
    apply_unscoreable_prior: bool = False,
) -> LearnedRiskModel:
    """Fit and return a :class:`LearnedRiskModel` in one call (documented convenience wrapper).

    ``apply_unscoreable_prior`` defaults to ``False`` (the shipped learned-head behavior: score
    every pair from its calibrated features, no constant pool-prior clamp).
    """
    model = LearnedRiskModel(config=config, seed=seed, apply_unscoreable_prior=apply_unscoreable_prior)
    return model.fit(train_feature_sets, labels, eda_artifact=eda_artifact)


# --------------------------------------------------------------------------- #
# CV-AUC GATE (transfer sanity-check): group-by-pool CV of the LearnedRiskModel
# on the confirmed dev pairs. Reuses signal_separation's real-data machinery (DRY).
# --------------------------------------------------------------------------- #
@dataclass
class LearnedRiskCVGate:
    """Result of the group-by-pool CV-AUC gate for :class:`LearnedRiskModel`.

    Attributes:
        cv_auc_mean: Mean held-out AUC across the group-by-pool folds (the transfer sanity-check
            number; expected ~0.80-0.86, i.e. it should reproduce the ~0.85 separation).
        cv_auc_std: Std of the per-fold held-out AUCs.
        n_folds: Number of folds that were scoreable (both classes in train AND held-out pool).
        n_pos: Confirmed-target pairs measured.
        n_neg: Confirmed-non-target pairs measured.
        n_pools: Distinct pools among the confirmed pairs.
        caveats: Any sampling caveats (e.g. a ``max_pairs`` head-subset).
    """

    cv_auc_mean: float
    cv_auc_std: float
    n_folds: int
    n_pos: int
    n_neg: int
    n_pools: int
    caveats: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "cv_auc_mean": self.cv_auc_mean,
            "cv_auc_std": self.cv_auc_std,
            "n_folds": self.n_folds,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "n_pools": self.n_pools,
            "caveats": self.caveats,
        }


def learned_risk_cv_auc(
    *,
    config: Optional[PipelineConfig] = None,
    data_loader=None,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    max_pairs: Optional[int] = None,
) -> LearnedRiskCVGate:
    """Group-by-pool CV AUC of the :class:`LearnedRiskModel` on the confirmed dev pairs.

    This is the transfer sanity-check folded into the build: it trains the SAME model the generator
    uses (leave-one-pool-out) on the confirmed development pairs and reports the mean held-out AUC.
    It reuses :mod:`poker_collusion.discovery.signal_separation`'s bounded one-actions-pass
    machinery so the measured population is IDENTICAL to the one the ~0.85 multivariate AUC was
    measured on. Bounded exactly like the Phase-1 floor (only confirmed labelled pairs' development
    shared hands are read, one streaming pass).
    """
    from sklearn.metrics import roc_auc_score

    from poker_collusion.discovery.signal_separation import (
        _prepare_labelled_inputs,
        _signals_for_pair,
    )
    from poker_collusion.features.pair_features import build_pair_feature_set
    from poker_collusion.io import DataLoader
    from poker_collusion.pipeline import _obtain_eda_artifact
    from poker_collusion.validation.cv import derive_pool_map
    from poker_collusion.validation.phase1_baseline_cv import DEVELOPMENT_PHASE

    cfg = config or get_config()
    loader = data_loader or DataLoader(config=cfg)
    artifact = _obtain_eda_artifact(loader, cfg, eda_artifact)

    labels = loader.load_labels_frame()
    total_labelled = len(labels)

    caveats: List[str] = []
    if max_pairs is not None and max_pairs < total_labelled:
        labels = labels.head(int(max_pairs)).copy()
        caveats.append(
            f"SAMPLED: measured only the first {int(max_pairs)} labelled pairs (of "
            f"{total_labelled}) for tractability (Phase-1 head-subset convention)."
        )

    confirmed_pair_ids = [
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}
    ]

    inputs = _prepare_labelled_inputs(loader, labels, confirmed_pair_ids)
    pool_map = derive_pool_map(labels, config=cfg)

    # {pair_id -> status} for confirmed pairs (drop those with no dev shared hands -> no signal).
    status_by_pair: Dict[str, str] = {}
    for r in labels.itertuples(index=False):
        st = str(getattr(r, "label_status", ""))
        if st in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}:
            status_by_pair[str(getattr(r, "pair_id"))] = st

    # Build one development-phase PairFeatureSet per confirmed pair with signal.
    fs_by_pair: Dict[str, PairFeatureSet] = {}
    label_by_pair: Dict[str, int] = {}
    pool_by_pair: Dict[str, object] = {}
    for pair_id, status in status_by_pair.items():
        signals = _signals_for_pair(inputs, pair_id)
        if not signals:
            continue
        player_a, player_b = inputs.players[pair_id]
        fs = build_pair_feature_set(
            pair_id=pair_id,
            signals=signals,
            player_a=player_a,
            player_b=player_b,
            phase=DEVELOPMENT_PHASE,
            eda_artifact=artifact,
            config=cfg,
        )
        fs_by_pair[pair_id] = fs
        label_by_pair[pair_id] = 1 if status == CONFIRMED_TARGET else 0
        pool_by_pair[pair_id] = pool_map.get(str(pair_id), "UNKNOWN_POOL")

    pair_ids = list(fs_by_pair.keys())
    n_pos = sum(1 for p in pair_ids if label_by_pair[p] == 1)
    n_neg = sum(1 for p in pair_ids if label_by_pair[p] == 0)
    pools = []
    for p in pair_ids:
        if pool_by_pair[p] not in pools:
            pools.append(pool_by_pair[p])

    # Leave-one-pool-out CV: train a fresh LearnedRiskModel on all-but-one pool, score the held-out
    # pool. Skip folds whose train OR held-out set is single-class.
    aucs: List[float] = []
    for held in pools:
        train_pairs = [p for p in pair_ids if pool_by_pair[p] != held]
        test_pairs = [p for p in pair_ids if pool_by_pair[p] == held]
        y_tr = [label_by_pair[p] for p in train_pairs]
        y_te = [label_by_pair[p] for p in test_pairs]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        status_tr = {p: status_by_pair[p] for p in train_pairs}
        try:
            model = LearnedRiskModel(config=cfg)
            model.fit([fs_by_pair[p] for p in train_pairs], status_tr, eda_artifact=artifact)
            scored = model.score_pairs(
                [fs_by_pair[p] for p in test_pairs], eda_artifact=artifact
            )
            proba = [scored[p] for p in test_pairs]
            aucs.append(float(roc_auc_score(y_te, proba)))
        except Exception:
            continue

    if aucs:
        cv_mean = float(np.mean(aucs))
        cv_std = float(np.std(aucs))
    else:
        cv_mean, cv_std = 0.5, 0.0

    return LearnedRiskCVGate(
        cv_auc_mean=cv_mean,
        cv_auc_std=cv_std,
        n_folds=len(aucs),
        n_pos=n_pos,
        n_neg=n_neg,
        n_pools=len(pools),
        caveats=caveats,
    )
