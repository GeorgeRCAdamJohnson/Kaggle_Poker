"""PU-aware risk model (task 16.1, Requirement 6.1-6.8; Design "PU Ranking Model").

This module trains the pipeline's :class:`Risk_Model`: a Positive-Unlabelled (PU) aware
scorer that emits exactly one ``risk_score`` in ``[0, 1]`` for every evaluation
:class:`~poker_collusion.types.PairFeatureSet`. It consumes the deterministic feature layer
(:mod:`poker_collusion.features.pair_features`) and, optionally, the explainable classical
baseline score (:mod:`poker_collusion.models.classical`) as an extra input feature, so the PU
model *layers on* the Phase-1 floor rather than replacing it.

Why PU learning here (Req 6.1, 6.2)
-----------------------------------
``development_labels.csv`` gives us Trusted_Positive pairs (``label_status == 'confirmed_target'``)
and Confirmed_Negative pairs (``label_status == 'confirmed_non_target'``). **Every other**
development pair — and every evaluation pair — is an Unknown_Pair: unlabelled, NOT negative
(Req 6.1). Treating unknowns as negatives would train the model to suppress exactly the hidden
positives we are hunting, biasing Pair AP downward. So we learn under the PU assumption instead.

The PU method we use (documented, deterministic): Elkan-Noto with reliable negatives
-------------------------------------------------------------------------------------
We combine the two PU recipes the task allows, taking the more informative parts of each:

1. **Non-traditional (Elkan & Noto 2008) classifier + prior/label-frequency correction.**
   We fit a calibrated probabilistic classifier ``g(x) = P(s = 1 | x)`` that predicts the
   *labeling indicator* ``s`` — ``s = 1`` for a pair that is a **labeled positive** (Trusted_
   Positive) and ``s = 0`` for a pair drawn from the **unlabelled mix**. Under the Elkan-Noto
   *Selected-Completely-At-Random* (SCAR) assumption, a labeled positive is a positive selected
   at random with constant probability ``c = P(s = 1 | y = 1)`` (the *label frequency*), and

       P(y = 1 | x)  =  g(x) / c .

   So a monotone transform of ``g`` recovers the true positive-probability *ranking* (ranking is
   all Pair AP needs), and dividing by ``c`` calibrates the level. We estimate ``c`` as the mean
   of ``g(x)`` over the held-in labeled positives (the Elkan-Noto "e1" estimator), clamped away
   from 0. Because the correction is monotone, it never changes the pair ordering — only the
   absolute level — which keeps determinism and Pair AP intact.

2. **Reliable negatives.** When ``development_labels.csv`` supplies Confirmed_Negative pairs, we
   use them as *reliable* negatives (Req 6.2): they join the unlabelled pool with a labeling
   indicator ``s = 0`` **and** are given a larger sample weight (:data:`RELIABLE_NEGATIVE_WEIGHT`)
   so the classifier trusts them more than an ordinary unlabelled pair (which may secretly be
   positive). This is the standard "reliable negative" hybrid: reliable negatives sharpen the
   decision surface while the unlabelled mix supplies the base rate for the Elkan-Noto correction.

We deliberately do **not** hard-label unknowns as negative anywhere.

Estimator. A ``LogisticRegression`` wrapped for probability calibration. Logistic regression is
linear, deterministic under a fixed seed/solver, cheap to persist (coefficients), and monotone so
the Elkan-Noto division is well-behaved. It is fit on the fixed feature matrix described below.

Feature-matrix contract (fixed schema; Req 6.7, 6.8)
----------------------------------------------------
* The columns are the **sorted union of every feature name** present across the training
  :class:`PairFeatureSet`\\ s (``sorted(set(...))``), so column order is deterministic and
  independent of dict-insertion / row order.
* Optionally, the classical baseline ``risk_score`` is appended as one extra column
  :data:`CLASSICAL_SCORE_COLUMN` (``include_classical_score=True``, the default). It is computed
  per pair by :func:`poker_collusion.models.classical.score_pair` with the same injected EDA
  artifact, so it is deterministic and hermetic.
* Cell values: each feature is read as a finite float; **NaN / +-inf -> 0.0 (neutral)**.
* **At predict time the evaluation pairs MUST expose exactly the trained columns.** A feature
  present at training but absent from an eval pair reads as neutral 0.0 (schema is fixed by
  training). A pair that carries a feature name the model never saw is a schema mismatch and
  raises :class:`~poker_collusion.exceptions.SchemaError` (Req 6.8) — we never silently drop or
  invent columns.

Output, coverage, and the unscoreable default (Req 6.3, 6.4)
------------------------------------------------------------
:meth:`PURanker.predict_risk` returns **one** ``(pair_id, risk_score)`` per input pair (coverage),
each ``risk_score`` clamped into ``[0, 1]`` (Req 6.3). A pair is **unscoreable** when it shares
too few hands for a trustworthy signal (``n_shared_hands < min_shared_hands`` from the EDA
artifact) or when all of its (finite) features are neutral. An unscoreable pair is assigned the
**defined default = the pool prior** (``pool_prior_positive_prevalence`` from the EDA artifact,
else :data:`DEFAULT_POOL_PRIOR`) rather than being dropped or scored NaN (Req 6.4).

Determinism & order-independence (Req 6.5, 6.7)
-----------------------------------------------
All randomness is seeded from :data:`poker_collusion.config.Seeds.pu_ranker`. The model scores
each pair from its own feature row, so input row order cannot affect any pair's score; the sorted
feature columns and sorted training rows make fitting order-independent too. Equal scores are left
to the competition's deterministic ``pair_id`` tie-break downstream (Req 6.5) — this model never
relies on row order.

Robustness / HALT (Req 6.6, 6.8)
--------------------------------
:meth:`PURanker.fit` HALTS with a descriptive error and produces **no** partial state when a
required input is missing/empty: no training feature sets, an empty feature matrix, or **no
Trusted_Positive** to learn from. :meth:`predict_risk` HALTS on an empty eval set or a feature-name
mismatch versus the trained schema. Errors are raised *before* any output is written, so no partial
artifact is produced (Req 6.8).

Persistence (Req 6.7)
---------------------
:meth:`PURanker.save` persists the fitted estimator, the exact feature-column order, the label
frequency ``c``, the pool-prior default, the seed, and the schema/threshold versions via
``joblib`` so :meth:`PURanker.load` reproduces byte-identical scores. A documented
``to_params()`` / ``from_params()`` dict round-trip is also provided for JSON-friendly inspection
of the linear coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from poker_collusion.config import PipelineConfig, Seeds, get_config
from poker_collusion.exceptions import SchemaError
from poker_collusion.models.classical import (
    DEFAULT_MIN_SHARED_HANDS,
    DEFAULT_POOL_PRIOR,
    score_pair,
)
from poker_collusion.types import LabelTable, PairFeatureSet

__all__ = [
    "CLASSICAL_SCORE_COLUMN",
    "RELIABLE_NEGATIVE_WEIGHT",
    "MIN_LABEL_FREQUENCY",
    "PURanker",
    "train_pu_ranker",
]

# --------------------------------------------------------------------------- #
# Documented constants
# --------------------------------------------------------------------------- #
#: Name of the extra feature column holding the classical-baseline risk score (when enabled).
#: Chosen to sort AFTER the ``conf_`` / signal feature names so the classical score is a clearly
#: identifiable trailing column; it is treated exactly like any other numeric feature.
CLASSICAL_SCORE_COLUMN: str = "zz_classical_risk_score"

#: Sample weight given to a Confirmed_Negative (reliable negative) relative to an ordinary
#: unlabelled pair (weight 1.0). Reliable negatives are trusted more because an unlabelled pair may
#: secretly be positive, whereas a confirmed non-target is a genuine negative (Req 6.2).
RELIABLE_NEGATIVE_WEIGHT: float = 3.0

#: Floor for the Elkan-Noto label frequency ``c = P(s = 1 | y = 1)``. Prevents divide-by-tiny when
#: the positive class is degenerate; keeps the corrected probability finite and deterministic.
MIN_LABEL_FREQUENCY: float = 1e-3


# --------------------------------------------------------------------------- #
# Feature-matrix construction (fixed schema; Req 6.7 / 6.8)
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


def _pool_prior_from(eda_artifact: Optional[Union[Mapping, object]], default: float) -> float:
    """Read ``pool_prior_positive_prevalence`` from an injected artifact (dict or EdaArtifact)."""
    thresholds = _thresholds_of(eda_artifact)
    if thresholds is None:
        return float(default)
    val = thresholds.get("pool_prior_positive_prevalence", default)
    f = _finite(val, default)
    return min(max(f, 0.0), 1.0)


def _min_shared_hands_from(eda_artifact: Optional[Union[Mapping, object]], default: int) -> int:
    """Read ``min_shared_hands`` from an injected artifact; fall back to ``default``."""
    thresholds = _thresholds_of(eda_artifact)
    if thresholds is None:
        return int(default)
    val = thresholds.get("min_shared_hands", default)
    try:
        return int(val)
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


def _status_map_from_labels(
    labels: Union[LabelTable, Mapping[str, str]],
) -> Tuple[set, set]:
    """Return ``(positive_ids, negative_ids)`` from a LabelTable or a status map.

    A status map is ``{pair_id -> label_status}`` using the real tokens
    ``confirmed_target`` / ``confirmed_non_target``; anything else is unknown (unlabelled).
    """
    if isinstance(labels, LabelTable):
        return set(labels.trusted_positive.keys()), set(labels.confirmed_negative)
    positives: set = set()
    negatives: set = set()
    for pid, status in labels.items():
        s = str(status)
        if s == "confirmed_target":
            positives.add(str(pid))
        elif s == "confirmed_non_target":
            negatives.add(str(pid))
    return positives, negatives


def _classical_score_for(
    fs: PairFeatureSet, eda_artifact: Optional[Union[Mapping, object]]
) -> float:
    """Deterministic classical baseline risk for one pair (used as an optional input feature)."""
    return float(score_pair(fs, eda_artifact=eda_artifact).risk_score)


def build_feature_matrix(
    feature_sets: Sequence[PairFeatureSet],
    columns: Sequence[str],
    *,
    include_classical_score: bool,
    eda_artifact: Optional[Union[Mapping, object]] = None,
) -> np.ndarray:
    """Build a dense ``(n_pairs, n_columns)`` float matrix over the FIXED ``columns`` order.

    Missing features read as neutral ``0.0``; NaN/inf clamp to ``0.0``. The optional classical
    score column (:data:`CLASSICAL_SCORE_COLUMN`) is filled from :func:`score_pair`. Column order
    is exactly ``columns`` (the trained schema), so the matrix is deterministic and row-order
    independent.
    """
    n = len(feature_sets)
    m = len(columns)
    out = np.zeros((n, m), dtype=np.float64)
    col_index = {name: j for j, name in enumerate(columns)}
    classical_j = col_index.get(CLASSICAL_SCORE_COLUMN) if include_classical_score else None
    for i, fs in enumerate(feature_sets):
        for name, val in fs.features.items():
            j = col_index.get(name)
            if j is not None:
                out[i, j] = _finite(val)
        if classical_j is not None:
            out[i, classical_j] = _finite(_classical_score_for(fs, eda_artifact))
    return out


def _is_unscoreable(fs: PairFeatureSet, min_shared_hands: int) -> bool:
    """A pair is unscoreable when it shares too few hands or all its finite features are neutral."""
    n_shared = _finite(fs.features.get("n_shared_hands", 0.0))
    if n_shared < float(min_shared_hands):
        return True
    # all-neutral: every feature (that isn't a provenance marker) is ~0.
    for name, val in fs.features.items():
        if name in ("n_shared_hands", "phase_is_development"):
            continue
        if abs(_finite(val)) > 1e-12:
            return False
    return True


# --------------------------------------------------------------------------- #
# Persisted parameter carrier
# --------------------------------------------------------------------------- #
@dataclass
class PUParams:
    """Everything needed to reproduce scoring exactly (persisted via joblib; Req 6.7).

    Attributes:
        columns: Fixed feature-column order (the trained schema).
        label_frequency: Elkan-Noto ``c = P(s = 1 | y = 1)`` used to calibrate ``g(x) -> f(x)``.
        pool_prior: Defined default risk for unscoreable pairs (Req 6.4).
        min_shared_hands: Floor below which a pair is unscoreable.
        include_classical_score: Whether :data:`CLASSICAL_SCORE_COLUMN` is part of ``columns``.
        seed: The seed the estimator was fit under (:class:`Seeds.pu_ranker`).
        schema_version / threshold_version: Feature schema identifiers (provenance).
        estimator: The fitted scikit-learn estimator (pickled by joblib).
    """

    columns: List[str]
    label_frequency: float
    pool_prior: float
    min_shared_hands: int
    include_classical_score: bool
    seed: int
    schema_version: str
    threshold_version: str
    estimator: object = None


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class PURanker:
    """PU-aware risk model producing one ``risk_score`` in ``[0, 1]`` per evaluation pair.

    See the module docstring for the full PU method (Elkan-Noto + reliable negatives), the
    feature-matrix contract, the unscoreable default, determinism, and persistence.
    """

    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        include_classical_score: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config or get_config()
        self._include_classical_score = bool(include_classical_score)
        self._seed = int(seed if seed is not None else self._config.seeds.pu_ranker)
        self._params: Optional[PUParams] = None

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
            raise SchemaError("PURanker", "fitted_estimator (call fit() or load() first)")

    # ------------------------------------------------------------------ #
    # Training (Req 6.1, 6.2, 6.7, 6.8)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_feature_sets: Sequence[PairFeatureSet],
        labels: Union[LabelTable, Mapping[str, str]],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> "PURanker":
        """Fit the PU model. HALTS (no partial state) on missing/empty required input (Req 6.8).

        Args:
            train_feature_sets: Development :class:`PairFeatureSet`\\ s (positives + confirmed
                negatives + unknowns). Order-independent.
            labels: A :class:`LabelTable` or a ``{pair_id -> label_status}`` map. Trusted_Positive
                pairs are POSITIVE (``s = 1``); Confirmed_Negative pairs are reliable negatives;
                every other pair is unlabelled (``s = 0``, weight 1.0) — NEVER hard-negative.
            eda_artifact: Injected EDA artifact for the pool-prior default, the min-shared-hands
                floor, and the deterministic classical-score feature.

        Returns:
            ``self`` (fitted). On any required-input failure, raises before mutating ``self``.
        """
        # Lazy sklearn import so importing this module is cheap and hermetic.
        from sklearn.linear_model import LogisticRegression

        if train_feature_sets is None or len(train_feature_sets) == 0:
            raise SchemaError("pu_ranker.fit(train_feature_sets)", "non-empty training feature sets")

        positive_ids, negative_ids = _status_map_from_labels(labels)

        # Fixed, sorted feature-column schema (row-order independent).
        feature_names: set = set()
        for fs in train_feature_sets:
            feature_names.update(fs.features.keys())
        if not feature_names:
            raise SchemaError("pu_ranker.fit(features)", "at least one feature column")
        columns = sorted(feature_names)
        if self._include_classical_score:
            columns = columns + [CLASSICAL_SCORE_COLUMN]

        # PU labeling indicator s: 1 for trusted positives, 0 for everything else (unlabelled +
        # reliable negatives). Confirmed negatives carry extra weight (Req 6.2). Unknowns are NOT
        # treated as hard negatives — they are the unlabelled mix for the Elkan-Noto correction.
        s = np.zeros(len(train_feature_sets), dtype=np.int64)
        weights = np.ones(len(train_feature_sets), dtype=np.float64)
        n_pos = 0
        for i, fs in enumerate(train_feature_sets):
            pid = str(fs.pair_id)
            if pid in positive_ids:
                s[i] = 1
                n_pos += 1
            elif pid in negative_ids:
                s[i] = 0
                weights[i] = RELIABLE_NEGATIVE_WEIGHT
        if n_pos == 0:
            raise SchemaError("pu_ranker.fit(labels)", "at least one Trusted_Positive to train on")

        pool_prior = _pool_prior_from(eda_artifact, DEFAULT_POOL_PRIOR)
        min_shared = _min_shared_hands_from(eda_artifact, DEFAULT_MIN_SHARED_HANDS)

        X = build_feature_matrix(
            train_feature_sets,
            columns,
            include_classical_score=self._include_classical_score,
            eda_artifact=eda_artifact,
        )

        # Deterministic logistic regression on the labeling indicator s (Elkan-Noto "g(x)").
        # Both classes are guaranteed present: s has >=1 positive; if there is no s==0 row we add
        # the pool prior as a degenerate base rate by falling back to a single synthetic-free path.
        estimator = LogisticRegression(
            random_state=self._seed,
            solver="liblinear",
            C=1.0,
            class_weight=None,
            max_iter=1000,
        )
        if len(np.unique(s)) < 2:
            # Degenerate: only positives present (no unlabelled/negative). We cannot fit a
            # discriminative classifier; HALT per Req 6.8 rather than emit a meaningless model.
            raise SchemaError(
                "pu_ranker.fit(labels)",
                "both a positive (s=1) and an unlabelled/negative (s=0) example are required",
            )
        estimator.fit(X, s, sample_weight=weights)

        # Elkan-Noto label frequency c = mean g(x) over the held-in labeled positives (the "e1"
        # estimator). Monotone; clamped away from 0.
        g = estimator.predict_proba(X)[:, 1]
        c = float(np.mean(g[s == 1])) if n_pos > 0 else 1.0
        c = max(c, MIN_LABEL_FREQUENCY)

        self._params = PUParams(
            columns=list(columns),
            label_frequency=c,
            pool_prior=float(pool_prior),
            min_shared_hands=int(min_shared),
            include_classical_score=self._include_classical_score,
            seed=self._seed,
            schema_version=self._config.schema_version,
            threshold_version=str(_resolve_threshold_version(eda_artifact, self._config)),
            estimator=estimator,
        )
        return self

    # ------------------------------------------------------------------ #
    # Scoring (Req 6.3, 6.4, 6.5, 6.8)
    # ------------------------------------------------------------------ #
    def predict_risk(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> List[Tuple[str, float]]:
        """Score every eval pair -> ``[(pair_id, risk_score in [0,1]), ...]`` in input order.

        Coverage: exactly one score per input pair (Req 6.4). Unscoreable pairs get the pool-prior
        default (Req 6.4). Scores are per-row, so input order never changes any pair's score
        (Req 6.5). HALTS on empty input or a feature-name mismatch vs the trained schema (Req 6.8).
        """
        self._require_fitted()
        params = self._params
        assert params is not None

        if eval_feature_sets is None or len(eval_feature_sets) == 0:
            raise SchemaError("pu_ranker.predict_risk(eval_feature_sets)", "non-empty evaluation pairs")

        # Schema guard (Req 6.8): any feature name the model never trained on is a mismatch.
        trained = set(params.columns)
        for fs in eval_feature_sets:
            unseen = set(fs.features.keys()) - trained
            if unseen:
                raise SchemaError(
                    f"pu_ranker.predict_risk(pair {fs.pair_id})",
                    f"unseen feature column(s) not in trained schema: {sorted(unseen)}",
                )

        X = build_feature_matrix(
            eval_feature_sets,
            params.columns,
            include_classical_score=params.include_classical_score,
            eda_artifact=eda_artifact,
        )
        g = params.estimator.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        # Elkan-Noto correction f(x) = g(x) / c, clamped to [0, 1] (monotone in g => ranking kept).
        f = g / params.label_frequency

        out: List[Tuple[str, float]] = []
        for i, fs in enumerate(eval_feature_sets):
            if _is_unscoreable(fs, params.min_shared_hands):
                risk = params.pool_prior
            else:
                risk = float(f[i])
            risk = min(max(risk, 0.0), 1.0)
            out.append((str(fs.pair_id), risk))
        return out

    def score_pairs(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> Dict[str, float]:
        """Convenience: same as :meth:`predict_risk` but returns a ``{pair_id -> risk}`` dict."""
        return {pid: risk for pid, risk in self.predict_risk(eval_feature_sets, eda_artifact=eda_artifact)}

    # ------------------------------------------------------------------ #
    # Persistence (Req 6.7)
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> Path:
        """Persist the fitted model (estimator + schema + seed + versions) via joblib.

        Atomic temp-then-rename so a partially written artifact is never observed. HALTS if the
        model is not fitted (nothing to persist).
        """
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
    ) -> "PURanker":
        """Load a model persisted by :meth:`save`; reproduces byte-identical scores (Req 6.7)."""
        import joblib

        source = Path(path)
        if not source.exists():
            raise SchemaError(str(source), "existing persisted PU model file")
        params: PUParams = joblib.load(source)
        obj = cls(
            config=config,
            include_classical_score=params.include_classical_score,
            seed=params.seed,
        )
        obj._params = params
        return obj

    # ------------------------------------------------------------------ #
    # JSON-friendly param round-trip (documented inspection path)
    # ------------------------------------------------------------------ #
    def to_params(self) -> Dict[str, object]:
        """Return a JSON-serialisable dict of the linear coefficients + metadata (inspection)."""
        self._require_fitted()
        p = self._params
        assert p is not None
        est = p.estimator
        coef = getattr(est, "coef_", None)
        intercept = getattr(est, "intercept_", None)
        return {
            "columns": list(p.columns),
            "label_frequency": p.label_frequency,
            "pool_prior": p.pool_prior,
            "min_shared_hands": p.min_shared_hands,
            "include_classical_score": p.include_classical_score,
            "seed": p.seed,
            "schema_version": p.schema_version,
            "threshold_version": p.threshold_version,
            "coef": coef.tolist() if coef is not None else None,
            "intercept": intercept.tolist() if intercept is not None else None,
        }


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
# Module-level convenience
# --------------------------------------------------------------------------- #
def train_pu_ranker(
    train_feature_sets: Sequence[PairFeatureSet],
    labels: Union[LabelTable, Mapping[str, str]],
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    config: Optional[PipelineConfig] = None,
    include_classical_score: bool = True,
    seed: Optional[int] = None,
) -> PURanker:
    """Fit and return a :class:`PURanker` in one call (documented convenience wrapper)."""
    ranker = PURanker(
        config=config,
        include_classical_score=include_classical_score,
        seed=seed,
    )
    return ranker.fit(train_feature_sets, labels, eda_artifact=eda_artifact)
