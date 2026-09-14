"""Behavior classifier with shared ``risk_score`` and none-thresholding (task 17.1).

Requirements 7.1-7.6, 12.3; Design "Behavior_Classifier"; ``RESEARCH_DOSSIER.md`` §1.3.

This module assigns every evaluation :class:`~poker_collusion.types.PairFeatureSet` exactly one
:class:`~poker_collusion.types.BehaviorLabel` in
``{none, directed_transfer, soft_play, coordinated_isolation, other_coordination}`` (Req 7.1) and
exposes the per-family score accounting used by **Behavior MAP** (Req 7.4, 12.3).

The classifier decides the LABEL only; it does NOT recompute the risk_score
-----------------------------------------------------------------------------
The ``risk_score`` is produced once by :mod:`poker_collusion.models.pu_ranker` and is a **shared
lever** across Pair AP and Behavior MAP (dossier Exploit E1). :meth:`BehaviorClassifier.predict`
takes that ``risk_score`` in as a ``{pair_id -> float}`` mapping and never re-derives it. The
disclosed-family head only picks *which* label a coordinated pair gets.

Routing rules (defaults documented; all thresholds configurable — task 18 calibrates them)
------------------------------------------------------------------------------------------
For each pair, given its shared ``risk_score`` ``r`` and the disclosed-family head's calibrated
class probabilities:

1. **none-thresholding (Req 7.6, Property 8 — behavior only; test is task 17.2).**
   ``r < coordination_threshold`` -> :data:`BehaviorLabel.NONE`. Default
   :data:`DEFAULT_COORDINATION_THRESHOLD`.
2. **Route coordinated-but-untemplated pairs to ``other_coordination`` (Req 7.3).**
   A pair with ``r >= coordination_threshold`` is coordinated; if the disclosed-family head is
   **ambiguous** — the top class probability minus the runner-up (the decision *margin*) is below
   :data:`other_coordination_margin` (default :data:`DEFAULT_OTHER_COORDINATION_MARGIN`) — the pair
   does not confidently fit one of the three Disclosed_Families, so it is routed to the
   behavior-agnostic catch-all :data:`BehaviorLabel.OTHER_COORDINATION`.
3. **Confident disclosed family.** Otherwise the pair is labelled with the head's argmax
   disclosed family (``directed_transfer`` / ``soft_play`` / ``coordinated_isolation``).

Per-family scoring for Behavior MAP (Req 7.4, 12.3; dossier §1.3)
-----------------------------------------------------------------
Behavior MAP is a macro one-vs-rest AP over **exactly the three** Disclosed_Families;
``other_coordination`` and ``none`` are **never** OvR classes (Req 7.5). For each disclosed family
the per-pair OvR score is the shared ``risk_score`` when that family is predicted **else hard 0**
(``np.where(predicted == family, risk, 0.0)``). :meth:`per_family_scores` returns
``{family -> {pair_id -> risk_if_predicted_else_0}}`` for the three disclosed families only
(``other_coordination`` / ``none`` excluded), ready to feed Behavior MAP.

Consequence exposed in the API (Req 7.5): because ``other_coordination`` is excluded from the
3-way macro, routing a **true** disclosed-family pair to ``other_coordination`` (or ``none``) is a
pure Behavior-MAP **loss** — that pair scores 0 in the family it truly belongs to. So the routing
margin is a tunable trade-off: liberal ``other_coordination`` is *free* against the metric only
for pairs that are NOT truly a disclosed family (dossier Exploit E2). The margin is configurable
precisely so task 18 can calibrate it against this loss.

Feature-matrix / schema-guard / persistence conventions
--------------------------------------------------------
Mirrors :mod:`poker_collusion.models.pu_ranker` exactly: a fixed, sorted feature-column schema
(``sorted(set(...))``); missing feature -> neutral ``0.0``; NaN/inf -> ``0.0``; an unseen feature
column at predict time raises :class:`~poker_collusion.exceptions.SchemaError` (Req: HALT). All
randomness is seeded from :data:`poker_collusion.config.Seeds.behavior_classifier`. :meth:`save` /
:meth:`load` persist the fitted estimator, feature columns, class order, seed, and schema/threshold
versions via ``joblib`` (atomic temp-then-rename), reproducing identical labels.

HALT (no partial state)
-----------------------
:meth:`fit` raises before mutating ``self`` on empty training feature sets, no feature columns, or
**no Trusted_Positive** carrying a disclosed family. :meth:`predict` raises on an empty eval set,
a feature-schema mismatch, or a ``risk_scores`` mapping missing a pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from poker_collusion.config import ExploitConfig, PipelineConfig, get_config
from poker_collusion.exceptions import SchemaError
from poker_collusion.models.pu_ranker import build_feature_matrix
from poker_collusion.types import (
    DISCLOSED_FAMILIES,
    BehaviorLabel,
    LabelTable,
    PairFeatureSet,
)

__all__ = [
    "DEFAULT_COORDINATION_THRESHOLD",
    "DEFAULT_OTHER_COORDINATION_MARGIN",
    "DISCLOSED_FAMILY_TOKENS",
    "BehaviorClassifier",
    "train_behavior_classifier",
]

# --------------------------------------------------------------------------- #
# Documented defaults (configurable; task 18 calibrates them)
# --------------------------------------------------------------------------- #
#: Default coordination threshold for none-thresholding (Req 7.6). A pair whose shared
#: ``risk_score`` is below this is labelled ``none``. Chosen as a modest default well above the
#: pool prior so only pairs the PU ranker lifts materially are treated as coordinated; task 18
#: calibrates it on local CV.
DEFAULT_COORDINATION_THRESHOLD: float = 0.5

#: Default decision-margin below which a coordinated pair's disclosed-family prediction is treated
#: as too ambiguous to name a family, and the pair is routed to ``other_coordination`` (Req 7.3).
#: The margin is ``p(top family) - p(runner-up family)`` from the calibrated head. Default kept
#: small so only genuinely ambiguous pairs are diverted (routing a TRUE disclosed-family pair to
#: ``other_coordination`` is a Behavior-MAP loss, Req 7.5); task 18 calibrates it.
DEFAULT_OTHER_COORDINATION_MARGIN: float = 0.15

#: The three Disclosed_Family label tokens, in a fixed deterministic order. ``other_coordination``
#: and ``none`` are deliberately absent (excluded from Behavior MAP, Req 7.5; dossier §1.3).
DISCLOSED_FAMILY_TOKENS: Tuple[str, ...] = tuple(str(f) for f in DISCLOSED_FAMILIES)


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #
def _disclosed_family_labels(
    labels: Union[LabelTable, Mapping[str, str]],
) -> Dict[str, str]:
    """Return ``{pair_id -> disclosed_family_token}`` for Trusted_Positive pairs.

    Accepts a :class:`LabelTable` (``trusted_positive`` maps pair_id -> behavior family) or a plain
    ``{pair_id -> behavior_family}`` map. Only the three Disclosed_Families are kept; any other
    token (e.g. ``other_coordination``, ``none``, or an unknown status) is dropped — the head is
    trained solely to distinguish the three disclosed families (Req 7.2).
    """
    if isinstance(labels, LabelTable):
        raw = labels.trusted_positive
    else:
        raw = labels
    allowed = set(DISCLOSED_FAMILY_TOKENS)
    out: Dict[str, str] = {}
    for pid, fam in raw.items():
        token = str(fam)
        if token in allowed:
            out[str(pid)] = token
    return out


# --------------------------------------------------------------------------- #
# Persisted parameter carrier
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorParams:
    """Everything needed to reproduce labelling exactly (persisted via joblib).

    Attributes:
        columns: Fixed feature-column order (the trained schema; mirrors pu_ranker).
        classes: The disclosed-family class tokens in the estimator's ``classes_`` order.
        seed: The seed the estimator was fit under (:class:`Seeds.behavior_classifier`).
        schema_version / threshold_version: Feature schema identifiers (provenance).
        estimator: The fitted scikit-learn multiclass estimator (pickled by joblib), or ``None``
            for the degenerate single-family fallback.
        single_family: The lone disclosed family when only one family was present in training
            (no discriminative head can be fit); ``None`` otherwise.
    """

    columns: List[str]
    classes: List[str]
    seed: int
    schema_version: str
    threshold_version: str
    estimator: object = None
    single_family: Optional[str] = None


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #
class BehaviorClassifier:
    """Assigns one :class:`BehaviorLabel` per pair using the shared ``risk_score`` for routing.

    See the module docstring for the routing rules (none-thresholding, other_coordination
    routing, confident disclosed family), the per-family scoring for Behavior MAP, and the
    feature-matrix / schema-guard / persistence conventions (mirroring
    :class:`~poker_collusion.models.pu_ranker.PURanker`).
    """

    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config or get_config()
        self._seed = int(seed if seed is not None else self._config.seeds.behavior_classifier)
        self._params: Optional[BehaviorParams] = None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        p = self._params
        return p is not None and (p.estimator is not None or p.single_family is not None)

    @property
    def columns(self) -> List[str]:
        self._require_fitted()
        return list(self._params.columns)  # type: ignore[union-attr]

    @property
    def classes(self) -> List[str]:
        self._require_fitted()
        return list(self._params.classes)  # type: ignore[union-attr]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise SchemaError(
                "BehaviorClassifier", "fitted_estimator (call fit() or load() first)"
            )

    # ------------------------------------------------------------------ #
    # Training (Req 7.2) — HALT (no partial state) on missing/empty input
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_feature_sets: Sequence[PairFeatureSet],
        label_table: Union[LabelTable, Mapping[str, str]],
        *,
        eda_artifact: Optional[Union[Mapping, object]] = None,
    ) -> "BehaviorClassifier":
        """Fit the disclosed-family head on Trusted_Positive labels (Req 7.2).

        Trains a deterministic multinomial :class:`~sklearn.linear_model.LogisticRegression` over
        the PairFeatureSet feature matrix (same style as pu_ranker) to distinguish the three
        Disclosed_Families. Only Trusted_Positive pairs carrying one of the three disclosed families
        are used as training rows; ``other_coordination`` / ``none`` / unknown pairs are not head
        training data (the head only names disclosed families; routing happens at predict time).

        HALTS (raises, no partial state) when: ``train_feature_sets`` is empty, no feature columns
        exist, or no Trusted_Positive carries a disclosed family.
        """
        from sklearn.linear_model import LogisticRegression

        if train_feature_sets is None or len(train_feature_sets) == 0:
            raise SchemaError(
                "behavior_classifier.fit(train_feature_sets)", "non-empty training feature sets"
            )

        family_by_pair = _disclosed_family_labels(label_table)
        if not family_by_pair:
            raise SchemaError(
                "behavior_classifier.fit(label_table)",
                "at least one Trusted_Positive carrying a disclosed behavior family",
            )

        # Fixed, sorted feature-column schema (row-order independent; mirrors pu_ranker). The
        # classifier does NOT append the classical-score column — it labels from the raw feature
        # layer and takes the shared risk_score in at predict time (never recomputes it).
        feature_names: set = set()
        for fs in train_feature_sets:
            feature_names.update(fs.features.keys())
        if not feature_names:
            raise SchemaError("behavior_classifier.fit(features)", "at least one feature column")
        columns = sorted(feature_names)

        # Training rows = Trusted_Positive pairs with a disclosed family, in input order.
        rows: List[PairFeatureSet] = []
        y: List[str] = []
        for fs in train_feature_sets:
            fam = family_by_pair.get(str(fs.pair_id))
            if fam is not None:
                rows.append(fs)
                y.append(fam)

        present_families = sorted(set(y))

        if len(present_families) < 2:
            # Degenerate: only one disclosed family present -> a discriminative head is meaningless.
            # Record the lone family; predict() will assign it to every confident coordinated pair.
            self._params = BehaviorParams(
                columns=list(columns),
                classes=list(present_families),
                seed=self._seed,
                schema_version=self._config.schema_version,
                threshold_version=str(_resolve_threshold_version(eda_artifact, self._config)),
                estimator=None,
                single_family=present_families[0],
            )
            return self

        X = build_feature_matrix(
            rows,
            columns,
            include_classical_score=False,
            eda_artifact=eda_artifact,
        )

        # sklearn's ``lbfgs`` solver fits softmax (multinomial) multiclass by default, which is
        # deterministic under a fixed seed/solver — exactly what we want (the ``multi_class`` kwarg
        # was removed in newer sklearn; multinomial is now the default for lbfgs).
        estimator = LogisticRegression(
            random_state=self._seed,
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
        )
        estimator.fit(X, np.asarray(y, dtype=object))

        self._params = BehaviorParams(
            columns=list(columns),
            classes=[str(c) for c in estimator.classes_],
            seed=self._seed,
            schema_version=self._config.schema_version,
            threshold_version=str(_resolve_threshold_version(eda_artifact, self._config)),
            estimator=estimator,
            single_family=None,
        )
        return self

    # ------------------------------------------------------------------ #
    # Prediction (Req 7.1, 7.3, 7.6)
    # ------------------------------------------------------------------ #
    def _schema_guard(self, eval_feature_sets: Sequence[PairFeatureSet]) -> None:
        """Any feature name the model never trained on is a schema mismatch (HALT)."""
        trained = set(self._params.columns)  # type: ignore[union-attr]
        for fs in eval_feature_sets:
            unseen = set(fs.features.keys()) - trained
            if unseen:
                raise SchemaError(
                    f"behavior_classifier.predict(pair {fs.pair_id})",
                    f"unseen feature column(s) not in trained schema: {sorted(unseen)}",
                )

    def _family_proba(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        *,
        eda_artifact: Optional[Union[Mapping, object]],
    ) -> Tuple[np.ndarray, List[str]]:
        """Return ``(proba[n, k], class_tokens)`` for the disclosed-family head.

        For the degenerate single-family fallback there is no estimator; return a one-column
        certainty matrix for the lone family (margin defined as 1.0 so it is never ambiguous).
        """
        params = self._params
        assert params is not None
        if params.estimator is None:
            fam = params.single_family or DISCLOSED_FAMILY_TOKENS[0]
            proba = np.ones((len(eval_feature_sets), 1), dtype=np.float64)
            return proba, [fam]
        X = build_feature_matrix(
            eval_feature_sets,
            params.columns,
            include_classical_score=False,
            eda_artifact=eda_artifact,
        )
        proba = params.estimator.predict_proba(X)  # type: ignore[union-attr]
        return proba, list(params.classes)

    def predict(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        risk_scores: Mapping[str, float],
        *,
        coordination_threshold: float = DEFAULT_COORDINATION_THRESHOLD,
        other_coordination_margin: float = DEFAULT_OTHER_COORDINATION_MARGIN,
        eda_artifact: Optional[Union[Mapping, object]] = None,
        exploits: Optional[ExploitConfig] = None,
    ) -> List[Tuple[str, BehaviorLabel]]:
        """Assign one :class:`BehaviorLabel` per eval pair (coverage; Req 7.1), in input order.

        Uses the SHARED ``risk_scores`` mapping (from the PU ranker) for routing — it is never
        recomputed here. Rules (see module docstring):

        * ``risk < coordination_threshold`` -> ``none`` (Req 7.6).
        * coordinated but ambiguous head (top-minus-runner-up margin ``< other_coordination_margin``)
          -> ``other_coordination`` (Req 7.3).
        * otherwise -> the head's argmax disclosed family.

        Discovery-Exploit wiring (task 18; Req 12.1/12.3/12.4). An optional :class:`ExploitConfig`
        (defaults to the pipeline config's) toggles two ledger adoptions deterministically:

        * **E1** (``e1_risk_calibration``) — the shared ``risk`` is monotonically recalibrated via
          :meth:`ExploitConfig.calibrate_risk` before the ``none``-threshold comparison. Strictly
          monotone, so it never reorders pairs; it only re-spaces the levels (Pair-AP-safe).
        * **E2** (``e2_liberal_other_coordination``) — the effective ``other_coordination`` routing
          margin is widened by :attr:`ExploitConfig.effective_other_coordination_margin_delta`, so
          more ambiguous coordinated pairs are routed to the (Behavior-MAP-excluded) catch-all.

        Both are OFF by default unless CONFIRMED in the ledger (dossier §4). HALTS on an empty eval
        set, a feature-schema mismatch, or a ``risk_scores`` mapping missing a pair (no partial
        output).
        """
        self._require_fitted()
        exploits = exploits if exploits is not None else self._config.exploits
        effective_margin = float(other_coordination_margin) + (
            exploits.effective_other_coordination_margin_delta if exploits is not None else 0.0
        )

        if eval_feature_sets is None or len(eval_feature_sets) == 0:
            raise SchemaError(
                "behavior_classifier.predict(eval_feature_sets)", "non-empty evaluation pairs"
            )
        if risk_scores is None:
            raise SchemaError("behavior_classifier.predict(risk_scores)", "a risk_scores mapping")

        # Every pair must carry a shared risk_score (coverage of the shared lever).
        missing = [str(fs.pair_id) for fs in eval_feature_sets if str(fs.pair_id) not in risk_scores]
        if missing:
            raise SchemaError(
                "behavior_classifier.predict(risk_scores)",
                f"risk_score for pair(s): {sorted(missing)}",
            )

        self._schema_guard(eval_feature_sets)

        proba, class_tokens = self._family_proba(eval_feature_sets, eda_artifact=eda_artifact)

        out: List[Tuple[str, BehaviorLabel]] = []
        for i, fs in enumerate(eval_feature_sets):
            pid = str(fs.pair_id)
            # E1: monotone recalibration of the shared risk (identity when the flag is off).
            risk = (
                exploits.calibrate_risk(float(risk_scores[pid]))
                if exploits is not None
                else float(risk_scores[pid])
            )

            # 1. none-thresholding (Req 7.6, Property 8 behavior).
            if risk < float(coordination_threshold):
                out.append((pid, BehaviorLabel.NONE))
                continue

            # Coordinated: inspect the disclosed-family head.
            row = proba[i]
            top_idx = int(np.argmax(row))
            top_p = float(row[top_idx])
            if row.shape[0] >= 2:
                runner_up = float(np.partition(row, -2)[-2])
            else:
                runner_up = 0.0
            margin = top_p - runner_up

            # 2. ambiguous -> other_coordination catch-all (Req 7.3); E2 widens the margin.
            if margin < effective_margin:
                out.append((pid, BehaviorLabel.OTHER_COORDINATION))
                continue

            # 3. confident disclosed family.
            out.append((pid, BehaviorLabel(class_tokens[top_idx])))
        return out

    def predict_labels(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        risk_scores: Mapping[str, float],
        *,
        coordination_threshold: float = DEFAULT_COORDINATION_THRESHOLD,
        other_coordination_margin: float = DEFAULT_OTHER_COORDINATION_MARGIN,
        eda_artifact: Optional[Union[Mapping, object]] = None,
        exploits: Optional[ExploitConfig] = None,
    ) -> Dict[str, BehaviorLabel]:
        """Convenience: :meth:`predict` as a ``{pair_id -> BehaviorLabel}`` dict."""
        return {
            pid: label
            for pid, label in self.predict(
                eval_feature_sets,
                risk_scores,
                coordination_threshold=coordination_threshold,
                other_coordination_margin=other_coordination_margin,
                eda_artifact=eda_artifact,
                exploits=exploits,
            )
        }

    # ------------------------------------------------------------------ #
    # Per-family scoring for Behavior MAP (Req 7.4, 7.5, 12.3; dossier §1.3)
    # ------------------------------------------------------------------ #
    def per_family_scores(
        self,
        eval_feature_sets: Sequence[PairFeatureSet],
        risk_scores: Mapping[str, float],
        *,
        coordination_threshold: float = DEFAULT_COORDINATION_THRESHOLD,
        other_coordination_margin: float = DEFAULT_OTHER_COORDINATION_MARGIN,
        eda_artifact: Optional[Union[Mapping, object]] = None,
        exploits: Optional[ExploitConfig] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return ``{family -> {pair_id -> risk_if_predicted_else_0}}`` for Behavior MAP.

        Only the three Disclosed_Families are keys (``other_coordination`` and ``none`` are
        EXCLUDED, Req 7.5 / dossier §1.3). For each family the per-pair OvR score is the shared
        ``risk_score`` when that family is the predicted label, else **hard 0**
        (``np.where(predicted == family, risk, 0.0)``). A pair routed to ``other_coordination`` or
        ``none`` therefore scores 0 in every disclosed family — the documented scoring loss for
        misrouting a true disclosed-family pair (Req 7.5).

        The per-family OvR score uses the SAME E1-recalibrated risk the routing used when the E1
        exploit is enabled (task 18), so Behavior MAP sees the recalibrated levels consistently.
        """
        exploits = exploits if exploits is not None else self._config.exploits
        labels = self.predict(
            eval_feature_sets,
            risk_scores,
            coordination_threshold=coordination_threshold,
            other_coordination_margin=other_coordination_margin,
            eda_artifact=eda_artifact,
            exploits=exploits,
        )
        family_maps: Dict[str, Dict[str, float]] = {
            fam: {} for fam in DISCLOSED_FAMILY_TOKENS
        }
        for pid, label in labels:
            token = str(label)
            risk = (
                exploits.calibrate_risk(float(risk_scores[pid]))
                if exploits is not None
                else float(risk_scores[pid])
            )
            for fam in DISCLOSED_FAMILY_TOKENS:
                family_maps[fam][pid] = risk if token == fam else 0.0
        return family_maps

    # ------------------------------------------------------------------ #
    # Persistence (mirrors pu_ranker; atomic temp-then-rename)
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> Path:
        """Persist the fitted classifier (estimator + schema + classes + seed + versions)."""
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
    ) -> "BehaviorClassifier":
        """Load a classifier persisted by :meth:`save`; reproduces identical labels."""
        import joblib

        source = Path(path)
        if not source.exists():
            raise SchemaError(str(source), "existing persisted behavior classifier file")
        params: BehaviorParams = joblib.load(source)
        obj = cls(config=config, seed=params.seed)
        obj._params = params
        return obj

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
            "classes": list(p.classes),
            "single_family": p.single_family,
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
def train_behavior_classifier(
    train_feature_sets: Sequence[PairFeatureSet],
    label_table: Union[LabelTable, Mapping[str, str]],
    *,
    eda_artifact: Optional[Union[Mapping, object]] = None,
    config: Optional[PipelineConfig] = None,
    seed: Optional[int] = None,
) -> BehaviorClassifier:
    """Fit and return a :class:`BehaviorClassifier` in one call (documented convenience wrapper)."""
    clf = BehaviorClassifier(config=config, seed=seed)
    return clf.fit(train_feature_sets, label_table, eda_artifact=eda_artifact)
