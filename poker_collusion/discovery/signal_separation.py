"""Phase -1 Discovery VALIDATION: measure how well each feature separates confirmed-target
from confirmed-non-target dev pairs on the REAL development labels (post-hoc).

Why this module exists
----------------------
The Phase -1 HARD GATE (RESEARCH_DOSSIER standing rule) was meant to block modeling until the
Discovery findings were *measured*. In practice the confounder-contrast hypotheses (H14-H17) were
written down but left ``OPEN -- PENDING DATA`` and never measured once the real data arrived; two
real submissions then scored *below* the do-nothing baseline (Phase-1 0.05713, Phase-2 0.06236 vs
sample 0.08404). Since the leaderboard metric is 70% Pair AP (a RANKING), the whole problem reduces
to: **does any feature we already compute actually separate positive from negative pairs?**

This module answers that with numbers, using the SAME bounded, leak-safe machinery the phase gates
use, so it is honest and reproducible:

* one bounded pass over ``actions.parquet`` for the confirmed dev pairs (the floor's
  ``_prepare_labelled_inputs`` cached-inputs pattern -- the 18M-row file is never materialised
  whole);
* a DEVELOPMENT-phase :class:`~poker_collusion.types.PairFeatureSet` per confirmed pair
  (``build_pair_feature_set`` with ``phase="development"``), exactly like Phase-1/Phase-2;
* group-by-pool folds (pool = ``hands.table_id`` via ``derive_pool_map``) so within-pool leakage
  is not over-read.

What it measures (per feature + the current classical risk_score)
-----------------------------------------------------------------
For every numeric feature in the feature dict AND for the current classical
:func:`~poker_collusion.models.classical.score_pair` risk:

* **ROC AUC (whole-sample, orientation-max).** ``roc_auc_score(y, x)`` and ``roc_auc_score(y, -x)``;
  we keep the max and record the orientation (``+`` = higher is more suspicious, ``-`` = lower is).
* **Group-by-pool CV AUC (mean +/- std).** Leave-one-pool-out (or K grouped folds) AUC, computed on
  the held-out pool each time with the orientation fixed from the whole-sample fit, so we do not
  over-read within-pool structure. Folds where the held-out pool has only one class are skipped.
* **Cohen's d.** ``(mean_pos - mean_neg) / pooled_std`` -- a standardized separation.

Features are ranked by CV AUC descending. A **multivariate upper bound** is added: ONE logistic
regression over ALL features with the same group-by-pool CV, whose mean held-out AUC is roughly the
best a linear model over the CURRENT features can do.

A couple of CHEAP derived candidates (from what the caches already expose) are also measured:
partner-vs-field directedness contrast magnitude, co-seat repetition rate, and multi-hand directed-
flow consistency. Anything that would need data we do not cheaply have (e.g. action latency/timing)
is recorded as ``NOT MEASURED -- needs X`` rather than faked.

This is a BOUNDED investigation: NO production pipeline changes, NO submission, NO score-chasing.
The ``__main__`` runs it on the real data and writes a plain-ASCII findings report to disk (the
execute shell swallows stdout, so the report must be READ from the file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from poker_collusion.config import PipelineConfig, get_config
from poker_collusion.features.eda import EdaArtifact
from poker_collusion.io import DataLoader
from poker_collusion.models.classical import score_pair
from poker_collusion.pipeline import _obtain_eda_artifact
from poker_collusion.validation.cv import (
    CONFIRMED_NON_TARGET,
    CONFIRMED_TARGET,
    derive_pool_map,
)
from poker_collusion.validation.phase1_baseline_cv import (
    DEVELOPMENT_PHASE,
    _LabelledInputs,
    _prepare_labelled_inputs,
    _signals_for_pair,
)
from poker_collusion.features.pair_features import build_pair_feature_set

try:  # sklearn is a declared dependency (1.9.0); guard only to give a clear error.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except Exception:  # pragma: no cover - exercised only if sklearn is missing
    _SKLEARN = False

__all__ = [
    "FeatureSeparation",
    "SeparationReport",
    "auc_orientation_max",
    "cohens_d",
    "grouped_cv_auc",
    "build_confirmed_feature_matrix",
    "measure_separation",
    "measure_separation_from_real_data",
    "format_report",
    "write_report",
]

# The classical risk pseudo-feature name (kept distinct from the raw feature dict keys).
CLASSICAL_RISK_KEY = "classical_risk_score"

# A ~0.65 CV-AUC bar is the "plausibly lifts PairAP above baseline" threshold the task names.
USEFUL_AUC_BAR = 0.65


# --------------------------------------------------------------------------- #
# Core statistics (pure; unit-tested on synthetic data)
# --------------------------------------------------------------------------- #
def auc_orientation_max(y: np.ndarray, x: np.ndarray) -> Tuple[float, str]:
    """Return ``(max AUC over both orientations, orientation)`` for a scalar score ``x``.

    ``orientation == "+"`` means higher ``x`` => more suspicious (AUC of ``x`` as a score);
    ``"-"`` means lower ``x`` => more suspicious (AUC of ``-x``). Constant / degenerate scores
    return ``(0.5, "+")``. NaNs are treated as the column mean so a feature is never dropped
    silently.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x = _fill_nan(x)
    if len(np.unique(y)) < 2 or np.allclose(x, x[0]):
        return 0.5, "+"
    a_pos = float(roc_auc_score(y, x))
    a_neg = float(roc_auc_score(y, -x))
    if a_pos >= a_neg:
        return a_pos, "+"
    return a_neg, "-"


def cohens_d(y: np.ndarray, x: np.ndarray) -> float:
    """Standardized mean separation ``(mean_pos - mean_neg) / pooled_std`` (0.0 if undefined)."""
    y = np.asarray(y, dtype=float)
    x = _fill_nan(np.asarray(x, dtype=float))
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    vp, vn = np.var(pos, ddof=1), np.var(neg, ddof=1)
    np_, nn = len(pos), len(neg)
    pooled = math.sqrt(((np_ - 1) * vp + (nn - 1) * vn) / max(np_ + nn - 2, 1))
    if pooled <= 0.0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def grouped_cv_auc(
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
    orientation: str = "+",
) -> Tuple[float, float, int]:
    """Leave-one-group-out CV AUC for a scalar score, orientation fixed.

    For each distinct group we hold it out, orient the score (``x`` or ``-x``), and compute the
    held-out AUC on that group's members. Groups whose held-out members are single-class are
    skipped (AUC undefined there). Returns ``(mean, std, n_folds_used)``; ``(0.5, 0.0, 0)`` when no
    fold is scoreable.
    """
    y = np.asarray(y, dtype=float)
    x = _fill_nan(np.asarray(x, dtype=float))
    groups = np.asarray(groups)
    sign = 1.0 if orientation == "+" else -1.0
    xs = sign * x

    aucs: List[float] = []
    for g in _stable_unique(groups):
        mask = groups == g
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            continue
        xx = xs[mask]
        if np.allclose(xx, xx[0]):
            aucs.append(0.5)
            continue
        aucs.append(float(roc_auc_score(yy, xx)))
    if not aucs:
        return 0.5, 0.0, 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with the finite column mean (or 0.0 if none finite)."""
    x = np.array(x, dtype=float, copy=True)
    finite = np.isfinite(x)
    if not finite.all():
        fill = float(x[finite].mean()) if finite.any() else 0.0
        x[~finite] = fill
    return x


def _stable_unique(values: np.ndarray) -> List:
    """Distinct values in first-seen order (deterministic; groups may be strings)."""
    seen: set = set()
    out: List = []
    for v in values.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Result carriers
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSeparation:
    """Discriminative power of one feature separating positive vs negative pairs."""

    name: str
    auc_whole: float
    orientation: str
    cv_auc_mean: float
    cv_auc_std: float
    cv_folds: int
    cohens_d: float
    mean_pos: float
    mean_neg: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "auc_whole": self.auc_whole,
            "orientation": self.orientation,
            "cv_auc_mean": self.cv_auc_mean,
            "cv_auc_std": self.cv_auc_std,
            "cv_folds": self.cv_folds,
            "cohens_d": self.cohens_d,
            "mean_pos": self.mean_pos,
            "mean_neg": self.mean_neg,
        }


@dataclass
class SeparationReport:
    """The full ranked separation table + multivariate upper bound + verdict."""

    features: List[FeatureSeparation]
    multivariate_cv_auc_mean: float
    multivariate_cv_auc_std: float
    multivariate_folds: int
    n_pos: int
    n_neg: int
    n_pools: int
    best_feature: Optional[str]
    best_cv_auc: float
    useful_bar: float = USEFUL_AUC_BAR
    caveats: List[str] = field(default_factory=list)
    not_measured: List[str] = field(default_factory=list)

    @property
    def verdict_useful(self) -> bool:
        """True iff any feature OR the multivariate model clears the useful CV-AUC bar."""
        return (self.best_cv_auc >= self.useful_bar) or (
            self.multivariate_cv_auc_mean >= self.useful_bar
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "features": [f.as_dict() for f in self.features],
            "multivariate_cv_auc_mean": self.multivariate_cv_auc_mean,
            "multivariate_cv_auc_std": self.multivariate_cv_auc_std,
            "multivariate_folds": self.multivariate_folds,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "n_pools": self.n_pools,
            "best_feature": self.best_feature,
            "best_cv_auc": self.best_cv_auc,
            "useful_bar": self.useful_bar,
            "verdict_useful": self.verdict_useful,
            "caveats": self.caveats,
            "not_measured": self.not_measured,
        }


# --------------------------------------------------------------------------- #
# Feature-matrix assembly (from already-built feature sets)
# --------------------------------------------------------------------------- #
def build_confirmed_feature_matrix(
    inputs: _LabelledInputs,
    labels: pd.DataFrame,
    pool_map: Mapping[str, object],
    artifact: Optional[EdaArtifact],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """Build the (feature matrix, y, groups, feature_names, derived_names) for confirmed pairs.

    One row per confirmed pair that has development shared hands. Columns are every numeric feature
    in the pair's development-phase :class:`PairFeatureSet` PLUS the classical risk pseudo-feature
    PLUS a couple of cheap derived candidates. ``y`` = 1 for confirmed_target, 0 for
    confirmed_non_target; ``groups`` = the pair's pool (``hands.table_id``). Pairs with no dev shared
    hands are dropped (no signal -- recorded as a caveat by the caller).
    """
    status_by_pair: Dict[str, str] = {}
    for r in labels.itertuples(index=False):
        st = str(getattr(r, "label_status", ""))
        if st in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}:
            status_by_pair[str(getattr(r, "pair_id"))] = st

    rows: List[Dict[str, float]] = []
    ys: List[int] = []
    grps: List[object] = []
    feat_names: set = set()

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
            config=config,
        )
        rec: Dict[str, float] = {k: float(v) for k, v in fs.features.items()}
        # Current classical risk (the thing that actually ranks Pair AP today).
        rec[CLASSICAL_RISK_KEY] = float(score_pair(fs, eda_artifact=artifact).risk_score)
        # Cheap derived candidates from what the feature dict already exposes.
        rec.update(_cheap_derived_candidates(fs.features))

        feat_names.update(rec.keys())
        rows.append(rec)
        ys.append(1 if status == CONFIRMED_TARGET else 0)
        grps.append(pool_map.get(str(pair_id), "UNKNOWN_POOL"))

    ordered = sorted(feat_names)
    if rows:
        frame = pd.DataFrame(rows, columns=ordered).fillna(0.0)
    else:
        frame = pd.DataFrame(columns=ordered)
    derived = sorted(_cheap_derived_candidates({}).keys())
    return frame, np.asarray(ys, dtype=int), np.asarray(grps, dtype=object), ordered, derived


def _cheap_derived_candidates(features: Mapping[str, float]) -> Dict[str, float]:
    """A couple of Dossier-listed ideas that are cheaply derivable from existing features.

    * ``deriv_directedness_abs`` -- |partner-vs-field directedness contrast| (H14). The signed
      ``conf_directedness_contrast`` is symmetric about 0 (A->B vs B->A), so its magnitude is the
      "how far from the field baseline in EITHER direction" candidate.
    * ``deriv_avoidance_gap_pos`` -- max(0, avoidance gap) (H15): only a POSITIVE gap (avoids the
      partner more than the field) is the collusion tell; negative gaps are not.
    * ``deriv_flow_consistency`` -- multi-hand directed-flow consistency: ``|value_flow_mean| /
      (value_flow_absmean + eps)`` in [0,1]. Near 1 => flow points the SAME way every hand
      (directed_transfer); near 0 => sign churns (variance/streaks).

    Passing ``{}`` returns the candidate keys with 0.0 so callers can enumerate the names.
    """
    eps = 1e-9

    def g(k: str) -> float:
        try:
            v = float(features.get(k, 0.0))
            return v if math.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    directedness_abs = abs(g("conf_directedness_contrast"))
    avoidance_gap_pos = max(0.0, g("conf_avoidance_gap"))
    absmean = g("value_flow_absmean")
    flow_consistency = abs(g("value_flow_mean")) / (absmean + eps) if absmean > 0 else 0.0
    return {
        "deriv_directedness_abs": float(directedness_abs),
        "deriv_avoidance_gap_pos": float(avoidance_gap_pos),
        "deriv_flow_consistency": float(min(max(flow_consistency, 0.0), 1.0)),
    }


# --------------------------------------------------------------------------- #
# Measurement over an already-built matrix (pure; unit-testable)
# --------------------------------------------------------------------------- #
def measure_separation(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    caveats: Optional[Sequence[str]] = None,
    not_measured: Optional[Sequence[str]] = None,
) -> SeparationReport:
    """Compute the per-feature separation table + multivariate upper bound from a matrix.

    Pure over its inputs (no I/O), so the unit test drives it on tiny synthetic data.
    """
    if not _SKLEARN:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for signal-separation measurement.")

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_pools = len(_stable_unique(groups))

    feats: List[FeatureSeparation] = []
    for name in frame.columns:
        x = frame[name].to_numpy(dtype=float)
        auc_whole, orientation = auc_orientation_max(y, x)
        cv_mean, cv_std, cv_folds = grouped_cv_auc(y, x, groups, orientation=orientation)
        d = cohens_d(y, x)
        xf = _fill_nan(x)
        mp = float(xf[y == 1].mean()) if n_pos else 0.0
        mn = float(xf[y == 0].mean()) if n_neg else 0.0
        feats.append(
            FeatureSeparation(
                name=name,
                auc_whole=auc_whole,
                orientation=orientation,
                cv_auc_mean=cv_mean,
                cv_auc_std=cv_std,
                cv_folds=cv_folds,
                cohens_d=d,
                mean_pos=mp,
                mean_neg=mn,
            )
        )

    feats.sort(key=lambda f: (f.cv_auc_mean, f.auc_whole), reverse=True)

    mv_mean, mv_std, mv_folds = _multivariate_cv_auc(frame, y, groups)

    best = feats[0] if feats else None
    return SeparationReport(
        features=feats,
        multivariate_cv_auc_mean=mv_mean,
        multivariate_cv_auc_std=mv_std,
        multivariate_folds=mv_folds,
        n_pos=n_pos,
        n_neg=n_neg,
        n_pools=n_pools,
        best_feature=best.name if best else None,
        best_cv_auc=best.cv_auc_mean if best else 0.5,
        caveats=list(caveats or []),
        not_measured=list(not_measured or []),
    )


def _multivariate_cv_auc(
    frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> Tuple[float, float, int]:
    """Group-by-pool CV AUC of ONE standardized logistic regression over ALL features.

    Trains on all-but-one pool, scores the held-out pool. Skips folds whose train OR held-out set
    is single-class. Returns ``(mean, std, n_folds)``; ``(0.5, 0.0, 0)`` if nothing scoreable.
    """
    if frame.shape[1] == 0 or len(np.unique(y)) < 2:
        return 0.5, 0.0, 0
    X = np.nan_to_num(frame.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)

    aucs: List[float] = []
    for g in _stable_unique(groups):
        test_mask = groups == g
        train_mask = ~test_mask
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        try:
            clf.fit(X[train_mask], y_tr)
            proba = clf.predict_proba(X[test_mask])[:, 1]
            aucs.append(float(roc_auc_score(y_te, proba)))
        except Exception:
            continue
    if not aucs:
        return 0.5, 0.0, 0
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)


# --------------------------------------------------------------------------- #
# Real-data driver (bounded; one actions pass)
# --------------------------------------------------------------------------- #
def measure_separation_from_real_data(
    *,
    config: Optional[PipelineConfig] = None,
    data_loader: Optional[DataLoader] = None,
    eda_artifact: Optional[EdaArtifact] = None,
    max_pairs: Optional[int] = None,
) -> SeparationReport:
    """Build confirmed dev-pair features in ONE bounded actions pass and measure separation.

    Bounded exactly like the Phase-1 floor: only confirmed labelled pairs' development shared hands
    are read, in a single streaming pass over ``actions.parquet`` (Req 1.6). ``max_pairs`` caps the
    labelled subset (recorded as a caveat) if the full confirmed-pair pass is too slow.
    """
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
            f"{total_labelled}) for tractability -- the same head-subset convention as the "
            f"Phase-1 floor / Phase-2 gate."
        )

    confirmed_pair_ids = [
        str(getattr(r, "pair_id"))
        for r in labels.itertuples(index=False)
        if getattr(r, "label_status", None) in {CONFIRMED_TARGET, CONFIRMED_NON_TARGET}
    ]

    inputs = _prepare_labelled_inputs(loader, labels, confirmed_pair_ids)
    pool_map = derive_pool_map(labels, config=cfg)

    frame, y, groups, feat_names, derived = build_confirmed_feature_matrix(
        inputs, labels, pool_map, artifact, cfg
    )

    n_dropped = len(confirmed_pair_ids) - len(frame)
    if n_dropped > 0:
        caveats.append(
            f"{n_dropped} confirmed pair(s) had no development shared hands (no signal) and were "
            f"excluded from the separation matrix."
        )

    not_measured = [
        "action timing / inter-action latency contrast -- NOT MEASURED: needs per-action wall-clock "
        "timestamps, which are not present in the cached action rows (only ordinal action_no).",
        "field-baseline change-point over time (H17 vs strategy changes) -- NOT MEASURED: needs the "
        "ordered per-player-vs-field signal series over time (the conf_field_baseline_change_stub "
        "is fixed at 0.0 at the pair-aggregation layer).",
    ]

    report = measure_separation(
        frame, y, groups, caveats=caveats, not_measured=not_measured
    )
    return report


# --------------------------------------------------------------------------- #
# ASCII report rendering
# --------------------------------------------------------------------------- #
def format_report(report: SeparationReport, *, top: Optional[int] = None) -> str:
    """Render the separation report as a plain-ASCII findings document."""
    lines: List[str] = []
    lines.append("SIGNAL SEPARATION FINDINGS (Phase -1 Discovery validation, post-hoc)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"Confirmed dev pairs measured: n_pos={report.n_pos}  n_neg={report.n_neg}  "
        f"n_pools={report.n_pools}"
    )
    lines.append(
        f"Useful CV-AUC bar (plausibly lifts PairAP above baseline): >= {report.useful_bar:.2f}"
    )
    lines.append("")
    lines.append("Per-feature discriminative power (ranked by group-by-pool CV AUC desc):")
    lines.append("-" * 78)
    header = f"{'rank':>4}  {'feature':<34} {'cvAUC':>7} {'+/-std':>7} {'d':>7} {'or':>3} {'whole':>7}"
    lines.append(header)
    lines.append("-" * 78)
    items = report.features if top is None else report.features[:top]
    for i, f in enumerate(items, start=1):
        lines.append(
            f"{i:>4}  {f.name[:34]:<34} {f.cv_auc_mean:>7.3f} {f.cv_auc_std:>7.3f} "
            f"{f.cohens_d:>7.3f} {f.orientation:>3} {f.auc_whole:>7.3f}"
        )
    lines.append("-" * 78)
    lines.append("")
    lines.append(
        f"MULTIVARIATE upper bound (logistic over ALL features, group-by-pool CV): "
        f"AUC = {report.multivariate_cv_auc_mean:.3f} +/- {report.multivariate_cv_auc_std:.3f} "
        f"({report.multivariate_folds} folds)"
    )
    lines.append("")
    lines.append("VERDICT")
    lines.append("-" * 78)
    best = report.best_feature or "(none)"
    lines.append(
        f"Best single feature: {best} @ CV AUC {report.best_cv_auc:.3f}.  "
        f"Multivariate CV AUC {report.multivariate_cv_auc_mean:.3f}."
    )
    if report.verdict_useful:
        lines.append(
            f"USEFUL SIGNAL FOUND: at least one feature or the multivariate model clears the "
            f">= {report.useful_bar:.2f} bar -- a monotone re-ranking on it could plausibly lift "
            f"PairAP above the do-nothing baseline. Worth revisiting the risk model."
        )
    else:
        lines.append(
            f"NO USEFUL SIGNAL: no single feature and not the multivariate model clears the "
            f">= {report.useful_bar:.2f} bar. Every measured feature separates positives from "
            f"negatives barely better than chance on group-by-pool CV. CEILING IMPLICATION: no "
            f"monotone re-ranking of the CURRENT features can materially lift Pair AP, which is "
            f"consistent with BOTH real submissions scoring below the do-nothing baseline. A "
            f"materially better score needs genuinely NEW discriminative signal, not re-weighting."
        )
    lines.append("")
    if report.caveats:
        lines.append("CAVEATS")
        lines.append("-" * 78)
        for c in report.caveats:
            lines.append(f"  - {c}")
        lines.append("")
    if report.not_measured:
        lines.append("NOT MEASURED (recorded, not faked)")
        lines.append("-" * 78)
        for c in report.not_measured:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines)


def write_report(report: SeparationReport, path: Path, *, top: Optional[int] = None) -> Path:
    """Write the ASCII findings report to ``path`` (ASCII-encoded) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = format_report(report, top=top)
    path.write_text(text, encoding="ascii", errors="replace")
    return path


# --------------------------------------------------------------------------- #
# __main__: run on the real data, write the ASCII report (read the FILE, not stdout)
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase -1 signal-separation measurement.")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap the number of labelled pairs measured (documented subset). Default: all.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/signal_separation_report.txt",
        help="Where to write the ASCII findings report.",
    )
    args = parser.parse_args()

    cfg = get_config()
    report = measure_separation_from_real_data(config=cfg, max_pairs=args.max_pairs)
    out_path = write_report(report, Path(args.out))
    # stdout may be swallowed; the FILE is the deliverable. Print a breadcrumb anyway.
    print(f"WROTE {out_path} (n_pos={report.n_pos} n_neg={report.n_neg} "
          f"best_cv_auc={report.best_cv_auc:.3f} mv={report.multivariate_cv_auc_mean:.3f})")


if __name__ == "__main__":
    _main()
