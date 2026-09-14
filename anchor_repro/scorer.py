"""Canonical local scorer for the poker-anchor-reproduction spec (task 2.2).

design.md "Components and Interfaces" §1 (CanonicalScorer), Req 1.2, 1.6, 2.5, 2.6.

The :class:`CanonicalScorer` is the SINGLE, auditable local scoring recipe under
which EVERY anchor (ours and competitors') is measured, so anchors are mutually
comparable and never mix regimes (Req 2.5, 2.6). It does NOT re-implement the
official metric, the CV split, or the PU unknown-handling — all three are REUSED
verbatim from the existing ``poker_collusion`` package (contract Rule 1, Req 1.6):

* :func:`poker_collusion.validation.cv.build_fold_solution` builds the PU-correct
  per-fold solution frame — unknown pairs are never materialised as negatives, so
  the PU handling is inherited, not reinvented.
* :func:`poker_collusion.metric.production_metric.score_components` is the verbatim
  official metric (validated bit-for-bit against the reference and re-verified
  against the published kernel by the task-2.1 self-check). ONE call returns BOTH
  ``pair_ap`` (the 70%-weighted PairAP intermediate — the PRIMARY gate) and
  ``combined`` (the full official combined score — the SECONDARY diagnostic) from a
  single evaluation, never two divergent recomputations or splits.

The load-bearing honest constraint (design.md, contract Rules 1 & 9): this scorer
scores predictions whose ``pair_id``\\s ARE dev-holdout pairs (the PU-stress
table-disjoint split), NOT a frozen EVAL artifact. The eval artifacts' pair sets are
DISJOINT from the dev labels, so no local metric can reproduce an absolute LB number
— that is impossible and is stated as impossible, never faked (Req 2.4).

Task 2.3 adds two things to this class:

* :meth:`CanonicalScorer.integrity_report` — the frozen-EVAL-artifact scoreability /
  integrity check (hash, row count 112,540, pair-set-equals-sample, non-degenerate
  ranking), mirroring ``poker_collusion.experiments.gate_a_verify_floor``. It does NOT
  dev-score the eval artifact (impossible; disjoint pairs) and surfaces the disjoint-set
  sizes between the artifact and the dev holdout explicitly.
* The belt-and-suspenders coverage-mismatch guard: :meth:`score_dev_predictions`
  pre-checks the prediction-vs-solution pair-id intersection (belt) AND catches the
  official metric's own ``ParticipantVisibleError`` coverage-mismatch raise (suspenders),
  re-surfacing both as an explicit :class:`CoverageMismatchError` whose message names how
  many pair-ids are missing and how many are extra — never a silent zero or a fabricated
  number (Property 5).

Hard gate (Req 1.2, 1.4): before scoring anything, the scorer runs the task-2.1
official-metric self-check via :func:`require_metric_matches_kernel`. If the local
metric has drifted from the published kernel, the self-check raises and the scorer
refuses to run — a self-referential metric that silently disagrees with the external
ground truth is exactly the failure this spec exists to prevent.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

# Reuse verbatim from poker_collusion — do NOT re-implement metric / CV / PU handling.
from poker_collusion.config import PipelineConfig
from poker_collusion.metric.production_metric import ScoreComponents, score_components
from poker_collusion.metric.reference_public_metric import ParticipantVisibleError
from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.validation.cv import build_fold_solution

from anchor_repro.models import (
    CanonicalScore,
    DisjointSetDiagnostic,
    IntegrityReport,
    ScoringRecipe,
)
from anchor_repro.self_check import (
    MetricSelfCheckError,
    SelfCheckResult,
    require_metric_matches_kernel,
)

__all__ = ["CanonicalScorer", "CoverageMismatchError", "EVAL_ROW_COUNT"]

#: The row-id column the official metric requires.
_ROW_ID_COLUMN = "pair_id"

#: Default dev-label / dev-evidence filenames under ``config.input_dir`` (the verbatim
#: competition schema; see poker_collusion.validation.cv and phase1_baseline_cv).
_DEV_LABELS_FILENAME = "development_labels.csv"
_DEV_EVIDENCE_FILENAME = "development_evidence.csv"

#: The eval sample-submission filename under ``config.input_dir`` (Kaggle's eval pair
#: set; the frozen EVAL artifacts must match it row-for-row to be scoreable).
_SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"

#: The MEASURED eval row count recorded during design (design.md load-bearing fact):
#: the frozen EVAL artifacts and the eval sample submission each have 112,540 rows.
EVAL_ROW_COUNT: int = 112_540

#: The official metric's coverage-mismatch message shape, e.g.
#: ``"pair_id coverage mismatch: 3 missing and 5 extra."`` (verbatim from
#: reference_public_metric.score). Matched so the suspenders-catch can confirm it is
#: the coverage raise and re-surface the same missing/extra numbers, never any other
#: ParticipantVisibleError (which is re-raised untouched).
_COVERAGE_MISMATCH_RE = re.compile(
    r"pair_id coverage mismatch: (?P<missing>\d+) missing and (?P<extra>\d+) extra"
)


class CoverageMismatchError(ParticipantVisibleError):
    """Raised by the belt pre-check when prediction and reference pair sets differ.

    Subclasses the official metric's :class:`ParticipantVisibleError` so callers
    (and the existing scorer tests) that catch the metric's own coverage raise also
    catch this pre-check refusal — both the belt (pre-check) and the suspenders
    (catch-and-re-surface) paths raise this SAME explicit type, and its message names
    the disjoint-set sizes (how many pair-ids are missing and extra), never a silent
    zero or a fabricated number (design.md "Coverage-mismatch guard", Property 5,
    Req 2.4).

    Attributes:
        diagnostic: The :class:`DisjointSetDiagnostic` carrying the measured
            missing/extra/intersection sizes.
    """

    def __init__(self, diagnostic: DisjointSetDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.describe())


def _pair_id_set(frame: pd.DataFrame) -> set[str]:
    """Return the set of ``pair_id`` values of a frame as strings (metric semantics)."""
    if _ROW_ID_COLUMN not in frame.columns:
        raise ValueError(
            f"frame is missing the required {_ROW_ID_COLUMN!r} column; "
            f"got columns {sorted(frame.columns)}."
        )
    return {str(pid) for pid in frame[_ROW_ID_COLUMN]}


def _disjoint_diagnostic(reference_ids: Iterable[str], predicted_ids: Iterable[str]) -> DisjointSetDiagnostic:
    """Measure the disjoint-set sizes between a reference and a predicted id set.

    ``missing``/``extra`` mirror the official metric's own definitions so the belt
    pre-check and the metric's suspenders raise report the SAME numbers:
    ``missing = |reference - predicted|``, ``extra = |predicted - reference|``.
    """
    reference = set(reference_ids)
    predicted = set(predicted_ids)
    intersection = reference & predicted
    return DisjointSetDiagnostic(
        n_reference=len(reference),
        n_predicted=len(predicted),
        n_intersection=len(intersection),
        n_missing=len(reference - predicted),
        n_extra=len(predicted - reference),
    )


def _sha256(path: Path) -> str:
    """Stream a SHA-256 hex digest of ``path`` (byte-integrity anchor, mirrors gate_a)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CanonicalScorer:
    """Score dev-holdout predictions under the single canonical recipe.

    Alongside :meth:`score_dev_predictions` (task 2.2), this class provides
    :meth:`integrity_report` (task 2.3): the frozen-EVAL-artifact scoreability check
    plus the impossibility / coverage-mismatch guard that surfaces the disjoint-set
    sizes explicitly, never a silent zero or a fabricated number.

    Args:
        recipe: The frozen :class:`ScoringRecipe`. Every anchor this scorer produces
            records this recipe's ``recipe_id`` so anchors are mutually comparable
            (Req 2.5, 2.6).
        config: The :class:`PipelineConfig` that locates the local data root
            (``config.input_dir``). Dev labels / evidence are loaded lazily from
            there the first time they are needed, unless injected (see ``labels`` /
            ``evidence``), so unit tests can drive the scorer with synthetic frames
            and never touch the real data files.
        labels: Optional pre-loaded dev-label frame (``pair_id``, ``label_status``,
            ``behavior_family``); when given, no ``development_labels.csv`` read
            happens. Used by tests and by callers that already hold the frame.
        evidence: Optional pre-loaded long-form dev-evidence frame (``pair_id``,
            ``hand_id`` and optionally ``evidence_rank``). ``None`` means no evidence
            hands are filled (positives get ``NO_EVIDENCE`` slots).

    The scorer runs the task-2.1 official-metric self-check exactly ONCE, lazily, on
    the first scoring call (cached thereafter). A self-check FAIL raises
    :class:`MetricSelfCheckError` and the scorer refuses to score (Req 1.2, 1.4).
    """

    def __init__(
        self,
        recipe: ScoringRecipe,
        config: PipelineConfig,
        *,
        labels: Optional[pd.DataFrame] = None,
        evidence: Optional[pd.DataFrame] = None,
        sample_submission: Optional[pd.DataFrame] = None,
    ) -> None:
        self.recipe = recipe
        self.config = config
        self._labels: Optional[pd.DataFrame] = labels
        self._evidence: Optional[pd.DataFrame] = evidence
        #: Optional pre-loaded eval sample submission (the eval pair set the frozen
        #: EVAL artifacts must match). Loaded lazily from ``config.input_dir`` on first
        #: use by :meth:`integrity_report` unless injected (tests supply a synthetic
        #: sample so no real data files are touched).
        self._sample_submission: Optional[pd.DataFrame] = sample_submission
        #: Cached passing self-check result; ``None`` until the gate has run.
        self._self_check: Optional[SelfCheckResult] = None

    # --------------------------------------------------------------------- #
    # Hard gate: the official-metric self-check (task 2.1) must pass first   #
    # --------------------------------------------------------------------- #
    def ensure_metric_verified(self) -> SelfCheckResult:
        """Run (once, cached) the official-metric self-check; refuse to run on FAIL.

        Wires in task 2.1's hard gate: the local verbatim official metric must equal
        the published kernel's own ``score()`` on a fixed example. A FAIL is a hard
        release blocker — the scorer must not judge anything with a drifted metric
        (Req 1.2, 1.4; design.md "the scorer refuses to run").

        Returns:
            The passing :class:`SelfCheckResult` (cached after the first success).

        Raises:
            MetricSelfCheckError: If the local metric does not equal the kernel score.
        """
        if self._self_check is None:
            # require_metric_matches_kernel raises MetricSelfCheckError on drift.
            self._self_check = require_metric_matches_kernel()
        return self._self_check

    # --------------------------------------------------------------------- #
    # Dev-label / evidence access (lazy, injectable)                         #
    # --------------------------------------------------------------------- #
    def _dev_labels(self) -> pd.DataFrame:
        """Return the dev-label frame, loading it lazily from config on first use."""
        if self._labels is None:
            path = Path(self.config.input_dir) / _DEV_LABELS_FILENAME
            self._labels = pd.read_csv(path)
        return self._labels

    def _dev_evidence(self) -> Optional[pd.DataFrame]:
        """Return the dev-evidence frame (lazy). Missing file ⇒ ``None`` (no evidence)."""
        if self._evidence is None:
            path = Path(self.config.input_dir) / _DEV_EVIDENCE_FILENAME
            if path.is_file():
                self._evidence = pd.read_csv(path)
        return self._evidence

    # --------------------------------------------------------------------- #
    # The canonical scoring pass                                             #
    # --------------------------------------------------------------------- #
    def score_dev_predictions(self, dev_predictions: pd.DataFrame) -> CanonicalScore:
        """Score a dev-holdout prediction frame under the canonical recipe.

        The ``dev_predictions`` frame is a submission whose ``pair_id``\\s ARE dev
        holdout pairs (the PU-stress table-disjoint split), with the production-metric
        schema: ``pair_id``, ``risk_score`` in ``[0, 1]``, ``predicted_behavior`` in
        the allowed set, and the five ``evidence_hand_*`` columns. It is NOT a frozen
        EVAL artifact (those have disjoint pair sets — see the class docstring / task
        2.3's ``integrity_report``).

        The scorer:

        1. Runs the official-metric self-check gate first and refuses to run on FAIL
           (Req 1.2, 1.4).
        2. Builds the PU-correct solution for the prediction's ``pair_id``\\s via the
           REUSED :func:`build_fold_solution` (unknowns never materialised as
           negatives — PU handling inherited, not reinvented).
        3. Scores it with the REUSED :func:`score_components` in a SINGLE evaluation,
           surfacing BOTH ``pair_ap`` (PRIMARY gate) and ``combined`` (SECONDARY
           diagnostic) from that one pass — never two divergent recomputations.

        Args:
            dev_predictions: The dev-holdout submission frame described above.

        Returns:
            A :class:`CanonicalScore` carrying ``pair_ap`` (the anchor's
            ``local_holdout_ap`` and the PRIMARY gate), ``combined`` (the SECONDARY
            diagnostic), and the full :class:`ScoreComponents` breakdown — all from
            the same single metric pass.

        Raises:
            MetricSelfCheckError: If the official-metric self-check fails.
            ValueError: If ``dev_predictions`` lacks the required ``pair_id`` column.
            CoverageMismatchError: If the built dev-holdout solution and the predictions
                do not cover the same ``pair_id`` set — raised either by the belt
                pre-check before the metric runs, or by re-surfacing the official
                metric's own coverage-mismatch ``ParticipantVisibleError``. Its message
                names how many pair-ids are missing and how many are extra — never a
                silent zero or a fabricated number (Property 5, Req 2.4).
                :class:`CoverageMismatchError` subclasses ``ParticipantVisibleError`` so
                existing ``except ParticipantVisibleError`` callers still catch it.
            ParticipantVisibleError: If the submission is otherwise invalid (a genuine,
                non-coverage submission-contract violation from the official metric).
        """
        # (1) Hard gate: the metric must match the published kernel or we refuse.
        self.ensure_metric_verified()

        if _ROW_ID_COLUMN not in dev_predictions.columns:
            raise ValueError(
                f"dev_predictions is missing the required {_ROW_ID_COLUMN!r} column; "
                f"got columns {sorted(dev_predictions.columns)}."
            )

        # (2) Build the PU-correct solution for exactly the predicted pair-ids, reusing
        # poker_collusion's builder so unknown pairs are never scored as negatives.
        pair_ids = [str(pid) for pid in dev_predictions[_ROW_ID_COLUMN]]
        solution = build_fold_solution(
            self._dev_labels(), pair_ids, evidence=self._dev_evidence()
        )

        # (2b) BELT (Change B pre-check): before invoking the metric, confirm the
        # prediction pair-id set equals the built dev-holdout solution set. If a pair
        # was dropped as unknown (PU) or an eval id slipped in, the sets differ and we
        # refuse with a diagnostic naming the disjoint-set sizes — never a silent zero
        # or a fabricated number (Property 5, Req 2.4). This raises the SAME explicit
        # CoverageMismatchError (a ParticipantVisibleError subclass) that the metric's
        # own raise is re-surfaced as below, so callers see one consistent type.
        diagnostic = _disjoint_diagnostic(
            _pair_id_set(solution), _pair_id_set(dev_predictions)
        )
        if not diagnostic.is_equal:
            raise CoverageMismatchError(diagnostic)

        # (3) ONE official-metric evaluation yields both scores + the full breakdown.
        # Copies are passed so the metric's in-place index sort cannot mutate caller
        # frames, and the two scores can never diverge across passes/splits.
        # SUSPENDERS (Change B catch): if the belt is somehow bypassed (e.g. an
        # upstream re-index), the metric's OWN coverage-mismatch ParticipantVisibleError
        # is re-surfaced as the same explicit CoverageMismatchError, never swallowed
        # into a silent zero. Any OTHER ParticipantVisibleError is a genuine
        # submission-contract violation and is re-raised untouched.
        try:
            components: ScoreComponents = score_components(
                solution.copy(), dev_predictions.copy(), _ROW_ID_COLUMN
            )
        except ParticipantVisibleError as exc:
            recovered = self._coverage_mismatch_from_metric(exc, solution, dev_predictions)
            if recovered is not None:
                raise recovered from exc
            raise

        return CanonicalScore(
            pair_ap=components.pair_ap,       # PRIMARY gate == anchor local_holdout_ap
            combined=components.combined,     # SECONDARY diagnostic
            components=components,            # full breakdown from the SAME pass
        )

    # --------------------------------------------------------------------- #
    # Coverage-mismatch / impossibility guard (task 2.3, Change B)          #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _coverage_mismatch_from_metric(
        exc: ParticipantVisibleError,
        reference: pd.DataFrame,
        predicted: pd.DataFrame,
    ) -> Optional[CoverageMismatchError]:
        """Re-surface the metric's coverage raise as an explicit CoverageMismatchError.

        Returns a :class:`CoverageMismatchError` (naming the disjoint-set sizes) IFF
        ``exc`` is the metric's own ``pair_id coverage mismatch: N missing and M extra``
        raise; otherwise returns ``None`` so the caller re-raises the original
        (genuine, non-coverage) submission-contract error untouched. The re-surfaced
        error recomputes the missing/extra sizes from the frames so the diagnostic is
        our own MEASURED number, consistent with the belt pre-check (Property 5).
        """
        if _COVERAGE_MISMATCH_RE.search(str(exc)) is None:
            return None
        try:
            diagnostic = _disjoint_diagnostic(
                _pair_id_set(reference), _pair_id_set(predicted)
            )
        except ValueError:  # pragma: no cover - reference/predicted always have pair_id here
            return None
        return CoverageMismatchError(diagnostic)

    def dev_holdout_disjoint_diagnostic(
        self, predictions: pd.DataFrame
    ) -> DisjointSetDiagnostic:
        """Measure the disjoint-set sizes between ``predictions`` and the dev labels.

        Surfaces, WITHOUT scoring, how many pair-ids are missing and extra between a
        prediction frame and the confirmed dev labels — the honest number the
        impossibility guard reports for an eval artifact (its pairs are disjoint from
        the dev labels, so ``n_intersection == 0``; design.md, Req 2.4, contract
        Rules 1 & 9). Used by :meth:`integrity_report`.
        """
        dev_ids = {str(pid) for pid in self._dev_labels()[_ROW_ID_COLUMN]}
        return _disjoint_diagnostic(dev_ids, _pair_id_set(predictions))

    def integrity_report(
        self,
        eval_artifact: Union[pd.DataFrame, str, Path],
        *,
        expected_row_count: int = EVAL_ROW_COUNT,
    ) -> IntegrityReport:
        """Scoreability / integrity of a frozen EVAL artifact (mirrors gate_a_verify_floor).

        Checks a frozen EVAL artifact's hash, row count (112,540), pair-set-equals-sample,
        and non-degenerate ranking — the SAME falsifiable checks the existing
        ``poker_collusion.experiments.gate_a_verify_floor`` runs against the 0.44519
        floor artifact. It DOES NOT attempt to dev-score the eval artifact: the eval
        pairs are DISJOINT from the dev labels, so a dev score is impossible (there is
        nothing to score), and claiming one would be fabrication (design.md, contract
        Rules 1 & 9, Req 2.4). Instead it surfaces the disjoint-set sizes between the
        artifact and the dev holdout explicitly (``n_intersection`` is expected to be 0).

        Args:
            eval_artifact: Either an already-loaded artifact frame, or a path to a
                ``submission_best_*.csv`` file. When a path is given the SHA-256 hash
                and byte size are computed from the file; when a frame is given they
                are recorded as unavailable (empty hash / 0 bytes) since no file backs
                it.
            expected_row_count: The eval row count to check against (default
                :data:`EVAL_ROW_COUNT` = 112,540).

        Returns:
            An :class:`IntegrityReport`. ``scoreable`` is ``True`` only when the
            artifact is intact (schema + unit-interval risk), has the expected row
            count, its pair-id set equals the eval sample submission, and its ranking
            is non-degenerate — the necessary condition to have been a leaderboard-valid
            file. ``scoreable`` is NEVER a claim that the artifact can be dev-scored.

        Raises:
            MetricSelfCheckError: If the official-metric self-check fails (same hard
                gate as scoring — a drifted metric must judge nothing, Req 1.2, 1.4).
            ValueError: If the artifact frame lacks the required ``pair_id`` column.
        """
        # Same hard gate as scoring: refuse to judge anything with a drifted metric.
        self.ensure_metric_verified()

        notes: list[str] = []

        # ---- Load frame + hash/bytes (hash only available when a path is given) ----
        if isinstance(eval_artifact, (str, Path)):
            path = Path(eval_artifact)
            sha256 = _sha256(path)
            n_bytes = path.stat().st_size
            frame = pd.read_csv(path, dtype={_ROW_ID_COLUMN: str})
            notes.append(f"MEASURED: sha256/bytes computed from file {path.name!r}.")
        else:
            frame = eval_artifact
            sha256 = ""
            n_bytes = 0
            notes.append(
                "MEASURED: an in-memory frame was supplied; sha256/bytes are "
                "unavailable (no file backs it)."
            )

        if _ROW_ID_COLUMN not in frame.columns:
            raise ValueError(
                f"eval_artifact is missing the required {_ROW_ID_COLUMN!r} column; "
                f"got columns {sorted(frame.columns)}."
            )

        # ---- INTEGRITY: shape + risk stats (mirrors gate_a) ----
        n_rows = int(len(frame))
        row_count_ok = n_rows == expected_row_count
        columns_match = tuple(frame.columns) == SUBMISSION_COLUMNS
        risk = pd.to_numeric(frame.get("risk_score"), errors="coerce")
        risk_notna = risk.notna()
        risk_in_unit_interval = bool(risk_notna.all() and risk.between(0.0, 1.0).all())
        risk_min = float(risk.min()) if risk_notna.any() else float("nan")
        risk_max = float(risk.max()) if risk_notna.any() else float("nan")
        risk_mean = float(risk.mean()) if risk_notna.any() else float("nan")
        risk_n_distinct = int(risk.nunique())
        # Non-degenerate ranking, matching gate_a's clause (>1000 distinct + spread).
        ranking_non_degenerate = bool(risk_n_distinct > 1000 and risk_max > risk_min)

        # ---- CONSISTENCY: pair-set equals the eval sample submission ----
        sample_diagnostic: Optional[DisjointSetDiagnostic] = None
        pair_set_equals_sample: Optional[bool] = None
        sample = self._eval_sample_submission()
        if sample is not None:
            sample_diagnostic = _disjoint_diagnostic(
                _pair_id_set(sample), _pair_id_set(frame)
            )
            pair_set_equals_sample = sample_diagnostic.is_equal
            notes.append(
                "MEASURED vs eval sample submission — " + sample_diagnostic.describe()
            )
        else:
            notes.append(
                "eval sample submission unavailable; pair-set-equals-sample not checked."
            )

        # ---- IMPOSSIBILITY: surface the eval-vs-dev disjoint sizes explicitly ----
        dev_disjoint_diagnostic: Optional[DisjointSetDiagnostic] = None
        try:
            dev_disjoint_diagnostic = self.dev_holdout_disjoint_diagnostic(frame)
            notes.append(
                "MEASURED vs dev holdout labels — " + dev_disjoint_diagnostic.describe()
            )
            if dev_disjoint_diagnostic.is_disjoint:
                notes.append(
                    "IMPOSSIBILITY (Req 2.4): the artifact's pair-ids are DISJOINT from "
                    "the dev labels, so this eval artifact CANNOT be dev-scored — no "
                    "absolute LB number is reproduced (stated impossible, never faked)."
                )
        except FileNotFoundError:
            notes.append(
                "dev labels unavailable; eval-vs-dev disjoint sizes not computed."
            )

        # ---- Scoreability verdict (integrity/scoreability only, NOT a dev score) ----
        scoreable = bool(
            columns_match
            and row_count_ok
            and risk_in_unit_interval
            and ranking_non_degenerate
            and (pair_set_equals_sample is True)
        )

        return IntegrityReport(
            sha256=sha256,
            n_bytes=n_bytes,
            n_rows=n_rows,
            expected_row_count=expected_row_count,
            row_count_ok=row_count_ok,
            columns_match=columns_match,
            pair_set_equals_sample=pair_set_equals_sample,
            sample_diagnostic=sample_diagnostic,
            risk_min=risk_min,
            risk_max=risk_max,
            risk_mean=risk_mean,
            risk_n_distinct=risk_n_distinct,
            risk_in_unit_interval=risk_in_unit_interval,
            ranking_non_degenerate=ranking_non_degenerate,
            dev_disjoint_diagnostic=dev_disjoint_diagnostic,
            scoreable=scoreable,
            notes=notes,
        )

    def _eval_sample_submission(self) -> Optional[pd.DataFrame]:
        """Return the eval sample submission (lazy). Missing file ⇒ ``None``."""
        if self._sample_submission is None:
            path = Path(self.config.input_dir) / _SAMPLE_SUBMISSION_FILENAME
            if path.is_file():
                self._sample_submission = pd.read_csv(path, dtype={_ROW_ID_COLUMN: str})
        return self._sample_submission
