"""Official-metric self-check for the poker-anchor-reproduction canonical scorer.

Task 2.1 of the ``poker-anchor-reproduction`` spec (design.md §2 "MetricSelfCheck",
Req 1.2 / 1.4). Before the canonical scorer is trusted to judge ANYTHING, this
module re-verifies that the verbatim local official-metric copy
(:func:`poker_collusion.metric.reference_public_metric.score`) produces exactly the
same number as the competition metric KERNEL's own ``score()`` on a known input.

Why this exists (accountability contract Rules 1, 9):
    The local ``reference_public_metric`` is documented as a character-for-character
    copy of the kernel. "Documented as" is an assumption, not a verified fact. This
    self-check turns that assumption into a MEASURED equality against the kernel's
    OWN code, executed fresh from the notebook — so a silent drift between the local
    copy and the published metric is a hard release blocker, not an invisible bug.

The kernel notebook (``data/poker/_metric_kernel/slash-poker-competition-metric.ipynb``)
contains exactly one code cell: the metric definition. It exposes **no** embedded
``(solution, submission, expected_score)`` example. Per the task's fallback path, this
module therefore:

  1. extracts the kernel's own ``score()`` cell from the notebook JSON and executes it
     in an ISOLATED namespace (the kernel's real code, not our re-import), then
  2. constructs a small FIXED ``(solution, submission)`` pair, and
  3. asserts ``reference_public_metric.score(...)`` equals the freshly-executed kernel
     ``score(...)`` to full float tolerance (exact bit equality, not ``approx``).

A FAIL is a hard gate: :func:`require_metric_matches_kernel` raises
:class:`MetricSelfCheckError` so the scorer refuses to run (design.md: "FAIL =>
release blocker; the scorer refuses to run").

Reuse, not re-implementation (Req 1.6): the LOCAL side of the comparison is the
existing verbatim ``reference_public_metric.score`` — this module does not create a
second copy of the metric. The KERNEL side is executed from the notebook purely to
verify the local copy against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from poker_collusion.metric import reference_public_metric

__all__ = [
    "SelfCheckResult",
    "MetricSelfCheckError",
    "KERNEL_NOTEBOOK_RELPATH",
    "locate_kernel_notebook",
    "extract_kernel_score_fn",
    "build_fixed_example",
    "verify_metric_against_kernel",
    "require_metric_matches_kernel",
]

#: Path to the official metric kernel notebook, relative to the ``poker/`` tree root
#: (the directory that contains both the ``anchor_repro`` package and ``data/``).
KERNEL_NOTEBOOK_RELPATH = Path("data/poker/_metric_kernel/slash-poker-competition-metric.ipynb")

#: The row-id column the official metric requires.
_ROW_ID_COLUMN = "pair_id"


class MetricSelfCheckError(RuntimeError):
    """Raised when the local official-metric copy does NOT equal the kernel's score.

    This is a hard release blocker (Req 1.4): if the local
    ``reference_public_metric.score`` has drifted from the published kernel metric,
    every downstream anchor would be measured with a divergent metric, so the
    canonical scorer MUST refuse to run. Failing loudly here upholds the
    accountability contract (a self-referential metric that silently disagrees with
    the external ground truth is exactly the failure this spec exists to prevent).
    """


@dataclass(frozen=True)
class SelfCheckResult:
    """The outcome of verifying the local metric against the kernel's own ``score()``.

    Attributes:
        passed: ``True`` iff the local metric equals the kernel metric to full float
            tolerance on the fixed example(s). A ``False`` here is a release blocker.
        local_score: The value ``reference_public_metric.score`` returned.
        kernel_score: The value the freshly-executed kernel ``score()`` returned.
        abs_diff: ``abs(local_score - kernel_score)`` (0.0 on a passing check).
        kernel_notebook: The resolved path to the kernel notebook that was executed.
        example_name: A short label for the fixed example used.
        detail: Human-readable summary (MEASURED, contract Rule 9).
        extra_checks: Optional per-example ``(name, local, kernel, abs_diff)`` records
            when more than one fixed example is compared, for a fuller audit trail.
    """

    passed: bool
    local_score: float
    kernel_score: float
    abs_diff: float
    kernel_notebook: Path
    example_name: str
    detail: str
    extra_checks: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Locating and extracting the kernel's own score() cell                       #
# --------------------------------------------------------------------------- #
def locate_kernel_notebook(explicit: Optional[Path] = None) -> Path:
    """Resolve the official metric kernel notebook path.

    Args:
        explicit: If given, this exact path is used (and must exist).

    Resolution order when ``explicit`` is None:
        1. ``<poker-tree-root>/data/poker/_metric_kernel/...`` where the tree root is
           the parent of this ``anchor_repro`` package (the common case).
        2. The current working directory joined with :data:`KERNEL_NOTEBOOK_RELPATH`.

    Returns:
        The resolved, existing notebook path.

    Raises:
        MetricSelfCheckError: If no candidate path exists — the self-check cannot run
            without the kernel, and a missing kernel is a blocker, never a silent pass.
    """
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        # anchor_repro/self_check.py -> anchor_repro/ -> poker/ (tree root)
        tree_root = Path(__file__).resolve().parent.parent
        candidates.append(tree_root / KERNEL_NOTEBOOK_RELPATH)
        candidates.append(Path.cwd() / KERNEL_NOTEBOOK_RELPATH)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    raise MetricSelfCheckError(
        f"official metric kernel notebook not found; tried: {tried}. "
        "The self-check cannot verify the local metric without the kernel and "
        "refuses to pass silently (accountability contract Rules 1, 9)."
    )


def _iter_code_cells(notebook: Dict[str, Any]) -> List[str]:
    """Return the source text of every code cell in a parsed .ipynb, in order."""
    sources: List[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        sources.append(src)
    return sources


def extract_kernel_score_fn(notebook_path: Optional[Path] = None) -> Callable[..., float]:
    """Extract and execute the kernel's own ``score()`` from the notebook.

    Parses the notebook JSON, concatenates its code cells, and executes them in a
    FRESH, ISOLATED namespace so the returned ``score`` is the KERNEL's code — not a
    re-import of our local ``reference_public_metric``. This is the whole point of the
    self-check: compare our copy against the published source, not against itself.

    Note on safety: the metric kernel is trusted first-party competition tooling (not
    an untrusted competitor notebook — those go through the sandboxed runner). It is
    still executed in an isolated ``exec`` namespace with no arguments and no captured
    globals, and only the resulting ``score`` callable is used.

    Args:
        notebook_path: Explicit notebook path; resolved via
            :func:`locate_kernel_notebook` when None.

    Returns:
        The kernel's own ``score(solution, submission, row_id_column_name)`` callable.

    Raises:
        MetricSelfCheckError: If the notebook cannot be parsed or defines no ``score``.
    """
    path = locate_kernel_notebook(notebook_path)
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - IO/format guard
        raise MetricSelfCheckError(
            f"could not read/parse kernel notebook {path}: {exc}"
        ) from exc

    code = "\n\n".join(_iter_code_cells(notebook))
    if "def score(" not in code:
        raise MetricSelfCheckError(
            f"kernel notebook {path} defines no score() cell to verify against."
        )

    namespace: Dict[str, Any] = {}
    try:
        exec(compile(code, str(path), "exec"), namespace)  # noqa: S102 - trusted kernel
    except Exception as exc:  # pragma: no cover - kernel-execution guard
        raise MetricSelfCheckError(
            f"kernel notebook {path} failed to execute its metric cell(s): {exc}"
        ) from exc

    kernel_score = namespace.get("score")
    if not callable(kernel_score):
        raise MetricSelfCheckError(
            f"kernel notebook {path} did not produce a callable score()."
        )
    return kernel_score


# --------------------------------------------------------------------------- #
# Fixed example construction                                                  #
# --------------------------------------------------------------------------- #
def _evidence_row(hands: List[str]) -> Dict[str, str]:
    """Fill the five evidence columns from a hand-id list, padding with the sentinel."""
    padded = list(hands) + [reference_public_metric.NO_EVIDENCE] * (5 - len(hands))
    return {f"evidence_hand_{i}": padded[i - 1] for i in range(1, 6)}


def build_fixed_example() -> Dict[str, Any]:
    """Construct a small FIXED ``(solution, submission, name)`` example.

    The kernel notebook embeds no example, so we build a deterministic one that
    exercises every component of the metric (PairAP, EvidenceMAP@5, BehaviorMAP) with
    a mix of hits and misses, so a divergence in ANY component would surface as a
    score difference:

    * five pairs, three true positives (one per disclosed TARGET behavior family) and
      two negatives — so all three behavior families and the pair-AP ranking matter;
    * an IMPERFECT submission (partial evidence hits, a rank inversion, a middling
      risk on one positive) so the score is strictly inside ``(0, 1)`` rather than a
      degenerate 0 or 1 that could hide a discrepancy.

    Returns:
        A dict with ``solution`` (pd.DataFrame), ``submission`` (pd.DataFrame), and a
        ``name`` label.
    """
    solution = pd.DataFrame(
        [
            {"pair_id": "P1", "risk_score": 1, "predicted_behavior": "directed_transfer",
             **_evidence_row(["HA", "HB"])},
            {"pair_id": "P2", "risk_score": 1, "predicted_behavior": "soft_play",
             **_evidence_row(["HC"])},
            {"pair_id": "P3", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
            {"pair_id": "P4", "risk_score": 0, "predicted_behavior": "none",
             **_evidence_row([])},
            {"pair_id": "P5", "risk_score": 1, "predicted_behavior": "coordinated_isolation",
             **_evidence_row(["HD", "HE", "HF"])},
        ]
    )
    submission = pd.DataFrame(
        [
            # correct family, one evidence hit + one miss, high-but-not-1 risk
            {"pair_id": "P1", "risk_score": 0.90, "predicted_behavior": "directed_transfer",
             **_evidence_row(["HA", "HX"])},
            # correct family, evidence hit at rank 2 (rank inversion), middling risk
            {"pair_id": "P2", "risk_score": 0.55, "predicted_behavior": "soft_play",
             **_evidence_row(["HY", "HC"])},
            # true negative, low risk
            {"pair_id": "P3", "risk_score": 0.20, "predicted_behavior": "none",
             **_evidence_row([])},
            # true negative, slightly higher (still below the positives) risk
            {"pair_id": "P4", "risk_score": 0.30, "predicted_behavior": "none",
             **_evidence_row([])},
            # correct family, two of three evidence hits, high risk
            {"pair_id": "P5", "risk_score": 0.80, "predicted_behavior": "coordinated_isolation",
             **_evidence_row(["HD", "HZ", "HF"])},
        ]
    )
    return {"solution": solution, "submission": submission, "name": "fixed_5pair_mixed"}


# --------------------------------------------------------------------------- #
# The self-check                                                              #
# --------------------------------------------------------------------------- #
def verify_metric_against_kernel(
    notebook_path: Optional[Path] = None,
) -> SelfCheckResult:
    """Verify the local official metric equals the kernel's own ``score()``.

    Extracts the kernel's ``score()`` fresh from the notebook, builds a fixed
    ``(solution, submission)`` example, scores it with BOTH the local
    ``reference_public_metric.score`` and the kernel ``score``, and asserts EXACT
    float equality (full tolerance — the local copy is meant to be byte-identical, so
    the two must return the identical float, not merely a close one).

    This never mutates any input frame and executes no untrusted competitor code.

    Args:
        notebook_path: Explicit kernel notebook path; auto-resolved when None.

    Returns:
        A :class:`SelfCheckResult`. ``passed`` is ``True`` only when the local and
        kernel scores are exactly equal. This function does NOT raise on a mismatch —
        it records it; use :func:`require_metric_matches_kernel` for the hard gate.

    Raises:
        MetricSelfCheckError: Only for setup failures (missing/unparseable kernel, no
            ``score`` cell) — i.e. when the check could not be run at all. A genuine
            score MISMATCH returns ``passed=False`` rather than raising here.
    """
    kernel_score = extract_kernel_score_fn(notebook_path)
    resolved_notebook = locate_kernel_notebook(notebook_path)
    example = build_fixed_example()

    solution = example["solution"]
    submission = example["submission"]
    name = example["name"]

    # Score with the LOCAL verbatim copy and the freshly-executed KERNEL code.
    # Pass copies so neither callable can observe the other's in-place index sorting.
    local_value = float(
        reference_public_metric.score(solution.copy(), submission.copy(), _ROW_ID_COLUMN)
    )
    kernel_value = float(
        kernel_score(solution.copy(), submission.copy(), _ROW_ID_COLUMN)
    )

    abs_diff = abs(local_value - kernel_value)
    # Full float tolerance: the local copy is a character-for-character extraction of
    # the kernel, so the returned floats must be EXACTLY equal (identical operations on
    # identical inputs). Any difference at all is a drift and a release blocker.
    passed = local_value == kernel_value

    if passed:
        detail = (
            f"MEASURED: local reference_public_metric.score == kernel score exactly "
            f"({local_value!r}) on fixed example {name!r}; abs_diff={abs_diff!r}. "
            f"The local official-metric copy matches the published kernel "
            f"({resolved_notebook.name}); scorer cleared to run."
        )
    else:
        detail = (
            f"MEASURED FAILURE: local reference_public_metric.score ({local_value!r}) "
            f"!= kernel score ({kernel_value!r}) on fixed example {name!r}; "
            f"abs_diff={abs_diff!r}. The local metric has DRIFTED from the published "
            f"kernel ({resolved_notebook.name}) — hard release blocker; the scorer "
            f"must refuse to run (Req 1.4)."
        )

    return SelfCheckResult(
        passed=passed,
        local_score=local_value,
        kernel_score=kernel_value,
        abs_diff=abs_diff,
        kernel_notebook=resolved_notebook,
        example_name=name,
        detail=detail,
        extra_checks=[
            {
                "name": name,
                "local": local_value,
                "kernel": kernel_value,
                "abs_diff": abs_diff,
            }
        ],
    )


def require_metric_matches_kernel(
    notebook_path: Optional[Path] = None,
) -> SelfCheckResult:
    """Hard gate: run the self-check and RAISE if the local metric drifted.

    Downstream code (the canonical scorer, task 2.2) calls this before scoring
    anything so that a metric drift is a release blocker, not a silent
    miscalibration (Req 1.4; design.md "FAIL => release blocker; the scorer refuses
    to run").

    Args:
        notebook_path: Explicit kernel notebook path; auto-resolved when None.

    Returns:
        The passing :class:`SelfCheckResult`.

    Raises:
        MetricSelfCheckError: If the local metric does not equal the kernel score.
    """
    result = verify_metric_against_kernel(notebook_path)
    if not result.passed:
        raise MetricSelfCheckError(result.detail)
    return result
