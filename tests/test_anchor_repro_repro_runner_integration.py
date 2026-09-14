"""Integration smoke test for the sandboxed reproduction runner (task 7.3).

design.md "Components and Interfaces" §4 (SandboxedReproRunner) + "Error Handling"
(BLOCKED vs partial). Requirements 1.5, 3.1, 3.3. Governed end-to-end by the workspace
``reverse-engineering-accountability`` contract and the safety guardrails: the run is
sandboxed (network off, path shim, credential-strip, working-dir jail, wall-clock
timeout); it NEVER transmits code/data anywhere and NEVER fabricates a ``SUCCEEDED``.

This is ONE representative run of the real, UNTRUSTED 0.64469 competitor notebook
(``poker/_refs/honghanh/detecting-collusive-value-transfer-in-6-max-poker.ipynb``)
against the local competition data root (``poker/data/poker``). Because it executes
real, untrusted, side-effecting code it is NOT property-tested (a single smoke run,
same convention as ``test_anchor_repro_ladder_integration.py``) and is marked
``@pytest.mark.integration`` so the fast unit/property suite can exclude it.

What it asserts is the STATE MACHINE and the sandbox behavior, NOT reproduction of the
0.64469 leaderboard number. The target notebook is a ~30-40 min full pipeline; within a
modest test budget it may legitimately BLOCK at ``compute`` (timeout), ``deps`` (an
export/runtime gap), ``data_path``, or ``runtime`` — every one of those is an HONEST
terminal state, not a test failure. The ONLY hard requirements are:

* :meth:`SandboxedReproRunner.run` returns exactly ONE terminal state — either
  ``SUCCEEDED`` (with a ``submission_path`` that exists and is a schema-valid
  ``submission.csv`` validated against the real ``sample_submission.csv``), OR
  ``BLOCKED`` (with a ``failure_stage`` in ``{deps, data_path, compute, runtime}`` and a
  non-empty ``failure_detail``). It NEVER returns partial success and NEVER returns a
  ``RUNNING`` result to the caller.
* The audit trail is populated: ``dependency_manifest`` and ``data_path_manifest`` are
  both non-empty regardless of outcome.

The test asserts NEITHER a specific outcome (contract Rule 9 — state the uncertainty;
both terminal states are honest) NOR that a SUCCEEDED was reached. It skips cleanly if
the notebook or the local data root is absent (contract Rules 1 & 9 — never fabricate).

Requirements: 1.5, 3.1, 3.3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anchor_repro.repro_runner import (
    STAGE_COMPUTE,
    STAGE_DATA_PATH,
    STAGE_DEPS,
    STAGE_RUNTIME,
    STATUS_BLOCKED,
    STATUS_SUCCEEDED,
    ReproResult,
    SandboxedReproRunner,
)

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------- #
# Real notebook / real data locations (resolved relative to this file so the   #
# test runs from anywhere), mirroring the sibling integration tests.           #
# --------------------------------------------------------------------------- #
_POKER_ROOT = Path(__file__).resolve().parents[1]  # .../kAGGLE/poker
_DATA_DIR = _POKER_ROOT / "data" / "poker"
_SAMPLE_SUBMISSION = _DATA_DIR / "sample_submission.csv"
_NOTEBOOK = (
    _POKER_ROOT
    / "_refs"
    / "honghanh"
    / "detecting-collusive-value-transfer-in-6-max-poker.ipynb"
)

#: Modest wall-clock budget: the notebook is a ~30-40 min full pipeline, so a
#: compute/timeout BLOCK inside this budget is an ACCEPTABLE, honest terminal state —
#: the test asserts the state machine + sandbox, not reproduction of the LB number.
_TIMEOUT_S = 90

#: The four honest BLOCKED stages the state machine may pin (design "Error Handling").
_VALID_STAGES = {STAGE_DEPS, STAGE_DATA_PATH, STAGE_COMPUTE, STAGE_RUNTIME}


def _require_notebook_and_data() -> None:
    """Skip cleanly (never fabricate) if the notebook or the data root is absent."""
    if not _NOTEBOOK.is_file():
        pytest.skip(
            f"Untrusted 0.64469 notebook not present at {_NOTEBOOK}; cannot run the "
            "sandboxed repro smoke test (skipping rather than fabricating a result)."
        )
    if not _DATA_DIR.is_dir():
        pytest.skip(
            f"Local competition data root not present at {_DATA_DIR}; the path shim "
            "has nothing to map the notebook's data discovery onto (skipping)."
        )


def _assert_schema_valid_submission(submission_path: Path) -> None:
    """Assert a captured ``submission.csv`` is schema-valid against sample_submission.

    Uses the official ``poker_collusion.submission.writer.validate_submission`` (the
    same contract the runner enforces internally) so the SUCCEEDED path is independently
    confirmed to have emitted a leaderboard-shaped submission, not merely a file.
    """
    from poker_collusion.submission.writer import (
        SubmissionValidationError,
        validate_submission,
    )

    assert submission_path.is_file(), (
        f"SUCCEEDED must yield a submission_path that exists; got {submission_path}"
    )
    frame = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    if not _SAMPLE_SUBMISSION.is_file():
        pytest.skip(
            f"sample_submission.csv absent at {_SAMPLE_SUBMISSION}; cannot schema-"
            "validate the captured SUCCEEDED submission (skipping the validation "
            "rather than asserting an unverifiable pass)."
        )
    sample = pd.read_csv(_SAMPLE_SUBMISSION, dtype=str, keep_default_na=False)
    try:
        validate_submission(frame, sample)
    except SubmissionValidationError as exc:  # pragma: no cover - honest failure surface
        raise AssertionError(
            "a SUCCEEDED run must capture a schema-valid submission.csv, but it "
            f"failed the official submission contract: {exc}"
        ) from exc


@pytest.mark.integration
def test_sandboxed_repro_runner_reaches_a_coherent_terminal_state() -> None:
    """One sandboxed run of the untrusted 0.64469 notebook against local data.

    Asserts a COHERENT TERMINAL state (never a specific outcome, never partial):

    * ``SUCCEEDED`` ⇒ ``submission_path`` exists and is a schema-valid ``submission.csv``
      (validated against the real ``sample_submission.csv``), and the failure fields are
      cleared; OR
    * ``BLOCKED`` ⇒ ``failure_stage`` ∈ {deps, data_path, compute, runtime} with a
      non-empty ``failure_detail``, and no ``submission_path``.

    In BOTH cases the run is exactly one terminal value (never ``RUNNING``, never
    partial) and the ``dependency_manifest`` / ``data_path_manifest`` audit trails are
    populated. A compute/timeout BLOCK within the modest budget is an acceptable pass —
    the test governs the state machine + sandbox, not reproduction of the LB score.
    """
    _require_notebook_and_data()

    runner = SandboxedReproRunner()
    result = runner.run(_NOTEBOOK, local_data_dir=_DATA_DIR, timeout_s=_TIMEOUT_S)

    # ------------------------------------------------------------------ #
    # Coarse shape: a real ReproResult for THIS notebook.                #
    # ------------------------------------------------------------------ #
    assert isinstance(result, ReproResult)
    assert result.notebook == str(_NOTEBOOK)

    # ------------------------------------------------------------------ #
    # Exactly ONE terminal state — never RUNNING, never partial (Req 3.3).#
    # ------------------------------------------------------------------ #
    assert result.status in {STATUS_SUCCEEDED, STATUS_BLOCKED}, (
        f"run() must return a terminal state, never RUNNING/partial; got "
        f"{result.status!r}"
    )
    # The two boolean views must agree with the single status (no partial/ambiguous).
    assert result.succeeded == (result.status == STATUS_SUCCEEDED)
    assert result.blocked == (result.status == STATUS_BLOCKED)
    assert result.succeeded != result.blocked, (
        "the result must be EXACTLY one terminal value (SUCCEEDED xor BLOCKED)"
    )

    # ------------------------------------------------------------------ #
    # Audit trail populated regardless of outcome (contract Rule 9).      #
    # ------------------------------------------------------------------ #
    assert result.dependency_manifest, "dependency_manifest must be populated"
    assert result.data_path_manifest, "data_path_manifest must be populated"

    # ------------------------------------------------------------------ #
    # Per-terminal-state coherence.                                       #
    # ------------------------------------------------------------------ #
    if result.status == STATUS_SUCCEEDED:
        # Honest SUCCEEDED: a real, schema-valid submission was captured; failure
        # fields cleared. We NEVER fabricate this — it only asserts IF it happened.
        assert result.failure_stage is None
        assert result.failure_detail is None
        assert result.submission_path is not None
        _assert_schema_valid_submission(Path(result.submission_path))
        print(
            "\n[repro e2e] SUCCEEDED: captured schema-valid submission at "
            f"{result.submission_path}"
        )
    else:  # STATUS_BLOCKED — an honest, pinned terminal state.
        assert result.submission_path is None, (
            "a BLOCKED run must not carry a submission_path"
        )
        assert result.failure_stage in _VALID_STAGES, (
            f"BLOCKED must pin a stage in {sorted(_VALID_STAGES)}; got "
            f"{result.failure_stage!r}"
        )
        assert result.failure_detail and result.failure_detail.strip(), (
            "a BLOCKED run must record a non-empty failure_detail (exact reason)"
        )
        print(
            f"\n[repro e2e] BLOCKED at stage={result.failure_stage!r} — "
            f"{result.failure_detail[:200]!r} "
            "(an honest terminal state; a compute/timeout block within the modest "
            "test budget is acceptable — the notebook is a ~30-40 min pipeline)."
        )
