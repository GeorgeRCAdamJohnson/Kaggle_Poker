"""Direct (owned-source) reproduction runner for competitor notebooks we TRUST.

Contract note (reverse-engineering-accountability, Rule 11 + the explicit user
decision): the sandboxed :class:`~anchor_repro.repro_runner.SandboxedReproRunner`
treats a notebook as UNTRUSTED and runs it behind an ``open``/``Path`` shim, a
data-junction, and a builtins monkeypatch. For the 0.64469 competitor notebook we
OWN and TRUST the source, and that shim kept colliding with the notebook's real
needs (``polars.write_csv`` bypasses the ``open`` shim, EDA ``plt.show`` blocks,
``/kaggle/working`` writes, UTF-8 stdio, library resource reads). Rather than keep
fighting the sandbox, this module runs the notebook's pipeline DIRECTLY in a child
process with only the minimal, honest overrides needed to make it run headless
against the local data:

* **matplotlib headless** — ``MPLBACKEND=Agg`` in the child env AND an injected
  ``matplotlib.use("Agg")`` + ``plt.show = lambda *a, **k: None`` shim at the top,
  so EDA cells never open a blocking GUI window.
* **data root** — an injected ``find_data_dir()`` override (placed AFTER the
  notebook body, so it wins over the notebook's own rglob-based definition) that
  returns the local ``poker/data/poker`` root.
* **output dir** — an injected ``OUTPUT_DIR`` override pointing at an EXPLICIT,
  PERSISTENT output dir passed in by the caller, so the notebook's feature cache
  (``prepared_pu24k``) AND ``submission.csv`` land where we can capture them and a
  rerun reuses the warm cache (~10-15 min) instead of a cold build (~30-40 min).
* **UTF-8 stdio** — ``PYTHONUTF8=1`` / ``PYTHONIOENCODING=utf-8`` in the child;
  the parent decodes ``encoding="utf-8", errors="replace"`` so a non-cp1252 byte in
  the notebook's prints (e.g. the arrow glyph) never tears the run down.
* **explicit pipeline drive** — the notebook's trailing ``RUN_FULL_PIPELINE`` /
  ``main()`` / ``evidence_lift_eda()`` driver cells are stripped from the body and
  ``main()`` is called from an injected footer AFTER the overrides, so the
  post-submission EDA cell cannot crash the run once ``submission.csv`` exists.

Honesty contract (Rules 1, 3, 9):

* A job is ``SUCCEEDED`` ONLY when ``submission.csv`` exists AND validates against
  the local ``sample_submission.csv`` via
  :func:`poker_collusion.submission.writer.validate_submission`. Never partial.
* On timeout / non-zero exit / missing / invalid submission the job is ``BLOCKED``
  with the real failure stage and the decodable traceback tail. No fabrication.
* The emitted ``submission.csv`` is an **EVAL** submission (112,540 eval pairs,
  disjoint from the dev labels). This module NEVER scores it as a dev AP — that is
  a separate downstream step (re-run the recipe on the DEV holdout). It only
  reports existence + row count + schema validity.

The data is read-only; the notebook source is never modified.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Reuse the notebook -> script export (with __future__ hoisting) from the sandboxed
# runner; we do NOT reuse its shim header (that is the whole point of "direct").
from anchor_repro.repro_runner import _export_notebook_to_script

__all__ = [
    "STATUS_SUCCEEDED",
    "STATUS_BLOCKED",
    "STAGE_DEPS",
    "STAGE_DATA_PATH",
    "STAGE_COMPUTE",
    "STAGE_RUNTIME",
    "DirectReproResult",
    "DirectReproJob",
    "build_direct_script",
    "run_direct_job",
    "run_jobs_parallel",
    "competitor_064469_job",
    "DEFAULT_TIMEOUT_S",
]

STATUS_SUCCEEDED: str = "SUCCEEDED"
STATUS_BLOCKED: str = "BLOCKED"

STAGE_DEPS: str = "deps"
STAGE_DATA_PATH: str = "data_path"
STAGE_COMPUTE: str = "compute"
STAGE_RUNTIME: str = "runtime"

#: Hard wall-clock cap: the user will not wait longer than 3 hours.
DEFAULT_TIMEOUT_S: int = 10800

_SUBMISSION_FILENAME: str = "submission.csv"
#: Expected eval submission row count (112,540 eval pairs).
_EXPECTED_EVAL_ROWS: int = 112540


# --------------------------------------------------------------------------- #
# Result + job models                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DirectReproResult:
    """Terminal result of a direct notebook reproduction (never partial).

    Attributes:
        name: The job name.
        status: :data:`STATUS_SUCCEEDED` or :data:`STATUS_BLOCKED`.
        elapsed_s: Measured wall-clock seconds for the child process.
        submission_path: Path to the emitted ``submission.csv`` on success (it lives
            in ``output_dir``, persistent); ``None`` when blocked.
        n_rows: Row count of the emitted submission on success; ``None`` otherwise.
        failure_stage: One of the STAGE_* vocab on BLOCKED; ``None`` on success.
        failure_detail: Human-readable pinned reason on BLOCKED; ``None`` on success.
        output_dir: The persistent output dir (cache + submission + log live here).
        log_path: Path to the captured child stdout+stderr log.
        returncode: Child process exit code (``None`` on timeout).
    """

    name: str
    status: str
    elapsed_s: float
    output_dir: str
    log_path: str
    submission_path: Optional[str] = None
    n_rows: Optional[int] = None
    failure_stage: Optional[str] = None
    failure_detail: Optional[str] = None
    returncode: Optional[int] = None

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_SUCCEEDED

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_s": round(self.elapsed_s, 2),
            "submission_path": self.submission_path,
            "n_rows": self.n_rows,
            "failure_stage": self.failure_stage,
            "failure_detail": self.failure_detail,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "returncode": self.returncode,
        }


@dataclass(frozen=True)
class DirectReproJob:
    """One reproduction job = a notebook + a persistent output dir + a data root.

    Designed so ladder-recipe reconstructions can be added as additional parallel
    jobs later: each job is independent, writes into its OWN ``output_dir``, and its
    child process is driven by :func:`run_direct_job`.

    Attributes:
        name: Short unique name (used in logs + the outcome JSON filename).
        notebook: Path to the ``.ipynb`` to run directly (trusted, owned source).
        data_dir: The local competition data root (``poker/data/poker``), read-only.
        output_dir: Persistent output dir (created before the run); the notebook's
            cache + ``submission.csv`` + log land here.
        timeout_s: Wall-clock hard cap in seconds (default 3h).
        python_executable: Interpreter for the child; defaults to ``sys.executable``.
    """

    name: str
    notebook: Union[str, Path]
    data_dir: Union[str, Path]
    output_dir: Union[str, Path]
    timeout_s: int = DEFAULT_TIMEOUT_S
    python_executable: Optional[str] = None


# --------------------------------------------------------------------------- #
# Script assembly                                                               #
# --------------------------------------------------------------------------- #

#: Trailing driver lines in the notebook body we strip so we can drive main()
#: ourselves under our own OUTPUT_DIR/find_data_dir overrides. Matched as substrings
#: of a stripped line; only the top-level driver calls are targeted.
_STRIP_TOPLEVEL_CALLS: Tuple[str, ...] = (
    "evidence_lift_eda()",  # post-submission EDA; must not crash the captured run
)


def _neutralize_trailing_driver(body: str) -> str:
    """Blank the notebook's own top-level ``main()`` driver + post-EDA call.

    The notebook ends with::

        RUN_FULL_PIPELINE = True
        if RUN_FULL_PIPELINE:
            main()
        else:
            print(...)
        ...
        evidence_lift_eda()

    We keep every DEFINITION (``def main``, ``def evidence_lift_eda``, all feature
    code) but neutralize the *top-level invocations* so our injected footer drives
    ``main()`` after our overrides are installed. This is a pure relocation of WHEN
    ``main()`` runs — it changes nothing the pipeline computes.

    ``RUN_FULL_PIPELINE`` is forced to ``False`` so the notebook's own
    ``if RUN_FULL_PIPELINE: main()`` block does not fire (our footer calls ``main()``
    explicitly). Any bare top-level ``evidence_lift_eda()`` call is blanked so the
    post-submission EDA cell cannot fail the run after ``submission.csv`` is written.
    """
    out_lines: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        # Force the notebook's pipeline gate off; our footer drives main() instead.
        if stripped.startswith("RUN_FULL_PIPELINE") and "=" in stripped and "==" not in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}RUN_FULL_PIPELINE = False  # neutralized: footer drives main()")
            continue
        # Blank a bare top-level post-submission EDA call (no leading indent = top level).
        if any(stripped == call for call in _STRIP_TOPLEVEL_CALLS) and line == stripped:
            out_lines.append(f"# neutralized top-level call: {stripped}")
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


#: Injected header: forces matplotlib headless BEFORE any notebook cell imports it,
#: and makes plt.show() a no-op so EDA cells never block. Authored here (trusted).
_HEADLESS_HEADER = '''\
# ==== direct_repro header (AUTHORED BY THE RUNNER) — headless matplotlib ====
import os as _dr_os
_dr_os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib as _dr_mpl
    _dr_mpl.use("Agg", force=True)
    import matplotlib.pyplot as _dr_plt
    _dr_plt.show = lambda *a, **k: None  # EDA plt.show() -> no-op (never blocks)
except Exception as _dr_exc:  # matplotlib may not be imported yet; header env still applies
    pass
# ==== end direct_repro header ====
'''


def _build_footer(data_dir: Path, output_dir: Path) -> str:
    """Injected footer placed AFTER the notebook body.

    Overrides ``find_data_dir`` and ``OUTPUT_DIR`` in the module globals (so they win
    over the notebook's own definitions), re-points any already-defined ``DATA_DIR``
    for the post-cache EDA, then explicitly drives ``main()``. Uses ``repr`` so
    Windows backslashes are escaped safely.
    """
    data_literal = repr(str(data_dir))
    out_literal = repr(str(output_dir))
    return f'''\
# ==== direct_repro footer (AUTHORED BY THE RUNNER) — drive the pipeline ====
import pathlib as _dr_pathlib

_DR_DATA_DIR = _dr_pathlib.Path({data_literal}).resolve()
_DR_OUTPUT_DIR = _dr_pathlib.Path({out_literal}).resolve()
_DR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_dir(*_a, **_k):
    """Override: return the local competition data root (wins over notebook's rglob)."""
    return _DR_DATA_DIR


# Repoint the module-level globals the notebook body reads.
OUTPUT_DIR = _DR_OUTPUT_DIR
DATA_DIR = _DR_DATA_DIR
DATA_OK = True

print("direct_repro: DATA_DIR =", _DR_DATA_DIR, flush=True)
print("direct_repro: OUTPUT_DIR =", _DR_OUTPUT_DIR, flush=True)
print("direct_repro: driving main() ...", flush=True)
main()
print("direct_repro: main() returned; submission at", _DR_OUTPUT_DIR / "submission.csv", flush=True)
# ==== end direct_repro footer ====
'''


def build_direct_script(
    notebook: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> str:
    """Assemble the full runnable script: header + notebook body + footer.

    The notebook's CODE cells are exported (``__future__`` imports hoisted to the top
    per :func:`_export_notebook_to_script`), the trailing driver is neutralized, and
    the headless header + pipeline-driving footer are wrapped around it.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    future_imports, body = _export_notebook_to_script(Path(notebook))
    body = _neutralize_trailing_driver(body)
    future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
    footer = _build_footer(data_dir, output_dir)
    return (
        future_block
        + _HEADLESS_HEADER
        + "\n\n"
        + body
        + "\n\n"
        + footer
    )


def _child_env(output_dir: Path) -> Dict[str, str]:
    """Child env: headless matplotlib + UTF-8 stdio + jailed mpl cache.

    Inherits the parent env (so ``poker_collusion`` stays importable via PYTHONPATH),
    then forces the minimal overrides. We do NOT strip credentials here (this is a
    TRUSTED, owned notebook run directly — the sandbox strip belongs to the untrusted
    path), but we DO force Agg + UTF-8 so the run is deterministic and non-blocking.
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output_dir / ".mpl_cache")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # The notebook's find_data_dir would try /kaggle/input first; our footer overrides
    # it, but clear the Kaggle marker anyway so no cell branches into a Kaggle path.
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    return env


def _tail(text: Optional[str], limit: int = 3000) -> str:
    if not text:
        return ""
    return text[-limit:]


def _diagnose_failure(log_text: str) -> Tuple[str, str]:
    """Pin a non-zero exit to a (stage, detail) from the decodable child log."""
    tail = _tail(log_text)
    import re

    m = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", log_text)
    if m or "ImportError" in log_text:
        module = m.group(1) if m else "<unresolved import>"
        return STAGE_DEPS, f"missing dependency {module!r}. Traceback tail:\n{tail}"
    if "MemoryError" in log_text:
        return STAGE_COMPUTE, f"ran out of memory (MemoryError). Traceback tail:\n{tail}"
    fm = re.search(r"FileNotFoundError: .*?([^\s'\"]+\.(?:csv|parquet|json|npy|txt))", log_text)
    if fm or "FileNotFoundError" in log_text:
        target = fm.group(1) if fm else "<file>"
        return STAGE_DATA_PATH, f"a required file was not found: {target}. Traceback tail:\n{tail}"
    return STAGE_RUNTIME, f"uncaught exception in a notebook cell. Traceback tail:\n{tail}"


def _validate_submission(submission_path: Path, data_dir: Path) -> Tuple[bool, int, str]:
    """Validate the emitted submission against the local sample_submission.csv.

    Returns ``(ok, n_rows, reason)``; ``reason`` empty on success. Reuses the official
    :func:`poker_collusion.submission.writer.validate_submission` so a locally-valid
    file is leaderboard-valid (schema, row count, exact eval pair-id set).
    """
    try:
        import pandas as pd
        from poker_collusion.submission.writer import (
            SubmissionValidationError,
            validate_submission,
        )
    except Exception as exc:
        return False, 0, f"could not import the submission validator: {exc}"

    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.is_file():
        return False, 0, f"sample_submission.csv not found under {data_dir}"
    try:
        frame = pd.read_csv(submission_path, dtype={"pair_id": str})
        sample = pd.read_csv(sample_path, dtype={"pair_id": str})
    except Exception as exc:
        return False, 0, f"could not read submission/sample csv: {exc}"

    n_rows = int(len(frame))
    try:
        validate_submission(frame, sample)
    except SubmissionValidationError as exc:
        return False, n_rows, str(exc)
    return True, n_rows, ""


# --------------------------------------------------------------------------- #
# Single-job execution                                                          #
# --------------------------------------------------------------------------- #


def run_direct_job(job: DirectReproJob) -> DirectReproResult:
    """Run ONE reproduction job directly in a child process, timed + capped.

    Writes the assembled script + captured log into ``job.output_dir`` (persistent),
    runs it as a child with the headless/UTF-8 env under ``job.timeout_s``, and
    returns a terminal :class:`DirectReproResult`. Also writes the outcome JSON to
    ``<output_dir>/_direct_repro_result.json``.
    """
    notebook = Path(job.notebook)
    data_dir = Path(job.data_dir).resolve()
    output_dir = Path(job.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"_{job.name}_run.log"
    result_json = output_dir / "_direct_repro_result.json"
    script_path = output_dir / f"_{job.name}_direct.py"
    interpreter = job.python_executable or sys.executable

    def blocked(stage: str, detail: str, elapsed: float, rc: Optional[int]) -> DirectReproResult:
        res = DirectReproResult(
            name=job.name,
            status=STATUS_BLOCKED,
            elapsed_s=elapsed,
            output_dir=str(output_dir),
            log_path=str(log_path),
            failure_stage=stage,
            failure_detail=detail,
            returncode=rc,
        )
        _write_result_json(result_json, res)
        return res

    # ---- pre-checks (fail fast with an honest stage) ----
    if not notebook.exists():
        return blocked(STAGE_RUNTIME, f"notebook not found: {notebook}", 0.0, None)
    if not data_dir.is_dir():
        return blocked(STAGE_DATA_PATH, f"data_dir not found: {data_dir}", 0.0, None)

    try:
        script = build_direct_script(notebook, data_dir, output_dir)
    except Exception as exc:
        return blocked(STAGE_RUNTIME, f"failed to assemble the direct script: {exc}", 0.0, None)
    script_path.write_text(script, encoding="utf-8")

    env = _child_env(output_dir)
    start = time.monotonic()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"# direct_repro job={job.name} notebook={notebook}\n")
        log_handle.write(f"# data_dir={data_dir}\n# output_dir={output_dir}\n")
        log_handle.write(f"# timeout_s={job.timeout_s} interpreter={interpreter}\n\n")
        log_handle.flush()
        try:
            completed = subprocess.run(
                [interpreter, str(script_path)],
                cwd=str(output_dir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=job.timeout_s,
            )
            rc: Optional[int] = completed.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            log_handle.write(f"\n# TIMEOUT after {job.timeout_s}s\n")
            return blocked(
                STAGE_COMPUTE,
                f"wall-clock timeout after {job.timeout_s}s (3h hard cap); the "
                "notebook did not emit a submission in time.",
                elapsed,
                None,
            )
        except (OSError, ValueError) as exc:
            elapsed = time.monotonic() - start
            return blocked(
                STAGE_DEPS,
                f"could not launch the child interpreter {interpreter!r}: {exc}",
                elapsed,
                None,
            )

    elapsed = time.monotonic() - start
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log_text = ""

    if rc != 0:
        stage, detail = _diagnose_failure(log_text)
        return blocked(stage, f"child exited {rc}: {detail}", elapsed, rc)

    # ---- capture + validate the emitted submission (never partial) ----
    emitted = output_dir / _SUBMISSION_FILENAME
    if not emitted.is_file():
        return blocked(
            STAGE_RUNTIME,
            f"child exited 0 but emitted no {_SUBMISSION_FILENAME!r} in {output_dir}; "
            "nothing to capture (never reported as partial success).",
            elapsed,
            rc,
        )

    ok, n_rows, reason = _validate_submission(emitted, data_dir)
    if not ok:
        return blocked(
            STAGE_RUNTIME,
            f"emitted {_SUBMISSION_FILENAME!r} ({n_rows} rows) failed the official "
            f"submission contract: {reason}",
            elapsed,
            rc,
        )

    res = DirectReproResult(
        name=job.name,
        status=STATUS_SUCCEEDED,
        elapsed_s=elapsed,
        output_dir=str(output_dir),
        log_path=str(log_path),
        submission_path=str(emitted),
        n_rows=n_rows,
        returncode=rc,
    )
    _write_result_json(result_json, res)
    return res


def _write_result_json(path: Path, result: DirectReproResult) -> None:
    payload = dict(result.to_json())
    payload["expected_eval_rows"] = _EXPECTED_EVAL_ROWS
    payload["rows_match_expected"] = (result.n_rows == _EXPECTED_EVAL_ROWS)
    payload["note"] = (
        "EVAL submission (112,540 eval pairs, disjoint from dev labels). NOT a dev "
        "AP; ladder scoring is a separate downstream step (re-run recipe on the DEV "
        "holdout)."
    )
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Parallel driver                                                               #
# --------------------------------------------------------------------------- #


def run_jobs_parallel(
    jobs: List[DirectReproJob],
    max_concurrency: int = 4,
    on_result: Optional[Callable[[DirectReproResult], None]] = None,
) -> Dict[str, DirectReproResult]:
    """Run up to ``max_concurrency`` reproduction jobs concurrently.

    Each job runs its OWN child process into its OWN persistent ``output_dir`` with
    its OWN log — fully independent, so ladder-recipe reconstructions can be added as
    more jobs later. A thread pool is used because :func:`run_direct_job` blocks on a
    child ``subprocess`` (the heavy work is in the child, not the driver thread), and
    the RAM cap is enforced by capping concurrency (each feature build ~2GB observed;
    default 4 keeps total under ~8-10GB, safe on 14GB free).

    Args:
        jobs: The jobs to run.
        max_concurrency: Max simultaneous child processes (RAM-safety cap).
        on_result: Optional callback fired as each job finishes (for progress).

    Returns:
        ``{job.name: DirectReproResult}`` for every job.
    """
    if not jobs:
        return {}
    workers = max(1, min(max_concurrency, len(jobs)))
    results: Dict[str, DirectReproResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_name = {pool.submit(run_direct_job, job): job.name for job in jobs}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                res = future.result()
            except Exception as exc:  # a driver-level crash, not a child failure
                res = DirectReproResult(
                    name=name,
                    status=STATUS_BLOCKED,
                    elapsed_s=0.0,
                    output_dir="",
                    log_path="",
                    failure_stage=STAGE_RUNTIME,
                    failure_detail=f"driver-level exception: {exc}",
                )
            results[name] = res
            if on_result is not None:
                on_result(res)
    return results


# --------------------------------------------------------------------------- #
# The 0.64469 competitor job factory                                            #
# --------------------------------------------------------------------------- #


def competitor_064469_job(
    poker_root: Union[str, Path],
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> DirectReproJob:
    """Build the 0.64469 competitor reproduction job.

    Args:
        poker_root: The ``poker/`` project root (contains ``data/poker`` and
            ``_refs/honghanh``).
        timeout_s: Wall-clock hard cap (default 3h).
    """
    poker_root = Path(poker_root)
    notebook = (
        poker_root
        / "_refs"
        / "honghanh"
        / "detecting-collusive-value-transfer-in-6-max-poker.ipynb"
    )
    data_dir = poker_root / "data" / "poker"
    output_dir = poker_root / "outputs" / "poker_collusion" / "repro_064469"
    return DirectReproJob(
        name="repro_064469",
        notebook=notebook,
        data_dir=data_dir,
        output_dir=output_dir,
        timeout_s=timeout_s,
    )
