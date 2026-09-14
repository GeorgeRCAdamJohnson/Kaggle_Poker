"""Direct (owned-source) reproduction runner for the nomannic PU-aware evidence
ranker notebook — a sibling of :mod:`anchor_repro.direct_repro` (the 0.64469
honghanh runner), adapted to the STRUCTURAL differences of the nomannic notebook.

Why a separate runner (contract Rule 6 — reuse, do not reinvent; but honestly note
where the two notebooks differ):

* honghanh wraps its whole pipeline in a ``main()`` and discovers its data root via
  a ``find_data_dir()`` rglob; :mod:`direct_repro` therefore NEUTRALIZES the trailing
  ``main()`` driver and injects a footer that overrides ``find_data_dir`` +
  ``OUTPUT_DIR`` and calls ``main()`` after the overrides are installed.
* nomannic has NO ``main()`` — it runs as top-level cells in order — and it does NOT
  use ``find_data_dir``. Instead it hard-codes THREE Kaggle paths as string literals:
    - ``DATA_DIR  = Path("/kaggle/input/competitions/detect-suspicious-value-transfers-in-poker")``  (cell 2)
    - ``PREP_DIR  = Path("/kaggle/working/prepared_v2")``                                            (cell 29)
    - ``submission.write_csv("/kaggle/working/submission.csv")``                                     (cell 40)
  So the footer-drives-``main()`` strategy does not apply. This runner instead does
  TARGETED, EXACT-STRING path-literal rewrites on the exported script so those three
  Kaggle paths point at the local data root / a persistent output dir, and injects a
  headless header that makes ``display`` / ``plt.show`` no-ops (the notebook's EDA
  cells call the IPython ``display`` builtin, which does not exist in a plain
  ``python`` child, so it MUST be stubbed or the EDA cells crash the run).

Everything else — the child-process isolation, UTF-8 stdio, Agg matplotlib, the
timeout cap, and the "SUCCEEDED only when submission.csv exists AND validates against
the local sample_submission.csv" honesty contract — is reused verbatim from
:mod:`direct_repro`. The data is read-only; the notebook source is never modified.

Contract notes (Rules 1, 3, 9):
* SUCCEEDED requires a captured, schema-valid ``submission.csv`` — never partial.
* On any failure the job is BLOCKED with the real stage + decodable traceback tail;
  no fabricated submission.
* Reported deviations (e.g. that this notebook DOES use xgboost, contrary to the
  task brief) are stated plainly, not hidden.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Union

# Reuse the trusted machinery from the honghanh direct runner.
from anchor_repro.direct_repro import (  # noqa: F401  (re-exported vocab)
    STATUS_SUCCEEDED,
    STATUS_BLOCKED,
    STAGE_DEPS,
    STAGE_DATA_PATH,
    STAGE_COMPUTE,
    STAGE_RUNTIME,
    DEFAULT_TIMEOUT_S,
    DirectReproJob,
    DirectReproResult,
    run_direct_job,
    _child_env,  # noqa: F401  (kept for parity; run_direct_job uses it internally)
)
from anchor_repro.repro_runner import _export_notebook_to_script

__all__ = [
    "build_nomannic_script",
    "nomannic_job",
    "run_nomannic",
]


# --------------------------------------------------------------------------- #
# The three hard-coded Kaggle path literals in the nomannic notebook.           #
# These are matched EXACTLY (as they appear in the exported cell source) and    #
# rewritten to local paths. If a match is missing the build fails loudly so we  #
# never silently run against the wrong (Kaggle) path.                           #
# --------------------------------------------------------------------------- #

_KAGGLE_DATA_LITERAL = '"/kaggle/input/competitions/detect-suspicious-value-transfers-in-poker"'
_KAGGLE_PREP_LITERAL = '"/kaggle/working/prepared_v2"'
_KAGGLE_SUB_LITERAL = '"/kaggle/working/submission.csv"'


#: Injected header (AUTHORED BY THE RUNNER, trusted): headless matplotlib BEFORE any
#: cell imports it, and IPython ``display`` + ``plt.show`` stubbed to no-ops so the
#: notebook's EDA cells (which call the notebook-only ``display`` builtin) run headless
#: in a plain ``python`` child instead of raising ``NameError: display``.
_NOMANNIC_HEADER = '''\
# ==== direct_repro_nomannic header (AUTHORED BY THE RUNNER) ====
import os as _dr_os
_dr_os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib as _dr_mpl
    _dr_mpl.use("Agg", force=True)
    import matplotlib.pyplot as _dr_plt
    _dr_plt.show = lambda *a, **k: None  # EDA plt.show() -> no-op (never blocks)
except Exception:
    pass
import builtins as _dr_builtins
# The IPython `display` builtin does not exist in a plain python child; the notebook's
# EDA cells call it. Make it a no-op so EDA cells run headless instead of NameError-ing.
if not hasattr(_dr_builtins, "display"):
    _dr_builtins.display = lambda *a, **k: None
def display(*a, **k):  # also bind at module scope (some cells shadow builtins lookup)
    return None
# ==== end header ====
'''


def _rewrite_kaggle_paths(body: str, data_dir: Path, output_dir: Path) -> str:
    """Rewrite the three hard-coded Kaggle path literals to local paths.

    * DATA_DIR literal  -> the local ``data/poker`` root (read-only competition data).
    * PREP_DIR literal  -> ``<output_dir>/prepared_v2`` (persistent warm feature cache).
    * submission path   -> ``<output_dir>/submission.csv`` (captured + validated).

    Uses ``repr(str(path))`` so Windows backslashes are escaped safely. Every
    substitution is asserted present so a moved/renamed literal fails the build
    loudly rather than silently running against the Kaggle mount.
    """
    data_literal = repr(str(data_dir.resolve()))
    prep_literal = repr(str((output_dir / "prepared_v2").resolve()))
    sub_literal = repr(str((output_dir / "submission.csv").resolve()))

    substitutions: List[Tuple[str, str, str]] = [
        (_KAGGLE_DATA_LITERAL, data_literal, "DATA_DIR"),
        (_KAGGLE_PREP_LITERAL, prep_literal, "PREP_DIR"),
        (_KAGGLE_SUB_LITERAL, sub_literal, "submission path"),
    ]
    for needle, replacement, label in substitutions:
        if needle not in body:
            raise ValueError(
                f"expected Kaggle {label} literal {needle!r} not found in the exported "
                "notebook body; the notebook changed and the path remap is unsafe."
            )
        body = body.replace(needle, replacement)
    return body


def build_nomannic_script(
    notebook: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> str:
    """Assemble the runnable script: header + path-remapped notebook body.

    The notebook's CODE cells are exported (``__future__`` hoisted per
    :func:`_export_notebook_to_script`), the three Kaggle path literals are rewritten
    to local paths, and the headless/``display``-stub header is prepended. No footer
    is needed: the pipeline is top-level and writes ``submission.csv`` itself (now at
    the remapped local path).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    future_imports, body = _export_notebook_to_script(Path(notebook))
    body = _rewrite_kaggle_paths(body, data_dir, output_dir)
    future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
    return future_block + _NOMANNIC_HEADER + "\n\n" + body + "\n"


def nomannic_job(
    poker_root: Union[str, Path],
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> DirectReproJob:
    """Build the nomannic reproduction job (mirrors :func:`competitor_064469_job`)."""
    poker_root = Path(poker_root)
    notebook = (
        poker_root
        / "_refs"
        / "nomannic"
        / "poker-collusion-pu-aware-evidence-ranker.ipynb"
    )
    data_dir = poker_root / "data" / "poker"
    output_dir = poker_root / "outputs" / "poker_collusion" / "repro_nomannic"
    return DirectReproJob(
        name="repro_nomannic",
        notebook=notebook,
        data_dir=data_dir,
        output_dir=output_dir,
        timeout_s=timeout_s,
    )


def run_nomannic(poker_root: Union[str, Path], timeout_s: int = DEFAULT_TIMEOUT_S) -> DirectReproResult:
    """Assemble + run the nomannic job in a child process, returning a terminal result.

    Monkeypatches :func:`anchor_repro.direct_repro.build_direct_script` for the duration
    of this call so the shared :func:`run_direct_job` (which owns the child-process
    isolation, UTF-8 stdio, timeout, and submission validation) assembles the script
    with THIS runner's nomannic-specific header + path remap instead of the honghanh
    footer strategy.
    """
    import anchor_repro.direct_repro as _dr

    job = nomannic_job(poker_root, timeout_s=timeout_s)
    original_builder = _dr.build_direct_script
    _dr.build_direct_script = build_nomannic_script  # type: ignore[assignment]
    try:
        return run_direct_job(job)
    finally:
        _dr.build_direct_script = original_builder  # type: ignore[assignment]


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[1]  # poker/
    print(f"[nomannic repro] poker_root={root}", flush=True)
    start = time.monotonic()
    result = run_nomannic(root)
    elapsed = time.monotonic() - start
    print(f"[nomannic repro] status={result.status} elapsed={elapsed:.1f}s", flush=True)
    print(f"[nomannic repro] {result.to_json()}", flush=True)
    sys.exit(0 if result.succeeded else 1)
