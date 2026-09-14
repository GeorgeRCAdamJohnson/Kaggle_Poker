"""Faithful reproduction of the lamhuy_topological competitor notebook (dossier §84).

This module reproduces the ``lamhuy_topological`` Kaggle notebook
(``_refs/lamhuy_topological/topological-collusion-dynamics-value-transfer-eda.ipynb``)
as a NEW local anchor, mirroring how :mod:`anchor_repro.competitor_recipe` reproduced
the honghanh 0.64469 notebook. Governed by the workspace
``reverse-engineering-accountability`` contract.

BUILD ONLY — this NEVER submits to Kaggle and NEVER touches ``submission.csv`` /
``submission_best_*.csv`` in the repo root. The eval submission the notebook produces
is written under ``outputs/poker_collusion/repro_lamhuy/`` only.

Why this module exists (§84).
-----------------------------
The lamhuy notebook beat honghanh on the real leaderboard (LB 0.64262). It is
honghanh's lineage scaled up: a triple-GBDT risk head (XGB 0.40 + LGBM 0.35 +
CatBoost 0.25), decoupled OvR specialists unioned into the risk (final risk =
0.50*triple + 0.50*decoupled), 60-unknown-per-table PU sampling with PU-stress eval
weighting, table-relative percentile + EB-shrunk/Welch-t partner-field contrast
features, and a dual-engine evidence retriever (specialist quad-ensemble 0.75 +
global 0.25) scored over the top 50k pairs. We reproduce it verbatim to get a local
anchor whose OOF projected-composite and eval submission we can compare to honghanh.

How it runs (faithful, in a CHILD PROCESS — mirrors competitor_recipe.py).
--------------------------------------------------------------------------
The notebook runs top-to-bottom (no ``main()`` guard). We:

1. Export its CODE cells to a runnable ``.py`` body via
   :func:`anchor_repro.repro_runner._export_notebook_to_script` (``__future__``
   hoisting handled there).
2. STRING-REPLACE the two Kaggle path literals so everything resolves locally:
   ``Path("/kaggle/input")`` -> the local data PARENT (``poker/data``), so the
   notebook's ``INPUT_ROOT.rglob("players.parquet")`` finds ``poker/data/poker``;
   and ``Path("/kaggle/working")`` -> the local scratch PARENT
   (``.../repro_lamhuy``), so ``PREP_DIR`` becomes
   ``.../repro_lamhuy/prepared_v13`` and the final ``submission.csv`` becomes
   ``.../repro_lamhuy/submission.csv``.
3. NEUTRALIZE the EDA/plotting: ``MPLBACKEND=Agg`` + monkeypatched
   ``plt.show``/``plt.tight_layout``, AND every code cell that touches matplotlib or
   seaborn is wrapped in a try/except so a plotting failure prints a warning but does
   NOT abort the model + submission pipeline (the models + submission are what
   matter).
4. Run the child end-to-end (triple-GBDT risk CV, decoupled specialists, behavior
   calibration, dual-engine evidence, eval inference). This is ~1-2h; the child
   wall-clock cap is 7200s.
5. Capture the notebook's OWN printed projected-composite components (its cell 33:
   ``OOF PU Stress Pair AP`` / ``OOF Evidence MAP@5`` / ``OOF PU Stress Behavior MAP``
   / ``PROJECTED COMPOSITE SCORE``) from the child log.

Then in the PARENT (:func:`main`) we VALIDATE the produced eval CSV against the
official contract, COMPARE it to honghanh's eval submission, and write a JSON summary
— never fabricating a CSV or a score if the child fails (contract Rules 1, 9).

Run:  ``python -m anchor_repro.lamhuy_recipe``  (from ``poker/``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from anchor_repro.repro_runner import _export_notebook_to_script

__all__ = [
    "LamhuyPaths",
    "default_lamhuy_paths",
    "build_child_script",
    "run_child",
    "validate_submission_csv",
    "compare_to_honghanh",
    "main",
]

#: Human-readable id of this anchor (dossier §84).
LAMHUY_RECIPE_ID: str = "lamhuy_topological_repro"

#: The real leaderboard score the notebook achieved (recorded for the audit trail;
#: NOT a local claim — the parent measures its OWN local validation numbers).
LAMHUY_REPORTED_LB: float = 0.64262

#: Child wall-clock cap. The full notebook (triple-GBDT CV + specialists + dual-engine
#: evidence over 3M+ hand rows + eval inference) is ~1-2h; 7200s is the requested cap.
_CHILD_TIMEOUT_S: int = 7200

#: The official eval-submission contract.
_SUBMISSION_COLUMNS: List[str] = [
    "pair_id",
    "risk_score",
    "predicted_behavior",
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
]
_EVIDENCE_COLS: List[str] = [f"evidence_hand_{i}" for i in range(1, 6)]
_EXPECTED_ROWS: int = 112_540


@dataclass(frozen=True)
class LamhuyPaths:
    """Resolved on-disk paths the lamhuy reproduction uses.

    Attributes:
        notebook: The lamhuy competitor notebook (untrusted source; its OWN code is
            exported and run behind the path/plot shims).
        data_dir: The local competition data root (``poker/data/poker``), read-only.
        data_parent: ``data_dir.parent`` (``poker/data``) — what ``/kaggle/input`` is
            remapped to, so the notebook's ``rglob("players.parquet")`` finds the data.
        scratch_parent: The local scratch parent (``.../repro_lamhuy``) — what
            ``/kaggle/working`` is remapped to; ``PREP_DIR`` becomes
            ``scratch_parent/prepared_v13`` and ``submission.csv`` lands here.
        submission_csv: The produced eval submission (``scratch_parent/submission.csv``).
        prepared_dir: The feature cache (``scratch_parent/prepared_v13``), KEPT.
        honghanh_submission: honghanh's eval submission for the comparison
            (``repro_064469/submission.csv``).
        output_dir: ``scratch_parent`` (where the child log + summary are written).
    """

    notebook: Path
    data_dir: Path
    data_parent: Path
    scratch_parent: Path
    submission_csv: Path
    prepared_dir: Path
    honghanh_submission: Path
    output_dir: Path

    def missing(self) -> List[Path]:
        """Return required INPUTS that are absent from disk."""
        needed = [
            self.notebook,
            self.data_dir / "players.parquet",
            self.data_dir / "hands.parquet",
            self.data_dir / "seats.parquet",
            self.data_dir / "actions.parquet",
            self.data_dir / "development_labels.csv",
            self.data_dir / "development_evidence.csv",
            self.data_dir / "evaluation_pairs.csv",
            self.data_dir / "sample_submission.csv",
        ]
        return [p for p in needed if not Path(p).exists()]


def default_lamhuy_paths(poker_root: os.PathLike | str) -> LamhuyPaths:
    """Resolve the standard lamhuy reproduction paths under a ``poker/`` tree root."""
    root = Path(poker_root)
    data_dir = (root / "data" / "poker").resolve()
    scratch = (root / "outputs" / "poker_collusion" / "repro_lamhuy").resolve()
    return LamhuyPaths(
        notebook=(
            root
            / "_refs"
            / "lamhuy_topological"
            / "topological-collusion-dynamics-value-transfer-eda.ipynb"
        ).resolve(),
        data_dir=data_dir,
        data_parent=data_dir.parent,
        scratch_parent=scratch,
        submission_csv=scratch / "submission.csv",
        prepared_dir=scratch / "prepared_v13",
        honghanh_submission=(
            root / "outputs" / "poker_collusion" / "repro_064469" / "submission.csv"
        ).resolve(),
        output_dir=scratch,
    )


# --------------------------------------------------------------------------- #
# Child-script assembly: export notebook body, rewrite paths, shim plotting.    #
# --------------------------------------------------------------------------- #

#: Matches the cell markers `_export_notebook_to_script` emits.
_CELL_MARKER_RE = re.compile(r"^# --- notebook code cell (\d+) ---\s*$")

#: A code cell that touches these is treated as an EDA/plot cell and wrapped so a
#: failure does not abort the pipeline. `plt.`/`sns.` cover matplotlib + seaborn;
#: `.show(` / `tight_layout` cover the blocking draw calls.
_PLOT_TOKENS = ("plt.", "sns.", "tight_layout", ".show(", "subplots(")

#: Header authored HERE (trusted) — headless matplotlib + neutralised show/draw so no
#: EDA cell can block or open a window. `tight_layout` is also neutralised because the
#: notebook calls it before `show()` and it can warn/throw on the Agg backend with
#: complex layouts. Placed before the notebook body.
_HEADER = (
    "import os as _lh_os\n"
    '_lh_os.environ.setdefault("MPLBACKEND", "Agg")\n'
    "try:\n"
    "    import matplotlib as _lh_mpl\n"
    '    _lh_mpl.use("Agg", force=True)\n'
    "    import matplotlib.pyplot as _lh_plt\n"
    "    _lh_plt.show = lambda *a, **k: None\n"
    "    _lh_plt.tight_layout = lambda *a, **k: None\n"
    "    _lh_plt.pause = lambda *a, **k: None\n"
    "except Exception as _lh_e:\n"
    '    print("lamhuy_recipe: matplotlib shim skipped:", _lh_e, flush=True)\n'
    "\n"
)


def _rewrite_paths(body: str, paths: LamhuyPaths) -> str:
    """Rewrite the notebook's two Kaggle path literals to local paths.

    The notebook resolves ``DATA_DIR`` by ``INPUT_ROOT.rglob("players.parquet")`` under
    ``INPUT_ROOT = Path("/kaggle/input")`` — so remapping ``Path("/kaggle/input")`` to
    the local data PARENT (``poker/data``) makes the rglob find ``poker/data/poker``.
    ``PREP_DIR = Path("/kaggle/working/prepared_v13")`` and the final
    ``out_csv = Path("/kaggle/working/submission.csv")`` are remapped by rewriting
    ``Path("/kaggle/working")`` to the local scratch PARENT (``.../repro_lamhuy``),
    which yields ``.../repro_lamhuy/prepared_v13`` and
    ``.../repro_lamhuy/submission.csv``.

    Uses ``repr`` for Windows-safe path literals (backslashes escaped). We match both
    single- and double-quoted forms defensively.
    """
    data_parent_lit = repr(str(paths.data_parent))
    scratch_lit = repr(str(paths.scratch_parent))

    replacements = {
        'Path("/kaggle/input")': f"Path({data_parent_lit})",
        "Path('/kaggle/input')": f"Path({data_parent_lit})",
        'Path("/kaggle/working")': f"Path({scratch_lit})",
        "Path('/kaggle/working')": f"Path({scratch_lit})",
    }
    n_replaced = 0
    for old, new in replacements.items():
        count = body.count(old)
        if count:
            body = body.replace(old, new)
            n_replaced += count

    # Defensive: any REMAINING absolute /kaggle/working or /kaggle/input string
    # literal (e.g. a path built without the Path(...) wrapper) is remapped too, so
    # nothing escapes to a non-existent Kaggle mount. These are plain string prefixes.
    body = body.replace(
        '"/kaggle/working', '"' + str(paths.scratch_parent).replace("\\", "\\\\")
    )
    body = body.replace(
        "'/kaggle/working", "'" + str(paths.scratch_parent).replace("\\", "\\\\")
    )
    body = body.replace(
        '"/kaggle/input', '"' + str(paths.data_parent).replace("\\", "\\\\")
    )
    body = body.replace(
        "'/kaggle/input", "'" + str(paths.data_parent).replace("\\", "\\\\")
    )

    if n_replaced == 0:
        raise RuntimeError(
            "lamhuy_recipe: expected to rewrite Path(\"/kaggle/input\") and "
            'Path("/kaggle/working") literals but found none — the notebook path '
            "handling changed; refusing to run with unshimmed paths."
        )
    return body


def _wrap_plot_cells(body: str) -> str:
    """Wrap every EDA/plot code cell in a try/except so a plot failure never aborts.

    The exported body is a concatenation of cells delimited by
    ``# --- notebook code cell N ---`` markers. For each cell whose source touches
    matplotlib/seaborn, we indent the cell body under a ``try:`` and swallow any
    exception with a printed warning. Non-plot cells are emitted verbatim. This keeps
    the model + submission cells running exactly as written.
    """
    lines = body.splitlines()
    # Partition into (marker_line, [cell_body_lines]) blocks.
    blocks: List[tuple] = []
    cur_marker: Optional[str] = None
    cur_lines: List[str] = []
    preamble: List[str] = []
    for line in lines:
        if _CELL_MARKER_RE.match(line):
            if cur_marker is not None:
                blocks.append((cur_marker, cur_lines))
            elif cur_lines:
                preamble = cur_lines
            cur_marker = line
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_marker is not None:
        blocks.append((cur_marker, cur_lines))
    elif cur_lines and not blocks:
        preamble = cur_lines

    out: List[str] = list(preamble)
    for marker, cell_lines in blocks:
        cell_src = "\n".join(cell_lines)
        is_plot = any(tok in cell_src for tok in _PLOT_TOKENS)
        out.append(marker)
        if is_plot and cell_src.strip():
            cell_no = _CELL_MARKER_RE.match(marker).group(1)
            out.append("try:")
            for cl in cell_lines:
                out.append("    " + cl if cl.strip() else cl)
            out.append("except Exception as _lh_plot_err:")
            out.append(
                f'    print("lamhuy_recipe: EDA/plot cell {cell_no} failed '
                '(non-fatal):", repr(_lh_plot_err), flush=True)'
            )
        else:
            out.extend(cell_lines)
    return "\n".join(out)


def build_child_script(paths: LamhuyPaths) -> str:
    """Assemble the full child script: header + path-rewritten, plot-wrapped body."""
    future_imports, body = _export_notebook_to_script(Path(paths.notebook))
    body = _rewrite_paths(body, paths)
    body = _wrap_plot_cells(body)
    future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
    return future_block + _HEADER + body + "\n"


# --------------------------------------------------------------------------- #
# Child execution + log parsing                                                 #
# --------------------------------------------------------------------------- #

#: The notebook's own cell-33 print lines we parse for the projected composite. Anchor
#: on the exact label text so we read the notebook's OWN reported numbers (Rule 1).
_METRIC_PATTERNS: Dict[str, re.Pattern] = {
    "oof_pu_stress_pair_ap": re.compile(r"OOF PU Stress Pair AP:\s*([0-9.]+)"),
    "oof_evidence_map5": re.compile(r"OOF Evidence MAP@5:\s*([0-9.]+)"),
    "oof_pu_stress_behavior_map": re.compile(r"OOF PU Stress Behavior MAP:\s*([0-9.]+)"),
    "projected_composite": re.compile(r"PROJECTED COMPOSITE SCORE:\s*([0-9.]+)"),
}


def _parse_metrics_from_log(log_text: str) -> Dict[str, Optional[float]]:
    """Extract the notebook's cell-33 projected-composite components from the log.

    Returns the LAST match for each label (the final cell-33 print), or ``None`` if a
    label never appeared (⇒ honestly recorded as missing, never fabricated).
    """
    out: Dict[str, Optional[float]] = {}
    for key, pat in _METRIC_PATTERNS.items():
        matches = pat.findall(log_text)
        out[key] = float(matches[-1]) if matches else None
    return out


def run_child(paths: LamhuyPaths, timeout_s: int = _CHILD_TIMEOUT_S) -> Dict[str, object]:
    """Run the notebook reproduction in a child process; return a status dict.

    Writes the child script + log under ``output_dir``. Returns a dict with keys:
    ``returncode``, ``timed_out``, ``log_path``, ``script_path``, ``duration_s``,
    ``metrics`` (parsed cell-33 components), ``log_tail``. Never raises on a child
    failure — the caller inspects ``returncode``/``timed_out`` and reports honestly.
    """
    output_dir = Path(paths.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths.prepared_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "_lamhuy_repro.py"
    log_path = output_dir / "_lamhuy_repro.log"

    script = build_child_script(paths)
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output_dir / ".mpl_cache")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    (output_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    timed_out = False
    start = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(
            f"# lamhuy_recipe reproduction\n# notebook={paths.notebook}\n"
            f"# data_parent={paths.data_parent}\n# scratch_parent={paths.scratch_parent}\n"
        )
        log_handle.flush()
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(output_dir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    duration_s = time.time() - start

    log_text = ""
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass

    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "log_path": str(log_path),
        "script_path": str(script_path),
        "duration_s": duration_s,
        "metrics": _parse_metrics_from_log(log_text),
        "log_tail": log_text[-4000:],
    }


# --------------------------------------------------------------------------- #
# Parent-side validation of the produced eval CSV                               #
# --------------------------------------------------------------------------- #


def validate_submission_csv(
    submission_csv: Path, eval_pairs_csv: Path
) -> Dict[str, object]:
    """Validate the produced eval submission against the official contract.

    Checks (each reported pass/fail):
      * exactly 112540 rows;
      * 8-col schema equals the official column order;
      * pair set == ``evaluation_pairs.csv`` pair set;
      * ``risk_score`` in [0,1];
      * no nulls anywhere;
      * no duplicate evidence hand within a pair (excluding NO_EVIDENCE).

    Returns a dict of individual check results + an ``all_passed`` flag. Never
    fabricates — if the file is missing, every check is ``False`` with a reason.
    """
    checks: Dict[str, object] = {}
    if not Path(submission_csv).is_file():
        return {
            "file_exists": False,
            "reason": f"submission CSV not found at {submission_csv}",
            "all_passed": False,
        }
    checks["file_exists"] = True

    df = pd.read_csv(submission_csv)
    eval_pairs = pd.read_csv(eval_pairs_csv)

    checks["row_count_112540"] = bool(len(df) == _EXPECTED_ROWS)
    checks["row_count_actual"] = int(len(df))

    checks["schema_8col_exact"] = bool(list(df.columns) == _SUBMISSION_COLUMNS)
    checks["columns_actual"] = list(df.columns)

    sub_pairs = set(df["pair_id"].astype(str))
    exp_pairs = set(eval_pairs["pair_id"].astype(str))
    checks["pair_set_matches_eval"] = bool(sub_pairs == exp_pairs)
    checks["n_missing_pairs"] = int(len(exp_pairs - sub_pairs))
    checks["n_extra_pairs"] = int(len(sub_pairs - exp_pairs))

    if "risk_score" in df.columns:
        rs = pd.to_numeric(df["risk_score"], errors="coerce")
        checks["risk_score_in_0_1"] = bool(
            rs.notna().all() and float(rs.min()) >= 0.0 and float(rs.max()) <= 1.0
        )
        checks["risk_score_min"] = float(rs.min()) if rs.notna().any() else None
        checks["risk_score_max"] = float(rs.max()) if rs.notna().any() else None
    else:
        checks["risk_score_in_0_1"] = False

    checks["no_nulls"] = bool(int(df.isnull().sum().sum()) == 0)
    checks["total_null_cells"] = int(df.isnull().sum().sum())

    # Duplicate evidence hand within a pair (NO_EVIDENCE excluded — repeats are legal).
    present_evidence = [c for c in _EVIDENCE_COLS if c in df.columns]
    dup_rows = 0
    if len(present_evidence) == 5:
        ev = df[present_evidence].astype(str)
        for row in ev.itertuples(index=False, name=None):
            real = [h for h in row if h != "NO_EVIDENCE"]
            if len(real) != len(set(real)):
                dup_rows += 1
    checks["no_duplicate_evidence_within_pair"] = bool(dup_rows == 0)
    checks["n_rows_with_duplicate_evidence"] = int(dup_rows)

    core = [
        "file_exists",
        "row_count_112540",
        "schema_8col_exact",
        "pair_set_matches_eval",
        "risk_score_in_0_1",
        "no_nulls",
        "no_duplicate_evidence_within_pair",
    ]
    checks["all_passed"] = bool(all(bool(checks.get(k)) for k in core))
    return checks


# --------------------------------------------------------------------------- #
# Parent-side comparison to honghanh                                            #
# --------------------------------------------------------------------------- #


def compare_to_honghanh(
    lamhuy_csv: Path, honghanh_csv: Path
) -> Dict[str, object]:
    """Compare the lamhuy eval submission to honghanh's on the shared eval pairs.

    Reports: Spearman of lamhuy vs honghanh ``risk_score``; top-500 (by risk) overlap;
    lamhuy behavior distribution counts; how many pairs get non-"none" behavior;
    evidence coverage (non-NO_EVIDENCE slots) for both. If honghanh's CSV is absent the
    comparison is honestly recorded as unavailable (never fabricated).
    """
    out: Dict[str, object] = {}
    lam = pd.read_csv(lamhuy_csv)

    # lamhuy self-describing stats (independent of honghanh).
    beh = lam["predicted_behavior"].astype(str).value_counts()
    out["lamhuy_behavior_distribution"] = {k: int(v) for k, v in beh.items()}
    out["lamhuy_n_non_none_behavior"] = int((lam["predicted_behavior"].astype(str) != "none").sum())
    present_ev = [c for c in _EVIDENCE_COLS if c in lam.columns]
    lam_ev_slots = int(sum((lam[c].astype(str) != "NO_EVIDENCE").sum() for c in present_ev))
    out["lamhuy_evidence_slots_filled"] = lam_ev_slots
    out["lamhuy_evidence_total_slots"] = int(len(lam) * 5)
    out["lamhuy_pairs_with_any_evidence"] = int(
        (lam[present_ev].astype(str) != "NO_EVIDENCE").any(axis=1).sum()
    ) if len(present_ev) == 5 else None

    if not Path(honghanh_csv).is_file():
        out["honghanh_available"] = False
        out["reason"] = f"honghanh submission not found at {honghanh_csv}"
        return out
    out["honghanh_available"] = True

    hon = pd.read_csv(honghanh_csv)
    merged = lam[["pair_id", "risk_score"]].merge(
        hon[["pair_id", "risk_score"]],
        on="pair_id",
        how="inner",
        suffixes=("_lam", "_hon"),
    )
    out["n_shared_pairs"] = int(len(merged))

    if len(merged) > 1:
        lam_rank = merged["risk_score_lam"].rank()
        hon_rank = merged["risk_score_hon"].rank()
        # Spearman = Pearson of the ranks.
        spearman = float(np.corrcoef(lam_rank.values, hon_rank.values)[0, 1])
        out["spearman_risk_vs_honghanh"] = spearman
    else:
        out["spearman_risk_vs_honghanh"] = None

    top_n = 500
    lam_top = set(
        lam.sort_values("risk_score", ascending=False)["pair_id"].astype(str).head(top_n)
    )
    hon_top = set(
        hon.sort_values("risk_score", ascending=False)["pair_id"].astype(str).head(top_n)
    )
    out["top_500_overlap_count"] = int(len(lam_top & hon_top))
    out["top_500_overlap_frac"] = float(len(lam_top & hon_top) / top_n)

    # honghanh comparison stats (for context).
    hon_beh = hon["predicted_behavior"].astype(str).value_counts()
    out["honghanh_behavior_distribution"] = {k: int(v) for k, v in hon_beh.items()}
    out["honghanh_n_non_none_behavior"] = int(
        (hon["predicted_behavior"].astype(str) != "none").sum()
    )
    hon_present_ev = [c for c in _EVIDENCE_COLS if c in hon.columns]
    out["honghanh_evidence_slots_filled"] = int(
        sum((hon[c].astype(str) != "NO_EVIDENCE").sum() for c in hon_present_ev)
    )
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def _cleanup_temp_scripts(paths: LamhuyPaths) -> List[str]:
    """Remove the scratch temp SCRIPT only. KEEP submission CSV, child log, cache."""
    removed: List[str] = []
    script_path = Path(paths.output_dir) / "_lamhuy_repro.py"
    try:
        if script_path.is_file():
            script_path.unlink()
            removed.append(str(script_path))
    except Exception:
        pass
    return removed


def main() -> int:
    """Run the full lamhuy reproduction + validation + comparison; write JSON summary.

    Returns a process exit code (0 on a produced-and-validated CSV, 1 otherwise). The
    summary JSON is ALWAYS written (recording the honest outcome), even on a child
    failure.
    """
    poker_root = Path(__file__).resolve().parents[1]
    paths = default_lamhuy_paths(poker_root)

    summary: Dict[str, object] = {
        "recipe_id": LAMHUY_RECIPE_ID,
        "reported_lb": LAMHUY_REPORTED_LB,
        "notebook": str(paths.notebook),
        "data_dir": str(paths.data_dir),
        "scratch_parent": str(paths.scratch_parent),
        "submission_csv": str(paths.submission_csv),
        "prepared_dir": str(paths.prepared_dir),
    }

    missing = paths.missing()
    if missing:
        summary["status"] = "BLOCKED"
        summary["failure_stage"] = "data_path"
        summary["failure_detail"] = "missing inputs: " + ", ".join(str(p) for p in missing)
        _write_summary(paths, summary)
        print("BLOCKED: missing inputs:", *missing, sep="\n  ")
        return 1

    print(f"lamhuy_recipe: running notebook reproduction (child cap {_CHILD_TIMEOUT_S}s)...")
    print(f"  notebook = {paths.notebook}")
    print(f"  data_parent -> {paths.data_parent}")
    print(f"  scratch_parent -> {paths.scratch_parent}")
    child = run_child(paths, timeout_s=_CHILD_TIMEOUT_S)
    summary["child"] = {
        "returncode": child["returncode"],
        "timed_out": child["timed_out"],
        "duration_s": round(float(child["duration_s"]), 1),
        "log_path": child["log_path"],
    }
    summary["notebook_oof_projected_composite"] = child["metrics"]

    child_ok = (not child["timed_out"]) and (child["returncode"] == 0)
    csv_exists = Path(paths.submission_csv).is_file()

    if not csv_exists:
        summary["status"] = "BLOCKED"
        summary["failure_stage"] = "compute" if child["timed_out"] else "runtime"
        summary["failure_detail"] = (
            f"child timed_out={child['timed_out']} returncode={child['returncode']}; "
            f"no submission.csv produced. log tail:\n{child['log_tail']}"
        )
        _write_summary(paths, summary)
        print("\n=== BLOCKED: no submission.csv produced ===")
        print(f"child returncode={child['returncode']} timed_out={child['timed_out']}")
        print("--- child log tail ---")
        print(child["log_tail"])
        return 1

    # CSV exists — validate + compare even if the child returned nonzero (e.g. a
    # post-submission assertion), recording the child status honestly.
    validation = validate_submission_csv(paths.submission_csv, paths.data_dir / "evaluation_pairs.csv")
    comparison = compare_to_honghanh(paths.submission_csv, paths.honghanh_submission)
    summary["validation"] = validation
    summary["honghanh_comparison"] = comparison
    summary["eval_csv_path"] = str(paths.submission_csv)

    if child_ok and validation.get("all_passed"):
        summary["status"] = "SUCCEEDED"
    else:
        summary["status"] = "PRODUCED_WITH_WARNINGS"
        summary["note"] = (
            f"submission.csv produced (child_ok={child_ok}, "
            f"all_checks_passed={validation.get('all_passed')})"
        )

    removed = _cleanup_temp_scripts(paths)
    summary["cleaned_temp_scripts"] = removed

    _write_summary(paths, summary)
    _print_report(paths, child, validation, comparison)
    return 0 if (child_ok and validation.get("all_passed")) else 1


def _write_summary(paths: LamhuyPaths, summary: Dict[str, object]) -> None:
    Path(paths.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(paths.output_dir) / "_repro_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nlamhuy_recipe: wrote summary -> {out_path}")


def _print_report(
    paths: LamhuyPaths,
    child: Dict[str, object],
    validation: Dict[str, object],
    comparison: Dict[str, object],
) -> None:
    m = child["metrics"]
    print("\n" + "=" * 64)
    print("NOTEBOOK OOF PROJECTED COMPOSITE (from its own cell 33):")
    print(f"  OOF PU Stress Pair AP:       {m.get('oof_pu_stress_pair_ap')}")
    print(f"  OOF Evidence MAP@5:          {m.get('oof_evidence_map5')}")
    print(f"  OOF PU Stress Behavior MAP:  {m.get('oof_pu_stress_behavior_map')}")
    print(f"  PROJECTED COMPOSITE SCORE:   {m.get('projected_composite')}")
    print("-" * 64)
    print("VALIDATION CHECKS:")
    for k, v in validation.items():
        print(f"  {k}: {v}")
    print("-" * 64)
    print("HONGHANH COMPARISON:")
    for k, v in comparison.items():
        print(f"  {k}: {v}")
    print("-" * 64)
    print(f"EVAL CSV PATH: {paths.submission_csv}")
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
