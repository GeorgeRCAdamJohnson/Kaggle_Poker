"""Timing probe (NOT a pytest): run the vectorized fast pipeline on the REAL data/poker.

Runs :func:`poker_collusion.pipeline_fast.run_baseline_pipeline_fast` against the real
competition inputs, times it end-to-end, summarizes the produced ``submission.csv``, and writes
a plain-ASCII report. Prints stage progress so a slow stage is visible rather than hanging.

Usage (from workspace root, PYTHONPATH = workspace root):
    python -m poker_collusion.timing_probe_fast
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from poker_collusion.config import PipelineConfig
from poker_collusion.pipeline_fast import run_baseline_pipeline_fast

_REPORT_PATH = Path("outputs/poker_collusion/timing_probe_fast_report.txt")


def _summarize(submission_path: Path) -> dict:
    frame = pd.read_csv(submission_path)
    risk = pd.to_numeric(frame["risk_score"], errors="coerce").fillna(0.0)
    behavior = frame["predicted_behavior"].astype(str)
    evidence_cols = [f"evidence_hand_{i}" for i in range(1, 6)]
    has_real_evidence = (
        frame[evidence_cols].astype(str).ne("NO_EVIDENCE").any(axis=1)
    )
    return {
        "rows": int(len(frame)),
        "nonzero_risk": int((risk > 0.0).sum()),
        "non_none_behavior": int(behavior.ne("none").sum()),
        "rows_with_real_evidence": int(has_real_evidence.sum()),
        "risk_max": float(risk.max()) if len(frame) else 0.0,
        "risk_mean": float(risk.mean()) if len(frame) else 0.0,
    }


def main() -> int:
    print("[probe] starting fast full-submission run on real data/poker ...", flush=True)
    cfg = PipelineConfig()  # auto-detects data/poker + outputs/poker_collusion
    print(f"[probe] input_dir  = {cfg.input_dir}", flush=True)
    print(f"[probe] output_dir = {cfg.output_dir}", flush=True)

    t0 = time.perf_counter()
    submission_path = run_baseline_pipeline_fast(cfg)
    elapsed = time.perf_counter() - t0
    print(f"[probe] run complete in {elapsed:.1f}s -> {submission_path}", flush=True)

    summary = _summarize(submission_path)
    print(f"[probe] summary: {summary}", flush=True)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "poker_collusion vectorized full-submission timing probe",
        "=======================================================",
        f"submission_path        : {submission_path}",
        f"wall_clock_seconds     : {elapsed:.2f}",
        f"wall_clock_minutes     : {elapsed / 60.0:.2f}",
        "",
        "submission summary",
        "------------------",
        f"rows                   : {summary['rows']}",
        f"nonzero_risk           : {summary['nonzero_risk']}",
        f"non_none_behavior      : {summary['non_none_behavior']}",
        f"rows_with_real_evidence: {summary['rows_with_real_evidence']}",
        f"risk_max               : {summary['risk_max']:.6f}",
        f"risk_mean              : {summary['risk_mean']:.6f}",
        "",
    ]
    _REPORT_PATH.write_text("\n".join(lines), encoding="ascii")
    print(f"[probe] report written to {_REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
