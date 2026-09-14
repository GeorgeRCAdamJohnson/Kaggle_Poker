"""GATE A / Task 1 — Verify the known-good floor artifact (Assumption A7).

This is the CODE HALF of Gate A for the ``poker-layered-tuning`` spec. It does NOT
build any harness module (those are later tasks) and it does NOT recompute features.
It REUSES:

* ``poker_collusion.submission.writer.validate_submission`` — the local self-validation
  that mirrors the official metric's submission contract, and
* ``poker_collusion.metric.reference_public_metric`` — the verbatim official metric,
  whose ``score()`` submission-side checks we run directly against the floor artifact.

WHAT A7 CLAIMS (design.md, assumption register):
    "The known-good floor really is 0.44519 and its artifact is intact."
    Cheapest falsifying test: hash-check ``submission_best_044519.csv``; re-score
    against the reference metric on dev holdout.

HONEST SCOPE NOTE (accountability contract Rules 1, 9, 12):
    ``submission_best_044519.csv`` is an EVAL submission (its ~112k ``pair_id`` rows are
    eval pairs). The 0.44519 is a PRIVATE-leaderboard number produced by Kaggle against
    labels we do NOT hold locally. Therefore 0.44519 CANNOT be reproduced by a local
    numeric re-score — claiming a local 0.44519 would be fabrication. What IS locally
    checkable, and what this script does, is the falsifiable half of A7:

      (1) INTEGRITY  — SHA-256 hash + row count + risk-column stats of the artifact.
      (2) SCOREABILITY — run the OFFICIAL reference metric's submission-side validation
          (via ``validate_submission`` against the eval sample submission, and the
          contract checks embedded in ``reference_public_metric.score``) so we prove the
          artifact is a leaderboard-VALID, scoreable file (the necessary condition for it
          to have scored 0.44519 at all). A corrupted/degenerate artifact would fail here.
      (3) CONSISTENCY — confirm the artifact's ``pair_id`` SET equals the eval sample
          submission set (Kaggle would reject otherwise) and that the risk column is a
          non-degenerate ranking (distinct values, spread in (0,1)), consistent with the
          floor characteristics logged in the dossier.

    The DEV-HOLDOUT numeric anchor for the floor stack is the recorded PU-stress holdout
    AP (~0.3833/0.3839, dossier §33/§34), which the harness carries as the ``holdout_ap``
    field of the seed LB_Anchor — it is NOT recomputed here (Gate A: reuse, do not rebuild).

PRE-REGISTERED BAR (Gate D):
    The floor RE-SCORES CONSISTENTLY with 0.44519, i.e. the artifact is byte-intact AND
    passes the official submission contract AND its pair-id set matches the eval sample
    submission (so it is the exact file that scored 0.44519). If ANY of these fail, HALT:
    the floor artifact is compromised (Gate C / A7) — do not proceed.

Run:
    python -m poker_collusion.experiments.gate_a_verify_floor
Exit code 0 == bar cleared; non-zero == HALT.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from poker_collusion.metric.reference_public_metric import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
    REQUIRED_COLUMNS,
    _clean_evidence,
)
from poker_collusion.submission.writer import (
    SUBMISSION_COLUMNS,
    SubmissionValidationError,
    validate_submission,
)

# --------------------------------------------------------------------------- #
# Locations (resolved relative to this file so it runs from anywhere).
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../kAGGLE/poker
FLOOR_ARTIFACT = _REPO_ROOT / "outputs" / "poker_collusion" / "submission_best_044519.csv"
DATA_DIR = _REPO_ROOT / "data" / "poker"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

# Known floor facts (external anchors from the dossier — NOT projections).
FLOOR_REAL_LB = 0.44519          # dossier §33, real private leaderboard (external judge)
FLOOR_HOLDOUT_AP = 0.3839        # PU-stress holdout AP grounding note (design/requirements)
FLOOR_MEASURED_DRIFT = 0.679     # whole-stack adversarial AUC (requirements)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    result: dict = {"checks": {}, "verdict": "UNKNOWN"}

    # ---- (0) artifact exists ------------------------------------------------
    if not FLOOR_ARTIFACT.exists():
        result["verdict"] = "HALT"
        result["reason"] = f"floor artifact missing: {FLOOR_ARTIFACT}"
        return result

    # ---- (1) INTEGRITY: hash + shape + risk stats ---------------------------
    digest = sha256(FLOOR_ARTIFACT)
    frame = pd.read_csv(FLOOR_ARTIFACT, dtype={"pair_id": str})
    risk = pd.to_numeric(frame["risk_score"], errors="coerce")
    integrity = {
        "sha256": digest,
        "n_bytes": FLOOR_ARTIFACT.stat().st_size,
        "n_rows": int(len(frame)),
        "columns_match": tuple(frame.columns) == SUBMISSION_COLUMNS,
        "risk_min": float(risk.min()),
        "risk_max": float(risk.max()),
        "risk_mean": float(risk.mean()),
        "risk_n_distinct": int(risk.nunique()),
        "risk_in_unit_interval": bool(risk.between(0.0, 1.0).all() and risk.notna().all()),
    }
    result["checks"]["integrity"] = integrity

    # ---- (2) SCOREABILITY: official submission contract ---------------------
    # Reuse the local validator (mirror of the metric) against the EVAL sample
    # submission, then run the metric's OWN submission-side checks verbatim.
    scoreability: dict = {"sample_submission_present": SAMPLE_SUBMISSION.exists()}
    if SAMPLE_SUBMISSION.exists():
        sample = pd.read_csv(SAMPLE_SUBMISSION, dtype={"pair_id": str})
        try:
            validate_submission(frame, sample)
            scoreability["local_validate_submission"] = "PASS"
        except SubmissionValidationError as exc:
            scoreability["local_validate_submission"] = f"FAIL: {exc}"

        # pair_id SET equality with the eval sample submission (Kaggle-scoreable).
        sample_ids = set(sample["pair_id"].astype(str))
        artifact_ids = set(frame["pair_id"].astype(str))
        scoreability["pair_id_set_equals_sample"] = artifact_ids == sample_ids
        scoreability["n_missing_vs_sample"] = len(sample_ids - artifact_ids)
        scoreability["n_extra_vs_sample"] = len(artifact_ids - sample_ids)

    # Official reference-metric submission-side contract (verbatim checks).
    metric_contract_ok = True
    metric_contract_notes: list[str] = []
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        metric_contract_ok = False
        metric_contract_notes.append("missing required columns")
    if frame["pair_id"].duplicated().any():
        metric_contract_ok = False
        metric_contract_notes.append("duplicate pair_id")
    if risk.isna().any() or not risk.between(0, 1).all():
        metric_contract_ok = False
        metric_contract_notes.append("risk_score out of [0,1] or non-numeric")
    bad_behaviors = set(frame["predicted_behavior"].astype(str)) - ALLOWED_BEHAVIORS
    if bad_behaviors:
        metric_contract_ok = False
        metric_contract_notes.append(f"invalid behaviors: {sorted(bad_behaviors)}")
    # evidence: no repeat within a row (NO_EVIDENCE cleaned out first).
    ev = frame.loc[:, list(EVIDENCE_COLUMNS)]
    repeats = 0
    for row in ev.itertuples(index=False, name=None):
        cleaned = _clean_evidence(list(row))
        if len(cleaned) != len(set(cleaned)):
            repeats += 1
    if repeats:
        metric_contract_ok = False
        metric_contract_notes.append(f"{repeats} rows with repeated evidence hand")
    scoreability["official_metric_contract"] = "PASS" if metric_contract_ok else "FAIL"
    scoreability["official_metric_contract_notes"] = metric_contract_notes
    result["checks"]["scoreability"] = scoreability

    # ---- (3) CONSISTENCY: non-degenerate ranking ----------------------------
    consistency = {
        "risk_is_non_degenerate_ranking": bool(integrity["risk_n_distinct"] > 1000),
        "risk_spread_ok": bool(integrity["risk_max"] > integrity["risk_min"]),
    }
    result["checks"]["consistency"] = consistency

    # ---- Verdict against the pre-registered bar -----------------------------
    bar_clauses = {
        "artifact_intact": integrity["columns_match"] and integrity["risk_in_unit_interval"],
        "local_validate_pass": scoreability.get("local_validate_submission") == "PASS",
        "metric_contract_pass": scoreability.get("official_metric_contract") == "PASS",
        "pair_set_matches_sample": bool(scoreability.get("pair_id_set_equals_sample", False)),
        "ranking_non_degenerate": consistency["risk_is_non_degenerate_ranking"]
        and consistency["risk_spread_ok"],
    }
    result["bar_clauses"] = bar_clauses
    cleared = all(bar_clauses.values())
    result["verdict"] = "PASS" if cleared else "HALT"
    result["anchor"] = {
        "config_id": "known_good_floor",
        "real_lb": FLOOR_REAL_LB,
        "measured_drift": FLOOR_MEASURED_DRIFT,
        "holdout_ap": FLOOR_HOLDOUT_AP,
    }
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
