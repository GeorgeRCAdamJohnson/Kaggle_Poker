"""Real candidate: lamhuy's ACTUAL persisted risk blended with field-contrast risk.

BUILD + VALIDATE ONLY. Does NOT submit to Kaggle and does NOT overwrite
``repro_lamhuy/submission.csv`` (Rule 7). Only ``risk_score`` changes vs
lamhuy's real submission; ``predicted_behavior`` and ``evidence_hand_1..5`` are
copied byte-identical, matching the exp3c/exp4c precedent (replace ONLY the
risk column, never touch behavior/evidence when they were not re-validated).

Honest scope note (Rule 9): the supporting evidence for this blend is the
confirmed-label lift measured against an APPROXIMATE base
(``lamhuy_field_contrast_blend.py``, +0.0191 at w=0.5), NOT against lamhuy's
real dev-holdout risk (no such cache exists for lamhuy in this repo, unlike
honghanh's persisted ``dev_holdout_oof_064469.parquet``). A conservative
weight is used here specifically because the blend partner is now lamhuy's
REAL risk, not the approximation the lift was measured against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from xgboost import XGBClassifier

from anchor_repro.lamhuy_field_contrast import FEATURE_COLUMNS

_SUBMISSION_COLUMNS = [
    "pair_id",
    "risk_score",
    "predicted_behavior",
    "evidence_hand_1",
    "evidence_hand_2",
    "evidence_hand_3",
    "evidence_hand_4",
    "evidence_hand_5",
]


def _fit_field_risk(root: Path) -> pd.DataFrame:
    prepared = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    report_dir = root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet")
    dev_field = pd.read_parquet(report_dir / "dev_field_contrast.parquet")
    eval_field = pd.read_parquet(report_dir / "eval_field_contrast.parquet")

    labeled = dev_pairs[dev_pairs["is_labeled"].astype(bool)].merge(
        dev_field, on="pair_id", how="inner", validate="one_to_one"
    )
    y = labeled["label"].astype(int).to_numpy()
    X = labeled[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=5051,
        n_jobs=-1,
    )
    model.fit(X, y)
    eval_X = eval_field[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({"pair_id": eval_field["pair_id"], "field_risk": model.predict_proba(eval_X)[:, 1]})


def validate_submission(frame: pd.DataFrame, eval_pairs_csv: Path) -> dict:
    checks: dict[str, object] = {}
    checks["row_count"] = int(len(frame))
    checks["schema_exact"] = bool(list(frame.columns) == _SUBMISSION_COLUMNS)
    eval_ids = set(pd.read_csv(eval_pairs_csv, usecols=["pair_id"])["pair_id"])
    frame_ids = set(frame["pair_id"])
    checks["missing_pairs"] = len(eval_ids - frame_ids)
    checks["extra_pairs"] = len(frame_ids - eval_ids)
    checks["risk_in_0_1"] = bool(frame["risk_score"].between(0.0, 1.0).all())
    checks["no_nulls"] = int(frame.isna().sum().sum()) == 0
    evidence_cols = [c for c in _SUBMISSION_COLUMNS if c.startswith("evidence_hand_")]
    dup = frame[evidence_cols].apply(
        lambda row: len([v for v in row if v != "NO_EVIDENCE"]) != len(set(v for v in row if v != "NO_EVIDENCE")),
        axis=1,
    )
    checks["n_rows_duplicate_evidence"] = int(dup.sum())
    checks["all_passed"] = bool(
        checks["schema_exact"]
        and checks["missing_pairs"] == 0
        and checks["extra_pairs"] == 0
        and checks["risk_in_0_1"]
        and checks["no_nulls"]
        and checks["n_rows_duplicate_evidence"] == 0
        and checks["row_count"] == len(eval_ids)
    )
    return checks


def build_candidate(root: Path, weight_field: float) -> dict:
    lamhuy_path = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "submission.csv"
    lamhuy = pd.read_csv(lamhuy_path)
    field = _fit_field_risk(root)

    merged = lamhuy.merge(field, on="pair_id", how="left", validate="one_to_one")
    if merged["field_risk"].isna().any():
        raise ValueError("field_risk missing for some eval pairs; refusing to fabricate a score")

    lamhuy_pct = rankdata(merged["risk_score"], method="average") / len(merged)
    field_pct = rankdata(merged["field_risk"], method="average") / len(merged)
    blended_pct = (1.0 - weight_field) * lamhuy_pct + weight_field * field_pct

    candidate = merged[_SUBMISSION_COLUMNS].copy()
    candidate["risk_score"] = blended_pct.clip(0.0, 1.0)

    report_dir = root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"candidate_lamhuy_field_blend_w{int(round(weight_field * 100)):03d}.csv"
    candidate.to_csv(out_path, index=False)

    eval_pairs_csv = root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13" / "eval_pairs.parquet"
    eval_ids_frame = pd.read_parquet(eval_pairs_csv, columns=["pair_id"])
    tmp_csv = report_dir / "_eval_pairs_ids.csv"
    eval_ids_frame.to_csv(tmp_csv, index=False)
    validation = validate_submission(candidate, tmp_csv)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    top500_overlap = len(
        set(candidate.nlargest(500, "risk_score")["pair_id"]) & set(lamhuy.nlargest(500, "risk_score")["pair_id"])
    )

    return {
        "note": (
            "BUILD + VALIDATE ONLY. Not submitted to Kaggle; does not overwrite lamhuy's real "
            "submission.csv (Rule 7). Supporting evidence is the confirmed-label lift measured "
            "against an APPROXIMATE base (+0.0191 at w=0.5, lamhuy_field_contrast_blend.py), NOT "
            "against lamhuy's real dev-holdout risk (no such cache exists here). This is a "
            "screened candidate, not a validated leaderboard prediction."
        ),
        "weight_field": weight_field,
        "candidate_path": str(out_path),
        "sha256": sha256,
        "top500_overlap_with_lamhuy": top500_overlap,
        "top500_overlap_fraction": top500_overlap / 500,
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weight-field", type=float, default=0.25)
    args = parser.parse_args()
    report = build_candidate(args.poker_root, args.weight_field)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
