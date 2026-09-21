"""Decompose OUR OWN V30 lineage into PairAP / EvidenceMAP@5 / BehaviorMAP on the dev holdout.

The context audit found: we have NEVER measured our own three components — only PairAP. Our shipped
evidence/behavior columns are copied verbatim from the honghanh reproduction. The metric is
0.70*PairAP + 0.20*Evidence + 0.10*Behavior, so 30% of the score has been treated as fixed/borrowed.

This scores our V30 dev-OOF risk (rank_sum_w50) with WHATEVER behavior+evidence heads we actually
ship, using the CanonicalScorer (same official metric, PU-correct dev solution). Compares to
honghanh's measured dev components (0.9747 / 0.3455 / 0.8935). Locates exactly where our 0.709 bleeds.

Run: python -m anchor_repro.v30_component_decomp
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anchor_repro.scorer import CanonicalScorer
from anchor_repro.models import ScoringRecipe
from poker_collusion.config import PipelineConfig

OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
SHIPPED = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior",
            "evidence_hand_1", "evidence_hand_2", "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"]


def _scorer():
    cfg = PipelineConfig()
    labels = pd.read_csv(Path(cfg.input_dir) / "development_labels.csv")
    ev_path = Path(cfg.input_dir) / "development_evidence.csv"
    evidence = pd.read_csv(ev_path) if ev_path.is_file() else None
    recipe = ScoringRecipe(recipe_id="v30_component_decomp_v1")
    return CanonicalScorer(recipe, cfg, labels=labels, evidence=evidence), labels, evidence


def main() -> int:
    scorer, labels, evidence = _scorer()
    conf_ids = set(labels["pair_id"].astype(str))
    print(f"confirmed dev pairs: {len(conf_ids)}")

    oof = pd.read_parquet(OOF)
    oof["pair_id"] = oof["pair_id"].astype(str)
    print(f"oof_rows columns: {list(oof.columns)}")
    print(f"oof_rows rows: {len(oof)}  unique pairs: {oof['pair_id'].nunique()}")

    # V30 dev risk = rank_sum_w50 (the 0.70904 risk), normalized to [0,1]
    risk = oof.drop_duplicates("pair_id").set_index("pair_id")["rank_sum_w50"].astype(float)
    r = (risk - risk.min()) / (risk.max() - risk.min() + 1e-12)

    # dev behavior/evidence: does the OOF carry heads? else use shipped (eval) heads is not valid for
    # dev pairs (disjoint). We need DEV behavior+evidence. Check what's available in oof_rows.
    have_beh = "predicted_behavior" in oof.columns
    have_ev = all(f"evidence_hand_{i}" in oof.columns for i in range(1, 6))
    print(f"oof has dev behavior head: {have_beh}   dev evidence head: {have_ev}")

    dev = pd.DataFrame({"pair_id": r.index.astype(str), "risk_score": r.values.clip(0, 1)})
    dev = dev[dev["pair_id"].isin(conf_ids)].copy()

    # ---- three scoring variants to isolate each component's contribution ----
    def score(frame, tag):
        f = frame.reindex(columns=SUB_COLS)
        try:
            s = scorer.score_dev_predictions(f)
            c = s.components
            print(f"\n[{tag}] PairAP={c.pair_ap:.4f}  EvidMAP@5={c.evidence_map:.4f}  "
                  f"BehavMAP={c.behavior_map:.4f}  combined={c.combined:.4f}")
            return {"tag": tag, "pair_ap": c.pair_ap, "evidence_map": c.evidence_map,
                    "behavior_map": c.behavior_map, "combined": c.combined}
        except Exception as e:
            print(f"\n[{tag}] SCORE FAILED: {e!r}")
            return {"tag": tag, "error": repr(e)}

    results = []

    # Variant A: risk only, behavior=none, NO_EVIDENCE  (pure PairAP; ev/beh floored)
    a = dev.copy()
    a["predicted_behavior"] = "none"
    for i in range(1, 6):
        a[f"evidence_hand_{i}"] = "NO_EVIDENCE"
    results.append(score(a, "A_risk_only_floored"))

    # Variant B: risk + DEV behavior/evidence heads if the OOF carries them
    if have_beh or have_ev:
        b = dev.merge(oof.drop_duplicates("pair_id")[["pair_id"] +
                      (["predicted_behavior"] if have_beh else []) +
                      ([f"evidence_hand_{i}" for i in range(1, 6)] if have_ev else [])],
                      on="pair_id", how="left")
        if not have_beh:
            b["predicted_behavior"] = "none"
        else:
            b["predicted_behavior"] = b["predicted_behavior"].fillna("none")
        for i in range(1, 6):
            col = f"evidence_hand_{i}"
            if col not in b.columns:
                b[col] = "NO_EVIDENCE"
            else:
                b[col] = b[col].fillna("NO_EVIDENCE")
        results.append(score(b, "B_v30_dev_heads"))
    else:
        print("\n[B] oof_rows carries NO dev behavior/evidence head -> our lineage ships PairAP-only "
              "on dev; behavior/evidence come ENTIRELY from the borrowed honghanh heads (eval only).")

    print("\n=== comparison to honghanh measured dev components ===")
    print("  honghanh_064469:  PairAP=0.9747  EvidMAP@5=0.3455  BehavMAP=0.8935  combined=0.8407")
    print("  (honghanh real LB = 0.64262; ours real LB = 0.70904)")
    Path("outputs/poker_collusion/component_decomposition").mkdir(parents=True, exist_ok=True)
    (Path("outputs/poker_collusion/component_decomposition") / "_v30_own_components.json").write_text(
        json.dumps({"results": results,
                    "honghanh_ref": {"pair_ap": 0.9747, "evidence_map": 0.3455, "behavior_map": 0.8935}},
                   indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
