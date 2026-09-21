"""Build OUR OWN evidence ranker: rank each positive pair's shared hands by planted-likelihood, top-5.

Our lineage emits EvidenceMAP@5 = 0.00 on dev (NO_EVIDENCE). The §127 fingerprint scores per-hand
planted-likelihood at AUC 0.92 but we NEVER wired it into an evidence head. The V30 dev hand-feature
cache (prepared_v13/dev_hand_features.parquet, 2.9M rows, all dev pairs' shared hands) already carries
the planted discriminators (transfer, contrib, opposite-flow, aggression, pot). We:

  1. label each dev shared hand planted(1)/not(0) via development_evidence.csv,
  2. train a per-hand planted ranker (pair-grouped OOF so no pair leaks into its own ranking),
  3. for each positive pair, rank its DEV shared hands by OOF score, emit top-5 hand_ids,
  4. score EvidenceMAP@5 via CanonicalScorer against the planted-evidence solution.

Target: beat 0.00, approach honghanh 0.35-0.46. PairAP/Behavior held (risk untouched; behavior='none'
here to isolate the evidence delta).

Run: python -m anchor_repro.v30_evidence_head
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from anchor_repro.gpu_config import xgb_params
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.models import ScoringRecipe
from poker_collusion.config import PipelineConfig

DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior",
            "evidence_hand_1", "evidence_hand_2", "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"]
SEED = 42

# per-hand planted discriminators available in dev_hand_features (from §127 separation)
HAND_FEATS = ["pot_bb", "pair_contribution_bb", "pair_pot_share", "contrib_gap_bb", "net_gap_bb",
              "max_win_bb", "max_loss_bb", "transfer_any_bb", "flow_oneway", "dump_vs_hole",
              "strong_hole_passive", "transfer_pot_ratio", "passivity_density", "hole_strength_gap",
              "both_showdown", "one_folded", "max_hole_strength", "pair_aggression", "pair_raises",
              "pair_calls", "pair_checks", "pair_postflop_checks", "max_amount_bb"]


def main() -> int:
    cfg = PipelineConfig()
    labels = pd.read_csv(Path(cfg.input_dir) / "development_labels.csv")
    evidence = pd.read_csv(Path(cfg.input_dir) / "development_evidence.csv")
    recipe = ScoringRecipe(recipe_id="v30_evidence_head_v1")
    scorer = CanonicalScorer(recipe, cfg, labels=labels, evidence=evidence)

    pos_ids = set(labels[labels["label"] == 1]["pair_id"].astype(str))
    planted = set(zip(evidence["pair_id"].astype(str), evidence["hand_id"].astype(str)))
    print(f"positive pairs: {len(pos_ids)}   planted (pair,hand): {len(planted)}")

    hf = pd.read_parquet(DEV_HF, columns=["pair_id", "hand_id"] + HAND_FEATS)
    hf["pair_id"] = hf["pair_id"].astype(str)
    hf["hand_id"] = hf["hand_id"].astype(str)
    # restrict to POSITIVE pairs' shared hands (evidence is scored over true positives only)
    hf = hf[hf["pair_id"].isin(pos_ids)].copy()
    hf["planted"] = [1 if (p, h) in planted else 0 for p, h in zip(hf["pair_id"], hf["hand_id"])]
    print(f"positive-pair shared hands: {len(hf)}   planted among them: {hf['planted'].sum()}")

    X = hf[HAND_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = hf["planted"].to_numpy().astype(np.int8)
    groups = hf["pair_id"].to_numpy()

    # per-hand planted ranker, pair-grouped OOF (a pair never leaks into its own ranking)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.05,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=4.0)
    oof = np.zeros(len(y), np.float32)
    gkf = GroupKFold(5)
    for tr, va in gkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    print(f"per-hand planted ranker (pair-grouped OOF): AUC={roc_auc_score(y, oof):.4f} "
          f"AP={average_precision_score(y, oof):.4f}")
    hf["score"] = oof

    # emit top-5 hands per positive pair by OOF planted-score
    top5 = (hf.sort_values(["pair_id", "score"], ascending=[True, False])
              .groupby("pair_id").head(5).groupby("pair_id")["hand_id"].apply(list))

    # build dev prediction frame over ALL confirmed pairs: positives get top-5, negatives NO_EVIDENCE
    oof_v30 = pd.read_parquet(OOF).drop_duplicates("pair_id")
    oof_v30["pair_id"] = oof_v30["pair_id"].astype(str)
    conf_ids = set(labels["pair_id"].astype(str))
    dev = oof_v30[oof_v30["pair_id"].isin(conf_ids)].copy()
    r = dev["rank_sum_w50"].astype(float)
    dev["risk_score"] = ((r - r.min()) / (r.max() - r.min() + 1e-12)).clip(0, 1)
    dev["predicted_behavior"] = "none"
    for i in range(1, 6):
        dev[f"evidence_hand_{i}"] = "NO_EVIDENCE"
    for pid, hands in top5.items():
        row = dev["pair_id"] == pid
        for i, h in enumerate(hands[:5], start=1):
            dev.loc[row, f"evidence_hand_{i}"] = h

    f = dev.reindex(columns=SUB_COLS)
    s = scorer.score_dev_predictions(f)
    c = s.components
    print(f"\n=== EVIDENCE HEAD scored on dev ===")
    print(f"  PairAP={c.pair_ap:.4f}  EvidMAP@5={c.evidence_map:.4f}  BehavMAP={c.behavior_map:.4f}  combined={c.combined:.4f}")
    print(f"  (our old evidence 0.0000; honghanh 0.3455-0.46)")

    Path("outputs/poker_collusion/component_decomposition").mkdir(parents=True, exist_ok=True)
    (Path("outputs/poker_collusion/component_decomposition") / "_v30_evidence_head.json").write_text(
        json.dumps({"per_hand_auc": float(roc_auc_score(y, oof)),
                    "per_hand_ap": float(average_precision_score(y, oof)),
                    "evidence_map": c.evidence_map, "pair_ap": c.pair_ap}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
