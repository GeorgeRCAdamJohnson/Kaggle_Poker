"""TUNE the behavior + evidence heads on the dev scorer before spending any submission.

Lesson (user): V30's shipped evidence/behavior are the MAX-TUNED honghanh heads. 0.67666 dropped
because we swapped them for untuned first-pass heads. So we must TUNE past the borrowed heads locally
(dev CanonicalScorer) and only submit if we beat them.

This sweeps:
  BEHAVIOR: routing rate x a confidence-margin gate (route to family only when the top specialist
            beats the 2nd by margin m; otherwise 'none') -> maximize dev BehaviorMAP.
  EVIDENCE: our per-hand planted ranker BLENDED with a heuristic strength (like honghanh w=0.75),
            sweep blend weight -> maximize dev EvidenceMAP@5, target beating 0.35-0.46.

All scored via CanonicalScorer on the 1860 dev confirmed pairs. Reports the best config for each head.
Run: python -m anchor_repro.tune_heads
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from anchor_repro.gpu_config import xgb_params
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.models import ScoringRecipe
from poker_collusion.config import PipelineConfig

OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior",
            "evidence_hand_1", "evidence_hand_2", "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"]
FAMILY = {0: "directed_transfer", 1: "soft_play", 2: "coordinated_isolation"}
HAND_FEATS = ["pot_bb", "pair_contribution_bb", "pair_pot_share", "contrib_gap_bb", "net_gap_bb",
              "max_win_bb", "max_loss_bb", "transfer_any_bb", "flow_oneway", "dump_vs_hole",
              "strong_hole_passive", "transfer_pot_ratio", "passivity_density", "hole_strength_gap",
              "both_showdown", "one_folded", "max_hole_strength", "pair_aggression", "pair_raises",
              "pair_calls", "pair_checks", "pair_postflop_checks", "max_amount_bb"]


def main() -> int:
    cfg = PipelineConfig()
    labels = pd.read_csv(Path(cfg.input_dir) / "development_labels.csv")
    evidence = pd.read_csv(Path(cfg.input_dir) / "development_evidence.csv")
    scorer = CanonicalScorer(ScoringRecipe(recipe_id="tune_heads_v1"), cfg, labels=labels, evidence=evidence)
    conf_ids = set(labels["pair_id"].astype(str))

    oof = pd.read_parquet(OOF).drop_duplicates("pair_id")
    oof["pair_id"] = oof["pair_id"].astype(str)
    dev = oof[oof["pair_id"].isin(conf_ids)].copy()
    r = dev["rank_sum_w50"].astype(float)
    dev["risk_score"] = ((r - r.min()) / (r.max() - r.min() + 1e-12)).clip(0, 1)
    S = dev[["specialist_directed", "specialist_soft", "specialist_isolation"]].to_numpy()
    Ssort = np.sort(S, axis=1)
    dev["margin"] = Ssort[:, -1] - Ssort[:, -2]          # confidence of the argmax family
    dev["fam"] = [FAMILY[i] for i in S.argmax(axis=1)]

    # ---------- BEHAVIOR tuning: rate x margin gate ----------
    def score_behavior(rate, margin_q):
        d = dev.copy()
        thr = d["risk_score"].quantile(1 - rate)
        mthr = d["margin"].quantile(margin_q) if margin_q > 0 else -1
        route = (d["risk_score"] >= thr) & (d["margin"] >= mthr)
        d["predicted_behavior"] = np.where(route, d["fam"], "none")
        for i in range(1, 6):
            d[f"evidence_hand_{i}"] = "NO_EVIDENCE"
        c = scorer.score_dev_predictions(d.reindex(columns=SUB_COLS)).components
        return c.behavior_map, c.pair_ap

    print("=== BEHAVIOR: rate x margin-gate -> dev BehaviorMAP (borrowed=0.08, honghanh=0.89) ===")
    best_b = (-1, None)
    for rate in [0.05, 0.10, 0.20, 0.35, 0.50, 1.0]:
        for mq in [0.0, 0.25, 0.50]:
            bm, pa = score_behavior(rate, mq)
            if bm > best_b[0]:
                best_b = (bm, (rate, mq))
            print(f"  rate={rate:<5} margin_q={mq:<5} BehaviorMAP={bm:.4f}")
    print(f"  BEST behavior: {best_b[0]:.4f} at rate,margin={best_b[1]}")

    # ---------- EVIDENCE tuning: ranker blended with heuristic strength ----------
    pos_ids = set(labels[labels["label"] == 1]["pair_id"].astype(str))
    planted = set(zip(evidence["pair_id"].astype(str), evidence["hand_id"].astype(str)))
    hf = pd.read_parquet(DEV_HF, columns=["pair_id", "hand_id"] + HAND_FEATS)
    hf["pair_id"] = hf["pair_id"].astype(str); hf["hand_id"] = hf["hand_id"].astype(str)
    hf = hf[hf["pair_id"].isin(pos_ids)].copy()
    hf["planted"] = [1 if (p, h) in planted else 0 for p, h in zip(hf["pair_id"], hf["hand_id"])]
    X = hf[HAND_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = hf["planted"].to_numpy().astype(np.int8)
    groups = hf["pair_id"].to_numpy()
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.05,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=4.0)
    oof_h = np.zeros(len(y), np.float32)
    for tr, va in GroupKFold(5).split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
        oof_h[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    hf["ml"] = oof_h
    # heuristic strength = the strongest single planted discriminator conjunction (transfer x pot x agg)
    def pr(a): return pd.Series(a).rank(pct=True).to_numpy()
    hf["heur"] = pr(hf["transfer_any_bb"]) * 0.4 + pr(hf["pair_contribution_bb"]) * 0.3 + pr(hf["pair_aggression"]) * 0.3

    r_all = oof.rename(columns={})  # dev risk frame
    devb = oof[oof["pair_id"].isin(conf_ids)].copy()
    rr = devb["rank_sum_w50"].astype(float)
    devb["risk_score"] = ((rr - rr.min()) / (rr.max() - rr.min() + 1e-12)).clip(0, 1)
    devb["predicted_behavior"] = "none"

    def score_evidence(w):
        hf["blend"] = w * pr(hf["ml"]) + (1 - w) * hf["heur"]
        top5 = (hf.sort_values(["pair_id", "blend"], ascending=[True, False])
                  .groupby("pair_id").head(5).groupby("pair_id")["hand_id"].apply(list)).to_dict()
        d = devb.copy()
        for i in range(1, 6):
            d[f"evidence_hand_{i}"] = "NO_EVIDENCE"
        for pid, hands in top5.items():
            row = d["pair_id"] == pid
            for i, h in enumerate(hands[:5], start=1):
                d.loc[row, f"evidence_hand_{i}"] = h
        c = scorer.score_dev_predictions(d.reindex(columns=SUB_COLS)).components
        return c.evidence_map

    print(f"\n=== EVIDENCE: ML-ranker x heuristic blend -> dev EvidenceMAP@5 (ours untuned=0.28, honghanh=0.35-0.46) ===")
    print(f"  per-hand ranker OOF AUC={roc_auc_score(y, oof_h):.4f}")
    best_e = (-1, None)
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        em = score_evidence(w)
        if em > best_e[0]:
            best_e = (em, w)
        print(f"  ml_weight={w:<5} EvidenceMAP@5={em:.4f}")
    print(f"  BEST evidence: {best_e[0]:.4f} at ml_weight={best_e[1]}")

    Path("outputs/poker_collusion/component_decomposition").mkdir(parents=True, exist_ok=True)
    (Path("outputs/poker_collusion/component_decomposition") / "_tune_heads.json").write_text(json.dumps(
        {"best_behavior_map": best_b[0], "best_behavior_cfg": best_b[1],
         "best_evidence_map": best_e[0], "best_evidence_ml_weight": best_e[1]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
