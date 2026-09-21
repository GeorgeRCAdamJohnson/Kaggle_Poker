"""Assemble the full EVAL submission: V30 risk (UNCHANGED 0.70904 ranking) + our OWN behavior + evidence heads.

- risk_score: EXACTLY the shipped V30 rank_sum_w50 eval values (ranking untouched -> PairAP unchanged).
- predicted_behavior: argmax of V30 eval specialists (directed/soft/iso); route top `RATE` fraction
  by risk to that family, rest -> 'none'.
- evidence_hand_1..5: per eval pair, top-5 hands by a per-hand planted ranker trained on ALL dev
  positives' planted flags (applied to eval pairs' shared hands).

Writes outputs/poker_collusion/heads_submission.csv. Prints a diff vs the shipped 0.70904 CSV
(rows changed) so we know exactly what we're changing. This is the PairAP-independent 30% lift.

Run: python -m anchor_repro.assemble_heads_submission
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from anchor_repro.gpu_config import xgb_params

SHIPPED = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
DEV_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
EVAL_HF = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
EVAL_SPEC = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/eval_specialist_scores.parquet")
OUT = Path("outputs/poker_collusion/heads_submission.csv")
FAMILY = {0: "directed_transfer", 1: "soft_play", 2: "coordinated_isolation"}
RATE = 0.12  # fraction of eval pairs (by risk) routed to a behavior family; rest 'none'
SEED = 42

HAND_FEATS = ["pot_bb", "pair_contribution_bb", "pair_pot_share", "contrib_gap_bb", "net_gap_bb",
              "max_win_bb", "max_loss_bb", "transfer_any_bb", "flow_oneway", "dump_vs_hole",
              "strong_hole_passive", "transfer_pot_ratio", "passivity_density", "hole_strength_gap",
              "both_showdown", "one_folded", "max_hole_strength", "pair_aggression", "pair_raises",
              "pair_calls", "pair_checks", "pair_postflop_checks", "max_amount_bb"]


def _train_evidence_ranker():
    import pandas as pd
    from poker_collusion.config import PipelineConfig
    cfg = PipelineConfig()
    labels = pd.read_csv(Path(cfg.input_dir) / "development_labels.csv")
    evidence = pd.read_csv(Path(cfg.input_dir) / "development_evidence.csv")
    pos_ids = set(labels[labels["label"] == 1]["pair_id"].astype(str))
    planted = set(zip(evidence["pair_id"].astype(str), evidence["hand_id"].astype(str)))
    hf = pd.read_parquet(DEV_HF, columns=["pair_id", "hand_id"] + HAND_FEATS)
    hf["pair_id"] = hf["pair_id"].astype(str); hf["hand_id"] = hf["hand_id"].astype(str)
    hf = hf[hf["pair_id"].isin(pos_ids)].copy()
    hf["planted"] = [1 if (p, h) in planted else 0 for p, h in zip(hf["pair_id"], hf["hand_id"])]
    X = hf[HAND_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = hf["planted"].to_numpy().astype(np.int8)
    pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.05,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=4.0)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=300)
    print(f"evidence ranker trained on {len(hf):,} dev-positive hands (planted={int(y.sum())})")
    return model


def main() -> int:
    sub = pd.read_csv(SHIPPED)
    sub["pair_id"] = sub["pair_id"].astype(str)
    print(f"shipped submission: {len(sub)} rows")

    # ---- behavior head: argmax specialist, route top-RATE by risk ----
    spec = pd.read_parquet(EVAL_SPEC)
    spec["pair_id"] = spec["pair_id"].astype(str)
    m = sub.merge(spec, on="pair_id", how="left")
    S = m[["specialist_directed", "specialist_soft", "specialist_isolation"]].fillna(0).to_numpy()
    fam = [FAMILY[i] for i in S.argmax(axis=1)]
    risk = m["risk_score"].astype(float).to_numpy()
    thr = np.quantile(risk, 1 - RATE)
    behavior = np.where(risk >= thr, fam, "none")
    print(f"behavior: routed {(behavior != 'none').sum()} pairs to a family (rate={RATE}); "
          f"dist={pd.Series(behavior).value_counts().to_dict()}")

    # ---- evidence head: top-5 planted-ranked hands per eval pair ----
    model = _train_evidence_ranker()
    eh = pd.read_parquet(EVAL_HF, columns=["pair_id", "hand_id"] + HAND_FEATS)
    eh["pair_id"] = eh["pair_id"].astype(str); eh["hand_id"] = eh["hand_id"].astype(str)
    Xe = eh[HAND_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    eh["score"] = model.predict(xgb.DMatrix(Xe))
    top5 = (eh.sort_values(["pair_id", "score"], ascending=[True, False])
              .groupby("pair_id").head(5).groupby("pair_id")["hand_id"].apply(list))
    print(f"evidence: scored {len(eh):,} eval hands, top-5 for {top5.shape[0]:,} pairs")

    # ---- assemble: risk UNCHANGED, behavior + evidence replaced ----
    out = sub.copy()
    out["predicted_behavior"] = behavior
    ev_cols = {f"evidence_hand_{i}": [] for i in range(1, 6)}
    top5_map = top5.to_dict()
    for pid in out["pair_id"]:
        hands = top5_map.get(pid, [])
        for i in range(1, 6):
            ev_cols[f"evidence_hand_{i}"].append(hands[i - 1] if i - 1 < len(hands) else "NO_EVIDENCE")
    for i in range(1, 6):
        out[f"evidence_hand_{i}"] = ev_cols[f"evidence_hand_{i}"]

    # sanity: risk identical to shipped
    assert np.allclose(out["risk_score"].astype(float), sub["risk_score"].astype(float)), "risk changed!"
    beh_changed = (out["predicted_behavior"] != sub["predicted_behavior"]).sum()
    ev_changed = (out["evidence_hand_1"] != sub["evidence_hand_1"]).sum()
    print(f"\nrisk_score: IDENTICAL to shipped (PairAP unchanged)")
    print(f"predicted_behavior changed on {beh_changed} rows; evidence_hand_1 changed on {ev_changed} rows")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
