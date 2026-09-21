"""Build OUR OWN behavior head from V30's ALREADY-COMPUTED per-family specialists.

Discovery: outputs/.../oof_rows.parquet ALREADY carries specialist_directed / specialist_soft /
specialist_isolation (per-family risk) + behavior_y (0=none,1=directed,2=soft,3=iso). V30 computed
these and we NEVER used them to emit predicted_behavior — we borrowed honghanh's head instead. This
wires our own specialists into a behavior head and scores BehaviorMAP@dev via CanonicalScorer.

BehaviorMAP = mean over the 3 TARGET families of OvR AP, where for family f the score vector is
risk_score where predicted_behavior==f else 0. So the lever is: (1) pick each pair's family by argmax
of the 3 specialists, (2) route enough high-risk pairs to a NON-'none' family so true positives enter
their family's ranking. We sweep the positive-rate (top fraction routed to a family) to maximize
dev BehaviorMAP, exactly like honghanh's best_rate rule.

Run: python -m anchor_repro.v30_behavior_head
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
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior",
            "evidence_hand_1", "evidence_hand_2", "evidence_hand_3", "evidence_hand_4", "evidence_hand_5"]
FAMILY = {1: "directed_transfer", 2: "soft_play", 3: "coordinated_isolation"}


def main() -> int:
    cfg = PipelineConfig()
    labels = pd.read_csv(Path(cfg.input_dir) / "development_labels.csv")
    ev_path = Path(cfg.input_dir) / "development_evidence.csv"
    evidence = pd.read_csv(ev_path) if ev_path.is_file() else None
    recipe = ScoringRecipe(recipe_id="v30_behavior_head_v1")
    scorer = CanonicalScorer(recipe, cfg, labels=labels, evidence=evidence)
    conf_ids = set(labels["pair_id"].astype(str))

    oof = pd.read_parquet(OOF).drop_duplicates("pair_id")
    oof["pair_id"] = oof["pair_id"].astype(str)
    dev = oof[oof["pair_id"].isin(conf_ids)].copy()
    print(f"dev pairs with specialists: {len(dev)}")

    # normalize V30 risk to [0,1]
    r = dev["rank_sum_w50"].astype(float)
    dev["risk_score"] = ((r - r.min()) / (r.max() - r.min() + 1e-12)).clip(0, 1)

    spec = dev[["specialist_directed", "specialist_soft", "specialist_isolation"]].to_numpy()
    # argmax family per pair (1=directed,2=soft,3=iso)
    fam_idx = spec.argmax(axis=1) + 1
    dev["fam"] = [FAMILY[i] for i in fam_idx]
    # a "behavior confidence" = how much the top specialist stands out (for routing order)
    dev["fam_conf"] = dev["risk_score"]  # route by risk (high-risk pairs get a family)

    def score_with_rate(rate: float):
        d = dev.copy()
        # top `rate` fraction by risk get their argmax family; rest -> 'none'
        thr = d["risk_score"].quantile(1 - rate)
        d["predicted_behavior"] = np.where(d["risk_score"] >= thr, d["fam"], "none")
        for i in range(1, 6):
            d[f"evidence_hand_{i}"] = "NO_EVIDENCE"
        f = d.reindex(columns=SUB_COLS)
        s = scorer.score_dev_predictions(f)
        c = s.components
        return c.pair_ap, c.evidence_map, c.behavior_map, c.combined

    print("\n=== BehaviorMAP vs positive-rate (fraction routed to a family) ===")
    best = (-1, None)
    results = []
    for rate in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 1.0]:
        pa, ev, bm, cb = score_with_rate(rate)
        print(f"  rate={rate:<5}  PairAP={pa:.4f}  Behav={bm:.4f}  combined={cb:.4f}")
        results.append({"rate": rate, "pair_ap": pa, "behavior_map": bm, "combined": cb})
        if bm > best[0]:
            best = (bm, rate)
    print(f"\nbest BehaviorMAP={best[0]:.4f} at rate={best[1]}  (borrowed head was 0.0806; honghanh 0.8935)")

    # also: pure argmax with NO 'none' floor (all pairs get a family)
    Path("outputs/poker_collusion/component_decomposition").mkdir(parents=True, exist_ok=True)
    (Path("outputs/poker_collusion/component_decomposition") / "_v30_behavior_head.json").write_text(
        json.dumps({"results": results, "best_behavior_map": best[0], "best_rate": best[1]}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
