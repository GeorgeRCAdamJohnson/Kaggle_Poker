"""E3 — graft V30's REFINEMENT layers onto the train-against-eval PU principle (E2).

User's point (§115/§116 correction): E2 is a BARE PU-learner on generic Layer-2 aggregates.
It plateaued across TUNING knobs, but it never got the REFINEMENTS that make V30 strong
(eval-honest 0.9203, §117). This tests whether the transport-clean PU principle COMPOSES with
V30's structure:

  E3a  PU-learner on the RICH V30 sequence features (128 cols) instead of generic aggregates.
       Isolates whether feature poverty was E2's bottleneck.
  E3b  PU FAMILY-SPECIALISTS: three train-against-eval PU heads (directed / soft / isolation,
       each = that family's confirmed positives vs unlabeled eval), unioned + rank-blended like
       V30's arms. Grafts V30's specialist+blend structure onto the HONEST training target.
  E3c  rank-blend the best E3 refinement with V30's rank_sum_w50 (0.9203) and judge eval-honest.

Thesis for why it could beat V30: E2/E3 train against the EVAL population (transport-clean),
while V30's specialists train against confirmed DEV labels (dev-selection-biased, §108/§110). If
V30's refinement power composes with E's honest target, E3 might exceed V30's dev-bias ceiling.

Gate: shared eval-honest harness (§115). Target to beat: V30 rank_sum_w50 = 0.9203.
Run:  python -m anchor_repro.detector_E3_refine
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
V30_EVAL_BASE = Path("outputs/poker_collusion/repro_lamhuy/submission.csv")
SEED = 42


def _prank(v):
    return pd.Series(np.asarray(v, np.float64)).rank(method="average", pct=True).to_numpy(np.float32)


def _seq_features():
    dev = pl.read_parquet(SEQ / "dev_sequence_features.parquet")
    ev = pl.read_parquet(SEQ / "eval_sequence_features.parquet")
    drop = {"pair_id", "player_1", "player_2", "table_id", "phase"}
    feats = [c for c in dev.columns if c not in drop and dev[c].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)]
    return dev, ev, feats


def _bagged_pu(Xd, y_pos_mask, Xe, n_bags, n_neg, depth, rounds, seed=SEED):
    """Bagged train-against-eval PU. y_pos_mask: bool over dev rows = this head's positives.
    Returns (dev_oof leak-free for positives+scored-for-rest, eval_scores)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_pos_mask)[0]
    rest_idx = np.where(~y_pos_mask)[0]
    n = len(Xd)
    oof = np.zeros(n, np.float64)
    oof_cnt = np.zeros(n, np.float64)
    rest_acc = np.zeros(len(rest_idx), np.float64)
    eval_acc = np.zeros(len(Xe), np.float64)
    for bag in range(n_bags):
        kf = KFold(5, shuffle=True, random_state=seed + bag)
        neg = Xe[rng.choice(len(Xe), n_neg, replace=False)]
        for tr_p, va_p in kf.split(pos_idx):
            tp = pos_idx[tr_p]; hp = pos_idx[va_p]
            Xtr = np.vstack([Xd[tp], neg]); ytr = np.concatenate([np.ones(len(tp)), np.zeros(n_neg)])
            spw = n_neg / max(len(tp), 1)
            m = xgb.train(xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=depth,
                                     eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                                     reg_lambda=6.0, scale_pos_weight=float(spw)),
                          xgb.DMatrix(Xtr, label=ytr), num_boost_round=rounds)
            oof[hp] += m.predict(xgb.DMatrix(Xd[hp])); oof_cnt[hp] += 1
            rest_acc += m.predict(xgb.DMatrix(Xd[rest_idx])) / 5.0
            eval_acc += m.predict(xgb.DMatrix(Xe)) / 5.0
    out = np.zeros(n, np.float32)
    out[pos_idx] = (oof[pos_idx] / np.clip(oof_cnt[pos_idx], 1, None)).astype(np.float32)
    out[rest_idx] = (rest_acc / n_bags).astype(np.float32)
    return out, (eval_acc / n_bags).astype(np.float32)


def main() -> int:
    H = Harness()
    oof = pl.read_parquet(SEQ / "oof_rows.parquet")
    # align sequence features to oof_rows order by pair_id
    dev_seq, eval_seq, feats = _seq_features()
    dev = oof.join(dev_seq.select(["pair_id", *feats]), on="pair_id", how="left")
    y = dev["y"].to_numpy().astype(np.int8)
    behavior_y = dev["behavior_y"].to_numpy().astype(np.int8)
    known = dev["known"].to_numpy().astype(bool)
    dev_ids = dev["pair_id"].to_list()
    Xd = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()

    ev = eval_seq
    Xe = ev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    eval_ids = ev["pair_id"].to_list()
    print(f"rich seq basis: {len(feats)} feats  dev={len(Xd)} eval={len(Xe)} device={xgb_device()}")
    print(f"TARGET: V30 rank_sum_w50 = 0.9203 eval-honest | E2 push = 0.65 | consistency 0.61\n")

    results = {}

    # ---- E3a: single PU head on rich seq features (positives = any confirmed positive) ----
    t = time.time()
    pos_any = (y == 1)
    doof, escore = _bagged_pu(Xd, pos_any, Xe, n_bags=8, n_neg=20000, depth=5, rounds=400)
    r = H.judge("E3a_rich_single", doof, dev_ids, escore, eval_ids)
    results["E3a_rich_single"] = (r, doof, escore)
    print(f"  E3a_rich_single   eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f} AUC={r.auc:.4f} ({time.time()-t:.0f}s)")

    # ---- E3b: PU family-specialists (directed=1, soft=2, isolation=3) unioned + rank-blended ----
    t = time.time()
    fam_dev, fam_eval = {}, {}
    for fid, fname in [(1, "directed"), (2, "soft"), (3, "isolation")]:
        mask = (behavior_y == fid)
        d, e = _bagged_pu(Xd, mask, Xe, n_bags=8, n_neg=20000, depth=5, rounds=400)
        fam_dev[fname] = d; fam_eval[fname] = e
        ap = average_precision_score(y[known], d[known])
        print(f"    PU-specialist {fname}: confirmed_AP(any-pos)={ap:.4f}  (pos={int(mask.sum())})")
    Dfam = np.column_stack([fam_dev["directed"], fam_dev["soft"], fam_dev["isolation"]])
    Efam = np.column_stack([fam_eval["directed"], fam_eval["soft"], fam_eval["isolation"]])
    # union sum + max like V30 arms
    for union, uname in [(np.clip(Dfam.sum(1), 0, 1), "sum"), (Dfam.max(1), "max")]:
        eunion = np.clip(Efam.sum(1), 0, 1) if uname == "sum" else Efam.max(1)
        r = H.judge(f"E3b_pu_spec_{uname}", union.astype(np.float32), dev_ids, eunion.astype(np.float32), eval_ids)
        results[f"E3b_pu_spec_{uname}"] = (r, union, eunion)
        print(f"  E3b_pu_spec_{uname}   eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f} AUC={r.auc:.4f}")
    print(f"    (E3b specialists built in {time.time()-t:.0f}s)")

    # ---- E3c: rank-blend the best E3 arm with V30 rank_sum_w50 ----
    # V30 arm on dev (from oof) and eval (rebuild from base+specialists)
    v30_dev = oof["rank_sum_w50"].to_numpy().astype(np.float32)
    ebase = pl.read_csv(V30_EVAL_BASE).select(["pair_id", "risk_score"])
    espec = pl.read_parquet(SEQ / "eval_specialist_scores.parquet")
    ejoin = ebase.join(espec, on="pair_id", how="inner")
    eb = ejoin["risk_score"].to_numpy().astype(np.float32)
    esp = np.column_stack([ejoin["specialist_directed"].to_numpy(), ejoin["specialist_soft"].to_numpy(),
                           ejoin["specialist_isolation"].to_numpy()]).astype(np.float32)
    v30_eval = np.clip(0.5 * _prank(eb) + 0.5 * _prank(np.clip(esp.sum(1), 0, 1)), 0, 1).astype(np.float32)
    v30_eval_ids = ejoin["pair_id"].to_list()

    # pick best E3 by eval-honest
    best_name = max(results, key=lambda k: results[k][0].eval_honest_ap)
    _, best_doof, best_escore = results[best_name]
    print(f"\n  best E3 arm = {best_name} ({results[best_name][0].eval_honest_ap:.4f}); blending with V30 w50 (0.9203)")
    # align E3 eval score to V30 eval id order
    e3_emap = dict(zip(eval_ids, best_escore))
    e3_e_aligned = np.array([e3_emap[p] for p in v30_eval_ids], np.float32)
    e3_dmap = dict(zip(dev_ids, best_doof))
    # dev: judge blends via harness (needs dev_oof over labeled + eval over eval_ids)
    for bw in [0.1, 0.2, 0.3, 0.5]:
        dblend = (1 - bw) * _prank(v30_dev) + bw * _prank(best_doof)
        eblend = (1 - bw) * _prank(v30_eval) + bw * _prank(e3_e_aligned)
        r = H.judge(f"E3c_v30blend_w{int(bw*100)}", dblend.astype(np.float32), dev_ids,
                    eblend.astype(np.float32), v30_eval_ids)
        results[f"E3c_v30blend_w{int(bw*100)}"] = (r, dblend, eblend)
        print(f"  E3c_v30blend_w{int(bw*100):02d}   eval_honest={r.eval_honest_ap:.4f} "
              f"(V30 alone 0.9203, delta {r.eval_honest_ap-0.9203:+.4f})")

    print("\n================ E3 REFINEMENT SUMMARY (gate = eval_honest_AP) ================")
    for name, (r, _, _) in sorted(results.items(), key=lambda kv: -kv[1][0].eval_honest_ap):
        tag = "  <== beats V30" if r.eval_honest_ap > 0.9203 else ""
        print(f"  {name:<22} {r.eval_honest_ap:.4f}{tag}")
    print(f"  {'V30 rank_sum_w50':<22} 0.9203  (reference)")

    out = {name: {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc}
           for name, (r, _, _) in results.items()}
    out["_v30_reference"] = 0.9203
    (CACHE / "_E3_refine_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote _E3_refine_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
