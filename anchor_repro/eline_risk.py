"""E-LINE RISK column — assemble the sequence-feature PU specialists into ONE risk.

The E-line's own risk head, standing alone (no V30 base borrowing). E4 proved the three
sequence-feature PU specialists are individually strong (directed 0.93, soft 0.94, iso 0.76)
but a raw sum/max union throws away signal (0.64-0.68). This builds the COMBINATION LAYER the
union lacked: percentile-rank each specialist, then combine via rank-blend arms, AND train an
E-line BASE PU head (all-positives on the full sequence basis) to blend with — mirroring V30's
base+specialist arm structure but entirely on the honest train-against-eval target.

Persists EVERYTHING (dev OOF + eval scores for base + 3 specialists + assembled arms) so the
behavior and evidence columns and the final assembly can reuse them with no recompute.

Run:  python -m anchor_repro.eline_risk
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
from anchor_repro.eval_honest_harness import Harness, weighted_ap, CACHE, D

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
OUT = CACHE / "eline"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _prank(v):
    return pd.Series(np.asarray(v, np.float64)).rank(method="average", pct=True).to_numpy(np.float32)


def _seq_data():
    dev = pl.read_parquet(SEQ / "dev_sequence_features.parquet")
    ev = pl.read_parquet(SEQ / "eval_sequence_features.parquet")
    num = [c for c in dev.columns
           if c not in ("pair_id", "player_1", "player_2", "table_id", "phase")
           and dev[c].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32, pl.UInt32)]
    groups = {
        "directed": [c for c in num if "resp" in c or "directed" in c] + ["shared_hands"],
        "soft": [c for c in num if "adj" in c or "soft" in c or "check_check" in c] + ["shared_hands"],
        "isolation": [c for c in num if "sand" in c or "iso" in c or "squeeze" in c] + ["shared_hands"],
        "all": sorted(num),
    }
    return dev, ev, groups


def _bagged_pu(Xd, pos_mask, Xe, n_bags, n_neg, depth, rounds, seed=SEED):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(pos_mask)[0]
    rest_idx = np.where(~pos_mask)[0]
    n = len(Xd)
    oof = np.zeros(n, np.float64); cnt = np.zeros(n, np.float64)
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
                                     eta=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
                                     reg_lambda=5.0, scale_pos_weight=float(spw)),
                          xgb.DMatrix(Xtr, label=ytr), num_boost_round=rounds)
            oof[hp] += m.predict(xgb.DMatrix(Xd[hp])); cnt[hp] += 1
            rest_acc += m.predict(xgb.DMatrix(Xd[rest_idx])) / 5.0
            eval_acc += m.predict(xgb.DMatrix(Xe)) / 5.0
    out = np.zeros(n, np.float32)
    out[pos_idx] = (oof[pos_idx] / np.clip(cnt[pos_idx], 1, None)).astype(np.float32)
    out[rest_idx] = (rest_acc / n_bags).astype(np.float32)
    return out, (eval_acc / n_bags).astype(np.float32)


def main() -> int:
    H = Harness()
    oof = pl.read_parquet(SEQ / "oof_rows.parquet")
    dev_seq, eval_seq, groups = _seq_data()
    dev = oof.join(dev_seq.select(["pair_id", *groups["all"]]), on="pair_id", how="left")
    y = dev["y"].to_numpy().astype(np.int8)
    fam = dev["behavior_y"].to_numpy().astype(np.int8)
    known = dev["known"].to_numpy().astype(bool)
    dev_ids = dev["pair_id"].to_list()
    eval_ids = eval_seq["pair_id"].to_list()
    # H.judge() aligns dev_oof to the labeled harness pairs internally by pair_id, so no manual
    # weight vector is needed here (dev has 25,860 rows incl PU pairs not in the labeled set).
    print(f"device={xgb_device()} dev={len(y)} eval={len(eval_ids)}")

    def X(cols, frame):
        return frame.select(cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()

    # ---- BASE PU head (all positives, full sequence basis) ----
    t = time.time()
    Xd_all, Xe_all = X(groups["all"], dev), X(groups["all"], eval_seq)
    base_d, base_e = _bagged_pu(Xd_all, y == 1, Xe_all, n_bags=10, n_neg=20000, depth=5, rounds=450)
    print(f"  base PU head built ({time.time()-t:.0f}s)")

    # ---- 3 family specialists on family-specific sequence features ----
    spec_d, spec_e = {}, {}
    for fid, fname in [(1, "directed"), (2, "soft"), (3, "isolation")]:
        t = time.time()
        Xdf, Xef = X(groups[fname], dev), X(groups[fname], eval_seq)
        d, e = _bagged_pu(Xdf, fam == fid, Xef, n_bags=10, n_neg=20000, depth=5, rounds=450)
        spec_d[fname], spec_e[fname] = d, e
        keep = known & ((fam == 0) | (fam == fid))
        ap = average_precision_score((fam[keep] == fid).astype(int), d[keep])
        print(f"  specialist {fname}: AP(vs neg)={ap:.4f} ({time.time()-t:.0f}s)")

    # ---- combination layer: build arms (rank-blend base + specialist union) ----
    Dspec = np.column_stack([spec_d["directed"], spec_d["soft"], spec_d["isolation"]])
    Espec = np.column_stack([spec_e["directed"], spec_e["soft"], spec_e["isolation"]])
    d_usum, e_usum = np.clip(Dspec.sum(1), 0, 1), np.clip(Espec.sum(1), 0, 1)
    d_umax, e_umax = Dspec.max(1), Espec.max(1)

    def judge(name, dv, ev):
        r = H.judge(name, dv.astype(np.float32), dev_ids, ev.astype(np.float32), eval_ids)
        print(f"  {name:<22} eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f}")
        return r.eval_honest_ap

    print("\n=== E-line risk arms (eval-honest) ===")
    arms_dev, arms_eval, arm_scores = {}, {}, {}
    # rank-normalized building blocks
    dbr, ebr = _prank(base_d), _prank(base_e)
    dsr, esr = _prank(d_usum), _prank(e_usum)
    dmr, emr = _prank(d_umax), _prank(e_umax)
    # per-specialist rank then mean (proper multi-head combination)
    d_specmean = np.mean([_prank(spec_d[f]) for f in ("directed", "soft", "isolation")], axis=0)
    e_specmean = np.mean([_prank(spec_e[f]) for f in ("directed", "soft", "isolation")], axis=0)

    candidates = {
        "base_only": (base_d, base_e),
        "union_sum": (d_usum, e_usum),
        "union_max": (d_umax, e_umax),
        "specrank_mean": (d_specmean, e_specmean),
    }
    for w_spec in [0.3, 0.5, 0.7]:
        candidates[f"base+specmean_w{int(w_spec*100)}"] = (
            (1 - w_spec) * dbr + w_spec * d_specmean,
            (1 - w_spec) * ebr + w_spec * e_specmean)
        candidates[f"base+unionmax_w{int(w_spec*100)}"] = (
            (1 - w_spec) * dbr + w_spec * dmr,
            (1 - w_spec) * ebr + w_spec * emr)
    for name, (dv, ev) in candidates.items():
        arm_scores[name] = judge(name, dv, ev)
        arms_dev[name], arms_eval[name] = dv, ev

    best = max(arm_scores, key=arm_scores.get)
    print(f"\nBEST E-line risk arm: {best} = {arm_scores[best]:.4f} eval-honest")

    # persist everything for behavior/evidence/assembly
    pl.DataFrame({"pair_id": dev_ids, "base": base_d,
                  "spec_directed": spec_d["directed"], "spec_soft": spec_d["soft"],
                  "spec_isolation": spec_d["isolation"],
                  "risk_best": arms_dev[best].astype(np.float32)}).write_parquet(OUT / "risk_dev.parquet")
    pl.DataFrame({"pair_id": eval_ids, "base": base_e,
                  "spec_directed": spec_e["directed"], "spec_soft": spec_e["soft"],
                  "spec_isolation": spec_e["isolation"],
                  "risk_best": arms_eval[best].astype(np.float32)}).write_parquet(OUT / "risk_eval.parquet")
    (OUT / "_risk_summary.json").write_text(json.dumps(
        {"best_arm": best, "arm_scores": arm_scores}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}/risk_dev.parquet, risk_eval.parquet, _risk_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
