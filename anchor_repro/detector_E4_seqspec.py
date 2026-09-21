"""E4 — PU family-specialists on the FULL V30 sequence basis (all three families targeted).

Diagnosis (e2_diagnose.py): E2 misses all three families because it has 7 count-based features
vs V30's 122 ordered-sequence features. The gap tracks sequence-dependence per family:
  directed  -0.08 (61 seq feats missing)
  soft      -0.10 (28 seq feats missing)
  isolation -0.23 (33 seq feats missing)

E4 = PU family-specialists (E3b structure) BUT on the FULL V30 sequence feature basis (E3a
features), targeting ALL three families. This is the version the dossier's own refinement
ledger (§6.3/§6.4) points to: sequence features for the family structure, trained against the
eval population (PU-honest).

Unlike E3a (monolithic head on seq features, HURT: 0.56) and E3b (specialists on count basis,
helped a little: 0.68), this gives each family's specialist the CORRECT feature basis for THAT
family's motif. The dossier says each family is a different ordered motif, so each specialist
should see its own seq_* columns.

Gate: eval-honest harness. Target: V30 rank_sum_w50 = 0.9203.

Run:  python -m anchor_repro.detector_E4_seqspec
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

# Per-family feature groups (from e2_diagnose: seq_resp_* for directed, seq_adj_* for soft,
# seq_sand_* for isolation) + shared base features from the seq frame.
_SHARED = ["shared_hands"]  # structural feature available to all


def _seq_data():
    dev = pl.read_parquet(SEQ / "dev_sequence_features.parquet")
    ev = pl.read_parquet(SEQ / "eval_sequence_features.parquet")
    all_numeric = [c for c in dev.columns
                   if c not in ("pair_id", "player_1", "player_2", "table_id", "phase")
                   and dev[c].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32, pl.UInt32)]
    dir_feats = [c for c in all_numeric if "resp" in c or "directed" in c] + _SHARED
    soft_feats = [c for c in all_numeric if "adj" in c or "soft" in c or "check_check" in c] + _SHARED
    iso_feats = [c for c in all_numeric if "sand" in c or "iso" in c or "squeeze" in c] + _SHARED
    all_feats = sorted(set(dir_feats + soft_feats + iso_feats))
    return dev, ev, {"directed": dir_feats, "soft": soft_feats, "isolation": iso_feats, "all": all_feats}


def _prank(v):
    return pd.Series(np.asarray(v, np.float64)).rank(method="average", pct=True).to_numpy(np.float32)


def _bagged_pu(Xd, pos_mask, Xe, n_bags, n_neg, depth, rounds, seed=SEED):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(pos_mask)[0]
    rest_idx = np.where(~pos_mask)[0]
    n = len(Xd)
    oof = np.zeros(n, np.float64); oof_cnt = np.zeros(n, np.float64)
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
            oof[hp] += m.predict(xgb.DMatrix(Xd[hp])); oof_cnt[hp] += 1
            rest_acc += m.predict(xgb.DMatrix(Xd[rest_idx])) / 5.0
            eval_acc += m.predict(xgb.DMatrix(Xe)) / 5.0
    out = np.zeros(n, np.float32)
    out[pos_idx] = (oof[pos_idx] / np.clip(oof_cnt[pos_idx], 1, None)).astype(np.float32)
    out[rest_idx] = (rest_acc / n_bags).astype(np.float32)
    return out, (eval_acc / n_bags).astype(np.float32)


def main() -> int:
    H = Harness()
    oof_base = pl.read_parquet(SEQ / "oof_rows.parquet")
    dev_seq, eval_seq, feat_groups = _seq_data()
    dev = oof_base.join(dev_seq.select(["pair_id", *feat_groups["all"]]), on="pair_id", how="left")
    y = dev["y"].to_numpy().astype(np.int8)
    fam = dev["behavior_y"].to_numpy().astype(np.int8)
    known = dev["known"].to_numpy().astype(bool)
    dev_ids = dev["pair_id"].to_list()
    eval_ids = eval_seq["pair_id"].to_list()
    print(f"device={xgb_device()}  dev={len(y)} eval={len(eval_ids)}")
    print(f"TARGET: V30 0.9203 | E3b 0.677 | E2 0.65 | consistency 0.61\n")

    results = {}

    # --- E4a: single PU head on ALL seq features (control: is monolithic + seq features still bad?) ---
    t = time.time()
    feats = feat_groups["all"]
    Xd_all = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    Xe_all = eval_seq.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    doof, escore = _bagged_pu(Xd_all, y == 1, Xe_all, n_bags=8, n_neg=20000, depth=5, rounds=400)
    r = H.judge("E4a_mono_allseq", doof, dev_ids, escore, eval_ids)
    results["E4a_mono_allseq"] = r
    print(f"  E4a_mono_allseq   eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f} ({time.time()-t:.0f}s)")

    # --- E4b: PU specialists on FAMILY-SPECIFIC seq features + union + rank-blend ---
    t = time.time()
    fam_dev, fam_eval = {}, {}
    for fid, fname, fgroup in [(1, "directed", "directed"), (2, "soft", "soft"), (3, "isolation", "isolation")]:
        ff = feat_groups[fgroup]
        Xdf = dev.select(ff).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
        Xef = eval_seq.select(ff).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
        mask = (fam == fid)
        d, e = _bagged_pu(Xdf, mask, Xef, n_bags=8, n_neg=20000, depth=5, rounds=400)
        fam_dev[fname] = d; fam_eval[fname] = e
        # per-family AP (this family vs negatives only)
        keep = known & ((fam == 0) | (fam == fid))
        yf = (fam[keep] == fid).astype(int)
        ap = average_precision_score(yf, d[keep])
        print(f"    specialist {fname}: AP(vs neg)={ap:.4f}  (n_pos={int(mask.sum())}, feats={len(ff)})")

    # union sum/max
    Dfam = np.column_stack([fam_dev["directed"], fam_dev["soft"], fam_dev["isolation"]])
    Efam = np.column_stack([fam_eval["directed"], fam_eval["soft"], fam_eval["isolation"]])
    for uname, du, eu in [("sum", np.clip(Dfam.sum(1), 0, 1), np.clip(Efam.sum(1), 0, 1)),
                          ("max", Dfam.max(1), Efam.max(1))]:
        r = H.judge(f"E4b_spec_{uname}", du.astype(np.float32), dev_ids, eu.astype(np.float32), eval_ids)
        results[f"E4b_spec_{uname}"] = r
        print(f"  E4b_spec_{uname}    eval_honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f}")

    # rank-blend with V30
    v30_dev = oof_base["rank_sum_w50"].to_numpy().astype(np.float32)
    ebase = pl.read_csv(V30_EVAL_BASE).select(["pair_id", "risk_score"])
    espec = pl.read_parquet(SEQ / "eval_specialist_scores.parquet")
    ej = ebase.join(espec, on="pair_id", how="inner")
    eb = ej["risk_score"].to_numpy().astype(np.float32)
    esp = np.column_stack([ej["specialist_directed"].to_numpy(), ej["specialist_soft"].to_numpy(),
                           ej["specialist_isolation"].to_numpy()]).astype(np.float32)
    v30_eval = np.clip(0.5 * _prank(eb) + 0.5 * _prank(np.clip(esp.sum(1), 0, 1)), 0, 1).astype(np.float32)
    v30_eval_ids = ej["pair_id"].to_list()

    best_name = max(results, key=lambda k: results[k].eval_honest_ap)
    best_r = results[best_name]
    print(f"\n  best E4 arm = {best_name} ({best_r.eval_honest_ap:.4f})")
    # get the right dev/eval scores for blending
    if "spec_sum" in best_name:
        bd, be = np.clip(Dfam.sum(1), 0, 1), np.clip(Efam.sum(1), 0, 1)
    elif "spec_max" in best_name:
        bd, be = Dfam.max(1), Efam.max(1)
    else:
        bd, be = doof, escore
    e4_emap = dict(zip(eval_ids, be))
    e4_ea = np.array([e4_emap.get(p, np.nanmedian(be)) for p in v30_eval_ids], np.float32)
    print("  blending with V30:")
    for bw in [0.05, 0.1, 0.2, 0.3, 0.5]:
        db = (1 - bw) * _prank(v30_dev) + bw * _prank(bd)
        eb = (1 - bw) * _prank(v30_eval) + bw * _prank(e4_ea)
        r = H.judge(f"E4c_blend_w{int(bw*100)}", db.astype(np.float32), dev_ids, eb.astype(np.float32), v30_eval_ids)
        results[f"E4c_blend_w{int(bw*100)}"] = r
        print(f"    blend w={bw:.2f}: eval_honest={r.eval_honest_ap:.4f} ({r.eval_honest_ap-0.9203:+.4f} vs V30)")

    print(f"\n  ({time.time()-t:.0f}s for specialists + blends)")

    print("\n================ E4 SUMMARY (gate = eval_honest_AP) ================")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].eval_honest_ap):
        tag = " <== beats V30" if r.eval_honest_ap > 0.9203 else ""
        print(f"  {name:<22} {r.eval_honest_ap:.4f}{tag}")
    print(f"  {'V30 rank_sum_w50':<22} 0.9203  (target)")
    print(f"  {'E3b pu_spec(count)':<22} 0.6768  (prior)")
    print(f"  {'E2 bare':<22} 0.6500  (prior)")

    out = {name: {"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc}
           for name, r in results.items()}
    (CACHE / "_E4_seqspec_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote _E4_seqspec_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
