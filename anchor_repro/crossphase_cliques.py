"""Last two honghanh §9 levers: CROSS-PHASE history (#3) + CLIQUES (#4). Fast (cached substrate).

#3 CROSS-PHASE: every eval player appeared in development (100%). Build each player's DEV-period
   behavioral baseline (aggression rate, showdown rate, net/contribution profile, transfer tendency)
   from the cached seat_full + action_signals + hand phase. For each eval pair, attach both members'
   dev baselines and derive pair features (baseline similarity, combined aggression, etc). The dev
   baseline is a richer per-player prior than anything computed within eval alone.

#4 CLIQUES: 693 colluders / 372 pairs -> some players sit in multiple collusive pairs. Propagate a
   base pair-risk over the co-play graph: a pair (B,C) is boosted if B or C also appears in ANOTHER
   high-risk pair. One-hop neighbor-risk aggregation. HONEST CAVEAT (EXC-2): 0 positive dev players
   carry into eval, so there is no labeled clique anchor in eval -> measure, do not assume.

Judged eval-honest vs 0.709. Uses cached action_signals.parquet + seat_full.parquet (built §131).
Run:  python -m anchor_repro.crossphase_cliques
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device
from anchor_repro.eval_honest_harness import Harness, CACHE, D

SEAT_FULL = CACHE / "seat_full.parquet"
ACT_SIG = CACHE / "action_signals.parquet"
OUT = CACHE / "crossphase"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


# ---------------------------------------------------------------------------
# #3 CROSS-PHASE player baselines (from DEVELOPMENT hands only)
# ---------------------------------------------------------------------------
def _player_dev_baseline() -> pl.DataFrame:
    phase = pl.scan_parquet(D / "hands.parquet").select(["hand_id", "phase", "big_blind"])
    seat = pl.scan_parquet(SEAT_FULL).select(
        ["hand_id", "player_id", "net_chips", "total_contribution", "folded", "went_to_showdown", "true_strength"])
    act = pl.scan_parquet(ACT_SIG).group_by(["hand_id", "player_id"]).agg(
        pl.col("n_agg").sum().alias("n_agg"), pl.col("n_act").sum().alias("n_act"),
        pl.col("n_fold").sum().alias("n_fold"), pl.col("n_check").sum().alias("n_check"))
    dev = (seat.join(phase, on="hand_id", how="left").filter(pl.col("phase") == "development")
           .join(act, on=["hand_id", "player_id"], how="left")
           .with_columns((pl.col("net_chips") / pl.col("big_blind")).alias("net_bb")))
    return dev.group_by("player_id").agg(
        pl.len().alias("dev_hands"),
        pl.col("net_bb").mean().alias("dev_net_mean"),
        pl.col("net_bb").std().alias("dev_net_std"),
        (pl.col("n_agg") / (pl.col("n_act") + 1)).mean().alias("dev_agg_rate"),
        pl.col("folded").cast(pl.Float64).mean().alias("dev_fold_rate"),
        pl.col("went_to_showdown").cast(pl.Float64).mean().alias("dev_sd_rate"),
        pl.col("true_strength").mean().alias("dev_str_mean"),
        (pl.col("total_contribution") / pl.col("big_blind")).mean().alias("dev_contrib_mean"),
    ).collect(engine="streaming")


def _crossphase_features(pairs: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    b = base
    a1 = pairs.select(["pair_id", pl.col("player_1").alias("player_id")]).join(b, on="player_id", how="left")
    a2 = pairs.select(["pair_id", pl.col("player_2").alias("player_id")]).join(b, on="player_id", how="left")
    cols = [c for c in b.columns if c != "player_id"]
    a1 = a1.rename({c: f"a_{c}" for c in cols}).drop("player_id")
    a2 = a2.rename({c: f"b_{c}" for c in cols}).drop("player_id")
    j = a1.join(a2, on="pair_id")
    return j.with_columns(
        # both members' dev baselines + similarity/combination
        (pl.col("a_dev_agg_rate") + pl.col("b_dev_agg_rate")).alias("cp_agg_sum"),
        (pl.col("a_dev_agg_rate") - pl.col("b_dev_agg_rate")).abs().alias("cp_agg_gap"),
        (pl.col("a_dev_net_mean") + pl.col("b_dev_net_mean")).alias("cp_net_sum"),
        (pl.col("a_dev_sd_rate") + pl.col("b_dev_sd_rate")).alias("cp_sd_sum"),
        (pl.col("a_dev_str_mean") - pl.col("b_dev_str_mean")).abs().alias("cp_str_gap"),
        (pl.col("a_dev_fold_rate") + pl.col("b_dev_fold_rate")).alias("cp_fold_sum"),
        (pl.col("a_dev_contrib_mean") + pl.col("b_dev_contrib_mean")).alias("cp_contrib_sum"),
        pl.min_horizontal("a_dev_hands", "b_dev_hands").alias("cp_min_hands"),
    )


CP_FEATS = ["cp_agg_sum", "cp_agg_gap", "cp_net_sum", "cp_sd_sum", "cp_str_gap", "cp_fold_sum",
            "cp_contrib_sum", "cp_min_hands", "a_dev_agg_rate", "b_dev_agg_rate",
            "a_dev_net_std", "b_dev_net_std", "a_dev_sd_rate", "b_dev_sd_rate"]


# ---------------------------------------------------------------------------
# #4 CLIQUE propagation over the co-play graph
# ---------------------------------------------------------------------------
def _clique_features(pairs: pl.DataFrame, base_risk: dict) -> pl.DataFrame:
    """Neighbor-risk: for each pair (u,v), the max/mean base risk of OTHER pairs containing u or v."""
    # player -> list of (partner, risk) from all pairs
    rows = []
    for r in pairs.iter_rows(named=True):
        pid, u, v = r["pair_id"], r["player_1"], r["player_2"]
        risk = base_risk.get(pid, 0.0)
        rows.append((u, pid, risk)); rows.append((v, pid, risk))
    pr = pl.DataFrame(rows, schema=["player_id", "pair_id", "risk"], orient="row")
    # per player: max/mean/2nd-highest risk among their pairs
    pstat = pr.group_by("player_id").agg(
        pl.col("risk").max().alias("p_max"),
        pl.col("risk").mean().alias("p_mean"),
        pl.len().alias("p_npairs"),
        pl.col("risk").sort(descending=True).alias("p_sorted"),
    ).with_columns(
        pl.col("p_sorted").list.get(1, null_on_oob=True).alias("p_2nd").fill_null(0.0)).drop("p_sorted")
    # for each pair, neighbor risk = each member's OTHER-pair risk (exclude self via 2nd-highest proxy)
    a = pairs.select(["pair_id", pl.col("player_1").alias("player_id")]).join(pstat, on="player_id", how="left")
    b = pairs.select(["pair_id", pl.col("player_2").alias("player_id")]).join(pstat, on="player_id", how="left")
    a = a.rename({"p_max": "a_nmax", "p_mean": "a_nmean", "p_npairs": "a_ndeg", "p_2nd": "a_n2nd"}).drop("player_id")
    b = b.rename({"p_max": "b_nmax", "p_mean": "b_nmean", "p_npairs": "b_ndeg", "p_2nd": "b_n2nd"}).drop("player_id")
    j = a.join(b, on="pair_id")
    return j.with_columns(
        pl.max_horizontal("a_n2nd", "b_n2nd").alias("clique_neighbor_max"),
        (pl.col("a_nmean") + pl.col("b_nmean")).alias("clique_neighbor_mean"),
        (pl.col("a_ndeg") + pl.col("b_ndeg")).alias("clique_degree"),
    ).select(["pair_id", "clique_neighbor_max", "clique_neighbor_mean", "clique_degree", "a_n2nd", "b_n2nd"])


def main() -> int:
    H = Harness()
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])

    # ---------------- #3 CROSS-PHASE ----------------
    t = time.time()
    base = _player_dev_baseline()
    print(f"player dev baselines: {base.height:,} ({time.time()-t:.0f}s)")
    dev_cp = _crossphase_features(labels, base)
    eval_cp = _crossphase_features(eval_pairs, base)

    # ---------------- #4 CLIQUES ----------------
    # base risk = V30 rank_sum_w50 (dev oof) for dev; V30 eval risk for eval
    seq = pl.read_parquet("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
    dev_risk = dict(zip(seq["pair_id"].to_list(), seq["rank_sum_w50"].to_numpy()))
    v30csv = pl.read_csv("outputs/poker_collusion/lamhuy_sequence_confirmation/candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv")
    eval_risk = dict(zip(v30csv["pair_id"].to_list(), v30csv["risk_score"].to_numpy()))
    dev_cl = _clique_features(labels, dev_risk)
    eval_cl = _clique_features(eval_pairs, eval_risk)
    print(f"clique features built ({time.time()-t:.0f}s)")

    # combine both feature sets
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    ptbl = (pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"])
            .join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
            .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
            .select(["pair_id", "table_id"]).collect())

    def assemble(pairs, cp, cl):
        return pairs.join(cp, on="pair_id", how="left").join(cl, on="pair_id", how="left")
    dev = assemble(labels, dev_cp, dev_cl).join(ptbl, on="pair_id", how="left")
    evf = assemble(eval_pairs, eval_cp, eval_cl)

    CL_FEATS = ["clique_neighbor_max", "clique_neighbor_mean", "clique_degree", "a_n2nd", "b_n2nd"]
    y = dev["label"].to_numpy().astype(np.int8)
    g = dev["table_id"].fill_null("?").to_numpy()

    def eval_block(feats, tag):
        X = dev.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        print(f"\n=== {tag} univariate AUC ===")
        for c in feats:
            v = X[c].to_numpy()
            try:
                a = roc_auc_score(y, v); print(f"  {max(a,1-a):.4f}  {c}")
            except Exception: pass
        oof = np.zeros(len(y), np.float32)
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        pp = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                        subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
        for tr, va in sgkf.split(X, y, g):
            spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            m = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=300)
            oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        full = xgb.train({**pp, "scale_pos_weight": float(spw)}, xgb.DMatrix(X, label=y), num_boost_round=300)
        Xe = evf.select(feats).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        escore = full.predict(xgb.DMatrix(Xe))
        r = H.judge(tag, oof, dev["pair_id"].to_list(), escore, evf["pair_id"].to_list())
        print(f"  {tag}: eval-honest={r.eval_honest_ap:.4f} confirmed={r.confirmed_ap:.4f} AUC={r.auc:.4f}")
        return r

    r_cp = eval_block(CP_FEATS, "crossphase")
    r_cl = eval_block(CL_FEATS, "cliques")
    r_both = eval_block(CP_FEATS + CL_FEATS, "crossphase+cliques")

    print(f"\n=== honghanh §9 lever audit COMPLETE ===")
    print(f"  #1 templates      0.3141 (prior)")
    print(f"  #2 position       ~chance (prior)")
    print(f"  #3 cross-phase    {r_cp.eval_honest_ap:.4f}")
    print(f"  #4 cliques        {r_cl.eval_honest_ap:.4f}")
    print(f"  #3+#4 combined    {r_both.eval_honest_ap:.4f}")
    print(f"  bar = V30 real LB 0.70904")
    (OUT / "_crossphase_cliques_summary.json").write_text(json.dumps(
        {"crossphase": r_cp.eval_honest_ap, "cliques": r_cl.eval_honest_ap,
         "combined": r_both.eval_honest_ap}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
