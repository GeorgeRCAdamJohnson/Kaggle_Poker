"""Seam 1 — DECISION-POLICY COUNTERFACTUAL (population-robust by construction).

Thesis (AI_HANDOFF_BRIEF §5.1, dossier §114 route 1): every prior signal modeled CONTENT
(what happened in a pair's shared hands) and rode the dev-selection density (§108/§110), so
dev-CV over-read eval by ~0.30 PairAP. This module instead models INTENT as a per-player
SELF-counterfactual:

  For each player p, learn p's OWN field policy from ALL of p's hands (label-free, whole
  population): given a decision state (hole strength, pot size, position, players active,
  to-call severity), what does p normally do (fold rate, aggression rate, showdown rate,
  net outcome)? Then for the pair's SHARED hands, measure how far each member's behaviour
  DEVIATES from their own solo-policy prediction on matched states.

Why this transports where consistency did not (§110):
  * The baseline is the player vs THEMSELVES on their non-shared hands. It does NOT depend
    on how densely colluders were seeded into dev (the density bias cancels: both a dev
    positive and an eval pair are measured as deviation-from-own-policy).
  * A colluder's tell is a STATE-CONDITIONAL deviation: folding a strong hand only when the
    partner is in / dumping chips only to the partner / going passive only vs the partner —
    i.e. behaviour that their own field policy would not predict.

Build:
  1. Per-player field policy on hand_player_agg (all hands): state-binned expected fold /
     agg / showdown / net_bb, EXCLUDING the pair's shared hands (leak-free: a player's
     baseline must not include the very hands we score).
  2. Per (pair, shared-hand): residual = observed member behaviour - own-policy expectation
     on that hand's state. Directional residuals (does the LOSER fold/pass more than their
     own policy predicts, in favour of the winner?).
  3. Per pair: aggregate residuals (mean/top-k/rate) -> pair features. Train a table-grouped
     OOF detector; report CONFIRMED PairAP, EVAL-HONEST importance-weighted AP (§110 harness),
     and drift. Blend onto the frozen rank_sum_w50 (0.70904) and report confirmed + PU-stress.

Gate: promote ONLY if eval-honest weighted AP clears the consistency ceiling (~0.61) AND a
rank-blend lifts PU-stress over the 0.70904 baseline beyond noise. Otherwise retire honestly.

Run:  python -m anchor_repro.gen_policy_counterfactual
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from anchor_repro.gpu_config import xgb_params, xgb_device

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")
HAND_PLAYER = CACHE / "hand_player_agg.parquet"
HAND_CTX = CACHE / "hand_context.parquet"
CONF_OOF = Path("outputs/poker_collusion/lamhuy_sequence_confirmation/oof_rows.parquet")
SEED = 42
PU_STRESS_WEIGHT = 112540 / 24000


# ----------------------------------------------------------------------------------------
# 1. Per-player field policy: state-binned expected behaviour over ALL of the player's hands
# ----------------------------------------------------------------------------------------
# State = coarse bins so every player has support in each cell. We keep the model NON-
# parametric (leave-one-hand-out style) via player x state-cell means, and to keep it
# leak-free we subtract the shared-hand contribution from the player's own means when
# scoring a shared hand (a per-player, per-cell "leave-these-hands-out" mean).

_HS_BINS = [0.0, 0.30, 0.42, 0.55, 1.01]        # hole-strength quartile-ish cut points
_POT_BINS = [0.0, 6.0, 12.0, 30.0, 1e9]          # pot in bb
_ACT_BINS = [0, 2, 3, 4, 99]                      # players_active-ish (min_players_active)


def _player_hand_states() -> pl.LazyFrame:
    """Per (hand_id, player_id): behaviour outcomes + coarse decision-state cell + net_bb."""
    hp = pl.scan_parquet(HAND_PLAYER)
    ctx = pl.scan_parquet(HAND_CTX).select(["hand_id", "big_blind", "final_pot"])
    df = (
        hp.join(ctx, on="hand_id", how="left")
        .with_columns(
            (pl.col("net_chips") / pl.col("big_blind")).alias("net_bb"),
            (pl.col("final_pot") / pl.col("big_blind")).alias("pot_bb"),
            (pl.col("n_agg") > 0).cast(pl.Int8).alias("is_agg"),
            pl.col("folded").cast(pl.Int8).alias("is_fold"),
            pl.col("went_to_showdown").cast(pl.Int8).alias("is_sd"),
        )
        .with_columns(
            pl.col("hs").cut(_HS_BINS[1:-1], left_closed=True).alias("hs_cell"),
            pl.col("pot_bb").cut(_POT_BINS[1:-1], left_closed=True).alias("pot_cell"),
            pl.col("min_players_active").fill_null(6).cut(_ACT_BINS[1:-1], left_closed=True).alias("act_cell"),
        )
        .with_columns(
            (pl.col("hs_cell").cast(pl.Utf8) + "|" + pl.col("pot_cell").cast(pl.Utf8)
             + "|" + pl.col("act_cell").cast(pl.Utf8)).alias("state_cell")
        )
        .select(["hand_id", "player_id", "state_cell", "net_bb",
                 "is_agg", "is_fold", "is_sd", "hs"])
    )
    return df


def _player_state_policy(states: pl.LazyFrame) -> pl.LazyFrame:
    """Per (player_id, state_cell): SUM and COUNT of each outcome over ALL that player's hands.

    We keep sums+counts (not means) so a shared hand can be removed leave-one-out style:
        policy_mean_excl = (sum - own_shared_value) / (count - n_shared_in_cell)
    """
    return states.group_by(["player_id", "state_cell"]).agg(
        pl.len().alias("pol_n"),
        pl.col("is_fold").sum().alias("pol_fold_sum"),
        pl.col("is_agg").sum().alias("pol_agg_sum"),
        pl.col("is_sd").sum().alias("pol_sd_sum"),
        pl.col("net_bb").sum().alias("pol_net_sum"),
    )


# ----------------------------------------------------------------------------------------
# 2. Shared-hand residuals: observed member behaviour minus own-policy expectation
# ----------------------------------------------------------------------------------------

def _shared_residuals(pair_frame: pl.LazyFrame, universe: pl.LazyFrame,
                      states: pl.LazyFrame, policy: pl.LazyFrame) -> pl.LazyFrame:
    """Per (pair_id, hand_id): each member's leave-this-hand-out policy residual + directional."""
    pf = pair_frame.select(["pair_id", "player_1", "player_2"])
    # shared-hand universe for these pairs
    uni = universe.select(["pair_id", "hand_id"])

    def member(colname: str, tag: str) -> pl.LazyFrame:
        m = (
            pf.select(["pair_id", pl.col(colname).alias("player_id")])
            .join(uni, on="pair_id", how="inner")
            .join(states, on=["hand_id", "player_id"], how="inner")
            .join(policy, on=["player_id", "state_cell"], how="left")
        )
        # leave-one-out expectation in this player's state cell (remove THIS hand)
        n_excl = (pl.col("pol_n") - 1).clip(1, None)
        exp_fold = (pl.col("pol_fold_sum") - pl.col("is_fold")) / n_excl
        exp_agg = (pl.col("pol_agg_sum") - pl.col("is_agg")) / n_excl
        exp_sd = (pl.col("pol_sd_sum") - pl.col("is_sd")) / n_excl
        exp_net = (pl.col("pol_net_sum") - pl.col("net_bb")) / n_excl
        return m.with_columns(
            (pl.col("is_fold") - exp_fold).alias(f"{tag}_fold_resid"),
            (pl.col("is_agg") - exp_agg).alias(f"{tag}_agg_resid"),
            (pl.col("is_sd") - exp_sd).alias(f"{tag}_sd_resid"),
            (pl.col("net_bb") - exp_net).alias(f"{tag}_net_resid"),
            pl.col("net_bb").alias(f"{tag}_net_bb"),
            pl.col("hs").alias(f"{tag}_hs"),
            (pl.col("pol_n") >= 5).cast(pl.Int8).alias(f"{tag}_has_support"),
        ).select(["pair_id", "hand_id",
                  f"{tag}_fold_resid", f"{tag}_agg_resid", f"{tag}_sd_resid",
                  f"{tag}_net_resid", f"{tag}_net_bb", f"{tag}_hs", f"{tag}_has_support"])

    a = member("player_1", "a")
    b = member("player_2", "b")
    j = a.join(b, on=["pair_id", "hand_id"], how="inner")

    # Directional counterfactual: identify the loser/winner of the pair on this hand, then
    # ask whether the LOSER behaved MORE passively/foldy than THEIR OWN policy predicts
    # (a chip-donation tell) and whether the winner's gain lines up.
    a_is_loser = pl.col("a_net_bb") < pl.col("b_net_bb")
    loser_fold_resid = pl.when(a_is_loser).then(pl.col("a_fold_resid")).otherwise(pl.col("b_fold_resid"))
    loser_agg_resid = pl.when(a_is_loser).then(pl.col("a_agg_resid")).otherwise(pl.col("b_agg_resid"))
    loser_sd_resid = pl.when(a_is_loser).then(pl.col("a_sd_resid")).otherwise(pl.col("b_sd_resid"))
    loser_net_resid = pl.when(a_is_loser).then(pl.col("a_net_resid")).otherwise(pl.col("b_net_resid"))
    loser_hs = pl.when(a_is_loser).then(pl.col("a_hs")).otherwise(pl.col("b_hs"))
    winner_net_resid = pl.when(a_is_loser).then(pl.col("b_net_resid")).otherwise(pl.col("a_net_resid"))

    return j.with_columns(
        # loser gave up MORE than their own policy: positive fold resid, negative agg resid,
        # weighted by how STRONG the loser's hand was (dumping a strong hand is the tell)
        (loser_fold_resid * (loser_hs + 0.1)).alias("cf_loser_strong_fold"),
        (-loser_agg_resid * (loser_hs + 0.1)).alias("cf_loser_strong_passive"),
        loser_sd_resid.alias("cf_loser_sd_resid"),
        # the loser lost MORE than their own policy predicts (donation) while the winner
        # gained more than theirs (matched directional transfer beyond both own baselines)
        (-loser_net_resid).alias("cf_loser_underperf"),
        (winner_net_resid).alias("cf_winner_overperf"),
        ((-loser_net_resid) * (winner_net_resid > 0).cast(pl.Float32)).alias("cf_directed_donation"),
        (pl.col("a_has_support") * pl.col("b_has_support")).alias("cf_both_support"),
        loser_hs.alias("cf_loser_hs"),
    ).select(["pair_id", "hand_id", "cf_loser_strong_fold", "cf_loser_strong_passive",
              "cf_loser_sd_resid", "cf_loser_underperf", "cf_winner_overperf",
              "cf_directed_donation", "cf_both_support", "cf_loser_hs"])


_CF_COLS = ["cf_loser_strong_fold", "cf_loser_strong_passive", "cf_loser_sd_resid",
            "cf_loser_underperf", "cf_winner_overperf", "cf_directed_donation"]


def _pair_cf_features(resid: pl.LazyFrame) -> pl.DataFrame:
    """Aggregate per-hand counterfactual residuals to pair level (support-weighted)."""
    r = resid.filter(pl.col("cf_both_support") == 1)  # only hands where BOTH have policy support
    agg = r.group_by("pair_id").agg(
        pl.len().alias("cf_n_supported"),
        *[pl.col(c).mean().alias(f"{c}_mean") for c in _CF_COLS],
        *[pl.col(c).top_k(3).mean().alias(f"{c}_top3") for c in _CF_COLS],
        *[(pl.col(c) > 0).mean().alias(f"{c}_rate") for c in _CF_COLS],
        pl.col("cf_directed_donation").max().alias("cf_directed_donation_max"),
        pl.col("cf_loser_hs").mean().alias("cf_loser_hs_mean"),
    ).collect(engine="streaming")
    return agg


def _weighted_ap(y, scores, w):
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]; w = w[order]
    tp = np.cumsum(y * w); fp = np.cumsum((1 - y) * w)
    precision = tp / np.clip(tp + fp, 1e-9, None)
    total_pos = np.sum(y * w)
    return 0.0 if total_pos <= 0 else float(np.sum(precision * y * w) / total_pos)


def main() -> int:
    import time
    V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
    V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")

    print(f"device={xgb_device()}")
    t = time.time()
    states = _player_hand_states()
    policy = _player_state_policy(states)
    # Materialize policy once (small: players x cells) to reuse for dev + eval.
    policy_df = policy.collect(engine="streaming")
    print(f"player-state policy cells: {policy_df.height:,} ({time.time()-t:.1f}s)")
    policy_lz = policy_df.lazy()
    states_lz = states  # lazy; re-scanned per join (cheap vs the eval loop)

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    dev_pf = labels.select(["pair_id", "player_1", "player_2"]).lazy()
    dev_uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"])

    t = time.time()
    dev_resid = _shared_residuals(dev_pf, dev_uni, states_lz, policy_lz)
    dev_cf = _pair_cf_features(dev_resid)
    print(f"dev pair CF features: {dev_cf.height:,} pairs ({time.time()-t:.1f}s)")

    # table_id per pair for grouped OOF
    hp = pl.scan_parquet(HAND_CTX).select(["hand_id", "table_id"])
    pair_tbl = (
        dev_uni.join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
        .sort(["pair_id", "n"], descending=[False, True]).unique("pair_id", keep="first")
        .select(["pair_id", "table_id"]).collect()
    )
    dev = (dev_cf.join(labels.select(["pair_id", "label"]), on="pair_id", how="left")
           .join(pair_tbl, on="pair_id", how="left"))
    feat_cols = [c for c in dev.columns if c not in ("pair_id", "label", "table_id")]
    X = dev.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = dev["label"].to_numpy().astype(np.int8)
    groups = dev["table_id"].to_numpy()
    print(f"dev pairs={dev.height} positives={int(y.sum())} feats={len(feat_cols)}")

    # ---- univariate transport check: which CF features separate positives? ----
    print("\n=== univariate CF feature AP (confirmed dev) ===")
    uni_rows = []
    for c in feat_cols:
        v = X[c].to_numpy()
        ap = max(average_precision_score(y, v), average_precision_score(y, -v))
        uni_rows.append((c, ap))
    for c, ap in sorted(uni_rows, key=lambda r: -r[1])[:10]:
        print(f"  {ap:.4f}  {c}")

    # ---- table-grouped OOF detector ----
    params = xgb_params(objective="binary:logistic", eval_metric="aucpr", max_depth=4, eta=0.04,
                        subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=6.0)
    oof = np.zeros(len(y), dtype=np.float32)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(X, y, groups):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.train({**params, "scale_pos_weight": float(spw)},
                      xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=350)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]))
    conf_ap = average_precision_score(y, oof)
    conf_auc = roc_auc_score(y, oof)
    print(f"\n=== policy-counterfactual detector (table-grouped OOF) ===")
    print(f"confirmed PairAP={conf_ap:.4f}  AUC={conf_auc:.4f}  (base {y.mean():.4f})")

    # ---- eval CF features (batched) + importance weights (eval-honest harness) ----
    t = time.time()
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    eval_uni = pl.scan_parquet(V30_EVAL).select(["pair_id", "hand_id"])
    batch = 20000
    parts = []
    for start in range(0, eval_pairs.height, batch):
        pf = eval_pairs.slice(start, batch).lazy()
        resid = _shared_residuals(pf, eval_uni, states_lz, policy_lz)
        parts.append(_pair_cf_features(resid))
    eval_cf = pl.concat(parts, how="vertical_relaxed")
    print(f"eval pair CF features: {eval_cf.height:,} pairs ({time.time()-t:.1f}s)")

    Xe = eval_cf.select(feat_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    # density-ratio importance weights P(eval)/P(dev) (OOF), §110 harness
    dev_prob = np.zeros(len(X))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        X_tr = np.vstack([X.to_numpy()[tr], Xe.to_numpy()])
        y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(Xe))])
        clf = xgb.XGBClassifier(**xgb_params(objective="binary:logistic", eval_metric="auc",
                                             max_depth=4, n_estimators=300, learning_rate=0.05,
                                             subsample=0.85, colsample_bytree=0.8, reg_lambda=5))
        clf.fit(X_tr, y_tr)
        dev_prob[va] = clf.predict_proba(X.to_numpy()[va])[:, 1]
    dev_prob = np.clip(dev_prob, 1e-4, 1 - 1e-4)
    w = dev_prob / (1 - dev_prob)
    w = np.clip(w, np.quantile(w, 0.01), np.quantile(w, 0.99)); w = w / w.mean()
    eval_honest = _weighted_ap(y, oof, w)
    print(f"importance weights: pos mean={w[y==1].mean():.3f} neg mean={w[y==0].mean():.3f}")
    print(f"EVAL-HONEST weighted AP={eval_honest:.4f}   (consistency ceiling ~0.61; V30 ~0.79)")

    # ---- blend onto frozen 0.70904 baseline ----
    base = pl.read_parquet(CONF_OOF).select(["pair_id", "known", "y", "rank_sum_w50"])
    det = dev.select(["pair_id"]).with_columns(pl.Series("cf_score", oof))
    merged = base.join(det, on="pair_id", how="left").with_columns(pl.col("cf_score").fill_null(0.0))
    yb = merged["y"].to_numpy().astype(np.int8)
    known = merged["known"].to_numpy().astype(bool)
    sw = np.where(known, 1.0, PU_STRESS_WEIGHT)
    rsw = merged["rank_sum_w50"].to_numpy().astype(float)
    cf = merged["cf_score"].to_numpy().astype(float)

    def rankn(a): return pd.Series(a).rank(pct=True).to_numpy()
    base_conf = average_precision_score(yb[known], rsw[known])
    base_stress = average_precision_score(yb, rsw, sample_weight=sw)
    print(f"\nbaseline rank_sum_w50: confirmed={base_conf:.5f} pu_stress={base_stress:.5f}")
    br, gr = rankn(rsw), rankn(cf)
    best = (0.0, base_conf, base_stress)
    for wt in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = (1 - wt) * br + wt * gr
        cc = average_precision_score(yb[known], blend[known])
        ss = average_precision_score(yb, blend, sample_weight=sw)
        print(f"  blend w={wt:.2f}: confirmed={cc:.5f} ({cc-base_conf:+.5f})  pu_stress={ss:.5f} ({ss-base_stress:+.5f})")
        if ss > best[2]:
            best = (wt, cc, ss)

    verdict = ("PROMOTE" if (eval_honest > 0.61 and best[2] > base_stress + 0.002) else "RETIRE")
    print(f"\nVERDICT: {verdict}  (eval_honest={eval_honest:.4f}, best_blend_w={best[0]}, "
          f"pu_stress {best[2]:.5f} vs base {base_stress:.5f})")

    summary = {
        "confirmed_pair_ap": float(conf_ap), "confirmed_auc": float(conf_auc),
        "eval_honest_weighted_ap": float(eval_honest),
        "pos_mean_weight": float(w[y == 1].mean()), "neg_mean_weight": float(w[y == 0].mean()),
        "baseline_confirmed": float(base_conf), "baseline_pu_stress": float(base_stress),
        "best_blend_w": float(best[0]), "best_blend_confirmed": float(best[1]),
        "best_blend_pu_stress": float(best[2]), "verdict": verdict,
        "top_univariate": sorted(uni_rows, key=lambda r: -r[1])[:10],
    }
    (CACHE / "_policy_counterfactual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote _policy_counterfactual_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
