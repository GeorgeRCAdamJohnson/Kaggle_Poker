"""Decorrelation check for the per-action EV-LOSS channel BEFORE building it fully.

User caught: hosen42 ALREADY has outcome-side equity (rk_final, loser_beats_winner_final, fold_better_
hand). §32 also did made-hand-surrendered. The ONLY new thing is DECISION-side EV-loss: chips of equity
each postflop action gave up vs the policy-optimal, directed at the partner. Before investing in the
full continuous-equity EV computation, build a CHEAP proxy of per-action EV-loss and check it is
DECORRELATED from their existing equity-outcome pair features. If corr is high, it's §32 again -> skip.
If low, it's genuinely new signal worth the full build.

Proxy EV-loss per postflop action:
  - called/checked facing a bet while holding a WEAK made-hand rank (rk >= 4 of 6) -> gave up chips
    (call_amount) with poor equity  => ev_loss += to_call_bb (proxy for the -EV of the call)
  - folded while holding a STRONG rank (rk <= 2) facing a small bet -> surrendered equity => ev_loss += pot proxy
Aggregate to pair (partner-vs-field normalized like the winning channels), correlate vs their
loser_beats_winner_final_mean / fold_better_hand_mean / cat_gap_final_mean pair features + base OOF.
Run:  python -m anchor_repro.evloss_decorr_check
"""
from __future__ import annotations
import time, numpy as np, pandas as pd, polars as pl
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
AC=STEP3/"action_context.parquet"; PS=STEP3/"player_strength.parquet"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def build_evloss_perhand() -> pl.DataFrame:
    strength=pl.scan_parquet(PS).select(["hand_id","player_id","rk_flop","rk_turn","rk_river"])
    acts=(pl.scan_parquet(AC).filter(pl.col("street_no")>=1)
          .join(strength,on=["hand_id","player_id"],how="left")
          .with_columns(pl.when(pl.col("street_no")==1).then(pl.col("rk_flop"))
                        .when(pl.col("street_no")==2).then(pl.col("rk_turn"))
                        .otherwise(pl.col("rk_river")).fill_null(6).clip(1,6).alias("rk"))
          .with_columns(
              # -EV call/check with a weak hand facing a bet: gave up ~to_call chips of -EV
              (((pl.col("action").is_in(["call"])) & (pl.col("to_call")>0) & (pl.col("rk")>=4))
               .cast(pl.Float32) * pl.col("to_call_bb").fill_null(0)).alias("evloss_call_weak"),
              # folded a strong hand (rk<=2) facing a bet: surrendered equity ~ amount already invested proxy
              (((pl.col("action")=="fold") & (pl.col("to_call")>0) & (pl.col("rk")<=2))
               .cast(pl.Float32) * pl.col("to_call_bb").fill_null(0)).alias("evloss_fold_strong"),
          ))
    return acts.group_by(["hand_id","player_id"]).agg(
        pl.col("evloss_call_weak").sum().alias("evloss_call_weak"),
        pl.col("evloss_fold_strong").sum().alias("evloss_fold_strong"),
    ).collect(engine="streaming")

def main():
    log("building proxy EV-loss per hand...")
    el=build_evloss_perhand()
    el=el.with_columns((pl.col("evloss_call_weak")+pl.col("evloss_fold_strong")).alias("evloss"))
    log(f"evloss per-hand {el.height:,}; mean evloss={el['evloss'].mean():.3f}")
    # aggregate to dev pairs (sum over shared hands, both members)
    ph=pl.scan_parquet(STEP3/"dev_pair_hands.parquet").select(["pair_id","hand_id","player_1","player_2"])
    a=ph.join(el.lazy(),left_on=["hand_id","player_1"],right_on=["hand_id","player_id"],how="left").rename({"evloss":"a_el"})
    j=(a.select(["pair_id","hand_id","a_el"]).join(
        ph.join(el.lazy(),left_on=["hand_id","player_2"],right_on=["hand_id","player_id"],how="left").select(["pair_id","hand_id",pl.col("evloss").alias("b_el")]),
        on=["pair_id","hand_id"]))
    pair_el=(j.with_columns((pl.col("a_el").fill_null(0)+pl.col("b_el").fill_null(0)).alias("h_el"))
             .group_by("pair_id").agg(pl.col("h_el").mean().alias("evloss_mean"),
                                      pl.col("h_el").top_k(3).mean().alias("evloss_top3"),
                                      pl.col("h_el").sum().alias("evloss_sum")).collect())
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet").merge(pair_el.to_pandas(),on="pair_id",how="left")
    for c in ["evloss_mean","evloss_top3","evloss_sum"]: dev[c]=dev[c].fillna(0)
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    oof_base=np.load(EDGE/"oof_base.npy")

    print("\n=== EV-loss proxy: univariate AUC + DECORRELATION vs hosen42 equity-outcome feats ===")
    equity_feats=[c for c in dev.columns if c in
                  ("loser_beats_winner_final_mean","fold_better_hand_mean","cat_gap_final_mean","loser_best_final_mean","winner_best_final_mean")]
    for c in ["evloss_mean","evloss_top3","evloss_sum"]:
        v=dev[c].to_numpy()
        a_all=roc_auc_score(y,v); a_all=max(a_all,1-a_all)
        corr_base=spearmanr(v,oof_base).correlation
        corrs={ef: round(spearmanr(v,dev[ef].fillna(0).to_numpy()).correlation,3) for ef in equity_feats}
        print(f"  {c:14} AUC={a_all:.4f}  corr(base_oof)={corr_base:.3f}  corr(equity feats)={corrs}")
    maxcorr=max(abs(spearmanr(dev['evloss_top3'].to_numpy(), dev[ef].fillna(0).to_numpy()).correlation) for ef in equity_feats)
    print(f"\n  VERDICT: max |corr| with existing equity feats = {maxcorr:.3f}  -> "
          f"{'REDUNDANT (skip, it is §32 again)' if maxcorr>0.5 else 'DECORRELATED -> worth the full continuous-equity build'}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
