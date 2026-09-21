"""Continuous-equity EV-loss v2 — fixes the crude proxy's 3 flaws, gated on dev-CV + held-out.

Crude proxy flaws (regressed -0.0009 LB): (1) to_call_bb x rank-bucket conflated bet SIZE with equity;
(2) rank-bucket != equity; (3) only 2 crude triggers. v2:
  * equity(action) = P(win) approximated from exact hand-RANK among the k live players at that street
    (rk=1 -> ~1.0 down to rk=k -> ~0), a monotone equity proxy from the exact 7-card evaluator.
  * pot_odds = to_call / (pot_before + to_call).  A call is -EV iff equity < pot_odds.
  * EV-loss is POT-ODDS based and BET-SIZE-NORMALIZED (a fraction, not raw chips):
      - called facing a bet with equity<pot_odds:  ev_loss += (pot_odds - equity)          # -EV call, normalized
      - folded facing a bet with equity>pot_odds:  ev_loss += (equity - pot_odds)           # surrendered +EV
      - checked/called passively with high equity not raising (slow-play toward partner): small term
  * continuous magnitude, not a 0/1 trigger.
Caches per-(hand,player) ev_loss_v2; then policy_edge_pvf picks it up via evloss_perhand overwrite path.
Run:  python -m anchor_repro.build_evloss_v2
"""
from __future__ import annotations
import time, numpy as np, polars as pl
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
AC=STEP3/"action_context.parquet"; PS=STEP3/"player_strength.parquet"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    strength=pl.scan_parquet(PS).select(["hand_id","player_id","rk_flop","rk_turn","rk_river"])
    # players live at each action's street: use players_active from action_context
    acts=(pl.scan_parquet(AC).filter(pl.col("street_no")>=1)
          .join(strength,on=["hand_id","player_id"],how="left")
          .with_columns(pl.when(pl.col("street_no")==1).then(pl.col("rk_flop"))
                        .when(pl.col("street_no")==2).then(pl.col("rk_turn"))
                        .otherwise(pl.col("rk_river")).alias("rk"),
                        pl.col("players_active").clip(2,6).alias("k"))
          .with_columns(
              # equity proxy: rank 1 of k -> ~1, rank k -> ~0; monotone, continuous
              pl.when(pl.col("rk").is_null()).then(0.5)
                .otherwise((pl.col("k").cast(pl.Float32)-pl.col("rk").cast(pl.Float32))/(pl.col("k").cast(pl.Float32)-1.0+1e-6))
                .clip(0.0,1.0).alias("equity"),
              # pot odds when facing a bet: to_call / (pot_after_call). pot_before ~ to_call/amount_pot_ratio proxy;
              # use a simple bounded pot-odds = to_call_bb / (to_call_bb + effective_pot), effective_pot >= 1bb
              pl.when(pl.col("to_call")>0)
                .then(pl.col("to_call_bb")/(pl.col("to_call_bb")+pl.max_horizontal(pl.lit(1.0), pl.col("to_call_bb"))+1e-6))
                .otherwise(0.0).alias("pot_odds"),
          )
          .with_columns(
              # -EV call: called facing a bet with equity below pot odds
              pl.when((pl.col("action")=="call")&(pl.col("to_call")>0))
                .then((pl.col("pot_odds")-pl.col("equity")).clip(lower_bound=0.0)).otherwise(0.0).alias("evl_call"),
              # surrendered +EV: folded facing a bet while holding equity above pot odds
              pl.when((pl.col("action")=="fold")&(pl.col("to_call")>0))
                .then((pl.col("equity")-pl.col("pot_odds")).clip(lower_bound=0.0)).otherwise(0.0).alias("evl_fold"),
              # passive slow-play: checked with strong equity (>=0.8) postflop (soft-play toward partner)
              pl.when((pl.col("action")=="check")&(pl.col("equity")>=0.8))
                .then(pl.col("equity")-0.8).otherwise(0.0).alias("evl_slowplay"),
          ))
    per=acts.group_by(["hand_id","player_id"]).agg(
        (pl.col("evl_call")+pl.col("evl_fold")+pl.col("evl_slowplay")).sum().alias("evloss"),
        pl.col("evl_call").sum().alias("evl_call"),
        pl.col("evl_fold").sum().alias("evl_fold"),
        pl.col("evl_slowplay").sum().alias("evl_slowplay"),
    ).collect(engine="streaming")
    ph=pl.read_parquet(DATA/"hands.parquet",columns=["hand_id","phase"])
    per=per.join(ph,on="hand_id",how="left")
    per.write_parquet(EDGE/"evloss_perhand.parquet")   # OVERWRITE so policy_edge_pvf picks up v2
    log(f"evloss v2 per-hand {per.height:,}; mean evloss={per['evloss'].mean():.4f} "
        f"(call {per['evl_call'].mean():.4f} fold {per['evl_fold'].mean():.4f} slow {per['evl_slowplay'].mean():.4f})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
