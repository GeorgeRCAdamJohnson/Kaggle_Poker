"""DECISION-EQUITY block done RIGHT — the named differentiator of the LB 0.82412 approach.

My crude EV-loss (§140/§141) failed because it used rank-buckets x to_call. The REAL decision-equity
uses the per-action raw fields we have: pot_before, to_call, stack_before, amount_to + the EXACT
7-card equity at that street. At each postflop decision facing a bet:
  pot_odds = to_call / (pot_before + to_call)                      # equity needed to call profitably
  equity   = exact hand strength -> P(best) proxy among live players at that street (from 7-card eval)
  A -EV CALL   = called with equity < pot_odds        -> ev_mistake = (pot_odds - equity)*(pot_before+to_call)  [chips knowingly given up]
  A -EV FOLD   = folded with equity > pot_odds + margin-> ev_surrender = (equity - pot_odds)*pot_before          [equity surrendered]
  A SLOWPLAY   = checked/called with very high equity (>=0.85) not raising -> passive with the nuts
The MAGNITUDE in CHIPS (not a 0/1 flag, not to_call alone) is the signal. A colluder makes these
mistakes DIRECTED at the partner; a bad player makes them randomly -> the pair-level (partner-vs-field)
and per-hand (evidence) aggregation separates them.

Caches per-(hand,player): ev_mistake_call, ev_surrender_fold, slowplay, + counts, both phases.
Uses exact hand rank among live players at each street (from player_strength cache rk_flop/turn/river).
Run:  python -m anchor_repro.decision_equity
"""
from __future__ import annotations
import time, numpy as np, polars as pl
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
PS=STEP3/"player_strength.parquet"
OUT=EDGE/"decision_equity.parquet"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    # rank among the players DEALT that hand (rk 1..k). Convert to an equity proxy vs live field.
    strength=pl.scan_parquet(PS).select(["hand_id","player_id","rk_flop","rk_turn","rk_river"])
    acts=(pl.scan_parquet(DATA/"actions.parquet")
          .filter(pl.col("street")!="preflop")
          .join(strength,on=["hand_id","player_id"],how="left")
          .with_columns(pl.when(pl.col("street")=="flop").then(pl.col("rk_flop"))
                        .when(pl.col("street")=="turn").then(pl.col("rk_turn"))
                        .otherwise(pl.col("rk_river")).alias("rk"))
          .with_columns(
              pl.col("players_active").clip(2,6).cast(pl.Float64).alias("k"),
              (pl.col("to_call").cast(pl.Float64)/(pl.col("pot_before").cast(pl.Float64)+pl.col("to_call").cast(pl.Float64)+1e-6)).clip(0,1).alias("pot_odds"),
          )
          .with_columns(
              # equity proxy: rank 1 of k -> ~1.0, rank k -> ~0. Continuous, from EXACT 7-card rank.
              pl.when(pl.col("rk").is_null()).then(0.5)
                .otherwise(((pl.col("k")-pl.col("rk").cast(pl.Float64))/(pl.col("k")-1.0+1e-6))).clip(0,1).alias("equity"),
          )
          .with_columns(
              # -EV CALL: called facing a bet with equity below pot odds, magnitude in CHIPS (bb)
              pl.when((pl.col("action")=="call")&(pl.col("to_call")>0)&(pl.col("equity")<pl.col("pot_odds")))
                .then((pl.col("pot_odds")-pl.col("equity"))*(pl.col("pot_before")+pl.col("to_call")))
                .otherwise(0.0).alias("ev_mistake_call"),
              # -EV FOLD: folded facing a bet holding equity ABOVE pot odds (surrendered a +EV spot)
              pl.when((pl.col("action")=="fold")&(pl.col("to_call")>0)&(pl.col("equity")>pl.col("pot_odds")+0.05))
                .then((pl.col("equity")-pl.col("pot_odds"))*pl.col("pot_before"))
                .otherwise(0.0).alias("ev_surrender_fold"),
              # SLOWPLAY: passive (check/call) with near-nut equity instead of raising
              pl.when((pl.col("action").is_in(["check","call"]))&(pl.col("equity")>=0.85))
                .then(pl.col("equity")-0.85).otherwise(0.0).alias("slowplay"),
          ))
    # normalize chip magnitudes by big_blind
    hands=pl.scan_parquet(DATA/"hands.parquet").select(["hand_id","big_blind","phase"])
    acts=acts.join(hands,on="hand_id",how="left").with_columns(
        (pl.col("ev_mistake_call")/pl.col("big_blind")).alias("ev_mistake_call_bb"),
        (pl.col("ev_surrender_fold")/pl.col("big_blind")).alias("ev_surrender_fold_bb"))
    per=acts.group_by(["hand_id","player_id"]).agg(
        pl.first("phase").alias("phase"),
        pl.col("ev_mistake_call_bb").sum().alias("ev_mistake_call"),
        pl.col("ev_surrender_fold_bb").sum().alias("ev_surrender_fold"),
        pl.col("slowplay").sum().alias("slowplay"),
        ((pl.col("ev_mistake_call")>0)|(pl.col("ev_surrender_fold")>0)).sum().alias("ev_mistake_ct"),
    ).with_columns((pl.col("ev_mistake_call")+pl.col("ev_surrender_fold")).alias("ev_mistake_total")).collect(engine="streaming")
    per.write_parquet(OUT)
    log(f"decision-equity per-hand {per.height:,}; mean total={per['ev_mistake_total'].mean():.3f} "
        f"(call {per['ev_mistake_call'].mean():.3f} fold {per['ev_surrender_fold'].mean():.3f} slow {per['slowplay'].mean():.3f})")
    print("phases:", per.group_by("phase").len().to_dict(as_series=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
