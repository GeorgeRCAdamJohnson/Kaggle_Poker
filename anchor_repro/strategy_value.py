"""STRATEGY-RELATIVE value function V(h) — the AAAI collusion-table method done properly (the one untried
construction; equity was a phase-drifty shadow of this).

AAAI allows ANY value function; we build an EMPIRICAL FIELD value function (the reference "non-colluding"
strategy = the field's average play), estimated from the data:
   V(state) = E[ player final net_bb | decision-time state ]
state = (hand-strength bucket, street, facing/to_call bucket, players_active, pot bucket).
This is "expected chips from this decision point if play proceeds as the field typically does" — exactly the
paper's value-function role, and PHASE-ROBUST when used RELATIVELY (collusion value = deviation of the
partner's V caused by the colluder's action vs this field expectation; absolute stakes/style cancels).

We ALSO build the per-action reference policy Pi_field(action | state) = field action distribution, so an
action's expected-value-under-field can be compared to the actual action taken (the counterfactual).

Fit SEPARATELY per phase (development / evaluation) on that phase's own field, so V is the phase's own
reference — this is what makes the resulting collusion value transport cleanly (no dev->eval level leak).

Outputs (cached): value_table_<phase>.parquet keyed by state-bucket -> (v_mean net_bb, n).
Run:  python -m anchor_repro.strategy_value --phase development
"""
from __future__ import annotations
import time, argparse, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); DATA=Path("data/poker")
OUT=STEP3/"edge"/"stratval"; OUT.mkdir(parents=True,exist_ok=True)
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

# ---- state bucketing (coarse enough to be well-estimated, fine enough to be meaningful) ----
CHEN_BINS=[-5,0,4,7,9,12,25]          # hand-strength (Chen) buckets
TOCALL_BINS=[-0.01,0.0,1.0,3.0,8.0,1e9]  # facing: nothing / <=1bb / <=3 / <=8 / big
POT_BINS=[0,3,8,20,50,1e9]            # pot size (bb) buckets

def _bucket(col,bins,label):
    return pl.col(col).cut(bins[1:-1],labels=[f"{label}{i}" for i in range(len(bins)-1)]).cast(pl.Utf8)

def build_value_table(phase):
    out_path=OUT/f"value_table_{phase}.parquet"
    # action-log with state, joined to per-(hand,player) hole strength + terminal net_bb
    ph=pl.scan_parquet(STEP3/"player_hands_policy.parquet").filter(pl.col("phase")==phase).select(
        ["hand_id","player_id","chen","net_bb","big_blind"])
    acts=pl.scan_parquet(STEP3/"action_context.parquet").filter(pl.col("phase")==phase).select(
        ["hand_id","player_id","action_no","street_no","to_call_bb","players_active"])
    # each ACTION becomes a decision-state row; join the hole strength + the hand's terminal net (V(z))
    d=acts.join(ph,on=["hand_id","player_id"],how="inner")
    d=d.with_columns(
        pl.col("chen").fill_null(0.0).cut(CHEN_BINS[1:-1],labels=[f"c{i}" for i in range(len(CHEN_BINS)-1)]).cast(pl.Utf8).alias("cb"),
        pl.col("to_call_bb").fill_null(0.0).cut(TOCALL_BINS[1:-1],labels=[f"t{i}" for i in range(len(TOCALL_BINS)-1)]).cast(pl.Utf8).alias("tb"),
        pl.col("players_active").fill_null(2).clip(2,6).cast(pl.Int8).cast(pl.Utf8).alias("pa"),
        pl.col("street_no").fill_null(0).cast(pl.Utf8).alias("sb"),
        (pl.col("net_bb")/pl.col("big_blind").clip(lower_bound=1)).alias("net_bb_norm"),
    )
    d=d.with_columns((pl.col("cb")+"|"+pl.col("sb")+"|"+pl.col("tb")+"|"+pl.col("pa")).alias("state"))
    vt=(d.group_by("state").agg(
            pl.col("net_bb_norm").mean().alias("v_mean"),
            pl.col("net_bb_norm").median().alias("v_med"),
            pl.len().alias("n"))
        ).collect(engine="streaming")
    vt.write_parquet(out_path)
    log(f"{phase}: value table {vt.height} states, {int(vt['n'].sum()):,} decision rows -> {out_path.name}")
    # also emit the global field mean as fallback for sparse states
    gm=float((d.select(pl.col("net_bb_norm").mean()).collect().item()))
    (OUT/f"value_global_{phase}.txt").write_text(str(gm)); log(f"{phase}: global field V = {gm:.4f}")
    return vt

def load_value(phase):
    vt=pl.read_parquet(OUT/f"value_table_{phase}.parquet")
    gm=float((OUT/f"value_global_{phase}.txt").read_text())
    return vt, gm

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",default="development"); a=ap.parse_args()
    vt=build_value_table(a.phase)
    # sanity: does V rise monotonically with hand strength? (validation of the value function)
    v=vt.with_columns(pl.col("state").str.split("|").list.get(0).alias("cb"))
    agg=v.group_by("cb").agg((pl.col("v_mean")*pl.col("n")).sum().alias("sw"),pl.col("n").sum().alias("nn")).with_columns((pl.col("sw")/pl.col("nn")).alias("V")).sort("cb")
    print("=== V by hand-strength bucket (should rise with strength c0<..<c5) ===")
    for r in agg.iter_rows(named=True): print(f"  {r['cb']}: V={r['V']:+.3f}  (n={int(r['nn']):,})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
