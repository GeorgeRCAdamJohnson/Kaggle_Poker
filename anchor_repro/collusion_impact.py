"""PER-HAND COLLUSION IMPACT — the AAAI collusion-table paradigm (§186), the host's actual evidence-selection
axis. Approximates the value-function counterfactual impact Cg(j,k) at STREET granularity using the GPU
per-hand equity engine as the value function Vi(h) ~= equity_i(street) * pot.

The host labels evidence by IMPACT (Δexpected-outcome vs a non-collusive baseline), NOT collusiveness of
actions (§184) and NOT magnitude (§187, magnitude rejected). Per-hand collusion-impact features (within-pair,
phase-immune, §185 set-metric):
  ci_ev_held_lost    : member's PEAK equity*pot (best expected value in the hand) that they did NOT realize
                       (net<0) while the PARTNER won -> joint EV the pair kept in-pair by NOT contesting
  ci_ev_surrender    : Σ_street (equity_member*pot) over streets where member had equity>partner yet ended
                       net-negative to partner's gain -> surrendered expected value, magnitude-weighted
  ci_marginal        : directed impact = (value the pair captured jointly) minus (equity-implied fair value)
                       = pair realized net  -  pair expected net (from peak equities) -> EXCESS over fair play
  ci_impact_ratio    : ci_marginal normalized by pot (impact per chip at risk) — small pots CAN score high
  ci_peak_eq_gap     : how far ahead (by equity) the losing member was at their peak street (subtle dumps)
Key vs §175 equity-surrender: that was a THRESHOLD FLAG (count of >0.65-eq folds). This is a signed CONTINUOUS
counterfactual UTILITY DELTA (Δequity*pot), the actual paradigm. Different construction.
Keyed (pair_id, hand_id). Run standalone (dev, single-feature within-pair MAP@5):
  python -m anchor_repro.collusion_impact
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

CI_FEATS=["ci_ev_held_lost","ci_ev_surrender","ci_marginal","ci_impact_ratio","ci_peak_eq_gap"]

def build_impact(phase, eq_path, hf_glob, keep_pairs=None):
    eq=pl.read_parquet(eq_path)  # pair_id,hand_id,member,street,equity,n_live
    # peak equity per member across streets (best expected value point in the hand) + preflop equity
    peak=(eq.group_by(["pair_id","hand_id","member"]).agg(pl.col("equity").max().alias("peak_eq"))
            .pivot(values="peak_eq",index=["pair_id","hand_id"],on="member",aggregate_function="first")
            .rename({"1":"m1_peak","2":"m2_peak"}))
    HF=sorted(STEP3.glob(hf_glob))
    cols=["pair_id","hand_id","pot_bb",
          "p1_folded","p2_folded","p1_won_share","p2_won_share","p1_net_bb","p2_net_bb",
          "p1_n_aggr","p2_n_aggr"]
    hf=pl.concat([pl.scan_parquet(f).select(cols).collect() for f in HF])
    if keep_pairs is not None: hf=hf.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    d=hf.join(peak,on=["pair_id","hand_id"],how="left").with_columns(
        pl.col("m1_peak").fill_null(0.0), pl.col("m2_peak").fill_null(0.0), pl.col("pot_bb").fill_null(0.0))
    d=d.with_columns(
        (pl.col("p1_won_share")>0).alias("p1_won"), (pl.col("p2_won_share")>0).alias("p2_won"),
        # expected value at peak = peak_eq * pot (chips the member "should" expect at their best point)
        (pl.col("m1_peak")*pl.col("pot_bb")).alias("m1_ev"), (pl.col("m2_peak")*pl.col("pot_bb")).alias("m2_ev"),
    )
    d=d.with_columns(
        # EV held-but-lost: member had high peak EV, ended net<0, partner won -> joint EV kept in-pair
        (pl.when((pl.col("p1_net_bb")<0)&pl.col("p2_won")).then(pl.col("m1_ev")).otherwise(0.0)
         + pl.when((pl.col("p2_net_bb")<0)&pl.col("p1_won")).then(pl.col("m2_ev")).otherwise(0.0)).alias("ci_ev_held_lost"),
        # EV surrender: peak-EV of the member who was AHEAD by equity but ended net-negative to partner
        (pl.when((pl.col("m1_peak")>pl.col("m2_peak"))&(pl.col("p1_net_bb")<0)&pl.col("p2_won")).then(pl.col("m1_ev")).otherwise(0.0)
         + pl.when((pl.col("m2_peak")>pl.col("m1_peak"))&(pl.col("p2_net_bb")<0)&pl.col("p1_won")).then(pl.col("m2_ev")).otherwise(0.0)).alias("ci_ev_surrender"),
        # marginal impact: pair realized net MINUS pair equity-expected net = EXCESS over fair play
        ((pl.col("p1_net_bb")+pl.col("p2_net_bb")) - (pl.col("m1_ev")+pl.col("m2_ev")-pl.col("pot_bb"))).alias("ci_marginal_raw"),
        # peak equity gap of the loser (how far ahead the surrendering member was) — subtle dumps score high
        (pl.when(pl.col("p1_net_bb")<pl.col("p2_net_bb")).then(pl.col("m1_peak")-pl.col("m2_peak"))
          .otherwise(pl.col("m2_peak")-pl.col("m1_peak"))).alias("ci_peak_eq_gap"),
    )
    d=d.with_columns(
        pl.col("ci_marginal_raw").abs().alias("ci_marginal"),
        (pl.col("ci_marginal_raw").abs()/(pl.col("pot_bb")+1.0)).alias("ci_impact_ratio"),
    )
    return d.select(["pair_id","hand_id"]+CI_FEATS)

CI_PAIR_FEATS=["cip_marginal_sum","cip_marginal_mean","cip_marginal_p95","cip_marginal_max",
               "cip_surrender_sum","cip_surrender_mean","cip_impact_ratio_mean","cip_peak_gap_mean",
               "cip_peak_gap_p95","cip_held_lost_sum","cip_pos_rate"]

def build_pair_impact(phase, eq_path, hf_glob, keep_pairs=None):
    """PAIR-LEVEL aggregate of per-hand collusion-impact = the AAAI collusion-table pair score (§189).
    Outcome-relative-to-fair-play => phase-robust RISK feature (does this pair collude?). Keyed pair_id."""
    d=build_impact(phase,eq_path,hf_glob,keep_pairs=keep_pairs)  # per-hand ci_*
    agg=(d.group_by("pair_id").agg(
        pl.col("ci_marginal").sum().alias("cip_marginal_sum"),
        pl.col("ci_marginal").mean().alias("cip_marginal_mean"),
        pl.col("ci_marginal").quantile(0.95).alias("cip_marginal_p95"),
        pl.col("ci_marginal").max().alias("cip_marginal_max"),
        pl.col("ci_ev_surrender").sum().alias("cip_surrender_sum"),
        pl.col("ci_ev_surrender").mean().alias("cip_surrender_mean"),
        pl.col("ci_impact_ratio").mean().alias("cip_impact_ratio_mean"),
        pl.col("ci_peak_eq_gap").mean().alias("cip_peak_gap_mean"),
        pl.col("ci_peak_eq_gap").quantile(0.95).alias("cip_peak_gap_p95"),
        pl.col("ci_ev_held_lost").sum().alias("cip_held_lost_sum"),
        (pl.col("ci_ev_surrender")>0).mean().alias("cip_pos_rate"),
    ))
    return agg

def main():
    from anchor_repro.seq_evidence import build_dh, map5, TARGET_BEHAVIORS
    dh=build_dh(); keep=set(dh["pair_id"].to_list())
    ci=build_impact("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=keep)
    dh=dh.join(ci,on=["pair_id","hand_id"],how="left").to_pandas()
    for c in CI_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    pos=dh[dh["label"]==1].copy()
    log(f"labelled hands {len(dh):,}; coverage {(dh['ci_marginal']!=0).mean():.3f}")
    log("=== per-hand collusion-impact: single-feature within-pair MAP@5 (magnitude rules maxed 0.18) ===")
    for c in CI_FEATS:
        m=map5(pos.assign(hand_score=pos[c]))
        fam={fm:map5(pos[pos['behavior_family']==fm].assign(hand_score=lambda d:d[c])) for fm in TARGET_BEHAVIORS}
        log(f"  {c:18s} MAP@5 {m:.4f}  fam={{ {', '.join(f'{k[:4]}:{v:.3f}' for k,v in fam.items())} }}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
