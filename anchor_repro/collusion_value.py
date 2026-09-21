"""PER-ACTION COLLUSION VALUES from the strategy value function (§Task 3, AAAI collusion table).

For each action by member k in a shared hand, its V-delta = V(state_after k acts) - V(state_before) tells
how much k's action changed EXPECTED value under the field reference. AAAI C(j,k) = sum of k's action
V-deltas that accrue to partner j. Empirical attribution (data-grounded, no game-tree sim):
  - When k takes a value-CEDING action (fold / check / call-not-raise) from a HIGH-V state (strong hand,
    ahead) AND the partner is still live and ends up winning the pot -> k ceded expected value that the
    partner captured. The ceded amount = V(k's pre-action state) - V(k's post-action state), i.e. the drop
    in k's own expectation, DIRECTED to the partner when partner benefits.
  - MI (marginal / differential): the same ceding directed at the PARTNER minus the same ceding directed at
    the FIELD (when no partner in pot / partner not the beneficiary). This is the partner-vs-others contrast
    that separates intentional collusion from a player who just folds a lot (phase-robust, §AAAI).

Everything is a per-hand quantity; we emit per-shared-hand columns joined to dev/eval pair-hands, ready for
count-invariant RATE aggregation in Task 4 (mean per shared hand, no sums, no co-occurrence reference).
Run:  python -m anchor_repro.collusion_value --phase development
"""
from __future__ import annotations
import time, argparse, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from anchor_repro.strategy_value import load_value, CHEN_BINS, TOCALL_BINS
STEP3=Path("outputs/poker_collusion/hosen42_step3")
OUT=STEP3/"edge"/"stratval"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def _state_expr():
    return (pl.col("chen").fill_null(0.0).cut(CHEN_BINS[1:-1],labels=[f"c{i}" for i in range(len(CHEN_BINS)-1)]).cast(pl.Utf8)
            +"|"+pl.col("street_no").fill_null(0).cast(pl.Utf8)
            +"|"+pl.col("to_call_bb").fill_null(0.0).cut(TOCALL_BINS[1:-1],labels=[f"t{i}" for i in range(len(TOCALL_BINS)-1)]).cast(pl.Utf8)
            +"|"+pl.col("players_active").fill_null(2).clip(2,6).cast(pl.Int8).cast(pl.Utf8))

def build_collusion_value(phase, pair_hands_path):
    """Per (pair_id, hand_id): per-action ceded-value directed to partner vs field. Emits per-hand cols."""
    vt,gm=load_value(phase)
    vmap=dict(zip(vt["state"].to_list(),vt["v_mean"].to_list()))
    # ordered action log with state + hole strength; only the pair members' hands matter, but we need
    # the whole hand to know who won -> use player_hands_policy for outcomes.
    ph=pl.scan_parquet(STEP3/"player_hands_policy.parquet").filter(pl.col("phase")==phase).select(
        ["hand_id","player_id","chen","net_bb","big_blind","won_share","folded"])
    acts=pl.scan_parquet(STEP3/"action_context.parquet").filter(pl.col("phase")==phase).select(
        ["hand_id","player_id","action_no","street_no","to_call_bb","players_active","action","is_aggr"])
    d=acts.join(ph,on=["hand_id","player_id"],how="inner").with_columns(_state_expr().alias("state"))
    d=d.with_columns(pl.col("state").replace_strict(vmap,default=gm).alias("V"))
    # per (hand,player) sequence of V; ceded value on a passive action = drop from that player's PEAK V
    # to their terminal (0 if they won). Approx per-action: value-ceding = max(0, peak_V_before_pass - V_pass)
    seq=(d.sort(["hand_id","player_id","action_no"])
           .group_by(["hand_id","player_id"]).agg(
               pl.col("V").max().alias("peakV"),
               pl.col("V").last().alias("lastV"),
               (pl.col("action")=="fold").any().alias("did_fold"),
               (~pl.col("is_aggr")).sum().alias("n_passive"),
               pl.len().alias("n_act")))
    seq=seq.join(ph.select(["hand_id","player_id","net_bb","big_blind","won_share","folded"]),on=["hand_id","player_id"],how="left")
    # ceded value (per player, per hand): peak expectation they had minus what they realized, when they went passive/folded
    seq=seq.with_columns(
        (pl.col("net_bb")/pl.col("big_blind").clip(lower_bound=1)).alias("net_norm"),
    ).with_columns(
        # value ceded = how far below their PEAK expectation they ended, only counting when they gave up (folded/passive) a positive-V spot
        pl.max_horizontal(pl.lit(0.0), pl.col("peakV")-pl.col("net_norm")).alias("ceded_raw"),
        (pl.col("peakV")>0.3).alias("had_edge"),   # had a genuinely +EV spot
    ).with_columns(
        (pl.col("ceded_raw")*((pl.col("did_fold"))|(pl.col("n_passive")>0)).cast(pl.Float32)).alias("ceded"),
    ).collect(engine="streaming")

    # now map to pairs: for each pair's shared hand, member A and B ceded values + who won
    pairs=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2"])
    sm=seq.select(["hand_id","player_id","ceded","had_edge","won_share","folded","peakV"])
    a=pairs.join(sm.rename({c:f"a_{c}" for c in ["player_id","ceded","had_edge","won_share","folded","peakV"]}),
                 left_on=["hand_id","player_1"],right_on=["hand_id","a_player_id"],how="left")
    a=a.join(sm.rename({c:f"b_{c}" for c in ["player_id","ceded","had_edge","won_share","folded","peakV"]}),
             left_on=["hand_id","player_2"],right_on=["hand_id","b_player_id"],how="left")
    a=a.with_columns([pl.col(c).fill_null(0.0) for c in ["a_ceded","b_ceded","a_won_share","b_won_share","a_peakV","b_peakV"]])
    a=a.with_columns(
        (pl.col("a_won_share")>0).alias("a_won"),(pl.col("b_won_share")>0).alias("b_won"))
    # DIRECTED collusion value per hand: value A ceded that B captured (A gave up +EV spot AND B won) + symmetric
    a=a.with_columns(
        (pl.when(pl.col("b_won")).then(pl.col("a_ceded")).otherwise(0.0)
         + pl.when(pl.col("a_won")).then(pl.col("b_ceded")).otherwise(0.0)).alias("cv_directed"),
        # TI: joint value swing = both members' ceded that stayed in-pair (either won)
        (pl.when(pl.col("a_won")|pl.col("b_won")).then(pl.col("a_ceded")+pl.col("b_ceded")).otherwise(0.0)).alias("cv_ti"),
        # MI proxy: directed-to-partner ceding minus ceding that went to the field (neither won)
        (pl.when(pl.col("b_won")).then(pl.col("a_ceded")).otherwise(0.0)
         + pl.when(pl.col("a_won")).then(pl.col("b_ceded")).otherwise(0.0)
         - pl.when(~(pl.col("a_won")|pl.col("b_won"))).then(pl.col("a_ceded")+pl.col("b_ceded")).otherwise(0.0)).alias("cv_mi"),
        # flag: an edge was ceded to the partner this hand
        (((pl.col("a_had_edge"))&(pl.col("b_won"))) | ((pl.col("b_had_edge"))&(pl.col("a_won")))).cast(pl.Int8).alias("cv_edge_to_partner"),
    )
    out=a.select(["pair_id","hand_id","cv_directed","cv_ti","cv_mi","cv_edge_to_partner"])
    p=OUT/f"collusion_value_{phase}.parquet"; out.write_parquet(p)
    log(f"{phase}: collusion-value per-hand {out.height:,} rows -> {p.name}; "
        f"mean cv_directed {out['cv_directed'].mean():.4f} edge_to_partner rate {out['cv_edge_to_partner'].mean():.4f}")
    return out

CV_FEATS=["cvd_rate","cvti_rate","cvmi_rate","cvedge_rate"]

def build_cv_features(phase):
    """Count-invariant per-pair RATE features (mean per shared hand — no sums, no solo baseline, no
    co-occurrence reference). Reads the cached per-hand collusion_value_{phase}.parquet (build it first)."""
    p=OUT/f"collusion_value_{phase}.parquet"
    cv=pl.read_parquet(p)
    return (cv.group_by("pair_id").agg(
        pl.col("cv_directed").mean().alias("cvd_rate"),
        pl.col("cv_ti").mean().alias("cvti_rate"),
        pl.col("cv_mi").mean().alias("cvmi_rate"),
        pl.col("cv_edge_to_partner").mean().alias("cvedge_rate")))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",default="development"); a=ap.parse_args()
    ph={"development":STEP3/"dev_pair_hands.parquet","evaluation":STEP3/"eval_pair_hands.parquet"}[a.phase]
    build_collusion_value(a.phase, ph)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
