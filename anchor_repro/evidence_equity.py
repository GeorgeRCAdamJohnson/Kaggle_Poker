"""EVIDENCE-SIDE equity features (the last clean lever, §174 plan #1).

Equity is PHASE-FRAGILE as a PAIR-LEVEL risk signal (drifts dev->eval, §172 adversarial 0.716). But the
evidence sub-metric is a WITHIN-PAIR ranking of a confirmed pair's own hands -> the equity NEVER crosses
the dev/eval boundary, so it CANNOT drift here. We use equity to rank WHICH hand is the smoking gun, not
to score the pair. No gate needed (within-pair == phase-immune, proven 1:1 local==LB).

Per (pair_id, hand_id) we emit a compact equity-evidence block:
  eq_m1_pf, eq_m2_pf     : preflop equity of each member
  eq_max_pf, eq_min_pf   : max / min member equity preflop
  eq_gap_pf              : |m1-m2| preflop equity gap
  eq_surr_hand           : equity SURRENDERED (high-eq member folded/no-aggr AND partner benefited) -- the
                           directed_transfer / soft_play smoking gun (exact §170 surrender logic, per-hand)
  eq_checkdown_hi        : both high-eq, both to showdown, neither aggressed          -- soft_play
  eq_winner_was_ahead    : the pair member who won net had the higher preflop equity (legit win = LOW signal;
                           winning WITHOUT the equity edge = suspicious)
  eq_loser_gave_edge     : the losing member had >= the winning member's equity (gave up the better hand)

These are exactly the hands the host marked as evidence for transfer/soft_play (a strong hand folded to the
partner). Joined onto the per-hand evidence frame by (pair_id, hand_id).

Run standalone (dev, vs is_evidence): python -m anchor_repro.evidence_equity
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)
T=0.65  # high-equity threshold (same as pair-level surrender)

EQEV_FEATS=["eq_m1_pf","eq_m2_pf","eq_max_pf","eq_min_pf","eq_gap_pf",
            "eq_surr_hand","eq_checkdown_hi","eq_winner_was_ahead","eq_loser_gave_edge"]

def build_perhand(phase, eq_path, hf_glob, keep_pairs=None):
    """Return a polars frame keyed (pair_id, hand_id) with EQEV_FEATS. Phase-immune within-pair evidence."""
    eq=pl.read_parquet(eq_path)  # pair_id,hand_id,member,street,equity,n_live
    # preflop equity per member (street 0), wide
    pf=(eq.filter(pl.col("street")==0)
          .pivot(values="equity",index=["pair_id","hand_id"],on="member",aggregate_function="first")
          .rename({"1":"eq_m1_pf","2":"eq_m2_pf"}))
    HF=sorted(STEP3.glob(hf_glob))
    cols=["pair_id","hand_id",
          "p1_folded","p2_folded","p1_won_share","p2_won_share","p1_net_bb","p2_net_bb",
          "p1_n_aggr","p2_n_aggr","p1_went_to_showdown","p2_went_to_showdown"]
    hf=pl.concat([pl.scan_parquet(f).select(cols).collect() for f in HF])
    if keep_pairs is not None:
        hf=hf.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    d=hf.join(pf,on=["pair_id","hand_id"],how="left")
    d=d.with_columns(
        pl.col("eq_m1_pf").fill_null(0.0), pl.col("eq_m2_pf").fill_null(0.0),
    )
    d=d.with_columns(
        pl.max_horizontal("eq_m1_pf","eq_m2_pf").alias("eq_max_pf"),
        pl.min_horizontal("eq_m1_pf","eq_m2_pf").alias("eq_min_pf"),
        (pl.col("eq_m1_pf")-pl.col("eq_m2_pf")).abs().alias("eq_gap_pf"),
        (pl.col("p1_won_share")>0).alias("p1_won"), (pl.col("p2_won_share")>0).alias("p2_won"),
        (pl.col("eq_m1_pf")>=T).alias("m1_hi"), (pl.col("eq_m2_pf")>=T).alias("m2_hi"),
    )
    d=d.with_columns(
        # equity surrendered: high-eq member folded/no-aggr AND partner won AND this member lost net
        (pl.when(pl.col("m1_hi") & ((pl.col("p1_folded"))|(pl.col("p1_n_aggr")==0)) & pl.col("p2_won") & (pl.col("p1_net_bb")<0)).then(pl.col("eq_m1_pf")).otherwise(0.0)
         + pl.when(pl.col("m2_hi") & ((pl.col("p2_folded"))|(pl.col("p2_n_aggr")==0)) & pl.col("p1_won") & (pl.col("p2_net_bb")<0)).then(pl.col("eq_m2_pf")).otherwise(0.0)).alias("eq_surr_hand"),
        # both high-eq, both to showdown, neither aggressed
        (pl.col("m1_hi") & pl.col("m2_hi") & pl.col("p1_went_to_showdown") & pl.col("p2_went_to_showdown") & (pl.col("p1_n_aggr")==0) & (pl.col("p2_n_aggr")==0)).cast(pl.Float32).alias("eq_checkdown_hi"),
        # winner net was the member with the higher preflop equity? (legit) -> low signal; else suspicious
        (((pl.col("p1_net_bb")>pl.col("p2_net_bb")) & (pl.col("eq_m1_pf")>=pl.col("eq_m2_pf")))
         | ((pl.col("p2_net_bb")>pl.col("p1_net_bb")) & (pl.col("eq_m2_pf")>=pl.col("eq_m1_pf")))).cast(pl.Float32).alias("eq_winner_was_ahead"),
        # the losing member had >= the winning member's equity (surrendered the better hand)
        (((pl.col("p1_net_bb")<pl.col("p2_net_bb")) & (pl.col("eq_m1_pf")>=pl.col("eq_m2_pf")))
         | ((pl.col("p2_net_bb")<pl.col("p1_net_bb")) & (pl.col("eq_m2_pf")>=pl.col("eq_m1_pf")))).cast(pl.Float32).alias("eq_loser_gave_edge"),
    )
    return d.select(["pair_id","hand_id"]+EQEV_FEATS)


def main():
    """Standalone check: how well does each equity-evidence feature (alone) rank the true evidence hand
    WITHIN each positive pair (host MAP@5-style single-feature ranking)?"""
    from anchor_repro.seq_evidence import build_dh, map5, TARGET_BEHAVIORS
    dh=build_dh()
    keep=set(dh["pair_id"].to_list())
    per=build_perhand("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",keep_pairs=keep)
    dh=dh.join(per,on=["pair_id","hand_id"],how="left").to_pandas()
    for c in EQEV_FEATS: dh[c]=dh[c].fillna(0.0).astype("float32")
    cov=(dh["eq_max_pf"]>0).mean()
    log(f"labelled hands {len(dh):,}, equity coverage {cov:.3f}")
    pos=dh[dh["label"]==1].copy()
    log("=== per-hand equity-evidence: single-feature within-pair MAP@5 (evidence base ~0.53-0.60) ===")
    import numpy as np
    for c in EQEV_FEATS:
        m=map5(pos, score_col=c)
        # per family too
        fam={fm:map5(pos[pos["behavior_family"]==fm],score_col=c) for fm in TARGET_BEHAVIORS}
        log(f"  {m:.4f}  {c:22s} fam={{ {', '.join(f'{k[:4]}:{v:.3f}' for k,v in fam.items())} }}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
