"""PHASE-ROBUST collusion-impact: FIELD-NORMALIZED TI & MI (§190 Corrections 1,5; §191 fix).

§191: raw pair-level ci_* cleared the value gate (+0.0059) but DRIFTED (adversarial 0.734) because
equity*pot carries absolute stakes/style level that varies dev->eval. FIX = the proven pvf mechanism:
normalize each member's impact against THEIR OWN SOLO BASELINE (impact level on hands WITHOUT the partner),
leaving only the pair-specific EXCESS. That's exactly the AAAI Marginal-Impact axis (partner-vs-others) and
exactly why the pvf channels are phase-robust. Absolute level (drift) cancels; pair excess (signal) remains.

Per member i, per phase: compute per-hand IMPACT CONTRIBUTION
   imp_i(hand) = peak_eq_i * pot          (member's expected-value stake in the hand — the value fn proxy)
   surr_i(hand)= peak_eq_i * pot * 1[i ahead by eq AND i net<0 AND partner won]   (surrendered EV)
Solo baseline = mean imp_i / surr_i over ALL i's hands in this phase (with & without partner).
MI feature per pair = (mean over SHARED hands) - (solo field baseline), shrunk by shared exposure — the
partner-vs-field EXCESS. TI feature = the pair's JOINT excess (both members' shared-excess summed).
Emits phase-robust CIMI_FEATS keyed pair_id.
Run:  python -m anchor_repro.collusion_impact_mi
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
SHRINK=25.0
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

CIMI_FEATS=["mi_surr_sum","mi_surr_min","mi_imp_sum","mi_imp_min","ti_joint_excess","mi_surr_max"]

def _perhand_member(phase, eq_path, hf_glob, keep_pairs):
    """Return a long frame: one row per (pair_id, hand_id, member_slot) with that member's player_id,
    imp (peak_eq*pot) and surr (surrendered EV this hand)."""
    eq=pl.read_parquet(eq_path)
    peak=(eq.group_by(["pair_id","hand_id","member"]).agg(pl.col("equity").max().alias("peak_eq"))
            .pivot(values="peak_eq",index=["pair_id","hand_id"],on="member",aggregate_function="first")
            .rename({"1":"m1_peak","2":"m2_peak"}))
    HF=sorted(STEP3.glob(hf_glob))
    cols=["pair_id","hand_id","pot_bb","player_1","player_2",
          "p1_won_share","p2_won_share","p1_net_bb","p2_net_bb"]
    hf=pl.concat([pl.scan_parquet(f).select(cols).collect() for f in HF])
    if keep_pairs is not None: hf=hf.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    d=hf.join(peak,on=["pair_id","hand_id"],how="left").with_columns(
        pl.col("m1_peak").fill_null(0.0),pl.col("m2_peak").fill_null(0.0),pl.col("pot_bb").fill_null(0.0),
        (pl.col("p1_won_share")>0).alias("p1_won"),(pl.col("p2_won_share")>0).alias("p2_won"))
    d=d.with_columns(
        (pl.col("m1_peak")*pl.col("pot_bb")).alias("m1_imp"),(pl.col("m2_peak")*pl.col("pot_bb")).alias("m2_imp"),
        (pl.when((pl.col("m1_peak")>pl.col("m2_peak"))&(pl.col("p1_net_bb")<0)&pl.col("p2_won")).then(pl.col("m1_peak")*pl.col("pot_bb")).otherwise(0.0)).alias("m1_surr"),
        (pl.when((pl.col("m2_peak")>pl.col("m1_peak"))&(pl.col("p2_net_bb")<0)&pl.col("p1_won")).then(pl.col("m2_peak")*pl.col("pot_bb")).otherwise(0.0)).alias("m2_surr"),
    )
    m1=d.select([pl.col("pair_id"),pl.col("hand_id"),pl.col("player_1").alias("player_id"),pl.col("m1_imp").alias("imp"),pl.col("m1_surr").alias("surr")])
    m2=d.select([pl.col("pair_id"),pl.col("hand_id"),pl.col("player_2").alias("player_id"),pl.col("m2_imp").alias("imp"),pl.col("m2_surr").alias("surr")])
    return pl.concat([m1,m2]), d

def build_mi(phase, eq_path, hf_glob, keep_pairs=None):
    long, d = _perhand_member(phase, eq_path, hf_glob, keep_pairs)
    # solo field baseline per player over ALL their hands in this phase (with & without partner)
    base=long.group_by("player_id").agg(
        pl.len().alias("bn"), pl.col("imp").mean().alias("b_imp"), pl.col("surr").mean().alias("b_surr"))
    # per-member shared aggregate
    def side(impcol,surrcol,pcol,key):
        return (d.group_by("pair_id").agg(
            pl.len().alias("n_shared"),
            pl.col(impcol).mean().alias(f"{key}_imp_sh"),
            pl.col(surrcol).mean().alias(f"{key}_surr_sh"),
            pl.first(pcol).alias(f"{key}_pid")))
    a=side("m1_imp","m1_surr","player_1","a"); b=side("m2_imp","m2_surr","player_2","b").drop("n_shared")
    j=a.join(b,on="pair_id")
    j=(j.join(base.rename({"player_id":"a_pid","bn":"a_bn","b_imp":"a_bimp","b_surr":"a_bsurr"}),on="a_pid",how="left")
         .join(base.rename({"player_id":"b_pid","bn":"b_bn","b_imp":"b_bimp","b_surr":"b_bsurr"}),on="b_pid",how="left"))
    def pvf(sh,bm,bn):
        field=(bm*bn - sh*pl.col("n_shared"))/(bn-pl.col("n_shared")).clip(lower_bound=1)
        return (sh-field)*pl.col("n_shared")/(pl.col("n_shared")+SHRINK)
    j=j.with_columns(
        pvf(pl.col("a_surr_sh"),pl.col("a_bsurr"),pl.col("a_bn")).alias("a_surr_mi"),
        pvf(pl.col("b_surr_sh"),pl.col("b_bsurr"),pl.col("b_bn")).alias("b_surr_mi"),
        pvf(pl.col("a_imp_sh"),pl.col("a_bimp"),pl.col("a_bn")).alias("a_imp_mi"),
        pvf(pl.col("b_imp_sh"),pl.col("b_bimp"),pl.col("b_bn")).alias("b_imp_mi"),
    ).with_columns(
        (pl.col("a_surr_mi")+pl.col("b_surr_mi")).alias("mi_surr_sum"),
        pl.min_horizontal("a_surr_mi","b_surr_mi").alias("mi_surr_min"),
        pl.max_horizontal("a_surr_mi","b_surr_mi").alias("mi_surr_max"),
        (pl.col("a_imp_mi")+pl.col("b_imp_mi")).alias("mi_imp_sum"),
        pl.min_horizontal("a_imp_mi","b_imp_mi").alias("mi_imp_min"),
        (pl.col("a_surr_mi")+pl.col("b_surr_mi")+pl.col("a_imp_mi")+pl.col("b_imp_mi")).alias("ti_joint_excess"),
    )
    return j.select(["pair_id"]+CIMI_FEATS)

def main():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","behavior_family","is_labeled"])
    mi=build_mi("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet")
    m=dp.join(mi,on="pair_id",how="left")
    y=m["label"].to_numpy(); lab=m["is_labeled"].to_numpy()
    log(f"MI feats built dev {mi.shape}")
    print("\n=== field-normalized MI/TI: univariate AUC (labelled) ===")
    for c in CIMI_FEATS:
        v=m[c].fill_null(0).to_numpy().astype(float)
        try: a=roc_auc_score(y[lab],v[lab]); print(f"  {c:18s} AUC {max(a,1-a):.4f}")
        except Exception: pass
    return 0

if __name__=="__main__":
    raise SystemExit(main())
