"""n_shared-INVARIANT per-hand collusion-impact RATES (§194 corrected construction of the AAAI axis).

§191 raw ci_* drifted (equity*pot carries stakes level). §192/§193 field-norm-vs-SOLO-baseline leaked
(dev colluders ~fully shared -> baseline collapses; and dev shared_hands=120 vs eval=86 regime shift). §194
fix: build features that are INVARIANT to shared-hand-count and NEVER reference co-occurrence:
  - per-hand RATES (mean over shared hands), never sums -> identical at 86 or 121 hands.
  - normalize equity contribution against a GLOBAL FIELD baseline (all players in the phase at similar
    equity/pot), NOT the pair's own solo hands -> no co-occurrence reference, absolute style cancels.
  - the impact is the FRACTION of high-equity spots the pair converted to partner benefit ABOVE the field
    rate of converting such spots to ANY benefit -> strategy-relative, count-invariant, phase-robust.

Per shared hand define surrender EVENT (member ahead by equity, net<0, partner won) and its equity weight.
Pair features (all RATES / field-relative, n_shared-invariant):
  cir_surr_rate       : fraction of shared hands that are surrender events (rate, not count)
  cir_surr_eqwt       : mean surrendered-equity per shared hand (bounded [0,1]-ish, pot-normalized)
  cir_impact_excess   : mean per-hand (pair realized frac of pot) - (pair equity-expected frac of pot)
                        = per-hand excess-over-fair, a RATE in pot units, no sums
  cir_hi_eq_fold_rate : rate of hands where a member folds a >0.6-equity hand to partner's benefit
  cir_field_rel_surr  : cir_surr_rate MINUS the global field's high-eq-fold rate (field = all players)
Run:  python -m anchor_repro.collusion_impact_rate
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

CIR_FEATS=["cir_surr_rate","cir_surr_eqwt","cir_impact_excess","cir_hi_eq_fold_rate","cir_field_rel_surr","cir_impact_p75"]
# clean subset for the gate: the count-invariant, above-noise-AUC features (drop cir_impact_p75: AUC 0.50 +
# the only nonzero shared_hands corr; cir_field_rel_surr == cir_surr_rate shifted, keep one)
CIR_GATE=["cir_surr_rate","cir_surr_eqwt","cir_hi_eq_fold_rate","cir_impact_excess"]

def _perhand(phase, eq_path, hf_glob, keep_pairs):
    eq=pl.read_parquet(eq_path)
    peak=(eq.group_by(["pair_id","hand_id","member"]).agg(pl.col("equity").max().alias("peak_eq"))
            .pivot(values="peak_eq",index=["pair_id","hand_id"],on="member",aggregate_function="first")
            .rename({"1":"m1_peak","2":"m2_peak"}))
    HF=sorted(STEP3.glob(hf_glob))
    cols=["pair_id","hand_id","pot_bb","player_1","player_2",
          "p1_folded","p2_folded","p1_won_share","p2_won_share","p1_net_bb","p2_net_bb","p1_n_aggr","p2_n_aggr"]
    hf=pl.concat([pl.scan_parquet(f).select(cols).collect() for f in HF])
    if keep_pairs is not None: hf=hf.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    d=hf.join(peak,on=["pair_id","hand_id"],how="left").with_columns(
        pl.col("m1_peak").fill_null(0.0),pl.col("m2_peak").fill_null(0.0),pl.col("pot_bb").fill_null(0.0),
        (pl.col("p1_won_share")>0).alias("p1_won"),(pl.col("p2_won_share")>0).alias("p2_won"))
    d=d.with_columns(
        # surrender event this hand (either member): ahead by equity, net<0, partner won
        ((pl.col("m1_peak")>pl.col("m2_peak"))&(pl.col("p1_net_bb")<0)&pl.col("p2_won") |
         (pl.col("m2_peak")>pl.col("m1_peak"))&(pl.col("p2_net_bb")<0)&pl.col("p1_won")).cast(pl.Int8).alias("surr_ev"),
        # surrendered equity (the ahead member's peak eq), 0 else — a per-hand FRACTION in [0,1]
        (pl.when((pl.col("m1_peak")>pl.col("m2_peak"))&(pl.col("p1_net_bb")<0)&pl.col("p2_won")).then(pl.col("m1_peak"))
         .when((pl.col("m2_peak")>pl.col("m1_peak"))&(pl.col("p2_net_bb")<0)&pl.col("p1_won")).then(pl.col("m2_peak"))
         .otherwise(0.0)).alias("surr_eq"),
        # high-eq fold to partner benefit: member folded a >0.6 peak-eq hand and partner won
        (((pl.col("m1_peak")>=0.6)&(pl.col("p1_folded"))&pl.col("p2_won")) |
         ((pl.col("m2_peak")>=0.6)&(pl.col("p2_folded"))&pl.col("p1_won"))).cast(pl.Int8).alias("hi_eq_fold"),
        # per-hand impact excess as a RATE: pair realized pot-fraction minus equity-expected pot-fraction
        (((pl.col("p1_won_share")+pl.col("p2_won_share"))) - pl.max_horizontal("m1_peak","m2_peak")).alias("impact_excess_hand"),
    )
    return d

def build_rate(phase, eq_path, hf_glob, keep_pairs=None):
    d=_perhand(phase,eq_path,hf_glob,keep_pairs)
    # GLOBAL field baseline: high-eq-fold rate + surr rate across ALL pairs' hands this phase (no pair ref)
    field_surr=d["surr_ev"].mean(); field_fold=d["hi_eq_fold"].mean()
    log(f"  {phase} field baselines: surr_rate={field_surr:.4f} hi_eq_fold_rate={field_fold:.4f}")
    agg=(d.group_by("pair_id").agg(
        pl.col("surr_ev").mean().alias("cir_surr_rate"),
        pl.col("surr_eq").mean().alias("cir_surr_eqwt"),
        pl.col("impact_excess_hand").mean().alias("cir_impact_excess"),
        pl.col("impact_excess_hand").quantile(0.75).alias("cir_impact_p75"),
        pl.col("hi_eq_fold").mean().alias("cir_hi_eq_fold_rate"),
    ).with_columns(
        (pl.col("cir_surr_rate")-field_surr).alias("cir_field_rel_surr"),
    ))
    return agg.select(["pair_id"]+CIR_FEATS)

def main():
    from sklearn.metrics import roc_auc_score
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","is_labeled","shared_hands"])
    r=build_rate("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet")
    m=dp.join(r,on="pair_id",how="left").to_pandas()
    y=m["label"].to_numpy(); lab=m["is_labeled"].to_numpy().astype(bool)
    log(f"CIR feats built dev {r.shape}")
    print("\n=== n_shared-invariant impact RATES: univariate AUC (labeled) + corr with shared_hands (want ~0) ===")
    for c in CIR_FEATS:
        v=m[c].fillna(0).to_numpy().astype(float)
        a=roc_auc_score(y[lab],v[lab]); csh=np.corrcoef(v,m["shared_hands"].fillna(0))[0,1]
        print(f"  {c:20s} AUC {max(a,1-a):.4f}  corr(shared_hands) {csh:+.3f}")
    print("  (corr(shared_hands) near 0 = count-invariant = won't mis-transfer on dev120->eval86)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
