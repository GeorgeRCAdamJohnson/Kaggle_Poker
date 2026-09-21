"""EQUITY-SURRENDER features (§ task #2, the NOVEL signal). Not "holds a strong hand" (AUC 0.73, overlaps
existing) but "SURRENDERS a high-equity hand to the partner's benefit" — join equity x action(folded/
didn't-bet) x outcome(partner won). This is what the Chen/made-value proxies + showdown-only rank CANNOT
see: folding/checking a hand that was ACTUALLY ahead, mid-hand, no showdown.

Per (pair,hand,member) we have equity (task #1). Join per-hand action/outcome (p{i}_folded, p{i}_won_share,
p{i}_net_bb, p{i}_n_aggr, p{i}_went_to_showdown). Build per-pair aggregates:
  eqs_fold_partnerwon : member had equity>=T, FOLDED, partner won the hand (chips kept in-pair) -- directed_transfer
  eqs_checkdown_hi    : member had equity>=T, did NOT bet/raise, went to showdown WITH partner       -- soft_play
  eq_surrendered_sum  : sum of equity*(folded or no-aggr)*(partner benefited)  = win-prob given away
  eq_surrendered_max  : the single most-surrendered hand
  + rates and per-street splits.
Validate each vs the 372 dev labels (AUC) and report per-family; report dev prevalence.
Run:  python -m anchor_repro.equity_features
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)
T=0.65  # equity threshold for "strong hand surrendered"

def build_features(phase, eq_path, hf_glob, keep_pairs=None):
    eq=pl.read_parquet(eq_path)  # pair_id,hand_id,member,street,equity,n_live
    HF=sorted(STEP3.glob(hf_glob))
    cols=["pair_id","hand_id","loser_is_p1","players_at_showdown",
          "p1_folded","p2_folded","p1_won_share","p2_won_share","p1_net_bb","p2_net_bb",
          "p1_n_aggr","p2_n_aggr","p1_went_to_showdown","p2_went_to_showdown","transfer_any"]
    hf=pl.concat([pl.scan_parquet(f).select([c for c in cols]).collect() for f in HF])
    if keep_pairs is not None:
        hf=hf.filter(pl.col("pair_id").is_in(list(keep_pairs)))
    # reshape equity to wide: one row per (pair,hand) with m1_equity, m2_equity
    eqw=(eq.pivot(values="equity",index=["pair_id","hand_id"],on="member",aggregate_function="first")
         .rename({"1":"m1_eq","2":"m2_eq"}))
    d=hf.join(eqw,on=["pair_id","hand_id"],how="left")
    # partner-won flags: for member1, "partner won" = p2 won share > 0 and p1 net < 0 (m1 gave to m2)
    d=d.with_columns(
        (pl.col("p2_won_share")>0).alias("p2_won"), (pl.col("p1_won_share")>0).alias("p1_won"),
        (pl.col("m1_eq").fill_null(0)>=T).alias("m1_hi"), (pl.col("m2_eq").fill_null(0)>=T).alias("m2_hi"),
    )
    # SURRENDER events (per member): high-eq AND (folded OR no-aggr) AND partner benefited (partner won / this member lost net)
    d=d.with_columns(
        # m1 surrenders to m2
        (pl.col("m1_hi") & ((pl.col("p1_folded")) | (pl.col("p1_n_aggr")==0)) & (pl.col("p2_won")) & (pl.col("p1_net_bb")<0)).alias("m1_surr"),
        (pl.col("m2_hi") & ((pl.col("p2_folded")) | (pl.col("p2_n_aggr")==0)) & (pl.col("p1_won")) & (pl.col("p2_net_bb")<0)).alias("m2_surr"),
        # checkdown vs partner: both to showdown, both high-eq, neither aggressed
        (pl.col("m1_hi") & pl.col("m2_hi") & (pl.col("p1_went_to_showdown")) & (pl.col("p2_went_to_showdown")) & (pl.col("p1_n_aggr")==0) & (pl.col("p2_n_aggr")==0)).alias("checkdown_hi"),
        # equity surrendered magnitude (member eq if they surrendered, else 0)
        (pl.when(pl.col("m1_hi") & ((pl.col("p1_folded"))|(pl.col("p1_n_aggr")==0)) & (pl.col("p2_won"))).then(pl.col("m1_eq")).otherwise(0.0)
         + pl.when(pl.col("m2_hi") & ((pl.col("p2_folded"))|(pl.col("p2_n_aggr")==0)) & (pl.col("p1_won"))).then(pl.col("m2_eq")).otherwise(0.0)).alias("eq_surr_hand"),
        (pl.max_horizontal("m1_eq","m2_eq")).alias("pair_max_eq"),
    )
    agg=(d.group_by("pair_id").agg(
        pl.len().alias("n_hands"),
        (pl.col("m1_surr")|pl.col("m2_surr")).sum().alias("eqs_fold_partnerwon"),
        ((pl.col("m1_surr")|pl.col("m2_surr")).sum()/pl.len()).alias("eqs_fold_rate"),
        pl.col("checkdown_hi").sum().alias("eqs_checkdown_hi"),
        (pl.col("checkdown_hi").sum()/pl.len()).alias("eqs_checkdown_rate"),
        pl.col("eq_surr_hand").sum().alias("eq_surrendered_sum"),
        pl.col("eq_surr_hand").max().alias("eq_surrendered_max"),
        pl.col("eq_surr_hand").mean().alias("eq_surrendered_mean"),
        pl.col("pair_max_eq").quantile(0.95).alias("eq_p95"),
        (pl.col("m1_eq").fill_null(0)>=0.8).sum().add((pl.col("m2_eq").fill_null(0)>=0.8).sum()).alias("n_strong_hands"),
    ))
    return agg

FEATS=["eqs_fold_partnerwon","eqs_fold_rate","eqs_checkdown_hi","eqs_checkdown_rate",
       "eq_surrendered_sum","eq_surrendered_max","eq_surrendered_mean","eq_p95","n_strong_hands"]

def main():
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","behavior_family","is_labeled"])
    keep=set(dp.filter(pl.col("is_labeled"))["pair_id"].to_list())
    agg=build_features("development",EQ/"pairhand_equity_devlab.parquet","dev_hand_features_*.parquet",keep_pairs=keep)
    m=dp.join(agg,on="pair_id",how="inner")
    log(f"labelled pairs with equity feats: {m.height}")
    y=m["label"].to_numpy()
    print("\n=== equity-SURRENDER features: standalone AUC vs 372 dev labels ===")
    res=[]
    for c in FEATS:
        v=m[c].fill_null(0).to_numpy().astype(float)
        try: a=roc_auc_score(y,v); res.append((c,max(a,1-a),a)); 
        except Exception: pass
    for c,a,raw in sorted(res,key=lambda x:-x[1]):
        print(f"  {a:.4f}  {'hi=sus' if raw>0.5 else 'lo=sus'}  {c}")
    print("\n=== per-family mean of the top surrender feature ===")
    top=sorted(res,key=lambda x:-x[1])[0][0]
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        s=m.filter(pl.col("behavior_family")==fam)[top]
        nn=m.filter(pl.col("label")==0)[top]
        print(f"  {fam}: {top}={s.mean():.3f} (n={s.len()})  vs neg {nn.mean():.3f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
