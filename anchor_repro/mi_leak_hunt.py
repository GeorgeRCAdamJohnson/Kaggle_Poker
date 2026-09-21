"""HUNT THE LEAK in field-normalized MI/TI (§192): +0.103 eval-weighted is too good -> almost certainly
leakage. Test the hypotheses directly on dev labelled pairs.

H1 SOLO-BASELINE COLLAPSE: does the field baseline include the pair's shared hands? For players who
   appear ~only with this partner, base_n ~= n_shared -> field denominator clips -> excess = raw outcome.
   Measure: distribution of base_n vs n_shared for POSITIVE pairs' members. If base_n ~= n_shared, leak.
H2 OUTCOME LEAKAGE: the surr term uses p*_net_bb / p*_won (hand outcome). Are the MI feats basically a
   re-encoding of the pair's win/net rate (which correlates with the label via the generator)? Correlate
   ti_joint_excess with a pure outcome feature (pair joint net) — if ~1.0, it's outcome not collusion.
H3 HELD-OUT-PAIR sanity: recompute solo baseline EXCLUDING the pair's own shared hands; re-check AUC. If
   AUC collapses, the signal WAS the shared hands leaking into their own baseline.
Run:  python -m anchor_repro.mi_leak_hunt
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from anchor_repro.collusion_impact_mi import _perhand_member, SHRINK, CIMI_FEATS, build_mi
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def main():
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","is_labeled"])
    long, d = _perhand_member("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet",None)
    pids=d.group_by("pair_id").agg(pl.first("player_1").alias("player_1"),pl.first("player_2").alias("player_2"))
    dp=dp.join(pids,on="pair_id",how="left")

    # H1: base_n (all a player's hands) vs n_shared (hands with THIS partner), for positive pairs
    base=long.group_by("player_id").agg(pl.len().alias("bn"))
    nsh=d.group_by("pair_id").agg(pl.len().alias("n_shared"))
    m=dp.join(nsh,on="pair_id",how="left")
    m=m.join(base.rename({"player_id":"player_1","bn":"a_bn"}),on="player_1",how="left")
    m=m.join(base.rename({"player_id":"player_2","bn":"b_bn"}),on="player_2",how="left").to_pandas()
    pos=m[m["label"]==1]
    log("=== H1 solo-baseline collapse (base_n vs n_shared for POSITIVE pairs) ===")
    ratio=pos["n_shared"]/pos[["a_bn","b_bn"]].min(axis=1)
    log(f"  n_shared / min(member total hands): mean {ratio.mean():.3f} median {ratio.median():.3f} p90 {ratio.quantile(.9):.3f} max {ratio.max():.3f}")
    log(f"  fraction of positive pairs where a member's hands are >80% shared: {(ratio>0.8).mean():.3f}")
    log(f"  (if high -> field baseline collapses to shared -> excess re-encodes pair outcome = LEAK)")

    # H2: outcome correlation. pair joint net (pure outcome) vs ti_joint_excess
    outc=(d.with_columns((pl.col("p1_net_bb")+pl.col("p2_net_bb")).alias("joint_net"))
            .group_by("pair_id").agg(pl.col("joint_net").mean().alias("pair_joint_net"),
                                     (pl.col("p1_won")|pl.col("p2_won")).mean().alias("pair_win_rate")))
    mi=build_mi("development",EQ/"pairhand_equity_dev.parquet","dev_hand_features_*.parquet")
    j=dp.join(mi,on="pair_id",how="left").join(outc,on="pair_id",how="left").to_pandas()
    lab=j["is_labeled"].to_numpy(); y=j["label"].to_numpy()
    log("=== H2 outcome leakage ===")
    for c in ["ti_joint_excess","mi_imp_sum","mi_surr_sum"]:
        cc=np.corrcoef(j[c].fillna(0),j["pair_joint_net"].fillna(0))[0,1]
        cw=np.corrcoef(j[c].fillna(0),j["pair_win_rate"].fillna(0))[0,1]
        log(f"  corr({c:16s}, pair_joint_net)={cc:+.3f}  corr(.,pair_win_rate)={cw:+.3f}")
    # is a PURE outcome feature already as good?
    for c in ["pair_joint_net","pair_win_rate"]:
        a=roc_auc_score(y[lab],j[c].fillna(0).to_numpy()[lab]); log(f"  pure outcome {c:16s} AUC {max(a,1-a):.4f}  (vs ti_joint_excess 0.949)")

    # H3: recompute MI with solo baseline EXCLUDING the pair's shared hands (true field)
    log("=== H3 solo baseline EXCLUDING shared hands (true leave-pair-out field) ===")
    # per player total imp/surr and count over ALL hands
    tot=long.group_by("player_id").agg(pl.len().alias("bn"),pl.col("imp").sum().alias("s_imp"),pl.col("surr").sum().alias("s_surr"))
    # per (pair, member) shared sum/count
    a_sh=d.group_by("pair_id").agg(pl.len().alias("n"),pl.col("m1_imp").sum().alias("a_imp_s"),pl.col("m1_surr").sum().alias("a_surr_s"),pl.first("player_1").alias("pid"))
    b_sh=d.group_by("pair_id").agg(pl.col("m2_imp").sum().alias("b_imp_s"),pl.col("m2_surr").sum().alias("b_surr_s"),pl.first("player_2").alias("pid2"))
    aj=a_sh.join(tot.rename({"player_id":"pid"}),on="pid",how="left")
    # TRUE field mean = (player total - shared) / (player_n - shared_n), guard n>shared
    aj=aj.with_columns(
        ((pl.col("s_imp")-pl.col("a_imp_s"))/(pl.col("bn")-pl.col("n")).clip(lower_bound=1)).alias("a_field_imp"),
        (pl.col("a_imp_s")/pl.col("n")).alias("a_sh_imp"),
        (pl.col("bn")-pl.col("n")).alias("a_out_n"),
    )
    aj=aj.with_columns(((pl.col("a_sh_imp")-pl.col("a_field_imp"))*pl.col("n")/(pl.col("n")+SHRINK)).alias("a_imp_mi_true"))
    jj=dp.join(aj.select(["pair_id","a_imp_mi_true","a_out_n"]),on="pair_id",how="left").to_pandas()
    lab2=jj["is_labeled"].to_numpy(); 
    frac_noout=(jj["a_out_n"].fillna(0)<=0).mean()
    log(f"  fraction of pairs where member A has ZERO non-shared hands (field undefined): {frac_noout:.3f}")
    aa=roc_auc_score(jj["label"].to_numpy()[lab2], jj["a_imp_mi_true"].fillna(0).to_numpy()[lab2])
    log(f"  a_imp_mi with TRUE leave-pair-out field: AUC {max(aa,1-aa):.4f}  (vs 0.827 in-baseline version)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
