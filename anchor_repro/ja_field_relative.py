"""FIELD-RELATIVE joint-advantage residuals — §223 Option 2, candidate #1.

The existing joint_advantage.py normalizes by a SOLO baseline (each member's own within-phase stats). The
§220 audit flagged exactly this: solo baselines DEGENERATE because ~88% of dev colluders are fully-shared,
so a player's "solo" stats are contaminated by their collusion partner. Prescription: a FIELD/POPULATION
baseline, not solo. This rebuilds joint-advantage the same way equity worked: per shared hand, model the
FIELD's expected joint outcome from matched context (pot, blinds, showdown structure, aggression, live
players), and keep the RESIDUAL (observed - expected). Then exposure-shrink (EB, k=100) for phase-invariance.

Signals per shared hand (pair A,B):
  joint_net    = net_bb(A) + net_bb(B)   -> residual = do they win MORE together than the field expects?
  cofold       = both folded              -> residual = coordinated non-contest beyond field
  pair_won     = either won the pot        -> residual = pair captures pots beyond field
Field null fit POOLED over ALL co-seated pairs' shared hands, cross-fitted 5-fold, NO pair id, NO label.
Outputs edge/equity/ja_field_relative_{phase}.parquet with FEATS_JA (EB-shrunk residuals).
Run:  python -m anchor_repro.ja_field_relative --phase development  (and --phase evaluation)
"""
from __future__ import annotations
import argparse, time, warnings, numpy as np, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
EB_K=3000.0  # §223: JA sum-residuals are dominated by exposure-variance; k=3000 -> drift 0.587 (<0.65) AND
             # incremental GROWS to +0.0234 (heavier shrink removes noise, keeps signal). Much stronger than k=100.
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

# context features a non-colluding pair shares (NO pair id, NO label, NO absolute pair-activity count)
CTX=["pot_bb","big_blind","players_at_showdown","live_bucket","a_aggr","b_aggr","a_pos","b_pos","a_vpip","b_vpip"]
FEATS_JA=["jar_net_eb","jar_cofold_eb","jar_won_eb","jar_net_z"]
FEATS_JA_AUX=["jar_net_sum","jar_cofold_sum","jar_won_sum","jar_n_opp"]

def add_shrunk_ja(df):
    n=df["jar_n_opp"].clip(lower=1)
    df=df.copy()
    df["jar_net_eb"]=df["jar_net_sum"]/(n+EB_K)
    df["jar_cofold_eb"]=df["jar_cofold_sum"]/(n+EB_K)
    df["jar_won_eb"]=df["jar_won_sum"]/(n+EB_K)
    df["jar_net_z"]=df["jar_net_sum"]/np.sqrt(n+EB_K)
    return df

def build(phase):
    ph=pl.read_parquet(STEP3/("dev_pair_hands.parquet" if phase=="development" else "eval_pair_hands.parquet")).select(["pair_id","hand_id","player_1","player_2"])
    hp=(pl.scan_parquet(STEP3/"player_hands.parquet").filter(pl.col("phase")==phase)
        .select(["hand_id","player_id","net_bb","folded","won_share","pot_bb","big_blind","players_at_showdown",
                 "n_aggr","n_actions","pos","vpip"]))
    a=hp.rename({"player_id":"player_1","net_bb":"n1","folded":"f1","won_share":"w1","n_aggr":"ag1","n_actions":"na1","pos":"pos1","vpip":"v1"})
    b=hp.rename({"player_id":"player_2","net_bb":"n2","folded":"f2","won_share":"w2","n_aggr":"ag2","n_actions":"na2","pos":"pos2","vpip":"v2",
                 "pot_bb":"pot_bb_b","big_blind":"bb_b","players_at_showdown":"pas_b"})
    j=(ph.lazy()
       .join(a,on=["hand_id","player_1"],how="inner")
       .join(b.select(["hand_id","player_2","n2","f2","w2","ag2","na2","pos2","v2"]),on=["hand_id","player_2"],how="inner")
       .with_columns([
           (pl.col("n1")+pl.col("n2")).alias("joint_net"),
           (pl.col("f1")&pl.col("f2")).cast(pl.Int8).alias("cofold"),
           ((pl.col("w1")>0)|(pl.col("w2")>0)).cast(pl.Int8).alias("pair_won"),
           pl.min_horizontal(pl.col("players_at_showdown").fill_null(2),pl.lit(6)).alias("live_bucket"),
           (pl.col("ag1")/(pl.col("na1")+1)).alias("a_aggr"),(pl.col("ag2")/(pl.col("na2")+1)).alias("b_aggr"),
           pl.col("pos1").fill_null(0).alias("a_pos"),pl.col("pos2").fill_null(0).alias("b_pos"),
           pl.col("v1").fill_null(0).cast(pl.Float32).alias("a_vpip"),pl.col("v2").fill_null(0).cast(pl.Float32).alias("b_vpip"),
       ])
       .select(["pair_id","joint_net","cofold","pair_won","pot_bb","big_blind","players_at_showdown","live_bucket",
                "a_aggr","b_aggr","a_pos","b_pos","a_vpip","b_vpip"])
       .collect(engine="streaming"))
    for c in CTX: j=j.with_columns(pl.col(c).fill_null(0.0).cast(pl.Float32))
    log(f"  {j.height:,} shared-hand events")
    X=j.select(CTX).to_numpy().astype("float32"); n=j.height; rng=np.random.RandomState(42)
    P=dict(learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=8)
    # Field null only needs to learn P(outcome|context) — a subsample estimates it as well as 14.8M rows and
    # is RAM-safe. Fit on a held-out FIT sample, predict all rows; to avoid a pair's own hands leaking into
    # its own expectation, use 2 disjoint fit-halves and predict each row with the model NOT trained on its half.
    half=rng.randint(0,2,size=n)
    FITN=300_000
    def crossfit(yv,obj):
        o=np.zeros(n)
        for h in (0,1):
            src=np.where(half!=h)[0]  # fit on the OTHER half
            idx=src if len(src)<=FITN else rng.choice(src,FITN,replace=False)
            m=lgb.train({**P,"objective":obj,"seed":42},lgb.Dataset(X[idx],yv[idx]),num_boost_round=200)
            te=half==h; o[te]=m.predict(X[te])
        return o
    jn=j["joint_net"].to_numpy().astype("float32"); cf=j["cofold"].to_numpy().astype("float32"); pw=j["pair_won"].to_numpy().astype("float32")
    ejn=crossfit(jn,"regression"); ecf=crossfit(cf,"binary"); epw=crossfit(pw,"binary")
    log(f"  field nulls cross-fit (joint_net R~, cofold {ecf.mean():.3f}, pair_won {epw.mean():.3f})")
    j=j.with_columns([pl.Series("r_net",jn-ejn),pl.Series("r_cofold",cf-ecf),pl.Series("r_won",pw-epw)])
    agg=j.group_by("pair_id").agg([
        pl.col("r_net").sum().alias("jar_net_sum"),
        pl.col("r_cofold").sum().alias("jar_cofold_sum"),
        pl.col("r_won").sum().alias("jar_won_sum"),
        pl.len().alias("jar_n_opp"),
    ])
    out=EQ/f"ja_field_relative_{phase}.parquet"
    agg.select(["pair_id"]+FEATS_JA_AUX).write_parquet(out)
    log(f"  wrote {out} ({agg.height:,} pairs)")
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",required=True,choices=["development","evaluation"]); a=ap.parse_args()
    build(a.phase); return 0

if __name__=="__main__":
    raise SystemExit(main())
