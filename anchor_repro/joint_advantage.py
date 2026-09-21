"""SOLO-BASELINE-NORMALIZED features (§171 joint-advantage + §172 equity drift-fix, unified).

The collusion literature detects collusion by the ADVANTAGE it confers vs independent play, not by
behavior pattern-matching. And §172 showed our raw equity features DRIFT because they carry each player's
absolute style. Both are fixed the same way: normalize each pair against EACH MEMBER'S OWN SOLO BASELINE
(computed WITHIN-PHASE so the normalization is phase-local -> the excess transfers dev->eval).

Features (all pair-level, phase-robust by construction):
  ja_net_excess    : combined net_bb of A+B in SHARED hands - (solo_mean_net_A + solo_mean_net_B)*n_shared
                     normalized per shared hand. >0 = they win TOGETHER beyond solo baselines (the core
                     joint-advantage / collusion-table signal, family-agnostic).
  ja_winrate_excess: pair's shared-hand combined win-rate - sum of solo win-rates.
  ja_cofold_excess : rate both fold in same hand - product of solo fold rates (coordinated non-contest).
  eqx_surr_excess  : pair high-eq-fold-to-partner rate - each member's solo high-eq-fold rate (drift-
                     hardened equity-surrender; needs equity cache).
Solo baselines from player_hands (all a player's hands in that phase). Validate vs 372 labels + gate.
Run:  python -m anchor_repro.joint_advantage
"""
from __future__ import annotations
import time, warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EQ=STEP3/"edge"/"equity"
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def solo_baseline(phase):
    """Per-player solo stats WITHIN phase (the phase-local reference)."""
    ph=(pl.scan_parquet(STEP3/"player_hands.parquet").filter(pl.col("phase")==phase)
        .group_by("player_id").agg(
            pl.col("net_bb").mean().alias("solo_net"),
            (pl.col("won_share")>0).mean().alias("solo_winrate"),
            pl.col("folded").cast(pl.Float64).mean().alias("solo_foldrate"),
            (pl.col("n_aggr")/(pl.col("n_actions")+1)).mean().alias("solo_aggr"),
            pl.len().alias("solo_hands")).collect())
    return ph

def build(phase, pair_hands_path):
    """pair-level solo-baseline-normalized features."""
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","player_1","player_2"])
    base=solo_baseline(phase)
    sb=dict(zip(base["player_id"],zip(base["solo_net"],base["solo_winrate"],base["solo_foldrate"])))
    # per shared hand, both members' net_bb / folded / won
    php=STEP3/"player_hands.parquet"
    hp=(pl.scan_parquet(php).filter(pl.col("phase")==phase).select(["hand_id","player_id","net_bb","folded","won_share"]))
    j=(ph.lazy()
       .join(hp.rename({"player_id":"player_1","net_bb":"n1","folded":"f1","won_share":"w1"}),on=["hand_id","player_1"],how="inner")
       .join(hp.rename({"player_id":"player_2","net_bb":"n2","folded":"f2","won_share":"w2"}),on=["hand_id","player_2"],how="inner")
       .with_columns((pl.col("n1")+pl.col("n2")).alias("joint_net"),
                     ((pl.col("w1")>0)|(pl.col("w2")>0)).cast(pl.Int8).alias("pair_won"),
                     (pl.col("f1")&pl.col("f2")).cast(pl.Int8).alias("both_fold"))
       .group_by(["pair_id","player_1","player_2"]).agg(
           pl.len().alias("nsh"),
           pl.col("joint_net").mean().alias("joint_net_mean"),
           pl.col("pair_won").mean().alias("pair_winrate"),
           pl.col("both_fold").mean().alias("cofold_rate"))
       .collect(engine="streaming").to_pandas())
    # normalize by solo baselines
    def g(pid,i): return sb.get(pid,(0.0,0.0,0.0))[i]
    j["s1_net"]=j["player_1"].map(lambda p:g(p,0)); j["s2_net"]=j["player_2"].map(lambda p:g(p,0))
    j["s1_wr"]=j["player_1"].map(lambda p:g(p,1));  j["s2_wr"]=j["player_2"].map(lambda p:g(p,1))
    j["s1_fr"]=j["player_1"].map(lambda p:g(p,2));  j["s2_fr"]=j["player_2"].map(lambda p:g(p,2))
    j["ja_net_excess"]=j["joint_net_mean"]-(j["s1_net"]+j["s2_net"])
    j["ja_winrate_excess"]=j["pair_winrate"]-(j["s1_wr"]+j["s2_wr"]).clip(upper=1.0)
    j["ja_cofold_excess"]=j["cofold_rate"]-(j["s1_fr"]*j["s2_fr"])
    return pl.from_pandas(j[["pair_id","ja_net_excess","ja_winrate_excess","ja_cofold_excess","joint_net_mean","pair_winrate","cofold_rate","nsh"]])

JA_FEATS=["ja_net_excess","ja_winrate_excess","ja_cofold_excess","joint_net_mean","pair_winrate","cofold_rate"]

def main():
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    dp=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","label","behavior_family","is_labeled"])
    ja=build("development",STEP3/"dev_pair_hands.parquet")
    m=dp.join(ja,on="pair_id",how="left").to_pandas()
    lab=m["is_labeled"].astype(bool).to_numpy(); y=m["label"].fillna(0).astype(int).to_numpy()
    log(f"joint-advantage feats built; labelled {lab.sum()}")
    print("\n=== joint-advantage standalone AUC vs 372 dev labels ===")
    res=[]
    for c in JA_FEATS:
        v=m[c].fillna(m[c].median()).to_numpy()[lab]; a=roc_auc_score(y[lab],v); res.append((c,max(a,1-a),a))
    for c,a,raw in sorted(res,key=lambda x:-x[1]): print(f"  {a:.4f}  {'hi=sus' if raw>0.5 else 'lo=sus'}  {c}")
    print("\n=== per-family mean of top feature ===")
    top=sorted(res,key=lambda x:-x[1])[0][0]
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        s=m[(m["behavior_family"]==fam)][top]; nn=m[(m["label"]==0)&lab][top]
        print(f"  {fam}: {top}={s.mean():.4f} (n={len(s)})  vs neg {nn.mean():.4f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
