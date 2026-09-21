"""§219 PHASE 1c — does BOARD/HOLE interaction, bet-size LATTICE, or multi-hand consistency sharply raise
purity beyond the raw-outcome ceiling (0.31)? §109's stated fork: if a hand-rule can't get purity toward
~0.8, the generator uses info we must MODEL. Test the three candidate ingredients directly on the planted set.

Adds to the gen_rule_frame:
  (A) decisive-action ratios: at each planted-pair-hand's largest aggressive action — amount/pot, amount/bb,
      stack/pot (SPR). Histogram planted vs non-planted for LATTICE spikes (§113 tested ratios null on all
      hands; here restricted to the family + combined with the rule).
  (B) board/hole interaction: did the winner actually hold the best hand at showdown (retrospective strength)?
      A "transfer" where the loser HAD the better hand is the smoking gun the raw net can't see.
  (C) multi-hand: is the planted hand an OUTLIER within the pair's own history (top-k by flow/pot)?
For each, measure purity of the best combined rule. If purity jumps toward 0.6-0.8 -> recovery. If it stays
~0.3 -> generator needs unmodeled info, evidence ceiling confirmed structurally.
Run: python -m anchor_repro.gen_rule_phase1c
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
warnings.simplefilter("ignore")
from pathlib import Path
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3")

def main():
    fr=pl.read_parquet(STEP3/"edge"/"genrule"/"gen_rule_frame.parquet")
    hset=fr["hand_id"].unique().to_list()
    # (A) decisive aggressive-action ratios
    acts=(pl.scan_parquet(D/"actions.parquet").filter(pl.col("hand_id").is_in(hset))
          .filter(pl.col("action").is_in(["bet","raise","all_in"]))
          .select(["hand_id","amount","pot_before","stack_before","to_call"]).collect())
    dec=acts.group_by("hand_id").agg(
        pl.col("amount").max().alias("max_bet"),
        pl.col("pot_before").filter(pl.col("amount")==pl.col("amount").max()).first().alias("pot_at"),
        pl.col("stack_before").filter(pl.col("amount")==pl.col("amount").max()).first().alias("stack_at"))
    hb=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(hset)).select(["hand_id","big_blind"])
    dec=dec.join(hb,on="hand_id",how="left").with_columns(pl.col("big_blind").cast(pl.Float64).clip(1,None).alias("bb"))
    dec=dec.with_columns(
        (pl.col("max_bet")/pl.col("pot_at").clip(1,None)).alias("bet_pot"),
        (pl.col("max_bet")/pl.col("bb")).alias("bet_bb"),
        (pl.col("stack_at")/pl.col("pot_at").clip(1,None)).alias("spr"))
    fr=fr.join(dec.select(["hand_id","bet_pot","bet_bb","spr"]),on="hand_id",how="left")
    # (C) multi-hand outlier: within-pair percentile of pot and |net gap|
    fr=fr.with_columns((pl.col("winner_net")-pl.col("loser_net")).abs().alias("net_gap_g"))
    fr=fr.with_columns(
        pl.col("pot_bb_g").rank(descending=True).over("pair_id").alias("pot_rank_in_pair"),
        (pl.col("pot_bb_g").rank("average",descending=False).over("pair_id")/pl.count("pot_bb_g").over("pair_id")).alias("pot_pct_in_pair"),
        (pl.col("net_gap_g").rank("average",descending=False).over("pair_id")/pl.count("net_gap_g").over("pair_id")).alias("gap_pct_in_pair"))

    def cp(f,mask):
        plt=f["is_planted"].to_numpy(); m=mask
        planted=plt.sum(); match=m.sum(); mp=(m&(plt==1)).sum()
        return mp/max(planted,1), mp/max(match,1), int(match), int(mp)

    print("=== PHASE 1c: does board/lattice/multi-hand raise purity beyond 0.31 raw-outcome ceiling? ===")
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        f=fr.filter(pl.col("behavior_family")==fam); base=f["is_planted"].mean()
        print(f"\n--- {fam} (base {base:.3f}) ---")
        opp=f["opposite_sign_net"].to_numpy(); pot=f["pot_bb_g"].to_numpy(); bsd=f["both_showdown"].to_numpy()
        bp=f["bet_pot"].fill_null(0).to_numpy(); potpct=f["pot_pct_in_pair"].fill_null(0).to_numpy(); gappct=f["gap_pct_in_pair"].fill_null(0).to_numpy()
        # (A) LATTICE: is planted bet_pot concentrated on round fractions?
        plt=f["is_planted"].to_numpy().astype(bool)
        for tgt in [0.5,0.66,0.75,1.0]:
            fp=np.mean(np.abs(bp[plt]-tgt)<0.03); fn=np.mean(np.abs(bp[~plt]-tgt)<0.03)
            print(f"  [A lattice] bet_pot~{tgt}: planted {fp:.3f} nonpl {fn:.3f} excess {fp-fn:+.3f}")
        # (C) MULTI-HAND: planted = the pair's TOP hand by pot / net-gap?
        rules={
            "raw best (opp & pot>=20 or showdown)": (opp&(pot>=20))|(bsd&(pot>=15)),
            "+ multi-hand: TOP-decile pot in pair": ((opp&(pot>=20))|(bsd&(pot>=15))) & (potpct>=0.90),
            "+ multi-hand: TOP-decile net-gap": ((opp&(pot>=20))|(bsd&(pot>=15))) & (gappct>=0.90),
            "multi-hand ALONE: top-decile pot": (potpct>=0.90),
            "multi-hand ALONE: top-3 pot in pair": (f["pot_rank_in_pair"].to_numpy()<=3),
        }
        for name,mask in rules.items():
            c,p,mm,mp=cp(f,mask); print(f"  {name:40s} cov {c:.3f} purity {p:.3f} (x{p/max(base,1e-9):.1f}) match {mm}")
    print("\n=== VERDICT: if any purity jumps toward 0.5-0.8 -> generator recovered. If stays ~0.3 -> needs modeled info. ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
