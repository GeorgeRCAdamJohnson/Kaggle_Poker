"""§214 — OUTSIDER-EXTRACTION hypothesis (from eyeballing real evidence hands). The MISSED true-evidence
hands are BIG pots where ONE colluder wins a lot funded by OUTSIDER money while the PARTNER is dealt in
(folds early / supports / squeezes). Our features measure pair<->pair chip flow (small); the host labels
pair-extracts-from-FIELD (big, outsider-funded). We measured the WRONG direction of chip flow.

Build per (pair,hand) features capturing outsider-funded extraction:
  ox_winner_net    : max(net) of the two members (how much one member won)
  ox_from_outsider : winner's net that came from OUTSIDERS = winner_net - (partner's loss to winner)
                     approx: winner_net if partner also lost only a little (partner not the funder)
  ox_pot_to_winner : pot size * (one member won it) -- big outsider pot captured by the pair
  ox_partner_folded_early : partner folded preflop/early while winner went on to take a big pot
  ox_joint_take    : (member1 net + member2 net) when POSITIVE and large = pair's net take from the table
Test single-feature within-pair MAP@5 (beat 0.24) + ev-vs-nonev separation + per family.
Run:  python -m anchor_repro.outsider_extraction_probe
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3")

def main():
    from anchor_repro.seq_evidence import build_dh, map5, TARGET_BEHAVIORS
    dh=build_dh().to_pandas()
    if "player_1" not in dh.columns:
        lab=pl.read_csv(D/"development_labels.csv").filter(pl.col("label")==1).select(["pair_id","player_1","player_2"]).to_pandas()
        dh=dh.merge(lab,on="pair_id",how="left")
    hset=set(dh["hand_id"].unique().tolist())
    seats=pl.read_parquet(D/"seats.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","player_id","net_chips","folded","went_to_showdown"]).to_pandas()
    hands=pl.read_parquet(D/"hands.parquet").filter(pl.col("hand_id").is_in(list(hset))).select(["hand_id","final_pot","big_blind"]).to_pandas()
    net={(r.hand_id,r.player_id):(r.net_chips,r.folded,r.went_to_showdown) for r in seats.itertuples()}
    pot={r.hand_id:(r.final_pot,r.big_blind) for r in hands.itertuples()}
    rows=[]
    for t in dh.itertuples():
        a=net.get((t.hand_id,t.player_1)); b=net.get((t.hand_id,t.player_2)); pb=pot.get(t.hand_id,(0,1))
        if a is None or b is None: rows.append((0,0,0,0,0)); continue
        an,af,_=a; bn,bf,_=b; bb=max(pb[1],1); P=pb[0]/bb
        wnet=max(an,bn)/bb; lnet=min(an,bn)/bb
        joint=(an+bn)/bb
        winner_net=max(an,bn)/bb
        # from-outsider: winner won more than the partner lost -> difference came from the field
        from_out=(max(an,bn)+min(an,bn))/bb if (max(an,bn)>0) else 0.0  # = joint net when a member won
        partner_folded=1.0 if ((an>bn and bf) or (bn>an and af)) else 0.0
        pot_to_winner=P if max(an,bn)>0 else 0.0
        rows.append((winner_net, from_out, pot_to_winner, partner_folded, joint))
    F=pd.DataFrame(rows,columns=["ox_winner_net","ox_from_outsider","ox_pot_to_winner","ox_partner_folded","ox_joint_take"])
    for c in F.columns: dh[c]=F[c].values.astype(float)
    pos=dh[dh["label"]==1].copy(); ev=pos[pos["is_evidence"]]; nv=pos[~pos["is_evidence"]]
    print(f"labeled hands {len(dh):,} evidence {int(dh['is_evidence'].sum())}")
    print("\n=== outsider-extraction: ev vs nonev mean + single-feature within-pair MAP@5 (beat 0.24!) ===")
    for c in F.columns:
        m=map5(pos.assign(hand_score=pos[c]))
        print(f"  {c:18s} ev {ev[c].mean():8.2f} nonev {nv[c].mean():8.2f} sep {ev[c].mean()-nv[c].mean():+8.2f} | MAP@5 {m:.4f}")
    print("\n=== per family (ox_joint_take / ox_pot_to_winner ev vs nonev) ===")
    for fam in TARGET_BEHAVIORS:
        f=pos[pos["behavior_family"]==fam]; fe=f[f["is_evidence"]]; fn=f[~f["is_evidence"]]
        print(f"  {fam:22s} joint_take {fe['ox_joint_take'].mean():7.2f}/{fn['ox_joint_take'].mean():7.2f} | pot_to_winner {fe['ox_pot_to_winner'].mean():7.2f}/{fn['ox_pot_to_winner'].mean():7.2f}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
