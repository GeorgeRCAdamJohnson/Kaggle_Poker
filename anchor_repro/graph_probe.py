"""§208 — GRAPH/GNN existence-probe (context-free agent's idea #1, the last untested lever). Before any GNN
build, the cheap question: does an EVAL-COMPUTABLE global-graph feature separate the label? Everything global
tried so far was label-only (rings §199 coincidence, timing §205 label-only). A GNN is only worth building if
the co-play graph carries pair-label signal computable WITHOUT labels.

Build the full candidate co-play graph over ALL dev labeled pairs (nodes=players, edges=candidate pairs), and
compute eval-computable topology features per pair:
  g_deg_max/min : max/min player degree (how many candidate partners each has)
  g_deg_gap     : |deg1-deg2|
  g_common_nbr  : shared neighbors of the two players (triangle potential)
  g_jaccard     : Jaccard of the two players' neighbor sets
  g_clust       : mean local clustering coeff of the two nodes
  g_pref_attach : deg1*deg2 (preferential attachment)
All label-free. Test AUC vs label on the 1860 dev labeled pairs + corr(shared_hands) leak check + drift-safety
(dev/eval mean match). Pre-registered: max AUC <0.55 => graph is label-only/coincidence too, GNN NOT worth
building. >0.60 clean (not shared_hands leak) => real, GNN worth scoping.
Run:  python -m anchor_repro.graph_probe
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3")

def main():
    # full candidate graph from dev_pair_features (all ~122k candidate pairs = the real co-play graph)
    dpf=pl.read_parquet(STEP3/"dev_pair_features.parquet").select(["pair_id","player_1","player_2"]).to_pandas()
    lab=pl.read_csv(D/"development_labels.csv").to_pandas()  # 1860 labeled pairs w/ label
    # adjacency over ALL candidate pairs (label-free)
    nbr=defaultdict(set)
    for a,b in zip(dpf["player_1"],dpf["player_2"]): nbr[a].add(b); nbr[b].add(a)
    deg={p:len(s) for p,s in nbr.items()}
    print(f"co-play graph: {len(deg)} players, {len(dpf)} candidate edges; deg mean {np.mean(list(deg.values())):.1f} max {max(deg.values())}")

    def feats(a,b):
        da,db=deg.get(a,0),deg.get(b,0)
        na,nb=nbr.get(a,set()),nbr.get(b,set())
        common=len(na&nb); union=len(na|nb)
        jac=common/union if union else 0.0
        # local clustering of a: edges among a's neighbors / possible
        return da,db,abs(da-db),common,jac,da*db
    rows=[feats(a,b) for a,b in zip(lab["player_1"],lab["player_2"])]
    G=pd.DataFrame(rows,columns=["g_deg1","g_deg2","g_deg_gap","g_common_nbr","g_jaccard","g_pref_attach"])
    G["g_deg_max"]=G[["g_deg1","g_deg2"]].max(axis=1); G["g_deg_min"]=G[["g_deg1","g_deg2"]].min(axis=1)
    y=lab["label"].to_numpy()
    # shared_hands for leak check
    sh=pl.read_parquet(STEP3/"dev_pairs.parquet").select(["pair_id","shared_hands"]).to_pandas()
    lab=lab.merge(sh,on="pair_id",how="left")
    print("\n=== eval-computable graph features: label AUC + corr(shared_hands) ===")
    GF=["g_deg_max","g_deg_min","g_deg_gap","g_common_nbr","g_jaccard","g_pref_attach"]
    best=0.5
    for c in GF:
        v=G[c].fillna(0).to_numpy().astype(float)
        a=roc_auc_score(y,v); csh=np.corrcoef(v,lab["shared_hands"].fillna(0))[0,1]
        aa=max(a,1-a); best=max(best,aa)
        print(f"  {c:14s} AUC {aa:.4f} ({'hi=pos' if a>0.5 else 'lo=pos'})  corr(shared_hands) {csh:+.3f}")
    print(f"\n=== PRE-REGISTERED verdict ===")
    if best<0.55: print(f"  max AUC {best:.3f} < 0.55 -> graph topology is NOT a label signal. GNN NOT worth building. CLOSED.")
    elif best>0.60: print(f"  max AUC {best:.3f} > 0.60 -> real graph signal (verify not shared_hands leak). GNN worth scoping.")
    else: print(f"  max AUC {best:.3f} in [0.55,0.60] -> marginal; report, lean CLOSE.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
