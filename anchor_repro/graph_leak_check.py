"""§208 — Is the AUC~1.0 graph-degree signal (graph_probe.py) REAL or a labeled-set SELECTION artifact?
Leak-until-proven (§200: any gain >+0.02, let alone AUC 1.0, is a leak until shown otherwise). Direct checks:
(A) Degree distribution: dev-labeled POS vs NEG vs the FULL dev candidate set vs the EVAL candidate set. If
    dev-labeled pairs (both pos AND neg) sit in a different degree regime than the eval scored population,
    the feature is calibrated to the labeled-set construction and won't transfer (the §194 regime trap).
(B) The killer: within the EVAL candidate graph, degree cannot separate a label we don't have — but we can
    check whether the DEV degree signal is an artifact of labeled-pair SAMPLING by comparing: do dev-labeled
    pairs even have the same degree distribution as random dev candidate pairs? If dev-labeled pairs are a
    low-degree SUBSET of dev candidates, the whole 'degree->label' is 'this pair was selected for labeling',
    not 'this pair colludes'.
Run:  python -m anchor_repro.graph_leak_check
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from collections import defaultdict
D=Path("data/poker"); STEP3=Path("outputs/poker_collusion/hosen42_step3")

def degmap(dpf):
    nbr=defaultdict(set)
    for a,b in zip(dpf["player_1"],dpf["player_2"]): nbr[a].add(b); nbr[b].add(a)
    return {p:len(s) for p,s in nbr.items()}

def pairdeg(dpf,deg):
    return np.array([max(deg.get(a,0),deg.get(b,0)) for a,b in zip(dpf["player_1"],dpf["player_2"])])

def main():
    dpf=pl.read_parquet(STEP3/"dev_pair_features.parquet").select(["pair_id","player_1","player_2"]).to_pandas()
    evf=pl.read_parquet(STEP3/"eval_pair_features.parquet").select(["pair_id","player_1","player_2"]).to_pandas()
    lab=pl.read_csv(D/"development_labels.csv").to_pandas()
    ddeg=degmap(dpf); edeg=degmap(evf)

    # (A) degree regimes
    lab_pos=lab[lab["label"]==1]; lab_neg=lab[lab["label"]==0]
    pd_all=pairdeg(dpf,ddeg); pe_all=pairdeg(evf,edeg)
    pd_pos=pairdeg(lab_pos,ddeg); pd_neg=pairdeg(lab_neg,ddeg)
    print("=== (A) max-degree regime by group ===")
    for name,arr in [("dev-LABELED-pos",pd_pos),("dev-LABELED-neg",pd_neg),("dev-ALL-candidate",pd_all),("EVAL-ALL-candidate",pe_all)]:
        print(f"  {name:20s} mean {arr.mean():5.1f} median {np.median(arr):4.0f} p10 {np.percentile(arr,10):4.0f} p90 {np.percentile(arr,90):4.0f}")

    # (B) are labeled pairs a low-degree SUBSET of dev candidates? (selection artifact)
    print("\n=== (B) selection-artifact test ===")
    print(f"  dev-ALL-candidate max-deg mean {pd_all.mean():.1f}; dev-LABELED (pos+neg) mean {pairdeg(lab,ddeg).mean():.1f}")
    labeled_ids=set(lab['pair_id']); dpf_lab=dpf[dpf['pair_id'].isin(labeled_ids)]; dpf_unlab=dpf[~dpf['pair_id'].isin(labeled_ids)]
    print(f"  labeled dev pairs max-deg mean {pairdeg(dpf_lab,ddeg).mean():.1f} | UNLABELED dev pairs mean {pairdeg(dpf_unlab,ddeg).mean():.1f}")
    print("  -> if LABELED (pos+neg alike) are a distinct low-degree subset, degree encodes 'was labeled', not 'colludes'.")
    # crucial: does degree separate pos from neg WITHIN the labeled set only because BOTH are low vs the field,
    # or is pos genuinely lower than neg? (already saw AUC~1.0; show the actual medians)
    print(f"\n  pos median max-deg {np.median(pd_pos):.0f} | neg median max-deg {np.median(pd_neg):.0f} | eval median {np.median(pe_all):.0f}")
    print("  KEY: if eval candidates sit at the FIELD degree (high), a model keyed on low-degree=pos scores")
    print("  ~ALL eval pairs as negative -> the AUC 1.0 does NOT transfer (labeled-set construction artifact).")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
