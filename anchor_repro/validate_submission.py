"""Validate a submission CSV against the competition schema + our byte-reference.
Usage: python -m anchor_repro.validate_submission outputs/poker_collusion/evidence_familycond_submission.csv
"""
from __future__ import annotations
import sys, hashlib, pandas as pd

COLS=["pair_id","risk_score","predicted_behavior","evidence_hand_1","evidence_hand_2","evidence_hand_3","evidence_hand_4","evidence_hand_5"]
BEHAVIORS={"none","directed_transfer","soft_play","coordinated_isolation"}
EV=[f"evidence_hand_{i}" for i in range(1,6)]
REF_SHA="2ba41953a89fa752"

def main(path):
    df=pd.read_csv(path)
    assert list(df.columns)==COLS, f"columns {df.columns.tolist()}"
    assert df["pair_id"].is_unique, "pair_id not unique"
    assert df["risk_score"].between(0,1).all(), "risk_score out of [0,1]"
    assert set(df["predicted_behavior"].unique())<=BEHAVIORS, "bad behavior label"
    assert df[EV].isna().sum().sum()==0, "null evidence cells"
    # no duplicate non-sentinel evidence hand within a pair
    dups=0
    for _,r in df[EV].iterrows():
        vals=[v for v in r.tolist() if v!="NO_EVIDENCE"]
        if len(vals)!=len(set(vals)): dups+=1
    assert dups==0, f"{dups} rows with duplicate evidence hands"
    sha=hashlib.sha256(open(path,"rb").read()).hexdigest()
    print(f"VALID: {len(df):,} rows, schema OK, risk in [0,1], behaviors OK, no dup/null evidence")
    print(f"behavior counts: {df['predicted_behavior'].value_counts().to_dict()}")
    print(f"sha256 prefix: {sha[:16]}  (reference {REF_SHA}: {'MATCH' if sha.startswith(REF_SHA) else 'differs'})")
    return 0

if __name__=="__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv)>1 else "outputs/poker_collusion/evidence_familycond_submission.csv"))
