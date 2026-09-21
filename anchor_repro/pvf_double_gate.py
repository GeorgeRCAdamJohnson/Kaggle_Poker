"""Double-gate evaluator for PVF channel sets: dev-CV eval-mirrored PairAP AND held-out-table PairAP.

Lesson §140: dev-CV misranked the crude EV-loss proxy at +0.0017 while the LB said -0.0009. dev-CV
resolution floor is ~±0.001 at this level, so we add a SECOND validator: hold 2 of 5 table-folds
FULLY out of training and measure eval-mirrored PairAP on those held-out tables (positives vs ALL
eligible pairs on unseen tables). A channel must clear BOTH the full-CV and the held-out gate above
the postflop+tight base before it earns a submission.

Usage: eval_channel_set(extra_feats) where extra_feats is a subset of pvf columns to ADD to the
hosen42 baseline 221 feats. Reports full-CV AP + held-out AP for baseline, base+postflop+tight
(the 0.81092 config), and base+the given extra set.
Run:  python -m anchor_repro.pvf_double_gate  [names of pvf feats to test, default = all]
"""
from __future__ import annotations
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score
from anchor_repro.policy_edge_pvf import pvf_features, PVF_FEATS
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"
SEED,N_FOLDS=42,5
POSTFLOP_TIGHT=["post_loose_pf_sum","post_loose_pf_min","total_loose_pf_sum","total_loose_pf_min",
                "both_post_min_top3","both_post_min_mean","both_post_surp_ct","tight_pf_sum","tight_pf_max"]  # the 0.81092 config
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def _prep():
    dev=pd.read_parquet(STEP3/"dev_pair_features.parquet")
    de=pvf_features(STEP3/"dev_pair_hands.parquet","development").to_pandas()
    dev=dev.merge(de,on="pair_id",how="left")
    for c in PVF_FEATS: dev[c]=dev[c].astype("float32").fillna(0)
    return dev

def _two_stage_oof(dev, feats, folds, y, labm, heldout_folds):
    """OOF over ALL folds (full-CV gate) but TRAIN never uses held-out folds (so held-out preds are
    truly unseen-table). Returns oof over all rows; held-out mask = rows in heldout_folds."""
    X=dev[feats].astype("float32")
    P=dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.5,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,verbose=-1,num_threads=-1)
    SEEDS=[SEED,SEED+11,SEED+23]
    heldout=np.isin(folds,heldout_folds)
    # stage-1 (single seed), train only on NON-heldout folds; predict all
    w1=np.where(labm,np.where(y==1,5.0,1.0),0.1); oof1=np.zeros(len(dev))
    for f in range(N_FOLDS):
        if f in heldout_folds: continue
        tr=(folds!=f)&(~heldout)&(w1>0); # never train on heldout
    # simpler: single model per left-out fold among the non-heldout training pool
    train_pool=~heldout
    # stage-1 to get pseudo split (CV within training pool for the non-heldout, direct predict heldout)
    for f in range(N_FOLDS):
        te=folds==f
        tr=train_pool&(folds!=f)&(w1>0)
        m=lgb.train({**P,"seed":SEED},lgb.Dataset(X[tr],y[tr],weight=w1[tr]),num_boost_round=750)
        oof1[te]=m.predict(X[te])
    pos=oof1[labm&(y==1)&train_pool]; ta,tp=np.quantile(pos,0.05),np.quantile(pos,0.50)
    unl=~labm; amb=unl&(oof1>=ta)&(oof1<tp)&train_pool; ps=unl&(oof1>=tp)&train_pool
    y2=np.where(ps,1,y); w2=np.where(labm,np.where(y==1,5.0,1.0),np.where(ps,1.0,np.where(amb,0.0,0.1)))
    w2=np.where(train_pool,w2,0.0)  # zero weight on held-out so they are never trained on
    oof=np.zeros(len(dev))
    for f in range(N_FOLDS):
        te=folds==f
        tr=train_pool&(folds!=f)&(w2>0)
        for sd in SEEDS:
            m=lgb.train({**P,"seed":sd},lgb.Dataset(X[tr],y2[tr],weight=w2[tr]),num_boost_round=750)
            oof[te]+=m.predict(X[te])/len(SEEDS)
    return oof, heldout

def main():
    dev=_prep()
    y=dev["label"].fillna(0).astype(int).to_numpy(); labm=dev["is_labeled"].to_numpy().astype(bool)
    TABLES=sorted(dev["table_id"].unique().tolist()); rng=np.random.RandomState(SEED)
    TF={t:int(v) for t,v in zip(TABLES,rng.permutation(len(TABLES))%N_FOLDS)}; folds=dev["table_id"].map(TF).to_numpy()
    META={"pair_id","table_id","player_1","player_2","label","behavior_family","is_labeled","fold","shared_hands","chunk","pred_family"}
    base_feats=[c for c in dev.columns if c not in META and not c.startswith("sus_") and c!="top5s_same_loser"
                and c not in PVF_FEATS and not c.startswith("evl_pf") and not c.startswith("deq_pf")]
    heldout_folds=[3,4]  # 2 of 5 tables fully held out for the second gate
    extra = sys.argv[1:] if len(sys.argv)>1 else PVF_FEATS

    def gate(feats,tag):
        oof,ho=_two_stage_oof(dev,feats,folds,y,labm,heldout_folds)
        full=average_precision_score(y,oof)
        hap=average_precision_score(y[ho],oof[ho])
        log(f"[{tag}] full-CV AP={full:.4f}  held-out(2 tables) AP={hap:.4f}")
        return full,hap

    log(f"base feats {len(base_feats)}; testing extra={extra}")
    b_full,b_ho=gate(base_feats,"baseline-221")
    pt_full,pt_ho=gate(base_feats+POSTFLOP_TIGHT,"base+postflop+tight (0.81092 cfg)")
    x_full,x_ho=gate(base_feats+POSTFLOP_TIGHT+[c for c in extra if c not in POSTFLOP_TIGHT],"base+PT+extra")
    print("\n=== DOUBLE GATE SUMMARY (must beat 0.81092-cfg on BOTH to earn a submission) ===")
    print(f"  baseline-221           full {b_full:.4f} | held-out {b_ho:.4f}")
    print(f"  +postflop+tight (best) full {pt_full:.4f} | held-out {pt_ho:.4f}")
    print(f"  +PT+extra              full {x_full:.4f} | held-out {x_ho:.4f}")
    print(f"  extra delta vs PT: full {x_full-pt_full:+.4f} | held-out {x_ho-pt_ho:+.4f}  "
          f"({'STAGE IT' if (x_full>pt_full+0.001 and x_ho>pt_ho+0.001) else 'do NOT stage (fails a gate)'})")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
