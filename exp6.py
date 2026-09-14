# -*- coding: utf-8 -*-
# EXP6b PU-LABEL sweep (fixed soft-label bug: use XGBRegressor for fractional labels).
# Foundation = v5+MFg+DIRc. JUDGE = pair-disjoint TABLE holdout PU-stress AP. Baseline = fixed unl=0.35.
# c (Elkan-Noto) ~0.3883 already estimated. Require holdout >= +0.010 to consider; hold-unless-0.60.
import warnings; warnings.filterwarnings('ignore')
import traceback
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
OUT=Path('_exp6.txt'); OUT.write_text('',encoding='utf-8')
def log(m): open(OUT,'a',encoding='utf-8').write(str(m)+'\n')
try:
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier, XGBRegressor
    D=Path('data/poker'); V5=Path('outputs/poker_collusion/feature_cache/v5'); V2=Path('outputs/poker_collusion/feature_cache/v2'); V7=Path('outputs/poker_collusion/feature_cache/v7')
    dp=pd.read_parquet(V2/'dev_pairs_v2.parquet')
    dev=pd.read_parquet(V5/'dev_v5.parquet').merge(dp[['pair_id','p_low','label']],on='pair_id',how='left')
    Mgd=np.load(V7/'_MFg_dev.npy'); DIRd=np.load(V7/'_DIRc_dev.npy')
    feats5=[c for c in dev.columns if c not in ('pair_id','p_low','label','behavior_family')]
    Xd5=dev[feats5].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    X=np.hstack([Xd5,Mgd,DIRd]).astype(np.float32)
    lab=dev['label'].to_numpy(); y=(lab==1).astype(int); known=(lab>=0); unl=(lab==-1)
    seatsc=pl.scan_parquet(D/'seats.parquet').select(['hand_id','player_id']); handsc=pl.scan_parquet(D/'hands.parquet').select(['hand_id','table_id'])
    p2t=(seatsc.join(handsc,on='hand_id').group_by('player_id').agg(pl.col('table_id').mode().first().alias('t')).collect()); p2t=dict(zip(p2t['player_id'].to_list(),p2t['t'].to_list()))
    dev['table_id']=dev['p_low'].map(p2t)
    n_pu=int(unl.sum()); sw=np.where(known,1.0,112540/max(n_pu,1))
    tabs=sorted(set(str(t) for t in dev['table_id'].fillna('NA'))); rng=np.random.default_rng(7); rng.shuffle(tabs); cut=int(len(tabs)*0.6)
    tr=dev['table_id'].astype(str).isin(set(tabs[:cut])).to_numpy(); ho=~tr
    def clf(fw):
        m=XGBClassifier(n_estimators=700,learning_rate=0.03,max_depth=4,min_child_weight=5,subsample=0.85,colsample_bytree=0.8,reg_lambda=6,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',n_jobs=-1,random_state=42)
        m.fit(X[tr],y[tr],sample_weight=fw[tr]); return average_precision_score(y[ho],m.predict_proba(X[ho])[:,1],sample_weight=sw[ho])
    fw0=np.where(y==1,2.0,np.where(known,1.0,0.35)); base=clf(fw0); log('BASELINE (fixed unl=0.35) holdout: %.4f'%base)
    # Elkan-Noto c
    g=XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=3,subsample=0.9,colsample_bytree=0.8,reg_lambda=5,objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=-1,random_state=1)
    g.fit(X[tr],(lab==1).astype(int)[tr]); vp=(lab[ho]==1); c=g.predict_proba(X[ho])[:,1][vp].mean() if vp.sum()>0 else 0.5
    log('Elkan-Noto c: %.4f'%c)
    results={'baseline_0.35':base}
    # weight sweeps
    for w in [0.15,0.2,0.25,0.5]:
        results['unl_%.2f'%w]=clf(np.where(y==1,2.0,np.where(known,1.0,w)))
        log('scheme unl_%.2f       holdout %.4f (d %+.4f)'%(w,results['unl_%.2f'%w],results['unl_%.2f'%w]-base))
    results['pos3_unl0.2']=clf(np.where(y==1,3.0,np.where(known,1.0,0.2))); log('scheme pos3_unl0.2     holdout %.4f (d %+.4f)'%(results['pos3_unl0.2'],results['pos3_unl0.2']-base))
    wen=float(np.clip((1-c)/max(c,1e-3),0.05,1.5)); results['en_c_scaled']=clf(np.where(y==1,2.0,np.where(known,1.0,wen))); log('scheme en_c_scaled(w=%.3f) holdout %.4f (d %+.4f)'%(wen,results['en_c_scaled'],results['en_c_scaled']-base))
    # FIXED soft-label scheme via REGRESSOR (fractional labels ok)
    pu=g.predict_proba(X)[:,1]; frac=np.clip((1-c)/max(c,1e-3)*pu,0,1)
    yb=y.astype(float).copy(); yb[unl]=frac[unl]
    r=XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=4,min_child_weight=5,subsample=0.85,colsample_bytree=0.8,reg_lambda=6,objective='reg:logistic',eval_metric='logloss',tree_method='hist',n_jobs=-1,random_state=42)
    r.fit(X[tr],yb[tr],sample_weight=fw0[tr]); ap=average_precision_score(y[ho],r.predict(X[ho]),sample_weight=sw[ho])
    results['en_softlabel_reg']=ap; log('scheme en_softlabel_reg holdout %.4f (d %+.4f)'%(ap,ap-base))
    best=max(results,key=results.get); log('BEST: %s = %.4f (delta %+.4f)'%(best,results[best],results[best]-base))
    log('GATE B (>=+0.010): %s'%('PASS' if (results[best]-base)>=0.010 else 'FAIL'))
    log('NOTE: holdout AP delta, NOT calibrated LB. hold-unless-0.60.')
    log('DONE')
except Exception as e:
    log('ERROR: '+repr(e)); log(traceback.format_exc())
