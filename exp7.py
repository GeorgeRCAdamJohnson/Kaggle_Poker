# -*- coding: utf-8 -*-
# EXP7 LAYER 2 SUPPRESSION/BACK-OFF (build ON TOP of 0.44519 foundation; nothing dropped).
# Signal = A and B CONTEST each other LESS than expected. A 'clash' = both active in a hand and one
# faces the other's aggression (partner bet/raised, this player has to_call>0 on same/next action same hand).
# suppression = expected_clashes - observed_clashes, normalized by co-active opportunity.
# Plus context: mutual-aggression rate when a STRONG outsider present vs absent (back-off tell).
# TESTS:
#  (X) CROSS-PREDICT: does this LABEL-FREE layer rank known labeled positives above chance? (AUC vs 0.5)
#  (A) drift < 0.65
#  (B) holdout on foundation >= +0.010
# Writes _exp7.txt. NO submission.
import warnings; warnings.filterwarnings('ignore')
import traceback
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd, polars as pl
OUT=Path('_exp7.txt'); OUT.write_text('',encoding='utf-8')
def log(m): open(OUT,'a',encoding='utf-8').write(str(m)+'\n')
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier
    D=Path('data/poker'); V5=Path('outputs/poker_collusion/feature_cache/v5'); V2=Path('outputs/poker_collusion/feature_cache/v2'); V7=Path('outputs/poker_collusion/feature_cache/v7')
    # --- per-hand: who was active postflop, who aggressed, who faced aggression ---
    a=pl.scan_parquet(D/'actions.parquet').select(['hand_id','action_no','street','player_id','action','to_call'])
    # players active postflop per hand (appear in flop/turn/river)
    post=a.filter(pl.col('street').is_in(['flop','turn','river']))
    active=post.group_by('hand_id').agg(pl.col('player_id').unique().alias('act')).collect()
    # aggressors postflop per hand
    aggr=post.filter(pl.col('action').is_in(['bet','raise','all_in'])).group_by('hand_id').agg(pl.col('player_id').unique().alias('agg')).collect()
    # players who FACED aggression (to_call>0 postflop)
    faced=post.filter(pl.col('to_call')>0).group_by('hand_id').agg(pl.col('player_id').unique().alias('faced')).collect()
    amap=dict(zip(active['hand_id'].to_list(),active['act'].to_list()))
    ggmap=dict(zip(aggr['hand_id'].to_list(),aggr['agg'].to_list()))
    fcmap=dict(zip(faced['hand_id'].to_list(),faced['faced'].to_list()))
    seats=pl.scan_parquet(D/'seats.parquet').select(['hand_id','player_id','seat_no'])
    hands=pl.scan_parquet(D/'hands.parquet').select(['hand_id','table_id','phase'])
    def build_supp(phase, pairs_df):
        hp=(seats.join(hands.filter(pl.col('phase')==phase).select(['hand_id']),on='hand_id')
              .group_by('hand_id').agg(pl.col('player_id').sort_by('seat_no').alias('pl')))
        fr=[hp.select(['hand_id',pl.col('pl').list.get(i).alias('a'),pl.col('pl').list.get(j).alias('b')]) for i,j in combinations(range(6),2)]
        pj=pl.concat(fr).with_columns(pl.min_horizontal('a','b').alias('p_low'),pl.max_horizontal('a','b').alias('p_high')).drop(['a','b'])
        pj=pj.join(pl.from_pandas(pairs_df[['pair_id','p_low','p_high']]).lazy(),on=['p_low','p_high'],how='inner').collect().to_pandas()
        hid=pj['hand_id'].values; lo=pj['p_low'].values; hi=pj['p_high'].values
        both_active=np.zeros(len(pj)); clash=np.zeros(len(pj))
        for k in range(len(pj)):
            act=amap.get(hid[k]); 
            if act is None: continue
            acts=set(act)
            if lo[k] in acts and hi[k] in acts:
                both_active[k]=1
                agg=ggmap.get(hid[k],()); fc=fcmap.get(hid[k],())
                aggs=set(agg); fcs=set(fc)
                # clash = one of pair aggressed AND the other faced aggression (they contested each other)
                if (lo[k] in aggs and hi[k] in fcs) or (hi[k] in aggs and lo[k] in fcs):
                    clash[k]=1
        pj['both_active']=both_active; pj['clash']=clash
        def agg_p(g):
            ba=g['both_active'].sum(); cl=g['clash'].sum()
            # observed clash rate given both active postflop
            obs_rate=cl/max(ba,1)
            # suppression = 1 - observed clash rate (high = they avoid each other), weighted by opportunity
            supp=(1.0-obs_rate)
            supp_w=supp*np.log1p(ba)   # more meaningful with more co-active hands
            return pd.Series({'s_clash_rate':obs_rate,'s_suppression':supp,'s_supp_weighted':supp_w,'s_coactive':np.log1p(ba)})
        ag=pj.groupby('pair_id').apply(agg_p).reset_index()
        return pairs_df[['pair_id']].merge(ag,on='pair_id',how='left').fillna(0)
    dp=pd.read_parquet(V2/'dev_pairs_v2.parquet')
    dev=pd.read_parquet(V5/'dev_v5.parquet').merge(dp[['pair_id','p_low','p_high','label']],on='pair_id',how='left')
    epk=pd.read_csv(D/'evaluation_pairs.csv'); epk['p_low']=epk[['player_1','player_2']].min(1); epk['p_high']=epk[['player_1','player_2']].max(1)
    ev=pd.read_parquet(V5/'eval_v5.parquet').merge(epk[['pair_id','p_low','p_high']],on='pair_id',how='left')
    log('build suppression dev...'); Sd=build_supp('development',dev); log('dev done')
    log('build suppression eval...'); Se=build_supp('evaluation',ev); log('eval done')
    scols=[c for c in Sd.columns if c!='pair_id']; Sdv=Sd[scols].to_numpy(np.float32); Sev=Se[scols].to_numpy(np.float32)
    log('cols: '+str(scols))
    # (X) CROSS-PREDICT: do LABEL-FREE suppression feats rank known positives above chance?
    lab=dev['label'].to_numpy(); known=(lab>=0); yk=(lab[known]==1).astype(int)
    for k,c in enumerate(scols):
        au=roc_auc_score(yk,Sdv[known,k]) if len(np.unique(yk))>1 else 0.5
        log('  X-predict %-16s AUC vs known labels: %.3f'%(c,au))
    # (A) drift
    Xav=np.vstack([Sdv,Sev]); yav=np.r_[np.zeros(len(Sdv)),np.ones(len(Sev))]; idx=np.random.default_rng(0).permutation(len(Xav))
    av=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.1,eval_metric='auc',tree_method='hist',n_jobs=-1,random_state=0); av.fit(Xav[idx[:len(idx)//2]],yav[idx[:len(idx)//2]])
    dA=roc_auc_score(yav[idx[len(idx)//2:]],av.predict_proba(Xav[idx[len(idx)//2:]])[:,1]); log('GATE A drift AUC: %.3f -> %s'%(dA,'PASS' if dA<0.65 else 'FAIL'))
    # (B) holdout on foundation
    Mgd=np.load(V7/'_MFg_dev.npy'); DIRd=np.load(V7/'_DIRc_dev.npy')
    feats5=[c for c in dev.columns if c not in ('pair_id','p_low','p_high','label','behavior_family')]
    Xd5=dev[feats5].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    Xf=np.hstack([Xd5,Mgd,DIRd]).astype(np.float32)
    seatsc=pl.scan_parquet(D/'seats.parquet').select(['hand_id','player_id']); handsc=pl.scan_parquet(D/'hands.parquet').select(['hand_id','table_id'])
    p2t=(seatsc.join(handsc,on='hand_id').group_by('player_id').agg(pl.col('table_id').mode().first().alias('t')).collect()); p2t=dict(zip(p2t['player_id'].to_list(),p2t['t'].to_list()))
    dev['table_id']=dev['p_low'].map(p2t)
    y=(lab==1).astype(int); n_pu=int((lab==-1).sum()); sw=np.where(known,1.0,112540/max(n_pu,1)); fw=np.where(y==1,2.0,np.where(known,1.0,0.35))
    tabs=sorted(set(str(t) for t in dev['table_id'].fillna('NA'))); rng=np.random.default_rng(7); rng.shuffle(tabs); cut=int(len(tabs)*0.6)
    tr=dev['table_id'].astype(str).isin(set(tabs[:cut])).to_numpy(); ho=~tr
    def hap(X):
        m=XGBClassifier(n_estimators=700,learning_rate=0.03,max_depth=4,min_child_weight=5,subsample=0.85,colsample_bytree=0.8,reg_lambda=6,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',n_jobs=-1,random_state=42)
        m.fit(X[tr],y[tr],sample_weight=fw[tr]); return average_precision_score(y[ho],m.predict_proba(X[ho])[:,1],sample_weight=sw[ho])
    base=hap(Xf); new=hap(np.hstack([Xf,Sdv]).astype(np.float32)); log('GATE B: foundation %.4f +supp %.4f delta %+.4f -> %s'%(base,new,new-base,'PASS' if (new-base)>=0.010 else 'FAIL'))
    np.save(V7/'_SUPP_dev.npy',Sdv); np.save(V7/'_SUPP_eval.npy',Sev)
    log('saved suppression feats (for concordance layer regardless of B)')
    log('DONE')
except Exception as e:
    log('ERROR: '+repr(e)); log(traceback.format_exc())
