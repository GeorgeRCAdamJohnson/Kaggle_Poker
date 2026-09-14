# -*- coding: utf-8 -*-
# LOOP VECTORIZED: optimize 0.44519 best via interaction/structural features. Gates decide.
# GATE A drift<0.65 ; GATE B holdout on top of foundation(v5+MFg+DIRc) >= +0.010. Greedy stack A-passers.
# Fully vectorized (no per-hand python loop). Writes _loopv.txt. NO submission from here.
import warnings; warnings.filterwarnings('ignore')
import traceback, json
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd, polars as pl
OUT=Path('_loopv.txt'); OUT.write_text('',encoding='utf-8')
def log(m):
    open(OUT,'a',encoding='utf-8').write(str(m)+'\n')
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier
    D=Path('data/poker'); V5=Path('outputs/poker_collusion/feature_cache/v5'); V2=Path('outputs/poker_collusion/feature_cache/v2'); V7=Path('outputs/poker_collusion/feature_cache/v7')
    made=pd.read_parquet(V7/'showdown_made_strength.parquet'); made_map=made.set_index(['hand_id','player_id'])['made'].to_dict()
    a=pl.scan_parquet(D/'actions.parquet').select(['hand_id','action_no','player_id','action','street'])
    aggr=(a.filter(pl.col('action').is_in(['bet','raise','all_in'])).group_by('hand_id').agg(pl.col('player_id').sort_by('action_no').last().alias('last_aggr')))
    smap={'preflop':0,'flop':1,'turn':2,'river':3}
    folds=(a.filter(pl.col('action')=='fold').select(['hand_id','player_id','street']).rename({'player_id':'folder'})
             .join(aggr,on='hand_id',how='left').with_columns(pl.col('street').replace_strict(smap,default=0).alias('st_idx'))).collect().to_pandas()
    seats=pl.scan_parquet(D/'seats.parquet').select(['hand_id','player_id','seat_no','net_chips','won_share'])
    hands=pl.scan_parquet(D/'hands.parquet').select(['hand_id','table_id','phase'])
    winner=(seats.filter(pl.col('won_share')>0).group_by('hand_id').agg(pl.col('player_id').sort_by('won_share').last().alias('winner'))).collect().to_pandas()
    def build_inter(phase,pairs_df):
        # per-hand pair expansion (vectorized): all 15 seat combos joined to target pairs
        hp=(seats.join(hands.filter(pl.col('phase')==phase).select(['hand_id']),on='hand_id')
              .group_by('hand_id').agg(pl.col('player_id').sort_by('seat_no').alias('pl')))
        fr=[hp.select(['hand_id',pl.col('pl').list.get(i).alias('a'),pl.col('pl').list.get(j).alias('b')]) for i,j in combinations(range(6),2)]
        pj=pl.concat(fr).with_columns(pl.min_horizontal('a','b').alias('p_low'),pl.max_horizontal('a','b').alias('p_high')).drop(['a','b'])
        pj=pj.join(pl.from_pandas(pairs_df[['pair_id','p_low','p_high']]).lazy(),on=['p_low','p_high'],how='inner').collect().to_pandas()
        nc=seats.collect().to_pandas()[['hand_id','player_id','net_chips']]
        # merge net_chips for low and high
        pj=pj.merge(nc.rename(columns={'player_id':'p_low','net_chips':'n1'}),on=['hand_id','p_low'],how='left')
        pj=pj.merge(nc.rename(columns={'player_id':'p_high','net_chips':'n2'}),on=['hand_id','p_high'],how='left')
        pj['n1']=pj['n1'].fillna(0); pj['n2']=pj['n2'].fillna(0)
        # made strength
        pj['m1']=[made_map.get((h,p),0) for h,p in zip(pj['hand_id'],pj['p_low'])]
        pj['m2']=[made_map.get((h,p),0) for h,p in zip(pj['hand_id'],pj['p_high'])]
        # winner
        wmap=dict(zip(winner['hand_id'],winner['winner']))
        pj['win']=pj['hand_id'].map(wmap)
        # fold-to-partner: build map hand->list(folder,aggr,st)
        fmap={}
        for h,fo,ag,st in zip(folds['hand_id'],folds['folder'],folds['last_aggr'],folds['st_idx']):
            fmap.setdefault(h,[]).append((fo,ag,st))
        # vectorized-ish surrender side via apply on grouped small lists (per row lookup)
        surr_side=np.zeros(len(pj)); fold_st=np.full(len(pj),-1)
        hid=pj['hand_id'].values; lo=pj['p_low'].values; hi=pj['p_high'].values
        for k in range(len(pj)):
            for (fo,ag,st) in fmap.get(hid[k],()):
                if fo==lo[k] and ag==hi[k]: surr_side[k]=-1; fold_st[k]=st
                elif fo==hi[k] and ag==lo[k]: surr_side[k]=1; fold_st[k]=st
        pj['surr_side']=surr_side; pj['fold_st']=fold_st
        pj['wa']=(pj['win']==pj['p_low']).astype(int); pj['wb']=(pj['win']==pj['p_high']).astype(int)
        # interaction primitives
        pj['strong_surr']=np.where((pj['surr_side']==-1)&(pj['wb']==1)&(pj['m1']>=200),pj['m1'],
                          np.where((pj['surr_side']==1)&(pj['wa']==1)&(pj['m2']>=200),pj['m2'],0.0))
        pj['benef']=np.where((pj['surr_side']==-1)&(pj['wb']==1),1,np.where((pj['surr_side']==1)&(pj['wa']==1),-1,0))
        def agg(g):
            i1=(g['strong_surr']>0).mean(); i1sev=np.log1p(g['strong_surr'].sum())
            bpos=(g['benef']==1).sum(); bneg=(g['benef']==-1).sum(); recip=abs(bpos-bneg)/(bpos+bneg+1)
            sts=g.loc[g['surr_side']!=0,'fold_st']
            if len(sts)>0:
                vc=sts.value_counts(normalize=True).values; ent=-(vc*np.log(vc+1e-9)).sum()/np.log(4)
            else: ent=1.0
            timing=(1-ent)*recip
            pwr=(g['wa']|g['wb']).mean(); exc=pwr-(2.0/6.0)
            return pd.Series({'i1_strong_surr_rate':i1,'i1_strong_surr_sev':i1sev,'i2_recip_asym':recip,'i3_timing_conc':timing,'i4_excess_cowin':exc})
        ag=pj.groupby('pair_id').apply(agg).reset_index()
        return pairs_df[['pair_id']].merge(ag,on='pair_id',how='left').fillna(0)
    dp=pd.read_parquet(V2/'dev_pairs_v2.parquet')
    dev=pd.read_parquet(V5/'dev_v5.parquet').merge(dp[['pair_id','p_low','p_high','label']],on='pair_id',how='left')
    epk=pd.read_csv(D/'evaluation_pairs.csv'); epk['p_low']=epk[['player_1','player_2']].min(1); epk['p_high']=epk[['player_1','player_2']].max(1)
    ev=pd.read_parquet(V5/'eval_v5.parquet').merge(epk[['pair_id','p_low','p_high']],on='pair_id',how='left')
    log('build dev...'); Id=build_inter('development',dev); log('dev done')
    log('build eval...'); Ie=build_inter('evaluation',ev); log('eval done')
    icols=[c for c in Id.columns if c!='pair_id']; Idv=Id[icols].to_numpy(np.float32); Iev=Ie[icols].to_numpy(np.float32)
    Mgd=np.load(V7/'_MFg_dev.npy'); DIRd=np.load(V7/'_DIRc_dev.npy')
    feats5=[c for c in dev.columns if c not in ('pair_id','p_low','p_high','label','behavior_family')]
    Xd5=dev[feats5].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    Xfound=np.hstack([Xd5,Mgd,DIRd]).astype(np.float32)
    seatsc=pl.scan_parquet(D/'seats.parquet').select(['hand_id','player_id']); handsc=pl.scan_parquet(D/'hands.parquet').select(['hand_id','table_id'])
    p2t=(seatsc.join(handsc,on='hand_id').group_by('player_id').agg(pl.col('table_id').mode().first().alias('t')).collect()); p2t=dict(zip(p2t['player_id'].to_list(),p2t['t'].to_list()))
    dev['table_id']=dev['p_low'].map(p2t)
    lab=dev['label'].to_numpy(); y=(lab==1).astype(int); known=(lab>=0)
    n_pu=int((lab==-1).sum()); sw=np.where(known,1.0,112540/max(n_pu,1)); fw=np.where(y==1,2.0,np.where(known,1.0,0.35))
    tabs=sorted(set(str(t) for t in dev['table_id'].fillna('NA'))); rng=np.random.default_rng(7); rng.shuffle(tabs); cut=int(len(tabs)*0.6)
    tr=dev['table_id'].astype(str).isin(set(tabs[:cut])).to_numpy(); ho=~tr
    def hap(X):
        m=XGBClassifier(n_estimators=700,learning_rate=0.03,max_depth=4,min_child_weight=5,subsample=0.85,colsample_bytree=0.8,reg_lambda=6,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',n_jobs=-1,random_state=42)
        m.fit(X[tr],y[tr],sample_weight=fw[tr]); return average_precision_score(y[ho],m.predict_proba(X[ho])[:,1],sample_weight=sw[ho])
    base=hap(Xfound); log('foundation holdout AP: %.4f'%base)
    apass=[]
    for k,c in enumerate(icols):
        xa=np.vstack([Idv[:,k:k+1],Iev[:,k:k+1]]); ya=np.r_[np.zeros(len(Idv)),np.ones(len(Iev))]; idx=np.random.default_rng(0).permutation(len(xa))
        av=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=0.1,eval_metric='auc',tree_method='hist',n_jobs=-1,random_state=0)
        av.fit(xa[idx[:len(idx)//2]],ya[idx[:len(idx)//2]]); auc=roc_auc_score(ya[idx[len(idx)//2:]],av.predict_proba(xa[idx[len(idx)//2:]])[:,1])
        d=hap(np.hstack([Xfound,Idv[:,k:k+1]]).astype(np.float32))-base
        Ap=auc<0.65; log('feat %-22s drift %.3f holdout %+.4f A=%s B=%s'%(c,auc,d,'P' if Ap else 'F','P' if d>=0.010 else 'F'))
        if Ap: apass.append(k)
    log('A-passing: '+str([icols[k] for k in apass]))
    chosen=[]; cur=base; curX=Xfound; rem=list(apass)
    while rem:
        best=None
        for k in rem:
            ap=hap(np.hstack([curX,Idv[:,k:k+1]]).astype(np.float32))
            if best is None or ap>best[1]: best=(k,ap)
        if best[1]-cur>0.0005:
            chosen.append(best[0]); curX=np.hstack([curX,Idv[:,best[0]:best[0]+1]]).astype(np.float32)
            log('+ stack %-22s cum holdout %.4f (d %+.4f)'%(icols[best[0]],best[1],best[1]-base)); cur=best[1]; rem.remove(best[0])
        else: break
    log('FINAL stacked: '+str([icols[k] for k in chosen]))
    log('foundation %.4f -> stacked %.4f (delta %+.4f)'%(base,cur,cur-base))
    log('NOTE: HOLDOUT AP delta, NOT calibrated LB. 0.60 bar not asserted from this.')
    if chosen:
        np.save(V7/'_INTER_dev.npy',Idv[:,chosen]); np.save(V7/'_INTER_eval.npy',Iev[:,chosen]); (V7/'_INTER_cols.json').write_text(json.dumps([icols[k] for k in chosen]))
        log('saved chosen interaction feats')
    log('DONE')
except Exception as e:
    log('ERROR: '+repr(e)); log(traceback.format_exc())
