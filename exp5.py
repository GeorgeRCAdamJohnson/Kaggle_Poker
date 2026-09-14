# -*- coding: utf-8 -*-
# EXP5 TRIADIC/RING: encode 3+ player ring-collusion signal onto each pair. Gates decide.
# For pair (A,B): shared-beneficiary (both feed same 3rd), ring-beneficiary inflow concentration,
# feeder-cluster membership. All per-table normalized (no population leak). GATE A drift<0.65,
# GATE B holdout on foundation(v5+MFg+DIRc) >= +0.010. NO submission from here. Writes _exp5.txt.
import warnings; warnings.filterwarnings('ignore')
import traceback, json
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd, polars as pl
OUT=Path('_exp5.txt'); OUT.write_text('',encoding='utf-8')
def log(m): open(OUT,'a',encoding='utf-8').write(str(m)+'\n')
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier
    D=Path('data/poker'); V5=Path('outputs/poker_collusion/feature_cache/v5'); V2=Path('outputs/poker_collusion/feature_cache/v2'); V7=Path('outputs/poker_collusion/feature_cache/v7')
    seats=pl.scan_parquet(D/'seats.parquet').select(['hand_id','player_id','net_chips','won_share'])
    hands=pl.scan_parquet(D/'hands.parquet').select(['hand_id','table_id','phase'])
    winner=(seats.filter(pl.col('won_share')>0).group_by('hand_id').agg(pl.col('player_id').sort_by('won_share').last().alias('winner'))).collect().to_pandas()
    def build_ring(phase, pairs_df):
        s=seats.join(hands.filter(pl.col('phase')==phase).select(['hand_id','table_id']),on='hand_id').collect().to_pandas()
        wmap=dict(zip(winner['hand_id'],winner['winner']))
        s['winner']=s['hand_id'].map(wmap)
        # directed flow: in each hand, losers (net<0) feed the winner. Build (feeder->beneficiary) chip sums per table.
        s['is_win']=(s['player_id']==s['winner'])
        # winner net per hand
        win_net=s[s['is_win']].groupby('hand_id')['net_chips'].sum().to_dict()
        # feeder rows: net_chips<0
        losers=s[s['net_chips']<0].copy(); losers['ben']=losers['hand_id'].map(wmap)
        losers=losers[losers['ben'].notna() & (losers['ben']!=losers['player_id'])]
        # edge weight feeder->ben = -net_chips (chips lost while ben won that hand)
        losers['w']=-losers['net_chips']
        edge=losers.groupby(['player_id','ben'])['w'].sum().reset_index()  # feeder,ben,chips
        # per-beneficiary: total inflow + number of distinct feeders + top-feeder share (ring = many feeders OR concentrated)
        ben_in=edge.groupby('ben')['w'].sum().to_dict()
        ben_nf=edge.groupby('ben')['player_id'].nunique().to_dict()
        # per player: total fed out + to how many distinct beneficiaries
        feed_out=edge.groupby('player_id')['w'].sum().to_dict()
        feed_nb=edge.groupby('player_id')['ben'].nunique().to_dict()
        # shared beneficiary sets: for each player, set of beneficiaries they feed (top by chips)
        pl_bens={}
        for pid,g in edge.groupby('player_id'):
            gg=g.sort_values('w',ascending=False)
            pl_bens[pid]=set(gg['ben'].head(5).tolist())
        # directed edge dict for pair-direct flow
        ed=dict(zip(zip(edge['player_id'],edge['ben']),edge['w']))
        # table normalizer: total chips moved per table
        s_tab=dict(zip(s['player_id'],s['table_id'])) if 'table_id' in s.columns else {}
        rows=[]
        lo=pairs_df['p_low'].values; hi=pairs_df['p_high'].values; pid_arr=pairs_df['pair_id'].values
        for k in range(len(pairs_df)):
            a=lo[k]; b=hi[k]
            # shared beneficiary: do A and B feed a COMMON third player?
            sba=pl_bens.get(a,set()); sbb=pl_bens.get(b,set())
            shared=sba & sbb; shared.discard(a); shared.discard(b)
            shared_ben=1.0 if len(shared)>0 else 0.0
            shared_ben_n=float(len(shared))
            # ring-beneficiary: is A (or B) a beneficiary with many feeders? normalized inflow concentration
            def ben_score(x):
                nf=ben_nf.get(x,0); return nf
            ring_ben_max=float(max(ben_score(a),ben_score(b)))
            # direct pair flow asymmetry (feeder->ben) normalized
            wab=ed.get((a,b),0.0); wba=ed.get((b,a),0.0)
            pair_dir_asym=abs(wab-wba)/(wab+wba+1.0)
            # is one of the pair feeding the OTHER as part of a multi-feeder ring on that ben?
            ring_pair_feed=0.0
            if wab>0 and ben_nf.get(b,0)>=2: ring_pair_feed=1.0   # A feeds B and B has >=2 feeders
            if wba>0 and ben_nf.get(a,0)>=2: ring_pair_feed=1.0
            rows.append((pid_arr[k],shared_ben,shared_ben_n,ring_ben_max,pair_dir_asym,ring_pair_feed))
        R=pd.DataFrame(rows,columns=['pair_id','r_shared_ben','r_shared_ben_n','r_ring_ben_max','r_pair_dir_asym','r_ring_pair_feed'])
        return pairs_df[['pair_id']].merge(R,on='pair_id',how='left').fillna(0)
    dp=pd.read_parquet(V2/'dev_pairs_v2.parquet')
    dev=pd.read_parquet(V5/'dev_v5.parquet').merge(dp[['pair_id','p_low','p_high','label']],on='pair_id',how='left')
    epk=pd.read_csv(D/'evaluation_pairs.csv'); epk['p_low']=epk[['player_1','player_2']].min(1); epk['p_high']=epk[['player_1','player_2']].max(1)
    ev=pd.read_parquet(V5/'eval_v5.parquet').merge(epk[['pair_id','p_low','p_high']],on='pair_id',how='left')
    log('build ring dev...'); Rd=build_ring('development',dev); log('dev done')
    log('build ring eval...'); Re=build_ring('evaluation',ev); log('eval done')
    rcols=[c for c in Rd.columns if c!='pair_id']; Rdv=Rd[rcols].to_numpy(np.float32); Rev=Re[rcols].to_numpy(np.float32)
    # normalize r_shared_ben_n and r_ring_ben_max (counts) -> could leak; log-scale + will check drift per col
    for k,c in enumerate(rcols):
        if c in ('r_shared_ben_n','r_ring_ben_max'):
            Rdv[:,k]=np.log1p(Rdv[:,k]); Rev[:,k]=np.log1p(Rev[:,k])
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
    for k,c in enumerate(rcols):
        xa=np.vstack([Rdv[:,k:k+1],Rev[:,k:k+1]]); ya=np.r_[np.zeros(len(Rdv)),np.ones(len(Rev))]; idx=np.random.default_rng(0).permutation(len(xa))
        av=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=0.1,eval_metric='auc',tree_method='hist',n_jobs=-1,random_state=0)
        av.fit(xa[idx[:len(idx)//2]],ya[idx[:len(idx)//2]]); auc=roc_auc_score(ya[idx[len(idx)//2:]],av.predict_proba(xa[idx[len(idx)//2:]])[:,1])
        d=hap(np.hstack([Xfound,Rdv[:,k:k+1]]).astype(np.float32))-base
        Ap=auc<0.65; log('feat %-20s drift %.3f holdout %+.4f A=%s B=%s'%(c,auc,d,'P' if Ap else 'F','P' if d>=0.010 else 'F'))
        if Ap: apass.append(k)
    # full-block gate + greedy stack
    if apass:
        Xa=np.vstack([Rdv[:,apass],Rev[:,apass]]); ya=np.r_[np.zeros(len(Rdv)),np.ones(len(Rev))]; idx=np.random.default_rng(0).permutation(len(Xa))
        av=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.1,eval_metric='auc',tree_method='hist',n_jobs=-1,random_state=0); av.fit(Xa[idx[:len(idx)//2]],ya[idx[:len(idx)//2]])
        blk=roc_auc_score(ya[idx[len(idx)//2:]],av.predict_proba(Xa[idx[len(idx)//2:]])[:,1]); log('block drift AUC (A-passers): %.3f'%blk)
    chosen=[]; cur=base; curX=Xfound; rem=list(apass)
    while rem:
        best=None
        for k in rem:
            ap=hap(np.hstack([curX,Rdv[:,k:k+1]]).astype(np.float32))
            if best is None or ap>best[1]: best=(k,ap)
        if best[1]-cur>0.0005:
            chosen.append(best[0]); curX=np.hstack([curX,Rdv[:,best[0]:best[0]+1]]).astype(np.float32); log('+ stack %-20s cum %.4f (d %+.4f)'%(rcols[best[0]],best[1],best[1]-base)); cur=best[1]; rem.remove(best[0])
        else: break
    log('FINAL stacked ring feats: '+str([rcols[k] for k in chosen]))
    log('foundation %.4f -> stacked %.4f (delta %+.4f)'%(base,cur,cur-base))
    log('NOTE: HOLDOUT AP delta, NOT calibrated LB. 0.60 bar not asserted.')
    if chosen:
        np.save(V7/'_RING_dev.npy',Rdv[:,chosen]); np.save(V7/'_RING_eval.npy',Rev[:,chosen]); (V7/'_RING_cols.json').write_text(json.dumps([rcols[k] for k in chosen]))
        log('saved chosen ring feats')
    log('DONE')
except Exception as e:
    log('ERROR: '+repr(e)); log(traceback.format_exc())
