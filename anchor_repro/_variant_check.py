import polars as pl
a = pl.scan_parquet('data/poker/actions.parquet').collect()
print('action types:', a.group_by('action').len().sort('len', descending=True).to_dicts())
h = pl.scan_parquet('data/poker/hands.parquet').select(['hand_id', 'big_blind']).collect()
br = a.filter(pl.col('action').is_in(['bet', 'raise', 'raises', 'b', 'r'])).join(h, on='hand_id', how='left')
print('bet/raise rows:', br.height)
if br.height > 0:
    br = br.with_columns((pl.col('amount_to') / pl.col('big_blind')).alias('to_bb'))
    print('bet/raise amount_to / BB quantiles:')
    for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        print(f'  p{int(p*100)}: {br["to_bb"].quantile(p):.3f}')
    print('  n distinct to_bb (rounded 2dp):', br['to_bb'].round(2).n_unique())
    print('  first 30 sorted distinct:', sorted(br['to_bb'].round(2).unique().to_list())[:30])
    # all-in tell: NL allows amount_to == full stack; Limit never does
    allin = a.filter(pl.col('action').str.contains('all')) if a['action'].dtype == pl.Utf8 else br.head(0)
    print('  actions containing "all":', a.filter(pl.col('action').str.contains('(?i)all')).height)
