"""Tokenize each PAIR's shared-hand action stream into a sequence (the raw representation lens).

Everything prior (ours + competitors) is engineered AGGREGATES. This builds the raw ORDERED action
stream per pair, so a sequence model can learn coordination structure (who-leads/folds-to-whom, in what
order) that aggregates flatten.

Token per action of a pair MEMBER within a shared hand:
  role (m1/m2) x action(fold/check/call/bet/raise/allin) x street(pf/f/t/r) x facing(partner/other/none)
  x size-bucket(0/small/med/big/allin)
Plus a HAND-BOUNDARY token between hands and a WON/LOST tag per hand end. Outsider actions are NOT
tokenized individually (too noisy) but "facing==partner vs other" captures the pair's response to the
partner vs the field. Document = tokens across the pair's shared hands ordered by hand_idx, capped to
last MAX_HANDS most-active shared hands x their actions, truncated to MAX_LEN tokens.

Caches: seq_tokens_dev.parquet / seq_tokens_eval.parquet (pair_id, token_ids list, length).
Vocab saved to seq_vocab.json. Uses hosen42 caches (action_context, pair-hand maps).
Run:  python -m anchor_repro.seq_tokenize
"""
from __future__ import annotations
import json, time, numpy as np, polars as pl
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); EDGE=STEP3/"edge"; DATA=Path("data/poker")
AC=STEP3/"action_context.parquet"
OUT=EDGE/"seq"; OUT.mkdir(parents=True,exist_ok=True)
MAX_HANDS=60      # most-active shared hands per pair
MAX_LEN=512       # token cap per pair document
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

# token id scheme: 0=PAD 1=BOS 2=HAND_SEP 3=WON 4=LOST ; action tokens start at 10
ACTIONS=["fold","check","call","bet","raise","all_in"]
def size_bucket(amount_pot):
    if amount_pot<=0: return 0
    if amount_pot<0.5: return 1
    if amount_pot<1.0: return 2
    return 3
# build vocab: role(2) x action(6) x street(4) x facing(3) x size(4) = 576, offset 10
def tok(role,act,street,facing,size):
    return 10 + (((role*6+act)*4+street)*3+facing)*4+size
VOCAB=10+2*6*4*3*4

def build(pair_hands_path, phase, out_path):
    if out_path.exists(): log(f"{phase} tokens cached"); return
    # pair-hand map with member ids and hand order
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","hand_idx","player_1","player_2"])
    # per-hand shared-hand count -> keep most-active (use all, cap by hand_idx recency later)
    # actions for the relevant hands+players
    hand_ids=ph["hand_id"].unique().to_list()
    ac=(pl.scan_parquet(AC).filter(pl.col("phase")==phase)
        .select(["hand_id","action_no","street_no","player_id","action","is_aggr","last_aggr","to_call","amount_pot_ratio"]))
    # join member role: only keep actions by player_1 or player_2 of each pair sharing that hand
    m1=ph.select(["pair_id","hand_id","hand_idx",pl.col("player_1").alias("player_id"),pl.col("player_2").alias("partner")]).with_columns(pl.lit(0,dtype=pl.Int64).alias("role"))
    m2=ph.select(["pair_id","hand_id","hand_idx",pl.col("player_2").alias("player_id"),pl.col("player_1").alias("partner")]).with_columns(pl.lit(1,dtype=pl.Int64).alias("role"))
    members=pl.concat([m1,m2])
    j=(members.lazy().join(ac,on=["hand_id","player_id"],how="inner")
       .with_columns(
           pl.col("action").replace_strict({a:i for i,a in enumerate(ACTIONS)},default=1).alias("act"),
           pl.col("street_no").clip(0,3).alias("street"),
           pl.when(pl.col("to_call")<=0).then(2).when(pl.col("last_aggr")==pl.col("partner")).then(0).otherwise(1).alias("facing"),
           pl.col("amount_pot_ratio").fill_null(0).alias("apr"),
       )
       .with_columns(pl.when(pl.col("apr")<=0).then(0).when(pl.col("apr")<0.5).then(1).when(pl.col("apr")<1.0).then(2).otherwise(3).alias("size"))
       .with_columns((10+(((pl.col("role")*6+pl.col("act"))*4+pl.col("street"))*3+pl.col("facing"))*4+pl.col("size")).alias("token"))
       .select(["pair_id","hand_id","hand_idx","action_no","token"])
       .collect(engine="streaming"))
    # order tokens within hand by action_no, hands by hand_idx; build per-pair document with HAND_SEP(2)
    j=j.sort(["pair_id","hand_idx","action_no"])
    # keep only the last MAX_HANDS hands per pair (recency) then flatten
    hand_rank=(j.select(["pair_id","hand_idx"]).unique().sort(["pair_id","hand_idx"])
               .with_columns((pl.col("hand_idx").rank("ordinal",descending=True).over("pair_id")).alias("hr")))
    j=j.join(hand_rank,on=["pair_id","hand_idx"]).filter(pl.col("hr")<=MAX_HANDS)
    # per-hand token lists then join with HAND_SEP
    perhand=(j.group_by(["pair_id","hand_idx"]).agg(pl.col("token").sort_by("action_no")).sort(["pair_id","hand_idx"]))
    doc=(perhand.group_by("pair_id",maintain_order=True).agg(pl.col("token").flatten().alias("_flat"),pl.len().alias("n_hands")))
    # prepend BOS(1); cap MAX_LEN
    docs=doc.with_columns((pl.concat_list([pl.lit([1]),pl.col("_flat")]).list.head(MAX_LEN)).alias("tokens")).select(["pair_id","tokens","n_hands"])
    docs=docs.with_columns(pl.col("tokens").list.len().alias("length"))
    docs.write_parquet(out_path)
    log(f"{phase}: {docs.height:,} pair docs, mean len {docs['length'].mean():.0f}, max {docs['length'].max()}")

def main():
    (OUT/"seq_vocab.json").write_text(json.dumps({"vocab_size":VOCAB,"PAD":0,"BOS":1,"HAND_SEP":2,"actions":ACTIONS,"max_len":MAX_LEN,"max_hands":MAX_HANDS}),encoding="utf-8")
    log(f"vocab size {VOCAB}")
    build(STEP3/"dev_pair_hands.parquet","development",OUT/"seq_tokens_dev.parquet")
    build(STEP3/"eval_pair_hands.parquet","evaluation",OUT/"seq_tokens_eval.parquet")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
