"""Per-(pair, hand) token sequences for EVIDENCE ranking (task #2).

The pair-level tokenizer (seq_tokenize.py) flattens all shared hands into ONE pair document.
For evidence we must rank hands WITHIN a positive pair, so we need a token sequence PER HAND
keyed by (pair_id, hand_id). This reproduces the EXACT token scheme of seq_tokenize.py but emits
one row per (pair_id, hand_id) instead of flattening, so the trained encoder can embed each hand
independently and a within-pair ranker can pick the true evidence hands.

Token per action of a pair MEMBER within a shared hand (identical to seq_tokenize.py):
  10 + (((role*6+act)*4+street)*3+facing)*4+size ; role m1/m2, act fold..all_in, street pf..r,
  facing partner/other/none, size 0/small/med/big. Each hand doc = [BOS, ...its tokens] capped MAX_LEN.

Caches: seq_perhand_dev.parquet / seq_perhand_eval.parquet (pair_id, hand_id, hand_idx, tokens, length).
Run:  python -m anchor_repro.seq_perhand_tokens
"""
from __future__ import annotations
import json, time, polars as pl
from pathlib import Path
STEP3=Path("outputs/poker_collusion/hosen42_step3"); AC=STEP3/"action_context.parquet"
OUT=STEP3/"edge"/"seq"
MAX_LEN=128        # per-HAND token cap (a single hand is short; 128 is generous)
ACTIONS=["fold","check","call","bet","raise","all_in"]
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}",flush=True)

def build(pair_hands_path, phase, out_path):
    if out_path.exists(): log(f"{phase} per-hand tokens cached"); return
    ph=pl.read_parquet(pair_hands_path).select(["pair_id","hand_id","hand_idx","player_1","player_2"])
    ac=(pl.scan_parquet(AC).filter(pl.col("phase")==phase)
        .select(["hand_id","action_no","street_no","player_id","action","is_aggr","last_aggr","to_call","amount_pot_ratio"]))
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
    # per-HAND token list (ordered by action_no across BOTH members), prepend BOS(1), cap MAX_LEN
    perhand=(j.sort(["pair_id","hand_idx","action_no"])
             .group_by(["pair_id","hand_id","hand_idx"],maintain_order=True)
             .agg(pl.col("token").sort_by("action_no").alias("_tok")))
    docs=(perhand.with_columns((pl.concat_list([pl.lit([1]),pl.col("_tok")]).list.head(MAX_LEN)).alias("tokens"))
          .select(["pair_id","hand_id","hand_idx","tokens"])
          .with_columns(pl.col("tokens").list.len().alias("length")))
    docs.write_parquet(out_path)
    log(f"{phase}: {docs.height:,} pair-hand docs, mean len {docs['length'].mean():.1f}, max {docs['length'].max()}")

def main():
    build(STEP3/"dev_pair_hands.parquet","development",OUT/"seq_perhand_dev.parquet")
    build(STEP3/"eval_pair_hands.parquet","evaluation",OUT/"seq_perhand_eval.parquet")

if __name__=="__main__":
    raise SystemExit(main())
