import polars as pl

d = pl.read_parquet("outputs/poker_collusion/feature_cache/dev_evidence_candidates.parquet")
e = pl.scan_parquet("outputs/poker_collusion/feature_cache/eval_evidence_candidates.parquet")
print("DEV cand: rows", d.height, "distinct pairs", d["pair_id"].n_unique(), "planted", int(d["is_planted"].sum()))
en = e.select(pl.col("pair_id").n_unique().alias("np"), pl.len().alias("n")).collect()
print("EVAL cand: rows", en["n"][0], "distinct pairs", en["np"][0])
print("DEV cands/pair mean:", round(d.height / d["pair_id"].n_unique(), 1))
print("EVAL cands/pair mean:", round(en["n"][0] / en["np"][0], 1))
lab = pl.read_csv("data/poker/development_labels.csv").select(["pair_id", "label"])
dj = d.select("pair_id").unique().join(lab, on="pair_id", how="left")
print("DEV cand pairs by label:", dj["label"].value_counts().sort("label").to_dicts())
# per-hand planted rate among dev candidates (the target base rate)
print("DEV per-hand planted rate:", round(float(d["is_planted"].mean()), 4))
