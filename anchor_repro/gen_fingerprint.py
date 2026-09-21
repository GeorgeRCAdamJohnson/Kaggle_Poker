"""Generator fingerprint test: do planted hands cluster on specific pot/bet/stack ratios?

If the synthetic planter injects collusion on ratio-locked or round-number targets, planted
hands will show SPIKES/DISCRETENESS in these quantities that non-planted hands do not. That
would be a near-deterministic signature (the discrete-unlock candidate). If the distributions
look the same (just shifted), there is no ratio-lock and this idea is dead.

Measures, planted vs non-planted (within positive pairs, from the action stream at the
DECISIVE action = the last aggressive action of the pair member):
  * pot_bb, final_pot / big_blind
  * bet amount / pot_before (bet-size fraction)
  * amount / big_blind (round bet sizes?)
  * stack_before / pot_before (SPR)
  * fractional part of these (are they integers / on a lattice?)

Run:  python -m anchor_repro.gen_fingerprint
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

D = Path("data/poker")
CACHE = Path("outputs/poker_collusion/generator_cache")


def main() -> int:
    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    pos = labels.filter(pl.col("label") == 1)
    evidence = pl.read_csv(D / "development_evidence.csv").select(["pair_id", "hand_id"]).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("is_planted")
    )
    # Map: which (player, hand) belong to a positive pair, and is that hand planted.
    pair_players = pl.concat([
        pos.select(["pair_id", pl.col("player_1").alias("player_id")]),
        pos.select(["pair_id", pl.col("player_2").alias("player_id")]),
    ])
    # Shared-hand universe for positive pairs (from dev_layer2, already restricted).
    shared = pl.scan_parquet(CACHE / "dev_layer2.parquet").select(["pair_id", "hand_id"]).collect()
    shared = shared.join(pos.select("pair_id"), on="pair_id", how="inner")
    shared = shared.join(evidence, on=["pair_id", "hand_id"], how="left").with_columns(
        pl.col("is_planted").fill_null(0)
    )
    planted_hands = set(shared.filter(pl.col("is_planted") == 1)["hand_id"].to_list())
    shared_hands = set(shared["hand_id"].to_list())
    print(f"positive-pair shared hands={len(shared_hands):,} planted hands={len(planted_hands):,}")

    # Pull actions for those hands; join blinds.
    acts = (
        pl.scan_parquet(D / "actions.parquet")
        .filter(pl.col("hand_id").is_in(list(shared_hands)))
        .filter(pl.col("action").is_in(["bet", "raise", "all_in"]))
        .join(pl.scan_parquet(D / "hands.parquet").select(["hand_id", "big_blind"]), on="hand_id")
        .with_columns(
            (pl.col("amount") / pl.col("big_blind")).alias("amount_bb"),
            (pl.col("amount") / pl.col("pot_before").clip(1, None)).alias("bet_pot_frac"),
            (pl.col("pot_before") / pl.col("big_blind")).alias("pot_bb"),
            (pl.col("stack_before") / pl.col("pot_before").clip(1, None)).alias("spr"),
            pl.col("hand_id").is_in(list(planted_hands)).alias("is_planted"),
        )
        .collect(engine="streaming")
    )
    p = acts.filter(pl.col("is_planted"))
    n = acts.filter(~pl.col("is_planted"))
    print(f"aggressive actions: planted-hand={p.height:,} nonplanted-hand={n.height:,}")

    def describe(col):
        pv = p[col].drop_nulls().to_numpy()
        nv = n[col].drop_nulls().to_numpy()
        return pv, nv

    print("\n=== distribution comparison (planted vs nonplanted aggressive actions) ===")
    for col in ["amount_bb", "bet_pot_frac", "pot_bb", "spr"]:
        pv, nv = describe(col)
        print(f"{col}: planted[med={np.median(pv):.3f} p25={np.percentile(pv,25):.3f} p75={np.percentile(pv,75):.3f}] "
              f"nonpl[med={np.median(nv):.3f} p25={np.percentile(nv,25):.3f} p75={np.percentile(nv,75):.3f}]")

    # DISCRETENESS test: are amounts on a lattice? fraction that are round multiples.
    print("\n=== discreteness / ratio-lock test ===")
    for name, arr_p, arr_n in [
        ("amount_bb", p["amount_bb"].to_numpy(), n["amount_bb"].to_numpy()),
        ("bet_pot_frac", p["bet_pot_frac"].to_numpy(), n["bet_pot_frac"].to_numpy()),
    ]:
        ap = arr_p[np.isfinite(arr_p)]; an = arr_n[np.isfinite(arr_n)]
        # fraction whose value is within 0.02 of a "round" pot fraction {0.5,0.66,0.75,1.0} or integer bb
        for tgt in [0.5, 0.66, 0.75, 1.0]:
            fp = np.mean(np.abs(ap - tgt) < 0.02)
            fn = np.mean(np.abs(an - tgt) < 0.02)
            if name == "bet_pot_frac":
                print(f"  {name} near {tgt}: planted={fp:.3f} nonpl={fn:.3f}")
        if name == "amount_bb":
            fp_int = np.mean(np.abs(ap - np.round(ap)) < 1e-6)
            fn_int = np.mean(np.abs(an - np.round(an)) < 1e-6)
            print(f"  {name} exact-integer bb: planted={fp_int:.3f} nonpl={fn_int:.3f}")

    # Most concentrated single values in planted (are there spike values?)
    print("\n=== top repeated exact values (planted) ===")
    for col in ["amount_bb", "bet_pot_frac"]:
        vc = p.group_by(col).len().sort("len", descending=True).head(6)
        total = p.height
        print(f"  {col}: " + ", ".join(f"{r[col]:.3f}x{r['len']}({r['len']/total*100:.1f}%)" for r in vc.iter_rows(named=True)))

    # KS-style separation: can a single ratio threshold separate planted from nonplanted?
    from sklearn.metrics import roc_auc_score
    print("\n=== single-ratio planted-vs-nonplanted AUC (per aggressive action) ===")
    yb = acts["is_planted"].to_numpy().astype(int)
    for col in ["amount_bb", "bet_pot_frac", "pot_bb", "spr"]:
        v = acts[col].fill_null(0).to_numpy()
        v = np.where(np.isfinite(v), v, 0)
        auc = max(roc_auc_score(yb, v), roc_auc_score(yb, -v))
        print(f"  {col:14s} AUC={auc:.4f}")

    summary = {"planted_actions": int(p.height), "nonplanted_actions": int(n.height)}
    (CACHE / "_fingerprint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
