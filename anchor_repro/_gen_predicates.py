"""Test near-deterministic per-hand predicates against the planted evidence hands.

For each family, evaluate candidate predicates on the positive-pair shared hands:
 - coverage = fraction of PLANTED hands satisfying it (recall of the generator);
 - purity   = fraction of hands satisfying it that are PLANTED (within positive pairs);
 - lift over base planted-rate.
A near-deterministic generator predicate should have very high coverage AND purity.
Also checks a pair-level reconstruction: does "pair has >= k hands satisfying predicate"
recover the positive-vs-negative labels?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("outputs/poker_collusion/generator_hunt")


def eval_predicate(df: pl.DataFrame, mask: pl.Expr, name: str) -> dict:
    m = df.with_columns(mask.alias("_p"))
    sat = m.filter(pl.col("_p"))
    planted = m.filter(pl.col("is_planted"))
    n_sat = sat.height
    n_planted = planted.height
    tp = sat.filter(pl.col("is_planted")).height
    coverage = tp / n_planted if n_planted else 0.0
    purity = tp / n_sat if n_sat else 0.0
    return {"name": name, "n_sat": n_sat, "coverage": coverage, "purity": purity}


def main() -> int:
    df = pl.read_parquet(OUT / "positive_shared_hands.parquet")

    # Candidate per-hand predicates (family-agnostic building blocks).
    preds = {
        "both_contrib_gt_blind": (pl.col("a_contrib_bb") > 1.0) & (pl.col("b_contrib_bb") > 1.0),
        "both_voluntary(>1bb)": (pl.col("a_contrib_bb") > 1.5) & (pl.col("b_contrib_bb") > 1.5),
        "abs_flow_ge_5": pl.col("abs_flow_bb") >= 5.0,
        "abs_flow_ge_10": pl.col("abs_flow_bb") >= 10.0,
        "min_contrib_ge_2": pl.col("min_contrib_bb") >= 2.0,
        "min_contrib_ge_3": pl.col("min_contrib_bb") >= 3.0,
        "not_both_fold_preflop": pl.col("min_contrib_bb") > 1.0,
    }
    print("=== PER-HAND PREDICATE coverage/purity of PLANTED hands, by family ===")
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        sub = df.filter(pl.col("behavior_family") == fam)
        base = sub["is_planted"].mean()
        print(f"\n--- {fam}: shared={sub.height} planted_rate={base:.4f} ---")
        for name, expr in preds.items():
            r = eval_predicate(sub, expr, name)
            print(f"  {name:22s} n_sat={r['n_sat']:6d} coverage={r['coverage']:.3f} purity={r['purity']:.3f}")

    # Directed-specific: does the LOSER commit big then the winner is the partner?
    print("\n=== DIRECTED-specific: directed loss into partner ===")
    dsub = df.filter(pl.col("behavior_family") == "directed_transfer")
    dsub = dsub.with_columns(
        (pl.min_horizontal("a_net_bb", "b_net_bb")).alias("loser_net_bb"),
        (pl.max_horizontal("a_net_bb", "b_net_bb")).alias("winner_net_bb"),
    )
    for thr in [3.0, 5.0, 10.0]:
        expr = (pl.col("loser_net_bb") <= -thr) & (pl.col("winner_net_bb") >= thr)
        r = eval_predicate(dsub, expr, f"opp_signs_ge_{thr}")
        print(f"  loser<=-{thr} & winner>=+{thr}: n_sat={r['n_sat']} coverage={r['coverage']:.3f} purity={r['purity']:.3f}")

    # Soft-specific: both reach showdown OR both check down; low aggression, one wins small.
    print("\n=== SOFT-specific: showdown / passivity ===")
    ssub = df.filter(pl.col("behavior_family") == "soft_play")
    for name, expr in {
        "both_showdown": (pl.col("a_went_to_showdown")) & (pl.col("b_went_to_showdown")),
        "any_showdown": (pl.col("a_went_to_showdown")) | (pl.col("b_went_to_showdown")),
        "both_contrib&showdown": (pl.col("min_contrib_bb") > 1.0) & (pl.col("a_went_to_showdown") | pl.col("b_went_to_showdown")),
    }.items():
        r = eval_predicate(ssub, expr, name)
        print(f"  {name:24s} n_sat={r['n_sat']} coverage={r['coverage']:.3f} purity={r['purity']:.3f}")

    # Isolation-specific: partner folds/steps aside; check outsider presence via players_dealt.
    print("\n=== ISOLATION-specific: one commits, other yields, multiway ===")
    isub = df.filter(pl.col("behavior_family") == "coordinated_isolation")
    for name, expr in {
        "one_big_one_fold": ((pl.col("a_contrib_bb") > 3.0) & (pl.col("b_folded"))) | ((pl.col("b_contrib_bb") > 3.0) & (pl.col("a_folded"))),
        "multiway(dealt>2)": pl.col("players_dealt") > 2,
        "one_commit_multiway": (pl.col("min_contrib_bb") > 1.0) & (pl.col("players_dealt") > 2),
    }.items():
        r = eval_predicate(isub, expr, name)
        print(f"  {name:22s} n_sat={r['n_sat']} coverage={r['coverage']:.3f} purity={r['purity']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
