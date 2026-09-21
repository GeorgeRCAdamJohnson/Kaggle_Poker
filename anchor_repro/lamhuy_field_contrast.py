"""Partner-vs-field counterfactual contrast: isolate pair-specific deviation from a
player's natural (field) baseline.

BUILD + MEASURE ONLY. LamhuY remains the risk foundation. For each player in a
candidate pair, this compares their behavior on hands where the OTHER member was
ALSO seated ("with") against hands where they were not ("without" -- the field
baseline). The contrast (with - without) cancels a player's own skill/style/
variance and isolates the relationship-specific deviation.

Reuses the existing ``player_hand_features.parquet`` cache verbatim (Rule 6) --
the same per-(hand, player) stat table lamhuy's own ``add_partner_field_contrasts``
joins against -- so no new hand-level computation is invented.

Extension beyond a plain pairwise contrast (the untested piece): a joint
THIRD-PLAYER isolation contrast for ``coordinated_isolation`` -- does the PAIR,
together, suppress/fold-out a specific outsider more than the field suppresses
that same outsider at the same table? Built as its own gated feature block,
independent of lamhuy's monolithic risk head, so it can be inspected and
composed transparently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: The per-hand player stat columns the contrast is built from (verbatim from
#: the lamhuy notebook's own FIELD_COLS, all already present in the cache).
FIELD_COLS: list[str] = [
    "net_bb",
    "aggressive_actions",
    "raises",
    "calls",
    "checks",
    "folds",
    "folds_facing_bet",
    "all_ins",
    "contribution_bb",
]

#: Empirical-Bayes shrinkage pseudo-count (same discipline as aggregation.py's
#: DEFAULT_PRIOR_STRENGTH-style shrink: with-sample-size n, weight = n/(n+ALPHA)).
SHRINK_ALPHA: float = 25.0

_STAT_SUFFIXES = ("with_mean", "without_mean", "contrast", "contrast_shrunk")
FEATURE_COLUMNS: list[str] = [
    f"{side}_{col}_{suffix}"
    for side in ("p1", "p2")
    for col in FIELD_COLS
    for suffix in _STAT_SUFFIXES
] + [
    "pair_net_contrast",
    "directed_asymmetry",
    "pair_agr_contrast",
    "soft_contrast",
    "pair_raise_contrast",
    "pair_net_contrast_shrunk",
    "directed_asymmetry_shrunk",
]


def _one_sided_contrast(
    pairs: pd.DataFrame, player_hands: pd.DataFrame, self_col: str, partner_col: str, prefix: str
) -> pd.DataFrame:
    """Per-pair with/without means + contrast for ONE member's stats (Rule 6 reuse)."""
    self_hands = pairs[["pair_id", self_col, partner_col]].merge(
        player_hands, left_on=self_col, right_on="player_id", how="inner"
    )
    presence = player_hands[["hand_id", "player_id"]].rename(columns={"player_id": "_partner_present"})
    self_hands = self_hands.merge(
        presence, left_on=["hand_id", partner_col], right_on=["hand_id", "_partner_present"], how="left"
    )
    self_hands["_with"] = self_hands["_partner_present"].notna()

    grouped = self_hands.groupby(["pair_id", "_with"])[FIELD_COLS].mean()
    counts = self_hands.groupby(["pair_id", "_with"]).size()

    with_mean = grouped.xs(True, level="_with").reindex(pairs["pair_id"])
    without_mean = grouped.xs(False, level="_with").reindex(pairs["pair_id"])
    n_with = counts.xs(True, level="_with").reindex(pairs["pair_id"]).fillna(0.0)

    result = pd.DataFrame(index=pairs["pair_id"])
    for col in FIELD_COLS:
        w = with_mean[col].fillna(0.0)
        wo = without_mean[col].fillna(0.0)
        contrast = w - wo
        shrink_weight = n_with / (n_with + SHRINK_ALPHA)
        result[f"{prefix}_{col}_with_mean"] = w
        result[f"{prefix}_{col}_without_mean"] = wo
        result[f"{prefix}_{col}_contrast"] = contrast
        result[f"{prefix}_{col}_contrast_shrunk"] = contrast * shrink_weight
    return result.reset_index()


def build_field_contrast_features(pairs: pd.DataFrame, player_hands: pd.DataFrame) -> pd.DataFrame:
    """Build the same pair-level partner-vs-field contrast features for dev and eval.

    Args:
        pairs: ``pair_id, player_1, player_2`` (dev_pairs.parquet / eval_pairs.parquet).
        player_hands: the phase-filtered slice of ``player_hand_features.parquet``
            (``hand_id, player_id`` + ``FIELD_COLS``), ALREADY restricted to the
            matching phase (development for dev pairs, evaluation for eval pairs).
    """
    keys = pairs[["pair_id", "player_1", "player_2"]].drop_duplicates("pair_id")
    stats = player_hands[["hand_id", "player_id", *FIELD_COLS]].copy()
    stats[FIELD_COLS] = stats[FIELD_COLS].astype("float32")

    p1_side = _one_sided_contrast(keys, stats, "player_1", "player_2", "p1")
    p2_side = _one_sided_contrast(keys, stats, "player_2", "player_1", "p2")
    result = p1_side.merge(p2_side, on="pair_id", how="inner", validate="one_to_one")

    result["pair_net_contrast"] = result["p1_net_bb_contrast"] + result["p2_net_bb_contrast"]
    result["directed_asymmetry"] = (result["p1_net_bb_contrast"] - result["p2_net_bb_contrast"]).abs()
    result["pair_agr_contrast"] = result["p1_aggressive_actions_contrast"] + result["p2_aggressive_actions_contrast"]
    result["soft_contrast"] = -result["pair_agr_contrast"]
    result["pair_raise_contrast"] = result["p1_raises_contrast"] + result["p2_raises_contrast"]
    result["pair_net_contrast_shrunk"] = (
        result["p1_net_bb_contrast_shrunk"] + result["p2_net_bb_contrast_shrunk"]
    )
    result["directed_asymmetry_shrunk"] = (
        result["p1_net_bb_contrast_shrunk"] - result["p2_net_bb_contrast_shrunk"]
    ).abs()

    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    return result[["pair_id", *FEATURE_COLUMNS]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poker-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepared = args.poker_root / "outputs" / "poker_collusion" / "repro_lamhuy" / "prepared_v13"
    report_dir = args.poker_root / "outputs" / "poker_collusion" / "lamhuy_temporal_direction"
    report_dir.mkdir(parents=True, exist_ok=True)

    player_hands = pd.read_parquet(
        prepared / "player_hand_features.parquet", columns=["hand_id", "player_id", "phase", *FIELD_COLS]
    )
    dev_pairs = pd.read_parquet(prepared / "dev_pairs.parquet", columns=["pair_id", "player_1", "player_2"])
    eval_pairs = pd.read_parquet(prepared / "eval_pairs.parquet", columns=["pair_id", "player_1", "player_2"])

    dev_features = build_field_contrast_features(
        dev_pairs, player_hands[player_hands["phase"] == "development"]
    )
    dev_features.to_parquet(report_dir / "dev_field_contrast.parquet", index=False)
    print(json.dumps({"stage": "dev_done", "n_pairs": int(len(dev_features))}))

    eval_features = build_field_contrast_features(
        eval_pairs, player_hands[player_hands["phase"] == "evaluation"]
    )
    eval_features.to_parquet(report_dir / "eval_field_contrast.parquet", index=False)
    print(json.dumps({"stage": "eval_done", "n_pairs": int(len(eval_features))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
