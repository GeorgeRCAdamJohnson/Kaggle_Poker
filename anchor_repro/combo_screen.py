"""§60/§61 combo screen — floor-guaranteed rank blends of VERIFIED levers.

Pre-registered bar: RESEARCH_DOSSIER §60 (BINDING). Trusted-scorer scope: §57.
Governed by the workspace ``reverse-engineering-accountability`` contract
(Rules 1, 2, 3, 6, 9, 12, 15).

WHAT THIS DOES (and only this)
------------------------------
Screens combinations of the three independently LB-verified Tier-1 levers on the
TRUSTED dev-holdout scorer (§57 — trustworthy RELATIVE ranking in ~0.33-0.64 LB) to
pick the single best FLOOR-GUARANTEED blend to submit — WITHOUT touching the LB.

The three levers (each a dev-holdout prediction frame reused from its existing runner,
contract Rule 6):

* floor stack v5+MFg+DIRc  — dev-holdout PairAP 0.8934 (real LB 0.44519). KNOWN-GOOD
  FLOOR.  Runner: ``recipe_registry.rerun_floor_stack`` (760 confirmed pairs).
* honghanh PU risk head    — dev-holdout PairAP 0.9746 (real LB 0.64262). Runner:
  ``competitor_recipe.rerun_competitor_recipe`` (1,860 confirmed OOF pairs).
* nomannic risk head       — dev-holdout PairAP 0.9376 (real LB 0.56202). Runner:
  ``nomannic_recipe.rerun_nomannic_recipe`` (1,860 confirmed OOF pairs).

COMPOSITION (contract Rule 15 — floor-guaranteed, w=0 allowed, blend on RANKS)
------------------------------------------------------------------------------
For each lever L in {honghanh, nomannic} and for the averaged pair {honghanh+nomannic}
we form, over the COMMON confirmed dev-holdout pairs (the intersection of all pair
sets so every combo is scored on ONE common set):

    blended_rank = (1 - w) * rank(floor_risk) + w * rank(lever_risk)

with ``w`` swept over a grid INCLUDING 0.0 (default {0.0, 0.1, ..., 1.0}). ``w=0``
recovers the floor exactly, so the floor is a HARD FLOOR: a lever that only hurts
collapses to w=0 and cannot regress below the floor-alone PairAP on the common set.

The blended rank is min-max normalised into [0, 1] and written as ``risk_score`` on a
CanonicalScorer-ready frame (PairAP is a global ranking metric — normalisation is
monotone and does not change the ranking, only satisfies the [0,1] contract). Each
w's frame is scored via ``CanonicalScorer.score_dev_predictions`` → canonical
confirmed-only PairAP.

DECISION RULE (§60 PRE-REGISTERED, bars FIXED — Rules 2, 3, 12, 15)
-------------------------------------------------------------------
Winner = the combo with the highest common-set PairAP that BEATS the floor-alone
common-set PairAP beyond ``margin`` AND whose best ``w`` is > 0. If no combo clears
that bar, the result is a NULL (report it; floor stays live-best; do NOT tune the
grid/features to manufacture a lift).

This module BUILDS + SCREENS ONLY. It NEVER submits to Kaggle, NEVER overwrites the
live ``submission.csv`` / ``submission_best_*.csv``. The winning eval submission is
written to a NEW path by :func:`build_combo_eval_submission`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_W_GRID",
    "DEFAULT_MARGIN",
    "BlendSweep",
    "ScreenResult",
    "rank_blend_frame",
    "common_confirmed_pairs",
    "sweep_blend",
    "screen_combos",
    "build_combo_eval_submission",
    "validate_eval_submission",
]

#: The pre-registered w grid — INCLUDES 0.0 so the floor is a hard floor (§60/Rule 15).
DEFAULT_W_GRID: Tuple[float, ...] = tuple(round(x, 4) for x in np.linspace(0.0, 1.0, 11))

#: Beat-the-floor margin below which a lift is treated as noise → NULL (§60/Rule 12).
#: Small but non-zero so a coincidental 4th-decimal wiggle is not called a "win".
DEFAULT_MARGIN: float = 1e-4


# --------------------------------------------------------------------------- #
# Rank-blend primitives                                                        #
# --------------------------------------------------------------------------- #


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Average-rank of ``values`` (ties share the mean rank), ascending.

    Uses ``pandas.Series.rank(method="average")`` — a higher raw risk → higher rank,
    matching the direction PairAP rewards.
    """
    return pd.Series(list(values), dtype=float).rank(method="average").to_numpy(float)


def rank_blend_frame(
    common_ids: Sequence[str],
    floor_risk: Sequence[float],
    lever_risk: Sequence[float],
    w: float,
) -> pd.DataFrame:
    """Build a CanonicalScorer-ready frame for one blend weight ``w``.

    ``blended_rank = (1-w)*rank(floor) + w*rank(lever)`` over the COMMON pairs, then
    min-max normalised into [0,1] as ``risk_score`` (monotone; does not change the
    ranking PairAP measures). Behavior/evidence columns are neutral placeholders —
    PairAP depends only on ``risk_score`` (same convention the runners use).

    Args:
        common_ids: The common confirmed pair ids (one row each, in a fixed order).
        floor_risk: Floor risk aligned to ``common_ids``.
        lever_risk: Lever risk aligned to ``common_ids``.
        w: Blend weight in [0,1]. w=0 == floor alone (hard floor); w=1 == lever alone.

    Returns:
        A frame with columns ``pair_id``, ``risk_score`` (+ neutral behavior/evidence).
    """
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"w must be in [0,1]; got {w!r}")
    fr = _ranks(floor_risk)
    lr = _ranks(lever_risk)
    blended = (1.0 - w) * fr + w * lr
    span = blended.max() - blended.min()
    if span <= 0:
        norm = np.full(len(blended), 0.5, dtype=float)
    else:
        norm = (blended - blended.min()) / span
    frame = pd.DataFrame(
        {
            "pair_id": [str(p) for p in common_ids],
            "risk_score": norm.astype(float).clip(0.0, 1.0),
            "predicted_behavior": "none",
        }
    )
    for col in (
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    ):
        frame[col] = "NO_EVIDENCE"
    return frame


def common_confirmed_pairs(
    frames: Dict[str, pd.DataFrame],
) -> List[str]:
    """Return the sorted intersection of ``pair_id`` across every prediction frame.

    So every combo is scored on ONE common pair set (§60 alignment requirement).
    """
    sets = [set(str(p) for p in f["pair_id"]) for f in frames.values()]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


# --------------------------------------------------------------------------- #
# The sweep + screen                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BlendSweep:
    """The full w-sweep for one lever↔floor blend on the common pair set.

    Attributes:
        blend_id: Which blend (e.g. ``"honghanh"``, ``"nomannic"``, ``"hh+nm_avg"``).
        w_grid: The swept weights (includes 0.0).
        pair_ap: PairAP at each w (aligned to ``w_grid``).
        best_w: The w maximising PairAP.
        best_pair_ap: PairAP at ``best_w``.
        floor_alone_pair_ap: PairAP at w=0 on this common set (== the floor baseline).
    """

    blend_id: str
    w_grid: Tuple[float, ...]
    pair_ap: Tuple[float, ...]
    best_w: float
    best_pair_ap: float
    floor_alone_pair_ap: float

    def sweep_rows(self) -> List[Tuple[float, float]]:
        """(w, PairAP) rows for auditable logging."""
        return list(zip(self.w_grid, self.pair_ap))


@dataclass
class ScreenResult:
    """The screen verdict across all blends (§60 decision rule applied).

    Attributes:
        common_ids: The common confirmed pair set (sorted).
        floor_alone_pair_ap: Floor-alone PairAP on the common set (baseline to beat).
        sweeps: One :class:`BlendSweep` per blend.
        winner_blend_id: The winning blend id, or ``None`` for a NULL verdict.
        winner_w: The winning w (>0), or ``None`` for NULL.
        winner_pair_ap: The winner's PairAP, or ``None`` for NULL.
        delta_over_floor: winner_pair_ap - floor_alone_pair_ap, or 0.0 for NULL.
        verdict: ``"WINNER"`` or ``"NULL"``.
        margin: The beat-the-floor margin used.
    """

    common_ids: List[str]
    floor_alone_pair_ap: float
    sweeps: List[BlendSweep]
    winner_blend_id: Optional[str] = None
    winner_w: Optional[float] = None
    winner_pair_ap: Optional[float] = None
    delta_over_floor: float = 0.0
    verdict: str = "NULL"
    margin: float = DEFAULT_MARGIN

    @property
    def n_common(self) -> int:
        return len(self.common_ids)


def sweep_blend(
    scorer,
    blend_id: str,
    common_ids: Sequence[str],
    floor_risk: Sequence[float],
    lever_risk: Sequence[float],
    w_grid: Sequence[float] = DEFAULT_W_GRID,
) -> BlendSweep:
    """Sweep ``w`` for one lever↔floor blend, scoring PairAP at each w.

    Args:
        scorer: A ``CanonicalScorer`` whose ``score_dev_predictions`` scores a frame.
        blend_id: Blend label.
        common_ids: Common confirmed pair ids (fixed order).
        floor_risk: Floor risk aligned to ``common_ids``.
        lever_risk: Lever risk aligned to ``common_ids``.
        w_grid: Weights to sweep (must include 0.0 for the floor-guarantee).

    Returns:
        A :class:`BlendSweep`.
    """
    aps: List[float] = []
    for w in w_grid:
        frame = rank_blend_frame(common_ids, floor_risk, lever_risk, w)
        score = scorer.score_dev_predictions(frame)
        aps.append(float(score.pair_ap))
    aps_t = tuple(aps)
    grid_t = tuple(float(w) for w in w_grid)
    best_idx = int(np.argmax(aps_t))
    floor_alone = float(aps_t[grid_t.index(0.0)]) if 0.0 in grid_t else float("nan")
    return BlendSweep(
        blend_id=blend_id,
        w_grid=grid_t,
        pair_ap=aps_t,
        best_w=grid_t[best_idx],
        best_pair_ap=aps_t[best_idx],
        floor_alone_pair_ap=floor_alone,
    )


def screen_combos(
    scorer,
    floor_preds: pd.DataFrame,
    honghanh_preds: pd.DataFrame,
    nomannic_preds: pd.DataFrame,
    *,
    w_grid: Sequence[float] = DEFAULT_W_GRID,
    margin: float = DEFAULT_MARGIN,
) -> ScreenResult:
    """Run the full §60 combo screen and apply the pre-registered decision rule.

    Blends screened (all floor-guaranteed, w=0 allowed): honghanh↔floor,
    nomannic↔floor, {honghanh+nomannic averaged}↔floor — each on the COMMON confirmed
    pair intersection so all combos share ONE pair set. The floor-alone common-set
    PairAP is the baseline to beat.

    Winner = highest-PairAP combo whose best_w > 0 AND best_pair_ap > floor_alone +
    margin. Otherwise NULL (floor stays best; no submission).

    Args:
        scorer: A ``CanonicalScorer``.
        floor_preds: Floor dev-holdout frame (``pair_id``, ``risk_score``, ...).
        honghanh_preds: honghanh dev-holdout frame.
        nomannic_preds: nomannic dev-holdout frame.
        w_grid: Weights to sweep (includes 0.0).
        margin: Beat-the-floor margin.

    Returns:
        A :class:`ScreenResult`.
    """
    frames = {
        "floor": floor_preds,
        "honghanh": honghanh_preds,
        "nomannic": nomannic_preds,
    }
    common = common_confirmed_pairs(frames)
    if not common:
        raise ValueError(
            "no common confirmed pairs across floor/honghanh/nomannic frames; "
            "cannot screen on one common set (contract Rule 9 — refusing to fabricate)."
        )

    def risk_aligned(frame: pd.DataFrame) -> np.ndarray:
        by = dict(
            zip((str(p) for p in frame["pair_id"]), frame["risk_score"].astype(float))
        )
        return np.array([by[p] for p in common], dtype=float)

    floor_risk = risk_aligned(floor_preds)
    hh_risk = risk_aligned(honghanh_preds)
    nm_risk = risk_aligned(nomannic_preds)
    # The averaged pair lever = mean of the two levers' RANKS (rank-space average, so
    # the two verified heads contribute on the same monotone scale before blending).
    hh_nm_avg_rank = 0.5 * (_ranks(hh_risk) + _ranks(nm_risk))

    sweeps = [
        sweep_blend(scorer, "honghanh", common, floor_risk, hh_risk, w_grid),
        sweep_blend(scorer, "nomannic", common, floor_risk, nm_risk, w_grid),
        sweep_blend(scorer, "hh+nm_avg", common, floor_risk, hh_nm_avg_rank, w_grid),
    ]

    # Floor-alone baseline on the common set (identical across sweeps at w=0; take the
    # first, assert consistency for audit).
    floor_alone = sweeps[0].floor_alone_pair_ap
    for s in sweeps[1:]:
        # w=0 is floor rank alone regardless of lever → same PairAP up to float noise.
        assert abs(s.floor_alone_pair_ap - floor_alone) < 1e-9, (
            "floor-alone PairAP must be identical across blends at w=0 "
            f"({s.blend_id} gave {s.floor_alone_pair_ap} vs {floor_alone})"
        )

    result = ScreenResult(
        common_ids=common,
        floor_alone_pair_ap=floor_alone,
        sweeps=sweeps,
        margin=margin,
    )

    # §60 decision rule: best combo that beats floor beyond margin with best_w > 0.
    eligible = [
        s
        for s in sweeps
        if s.best_w > 0.0 and s.best_pair_ap > floor_alone + margin
    ]
    if eligible:
        winner = max(eligible, key=lambda s: s.best_pair_ap)
        result.winner_blend_id = winner.blend_id
        result.winner_w = winner.best_w
        result.winner_pair_ap = winner.best_pair_ap
        result.delta_over_floor = winner.best_pair_ap - floor_alone
        result.verdict = "WINNER"
    else:
        result.verdict = "NULL"
    return result


# --------------------------------------------------------------------------- #
# Winning combo → full eval submission (NEW path only; never overwrites live)  #
# --------------------------------------------------------------------------- #


def build_combo_eval_submission(
    floor_eval: pd.DataFrame,
    lever_eval: pd.DataFrame,
    w: float,
    out_path: Path,
) -> pd.DataFrame:
    """Build the full 112,540-row eval submission for the winning blend at ``w``.

    Rank-blends the floor's eval ``risk_score`` with the lever's eval ``risk_score``
    at the chosen ``w`` (SAME rank-blend as the screen), over the eval pair set. The
    behavior/evidence columns are REUSED VERBATIM from the floor eval artifact (the
    LB-verified 0.44519 behavior/evidence — the blend only changes ``risk_score``).

    Writes to ``out_path`` (a NEW path — this function NEVER overwrites the live
    ``submission.csv`` / ``submission_best_*.csv``; the caller passes combo_screen/).

    Args:
        floor_eval: Floor eval submission (112,540 rows, 8-col schema).
        lever_eval: Lever eval submission (same pair set).
        w: The winning blend weight (>0).
        out_path: Destination CSV path (parent created if absent).

    Returns:
        The written submission frame.
    """
    floor_eval = floor_eval.copy()
    floor_eval["pair_id"] = floor_eval["pair_id"].astype(str)
    lever_by = dict(
        zip(
            (str(p) for p in lever_eval["pair_id"]),
            lever_eval["risk_score"].astype(float),
        )
    )
    missing = [p for p in floor_eval["pair_id"] if p not in lever_by]
    if missing:
        raise ValueError(
            f"lever eval is missing {len(missing)} floor eval pair_ids "
            "(pair sets must match to rank-blend); refusing to fabricate."
        )
    lever_risk = np.array([lever_by[p] for p in floor_eval["pair_id"]], dtype=float)
    floor_risk = floor_eval["risk_score"].astype(float).to_numpy()

    fr = _ranks(floor_risk)
    lr = _ranks(lever_risk)
    blended = (1.0 - w) * fr + w * lr
    span = blended.max() - blended.min()
    norm = (blended - blended.min()) / span if span > 0 else np.full(len(blended), 0.5)

    out = floor_eval.copy()
    out["risk_score"] = norm.astype(float).clip(0.0, 1.0)
    # Reuse floor behavior/evidence columns verbatim (only risk_score changes).
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def validate_eval_submission(
    submission: pd.DataFrame,
    evaluation_pairs: pd.DataFrame,
) -> Dict[str, object]:
    """Validate the built eval submission against the §60 contract.

    Checks: 112,540 rows, 8-col schema, risk_score in [0,1], pair_ids == the
    evaluation_pairs set. Returns a dict of the checks (all must be True).
    """
    expected_cols = [
        "pair_id",
        "risk_score",
        "predicted_behavior",
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    ]
    sub_pairs = set(str(p) for p in submission["pair_id"])
    eval_pairs = set(str(p) for p in evaluation_pairs["pair_id"])
    rs = submission["risk_score"].astype(float)
    return {
        "n_rows": int(len(submission)),
        "n_rows_ok": len(submission) == 112_540,
        "schema_ok": list(submission.columns) == expected_cols,
        "risk_in_unit": bool((rs >= 0.0).all() and (rs <= 1.0).all()),
        "pairs_match_eval": sub_pairs == eval_pairs,
        "n_pairs": len(sub_pairs),
    }
