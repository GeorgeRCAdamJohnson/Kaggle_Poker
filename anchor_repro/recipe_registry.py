"""Ladder recipe-runner adapter — re-run reconstructable ladder recipes (task 5.8).

design.md "Components and Interfaces" §3 (LadderValidator / the injected
``recipe_runner``) and tasks.md 5.8. Requirements 2.1, 2.5, 2.6. Governed by the
workspace ``reverse-engineering-accountability`` contract.

Why this module exists (contract Rules 3 & 11 — debug the null, don't accept it).
--------------------------------------------------------------------------------
Task 5.5 measured a 0/9-scorable ladder. That was NOT a real null: it was an
EXECUTION GAP — :class:`anchor_repro.ladder.LadderValidator` had no injected
``recipe_runner``, so every artifact fell through to the honest BLOCKED default.
Investigation confirmed the ``submission_best_044519.csv`` recipe is fully
re-runnable from on-disk caches:

* ``feature_cache/v5/{dev,eval}_v5.parquet`` (25,860 dev rows × 60 cols, has
  ``pair_id``) — the ``feats5`` foundation block.
* ``feature_cache/v7/_MFg_{dev,eval}.npy`` (25,860 × 10) — the graded board-equity
  block ("MFg", dossier §32).
* ``feature_cache/v7/_DIRc_{dev,eval}.npy`` (25,860 × 8; cols in
  ``v7/_DIRc_cols.json``) — the drift-hardened directional block (dossier §33).

Together these are the ``cand_exp4c_directional`` LB-verified floor stack
(``tuning_harness/layers/registry.py`` :func:`known_good_floor_config`; dossier §33,
real LB 0.44519), re-run to a PU-stress holdout AP of ~0.3833 (dossier §34
whole-stack) / 0.3839 (§37 grounding note). This module closes the execution gap by
RE-RUNNING that recipe on the dev holdout exactly as ``poker/loopv.py`` builds it —
``Xfound = hstack([Xd5, Mgd, DIRd])`` fit as ONE XGB model on the seed-7 60/40
table-disjoint split — and emitting a dev-holdout prediction frame the
:class:`anchor_repro.scorer.CanonicalScorer` can score.

Reuse, not rebuild (contract Rule 6). This adapter does NOT recompute a single
feature: it loads the persisted caches verbatim (the exact matrices ``loopv.py``
wrote and the LB-verified stack was built from) and re-uses
:func:`poker_collusion.validation.cv.build_fold_solution` for the PU-correct scoring
target. It re-implements neither the metric, the CV split logic, nor the features.

The load-bearing honest constraint (contract Rules 1 & 9).
----------------------------------------------------------
The eval CSV (``submission_best_044519.csv``) has ``pair_id``\\s DISJOINT from the
dev labels, so it is NEVER dev-scored. Instead the recipe is re-run to produce
predictions over the DEV pairs, and only the CONFIRMED dev-holdout pairs (the ones
:func:`build_fold_solution` materialises) carry into the scored frame. The resulting
PairAP is a MEASURED LOCAL number that feeds only the RELATIVE rank-tracking gate —
it is NOT and cannot be an LB claim (Rule 1).

Two AP definitions differ, stated honestly (Rule 9).
----------------------------------------------------
``loopv.py`` records its holdout AP as ``average_precision_score`` over ALL holdout
pairs with PU sample weights (unknowns down-weighted). The canonical scorer computes
PairAP via the official metric over the CONFIRMED holdout pairs only (PU-correct;
unknowns never materialised). These are two different AP definitions on two different
row sets, so the canonical PairAP is EXPECTED to be in the neighbourhood of the
recorded ~0.3833–0.3839 but need not equal it. :func:`build_floor_recipe_runner`
records the loopv-style weighted-all AP alongside the canonical PairAP so the re-run
consistency can be checked without conflating the two, and any deviation is surfaced,
never silently accepted.

Per-artifact reconstructability (honest BLOCKED, contract Rules 3 & 9).
----------------------------------------------------------------------
Only ``submission_best_044519.csv`` has a documented, cache-complete recipe (the
floor stack). The other eight ladder artifacts (0.18462…0.41289) are earlier points
on the trajectory whose EXACT feature stacks are not reconstructable from the
versioned caches alone — the specific per-artifact feature composition that produced
each historical LB score is not recorded as a cache manifest, and fabricating one
would violate contract Rule 3 (no forced story) and Rule 9 (state the uncertainty).
So :func:`build_ladder_recipe_runner` returns a runner that reconstructs 044519 and
returns ``None`` (⇒ honest BLOCKED) for the other eight, and
:func:`reconstruction_report` names, per BLOCKED artifact, the specific missing
recipe input so the ladder point's ``blocked_reason`` can cite it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.validation.cv import build_fold_solution

from anchor_repro.artifact_names import ArtifactLB
from anchor_repro.ladder import RecipeRerun, RecipeRunner

__all__ = [
    "FLOOR_DIGITS",
    "FLOOR_RECIPE_ID",
    "FloorRerunResult",
    "ReconstructionStatus",
    "CacheLocations",
    "default_cache_locations",
    "build_floor_recipe_runner",
    "build_ladder_recipe_runner",
    "reconstruction_report",
    "MissingCacheError",
]

# --------------------------------------------------------------------------- #
# The one artifact whose recipe is documented + cache-complete (dossier §33).  #
# --------------------------------------------------------------------------- #

#: The LB digits of the floor artifact (``submission_best_044519.csv``, real LB
#: 0.44519). The ONLY ladder artifact whose exact recipe is reconstructable from the
#: on-disk caches (the ``cand_exp4c_directional`` floor stack, dossier §33).
FLOOR_DIGITS: str = "044519"

#: Human-readable id of the floor stack recipe (foundation v5 + MFg + DIRc).
FLOOR_RECIPE_ID: str = "cand_exp4c_directional"

#: The floor model's XGB hyperparameters, VERBATIM from ``poker/loopv.py`` ``hap()``
#: (the function that produced the recorded 0.3833/0.3839 holdout AP). Reused, not
#: re-tuned — changing these would no longer reproduce the recorded floor.
_FLOOR_XGB_PARAMS: Dict[str, object] = dict(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=6,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

#: The seed-7 60/40 table-disjoint split constants, VERBATIM from ``loopv.py``.
_SPLIT_SEED: int = 7
_SPLIT_CUT_FRAC: float = 0.6

#: The eval row count used for the PU sample-weight rescale in ``loopv.py``
#: (``112540 / n_pu``). Reused verbatim so the re-run matches the recorded floor.
_PU_WEIGHT_EVAL_ROWS: int = 112_540

#: Bookkeeping columns dropped from ``dev_v5.parquet`` to get the ``feats5``
#: foundation block (verbatim from ``loopv.py``: everything except these).
_V5_DROP_COLUMNS: Tuple[str, ...] = ("pair_id", "p_low", "p_high", "label", "behavior_family")

#: The recorded floor holdout AP band (dossier §34 whole-stack 0.3833; §37 grounding
#: note 0.3839). Used ONLY for the re-run consistency note, never as a pass/fail gate.
FLOOR_HOLDOUT_AP_LOW: float = 0.3833
FLOOR_HOLDOUT_AP_HIGH: float = 0.3839


class MissingCacheError(FileNotFoundError):
    """Raised when a cache the floor recipe needs is absent from disk.

    Surfaced so the adapter can report the SPECIFIC missing cache path (contract
    Rule 9 — name what is missing), rather than a silent failure.
    """


@dataclass(frozen=True)
class CacheLocations:
    """Resolved on-disk paths of the feature caches the floor recipe reuses.

    Every path is a persisted artifact ``loopv.py`` / the LB-verified stack was built
    from — this adapter reads them verbatim and never recomputes a feature (contract
    Rule 6).

    Attributes:
        dev_v5: ``feature_cache/v5/dev_v5.parquet`` (foundation ``feats5`` + pair_id).
        dev_pairs_v2: ``feature_cache/v2/dev_pairs_v2.parquet`` (pair_id, p_low,
            p_high, label, behavior_family — the PU label + pool-key source).
        mfg_dev: ``feature_cache/v7/_MFg_dev.npy`` (graded board-equity, 10 cols).
        dirc_dev: ``feature_cache/v7/_DIRc_dev.npy`` (directional block, 8 cols).
        seats: ``data/poker/seats.parquet`` (hand_id, player_id — for the pool map).
        hands: ``data/poker/hands.parquet`` (hand_id, table_id — for the pool map).
    """

    dev_v5: Path
    dev_pairs_v2: Path
    mfg_dev: Path
    dirc_dev: Path
    seats: Path
    hands: Path

    def missing(self) -> List[Path]:
        """Return the subset of required cache paths that are absent from disk."""
        return [p for p in (self.dev_v5, self.dev_pairs_v2, self.mfg_dev, self.dirc_dev, self.seats, self.hands) if not Path(p).is_file()]


def default_cache_locations(poker_root: Path) -> CacheLocations:
    """Resolve the standard cache locations under a ``poker/`` tree root.

    Args:
        poker_root: The ``.../kAGGLE/poker`` directory containing ``outputs`` and
            ``data``.

    Returns:
        A :class:`CacheLocations` with the verbatim on-disk paths.
    """
    root = Path(poker_root)
    fc = root / "outputs" / "poker_collusion" / "feature_cache"
    data = root / "data" / "poker"
    return CacheLocations(
        dev_v5=fc / "v5" / "dev_v5.parquet",
        dev_pairs_v2=fc / "v2" / "dev_pairs_v2.parquet",
        mfg_dev=fc / "v7" / "_MFg_dev.npy",
        dirc_dev=fc / "v7" / "_DIRc_dev.npy",
        seats=data / "seats.parquet",
        hands=data / "hands.parquet",
    )


@dataclass(frozen=True)
class FloorRerunResult:
    """The measured outcome of re-running the floor stack on the dev holdout.

    Attributes:
        dev_predictions: The dev-holdout prediction frame (production-metric schema)
            restricted to the CONFIRMED holdout pairs, ready for the CanonicalScorer.
        n_holdout_all: Number of ALL dev pairs in the holdout split (incl. PU
            unknowns) — the row set ``loopv.py`` computes its weighted AP over.
        n_holdout_confirmed: Number of CONFIRMED holdout pairs (the row set the
            canonical PairAP is computed over).
        loopv_weighted_ap: The loopv-style ``average_precision_score`` over ALL
            holdout pairs with PU sample weights — the number directly comparable to
            the recorded 0.3833/0.3839 (a MEASURED local re-run consistency figure,
            NOT an LB claim; contract Rules 1, 9).
        n_features: Total feature columns in the reconstructed ``Xfound`` (59 v5
            foundation + 10 MFg + 8 DIRc = 77, matching dossier §33's "77 feats total";
            recorded for audit).
    """

    dev_predictions: pd.DataFrame
    n_holdout_all: int
    n_holdout_confirmed: int
    loopv_weighted_ap: float
    n_features: int


# --------------------------------------------------------------------------- #
# The floor recipe re-run (dossier §33/§34, verbatim from loopv.py).           #
# --------------------------------------------------------------------------- #


def _load_pool_map(seats_path: Path, hands_path: Path) -> Dict[object, object]:
    """Build ``{player_id: table_id}`` from seats+hands (projected reads only).

    Mirrors ``loopv.py``'s ``p2t`` construction (mode = the table a player appears at
    most). Column-projected — never touches ``actions.parquet``.
    """
    seats = pd.read_parquet(seats_path, columns=["hand_id", "player_id"])
    hands = pd.read_parquet(hands_path, columns=["hand_id", "table_id"])
    merged = seats.merge(hands, on="hand_id", how="left").dropna(subset=["table_id"])
    # Most-frequent table per player (loopv uses polars .mode().first()).
    mode = (
        merged.groupby("player_id")["table_id"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    )
    return mode.to_dict()


def rerun_floor_stack(
    caches: CacheLocations,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame] = None,
) -> FloorRerunResult:
    """Re-run the ``cand_exp4c_directional`` floor stack on the dev holdout.

    Reconstructs the LB-verified 0.44519 stack EXACTLY as ``poker/loopv.py`` builds
    it (contract Rule 6 — reuse the persisted caches, do not rebuild features):

    1. ``Xd5`` = the ``feats5`` foundation columns of ``dev_v5.parquet`` (all columns
       except the id/pool/label/behavior bookkeeping), inf/NaN → 0.
    2. ``Mgd`` = ``_MFg_dev.npy`` (10 graded board-equity cols).
    3. ``DIRd`` = ``_DIRc_dev.npy`` (8 directional cols).
    4. ``Xfound = hstack([Xd5, Mgd, DIRd])`` — ONE model over the whole stack.
    5. Seed-7 60/40 table-disjoint split; PU sample weights
       (``sw`` rescales unknowns by ``112540/n_pu``; ``fw`` = 2 for positives, 1 for
       confirmed, 0.35 for unknowns) — all verbatim from ``loopv.py``.
    6. Fit ONE XGB (the ``hap`` hyperparameters) on the train split, predict the
       holdout.

    Then it emits a canonical-scorer-ready dev-holdout prediction frame restricted to
    the CONFIRMED holdout pairs (the ones :func:`build_fold_solution` materialises),
    and records the loopv-style weighted-all AP for the re-run consistency check.

    Args:
        caches: Resolved on-disk cache paths (see :class:`CacheLocations`).
        labels: The confirmed dev-label frame (``development_labels.csv``) — used to
            build the PU-correct scoring target's confirmed pair set.
        evidence: Optional dev-evidence frame for filling positive evidence slots.

    Returns:
        A :class:`FloorRerunResult`.

    Raises:
        MissingCacheError: If any required cache is absent (names the missing paths).
        ImportError: If ``xgboost`` is unavailable in the environment.
    """
    missing = caches.missing()
    if missing:
        raise MissingCacheError(
            "floor recipe cannot be reconstructed; missing cache file(s): "
            + ", ".join(str(p) for p in missing)
        )

    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    # --- 1-3: load the persisted stack verbatim (dev_v5 row order is authoritative;
    # the .npy caches were saved in that order by loopv.py, which hstacks them with no
    # reindex). We merge the PU label/pool keys from dev_pairs_v2 on pair_id keeping
    # dev_v5 order (exactly loopv's ``dev = dev_v5.merge(dp[...], on='pair_id')``).
    dev_v5 = pd.read_parquet(caches.dev_v5)
    dp = pd.read_parquet(caches.dev_pairs_v2)[["pair_id", "p_low", "p_high", "label"]]
    dev = dev_v5.merge(dp, on="pair_id", how="left")

    feats5 = [c for c in dev.columns if c not in _V5_DROP_COLUMNS]
    Xd5 = dev[feats5].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Mgd = np.load(caches.mfg_dev)
    DIRd = np.load(caches.dirc_dev)
    if not (len(Xd5) == len(Mgd) == len(DIRd) == len(dev)):
        raise MissingCacheError(
            "floor cache row-count mismatch (dev_v5="
            f"{len(Xd5)}, MFg={len(Mgd)}, DIRc={len(DIRd)}); the caches must be the "
            "aligned dev-row matrices loopv.py hstacks — refusing to fabricate an "
            "alignment (contract Rules 1, 9)."
        )
    Xfound = np.hstack([Xd5, Mgd, DIRd]).astype(np.float32)

    # --- 5: PU labels + sample weights (verbatim from loopv.py). ---
    lab = dev["label"].to_numpy()
    y = (lab == 1).astype(int)
    known = lab >= 0
    n_pu = int((lab == -1).sum())
    sw = np.where(known, 1.0, _PU_WEIGHT_EVAL_ROWS / max(n_pu, 1))
    fw = np.where(y == 1, 2.0, np.where(known, 1.0, 0.35))

    # --- 5: seed-7 60/40 table-disjoint split (verbatim from loopv.py). ---
    player_pool = _load_pool_map(caches.seats, caches.hands)
    dev_table = dev["p_low"].map(player_pool)
    tabs = sorted(set(str(t) for t in dev_table.fillna("NA")))
    rng = np.random.default_rng(_SPLIT_SEED)
    rng.shuffle(tabs)
    cut = int(len(tabs) * _SPLIT_CUT_FRAC)
    train_tabs = set(tabs[:cut])
    tr = dev_table.astype(str).isin(train_tabs).to_numpy()
    ho = ~tr

    # --- 6: fit ONE model on the train split, predict the holdout (verbatim hap). ---
    model = XGBClassifier(**_FLOOR_XGB_PARAMS)
    model.fit(Xfound[tr], y[tr], sample_weight=fw[tr])
    ho_proba = model.predict_proba(Xfound[ho])[:, 1]

    # loopv-style weighted AP over ALL holdout pairs (the number comparable to the
    # recorded 0.3833/0.3839). MEASURED local re-run figure, not an LB claim.
    loopv_weighted_ap = float(
        average_precision_score(y[ho], ho_proba, sample_weight=sw[ho])
    )

    # --- Emit a canonical-scorer-ready dev-holdout prediction frame. ---
    # Restrict to CONFIRMED holdout pairs (the pairs build_fold_solution materialises
    # — PU-correct: unknowns are never scored as negatives). The eval CSV is NEVER
    # touched here (its pairs are disjoint from these dev pairs).
    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    ho_pairs = dev["pair_id"].astype(str).to_numpy()[ho]
    ho_scores = ho_proba
    keep = np.array([pid in confirmed_ids for pid in ho_pairs])
    pred_pairs = ho_pairs[keep]
    pred_scores = ho_scores[keep]

    # Build the PU-correct solution over exactly these confirmed holdout pairs so the
    # emitted prediction frame's pair set matches what the CanonicalScorer will build.
    solution = build_fold_solution(labels, list(pred_pairs), evidence=evidence)
    solution_ids = {str(pid) for pid in solution["pair_id"]}
    # Align predictions to the solution's confirmed pairs (build_fold_solution may drop
    # any id lacking a confirmed status; keep only the intersection so pair sets match).
    score_by_pair = dict(zip((str(p) for p in pred_pairs), (float(s) for s in pred_scores)))
    kept_pairs = [pid for pid in solution["pair_id"].astype(str) if pid in score_by_pair]

    preds = pd.DataFrame({"pair_id": kept_pairs})
    preds["risk_score"] = [score_by_pair[pid] for pid in kept_pairs]
    # Clip to the unit interval the metric requires (probabilities already in [0,1]).
    preds["risk_score"] = preds["risk_score"].astype(float).clip(0.0, 1.0)
    # behavior + evidence heads are not what this re-run reconstructs (the floor
    # artifact's behavior/evidence heads are byte-identical to the 0.41289 schema per
    # dossier §33); fill the metric-required columns so the frame is scoreable. PairAP
    # (the PRIMARY gate quantity) depends only on risk_score, so these do not affect it.
    preds["predicted_behavior"] = "none"
    for col in SUBMISSION_COLUMNS:
        if col not in preds.columns:
            preds[col] = "NO_EVIDENCE"
    preds = preds[list(SUBMISSION_COLUMNS)]

    return FloorRerunResult(
        dev_predictions=preds,
        n_holdout_all=int(ho.sum()),
        n_holdout_confirmed=int(len(preds)),
        loopv_weighted_ap=loopv_weighted_ap,
        n_features=int(Xfound.shape[1]),
    )


# --------------------------------------------------------------------------- #
# Per-artifact reconstructability (honest BLOCKED reasons, contract Rule 9).   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReconstructionStatus:
    """Whether a ladder artifact's recipe is reconstructable, and why not if not.

    Attributes:
        digits: The artifact's LB digits (e.g. ``"044519"``).
        reconstructable: ``True`` iff a cache-complete, documented recipe exists.
        recipe_id: The recipe id when reconstructable (else empty).
        missing_input: The SPECIFIC missing recipe input naming why a non-
            reconstructable artifact is BLOCKED (contract Rule 9). Empty when
            reconstructable.
    """

    digits: str
    reconstructable: bool
    recipe_id: str = ""
    missing_input: str = ""


#: Per-artifact reconstruction verdicts. Only the floor (044519) is cache-complete;
#: the eight earlier trajectory points have no recorded cache manifest pinning their
#: exact feature stack, so they are honestly BLOCKED with the specific missing input
#: named (contract Rules 3 & 9 — no fabricated recipe).
_LADDER_DIGITS: Tuple[str, ...] = (
    "018462",
    "019029",
    "019281",
    "032580",
    "033131",
    "037063",
    "039372",
    "041289",
    "044519",
)


def _blocked_missing_input(digits: str) -> str:
    """The specific missing recipe input for a non-reconstructable ladder artifact."""
    return (
        f"no recorded cache manifest pins the exact feature stack that produced the "
        f"real LB {int(digits) / 100_000:.5f} artifact (submission_best_{digits}.csv); "
        "the versioned caches (v2/v3/v4/v6) hold intermediate feature blocks but the "
        "specific per-artifact composition + model config for this trajectory point is "
        "not documented as a reconstructable recipe (only the 0.44519 floor stack is — "
        "dossier §33). Reconstructing one would be a fabricated recipe (contract "
        "Rules 3 & 9), so this point is honestly BLOCKED."
    )


def reconstruction_report() -> Dict[str, ReconstructionStatus]:
    """Return the per-artifact reconstructability verdicts, keyed by LB digits.

    The single cache-complete, documented recipe is the floor stack (044519). Every
    other artifact is BLOCKED with a specific named missing input so a ladder point's
    ``blocked_reason`` can cite exactly what is unavailable (contract Rule 9).
    """
    report: Dict[str, ReconstructionStatus] = {}
    for digits in _LADDER_DIGITS:
        if digits == FLOOR_DIGITS:
            report[digits] = ReconstructionStatus(
                digits=digits,
                reconstructable=True,
                recipe_id=FLOOR_RECIPE_ID,
            )
        else:
            report[digits] = ReconstructionStatus(
                digits=digits,
                reconstructable=False,
                missing_input=_blocked_missing_input(digits),
            )
    return report


# --------------------------------------------------------------------------- #
# The RecipeRunner factories the LadderValidator consumes.                     #
# --------------------------------------------------------------------------- #


def build_floor_recipe_runner(
    caches: CacheLocations,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    evidence: Optional[pd.DataFrame] = None,
    on_result: Optional[Callable[[FloorRerunResult], None]] = None,
) -> RecipeRunner:
    """Build a :data:`anchor_repro.ladder.RecipeRunner` that re-runs ONLY the floor.

    The returned callback reconstructs the ``cand_exp4c_directional`` floor stack for
    ``submission_best_044519.csv`` (via :func:`rerun_floor_stack`) and returns a
    :class:`RecipeRerun` carrying the dev-holdout predictions; for every OTHER artifact
    it returns ``None`` (⇒ the ladder point is honestly BLOCKED).

    The re-run is cached (fitting the XGB once) so scoring the ladder repeatedly does
    not refit. If ``on_result`` is given it is called once with the
    :class:`FloorRerunResult` (so a caller can record the loopv-weighted AP + row
    counts for the re-run consistency note without re-running).

    Args:
        caches: Resolved cache paths.
        labels: Confirmed dev-label frame (for the PU-correct scoring target).
        canonical_recipe_id: The scorer's ``recipe.recipe_id`` — recorded on the
            :class:`RecipeRerun` so the :class:`LadderValidator`'s cross-regime guard
            passes (Req 2.6). The floor is scored under the same canonical recipe as
            every other point, so anchors stay mutually comparable.
        evidence: Optional dev-evidence frame.
        on_result: Optional one-shot callback receiving the :class:`FloorRerunResult`.

    Returns:
        A ``RecipeRunner`` callable ``(path, artifact) -> Optional[RecipeRerun]``.
    """
    cache: Dict[str, RecipeRerun] = {}

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if artifact.digits != FLOOR_DIGITS:
            return None  # honest BLOCKED for every non-floor artifact
        if FLOOR_DIGITS not in cache:
            result = rerun_floor_stack(caches, labels, evidence=evidence)
            if on_result is not None:
                on_result(result)
            detail = (
                f"re-ran {FLOOR_RECIPE_ID} floor stack (foundation v5 + MFg + DIRc, "
                f"{result.n_features} feats, one XGB, seed-7 60/40 table-disjoint) on "
                f"the dev holdout; canonical PairAP over {result.n_holdout_confirmed} "
                f"confirmed holdout pairs; loopv-style weighted-all AP over "
                f"{result.n_holdout_all} pairs = {result.loopv_weighted_ap:.4f} "
                f"(recorded floor {FLOOR_HOLDOUT_AP_LOW}-{FLOOR_HOLDOUT_AP_HIGH}; "
                "MEASURED local re-run figure, NOT an LB claim — contract Rules 1, 9)"
            )
            cache[FLOOR_DIGITS] = RecipeRerun(
                dev_predictions=result.dev_predictions,
                recipe_id=canonical_recipe_id,
                detail=detail,
            )
        return cache[FLOOR_DIGITS]

    return runner


def build_ladder_recipe_runner(
    caches: CacheLocations,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    evidence: Optional[pd.DataFrame] = None,
    on_result: Optional[Callable[[FloorRerunResult], None]] = None,
) -> RecipeRunner:
    """Build the full ladder runner (floor reconstructed, the other eight BLOCKED).

    Identical to :func:`build_floor_recipe_runner` today — the floor is the only
    cache-complete recipe, so this is the honest full-ladder runner: 044519 SCORABLE,
    the other eight BLOCKED (each with a specific named missing input via
    :func:`reconstruction_report`). Kept as a distinct entry point so a future task
    that reconstructs another trajectory point can extend it without changing callers.
    """
    return build_floor_recipe_runner(
        caches,
        labels,
        canonical_recipe_id=canonical_recipe_id,
        evidence=evidence,
        on_result=on_result,
    )
