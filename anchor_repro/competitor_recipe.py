"""Competitor-0.64469 dev-holdout recipe runner (§42 follow-up → ladder gate n=2).

design.md "Components and Interfaces" §3 (LadderValidator / the injected
``recipe_runner``); Requirements 2.1, 2.5, 2.6. Governed by the workspace
``reverse-engineering-accountability`` contract.

Why this module exists (§42 open follow-up; contract Rules 1, 2, 3, 9, 11, 16).
-------------------------------------------------------------------------------
Dossier §42 recorded the ladder gate as an INVESTIGATED NULL at n=1 — only the
``submission_best_044519.csv`` floor recovered a dev-holdout AP (via
:mod:`anchor_repro.recipe_registry`). §43 then SUCCESSFULLY reproduced the 0.64469
competitor notebook (``_refs/honghanh/detecting-collusive-value-transfer-in-6-max-
poker.ipynb``), leaving a WARM feature cache at
``outputs/poker_collusion/repro_064469/prepared_pu24k/``. This module closes the
follow-up by recovering the 0.64469 point's DEV-HOLDOUT PairAP so the ladder gate
reaches n=2 and a Spearman becomes computable at all.

The load-bearing constraint (contract Rule 1 — do NOT violate).
---------------------------------------------------------------
The 0.64469 ``submission.csv`` §43 produced is an EVAL submission (112,540 eval
pairs DISJOINT from the dev labels). It CANNOT be scored as a dev AP directly. To
get a ladder point comparable to the 044519 floor, the competitor recipe must be
RE-RUN on the DEV HOLDOUT (exactly as :mod:`recipe_registry` did for the floor) and
the resulting dev-pair predictions scored with the :class:`CanonicalScorer` under
the canonical recipe. This module does exactly that.

How the dev-holdout point is produced (faithful re-run, contract Rule 6 — reuse).
---------------------------------------------------------------------------------
The competitor's own recipe IS a table-disjoint holdout evaluation: 5-fold
``StratifiedGroupKFold`` grouped by ``table_id`` (a whole 30-player pool stays in
train or validation), so the OUT-OF-FOLD risk on each dev pair is a prediction made
by a model that never saw that pair's table — the definition of a table-disjoint dev
holdout. We therefore re-run the competitor's risk pipeline to produce OOF risk over
the 25,860 dev pairs, keep the OOF risk of the 1,860 CONFIRMED pairs (the ones the
:class:`CanonicalScorer` can PU-correctly score), and emit that as the dev-holdout
prediction frame. This mirrors :func:`recipe_registry.rerun_floor_stack`, whose
holdout is the seed-7 60/40 table split; here the holdout is the 5-fold grouped OOF.

To REUSE the competitor's exact feature + model code rather than re-implement it
(contract Rule 6, and to avoid a divergent re-build silently changing the recipe),
the heavy work runs in a CHILD PROCESS that imports the notebook's OWN functions
(exported via :func:`anchor_repro.repro_runner._export_notebook_to_script`) and its
warm ``prepared_pu24k`` cache. The child:

1. loads ``dev_pairs.parquet`` + ``dev_hand_features.parquet`` from the warm cache,
2. rebuilds the pair-level train matrix with the notebook's ``aggregate_pair_features``
   + ``add_player_baselines`` + ``add_partner_field_contrasts`` (verbatim),
3. runs the notebook's 5-fold ``StratifiedGroupKFold`` risk head + 3 family-OvR
   detectors with the notebook's exact ``risk_params`` / ``family_det_params`` /
   sample-weights (pos 2.0 / PU 0.35 / conf-neg 1.0),
4. forms the four risk combiners (overall / family_max / max_all / blend) and selects
   the one with the best PU-stress OOF AP (the notebook's own selection rule),
5. writes ``dev_holdout_oof_064469.parquet`` (``pair_id``, ``oof_risk``, per-combiner
   OOF risk, ``is_labeled``, ``is_pu``, ``label``) plus a small JSON of the measured
   PU-stress AP per combiner + the selected combiner.

The parent then loads that cache, keeps the CONFIRMED pairs' selected-combiner OOF
risk, and returns it as a :class:`RecipeRerun`. The re-run is cached on disk, so a
second ladder scoring reuses it (no refit).

Honesty (contract Rules 1, 9, 12).
----------------------------------
* The OOF dev-holdout PairAP this yields is a MEASURED LOCAL number. It is NOT the
  0.64469 leaderboard score and NOT the eval submission. It is labelled as such
  everywhere it is surfaced.
* The behavior/evidence heads are NOT reconstructed here — PairAP (the PRIMARY gate
  quantity) depends only on ``risk_score``, so the emitted frame fills
  ``predicted_behavior='none'`` / ``NO_EVIDENCE`` exactly as the floor runner does.
  The ``combined`` secondary score this produces therefore reflects PairAP with empty
  behavior/evidence heads and is recorded honestly as such, never as the competitor's
  full combined score.
* If the exact combiner cannot be reconstructed (missing cache, xgboost absent, a
  child failure), the runner records WHAT happened and returns ``None`` (⇒ the ladder
  point is honestly BLOCKED) — it never fabricates a number.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from poker_collusion.submission.writer import SUBMISSION_COLUMNS
from poker_collusion.validation.cv import build_fold_solution

from anchor_repro.artifact_names import ArtifactLB
from anchor_repro.ladder import RecipeRerun, RecipeRunner
from anchor_repro.recipe_registry import (
    FLOOR_DIGITS,
    CacheLocations,
    FloorRerunResult,
    build_floor_recipe_runner,
)
from anchor_repro.repro_runner import _export_notebook_to_script

__all__ = [
    "COMPETITOR_DIGITS",
    "COMPETITOR_RECIPE_ID",
    "COMPETITOR_REPORTED_PU_STRESS_AP",
    "CompetitorRerunResult",
    "CompetitorReconstructionError",
    "default_competitor_paths",
    "CompetitorPaths",
    "rerun_competitor_recipe",
    "build_competitor_recipe_runner",
    "build_two_point_ladder_runner",
]

#: The LB digits of the competitor artifact (real LB 0.64469).
COMPETITOR_DIGITS: str = "064469"

#: Human-readable id of the competitor recipe (PU-aware blend, dossier §43).
COMPETITOR_RECIPE_ID: str = "competitor_pu_value_transfer_064469"

#: The PU-stress OOF AP the notebook REPORTED for the winning ``blend`` combiner on
#: its 24k-PU run (writeup §6: blend 0.6174 vs overall 0.6055). Used ONLY for a
#: re-run consistency note, never as a pass/fail gate. This is the competitor's own
#: PU-stress-weighted AP over ALL dev pairs, a DIFFERENT AP definition than the
#: canonical confirmed-only PairAP — the two are not expected to be equal (Rule 9).
COMPETITOR_REPORTED_PU_STRESS_AP: float = 0.6174

#: Child wall-clock cap. A warm-cache re-run of just the risk + family CV (no eval
#: scoring, no evidence ranker) is far lighter than the full notebook; 1h is ample.
_CHILD_TIMEOUT_S: int = 3600

_OOF_CACHE_FILENAME: str = "dev_holdout_oof_064469.parquet"
_OOF_META_FILENAME: str = "dev_holdout_oof_064469_meta.json"


class CompetitorReconstructionError(RuntimeError):
    """Raised when the competitor dev-holdout re-run cannot be produced.

    Surfaced so the runner can report the SPECIFIC reason (missing cache, child
    failure) and return ``None`` (⇒ honest BLOCKED) rather than fabricate a number
    (contract Rules 1, 9).
    """


@dataclass(frozen=True)
class CompetitorPaths:
    """Resolved on-disk paths the competitor dev-holdout re-run reuses.

    Attributes:
        notebook: The 0.64469 competitor notebook (owned, trusted source; its OWN
            feature + model functions are imported by the child so the recipe is
            reused, not re-implemented).
        data_dir: The local competition data root (``poker/data/poker``), read-only.
        prepared_dir: The warm feature cache ``repro_064469/prepared_pu24k`` (§43).
        output_dir: Where the child writes the OOF cache + meta + log (``repro_064469``).
    """

    notebook: Path
    data_dir: Path
    prepared_dir: Path
    output_dir: Path

    def missing(self) -> List[Path]:
        """Return the required inputs that are absent from disk."""
        needed = [
            self.notebook,
            self.data_dir,
            self.prepared_dir / "dev_pairs.parquet",
            self.prepared_dir / "dev_hand_features.parquet",
            self.prepared_dir / "player_hands.parquet",
        ]
        return [p for p in needed if not Path(p).exists()]


def default_competitor_paths(poker_root: Union[str, Path]) -> CompetitorPaths:
    """Resolve the standard competitor re-run paths under a ``poker/`` tree root."""
    root = Path(poker_root)
    repro = root / "outputs" / "poker_collusion" / "repro_064469"
    return CompetitorPaths(
        notebook=root
        / "_refs"
        / "honghanh"
        / "detecting-collusive-value-transfer-in-6-max-poker.ipynb",
        data_dir=root / "data" / "poker",
        prepared_dir=repro / "prepared_pu24k",
        output_dir=repro,
    )


@dataclass(frozen=True)
class CompetitorRerunResult:
    """The measured outcome of re-running the competitor recipe on the dev holdout.

    Attributes:
        dev_predictions: The dev-holdout prediction frame (production-metric schema)
            restricted to the CONFIRMED dev pairs, ready for the CanonicalScorer.
        n_confirmed: Number of confirmed dev pairs carried into the scored frame.
        n_dev_all: Number of ALL dev pairs the OOF was produced over (incl. PU).
        selected_combiner: The risk combiner the notebook's PU-stress rule selected
            ("overall" / "family_max" / "max_all" / "blend").
        pu_stress_ap: The MEASURED PU-stress OOF AP of the selected combiner over ALL
            dev pairs — the number directly comparable to the notebook's reported
            0.6174 (a re-run consistency figure, NOT an LB claim; Rules 1, 9).
        combiner_pu_stress_ap: PU-stress OOF AP of every combiner (audit trail).
        n_features: Number of pair-level features in the reconstructed matrix (audit).
        reused_cache: ``True`` if a prior on-disk OOF cache was reused (no refit).
    """

    dev_predictions: pd.DataFrame
    n_confirmed: int
    n_dev_all: int
    selected_combiner: str
    pu_stress_ap: float
    combiner_pu_stress_ap: Dict[str, float]
    n_features: int
    reused_cache: bool


# --------------------------------------------------------------------------- #
# The child script: reuse the notebook's OWN functions to compute dev OOF.     #
# --------------------------------------------------------------------------- #

#: The child-script footer. Placed AFTER the exported notebook body so the
#: notebook's ``main()`` / EDA drivers are NOT invoked (we import its functions only)
#: and we run only the risk + family CV to produce dev-holdout OOF. Authored here
#: (trusted). Uses ``repr`` for Windows-safe path literals.
_CHILD_FOOTER_TEMPLATE = '''\
# ==== competitor_recipe child footer (AUTHORED BY THE RUNNER) ====
# Reuse the notebook's OWN feature + model code (imported above) to produce the
# DEV-HOLDOUT OOF risk. We do NOT call the notebook's main() (that builds the EVAL
# submission); we replicate ONLY its risk + family-OvR CV on the warm dev cache, so
# the recipe is reused verbatim, not re-implemented (contract Rule 6).
import json as _cr_json
import numpy as _cr_np
import pandas as _cr_pd
import polars as _cr_pl
from pathlib import Path as _CrPath
from sklearn.model_selection import StratifiedGroupKFold as _CrSGKF
from xgboost import XGBClassifier as _CrXGB

_CR_PREP = _CrPath({prepared_literal})
_CR_OUT = _CrPath({output_literal})
_CR_DATA = _CrPath({data_literal})
_CR_OOF_PATH = _CR_OUT / {oof_name!r}
_CR_META_PATH = _CR_OUT / {meta_name!r}

print("competitor_recipe: prepared_dir =", _CR_PREP, flush=True)

_cr_dev_pairs = _cr_pl.read_parquet(_CR_PREP / "dev_pairs.parquet")
_cr_dev_hand_path = _CR_PREP / "dev_hand_features.parquet"
_cr_player_hand_path = _CR_PREP / "player_hands.parquet"

# Rebuild the pair-level train matrix with the notebook's OWN functions (verbatim).
_cr_hands = _cr_pl.scan_parquet(_CR_DATA / "hands.parquet")
_cr_seats = _cr_pl.scan_parquet(_CR_DATA / "seats.parquet")
_cr_eval_pairs = _cr_pl.read_csv(_CR_DATA / "evaluation_pairs.csv")

print("competitor_recipe: aggregating pair features (notebook code)...", flush=True)
_cr_train = aggregate_pair_features(_cr_dev_hand_path, _cr_dev_pairs)
_cr_train = add_player_baselines(
    _cr_train, player_phase_baselines(_cr_player_hand_path, "development")
)
_cr_train = add_partner_field_contrasts(
    _cr_train, _cr_seats, _cr_hands, _cr_player_hand_path, "development"
)

_cr_exclude = {{
    "pair_id", "player_1", "player_2", "label", "behavior_family",
    "table_id", "is_labeled", "is_pu",
}}
_cr_feature_cols = [
    c for c, dtype in _cr_train.schema.items()
    if dtype.is_numeric() and c not in _cr_exclude
]
_cr_X = matrix_from(_cr_train, _cr_feature_cols)
_cr_y = _cr_train["label"].fill_null(0).to_numpy().astype(_cr_np.int8)
_cr_is_labeled = _cr_train["is_labeled"].to_numpy().astype(bool)
_cr_is_pu = _cr_train["is_pu"].to_numpy().astype(bool)
_cr_n_eval = len(_cr_eval_pairs)
_cr_fit_weight = _cr_np.where(
    _cr_y == 1, POS_WEIGHT, _cr_np.where(_cr_is_pu, PU_WEIGHT, 1.0)
).astype(_cr_np.float32)
_cr_behavior_y = _cr_np.array(
    [BEHAVIOR_TO_ID.get(x, 0) for x in _cr_train["behavior_family"].to_list()],
    dtype=_cr_np.int8,
)
_cr_groups = _cr_train["table_id"].fill_null("NA").to_numpy()
print(
    f"competitor_recipe: train matrix {{_cr_X.shape[0]}} x {{_cr_X.shape[1]}} | "
    f"pos={{int((_cr_y==1).sum())}} conf_neg={{int((_cr_is_labeled & (_cr_y==0)).sum())}} "
    f"PU={{int(_cr_is_pu.sum())}}",
    flush=True,
)

# The notebook's EXACT risk + family-OvR hyperparameters.
_cr_risk_params = dict(
    n_estimators=700, learning_rate=0.035, max_depth=4, min_child_weight=6,
    subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5,
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    early_stopping_rounds=70, n_jobs=-1,
)
_cr_family_det_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=3, min_child_weight=4,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=4,
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    early_stopping_rounds=50, n_jobs=-1,
)

_cr_cv = _CrSGKF(n_splits=5, shuffle=True, random_state=SEED)
_cr_oof_risk = _cr_np.zeros(len(_cr_train), dtype=_cr_np.float32)
_cr_oof_family = _cr_np.zeros((len(_cr_train), 3), dtype=_cr_np.float32)

for _cr_fold, (_cr_tr, _cr_va) in enumerate(
    _cr_cv.split(_cr_X, _cr_behavior_y, _cr_groups), start=1
):
    _cr_rm = _CrXGB(**_cr_risk_params, random_state=SEED + _cr_fold)
    _cr_rm.fit(
        _cr_X.iloc[_cr_tr], _cr_y[_cr_tr], sample_weight=_cr_fit_weight[_cr_tr],
        eval_set=[(_cr_X.iloc[_cr_va], _cr_y[_cr_va])], verbose=False,
    )
    _cr_oof_risk[_cr_va] = _cr_rm.predict_proba(_cr_X.iloc[_cr_va])[:, 1]
    for _cr_k, _cr_family in enumerate(BEHAVIORS):
        _cr_yk = (_cr_behavior_y == _cr_k + 1).astype(_cr_np.int8)
        _cr_wk = _cr_np.where(
            _cr_yk == 1, POS_WEIGHT, _cr_np.where(_cr_is_pu, PU_WEIGHT, 1.0)
        ).astype(_cr_np.float32)
        _cr_fp = dict(_cr_family_det_params)
        if int(_cr_yk[_cr_va].sum()) == 0:
            _cr_fp.pop("early_stopping_rounds", None)
        _cr_det = _CrXGB(**_cr_fp, random_state=SEED + _cr_fold * 10 + _cr_k)
        _cr_fit_kw = dict(sample_weight=_cr_wk[_cr_tr], verbose=False)
        if int(_cr_yk[_cr_va].sum()) > 0:
            _cr_fit_kw["eval_set"] = [(_cr_X.iloc[_cr_va], _cr_yk[_cr_va])]
        _cr_det.fit(_cr_X.iloc[_cr_tr], _cr_yk[_cr_tr], **_cr_fit_kw)
        _cr_oof_family[_cr_va, _cr_k] = _cr_det.predict_proba(_cr_X.iloc[_cr_va])[:, 1]
    print(f"competitor_recipe: fold {{_cr_fold}}/5 done", flush=True)

# The notebook's four risk combiners + its PU-stress selection rule (verbatim).
_cr_cands = combine_risk_candidates(_cr_oof_risk, _cr_oof_family)
_cr_pu_ap = {{
    name: pu_stress_ap(_cr_y, scores, _cr_is_pu, _cr_n_eval)
    for name, scores in _cr_cands.items()
}}
_cr_best = max(_cr_pu_ap, key=lambda n: _cr_pu_ap[n])
print("competitor_recipe: combiner PU-stress AP =", _cr_pu_ap, flush=True)
print("competitor_recipe: selected combiner =", _cr_best, flush=True)

# Persist the OOF risk over ALL dev pairs + every combiner (audit trail).
_cr_out_df = _cr_pd.DataFrame({{
    "pair_id": _cr_train["pair_id"].to_list(),
    "is_labeled": _cr_is_labeled,
    "is_pu": _cr_is_pu,
    "label": _cr_y,
    "oof_risk": _cr_cands[_cr_best].astype(_cr_np.float32),
}})
for _cr_name, _cr_scores in _cr_cands.items():
    _cr_out_df[f"oof_{{_cr_name}}"] = _cr_scores.astype(_cr_np.float32)
_cr_out_df.to_parquet(_CR_OOF_PATH, index=False)

_cr_meta = {{
    "selected_combiner": _cr_best,
    "combiner_pu_stress_ap": {{k: float(v) for k, v in _cr_pu_ap.items()}},
    "pu_stress_ap": float(_cr_pu_ap[_cr_best]),
    "n_dev_all": int(len(_cr_train)),
    "n_confirmed": int(_cr_is_labeled.sum()),
    "n_features": int(_cr_X.shape[1]),
    "n_eval_for_pu_weight": int(_cr_n_eval),
}}
_CR_META_PATH.write_text(_cr_json.dumps(_cr_meta, indent=2), encoding="utf-8")
print("competitor_recipe: wrote OOF cache", _CR_OOF_PATH, flush=True)
print("competitor_recipe: DONE", flush=True)
# ==== end competitor_recipe child footer ====
'''


def _build_child_script(paths: CompetitorPaths) -> str:
    """Assemble the child script: notebook body (functions only) + our OOF footer.

    The notebook's CODE cells are exported (``__future__`` hoisted) and its trailing
    top-level driver lines (``RUN_FULL_PIPELINE = True`` / ``main()`` / the EDA call)
    are neutralised so importing the body defines the functions WITHOUT running the
    full EVAL pipeline. Our footer then drives ONLY the dev-holdout risk CV.
    """
    future_imports, body = _export_notebook_to_script(Path(paths.notebook))

    # Neutralise the notebook's own top-level drivers so we import functions only.
    neutralised: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUN_FULL_PIPELINE") and "=" in stripped and "==" not in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            neutralised.append(f"{indent}RUN_FULL_PIPELINE = False  # neutralized: footer drives dev OOF")
            continue
        if line == stripped and (stripped == "main()" or stripped == "evidence_lift_eda()"):
            neutralised.append(f"# neutralized top-level call: {stripped}")
            continue
        neutralised.append(line)
    body = "\n".join(neutralised)

    footer = _CHILD_FOOTER_TEMPLATE.format(
        prepared_literal=repr(str(Path(paths.prepared_dir).resolve())),
        output_literal=repr(str(Path(paths.output_dir).resolve())),
        data_literal=repr(str(Path(paths.data_dir).resolve())),
        oof_name=_OOF_CACHE_FILENAME,
        meta_name=_OOF_META_FILENAME,
    )
    future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
    # Headless matplotlib so any imported EDA helper that touches pyplot cannot block.
    header = (
        "import os as _cr_os\n"
        '_cr_os.environ.setdefault("MPLBACKEND", "Agg")\n'
        "try:\n"
        "    import matplotlib as _cr_mpl\n"
        '    _cr_mpl.use("Agg", force=True)\n'
        "    import matplotlib.pyplot as _cr_plt\n"
        "    _cr_plt.show = lambda *a, **k: None\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    return future_block + header + body + "\n\n" + footer


def _run_child(paths: CompetitorPaths, timeout_s: int) -> Dict[str, object]:
    """Run the child OOF re-run; return the parsed meta dict. Raises on any failure."""
    output_dir = Path(paths.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "_competitor_dev_oof.py"
    log_path = output_dir / "_competitor_dev_oof.log"

    script = _build_child_script(paths)
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output_dir / ".mpl_cache")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    (output_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"# competitor_recipe dev-OOF re-run\n# notebook={paths.notebook}\n")
        log_handle.flush()
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(output_dir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise CompetitorReconstructionError(
                f"competitor dev-OOF re-run timed out after {timeout_s}s; see {log_path}"
            ) from exc

    if completed.returncode != 0:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        raise CompetitorReconstructionError(
            f"competitor dev-OOF child exited {completed.returncode}; log tail:\n{tail}"
        )

    meta_path = output_dir / _OOF_META_FILENAME
    if not meta_path.is_file():
        raise CompetitorReconstructionError(
            f"child exited 0 but wrote no {_OOF_META_FILENAME}; see {log_path}"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _emit_dev_predictions(
    oof_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build the canonical-scorer-ready dev-holdout frame from the OOF cache.

    Keeps the CONFIRMED dev pairs' selected-combiner OOF risk, restricted to the
    pairs :func:`build_fold_solution` materialises (PU-correct: unknowns never
    scored). Fills the metric-required behavior/evidence columns with the neutral
    values (PairAP — the PRIMARY gate — depends only on ``risk_score``), exactly as
    :func:`recipe_registry.rerun_floor_stack` does.
    """
    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    conf = oof_df[oof_df["is_labeled"].astype(bool)].copy()
    conf["pair_id"] = conf["pair_id"].astype(str)
    conf = conf[conf["pair_id"].isin(confirmed_ids)]

    score_by_pair = dict(zip(conf["pair_id"], conf["oof_risk"].astype(float)))

    # Align to the PU-correct solution's confirmed pairs so pair sets match exactly.
    solution = build_fold_solution(labels, list(score_by_pair.keys()), evidence=evidence)
    kept = [pid for pid in solution["pair_id"].astype(str) if pid in score_by_pair]

    preds = pd.DataFrame({"pair_id": kept})
    preds["risk_score"] = [score_by_pair[pid] for pid in kept]
    preds["risk_score"] = preds["risk_score"].astype(float).clip(0.0, 1.0)
    preds["predicted_behavior"] = "none"
    for col in SUBMISSION_COLUMNS:
        if col not in preds.columns:
            preds[col] = "NO_EVIDENCE"
    return preds[list(SUBMISSION_COLUMNS)]


def rerun_competitor_recipe(
    paths: CompetitorPaths,
    labels: pd.DataFrame,
    *,
    evidence: Optional[pd.DataFrame] = None,
    timeout_s: int = _CHILD_TIMEOUT_S,
    force: bool = False,
) -> CompetitorRerunResult:
    """Re-run the competitor recipe on the dev holdout → a scorable prediction frame.

    Reuses the warm ``prepared_pu24k`` cache and the notebook's OWN functions (in a
    child process) to produce OOF risk over the dev pairs, selects the combiner by the
    notebook's PU-stress rule, and emits the CONFIRMED pairs' OOF risk as a
    dev-holdout prediction frame the :class:`CanonicalScorer` can score.

    The OOF cache is written to ``output_dir`` and REUSED on a subsequent call (unless
    ``force=True``), so scoring the ladder repeatedly does not refit.

    Args:
        paths: Resolved competitor re-run paths.
        labels: The confirmed dev-label frame (for the PU-correct scoring target).
        evidence: Optional dev-evidence frame (for positive evidence slots).
        timeout_s: Child wall-clock cap.
        force: Recompute even if an OOF cache exists.

    Returns:
        A :class:`CompetitorRerunResult`.

    Raises:
        CompetitorReconstructionError: If a required input is missing or the child
            re-run fails (the runner turns this into an honest BLOCKED, never a
            fabricated number).
    """
    missing = paths.missing()
    if missing:
        raise CompetitorReconstructionError(
            "competitor recipe cannot be re-run; missing input(s): "
            + ", ".join(str(p) for p in missing)
        )

    output_dir = Path(paths.output_dir).resolve()
    oof_path = output_dir / _OOF_CACHE_FILENAME
    meta_path = output_dir / _OOF_META_FILENAME

    reused = False
    if oof_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reused = True
    else:
        meta = _run_child(paths, timeout_s)

    oof_df = pd.read_parquet(oof_path)
    preds = _emit_dev_predictions(oof_df, labels, evidence=evidence)

    return CompetitorRerunResult(
        dev_predictions=preds,
        n_confirmed=int(len(preds)),
        n_dev_all=int(meta.get("n_dev_all", len(oof_df))),
        selected_combiner=str(meta.get("selected_combiner", "")),
        pu_stress_ap=float(meta.get("pu_stress_ap", float("nan"))),
        combiner_pu_stress_ap={
            str(k): float(v) for k, v in dict(meta.get("combiner_pu_stress_ap", {})).items()
        },
        n_features=int(meta.get("n_features", oof_df.shape[1])),
        reused_cache=reused,
    )


def build_competitor_recipe_runner(
    paths: CompetitorPaths,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    evidence: Optional[pd.DataFrame] = None,
    timeout_s: int = _CHILD_TIMEOUT_S,
    force: bool = False,
    on_result: Optional[Callable[[CompetitorRerunResult], None]] = None,
) -> RecipeRunner:
    """Build a :data:`anchor_repro.ladder.RecipeRunner` for the 0.64469 competitor.

    The returned callback re-runs the competitor recipe on the dev holdout for
    ``submission_best_064469.csv`` (via :func:`rerun_competitor_recipe`) and returns a
    :class:`RecipeRerun` carrying the dev-holdout predictions under the canonical
    recipe id; for every OTHER artifact it returns ``None`` (⇒ honest BLOCKED). The
    re-run is cached so scoring the ladder repeatedly does not refit.

    Args:
        paths: Resolved competitor re-run paths.
        labels: Confirmed dev-label frame (for the PU-correct scoring target).
        canonical_recipe_id: The scorer's ``recipe.recipe_id`` — recorded on the
            :class:`RecipeRerun` so the ladder's cross-regime guard passes (Req 2.6).
        evidence: Optional dev-evidence frame.
        timeout_s: Child wall-clock cap.
        force: Recompute even if an OOF cache exists.
        on_result: Optional one-shot callback receiving the CompetitorRerunResult.

    Returns:
        A ``RecipeRunner`` callable ``(path, artifact) -> Optional[RecipeRerun]``.
    """
    cache: Dict[str, RecipeRerun] = {}

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if artifact.digits != COMPETITOR_DIGITS:
            return None  # honest BLOCKED for every non-competitor artifact
        if COMPETITOR_DIGITS not in cache:
            result = rerun_competitor_recipe(
                paths, labels, evidence=evidence, timeout_s=timeout_s, force=force
            )
            if on_result is not None:
                on_result(result)
            detail = (
                f"re-ran {COMPETITOR_RECIPE_ID} (5-fold StratifiedGroupKFold by "
                f"table_id, {result.n_features} pair feats, risk head + 3 family OvR, "
                f"combiner={result.selected_combiner}) on the dev holdout; OOF risk over "
                f"{result.n_dev_all} dev pairs, canonical PairAP over "
                f"{result.n_confirmed} confirmed pairs; PU-stress OOF AP="
                f"{result.pu_stress_ap:.4f} (notebook reported "
                f"{COMPETITOR_REPORTED_PU_STRESS_AP}; MEASURED local re-run figure, NOT "
                "an LB claim — contract Rules 1, 9)"
                + ("; reused OOF cache" if result.reused_cache else "")
            )
            cache[COMPETITOR_DIGITS] = RecipeRerun(
                dev_predictions=result.dev_predictions,
                recipe_id=canonical_recipe_id,
                detail=detail,
            )
        return cache[COMPETITOR_DIGITS]

    return runner


# --------------------------------------------------------------------------- #
# The two-point ladder runner (floor 044519 + competitor 064469).              #
# --------------------------------------------------------------------------- #


def build_two_point_ladder_runner(
    caches: CacheLocations,
    competitor_paths: CompetitorPaths,
    labels: pd.DataFrame,
    *,
    canonical_recipe_id: str,
    evidence: Optional[pd.DataFrame] = None,
    timeout_s: int = _CHILD_TIMEOUT_S,
    force: bool = False,
    on_floor_result: Optional[Callable[[FloorRerunResult], None]] = None,
    on_competitor_result: Optional[Callable[[CompetitorRerunResult], None]] = None,
) -> RecipeRunner:
    """Build a ``RecipeRunner`` returning BOTH scorable ladder points.

    Composes the floor runner (:func:`recipe_registry.build_floor_recipe_runner`,
    which reconstructs ``submission_best_044519.csv`` → dev-holdout PairAP) with the
    competitor runner (:func:`build_competitor_recipe_runner`, which re-runs the
    0.64469 recipe → dev-holdout PairAP). For ``044519`` the floor runner answers; for
    ``064469`` the competitor runner answers; for every OTHER artifact BOTH return
    ``None`` (⇒ honest BLOCKED). Scoring the resulting ladder with
    :meth:`LadderValidator.score_ladder` yields the two mutually-comparable
    (real_lb, local_pair_ap, local_combined) points that take the ladder gate to n=2
    so :meth:`LadderValidator.rank_tracking` can compute a Spearman at all.

    Both points are scored under the SAME ``canonical_recipe_id`` so they are mutually
    comparable and the ladder's cross-regime guard passes (Req 2.6). Each re-run is
    cached (floor fits once, competitor reuses its on-disk OOF cache).

    Args:
        caches: The floor-recipe feature caches (see :class:`CacheLocations`).
        competitor_paths: The competitor re-run paths (see :class:`CompetitorPaths`).
        labels: Confirmed dev-label frame (for the PU-correct scoring target).
        canonical_recipe_id: The scorer's ``recipe.recipe_id`` (recorded on both).
        evidence: Optional dev-evidence frame.
        timeout_s: Competitor child wall-clock cap.
        force: Recompute the competitor OOF even if a cache exists.
        on_floor_result: Optional one-shot callback for the FloorRerunResult.
        on_competitor_result: Optional one-shot callback for the CompetitorRerunResult.

    Returns:
        A ``RecipeRunner`` callable ``(path, artifact) -> Optional[RecipeRerun]``.
    """
    floor_runner = build_floor_recipe_runner(
        caches,
        labels,
        canonical_recipe_id=canonical_recipe_id,
        evidence=evidence,
        on_result=on_floor_result,
    )
    competitor_runner = build_competitor_recipe_runner(
        competitor_paths,
        labels,
        canonical_recipe_id=canonical_recipe_id,
        evidence=evidence,
        timeout_s=timeout_s,
        force=force,
        on_result=on_competitor_result,
    )

    def runner(path: Path, artifact: ArtifactLB) -> Optional[RecipeRerun]:
        if artifact.digits == FLOOR_DIGITS:
            return floor_runner(path, artifact)
        if artifact.digits == COMPETITOR_DIGITS:
            return competitor_runner(path, artifact)
        return None  # honest BLOCKED for the other trajectory artifacts

    return runner
