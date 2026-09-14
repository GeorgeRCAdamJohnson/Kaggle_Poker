"""Per-model 3-component metric decomposition on the dev holdout (dossier §73).

BUILD + MEASURE ONLY. This module NEVER submits to Kaggle and NEVER writes to
``submission.csv`` / ``submission_best_*.csv``. It writes only its own JSON summary
under ``outputs/poker_collusion/component_decomposition/``.

Why this exists (dossier §73; contract Rules 1, 6, 9).
------------------------------------------------------
The LB metric is ``0.70*PairAP + 0.20*EvidenceMAP@5 + 0.10*BehaviorMAP`` (dossier §1,
CONFIRMED). Every dev-holdout number measured so far is PairAP ONLY: the floor /
nomannic / competitor dev-holdout runners all fill ``predicted_behavior="none"`` and
``evidence="NO_EVIDENCE"`` (see :mod:`anchor_repro.recipe_registry`,
:mod:`anchor_repro.nomannic_recipe`, :mod:`anchor_repro.competitor_recipe`), so the
EvidenceMAP and BehaviorMAP components have NEVER been measured on the dev holdout.
This module measures all three components, per model, to locate the weak component and
bracket how much of honghanh's LB is ranking vs the two neglected components.

Coverage (contract Rule 9 — honest about what is measurable):
-------------------------------------------------------------
* honghanh (LB 0.64469): FULL 3-component breakdown. Its notebook has a real behavior
  head (per-family OvR / multiclass OOF + a positive-rate rule) AND a real evidence
  ranker (5-fold ``XGBRanker`` OOF blended with a per-family heuristic → top-5 hands
  per pair). We RE-RUN the notebook's OWN functions (contract Rule 6) in a child
  process on the warm ``prepared_pu24k`` cache to produce a dev-holdout frame that
  carries REAL behavior + evidence heads (not neutral placeholders), then score all
  three components with the :class:`CanonicalScorer`.
* floor (LB 0.44519) and nomannic (LB 0.56202): PairAP is measured (reusing their
  existing dev-holdout runners). Their runners expose ONLY a risk head — no dev
  behavior head and no dev evidence ranker are available without a large rebuild that
  is out of scope here (dossier §73 practical scope). So their behavior_map /
  evidence_map are reported as NOT-MEASURABLE-WITHOUT-REBUILD, never fabricated.

Implied-eval-PairAP brackets for honghanh (dossier §73 question 2):
-------------------------------------------------------------------
Given the real LB ``0.64262`` (honghanh's on-LB score for the emitted submission),
solve ``LB = 0.70*evalPairAP + 0.20*evalEvidence + 0.10*evalBehavior`` for evalPairAP
under two brackets:
  (a) eval evidence = behavior = 0  ->  evalPairAP = LB / 0.70
  (b) eval evidence/behavior = the MEASURED DEV values  ->
        evalPairAP = (LB - 0.20*devEvidence - 0.10*devBehavior) / 0.70
Bracket (a) is the upper bound on how much of the LB could be pure ranking; bracket
(b) assumes the eval components equal the dev ones (an assumption, stated as such —
contract Rules 4, 9: the dev components are NOT proven to transfer to eval).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from anchor_repro.competitor_recipe import CompetitorPaths, default_competitor_paths
from anchor_repro.nomannic_recipe import (
    default_nomannic_cache_locations,
    rerun_nomannic_recipe,
)
from anchor_repro.recipe_registry import default_cache_locations, rerun_floor_stack
from anchor_repro.repro_runner import _export_notebook_to_script
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.models import ScoringRecipe
from poker_collusion.config import PipelineConfig
from poker_collusion.submission.writer import SUBMISSION_COLUMNS

__all__ = [
    "HONGHANH_REAL_LB",
    "ComponentResult",
    "run_component_decomposition",
]

#: honghanh's on-LB score for the submission whose components we bracket (dossier §73).
HONGHANH_REAL_LB: float = 0.64262

#: Child wall-clock cap. The behavior + evidence heads on the WARM dev cache (risk CV
#: is already cached in the meta; we recompute the OOF family + evidence ranker) are
#: light; 1h is ample headroom.
_CHILD_TIMEOUT_S: int = 3600

_RICH_OOF_FILENAME: str = "dev_holdout_full_064469.parquet"
_RICH_META_FILENAME: str = "dev_holdout_full_064469_meta.json"

_NOT_MEASURABLE = "NOT-MEASURABLE-WITHOUT-REBUILD"


# --------------------------------------------------------------------------- #
# The augmented child footer — reuse the notebook's OWN behavior + evidence    #
# code to produce a dev-holdout frame with REAL heads (contract Rule 6).       #
# --------------------------------------------------------------------------- #

#: Footer placed AFTER the exported notebook body (functions only; the notebook's
#: main()/EDA drivers are neutralised). It replicates ONLY the notebook's dev-side
#: risk + family OOF + behavior assignment + evidence-ranker OOF, then writes a rich
#: dev-holdout frame (pair_id, is_labeled, label, oof_risk, predicted_behavior,
#: evidence_hand_1..5). AUTHORED here (trusted); uses repr() for Windows-safe paths.
_CHILD_FOOTER_TEMPLATE = '''\
# ==== component_decomposition child footer (AUTHORED BY THE RUNNER) ====
# Reuse the notebook's OWN feature + model + behavior + evidence code (imported above)
# to produce a DEV-HOLDOUT frame carrying REAL behavior + evidence heads. We do NOT
# call the notebook's main() (that builds the EVAL submission); we replicate ONLY its
# dev-side OOF risk + family OvR + behavior assignment + evidence-ranker OOF, so the
# heads are the notebook's real heads, not neutral placeholders (contract Rule 6).
import json as _cd_json
import gc as _cd_gc
import numpy as _cd_np
import pandas as _cd_pd
import polars as _cd_pl
from pathlib import Path as _CdPath
from sklearn.model_selection import StratifiedGroupKFold as _CdSGKF
from sklearn.metrics import average_precision_score as _cd_aps
from xgboost import XGBClassifier as _CdXGB, XGBRanker as _CdXGBRanker

_CD_PREP = _CdPath({prepared_literal})
_CD_OUT = _CdPath({output_literal})
_CD_DATA = _CdPath({data_literal})
_CD_OOF_PATH = _CD_OUT / {oof_name!r}
_CD_META_PATH = _CD_OUT / {meta_name!r}

print("component_decomposition: prepared_dir =", _CD_PREP, flush=True)

_cd_dev_pairs = _cd_pl.read_parquet(_CD_PREP / "dev_pairs.parquet")
_cd_dev_hand_path = _CD_PREP / "dev_hand_features.parquet"
_cd_player_hand_path = _CD_PREP / "player_hands.parquet"

_cd_hands = _cd_pl.scan_parquet(_CD_DATA / "hands.parquet")
_cd_seats = _cd_pl.scan_parquet(_CD_DATA / "seats.parquet")
_cd_eval_pairs = _cd_pl.read_csv(_CD_DATA / "evaluation_pairs.csv")
_cd_labels = _cd_pl.read_csv(_CD_DATA / "development_labels.csv")
_cd_dev_evidence = _cd_pl.read_csv(_CD_DATA / "development_evidence.csv").select(
    ["pair_id", "hand_id"]
).unique()

# Rebuild the pair-level train matrix with the notebook's OWN functions (verbatim).
print("component_decomposition: aggregating pair features (notebook code)...", flush=True)
_cd_train = aggregate_pair_features(_cd_dev_hand_path, _cd_dev_pairs)
_cd_train = add_player_baselines(
    _cd_train, player_phase_baselines(_cd_player_hand_path, "development")
)
_cd_train = add_partner_field_contrasts(
    _cd_train, _cd_seats, _cd_hands, _cd_player_hand_path, "development"
)

_cd_exclude = {{
    "pair_id", "player_1", "player_2", "label", "behavior_family",
    "table_id", "is_labeled", "is_pu",
}}
_cd_feature_cols = [
    c for c, dtype in _cd_train.schema.items()
    if dtype.is_numeric() and c not in _cd_exclude
]
_cd_X = matrix_from(_cd_train, _cd_feature_cols)
_cd_y = _cd_train["label"].fill_null(0).to_numpy().astype(_cd_np.int8)
_cd_is_labeled = _cd_train["is_labeled"].to_numpy().astype(bool)
_cd_is_pu = _cd_train["is_pu"].to_numpy().astype(bool)
_cd_n_eval = len(_cd_eval_pairs)
_cd_fit_weight = _cd_np.where(
    _cd_y == 1, POS_WEIGHT, _cd_np.where(_cd_is_pu, PU_WEIGHT, 1.0)
).astype(_cd_np.float32)
_cd_behavior_y = _cd_np.array(
    [BEHAVIOR_TO_ID.get(x, 0) for x in _cd_train["behavior_family"].to_list()],
    dtype=_cd_np.int8,
)
_cd_groups = _cd_train["table_id"].fill_null("NA").to_numpy()
print(
    f"component_decomposition: train {{_cd_X.shape[0]}} x {{_cd_X.shape[1]}} | "
    f"pos={{int((_cd_y==1).sum())}} PU={{int(_cd_is_pu.sum())}}",
    flush=True,
)

# The notebook's EXACT risk + family-OvR + multiclass-behavior hyperparameters.
_cd_risk_params = dict(
    n_estimators=700, learning_rate=0.035, max_depth=4, min_child_weight=6,
    subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5,
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    early_stopping_rounds=70, n_jobs=-1,
)
_cd_family_det_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=3, min_child_weight=4,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=4,
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    early_stopping_rounds=50, n_jobs=-1,
)
_cd_behavior_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=3, min_child_weight=4,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=4,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    tree_method="hist", early_stopping_rounds=50, n_jobs=-1,
)

_cd_cv = _CdSGKF(n_splits=5, shuffle=True, random_state=SEED)
_cd_oof_risk = _cd_np.zeros(len(_cd_train), dtype=_cd_np.float32)
_cd_oof_family_det = _cd_np.zeros((len(_cd_train), 3), dtype=_cd_np.float32)
_cd_oof_behavior = _cd_np.zeros((len(_cd_train), 3), dtype=_cd_np.float32)
_cd_fold_id = _cd_np.full(len(_cd_train), -1, dtype=_cd_np.int8)

for _cd_fold, (_cd_tr, _cd_va) in enumerate(
    _cd_cv.split(_cd_X, _cd_behavior_y, _cd_groups)
):
    _cd_fold_id[_cd_va] = _cd_fold
    _cd_rm = _CdXGB(**_cd_risk_params, random_state=SEED + _cd_fold + 1)
    _cd_rm.fit(
        _cd_X.iloc[_cd_tr], _cd_y[_cd_tr], sample_weight=_cd_fit_weight[_cd_tr],
        eval_set=[(_cd_X.iloc[_cd_va], _cd_y[_cd_va])], verbose=False,
    )
    _cd_oof_risk[_cd_va] = _cd_rm.predict_proba(_cd_X.iloc[_cd_va])[:, 1]
    # Family OvR detectors (3 one-vs-rest) — the notebook's family head.
    for _cd_k, _cd_family in enumerate(BEHAVIORS):
        _cd_yk = (_cd_behavior_y == _cd_k + 1).astype(_cd_np.int8)
        _cd_wk = _cd_np.where(
            _cd_yk == 1, POS_WEIGHT, _cd_np.where(_cd_is_pu, PU_WEIGHT, 1.0)
        ).astype(_cd_np.float32)
        _cd_fp = dict(_cd_family_det_params)
        if int(_cd_yk[_cd_va].sum()) == 0:
            _cd_fp.pop("early_stopping_rounds", None)
        _cd_det = _CdXGB(**_cd_fp, random_state=SEED + _cd_fold * 10 + _cd_k)
        _cd_fit_kw = dict(sample_weight=_cd_wk[_cd_tr], verbose=False)
        if int(_cd_yk[_cd_va].sum()) > 0:
            _cd_fit_kw["eval_set"] = [(_cd_X.iloc[_cd_va], _cd_yk[_cd_va])]
        _cd_det.fit(_cd_X.iloc[_cd_tr], _cd_yk[_cd_tr], **_cd_fit_kw)
        _cd_oof_family_det[_cd_va, _cd_k] = _cd_det.predict_proba(_cd_X.iloc[_cd_va])[:, 1]
    # Multiclass behavior model (trained on positives only) — the notebook's other head.
    _cd_pos_tr = _cd_tr[_cd_behavior_y[_cd_tr] > 0]
    _cd_pos_va = _cd_va[_cd_behavior_y[_cd_va] > 0]
    if len(_cd_pos_tr):
        _cd_bp = dict(_cd_behavior_params)
        if len(_cd_pos_va) == 0:
            _cd_bp.pop("early_stopping_rounds", None)
        _cd_bm = _CdXGB(**_cd_bp, random_state=SEED + _cd_fold)
        _cd_bkw = dict(verbose=False)
        if len(_cd_pos_va):
            _cd_bkw["eval_set"] = [(_cd_X.iloc[_cd_pos_va], _cd_behavior_y[_cd_pos_va] - 1)]
        _cd_bm.fit(_cd_X.iloc[_cd_pos_tr], _cd_behavior_y[_cd_pos_tr] - 1, **_cd_bkw)
        _cd_oof_behavior[_cd_va] = _cd_bm.predict_proba(_cd_X.iloc[_cd_va])
    print(f"component_decomposition: risk/family fold {{_cd_fold + 1}}/5 done", flush=True)

# --- Behavior head: pick OvR vs multiclass by positive-family accuracy (notebook rule).
_cd_family_multi = _cd_oof_behavior.argmax(1)
_cd_family_ovr = _cd_oof_family_det.argmax(1)
_cd_pos = _cd_y == 1
_cd_acc_multi = float((_cd_family_multi[_cd_pos] + 1 == _cd_behavior_y[_cd_pos]).mean()) if _cd_pos.any() else 0.0
_cd_acc_ovr = float((_cd_family_ovr[_cd_pos] + 1 == _cd_behavior_y[_cd_pos]).mean()) if _cd_pos.any() else 0.0
_cd_use_ovr = _cd_acc_ovr >= _cd_acc_multi
_cd_family_oof = _cd_family_ovr if _cd_use_ovr else _cd_family_multi
print(
    f"component_decomposition: positive-family accuracy multi={{_cd_acc_multi:.4f}} "
    f"ovr={{_cd_acc_ovr:.4f}} use_ovr={{_cd_use_ovr}}",
    flush=True,
)

# --- Positive-rate rule (notebook's select_positive_rate) on the DEV OOF. ---
_cd_rate_grid = _cd_np.array(
    [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02]
)
_cd_best_rate = select_positive_rate(_cd_oof_risk, _cd_family_oof, _cd_behavior_y, _cd_rate_grid)
print(f"component_decomposition: selected positive rate {{_cd_best_rate:.2%}}", flush=True)
# Assign predicted_behavior on the DEV pairs exactly as the notebook does on eval:
# active = top-rate by risk; active pairs -> family name, else "none".
_cd_active = top_mask(_cd_oof_risk, _cd_best_rate)
_cd_predicted_behavior = _cd_np.full(len(_cd_oof_risk), "none", dtype=object)
_cd_predicted_behavior[_cd_active] = _cd_np.array(BEHAVIORS)[_cd_family_oof[_cd_active]]

# --- Evidence head: the notebook's 5-fold XGBRanker OOF over dev positives' hands. ---
print("component_decomposition: training evidence ranker (notebook code)...", flush=True)
_cd_hand_feats = numeric_hand_features(_cd_dev_hand_path)
_cd_rank_base = [c for c in RANK_BASE_CANDIDATES if c in _cd_hand_feats][:20]
_cd_rank_pct = [f"{{c}}_pair_pct" for c in _cd_rank_base]
_cd_ev_feature_cols = _cd_hand_feats + _cd_rank_pct + FAMILY_FEATURES

_cd_positive_info = (
    _cd_labels.filter(_cd_pl.col("behavior_family").is_in(BEHAVIORS))
    .select(["pair_id", "behavior_family"])
    .with_columns(
        _cd_pl.when(_cd_pl.col("behavior_family") == "directed_transfer").then(1)
        .when(_cd_pl.col("behavior_family") == "soft_play").then(2)
        .otherwise(3).cast(_cd_pl.Int8).alias("behavior_id_true")
    )
)
_cd_ev_keys = _cd_dev_evidence.select(["pair_id", "hand_id"]).unique().with_columns(
    _cd_pl.lit(1, dtype=_cd_pl.Int8).alias("is_evidence")
)
_cd_pair_fold = _cd_pl.DataFrame({{
    "pair_id": _cd_train["pair_id"],
    "fold": _cd_fold_id,
    "behavior_id_pred": (_cd_family_oof + 1).astype(_cd_np.int8),
}})
_cd_ev_base = (
    _cd_pl.scan_parquet(_cd_dev_hand_path)
    .select(["pair_id", "hand_id"] + _cd_hand_feats)
    .join(_cd_positive_info.lazy(), on="pair_id")
    .join(_cd_ev_keys.lazy(), on=["pair_id", "hand_id"], how="left")
    .with_columns(_cd_pl.col("is_evidence").fill_null(0))
    .collect()
    .join(_cd_pair_fold, on="pair_id")
    .with_columns([
        (_cd_pl.col(c).rank(method="average").over("pair_id") / _cd_pl.len().over("pair_id"))
        .cast(_cd_pl.Float32).alias(f"{{c}}_pair_pct")
        for c in _cd_rank_base
    ])
    .with_row_index("_row")
)

_cd_rank_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=4, min_child_weight=2,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=4,
    objective="rank:pairwise", eval_metric="map@5", tree_method="hist", n_jobs=-1,
)
_cd_ev_oof = _cd_np.zeros(len(_cd_ev_base), dtype=_cd_np.float32)
for _cd_efold in range(5):
    _cd_tr_part = _cd_ev_base.filter(_cd_pl.col("fold") != _cd_efold)
    _cd_va_part = _cd_ev_base.filter(_cd_pl.col("fold") == _cd_efold)
    if _cd_va_part.height == 0 or _cd_tr_part.height == 0:
        continue
    _, _cd_xtr, _cd_ytr, _cd_gtr = evidence_matrix(_cd_tr_part, _cd_ev_feature_cols, "behavior_id_true")
    _cd_vo, _cd_xva, _cd_yva, _cd_gva = evidence_matrix(_cd_va_part, _cd_ev_feature_cols, "behavior_id_pred")
    _cd_rk = _CdXGBRanker(**_cd_rank_params, early_stopping_rounds=60, random_state=SEED + _cd_efold)
    _cd_rk.fit(_cd_xtr, _cd_ytr, group=_cd_gtr, eval_set=[(_cd_xva, _cd_yva)], eval_group=[_cd_gva], verbose=False)
    _cd_ev_oof[_cd_vo["_row"].to_numpy()] = _cd_rk.predict(_cd_xva)
    print(f"component_decomposition: evidence fold {{_cd_efold + 1}}/5 done", flush=True)

# Notebook's own OOF Evidence MAP@5 (dev), for a re-run consistency note.
_cd_notebook_ev_map5 = evidence_map5(
    _cd_ev_base["pair_id"].to_numpy(),
    _cd_ev_base["is_evidence"].to_numpy(),
    _cd_ev_oof,
)
print(f"component_decomposition: notebook OOF Evidence MAP@5 (dev) = {{_cd_notebook_ev_map5:.4f}}", flush=True)

# Blend the ranker OOF with the per-family heuristic (notebook's blend_within_pair),
# then take top-5 hands per pair (the notebook's evidence selection).
_cd_ev_flagged = add_family_flags(_cd_ev_base, "behavior_id_pred")
_cd_bid = _cd_ev_flagged["behavior_id_pred"].to_numpy()
_cd_heur = _cd_np.where(
    _cd_bid == 1, _cd_ev_flagged["directed_signal"].to_numpy(),
    _cd_np.where(_cd_bid == 2, _cd_ev_flagged["soft_signal"].to_numpy(),
    _cd_np.where(_cd_bid == 3, _cd_ev_flagged["isolation_signal"].to_numpy(),
                 _cd_ev_flagged["other_signal"].to_numpy())),
).astype(_cd_np.float32)
# _cd_ev_base rows are in _row order; _cd_ev_oof is aligned to that order.
_cd_blended = blend_within_pair(_cd_ev_base, _cd_ev_oof, _cd_heur)
_cd_topk = (
    _cd_ev_base.select(["pair_id", "hand_id"])
    .with_columns(_cd_pl.Series("_score", _cd_blended))
    .group_by("pair_id")
    .agg(_cd_pl.col("hand_id").sort_by("_score", descending=True).head(5).alias("hands"))
    .with_columns([
        _cd_pl.col("hands").list.get(i, null_on_oob=True).fill_null("NO_EVIDENCE").alias(f"evidence_hand_{{i + 1}}")
        for i in range(5)
    ])
    .drop("hands")
)

# --- Assemble the rich dev-holdout frame (ALL dev pairs; scorer keeps confirmed). ---
_cd_out = _cd_pd.DataFrame({{
    "pair_id": _cd_train["pair_id"].to_list(),
    "is_labeled": _cd_is_labeled,
    "label": _cd_y,
    "oof_risk": _cd_oof_risk.astype(_cd_np.float32),
    "predicted_behavior": _cd_predicted_behavior.astype(str),
}})
_cd_ev_wide_pd = _cd_topk.to_pandas()
_cd_out = _cd_out.merge(_cd_ev_wide_pd, on="pair_id", how="left")
for _cd_i in range(1, 6):
    _cd_col = f"evidence_hand_{{_cd_i}}"
    if _cd_col not in _cd_out.columns:
        _cd_out[_cd_col] = "NO_EVIDENCE"
    _cd_out[_cd_col] = _cd_out[_cd_col].fillna("NO_EVIDENCE")
_cd_out.to_parquet(_CD_OOF_PATH, index=False)

_cd_meta = {{
    "use_ovr_family": bool(_cd_use_ovr),
    "acc_multi": float(_cd_acc_multi),
    "acc_ovr": float(_cd_acc_ovr),
    "best_rate": float(_cd_best_rate),
    "n_active_dev": int(_cd_active.sum()),
    "notebook_oof_evidence_map5_dev": float(_cd_notebook_ev_map5),
    "n_dev_all": int(len(_cd_train)),
    "n_confirmed": int(_cd_is_labeled.sum()),
    "behavior_rule": (
        "predicted_behavior = BEHAVIORS[argmax family OOF] for pairs in the top "
        "select_positive_rate(risk) fraction, else 'none' (notebook's eval rule "
        "applied to the dev OOF)"
    ),
    "evidence_rule": (
        "5-fold XGBRanker OOF over dev positives' hands blended (w=0.75) with the "
        "per-family heuristic signal, top-5 hands per pair (notebook verbatim)"
    ),
}}
_CD_META_PATH.write_text(_cd_json.dumps(_cd_meta, indent=2), encoding="utf-8")
print("component_decomposition: wrote rich dev-holdout frame", _CD_OOF_PATH, flush=True)
print("component_decomposition: DONE", flush=True)
# ==== end component_decomposition child footer ====
'''


def _build_child_script(paths: CompetitorPaths) -> str:
    """Assemble the child script: notebook body (functions only) + our rich-head footer."""
    future_imports, body = _export_notebook_to_script(Path(paths.notebook))

    neutralised: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUN_FULL_PIPELINE") and "=" in stripped and "==" not in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            neutralised.append(f"{indent}RUN_FULL_PIPELINE = False  # neutralized")
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
        oof_name=_RICH_OOF_FILENAME,
        meta_name=_RICH_META_FILENAME,
    )
    future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
    header = (
        "import os as _cd_os\n"
        '_cd_os.environ.setdefault("MPLBACKEND", "Agg")\n'
        "try:\n"
        "    import matplotlib as _cd_mpl\n"
        '    _cd_mpl.use("Agg", force=True)\n'
        "    import matplotlib.pyplot as _cd_plt\n"
        "    _cd_plt.show = lambda *a, **k: None\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    return future_block + header + body + "\n\n" + footer


def _run_child(paths: CompetitorPaths, timeout_s: int) -> Dict[str, object]:
    """Run the rich-head child; return the parsed meta dict. Raises on any failure."""
    output_dir = Path(paths.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "_component_decomp_child.py"
    log_path = output_dir / "_component_decomp_child.log"

    script_path.write_text(_build_child_script(paths), encoding="utf-8")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output_dir / ".mpl_cache")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    (output_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"# component_decomposition rich dev-head re-run\n# notebook={paths.notebook}\n")
        log_handle.flush()
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

    if completed.returncode != 0:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        except Exception:
            pass
        raise RuntimeError(
            f"component_decomposition child exited {completed.returncode}; log tail:\n{tail}"
        )

    meta_path = output_dir / _RICH_META_FILENAME
    if not meta_path.is_file():
        raise RuntimeError(f"child exited 0 but wrote no {_RICH_META_FILENAME}; see {log_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Result container + the driver.                                               #
# --------------------------------------------------------------------------- #


@dataclass
class ComponentResult:
    """One model's 3-component dev-holdout breakdown (+ coverage notes)."""

    model: str
    pair_ap: Optional[float]
    evidence_map: Optional[float]
    behavior_map: Optional[float]
    combined: Optional[float]
    coverage_notes: str
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "pair_ap": self.pair_ap,
            "evidence_map": self.evidence_map,
            "behavior_map": self.behavior_map,
            "combined": self.combined,
            "coverage_notes": self.coverage_notes,
            **({"extra": self.extra} if self.extra else {}),
        }


def _make_scorer(config: PipelineConfig, labels: pd.DataFrame, evidence: pd.DataFrame) -> CanonicalScorer:
    recipe = ScoringRecipe(recipe_id="component_decomposition_v1")
    return CanonicalScorer(recipe, config, labels=labels, evidence=evidence)


def _honghanh_result(
    scorer: CanonicalScorer,
    paths: CompetitorPaths,
    labels: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    force: bool,
    timeout_s: int,
) -> ComponentResult:
    """FULL 3-component breakdown for honghanh (real behavior + evidence heads)."""
    output_dir = Path(paths.output_dir).resolve()
    oof_path = output_dir / _RICH_OOF_FILENAME
    meta_path = output_dir / _RICH_META_FILENAME

    if oof_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reused = True
    else:
        meta = _run_child(paths, timeout_s)
        reused = False

    rich = pd.read_parquet(oof_path)

    # Keep confirmed dev pairs; carry the REAL behavior + evidence heads into a
    # production-metric-schema frame. The scorer builds the PU-correct solution and
    # scores all three components in one pass.
    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    conf = rich[rich["is_labeled"].astype(bool)].copy()
    conf["pair_id"] = conf["pair_id"].astype(str)
    conf = conf[conf["pair_id"].isin(confirmed_ids)]

    preds = pd.DataFrame({"pair_id": conf["pair_id"].to_numpy()})
    preds["risk_score"] = conf["oof_risk"].astype(float).clip(0.0, 1.0).to_numpy()
    preds["predicted_behavior"] = conf["predicted_behavior"].astype(str).to_numpy()
    for i in range(1, 6):
        col = f"evidence_hand_{i}"
        preds[col] = conf[col].astype(str).to_numpy() if col in conf.columns else "NO_EVIDENCE"
    preds = preds[list(SUBMISSION_COLUMNS)]

    score = scorer.score_dev_predictions(preds)
    c = score.components
    notes = (
        "FULL 3-component breakdown. Behavior + evidence heads are the notebook's OWN "
        "heads re-run on the dev holdout (behavior: "
        f"{meta.get('behavior_rule', '')}; evidence: {meta.get('evidence_rule', '')}). "
        f"use_ovr_family={meta.get('use_ovr_family')}, best_rate={meta.get('best_rate')}, "
        f"notebook_oof_evidence_map5_dev={meta.get('notebook_oof_evidence_map5_dev')}, "
        f"n_confirmed_scored={len(preds)}. "
        + ("reused rich OOF cache" if reused else "recomputed rich OOF cache")
    )
    return ComponentResult(
        model="honghanh_064469",
        pair_ap=c.pair_ap,
        evidence_map=c.evidence_map,
        behavior_map=c.behavior_map,
        combined=c.combined,
        coverage_notes=notes,
        extra={
            "use_ovr_family": meta.get("use_ovr_family"),
            "best_rate": meta.get("best_rate"),
            "notebook_oof_evidence_map5_dev": meta.get("notebook_oof_evidence_map5_dev"),
            "n_confirmed_scored": len(preds),
        },
    )


def _floor_result(
    scorer: CanonicalScorer, poker_root: Path, labels: pd.DataFrame, evidence: pd.DataFrame
) -> ComponentResult:
    """PairAP for the floor (risk-only runner); behavior/evidence not measurable here."""
    try:
        result = rerun_floor_stack(default_cache_locations(poker_root), labels, evidence=evidence)
        score = scorer.score_dev_predictions(result.dev_predictions)
        return ComponentResult(
            model="floor_044519",
            pair_ap=score.components.pair_ap,
            evidence_map=None,
            behavior_map=None,
            combined=None,
            coverage_notes=(
                f"PairAP MEASURED (reused floor runner, {result.n_holdout_confirmed} confirmed "
                f"holdout pairs; loopv weighted-all AP={result.loopv_weighted_ap:.4f}). "
                "behavior_map / evidence_map = " + _NOT_MEASURABLE + ": the floor runner "
                "exposes ONLY a risk head — no dev behavior head and no dev evidence ranker "
                "exist without a large rebuild (out of §73 scope). Not fabricated (Rule 9)."
            ),
            extra={"loopv_weighted_ap": result.loopv_weighted_ap},
        )
    except Exception as exc:  # noqa: BLE001 - surface honestly, never fabricate
        return ComponentResult(
            model="floor_044519",
            pair_ap=None, evidence_map=None, behavior_map=None, combined=None,
            coverage_notes=f"BLOCKED: floor PairAP re-run failed: {exc!r}",
        )


def _nomannic_result(
    scorer: CanonicalScorer, poker_root: Path, labels: pd.DataFrame, evidence: pd.DataFrame
) -> ComponentResult:
    """PairAP for nomannic (risk-only runner); behavior/evidence not measurable here."""
    try:
        result = rerun_nomannic_recipe(
            default_nomannic_cache_locations(poker_root), labels, evidence=evidence
        )
        score = scorer.score_dev_predictions(result.dev_predictions)
        return ComponentResult(
            model="nomannic_056202",
            pair_ap=score.components.pair_ap,
            evidence_map=None,
            behavior_map=None,
            combined=None,
            coverage_notes=(
                f"PairAP MEASURED (reused nomannic runner, {result.n_confirmed} confirmed "
                f"holdout pairs; labeled OOF AP={result.labeled_ap:.4f}). Note: nomannic's "
                "pair basis FAILED the adversarial drift gate (dossier §52), so its dev PairAP "
                "is NOT a trustworthy LB predictor (Rule 17). behavior_map / evidence_map = "
                + _NOT_MEASURABLE + ": risk-only runner, no dev behavior/evidence head without "
                "a large rebuild (out of §73 scope). Not fabricated (Rule 9)."
            ),
            extra={"labeled_ap": result.labeled_ap},
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentResult(
            model="nomannic_056202",
            pair_ap=None, evidence_map=None, behavior_map=None, combined=None,
            coverage_notes=f"BLOCKED: nomannic PairAP re-run failed: {exc!r}",
        )


def _implied_eval_pair_ap(lb: float, dev_evidence: Optional[float], dev_behavior: Optional[float]) -> Dict[str, object]:
    """honghanh implied-eval-PairAP brackets (dossier §73 question 2)."""
    bracket_a = lb / 0.70  # eval evidence = behavior = 0
    out: Dict[str, object] = {
        "lb": lb,
        "bracket_a_evidence_behavior_zero": bracket_a,
        "bracket_a_note": "evalPairAP = LB / 0.70 (assumes eval evidence=behavior=0; upper bound on ranking share).",
    }
    if dev_evidence is not None and dev_behavior is not None:
        bracket_b = (lb - 0.20 * dev_evidence - 0.10 * dev_behavior) / 0.70
        out["bracket_b_eval_equals_dev_components"] = bracket_b
        out["bracket_b_note"] = (
            "evalPairAP = (LB - 0.20*devEvidence - 0.10*devBehavior) / 0.70; ASSUMES the eval "
            "components equal the measured DEV components (NOT proven to transfer — Rules 4, 9)."
        )
        out["dev_evidence_used"] = dev_evidence
        out["dev_behavior_used"] = dev_behavior
    else:
        out["bracket_b_eval_equals_dev_components"] = None
        out["bracket_b_note"] = "bracket (b) needs measured dev evidence + behavior (honghanh only)."
    return out


def run_component_decomposition(
    poker_root: Path,
    *,
    force: bool = False,
    timeout_s: int = _CHILD_TIMEOUT_S,
) -> Dict[str, object]:
    """Run the full §73 per-model 3-component decomposition. BUILD + MEASURE ONLY."""
    poker_root = Path(poker_root)
    config = PipelineConfig()
    data_dir = Path(config.input_dir)
    labels = pd.read_csv(data_dir / "development_labels.csv")
    evidence = pd.read_csv(data_dir / "development_evidence.csv")

    scorer = _make_scorer(config, labels, evidence)
    comp_paths = default_competitor_paths(poker_root)

    results: List[ComponentResult] = []
    # honghanh first (the highest-value full breakdown).
    hh = _honghanh_result(scorer, comp_paths, labels, evidence, force=force, timeout_s=timeout_s)
    results.append(hh)
    results.append(_floor_result(scorer, poker_root, labels, evidence))
    results.append(_nomannic_result(scorer, poker_root, labels, evidence))

    brackets = _implied_eval_pair_ap(HONGHANH_REAL_LB, hh.evidence_map, hh.behavior_map)

    summary = {
        "note": (
            "BUILD + MEASURE ONLY (dossier §73). Dev-holdout 3-component decomposition. "
            "PairAP/EvidenceMAP/BehaviorMAP are MEASURED LOCAL dev numbers, NOT LB claims. "
            "The dev components are NOT proven to transfer to eval (Rules 4, 9)."
        ),
        "metric_weights": {"pair_ap": 0.70, "evidence_map": 0.20, "behavior_map": 0.10},
        "models": {r.model: r.as_dict() for r in results},
        "honghanh_implied_eval_pair_ap_brackets": brackets,
    }

    out_dir = poker_root / "outputs" / "poker_collusion" / "component_decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_table(results, brackets, out_dir / "_summary.json")
    return summary


def _fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def _print_table(results: List[ComponentResult], brackets: Dict[str, object], summary_path: Path) -> None:
    print()
    print("=" * 88)
    print("§73 DEV-HOLDOUT 3-COMPONENT DECOMPOSITION (BUILD + MEASURE ONLY; NOT LB claims)")
    print("LB metric = 0.70*PairAP + 0.20*EvidenceMAP@5 + 0.10*BehaviorMAP")
    print("=" * 88)
    header = f"{'model':<20}{'PairAP':>12}{'EvidMAP@5':>12}{'BehavMAP':>12}{'combined':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.model:<20}{_fmt(r.pair_ap):>12}{_fmt(r.evidence_map):>12}{_fmt(r.behavior_map):>12}{_fmt(r.combined):>12}")
    print("-" * len(header))
    print()
    print("Coverage notes:")
    for r in results:
        print(f"  [{r.model}] {r.coverage_notes}")
    print()
    print("honghanh implied-eval-PairAP brackets (LB = %.5f):" % brackets["lb"])
    print(f"  (a) eval evidence=behavior=0  -> evalPairAP = {_fmt(brackets['bracket_a_evidence_behavior_zero'])}")
    b = brackets.get("bracket_b_eval_equals_dev_components")
    print(f"  (b) eval components = dev     -> evalPairAP = {_fmt(b)}")
    print(f"      (dev evidence used={_fmt(brackets.get('dev_evidence_used'))}, dev behavior used={_fmt(brackets.get('dev_behavior_used'))})")
    print()
    print(f"Wrote summary: {summary_path}")
    print("=" * 88)


def _main() -> None:
    parser = argparse.ArgumentParser(description="§73 per-model 3-component dev-holdout decomposition.")
    parser.add_argument("--poker-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--force", action="store_true", help="Recompute honghanh's rich OOF cache.")
    parser.add_argument("--timeout", type=int, default=_CHILD_TIMEOUT_S)
    args = parser.parse_args()
    run_component_decomposition(Path(args.poker_root), force=args.force, timeout_s=args.timeout)


if __name__ == "__main__":
    _main()
