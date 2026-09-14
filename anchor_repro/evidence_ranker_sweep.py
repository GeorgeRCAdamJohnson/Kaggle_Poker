"""Evidence-ranker improvement sweep on the dev holdout (dossier §75).

BUILD + DEV-TUNE ONLY. This module NEVER submits to Kaggle and NEVER writes to
``submission.csv`` / ``submission_best_*.csv``. It writes only its own JSON summary
and (optionally) a cached eval-evidence parquet under
``outputs/poker_collusion/evidence_ranker/``.

Why this exists (dossier §75; contract Rules 6, 7, 13, 15, 18).
---------------------------------------------------------------
The LB metric is ``0.70*PairAP + 0.20*EvidenceMAP@5 + 0.10*BehaviorMAP`` (dossier §1).
§74 measured honghanh's dev components: PairAP 0.9747, BehaviorMAP 0.8935, and
EvidenceMAP@5 only ``0.34549`` — the weakest component and worth 20% of the LB. This
module sweeps the evidence RANKER (dossier §75) to try to push the CANONICAL dev
EvidenceMAP@5 past the pre-registered bar ``0.3455 + 0.02 = 0.3655``.

Two evidence numbers are kept visible side by side throughout (contract Rule 9):
  * ``notebook_evidence_map5`` — the notebook's OWN internal ``evidence_map5`` over the
    OOF ranker scores on the dev positives' candidate hands (≈0.4607 for the floor).
  * ``canonical_evidence_map`` — the number that MATTERS: the top-5 evidence hands per
    positive pair scored through :class:`CanonicalScorer` (≈0.3455 for the floor).
The two are DIFFERENT definitions (§74) — the canonical one is the deliverable.

How EvidenceMAP is isolated (so only ``evidence_hand_*`` changes across configs).
---------------------------------------------------------------------------------
The CanonicalScorer scores a full prediction frame over the 1,860 confirmed dev pairs
and returns all three components in one pass. EvidenceMAP@5 depends ONLY on the five
``evidence_hand_*`` columns of the confirmed POSITIVE pairs. To isolate it, we hold
``risk_score`` and ``predicted_behavior`` FIXED at honghanh's own dev-holdout heads
(reused verbatim from the §73 ``dev_holdout_full_064469.parquet`` cache written by
:mod:`anchor_repro.component_decomposition`) and swap in ONLY each config's
``evidence_hand_*`` for the positive pairs. PairAP and BehaviorMAP are therefore
identical across every config, so the canonical EvidenceMAP@5 delta is attributable
to the ranker change alone.

The evidence candidate table + all OOF rankings are produced in a CHILD PROCESS that
imports the notebook's OWN evidence functions (``numeric_hand_features``,
``RANK_BASE_CANDIDATES``, ``FAMILY_FEATURES``, ``evidence_matrix``, ``add_family_flags``,
``blend_within_pair``, ``evidence_map5``) via
:func:`anchor_repro.repro_runner._export_notebook_to_script`, with the notebook's
``main()`` neutralised (contract Rule 6) — exactly the pattern of
:mod:`anchor_repro.component_decomposition`.

The sweep (dossier §75; bounded, sequential — best-of-each carries forward, Rule 18).
-------------------------------------------------------------------------------------
  (a) BLEND WEIGHT on the base-feature ``rank:pairwise`` ranker: w in
      {0.0, 0.25, 0.5, 0.75, 1.0} (w = weight on the ranker; 0.75 is the floor).
  (b) FEATURE ENRICHMENT: add within-pair percentile ranks of a set of collusion
      signals, retrain the ranker, re-sweep w — keep best-w.
  (c) PER-FAMILY vs GLOBAL ranker (best config so far).
  (d) OBJECTIVE: rank:pairwise vs rank:ndcg vs rank:map (best config so far).

If the best config clears the bar, the eval evidence for that config is produced over
ALL 112,540 eval pairs (using a ranker trained on all dev positives) and cached to
``eval_evidence_best.parquet`` so a candidate CSV could be assembled LATER — this
module does NOT assemble or submit any CSV (contract Rule 7).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from anchor_repro.competitor_recipe import CompetitorPaths, default_competitor_paths
from anchor_repro.component_decomposition import (
    _RICH_OOF_FILENAME,
    _RICH_META_FILENAME,
    run_component_decomposition,
)
from anchor_repro.repro_runner import _export_notebook_to_script
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.models import ScoringRecipe
from poker_collusion.config import PipelineConfig
from poker_collusion.submission.writer import SUBMISSION_COLUMNS

__all__ = [
    "FLOOR_CANONICAL_EVIDENCE_MAP",
    "SWEEP_BAR_DELTA",
    "run_evidence_ranker_sweep",
]

#: The §74 canonical dev EvidenceMAP@5 floor the sweep must reproduce (honghanh's
#: current XGBRanker w=0.75, base features, rank:pairwise). If the reproduced floor
#: does not match this within tolerance, the sweep is built on a broken baseline and
#: must NOT be trusted (contract Rule 11) — the runner reports the discrepancy and
#: refuses to declare any winner.
FLOOR_CANONICAL_EVIDENCE_MAP: float = 0.34549

#: Tolerance for the floor reproduction (canonical EvidenceMAP@5 is deterministic
#: given the fixed cache; allow a small float-comparison / re-fit jitter margin).
FLOOR_TOLERANCE: float = 0.004

#: The pre-registered bar delta (dossier §75). A config clears iff canonical dev
#: EvidenceMAP@5 >= FLOOR + SWEEP_BAR_DELTA.
SWEEP_BAR_DELTA: float = 0.02

#: Child wall-clock cap (§75: warm cache makes it fast; 1h is ample headroom).
_CHILD_TIMEOUT_S: int = 3600

#: Filenames the child writes under ``output_dir`` (the evidence_ranker out dir).
_SWEEP_EVIDENCE_FILENAME: str = "_sweep_config_evidence.parquet"
_SWEEP_CHILD_META_FILENAME: str = "_sweep_child_meta.json"
_EVAL_EVIDENCE_FILENAME: str = "eval_evidence_best.parquet"


# --------------------------------------------------------------------------- #
# The child footer — reuse the notebook's OWN evidence machinery, run the sweep #
# of OOF rankers, and emit per-config top-5 evidence hands for the dev positives.#
# --------------------------------------------------------------------------- #

_CHILD_FOOTER_TEMPLATE = '''\
# ==== evidence_ranker_sweep child footer (AUTHORED BY THE RUNNER) ====
# Reuse the notebook's OWN evidence functions (imported above) to build the dev
# positive evidence candidate table with honghanh's CV fold ids + is_evidence labels,
# then sweep OOF evidence rankers (blend-weight / enrichment / per-family / objective)
# and emit each config's top-5 evidence hands per positive pair (contract Rule 6, 13).
# We do NOT call the notebook's main(); we replicate ONLY its evidence-ranker path.
import json as _es_json
import gc as _es_gc
import numpy as _es_np
import pandas as _es_pd
import polars as _es_pl
from pathlib import Path as _EsPath
from sklearn.model_selection import StratifiedGroupKFold as _EsSGKF
from xgboost import XGBClassifier as _EsXGB, XGBRanker as _EsXGBRanker

_ES_PREP = _EsPath({prepared_literal})
_ES_OUT = _EsPath({output_literal})
_ES_DATA = _EsPath({data_literal})
_ES_EV_PARQUET = _ES_OUT / {ev_name!r}
_ES_META_PATH = _ES_OUT / {meta_name!r}
_ES_EVAL_PARQUET = _ES_OUT / {eval_name!r}
_ES_BAR = float({bar_literal})
_ES_EMIT_EVAL = bool({emit_eval_literal})

print("evidence_ranker_sweep: prepared_dir =", _ES_PREP, flush=True)

_es_dev_pairs = _es_pl.read_parquet(_ES_PREP / "dev_pairs.parquet")
_es_dev_hand_path = _ES_PREP / "dev_hand_features.parquet"
_es_eval_hand_path = _ES_PREP / "eval_hand_features.parquet"
_es_eval_pairs_path = _ES_PREP / "eval_pairs.parquet"
_es_player_hand_path = _ES_PREP / "player_hands.parquet"

_es_hands = _es_pl.scan_parquet(_ES_DATA / "hands.parquet")
_es_seats = _es_pl.scan_parquet(_ES_DATA / "seats.parquet")
_es_eval_pairs_csv = _es_pl.read_csv(_ES_DATA / "evaluation_pairs.csv")
_es_labels = _es_pl.read_csv(_ES_DATA / "development_labels.csv")
_es_dev_evidence = (
    _es_pl.read_csv(_ES_DATA / "development_evidence.csv")
    .select(["pair_id", "hand_id"]).unique()
)

# --- Rebuild the pair-level train matrix with the notebook's OWN functions so we can
#     recover the SAME 5-fold StratifiedGroupKFold(table_id) fold ids + family OOF the
#     notebook (and the §73 rich cache) use, then derive behavior_id_pred per pair.
print("evidence_ranker_sweep: aggregating pair features (notebook code)...", flush=True)
_es_train = aggregate_pair_features(_es_dev_hand_path, _es_dev_pairs)
_es_train = add_player_baselines(
    _es_train, player_phase_baselines(_es_player_hand_path, "development")
)
_es_train = add_partner_field_contrasts(
    _es_train, _es_seats, _es_hands, _es_player_hand_path, "development"
)

_es_exclude = {{
    "pair_id", "player_1", "player_2", "label", "behavior_family",
    "table_id", "is_labeled", "is_pu",
}}
_es_feature_cols = [
    c for c, dtype in _es_train.schema.items()
    if dtype.is_numeric() and c not in _es_exclude
]
_es_X = matrix_from(_es_train, _es_feature_cols)
_es_y = _es_train["label"].fill_null(0).to_numpy().astype(_es_np.int8)
_es_is_pu = _es_train["is_pu"].to_numpy().astype(bool)
_es_behavior_y = _es_np.array(
    [BEHAVIOR_TO_ID.get(x, 0) for x in _es_train["behavior_family"].to_list()],
    dtype=_es_np.int8,
)
_es_groups = _es_train["table_id"].fill_null("NA").to_numpy()

# Family-OvR OOF (matches the notebook / §73 footer) to derive behavior_id_pred.
_es_family_det_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=3, min_child_weight=4,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=4,
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    early_stopping_rounds=50, n_jobs=-1,
)
_es_multi_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=3, min_child_weight=4,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=4,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    tree_method="hist", early_stopping_rounds=50, n_jobs=-1,
)
_es_cv = _EsSGKF(n_splits=5, shuffle=True, random_state=SEED)
_es_oof_family_det = _es_np.zeros((len(_es_train), 3), dtype=_es_np.float32)
_es_oof_multi = _es_np.zeros((len(_es_train), 3), dtype=_es_np.float32)
_es_fold_id = _es_np.full(len(_es_train), -1, dtype=_es_np.int8)
for _es_fold, (_es_tr, _es_va) in enumerate(_es_cv.split(_es_X, _es_behavior_y, _es_groups)):
    _es_fold_id[_es_va] = _es_fold
    for _es_k in range(3):
        _es_yk = (_es_behavior_y == _es_k + 1).astype(_es_np.int8)
        _es_wk = _es_np.where(
            _es_yk == 1, POS_WEIGHT, _es_np.where(_es_is_pu, PU_WEIGHT, 1.0)
        ).astype(_es_np.float32)
        _es_fp = dict(_es_family_det_params)
        if int(_es_yk[_es_va].sum()) == 0:
            _es_fp.pop("early_stopping_rounds", None)
        _es_det = _EsXGB(**_es_fp, random_state=SEED + _es_fold * 10 + _es_k)
        _es_fkw = dict(sample_weight=_es_wk[_es_tr], verbose=False)
        if int(_es_yk[_es_va].sum()) > 0:
            _es_fkw["eval_set"] = [(_es_X.iloc[_es_va], _es_yk[_es_va])]
        _es_det.fit(_es_X.iloc[_es_tr], _es_yk[_es_tr], **_es_fkw)
        _es_oof_family_det[_es_va, _es_k] = _es_det.predict_proba(_es_X.iloc[_es_va])[:, 1]
    _es_pos_tr = _es_tr[_es_behavior_y[_es_tr] > 0]
    _es_pos_va = _es_va[_es_behavior_y[_es_va] > 0]
    if len(_es_pos_tr):
        _es_bp = dict(_es_multi_params)
        if len(_es_pos_va) == 0:
            _es_bp.pop("early_stopping_rounds", None)
        _es_bm = _EsXGB(**_es_bp, random_state=SEED + _es_fold)
        _es_bkw = dict(verbose=False)
        if len(_es_pos_va):
            _es_bkw["eval_set"] = [(_es_X.iloc[_es_pos_va], _es_behavior_y[_es_pos_va] - 1)]
        _es_bm.fit(_es_X.iloc[_es_pos_tr], _es_behavior_y[_es_pos_tr] - 1, **_es_bkw)
        _es_oof_multi[_es_va] = _es_bm.predict_proba(_es_X.iloc[_es_va])
    print(f"evidence_ranker_sweep: family fold {{_es_fold + 1}}/5 done", flush=True)

# Pick OvR vs multiclass by positive-family accuracy (the §73 rule) so behavior_id_pred
# matches the fixed dev heads exactly.
_es_family_multi = _es_oof_multi.argmax(1)
_es_family_ovr = _es_oof_family_det.argmax(1)
_es_pos = _es_y == 1
_es_acc_multi = float((_es_family_multi[_es_pos] + 1 == _es_behavior_y[_es_pos]).mean()) if _es_pos.any() else 0.0
_es_acc_ovr = float((_es_family_ovr[_es_pos] + 1 == _es_behavior_y[_es_pos]).mean()) if _es_pos.any() else 0.0
_es_family_oof = _es_family_ovr if _es_acc_ovr >= _es_acc_multi else _es_family_multi
print(
    f"evidence_ranker_sweep: family acc multi={{_es_acc_multi:.4f}} ovr={{_es_acc_ovr:.4f}}",
    flush=True,
)

# --- Build the evidence candidate table for the dev POSITIVES (notebook verbatim). ---
_es_hand_feats = numeric_hand_features(_es_dev_hand_path)
_es_rank_base = [c for c in RANK_BASE_CANDIDATES if c in _es_hand_feats][:20]
_es_rank_pct = [f"{{c}}_pair_pct" for c in _es_rank_base]
_es_base_feature_cols = _es_hand_feats + _es_rank_pct + FAMILY_FEATURES

_es_positive_info = (
    _es_labels.filter(_es_pl.col("behavior_family").is_in(BEHAVIORS))
    .select(["pair_id", "behavior_family"])
    .with_columns(
        _es_pl.when(_es_pl.col("behavior_family") == "directed_transfer").then(1)
        .when(_es_pl.col("behavior_family") == "soft_play").then(2)
        .otherwise(3).cast(_es_pl.Int8).alias("behavior_id_true")
    )
)
_es_ev_keys = _es_dev_evidence.select(["pair_id", "hand_id"]).unique().with_columns(
    _es_pl.lit(1, dtype=_es_pl.Int8).alias("is_evidence")
)
_es_pair_fold = _es_pl.DataFrame({{
    "pair_id": _es_train["pair_id"],
    "fold": _es_fold_id,
    "behavior_id_pred": (_es_family_oof + 1).astype(_es_np.int8),
}})
# Within-pair percentile ranks of the enrichment signals (config b). Any absent signal
# is simply skipped (contract Rule 9 — no fabricated column).
_ES_ENRICH_SIGNALS = [
    "transfer_any_bb", "pair_pot_share", "max_win_bb", "strong_hole_passive",
    "dump_vs_hole", "both_showdown", "net_gap_bb",
]
_es_enrich_present = [c for c in _ES_ENRICH_SIGNALS if c in _es_hand_feats]
_es_enrich_pct = [f"{{c}}_enrich_pct" for c in _es_enrich_present]
print("evidence_ranker_sweep: enrichment signals present =", _es_enrich_present, flush=True)

_es_ev_base = (
    _es_pl.scan_parquet(_es_dev_hand_path)
    .select(["pair_id", "hand_id"] + _es_hand_feats)
    .join(_es_positive_info.lazy(), on="pair_id")
    .join(_es_ev_keys.lazy(), on=["pair_id", "hand_id"], how="left")
    .with_columns(_es_pl.col("is_evidence").fill_null(0))
    .collect()
    .join(_es_pair_fold, on="pair_id")
    .with_columns([
        (_es_pl.col(c).rank(method="average").over("pair_id") / _es_pl.len().over("pair_id"))
        .cast(_es_pl.Float32).alias(f"{{c}}_pair_pct")
        for c in _es_rank_base
    ])
    .with_columns([
        (_es_pl.col(c).rank(method="average").over("pair_id") / _es_pl.len().over("pair_id"))
        .cast(_es_pl.Float32).alias(f"{{c}}_enrich_pct")
        for c in _es_enrich_present
    ])
    .with_row_index("_row")
)
_es_n_pairs = _es_ev_base["pair_id"].n_unique()
print(f"evidence_ranker_sweep: {{_es_ev_base.height}} candidate hands over {{_es_n_pairs}} dev positives", flush=True)

_es_base_rank_params = dict(
    n_estimators=500, learning_rate=0.04, max_depth=4, min_child_weight=2,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=4,
    tree_method="hist", n_jobs=-1,
)

def _es_oof_ranker(feature_cols, objective, per_family):
    """5-fold OOF ranker scores over _es_ev_base rows (aligned to _row).

    per_family: if True, train a separate ranker per behavior_id_true within each
    training fold and predict each validation hand with its predicted family's model
    (falling back to a global model for unseen families). Otherwise one global ranker.
    Returns an np.float32 array aligned to _es_ev_base["_row"].
    """
    _oof = _es_np.zeros(_es_ev_base.height, dtype=_es_np.float32)
    _params = dict(_es_base_rank_params)
    _params["objective"] = objective
    _params["eval_metric"] = "map@5"
    for _f in range(5):
        _tr_part = _es_ev_base.filter(_es_pl.col("fold") != _f)
        _va_part = _es_ev_base.filter(_es_pl.col("fold") == _f)
        if _va_part.height == 0 or _tr_part.height == 0:
            continue
        _, _xtr, _ytr, _gtr = evidence_matrix(_tr_part, feature_cols, "behavior_id_true")
        _vo, _xva, _yva, _gva = evidence_matrix(_va_part, feature_cols, "behavior_id_pred")
        if per_family:
            # Global fallback model (trained on the whole train fold).
            _gm = _EsXGBRanker(**_params, random_state=SEED + _f)
            _gm.fit(_xtr, _ytr, group=_gtr, verbose=False)
            _va_sorted = add_family_flags(_va_part, "behavior_id_pred").sort(["pair_id", "hand_id"])
            _bid_va = _va_sorted["behavior_id_pred"].to_numpy()
            _rows_va = _vo["_row"].to_numpy()
            _pred = _gm.predict(_xva).astype(_es_np.float32)
            for _fam in (1, 2, 3):
                _tr_fam = _tr_part.filter(_es_pl.col("behavior_id_true") == _fam)
                if _tr_fam.height == 0 or _tr_fam["pair_id"].n_unique() < 2:
                    continue
                _, _xf, _yf, _gf = evidence_matrix(_tr_fam, feature_cols, "behavior_id_true")
                if int(_yf.sum()) == 0:
                    continue
                _fm = _EsXGBRanker(**_params, random_state=SEED + _f * 10 + _fam)
                _fm.fit(_xf, _yf, group=_gf, verbose=False)
                _mask = _bid_va == _fam
                if _mask.any():
                    _pred[_mask] = _fm.predict(_xva[_mask]).astype(_es_np.float32)
            _oof[_rows_va] = _pred
        else:
            _rk = _EsXGBRanker(**_params, early_stopping_rounds=60, random_state=SEED + _f)
            _rk.fit(_xtr, _ytr, group=_gtr, eval_set=[(_xva, _yva)], eval_group=[_gva], verbose=False)
            _oof[_vo["_row"].to_numpy()] = _rk.predict(_xva).astype(_es_np.float32)
    return _oof

# Per-family heuristic signal (notebook's blend heuristic), aligned to _row order.
_es_flagged = add_family_flags(_es_ev_base, "behavior_id_pred")
_es_bid = _es_flagged["behavior_id_pred"].to_numpy()
_es_heur = _es_np.where(
    _es_bid == 1, _es_flagged["directed_signal"].to_numpy(),
    _es_np.where(_es_bid == 2, _es_flagged["soft_signal"].to_numpy(),
    _es_np.where(_es_bid == 3, _es_flagged["isolation_signal"].to_numpy(),
                 _es_flagged["other_signal"].to_numpy())),
).astype(_es_np.float32)

def _es_topk_frame(config_name, blended):
    """Top-5 hands per positive pair for a blended score → tidy long frame."""
    _f = (
        _es_ev_base.select(["pair_id", "hand_id"])
        .with_columns(_es_pl.Series("_score", blended))
        .group_by("pair_id")
        .agg(_es_pl.col("hand_id").sort_by("_score", descending=True).head(5).alias("hands"))
        .with_columns([
            _es_pl.col("hands").list.get(i, null_on_oob=True).alias(f"evidence_hand_{{i + 1}}")
            for i in range(5)
        ])
        .drop("hands")
        .with_columns(_es_pl.lit(config_name).alias("config"))
    )
    return _f

def _es_run_config(config_name, feature_cols, objective, per_family, w):
    """Produce OOF ranker → blended → top-5 frame + notebook evidence_map5 for a config."""
    _oof = _es_oof_ranker(feature_cols, objective, per_family)
    _nb_map = evidence_map5(
        _es_ev_base["pair_id"].to_numpy(), _es_ev_base["is_evidence"].to_numpy(), _oof
    )
    if w >= 1.0:
        _blended = _oof
    else:
        _blended = blend_within_pair(_es_ev_base, _oof, _es_heur, w_ranker=w)
    _frame = _es_topk_frame(config_name, _blended)
    print(f"evidence_ranker_sweep: config {{config_name}} notebook_map5={{_nb_map:.4f}}", flush=True)
    return _frame, float(_nb_map), _oof

# --------------------------------------------------------------------------- #
# The bounded sequential sweep (dossier §75).                                  #
# --------------------------------------------------------------------------- #
_es_frames = []
_es_meta_configs = {{}}

# (a) BLEND WEIGHT on base features, rank:pairwise. Reuse the SAME base OOF for all w.
_es_base_oof = _es_oof_ranker(_es_base_feature_cols, "rank:pairwise", False)
_es_base_nb_map = float(evidence_map5(
    _es_ev_base["pair_id"].to_numpy(), _es_ev_base["is_evidence"].to_numpy(), _es_base_oof
))
print(f"evidence_ranker_sweep: base rank:pairwise notebook_map5={{_es_base_nb_map:.4f}}", flush=True)
for _es_w in (0.0, 0.25, 0.5, 0.75, 1.0):
    _es_name = f"a_blendw_{{_es_w:.2f}}_base_pairwise"
    if _es_w >= 1.0:
        _es_bl = _es_base_oof
    else:
        _es_bl = blend_within_pair(_es_ev_base, _es_base_oof, _es_heur, w_ranker=_es_w)
    _es_frames.append(_es_topk_frame(_es_name, _es_bl))
    _es_meta_configs[_es_name] = {{
        "notebook_evidence_map5": _es_base_nb_map,
        "family": "blend_weight", "w": _es_w, "objective": "rank:pairwise",
        "features": "base", "per_family": False,
    }}

# (b) FEATURE ENRICHMENT: base features + enrichment percentile ranks; retrain; sweep w.
_es_enriched_cols = _es_base_feature_cols + _es_enrich_pct
if _es_enrich_pct:
    _es_enr_oof = _es_oof_ranker(_es_enriched_cols, "rank:pairwise", False)
    _es_enr_nb_map = float(evidence_map5(
        _es_ev_base["pair_id"].to_numpy(), _es_ev_base["is_evidence"].to_numpy(), _es_enr_oof
    ))
    print(f"evidence_ranker_sweep: enriched rank:pairwise notebook_map5={{_es_enr_nb_map:.4f}}", flush=True)
    for _es_w in (0.0, 0.25, 0.5, 0.75, 1.0):
        _es_name = f"b_enrich_blendw_{{_es_w:.2f}}"
        if _es_w >= 1.0:
            _es_bl = _es_enr_oof
        else:
            _es_bl = blend_within_pair(_es_ev_base, _es_enr_oof, _es_heur, w_ranker=_es_w)
        _es_frames.append(_es_topk_frame(_es_name, _es_bl))
        _es_meta_configs[_es_name] = {{
            "notebook_evidence_map5": _es_enr_nb_map,
            "family": "enrichment", "w": _es_w, "objective": "rank:pairwise",
            "features": "enriched", "per_family": False,
            "enrich_signals_used": _es_enrich_present,
        }}

# The parent decides the winning w for (a)/(b) after canonical scoring, but the child
# must pick a config to carry into (c)/(d). We carry the ENRICHED features (if any) at
# the FLOOR w=0.75 for (c)/(d) since w=0.75 is honghanh's tuned blend (a sensible,
# bounded default; the parent still scores every (c)/(d) variant canonically).
_es_carry_cols = _es_enriched_cols if _es_enrich_pct else _es_base_feature_cols
_es_carry_feat_tag = "enriched" if _es_enrich_pct else "base"

# (c) PER-FAMILY vs GLOBAL (carry features, rank:pairwise, w=0.75).
for _es_pf, _es_tag in ((False, "global"), (True, "perfamily")):
    _es_name = f"c_{{_es_tag}}_{{_es_carry_feat_tag}}_pairwise_w0.75"
    _es_fr, _es_nb, _ = _es_run_config(_es_name, _es_carry_cols, "rank:pairwise", _es_pf, 0.75)
    _es_frames.append(_es_fr)
    _es_meta_configs[_es_name] = {{
        "notebook_evidence_map5": _es_nb, "family": "per_family_vs_global",
        "w": 0.75, "objective": "rank:pairwise", "features": _es_carry_feat_tag,
        "per_family": _es_pf,
    }}

# (d) OBJECTIVE sweep (carry features, global, w=0.75).
for _es_obj in ("rank:pairwise", "rank:ndcg", "rank:map"):
    _es_name = f"d_obj_{{_es_obj.replace(':', '_')}}_{{_es_carry_feat_tag}}_w0.75"
    _es_fr, _es_nb, _ = _es_run_config(_es_name, _es_carry_cols, _es_obj, False, 0.75)
    _es_frames.append(_es_fr)
    _es_meta_configs[_es_name] = {{
        "notebook_evidence_map5": _es_nb, "family": "objective",
        "w": 0.75, "objective": _es_obj, "features": _es_carry_feat_tag,
        "per_family": False,
    }}

# Persist all per-config top-5 evidence hands (long: config, pair_id, evidence_hand_*).
_es_all = _es_pl.concat(_es_frames, how="vertical_relaxed")
_es_all.write_parquet(_ES_EV_PARQUET)
print("evidence_ranker_sweep: wrote per-config evidence", _ES_EV_PARQUET, flush=True)

_es_child_meta = {{
    "configs": _es_meta_configs,
    "n_dev_positive_pairs": int(_es_n_pairs),
    "n_candidate_hands": int(_es_ev_base.height),
    "enrich_signals_present": _es_enrich_present,
    "carry_features": _es_carry_feat_tag,
    "base_feature_count": int(len(_es_base_feature_cols)),
    "enriched_feature_count": int(len(_es_enriched_cols)),
}}
_ES_META_PATH.write_text(_es_json.dumps(_es_child_meta, indent=2), encoding="utf-8")

# --------------------------------------------------------------------------- #
# (5) EVAL evidence for the requested best config (only if the parent asked).  #
#     The parent writes {{output_dir}}/_eval_request.json BEFORE re-running the  #
#     child with EMIT_EVAL=1 once it knows the canonical winner. We honor that. #
# --------------------------------------------------------------------------- #
if _ES_EMIT_EVAL:
    _es_req_path = _ES_OUT / "_eval_request.json"
    _es_req = _es_json.loads(_es_req_path.read_text(encoding="utf-8"))
    _es_feat_tag = _es_req["features"]
    _es_objective = _es_req["objective"]
    _es_per_family = bool(_es_req["per_family"])
    _es_w = float(_es_req["w"])
    _es_feature_cols = _es_enriched_cols if (_es_feat_tag == "enriched" and _es_enrich_pct) else _es_base_feature_cols
    print(f"evidence_ranker_sweep: EVAL evidence for {{_es_req}}", flush=True)

    # Train a SHARED ranker on ALL dev positives for the winning config (notebook's
    # shared_ranker pattern), then score every eval pair's candidate hands in chunks.
    _es_params = dict(_es_base_rank_params)
    _es_params["objective"] = _es_objective
    _es_params["eval_metric"] = "map@5"
    _, _es_x_ev, _es_y_ev, _es_g_ev = evidence_matrix(_es_ev_base, _es_feature_cols, "behavior_id_true")
    _es_shared_models = {{}}
    _es_global_model = _EsXGBRanker(**_es_params, random_state=SEED)
    _es_global_model.fit(_es_x_ev, _es_y_ev, group=_es_g_ev, verbose=False)
    if _es_per_family:
        for _es_fam in (1, 2, 3):
            _es_tr_fam = _es_ev_base.filter(_es_pl.col("behavior_id_true") == _es_fam)
            if _es_tr_fam.height == 0 or _es_tr_fam["pair_id"].n_unique() < 2:
                continue
            _, _es_xf, _es_yf, _es_gf = evidence_matrix(_es_tr_fam, _es_feature_cols, "behavior_id_true")
            if int(_es_yf.sum()) == 0:
                continue
            _es_fm = _EsXGBRanker(**_es_params, random_state=SEED + _es_fam)
            _es_fm.fit(_es_xf, _es_yf, group=_es_gf, verbose=False)
            _es_shared_models[_es_fam] = _es_fm

    # Eval behavior_id_pred: use honghanh's eval submission behavior if available; else
    # default all-eval to family 1 (directed) so evidence is produced for ALL pairs
    # (the parent's request records which). We read the emitted eval submission.
    _es_eval_pairs = _es_pl.read_parquet(_es_eval_pairs_path)
    _es_sub_path = _ES_OUT.parent / "repro_064469" / "submission.csv"
    _es_beh_map = None
    if _es_sub_path.is_file():
        _es_sub = _es_pl.read_csv(_es_sub_path).select(["pair_id", "predicted_behavior"])
        _es_beh_map = _es_sub.with_columns(
            _es_pl.when(_es_pl.col("predicted_behavior") == "directed_transfer").then(1)
            .when(_es_pl.col("predicted_behavior") == "soft_play").then(2)
            .when(_es_pl.col("predicted_behavior") == "coordinated_isolation").then(3)
            .otherwise(1).cast(_es_pl.Int8).alias("behavior_id_pred")
        ).select(["pair_id", "behavior_id_pred"])
    if _es_beh_map is None:
        _es_beh_map = _es_eval_pairs.select("pair_id").with_columns(
            _es_pl.lit(1, dtype=_es_pl.Int8).alias("behavior_id_pred")
        )

    _es_pair_ids = _es_eval_pairs["pair_id"].to_list()
    _es_chunk = 4000
    _es_top_chunks = []
    for _es_start in range(0, len(_es_pair_ids), _es_chunk):
        _es_ch = _es_pair_ids[_es_start:_es_start + _es_chunk]
        _es_part = (
            _es_pl.scan_parquet(_es_eval_hand_path)
            .filter(_es_pl.col("pair_id").is_in(_es_ch))
            .select(["pair_id", "hand_id"] + _es_hand_feats)
            .join(_es_beh_map.lazy(), on="pair_id", how="left")
            .with_columns(_es_pl.col("behavior_id_pred").fill_null(1))
            .collect()
            .with_columns([
                (_es_pl.col(c).rank(method="average").over("pair_id") / _es_pl.len().over("pair_id"))
                .cast(_es_pl.Float32).alias(f"{{c}}_pair_pct")
                for c in _es_rank_base
            ])
            .with_columns([
                (_es_pl.col(c).rank(method="average").over("pair_id") / _es_pl.len().over("pair_id"))
                .cast(_es_pl.Float32).alias(f"{{c}}_enrich_pct")
                for c in _es_enrich_present
            ])
        )
        _es_part = add_family_flags(_es_part, "behavior_id_pred")
        _es_mtx = _es_np.nan_to_num(
            _es_part.select(_es_feature_cols).to_numpy().astype(_es_np.float32), nan=0, posinf=0, neginf=0
        )
        _es_scores = _es_global_model.predict(_es_mtx).astype(_es_np.float32)
        if _es_per_family and _es_shared_models:
            _es_bidv = _es_part["behavior_id_pred"].to_numpy()
            for _es_fam, _es_fm in _es_shared_models.items():
                _es_m = _es_bidv == _es_fam
                if _es_m.any():
                    _es_scores[_es_m] = _es_fm.predict(_es_mtx[_es_m]).astype(_es_np.float32)
        _es_bidv = _es_part["behavior_id_pred"].to_numpy()
        _es_heur_e = _es_np.where(
            _es_bidv == 1, _es_part["directed_signal"].to_numpy(),
            _es_np.where(_es_bidv == 2, _es_part["soft_signal"].to_numpy(),
            _es_np.where(_es_bidv == 3, _es_part["isolation_signal"].to_numpy(),
                         _es_part["other_signal"].to_numpy())),
        ).astype(_es_np.float32)
        if _es_w >= 1.0:
            _es_bl = _es_scores
        else:
            _es_bl = blend_within_pair(_es_part, _es_scores, _es_heur_e, w_ranker=_es_w)
        _es_top_chunks.append(
            _es_part.select(["pair_id", "hand_id"])
            .with_columns(_es_pl.Series("_score", _es_bl))
            .group_by("pair_id")
            .agg(_es_pl.col("hand_id").sort_by("_score", descending=True).head(5).alias("hands"))
        )
        del _es_part, _es_mtx
        _es_gc.collect()
        print(f"evidence_ranker_sweep: eval chunk {{_es_start // _es_chunk + 1}}", flush=True)

    _es_eval_wide = (
        _es_pl.concat(_es_top_chunks)
        .with_columns([
            _es_pl.col("hands").list.get(i, null_on_oob=True).fill_null("NO_EVIDENCE").alias(f"evidence_hand_{{i + 1}}")
            for i in range(5)
        ])
        .drop("hands")
    )
    _es_eval_wide.write_parquet(_ES_EVAL_PARQUET)
    print("evidence_ranker_sweep: wrote eval evidence", _ES_EVAL_PARQUET, "rows", _es_eval_wide.height, flush=True)

print("evidence_ranker_sweep: DONE", flush=True)
# ==== end evidence_ranker_sweep child footer ====
'''


def _build_child_script(paths: CompetitorPaths, out_dir: Path, *, emit_eval: bool) -> str:
    """Assemble the child: notebook body (functions only) + our sweep footer."""
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
        output_literal=repr(str(out_dir.resolve())),
        data_literal=repr(str(Path(paths.data_dir).resolve())),
        ev_name=_SWEEP_EVIDENCE_FILENAME,
        meta_name=_SWEEP_CHILD_META_FILENAME,
        eval_name=_EVAL_EVIDENCE_FILENAME,
        bar_literal=repr(FLOOR_CANONICAL_EVIDENCE_MAP + SWEEP_BAR_DELTA),
        emit_eval_literal=repr(bool(emit_eval)),
    )
    future_block = ("\n".join(future_imports) + "\n\n") if future_imports else ""
    header = (
        "import os as _es_os\n"
        '_es_os.environ.setdefault("MPLBACKEND", "Agg")\n'
        "try:\n"
        "    import matplotlib as _es_mpl\n"
        '    _es_mpl.use("Agg", force=True)\n'
        "    import matplotlib.pyplot as _es_plt\n"
        "    _es_plt.show = lambda *a, **k: None\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    return future_block + header + body + "\n\n" + footer


def _run_child(paths: CompetitorPaths, out_dir: Path, timeout_s: int, *, emit_eval: bool) -> Dict[str, object]:
    """Run the sweep child; return the parsed child meta dict. Raises on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_eval" if emit_eval else "_sweep"
    script_path = out_dir / f"_evidence_ranker{suffix}_child.py"
    log_path = out_dir / f"_evidence_ranker{suffix}_child.log"

    script_path.write_text(_build_child_script(paths, out_dir, emit_eval=emit_eval), encoding="utf-8")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(out_dir / ".mpl_cache")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    (out_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"# evidence_ranker_sweep child (emit_eval={emit_eval})\n# notebook={paths.notebook}\n")
        log_handle.flush()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(out_dir),
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
            f"evidence_ranker_sweep child exited {completed.returncode}; log tail:\n{tail}"
        )

    meta_path = out_dir / _SWEEP_CHILD_META_FILENAME
    if not meta_path.is_file():
        raise RuntimeError(f"child exited 0 but wrote no {_SWEEP_CHILD_META_FILENAME}; see {log_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Parent: score each config's evidence canonically with fixed risk/behavior.   #
# --------------------------------------------------------------------------- #


@dataclass
class ConfigResult:
    config_name: str
    canonical_evidence_map: float
    notebook_evidence_map5: float
    delta_vs_floor: float
    clears_bar: bool
    meta: Dict[str, object]


def _fixed_confirmed_frame(rich: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The confirmed-dev prediction frame with FIXED risk + behavior (evidence blank).

    Reuses honghanh's own dev-holdout risk + predicted_behavior heads (the §73 rich
    cache). Evidence columns are filled ``NO_EVIDENCE`` here and OVERWRITTEN per config
    for the positive pairs, so PairAP + BehaviorMAP are identical across every config
    and only EvidenceMAP@5 moves.
    """
    confirmed_ids = {str(pid) for pid in labels["pair_id"]}
    conf = rich[rich["is_labeled"].astype(bool)].copy()
    conf["pair_id"] = conf["pair_id"].astype(str)
    conf = conf[conf["pair_id"].isin(confirmed_ids)]

    base = pd.DataFrame({"pair_id": conf["pair_id"].to_numpy()})
    base["risk_score"] = conf["oof_risk"].astype(float).clip(0.0, 1.0).to_numpy()
    base["predicted_behavior"] = conf["predicted_behavior"].astype(str).to_numpy()
    for i in range(1, 6):
        base[f"evidence_hand_{i}"] = "NO_EVIDENCE"
    return base


def _score_config(
    scorer: CanonicalScorer,
    base_frame: pd.DataFrame,
    config_evidence: pd.DataFrame,
) -> float:
    """Return the canonical EvidenceMAP@5 for one config's evidence hands."""
    ev = config_evidence.copy()
    ev["pair_id"] = ev["pair_id"].astype(str)
    ev_cols = [f"evidence_hand_{i}" for i in range(1, 6)]
    ev_map = {row.pair_id: row for row in ev.itertuples(index=False)}

    frame = base_frame.copy()
    ev_lookup = ev.set_index("pair_id")
    overlap = frame["pair_id"].isin(ev_lookup.index)
    for col in ev_cols:
        vals = frame.loc[overlap, "pair_id"].map(ev_lookup[col])
        frame.loc[overlap, col] = vals.fillna("NO_EVIDENCE").astype(str).values
    frame = frame[list(SUBMISSION_COLUMNS)]

    score = scorer.score_dev_predictions(frame)
    return float(score.components.evidence_map)


def run_evidence_ranker_sweep(
    poker_root: Path,
    *,
    force: bool = False,
    timeout_s: int = _CHILD_TIMEOUT_S,
) -> Dict[str, object]:
    """Run the §75 evidence-ranker sweep. BUILD + DEV-TUNE ONLY (no submit)."""
    poker_root = Path(poker_root)
    config = PipelineConfig()
    data_dir = Path(config.input_dir)
    labels = pd.read_csv(data_dir / "development_labels.csv")
    evidence = pd.read_csv(data_dir / "development_evidence.csv")

    comp_paths = default_competitor_paths(poker_root)
    out_dir = poker_root / "outputs" / "poker_collusion" / "evidence_ranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (0) Ensure the §73 rich dev-holdout cache (fixed risk + behavior heads) exists.
    rich_oof = Path(comp_paths.output_dir).resolve() / _RICH_OOF_FILENAME
    rich_meta = Path(comp_paths.output_dir).resolve() / _RICH_META_FILENAME
    if not (rich_oof.is_file() and rich_meta.is_file()):
        print("evidence_ranker_sweep: rich dev-holdout cache missing; building via §73...", flush=True)
        run_component_decomposition(poker_root, force=False, timeout_s=timeout_s)
    rich = pd.read_parquet(rich_oof)
    fixed_meta = json.loads(rich_meta.read_text(encoding="utf-8"))

    scorer = CanonicalScorer(
        ScoringRecipe(recipe_id="evidence_ranker_sweep_v1"), config, labels=labels, evidence=evidence
    )
    base_frame = _fixed_confirmed_frame(rich, labels)

    # Sanity: the FIXED frame with NO_EVIDENCE should give EvidenceMAP@5 == 0 (the
    # true positives get no ranked hits). Confirms risk/behavior are held fixed and
    # EvidenceMAP is fully attributable to evidence hands.
    zero_frame = base_frame[list(SUBMISSION_COLUMNS)].copy()
    zero_score = scorer.score_dev_predictions(zero_frame)
    print(
        f"evidence_ranker_sweep: NO_EVIDENCE baseline -> pair_ap={zero_score.components.pair_ap:.4f} "
        f"behavior_map={zero_score.components.behavior_map:.4f} evidence_map={zero_score.components.evidence_map:.4f}",
        flush=True,
    )

    # (1)-(4) Run the child sweep (dev only; no eval yet).
    ev_parquet = out_dir / _SWEEP_EVIDENCE_FILENAME
    child_meta_path = out_dir / _SWEEP_CHILD_META_FILENAME
    if ev_parquet.is_file() and child_meta_path.is_file() and not force:
        child_meta = json.loads(child_meta_path.read_text(encoding="utf-8"))
        print("evidence_ranker_sweep: reused per-config evidence cache", flush=True)
    else:
        child_meta = _run_child(comp_paths, out_dir, timeout_s, emit_eval=False)

    all_ev = pd.read_parquet(ev_parquet)
    config_metas: Dict[str, Dict[str, object]] = child_meta["configs"]  # type: ignore[assignment]

    results: List[ConfigResult] = []
    for config_name, cmeta in config_metas.items():
        cfg_ev = all_ev[all_ev["config"] == config_name].drop(columns=["config"])
        canonical = _score_config(scorer, base_frame, cfg_ev)
        delta = canonical - FLOOR_CANONICAL_EVIDENCE_MAP
        results.append(
            ConfigResult(
                config_name=config_name,
                canonical_evidence_map=canonical,
                notebook_evidence_map5=float(cmeta.get("notebook_evidence_map5", float("nan"))),
                delta_vs_floor=delta,
                clears_bar=bool(canonical >= FLOOR_CANONICAL_EVIDENCE_MAP + SWEEP_BAR_DELTA),
                meta=dict(cmeta),
            )
        )

    # Locate the reproduced FLOOR config (honghanh's current: base, pairwise, w=0.75).
    floor_name = "a_blendw_0.75_base_pairwise"
    floor_result = next((r for r in results if r.config_name == floor_name), None)
    floor_reproduced = floor_result.canonical_evidence_map if floor_result else float("nan")
    floor_ok = (
        floor_result is not None
        and abs(floor_reproduced - FLOOR_CANONICAL_EVIDENCE_MAP) <= FLOOR_TOLERANCE
    )

    results_sorted = sorted(results, key=lambda r: r.canonical_evidence_map, reverse=True)
    best = results_sorted[0] if results_sorted else None

    eval_cached = False
    eval_note = "not attempted"
    if not floor_ok:
        eval_note = (
            "SKIPPED: floor did not reproduce within tolerance — sweep NOT trusted "
            "(contract Rule 11). No eval evidence produced."
        )
    elif best is not None and best.clears_bar:
        # (5) Produce + cache eval evidence for the winning config.
        req = {
            "config_name": best.config_name,
            "features": best.meta.get("features", "base"),
            "objective": best.meta.get("objective", "rank:pairwise"),
            "per_family": bool(best.meta.get("per_family", False)),
            "w": float(best.meta.get("w", 0.75)),
        }
        (out_dir / "_eval_request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
        try:
            _run_child(comp_paths, out_dir, timeout_s, emit_eval=True)
            eval_cached = (out_dir / _EVAL_EVIDENCE_FILENAME).is_file()
            eval_note = (
                f"CACHED eval evidence for winning config {best.config_name!r} to "
                f"{_EVAL_EVIDENCE_FILENAME} (NOT assembled into a CSV; NOT submitted)."
                if eval_cached
                else "eval child ran but wrote no parquet"
            )
        except Exception as exc:  # noqa: BLE001 - surface honestly
            eval_note = f"eval evidence production FAILED: {exc!r}"
    else:
        eval_note = "no config cleared the bar; eval evidence not produced (as specified)."

    summary = {
        "note": (
            "BUILD + DEV-TUNE ONLY (dossier §75). Canonical dev EvidenceMAP@5 sweep. "
            "risk_score + predicted_behavior are held FIXED at honghanh's own dev-holdout "
            "heads (§73 rich cache) across every config, so PairAP + BehaviorMAP are "
            "identical and only EvidenceMAP@5 moves. These are MEASURED LOCAL dev numbers, "
            "NOT LB claims, and are NOT proven to transfer to eval (contract Rules 4, 9). "
            "NO submission was made and no submission CSV was written."
        ),
        "bar": {
            "floor_canonical_evidence_map": FLOOR_CANONICAL_EVIDENCE_MAP,
            "delta": SWEEP_BAR_DELTA,
            "clears_threshold": FLOOR_CANONICAL_EVIDENCE_MAP + SWEEP_BAR_DELTA,
        },
        "floor_reproduction": {
            "expected": FLOOR_CANONICAL_EVIDENCE_MAP,
            "reproduced": floor_reproduced,
            "tolerance": FLOOR_TOLERANCE,
            "reproduced_ok": bool(floor_ok),
            "floor_config": floor_name,
        },
        "no_evidence_baseline": {
            "pair_ap": float(zero_score.components.pair_ap),
            "behavior_map": float(zero_score.components.behavior_map),
            "evidence_map": float(zero_score.components.evidence_map),
        },
        "fixed_heads_meta": fixed_meta,
        "child_meta": {
            k: v for k, v in child_meta.items() if k != "configs"
        },
        "configs": [
            {
                "config_name": r.config_name,
                "canonical_evidence_map": r.canonical_evidence_map,
                "notebook_evidence_map5": r.notebook_evidence_map5,
                "delta_vs_floor": r.delta_vs_floor,
                "clears_bar": r.clears_bar,
                "config_meta": r.meta,
            }
            for r in results_sorted
        ],
        "best_config": (
            {
                "config_name": best.config_name,
                "canonical_evidence_map": best.canonical_evidence_map,
                "notebook_evidence_map5": best.notebook_evidence_map5,
                "delta_vs_floor": best.delta_vs_floor,
                "clears_bar": best.clears_bar,
            }
            if best is not None
            else None
        ),
        "eval_evidence_cached": eval_cached,
        "eval_note": eval_note,
    }

    (out_dir / "_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_table(summary)
    return summary


def _print_table(summary: Dict[str, object]) -> None:
    fr = summary["floor_reproduction"]  # type: ignore[index]
    bar = summary["bar"]  # type: ignore[index]
    print()
    print("=" * 96)
    print("§75 EVIDENCE-RANKER SWEEP — CANONICAL dev EvidenceMAP@5 (BUILD + DEV-TUNE ONLY; NOT LB)")
    print("=" * 96)
    print(
        f"Floor reproduction: expected {fr['expected']:.4f}, reproduced "  # type: ignore[index]
        f"{fr['reproduced']:.4f}  -> {'OK' if fr['reproduced_ok'] else 'MISMATCH'}"  # type: ignore[index]
    )
    print(
        f"Bar: clears iff canonical >= {bar['clears_threshold']:.4f} "  # type: ignore[index]
        f"(floor {bar['floor_canonical_evidence_map']:.4f} + {bar['delta']:.4f})"  # type: ignore[index]
    )
    print("-" * 96)
    header = f"{'config':<40}{'canonicalMAP@5':>16}{'notebookMAP5':>14}{'delta':>10}{'clears':>8}"
    print(header)
    print("-" * 96)
    for c in summary["configs"]:  # type: ignore[index]
        print(
            f"{c['config_name']:<40}{c['canonical_evidence_map']:>16.4f}"
            f"{c['notebook_evidence_map5']:>14.4f}{c['delta_vs_floor']:>+10.4f}"
            f"{('YES' if c['clears_bar'] else 'no'):>8}"
        )
    print("-" * 96)
    best = summary["best_config"]  # type: ignore[index]
    if best is not None:
        print(
            f"BEST: {best['config_name']} canonical={best['canonical_evidence_map']:.4f} "
            f"delta={best['delta_vs_floor']:+.4f} clears_bar={best['clears_bar']}"
        )
    print(f"Eval evidence cached: {summary['eval_evidence_cached']} — {summary['eval_note']}")  # type: ignore[index]
    print("=" * 96)


def _main() -> None:
    parser = argparse.ArgumentParser(description="§75 evidence-ranker sweep (BUILD + DEV-TUNE only).")
    parser.add_argument("--poker-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--force", action="store_true", help="Recompute the per-config evidence cache.")
    parser.add_argument("--timeout", type=int, default=_CHILD_TIMEOUT_S)
    args = parser.parse_args()
    run_evidence_ranker_sweep(Path(args.poker_root), force=args.force, timeout_s=args.timeout)


if __name__ == "__main__":
    _main()
