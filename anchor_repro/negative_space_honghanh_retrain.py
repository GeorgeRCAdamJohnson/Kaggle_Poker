"""§69 — retrain honghanh's 0.64 risk head with the 3 negative-space RANK columns.

Pre-registration: RESEARCH_DOSSIER §69 (append-only; bar FIXED before this ran).
Governed by the workspace ``reverse-engineering-accountability`` contract
(Rules 1, 2, 3, 4, 6, 9, 11, 12, 15, 16, 17, 19). BUILD + HOLDOUT-OOF ONLY. This
module NEVER submits to Kaggle and NEVER overwrites ``submission.csv`` /
``submission_best_*.csv``.

The hypothesis (§69)
--------------------
Established on the FLOOR base: appending the 3 negative-space rank columns
(``ns_clash_rank, ns_contest_rank, ns_foldyield_rank``) as SEPARATE features and
retraining lifted the LB by +0.034. Test whether the SAME FORM — the 3 ns columns
appended as separate features and RETRAINED — lifts the STRONGER honghanh 0.64 head
on honghanh's OWN 5-fold table-grouped holdout (StratifiedGroupKFold by table_id,
SEED=42, the notebook's exact risk_params/family params/combiners/pu_stress rule).

Pre-registered bar (Rule 2, BINDING, FIXED before any number was produced)
--------------------------------------------------------------------------
* success threshold: delta = augmented_confirmed_PairAP - baseline_confirmed_PairAP
  must be >= +0.005 on honghanh's OWN 5-fold table-grouped OOF (the external-to-this-
  build held-out judge: a pair's OOF risk is predicted by a model that never saw that
  pair's table — a table-disjoint holdout).
* baseline/null it must beat: honghanh-ALONE confirmed PairAP (the 271-feature head,
  ``repro_064469/dev_holdout_oof_064469.parquet``, already on disk).
* judged on: the CONFIRMED (is_labeled==True) pairs' selected-combiner OOF risk,
  scored by the CanonicalScorer's canonical confirmed-only PairAP (identical
  construction to ``competitor_recipe._emit_dev_predictions`` + §67).
If delta < +0.005 the result is an HONEST NULL (Rule 3): no eval CSV is built, the
floor/honghanh live-bests are untouched.

Faithful reuse (Rule 6 — reuse, do not reinvent)
------------------------------------------------
The heavy CV runs in a CHILD PROCESS built from ``competitor_recipe._build_child_script``'s
machinery: the notebook's OWN feature functions (``aggregate_pair_features``,
``add_player_baselines``, ``add_partner_field_contrasts``, ``matrix_from``) rebuild the
271-feature pair matrix, and the notebook's EXACT 5-fold StratifiedGroupKFold risk head
+ 3 family-OvR detectors + 4 combiners + pu_stress selection run verbatim. The ONLY
modifications vs honghanh's child (all in an AUTHORED footer):
  (a) after ``_cr_train`` is built and BEFORE ``matrix_from``, LEFT-JOIN the 3 ns rank
      columns onto ``_cr_train`` by pair_id (polars join; nulls -> 0.0);
  (b) the 3 ns column names are appended to ``_cr_feature_cols`` (matrix -> 274 feats);
  (c) output paths point to ``repro_064469_ns`` (honghanh's original cache untouched);
  (d) AFTER the CV, ALSO fit the risk head on ALL dev rows (274 feats) and predict the
      112,540 eval pairs (eval matrix built by the SAME notebook functions on the eval
      pool + the EVAL ns 3-rank block), writing ``eval_risk`` to the augmented cache.

Row-set parity note (Rule 9 — stated uncertainty, not fabrication)
------------------------------------------------------------------
honghanh's dev_pairs (25,860) and the floor's ``dev_v5`` (25,860) share ALL 1,860
CONFIRMED pairs but only 6,736 of the 19,124 PU-unknown negatives (the PU negatives are
independently sampled). The ns block is therefore built over honghanh's OWN dev pairs
(``pair_id, a=player_1, b=player_2`` from ``dev_pairs.parquet``) so that ALL 25,860
training rows carry a real ns value — NOT over ``dev_v5`` (which would leave ~19k
honghanh training rows at ns=0, a composition bug that would bias the CV against the
idea — Rule 11: correct the EXECUTION, do not let it become a verdict on the idea). The
same ``build_negative_space`` code path, same k=50, and same field-exclusion positive
players (identical set to ``_dev_positive_players``, verified) are used, so this is the
negative-space idea executed CORRECTLY on honghanh's row set, not a divergent build.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from poker_collusion.config import PipelineConfig

from anchor_repro.competitor_recipe import (
    _build_child_script,
    _CHILD_FOOTER_TEMPLATE,
    _emit_dev_predictions,
    default_competitor_paths,
)
from anchor_repro.models import ScoringRecipe
from anchor_repro.scorer import CanonicalScorer
from anchor_repro.negative_space import (
    NEGATIVE_SPACE_RANK_COLUMNS,
    build_negative_space,
    default_negative_space_cache_locations,
    negative_space_drift_gate,
    _dev_positive_players,
)
from anchor_repro.combo_screen import validate_eval_submission

__all__ = [
    "run_ns_honghanh_retrain",
]

#: §69 pre-registered success bar (BINDING, FIXED).
_BAR: float = 0.005

#: The ns shrinkage k this experiment is fit at (recorded; not swept here — the FLOOR
#: +0.034 was established at the k the block was shipped with; §69 tests the SAME form).
_K: float = 50.0

#: Child wall-clock cap (CV ~1h + a full-dev fit + eval predict). 5400s per the spec.
_CHILD_TIMEOUT_S: int = 5400

_NS_OUT_SUBDIR = ("outputs", "poker_collusion", "repro_064469_ns")
_NS_STAGING_SUBDIR = ("outputs", "poker_collusion", "ns_on_honghanh")

_AUG_OOF_NAME = "dev_holdout_oof_064469_ns.parquet"
_AUG_META_NAME = "dev_holdout_oof_064469_ns_meta.json"
_EVAL_RISK_NAME = "eval_risk_064469_ns.parquet"
_NS_DEV_STAGE = "_ns_dev_3rank_k50.parquet"
_NS_EVAL_STAGE = "_ns_eval_3rank_k50.parquet"


# --------------------------------------------------------------------------- #
# The AUGMENTED child footer — modeled on competitor_recipe._CHILD_FOOTER_     #
# TEMPLATE, with (a) the ns left-join, (b) ns cols in _cr_feature_cols,        #
# (c) repro_064469_ns output paths, (d) a full-dev-fit + eval-predict tail.    #
# --------------------------------------------------------------------------- #

_AUG_FOOTER = '''\
# ==== negative_space_honghanh_retrain child footer (AUTHORED BY THE RUNNER) ====
# Reuse the notebook's OWN feature + model code (imported above), identical to the
# honghanh dev-OOF child, with the ONLY diffs: (a) LEFT-JOIN the 3 ns rank columns
# onto _cr_train by pair_id BEFORE matrix_from, (b) include the 3 ns cols in
# _cr_feature_cols (matrix -> 274), (c) outputs to repro_064469_ns, (d) a full-dev
# fit + eval predict tail. Everything else (5-fold SGKF, SEED, risk_params, family
# params, combiners, pu_stress selection) is verbatim honghanh (contract Rule 6).
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
_CR_EVAL_RISK_PATH = _CR_OUT / {eval_risk_name!r}
_CR_NS_DEV_PATH = _CrPath({ns_dev_literal})
_CR_NS_EVAL_PATH = _CrPath({ns_eval_literal})
_CR_NS_COLS = list({ns_cols!r})

_CR_OUT.mkdir(parents=True, exist_ok=True)
print("ns_honghanh_retrain: prepared_dir =", _CR_PREP, flush=True)

_cr_dev_pairs = _cr_pl.read_parquet(_CR_PREP / "dev_pairs.parquet")
_cr_dev_hand_path = _CR_PREP / "dev_hand_features.parquet"
_cr_player_hand_path = _CR_PREP / "player_hands.parquet"
_cr_eval_pairs_prep = _cr_pl.read_parquet(_CR_PREP / "eval_pairs.parquet")
_cr_eval_hand_path = _CR_PREP / "eval_hand_features.parquet"

# Rebuild the pair-level train matrix with the notebook's OWN functions (verbatim).
_cr_hands = _cr_pl.scan_parquet(_CR_DATA / "hands.parquet")
_cr_seats = _cr_pl.scan_parquet(_CR_DATA / "seats.parquet")
_cr_eval_pairs = _cr_pl.read_csv(_CR_DATA / "evaluation_pairs.csv")

print("ns_honghanh_retrain: aggregating dev pair features (notebook code)...", flush=True)
_cr_train = aggregate_pair_features(_cr_dev_hand_path, _cr_dev_pairs)
_cr_train = add_player_baselines(
    _cr_train, player_phase_baselines(_cr_player_hand_path, "development")
)
_cr_train = add_partner_field_contrasts(
    _cr_train, _cr_seats, _cr_hands, _cr_player_hand_path, "development"
)

# --- (a) LEFT-JOIN the 3 ns rank columns onto _cr_train by pair_id (nulls -> 0.0). ---
_cr_ns_dev = _cr_pl.read_parquet(_CR_NS_DEV_PATH)
_cr_ns_dev = _cr_ns_dev.with_columns(_cr_pl.col("pair_id").cast(_cr_pl.Utf8))
_cr_train = _cr_train.with_columns(_cr_pl.col("pair_id").cast(_cr_pl.Utf8))
_cr_n_before = _cr_train.height
_cr_train = _cr_train.join(
    _cr_ns_dev.select(["pair_id"] + _CR_NS_COLS), on="pair_id", how="left"
)
_cr_train = _cr_train.with_columns(
    [_cr_pl.col(c).fill_null(0.0).cast(_cr_pl.Float64) for c in _CR_NS_COLS]
)
_cr_n_ns_matched = int(
    _cr_ns_dev.filter(_cr_pl.col("pair_id").is_in(_cr_train["pair_id"])).height
)
assert _cr_train.height == _cr_n_before, "ns left-join changed dev row count"
print(
    f"ns_honghanh_retrain: ns dev join matched {{_cr_n_ns_matched}}/{{_cr_train.height}} rows",
    flush=True,
)

_cr_exclude = {{
    "pair_id", "player_1", "player_2", "label", "behavior_family",
    "table_id", "is_labeled", "is_pu",
}}
_cr_feature_cols = [
    c for c, dtype in _cr_train.schema.items()
    if dtype.is_numeric() and c not in _cr_exclude
]
# --- (b) ensure the 3 ns cols are INCLUDED (numeric, not excluded => already in). ---
for _cr_nc in _CR_NS_COLS:
    assert _cr_nc in _cr_feature_cols, f"ns col {{_cr_nc}} missing from feature_cols"
_cr_X = matrix_from(_cr_train, _cr_feature_cols)
print(
    f"ns_honghanh_retrain: train matrix {{_cr_X.shape[0]}} x {{_cr_X.shape[1]}} "
    f"(expect 274 features incl 3 ns)",
    flush=True,
)
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
    f"ns_honghanh_retrain: pos={{int((_cr_y==1).sum())}} "
    f"conf_neg={{int((_cr_is_labeled & (_cr_y==0)).sum())}} PU={{int(_cr_is_pu.sum())}}",
    flush=True,
)

# The notebook's EXACT risk + family-OvR hyperparameters (verbatim).
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
    print(f"ns_honghanh_retrain: fold {{_cr_fold}}/5 done", flush=True)

# The notebook's four risk combiners + its PU-stress selection rule (verbatim).
_cr_cands = combine_risk_candidates(_cr_oof_risk, _cr_oof_family)
_cr_pu_ap = {{
    name: pu_stress_ap(_cr_y, scores, _cr_is_pu, _cr_n_eval)
    for name, scores in _cr_cands.items()
}}
_cr_best = max(_cr_pu_ap, key=lambda n: _cr_pu_ap[n])
print("ns_honghanh_retrain: combiner PU-stress AP =", _cr_pu_ap, flush=True)
print("ns_honghanh_retrain: selected combiner =", _cr_best, flush=True)

# Persist the AUGMENTED OOF risk over ALL dev pairs + every combiner (audit trail).
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
    "ns_cols": _CR_NS_COLS,
    "ns_dev_matched": int(_cr_n_ns_matched),
}}
_CR_META_PATH.write_text(_cr_json.dumps(_cr_meta, indent=2), encoding="utf-8")
print("ns_honghanh_retrain: wrote augmented OOF cache", _CR_OOF_PATH, flush=True)

# --- (d) full-dev fit (274 feats) + predict the 112,540 eval pairs. ---
print("ns_honghanh_retrain: building eval matrix (notebook code)...", flush=True)
_cr_eval_feats = aggregate_pair_features(_cr_eval_hand_path, _cr_eval_pairs_prep)
_cr_eval_feats = add_player_baselines(
    _cr_eval_feats, player_phase_baselines(_cr_player_hand_path, "evaluation")
)
_cr_eval_feats = add_partner_field_contrasts(
    _cr_eval_feats, _cr_seats, _cr_hands, _cr_player_hand_path, "evaluation"
)
_cr_ns_eval = _cr_pl.read_parquet(_CR_NS_EVAL_PATH)
_cr_ns_eval = _cr_ns_eval.with_columns(_cr_pl.col("pair_id").cast(_cr_pl.Utf8))
_cr_eval_feats = _cr_eval_feats.with_columns(_cr_pl.col("pair_id").cast(_cr_pl.Utf8))
_cr_eval_feats = _cr_eval_feats.join(
    _cr_ns_eval.select(["pair_id"] + _CR_NS_COLS), on="pair_id", how="left"
)
_cr_eval_feats = _cr_eval_feats.with_columns(
    [_cr_pl.col(c).fill_null(0.0).cast(_cr_pl.Float64) for c in _CR_NS_COLS]
)
# Align eval feature columns EXACTLY to the trained matrix columns (order + presence).
_cr_Xe = matrix_from(_cr_eval_feats, _cr_feature_cols)
print(
    f"ns_honghanh_retrain: eval matrix {{_cr_Xe.shape[0]}} x {{_cr_Xe.shape[1]}}",
    flush=True,
)
# Fit the risk head on ALL dev rows (no early stopping — no held-out fold here).
_cr_full_params = dict(_cr_risk_params)
_cr_full_params.pop("early_stopping_rounds", None)
_cr_full = _CrXGB(**_cr_full_params, random_state=SEED)
_cr_full.fit(_cr_X, _cr_y, sample_weight=_cr_fit_weight)
_cr_eval_risk = _cr_full.predict_proba(_cr_Xe)[:, 1].astype(_cr_np.float32)
_cr_eval_out = _cr_pd.DataFrame({{
    "pair_id": _cr_eval_feats["pair_id"].to_list(),
    "eval_risk": _cr_eval_risk,
}})
_cr_eval_out.to_parquet(_CR_EVAL_RISK_PATH, index=False)
print("ns_honghanh_retrain: wrote eval risk", _CR_EVAL_RISK_PATH, flush=True)
print("ns_honghanh_retrain: DONE", flush=True)
# ==== end negative_space_honghanh_retrain child footer ====
'''


def _build_augmented_child_script(
    competitor_paths,
    ns_out_dir: Path,
    ns_dev_stage: Path,
    ns_eval_stage: Path,
) -> str:
    """Assemble the augmented child: notebook body (functions only) + our ns footer.

    Reuses ``competitor_recipe._build_child_script`` to obtain the neutralised notebook
    body (its ``main()`` / EDA drivers disabled so importing defines functions only),
    then swaps its honghanh footer for our AUGMENTED footer. We rebuild the body via
    ``_build_child_script`` and strip everything from the honghanh footer sentinel on,
    guaranteeing the SAME neutralisation as honghanh's child (Rule 6).
    """
    base = _build_child_script(competitor_paths)
    sentinel = "# ==== competitor_recipe child footer"
    idx = base.find(sentinel)
    if idx == -1:
        raise RuntimeError(
            "could not locate competitor_recipe footer sentinel in the base child "
            "script; the reuse contract (Rule 6) is broken — refusing to guess."
        )
    body = base[:idx]

    footer = _AUG_FOOTER.format(
        prepared_literal=repr(str(Path(competitor_paths.prepared_dir).resolve())),
        output_literal=repr(str(Path(ns_out_dir).resolve())),
        data_literal=repr(str(Path(competitor_paths.data_dir).resolve())),
        oof_name=_AUG_OOF_NAME,
        meta_name=_AUG_META_NAME,
        eval_risk_name=_EVAL_RISK_NAME,
        ns_dev_literal=repr(str(Path(ns_dev_stage).resolve())),
        ns_eval_literal=repr(str(Path(ns_eval_stage).resolve())),
        ns_cols=tuple(NEGATIVE_SPACE_RANK_COLUMNS),
    )
    return body + footer


def _run_child(script: str, ns_out_dir: Path, log_path: Path, timeout_s: int) -> Dict:
    """Run the augmented child; return the parsed meta dict. Raises on any failure."""
    ns_out_dir.mkdir(parents=True, exist_ok=True)
    script_path = ns_out_dir / "_ns_honghanh_retrain_child.py"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(ns_out_dir / ".mpl_cache")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    (ns_out_dir / ".mpl_cache").mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write("# ns_honghanh_retrain augmented dev-OOF + eval re-run\n")
        log_handle.flush()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ns_out_dir),
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
            f"augmented child exited {completed.returncode}; log tail:\n{tail}"
        )
    meta_path = ns_out_dir / _AUG_META_NAME
    if not meta_path.is_file():
        raise RuntimeError(f"child exited 0 but wrote no {_AUG_META_NAME}; see {log_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _confirmed_pair_ap(
    oof_df: pd.DataFrame,
    scorer: CanonicalScorer,
    labels: pd.DataFrame,
    evidence: Optional[pd.DataFrame],
) -> float:
    """Canonical confirmed-only PairAP of an OOF frame (identical to §67 construction).

    Reuses ``competitor_recipe._emit_dev_predictions`` (build the confirmed dev-holdout
    frame from the OOF cache) then ``CanonicalScorer.score_dev_predictions().pair_ap``.
    """
    preds = _emit_dev_predictions(oof_df, labels, evidence=evidence)
    return float(scorer.score_dev_predictions(preds).pair_ap)


def run_ns_honghanh_retrain(poker_root: Path, *, force: bool = False) -> Dict:
    """Run the §69 experiment end-to-end (BUILD + HOLDOUT-OOF ONLY; never submits).

    Returns the summary dict (also written to ``ns_on_honghanh/_ns_honghanh_retrain_
    summary.json``).
    """
    poker_root = Path(poker_root).resolve()
    data_dir = poker_root / "data" / "poker"
    dev_labels = data_dir / "development_labels.csv"
    dev_evidence = data_dir / "development_evidence.csv"
    eval_pairs_csv = data_dir / "evaluation_pairs.csv"

    ns_out_dir = poker_root.joinpath(*_NS_OUT_SUBDIR)
    stage_dir = poker_root.joinpath(*_NS_STAGING_SUBDIR)
    stage_dir.mkdir(parents=True, exist_ok=True)
    ns_out_dir.mkdir(parents=True, exist_ok=True)

    competitor_paths = default_competitor_paths(poker_root)
    missing = competitor_paths.missing()
    if missing:
        raise FileNotFoundError(
            "honghanh warm cache missing input(s): "
            + ", ".join(str(p) for p in missing)
        )
    caches = default_negative_space_cache_locations(poker_root)
    ns_missing = caches.missing()
    if ns_missing:
        raise FileNotFoundError(
            "negative-space caches missing: " + ", ".join(str(p) for p in ns_missing)
        )

    # ---- (1) Build the DEV ns 3-rank block over honghanh's OWN dev pairs (k=50). ----
    prepared = Path(competitor_paths.prepared_dir)
    hh_dev = pd.read_parquet(prepared / "dev_pairs.parquet", columns=["pair_id", "player_1", "player_2"])
    dev_pairs_ab = pd.DataFrame(
        {
            "pair_id": hh_dev["pair_id"].astype(str).to_numpy(),
            "a": hh_dev["player_1"].to_numpy(),
            "b": hh_dev["player_2"].to_numpy(),
        }
    )
    pos_players = _dev_positive_players(caches)
    print(f"[§69] building DEV ns block over {len(dev_pairs_ab)} honghanh pairs (k={_K})...", flush=True)
    dev_block = build_negative_space(
        dev_pairs_ab, "development", caches, k=_K, positive_players=pos_players
    )
    ns_dev_stage = stage_dir / _NS_DEV_STAGE
    dev_ns = dev_block[["pair_id", *NEGATIVE_SPACE_RANK_COLUMNS]].copy()
    dev_ns["pair_id"] = dev_ns["pair_id"].astype(str)
    dev_ns.to_parquet(ns_dev_stage, index=False)

    # ---- Build the EVAL ns 3-rank block over honghanh's OWN eval pairs (k=50). ----
    hh_eval = pd.read_parquet(prepared / "eval_pairs.parquet", columns=["pair_id", "player_1", "player_2"])
    eval_pairs_ab = pd.DataFrame(
        {
            "pair_id": hh_eval["pair_id"].astype(str).to_numpy(),
            "a": hh_eval["player_1"].to_numpy(),
            "b": hh_eval["player_2"].to_numpy(),
        }
    )
    print(f"[§69] building EVAL ns block over {len(eval_pairs_ab)} eval pairs (k={_K})...", flush=True)
    eval_block = build_negative_space(
        eval_pairs_ab, "evaluation", caches, k=_K, positive_players=set()
    )
    ns_eval_stage = stage_dir / _NS_EVAL_STAGE
    eval_ns = eval_block[["pair_id", *NEGATIVE_SPACE_RANK_COLUMNS]].copy()
    eval_ns["pair_id"] = eval_ns["pair_id"].astype(str)
    eval_ns.to_parquet(ns_eval_stage, index=False)

    # ---- (2-4) Run the augmented child (CV + full-dev fit + eval predict). ----
    aug_oof_path = ns_out_dir / _AUG_OOF_NAME
    aug_meta_path = ns_out_dir / _AUG_META_NAME
    eval_risk_path = ns_out_dir / _EVAL_RISK_NAME
    log_path = ns_out_dir / "_child.log"

    if aug_oof_path.is_file() and aug_meta_path.is_file() and eval_risk_path.is_file() and not force:
        print("[§69] reusing existing augmented cache (force=False)", flush=True)
        meta = json.loads(aug_meta_path.read_text(encoding="utf-8"))
    else:
        script = _build_augmented_child_script(
            competitor_paths, ns_out_dir, ns_dev_stage, ns_eval_stage
        )
        print(f"[§69] running augmented child (timeout {_CHILD_TIMEOUT_S}s); log -> {log_path}", flush=True)
        meta = _run_child(script, ns_out_dir, log_path, _CHILD_TIMEOUT_S)

    n_features = int(meta.get("n_features", -1))
    selected_combiner_aug = str(meta.get("selected_combiner", ""))

    # ---- (5) Score BOTH oof parquets' confirmed PairAP the §67 way. ----
    labels = pd.read_csv(dev_labels)
    evidence = pd.read_csv(dev_evidence) if dev_evidence.is_file() else None
    scorer = CanonicalScorer(
        ScoringRecipe(),
        PipelineConfig(input_dir=data_dir),
        labels=labels,
        evidence=evidence,
    )

    honghanh_oof_path = Path(competitor_paths.output_dir) / "dev_holdout_oof_064469.parquet"
    honghanh_meta_path = Path(competitor_paths.output_dir) / "dev_holdout_oof_064469_meta.json"
    baseline_oof = pd.read_parquet(honghanh_oof_path)
    augmented_oof = pd.read_parquet(aug_oof_path)

    baseline_pair_ap = _confirmed_pair_ap(baseline_oof, scorer, labels, evidence)
    augmented_pair_ap = _confirmed_pair_ap(augmented_oof, scorer, labels, evidence)
    delta = augmented_pair_ap - baseline_pair_ap
    clears_bar = bool(delta >= _BAR)

    selected_combiner_base = ""
    if honghanh_meta_path.is_file():
        selected_combiner_base = str(
            json.loads(honghanh_meta_path.read_text(encoding="utf-8")).get("selected_combiner", "")
        )

    # ---- (6) Drift gate for INFORMATION ONLY (not a veto). ----
    try:
        drift = negative_space_drift_gate(caches, k=_K)
        drift_auc = float(drift.auc)
        drift_pass = bool(drift.passed)
    except Exception as exc:  # pragma: no cover - informational only
        drift_auc = float("nan")
        drift_pass = False
        print(f"[§69] drift gate raised (informational): {exc!r}", flush=True)

    print("\n[§69] ===== RESULTS =====", flush=True)
    print(f"  baseline_pair_ap  = {baseline_pair_ap:.6f}", flush=True)
    print(f"  augmented_pair_ap = {augmented_pair_ap:.6f}", flush=True)
    print(f"  delta             = {delta:+.6f}  (bar >= {_BAR})", flush=True)
    print(f"  clears_bar        = {clears_bar}", flush=True)
    print(f"  n_features        = {n_features} (expect 274)", flush=True)
    print(f"  drift_auc         = {drift_auc:.4f}  {'PASS' if drift_pass else 'FAIL'} (<0.65, INFO ONLY)", flush=True)
    print(f"  selected_combiner base={selected_combiner_base!r} aug={selected_combiner_aug!r}", flush=True)

    eval_candidate_path: Optional[str] = None
    spearman: Optional[float] = None
    validation: Optional[Dict] = None

    if clears_bar:
        # ---- (7) Build the eval candidate CSV (NEW path; never overwrites live). ----
        honghanh_eval = pd.read_csv(Path(competitor_paths.output_dir) / "submission.csv")
        honghanh_eval["pair_id"] = honghanh_eval["pair_id"].astype(str)
        aug_eval_risk = pd.read_parquet(eval_risk_path)
        aug_eval_risk["pair_id"] = aug_eval_risk["pair_id"].astype(str)
        risk_by = dict(zip(aug_eval_risk["pair_id"], aug_eval_risk["eval_risk"].astype(float)))

        missing_eval = [p for p in honghanh_eval["pair_id"] if p not in risk_by]
        if missing_eval:
            raise RuntimeError(
                f"augmented eval risk missing {len(missing_eval)} honghanh eval pair_ids; "
                "refusing to fabricate (Rule 9)."
            )
        new_risk = np.array([risk_by[p] for p in honghanh_eval["pair_id"]], dtype=float)
        # Normalize to [0,1] by min-max (ranking is what matters; monotone).
        span = new_risk.max() - new_risk.min()
        norm_risk = (new_risk - new_risk.min()) / span if span > 0 else np.full(len(new_risk), 0.5)

        # Spearman vs honghanh base eval risk (rank correlation).
        base_risk = honghanh_eval["risk_score"].astype(float).to_numpy()
        spearman = float(
            pd.Series(norm_risk).corr(pd.Series(base_risk), method="spearman")
        )

        out = honghanh_eval.copy()
        out["risk_score"] = norm_risk.clip(0.0, 1.0)
        eval_candidate = stage_dir / "ns_honghanh_retrain_augmented.csv"
        out.to_csv(eval_candidate, index=False)
        eval_candidate_path = str(eval_candidate)

        eval_pairs_df = pd.read_csv(eval_pairs_csv)
        validation = validate_eval_submission(out, eval_pairs_df)
        print(f"  eval candidate    = {eval_candidate_path}", flush=True)
        print(f"  eval validation   = {validation}", flush=True)
        print(f"  spearman vs base  = {spearman:.6f}", flush=True)
    else:
        print("  HONEST NULL: delta below the +0.005 bar — NO eval CSV built. "
              "honghanh/floor live-bests untouched (Rules 2, 3, 12).", flush=True)

    summary = {
        "k": _K,
        "bar": _BAR,
        "baseline_pair_ap": baseline_pair_ap,
        "augmented_pair_ap": augmented_pair_ap,
        "delta": delta,
        "clears_bar": clears_bar,
        "n_features": n_features,
        "drift_auc": drift_auc,
        "drift_pass": drift_pass,
        "n_confirmed": int(meta.get("n_confirmed", -1)),
        "selected_combiner_baseline": selected_combiner_base,
        "selected_combiner_augmented": selected_combiner_aug,
        "combiner_pu_stress_ap_augmented": meta.get("combiner_pu_stress_ap", {}),
        "eval_candidate": eval_candidate_path,
        "eval_validation": validation,
        "spearman_vs_honghanh_base": spearman,
        "ns_dev_matched": int(meta.get("ns_dev_matched", -1)),
        "note": (
            "ns built over honghanh's OWN dev/eval pairs (all rows populated); "
            "confirmed scoring pairs identical to dev_v5. BUILD+HOLDOUT-OOF only; "
            "no Kaggle submit; submission.csv/submission_best_* untouched."
        ),
    }
    summary_path = stage_dir / "_ns_honghanh_retrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[§69] wrote summary -> {summary_path}", flush=True)
    return summary


def _find_poker_root() -> Path:
    """Resolve the ``poker/`` root from this file location (anchor_repro/..)."""
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    run_ns_honghanh_retrain(_find_poker_root())
