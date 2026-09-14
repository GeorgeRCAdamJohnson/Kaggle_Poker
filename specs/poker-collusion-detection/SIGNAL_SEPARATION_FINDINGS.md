# SIGNAL SEPARATION FINDINGS — Phase −1 Discovery validation (post-hoc)

> **What this is.** The measured separation evidence the Phase −1 HARD GATE was supposed to require
> *before* any modeling, but which was skipped: for every feature we already compute (all `conf_*`
> confounder features plus the value-flow / aggression / isolation / episodic features) and for the
> current classical `risk_score`, how well does it separate **confirmed-target** from
> **confirmed-non-target** development pairs on the REAL `development_labels.csv`, under
> **group-by-pool** cross-validation (pool = `hands.table_id`)?
>
> **Bounded investigation.** No production pipeline changes, no submission, no score-chasing.
> Deliverable = ranked separation table + honest verdict + Dossier updates.
>
> **Reproduce.** `python -m poker_collusion.discovery.signal_separation --out outputs/signal_separation_report.txt`
> from `…\poker` with `PYTHONPATH=…\poker`. Importable API + synthetic unit test:
> `poker_collusion/discovery/signal_separation.py`, `poker_collusion/tests/test_signal_separation.py`
> (7 tests, all passing). ASCII report: `poker/outputs/signal_separation_report.txt`.

## How it was measured

- **Population.** All confirmed labelled dev pairs with development shared hands:
  **n_pos = 372**, **n_neg = 1,488**, across **n_pools = 397** tables. (No cap needed — the full
  confirmed-pair single-pass over `actions.parquet` completed; same bounded cached-inputs pattern
  as the Phase-1 floor / Phase-2 gate, so the 18.6M-row file is streamed once, never materialised.)
- **Features.** The exact development-phase `PairFeatureSet` each pair emits (`build_pair_feature_set`,
  `phase="development"`), plus the current `classical_risk_score` (`models.classical.score_pair`),
  plus three cheap derived candidates from the existing caches (see below).
- **Per-feature discriminative power.**
  - **ROC AUC (orientation-max):** `roc_auc_score` of the feature and of its negation; keep the max
    and record orientation (`+` higher = more suspicious, `−` lower = more suspicious).
  - **Group-by-pool CV AUC (mean ± std):** leave-one-pool-out held-out AUC with orientation fixed;
    single-class held-out pools skipped. This is the honest number (guards within-pool leakage).
  - **Cohen's d:** `(mean_pos − mean_neg) / pooled_std`.
- **Multivariate upper bound:** ONE standardized logistic regression over ALL features, same
  group-by-pool CV — roughly the best a linear model over the CURRENT features can do.

## Ranked separation table (top 16 by group-by-pool CV AUC)

| # | feature | CV AUC (mean ± std) | Cohen's d | orient | whole-sample AUC |
|---|---------|--------------------:|----------:|:------:|-----------------:|
| 1 | `value_flow_abs_p95` | **0.886 ± 0.207** | 1.244 | + | 0.882 |
| 2 | `value_flow_abs_p90` | 0.831 ± 0.235 | 0.805 | + | 0.843 |
| 3 | `value_flow_abs_mean` | 0.799 ± 0.282 | 1.386 | + | 0.807 |
| 4 | `value_flow_absmean` | 0.799 ± 0.282 | 1.386 | + | 0.807 |
| 5 | `aggression_asymmetry_mean` | 0.782 ± 0.299 | 1.128 | + | 0.767 |
| 6 | `conf_avoidance_gap` (H15) | 0.782 ± 0.299 | −1.128 | − | 0.767 |
| 7 | `isolation_mean` | 0.732 ± 0.305 | −0.783 | − | 0.712 |
| 8 | `deriv_directedness_abs` (H14 magnitude) | 0.725 ± 0.322 | 1.190 | + | 0.733 |
| 9 | `value_flow_max_pos` | 0.710 ± 0.302 | 0.724 | + | 0.716 |
| 10 | `episodic_peak` | 0.700 ± 0.315 | 0.804 | + | 0.724 |
| 11 | `value_flow_abs_max` / `value_flow_max_abs` | 0.700 ± 0.315 | 0.804 | + | 0.724 |
| 13 | `flagged_rate_shrunk` | 0.692 ± 0.315 | 0.634 | + | 0.677 |
| 14 | `flagged_rate` | 0.689 ± 0.320 | 0.626 | + | 0.675 |
| 15 | `episodic_score` | 0.683 ± 0.299 | 0.717 | + | 0.712 |
| 16 | **`classical_risk_score`** | **0.667 ± 0.353** | 0.694 | + | 0.681 |

**Confounder-feature (`conf_*`) verdicts, as built:**

| feature | CV AUC | Cohen's d | read |
|---------|-------:|----------:|------|
| `conf_avoidance_gap` (H15) | **0.782** | −1.128 | **CONFIRMED** — strong; mirrors `aggression_asymmetry_mean`. |
| `conf_joint_isolation_rate` (H16) | 0.515 | 0.075 | **REFUTED** — chance (raw `isolation_mean` 0.732 carries the isolation signal instead). |
| `conf_temporal_concentration` (H17) | 0.529 | −0.140 | **REFUTED** — chance as a standalone separator. |
| `conf_directedness_contrast` (H14, signed) | 0.487 | 0.047 | **REFUTED as built** — signed A→B vs B→A cancels; but its **magnitude** separates (row 8). |
| `conf_field_baseline_change_stub` (H17 strategy-change) | 0.500 | 0.000 | **NOT MEASURED** — fixed 0.0 stub at this layer. |

**Multivariate upper bound (logistic over ALL features, group-by-pool CV): AUC = 0.850 ± 0.254 (240 folds).**

## Honest verdict

**On development labels, discriminative signal clearly EXISTS.** Multiple raw features clear the
useful bar (CV AUC ≥ ~0.65) by a wide margin — the value-flow tail/mean stats and the
aggression-avoidance gap top out at **0.78–0.89** with large effect sizes (Cohen's d ≈ 1.1–1.4), and
the multivariate linear ceiling is **0.85**. So the answer to "is there ANY feature or model that
plausibly separates positives from negatives on dev?" is an emphatic **yes**.

**But this must be read against the leaderboard, not in isolation — and there it is a warning, not a
victory.** Both real submissions scored *below* the do-nothing baseline (Phase-1 0.05713, Phase-2
0.06236 vs sample 0.08404), yet dev-label separation is 0.85–0.89. Those two facts only reconcile
one way: **the dev-label ceiling is high but the signal did not TRANSFER to the private evaluation
Pair AP.** The bottleneck is therefore *not* "no raw signal in the features" (the earlier
Leaderboard-Results read) and *not* risk calibration alone — it is a **train→eval transfer gap**.
The most likely, cheaply-checkable culprits, in priority order:

1. **Dev-vs-eval feature availability / definition drift.** The strongest separators are
   `value_flow_*` and `aggression_asymmetry_mean` computed over each pair's *development* shared
   hands. On the real submission these are computed over *evaluation* shared hands, and the
   production evidence/validity gates behave differently across phases. If the eval-period signal is
   sparser or shaped differently, a dev-fit ranking need not hold on eval.
2. **PU-label optimism.** Confirmed negatives are a *curated* sample, not a random draw from the
   112,540 evaluation pairs. AUC that separates confirmed-target from confirmed-non-target can be
   far higher than AP over the true rare-positive evaluation population (prevalence ≈ 0.214%).
3. **The current `classical_risk_score` under-uses the signal.** It ranks at CV AUC **0.667** —
   above chance but well below the best single raw feature (0.886) and the linear ceiling (0.85).
   The combiner is leaving a lot of the available dev separation on the table.

**Ceiling implication (stated plainly).** There is real headroom in the *current* features on dev
labels — a better combiner (or a learned linear model) could push a dev-CV ranking from ~0.67 to
~0.85 AUC. Whether that converts to Pair AP above baseline on the private eval set is **unproven and
doubted**, because the same features already shipped and lost to a constant. The highest-leverage
next step is therefore **not** more feature engineering but a **transfer diagnostic**: measure these
same per-feature AUCs on an eval-period-defined feature set (and against a *random* unknown-pair
negative sample, not just confirmed negatives) to see how much of the 0.85 survives. If it collapses,
the fix is distributional (dev/eval alignment, PU handling), not a new signal.

## Not measured (recorded, not faked)

- **Action timing / inter-action latency contrast** — needs per-action wall-clock timestamps, which
  are **not** in the cached action rows (only ordinal `action_no`). Would require a new `actions`
  projection.
- **Field-baseline change-point over time** (H17 vs *strategy changes*) — needs the ordered
  per-player-vs-field signal series over time; `conf_field_baseline_change_stub` is fixed at 0.0 at
  the pair-aggregation layer.

## Process lesson

This measurement should have run **before** any modeling. The Phase −1 gate documented H14–H17 as
hypotheses and then let modeling proceed with them `OPEN — PENDING DATA`; the numbers here (that the
signed H14 contrast and H16/H17 `conf_*` features are near chance, while a couple of raw stats and
the *magnitude* of H14 carry almost all the separation) would have redirected feature and combiner
design from day one — and, more importantly, would have surfaced the **transfer question** before two
scored submissions. The gate must require **measured per-feature separation on dev labels (AUC under
group-by-pool CV)**, not just written hypotheses, before modeling begins.
