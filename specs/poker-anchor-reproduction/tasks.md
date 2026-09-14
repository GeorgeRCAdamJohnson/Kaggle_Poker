# Implementation Plan: poker-anchor-reproduction

## Overview

This plan turns REAL code with KNOWN real leaderboard scores into REAL `(local-holdout-AP, real-LB)` anchors. It is implemented in **Python** as a new `anchor_repro` package under the existing `poker/` tree, and it **reuses** the existing `poker_collusion` package verbatim — it does NOT re-implement the official metric, the CV split, or the Spearman statistic. Reused modules:

- `poker_collusion/metric/reference_public_metric.py` (verbatim official score) — Req 1.2/1.3 already satisfied by this existing local copy.
- `poker_collusion/metric/production_metric.py` (`score_components`).
- `poker_collusion/validation/cv.py` (`CVHarness`, `build_fold_solution`, pool-disjoint folds).
- `poker_collusion/validation/lb_cv.py` (`spearman_corr`).
- `poker_collusion/config.py` (`PipelineConfig` input-dir shim, `Seeds`).

The mandated build order is enforced by task ordering and the dependency graph: **(1) tooling + canonical scorer + official-metric self-check → (2) validate the scorer against our own nine known-LB artifacts (the rank-tracking external-judge gate) → (3) reproduce the 0.64469 competitor notebook as a new anchor, gated on step 2's verdict.** A scorer that cannot rank-track our own known ladder is not trusted to anchor anyone else's code.

Property-based tests use `hypothesis` (already present in the repo `.hypothesis/`), min 100 iterations each, tagged **Feature: poker-anchor-reproduction, Property {n}**. Property 3 (metric equivalence) is a single fixed-example self-check, not a 100-iteration property.

Kaggle submission is explicitly OUT OF SCOPE — no submission tasks appear below.

## Tasks

- [x] 1. Set up the `anchor_repro` package, dependencies, and data models (build order step 1 — tooling first)
  - [x] 1.1 Scaffold the `poker/anchor_repro/` package and record dependencies
    - Create `poker/anchor_repro/__init__.py` importing from `poker_collusion` (no metric/CV/Spearman re-implementation).
    - Add `polars` (pinned) to `poker/pyproject.toml` and write `poker/anchor_repro/requirements.lock` so the lowest-friction-notebook setup is repeatable; note `lightgbm`+`catboost` notebooks are out of scope for step 3.
    - _Requirements: 1.1, 1.6_

  - [x] 1.2 Implement the `ScoringRecipe` and `CanonicalScore` frozen data models
    - Define `ScoringRecipe(recipe_id, metric, primary_ap_component, secondary_component, split, unknown_handling, holdout_seed)` as a frozen dataclass with defaults `canonical_v1`, `reference_public_metric.score`, `pair_ap` (PRIMARY rank-tracked component), `combined` (SECONDARY diagnostic = full official combined score), `table_disjoint_holdout_40pct`, `pu_confirmed_only`, `Seeds.cv_split`.
    - Define `CanonicalScore(pair_ap, combined, components)` as a frozen dataclass holding both local scores plus the full `ScoreComponents` breakdown from a single scoring pass; `pair_ap` is the PRIMARY gate quantity and the anchor's `local_holdout_ap`, `combined` is the SECONDARY diagnostic.
    - Add serialization to a provenance dict so the recipe (naming primary + secondary components) can be embedded in every anchor.
    - _Requirements: 2.5, 2.6_

  - [x] 1.3 Implement the filename→real-LB parser
    - Parse `submission_best_<d>.csv` → `real_lb = int(d)/1e5`; expose formatting back to `<d>` for round-trip.
    - _Requirements: 2.1, 4.1_

  - [x] 1.4 Write property test for the filename→real-LB parse round-trip
    - **Feature: poker-anchor-reproduction, Property 7: For any artifact filename `submission_best_<d>.csv`, the parsed `real_lb` SHALL satisfy `real_lb == int(d)/1e5` and re-formatting SHALL reproduce `<d>`.**
    - Generate random 5-digit scores with `hypothesis` (min 100 iterations); assert the round-trip.
    - **Validates: Requirements 2.1, 4.1**

  - [x] 1.5 Write unit test for the parser on the exact nine real filenames
    - Assert the nine known filenames parse to `0.18462 … 0.44519`.
    - _Requirements: 2.1_

- [x] 2. Implement the CanonicalScorer and its official-metric self-check
  - [x] 2.1 Implement `MetricSelfCheck.verify_metric_against_kernel`
    - Load the fixed `(solution, submission, expected_score)` example from `data/poker/_metric_kernel/slash-poker-competition-metric.ipynb` (or construct a small fixed pair and extract the kernel's own `score()` cell); assert `reference_public_metric.score(...)` equals the kernel score to full float tolerance; FAIL is a hard release blocker.
    - _Requirements: 1.2, 1.4_

  - [x] 2.2 Implement `CanonicalScorer.score_dev_predictions`
    - Reuse `build_fold_solution` (`validation/cv.py`) and `score_components` (`metric/production_metric.py`) so PU unknown-handling (unknowns never materialized as negatives) is inherited; return a `CanonicalScore` carrying BOTH `pair_ap` (the metric's 70%-weighted PairAP intermediate, PRIMARY gate) and `combined` (the full official combined score, SECONDARY diagnostic) plus the full `ScoreComponents`, all from a SINGLE `reference_public_metric.score` evaluation (never two divergent recomputations/splits); `CanonicalScore.pair_ap` is the anchor's `local_holdout_ap` and `combined` is recorded alongside it.
    - Refuse to run unless `MetricSelfCheck` passed (wire in task 2.1's gate).
    - _Requirements: 1.2, 1.6, 2.5, 2.6_

  - [x] 2.3 Implement `CanonicalScorer.integrity_report` and the impossibility + coverage-mismatch guard
    - `integrity_report` checks a frozen EVAL artifact's hash, row count (112,540), pair-set-equals-sample, and non-degenerate ranking (mirrors `gate_a_verify_floor`); does NOT dev-score the eval artifact, and surfaces the disjoint-set sizes explicitly.
    - When a frame's pair-ids are eval (disjoint from dev labels), raise an explicit error naming the disjoint-set impossibility instead of returning a silent zero or fabricated number.
    - Belt-and-suspenders coverage-mismatch handling (lighter-priority "watch and crank" hardening): before scoring, pre-check the pair-id intersection between the prediction frame and the dev holdout AND also catch the official metric's own `ParticipantVisibleError` coverage-mismatch raise; in both paths the diagnostic names the disjoint-set sizes (how many pair-ids are missing and how many are extra) — never a silent zero or fabricated number.
    - _Requirements: 1.2, 2.4, 4.1_

  - [x] 2.4 Write property test for canonical scorer determinism
    - **Feature: poker-anchor-reproduction, Property 1: For any dev-prediction frame scored twice under the same `ScoringRecipe`, the `CanonicalScorer` SHALL return an identical `CanonicalScore` — identical `pair_ap`, identical `combined`, and identical `ScoreComponents`.**
    - Generate random valid dev-prediction frames with `hypothesis` (min 100 iterations); assert double-scoring returns identical `pair_ap`, `combined`, and `components`.
    - **Validates: Requirements 1.2, 2.5, 2.6**

  - [x] 2.5 Write property test for the impossibility + coverage-mismatch guard
    - **Feature: poker-anchor-reproduction, Property 5: For any eval artifact and dev-label set whose pair-id sets are not equal (empty intersection, or any missing/extra pair), the scorer SHALL NOT return an absolute LB-score reproduction: it SHALL either pre-check the pair-id intersection and refuse, or surface the metric's own `ParticipantVisibleError` coverage-mismatch raise — never a silent zero or fabricated number — and in both paths the diagnostic SHALL name the disjoint-set sizes.**
    - Generate disjoint / missing / extra eval-vs-dev pair sets with `hypothesis` (min 100 iterations); exercise BOTH the pre-check path and the metric-raise path; assert no absolute-match path exists, the scorer errors explicitly, and the diagnostic names the disjoint-set sizes (missing and extra counts).
    - **Validates: Requirements 2.4, 4.1**

  - [x] 2.6 Write the official-metric equivalence self-check (fixed example, not a 100-iteration property)
    - **Feature: poker-anchor-reproduction, Property 3: For any fixed (solution, submission) input from the metric-kernel example, the local `reference_public_metric.score` SHALL equal the kernel's own score to full float tolerance; if not, the scorer SHALL refuse to run.**
    - Single fixed-example equality assertion (integration self-check), executed once.
    - **Validates: Requirements 1.2, 1.4**

  - [x] 2.7 Write unit test for `integrity_report` on a real artifact copy
    - Use a copy of a real artifact (never the artifact itself); assert row count 112,540, pair-set equals sample, non-degenerate ranking (matches `gate_a_verify_floor`).
    - _Requirements: 1.2_

- [x] 3. Checkpoint - tooling + canonical scorer + self-check complete
  - Ensure all tests pass, ask the user if questions arise. Do NOT proceed to the ladder gate until the metric self-check passes.

- [x] 4. Pre-register the rank-tracking bar (before any ladder scoring — contract Rule 2)
  - [x] 4.1 Write the pre-registered gate to the dossier BEFORE scoring the ladder
    - Append to `RESEARCH_DOSSIER.md`: held-out judge = real LB order of the nine artifacts; statistic = Spearman via `lb_cv.spearman_corr`; PRIMARY metric (the gate) = PairAP (`local_pair_ap`) — fixed before any ladder is scored because the official metric is 70% PairAP; SECONDARY diagnostic = the full combined official score (`local_combined`), reported every run regardless of whether it agrees with the primary; contract-minimum bar `spearman > 0.0` strictly per metric; RECOMMENDED operational bar `spearman >= 0.7` on the PRIMARY (PairAP), also computed/reported for the SECONDARY combined; the explicit n=9 significance/CI caveat (a near-zero Spearman over nine points is indistinguishable from noise); and that a PairAP-vs-combined disagreement is a first-class finding, never cherry-picked. Record both metrics, both bars, and the uncertainty before any result exists.
    - _Requirements: 2.2, 4.2_

- [x] 5. Implement the LadderValidator and run the external-judge gate (build order step 2)
  - [x] 5.1 Implement `LadderPoint` / `MetricRankTracking` / `RankTrackingVerdict` data models and `LadderValidator.score_ladder`
    - Define `LadderPoint(config_id, real_lb, local_pair_ap, local_combined, recipe_id, source)` carrying BOTH the primary PairAP and secondary combined local scores per artifact; define `MetricRankTracking(metric_name, spearman, threshold, passed, verdict)` for ONE metric; define `RankTrackingVerdict(n_points, ci_note, primary, secondary, agree, disagreement_note, passed, verdict, message)` bundling the primary (PairAP) + secondary (combined) `MetricRankTracking` results.
    - Score ALL nine known-LB artifacts under the canonical scorer via `CanonicalScorer`; from each artifact's single `CanonicalScore` record both `local_pair_ap` and `local_combined` into the `LadderPoint`, recording per artifact which recipe/split was used and which artifacts are scorable vs BLOCKED (recipe unavailable), reported honestly.
    - _Requirements: 2.1, 2.5, 2.6_

  - [x] 5.2 Implement `LadderValidator.rank_tracking` gate
    - Compute Spearman for BOTH metrics (reuse `lb_cv.spearman_corr`): `primary = spearman_corr(local_pair_ap, real_lb)`, `secondary = spearman_corr(local_combined, real_lb)`; for EACH metric set `passed = spearman >= 0.7 AND spearman > 0.0` and populate its `MetricRankTracking.verdict` in {`TRUSTED`, `WEAK`, `NULL`}.
    - Set the overall `RankTrackingVerdict.passed`/`verdict` equal to the PRIMARY (PairAP) result (the pre-registered gate; the secondary never overrides it); populate `agree` (do primary and secondary land on the same pass/verdict tier?) and a non-empty `disagreement_note` when they disagree, recording it as a first-class dossier finding — not reconciled or cherry-picked; populate the n=9 `ci_note`. Never claim absolute-score reproduction (only relative rank-tracking).
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 5.3 Write property test for rank-tracking gate monotonicity (both metrics)
    - **Feature: poker-anchor-reproduction, Property 4: For any ladder of `(real_lb, local_pair_ap, local_combined)` points, EACH per-metric `MetricRankTracking.passed` flag (primary PairAP and secondary combined) SHALL be monotone in that metric's local-ordering agreement with the real-LB ordering; a rank-improving perturbation never decreases that metric's Spearman, a metric is TRUSTED only when `spearman >= threshold` AND `spearman > 0.0`, and the overall `RankTrackingVerdict.passed`/`verdict` SHALL equal the PRIMARY (PairAP) result.**
    - Generate random ladders (both metrics) with `hypothesis` (min 100 iterations); assert per-metric Spearman non-decreasing under rank-improving perturbations, the per-metric pass-flag rule, and that overall verdict mirrors the primary.
    - **Validates: Requirements 2.2, 2.3**

  - [x] 5.4 Write property test for scorer–recipe consistency across artifacts
    - **Feature: poker-anchor-reproduction, Property 2: For any set of artifacts scored to produce anchors, every resulting anchor SHALL record the SAME `recipe_id`; scoring the same dev-prediction frame under two anchors' recorded recipes SHALL yield the same local holdout AP (no cross-regime mixing).**
    - Generate random anchor batches with `hypothesis` (min 100 iterations); assert single `recipe_id` and equal AP under recorded recipes.
    - **Validates: Requirements 2.5, 2.6, 3.5**

  - [x] 5.5 Write integration smoke test for ladder scoring end-to-end
    - One representative run over the real dev holdout via `CVHarness` for whichever artifacts have a re-runnable recipe; record scorable vs BLOCKED honestly. Not property-tested.
    - _Requirements: 2.1, 2.5_

  - [x] 5.7 Write property test for dual-metric verdict always reported, never cherry-picked
    - **Feature: poker-anchor-reproduction, Property 8: For any scored ladder, the `RankTrackingVerdict` SHALL contain BOTH the primary (PairAP) and secondary (combined) `MetricRankTracking` results; the overall `passed`/`verdict` SHALL equal the primary (PairAP) result independent of the secondary; and whenever the two metrics disagree on their pass flag the verdict SHALL set `agree=False` and record a non-empty `disagreement_note`.**
    - Generate ladders with `hypothesis` (min 100 iterations) where PairAP and combined either agree or conflict; assert both `MetricRankTracking` results are present, the overall verdict equals the primary regardless of the secondary, and a disagreement sets `agree=False` with a non-empty `disagreement_note`.
    - **Validates: Requirements 2.2, 2.3, 4.1, 4.2**

  - [x] 5.8 Build the ladder recipe-runner adapter and re-run every reconstructable ladder recipe (debug the n=0 null - contract Rules 3, 11)
    - CONTEXT: task 5.5 measured a 0/9-scorable ladder because `LadderValidator` has no injected `recipe_runner`, NOT because the ladder is unscorable. Investigation confirmed the null is an EXECUTION GAP, not a real null: the `submission_best_044519.csv` recipe is fully re-runnable from on-disk caches (`feature_cache/v5/{dev,eval}_v5.parquet` 25,860 rows; `v7/_MFg_{dev,eval}.npy` 25,860x10; `v7/_DIRc_{dev,eval}.npy` 25,860x8) - the `cand_exp4c_directional` floor stack in `tuning_harness/layers/registry.py`, re-run to `holdout_ap=0.3839` in dossier section 37. This task closes that gap (contract Rule 11).
    - Implement a `RecipeRunner` adapter (e.g. `anchor_repro/recipe_registry.py`) that, per artifact, REUSES the existing feature caches + `tuning_harness/layers/registry.py` recipe definitions to re-run the recipe on the dev holdout via `CVHarness`/`build_fold_solution` and emit a dev-holdout prediction frame the `CanonicalScorer` scores - NEVER dev-scoring the disjoint eval CSV.
    - For `submission_best_044519.csv`: reconstruct the floor stack (foundation v5 + board_equity MFg + directional DIRc, RETRAIN-composed exactly as `loopv.py` hstacks `[Xd5, Mgd, DIRd]` and fits one model), fit on the dev holdout, emit predictions -> a SCORABLE point. Assert the recovered dev-holdout PairAP is consistent with the section 37/33 recorded `holdout_ap ~= 0.3833-0.3839` (a re-run sanity check, not a new claim).
    - For the other eight artifacts (0.18462...0.41289): attempt reconstruction from the versioned caches (v2/v3/v4/v6) + the candidate definitions in the dossier/experiment scripts. Mark an artifact BLOCKED ONLY where its exact recipe genuinely cannot be reconstructed, with the specific missing cache/config named (an honest, per-artifact BLOCKED - never a blanket give-up). Report the scorable-vs-BLOCKED split with reasons.
    - Adversarial-validation note (contract Rule 5): a re-run's dev-holdout AP is a MEASURED local number, not an LB claim; it only feeds the RELATIVE rank-tracking gate.
    - Add a unit/integration test asserting >=1 SCORABLE point (044519) with a finite dev-holdout PairAP, and that any BLOCKED artifact names its missing recipe input.
    - _Requirements: 2.1, 2.5, 2.6_

  - [x] 5.6 Record the ladder verdict to the dossier (measured result + verdict)
    - Run `score_ladder` WITH the task-5.8 recipe-runner adapter, then `rank_tracking`, and append the recovered `(real_lb, local_pair_ap, local_combined)` points; the measured Spearman for BOTH metrics (primary PairAP and secondary combined) over the ACTUAL number of scorable points recovered; the TRUSTED/WEAK/NULL verdict for each; and any PairAP-vs-combined disagreement (`agree=False` + `disagreement_note`) as a first-class finding - record BOTH metric results, never cherry-pick whichever looks better. Label each number MEASURED (local AP/combined, Spearman) vs EXTERNALLY VERIFIED (real LB). State the exact n (scorable points) and its CI caveat.
    - If, after task 5.8, fewer than 2 points are scorable, record that as an honest, INVESTIGATED null naming which artifacts were BLOCKED and why (recipe genuinely unreconstructable) - distinct from the earlier un-wired n=0. The overall verdict mirrors the PRIMARY (PairAP); a PRIMARY NULL is a first-class deliverable and blocks trusting the scorer for anchoring.
    - _Requirements: 2.3, 4.1, 4.2_

- [x] 6. Checkpoint - external-judge gate evaluated
  - Ensure all tests pass, ask the user if questions arise. Only proceed to step 3 with the gate verdict recorded; if NULL/WEAK, step 3 still runs but its anchor is flagged `scorer_trusted=false`.

- [x] 7. Implement the sandboxed reproduction runner (build order step 3 — untrusted-code execution)
  - [x] 7.1 Implement `SandboxedReproRunner.read_as_untrusted`
    - Parse notebook cells WITHOUT executing; extract declared deps and every data-path read; flag any network/subprocess/credential-env/filesystem-escape calls for review. Treat ALL notebook content as untrusted regardless of source (no trusted-source exemption).
    - _Requirements: 1.5, 3.1_

  - [x] 7.2 Implement `SandboxedReproRunner.run` with sandbox controls and status handling
    - Execute behind: (a) an explicit path shim mapping the notebook's data-root discovery to `poker/data/poker` (reads outside fail closed); (b) network egress disabled; (c) credential-bearing env vars stripped; (d) a working-dir jail + wall-clock timeout. Capture the emitted `submission.csv` on success and validate it against the official submission contract.
    - Implement `ReproResult` status handling: `RUNNING` → `SUCCEEDED` or `BLOCKED` with `failure_stage` in {`deps`,`data_path`,`compute`,`runtime`} and exact detail; status MAY sit at `RUNNING` while a failure is diagnosed and flip to `BLOCKED` only once the cause is pinned. Never report partial success.
    - _Requirements: 1.5, 3.1, 3.3_

  - [x] 7.3 Write integration smoke test for the sandboxed repro runner
    - One run of the 0.64469 notebook; assert either `SUCCEEDED` with a schema-valid `submission.csv` or `BLOCKED` with a recorded stage/reason. Not property-tested (executes real, untrusted, side-effecting code).
    - _Requirements: 1.5, 3.1, 3.3_

- [x] 8. Implement the append-only AnchorStore and record the 0.64469 anchor
  - [x] 8.1 Implement the `AnchorStore` (append-only, `lb_anchors.json` schema)
    - Load/append against the existing `.kiro/specs/poker-collusion-detection/lb_anchors.json` schema (preserve parent fields; add `recipe_id` inside `source`/provenance). Refuse to edit/delete an existing anchor; refuse to persist `verified=true` without a real external LB; atomic writes (temp file + rename).
    - _Requirements: 3.5, 4.2_

  - [x] 8.2 Write property test for append-only anchor-store integrity
    - **Feature: poker-anchor-reproduction, Property 6: For any sequence of `append` operations, every previously stored anchor SHALL remain byte-unchanged, none SHALL be deleted, and an anchor SHALL NOT be persisted with `verified=true` unless backed by a real external LB value.**
    - Generate random append sequences with `hypothesis` (min 100 iterations); assert prior anchors unchanged, none deleted, no unverified `verified=true`.
    - **Validates: Requirements 3.5, 4.2**

  - [x] 8.3 Write unit test for anchor-store schema compatibility
    - Assert an appended anchor is schema-compatible with the existing `lb_anchors.json` so the parent `poker-layered-tuning` harness can consume it.
    - _Requirements: 3.5_

  - [x] 8.4 Score the reproduced submission and record the anchor
    - If step 7 `SUCCEEDED`, score the captured `submission.csv` under the canonical scorer (same recipe as step 2) and record the `(config_id, real_lb=0.64469, local_holdout_ap)` anchor via `AnchorStore` with provenance noting reproduced-competitor origin, `recipe_id`, notebook sha256, and the ladder gate verdict; set `verified=true` only if backed by the external LB.
    - If step 7 is `BLOCKED`, record reproduction as BLOCKED with the exact stage/reason; do not fabricate an anchor.
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 8.5 Record the new-point consistency check to the dossier
    - Report whether the reproduced 0.64469 local AP is consistent with the step-2 rank-tracking (does a real 0.64469 sit above our real 0.44519 on the local metric?) — a consistency check of the validator on a NEW real point. Label MEASURED vs EXTERNALLY VERIFIED.
    - _Requirements: 3.4, 4.1_

- [x] 9. Honest reporting, audit trail, and layered-tuning boundary
  - [x] 9.1 Append the pre-registered bars, measured results, and verdicts to `RESEARCH_DOSSIER.md`
    - Consolidate: every produced number labelled MEASURED (local PairAP, local combined, both Spearmans), EXTERNALLY VERIFIED (real LB from filename), or ASSUMED/PROJECTED; record BOTH the primary (PairAP) and secondary (combined) rank-tracking results and any PairAP-vs-combined disagreement verbatim as a first-class finding (no cherry-picking); append-only, externally auditable.
    - _Requirements: 4.1, 4.2_

  - [x] 9.2 Record the layered-tuning recommendation without auto-modifying it
    - Record whether the new multi-point anchor set supports re-fitting the `poker-layered-tuning` projection OR whether the rank-tracking null means the projection stays disabled. MAY emit a notification that an update is available, but the notification itself performs NO modification (implementation discipline, not an active block). Record competitor-revealed signals as hypotheses to test, not verified improvements.
    - _Requirements: 4.3, 4.4, 4.5, 4.6_

- [x] 10. Final checkpoint - ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (tests) and can be skipped for a faster path, but the property tests directly validate the eight Correctness Properties and are recommended.
- Each task references specific requirement clauses and/or the property it satisfies for traceability.
- The mandated build order is enforced: tooling + scorer + self-check (tasks 1–3) → pre-register bar + ladder gate (tasks 4–6) → sandboxed reproduction + anchor (tasks 7–8). Task 8.4 (reproduction anchor) is downstream of the task 5–6 gate.
- The official metric, CV split, and Spearman are REUSED from `poker_collusion`, never re-implemented (Req 1.2, 1.6).
- Property 3 is a fixed-example self-check (task 2.6), not a 100-iteration property; all other property tests use `hypothesis` at min 100 iterations.
- No Kaggle submission tasks — submission is out of scope.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "1.5", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "4.1"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2", "5.4", "5.5"] },
    { "id": 6, "tasks": ["5.3", "5.7", "5.8", "7.1"] },
    { "id": 7, "tasks": ["5.6", "7.2", "8.1"] },
    { "id": 8, "tasks": ["7.3", "8.2", "8.3"] },
    { "id": 9, "tasks": ["8.4"] },
    { "id": 10, "tasks": ["8.5", "9.1"] },
    { "id": 11, "tasks": ["9.2"] }
  ]
}
```
