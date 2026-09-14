# Implementation Plan: Poker Layered Tuning

## Overview

This plan implements the layered detection + LB-calibrated tuning harness as a thin orchestration layer over the existing `poker/poker_collusion` package. Nothing here rebuilds the metric, CV harness, feature caches, PU ranker, or submission writer — those are reused as the substrate and wrapped as harness components.

The task order is dictated by **Requirement 8 / Gate A**: the first implementation task produces a submission-ready candidate and obtains a REAL leaderboard score (an external checkpoint) BEFORE any elaboration task begins. No task builds a second layer on an unverified base. The sequence is: (1) early external checkpoint, then (2) harness components (registry, composer, drift gate, calibrator, holdout scorer, shrinkage), then (3) tuning + ablation, then (4) honest projection + submission gate + stop-loss state machine.

The 23 Correctness Properties from the design are implemented as Hypothesis property-based tests (>=100 iterations each, one test per property, tag `# Feature: poker-layered-tuning, Property {number}: {property_text}`). PBT is applicable because the calibrator, composer, shrinkage, ablation, projection, and submission gate are pure functions with universal invariants.

**Conventions used below**
- New harness code lives under `poker/poker_collusion/tuning_harness/` mirroring the design's module names (`layers/`, `composition/`, `gate/`, `holdout/`, `ablation/`, `tuning/`, `shrinkage/`, `projection/`, `submission/`).
- The LB_Anchor store is an append-only JSON/CSV alongside `poker-collusion-detection/RESEARCH_DOSSIER.md`.
- Per Gate D, every experiment/submission task carries a pre-registered success bar and an experiment/time budget; on budget overrun without clearing the bar the task escalates to the user for a keep/kill decision.
- Per Req 8.7, every completed task appends its pre-registered bar, held-out result, gate verdict, and keep/revert decision to `RESEARCH_DOSSIER.md`.

## Tasks

- [x] 1. GATE A — Early external checkpoint: submit the known-good floor and obtain a REAL leaderboard score before any elaboration
  - Reuse `submission/writer.py` and `metric/reference_public_metric.py` to validate and emit a submission CSV for the LB-verified floor stack (foundation v5 + board_equity MFg + directional DIRc), reusing the existing feature caches — do NOT recompute features
  - Confirm the emitted CSV byte-matches / re-scores to the existing `submission_best_044519.csv` floor artifact before proposing it (Assumption A7 falsifying test: hash-check + re-score against the reference metric on the dev holdout)
  - Produce the single submission-ready candidate, obtain its real leaderboard score, and record it as the first `LB_Anchor` tuple (config_id, real_lb, measured_drift, holdout_ap)
  - Pre-registered bar (Gate D): the real LB score reproduces the known floor 0.44519 (within submission rounding); budget: 1 submission. If it does not reproduce, HALT — the floor artifact is compromised (Gate C / A7) — and escalate to the user
  - Append the pre-registered bar, the real LB result, gate verdict, and keep/revert decision to `RESEARCH_DOSSIER.md` (Req 8.7)
  - _Requirements: 8.1, 8.4, 8.7, 7.4_

- [x] 2. Checkpoint — Do not proceed past Gate A until the real LB score has returned and confirms the floor
  - Ensure all tests pass, ask the user if questions arise.
  - Confirm the first LB_Anchor is recorded and the floor is verified; no elaboration task (3+) may start otherwise (Gate A hard constraint)
  - _Requirements: 8.1_

- [x] 3. Establish harness scaffolding, data models, and the LB_Anchor store
  - [x] 3.1 Create the harness package skeleton and shared data models
    - Create `tuning_harness/` package importing the existing `poker_collusion` package (reuse, not rebuild)
    - Define `CandidateConfig`, `CompositionSpec`, `Candidate`, `SubmissionReport`, `RankVector`, `FeatureFrame` views over the existing feature caches keyed by `pair_id`
    - Implement the append-only `LB_Anchor` store (JSON/CSV) alongside `RESEARCH_DOSSIER.md`, seeded with the anchor recorded in task 1
    - _Requirements: 7.1, 8.7_

- [x] 4. Layer registry with the known-good floor configuration
  - [x] 4.1 Implement `layers/registry.py`
    - Define the frozen `Layer` dataclass (name, kind monotone/conditional, feature_keys into existing caches, optional flag); `FOUNDATION` is not optional
    - Implement `known_good_floor_config()` returning the exact LB-verified stack: foundation v5 + board_equity (MFg) + directional (DIRc) with default knobs (the 0.44519 floor)
    - Register optional layers as toggleable blocks (board_equity, directional, iso, whipsaw, negative_space) referencing existing feature-cache keys — do NOT recompute features
    - _Requirements: 2.1, 2.3_

- [x] 5. Composer with hard floor guarantee
  - [x] 5.1 Implement `composition/composer.py`
    - Implement `rank_blend(base, layer, weight)` as a convex combination of ranks; `weight == 0.0` returns the base ranking exactly (order-identical)
    - Implement `retrain(base_features, layer_features, model_cfg)` refitting the reused baseline model on base+layer features (result trusted only after the combined set clears the Drift_Gate — wired in task 9)
    - Implement `compose(cfg)`: all optional layers OFF reproduces `known_good_floor_config()` exactly; report layers-ON best-effort; wrap reporting so any exception yields `layers_reported = None` and never blocks assembly
    - Implement corrupting-layer detection: a layer that reduces holdout AP at every positive weight in the tuning grid is flagged CORRUPTING (not merely NULL)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1_

  - [x] 5.2 Write property test for RANK_BLEND weight-zero identity
    - **Property 7: RANK_BLEND at weight zero equals the base ranking exactly**
    - **Validates: Requirements 2.2** (Assumption A5 falsifying test)

  - [x] 5.3 Write property test for all-OFF reproducing the floor
    - **Property 6: All optional layers OFF reproduces the floor exactly**
    - **Validates: Requirements 2.1** (Assumption A4 falsifying test)

  - [x] 5.4 Write property test for corrupting-layer flagging
    - **Property 8: A layer that hurts at every positive weight is flagged CORRUPTING**
    - **Validates: Requirements 2.5**

  - [x] 5.5 Write unit test for best-effort layer reporting and reporting-failure edge
    - Reporting succeeds: `layers_reported` equals layers-on (Req 2.3)
    - Injected reporting failure: valid `Candidate` returned, `layers_reported == None`, assembly not blocked (Req 2.4)
    - _Requirements: 2.3, 2.4_

- [x] 6. Drift_Gate (adversarial dev-vs-eval AUC)
  - [x] 6.1 Implement `gate/drift_gate.py`
    - Implement `adversarial_auc(feature_frame)`: train a dev-vs-eval classifier (target = is_eval_row) via grouped CV, return the dev-vs-eval AUC (high AUC = describes "which pool", i.e. drift)
    - Implement `evaluate_drift(feature_frame, calib)` returning `DriftResult` (auc, verdict, threshold, threshold_kind); classify with `calib.threshold` when TRUSTED, else PROVISIONAL
    - _Requirements: 1.1_

- [x] 7. Calibrator (Clean_Separation_Check + LB_Calibrated_Threshold + null surfacing)
  - [x] 7.1 Implement `gate/calibration.py`
    - Implement `clean_separation_check(anchors, floor_lb)`: PASS anchors = real_lb >= floor_lb, FAIL anchors = regressed; `separable` iff `max_pass_drift < min_fail_drift` strictly
    - Implement `calibrate(anchors, floor_lb)`: if not separable -> gate INVALID, `trusted = False`, no trusted threshold, `null_surfaced = True`; if separable AND >=3 anchors -> `LB_CALIBRATED` threshold placed in the open gap `(max_pass_drift, min_fail_drift)` (midpoint); else `PROVISIONAL` (~0.65) with reduced-confidence flag
    - Ensure the calibrator never raises on insufficient/non-separable anchors — it degrades to reduced-confidence
    - _Requirements: 1.1, 1.4, 1.5, 1.6, 1.7_

  - [x] 7.2 Write property test for calibrated-threshold derivation
    - **Property 1: Calibrated threshold is derived from anchors, never the fixed constant**
    - **Validates: Requirements 1.1**

  - [x] 7.3 Write property test for anchor re-classification
    - **Property 2: The calibrated threshold correctly re-classifies its own anchors**
    - **Validates: Requirements 1.2, 1.3**

  - [x] 7.4 Write property test for the separation predicate
    - **Property 3: Clean_Separation_Check equals the strict-order predicate**
    - **Validates: Requirements 1.4** (Assumption A1 falsifying test)

  - [x] 7.5 Write property test for overlap invalidation and null surfacing
    - **Property 4: Overlapping anchor ranges invalidate the gate and surface the null**
    - **Validates: Requirements 1.5**

  - [x] 7.6 Write property test for the trust branch
    - **Property 5: Trust is granted only when separable and sufficiently anchored**
    - **Validates: Requirements 1.6, 1.7**

  - [x] 7.7 Write unit test for the motivating bug (Req 1.8)
    - Concrete anchors 0.639 PASS / 0.644 PASS / 0.865 FAIL; assert foundation drift 0.679 classifies PASS under the calibrated gap between 0.644 and 0.865 (the exact regression the fixed 0.65 caused)
    - _Requirements: 1.2, 1.3, 1.8_

- [x] 8. Holdout scorer (table-disjoint PU-stress AP)
  - [x] 8.1 Implement `holdout/scorer.py`
    - Wrap the existing group-by-`table_id` CV (`validation/cv.py`) and `reference_public_metric.py` to produce PU-stress AP on the 40% table-disjoint, pair-disjoint split — reuse, do not reimplement the split or metric
    - _Requirements: 3.1, 5.4_

- [x] 9. Wire the gate sequence: Drift PASS first, then holdout is trustworthy
  - [x] 9.1 Compose the drift -> holdout sequence and mark RETRAIN trust
    - Sequence: Drift_Gate on the candidate feature set first; only if PASS is the holdout AP treated as a trustworthy LB predictor (Rule 17); a drift FAIL disqualifies the candidate and inverts trust in its holdout
    - Mark a RETRAIN composition trusted only if the combined (base + layer) feature set clears the Drift_Gate; a RETRAIN whose combined features FAIL is never trusted
    - _Requirements: 4.2_

  - [x] 9.2 Write property test for RETRAIN trust gating
    - **Property 12: A RETRAIN result is trusted only if its combined feature set passes the Drift_Gate**
    - **Validates: Requirements 4.2**

- [x] 10. Checkpoint — Ensure gate + composer + calibrator tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Append the harness-foundation verification (properties 1-8, 12) result to `RESEARCH_DOSSIER.md` (Req 8.7)
  - _Requirements: 8.7_

- [x] 11. Sample-size shrinkage for per-pair statistics
  - [x] 11.1 Implement `shrinkage/eb.py`
    - Implement `shrink(value, n, k, prior)` = `prior + (value - prior) * n / (n + k)`; at `n=0` returns prior, monotone in `n`, approaches value as `n` grows large (`n` = shared-hand count)
    - Implement `sweep_shrinkage(layer, k_grid)`: evaluate EVERY constant in the grid (never stop early), choose one whose top-K-by-score median shared-hand count is not below the population median
    - _Requirements: 6.1, 6.2_

  - [x] 11.2 Write property test for shrinkage monotonicity
    - **Property 17: Sample-size shrinkage is monotone in shared-hand count**
    - **Validates: Requirements 6.1** (generators cover n=0 boundary; Assumption A6)

  - [x] 11.3 Write property test for the exhaustive verified sweep
    - **Property 18: The shrinkage sweep is exhaustive and its choice is verified**
    - **Validates: Requirements 6.2**

- [x] 12. Ablation (leave-one-out)
  - [x] 12.1 Implement `ablation/leave_one_out.py`
    - Implement `ablate(cfg)`: for each ON layer report `full_holdout - holdout_without_layer`; recommend removal when contribution <= 0; return the leanest subset within a stated tolerance of the best holdout
    - Reuse the holdout scorer from task 8; no new metric or split
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 12.2 Write property test for leave-one-out contribution
    - **Property 9: Ablation reports each layer's leave-one-out contribution**
    - **Validates: Requirements 3.1**

  - [x] 12.3 Write property test for the removal recommendation
    - **Property 10: Non-positive contribution triggers a removal recommendation**
    - **Validates: Requirements 3.2**

  - [x] 12.4 Write property test for leanest-subset selection
    - **Property 11: The chosen subset is the leanest within tolerance of the best**
    - **Validates: Requirements 3.3**

- [x] 13. Two-level tuner (drift -> baseline -> layers)
  - [x] 13.1 Implement `tuning/tuner.py`
    - Implement `tune(cfg, calib)` invoking stages in exactly the order: resolve feature-set drift, then Baseline_Tuning, then Layer_Tuning; coarse structural grids only (RANK_BLEND weight grid includes 0.0)
    - Enforce two independent guarantees: (a) the selected config clears the Drift_Gate, AND (b) any tuned gain that increases measured drift past the threshold is rejected even when holdout AP improved; (b) holds independently of (a)
    - Record a layer as conditional/RETRAIN iff RANK_BLEND reduces holdout at every positive weight AND RETRAIN increases holdout; otherwise no conditional classification is mandated
    - Report the `LB_Projection` with uncertainty, distinguishing measured holdout from projected LB
    - Pre-registered bar (Gate D): a tuned config is accepted only if it clears the Drift_Gate and its projection does not regress the floor; budget: coarse grids only, escalate to user on overrun
    - _Requirements: 4.1, 4.3, 5.1, 5.2, 5.3, 5.4_

  - [x] 13.2 Write property test for tuning stage order
    - **Property 14: Tuning executes drift, then baseline, then layers, in that order**
    - **Validates: Requirements 5.1**

  - [x] 13.3 Write property test for the two independent guarantees
    - **Property 15: The two tuning guarantees hold independently**
    - **Validates: Requirements 5.3**

  - [x] 13.4 Write property test for conditional/RETRAIN classification
    - **Property 13: Conditional/RETRAIN classification holds exactly when both conditions hold**
    - **Validates: Requirements 4.3**

  - [x] 13.5 Write property test for measured-vs-projected reporting in tuning
    - **Property 16: Reports distinguish measured holdout from projected leaderboard**
    - **Validates: Requirements 5.4**

- [x] 14. Checkpoint — Ensure tuning, ablation, shrinkage tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Append the tuning/ablation/shrinkage verification (properties 9-11, 13-18) result to `RESEARCH_DOSSIER.md` (Req 8.7)
  - _Requirements: 8.7_

- [x] 15. LB_Projection with enforced fit range (no extrapolation)
  - [x] 15.1 Implement `projection/project.py`
    - Implement `ProjectionFit` (slope ~1.4235, intercept ~-0.1414, resid_std ~0.0022, fit_min_ap, fit_max_ap, n_points) fit on the LB_Anchors
    - Implement `project_lb(holdout_ap, fit)` returning (point, uncertainty, regime_verified); `regime_verified` is True only when `holdout_ap` is within `[fit_min_ap, fit_max_ap]`, else flagged unverified — never extrapolate
    - _Requirements: 5.4, 7.1_

  - [x] 15.2 Write property test for no-extrapolation flagging
    - **Property 23: Projections outside the fit range are flagged unverified (no extrapolation)**
    - **Validates: Requirements 5.4, 7.1** (enforces contract Rule 4; Assumption A3)

- [x] 16. Submission gate, anchor recording, and stop-loss HALT state machine
  - [x] 16.1 Implement `submission/gate.py`
    - Implement `submission_report(candidate)` emitting layers ON, holdout AP, measured drift, gate verdict, LB_Projection, and regime-verified flag; set `recommend_submit = False` if the Drift_Gate FAILs regardless of holdout AP
    - Implement `record_anchor_and_recalibrate(real_lb, candidate)`: append exactly one new LB_Anchor to the store and recalibrate gate + projection over the augmented set
    - Implement the guarded floor writer: `submission_best_044519.csv` is only overwritten through a write backed by a leaderboard-verified improvement; on refusal the artifact is left byte-for-byte unchanged and the refusal is logged to the dossier
    - Implement the submission-history state machine: two consecutive regressions vs the Known_Good_Floor emit a HALT signal blocking new candidates until the harness is re-examined (Gate C)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.3_

  - [x] 16.2 Write property test for report completeness
    - **Property 19: The submission report is complete**
    - **Validates: Requirements 7.1**

  - [x] 16.3 Write property test for drift-fail never recommended
    - **Property 20: A drift-failing candidate is never recommended for submission**
    - **Validates: Requirements 7.2, 8.3**

  - [x] 16.4 Write property test for anchor append + recalibration
    - **Property 21: Recording a real leaderboard result appends an anchor and recalibrates**
    - **Validates: Requirements 7.3**

  - [x] 16.5 Write property test for the guarded floor writer
    - **Property 22: The floor artifact is never overwritten without a verified improvement**
    - **Validates: Requirements 7.4**

  - [x] 16.6 Write unit test for the two-consecutive-regression HALT
    - Feed a submission history with two consecutive regressions vs the floor; assert the state machine emits HALT and blocks new candidates (Gate C)
    - _Requirements: 8.3_

- [x] 17. Integration tests on the real table-disjoint holdout and the floor-artifact guard
  - [x] 17.1 Write end-to-end integration test on the real holdout
    - Using `validation/cv.py` + `reference_public_metric.py`, compose the floor stack, score PU-stress AP, run the Drift_Gate, and confirm the projection lands within ~2*resid_std of the known floor LB (1-2 representative runs)
    - Pre-registered bar (Gate D): projection residual <= ~2*resid_std against the floor's real LB; budget: 1-2 runs, escalate on overrun (Assumption A3 falsifying test)
    - _Requirements: 5.4, 7.1_

  - [x] 17.2 Write floor-artifact guard integration test
    - Attempt a non-improving write against a temp-dir copy of the real `submission_best_044519.csv`; assert bytes unchanged
    - _Requirements: 7.4_

- [x] 18. Final checkpoint — Ensure all tests pass and the audit trail is complete
  - Ensure all tests pass, ask the user if questions arise.
  - Confirm every completed task's pre-registered bar, held-out result, gate verdict, and keep/revert decision is recorded in `RESEARCH_DOSSIER.md` (Req 8.7); confirm the accountability contract (Gate F) governed any conflicts
  - _Requirements: 8.7_

## Notes

- Tasks marked with `*` are optional test sub-tasks (property, unit, integration) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Every task references specific requirement sub-clauses for traceability; property test tasks additionally reference the design property number they validate.
- **Gate A is enforced by ordering**: task 1 obtains a real LB score and task 2 is a hard checkpoint; no elaboration task (3+) begins until the floor is externally verified.
- **Gate D** budgets and pre-registered bars are attached to the experiment/submission tasks (1, 13, 17); overrun escalates to the user for a keep/kill decision.
- **Gate C** stop-loss (two consecutive regressions -> HALT; drift-fail never submittable) is implemented in task 16 and covered by Property 20 + the HALT unit test.
- **Gate F**: the reverse-engineering-accountability contract is the binding authority; any conflict resolves in its favor.
- **Req 8.7** audit-trail appends are attached to the checkpoint tasks (10, 14, 18) and task 1, so the trail stays externally auditable.
- All 23 correctness properties map to exactly one property test; PBT runs a minimum of 100 iterations per property with the required tag format.
- The harness reuses the existing `poker_collusion` package throughout (metric, CV, feature caches, PU ranker, submission writer); no core component is rebuilt.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["3.1"] },
    { "id": 2, "tasks": ["4.1", "6.1", "8.1", "11.1", "15.1"] },
    { "id": 3, "tasks": ["5.1", "7.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "5.5", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "11.2", "11.3", "15.2", "9.1"] },
    { "id": 5, "tasks": ["9.2", "12.1"] },
    { "id": 6, "tasks": ["12.2", "12.3", "12.4", "13.1"] },
    { "id": 7, "tasks": ["13.2", "13.3", "13.4", "13.5", "16.1"] },
    { "id": 8, "tasks": ["16.2", "16.3", "16.4", "16.5", "16.6", "17.1", "17.2"] }
  ]
}
```
