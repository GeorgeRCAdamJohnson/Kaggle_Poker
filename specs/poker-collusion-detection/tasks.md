# Implementation Plan: Detect Suspicious Value Transfers in Poker

## Overview

This plan implements the phased, gated attack pipeline from the design. The order is
non-negotiable: **Phase −1 (Discovery) is a hard gate** — no feature engineering,
modeling, or submission tuning may begin until the metric contract is written and the
Discovery findings + exploit ledger are documented and reviewed in `RESEARCH_DOSSIER.md`.
**Phase 0** re-implements the three-part metric bit-for-bit and builds the PU-correct,
group-by-pool, family-stratified local CV harness. **Phase 1** ships an explainable
classical baseline end-to-end (the floor). **Phase 2** layers the PU ranking model and
behavior classifier. **Phase 3** adds a learned evidence ranker and joint tuning. The
**Final** phase produces the reproducibility manifest, seeds, five case reviews, and the
SDK-shadowing guard.

Language: **Python** (pandas/numpy, PyTorch stack, Hypothesis for property tests). Runs on
Windows PowerShell and in the Kaggle notebook; GCP available for training.

Each phase gate is encoded as a task or sub-task. The 16 correctness properties from the
design are implemented as Hypothesis property tests, each tagged
`# Feature: poker-collusion-detection, Property {n}: {property_text}`, and marked optional (`*`).
Core implementation sub-tasks are never optional.

Every task references the requirement(s) it satisfies (by number) and, where relevant, the
design property or section.

## Tasks

- [x] 1. Project scaffolding and shared foundations
  - Create the package layout: `poker_collusion/` with sub-packages `io/`, `features/`, `models/`, `evidence/`, `metric/`, `submission/`, `validation/`, `discovery/`, plus `tests/` and `configs/`
  - Add `pyproject.toml`/`requirements.txt` pinning exact versions (pandas, numpy, pyarrow, scikit-learn, torch, hypothesis, pytest); prefer pinned/exact versions
  - Define the core data models from the design as frozen dataclasses in `poker_collusion/types.py`: `Pair`, `HandSignal`, `PairFeatureSet`, `LabelTable`, `PairPrediction`, the `NOT_APPLICABLE` sentinel, and the behavior-label enum `{none, directed_transfer, soft_play, coordinated_isolation, other_coordination}`
  - Add a central seeded-config module (`poker_collusion/config.py`) holding paths, seeds, row-group/column selections, and threshold-version identifiers
  - Add a `SchemaError(file, missing_key)` exception type
  - _Requirements: 1.5, 5.4, 11.2_

---

## Phase −1: Discovery & Adversarial Investigation (HARD GATE)

> No modeling, feature engineering, or submission tuning begins until task 3 (the gate) is
> complete. Phase 0 may only start once the metric contract (task 2.2) is written.

- [x] 2. Discovery data access and forensics
  - [x] 2.1 Download competition data and verify schema/join keys
    - Write `poker_collusion/discovery/fetch_data.py` (or a documented PowerShell/Kaggle-CLI step) to download the competition files into the configured input directory
    - Assert presence of `players.parquet`, `hands.parquet`, `seats.parquet`, `actions.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, `sample_submission.csv`
    - Verify join keys: `hand_id` links gameplay tables, `player_id` links player tables, and `(hand_id, action_no)` ordering exists in `actions.parquet`; confirm `hands.phase` values; record `actions.parquet` row count (expect 18,609,028)
    - Write findings to the dossier; raise `SchemaError` on any missing file/column
    - _Requirements: 1.1, 1.2, 1.3, 1.6_

  - [x] 2.2 Workstream A — metric code forensics → write the metric contract
    - Obtain and read the actual public competition metric code line-by-line
    - Write `RESEARCH_DOSSIER.md` section "Metric Contract" enumerating: equal-`risk_score` tie-break (expected deterministic `pair_id`), treatment of unranked/missing pairs (implicit score), Evidence MAP@5 with fewer than 5 hands and the literal `NO_EVIDENCE` sentinel, confirmation a missed target pair contributes **zero** to Evidence MAP@5, how absent classes score in Behavior MAP and that `other_coordination` is excluded, and the ~30/70 public/private split stratification (by behavior family) and grouping
    - Contract must be precise enough for Phase 0 to match the public code bit-for-bit
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.6_ (Design: Phase −1 Workstream A; Property 15)

  - [x] 2.3 Workstream B — data-generation reverse engineering
    - Using only public data, log hypotheses on: distribution artifacts (value ranges, discretization), ID/timestamp/ordering regularities in `hand_id`/timestamps/action ordering/seat assignment, pool/table construction regularities (400 pools × 30 players × ~5,000 hands), statistical tells separating planted positives from disclosed confounders (tilt, weak play, similar strategies, repeated opponent selection, streaks, strategy changes), and any signature of the hidden `other_coordination` family
    - Confirm or refute each hypothesis against `development_labels.csv`; record results in the dossier "Data-Generation Findings" section
    - _Requirements: 2.1, 2.2, 2.3_ (Design: Phase −1 Workstream B)

  - [x] 2.4 Workstream C — label & evidence structure audit
    - From `development_evidence.csv`, derive per-disclosed-family (`directed_transfer`, `soft_play`, `coordinated_isolation`) behavior-specific action signatures
    - Confirm the invariant that latent scenario activation alone is never evidence; confirm each planted evidence hand contains both players AND a public behavior-specific action
    - Verify `evaluation_pairs.csv` excludes publicly labelled pair IDs and pairs containing a publicly labelled positive player; report violations
    - Record the concrete per-family action patterns the evidence validity gate must reproduce in the dossier "Label/Evidence Signatures" section
    - _Requirements: 2.4, 2.5, 8.4_ (Design: Phase −1 Workstream C)

  - [x] 2.5 Workstream D — adversarial / exploit hunting (rules-compliant, logged)
    - Log each candidate exploit as a hypothesis in the dossier "Exploit Ledger": risk-score calibration alone lifting Behavior MAP; liberal `other_coordination` prediction (free in Behavior MAP, checked against Pair-AP calibration cost); a dominant evidence-selection heuristic; coverage vs. ranking precision under the zero-for-missed-target rule; cross-component trades among Pair AP / Evidence MAP@5 / Behavior MAP
    - Mark each exploit CONFIRMED or REFUTED only after local-CV evidence exists (local CV lands in Phase 0; leave exploits as OPEN here with the test plan recorded)
    - Discard on sight any candidate requiring private information; record the rules-compliance guardrail
    - _Requirements: 12.1, 12.3_ (Design: Phase −1 Workstream D)

  - [x] 2.6 Workstream E — create and structure `RESEARCH_DOSSIER.md`
    - Create `.kiro/specs/poker-collusion-detection/RESEARCH_DOSSIER.md` with sections mirroring the workstreams: Metric Contract, Data-Generation Findings, Label/Evidence Signatures, Exploit Ledger, plus an Open-Questions log
    - Add a standing rule at the top: no modeling begins until Discovery findings are written and reviewed; every later phase must consult and update the dossier
    - _Requirements: 11.4, 11.5_ (Design: Phase −1 Workstream E)

- [x] 3. HARD GATE — Discovery review checkpoint
  - Confirm the metric contract (task 2.2) is written and complete; confirm data-generation findings, label/evidence signatures, and the initial exploit ledger are documented and reviewed in the dossier
  - Ensure all Discovery findings are written down (not just in someone's head); ask the user if questions arise
  - Do not proceed to any feature engineering or modeling until this gate is cleared; Phase 0 may start once the metric contract exists

---

## Phase 0: Scoring Harness + Local CV

- [x] 4. Reference and production metric re-implementation
  - [x] 4.1 Implement an independent, deliberately-simple reference metric
    - In `tests/reference_metric.py`, implement Pair AP, Evidence MAP@5 (missed target pair → zero), and Behavior MAP (macro one-vs-rest over the 3 disclosed families, excluding `other_coordination`) as straightforwardly as possible, to serve as the Property-15 model-based oracle
    - _Requirements: 10.1, 10.2, 10.3, 10.4_ (Design: Testing Strategy)

  - [x] 4.2 Implement the production metric matching the metric contract
    - In `poker_collusion/metric/three_part.py`, implement the three components to match the Phase −1 metric contract bit-for-bit: deterministic `pair_id` tie-break, implicit score for unranked/missing pairs, `NO_EVIDENCE` handling and fewer-than-5 evidence, zero for missed target pairs, exclusion of `other_coordination` from Behavior MAP
    - Expose separate component scores and a combined summary
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 12.2_

  - [x] 4.3 Property test — local metric matches reference (PU-correct)
    - `# Feature: poker-collusion-detection, Property 15: Local metric matches a reference implementation (PU-correct)`
    - Model-based Hypothesis test comparing `three_part.py` to `reference_metric.py`; generators must include missed-target-pair cases (contribute zero) and `other_coordination` (excluded from Behavior MAP)
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 4.4 Reconciliation test — bit-for-bit against the actual public metric code
    - Run `three_part.py` and the actual published competition metric code on identical batteries covering every metric-contract edge case (equal-`risk_score` tie-breaks, unranked/missing pairs, fewer-than-5 and `NO_EVIDENCE` evidence, missed target pairs, absent Behavior-MAP classes); assert exact equality on all three components; divergence is a release blocker
    - _Requirements: 10.6_ (Design: Property 15, Phase −1 Workstream A)

  - [x] 4.5 Metric-contract acceptance check
    - Assert `RESEARCH_DOSSIER.md` metric-contract section exists and that every documented edge case, tie-break, constant, epsilon, rounding rule, default, and the split-construction rule has a corresponding assertion in the reconciliation battery; the re-implementation may not diverge from the contract without a matching contract update
    - _Requirements: 10.6_ (Design: Testing Strategy)

- [x] 5. Local CV harness and diagnostics
  - [x] 5.1 Build the group-by-pool, family-stratified, PU-correct CV harness
    - In `poker_collusion/validation/cv.py`, construct held-out splits grouped by `pool` (no pool leaks across folds), stratified by behavior family, and PU-correct so Unknown_Pairs are never scored as negatives
    - Wire the three-part metric into per-fold scoring; report the three components separately plus a combined summary
    - _Requirements: 10.1, 10.4, 10.5_ (Design: Principle 1, group-by-pool CV)

  - [x] 5.2 Add bootstrap confidence intervals over pairs
    - Compute bootstrap CIs for all three components; expose a noise-band comparison helper (`is_improvement_beyond_band`) used as the phase-gate acceptance rule
    - _Requirements: 12.2_ (Design: Validation Discipline)

  - [x] 5.3 Add the LB-vs-CV correlation diagnostic and submission-selection rule
    - Record per-submission local CV and public LB; implement the selection rule: maximize combined three-component summary on CV, break near-ties toward the narrower CV band, never select on public LB alone, trust CV on divergence
    - _Requirements: 12.4_ (Design: Submission-selection rule)

- [x] 6. Phase 0 gate checkpoint
  - Verify bit-for-bit metric match (tasks 4.4, 4.5 pass) and that the CV harness produces stable CIs; confirm consistency with the dossier
  - Ensure all tests pass, ask the user if questions arise

---

## Phase 1: Classical Baseline End-to-End (the floor)

- [x] 7. Data_Loader
  - [x] 7.1 Implement chunked, column-projected, per-pool loading and joins
    - In `poker_collusion/io/data_loader.py`, implement `DataLoader` per the design interface: `load_players/hands/seats`, `iter_actions(pool_id, columns)` (chunked, column-projected, ordered by `(hand_id, action_no)`), `load_labels` (trusted-positive / confirmed-negative / unknown), `load_evaluation_pairs`, `load_sample_submission`, `shared_hands(pair)` partitioned dev/eval; attach `hands.phase`
    - Raise `SchemaError(file, missing_key)` on missing key/column; never load `actions.parquet` whole
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 7.2 Fixture and edge-case tests for loading/joining
    - Tiny constructed tables verifying joins, shared-hand dev/eval partitioning, and descriptive `SchemaError` on missing files/columns
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 8. EDA & Label Audit
  - [x] 8.1 Implement EDA/audit and versioned threshold artifact
    - In `poker_collusion/features/eda.py`, compute shared-hand-count distribution per pair, PU counts (trusted-positive / confirmed-negative / unknown), disclosed-family label distribution, evidence-validity confirmation (each dev-evidence hand is a shared hand with a behavior-specific action), and evaluation-pair leakage checks
    - Persist `eda_artifact.json` with calibrated thresholds and a threshold version; feature engineering may read only these constants
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 8.2 Fixture tests for audit reporting
    - Verify PU counts, family distribution, and evidence-validity reporting on constructed tables
    - _Requirements: 2.2, 2.3, 2.4_

- [x] 9. Hand-Level Signal Extraction
  - [x] 9.1 Implement hand-level signals (decision-time context only)
    - In `poker_collusion/features/signals.py`, compute per shared hand: signed `value_flow`, `aggression_asymmetry`, `isolation`, `mi_conflict`, and per-family `behavior_action_flag`, using only `pot_before, stack_before, to_call, players_active, amount, amount_to`
    - Classify `all_in` as a call when `amount <= to_call` else aggressive; assign the `NOT_APPLICABLE` sentinel on zero denominators (never raise)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 9.2 Property test — directed value-flow is antisymmetric
    - `# Feature: poker-collusion-detection, Property 1: Directed value-flow is antisymmetric`
    - _Requirements: 3.1_

  - [x] 9.3 Property test — all-in classification follows the amount/to_call rule
    - `# Feature: poker-collusion-detection, Property 2: All-in classification follows the amount/to_call rule`
    - _Requirements: 3.4_

  - [x] 9.4 Property test — zero-denominator signals yield the sentinel, never an error
    - `# Feature: poker-collusion-detection, Property 3: Zero-denominator signals yield the sentinel, never an error`
    - _Requirements: 3.5_

- [x] 10. Episodic Aggregation and Pair_Feature_Set
  - [x] 10.1 Implement episodic aggregation
    - In `poker_collusion/features/aggregation.py`, aggregate hand-level signals with peak-preserving stats (max, p90/p95, burst counts, means), a temporal-concentration feature, and count-robust normalization (empirical-Bayes shrinkage toward the pool prior)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 10.2 Property test — episodic concentration never lowers the coordination signal
    - `# Feature: poker-collusion-detection, Property 4: Episodic concentration never lowers the coordination signal`
    - _Requirements: 4.1, 4.2_

  - [x] 10.3 Assemble the deterministic, schema-versioned Pair_Feature_Set
    - In `poker_collusion/features/pair_features.py`, build `PairFeatureSet` including value-flow, aggression-asymmetry, isolation, episodic features, and confounder-separating features (tilt, weak play, similar strategies, repeated opponent selection, streaks, strategy changes); identical definition for dev and eval periods; persist with schema + threshold version
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 10.4 Property test — feature computation is deterministic
    - `# Feature: poker-collusion-detection, Property 5: Feature computation is deterministic`
    - _Requirements: 5.3, 5.5_

- [x] 11. Classical Baseline Detectors
  - [x] 11.1 Implement explainable classical detectors with reason strings
    - In `poker_collusion/models/classical.py`, implement value-flow graph score, aggression-asymmetry & soft-play statistical tests, isolation-pressure ratio, mutual-information conflict avoidance, and collusion-table advantage (behavior-agnostic backbone for `other_coordination`); each emits a score and a human-readable reason string
    - Combine into a baseline `risk_score ∈ [0,1]` for every evaluation pair; assign a defined default (pool prior) when unscoreable
    - _Requirements: 5.1, 5.2, 6.3, 6.4_ (Design: Principle 3)

  - [x] 11.2 Property test — risk score coverage and range (baseline)
    - `# Feature: poker-collusion-detection, Property 6: Risk score coverage and range`
    - _Requirements: 6.3, 6.4_

- [x] 12. Evidence_Retriever
  - [x] 12.1 Implement evidence selection with hard validity gate and deterministic ranking
    - In `poker_collusion/evidence/retriever.py`, select 0–5 evaluation-period shared hands per pair, each passing the hard validity gate (both players present, evaluation period, ≥1 behavior-specific action reproducing the per-family signatures from Discovery workstream C); rank by evidence-strength descending with ascending-hand-id tie-break; fill remaining slots with `NO_EVIDENCE`; de-duplicate; every cell non-empty
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 12.2 Property test — every selected evidence hand is valid
    - `# Feature: poker-collusion-detection, Property 9: Every selected evidence hand is valid`
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 12.3 Property test — evidence slots are complete and non-empty
    - `# Feature: poker-collusion-detection, Property 10: Evidence slots are complete and non-empty`
    - _Requirements: 8.6, 8.7_

  - [x] 12.4 Property test — no repeated evidence hand within a pair
    - `# Feature: poker-collusion-detection, Property 11: No repeated evidence hand within a pair`
    - _Requirements: 8.8_

  - [x] 12.5 Property test — evidence positions sorted by strength then hand id
    - `# Feature: poker-collusion-detection, Property 12: Evidence positions are sorted by strength then hand id`
    - _Requirements: 8.5_

- [x] 13. Submission_Writer
  - [x] 13.1 Implement schema-matching writer with self-validation and atomic write
    - In `poker_collusion/submission/writer.py`, emit exactly the `sample_submission.csv` schema/column order to `/kaggle/working/submission.csv` (configurable local path otherwise); one row per evaluation pair_id copied unchanged; `risk_score ∈ [0,1]`; `predicted_behavior` in the allowed set
    - Self-validate identical headers, identical row count, exact pair-id set, non-empty evidence cells, and no repeated hand id per row; raise a descriptive error on mismatch; write temp-then-rename (atomic)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 13.2 Property test — submission matches the sample schema (round-trip)
    - `# Feature: poker-collusion-detection, Property 13: Submission matches the sample schema (round-trip)`
    - _Requirements: 9.1, 9.2, 9.6_

  - [x] 13.3 Property test — submission field domains
    - `# Feature: poker-collusion-detection, Property 14: Submission field domains`
    - _Requirements: 9.3, 9.4_

- [x] 14. Wire the Phase-1 baseline end-to-end
  - [x] 14.1 Assemble the single-entry-point baseline pipeline
    - In `poker_collusion/pipeline.py`, wire Data_Loader → EDA/audit → signals → aggregation → Pair_Feature_Set → classical detectors → evidence retriever → submission writer into one documented entry point runnable on Windows PowerShell and in the Kaggle notebook
    - Route classical `risk_score` and evidence into the writer; assign `none`/behavior labels via a simple threshold placeholder (full classifier arrives in Phase 2)
    - _Requirements: 11.1_

  - [x] 14.2 End-to-end integration test on a small synthetic pool
    - Run the pipeline on a tiny synthetic pool producing a schema-valid submission; assert acceptance-shape validity
    - _Requirements: 9.6, 11.1_

- [x] 15. Phase 1 gate checkpoint
  - Confirm the submission is accepted (schema-valid) and record the classical baseline's combined three-component CV score with CIs as the measurable floor; confirm consistency with the metric contract and dossier
  - Ensure all tests pass, ask the user if questions arise

---

## Phase 2: PU-Aware Ranking Model + Behavior Classifier

- [x] 16. PU Ranking Model
  - [x] 16.1 Implement the PU-aware risk model
    - In `poker_collusion/models/pu_ranker.py`, train on trusted positives (positive) + unknown pairs (unlabelled, not negative) + confirmed negatives (reliable negative) using PU-appropriate estimators; consume the classical baseline scores/features as inputs
    - Output exactly one `risk_score ∈ [0,1]` per evaluation pair; assign a defined default when unscoreable; halt with a descriptive error (no partial output) on missing/empty required input
    - Persist trained parameters and seeds for exact reproduction
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 16.2 Property test — risk score coverage and range (PU model)
    - `# Feature: poker-collusion-detection, Property 6: Risk score coverage and range`
    - _Requirements: 6.3, 6.4_

  - [x] 16.3 Property test — ranking independent of input row order
    - `# Feature: poker-collusion-detection, Property 7: Ranking and evidence ordering are independent of input row order`
    - _Requirements: 6.5, 8.5_

- [x] 17. Behavior_Classifier
  - [x] 17.1 Implement the behavior classifier with shared risk_score and none-thresholding
    - In `poker_collusion/models/behavior_classifier.py`, predict one of `{none, directed_transfer, soft_play, coordinated_isolation, other_coordination}`; train disclosed-family heads on trusted-positive labels; route coordinated-but-untemplated pairs to `other_coordination`; assign `none` below a configurable coordination threshold
    - Per-family score = shared `risk_score` when that family is predicted else 0; account for `other_coordination` exclusion so misrouting a true disclosed-family pair is treated as a scoring loss
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 12.3_

  - [x] 17.2 Property test — below-threshold pairs are routed to `none`
    - `# Feature: poker-collusion-detection, Property 8: Below-threshold pairs are routed to none`
    - _Requirements: 7.6_

- [x] 18. Adopt CV-confirmed exploits and calibrate the shared risk_score
  - Revisit the Discovery Exploit Ledger; run each OPEN exploit through the local CV harness; mark CONFIRMED or REFUTED with supporting CV evidence in the dossier
  - Adopt only exploits confirmed beyond the noise band (e.g., risk-score calibration for Behavior MAP, liberal `other_coordination` routing checked against Pair-AP cost, coverage-over-precision evidence rule); wire adopted routing/calibration into the model and behavior classifier
  - _Requirements: 12.1, 12.3, 12.4_ (Design: Phase −1 Workstream D)

- [x] 19. Phase 2 gate checkpoint
  - Confirm the Phase-2 pipeline beats the Phase-1 floor beyond the noise band on the combined summary with no component regression; confirm consistency with metric contract and dossier
  - Ensure all tests pass, ask the user if questions arise

---

## Phase 3: Learned Evidence Ranker + Joint Tuning

- [ ] 20. Learned evidence-strength ranker
  - [ ] 20.1 Implement a learned evidence-strength scorer behind the validity gate
    - In `poker_collusion/evidence/learned_ranker.py`, replace/augment the heuristic evidence-strength score with a learned scorer; the hard validity gate and deterministic ascending-hand-id tie-break remain unchanged
    - _Requirements: 8.5, 8.4_

  - [ ] 20.2 Property test — evidence positions sorted by strength then hand id (learned ranker)
    - `# Feature: poker-collusion-detection, Property 12: Evidence positions are sorted by strength then hand id`
    - _Requirements: 8.5_

- [ ] 21. Joint tuning across the three metric components
  - [ ] 21.1 Implement joint-tuning configuration and reporting
    - In `poker_collusion/validation/joint_tuning.py`, expose configuration controlling trade-offs among Pair AP, Evidence MAP@5, and Behavior MAP; report the effect on all three components for every configuration change so regressions are visible; select the final config by the combined three-component summary on held-out development data
    - _Requirements: 12.1, 12.2, 12.4_

  - [ ] 21.2 (Optional escalation) GNN detector for complex collusive patterns
    - Only if Phase-3 baseline is set: add a GNN escalation on the per-pool co-seating graph feeding the risk model; adopt only if it beats the prior configuration beyond the noise band
    - _Requirements: 6.6, 12.4_ (Design: Prior Art, Phase-3+ escalation)

- [ ] 22. Phase 3 gate checkpoint
  - Confirm the Phase-3 pipeline beats Phase 2 beyond the noise band with no component regression; confirm consistency with metric contract and dossier
  - Ensure all tests pass, ask the user if questions arise

---

## Final: Reproducibility, Winner Verification, and SDK Guard

- [ ] 23. Reproducibility manifest and seed persistence
  - [ ] 23.1 Implement the reproducibility manifest and seed/artifact persistence
    - In `poker_collusion/repro/manifest.py`, persist all random seeds, trained model artifacts, calibrated thresholds, and feature schema versions; produce a manifest listing input file versions, code version, seeds, and output checksums for the selected submission
    - _Requirements: 11.2, 11.3, 11.5_

  - [ ] 23.2 Property test — selected configuration reproduces the submission
    - `# Feature: poker-collusion-detection, Property 16: Selected configuration reproduces the submission`
    - Run the pipeline twice on identical input under fixed config/seed; assert byte-identical `submission.csv` (equal checksum)
    - _Requirements: 11.2_

- [ ] 24. Case-review generator for winner verification
  - Implement `poker_collusion/repro/case_review.py` generating, per submitted evidence, a case-review record with pair ID, evidence hand IDs, observable behavior, and a plausible benign alternative; draw the five required case reviews from the dossier's confirmed findings and case candidates
  - _Requirements: 11.4_

- [ ] 25. SDK-shadowing guard
  - [ ] 25.1 Implement the no-shadow guard
    - In `poker_collusion/repro/sdk_guard.py`, detect and remove any local stub module (e.g., `aicomp_sdk.py`/helper) that would shadow the environment-provided competition SDK before pipeline execution
    - _Requirements: 11.6_

  - [ ] 25.2 Smoke test — no local stub shadows the real SDK
    - Assert the guard detects and neutralizes a planted shadowing stub before execution
    - _Requirements: 11.6_

- [ ] 26. Final checkpoint
  - Confirm the selected configuration reproduces the submission (task 23.2 passes), the five case reviews and manifest exist, and the SDK guard runs at entry; confirm the winner-verification writeup is seeded from the dossier
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional (property/unit/integration tests) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific requirements for traceability; property tests reference the design property number.
- Phase −1 is a hard gate: task 3 must be cleared before any feature engineering or modeling; Phase 0 may begin once the metric contract (2.2) is written.
- Every phase gate (tasks 3, 6, 15, 19, 22, 26) verifies: consistency with the metric contract, consultation/update of `RESEARCH_DOSSIER.md`, and improvement beyond the bootstrap noise band relative to the prior phase.
- All 16 correctness properties are covered exactly once as Hypothesis tests (≥100 iterations) with the required tag comment; Property 15 additionally has the bit-for-bit reconciliation and metric-contract acceptance checks.
- Only coding, testing, and documentation tasks executable in the workspace are included (Python, Windows/PowerShell, Kaggle notebook, GCP training available).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "2.6"] },
    { "id": 3, "tasks": ["4.1", "4.2"] },
    { "id": 4, "tasks": ["4.3", "4.4", "4.5", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["7.2", "8.1", "9.1"] },
    { "id": 8, "tasks": ["8.2", "9.2", "9.3", "9.4", "10.1"] },
    { "id": 9, "tasks": ["10.2", "10.3"] },
    { "id": 10, "tasks": ["10.4", "11.1", "12.1"] },
    { "id": 11, "tasks": ["11.2", "12.2", "12.3", "12.4", "12.5", "13.1"] },
    { "id": 12, "tasks": ["13.2", "13.3", "14.1"] },
    { "id": 13, "tasks": ["14.2", "16.1"] },
    { "id": 14, "tasks": ["16.2", "16.3", "17.1"] },
    { "id": 15, "tasks": ["17.2", "18"] },
    { "id": 16, "tasks": ["20.1"] },
    { "id": 17, "tasks": ["20.2", "21.1"] },
    { "id": 18, "tasks": ["21.2", "23.1"] },
    { "id": 19, "tasks": ["23.2", "24", "25.1"] },
    { "id": 20, "tasks": ["25.2"] }
  ]
}
```
