# Requirements Document

## Introduction

This spec defines the **layered detection + LB-calibrated tuning system** for the Kaggle competition "Detect Suspicious Value Transfers in Poker". It is a distinct follow-on to the original `poker-collusion-detection` spec: that spec defined the detection pipeline; THIS spec defines the **validation, composition, and tuning methodology** that lets us improve the model without regressing — the exact discipline whose absence caused repeated leaderboard regressions.

The motivating failures, all real and logged in RESEARCH_DOSSIER.md:
- Rich feature stacks that scored high on internal CV regressed on the leaderboard (0.39 -> 0.30) because validation was self-referential.
- A drift gate with a fixed 0.65 threshold mis-flagged the LB-VERIFIED foundation (`bq+dir`, drift 0.679, real LB 0.44519) as a failure, while correctly catching a feature (nullprobe, drift 0.72+) that did regress. The threshold was calibrated on single-feature adds and does not generalize to multi-block configs.
- A promising feature (whipsaw for coordinated_isolation) was nearly discarded after one shallow pass before tuning revealed it was the strongest signal found.

The system must let us (a) reverse-engineer the synthetic generator's per-family fingerprints into features, (b) compose those features as independently toggleable layers on top of a known-good floor that can never regress, (c) validate every candidate through a gate calibrated against REAL leaderboard anchors rather than a guessed threshold, and (d) tune the baseline model and the layers without overfitting the validation holdout.

The deliverable is a reproducible harness that, given the current feature layers, reports each layer's contribution, projects a leaderboard score with a calibrated and honest uncertainty, and emits a single submission-ready candidate that is the leanest configuration clearing the LB-calibrated gate.

## Glossary

- **Layer**: A named, independently toggleable block of features (e.g. foundation, board_equity, directional, iso, whipsaw, negative_space) that can be switched ON or OFF in a candidate.
- **Foundation**: The LB-verified base feature set (v5 pair features), holdout AP ~0.3839, whose real leaderboard score (0.44519 with board_equity+directional) is the KNOWN-GOOD FLOOR.
- **Known_Good_Floor**: The best leaderboard-verified configuration; no candidate may be submitted that is projected below it, and turning all optional layers OFF must reproduce it exactly.
- **Holdout**: The pair-disjoint, table-disjoint 40% split the model never trains on; the internal judge.
- **Drift_Gate**: An adversarial-validation classifier that scores how separable dev rows are from eval rows on a feature set; high separation = distribution drift = untrustworthy.
- **LB_Anchor**: A (configuration, real_leaderboard_score, measured_drift, holdout_AP) tuple from an actually-submitted candidate, used to calibrate the gate and the projection.
- **LB_Calibrated_Threshold**: A drift threshold derived from LB_Anchors (the max drift among configs that did NOT regress vs the min drift among configs that DID), replacing the guessed 0.65.
- **Clean_Separation_Check**: A validity precondition on the drift gate requiring that the maximum measured drift among PASS anchors (configs at or above the Known_Good_Floor) is strictly below the minimum measured drift among FAIL anchors (configs that regressed); without this separation no single drift threshold separates the classes and the gate is disqualified as a discriminator.
- **LB_Projection**: A holdout-to-leaderboard estimate valid ONLY for drift-passing configs, fit on LB_Anchors, reported with its uncertainty and fit-point count.
- **Composition_Mode**: How a layer combines with the current model: RANK_BLEND (weighted rank average, weight tunable to 0, for monotone signals) or RETRAIN (added as features and refit, for conditional/non-linear signals), each with its own safety condition.
- **Ablation**: Leave-one-out measurement of each layer's unique contribution to holdout AP.
- **Baseline_Tuning**: Coarse tuning of the model hyperparameters and PU sample weights on a fixed feature set.
- **Layer_Tuning**: Tuning a layer's own knobs (shrinkage constant, thresholds, composition weight).
- **Sample_Size_Shrinkage**: Down-weighting a per-pair statistic toward a prior for pairs with few shared hands (value * n/(n+k)).
- **Generator_Fingerprint**: A per-behavior-family structural signature of the synthetic data generator, discovered by comparing planted evidence hands to ordinary hands.

## Requirements

### Requirement 1: LB-calibrated drift gate
**User Story:** As the model builder, I want the drift gate calibrated against real leaderboard outcomes, so that it stops rejecting known-good configurations and only blocks configurations that actually regress.

#### Acceptance Criteria
1. WHEN the drift gate is evaluated THEN the system SHALL classify a configuration using an LB_Calibrated_Threshold derived from LB_Anchors, NOT a fixed 0.65.
2. WHEN an LB_Anchor is a configuration that scored at or above the Known_Good_Floor on the real leaderboard THEN the gate SHALL classify that configuration's drift level as PASS.
3. WHEN an LB_Anchor is a configuration that regressed on the real leaderboard THEN the gate SHALL classify that configuration's drift level as FAIL.
4. WHEN the LB_Calibrated_Threshold is derived from LB_Anchors THEN the system SHALL first perform a Clean_Separation_Check that computes the maximum measured drift among PASS anchors and the minimum measured drift among FAIL anchors, and SHALL treat the anchors as SEPARABLE only IF the maximum drift among PASS anchors is strictly less than the minimum drift among FAIL anchors.
5. IF the Clean_Separation_Check fails because the PASS-anchor drift range and the FAIL-anchor drift range overlap THEN the system SHALL treat the drift gate as INVALID and untrustworthy, SHALL NOT report any LB_Calibrated_Threshold as trustworthy, and SHALL surface the null result that drift does not cleanly separate leaderboard outcomes.
6. WHEN the Clean_Separation_Check passes AND at least 3 LB_Anchors exist THEN the system SHALL trust the LB_Calibrated_Threshold to classify new configurations as PASS or FAIL; otherwise the system SHALL fall back to the PROVISIONAL threshold logic from criterion 7 and state the reduced confidence.
7. IF fewer than 3 LB_Anchors exist OR the Clean_Separation_Check fails THEN the system SHALL apply a PROVISIONAL threshold and SHALL report the threshold as PROVISIONAL and state the reduced confidence.
8. IF the known-good foundation (bq+dir, real LB 0.44519, drift ~0.679) is evaluated THEN the gate SHALL return PASS, using the trusted LB_Calibrated_Threshold when the Clean_Separation_Check passes and at least 3 LB_Anchors exist, and using the PROVISIONAL threshold logic from criterion 7 otherwise.

### Requirement 2: Toggleable layers with a guaranteed floor
**User Story:** As the model builder, I want each feature block to be an independently switchable layer with a floor guarantee, so that adding a layer can never silently corrupt the known-good model.

#### Acceptance Criteria
1. WHEN all optional layers are OFF THEN the system SHALL reproduce the Known_Good_Floor configuration exactly.
2. WHEN a layer is composed via RANK_BLEND with weight 0 THEN the resulting ranking SHALL equal the base ranking exactly.
3. WHEN a candidate is assembled THEN the system SHALL report which layers are ON on a best-effort basis.
4. IF layer reporting fails THEN candidate assembly SHALL proceed and the reporting failure SHALL NOT block assembly.
5. WHEN a layer is added AND it reduces holdout AP at every positive composition weight THEN the system SHALL flag it as CORRUPTING (displacing known-good signal), not merely null.

### Requirement 3: Ablation reporting
**User Story:** As the model builder, I want leave-one-out ablation for every layer, so that I submit the leanest configuration and never ship redundant layers.

#### Acceptance Criteria
1. WHEN a multi-layer candidate is evaluated THEN the system SHALL report each layer's unique contribution (full holdout minus holdout with that layer removed).
2. IF a layer's unique contribution is <= 0 THEN the system SHALL recommend removing it.
3. WHEN ablation completes THEN the system SHALL identify the leanest subset within a stated tolerance of the best holdout.

### Requirement 4: Composition-mode correctness
**User Story:** As the model builder, I want each layer composed by the mode appropriate to its signal shape, so that conditional signals are not destroyed by linear blending.

#### Acceptance Criteria
1. WHEN a layer is a monotone per-pair score THEN the system SHALL support RANK_BLEND composition with a holdout-tuned weight (0 allowed).
2. WHEN a layer is a conditional/non-linear signal THEN the system SHALL support RETRAIN composition and SHALL require the combined feature set to clear the LB-calibrated Drift_Gate before the result is trusted.
3. WHEN RANK_BLEND of a layer reduces holdout at every positive weight AND RETRAIN of the same layer increases holdout THEN the system SHALL record the layer as conditional and use RETRAIN; WHERE this performance condition does not both hold (RANK_BLEND does not underperform RETRAIN), no conditional/RETRAIN classification is mandated.

### Requirement 5: Two-level tuning without holdout overfitting
**User Story:** As the model builder, I want baseline-model and layer tuning sequenced and bounded, so that tuning improves the leaderboard rather than overfitting the holdout.

#### Acceptance Criteria
1. WHEN tuning begins THEN the system SHALL first resolve feature-set drift, THEN tune the Baseline model, THEN tune Layers, in that order.
2. WHEN tuning any knob THEN the system SHALL use coarse grids (structural steps), not fine micro-grids.
3. WHEN a tuned configuration is selected THEN the system SHALL enforce two independent guarantees: (a) a selected tuned configuration SHALL clear the LB-calibrated Drift_Gate, AND (b) a tuned gain that increases drift past the threshold SHALL be rejected; the rejection guarantee (b) SHALL hold independently of the selection check (a).
4. WHEN a tuned configuration is reported THEN the system SHALL report its LB_Projection with uncertainty, distinguishing measured holdout from projected LB.

### Requirement 6: Sample-size shrinkage for per-pair statistics
**User Story:** As the model builder, I want every per-pair statistic shrunk by its sample size, so that low-shared-hand pairs do not surface as small-sample noise.

#### Acceptance Criteria
1. WHEN a per-pair rate/z-score/excess enters a layer THEN it SHALL be shrunk by shared-hand count before ranking.
2. WHEN a shrinkage constant is chosen THEN the system SHALL complete a FULL sweep of the shrinkage constant even if an early constant passes the top-K median shared-hand verification (the sweep SHALL NOT stop early), and the choice SHALL be verified by checking that the top-K-by-score median shared-hand count is not below the population median.

### Requirement 7: Honest projection and submission gate
**User Story:** As the competitor, I want every submission decision backed by an LB-calibrated projection with stated uncertainty, so that we spend scarce submissions on candidates that are genuinely likely to improve.

#### Acceptance Criteria
1. WHEN a candidate is proposed for submission THEN the system SHALL emit: layers ON, holdout AP, measured drift, gate verdict, LB_Projection, and whether the projection regime is verified.
2. IF the candidate does not clear the LB-calibrated Drift_Gate THEN the system SHALL NOT recommend submission.
3. WHEN a candidate is submitted and its real LB score returns THEN the system SHALL append it as a new LB_Anchor and recalibrate the gate and projection.
4. The system SHALL NEVER overwrite the Known_Good_Floor submission artifact without a leaderboard-verified improvement.


### Requirement 8: Process gates and anti-rails (prevent the first spec's sprawl)
**User Story:** As the competitor who lost significant time when the first spec built a full pipeline on an unverified foundation, I want structural gates hard-wired into this spec, so that the work cannot sprawl, run on unchecked assumptions, or grind a dead path without an explicit stop.

#### Acceptance Criteria
1. GATE A (early external checkpoint): WHEN this spec's task list is executed THEN the FIRST task SHALL produce a submission-ready candidate and obtain a REAL leaderboard score before any elaboration task begins; no task may build a second layer on top of an unverified base.
2. GATE B (assumption register): WHEN any requirement or design decision rests on a load-bearing assumption THEN the system SHALL record (a) the assumption stated plainly, (b) the cheapest test that would falsify it, (c) the action if it is false; AND design SHALL NOT proceed past an unfalsified load-bearing assumption.
3. GATE C (stop-loss tripwires): IF two consecutive submissions regress versus the Known_Good_Floor THEN work SHALL HALT and the harness itself SHALL be re-examined before any new candidate; AND a configuration that fails the LB-calibrated Drift_Gate SHALL NOT be submittable regardless of holdout AP.
4. GATE D (scope/experiment budget): WHEN a task is defined THEN it SHALL carry a pre-registered success bar AND an experiment/time budget; IF the budget is exceeded without clearing the bar THEN the task SHALL escalate to the user for a keep/kill decision rather than continuing silently.
5. GATE E (adversarial self-review at phase gates): BEFORE transitioning requirements->design->tasks THEN the system SHALL run a devil's-advocate pass naming the single most-likely-wrong assumption and its cheapest falsifying test, and record the outcome.
6. GATE F (standing referee): The system SHALL treat the always-on reverse-engineering-accountability contract (rules 1-19) as the binding validation authority for every action in this spec; any conflict resolves in favor of the contract.
7. WHEN a task completes THEN the system SHALL record its pre-registered bar, the held-out result, the gate verdict, and a keep/revert decision in RESEARCH_DOSSIER.md, so the trail is externally auditable.
