# Requirements Document

## Introduction

This spec defines the attack plan for the Kaggle competition **"Detect Suspicious Value Transfers in Poker"**. The goal is to detect coordinated player relationships (collusion) in 2,000,000 synthetic six-player No-Limit Texas Hold'em hands, rank every evaluation pair by suspicion, predict the coordination behavior type, and submit up to 5 supporting evidence hands per pair.

The competition is scored by a three-part metric: **Pair AP** (ranking pairs by risk), **Evidence MAP@5** (ranked evidence hands vs private planted evidence, over true target pairs only), and **Behavior MAP** (macro one-vs-rest AP over the three disclosed target families). The system must optimize all three jointly.

Two structural challenges dominate the design. First, this is a **Positive-Unlabelled (PU) learning** problem: `development_labels.csv` lists trusted positives and confirmed non-target pairs, but every unlisted development pair is UNKNOWN — not negative. Second, coordination is **episodic**: a coordinated pair mixes manipulated hands with ordinary play, so the signal must be detected at the hand level and aggregated to the pair level without assuming the relationship is active across the whole timeline. Difficult confounders (tilt, weak play, similar strategies, repeated opponent selection, streaks, strategy changes) must be separated from genuine coordination.

The deliverable is a reproducible pipeline that ingests the data, engineers pair-level and hand-level features, trains a PU-aware ranking model and a behavior classifier, retrieves and ranks evidence hands, and produces a `submission.csv` matching `sample_submission.csv` exactly, plus the artifacts required for winner verification.

## Glossary

- **Pipeline**: The complete end-to-end system: data ingestion → EDA → feature engineering → PU ranking model → behavior classifier → evidence retrieval → submission generation → local validation.
- **Data_Loader**: The module that reads and joins the competition parquet/CSV files into analysis-ready tables.
- **Pair**: An unordered relationship between two players evaluated for coordination. Evaluation pairs are listed in `evaluation_pairs.csv`.
- **Development_Period**: The chronological first 60% of hands within each pool (`hands.phase`), used for training and label derivation.
- **Evaluation_Period**: The chronological final 40% of hands within each pool, from which risk scores, behaviors, and evidence hands are produced.
- **Pool**: One of 400 persistent groups of 30 players sharing a single `table_id` for approximately 5,000 hands.
- **Shared_Hand**: A hand in which BOTH players of a Pair are seated.
- **Target_Behavior**: One of the coordination classes: `directed_transfer`, `soft_play`, `coordinated_isolation`, `other_coordination`, or `none`.
- **Disclosed_Family**: One of the three publicly described coordination behaviors: `directed_transfer`, `soft_play`, `coordinated_isolation`.
- **Other_Coordination**: A fourth, undisclosed coordination mechanism that never appears in public positive labels; predicted when coordination fits none of the three Disclosed_Families.
- **PU_Learning**: Positive-Unlabelled learning; training from a set of trusted positives and unlabelled examples where unlabelled does not mean negative.
- **Trusted_Positive**: A development pair confirmed as coordinated in `development_labels.csv`.
- **Confirmed_Negative**: A development pair confirmed as a non-target (non-coordinated) pair in `development_labels.csv`.
- **Unknown_Pair**: A development pair not listed in `development_labels.csv`; label status is unknown.
- **Episodic_Activity**: The property that a coordinated Pair manipulates only a subset of its Shared_Hands.
- **Pair_Feature_Set**: The vector of engineered features describing a Pair, aggregated from hand-level signals.
- **Risk_Model**: The PU-aware model that outputs a risk_score (0–1) per evaluation Pair.
- **Behavior_Classifier**: The model that predicts the Target_Behavior for a Pair.
- **Evidence_Retriever**: The module that selects and ranks up to 5 evaluation-period Shared_Hands as evidence per Pair.
- **Behavior_Specific_Action**: An observable action in the public action log characteristic of a Disclosed_Family (e.g., a value dump, a checked-back strong hand, coordinated pressure).
- **Submission_Writer**: The module that produces `submission.csv` matching `sample_submission.csv`.
- **Local_Validator**: The module that reproduces the three-part competition metric on held-out development data.
- **Pair_AP**: Average Precision over all evaluation pairs ranked by risk_score.
- **Evidence_MAP5**: Mean Average Precision at 5 of ranked evidence hand IDs over true target pairs versus private planted evidence.
- **Behavior_MAP**: Macro one-vs-rest Average Precision averaged across the three Disclosed_Families only.

## Requirements

### Requirement 1: Data Ingestion and Joining

**User Story:** As a Kaggle competitor, I want the raw competition files loaded and joined into analysis-ready tables, so that all downstream feature engineering operates on consistent, correctly linked data.

#### Acceptance Criteria

1. THE Data_Loader SHALL read `players.parquet`, `hands.parquet`, `seats.parquet`, `actions.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, and `sample_submission.csv` from a configurable input directory.
2. THE Data_Loader SHALL join gameplay tables on `hand_id` and player tables on `player_id`, preserving the `(hand_id, action_no)` ordering of `actions.parquet`.
3. WHEN a hand is loaded, THE Data_Loader SHALL attach the `hands.phase` value so that each hand is labeled as belonging to the Development_Period or the Evaluation_Period.
4. THE Data_Loader SHALL expose, for any Pair, the set of Shared_Hands in which both players are seated, partitioned by Development_Period and Evaluation_Period.
5. IF a referenced `hand_id`, `player_id`, or expected column is missing from an input file, THEN THE Data_Loader SHALL raise a descriptive error identifying the file and the missing key.
6. WHERE memory limits prevent loading `actions.parquet` (18,609,028 rows) in full, THE Data_Loader SHALL support chunked or column-projected reads driven by a configurable column and row-group selection.

### Requirement 2: Exploratory Data Analysis and Label Audit

**User Story:** As a Kaggle competitor, I want a reproducible EDA and label audit, so that I understand the distributions, confirm the PU structure, and calibrate feature thresholds from measured statistics rather than guesses.

#### Acceptance Criteria

1. THE Pipeline SHALL produce summary statistics of Shared_Hand counts per Pair, showing that different pairs share differing numbers of hands.
2. THE Pipeline SHALL report the counts of Trusted_Positive pairs, Confirmed_Negative pairs, and Unknown_Pairs derived from `development_labels.csv`.
3. THE Pipeline SHALL report the distribution of Target_Behavior labels among Trusted_Positive pairs across the three Disclosed_Families.
4. WHEN auditing `development_evidence.csv`, THE Pipeline SHALL confirm that each listed evidence hand is a Shared_Hand for its Pair and contains at least one Behavior_Specific_Action in the public action log.
5. THE Pipeline SHALL verify that `evaluation_pairs.csv` excludes publicly labelled pair IDs and pairs containing a publicly labelled positive player, and SHALL report any violations found.
6. THE Pipeline SHALL persist EDA outputs (statistics and calibrated thresholds) to a versioned artifact so that feature engineering consumes fixed, documented values.

### Requirement 3: Hand-Level Coordination Signals

**User Story:** As a Kaggle competitor, I want per-hand signals computed for each pair of co-seated players, so that episodic coordination is captured at the level where the manipulated action is visible.

#### Acceptance Criteria

1. FOR each Shared_Hand of a Pair, THE Pipeline SHALL compute a directed value-flow signal equal to the net chips transferred between the two players within that hand, computed as the difference between chips one player contributes via `seats` and the chips that player receives in the hand outcome, expressed as a signed value in chips.
2. FOR each Shared_Hand of a Pair, THE Pipeline SHALL compute an aggression-asymmetry signal equal to the ratio of the count of bet-or-raise actions taken by each player against the other (from the ordered `actions` log) to that player's count of bet-or-raise actions against all other seated players in the same hand, where a lower ratio indicates greater mutual aggression avoidance.
3. FOR each Shared_Hand of a Pair, THE Pipeline SHALL compute an isolation signal equal to the ratio of the count of bet-or-raise actions the two players direct at other seated players to the count of bet-or-raise actions the two players direct at each other, using the ordered `actions` log.
4. WHEN an action has `action = all_in`, THE Pipeline SHALL classify the all-in as a call WHERE `amount` is less than or equal to `to_call`, and SHALL classify it as an aggressive action WHERE `amount` is greater than `to_call`.
5. IF a signal defined in criteria 2 or 3 has a zero denominator (no qualifying bet-or-raise opportunities exist for the relevant player or pair in the hand), THEN THE Pipeline SHALL assign that signal a defined sentinel value indicating "not applicable" and SHALL NOT raise an error or produce an undefined result.
6. THE Pipeline SHALL compute each hand-level signal using only decision-time context available in the action log (`pot_before`, `stack_before`, `to_call`, `players_active`, `amount`, `amount_to`).
7. THE Pipeline SHALL NOT use any information that is unavailable at the decision time of an action when computing hand-level signals.
8. FOR each Shared_Hand, THE Pipeline SHALL record a boolean indicating whether a Behavior_Specific_Action for any Disclosed_Family is present in the public action log, so that the hand can later qualify as evidence.

### Requirement 4: Episodic Activity Detection

**User Story:** As a Kaggle competitor, I want episodic coordination detected within a pair's timeline, so that a relationship that manipulates only some hands is not diluted by its ordinary play.

#### Acceptance Criteria

1. THE Pipeline SHALL aggregate hand-level signals into Pair_Feature_Set values using aggregations that preserve episodic peaks (for example maxima, high quantiles, and burst counts) in addition to means.
2. WHEN a Pair has manipulated a subset of Shared_Hands, THE Pipeline SHALL produce a higher coordination signal than a Pair with the same mean but no concentrated episode.
3. THE Pipeline SHALL compute a temporal-concentration feature that measures whether high-signal Shared_Hands cluster in time rather than being uniformly distributed.
4. THE Pipeline SHALL compute features that are robust to differing Shared_Hand counts across pairs, so that pairs sharing few hands are neither systematically inflated nor suppressed relative to pairs sharing many hands.

### Requirement 5: Pair Feature Engineering

**User Story:** As a Kaggle competitor, I want a documented Pair_Feature_Set combining all engineered signals, so that the Risk_Model and Behavior_Classifier train on a consistent representation that separates coordination from confounders.

#### Acceptance Criteria

1. THE Pipeline SHALL assemble a Pair_Feature_Set for every Pair that includes directed value-flow, aggression-asymmetry, isolation, and episodic-activity features.
2. THE Pipeline SHALL include features designed to distinguish coordination from the disclosed confounders: tilt, weak play, similar strategies, repeated opponent selection, streaks, and strategy changes.
3. THE Pipeline SHALL compute the identical Pair_Feature_Set definition for Development_Period training pairs and Evaluation_Period scoring pairs, so that no feature is available at training time but absent at scoring time.
4. THE Pipeline SHALL persist the Pair_Feature_Set with a schema version and the calibrated thresholds used, so that feature generation is reproducible.
5. FOR ALL pairs, computing the Pair_Feature_Set twice from the same input SHALL produce identical feature values (deterministic feature property).

### Requirement 6: PU-Aware Risk Model

**User Story:** As a Kaggle competitor, I want a risk model trained under Positive-Unlabelled assumptions, so that Unknown_Pairs are not incorrectly treated as negatives and Pair AP is maximized.

#### Acceptance Criteria

1. WHEN training the Risk_Model, THE Risk_Model SHALL use Trusted_Positive pairs as positive examples and SHALL treat every development pair not listed in `development_labels.csv` (an Unknown_Pair) as unlabelled rather than negative.
2. WHERE Confirmed_Negative pairs are present in `development_labels.csv`, THE Risk_Model SHALL use them as reliable negative examples.
3. THE Risk_Model SHALL output exactly one risk_score, expressed as an inclusive real number between 0.0 and 1.0, for every Pair listed in `evaluation_pairs.csv`, where a higher value indicates a higher likelihood of coordination.
4. IF the Risk_Model cannot compute a score for a Pair listed in `evaluation_pairs.csv`, THEN THE Risk_Model SHALL assign that Pair a defined default risk_score rather than omitting the Pair or producing an undefined value.
5. WHEN two pairs receive equal risk_scores, THE Pipeline SHALL rely on the competition's deterministic pair_id tie-break to produce a consistent ranking and SHALL NOT depend on input row ordering for correctness.
6. WHEN validating ranking quality, THE Risk_Model SHALL be evaluated with a PU-adjusted estimate of Pair_AP computed without assuming Unknown_Pairs are negatives.
7. THE Risk_Model SHALL persist its trained parameters and the random seeds used, so that re-running scoring produces identical risk_score values for the same `evaluation_pairs.csv` input.
8. IF a required input file for training or scoring is missing or empty, THEN THE Risk_Model SHALL halt with a descriptive error indication and SHALL NOT write a partial output file.

### Requirement 7: Behavior Classification

**User Story:** As a Kaggle competitor, I want a behavior classifier for the coordination families, so that Behavior MAP is maximized while correctly routing undisclosed coordination to other_coordination.

#### Acceptance Criteria

1. THE Behavior_Classifier SHALL predict exactly one Target_Behavior per Pair from the set `none`, `directed_transfer`, `soft_play`, `coordinated_isolation`, `other_coordination`.
2. THE Behavior_Classifier SHALL train its Disclosed_Family predictions using the Target_Behavior labels of Trusted_Positive pairs.
3. WHEN a Pair is predicted as coordinated but its features fit none of the three Disclosed_Families, THE Behavior_Classifier SHALL assign `other_coordination`.
4. THE Behavior_Classifier SHALL provide a per-family score consistent with the competition's Behavior_MAP definition, where the class score equals the risk_score when that family is predicted and 0 otherwise.
5. THE Pipeline SHALL account for the exclusion of `other_coordination` from Behavior_MAP when tuning the classifier, so that routing a true Disclosed_Family pair to `other_coordination` is treated as a scoring loss.
6. IF a Pair's risk_score is below a configurable coordination threshold, THEN THE Behavior_Classifier SHALL assign `none`.

### Requirement 8: Evidence Hand Retrieval and Ranking

**User Story:** As a Kaggle competitor, I want up to five evidence hands retrieved and ranked per pair, so that Evidence MAP@5 is maximized and submitted evidence satisfies all validity rules.

#### Acceptance Criteria

1. WHEN processing an evaluation Pair, THE Evidence_Retriever SHALL select at least 0 and at most 5 candidate evidence hands, where each selected hand is a Shared_Hand whose participant set includes both players of the Pair.
2. IF a candidate hand's participant set does not include both players of the Pair, THEN THE Evidence_Retriever SHALL NOT select that hand.
3. THE Evidence_Retriever SHALL select only hands whose play occurs within the Evaluation_Period, and IF a candidate hand occurs within the Development_Period, THEN THE Evidence_Retriever SHALL NOT select that hand.
4. THE Evidence_Retriever SHALL treat a hand as a valid evidence hand only when all of the following hold: the hand is a Shared_Hand containing both players of the Pair, the hand occurs within the Evaluation_Period, and the hand contains at least one Behavior_Specific_Action recorded in the public action log; and IF a candidate hand's collusion indication derives solely from latent scenario activation with no Behavior_Specific_Action in the public action log, THEN THE Evidence_Retriever SHALL NOT treat that hand as valid evidence.
5. WHEN two or more valid evidence hands are selected for a Pair, THE Evidence_Retriever SHALL assign them to positions `evidence_hand_1` through `evidence_hand_5` in descending order of their computed evidence-strength score, and IF two hands have equal evidence-strength scores, THEN THE Evidence_Retriever SHALL order them by ascending hand ID so that ordering is deterministic and repeatable across runs.
6. WHEN fewer than 5 valid evidence hands are available for a Pair, THE Evidence_Retriever SHALL fill each of the remaining evidence positions with the literal value `NO_EVIDENCE`.
7. THE Evidence_Retriever SHALL emit exactly one non-empty value in each of the 5 evidence positions for every Pair, where each value is either a selected valid hand ID or the literal `NO_EVIDENCE`.
8. WITHIN a single Pair's 5 evidence positions, THE Evidence_Retriever SHALL NOT assign the same hand ID to more than one position.

### Requirement 9: Submission File Generation

**User Story:** As a Kaggle competitor, I want a submission file that exactly matches the required template, so that the submission is accepted and scored without format rejection.

#### Acceptance Criteria

1. THE Submission_Writer SHALL produce one row per pair_id present in `evaluation_pairs.csv`, copying each pair_id unchanged.
2. THE Submission_Writer SHALL write the columns `pair_id`, `risk_score`, `predicted_behavior`, `evidence_hand_1`, `evidence_hand_2`, `evidence_hand_3`, `evidence_hand_4`, `evidence_hand_5` in the exact order and header of `sample_submission.csv`.
3. THE Submission_Writer SHALL write `risk_score` as a value between 0 and 1 inclusive.
4. THE Submission_Writer SHALL write `predicted_behavior` as one of `none`, `directed_transfer`, `soft_play`, `coordinated_isolation`, `other_coordination`.
5. THE Submission_Writer SHALL write the output file to `/kaggle/working/submission.csv` when executing in the Kaggle notebook environment, and to a configurable local path otherwise.
6. WHEN the Submission_Writer completes, THE Pipeline SHALL validate the output against `sample_submission.csv` for identical column headers, identical row count, and the exact set of pair_ids, and SHALL raise a descriptive error on any mismatch.
7. THE Submission_Writer SHALL confirm every evidence cell is non-empty and that no hand ID repeats within a row before the file is considered complete.

### Requirement 10: Local Validation Against the Three-Part Metric

**User Story:** As a Kaggle competitor, I want a local validator that reproduces the competition metric, so that I can compare experiments offline and trust leaderboard movement before submitting.

#### Acceptance Criteria

1. THE Local_Validator SHALL compute Pair_AP over a held-out set of development pairs ranked by predicted risk_score.
2. THE Local_Validator SHALL compute Evidence_MAP5 over held-out true target pairs by comparing ranked evidence hand IDs against the known development evidence hands, where a missed target pair contributes zero.
3. THE Local_Validator SHALL compute Behavior_MAP as macro one-vs-rest Average Precision across the three Disclosed_Families only, excluding `other_coordination`.
4. THE Local_Validator SHALL construct held-out splits that respect the PU structure, so that Unknown_Pairs are not scored as negatives in a way that misrepresents Pair_AP.
5. THE Local_Validator SHALL report the three component scores separately and a combined summary, so that trade-offs between the components are visible when tuning.
6. WHEN the public competition metric code is available, THE Local_Validator SHALL be reconciled against it and SHALL report any divergence in computed scores.

### Requirement 11: Reproducibility and Winner Verification

**User Story:** As a Kaggle competitor, I want the full pipeline to be reproducible and to emit the artifacts required for prize eligibility, so that I can satisfy winner verification within the 7-day window.

#### Acceptance Criteria

1. THE Pipeline SHALL execute end-to-end from raw inputs to `submission.csv` through a single documented entry point on Windows PowerShell and in the Kaggle notebook environment.
2. THE Pipeline SHALL fix and record all random seeds, so that re-running the selected configuration reproduces the submitted `submission.csv`.
3. THE Pipeline SHALL persist trained model artifacts, calibrated thresholds, and feature schema versions needed to reproduce the selected submission.
4. THE Pipeline SHALL generate, for any submitted evidence, a case-review record containing the pair ID, the evidence hand IDs, the observable behavior, and a plausible benign alternative, to support the five required case reviews.
5. THE Pipeline SHALL produce a reproducibility manifest listing input file versions, code version, seeds, and output checksums, to support the Kaggle Solution Writeup and public notebook or repository.
6. WHERE the Kaggle notebook shadows the real competition SDK or helper modules, THE Pipeline SHALL ensure no local stub module shadows the environment-provided modules before execution.

### Requirement 12: Joint Optimization of the Three Metric Components

**User Story:** As a Kaggle competitor, I want the pipeline tuned for all three metric components together, so that gains in one component are not silently traded away against another.

#### Acceptance Criteria

1. THE Pipeline SHALL expose configuration controlling the trade-offs among Pair_AP, Evidence_MAP5, and Behavior_MAP.
2. WHEN a configuration change improves one component, THE Local_Validator SHALL report the effect on all three components so that regressions are visible.
3. THE Pipeline SHALL share the risk_score between the Risk_Model output and the Behavior_MAP per-family score, consistent with the metric definition that a family's class score equals the risk_score when predicted.
4. THE Pipeline SHALL select the final submitted configuration by comparing the combined three-component summary across candidate configurations on held-out development data.
