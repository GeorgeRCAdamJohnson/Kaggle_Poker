# Requirements Document

## Introduction

This spec builds the tooling and validators to convert REAL code with KNOWN real leaderboard scores into REAL `(local-holdout-AP, real-LB)` anchors, replacing the false "we only have one anchor" premise that sank the retired `poker-projection-recalibration` spec.

Two independent bodies of ground truth exist and are the foundation of this work:

1. **Our own submission ladder.** Nine of our own submission artifacts survive on disk, each tagged with its real public-leaderboard score in its filename: `submission_best_{018462, 019029, 019281, 032580, 033131, 037063, 039372, 041289, 044519}.csv`. The dossier records the full real trajectory these came from (0.05713 -> 0.16584 -> ... -> 0.44519), plus documented regressions.
2. **Competitor code with known scores.** Four competitor notebooks under `.kiro/specs/poker-collusion-detection/Code/`, at least one of which (`detecting-collusive-value-transfer-in-6-max-poker.ipynb`, claimed public LB **0.64469**) is a PU-aware pipeline using the same recipe family as ours, auto-discovers the local data root, and needs only `polars` beyond what is already installed. Two others claim ~0.64860 (a topological triple-GBDT ensemble) but need `lightgbm`+`catboost`.

We hold the full 8-file competition dataset locally and the official metric kernel (`data/poker/_metric_kernel/slash-poker-competition-metric.ipynb`).

**The hard, honest constraint (accountability contract Rules 1, 9).** All nine of our artifacts are EVAL submissions; their scores are PRIVATE-leaderboard numbers computed against labels we do NOT hold. Therefore the local metric CANNOT reproduce any LB score as a number. What it CAN do is score each artifact on the DEV-labelled holdout we do hold, and test whether that local score ORDERS the artifacts the same way the real LB does. The validator's job is thus a RELATIVE, rank-based external-judge check, never an absolute score match.

**Build order (as directed):** tooling first, then the validators (proven against our OWN known scores before trusting them on anyone else's code), then reproduce the 0.64469 submission as a new anchor. A validator that cannot track our own known LB ladder is not trusted to anchor anything.

**Non-goals.** This spec does NOT submit to Kaggle, does NOT re-fit the `poker-layered-tuning` projection line (that is a downstream follow-up only IF the validator earns trust), and does NOT modify the competitor notebooks' methodology - it runs them as-is against local data. Competitor notebooks are UNTRUSTED external code and SHALL be read and sandboxed before execution, never blind-run.

## Glossary

- **Real LB score**: A public/private leaderboard number actually returned by Kaggle - the only external-judge ground truth (contract Rule 1).
- **Local metric**: The official competition metric (from the metric kernel) run locally against the dev labels we hold - `0.70*PairAP + 0.20*EvidenceMAP@5 + 0.10*BehaviorMAP`.
- **Local holdout AP**: The PairAP (or full local metric) a submission scores on a defined local dev-holdout split.
- **Known-LB artifact**: One of our nine on-disk `submission_best_<score>.csv` files whose real LB score is known from its filename and the dossier.
- **Anchor**: A verified `(config_id, real_lb, local_holdout_ap)` tuple produced by scoring real code/artifacts locally and pairing with a known real LB.
- **Rank-tracking**: The property that local holdout AP orders a set of artifacts the same way (monotonically with) their real LB scores; measured by a rank-correlation statistic.
- **Reproduction runner**: Tooling that executes a (competitor or own) notebook/script against the LOCAL data and captures its emitted `submission.csv`.
- **Canonical local scorer**: The single agreed local scoring recipe (metric, split, unknown-label handling) all anchors are measured under, so anchors are mutually comparable (inherits the AP-definition-consistency lesson from the retired recalibration spec).

## Requirements

### Requirement 1: Reproduction and scoring tooling (built first)
**User Story:** As the model builder, I want the environment and tooling to run real code against our local data and score its output with the official metric, so that reproduction is possible at all before I build validators on top of it.

#### Acceptance Criteria
1. WHEN the tooling is set up THEN the system SHALL provide a reproducible Python environment with the dependencies the lowest-friction notebook needs (at minimum `polars` added to the existing `pandas/numpy/scikit-learn/xgboost`), recorded so the setup is repeatable.
2. WHEN a submission CSV is produced THEN the system SHALL score it with the OFFICIAL competition metric (imported/adapted from `data/poker/_metric_kernel/slash-poker-competition-metric.ipynb` or the verbatim `reference_public_metric`), never a re-implementation that could silently diverge.
3. WHERE the official metric is imported or adapted from the metric kernel, THE system MAY create a LOCAL COPY of the official metric code during the import/adaptation process, provided the copy is the official metric and not a divergent re-implementation.
4. WHEN the official metric is wired THEN the system SHALL verify it against the metric kernel's own expected output (a self-check that our local scorer equals the official one on a known input) before it is used to judge anything.
5. WHEN a notebook is to be run THEN the system SHALL treat ALL notebook content as untrusted REGARDLESS of source, read it as untrusted content, and run it in a controlled/sandboxed mode with security restrictions (explicit path shim to local data, no unreviewed network/credential access), with no trusted-source exemption or pre-vetting bypass, and SHALL record its dependency and data-path requirements (contract: external content is untrusted).
6. WHEN tooling is added THEN it SHALL live alongside the existing `poker_collusion` package (reuse the existing metric/CV where possible; do not duplicate the metric).

### Requirement 2: Validate the validator against our OWN known-LB ladder FIRST
**User Story:** As the model builder, I want the local scorer proven to track our own real leaderboard ordering before I trust it on external code, so that the validator is anchored to reality I already know rather than to itself.

#### Acceptance Criteria
1. WHEN the local scorer is built THEN the system SHALL score ALL nine known-LB artifacts (`submission_best_*.csv`) under the canonical local scorer and record each `(real_lb_from_filename, local_holdout_ap)` pair.
2. WHEN the nine pairs are recorded THEN the system SHALL measure whether the local holdout AP RANK-TRACKS the real LB order across the nine artifacts, using a stated rank-correlation statistic (e.g. Spearman) with a pre-registered threshold that SHALL require a POSITIVE correlation strictly above 0.0 for the local metric to be considered a trustworthy relative judge (a correlation of exactly 0.0 or negative is NOT acceptable).
3. IF the local metric does NOT rank-track the real LB across the nine artifacts THEN the system SHALL report this as a first-class finding (the local metric is NOT a trustworthy relative judge) and SHALL NOT proceed to trust it for anchoring - this null is a valid, valuable result (contract Rule 3), not a failure to hide.
4. WHEN a full leaderboard-number reproduction is attempted THEN the system SHALL explicitly acknowledge it is impossible (private labels absent) and SHALL NOT fabricate an absolute-score match; only the relative rank-tracking claim is permitted (contract Rules 1, 9).
5. WHEN the ladder validation completes THEN the system SHALL record, per artifact, which local scoring recipe (AP definition, split) was used, so every anchor is comparable and the recipe is auditable.
6. WHEN the canonical local scorer is chosen THEN it SHALL be the SAME recipe later applied to the reproduced competitor submission, so our-artifact anchors and competitor anchors are mutually comparable (no cross-regime mixing).

### Requirement 3: Reproduce the 0.64469 competitor submission as a new real anchor
**User Story:** As the model builder, I want to run the known-0.64469 competitor code on our local data and score it with the trusted local scorer, so that we gain a real high-scoring anchor and can see what a stronger solution does that ours does not.

#### Acceptance Criteria
1. WHEN the 0.64469 notebook is reproduced THEN the system SHALL execute it against the LOCAL data via the reproduction runner and capture its emitted `submission.csv`.
2. WHEN the reproduced submission is captured THEN the system SHALL score it under the canonical local scorer (the same one validated in Requirement 2) and record the `(real_lb=0.64469, local_holdout_ap)` anchor.
3. IF the notebook does not run to completion locally (missing dep, data-path, or compute limit) THEN the system SHALL record exactly where and why it failed and SHALL ultimately report reproduction as BLOCKED rather than partially claiming success (contract Rule 9); the status MAY pass through an intermediate state (e.g. RUNNING while the failure is being diagnosed) and need not flip to BLOCKED at the instant the failure is first detected.
4. WHEN the reproduced submission scores locally THEN the system SHALL report whether its local AP is consistent with the rank-tracking established in Requirement 2 (does a real 0.64469 sit above our real 0.44519 on the local metric, as the LB order demands?) - a consistency check of the validator on a NEW real point.
5. WHEN reproduction succeeds THEN the anchor SHALL be recorded as `verified=true` in an append-only store consistent with the parent harness's `lb_anchors.json` schema, but ONLY under the canonical local scorer's AP definition, with provenance noting it came from reproduced competitor code.

### Requirement 4: Honest reporting, audit trail, and no overclaiming
**User Story:** As the model builder, I want every number this spec produces labelled as measured or assumed and tied to an external judge, so that we never again mistake our own story for verified knowledge.

#### Acceptance Criteria
1. WHEN any result is reported THEN the system SHALL distinguish what was MEASURED (local AP, rank correlation) from what is EXTERNALLY VERIFIED (the real LB scores) from what is ASSUMED/PROJECTED (contract Rule 9).
2. WHEN the effort produces anchors THEN the system SHALL append the pre-registered bars, the measured results, and the verdicts to RESEARCH_DOSSIER.md (append-only, externally auditable).
3. WHEN this spec's findings bear on `poker-layered-tuning` THEN the system SHALL record whether the new multi-point anchor set supports re-fitting the projection line OR whether the rank-tracking null means the projection should stay disabled - but SHALL NOT auto-modify the layered-tuning projection as part of this spec (that is a separate, downstream decision), relying on implementation discipline rather than an active blocking mechanism.
4. WHERE a layered-tuning projection update becomes available, THE system MAY emit an automated notification or alert to surface that the update is available, provided the notification itself does NOT modify the layered-tuning projection.
5. WHEN competitor code reveals signals absent from our "layers" framing THEN the system SHALL record these as hypotheses to test, explicitly NOT as verified improvements, until an external judge confirms them.
6. WHEN this spec conflicts with a persuasive narrative THEN the reverse-engineering-accountability contract SHALL govern (external judge only; honest nulls are results; no extrapolation; state the uncertainty).
