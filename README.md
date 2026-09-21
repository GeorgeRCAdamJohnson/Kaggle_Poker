# Kaggle_Poker — Detect Suspicious Value Transfers in Poker

Full methodology + code for the Kaggle competition
**detect-suspicious-value-transfers-in-poker** (detecting collusive value-transfer
dyads in 6-max No-Limit Hold'em from hand/action/seat logs).

This repo is deliberately transparent: it ships the complete research trail, not just
the winning code. The governing philosophy is in
`specs/poker-collusion-detection/RESEARCH_DOSSIER.md` — an append-only,
externally-auditable log of every pre-registered experiment, its locked prediction,
and its leaderboard verdict.

## The metric
Leaderboard score = 0.70*PairAP + 0.20*EvidenceMAP@5 + 0.10*BehaviorMAP.

## Selected submission — how to reproduce & verify
- **Selected submission:** `evidence_familycond_submission.csv`, public LB **0.83460**.
- **Reproduction:** see [`REPRODUCTION.md`](REPRODUCTION.md) for the full raw-files -> submission pipeline
  (staged A/B/C), environment, and the fast path via the shipped encoder in `artifacts/`.
  [`REPRODUCE_83460.ipynb`](REPRODUCE_83460.ipynb) regenerates the CSV and byte-checks it.
- **Verify:** `python -m anchor_repro.validate_submission outputs/poker_collusion/evidence_familycond_submission.csv`
  (schema + range + no-dup-evidence checks; expected SHA256 prefix `2ba41953a89fa752`).
- **Solution writeup:** [`SUBMISSION_writeup.md`](SUBMISSION_writeup.md) (<=1,500 words).
- **Five case reviews:** [`SUBMISSION_case_reviews.md`](SUBMISSION_case_reviews.md) — pair ID, evidence
  hand ID(s), observable suspicious behavior, and a plausible benign alternative per case.

## Verified leaderboard ladder (external judge = Kaggle LB)
- Phase-1 classical floor: 0.44519
- Policy-deviation risk (card-conditional entry surprise, partner-vs-field): 0.80422 -> 0.81092
- Learned sequence encoder (masked-action Transformer, distilled seq_risk): 0.82082
- Causal-ensemble risk: 0.82930
- Contextualized per-hand evidence embedding: 0.83185
- **Family-conditional evidence (3 per-family rankers, routed): 0.83460 — BEST/selected**

Risk saturated near 0.829; the final gains came from the evidence head, matching the metric arithmetic
(EvidenceMAP@5 was the largest remaining lever).

## The load-bearing lesson
The recurring limiter was FEATURE-TRANSPORT FIDELITY, not feature discovery: strong dev-CV gains repeatedly
collapsed on eval because a dev feature did not mean the same thing on the evaluation population. Every
feature block is built by ONE shared dev/eval function and must pass an adversarial dev-vs-eval drift gate;
mean-matching is not sufficient because rank structure can diverge while means match. Evidence ranking is a
within-pair (base-rate-free) problem and is phase-immune, so it transfers 1:1. See the RESEARCH_DOSSIER for
the full pre-registered experiment trail and honest negatives.

## Compliance
All signals infer coordination from gameplay only — actions, bet sizes, contributions, showdown holdings,
board, and hand strength. No ID formats, row/file ordering, or generator internals are used; candidates that
would have relied on such artifacts were identified and rejected (see writeup + dossier).

## Layout
- anchor_repro/ — pipeline code: risk/evidence/behavior heads, sequence encoder, drift gate, scorer, submission assembly, `validate_submission.py`.
- poker_collusion/ — the core pipeline package (metric, CV, features, models, tuning harness).
- artifacts/ — small pretrained weights (`_v2_encoder.pt`, `seq_vocab.json`) so the encoder training stage can be skipped for a fast verify.
- tests/ — unit + property + integration tests (metric parity, drift gate, tuner guarantees).
- specs/ — the full .kiro methodology: requirements, design, tasks, and the RESEARCH_DOSSIER (append-only experiment log through §225).

## NOT included (by design)
- data/ — competition data (not ours to redistribute; download from Kaggle).
- outputs/ — regenerable caches, models, and submission CSVs.
- _refs/ — competitor notebooks (their copyright).
- credentials of any kind.

## Reproducing
See [`REPRODUCTION.md`](REPRODUCTION.md) for the authoritative step-by-step. In short: install
`anchor_repro/requirements.lock`, place the competition files under `data/poker/`, then run the staged
pipeline (features -> sequence encoder -> risk/behavior/evidence -> family-conditional swap). The trained
encoder is shipped in `artifacts/` so the GPU training stage can be skipped for a fast reproduction of the
selected submission.
