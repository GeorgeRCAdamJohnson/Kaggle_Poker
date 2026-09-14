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

## Verified leaderboard ladder (external judge = Kaggle LB)
- Phase-1 classical floor: 0.44519 (v5 event detectors + MFg + DIRc)
- honghanh reproduction: 0.64262 (PU-aware value-transfer blend)
- Evidence fix (A): 0.65242 (pure-ranker evidence on honghanh, +0.0098)
- lamhuy V30 reproduction: 0.67838 (triple-GBDT + decoupled OvR + dual-engine evidence) BEST

## The load-bearing lesson
The recurring limiter was FEATURE-TRANSPORT FIDELITY, not feature discovery: strong
dev-CV gains repeatedly collapsed on the leaderboard because a dev feature did not mean
the same thing on eval (pool-prior clamp, R2 parity bug, drifting ranks, the graph
parity break). Every feature block must be built by ONE shared dev/eval function and
pass an adversarial drift gate — mean-matching is NOT sufficient, since rank structure
can diverge while means match. See RESEARCH_DOSSIER sections 89-94.

## Layout
- anchor_repro/ — competitor-recipe reproductions, drift gate, scorer, graph retest, consensus/triangulation, evidence sweep.
- poker_collusion/ — the core pipeline package (metric, CV, features, models, tuning harness).
- tests/ — unit + property + integration tests (metric parity, drift gate, tuner guarantees).
- specs/ — the full .kiro methodology: requirements, design, tasks, and the RESEARCH_DOSSIER.

## NOT included (by design)
- data/ — competition data (not ours to redistribute; download from Kaggle).
- outputs/ — regenerable caches, models, and submission CSVs.
- _refs/ — competitor notebooks (their copyright).
- credentials of any kind.

## Reproducing
Install deps (xgboost, lightgbm, catboost, polars, pandas, scikit-learn), place the
competition data under data/poker/, then run the recipe modules under anchor_repro/.
GPU (xgboost device=cuda) is supported and recommended for the triple-GBDT retrains.
