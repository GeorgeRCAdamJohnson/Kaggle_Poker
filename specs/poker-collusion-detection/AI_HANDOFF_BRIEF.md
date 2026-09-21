# AI Handoff Brief — Poker Collusion Detection Wall

**Purpose:** Give another AI model (e.g. a high-grade model in Copilot) everything it needs to
attack our current wall WITHOUT wasting effort on the dozens of things we already tried and killed.
Read this whole document before proposing anything. Then use the prompt templates at the end.

**Last updated:** 2026-09-15. Source of truth for all numbers: `RESEARCH_DOSSIER.md` §1, §6, §40–§114.

---

## 1. The competition in one paragraph

Kaggle-style competition: **"Detect Suspicious Value Transfers in Poker."** We are given a large
synthetic poker dataset (400 tables × 30 players × 5,000 hands = 60M+ hands worth of action logs,
~18.6M individual actions). A small set of player **pairs** are secretly colluding. We must output,
for every one of **112,540 evaluation pairs**, three things:

1. **`risk_score`** ∈ [0,1] — how likely the pair is colluding (drives the ranking metric).
2. **`predicted_behavior`** — which collusion family (`directed_transfer`, `soft_play`,
   `coordinated_isolation`, or `other_coordination` / `none`).
3. **`evidence_hand_1..5`** — the 5 specific hands that best show the collusion.

## 2. The exact scoring metric (CONFIRMED against the official kernel)

```
final = 0.70 · PairAP  +  0.20 · EvidenceMAP@5  +  0.10 · BehaviorMAP
```

- **PairAP (70%)** — Average Precision over ALL 112,540 pairs ranked by `risk_score` descending,
  against the hidden binary "is this pair colluding" truth. **This is the whole game.** Everything
  else is worth 30% combined.
- **EvidenceMAP@5 (20%)** — only scored over TRUE colluding pairs; for each, how many of our top-5
  evidence hands match the planted evidence hands. Denominator = `min(#planted, 5)`.
- **BehaviorMAP (10%)** — one-vs-rest AP over the three disclosed families only.

Full clause-by-clause detail is in `RESEARCH_DOSSIER.md` §1. Key gotchas: `risk_score` must be in
[0,1] (no clipping — out of range is rejected), every eval pair must be scored exactly once, ties
break by `pair_id` ascending.

## 3. Where we actually are (the numbers that matter)

| Milestone | Public LB | Note |
|---|---|---|
| Do-nothing baseline | 0.08404 | sample submission |
| Our old plateau (cand_18) | 0.39372 | 3-head XGB, 4 days stuck here |
| PU-aware reproductions (nomannic/honghanh) | 0.56–0.65 | reproduced strong public notebooks |
| lamhuy V30 reproduction | 0.67838 | faithful re-run of a top public notebook |
| **OUR CURRENT BEST** | **0.70904** | `candidate_v30_hnm_sequence_confirmation_rank_sum_w50.csv` — V30 risk + ordered-sequence family specialists (rank-sum, seed137) |
| 0.709 sequence-behavior variant | 0.70896 | near-identical; our 2nd-best |
| **The field's top scores** | **~0.79–0.93** | this is the wall we cannot cross |

**The wall: we are stuck at ~0.709. The leaders are at ~0.79 (public) up to ~0.89–0.93 (private).**
We need roughly **+0.08 to +0.18** and nothing we have tried in the last ~70 experiments crosses it.

## 4. WHY we are stuck (the single most important insight — do not skip)

This is a **synthetic dataset with a Positive-Unlabeled (PU) transfer problem**, and it has burned
every naive approach:

### 4a. The dev-selection bias (the root cause, MEASURED in §110)
- We have ~1,860 **labeled** development pairs (372 positive). We must score 112,540 **eval** pairs,
  almost all unlabeled.
- The labeled positives are **~3.5× denser in collusion** than the eval population. The generator
  seeded colluders densely into the labeled tables; eval is sparse.
- **Consequence:** any detector trained/validated on dev CV massively over-reads. We measured a
  detector at **dev-CV PairAP 0.90** whose **eval-honest PairAP was only ~0.61.** A ~0.30 illusion.
- We built an **eval-honest harness** (importance-weighted AP via a dev-vs-eval density-ratio
  classifier). **ANY new idea MUST be scored on this harness, not raw dev CV.** Raw dev CV lies here.

### 4b. "More complexity / more CV" REGRESSES on the LB — repeatedly
Every one of these raised dev CV and LOWERED the real LB:
- deeper GBT risk head; 3-way over-correlated ensembles; deeper/longer evidence rankers (d5, it500);
  PU-weight retuning; rich feature stacks (cand_23 → 0.369, below our own floor).
- **Rule for this dataset: modest complexity + model diversity generalizes; chasing CV overfits.**

### 4c. Discrete "unlock" hypotheses — ALL tested, ALL dead
We hunted for a deterministic generator signature. None exists in the public data:
- **Exact bet/pot/stack ratio fingerprints** (§113): planted hands bet the SAME sizes as normal
  hands (bet-pot-fraction AUC 0.5005). No ratio-lock. Dead.
- **AAAI joint-unit advantage** (§112): INVERTED here — colluding pairs do WORSE as a unit, because
  collusion is internal chip TRANSFER, not unit profit. Eval-honest AP 0.37. Dead as framed.
- **Pairwise consistency detector** (§110): eval-honest 0.61 < V30's 0.79. Real signal, but below
  what we already have. Not promotable.
- **Graph features, seat rotation, temporal concentration, metadata leaks:** all tested, all
  saturated or refuted (§2, §34, §1540).

### 4d. What the collusion actually IS (banked, confirmed)
- `directed_transfer`: one member loses a big pot to the other (directional net chip flow). LARGEST
  family and the one our models score best. The real population-robust tell is **directional funding
  — one member repeatedly wins while the partner loses in the same hands.**
- `soft_play`: both players passively check/call down, muted aggression toward each other.
- `coordinated_isolation`: both raise to squeeze out a third player, one collects.
- The tell is in **WHO wins/loses and the passivity structure**, not in special bet sizes.

## 5. What is GENUINELY still unexplored (the only fresh routes)

From the §114 persona crosswalk, exactly THREE paradigms are structurally different from everything
tried and remain live candidates. Everything else is already tested or a rebrand.

1. **Decision-policy counterfactual.** Not another aggregate. Model each player's decision at a
   specific action state (given partner raised, folder had board-relative strong hand, given
   stack/pot/position) and ask: *how abnormal is this action versus that same player's learned field
   policy?* We built board-equity surrender aggregates before; we never built the true conditional
   policy counterfactual.

2. **Victim-centric counterfactual.** Model a specific OUTSIDER's loss / fold / EV shift when a
   candidate pair is present and acting, versus that outsider's matched field baseline. Distinct
   from raw outsider-fold counts and from joint-unit advantage. Strongest non-sequence candidate.

3. **Raw sequence representation OR formal rule mining.** Either (a) self-supervised representation
   learning on all 18.6M raw actions (role/action/size/street tokens) → small MIL/pair heads, or
   (b) explicit program/decision-list/rule mining on the 1,817 evidence hands. We only ever built
   hand-engineered ordered-sequence AGGREGATES, never an end-to-end learned representation or a
   formal rule search. Caveat: labeled data is small (372 pairs), so a supervised transformer alone
   is likely data-starved — the defensible version is self-supervised first.

Two cheap descriptive probes worth doing before expensive models:
- Evidence-count distribution: `evidence_count ÷ shared_hands` by family/table — is there a generator
  planting-rate constant, or is it just top-5 annotation truncation?
- Cross-family hand signatures within a single labeled family — are "pure" family labels actually
  mixtures?

## 6. HARD CONSTRAINTS any proposal must respect

1. **No private / hidden information.** Nothing may read the hidden target set, planted-evidence
   hands, private-LB labels, or the public/private split. Anything requiring these is discarded on
   sight, not logged.
2. **Eval-honest gate is mandatory.** Every candidate is measured on the importance-weighted harness
   (§110), NOT raw dev CV. A dev-CV win is not evidence of anything here.
3. **One change per submission.** We have a limited daily submission budget. No arm-hopping.
4. **PairAP is 70% of the score.** Evidence and behavior micro-gains are largely exhausted; the
   payoff is in the risk ranking on the UNLABELED eval population.
5. **Beat 0.70904 or it doesn't ship.** A candidate must clear the current best on an eval-honest
   holdout before it's worth a submission.

## 7. The honest possibility

It is genuinely possible we are near the ceiling of publicly-derivable signal for the approaches
tried, and the ~0.89 field either has the generator's parameters or a modeling paradigm we haven't
found. A new model's job is to either (a) find a way to make one of the three fresh paradigms in §5
beat 0.709 eval-honestly, or (b) confirm the ceiling with a rigorous argument. Both are valuable.
Do NOT accept a proposal that just re-runs a killed idea with new words.

---

## 8. HOW TO PROMPT ANOTHER AI MODEL (Copilot high-grade models)

### 8a. General principles for prompting on THIS problem
- **Give it this whole brief as context.** Without it, every model re-proposes deeper GBTs, more
  features, and bigger ensembles — all of which are in our graveyard (§4b).
- **Force it to classify each idea as: already-tested / rebrand / genuinely-new.** Make it check
  against §4 and §5 before proposing.
- **Demand an eval-honest validation plan**, not a dev-CV number.
- **Ask for ONE concrete, buildable experiment**, not a menu of ten. We iterate one change per submit.
- **Require it to state the expected eval-honest PairAP and why it would transport** (population
  robustness), since that is exactly what kills ideas here.
- **Reject any proposal that needs hidden info** (§6.1).

### 8b. Copy-paste prompt template — "cold review, find the fresh route"

```
You are a Kaggle-grandmaster-level ML engineer. I'm stuck on a synthetic poker collusion-detection
competition. My current best public LB is 0.70904; the field leaders are at ~0.79–0.93. I need to
close that gap or rigorously confirm I'm at a ceiling.

CRITICAL: I have already tried and KILLED most obvious ideas. Before proposing ANYTHING, read the
attached brief (AI_HANDOFF_BRIEF.md) and for every idea you have, classify it as:
  (a) ALREADY-TESTED — matches something in section 4 of the brief; discard it.
  (b) REBRAND — a renamed version of a tested idea; discard it.
  (c) GENUINELY-NEW — structurally different from everything in sections 4 and 5.

Only (c) ideas are allowed. The known-fresh routes are in section 5 — you may build on those or
propose something equally novel, but you must justify why it's not already covered.

The core problem is a dev-selection / PU transfer bias (section 4a): dev CV lies by ~0.30 of PairAP.
Any idea you propose MUST come with an eval-honest validation plan (importance-weighted, not raw
dev CV) and a clear argument for WHY the signal would transport to the sparse unlabeled eval
population.

Constraints: no hidden/private info; PairAP is 70% of the metric; must beat 0.70904 eval-honestly;
one change per submission.

Deliver EXACTLY ONE concrete, buildable experiment: the feature/model, the exact construction, the
eval-honest validation, the expected eval-honest PairAP, and the failure mode that would kill it.
No menus. No hedging. If your honest conclusion is that I'm at the ceiling, say so and prove it.
```

### 8c. Copy-paste prompt template — "attack one specific fresh route"

Use this when you want depth on one of the three §5 paradigms. Swap in the paradigm.

```
Context: attached AI_HANDOFF_BRIEF.md. My best is 0.70904; the core obstacle is dev-selection bias
that makes dev CV over-read by ~0.30 PairAP (brief section 4a). I want to build ONE specific idea:

  [PASTE ONE OF:]
  - Decision-policy counterfactual: for each player action, how abnormal is it vs that same player's
    learned field policy, conditioned on action state (partner raised, board-relative hand strength,
    stack/pot/position)?
  - Victim-centric counterfactual: an outsider's loss/fold/EV shift when a candidate pair is present
    and acting, vs that outsider's matched field baseline.
  - Self-supervised sequence representation on 18.6M raw actions → small MIL/pair head.

Design the full build: exact feature/model construction from the raw action log; how to avoid
fitting dev-selection or annotation mechanics instead of the planting rule; the eval-honest
importance-weighted validation; the expected eval-honest PairAP and why it transports; and the
specific result that would prove it a dead end. Be concrete enough that I can implement it directly.
```

### 8d. What a GOOD answer looks like (and a BAD one)
- **GOOD:** names a specific action-state conditional, explains why it's population-robust (doesn't
  depend on dev collusion density), gives an eval-honest validation, predicts a number, and names
  its own failure mode.
- **BAD:** "train an XGBoost/LightGBM ensemble on more features with hyperparameter tuning and
  stratified CV." That is our graveyard (§4b). If a model says this, it did not read the brief.

---

## 9. Quick-reference: the graveyard (do not let any model resurrect these)

deeper GBT risk · 3-way correlated ensembles · deeper/longer evidence rankers (d5/it500) · PU-weight
tuning · rich feature stacks · exact bet/pot/stack ratio fingerprints · AAAI joint-unit advantage
(as unit-profit) · graph features on confirmed-label graphs · seat-rotation co-seating signal ·
temporal-concentration burst · metadata leaks (region/client/account-age — already model inputs) ·
generic family aggregates · raw consistency rates (eval-honest 0.61, below current best).

**Anything on this list is dead. A fresh idea must not be one of these in disguise.**
