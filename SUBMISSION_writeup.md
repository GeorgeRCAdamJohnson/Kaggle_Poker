# Detecting Coordinated Value Transfers in Poker — Solution Writeup

**Final selected submission:** `evidence_familycond_submission.csv` — public LB **0.83460**.
**Metric:** `0.70·PairAP + 0.20·EvidenceMAP@5 + 0.10·BehaviorMAP`.

All signals are inferred from gameplay only — actions, contributions, showdowns, and hand strength. No ID
formats, row/file ordering, or generator internals are used.

## 1. Framing

The task decomposes into three heads scored jointly: rank pairs by collusion risk (PairAP, 70%), retrieve up
to five evidence hands per flagged pair (EvidenceMAP@5, 20%), and label the behavior family (BehaviorMAP,
10%). Two facts shaped every decision:

- **Supervision is tiny and one-sided.** Only 1,860 trusted labels (372 positive pairs, 1,488 negatives)
  over ~122k candidate relationships. Unlabeled pairs are *not* negatives — we treat them as unlabeled and
  use positive-unlabeled (PU) training, never as a hard-negative loss.
- **Development and evaluation are different populations.** A signal that separates labeled positives on dev
  can still fail on eval if it keys on population composition rather than coordination. We adopted an
  **eval-weighted, table-disjoint validation gate** (adversarial dev-vs-eval reweighting on held-out table
  folds) as the trustworthy predictor of leaderboard movement, and rejected any feature whose gain came with
  dev-vs-eval drift (adversarial AUC ≥ 0.65).

## 2. Risk head (PairAP)

The risk model is a two-stage PU classifier (LightGBM) over three complementary feature groups, all
partner-vs-field normalized so a feature measures a *pair-specific* deviation, not one loose player:

1. **Base pair features** — shared-hand rates of both-vpip, both-to-showdown, directed net chip flow,
   fold/call/raise-vs-partner contrasts, showdown hand-strength gaps, and outsider-directed pressure.
2. **Policy-deviation channels (PT)** — a card-conditional entry policy `P(vpip | hole_class, position,
   facing_raise)` learned over ~12M seats; a pair that *both* enter or fold against the policy in a directed,
   partner-specific way is the core collusion tell. Extended with a field-normalized postflop-surprise and a
   "tight" channel (folding hands the policy says to play — the -EV surrender-to-partner deviation).
3. **Learned sequence risk (seq_risk)** — a masked-action Transformer pretrained self-supervised on the raw
   ordered action stream of every pair "document," then distilled into a single risk column. This captures
   micro-structure in the betting sequence that engineered aggregates miss, and was decorrelated from the
   engineered channels (adding it cleared the eval-weighted gate by >4× the noise band).

Stage 1 trains on trusted labels with class weighting; a PU relabeling step promotes high-confidence
unlabeled pairs to positive with a graded weight; stage 2 retrains on the augmented set. Cross-validation
splits **whole tables** so no player appears in both train and validation.

## 3. Evidence head (EvidenceMAP@5)

Evidence is a *within-pair* ranking (which of a flagged pair's shared hands are the planted evidence), which
makes it base-rate-free and phase-drift-immune — we verified 1:1 dev→leaderboard transfer, so it needs no
drift gate, only honest cross-validated MAP@5.

Our key gain here was **family-conditional ranking**. The three families leave mechanically distinct per-hand
signatures: directed_transfer is a net directed chip flow via a value-committing action; soft_play is
declined partner-directed aggression; coordinated_isolation is *joint* pressure on a third player. A single
generic ranker must average over these, which suppresses the family whose evidence looks least like the
others. We instead route each pair's hands to one of three per-family LightGBM rankers (matched to the pair's
predicted family), trained on `is_evidence` over per-hand behavioral features plus a **contextualized
sequence embedding** — each hand embedded by mean-pooling the trained encoder's per-token hidden states, so
a hand is represented in the context of the pair's whole session, not in isolation.

This lifted OOF MAP@5 from **0.5276** (engineered baseline) → **0.5710** (contextualized per-hand embedding)
→ **0.5989** (family-conditional routing). The largest single gain was on coordinated_isolation (+0.094),
whose joint-third-player evidence hands look nothing like the other families' and were being averaged away.

## 4. Behavior head (BehaviorMAP)

A family classifier trained on the 372 confirmed targets over the same pair features, applied to every pair,
with a prior-normalized argmax. Family accuracy on confirmed targets is high, so routing (used by both the
behavior head and the evidence ranker) is reliable.

## 5. What we tried that did *not* help (honest negatives)

We ran a disciplined "gate-or-reject" loop and rejected many plausible ideas because they failed the
eval-weighted transfer test rather than because they looked bad on dev: graph/network topology features (a
degree signal that hit AUC ~1.0 on dev but was a labeled-set selection artifact that inverts on eval);
timing-regularity of planted episodes (label-only, unreconstructable from eval-available features); a
domain-adversarial (gradient-reversal) risk encoder; multi-entity table context; and several equity- and
joint-advantage-based risk features whose per-hand information proved redundant with the existing features or
drifted across phases. A listwise `rank:map@5` evidence objective did not beat binary classification on our
feature basis. Recording these as measured nulls kept us from shipping dev-overfit signal.

## 6. Reproduction

The full reproduction lives in the linked code repository (`REPRODUCTION.md` for the staged pipeline,
`REPRODUCE_83460.ipynb` for a byte-verified regeneration, `anchor_repro/validate_submission.py` for schema
checks). This Kaggle notebook is the writeup itself, not the runnable pipeline. From the released files, the
repository regenerates the selected submission in five stages:

1. one pass over the action log to build per-hand pair features (Polars);
2. the card-conditional entry-policy tables and field baselines;
3. the masked-action sequence encoder (pretrain + distill) → `seq_risk`;
4. the two-stage PU risk model, the family classifier, and the three family-conditional evidence rankers;
5. assembly into `submission.csv`, with validity checks (schema, `pair_id` set, `risk_score ∈ [0,1]`, no
   duplicate evidence hands within a pair).

Cross-validation is table-disjoint throughout; the risk gate uses eval-weighted held-out folds. A full run
is a few hours on a single GPU (encoder) plus CPU (LightGBM). The trained encoder checkpoint is shipped in
the repository's `artifacts/`, so the encoder-training stage can be skipped and the submission regenerated in
minutes; the regenerated file is byte-identical (SHA256 prefix `2ba41953a89fa752`).

## 7. Compliance

Every feature is a function of gameplay: actions, bet sizes, contributions, showdown holdings, and hand
strength. We use no ID structure, no row/file ordering, and no generator internals. Candidate signals that
would have depended on such artifacts (e.g., the degree/selection artifact in §5) were explicitly identified
and discarded.

## 8. Limitations

Ranking a 0.2%-prevalence positive class to the very top of a sparse population is the residual difficulty:
we can characterize what makes colluders different, but concentrating the evidence ranking onto exactly the
handful of planted hands per pair — versus the larger set of hands that merely satisfy the necessary
conditions — is where headroom remains. We believe the strongest further gain is a *joint within-pair
set-selection* evidence model that scores all of a pair's candidate hands together rather than independently.
