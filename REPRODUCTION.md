# Reproduction — Detect Suspicious Value Transfers in Poker (LB 0.83460)

This document reproduces our selected submission `evidence_familycond_submission.csv` (public LB **0.83460**)
from the released competition files. Everything infers coordination from gameplay only.

## 0. Environment

- Python 3.11+ (developed on 3.14), single machine. A CUDA GPU is used only for the sequence encoder
  (pretrain + embed); everything else is CPU.
- Install pinned dependencies:
  ```bash
  pip install -r anchor_repro/requirements.lock
  ```
  Key packages: `polars`, `pandas`, `numpy`, `scikit-learn`, `lightgbm` (CPU), `xgboost`,
  `torch` (CUDA build matching your GPU; CPU works but the encoder step is slow).

## 1. Data layout

Place the released competition files under `data/poker/`:
```
data/poker/hands.parquet  seats.parquet  actions.parquet  players.parquet
           development_labels.csv  development_evidence.csv  evaluation_pairs.csv  sample_submission.csv
```

## 2. Pipeline stages (raw -> submission)

Run in order. Each stage writes caches consumed by the next; intermediate caches are large (~12 GB total)
and are regenerated here rather than shipped.

### Stage A — base pair/hand features and the policy-deviation risk stack
Builds the `hosen42_step3/` caches (pair features, per-hand features, pair-hands) and the entry-policy
tables, then the two-stage PU risk model with the policy-deviation (PT) channels.
```bash
python -m _refs.hosen42_policy.run_policy          # -> outputs/poker_collusion/hosen42_step3/*.parquet
python -m anchor_repro.build_pvf_submission        # policy-deviation + postflop + tight risk channels
```

### Stage B — the learned sequence encoder (GPU)
Tokenize each pair's ordered action document, pretrain the masked-action Transformer, fine-tune on the 372
labels, distill to a `seq_risk` column, and export the contextualized per-hand embeddings.
```bash
python -m anchor_repro.seq_tokenize                # seq_tokens_{dev,eval}.parquet + seq_vocab.json
python -m anchor_repro.seq_encoder_v2              # trains _v2_encoder.pt  (GPU; hours)
python -m anchor_repro.seq_ctx_evidence            # seq_ctx_emb_{dev,eval}.parquet
python -m anchor_repro.seq_v2_assemble             # -> seq_v2_submission.csv (risk stack, LB ~0.82082)
```
The trained checkpoint `_v2_encoder.pt` (6 MB) and `seq_vocab.json` are provided in `artifacts/` so this
stage can be skipped for a faster verify: drop them into
`outputs/poker_collusion/hosen42_step3/edge/seq/` and go straight to Stage C.

### Stage C — evidence upgrade and the family-conditional swap (the 0.83460 submission)
Adds the contextualized per-hand embedding to the evidence ranker (-> 0.83185 base), then routes each pair's
hands to per-family rankers and swaps the evidence columns in — keeping risk + behavior byte-identical.
```bash
python -m anchor_repro.seq_evidence_submit         # -> seq_evidence_submission.csv (LB 0.83185)
python -m anchor_repro.evidence_familycond_submit  # -> evidence_familycond_submission.csv (LB 0.83460)
```

## 3. Verify

```bash
python -m anchor_repro.validate_submission outputs/poker_collusion/evidence_familycond_submission.csv
# schema: 8 columns; 112,540 unique pair_ids; risk_score in [0,1];
# predicted_behavior in {none,directed_transfer,soft_play,coordinated_isolation};
# no duplicate/null evidence hand within a pair; prints SHA256 prefix.
```
Expected SHA256 prefix of the final CSV: `2ba41953a89fa752` (byte-identical regeneration).

## 4. What comes from where (audit)

- Risk = two-stage PU LightGBM over base pair features + policy-deviation (PT) channels + distilled
  `seq_risk`. Table-disjoint CV; eval-weighted held-out gate.
- Behavior = family classifier on the 372 confirmed targets, prior-normalized argmax.
- Evidence = three per-family LightGBM rankers over per-hand behavioral features + contextualized sequence
  embeddings, routed by predicted family; within-pair ranking (phase-immune).
- The final swap changes only `evidence_hand_1..5`; `risk_score` and `predicted_behavior` are asserted
  identical to the 0.83185 base.

## 5. Compliance

No feature uses ID formats, row/file ordering, or generator internals. Rejected candidates that would have
relied on such artifacts (e.g. a graph-degree signal that was a labeled-set selection artifact) are
documented and excluded. All inputs are gameplay: actions, bet sizes, contributions, showdown holdings,
board, and hand strength.

## Files
- `anchor_repro/` — all pipeline code
- `anchor_repro/requirements.lock` — pinned dependencies
- `artifacts/_v2_encoder.pt`, `artifacts/seq_vocab.json` — pretrained encoder (skip Stage B training)
- `REPRODUCE_83460.ipynb` — notebook that runs Stage C from prebuilt caches + byte-checks the output
- `SUBMISSION_writeup.md`, `SUBMISSION_case_reviews.md` — the required writeup and five case reviews
