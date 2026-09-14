# poker_collusion

Pipeline for the Kaggle competition **"Detect Suspicious Value Transfers in Poker"**.

Detects coordinated player relationships (collusion) across ~2,000,000 synthetic
six-player No-Limit Hold'em hands, ranks evaluation pairs by suspicion, predicts
the coordination behavior, retrieves up to 5 supporting evidence hands per pair,
and writes a `submission.csv` matching `sample_submission.csv` exactly.

## Package layout

```
poker_collusion/
  io/          # Data_Loader: chunked, column-projected reads + joins
  features/    # EDA/audit, hand-level signals, episodic aggregation, pair features
  models/      # classical baselines, PU ranking model, behavior classifier
  evidence/    # evidence retrieval + ranking
  metric/      # three-part competition metric re-implementation
  submission/  # submission writer + self-validation
  validation/  # local CV harness (group-by-pool, family-stratified, PU-correct)
  discovery/   # Phase -1 data access, forensics, dossier tooling
  tests/       # package-internal tests
  config.py    # central seeded configuration (paths, seeds, columns, versions)
  types.py     # core data models (Pair, HandSignal, PairFeatureSet, ...)
  exceptions.py# SchemaError(file, missing_key)
tests/         # top-level test suite (property-based + integration)
configs/       # experiment configuration files
```

## Install

```powershell
python -m pip install -r poker_collusion/requirements.txt
```

## Reproducibility

All seeds are fixed in `poker_collusion/config.py` (`Seeds`). Schema and threshold
versions are recorded so feature generation and scoring are reproducible for
winner verification.
