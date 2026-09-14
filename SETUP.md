# SETUP — bootstrapping a second machine (laptop / desktop)

GitHub holds the code + methodology; each machine keeps its own local `data/` and
`outputs/` (gitignored, regenerable). This is the exact sequence to stand up a fresh
machine so it reproduces the verified results.

## 1. Clone
```
git clone https://github.com/GeorgeRCAdamJohnson/Kaggle_Poker.git
cd Kaggle_Poker
```

## 2. Python + dependencies
Target interpreter: **Python 3.13** (verified on 3.13.14).
```
python -m pip install -r requirements.txt
```
On Python 3.13, lightgbm/catboost need prebuilt wheels (source builds fail). If the
line above errors on either, force binary wheels:
```
python -m pip install --only-binary=:all: lightgbm catboost
```
GPU: xgboost `device="cuda"` is used for the triple-GBDT retrains. It needs a CUDA GPU
+ matching driver. If none is present the code falls back to CPU `tree_method="hist"`
automatically (a small speed/RAM cost, identical methodology — note CPU-hist vs GPU-hist
give tiny tree differences, so model artifacts are not byte-identical across machines).

NOTE on pyproject.toml: its pins are HISTORICAL (older numpy/pandas, lists torch which
the poker recipes do NOT use, omits the GBDT libs). Use `requirements.txt` for the
poker pipeline — those are the versions that produced the verified LB ladder.

## 3. Competition data (NOT in the repo)
Download the competition data from Kaggle and place it under `data/poker/` so these
8 files exist:
```
data/poker/players.parquet
data/poker/hands.parquet
data/poker/seats.parquet
data/poker/actions.parquet
data/poker/development_labels.csv
data/poker/development_evidence.csv
data/poker/evaluation_pairs.csv
data/poker/sample_submission.csv
```
(Or copy them from the local backup at `D:\Poker\data\poker\`.)

## 4. Kaggle credentials (per-machine, never committed)
Put your `kaggle.json` API token in the OS-standard location (`%USERPROFILE%\.kaggle\`
on Windows / `~/.kaggle/` on Linux). Used for submissions and kernel pulls. It is
gitignored and must be set up independently on each machine.

## 5. Run
- Reproduce the best base (lamhuy V30, LB ~0.678): `python -m anchor_repro.lamhuy_recipe`
- Component decomposition / evidence sweep / graph retest / consensus: the other
  `anchor_repro/*.py` modules (each writes to `outputs/poker_collusion/...`).
- Tests: `pytest` (integration tests auto-skip if data/caches are absent).

## 6. Multi-machine hygiene (keep drift minor)
- `git pull` before you start; `git commit && git push` when you stop.
- The one file prone to conflict is `specs/poker-collusion-detection/RESEARCH_DOSSIER.md`
  — it is append-only, so conflicts (if any) are always at the tail: keep both sections
  and renumber. Never edit sections mid-file.
- `data/` and `outputs/` diverge per machine by design (regenerable). Don't expect
  byte-identical caches across a GPU machine and a CPU machine.
