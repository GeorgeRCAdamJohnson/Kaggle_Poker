"""FIELD-RELATIVE (residual) equity-surrender features — §220 ground-up rebuild.

DIAGNOSIS (§220, verified against _gate.log/_diag2.log): the original equity-surrender features
(equity_features.py) carry REAL non-redundant signal (+0.0084 incremental AUC, standalone AUC 0.79)
but were expressed as ABSOLUTE per-pair counts/sums/levels. Absolute activity magnitude encodes the
phase-specific population composition -> adversarial dev-vs-eval AUC 0.716 (drifts hard) -> killed as an
add-on (eval-weighted delta -0.0135).

FIX: express surrender as a FIELD-RELATIVE RESIDUAL. For each surrender-opportunity hand-event we model
the field's baseline P(surrender) from context a non-colluding pair would share (equity level, pot size,
blinds/stakes, showdown structure, aggression, live players). A pair's feature = observed surrenders MINUS
expected surrenders, summed/averaged over its hands. The residual subtracts the field baseline, so the
phase-level shift cancels by construction -> should be phase-invariant (drift < 0.65, target < 0.55) while
retaining the collusion signal (colluders surrender MORE than the field expects given identical context).

The null model is fit on ALL pairs' hand-events POOLED (the field), NOT per-pair, and is trained WITHOUT
any pair identity or label -> it learns "how often does anyone surrender in this situation." Colluders'
positive residual is exactly the excess the audit predicts.

Outputs per phase:
  edge/equity/field_relative_{phase}.parquet  with per-pair residual features (FEATS_FR).
Run:  python -m anchor_repro.equity_field_relative --phase development
      python -m anchor_repro.equity_field_relative --phase evaluation
"""
from __future__ import annotations
import argparse, time, warnings, numpy as np, polars as pl, lightgbm as lgb
warnings.simplefilter("ignore")
from pathlib import Path
STEP3 = Path("outputs/poker_collusion/hosen42_step3"); EQ = STEP3/"edge"/"equity"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
T = 0.65  # equity threshold matching equity_features.py "strong hand"

# per-hand columns needed to define surrender EVENTS + the field null-model context
HF_COLS = ["pair_id","hand_id","table_id","big_blind","pot_bb","players_at_showdown",
           "p1_folded","p2_folded","p1_won_share","p2_won_share","p1_net_bb","p2_net_bb",
           "p1_n_aggr","p2_n_aggr","p1_went_to_showdown","p2_went_to_showdown"]

# null-model context features (what a non-colluding pair shares; NO pair id, NO label, NO absolute-count)
CTX = ["m_eq","partner_eq","pot_bb","big_blind","players_at_showdown","live_bucket",
       "m_n_aggr","partner_n_aggr","m_went_sd","partner_went_sd"]

# NOTE (§220): raw residual sums drift jointly (0.68) purely from EXPOSURE variance — dev pairs play more
# hands (n_opp 15.9 vs eval 11.5), so summed residuals accumulate more variance in dev even though the
# residual DIRECTION is phase-invariant (means match to 4 decimals). Fix per drift diagnosis: EXPOSURE-SHRINK
# (empirical-Bayes) toward 0, which variance-stabilizes. At k=100 the joint adversarial drift is 0.615 (<0.65,
# GATE 1 pass). These EB-shrunk residuals are the SCORED set. Raw sums + counts kept as AUX (reference only).
EB_K = 100.0
FEATS_FR = [
    "frs_eb",     # resid_sum / (n_opp + k)  : empirical-Bayes shrunk mean residual (surrender excess vs field)
    "frs_z",      # resid_sum / sqrt(n_opp + k) : variance-stabilized residual score
    "frs_ebmag",  # excess_mag / (n_opp + k) : equity-magnitude-weighted EB residual
    "frs_ebpos",  # resid_pos / (n_opp + k)  : positive-only (surrendered-more-than-field) EB
]
# built and cached but held OUT of the scored set (drift/reference/derivation only)
FEATS_FR_AUX = ["fr_surr_resid_sum", "fr_surr_resid_mean", "fr_surr_resid_pos", "fr_surr_excess_mag",
                "fr_surr_lift", "fr_surr_obs_rate", "fr_surr_exp_rate", "fr_n_opp"]


def add_shrunk(df):
    """Attach the EB-shrunk scored features from the raw residual/exposure columns (used by the gate)."""
    import numpy as _np
    n = df["fr_n_opp"].clip(lower=1)
    df = df.copy()
    df["frs_eb"]     = df["fr_surr_resid_sum"]  / (n + EB_K)
    df["frs_z"]      = df["fr_surr_resid_sum"]  / _np.sqrt(n + EB_K)
    df["frs_ebmag"]  = df["fr_surr_excess_mag"] / (n + EB_K)
    df["frs_ebpos"]  = df["fr_surr_resid_pos"]  / (n + EB_K)
    return df


def _events(phase: str) -> pl.DataFrame:
    """Reshape per-hand + equity into one row per (pair,hand,member) surrender-OPPORTUNITY (member high-eq),
    with the observed-surrender flag and the field-context features. Two members per hand -> long form."""
    eq_path = EQ/f"pairhand_equity_{'dev' if phase=='development' else 'eval'}.parquet"
    eq = pl.read_parquet(eq_path)
    eqw = (eq.pivot(values="equity", index=["pair_id","hand_id"], on="member", aggregate_function="first")
             .rename({"1":"m1_eq","2":"m2_eq"}))
    if phase == "development":
        HF = sorted(STEP3.glob("dev_hand_features_*.parquet"))
    else:
        # eval is a single unsuffixed file; fall back to the suffixed glob if the layout ever changes
        HF = [STEP3/"eval_hand_features.parquet"] if (STEP3/"eval_hand_features.parquet").exists() \
             else sorted(STEP3.glob("eval_hand_features*.parquet"))
    hf = pl.concat([pl.scan_parquet(f).select(HF_COLS).collect() for f in HF])
    d = hf.join(eqw, on=["pair_id","hand_id"], how="left")
    d = d.with_columns([
        (pl.col("p2_won_share")>0).alias("p2_won"), (pl.col("p1_won_share")>0).alias("p1_won"),
        pl.col("m1_eq").fill_null(0.0), pl.col("m2_eq").fill_null(0.0),
        pl.min_horizontal(pl.col("players_at_showdown").fill_null(2), pl.lit(6)).alias("live_bucket"),
    ])
    # member-1 opportunity rows: m1 high-eq. observed surrender = folded-or-passive AND partner won AND m1 lost net
    m1 = d.filter(pl.col("m1_eq")>=T).select([
        pl.col("pair_id"),
        pl.col("m1_eq").alias("m_eq"), pl.col("m2_eq").alias("partner_eq"),
        pl.col("pot_bb"), pl.col("big_blind"), pl.col("players_at_showdown"), pl.col("live_bucket"),
        pl.col("p1_n_aggr").alias("m_n_aggr"), pl.col("p2_n_aggr").alias("partner_n_aggr"),
        pl.col("p1_went_to_showdown").cast(pl.Int8).alias("m_went_sd"),
        pl.col("p2_went_to_showdown").cast(pl.Int8).alias("partner_went_sd"),
        (((pl.col("p1_folded")) | (pl.col("p1_n_aggr")==0)) & (pl.col("p2_won")) & (pl.col("p1_net_bb")<0))
            .cast(pl.Int8).alias("obs_surr"),
    ])
    m2 = d.filter(pl.col("m2_eq")>=T).select([
        pl.col("pair_id"),
        pl.col("m2_eq").alias("m_eq"), pl.col("m1_eq").alias("partner_eq"),
        pl.col("pot_bb"), pl.col("big_blind"), pl.col("players_at_showdown"), pl.col("live_bucket"),
        pl.col("p2_n_aggr").alias("m_n_aggr"), pl.col("p1_n_aggr").alias("partner_n_aggr"),
        pl.col("p2_went_to_showdown").cast(pl.Int8).alias("m_went_sd"),
        pl.col("p1_went_to_showdown").cast(pl.Int8).alias("partner_went_sd"),
        (((pl.col("p2_folded")) | (pl.col("p2_n_aggr")==0)) & (pl.col("p1_won")) & (pl.col("p2_net_bb")<0))
            .cast(pl.Int8).alias("obs_surr"),
    ])
    ev = pl.concat([m1, m2]).with_columns([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in CTX])
    return ev


def build(phase: str):
    log(f"phase={phase}: building surrender-opportunity events")
    ev = _events(phase)
    log(f"  {ev.height:,} opportunity events; base surrender rate {ev['obs_surr'].mean():.4f}")
    # FIELD NULL MODEL: P(surrender | context). Fit on POOLED events of THIS phase, no pair id, no label.
    # Cross-fitted (5-fold on events) so a pair's expected value is never fit on its own rows -> no leak of
    # the pair's own surrender behavior into its expectation.
    X = ev.select(CTX).to_numpy().astype("float32")
    y = ev["obs_surr"].to_numpy().astype("float32")
    n = len(y); rng = np.random.RandomState(42); fold = rng.randint(0, 5, size=n)
    exp = np.zeros(n, dtype="float32")
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
             verbose=-1, num_threads=-1)
    for f in range(5):
        tr = fold != f; te = fold == f
        m = lgb.train({**P, "seed": 42}, lgb.Dataset(X[tr], y[tr]), num_boost_round=200)
        exp[te] = m.predict(X[te])
    ev = ev.with_columns(pl.Series("exp_surr", exp))
    ev = ev.with_columns((pl.col("obs_surr") - pl.col("exp_surr")).alias("resid"))
    ev = ev.with_columns([
        (pl.col("m_eq") * pl.col("resid")).alias("resid_mag"),
        pl.when(pl.col("resid") > 0).then(pl.col("resid")).otherwise(0.0).alias("resid_pos"),
    ])
    log(f"  field null cross-fit done; mean exp {exp.mean():.4f} (== base rate, sanity)")
    agg = ev.group_by("pair_id").agg([
        pl.col("resid").sum().alias("fr_surr_resid_sum"),
        pl.col("resid").mean().alias("fr_surr_resid_mean"),
        pl.col("resid_pos").sum().alias("fr_surr_resid_pos"),
        pl.col("resid_mag").sum().alias("fr_surr_excess_mag"),
        pl.col("obs_surr").mean().alias("fr_surr_obs_rate"),
        pl.col("exp_surr").mean().alias("fr_surr_exp_rate"),
        pl.len().alias("fr_n_opp"),
    ]).with_columns([
        (pl.col("fr_surr_obs_rate") / (pl.col("fr_surr_exp_rate") + 1e-3)).alias("fr_surr_lift"),
    ])
    out = EQ/f"field_relative_{phase}.parquet"
    aggpd = add_shrunk(agg.to_pandas())
    pl.from_pandas(aggpd.loc[:, ["pair_id"] + FEATS_FR + FEATS_FR_AUX]).write_parquet(out)
    log(f"  wrote {out}  ({agg.height:,} pairs)")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True, choices=["development","evaluation"])
    a = ap.parse_args(); build(a.phase); return 0

if __name__ == "__main__":
    raise SystemExit(main())
