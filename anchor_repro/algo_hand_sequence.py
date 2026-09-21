"""ALGO 4 (rebuilt HAND-LEVEL) — sequence transformer on raw ordered actions, dense per-hand target.

The pair-level sequence model was data-starved (372 pair labels -> confirmed AP 0.33). The DENSE
positive set is the 1,817 PLANTED HANDS (each ~12 actions, full sequences in actions.parquet) — 4.9x
more positives. This trains a hand-level sequence transformer: read the raw ordered pair-relative
action tokens of a SINGLE hand, predict P(planted). Then aggregate per-hand -> pair for risk, and
read evidence/behavior from the same model (like unified_perhand, but a SEQUENCE model over raw
actions instead of a GBT over aggregates).

Positives: 1,817 planted hands (pair-relative tokens: member-A/B/outsider x action x size x street).
Negatives: sampled hands from eval-population pairs' shared hands (background coordination-lookalikes).
The bet: reading the ORDERED interaction (A raises->B folds->A wins) separates planted from
lookalike better than aggregates did (unified per-hand GBT hit AUC 0.99 vs RANDOM background but
0.16 pair-risk on eval — the sequence model may read the DIRECTED ordering that aggregates blur).

Judged eval-honest (per-hand model AUC honest; pair-risk aggregate validated only by LB).
Run:  python -m anchor_repro.algo_hand_sequence  [--quick]
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from anchor_repro.eval_honest_harness import Harness, CACHE, D

V30_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
V30_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
OUT = CACHE / "algos"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

ACTIONS = {"fold": 0, "call": 1, "check": 2, "bet": 3, "raise": 4, "all_in": 5}
MAX_ACTS = 28
PAD = 0
VOCAB = 1 + 3 * 6 * 4 * 5  # role(1..3) x action(6) x street(4) x sizebucket(5)


def _hand_tokens(pairs: pl.DataFrame, universe: pl.LazyFrame, actions: pl.LazyFrame,
                 hand_filter: list | None = None) -> pl.DataFrame:
    """Per (pair_id, hand_id): ordered token list of pair-relative actions."""
    uni = universe.select(["pair_id", "hand_id"]).unique()
    pf = pairs.select(["pair_id", "player_1", "player_2"]).lazy()
    hands = uni.join(pf, on="pair_id", how="inner")
    if hand_filter is not None:
        hands = hands.filter(pl.col("hand_id").is_in(hand_filter))
    acts = actions.select(["hand_id", "action_no", "street", "player_id", "action", "amount", "pot_before"])
    j = hands.join(acts, on="hand_id", how="inner").filter(pl.col("action_no") < MAX_ACTS)
    j = j.with_columns(
        pl.when(pl.col("player_id") == pl.col("player_1")).then(1)
          .when(pl.col("player_id") == pl.col("player_2")).then(2).otherwise(3).alias("role"),
        pl.col("action").replace_strict(ACTIONS, default=0).alias("act_id"),
        pl.col("street").replace_strict({"preflop": 0, "flop": 1, "turn": 2, "river": 3}, default=0).alias("st"),
        (pl.col("amount") / pl.when(pl.col("pot_before") > 0).then(pl.col("pot_before")).otherwise(1)).alias("ar"),
    ).with_columns(
        pl.when(pl.col("amount") <= 0).then(0).otherwise(
            1 + (pl.col("ar") >= 0.5).cast(pl.Int32) + (pl.col("ar") >= 1.0).cast(pl.Int32) + (pl.col("ar") >= 2.0).cast(pl.Int32)
        ).alias("bucket")
    ).with_columns(
        (1 + ((pl.col("role") - 1) * 6 * 4 * 5 + pl.col("act_id") * 4 * 5 + pl.col("st") * 5 + pl.col("bucket"))).alias("tok")
    )
    return j.select(["pair_id", "hand_id", "action_no", "tok"]).collect(engine="streaming")


def _pack_hands(tok_df: pl.DataFrame):
    """Return (keys list[(pair_id,hand_id)], np.int64[N, MAX_ACTS])."""
    grouped = {}
    for r in tok_df.iter_rows(named=True):
        grouped.setdefault((r["pair_id"], r["hand_id"]), []).append((r["action_no"], r["tok"]))
    keys = list(grouped.keys())
    arr = np.zeros((len(keys), MAX_ACTS), np.int64)
    for i, k in enumerate(keys):
        acts = sorted(grouped[k])[:MAX_ACTS]
        for j, (_, t) in enumerate(acts):
            arr[i, j] = t
    return keys, arr


class HandTransformer(nn.Module):
    def __init__(self, d=56, nhead=4, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(MAX_ACTS, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, d * 2, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1))

    def forward(self, x):
        mask = (x == PAD)
        h = self.emb(x) + self.pos[None, :x.shape[1]]
        h = self.tr(h, src_key_padding_mask=mask)
        w = (~mask).float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def _predict(model, arr, dev_t, bs=4096):
    model.eval()
    out = np.zeros(len(arr), np.float32)
    with torch.no_grad():
        for s in range(0, len(arr), bs):
            xb = torch.tensor(arr[s:s+bs], device=dev_t)
            out[s:s+bs] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def main() -> int:
    quick = "--quick" in sys.argv
    H = Harness()
    dev_t = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev_t} vocab={VOCAB}")

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    evidence = pl.read_csv(D / "development_evidence.csv").select(["pair_id", "hand_id", "behavior_family"])
    actions = pl.scan_parquet(D / "actions.parquet")
    dev_uni = pl.scan_parquet(V30_DEV).select(["pair_id", "hand_id"])
    eval_uni = pl.scan_parquet(V30_EVAL).select(["pair_id", "hand_id"])

    # --- build hand tokens for ALL dev shared hands (positives + their own non-planted = context) ---
    t = time.time()
    dev_tok = _hand_tokens(labels, dev_uni, actions)
    keys, arr = _pack_hands(dev_tok)
    key_df = pl.DataFrame({"pair_id": [k[0] for k in keys], "hand_id": [k[1] for k in keys]})
    planted_set = set(evidence["hand_id"].to_list())
    is_planted = np.array([1 if k[1] in planted_set else 0 for k in keys], np.int8)
    # pool for grouped CV
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    tbl = key_df.lazy().join(hp, on="hand_id", how="left").collect()["table_id"].fill_null("?").to_numpy()
    print(f"dev hand-tokens: {len(keys):,} hands, planted={int(is_planted.sum())} ({time.time()-t:.0f}s)")

    # --- hand-level OOF: planted vs (dev non-planted + will add eval-bg at train) ---
    epochs = 6 if quick else 20
    oof = np.zeros(len(keys), np.float32)
    gkf = GroupKFold(5)
    t = time.time()
    for fold, (tr, va) in enumerate(gkf.split(arr, is_planted, tbl)):
        model = HandTransformer().to(dev_t)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Xtr = torch.tensor(arr[tr], device=dev_t); ytr = torch.tensor(is_planted[tr].astype(np.float32), device=dev_t)
        spw = float((is_planted[tr] == 0).sum() / max((is_planted[tr] == 1).sum(), 1))
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(tr), device=dev_t)
            for s in range(0, len(tr), 2048):
                idx = perm[s:s+2048]; opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(Xtr[idx]), ytr[idx], pos_weight=torch.tensor(spw, device=dev_t))
                loss.backward(); opt.step()
        oof[va] = _predict(model, arr[va], dev_t)
        print(f"  fold {fold+1}/5 ({time.time()-t:.0f}s)", flush=True)

    ph_auc = roc_auc_score(is_planted, oof)
    ph_ap = average_precision_score(is_planted, oof)
    print(f"\nHAND-level sequence model: planted AUC={ph_auc:.4f} AP={ph_ap:.4f} (n_planted={int(is_planted.sum())})")

    # --- aggregate per-hand OOF -> pair risk on dev (confirmed pairs only have candidates) ---
    dev_pair_coord = key_df.with_columns(pl.Series("coord", oof)).group_by("pair_id").agg(
        pl.col("coord").top_k(3).mean().alias("risk_top3"), pl.col("coord").max().alias("risk_max"))

    # --- full model + eval scoring (chunked, sub-batched) ---
    full = HandTransformer().to(dev_t)
    opt = torch.optim.Adam(full.parameters(), lr=1e-3, weight_decay=1e-4)
    Xall = torch.tensor(arr, device=dev_t); yall = torch.tensor(is_planted.astype(np.float32), device=dev_t)
    spw = float((is_planted == 0).sum() / max((is_planted == 1).sum(), 1))
    full.train()
    for ep in range(epochs):
        perm = torch.randperm(len(keys), device=dev_t)
        for s in range(0, len(keys), 2048):
            idx = perm[s:s+2048]; opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(full(Xall[idx]), yall[idx], pos_weight=torch.tensor(spw, device=dev_t))
            loss.backward(); opt.step()

    t = time.time()
    eval_pair_coord = {}
    CHUNK = 8000
    for start in range(0, eval_pairs.height, CHUNK):
        cp = eval_pairs.slice(start, CHUNK)
        ctok = _hand_tokens(cp, eval_uni, actions)
        if ctok.height == 0:
            continue
        ck, ca = _pack_hands(ctok)
        cscore = _predict(full, ca, dev_t)
        cdf = pl.DataFrame({"pair_id": [k[0] for k in ck], "coord": cscore}).group_by("pair_id").agg(
            pl.col("coord").top_k(3).mean().alias("risk_top3"))
        for r in cdf.iter_rows(named=True):
            eval_pair_coord[r["pair_id"]] = r["risk_top3"]
        print(f"  eval chunk {start//CHUNK+1}/{-(-eval_pairs.height//CHUNK)} ({time.time()-t:.0f}s)", flush=True)

    # judge eval-honest (dev risk over labeled pairs; only 372 pos have candidates -> floor others)
    dmap = dict(zip(dev_pair_coord["pair_id"].to_list(), dev_pair_coord["risk_top3"].to_numpy()))
    dfloor = min(dmap.values())
    dev_oof_risk = np.array([dmap.get(p, dfloor) for p in H.dev_pair_ids], np.float32)
    efloor = min(eval_pair_coord.values())
    escore = np.array([eval_pair_coord.get(p, efloor) for p in H.eval_pair_ids], np.float32)
    r = H.judge("hand_sequence", dev_oof_risk, H.dev_pair_ids, escore, H.eval_pair_ids)
    print(f"\n=== HAND-SEQUENCE (eval-honest; NOTE dev floors non-candidate pairs -> optimistic) ===")
    print(f"  planted-hand AUC={ph_auc:.4f}  |  {r}")
    print(f"  (bar = REAL LB 0.70904; harness number is an ESTIMATE, LB is the only real judge)")

    pl.DataFrame({"pair_id": H.eval_pair_ids, "risk_score": ((escore-escore.min())/(escore.max()-escore.min()+1e-9)).astype(np.float32)}).write_parquet(OUT / "hand_sequence_eval_risk.parquet")
    (OUT / "_hand_sequence_summary.json").write_text(json.dumps(
        {"planted_hand_auc": float(ph_auc), "planted_hand_ap": float(ph_ap),
         "eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/hand_sequence_eval_risk.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
