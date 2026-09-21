"""ALGO 4 — SEQUENCE MODEL on RAW ordered actions (the biggest swing; dossier §114 route 3).

Every prior model consumes AGGREGATES of the action stream. This consumes the RAW ordered
interaction between a pair's members within their shared hands — the coordination DYNAMICS that
mean/rate/percentile destroy. A Transformer over pair-relative action tokens learns motifs like
"A raises -> B folds -> A takes it", repeated, that no aggregate captures.

Tokenization (pair-relative, per shared hand): for each shared hand, the ordered actions become
tokens = (actor_role in {A,B,outsider}) x (action in {fold,call,check,bet,raise,all_in}) x
(bet-size bucket) x (street). A pair is a SET of its shared-hand sequences -> we encode each hand
with a small Transformer, mean/attention-pool hands to a pair vector, classify pos vs eval-background.

Because the representation is learned on the interaction structure (not dev-selection aggregates),
transfer should be BETTER than the aggregate lineage. Trained pos-vs-eval-background (dense) +
table-grouped OOF on dev. Judged eval-honest.

Run:  python -m anchor_repro.algo_sequence   [--quick]
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from anchor_repro.eval_honest_harness import Harness, CACHE, D

SEQ = Path("outputs/poker_collusion/lamhuy_sequence_confirmation")
V30_UNI_DEV = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/dev_hand_features.parquet")
V30_UNI_EVAL = Path("outputs/poker_collusion/repro_lamhuy/prepared_v13/eval_hand_features.parquet")
OUT = CACHE / "algos"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

ACTIONS = {"fold": 0, "call": 1, "check": 2, "bet": 3, "raise": 4, "all_in": 5}
MAX_HANDS = 40      # cap shared hands per pair (most have < 80; sample if more)
MAX_ACTS = 24       # cap actions per hand
PAD = 0


def _bucket_amount(amt, pot):
    # bet-size bucket relative to pot: 0=none,1=<0.5pot,2=0.5-1,3=1-2,4=>2
    r = amt / max(pot, 1)
    return 1 + int(r >= 0.5) + int(r >= 1.0) + int(r >= 2.0) if amt > 0 else 0


def _build_pair_sequences(pair_frame: pl.DataFrame, universe: pl.LazyFrame, actions: pl.LazyFrame):
    """Return {pair_id: LongTensor[n_hands, MAX_ACTS] of token ids} using pair-relative roles.

    token = role(0=pad,1=A,2=B,3=outsider) * 6*5*4 + action*5*4 + street*4 + bucket ... compacted.
    We build a compact integer token; vocab kept small.
    """
    # restrict actions to the pair-universe hands, join role
    uni = universe.select(["pair_id", "hand_id"]).unique()
    pf = pair_frame.select(["pair_id", "player_1", "player_2"]).lazy()
    hands = uni.join(pf, on="pair_id", how="inner")  # pair_id, hand_id, p1, p2
    # sample MAX_HANDS per pair deterministically (by hand_id) to bound size
    hands = hands.with_columns(pl.col("hand_id").rank("dense").over("pair_id").alias("hrank")).filter(pl.col("hrank") <= MAX_HANDS)
    acts = actions.select(["hand_id", "action_no", "street", "player_id", "action", "amount", "pot_before"])
    j = hands.join(acts, on="hand_id", how="inner")
    j = j.with_columns(
        pl.when(pl.col("player_id") == pl.col("player_1")).then(1)
          .when(pl.col("player_id") == pl.col("player_2")).then(2).otherwise(3).alias("role"),
        pl.col("action").replace_strict(ACTIONS, default=0).alias("act_id"),
        pl.col("street").replace_strict({"preflop":0,"flop":1,"turn":2,"river":3}, default=0).alias("st"),
        (pl.col("amount") / pl.when(pl.col("pot_before")>0).then(pl.col("pot_before")).otherwise(1)).alias("amt_ratio"),
    ).with_columns(
        (pl.when(pl.col("amount")<=0).then(0)
         .otherwise(1 + (pl.col("amt_ratio")>=0.5).cast(pl.Int32) + (pl.col("amt_ratio")>=1.0).cast(pl.Int32) + (pl.col("amt_ratio")>=2.0).cast(pl.Int32))).alias("bucket")
    ).with_columns(
        # compact token: role(1..3) act(0..5) st(0..3) bucket(0..4) -> id in [1, 3*6*4*5]
        (1 + ((pl.col("role")-1)*6*4*5 + pl.col("act_id")*4*5 + pl.col("st")*5 + pl.col("bucket"))).alias("tok")
    ).filter(pl.col("action_no") < MAX_ACTS)
    return j.select(["pair_id", "hand_id", "action_no", "tok"]).collect(engine="streaming")


VOCAB = 1 + 3*6*4*5  # 361


class HandSeqEncoder(nn.Module):
    def __init__(self, d=48, nhead=4, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(MAX_ACTS, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, d*2, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x):  # x [B, MAX_ACTS]
        h = self.emb(x) + self.pos[None, :x.shape[1]]
        mask = (x == PAD)
        h = self.tr(h, src_key_padding_mask=mask)
        # masked mean pool
        w = (~mask).float().unsqueeze(-1)
        return (h * w).sum(1) / w.sum(1).clamp(min=1)


class PairModel(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.hand_enc = HandSeqEncoder(d)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, hands_tok, hand_mask):  # [P, H, A], [P, H]
        P, Hn, A = hands_tok.shape
        he = self.hand_enc(hands_tok.reshape(P*Hn, A)).reshape(P, Hn, -1)
        w = hand_mask.float().unsqueeze(-1)
        pair_vec = (he * w).sum(1) / w.sum(1).clamp(min=1)   # pool hands -> pair
        return self.head(pair_vec).squeeze(-1)


def _pack(seq_df: pl.DataFrame, pair_ids):
    """pair_id -> (hands_tok [H,A], hand_mask [H]) padded tensors, for a list of pair_ids."""
    # build nested dict pair -> hand -> [tokens by action_no]
    by_pair = {}
    for row in seq_df.iter_rows(named=True):
        by_pair.setdefault(row["pair_id"], {}).setdefault(row["hand_id"], []).append((row["action_no"], row["tok"]))
    packed = {}
    for pid in pair_ids:
        hands = by_pair.get(pid, {})
        htoks = []
        for hid, acts in list(hands.items())[:MAX_HANDS]:
            acts.sort()
            toks = [t for _, t in acts][:MAX_ACTS]
            toks += [PAD]*(MAX_ACTS-len(toks))
            htoks.append(toks)
        if not htoks:
            htoks = [[PAD]*MAX_ACTS]
        while len(htoks) < 1:
            htoks.append([PAD]*MAX_ACTS)
        packed[pid] = htoks
    return packed


def main() -> int:
    quick = "--quick" in sys.argv
    H = Harness()
    dev_t = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev_t} vocab={VOCAB}")

    labels = pl.read_csv(D / "development_labels.csv").select(["pair_id", "player_1", "player_2", "label"])
    eval_pairs = pl.read_csv(D / "evaluation_pairs.csv").select(["pair_id", "player_1", "player_2"])
    actions = pl.scan_parquet(D / "actions.parquet")
    dev_uni = pl.scan_parquet(V30_UNI_DEV).select(["pair_id", "hand_id"])
    eval_uni = pl.scan_parquet(V30_UNI_EVAL).select(["pair_id", "hand_id"])

    t = time.time()
    dev_seq = _build_pair_sequences(labels, dev_uni, actions)
    print(f"dev sequences built: {dev_seq.height:,} tokens ({time.time()-t:.0f}s)")
    dev_ids = labels["pair_id"].to_list()
    y = labels["label"].to_numpy().astype(np.int8)
    dev_packed = _pack(dev_seq, dev_ids)

    # table groups
    hp = pl.scan_parquet(CACHE / "hand_context.parquet").select(["hand_id", "table_id"])
    ptbl = (dev_uni.join(hp, on="hand_id").group_by(["pair_id", "table_id"]).agg(pl.len().alias("n"))
            .sort(["pair_id","n"],descending=[False,True]).unique("pair_id",keep="first").select(["pair_id","table_id"]).collect())
    tmap = dict(zip(ptbl["pair_id"].to_list(), ptbl["table_id"].to_list()))
    groups = np.array([tmap.get(p, "?") for p in dev_ids])

    def batch_tensor(packed, ids):
        maxh = max(len(packed[p]) for p in ids)
        maxh = min(maxh, MAX_HANDS)
        B = len(ids)
        tok = np.zeros((B, maxh, MAX_ACTS), np.int64)
        hmask = np.zeros((B, maxh), np.float32)
        for i, p in enumerate(ids):
            hs = packed[p][:maxh]
            for hj, row in enumerate(hs):
                tok[i, hj] = row; hmask[i, hj] = 1.0
        return torch.tensor(tok), torch.tensor(hmask)

    from sklearn.model_selection import StratifiedGroupKFold
    oof = np.zeros(len(y), np.float32)
    epochs = 8 if quick else 25
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    t = time.time()
    for fold,(tr,va) in enumerate(sgkf.split(dev_ids, y, groups)):
        model = PairModel().to(dev_t)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_ids = [dev_ids[i] for i in tr]; ytr = torch.tensor(y[tr].astype(np.float32), device=dev_t)
        spw = (y[tr]==0).sum()/max((y[tr]==1).sum(),1)
        tok_tr, hm_tr = batch_tensor(dev_packed, tr_ids)
        tok_tr, hm_tr = tok_tr.to(dev_t), hm_tr.to(dev_t)
        model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(tr_ids), device=dev_t)
            for s in range(0, len(tr_ids), 256):
                idx = perm[s:s+256]
                opt.zero_grad()
                logit = model(tok_tr[idx], hm_tr[idx])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, ytr[idx], pos_weight=torch.tensor(float(spw), device=dev_t))
                loss.backward(); opt.step()
        model.eval()
        va_ids = [dev_ids[i] for i in va]
        tok_va, hm_va = batch_tensor(dev_packed, va_ids)
        with torch.no_grad():
            oof[va] = torch.sigmoid(model(tok_va.to(dev_t), hm_va.to(dev_t))).cpu().numpy()
        print(f"  fold {fold+1}/5 done ({time.time()-t:.0f}s)", flush=True)

    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"\nSEQUENCE model dev: confirmed AP={average_precision_score(y,oof):.4f} AUC={roc_auc_score(y,oof):.4f}")

    # eval sequences (batched build) + full-model scoring
    t = time.time()
    full = PairModel().to(dev_t)
    opt = torch.optim.Adam(full.parameters(), lr=1e-3, weight_decay=1e-4)
    spw = (y==0).sum()/max((y==1).sum(),1)
    tok_all, hm_all = batch_tensor(dev_packed, dev_ids); tok_all, hm_all = tok_all.to(dev_t), hm_all.to(dev_t)
    yall = torch.tensor(y.astype(np.float32), device=dev_t)
    full.train()
    for ep in range(epochs):
        perm = torch.randperm(len(dev_ids), device=dev_t)
        for s in range(0, len(dev_ids), 256):
            idx = perm[s:s+256]; opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(full(tok_all[idx], hm_all[idx]), yall[idx], pos_weight=torch.tensor(float(spw),device=dev_t))
            loss.backward(); opt.step()
    full.eval()
    # eval in chunks (112k pairs)
    escore = np.zeros(eval_pairs.height, np.float32)
    eval_ids = eval_pairs["pair_id"].to_list()
    CHUNK = 15000
    for start in range(0, eval_pairs.height, CHUNK):
        chunk_pairs = eval_pairs.slice(start, CHUNK)
        cseq = _build_pair_sequences(chunk_pairs, eval_uni, actions)
        cpacked = _pack(cseq, chunk_pairs["pair_id"].to_list())
        tok_c, hm_c = batch_tensor(cpacked, chunk_pairs["pair_id"].to_list())
        with torch.no_grad():
            escore[start:start+chunk_pairs.height] = torch.sigmoid(full(tok_c.to(dev_t), hm_c.to(dev_t))).cpu().numpy()
        print(f"  eval chunk {start//CHUNK+1}/{-(-eval_pairs.height//CHUNK)} ({time.time()-t:.0f}s)", flush=True)

    r = H.judge("sequence_model", oof, dev_ids, escore, eval_ids)
    print(f"\n=== SEQUENCE model (eval-honest) ===")
    print(f"  {r}")
    print(f"  vs V30 0.9203 -> delta {r.eval_honest_ap-0.9203:+.4f}")
    pl.DataFrame({"pair_id": eval_ids, "risk_score": ((escore-escore.min())/(escore.max()-escore.min()+1e-9)).astype(np.float32)}).write_parquet(OUT / "sequence_eval_risk.parquet")
    (OUT / "_sequence_summary.json").write_text(json.dumps({"eval_honest_ap": r.eval_honest_ap, "confirmed_ap": r.confirmed_ap, "auc": r.auc}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
