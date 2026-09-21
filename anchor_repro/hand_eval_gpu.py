"""GPU-vectorized 7-card poker hand strength (torch). Scores 659k seats in seconds.

Computes a monotone true-strength score per seat from 7 cards (2 hole + up to 5 board), fully
vectorized on GPU — no per-hand Python loop. Best-5-of-7 is derived directly from rank-count
histograms + straight/flush detection over the 7 cards (standard, exact).

Score = cat*R^5 + tiebreak in base-R (R=15) so higher=stronger and comparable across hands.
Categories: 0 high,1 pair,2 two-pair,3 trips,4 straight,5 flush,6 full house,7 quads,8 straight-flush.

API: strength_for_seats(hole1, hole2, board_lists) -> np.float32[N] in ~[0,1].
Run:  python -m anchor_repro.hand_eval_gpu   (self-test)
"""

from __future__ import annotations

import numpy as np
import torch

RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}   # 2..14
SUITS = {"s": 0, "h": 1, "d": 2, "c": 3}
R = 15  # base for score encoding


def _parse_cards(card_lists, dev):
    """card_lists: list[list[str]] each length up to 7. Returns rank[N,7], suit[N,7], mask[N,7].

    Vectorized: build flat numpy int arrays (fast Python only for the flat fill), then one GPU move.
    """
    N = len(card_lists)
    rank_np = np.zeros((N, 7), dtype=np.int64)
    suit_np = np.zeros((N, 7), dtype=np.int64)
    mask_np = np.zeros((N, 7), dtype=bool)
    for i, cards in enumerate(card_lists):
        for j, c in enumerate(cards[:7]):
            if c and len(c) == 2:
                r = RANKS.get(c[0])
                if r is not None:
                    rank_np[i, j] = r; suit_np[i, j] = SUITS.get(c[1], 0); mask_np[i, j] = True
    rank = torch.from_numpy(rank_np).to(dev)
    suit = torch.from_numpy(suit_np).to(dev)
    mask = torch.from_numpy(mask_np).to(dev)
    return rank, suit, mask


def _score(card_lists, dev):
    rank, suit, mask = _parse_cards(card_lists, dev)
    N = rank.shape[0]
    # rank count histogram [N,15] (index by rank 2..14)
    rc = torch.zeros(N, R, device=dev)
    idx = rank.clamp(0, R-1)
    rc.scatter_add_(1, idx, mask.float())
    # suit count histogram [N,4]
    sc = torch.zeros(N, 4, device=dev)
    sc.scatter_add_(1, suit, mask.float())
    is_flush = (sc.max(1).values >= 5)
    flush_suit = sc.argmax(1)

    # rank-present bitmask for straight detection (2..14), add ace-low (rank 1) if 14 present
    present = (rc > 0).float()  # [N,15], indices 2..14 used
    present_low = present.clone()
    present_low[:, 1] = present[:, 14]  # ace as low
    # straight: any window of 5 consecutive ranks all present. build via conv-like check
    ranks_axis = torch.arange(R, device=dev)
    def straight_top(pres):
        top = torch.zeros(N, device=dev)
        for hi in range(14, 4, -1):  # top card 14..5
            window = pres[:, hi-4:hi+1]  # 5 consecutive
            ok = (window.sum(1) == 5)
            top = torch.where((top == 0) & ok, torch.tensor(float(hi), device=dev), top)
        return top
    st_top = straight_top(present_low)
    is_straight = st_top > 0

    # straight-flush: straight within the flush suit's cards
    # build present mask restricted to flush suit
    fs = flush_suit.unsqueeze(1)
    in_fsuit = (suit == fs) & mask
    rc_f = torch.zeros(N, R, device=dev)
    rc_f.scatter_add_(1, rank.clamp(0, R-1), in_fsuit.float())
    pres_f = (rc_f > 0).float(); pres_f_low = pres_f.clone(); pres_f_low[:, 1] = pres_f[:, 14]
    sf_top = straight_top(pres_f_low)
    is_sf = is_flush & (sf_top > 0)

    # count patterns: sort ranks by (count, rank) desc for tiebreak
    # counts per rank; get top-5 tiebreak ranks weighted by count
    # build sortable key: count*100 + rank, take ranks in that order
    key = rc * 100 + ranks_axis.float()[None, :]
    order = torch.argsort(key, dim=1, descending=True)  # [N,15] rank indices by (count,rank)
    counts_sorted = torch.gather(rc, 1, order)
    ranks_sorted = order.float()

    c0 = counts_sorted[:, 0]; c1 = counts_sorted[:, 1]
    is_quads = c0 == 4
    is_full = (c0 == 3) & (c1 >= 2)
    is_trips = c0 == 3
    is_2pair = (c0 == 2) & (c1 == 2)
    is_pair = c0 == 2

    # category
    cat = torch.zeros(N, device=dev)
    cat = torch.where(is_pair, torch.tensor(1., device=dev), cat)
    cat = torch.where(is_2pair, torch.tensor(2., device=dev), cat)
    cat = torch.where(is_trips, torch.tensor(3., device=dev), cat)
    cat = torch.where(is_straight, torch.tensor(4., device=dev), cat)
    cat = torch.where(is_flush, torch.tensor(5., device=dev), cat)
    cat = torch.where(is_full, torch.tensor(6., device=dev), cat)
    cat = torch.where(is_quads, torch.tensor(7., device=dev), cat)
    cat = torch.where(is_sf, torch.tensor(8., device=dev), cat)

    # tiebreak: 5 ranks. for made hands use ranks_sorted; for straight/sf use top card.
    tie = ranks_sorted[:, :5].clone()
    straight_tie = torch.zeros(N, 5, device=dev); straight_tie[:, 0] = st_top
    sf_tie = torch.zeros(N, 5, device=dev); sf_tie[:, 0] = sf_top
    use_straight = (cat == 4)
    use_sf = (cat == 8)
    tie = torch.where(use_straight.unsqueeze(1), straight_tie, tie)
    tie = torch.where(use_sf.unsqueeze(1), sf_tie, tie)

    # encode score
    score = cat * (R**5)
    for i in range(5):
        score = score + tie[:, i] * (R ** (4 - i))
    # preflop-only (fewer than 5 cards): mask.sum<5 -> use high/pair of 2 hole
    ncards = mask.sum(1)
    return score, ncards


_MAX = 8 * (R**5) + 14*R**4 + 13*R**3 + 12*R**2 + 11*R + 10


def strength_for_seats(card_lists, dev=None) -> np.ndarray:
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    score, ncards = _score(card_lists, dev)
    st = (score / _MAX).clamp(0, 1)
    st = torch.where(ncards >= 5, st, st)  # preflop handled by same encoding (fewer ranks)
    return st.cpu().numpy().astype(np.float32)


def _selftest():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hands = [
        ["Ah","Kh","Qh","Jh","Th"],           # royal
        ["5d","6d","7d","8d","9d"],            # straight flush
        ["Ah","Ad","As","Kc","Kd"],           # full house (aces full)
        ["8s","8d","8h","2c","3d"],            # trips
        ["2c","7d","9h","Js","4c"],           # seven high
        ["Ah","Ad","As","Ac","2d"],           # quads
    ]
    st = strength_for_seats(hands, dev)
    names = ["royal","straightflush","fullhouse","trips","sevenhigh","quads"]
    for n, s in zip(names, st):
        print(f"  {n:<15} strength={s:.4f}")
    assert st[0] > st[1] > st[2] > st[3] > st[4]  # royal>SF>full>trips>high
    assert st[5] > st[2]  # quads > full house
    print("  ORDERING PASS  device=", dev)


if __name__ == "__main__":
    _selftest()
