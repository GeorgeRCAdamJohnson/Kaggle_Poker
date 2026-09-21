"""Exact 7-card poker hand evaluator (no external dependency), vectorized.

We have real hole+board cards on showdown seats but only ever used a crude rank-sum proxy (hs).
This computes the TRUE made-hand strength: a single monotone score where higher = stronger 5-card
hand out of 7 (2 hole + up to 5 board). Score encodes category (0=high..8=straight-flush) plus
tiebreak ranks, so it is directly comparable across hands.

Card strings: rank in '23456789TJQKA', suit in 'shdc' (e.g. 'Ah','Td','9c'). Board is space-sep.

Score layout (int, higher=better): cat*10^10 + r1*10^8 + r2*10^6 + r3*10^4 + r4*10^2 + r5.
Ranks are 2..14. We also expose a NORMALIZED [0,1] strength = score / max_score for modeling.

Vectorized path (`evaluate_frame`) works on a polars DataFrame with hole_card_1/2 + board_cards.
Run:  python -m anchor_repro.hand_eval   (self-test on known hands)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import polars as pl

RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
CAT_HIGH, CAT_PAIR, CAT_2PAIR, CAT_TRIPS, CAT_STRAIGHT, CAT_FLUSH, CAT_FULL, CAT_QUADS, CAT_SF = range(9)


def _card(cs: str):
    return RANKS[cs[0]], cs[1]


def _best5_score(cards):
    """cards: list of (rank,suit) length 5..7 -> best 5-card score int."""
    best = -1
    for combo in combinations(cards, 5):
        best = max(best, _score5(combo))
    return best


def _score5(five):
    ranks = sorted((r for r, _ in five), reverse=True)
    suits = [s for _, s in five]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # sort ranks by (count, rank) desc for tiebreak
    by = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    ordered = [r for r, _ in by]
    cvals = sorted(counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    is_straight, top = False, 0
    if len(uniq) >= 5:
        u = uniq
        # wheel A-5
        if set([14, 5, 4, 3, 2]).issubset(set(u)):
            is_straight, top = True, 5
        for i in range(len(u) - 4):
            w = u[i:i+5]
            if w[0] - w[4] == 4:
                is_straight, top = True, w[0]; break
    def enc(cat, tie):
        v = cat * 10**10
        for i, t in enumerate(tie[:5]):
            v += t * 10**(8 - 2*i)
        return v
    if is_straight and is_flush:
        return enc(CAT_SF, [top])
    if cvals[0] == 4:
        return enc(CAT_QUADS, ordered)
    if cvals[0] == 3 and cvals[1] >= 2:
        return enc(CAT_FULL, ordered)
    if is_flush:
        return enc(CAT_FLUSH, ranks)
    if is_straight:
        return enc(CAT_STRAIGHT, [top])
    if cvals[0] == 3:
        return enc(CAT_TRIPS, ordered)
    if cvals[0] == 2 and cvals[1] == 2:
        return enc(CAT_2PAIR, ordered)
    if cvals[0] == 2:
        return enc(CAT_PAIR, ordered)
    return enc(CAT_HIGH, ranks)


_MAXSCORE = CAT_SF * 10**10 + 14*10**8 + 13*10**6 + 12*10**4 + 11*10**2 + 10


def evaluate_seat(h1: str, h2: str, board: str):
    """Return (score_int, strength_0_1, category). board may be empty (preflop)."""
    if not h1 or not h2 or h1 == "" or h2 == "":
        return -1, 0.0, -1
    cards = [_card(h1), _card(h2)]
    if board:
        for c in board.split():
            if c:
                cards.append(_card(c))
    if len(cards) < 5:
        # preflop-only: rank by high card of the two hole cards + pair bonus
        r = sorted([cards[0][0], cards[1][0]], reverse=True)
        pair = cards[0][0] == cards[1][0]
        sc = (CAT_PAIR if pair else CAT_HIGH) * 10**10 + r[0]*10**8 + r[1]*10**6
        return sc, sc / _MAXSCORE, (CAT_PAIR if pair else CAT_HIGH)
    sc = _best5_score(cards)
    return sc, sc / _MAXSCORE, sc // 10**10


def evaluate_frame(df: pl.DataFrame, h1="hole_card_1", h2="hole_card_2", board="board_cards") -> pl.DataFrame:
    """Add true_strength (0..1), hand_cat, hand_score columns. Only rows with cards get real values."""
    rows = df.select([h1, h2, board]).to_dicts()
    scores = np.empty(len(rows), np.float64)
    strength = np.empty(len(rows), np.float64)
    cats = np.empty(len(rows), np.int32)
    for i, r in enumerate(rows):
        sc, st, cat = evaluate_seat(r.get(h1) or "", r.get(h2) or "", r.get(board) or "")
        scores[i] = sc; strength[i] = st; cats[i] = cat
    return df.with_columns(
        pl.Series("hand_score", scores),
        pl.Series("true_strength", strength.astype(np.float32)),
        pl.Series("hand_cat", cats),
    )


def _selftest():
    cases = [
        ("Ah", "Ad", "As Kc Kd", "full house / quads-ish"),
        ("Ah", "Kh", "Qh Jh Th", "royal flush"),
        ("2c", "7d", "9h Js 4c", "seven high junk"),
        ("8s", "8d", "8h 2c 3d", "trips"),
        ("Ts", "Js", "Qs Ks As", "royal flush spades"),
        ("5d", "6d", "7d 8d 9d", "straight flush"),
    ]
    print("=== hand_eval self-test (higher = stronger) ===")
    scored = []
    for h1, h2, b, desc in cases:
        sc, st, cat = evaluate_seat(h1, h2, b)
        catn = ["high","pair","2pair","trips","straight","flush","full","quads","SF"][cat]
        print(f"  {h1} {h2} | {b:<14} -> cat={catn:<9} strength={st:.4f}  ({desc})")
        scored.append((st, desc))
    # sanity: royal flush > straight flush > trips > seven high
    assert evaluate_seat("Ah","Kh","Qh Jh Th")[0] > evaluate_seat("8s","8d","8h 2c 3d")[0]
    assert evaluate_seat("8s","8d","8h 2c 3d")[0] > evaluate_seat("2c","7d","9h Js 4c")[0]
    print("  ORDERING checks PASS")


if __name__ == "__main__":
    _selftest()
