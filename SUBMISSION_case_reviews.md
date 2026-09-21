# Five Case Reviews — Detect Suspicious Value Transfers in Poker

Each case cites a submitted pair, the evidence hand(s) our ranker surfaced, the observable in-gameplay
behavior that drove the flag, and a plausible benign alternative. All observations are derived solely from
public gameplay (actions, contributions, showdowns) — no ID formats, ordering, or generator internals.

Note on method: our detector flags a *pattern across a pair's shared hands*, not any single hand. A single
hand in isolation is rarely conclusive; the evidence hands below are the strongest individual instances of
the pair-level pattern, and each has a legitimate one-hand explanation. That tension is the point of the
review.

---

## Case 1 — Directed transfer
- **Pair ID:** `PD5956DB3CCF8` (members `U05013CA31519`, `U20CA59C9CA53`)
- **Evidence hands:** `HA4449DCF8340CB`, `H86AD7E78C9C300`
- **Observable suspicious behavior:** In both hands `U05013CA31519` voluntarily puts chips in preflop to
  call `U20CA59C9CA53`'s raise, then folds to the partner's first postflop bet, transferring the pot to the
  partner. In `HA4449DCF8340CB` it calls a 3x raise with `2h3d` (a hand with almost no preflop equity) and
  folds the flop; in `H86AD7E78C9C300` it calls with `As2c`, flops top pair (Ace) on `Qs Ah 8d`, and still
  folds to a large bet. Folding top pair after voluntarily entering is the atypical action that drove the flag.
- **Plausible benign explanation:** `U05013CA31519` may simply be a loose-passive player who calls too wide
  preflop and folds weak-kicker top pair (A2) to a big bet out of position — a common, non-collusive leak.
  The `2h3d` call could be a blind-defense or a misclick-level loose call. Neither hand alone proves intent.

## Case 2 — Soft play
- **Pair ID:** `PC178304F6F93` (members `U2B73D38F0F64`, `U54EBF17596CB`)
- **Evidence hands:** `H22311A667D6951`, `H3965E8228C0661`
- **Observable suspicious behavior:** When only these two contest the pot, they repeatedly check it down
  across multiple streets rather than bet for value, then one makes a minimal river bet the other folds to —
  avoiding the large confrontations they take against other opponents. In `H22311A667D6951` both hold
  strong-ish hands (KQ suited vs K9 suited) yet check flop and turn heads-up before a tiny river bet ends it.
- **Plausible benign explanation:** Mutual pot-control with marginal made hands is standard, correct poker:
  neither wants to bloat a pot with a medium-strength holding out of position, so they check down and one
  takes it cheaply at the end. Two tight-passive players naturally produce many low-aggression shared pots.

## Case 3 — Coordinated isolation
- **Pair ID:** `PF653436402F6` (members `U023E561BC344`, `U10370091F79C`)
- **Evidence hands:** `H339F3BB44B285F`, `HF0BFAD22FA23D0`
- **Observable suspicious behavior:** The pair applies joint preflop pressure that clears the field: in
  `H339F3BB44B285F` one raises and the other folds immediately, and across the pair's hands their combined
  raising isolates third parties out of pots at an elevated rate. `HF0BFAD22FA23D0` shows a 450-chip preflop
  raising war between the two that ends with both folding — heavy chip commitment with no showdown.
- **Plausible benign explanation:** Two independently aggressive players seated together will mechanically
  produce frequent raise/re-raise dynamics and squeeze the table; the `HF0BFAD22FA23D0` 3-bet/4-bet-then-
  both-fold line is consistent with two loose-aggressive players each putting the other on a bluff and both
  backing down — spew, not coordination.

## Case 4 — Directed transfer
- **Pair ID:** `P8F02C81D9B3B` (members `U47902497B4AC`, `U80E0D24A8B57`)
- **Evidence hands:** `H520F9D51A2348B`, `HDFE0DF189E11C3`
- **Observable suspicious behavior:** `U80E0D24A8B57` leads into `U47902497B4AC` postflop (a flop bet of 23
  into a 16 pot) and then folds to a raise on the next street, giving up the pot after building it — the
  fund-then-surrender shape of a directed transfer. Net chips move from `U80E0D24A8B57` to `U47902497B4AC`
  in both hands.
- **Plausible benign explanation:** A standard failed semi-bluff: `U80E0D24A8B57` bets a draw or weak pair,
  gets raised, and correctly folds when the price is wrong. Bet-fold against a raise is textbook, not
  necessarily a gift; the observed chip flow is the ordinary result of losing the hand.

## Case 5 — Soft play
- **Pair ID:** `PE207D4A85DC0` (members `U66E37FF9B003`, `U9739E12A80FF`)
- **Evidence hands:** `HB20F602E8A5C70`, `H18F9B2CFE78EE6`
- **Observable suspicious behavior:** `U9739E12A80FF` repeatedly commits chips into `U66E37FF9B003` and then
  surrenders. In `H18F9B2CFE78EE6` it calls a preflop raise and two further raises into a 130-chip pot, leads
  the flop, then folds the turn to a small 19-chip bet — abandoning a large invested pot to a token bet. The
  pair's mutual postflop aggression is well below each player's aggression against the wider field (the
  partner-vs-field contrast our model keys on).
- **Plausible benign explanation:** The turn fold can be a disciplined laydown: `U9739E12A80FF` holds `9d8c`
  on a `2c 7h Jc 2h` board (missed everything, only backdoor equity gone) and correctly folds a busted hand
  despite prior investment — sunk-cost-aware, textbook-correct play. Heavy preflop calling followed by giving
  up unimproved is a common losing-but-honest style, not necessarily a coordinated surrender.

---

*Selected submission: `evidence_familycond_submission.csv` (public LB 0.83460). Method infers coordination
from gameplay features only; see solution writeup and reproduction notebook.*
