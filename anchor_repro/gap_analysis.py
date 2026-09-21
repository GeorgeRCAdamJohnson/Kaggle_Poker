"""Which component is our WEAKEST — by recoverable weighted points, not just lowest number."""

# name, weight, current_dev, ceiling, note
comps = [
    ("PairAP (risk)",   0.70, 0.92, 0.98, "eval-honest ~0.92; LB-saturated in our lineage (arm knob dead-even)"),
    ("Evidence-MAP@5",  0.20, 0.29, 0.867, "oracle retriever ceiling 0.867; we capture only 0.29"),
    ("Behavior-MAP",    0.10, 0.90, 1.00, "dev 0.90; already near ceiling"),
]

print(f"{'component':<18}{'wt':>5}{'cur':>7}{'ceil':>8}{'gap':>7}{'wtd_recov':>11}")
total = 0.0
for n, w, cur, ceil, note in comps:
    gap = ceil - cur
    wr = w * gap
    total += wr
    print(f"{n:<18}{w:>5.2f}{cur:>7.2f}{ceil:>8.3f}{gap:>7.3f}{wr:>11.4f}   {note}")
print(f"\nTOTAL weighted recoverable (to ceilings): {total:.4f}")

print("\nFraction of its OWN ceiling captured (lower = weaker):")
for n, w, cur, ceil, note in comps:
    print(f"  {n:<18} {cur/ceil*100:5.1f}%")

print("\nRealistic (not-oracle) recoverable, if evidence 0.29 -> 0.50 (learned rankers hit this):")
print(f"  Evidence gain: 0.20 * (0.50 - 0.29) = {0.20*(0.50-0.29):.4f} composite points")
print(f"  PairAP gain (saturated, realistic ~0): ~0.0000")
print(f"  Behavior gain (near ceiling, realistic ~0.02): {0.10*0.02:.4f}")
