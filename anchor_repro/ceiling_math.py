"""Can maximizing OUR components reach ~0.88? Honest arithmetic. composite = .70*PairAP + .20*Ev + .10*Beh."""

def comp(p, e, b):
    return 0.70 * p + 0.20 * e + 0.10 * b

print("What 0.88 REQUIRES vs what maximizing our components gives:\n")

scenarios = [
    ("Our shipped (0.70904)",            0.79, 0.29, 0.60, "LB-real: implied PairAP~0.79, our Ev/Beh"),
    ("Max OUR evidence (Ev->0.867 oracle)", 0.79, 0.867, 0.60, "evidence at its ORACLE ceiling, risk unchanged"),
    ("Max evidence + behavior perfect",  0.79, 0.867, 1.00, "Ev oracle + Beh perfect, risk unchanged"),
    ("EVERYTHING at our ceilings",        0.79, 0.867, 1.00, "same as above; risk is the binding limit"),
    ("---", 0,0,0,""),
    ("A 0.88 team (implied)",             0.88, 0.60, 0.80, "back-out: what a 0.88 needs"),
    ("A 0.88 via risk alone",             0.93, 0.60, 0.80, "if they carry it on PairAP"),
]
for name, p, e, b, note in scenarios:
    if name == "---":
        print("-" * 70); continue
    print(f"  {name:<34} PairAP={p:.2f} Ev={e:.2f} Beh={b:.2f} -> {comp(p,e,b):.4f}   {note}")

print("\nThe binding constraint:")
print("  Our PairAP is LB-limited to ~0.79 (eval-real; the 0.92 harness number is HOT).")
print("  With PairAP FROZEN at 0.79, the MAX composite we can reach even with")
print(f"    PERFECT evidence (0.867) AND perfect behavior (1.0) = {comp(0.79, 0.867, 1.0):.4f}")
print(f"    realistic evidence (0.50) + behavior (0.62)         = {comp(0.79, 0.50, 0.62):.4f}")
print()
print("  To reach 0.88 you need PairAP itself near:")
for target in [0.85, 0.88]:
    # solve for PairAP given generous Ev=0.60, Beh=0.80
    p_needed = (target - 0.20*0.60 - 0.10*0.80) / 0.70
    print(f"    composite {target}: PairAP must be >= {p_needed:.3f}  (we are ~0.79 real)")
