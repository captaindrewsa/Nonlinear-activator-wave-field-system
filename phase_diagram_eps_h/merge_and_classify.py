#!/usr/bin/env python3
"""Merge partA/partB raw results, re-apply the (corrected) strict
classification, and write the final strict CSV."""
import csv

PHI_LO = 0.533
DELTA_PHI = 0.3
A_THR = 0.15
R_THR = 5.0

rows = []
for fn in ["strict_partA.csv", "strict_partB.csv"]:
    rows += list(csv.DictReader(open(fn)))


def reclassify(r):
    code = int(float(r["regime_code"]))
    if code == -1:
        return -1  # numerically unstable / divergent -> keep
    phi_max = float(r["phi_max"])
    A = float(r["A"])
    ratio = float(r["freq_ratio"])
    if phi_max <= PHI_LO + DELTA_PHI:
        return 0  # collapse (phi_max alone)
    elif (A > A_THR) and (ratio >= R_THR):
        return 2  # breathing
    else:
        return 1  # stationary


for r in rows:
    r["regime_code"] = reclassify(r)

rows.sort(key=lambda r: (float(r["eps"]), float(r["h"])))

fieldnames = ["eps", "h", "A", "phi_max", "phi_min", "phi_mean",
              "dominant_freq", "freq_ratio", "regime_code"]
out = "/home/user/workspace/phase_map_eps_h_data_strict.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in fieldnames})

from collections import Counter
c = Counter(r["regime_code"] for r in rows)
print("Wrote", out)
print("regime counts:", dict(c))
# summary by eps
print("\nregime by eps (h=0.533 column):")
for r in rows:
    if abs(float(r["h"]) - 0.533) < 1e-9:
        print(f"  eps={float(r['eps']):4.1f} -> code {r['regime_code']}  "
              f"(A={float(r['A']) if r['A']==r['A'] else float('nan'):.3f}, "
              f"phi_max={float(r['phi_max']) if r['phi_max']==r['phi_max'] else float('nan'):.3f})")
