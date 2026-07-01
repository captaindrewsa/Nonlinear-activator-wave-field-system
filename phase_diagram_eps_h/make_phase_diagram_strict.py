#!/usr/bin/env python3
"""
Strict (eps, h) phase diagram from phase_map_eps_h_data_strict.csv.

Color scheme:
    dark blue -> 0 (collapse)
    orange    -> 1 (stationary autosoliton)
    red       -> 2 (breathing autosoliton)
    gray      -> -1 (numerically unstable / divergent), also marked with a cross

Only directly computed points are used (no analytic extrapolation).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

CSV = "/home/user/workspace/phase_map_eps_h_data_strict.csv"
PDF = "/home/user/workspace/fig_parameter_map_eps_h_strict.pdf"

COLOR = {-1: "#9e9e9e",   # gray   - numerically unstable
          0: "#1f3b73",   # dark blue - collapse
          1: "#ff7f0e",   # orange - stationary autosoliton
          2: "#d62728"}   # red    - breathing autosoliton
LABEL = {-1: "Numerically unstable (-1)",
          0: "Collapse (0)",
          1: "Stationary autosoliton (1)",
          2: "Breathing autosoliton (2)"}

rows = list(csv.DictReader(open(CSV)))
data = [(float(r["eps"]), float(r["h"]), int(float(r["regime_code"]))) for r in rows]

eps_vals = sorted(set(d[0] for d in data))
h_vals = sorted(set(d[1] for d in data))
lookup = {(e, h): c for e, h, c in data}

# regime grid (rows = h, cols = eps); map codes -1,0,1,2 -> 0,1,2,3
code_to_idx = {-1: 0, 0: 1, 1: 2, 2: 3}
grid = np.full((len(h_vals), len(eps_vals)), np.nan)
for j, e in enumerate(eps_vals):
    for i, h in enumerate(h_vals):
        if (e, h) in lookup:
            grid[i, j] = code_to_idx[lookup[(e, h)]]

fig, ax = plt.subplots(figsize=(10, 6.5))

cmap = ListedColormap([COLOR[-1], COLOR[0], COLOR[1], COLOR[2]])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

# cell edges (midpoints) for a clean tiled map
def edges(vals):
    vals = np.array(vals, dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2
    first = vals[0] - (vals[1] - vals[0]) / 2
    last = vals[-1] + (vals[-1] - vals[-2]) / 2
    return np.concatenate([[first], mids, [last]])

eps_edges = edges(eps_vals)
h_edges = edges(h_vals)

ax.pcolormesh(eps_edges, h_edges, grid, cmap=cmap, norm=norm,
              shading="flat", edgecolors="white", linewidth=0.6)

# overlay markers: small black dot for every computed point,
# and a black cross 'x' for numerically unstable (-1) points
for e, h, c in data:
    if c == -1:
        ax.plot(e, h, "x", color="black", markersize=9, markeredgewidth=2.0, zorder=6)
    else:
        ax.plot(e, h, "o", color="black", markersize=3, zorder=5)

ax.set_xlabel(r"$\varepsilon$", fontsize=14)
ax.set_ylabel(r"$h$", fontsize=14)
ax.set_title(r"Strict phase diagram in the $(\varepsilon,\,h)$ plane "
             r"(direct PDE integration)", fontsize=12.5)

ax.set_xticks(eps_vals)
ax.set_xticklabels([f"{e:g}" for e in eps_vals], rotation=45, ha="right", fontsize=9)
ax.set_yticks(h_vals)
ax.tick_params(labelsize=10)

handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=12,
                  markerfacecolor=COLOR[k], markeredgecolor="white",
                  label=LABEL[k]) for k in (0, 1, 2, -1)]
handles.append(Line2D([0], [0], marker="x", linestyle="none", markersize=9,
                      markeredgewidth=2.0, color="black",
                      label="unstable point marker"))
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=True, fontsize=9.5, title="Regime")

fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.16)
fig.savefig(PDF, bbox_inches="tight", pad_inches=0.4)
print("Wrote", PDF)
from collections import Counter
print("counts:", dict(Counter(c for _, _, c in data)))
