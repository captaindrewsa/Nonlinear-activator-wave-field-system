#!/usr/bin/env python3
"""
Build a strictly data-based (eps, h) phase diagram from the scan CSV.

Discrete colormap (consistent with the paper):
    blue (dark)  -> collapse (0)
    orange       -> stationary autosoliton (1)
    red          -> breathing autosoliton (2)

Only the simulated points are used (no analytic extrapolation).
Numerically unstable points (regime_code = -1, eps=1.5) are excluded.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

CSV = "/home/user/workspace/phase_map_eps_h_data.csv"
PDF = "/home/user/workspace/fig_parameter_map_eps_h_computed.pdf"

# discrete colors per regime code
COLOR = {0: "#1f3b73",   # dark blue  - collapse
         1: "#ff7f0e",   # orange     - stationary autosoliton
         2: "#d62728"}   # red        - breathing autosoliton
LABEL = {0: "Collapse (0)",
         1: "Stationary autosoliton (1)",
         2: "Breathing autosoliton (2)"}

rows = list(csv.DictReader(open(CSV)))
data = []
for r in rows:
    code = int(float(r["regime_code"]))
    if code < 0:
        continue  # skip numerically unstable points
    data.append((float(r["eps"]), float(r["h"]), code))

eps_vals = sorted(set(d[0] for d in data))
h_vals = sorted(set(d[1] for d in data))

# build a regime grid for the heatmap (rows = h, cols = eps)
grid = np.full((len(h_vals), len(eps_vals)), np.nan)
lookup = {(e, h): c for e, h, c in data}
for j, e in enumerate(eps_vals):
    for i, h in enumerate(h_vals):
        if (e, h) in lookup:
            grid[i, j] = lookup[(e, h)]

# ---- plot ----
fig, ax = plt.subplots(figsize=(10, 6.5))

cmap = ListedColormap([COLOR[0], COLOR[1], COLOR[2]])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

# pcolormesh with cell edges so each simulated point is a clear tile
de = np.diff(eps_vals)
dh = np.diff(h_vals)
eps_edges = np.concatenate([[eps_vals[0] - de[0] / 2],
                            (np.array(eps_vals[:-1]) + np.array(eps_vals[1:])) / 2,
                            [eps_vals[-1] + de[-1] / 2]])
h_edges = np.concatenate([[h_vals[0] - dh[0] / 2],
                          (np.array(h_vals[:-1]) + np.array(h_vals[1:])) / 2,
                          [h_vals[-1] + dh[-1] / 2]])

mesh = ax.pcolormesh(eps_edges, h_edges, grid, cmap=cmap, norm=norm,
                     shading="flat", edgecolors="white", linewidth=0.6)

# overlay the actual simulated points as markers
for e, h, c in data:
    ax.plot(e, h, "o", color="black", markersize=3, zorder=5)

ax.set_xlabel(r"$\varepsilon$", fontsize=14)
ax.set_ylabel(r"$h$", fontsize=14)
ax.set_title(r"Phase diagram in the $(\varepsilon,\,h)$ plane (simulated data)",
             fontsize=13)

ax.set_xticks(eps_vals)
ax.set_xticklabels([f"{e:g}" for e in eps_vals], rotation=45, ha="right", fontsize=9)
ax.set_yticks(h_vals)
ax.tick_params(labelsize=10)

# discrete legend
handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=12,
                  markerfacecolor=COLOR[k], markeredgecolor="white",
                  label=LABEL[k]) for k in (0, 1, 2)]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=True, fontsize=10, title="Regime")

fig.subplots_adjust(left=0.12, right=0.80, top=0.90, bottom=0.14)
fig.savefig(PDF, bbox_inches="tight", pad_inches=0.4)
print("Wrote", PDF)
print("eps used:", eps_vals)
print("h used:", h_vals)
from collections import Counter
print("regime counts (plotted):", dict(Counter(c for _, _, c in data)))
