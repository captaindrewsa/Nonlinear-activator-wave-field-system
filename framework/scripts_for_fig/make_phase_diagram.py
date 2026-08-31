#!/usr/bin/env python3
"""Build Fig. 4: baseline operational classification from scan CSV data.

Required CSV columns: eps, h, regime_code.
Optional: regime_label. Codes: -1 unstable, 0 collapsed,
1 weakly modulated/unresolved, 2 breathing.

Example:
    python make_phase_diagram.py phase_map_eps_h_data.csv cnsns_fig09_phasediagram_epsh.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


COLORS = {
    -1: "#9a9a9a",
     0: "#1f3b73",
     1: "#ff7f0e",
     2: "#d62728",
}
LABELS = {
    -1: "Numerically unstable (-1)",
     0: "Collapsed (0)",
     1: "Weakly modulated / unresolved (1)",
     2: "Breathing (2)",
}


def cell_edges(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two distinct values are required on each axis.")
    mids = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (mids[0] - values[0]), mids, values[-1] + (values[-1] - mids[-1])]


def load_rows(path: Path) -> list[tuple[float, float, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"eps", "h", "regime_code"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must have headers: eps,h,regime_code. "
                f"Found: {reader.fieldnames}"
            )
        rows = []
        for n, row in enumerate(reader, start=2):
            try:
                eps = float(row["eps"])
                h = float(row["h"])
                code = int(float(row["regime_code"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric data at CSV line {n}: {row}") from exc
            if code not in COLORS:
                raise ValueError(f"Unknown regime_code {code} at CSV line {n}.")
            rows.append((eps, h, code))
    if not rows:
        raise ValueError("The CSV contains no data rows.")
    return rows


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python make_phase_diagram.py INPUT.csv OUTPUT.pdf\n"
            "Required CSV columns: eps,h,regime_code"
        )

    csv_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])
    rows = load_rows(csv_path)

    eps_values = sorted({eps for eps, _, _ in rows})
    h_values = sorted({h for _, h, _ in rows})
    eps_index = {value: index for index, value in enumerate(eps_values)}
    h_index = {value: index for index, value in enumerate(h_values)}

    grid = np.full((len(h_values), len(eps_values)), np.nan)
    for eps, h, code in rows:
        grid[h_index[h], eps_index[eps]] = code

    finite_codes = [-1, 0, 1, 2]
    cmap = ListedColormap([COLORS[code] for code in finite_codes])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 5.9))
    ax.pcolormesh(
        cell_edges(eps_values),
        cell_edges(h_values),
        grid,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.7,
        zorder=1,
    )

    for eps, h, code in rows:
        if code == -1:
            ax.plot(eps, h, marker="x", color="black", markersize=8,
                    markeredgewidth=1.7, linestyle="none", zorder=3)
        else:
            ax.plot(eps, h, marker="o", color="black", markersize=2.7,
                    linestyle="none", zorder=3)

    ax.set_title(r"Baseline operational classification in the $(\varepsilon,h)$ plane", pad=10)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$h$")
    ax.set_xticks(eps_values)
    ax.set_xticklabels([f"{x:g}" for x in eps_values], rotation=45, ha="right")
    ax.set_yticks(h_values)
    ax.set_yticklabels([f"{y:.3f}" for y in h_values])
    ax.tick_params(direction="in", top=True, right=True)

    present_codes = sorted({code for _, _, code in rows})
    handles = []
    for code in present_codes:
        if code == -1:
            handles.append(Line2D([0], [0], marker="x", color="black", linestyle="none",
                                  markersize=8, markeredgewidth=1.7, label=LABELS[code]))
        else:
            handles.append(Line2D([0], [0], marker="s", linestyle="none", markersize=10,
                                  markerfacecolor=COLORS[code], markeredgecolor="white",
                                  label=LABELS[code]))
    handles.append(Line2D([0], [0], marker="o", color="black", linestyle="none",
                          markersize=3, label="Simulated point"))

    ax.legend(handles=handles, title="Regime", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=9, title_fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.72, top=0.88, bottom=0.18)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    counts = {LABELS[code]: sum(1 for _, _, value in rows if value == code) for code in present_codes}
    print(f"Wrote: {pdf_path}")
    print(f"Input: {csv_path}")
    print(f"Grid: {len(eps_values)} eps values x {len(h_values)} h values")
    print("Counts:", counts)


if __name__ == "__main__":
    main()
