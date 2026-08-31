#!/usr/bin/env python3
"""Build Figure 2: exploratory two-spot outcomes versus initial separation.

Required input CSV columns:
    d0, df

Optional columns:
    outcome  - one of: resolved, merged, decayed, separated, unresolved
    t_eval   - time at which df was measured (used only in annotation)

Rows with a numeric df are plotted as unconnected markers. Rows with a missing
or non-numeric df may still be shown as categorical outcomes if outcome is set.

Example:
    python make_two_spot_outcomes.py two_spot_outcomes.csv cnsns_fig07_two_spot_outcomes.pdf
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


OUTCOME_STYLE = {
    "resolved": {"label": "Two localised maxima resolved", "marker": "o", "color": "#3b4f7a"},
    "merged": {"label": "Merged", "marker": "D", "color": "#7a3e2b"},
    "decayed": {"label": "One or both spots decayed", "marker": "x", "color": "#6f6f6f"},
    "separated": {"label": "Separated / not bound", "marker": "^", "color": "#a16616"},
    "unresolved": {"label": "Unresolved", "marker": "s", "color": "#7a5e9b"},
}

ALIASES = {
    "two_spots": "resolved",
    "two_spot": "resolved",
    "two peaks": "resolved",
    "two_peaks": "resolved",
    "bound": "resolved",
    "merged": "merged",
    "merge": "merged",
    "decayed": "decayed",
    "decay": "decayed",
    "collapsed": "decayed",
    "separated": "separated",
    "separation": "separated",
    "unbound": "separated",
    "unresolved": "unresolved",
}


def numeric(value: str | None) -> float:
    if value is None:
        return float("nan")
    value = value.strip()
    if not value or value.lower() in {"nan", "na", "n/a", "none", "-"}:
        return float("nan")
    return float(value)


def canonical_outcome(value: str | None, df: float) -> str:
    if value is None or not value.strip():
        return "resolved" if math.isfinite(df) else "unresolved"
    key = value.strip().lower().replace("-", "_")
    return ALIASES.get(key, key if key in OUTCOME_STYLE else "unresolved")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python make_two_spot_outcomes.py INPUT.csv OUTPUT.pdf\n"
            "Required columns: d0,df; optional: outcome,t_eval"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "d0" not in reader.fieldnames or "df" not in reader.fieldnames:
            raise ValueError(
                "CSV must include headers d0 and df. "
                f"Found: {reader.fieldnames}"
            )
        rows = list(reader)

    data = []
    for line, row in enumerate(rows, start=2):
        d0 = numeric(row.get("d0"))
        df = numeric(row.get("df"))
        if not math.isfinite(d0):
            raise ValueError(f"CSV line {line}: d0 must be numeric.")
        outcome = canonical_outcome(row.get("outcome"), df)
        data.append((d0, df, outcome))

    data.sort(key=lambda item: item[0])
    d0_values = [item[0] for item in data]
    finite_df = [item[1] for item in data if math.isfinite(item[1])]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
    })
    fig, ax = plt.subplots(figsize=(6.7, 4.6))

    present = []
    for outcome, style in OUTCOME_STYLE.items():
        subset = [(d0, df) for d0, df, label in data if label == outcome]
        if not subset:
            continue
        present.append(outcome)
        valid = [(d0, df) for d0, df in subset if math.isfinite(df)]
        invalid = [(d0, df) for d0, df in subset if not math.isfinite(df)]

        if valid:
            x, y = zip(*valid)
            ax.scatter(
                x, y,
                s=62,
                marker=style["marker"],
                facecolors=style["color"] if style["marker"] != "x" else "none",
                edgecolors="black" if style["marker"] != "x" else style["color"],
                linewidths=0.75 if style["marker"] != "x" else 1.4,
                zorder=3,
            )

        if invalid:
            # Put categorical outcomes just below the numerical axis.
            y_marker = (min(finite_df) - 0.9) if finite_df else 0.0
            x, _ = zip(*invalid)
            ax.scatter(
                x, [y_marker] * len(x),
                s=62,
                marker=style["marker"],
                color=style["color"],
                linewidths=1.1,
                zorder=3,
            )

    ax.set_title("Two-spot outcomes versus initial separation", pad=9)
    ax.set_xlabel(r"initial centre-to-centre separation $d_0$")
    ax.set_ylabel(r"late-time separation estimate $d_f$")
    ax.grid(True, color="0.88", linewidth=0.7, zorder=0)
    ax.tick_params(direction="in", top=True, right=True)

    if d0_values:
        ax.set_xticks(sorted(set(d0_values)))
        ax.set_xlim(min(d0_values) - 0.6, max(d0_values) + 0.6)
    if finite_df:
        lower = max(0.0, min(finite_df) - 1.2)
        upper = max(finite_df) + 1.4
        ax.set_ylim(lower, upper)

    handles = [
        Line2D(
            [0], [0],
            marker=OUTCOME_STYLE[key]["marker"],
            linestyle="none",
            markersize=7,
            markerfacecolor=OUTCOME_STYLE[key]["color"] if OUTCOME_STYLE[key]["marker"] != "x" else "none",
            markeredgecolor="black" if OUTCOME_STYLE[key]["marker"] != "x" else OUTCOME_STYLE[key]["color"],
            color=OUTCOME_STYLE[key]["color"],
            label=OUTCOME_STYLE[key]["label"],
        )
        for key in present
    ]
    if handles:
        ax.legend(handles=handles, title="Late-time outcome", loc="best", frameon=True, framealpha=0.93)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Wrote: {output_path.resolve()}")
    print(f"Input rows: {len(data)}")
    print("No connecting line is drawn between sampled initial separations.")


if __name__ == "__main__":
    main()
