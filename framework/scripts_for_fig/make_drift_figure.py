#!/usr/bin/env python3
"""Build a three-panel Figure 3 for asymmetric two-spot transients.

Inputs
------
1) A CSV track file for every run, containing either:
   A. time and centre-of-mass coordinate columns, or
   B. time, x, y, phi columns that describe an activator field snapshot.
2) Two rendered activator snapshots for the baseline run, supplied as image files.

The script uses DR0 (baseline) and DR2 (larger-domain validation) by default.
DR3 may be added as an optional asymmetry-control curve. DR1 must not be used
in the late-time plot because it terminates near t=20.3.

Examples
--------
Using precomputed centre-of-mass tracks:
python make_drift_figure.py \
  --dr0-track /path/DR0base160/track.csv \
  --dr2-track /path/DR2domain240/track.csv \
  --dr3-track /path/DR3offsetdy4/track.csv \
  --snapshot-early /path/DR0_phi_t0120.png \
  --snapshot-late /path/DR0_phi_t1400.png \
  --output cnsns_fig08_drift_validation.pdf

If coordinate column names are unusual, specify them explicitly:
python make_drift_figure.py ... --time-col t --x-col x_cm --y-col y_cm

If a CSV stores a flattened field, use --field-mode and specify --nx, --ny,
--dx, plus its coordinate/value columns:
python make_drift_figure.py ... --field-mode --time-col time --x-col x \
  --y-col y --phi-col phi --nx 160 --ny 160 --dx 1.0
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


TIME_CANDIDATES = ("time", "t", "Time", "T")
XCM_CANDIDATES = ("x_cm", "xcm", "X_CM", "Xcm", "com_x", "cm_x", "xCOM")
YCM_CANDIDATES = ("y_cm", "ycm", "Y_CM", "Ycm", "com_y", "cm_y", "yCOM")
X_CANDIDATES = ("x", "X")
Y_CANDIDATES = ("y", "Y")
PHI_CANDIDATES = ("phi", "Phi", "varphi", "u")


def choose_column(names: list[str], requested: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in names:
            raise ValueError(f"Requested {kind} column {requested!r} not found. Available: {names}")
        return requested
    for name in candidates:
        if name in names:
            return name
    raise ValueError(f"Could not identify {kind} column. Available: {names}. Use --{kind.replace('_', '-')}.")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return reader.fieldnames, rows


def circular_mean_coordinate(coords: np.ndarray, weights: np.ndarray, length: float) -> float:
    theta = 2.0 * np.pi * coords / length
    s = np.sum(weights * np.sin(theta))
    c = np.sum(weights * np.cos(theta))
    angle = math.atan2(s, c)
    if angle < 0:
        angle += 2.0 * np.pi
    return length * angle / (2.0 * np.pi)


def unwrap_periodic(values: np.ndarray, length: float) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if delta > length / 2.0:
            out[i:] -= length
        elif delta < -length / 2.0:
            out[i:] += length
    return out


def load_cm_track(args: argparse.Namespace, track_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names, rows = read_csv_rows(track_path)
    t_col = choose_column(names, args.time_col, TIME_CANDIDATES, "time")

    if not args.field_mode:
        x_col = choose_column(names, args.x_col, XCM_CANDIDATES, "x_cm")
        y_col = choose_column(names, args.y_col, YCM_CANDIDATES, "y_cm")
        t = np.asarray([float(row[t_col]) for row in rows])
        x = np.asarray([float(row[x_col]) for row in rows])
        y = np.asarray([float(row[y_col]) for row in rows])
        return t, x, y

    x_col = choose_column(names, args.x_col, X_CANDIDATES, "x")
    y_col = choose_column(names, args.y_col, Y_CANDIDATES, "y")
    phi_col = choose_column(names, args.phi_col, PHI_CANDIDATES, "phi")
    if args.nx is None or args.ny is None or args.dx is None:
        raise ValueError("--field-mode requires --nx, --ny, and --dx.")

    values = {}
    for row in rows:
        t = float(row[t_col])
        values.setdefault(t, []).append((float(row[x_col]), float(row[y_col]), float(row[phi_col])))

    length_x = args.nx * args.dx
    length_y = args.ny * args.dx
    times, xs, ys = [], [], []
    for t in sorted(values):
        arr = np.asarray(values[t], dtype=float)
        weights = np.maximum(arr[:, 2] - args.phi_lo, 0.0)
        if weights.sum() <= 0:
            continue
        times.append(t)
        xs.append(circular_mean_coordinate(arr[:, 0], weights, length_x))
        ys.append(circular_mean_coordinate(arr[:, 1], weights, length_y))
    return np.asarray(times), np.asarray(xs), np.asarray(ys)


def displacement(t: np.ndarray, x: np.ndarray, y: np.ndarray, length: float, reference_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    x_unwrapped = unwrap_periodic(x, length)
    y_unwrapped = unwrap_periodic(y, length)
    ref_idx = int(np.argmin(np.abs(t - reference_time)))
    dr = np.hypot(x_unwrapped - x_unwrapped[ref_idx], y_unwrapped - y_unwrapped[ref_idx])
    return t, dr, np.array([x_unwrapped[ref_idx], y_unwrapped[ref_idx]])

def moving_average(values, window=401):
    """Centred moving average used only to visualise the slow trend."""
    values = np.asarray(values, dtype=float)

    if window < 3:
        return values.copy()

    if window % 2 == 0:
        window += 1

    if window >= len(values):
        window = max(3, len(values) // 5)
        if window % 2 == 0:
            window += 1

    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def add_snapshot(ax: plt.Axes, path: Path, panel_label: str, time_label: str) -> None:
    image = mpimg.imread(path)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(f"({panel_label}) {time_label}", fontsize=10, pad=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a three-panel drift-validation figure.")
    parser.add_argument("--dr0-track", required=True, help="Baseline DR0 track CSV")
    parser.add_argument("--dr2-track", required=True, help="Larger-domain DR2 track CSV")
    parser.add_argument("--dr3-track", help="Optional offset-asymmetry DR3 track CSV")
    parser.add_argument("--snapshot-early", required=True, help="Baseline snapshot near t=120")
    parser.add_argument("--snapshot-late", required=True, help="Baseline snapshot near t=1400")
    parser.add_argument("--output", default="cnsns_fig08_drift_validation.pdf", help="Output PDF path")
    parser.add_argument("--reference-time", type=float, default=120.0, help="Reference time for displacement")
    parser.add_argument("--dr0-length", type=float, default=160.0, help="DR0 domain side length")
    parser.add_argument("--dr2-length", type=float, default=240.0, help="DR2 domain side length")
    parser.add_argument("--dr3-length", type=float, default=160.0, help="DR3 domain side length")
    parser.add_argument("--time-col", help="Time column name")
    parser.add_argument("--x-col", help="CM x-column, or x-column in --field-mode")
    parser.add_argument("--y-col", help="CM y-column, or y-column in --field-mode")
    parser.add_argument("--field-mode", action="store_true", help="Compute CM from flattened field CSV")
    parser.add_argument("--phi-col", help="Activator-value column in --field-mode")
    parser.add_argument("--phi-lo", type=float, default=0.533, help="Background activator level")
    parser.add_argument("--nx", type=int, help="Grid nx in --field-mode")
    parser.add_argument("--ny", type=int, help="Grid ny in --field-mode")
    parser.add_argument("--dx", type=float, help="Grid spacing in --field-mode")
    args = parser.parse_args()

    dr0_path = Path(args.dr0_track)
    dr2_path = Path(args.dr2_track)
    dr3_path = Path(args.dr3_track) if args.dr3_track else None
    image_early = Path(args.snapshot_early)
    image_late = Path(args.snapshot_late)
    output = Path(args.output)
    for path in (dr0_path, dr2_path, image_early, image_late):
        if not path.exists():
            raise FileNotFoundError(path)
    if dr3_path and not dr3_path.exists():
        raise FileNotFoundError(dr3_path)

    t0, x0, y0 = load_cm_track(args, dr0_path)
    t2, x2, y2 = load_cm_track(args, dr2_path)
    t0, dr0, _ = displacement(t0, x0, y0, args.dr0_length, args.reference_time)
    t2, dr2, _ = displacement(t2, x2, y2, args.dr2_length, args.reference_time)

    dr3 = None
    if dr3_path:
        t3, x3, y3 = load_cm_track(args, dr3_path)
        t3, dr3, _ = displacement(t3, x3, y3, args.dr3_length, args.reference_time)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
    })
    fig = plt.figure(figsize=(7.15, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.88))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    add_snapshot(ax_a, image_early, "a", rf"$t\approx{args.reference_time:g}$")
    add_snapshot(ax_b, image_late, "b", r"$t\approx1400$")

    # Show fast breathing-scale fluctuations only as faint background traces.
    ax_c.plot(
        t0, dr0,
        color="#1f3b73",
        linewidth=0.45,
        alpha=0.18,
        zorder=1,
    )
    ax_c.plot(
        t2, dr2,
        color="#bc6c25",
        linewidth=0.45,
        alpha=0.18,
        zorder=1,
    )
    if dr3 is not None:
        ax_c.plot(
            t3, dr3,
            color="#6a994e",
            linewidth=0.45,
            alpha=0.16,
            zorder=1,
        )
    # Smooth curves are for visualising the slow displacement trend.
    smooth_window = 401
    dr0_smooth = moving_average(dr0, smooth_window)
    dr2_smooth = moving_average(dr2, smooth_window)
    ax_c.plot(
        t0, dr0_smooth,
        color="#1f3b73",
        linewidth=2.1,
        label=r"DR0: $L=160$, $\Delta x=1$",
        zorder=4,
    )
    ax_c.plot(
        t2, dr2_smooth,
        color="#bc6c25",
        linewidth=2.1,
        linestyle="--",
        label=r"DR2: $L=240$, $\Delta x=1$",
        zorder=4,
    )
    dr3_smooth = None
    if dr3 is not None:
        dr3_smooth = moving_average(dr3, smooth_window)
        ax_c.plot(
            t3, dr3_smooth,
            color="#6a994e",
            linewidth=1.9,
            linestyle=":",
            label=r"DR3: offset asymmetry",
            zorder=4,
        )
    # Final-time markers: the visible dots at the end of every smoothed curve.
    ax_c.scatter(
        [t0[-1]],
        [dr0_smooth[-1]],
        s=36,
        color="#1f3b73",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    ax_c.scatter(
        [t2[-1]],
        [dr2_smooth[-1]],
        s=36,
        color="#bc6c25",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    if dr3_smooth is not None:
        ax_c.scatter(
            [t3[-1]],
            [dr3_smooth[-1]],
            s=36,
            color="#6a994e",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
    # A compact numerical summary in the lower-right corner.
    final_text = (
        "Final $\\Delta R_{\\rm CM}$:\n"
        f"DR0 = {dr0[-1]:.2f}\n"
        f"DR2 = {dr2[-1]:.2f}"
    )
    if dr3 is not None:
        final_text += f"\nDR3 = {dr3[-1]:.2f}"
    ax_c.text(
        0.985,
        0.05,
        final_text,
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax_c.axvline(
        args.reference_time,
        color="0.35",
        linewidth=0.8,
        linestyle="-.",
        zorder=0,
    )
    
    ax_c.axvline(args.reference_time, color="0.35", linewidth=0.8, linestyle="-.", zorder=0)
    ax_c.set_xlabel(r"time $t$")
    ax_c.set_ylabel(r"$\Delta R_{\rm CM}(t)$")
    ax_c.set_title(r"(c) Centre-of-mass displacement", pad=7)
    ax_c.grid(True, color="0.87", linewidth=0.6)
    ax_c.set_xlim(left=0)
    ax_c.set_ylim(bottom=0)
    ax_c.set_ylim(0, 2.95)
    ax_c.legend(
        loc="upper left",
        bbox_to_anchor=(0.06, 0.99),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Wrote: {output}")
    print(f"DR0 final displacement: {dr0[-1]:.5g}")
    print(f"DR2 final displacement: {dr2[-1]:.5g}")
    if dr3 is not None:
        print(f"DR3 final displacement: {dr3[-1]:.5g}")


if __name__ == "__main__":
    main()
