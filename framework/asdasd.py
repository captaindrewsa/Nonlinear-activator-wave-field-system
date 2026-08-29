#!/usr/bin/env python3
"""
Export a clean publication-ready activator snapshot from an NPZ file.

Usage:
    python asdasd.py INPUT_SNAPSHOT.npz OUTPUT_IMAGE.png

Example:
    python asdasd.py snapshot2d_DR0_base160_t00001200.npz DR0_phi_t0120_clean.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def get_phi_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """Find the activator field using common key names."""
    candidate_keys = (
        "phi",
        "Phi",
        "PHI",
        "varphi",
        "u",
        "field_phi",
    )

    for key in candidate_keys:
        if key in npz.files:
            value = np.asarray(npz[key])
            if value.ndim == 2:
                return value

    two_dimensional = []
    for key in npz.files:
        value = np.asarray(npz[key])
        if value.ndim == 2:
            two_dimensional.append((key, value))

    if len(two_dimensional) == 1:
        key, value = two_dimensional[0]
        print(f"Using the only 2D array in the archive: {key}")
        return value

    available = ", ".join(npz.files)
    found = ", ".join(key for key, _ in two_dimensional) or "none"
    raise KeyError(
        "Cannot determine which NPZ array is the activator field phi. "
        f"Available keys: {available}. Two-dimensional arrays: {found}."
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage:\n"
            "  python asdasd.py INPUT_SNAPSHOT.npz OUTPUT_IMAGE.png"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_path) as snapshot:
        print("Available NPZ keys:", snapshot.files)
        phi = get_phi_array(snapshot)

    if phi.ndim != 2:
        raise ValueError(f"Expected a 2D phi array, got shape {phi.shape}")

    print(f"phi shape: {phi.shape}")
    print(f"phi range: [{np.nanmin(phi):.6g}, {np.nanmax(phi):.6g}]")

    # These bounds match the visual range used in the existing Figure 3.
    # Keep exactly the same vmin/vmax for the t=120 and t=1400 panels.
    vmin = -3.0
    vmax = 4.2

    # Crop limits in grid/physical coordinates. Edit only if needed,
    # but use identical limits for both time snapshots.
    x_min, x_max = 48, 112
    y_min, y_max = 48, 112

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)

    ax.imshow(
        phi,
        origin="lower",
        extent=[0, phi.shape[1], 0, phi.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        resample=False,
        aspect="equal",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output was not created correctly: {output_path}")

    print(f"Wrote: {output_path.resolve()}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()