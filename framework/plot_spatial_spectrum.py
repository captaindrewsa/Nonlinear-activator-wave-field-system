#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute azimuthally averaged 2D Fourier power spectra "
            "from late-time .npz snapshots produced by sim_framework."
        )
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to one simulation run directory containing config.json and snapshots2d/",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: <run_dir>/processed_spatial_spectrum",
    )
    parser.add_argument(
        "--t_min",
        type=float,
        default=480.0,
        help="Use only snapshots with t >= t_min. Default: 480.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=800.0,
        help="Use only snapshots with t <= t_max. Default: 800.",
    )
    parser.add_argument(
        "--max_snapshots",
        type=int,
        default=48,
        help="Maximum number of uniformly selected snapshots. Default: 48.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of radial-wavenumber bins. Default: 120.",
    )
    parser.add_argument(
        "--remove_mean",
        action="store_true",
        help="Subtract the spatial mean of each field before FFT.",
    )
    parser.add_argument(
        "--window",
        choices=["none", "hann"],
        default="hann",
        help="Spatial window before FFT. Default: hann.",
    )
    parser.add_argument(
        "--show_2d",
        action="store_true",
        help="Add 2D log-power spectrum panels for a representative phi snapshot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def time_from_filename(path: Path) -> float:
    """
    Extract simulation time from sim_framework snapshot filename.

    Expected examples:
      snapshot2d_run_name_t00004800.npz
      snapshot2d_run_name_t00008000_final.npz

    In the current framework, the integer after 't' is time * 10.
    Thus t00004800 corresponds to t = 480.0.
    """
    match = re.search(r"_t(\d+)(?:_final|_unstablelast)?\.npz$", path.name)
    if not match:
        return float("nan")

    return float(int(match.group(1))) / 10.0


def read_snapshot(path: Path) -> Tuple[float, Dict[str, np.ndarray]]:
    filename_time = time_from_filename(path)

    with np.load(path) as data:
        if "t" in data:
            try:
                stored_time = float(np.ravel(data["t"])[0])
            except (TypeError, ValueError, IndexError):
                stored_time = float("nan")
        else:
            stored_time = float("nan")

        if not np.isfinite(stored_time):
            t = filename_time
        else:
            t = stored_time

        fields = {}
        for name in ("phi", "psi", "v"):
            if name in data:
                fields[name] = np.asarray(data[name], dtype=np.float64)

    return t, fields


def snapshot_paths(run_dir: Path) -> List[Path]:
    snap_dir = run_dir / "snapshots_2d"
    paths = sorted(Path(p) for p in glob.glob(str(snap_dir / "*.npz")))
    if not paths:
        paths = sorted(Path(p) for p in glob.glob(str(run_dir / "*.npz")))
    return paths


def select_snapshots(
    paths: List[Path],
    t_min: float,
    t_max: float,
    max_snapshots: int,
) -> List[Tuple[Path, float]]:
    indexed = []
    for path in paths:
        try:
            t, _ = read_snapshot(path)
        except Exception:
            continue
        tol = 1e-6 * max(1.0, abs(t_min), abs(t_max))
        if np.isfinite(t) and (t_min - tol) <= t <= (t_max + tol):
            indexed.append((path, t))

    indexed.sort(key=lambda x: x[1])

    if not indexed:
        raise RuntimeError(
            f"No readable snapshots found in requested interval "
            f"[{t_min}, {t_max}]."
        )

    if len(indexed) <= max_snapshots:
        return indexed

    idx = np.linspace(0, len(indexed) - 1, max_snapshots).round().astype(int)
    return [indexed[i] for i in np.unique(idx)]


def hann2d(shape: Tuple[int, int]) -> np.ndarray:
    wx = np.hanning(shape[0])
    wy = np.hanning(shape[1])
    return np.outer(wx, wy)


def radial_wavenumber_grid(shape: Tuple[int, int], dx: float) -> Tuple[np.ndarray, float]:
    nx, ny = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    kr = np.hypot(kx_grid, ky_grid)
    k_nyquist = np.pi / dx
    return kr, k_nyquist


def radial_average(
    power: np.ndarray,
    kr: np.ndarray,
    bins: int,
    k_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, k_max, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    flat_k = kr.ravel()
    flat_p = power.ravel()

    valid = np.isfinite(flat_k) & np.isfinite(flat_p) & (flat_k <= k_max)
    bin_index = np.digitize(flat_k[valid], edges) - 1
    inside = (bin_index >= 0) & (bin_index < bins)

    sums = np.bincount(
        bin_index[inside],
        weights=flat_p[valid][inside],
        minlength=bins,
    )
    counts = np.bincount(bin_index[inside], minlength=bins)

    radial = sums / np.maximum(counts, 1)
    return centres, radial


def normalise_nonzero(k: np.ndarray, power: np.ndarray) -> np.ndarray:
    mask = k > 0.0
    total = np.trapz(power[mask], k[mask]) if np.any(mask) else 0.0
    if not np.isfinite(total) or total <= 0.0:
        return power.copy()
    return power / total


def high_k_fraction(k: np.ndarray, power: np.ndarray, cutoff: float) -> float:
    total_mask = k > 0.0
    high_mask = k >= cutoff

    total = np.trapz(power[total_mask], k[total_mask])
    high = np.trapz(power[high_mask], k[high_mask])

    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return float(high / total)


def dominant_wavenumber(k: np.ndarray, power: np.ndarray) -> float:
    mask = k > 0.0
    if not np.any(mask):
        return float("nan")
    kk = k[mask]
    pp = power[mask]
    return float(kk[np.argmax(pp)])


def compute_spectra(
    selected: List[Tuple[Path, float]],
    dx: float,
    bins: int,
    remove_mean: bool,
    window_mode: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], List[float]]:
    accum: Dict[str, List[np.ndarray]] = {"phi": [], "psi": [], "v": []}
    representative: Dict[str, np.ndarray] = {}
    used_times: List[float] = []

    kr = None
    k_nyquist = None
    k_max = None

    for snap_index, (path, t) in enumerate(selected):
        _, fields = read_snapshot(path)

        if not fields:
            continue

        first_field = next(iter(fields.values()))
        if kr is None:
            kr, k_nyquist = radial_wavenumber_grid(first_field.shape, dx)
            k_max = float(k_nyquist)

        spatial_window = (
            hann2d(first_field.shape) if window_mode == "hann" else np.ones(first_field.shape)
        )

        if snap_index == len(selected) // 2:
            representative = {name: arr.copy() for name, arr in fields.items()}

        for name in accum:
            if name not in fields:
                continue

            field = fields[name].copy()
            if remove_mean:
                field -= np.mean(field)

            field *= spatial_window
            fft_field = np.fft.fft2(field)
            power_2d = np.abs(fft_field) ** 2

            k, radial = radial_average(power_2d, kr, bins=bins, k_max=k_max)
            accum[name].append(radial)

        used_times.append(t)

    if kr is None or k_nyquist is None:
        raise RuntimeError("Could not construct wavenumber grid from snapshots.")

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for name, arrays in accum.items():
        if not arrays:
            continue

        stack = np.vstack(arrays)
        mean_power = np.mean(stack, axis=0)
        std_power = np.std(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean_power)

        result[name] = {
            "k": k,
            "mean": mean_power,
            "std": std_power,
            "norm": normalise_nonzero(k, mean_power),
            "norm_std": std_power / max(
                np.trapz(mean_power[k > 0], k[k > 0]),
                np.finfo(float).eps,
            ),
        }

    return result, representative, used_times


def save_csv(
    out_dir: Path,
    spectra: Dict[str, Dict[str, np.ndarray]],
    k_nyquist: float,
    used_times: List[float],
) -> None:
    fields = list(spectra.keys())
    k = spectra[fields[0]]["k"]

    columns = {
        "k": k,
        "k_over_k_nyquist": k / k_nyquist,
    }

    for name in fields:
        columns[f"{name}_power_mean"] = spectra[name]["mean"]
        columns[f"{name}_power_std"] = spectra[name]["std"]
        columns[f"{name}_power_normalised"] = spectra[name]["norm"]
        columns[f"{name}_power_normalised_std"] = spectra[name]["norm_std"]

    out_path = out_dir / "azimuthal_power_spectra.csv"
    header = list(columns.keys())
    data = np.column_stack([columns[key] for key in header])

    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.12e",
    )

    summary = {
        "n_snapshots": len(used_times),
        "t_min_used": float(np.min(used_times)),
        "t_max_used": float(np.max(used_times)),
        "k_nyquist": float(k_nyquist),
        "k_nyquist_over_2": float(k_nyquist / 2.0),
        "fields": {},
    }

    for name, item in spectra.items():
        summary["fields"][name] = {
            "dominant_k": dominant_wavenumber(item["k"], item["norm"]),
            "high_k_fraction_k_ge_knyquist_over_2": high_k_fraction(
                item["k"],
                item["norm"],
                k_nyquist / 2.0,
            ),
        }

    with (out_dir / "spectral_resolution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def plot_spectra(
    out_dir: Path,
    spectra: Dict[str, Dict[str, np.ndarray]],
    k_nyquist: float,
    used_times: List[float],
    show_2d: bool,
    representative: Dict[str, np.ndarray],
    dx: float,
    remove_mean: bool,
    window_mode: str,
) -> None:
    colours = {
        "phi": "#1f4e79",
        "psi": "#8b3f7a",
        "v": "#b05a1a",
    }
    labels = {
        "phi": r"$\phi$",
        "psi": r"$\psi$",
        "v": r"$v=\partial_t\psi$",
    }

    if show_2d and "phi" in representative:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.0, 8.4),
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        ax_field, ax_2d, ax_radial, ax_tail = axes.ravel()

        phi = representative["phi"].copy()
        if remove_mean:
            phi -= np.mean(phi)

        spatial_window = hann2d(phi.shape) if window_mode == "hann" else np.ones(phi.shape)
        phi_fft = np.fft.fftshift(np.fft.fft2(phi * spatial_window))
        phi_power = np.abs(phi_fft) ** 2

        nx, ny = phi.shape
        kx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
        ky = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=dx))

        im0 = ax_field.imshow(
            phi.T,
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )
        ax_field.set_title(r"Representative late-time $\phi(x,y)$")
        ax_field.set_xlabel(r"$x$ grid index")
        ax_field.set_ylabel(r"$y$ grid index")
        fig.colorbar(im0, ax=ax_field, fraction=0.046, pad=0.04)

        im1 = ax_2d.imshow(
            np.log10(phi_power.T + np.finfo(float).eps),
            origin="lower",
            extent=[kx.min(), kx.max(), ky.min(), ky.max()],
            cmap="magma",
            aspect="equal",
        )
        ax_2d.set_title(r"$\log_{10}|\widehat{\phi}(k_x,k_y)|^2$")
        ax_2d.set_xlabel(r"$k_x$")
        ax_2d.set_ylabel(r"$k_y$")
        fig.colorbar(im1, ax=ax_2d, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ax_radial, ax_tail = axes

    for name, item in spectra.items():
        k = item["k"]
        p = item["norm"]
        s = item["norm_std"]

        ax_radial.plot(
            k,
            p,
            color=colours.get(name, "black"),
            lw=1.6,
            label=labels.get(name, name),
        )

        if np.any(s > 0):
            ax_radial.fill_between(
                k,
                np.maximum(p - s, 0.0),
                p + s,
                color=colours.get(name, "black"),
                alpha=0.14,
                linewidth=0,
            )

    ax_radial.axvline(
        k_nyquist / 2.0,
        color="0.35",
        ls="--",
        lw=1.0,
        label=r"$k_{\rm Ny}/2$",
    )
    ax_radial.axvline(
        k_nyquist,
        color="0.15",
        ls=":",
        lw=1.2,
        label=r"$k_{\rm Ny}$",
    )
    ax_radial.set_xlabel(r"radial wavenumber $k$")
    ax_radial.set_ylabel(r"normalised azimuthal power")
    ax_radial.set_title("Late-time radial power spectrum")
    ax_radial.set_xlim(0.0, k_nyquist)
    ax_radial.legend(frameon=False, fontsize=9)
    ax_radial.grid(alpha=0.22)

    for name, item in spectra.items():
        k = item["k"]
        p = item["norm"]
        mask = k > 0.0

        ax_tail.semilogy(
            k[mask],
            np.maximum(p[mask], 1e-18),
            color=colours.get(name, "black"),
            lw=1.5,
            label=labels.get(name, name),
        )

    ax_tail.axvline(
        k_nyquist / 2.0,
        color="0.35",
        ls="--",
        lw=1.0,
        label=r"$k_{\rm Ny}/2$",
    )
    ax_tail.axvline(
        k_nyquist,
        color="0.15",
        ls=":",
        lw=1.2,
        label=r"$k_{\rm Ny}$",
    )
    ax_tail.set_xlabel(r"radial wavenumber $k$")
    ax_tail.set_ylabel(r"normalised power (log scale)")
    ax_tail.set_title("High-wavenumber spectral tail")
    ax_tail.set_xlim(0.0, k_nyquist)
    ax_tail.legend(frameon=False, fontsize=9)
    ax_tail.grid(alpha=0.22, which="both")

    t_label = (
        rf"$t\in[{np.min(used_times):.2f},\,{np.max(used_times):.2f}]$, "
        rf"$N_{{\rm snap}}={len(used_times)}$, "
        rf"$k_{{\rm Ny}}={k_nyquist:.3f}$"
    )
    fig.suptitle("Spatial spectral-resolution diagnostic: " + t_label, y=0.995)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"spatial_spectral_resolution.{ext}",
            dpi=220,
            bbox_inches="tight",
        )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find config.json in {run_dir}")

    config = load_json(config_path)
    dx = float(config.get("dx", 1.0))

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run_dir / "processed_spatial_spectrum"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = snapshot_paths(run_dir)
    print(f"run_dir: {run_dir}")
    print(f"candidate snapshot files found: {len(paths)}")

    for path in paths[:20]:
        try:
            t, fields = read_snapshot(path)
            print(
                f"file={path.name} | "
                f"parsed_t={t!r} | "
                f"keys={list(fields.keys())}"
            )
        except Exception as exc:
            print(f"file={path.name} | READ ERROR: {exc!r}")
    selected = select_snapshots(
        paths=paths,
        t_min=args.t_min,
        t_max=args.t_max,
        max_snapshots=args.max_snapshots,
    )

    spectra, representative, used_times = compute_spectra(
        selected=selected,
        dx=dx,
        bins=args.bins,
        remove_mean=args.remove_mean,
        window_mode=args.window,
    )

    if not spectra:
        raise RuntimeError("No phi/psi/v fields were found in selected snapshots.")

    _, k_nyquist = radial_wavenumber_grid(
        next(iter(representative.values())).shape,
        dx=dx,
    )

    save_csv(
        out_dir=out_dir,
        spectra=spectra,
        k_nyquist=k_nyquist,
        used_times=used_times,
    )

    plot_spectra(
        out_dir=out_dir,
        spectra=spectra,
        k_nyquist=k_nyquist,
        used_times=used_times,
        show_2d=args.show_2d,
        representative=representative,
        dx=dx,
        remove_mean=args.remove_mean,
        window_mode=args.window,
    )

    summary_path = out_dir / "spectral_resolution_summary.json"
    print(f"Selected snapshots: {len(used_times)}")
    print(f"Used time interval: [{min(used_times):.6f}, {max(used_times):.6f}]")
    print(f"Nyquist wavenumber: {k_nyquist:.8f}")
    print(f"Figure written to: {out_dir / 'spatial_spectral_resolution.pdf'}")
    print(f"CSV written to: {out_dir / 'azimuthal_power_spectra.csv'}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()