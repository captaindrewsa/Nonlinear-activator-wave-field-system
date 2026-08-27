#!/usr/bin/env python3
"""
analysis_validation.py
======================

Validation-oriented analysis of sim_framework 2D NPZ snapshots.

Input layout (compatible with npz_to_csv_gif.py)
-------------------------------------------------
run_dir/
  config.json
  snapshots_2d/
    snapshot2d_<run_id>_t0000000.npz
    ...

Each NPZ must contain phi and t; psi is optional. The script accepts either
one run (--run-id) or a collection of runs (--out-dir). For each run it writes
only below

  run_dir/processed/<analysis_name>/

and therefore does not overwrite the media products created by
npz_to_csv_gif.py.

Products per run
----------------
summary_timeseries.csv   per-snapshot scalar observables
summary.json              run-level metrics and classification
figures/*.png and *.pdf   time series, FFT, final radial profiles, optional
                           threshold-sweep and two-spot diagnostics
tail_fit.json              transparent zero-crossing/envelope tail estimate
threshold_sweep.csv       optional classification sensitivity results
two_spot_diagnostics.csv  optional component/separation/COM trajectory

Examples
--------
python analysis_validation.py --run-id path/to/run --analysis-name validation
python analysis_validation.py --out-dir path/to/collection --analysis-name mesh_dt
python analysis_validation.py --run-id path/to/run --analysis-name threshold --threshold-sweep
python analysis_validation.py --run-id path/to/run --analysis-name pair --two-spot

Dependencies
------------
numpy, matplotlib, and the local run_discovery.py module used by
npz_to_csv_gif.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_discovery import norm_path, processed_dir, resolve_targets, run_id_of, snapshot_dir_of

CORE_THRESHOLD = 1.5
MIN_COMPONENT_AREA = 20
EPS = 1e-12
DEFAULT_COLLAPSE_OFFSET = 0.30
DEFAULT_AMPLITUDE_THRESHOLD = 0.15
DEFAULT_FFT_PROMINENCE = 5.0
_RE_FRAMEWORK = re.compile(r"^snapshot2d_(.+?)_t(\d+)(_final)?\.npz$")


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def safe_slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-._")
    return value or "validation"


def analysis_dir_of(run_dir: str, analysis_name: str) -> str:
    return ensure_dir(processed_dir(run_dir, safe_slug(analysis_name)))


def parse_snapshot_name(path: str) -> Tuple[Optional[str], Optional[int], bool]:
    m = _RE_FRAMEWORK.match(os.path.basename(path))
    if not m:
        return None, None, False
    return m.group(1), int(m.group(2)), bool(m.group(3))


def collect_snapshots(run_dir: str) -> List[str]:
    snap_dir = snapshot_dir_of(run_dir)
    paths = sorted(glob.glob(os.path.join(snap_dir, "snapshot2d_*.npz")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(run_dir, "snapshot2d_*.npz")))
    return paths


def load_snapshot(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    with np.load(path) as z:
        phi = np.asarray(z["phi"], dtype=np.float64)
        psi = np.asarray(z["psi"], dtype=np.float64) if "psi" in z else np.zeros_like(phi)
        v = np.asarray(z["v"], dtype=np.float64) if "v" in z else None
        t_raw = z["t"]
        t = float(np.ravel(t_raw)[0])
    return phi, psi, v, t


def load_config(run_dir: str) -> Dict[str, Any]:
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_track(run_dir: str) -> Dict[str, np.ndarray]:
    path = Path(run_dir) / "track.csv"
    if not path.is_file():
        return {}

    columns: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, [])
                try:
                    columns[key].append(float(value))
                except (TypeError, ValueError):
                    columns[key].append(float("nan"))

    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def load_result(run_dir: str) -> Dict[str, Any]:
    path = Path(run_dir) / "result.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def config_value(cfg: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in cfg:
            return cfg[name]
    return default


def result_value(res: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in res:
            return res[name]
    return default


def resolve_dx(cfg: Dict[str, Any], res: Dict[str, Any], args: argparse.Namespace) -> float:
    raw = config_value(cfg, "dx", default=result_value(res, "dx", default=args.dx))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(args.dx)


def resolve_dt(cfg: Dict[str, Any], res: Dict[str, Any]) -> Optional[float]:
    raw = config_value(cfg, "dt", default=result_value(res, "dt", default=None))
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def periodic_delta(a: float, b: float, n: int) -> float:
    d = a - b
    return d - n * round(d / n)


def periodic_distance(c1: Dict[str, float], c2: Dict[str, float], shape: Tuple[int, int], dx: float) -> float:
    ddx = periodic_delta(c1["cx"], c2["cx"], shape[0])
    ddy = periodic_delta(c1["cy"], c2["cy"], shape[1])
    return float(dx * math.hypot(ddx, ddy))


def connected_components_4(mask: np.ndarray) -> List[np.ndarray]:
    """Non-periodic 4-connected components of a threshold mask.

    Core structures normally remain away from the domain boundaries in the
    validation runs. Boundary-crossing spots are still represented in the
    scalar global diagnostics; for exact boundary-spanning component tracking,
    extend this routine to periodic connectivity.
    """
    nx, ny = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            points: List[Tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                points.append((x, y))
                for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= xx < nx and 0 <= yy < ny and mask[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        stack.append((xx, yy))
            components.append(np.asarray(points, dtype=int))
    return components


def detect_components(phi: np.ndarray, threshold: float, min_area: int) -> List[Dict[str, float]]:
    mask = phi > threshold
    raw = [comp for comp in connected_components_4(mask) if len(comp) >= min_area]
    result: List[Dict[str, float]] = []
    for comp in raw:
        xs, ys = comp[:, 0], comp[:, 1]
        values = phi[xs, ys]
        weights = np.maximum(values - threshold, EPS)
        cx = float(np.sum(xs * weights) / np.sum(weights))
        cy = float(np.sum(ys * weights) / np.sum(weights))
        area = int(len(comp))
        result.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "r_eff": float(math.sqrt(area / math.pi)),
            "phi_mean": float(np.mean(values)),
            "phi_max": float(np.max(values)),
        })
    return sorted(result, key=lambda q: q["area"], reverse=True)


def weighted_global_centroid(phi: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    weights = np.maximum(phi - threshold, 0.0)
    mass = float(np.sum(weights))
    if mass <= EPS:
        return float("nan"), float("nan"), mass
    xs, ys = np.indices(phi.shape)
    return float(np.sum(xs * weights) / mass), float(np.sum(ys * weights) / mass), mass


def fft_metrics(t: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    if len(t) < 8:
        return {"frequency": float("nan"), "period": float("nan"), "p_max": float("nan"), "p_median": float("nan"), "r_fft": float("nan")}
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return {"frequency": float("nan"), "period": float("nan"), "p_max": float("nan"), "p_median": float("nan"), "r_fft": float("nan")}
    y = (x - np.mean(x)) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(y)) ** 2
    freq = np.fft.rfftfreq(len(y), d=dt)
    if len(power) <= 1:
        return {"frequency": float("nan"), "period": float("nan"), "p_max": float("nan"), "p_median": float("nan"), "r_fft": float("nan")}
    p = power[1:]
    f = freq[1:]
    peak_idx = int(np.argmax(p))
    pmax = float(p[peak_idx])
    background = np.delete(p, peak_idx)
    pmed = float(np.median(background)) if len(background) else EPS
    fdom = float(f[peak_idx])
    return {
        "frequency": fdom,
        "period": float(1.0 / fdom) if fdom > 0 else float("nan"),
        "p_max": pmax,
        "p_median": pmed,
        "r_fft": float(pmax / max(pmed, EPS)),
    }


def radial_profile(field: np.ndarray, cx: float, cy: float, dx: float, bins: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = np.indices(field.shape)
    r = np.hypot(xs - cx, ys - cy) * dx
    edges = np.linspace(0.0, float(np.max(r)), bins + 1)
    which = np.clip(np.digitize(r.ravel(), edges) - 1, 0, bins - 1)
    count = np.bincount(which, minlength=bins)
    sums = np.bincount(which, weights=field.ravel(), minlength=bins)
    profile = sums / np.maximum(count, 1)
    return 0.5 * (edges[:-1] + edges[1:]), profile


def local_maxima(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.array([], dtype=int)
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1


def fit_tail(r: np.ndarray, psi_r: np.ndarray, r_min: float, r_max: float) -> Dict[str, Any]:
    """A reproducible, deliberately conservative tail estimate.

    mu is inferred from zero-crossing spacings (mu = pi / spacing), and
    gamma is the least-squares slope of log of positive local peak magnitudes.
    The result is an empirical profile descriptor, not a spatial-eigenvalue
    calculation of the breathing state.
    """
    sel = (r >= r_min) & (r <= r_max) & np.isfinite(psi_r)
    rr, yy = r[sel], psi_r[sel]
    base = {"r_min": float(r_min), "r_max": float(r_max), "gamma": float("nan"), "mu": float("nan"), "zero_crossings": 0, "envelope_peaks": 0}
    if len(rr) < 10:
        return base
    signs = np.sign(yy)
    signs[signs == 0] = 1
    crosses = np.where(signs[1:] != signs[:-1])[0]
    base["zero_crossings"] = int(len(crosses))
    if len(crosses) >= 2:
        locations = 0.5 * (rr[crosses] + rr[crosses + 1])
        spacing = np.diff(locations)
        spacing = spacing[spacing > EPS]
        if len(spacing):
            base["mu"] = float(math.pi / np.median(spacing))
    envelope = np.abs(yy)
    peaks = local_maxima(envelope)
    peaks = peaks[envelope[peaks] > EPS]
    base["envelope_peaks"] = int(len(peaks))
    if len(peaks) >= 2:
        coeff = np.polyfit(rr[peaks], np.log(envelope[peaks]), 1)
        base["gamma"] = float(-coeff[0])
    return base


def classify(phi_max: float, phi_lo: Optional[float], a_pp: float, r_fft: float, collapse_offset: float, amplitude_threshold: float, fft_threshold: float) -> str:
    if phi_lo is not None and np.isfinite(phi_max) and phi_max <= phi_lo + collapse_offset:
        return "collapsed"
    if np.isfinite(a_pp) and np.isfinite(r_fft) and a_pp > amplitude_threshold and r_fft >= fft_threshold:
        return "breathing"
    return "weakly_modulated_or_unresolved"


def save_figure_both(fig: plt.Figure, figure_dir: str, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, f"{stem}.png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(figure_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(t: np.ndarray, phi_center: np.ndarray, phi_lo: Optional[float], phi_hi: Optional[float], figure_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, phi_center, lw=1.25, color="#175a9f")
    if phi_lo is not None:
        ax.axhline(phi_lo, color="0.4", ls="--", lw=0.8, label=r"$\phi_{\rm lo}$")
    if phi_hi is not None:
        ax.axhline(phi_hi, color="0.4", ls=":", lw=0.8, label=r"$\phi_{\rm hi}$")
    ax.set(xlabel="t", ylabel=r"$\phi_c(t)$", title="Central activator signal")
    if phi_lo is not None or phi_hi is not None:
        ax.legend(frameon=False)
    save_figure_both(fig, figure_dir, "timeseries_phi_center")


def plot_fft(t: np.ndarray, signal: np.ndarray, figure_dir: str) -> None:
    if len(signal) < 2:
        return
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return
    spectrum = np.abs(np.fft.rfft((signal - np.mean(signal)) * np.hanning(len(signal)))) ** 2
    freq = np.fft.rfftfreq(len(signal), dt)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(freq[1:], spectrum[1:], lw=1.1, color="#a12c2c")
    ax.set(xlabel="frequency", ylabel="Hann-windowed power", title="Central-signal spectrum")
    save_figure_both(fig, figure_dir, "fft_phi_center")


def plot_radial(r: np.ndarray, value: np.ndarray, ylabel: str, title: str, stem: str, figure_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(r, value, lw=1.15)
    ax.axhline(0.0, color="0.5", lw=0.75, ls=":")
    ax.set(xlabel="r (physical units)", ylabel=ylabel, title=title)
    save_figure_both(fig, figure_dir, stem)


def plot_threshold_sweep(rows: List[Dict[str, Any]], figure_dir: str) -> None:
    if not rows:
        return
    colours = {"collapsed": "#377eb8", "breathing": "#e41a1c", "weakly_modulated_or_unresolved": "#ff7f00"}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for label in sorted({str(r["label"]) for r in rows}):
        subset = [r for r in rows if r["label"] == label]
        ax.scatter([r["amplitude_threshold"] for r in subset], [r["fft_threshold"] for r in subset], s=35, label=label, color=colours.get(label, "0.5"), alpha=0.75)
    ax.set(xlabel="amplitude threshold", ylabel="FFT-prominence threshold", title="Classification sensitivity")
    ax.legend(frameon=False, fontsize=8)
    save_figure_both(fig, figure_dir, "threshold_sweep")


def plot_pair(t: np.ndarray, separation: np.ndarray, displacement: np.ndarray, figure_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    ax1.plot(t, separation, lw=1.2, color="#6a3d9a")
    ax1.set(ylabel="separation")
    ax2.plot(t, displacement, lw=1.2, color="#1b9e77")
    ax2.set(xlabel="t", ylabel="COM displacement")
    fig.suptitle("Two-spot diagnostics")
    save_figure_both(fig, figure_dir, "twospot_diagnostics")


def threshold_rows(t: np.ndarray, phi_c: np.ndarray, phi_lo: Optional[float], args: argparse.Namespace, base_window: Tuple[float, float]) -> List[Dict[str, Any]]:
    offsets = args.sweep_collapse_offsets or [0.20, 0.30, 0.40]
    amplitudes = args.sweep_amplitude_thresholds or [0.10, 0.15, 0.20]
    prominences = args.sweep_fft_prominences or [3.0, 5.0, 8.0]
    windows = [base_window] + (args.extra_windows or [])
    rows: List[Dict[str, Any]] = []
    for t0, t1 in windows:
        m = (t >= t0) & (t <= t1)
        if int(np.sum(m)) < 8:
            continue
        tw, yw = t[m], phi_c[m]
        a_pp = float(np.max(yw) - np.min(yw))
        fm = fft_metrics(tw, yw)
        pmax = float(np.max(yw))
        for offset in offsets:
            for amplitude in amplitudes:
                for prominence in prominences:
                    rows.append({
                        "window_t0": t0, "window_t1": t1,
                        "collapse_offset": offset,
                        "amplitude_threshold": amplitude,
                        "fft_threshold": prominence,
                        "a_pp": a_pp,
                        "r_fft": fm["r_fft"],
                        "period": fm["period"],
                        "label": classify(pmax, phi_lo, a_pp, fm["r_fft"], offset, amplitude, prominence),
                    })
    return rows


def process_run(run_dir: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    run_dir = norm_path(run_dir)
    cfg = load_config(run_dir)
    res = load_result(run_dir)
    track = load_track(run_dir)
    run_id = run_id_of(run_dir)
    dx = resolve_dx(cfg, res, args)
    dt_cfg = resolve_dt(cfg, res)

    status = str(result_value(res, "status", default="unknown"))
    valid_for_analysis = (status == "completed")

    files = collect_snapshots(run_dir)
    if not files:
        print(f"[skip] no snapshot2d*.npz in {run_dir}")
        return None

    out_dir = analysis_dir_of(run_dir, args.analysis_name)
    figure_dir = ensure_dir(Path(out_dir) / "figures")

    nx_cfg = config_value(cfg, "nx", default=None)
    ny_cfg = config_value(cfg, "ny", default=None)

    try:
        grid_nx = int(nx_cfg) if nx_cfg is not None else None
    except (TypeError, ValueError):
        grid_nx = None

    try:
        grid_ny = int(ny_cfg) if ny_cfg is not None else None
    except (TypeError, ValueError):
        grid_ny = None

    grid = {
        "nx": grid_nx,
        "ny": grid_ny,
        "dx": dx,
        "dt": dt_cfg,
        "Lx": float(grid_nx * dx) if grid_nx is not None else None,
        "Ly": float(grid_ny * dx) if grid_ny is not None else None,
    }

    if not valid_for_analysis:
        summary = {
            "run_id": run_id,
            "run_dir": run_dir,
            "analysis_name": args.analysis_name,
            "valid_for_analysis": False,
            "solver_status": status,
            "failure_reason": result_value(res, "failure_reason", default=None),
            "failure_step": result_value(res, "failure_step", default=None),
            "failure_time": result_value(res, "failure_time", default=None),
            "solver_max_abs_phi": result_value(res, "max_abs_phi", default=None),
            "solver_max_abs_psi": result_value(res, "max_abs_psi", default=None),
            "solver_max_abs_v": result_value(res, "max_abs_v", default=None),
            "n_snapshots": 0,
            "t_min": None,
            "t_max": result_value(res, "t_total_reached", default=None),
            "time_series_source": None,
            "n_track_points": 0,
            "grid": grid,
            "observation_window": {
                "t0": None,
                "t1": None,
            },
            "a_pp": None,
            "phi_max_window": None,
            "frequency": None,
            "period": None,
            "r_fft": None,
            "regime_label": "numerically_unstable",
            "classification_parameters": {
                "collapse_offset": args.collapse_offset,
                "amplitude_threshold": args.amplitude_threshold,
                "fft_prominence": args.fft_prominence,
            },
            "tail_fit": {},
            "threshold_sweep_rows": 0,
            "two_spot_rows": 0,
        }

        with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[skip-unstable] {run_id} status={status} reason={summary['failure_reason']}")
        return summary

    records: List[Dict[str, Any]] = []
    final_phi: Optional[np.ndarray] = None
    final_psi: Optional[np.ndarray] = None

    for path in files:
        phi, psi, v, t_val = load_snapshot(path)
        components = detect_components(phi, args.core_threshold, args.min_component_area)
        gx, gy, mass = weighted_global_centroid(phi, args.core_threshold)

        nx, ny = phi.shape
        center = float(phi[nx // 2, ny // 2])

        c0 = components[0] if len(components) >= 1 else {}
        c1 = components[1] if len(components) >= 2 else {}

        separation = periodic_distance(c0, c1, phi.shape, dx) if c0 and c1 else float("nan")

        records.append({
            "t": float(t_val),
            "phi_center": center,
            "phi_max": float(np.max(phi)),
            "phi_min": float(np.min(phi)),
            "psi_max_abs": float(np.max(np.abs(psi))),
            "n_components": len(components),
            "global_cx": gx,
            "global_cy": gy,
            "active_mass": mass,
            "c0_x": c0.get("cx", float("nan")),
            "c0_y": c0.get("cy", float("nan")),
            "c0_area": c0.get("area", float("nan")),
            "c0_r_eff": c0.get("r_eff", float("nan")),
            "c1_x": c1.get("cx", float("nan")),
            "c1_y": c1.get("cy", float("nan")),
            "c1_area": c1.get("area", float("nan")),
            "c1_r_eff": c1.get("r_eff", float("nan")),
            "separation": separation,
        })

        final_phi, final_psi = phi, psi

    if not records:
        print(f"[skip] no readable snapshots in {run_dir}")
        return None

    records.sort(key=lambda row: row["t"])
    t = np.asarray([r["t"] for r in records], dtype=float)
    phi_c = np.asarray([r["phi_center"] for r in records], dtype=float)

    track_t = track.get("t", np.array([], dtype=float))
    track_phi_c = track.get("phi_center", np.array([], dtype=float))
    valid_track = np.isfinite(track_t) & np.isfinite(track_phi_c)
    track_t = track_t[valid_track]
    track_phi_c = track_phi_c[valid_track]

    use_track = False
    if len(track_t) >= 8:
        if len(t) > 0:
            t_end = float(t[-1])
            span = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
            tol = max(5.0 * (dt_cfg or 0.0), 0.05 * span, 1e-9)
            use_track = np.isfinite(track_t[-1]) and (track_t[-1] >= t_end - tol)
        else:
            use_track = True

    time_t = track_t if use_track else t
    time_phi_c = track_phi_c if use_track else phi_c
    time_series_source = "track.csv" if use_track else "snapshots"

    with open(Path(out_dir) / "summary_timeseries.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    phi_lo_raw = config_value(cfg, "phi_lo", default=config_value(cfg, "h_bg", "h", default=None))
    phi_hi_raw = config_value(cfg, "phi_hi", default=None)

    phi_lo = float(phi_lo_raw) if phi_lo_raw is not None else None
    phi_hi = float(phi_hi_raw) if phi_hi_raw is not None else None

    t0 = args.window_t0 if args.window_t0 is not None else float(time_t[0] + 0.6 * (time_t[-1] - time_t[0]))
    t1 = args.window_t1 if args.window_t1 is not None else float(time_t[-1])

    in_window = (time_t >= t0) & (time_t <= t1)
    tw, yw = time_t[in_window], time_phi_c[in_window]

    amp = float(np.max(yw) - np.min(yw)) if len(yw) else float("nan")
    fft = fft_metrics(tw, yw)
    regime = classify(float(np.max(yw)) if len(yw) else float("nan"), phi_lo, amp, fft["r_fft"], args.collapse_offset, args.amplitude_threshold, args.fft_prominence)

    plot_timeseries(time_t, time_phi_c, phi_lo, phi_hi, figure_dir)
    if len(tw) >= 8:
        plot_fft(tw, yw, figure_dir)

    tail_result: Dict[str, Any] = {}
    if final_phi is not None and final_psi is not None:
        final_components = detect_components(final_phi, args.core_threshold, args.min_component_area)
        if final_components:
            core = final_components[0]

            r_phi, p_phi = radial_profile(final_phi, core["cx"], core["cy"], dx)
            r_psi, p_psi = radial_profile(final_psi, core["cx"], core["cy"], dx)

            plot_radial(r_phi, p_phi, r"$\phi(r)$", "Final radial activator profile", "radial_profile_phi", figure_dir)
            plot_radial(r_psi, p_psi, r"$\psi(r)$", "Final radial inhibitor profile", "radial_profile_psi", figure_dir)

            r_max = float(np.max(r_psi))
            tail_result = fit_tail(
                r_psi,
                p_psi,
                args.tail_r_min if args.tail_r_min is not None else 0.25 * r_max,
                args.tail_r_max if args.tail_r_max is not None else 0.90 * r_max,
            )
            tail_result["profile_time"] = float(t[-1])

            with open(Path(out_dir) / "tail_fit.json", "w", encoding="utf-8") as f:
                json.dump(tail_result, f, indent=2, ensure_ascii=False)

    sweep: List[Dict[str, Any]] = []
    if args.threshold_sweep:
        sweep = threshold_rows(time_t, time_phi_c, phi_lo, args, (t0, t1))
        if sweep:
            with open(Path(out_dir) / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
                writer.writeheader()
                writer.writerows(sweep)
            plot_threshold_sweep(sweep, figure_dir)

    two_spot_rows: List[Dict[str, Any]] = []
    if args.two_spot and records:
        start_x = records[0]["global_cx"]
        start_y = records[0]["global_cy"]
        nx = grid_nx if grid_nx is not None else (final_phi.shape[0] if final_phi is not None else 1)
        ny = grid_ny if grid_ny is not None else (final_phi.shape[1] if final_phi is not None else 1)

        for row in records:
            if all(np.isfinite([row["global_cx"], row["global_cy"], start_x, start_y])):
                displacement = dx * math.hypot(
                    periodic_delta(row["global_cx"], start_x, nx),
                    periodic_delta(row["global_cy"], start_y, ny),
                )
            else:
                displacement = float("nan")

            two_spot_rows.append({
                "t": row["t"],
                "n_components": row["n_components"],
                "separation": row["separation"],
                "global_cx": row["global_cx"],
                "global_cy": row["global_cy"],
                "com_displacement": displacement,
            })

        with open(Path(out_dir) / "two_spot_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(two_spot_rows[0].keys()))
            writer.writeheader()
            writer.writerows(two_spot_rows)

        plot_pair(
            t,
            np.asarray([q["separation"] for q in two_spot_rows], dtype=float),
            np.asarray([q["com_displacement"] for q in two_spot_rows], dtype=float),
            figure_dir,
        )

    summary = {
        "run_id": run_id,
        "run_dir": run_dir,
        "analysis_name": args.analysis_name,
        "valid_for_analysis": True,
        "solver_status": status,
        "failure_reason": result_value(res, "failure_reason", default=None),
        "failure_step": result_value(res, "failure_step", default=None),
        "failure_time": result_value(res, "failure_time", default=None),
        "solver_max_abs_phi": result_value(res, "max_abs_phi", default=None),
        "solver_max_abs_psi": result_value(res, "max_abs_psi", default=None),
        "solver_max_abs_v": result_value(res, "max_abs_v", default=None),
        "n_snapshots": len(records),
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
        "time_series_source": time_series_source,
        "n_track_points": int(len(track_t)),
        "grid": grid,
        "observation_window": {
            "t0": t0,
            "t1": t1,
        },
        "a_pp": amp,
        "phi_max_window": float(np.max(yw)) if len(yw) else float("nan"),
        "frequency": fft["frequency"],
        "period": fft["period"],
        "r_fft": fft["r_fft"],
        "regime_label": regime,
        "classification_parameters": {
            "collapse_offset": args.collapse_offset,
            "amplitude_threshold": args.amplitude_threshold,
            "fft_prominence": args.fft_prominence,
        },
        "tail_fit": tail_result,
        "threshold_sweep_rows": len(sweep),
        "two_spot_rows": len(two_spot_rows),
    }

    with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[ok] {run_id} regime={regime} a_pp={amp:.6g} period={fft['period']:.6g} -> {out_dir}")
    return summary


def parse_windows(items: Optional[List[str]]) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for item in items or []:
        parts = item.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Each --extra-window must be t0,t1")
        a, b = float(parts[0]), float(parts[1])
        if b <= a:
            raise argparse.ArgumentTypeError("Each extra window must obey t1 > t0")
        result.append((a, b))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation analysis for sim_framework NPZ runs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="Path to a single run directory")
    source.add_argument("--out-dir", help="Path containing multiple run directories")
    parser.add_argument("--analysis-name", required=True, help="Results go to processed/<analysis_name>/")
    parser.add_argument("--dx", type=float, default=1.0, help="Physical grid spacing used for radial/separation coordinates")
    parser.add_argument("--core-threshold", type=float, default=CORE_THRESHOLD)
    parser.add_argument("--min-component-area", type=int, default=MIN_COMPONENT_AREA)
    parser.add_argument("--window-t0", type=float, default=None)
    parser.add_argument("--window-t1", type=float, default=None)
    parser.add_argument("--collapse-offset", type=float, default=DEFAULT_COLLAPSE_OFFSET)
    parser.add_argument("--amplitude-threshold", type=float, default=DEFAULT_AMPLITUDE_THRESHOLD)
    parser.add_argument("--fft-prominence", type=float, default=DEFAULT_FFT_PROMINENCE)
    parser.add_argument("--tail-r-min", type=float, default=None)
    parser.add_argument("--tail-r-max", type=float, default=None)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--sweep-collapse-offsets", type=float, nargs="+", default=None)
    parser.add_argument("--sweep-amplitude-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--sweep-fft-prominences", type=float, nargs="+", default=None)
    parser.add_argument("--extra-window", dest="extra_window_text", action="append", default=None, metavar="T0,T1")
    parser.add_argument("--two-spot", action="store_true")
    return parser


def write_collection_manifest(out_dir: str, analysis_name: str, summaries: List[Dict[str, Any]]) -> None:
    root = Path(out_dir)
    stem = f"processed_{safe_slug(analysis_name)}_manifest"
    keys = [
        "run_id",
        "run_dir",
        "valid_for_analysis",
        "solver_status",
        "failure_reason",
        "n_snapshots",
        "regime_label",
        "a_pp",
        "period",
        "frequency",
        "r_fft",
    ]
    with open(root / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in keys})
    with open(root / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"analysis_name": analysis_name, "runs": summaries}, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.analysis_name = safe_slug(args.analysis_name)
    args.extra_windows = parse_windows(args.extra_window_text)
    targets = resolve_targets(args.run_id, args.out_dir)
    print(f"Discovered run directories: {len(targets)}")
    summaries = [result for target in targets if (result := process_run(target, args)) is not None]
    if args.out_dir and summaries:
        write_collection_manifest(norm_path(args.out_dir), args.analysis_name, summaries)
    print(f"Done. Analysed {len(summaries)} run(s).")


if __name__ == "__main__":
    main()
