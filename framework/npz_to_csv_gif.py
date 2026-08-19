#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npz_to_csv_gif.py
=================

Framework-compatible post-processing utility for 2D snapshot stacks produced by
sim_framework-style runs.

Main features
-------------
- Process either one run directory (--run-id) or a whole collection (--out-dir).
- Export per-run CSV / JSON summaries into run_dir/processed/... .
- Render phi / psi frames and build a GIF.
- Restrict rendering to a time window (--t_start, --t_end) or selected exact
  times (--times).
- Support multiple visualization strategies without overwriting previous media
  outputs by writing into media variants.
- Support dynamic, fixed, global, and symmetric scaling modes.
- Support optional signal transforms (none, log1p, asinh) before plotting.

Expected run layout
-------------------
run_dir/
  config.json
  snapshots_2d/
    snapshot2d_<tag>_t0000000.npz
    snapshot2d_<tag>_t0000030.npz
    ...

Media output layout
-------------------
run_dir/processed/media/<variant_name>/
  frames_phi/
  frames_psi/
  <tag>.gif
  render_manifest.json

Tabular outputs remain in run_dir/processed/media/ by default for backward
compatibility, while render products are isolated per variant.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from run_discovery import norm_path, resolve_targets, processed_dir, run_id_of, snapshot_dir_of

CENTER_THRESHOLD = 1.5
MIN_MASS = 20
CORE_THRESHOLD = 1.5
EPS = 1e-8
_RE_FRAMEWORK = re.compile(r"^snapshot2d_(.+?)_t(\d+)(_final)?\.npz$")


def ensure_dir(d: str) -> str:
    d = norm_path(d)
    os.makedirs(d, exist_ok=True)
    return d


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def parse_tag_time(fname: str) -> Tuple[Optional[str], Optional[int], Optional[bool]]:
    base = os.path.basename(str(fname))
    m = _RE_FRAMEWORK.match(base)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), bool(m.group(3))


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    z = np.load(path)
    phi = z["phi"].astype(np.float32)
    psi = z["psi"].astype(np.float32) if "psi" in z else np.zeros_like(phi, dtype=np.float32)
    t_raw = z["t"]
    t = float(t_raw[0]) if np.ndim(t_raw) > 0 else float(t_raw)
    return phi, psi, t


def collect_snapshots_for_run(run_dir: str) -> List[str]:
    snap_dir = snapshot_dir_of(run_dir)
    files = sorted(glob.glob(os.path.join(snap_dir, "snapshot2d_*.npz")))
    if files:
        return files
    return sorted(glob.glob(os.path.join(run_dir, "snapshot2d_*.npz")))


def detect_center(phi: np.ndarray, thresh: float = CENTER_THRESHOLD, min_mass: int = MIN_MASS) -> Tuple[float, float, int]:
    mask = phi > thresh
    mass = int(mask.sum())
    if mass < min_mass:
        return float("nan"), float("nan"), mass
    coords = np.argwhere(mask)
    vals = phi[mask]
    w = vals - thresh
    if np.all(w <= 0):
        w = np.ones_like(vals)
    cx = float(np.sum(coords[:, 0] * w) / np.sum(w))
    cy = float(np.sum(coords[:, 1] * w) / np.sum(w))
    return cx, cy, mass


def connected_components_4(mask: np.ndarray) -> List[np.ndarray]:
    nx, ny = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            pts = []
            while stack:
                x, y = stack.pop()
                pts.append((x, y))
                for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= xx < nx and 0 <= yy < ny and mask[xx, yy] and not visited[xx, yy]:
                        visited[xx, yy] = True
                        stack.append((xx, yy))
            comps.append(np.array(pts, dtype=int))
    return comps


def detect_core_2d(phi: np.ndarray, thresh: float = CORE_THRESHOLD, min_mass: int = MIN_MASS) -> Dict[str, Any]:
    mask = phi > thresh
    if int(mask.sum()) < min_mass:
        ix, iy = np.unravel_index(np.argmax(phi), phi.shape)
        x0 = max(0, ix - 2)
        x1 = min(phi.shape[0] - 1, ix + 2)
        y0 = max(0, iy - 2)
        y1 = min(phi.shape[1] - 1, iy + 2)
        core_patch = phi[x0:x1 + 1, y0:y1 + 1]
        return {
            "source": "fallback-max", "cx": float(ix), "cy": float(iy),
            "area": int((x1 - x0 + 1) * (y1 - y0 + 1)),
            "r_eff": float(math.sqrt(max((x1 - x0 + 1) * (y1 - y0 + 1), 1) / math.pi)),
            "bbox_x0": int(x0), "bbox_x1": int(x1), "bbox_y0": int(y0), "bbox_y1": int(y1),
            "phi_mean_core": float(np.mean(core_patch)), "phi_max_core": float(np.max(core_patch)),
            "n_components": 0, "orientation_deg": None, "eccentricity": None,
        }

    comps = sorted(connected_components_4(mask), key=lambda c: len(c), reverse=True)
    comp = comps[0]
    xs = comp[:, 0]
    ys = comp[:, 1]
    vals = phi[xs, ys]
    w = vals - thresh
    if np.all(w <= 0):
        w = np.ones_like(vals)

    cx = float(np.sum(xs * w) / np.sum(w))
    cy = float(np.sum(ys * w) / np.sum(w))
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = int(len(comp))
    r_eff = float(math.sqrt(area / math.pi))

    xcw = xs - cx
    ycw = ys - cy
    cxx = float(np.sum(w * xcw * xcw) / np.sum(w))
    cyy = float(np.sum(w * ycw * ycw) / np.sum(w))
    cxy = float(np.sum(w * xcw * ycw) / np.sum(w))
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    lam1 = float(max(evals[0], 0.0))
    lam2 = float(max(evals[1], 0.0))
    orientation = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - lam2 / max(lam1, 1e-12)))) if lam1 > 0 else None

    return {
        "source": "threshold", "cx": cx, "cy": cy, "area": area, "r_eff": r_eff,
        "bbox_x0": x0, "bbox_x1": x1, "bbox_y0": y0, "bbox_y1": y1,
        "phi_mean_core": float(np.mean(phi[xs, ys])), "phi_max_core": float(np.max(phi[xs, ys])),
        "n_components": int(len(comps)), "orientation_deg": orientation, "eccentricity": eccentricity,
    }


def dist_nan(a: float, b: float, c: float, d: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and np.isfinite(d)):
        return float("nan")
    return float(np.hypot(a - c, b - d))


def compute_crop_window(frames: List[np.ndarray], margin: int = 12, thresh: float = CORE_THRESHOLD) -> Tuple[int, int, int, int]:
    masks = [f > thresh for f in frames]
    union = np.any(np.stack(masks, axis=0), axis=0)
    pts = np.argwhere(union)
    if pts.size == 0:
        nx, ny = frames[0].shape
        return 0, nx, 0, ny
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    nx, ny = frames[0].shape
    x0 = max(0, x0 - margin)
    x1 = min(nx, x1 + margin + 1)
    y0 = max(0, y0 - margin)
    y1 = min(ny, y1 + margin + 1)
    return int(x0), int(x1), int(y0), int(y1)


def compute_dynamic_range(arr: np.ndarray, mode: str = "meanstd", k_sigma: float = 2.5, q_low: float = 0.02, q_high: float = 0.98, center_on_mean: bool = True) -> Tuple[float, float]:
    finite = np.asarray(arr, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if mode == "minmax":
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    elif mode == "meanstd":
        mu = float(np.mean(finite))
        sigma = max(float(np.std(finite)), EPS)
        if center_on_mean:
            vmin = mu - k_sigma * sigma
            vmax = mu + k_sigma * sigma
        else:
            vmin = float(np.min(finite))
            vmax = mu + k_sigma * sigma
    elif mode == "quantile":
        vmin = float(np.quantile(finite, q_low))
        vmax = float(np.quantile(finite, q_high))
    else:
        raise ValueError(f"Unknown dynamic range mode: {mode}")
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        mu = float(np.mean(finite))
        sigma = max(float(np.std(finite)), EPS)
        vmin = mu - sigma
        vmax = mu + sigma
    return vmin, vmax


def parse_times_arg(times_str: Optional[str]) -> Optional[List[float]]:
    if not times_str:
        return None
    vals = []
    for chunk in times_str.split(","):
        chunk = chunk.strip()
        if chunk:
            vals.append(float(chunk))
    return vals if vals else None


def choose_frames_by_time(times: Sequence[float], t_start: Optional[float], t_end: Optional[float], exact_times: Optional[Sequence[float]], tol: float) -> List[int]:
    idx = list(range(len(times)))
    if t_start is not None:
        idx = [i for i in idx if times[i] >= t_start]
    if t_end is not None:
        idx = [i for i in idx if times[i] <= t_end]
    if exact_times:
        picked = []
        for target in exact_times:
            best_i = None
            best_dt = None
            for i in idx:
                dt = abs(times[i] - target)
                if best_dt is None or dt < best_dt:
                    best_i = i
                    best_dt = dt
            if best_i is not None and best_dt is not None and best_dt <= tol:
                if best_i not in picked:
                    picked.append(best_i)
        return picked
    return idx


def safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "variant"


def auto_variant_name(args: argparse.Namespace, field: str = "phi") -> str:
    parts = []
    parts.append(f"{field}-{args.phi_scale if field == 'phi' else args.psi_scale}")
    if args.t_start is not None or args.t_end is not None:
        parts.append(f"t{args.t_start if args.t_start is not None else 'min'}-{args.t_end if args.t_end is not None else 'max'}")
    if args.times:
        parts.append("times-" + "_".join(str(int(t)) if float(t).is_integer() else str(t) for t in args.times[:5]))
    transform = args.phi_transform if field == "phi" else args.psi_transform
    if transform != "none":
        parts.append(transform)
    if (field == "phi" and args.phi_scale == "fixed") or (field == "psi" and args.psi_scale == "fixed"):
        vmin = args.phi_vmin if field == "phi" else args.psi_vmin
        vmax = args.phi_vmax if field == "phi" else args.psi_vmax
        parts.append(f"{vmin}_{vmax}")
    parts.append(f"zoom-{args.zoom}")
    return safe_slug("_".join(map(str, parts)))


def apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if transform == "none":
        return arr
    if transform == "log1p":
        return np.sign(arr) * np.log1p(np.abs(arr))
    if transform == "asinh":
        return np.arcsinh(arr)
    raise ValueError(f"Unknown transform: {transform}")


def resolve_scale(arr_show: np.ndarray, *, scale_mode: str, fixed_vmin: Optional[float], fixed_vmax: Optional[float], precomputed_vmin: Optional[float], precomputed_vmax: Optional[float], dynamic_mode: str, k_sigma: float, q_low: float, q_high: float) -> Tuple[float, float]:
    if scale_mode == "fixed":
        if fixed_vmin is None or fixed_vmax is None:
            raise ValueError("fixed scale requires both vmin and vmax")
        return float(fixed_vmin), float(fixed_vmax)
    if scale_mode == "global":
        if precomputed_vmin is None or precomputed_vmax is None:
            raise ValueError("global scale requires precomputed vmin/vmax")
        return float(precomputed_vmin), float(precomputed_vmax)
    if scale_mode == "symmetric":
        local_vmin, local_vmax = compute_dynamic_range(arr_show, mode=dynamic_mode, k_sigma=k_sigma, q_low=q_low, q_high=q_high)
        vmax = max(abs(local_vmin), abs(local_vmax))
        return -float(vmax), float(vmax)
    return compute_dynamic_range(arr_show, mode=dynamic_mode, k_sigma=k_sigma, q_low=q_low, q_high=q_high)


def render_frame(arr: np.ndarray, t: float, out_path: str | Path, title_prefix: str, cmap: str, gif_size: int, crop: Optional[Tuple[int, int, int, int]] = None, show_axes: bool = True, scale_mode: str = "dynamic", fixed_vmin: Optional[float] = None, fixed_vmax: Optional[float] = None, precomputed_vmin: Optional[float] = None, precomputed_vmax: Optional[float] = None, dynamic_mode: str = "meanstd", k_sigma: float = 2.5, q_low: float = 0.02, q_high: float = 0.98, transform: str = "none") -> None:
    out_path = ensure_parent_dir(out_path)
    arr_plot = apply_transform(arr, transform)
    if crop is not None:
        x0, x1, y0, y1 = crop
        arr_show = arr_plot[x0:x1, y0:y1]
        extent = [y0, y1 - 1, x0, x1 - 1]
    else:
        arr_show = arr_plot
        nx, ny = arr.shape
        extent = [0, ny - 1, 0, nx - 1]

    vmin, vmax = resolve_scale(
        arr_show,
        scale_mode=scale_mode,
        fixed_vmin=fixed_vmin,
        fixed_vmax=fixed_vmax,
        precomputed_vmin=precomputed_vmin,
        precomputed_vmax=precomputed_vmax,
        dynamic_mode=dynamic_mode,
        k_sigma=k_sigma,
        q_low=q_low,
        q_high=q_high,
    )

    dpi = 150
    fig_inches = max(6.0, gif_size / dpi)
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    im = ax.imshow(arr_show, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", resample=False, aspect="equal", extent=extent)
    ax.set_title(f"{title_prefix} t={t:.1f}", fontsize=12)
    if show_axes:
        ax.set_xlabel("y")
        ax.set_ylabel("x")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    label = title_prefix if transform == "none" else f"{title_prefix} [{transform}]"
    cbar.ax.set_ylabel(label, rotation=90)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_gif_from_pngs(frame_paths: List[str], gif_path: str | Path, fps: int = 6) -> None:
    if not frame_paths:
        return
    gif_path = ensure_parent_dir(gif_path)
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    duration_ms = int(1000 / max(1, fps))
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False, disposal=2)


def save_midline_csv(path: str, rows: List[List[float]], ny: int) -> None:
    path = str(ensure_parent_dir(path))
    header = ["t"] + [f"y{i}" for i in range(ny)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_track_csv(path: str, rows_track: List[List[float]], max_phi_local: List[float]) -> None:
    path = str(ensure_parent_dir(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "mass", "phi_center", "max_phi_local"])
        for row, mp in zip(rows_track, max_phi_local):
            w.writerow(row + [mp])


def save_core_timeseries_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    path = str(ensure_parent_dir(path))
    fieldnames = ["t", "cx", "cy", "track_x", "track_y", "dist_core_track", "area", "r_eff", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "phi_mean_core", "phi_max_core", "phi_global_max", "psi_global_max", "n_components", "orientation_deg", "eccentricity", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def save_manifest_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    path = str(ensure_parent_dir(path))
    fieldnames = ["run_id", "run_dir", "n_frames", "midline_phi_csv", "midline_psi_csv", "track_csv", "core_2d_csv", "core_2d_summary_json", "gif", "frames_phi_dir", "frames_psi_dir", "field_2d_stack_npz"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def make_media_dirs(run_dir: str, save_psi_frames: bool, variant_name: str) -> Tuple[str, str, str, str]:
    media_root = processed_dir(run_dir, "media")
    variant_dir = ensure_dir(os.path.join(media_root, variant_name))
    frames_phi_dir = ensure_dir(os.path.join(variant_dir, "frames_phi"))
    frames_psi_dir = os.path.join(variant_dir, "frames_psi")
    if save_psi_frames:
        frames_psi_dir = ensure_dir(frames_psi_dir)
    return media_root, variant_dir, frames_phi_dir, frames_psi_dir


def process_run(run_dir: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    run_dir = norm_path(run_dir)
    tag = run_id_of(run_dir)
    files = collect_snapshots_for_run(run_dir)
    if not files:
        print(f"[!] No snapshot2d_*.npz files found in {run_dir}")
        return None

    rows_phi, rows_psi, rows_track, rows_core = [], [], [], []
    phi_frames, psi_frames, phi_max_series, times, t_ints = [], [], [], [], []
    stack_t, stack_phi, stack_psi = [], [], []

    for path in files:
        phi, psi, t = load_npz(path)
        _, t_int, _ = parse_tag_time(path)
        nx, ny = phi.shape
        cx_track, cy_track, mass = detect_center(phi, thresh=args.core_threshold, min_mass=MIN_MASS)
        phi_center = float(phi[nx // 2, ny // 2])
        rows_phi.append([t] + phi[nx // 2, :].astype(float).tolist())
        rows_psi.append([t] + psi[nx // 2, :].astype(float).tolist())
        rows_track.append([t, cx_track, cy_track, mass, phi_center])
        core = detect_core_2d(phi, thresh=args.core_threshold, min_mass=MIN_MASS)
        rows_core.append({
            "t": float(t), "cx": float(core["cx"]), "cy": float(core["cy"]),
            "track_x": float(cx_track) if np.isfinite(cx_track) else float("nan"),
            "track_y": float(cy_track) if np.isfinite(cy_track) else float("nan"),
            "dist_core_track": dist_nan(core["cx"], core["cy"], cx_track, cy_track),
            "area": int(core["area"]), "r_eff": float(core["r_eff"]),
            "bbox_x0": int(core["bbox_x0"]), "bbox_x1": int(core["bbox_x1"]),
            "bbox_y0": int(core["bbox_y0"]), "bbox_y1": int(core["bbox_y1"]),
            "phi_mean_core": float(core["phi_mean_core"]), "phi_max_core": float(core["phi_max_core"]),
            "phi_global_max": float(np.max(phi)), "psi_global_max": float(np.max(psi)),
            "n_components": int(core["n_components"]), "orientation_deg": core["orientation_deg"],
            "eccentricity": core["eccentricity"], "source": core["source"],
        })
        phi_frames.append(phi)
        psi_frames.append(psi)
        phi_max_series.append(float(np.max(phi)))
        times.append(float(t))
        t_ints.append(int(t_int) if t_int is not None else int(round(10 * t)))
        if args.save_stack:
            stack_t.append(float(t))
            stack_phi.append(phi.astype(np.float32))
            stack_psi.append(psi.astype(np.float32))

    if not rows_phi:
        print(f"[!] No valid frames in {run_dir}")
        return None

    media_variant = safe_slug(args.media_variant) if args.media_variant else auto_variant_name(args)
    media_root, variant_dir, frames_phi_dir, frames_psi_dir = make_media_dirs(run_dir, not args.no_psi_frames, media_variant)
    print(f"\n[{tag}] frames={len(files)}")
    print(f" run_dir      -> {run_dir}")
    print(f" media_root   -> {media_root}")
    print(f" variant_dir  -> {variant_dir}")

    max_phi_local = [float(np.max(phi_max_series[max(0, i - 10):i + 1])) for i in range(len(phi_max_series))]

    mid_phi_path = os.path.join(media_root, f"midline_phi_t_y_{tag}.csv")
    mid_psi_path = os.path.join(media_root, f"midline_psi_t_y_{tag}.csv")
    tr_path = os.path.join(media_root, f"track_2d_{tag}.csv")
    core_csv_path = os.path.join(media_root, f"core_2d_timeseries_{tag}.csv")
    core_json_path = os.path.join(media_root, f"core_2d_summary_{tag}.json")
    gif_path = os.path.join(variant_dir, f"{tag}.gif")
    stack_path = os.path.join(media_root, f"field_2d_stack_{tag}.npz")
    render_manifest_path = os.path.join(variant_dir, "render_manifest.json")

    save_midline_csv(mid_phi_path, rows_phi, len(rows_phi[0]) - 1)
    save_midline_csv(mid_psi_path, rows_psi, len(rows_psi[0]) - 1)
    save_track_csv(tr_path, rows_track, max_phi_local)
    save_core_timeseries_csv(core_csv_path, rows_core)

    core_summary = {
        "run_id": tag, "run_dir": run_dir, "n_frames": len(times),
        "t_min": float(min(times)), "t_max": float(max(times)),
        "area_mean": float(np.mean([r["area"] for r in rows_core])),
        "area_max": float(np.max([r["area"] for r in rows_core])),
        "r_eff_mean": float(np.mean([r["r_eff"] for r in rows_core])),
        "phi_core_max": float(np.max([r["phi_max_core"] for r in rows_core])),
        "phi_global_max": float(np.max([r["phi_global_max"] for r in rows_core])),
        "n_components_max": int(np.max([r["n_components"] for r in rows_core])),
        "source_counts": {
            "threshold": int(sum(r["source"] == "threshold" for r in rows_core)),
            "fallback-max": int(sum(r["source"] == "fallback-max" for r in rows_core)),
        },
    }
    with ensure_parent_dir(core_json_path).open("w", encoding="utf-8") as f:
        json.dump(core_summary, f, ensure_ascii=False, indent=2)

    if args.save_stack:
        np.savez_compressed(
            ensure_parent_dir(stack_path),
            t=np.asarray(stack_t, dtype=np.float32),
            phi=np.stack(stack_phi, axis=0).astype(np.float32),
            psi=np.stack(stack_psi, axis=0).astype(np.float32),
        )

    selected_idx = choose_frames_by_time(times, args.t_start, args.t_end, args.times, args.time_tolerance)
    if not selected_idx:
        print(f"[!] No frames selected for rendering in {run_dir}")
        return None

    phi_frames_sel = [phi_frames[i] for i in selected_idx]
    psi_frames_sel = [psi_frames[i] for i in selected_idx]
    times_sel = [times[i] for i in selected_idx]
    t_ints_sel = [t_ints[i] for i in selected_idx]

    crop = compute_crop_window(phi_frames_sel, margin=args.zoom_margin, thresh=args.core_threshold) if args.zoom == "auto" else None

    phi_pre_vmin = phi_pre_vmax = None
    psi_pre_vmin = psi_pre_vmax = None
    if args.phi_scale == "global":
        phi_ref = phi_frames_sel if args.phi_global_ref == "window" else phi_frames
        phi_concat = np.concatenate([apply_transform(f, args.phi_transform).ravel() for f in phi_ref])
        phi_pre_vmin, phi_pre_vmax = compute_dynamic_range(phi_concat, mode=args.phi_norm, k_sigma=args.phi_k_sigma, q_low=args.phi_q_low, q_high=args.phi_q_high)
    if args.psi_scale == "global":
        psi_ref = psi_frames_sel if args.psi_global_ref == "window" else psi_frames
        psi_concat = np.concatenate([apply_transform(f, args.psi_transform).ravel() for f in psi_ref])
        psi_pre_vmin, psi_pre_vmax = compute_dynamic_range(psi_concat, mode=args.psi_norm, k_sigma=args.psi_k_sigma, q_low=args.psi_q_low, q_high=args.psi_q_high)

    phi_pngs: List[str] = []
    psi_pngs: List[str] = []
    for i, (phi, psi, t, t_int) in enumerate(zip(phi_frames_sel, psi_frames_sel, times_sel, t_ints_sel)):
        phi_png = os.path.join(frames_phi_dir, f"frame_{i:04d}_t{t_int:07d}.png")
        render_frame(phi, t, phi_png, "phi(x,y)", args.cmap_phi, args.gif_size, crop=crop, show_axes=not args.no_axes, scale_mode=args.phi_scale, fixed_vmin=args.phi_vmin, fixed_vmax=args.phi_vmax, precomputed_vmin=phi_pre_vmin, precomputed_vmax=phi_pre_vmax, dynamic_mode=args.phi_norm, k_sigma=args.phi_k_sigma, q_low=args.phi_q_low, q_high=args.phi_q_high, transform=args.phi_transform)
        phi_pngs.append(phi_png)
        if not args.no_psi_frames:
            psi_png = os.path.join(frames_psi_dir, f"frame_{i:04d}_t{t_int:07d}.png")
            render_frame(psi, t, psi_png, "psi(x,y)", args.cmap_psi, args.gif_size, crop=crop, show_axes=not args.no_axes, scale_mode=args.psi_scale, fixed_vmin=args.psi_vmin, fixed_vmax=args.psi_vmax, precomputed_vmin=psi_pre_vmin, precomputed_vmax=psi_pre_vmax, dynamic_mode=args.psi_norm, k_sigma=args.psi_k_sigma, q_low=args.psi_q_low, q_high=args.psi_q_high, transform=args.psi_transform)
            psi_pngs.append(psi_png)

    if not args.no_gif:
        build_gif_from_pngs(phi_pngs, gif_path, fps=args.fps)

    render_manifest = {
        "run_id": tag,
        "variant": media_variant,
        "variant_dir": variant_dir,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "times": args.times,
        "time_tolerance": args.time_tolerance,
        "selected_n_frames": len(selected_idx),
        "selected_times": times_sel,
        "phi": {
            "scale": args.phi_scale,
            "norm": args.phi_norm,
            "k_sigma": args.phi_k_sigma,
            "q_low": args.phi_q_low,
            "q_high": args.phi_q_high,
            "vmin": args.phi_vmin,
            "vmax": args.phi_vmax,
            "global_ref": args.phi_global_ref,
            "transform": args.phi_transform,
            "precomputed_vmin": phi_pre_vmin,
            "precomputed_vmax": phi_pre_vmax,
        },
        "psi": {
            "scale": args.psi_scale,
            "norm": args.psi_norm,
            "k_sigma": args.psi_k_sigma,
            "q_low": args.psi_q_low,
            "q_high": args.psi_q_high,
            "vmin": args.psi_vmin,
            "vmax": args.psi_vmax,
            "global_ref": args.psi_global_ref,
            "transform": args.psi_transform,
            "precomputed_vmin": psi_pre_vmin,
            "precomputed_vmax": psi_pre_vmax,
        },
        "zoom": args.zoom,
        "zoom_margin": args.zoom_margin,
        "crop": crop,
        "gif_path": None if args.no_gif else gif_path,
        "frames_phi_dir": frames_phi_dir,
        "frames_psi_dir": None if args.no_psi_frames else frames_psi_dir,
    }
    with ensure_parent_dir(render_manifest_path).open("w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)

    print(f" CSV  : {norm_path(mid_phi_path)}")
    print(f" CSV  : {norm_path(mid_psi_path)}")
    print(f" CSV  : {norm_path(tr_path)}")
    print(f" CSV  : {norm_path(core_csv_path)}")
    print(f" JSON : {norm_path(core_json_path)}")
    print(f" RJSON: {norm_path(render_manifest_path)}")
    if args.save_stack:
        print(f" NPZ  : {norm_path(stack_path)}")
    if not args.no_gif:
        print(f" GIF  : {norm_path(gif_path)}")
    print(f" PNG  : {norm_path(frames_phi_dir)}")
    if not args.no_psi_frames:
        print(f" PNG  : {norm_path(frames_psi_dir)}")

    return {
        "run_id": tag,
        "run_dir": run_dir,
        "n_frames": len(times_sel),
        "midline_phi_csv": os.path.basename(mid_phi_path),
        "midline_psi_csv": os.path.basename(mid_psi_path),
        "track_csv": os.path.basename(tr_path),
        "core_2d_csv": os.path.basename(core_csv_path),
        "core_2d_summary_json": os.path.basename(core_json_path),
        "gif": os.path.relpath(gif_path, run_dir) if not args.no_gif else "",
        "frames_phi_dir": os.path.relpath(frames_phi_dir, run_dir),
        "frames_psi_dir": os.path.relpath(frames_psi_dir, run_dir) if not args.no_psi_frames else "",
        "field_2d_stack_npz": os.path.basename(stack_path) if args.save_stack else "",
    }


def build_parser() -> argparse.ArgumentParser:
    examples = "\n".join([
        "Examples:",
        "  Process one run with default dynamic rendering:",
        "    python npz_to_csv_gif.py --run-id path/to/Q1DA_05",
        "",
        "  Render only a time window and store outputs in a dedicated media variant:",
        "    python npz_to_csv_gif.py --run-id path/to/Q1DA_05 \\",
        "      --t_start 500 --t_end 900 --media_variant late_window",
        "",
        "  Create a fixed-scale article triptych without overwriting previous renders:",
        "    python npz_to_csv_gif.py --run-id path/to/Q1DA_05 \\",
        "      --times 375,408,846 --phi_scale fixed --phi_vmin -15 --phi_vmax 15 \\",
        "      --media_variant article_triptych_fixed",
        "",
        "  Use global scaling within the selected window and apply asinh transform:",
        "    python npz_to_csv_gif.py --run-id path/to/Q1DA_05 \\",
        "      --t_start 780 --t_end 900 --phi_scale global --phi_global_ref window \\",
        "      --phi_transform asinh --media_variant article_late_asinh",
        "",
        "  Process all runs inside a collection directory:",
        "    python npz_to_csv_gif.py --out-dir path/to/models/q1_vector_diagonal_serial_a --save_stack",
    ])

    ap = argparse.ArgumentParser(
        description=(
            "Convert snapshot2d NPZ files into CSV / JSON summaries, PNG frames, and GIF animations. "
            "The tool is compatible with sim_framework-style run directories and supports reproducible "
            "visualization variants, time-window rendering, fixed/global scaling, and optional signal transforms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )

    target = ap.add_argument_group("Input selection")
    target.add_argument("--run-id", default=None, help="Path to a single run directory. Only this run will be processed.")
    target.add_argument("--out-dir", default=None, help="Path to a directory containing multiple run directories. All discovered runs will be processed.")

    timeg = ap.add_argument_group("Time selection")
    timeg.add_argument("--t_start", type=float, default=None, help="Render only frames with t >= t_start.")
    timeg.add_argument("--t_end", type=float, default=None, help="Render only frames with t <= t_end.")
    timeg.add_argument("--times", type=str, default=None, help="Comma-separated target times, for example: 375,408,846. The nearest available frame is chosen for each value.")
    timeg.add_argument("--time_tolerance", type=float, default=1e9, help="Maximum absolute difference allowed when matching --times to available snapshots. Default is very permissive.")

    varg = ap.add_argument_group("Media variant management")
    varg.add_argument("--media_variant", default=None, help="Explicit output subfolder name inside processed/media/. If omitted, an automatic name is generated from key visualization settings.")

    phig = ap.add_argument_group("Phi rendering options")
    phig.add_argument("--phi_scale", choices=["dynamic", "global", "fixed", "symmetric"], default="dynamic", help="Scaling strategy for phi frames: dynamic per-frame, global across reference frames, fixed manual limits, or symmetric dynamic range around zero.")
    phig.add_argument("--phi_norm", choices=["minmax", "meanstd", "quantile"], default="meanstd", help="Statistic used to estimate dynamic/global color range for phi.")
    phig.add_argument("--phi_k_sigma", type=float, default=2.5, help="Sigma multiplier used when --phi_norm meanstd is selected.")
    phig.add_argument("--phi_q_low", type=float, default=0.02, help="Lower quantile used when --phi_norm quantile is selected.")
    phig.add_argument("--phi_q_high", type=float, default=0.98, help="Upper quantile used when --phi_norm quantile is selected.")
    phig.add_argument("--phi_vmin", type=float, default=None, help="Manual lower color limit for phi. Required together with --phi_vmax when --phi_scale fixed is used.")
    phig.add_argument("--phi_vmax", type=float, default=None, help="Manual upper color limit for phi. Required together with --phi_vmin when --phi_scale fixed is used.")
    phig.add_argument("--phi_global_ref", choices=["full", "window"], default="window", help="Reference set for phi global scaling: full uses all run frames, window uses only selected frames.")
    phig.add_argument("--phi_transform", choices=["none", "log1p", "asinh"], default="none", help="Value transform applied before phi rendering. log1p is sign-preserving; asinh is often useful for large-amplitude fields.")
    phig.add_argument("--cmap_phi", default="magma", help="Matplotlib colormap for phi frames.")

    psig = ap.add_argument_group("Psi rendering options")
    psig.add_argument("--psi_scale", choices=["dynamic", "global", "fixed", "symmetric"], default="dynamic", help="Scaling strategy for psi frames.")
    psig.add_argument("--psi_norm", choices=["minmax", "meanstd", "quantile"], default="quantile", help="Statistic used to estimate dynamic/global color range for psi.")
    psig.add_argument("--psi_k_sigma", type=float, default=2.5, help="Sigma multiplier used when --psi_norm meanstd is selected.")
    psig.add_argument("--psi_q_low", type=float, default=0.02, help="Lower quantile used when --psi_norm quantile is selected.")
    psig.add_argument("--psi_q_high", type=float, default=0.98, help="Upper quantile used when --psi_norm quantile is selected.")
    psig.add_argument("--psi_vmin", type=float, default=None, help="Manual lower color limit for psi. Required with --psi_vmax when --psi_scale fixed is used.")
    psig.add_argument("--psi_vmax", type=float, default=None, help="Manual upper color limit for psi. Required with --psi_vmin when --psi_scale fixed is used.")
    psig.add_argument("--psi_global_ref", choices=["full", "window"], default="window", help="Reference set for psi global scaling.")
    psig.add_argument("--psi_transform", choices=["none", "log1p", "asinh"], default="none", help="Value transform applied before psi rendering.")
    psig.add_argument("--cmap_psi", default="viridis", help="Matplotlib colormap for psi frames.")
    psig.add_argument("--no_psi_frames", action="store_true", help="Do not render psi PNG frames.")

    outg = ap.add_argument_group("Output and rendering control")
    outg.add_argument("--fps", type=int, default=6, help="Frames per second for the output GIF.")
    outg.add_argument("--gif_size", type=int, default=1800, help="Target figure size used during PNG rendering. Larger values increase image size and render time.")
    outg.add_argument("--no_gif", action="store_true", help="Skip GIF generation and render only PNG frames.")
    outg.add_argument("--zoom", choices=["none", "auto"], default="auto", help="Auto computes a crop around the thresholded activity over the selected frames. none renders the full domain.")
    outg.add_argument("--zoom_margin", type=int, default=12, help="Extra pixels added around the automatically detected crop.")
    outg.add_argument("--no_axes", action="store_true", help="Hide axis ticks and labels in rendered images.")
    outg.add_argument("--save_stack", action="store_true", help="Also save field_2d_stack_<tag>.npz into processed/media/.")
    outg.add_argument("--core_threshold", type=float, default=CORE_THRESHOLD, help="Threshold used for center detection, crop estimation, and core metrics.")

    return ap


def validate_args(args: argparse.Namespace) -> None:
    args.times = parse_times_arg(args.times)
    if bool(args.run_id) == bool(args.out_dir):
        raise SystemExit("Exactly one of --run-id or --out-dir must be provided.")
    if args.phi_scale == "fixed" and (args.phi_vmin is None or args.phi_vmax is None):
        raise SystemExit("--phi_scale fixed requires both --phi_vmin and --phi_vmax.")
    if args.psi_scale == "fixed" and (args.psi_vmin is None or args.psi_vmax is None):
        raise SystemExit("--psi_scale fixed requires both --psi_vmin and --psi_vmax.")
    if args.t_start is not None and args.t_end is not None and args.t_start > args.t_end:
        raise SystemExit("--t_start must not be greater than --t_end.")
    if args.phi_vmin is not None and args.phi_vmax is not None and args.phi_vmin >= args.phi_vmax:
        raise SystemExit("phi limits must satisfy phi_vmin < phi_vmax.")
    if args.psi_vmin is not None and args.psi_vmax is not None and args.psi_vmin >= args.psi_vmax:
        raise SystemExit("psi limits must satisfy psi_vmin < psi_vmax.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    targets = resolve_targets(args.run_id, args.out_dir)
    print(f"Discovered run directories: {len(targets)}")

    manifest_rows: List[Dict[str, Any]] = []
    total = 0
    for run_dir in targets:
        rec = process_run(run_dir, args)
        if rec is not None:
            manifest_rows.append(rec)
            total += int(rec["n_frames"])

    if args.out_dir and manifest_rows:
        out_root = norm_path(args.out_dir)
        manifest_csv = os.path.join(out_root, "processed_manifest.csv")
        manifest_json = os.path.join(out_root, "processed_manifest.json")
        save_manifest_csv(manifest_csv, manifest_rows)
        with ensure_parent_dir(manifest_json).open("w", encoding="utf-8") as f:
            json.dump({"runs": manifest_rows}, f, ensure_ascii=False, indent=2)
        print(f"\nManifest CSV : {norm_path(manifest_csv)}")
        print(f"Manifest JSON: {norm_path(manifest_json)}")

    print(f"\nDone. Total rendered frames: {total}")


if __name__ == "__main__":
    main()
