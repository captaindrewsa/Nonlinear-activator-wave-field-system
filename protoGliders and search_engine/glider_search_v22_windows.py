#!/usr/bin/env python3
"""
glider_serach_v22_windows.py
================================================================================
Two-soliton interaction / glider search for the 2-field autosoliton PDE system.

MODEL
-----
Reaction term with two sigmoid gates (kappa=10, theta1=4, theta2=16):
    sigma_i(u) = 1 / (1 + exp(-kappa*(u - theta_i)))
    M(phi)     = phi * sigma1(phi^2) * (1 - sigma2(phi^2)) - phi

Integrated PDE system (D_psi = 0):
    d_t  phi = M(phi) + h + psi
    d_tt psi + eps * d_t psi = D_psi * Lap(psi) + Lap(phi)

Explicit time stepping (dt=0.003), reformulated with v = d_t psi:
    phi <- phi + dt*(M(phi) + h + psi)
    v   <- v   + dt*(Lap(phi) + D_psi*Lap(psi) - eps*v)
    psi <- psi + dt*v

NUMERICS
--------
- Grid NX=NY=224, dx=1, periodic boundaries.
- Spatial operator: 9-point isotropic Laplacian (torch.roll based).
- Backend auto-select: CUDA -> DirectML -> CPU (via torch).

PURPOSE (what this script actually does)
----------------------------------------
Scans a multi-parameter grid to look for moving/interacting localized
structures ("gliders") formed by TWO interacting autosoliton spots.
Scanned parameters:
    eps, h                       -- model parameters (damping, drive)
    dist, dy                     -- separation and vertical offset of the pair
    phase_shift                  -- relative breathing phase of the two spots
    kick, kick_type ('v'|'psi')  -- localized momentum/field kick
    amp_asym, rad_asym           -- amplitude / radius asymmetry between spots

WORKFLOW
--------
1. warm_seed_bank(): for each (eps, h, radius) build a single relaxed spot by
   evolving a radial disk (radius ~8, phi=PHI_HI on background PHI_LO) for
   T_WARM=160, storing N_CYCLE_SNAPSHOTS snapshots over one breathing cycle.
   Seeds are cached to disk (seed_cache/*.pt) keyed by an MD5 of the params.
2. build_pair_from_cycle(): place two seeds (left/right) at a chosen breathing
   phase, separation and asymmetry, then optionally apply a Gaussian kick.
3. run_case(): integrate for T_RUN=1400, sampling every MONITOR_EVERY steps:
   - detect_center(): weighted centroid of the region phi > CENTER_THRESHOLD,
   - track center (x, y), mass, and center value phi(cx, cy) over time,
   - store a space-time slice phi[NX//2, :] for each sampled time.
4. classify(): linear fit of the center trajectory ->
       label in {glider, pinned, pinned_breather, complex, decay}
   using mean mass, drift, propagation speed, fit R^2 and breathing amplitude.

OUTPUTS (per run, written to output_glider_v22_focus/)
------------------------------------------------------
- glider_scan_v22.csv     : one row per case (params + classification metrics),
                            appended incrementally; used for resume/skip.
- track_<tag>.csv         : time series t, x, y, mass, phi_center per case.
- space_time_<tag>.csv/png: space-time diagram phi(center row, y, t).
- preview_<tag>.gif       : optional preview for the best glider/complex cases.
- glider_candidates_v22.json, glider_summary_v22.json, velocity_map_v22.csv,
  progress_v22.jsonl, seed_cache_index_v22.json.

Runs are resumable: cases already present in the scan CSV, or with existing
track/space-time output files, are skipped (see case_already_done / seen_tags).

NOTE
----
This is the ORIGINAL exploratory two-spot glider-search code. The (eps, h)
phase-diagram scripts (phase_scan_*.py) reuse only its model, 9-point
Laplacian, stepping scheme and single-spot radial initial condition -- not the
pair-construction / glider-classification machinery.
================================================================================
"""
import os
import json
import csv
import time
import hashlib
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch

try:
    import torch_directml
    HAS_DML = False
except Exception:
    HAS_DML = False

OUT = "output_glider_v22_focus"
CACHE_DIR = os.path.join(OUT, "seed_cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SCAN_CSV = os.path.join(OUT, "glider_scan_v22.csv")
CANDIDATES_JSON = os.path.join(OUT, "glider_candidates_v22.json")
SUMMARY_JSON = os.path.join(OUT, "glider_summary_v22.json")
VELOCITY_CSV = os.path.join(OUT, "velocity_map_v22.csv")
PROGRESS_JSONL = os.path.join(OUT, "progress_v22.jsonl")

KAPPA = 10.0
THETA1 = 4.0
THETA2 = 16.0
PHI_LO = 0.533
PHI_HI = 3.9766096853487105
DW = 0.0

NX = 224
NY = 224
DX = 1.0
DT = 0.003
T_WARM = 160.0
T_RUN = 1400.0
FRAME_EVERY = 60
MONITOR_EVERY = 10
CENTER_THRESHOLD = 1.5
N_CYCLE_SNAPSHOTS = 16

#Облегченный поиск
EPS_LIST = [2.80, 2.78, 2.82, 2.90]
H_LIST = [0.533, 0.545]
DIST_LIST = [18, 14]
PHASE_LIST = [0.0, 0.25, 0.5]
DY_LIST = [0, 4]
KICK_LIST = [0.06, 0.0, 0.03]
KICK_TYPE_LIST = ["v", "psi"]
AMP_ASYM_LIST = [0.0, 0.15]
RAD_ASYM_LIST = [0, 1]
SAVE_TOP_GIFS = 4   # number of best glider/complex cases to render as preview GIFs

#Полный поиск
#EPS_LIST = [2.74, 2.78, 2.80, 2.82, 2.86, 2.90, 3.00]
#H_LIST = [0.533, 0.539, 0.545]
#DIST_LIST = [12, 14, 16, 18, 20]
#PHASE_LIST = [0.0, 0.125, 0.25, 0.375, 0.5]
#DY_LIST = [0, 2, 4, 6]
#KICK_LIST = [0.0, 0.015, 0.03, 0.045, 0.06]
#KICK_TYPE_LIST = ["v", "psi"]
#AMP_ASYM_LIST = [0.0, 0.08, 0.15]
#RAD_ASYM_LIST = [0, 1, 2]
#SAVE_TOP_GIFS = 8

#Старый вариант
# EPS_LIST = [2.7, 2.9, 3.2, 3.4]
# H_LIST = [0.52, 0.533, 0.545]
# DIST_LIST = [14, 18, 22, 26]
# PHASE_LIST = [0.0, 0.25, 0.5]
# DY_LIST = [0, 4]
# KICK_LIST = [0.0, 0.03, 0.06]
# KICK_TYPE_LIST = ["v", "psi"]
# AMP_ASYM_LIST = [0.0, 0.15]
# RAD_ASYM_LIST = [0, 1]
# SAVE_TOP_GIFS = 4

@dataclass
class CaseParams:
    eps: float
    h: float
    dist: int
    phase_shift: float
    dy: int
    kick: float
    kick_type: str
    amp_asym: float
    rad_asym: int

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if HAS_DML:
        return torch_directml.device(), "directml"
    return torch.device("cpu"), "cpu"

def sigma1(u):
    return 1.0 / (1.0 + torch.exp(-KAPPA * (u - THETA1)))

def sigma2(u):
    return 1.0 / (1.0 + torch.exp(-KAPPA * (u - THETA2)))

def M(phi):
    return phi * sigma1(phi * phi) * (1.0 - sigma2(phi * phi)) - phi

def laplacian9(u, dx):
    up = torch.roll(u, shifts=1, dims=0)
    dn = torch.roll(u, shifts=-1, dims=0)
    lf = torch.roll(u, shifts=1, dims=1)
    rt = torch.roll(u, shifts=-1, dims=1)
    ul = torch.roll(up, shifts=1, dims=1)
    ur = torch.roll(up, shifts=-1, dims=1)
    dl = torch.roll(dn, shifts=1, dims=1)
    dr = torch.roll(dn, shifts=-1, dims=1)
    return ((4.0 * (up + dn + lf + rt) + (ul + ur + dl + dr) - 20.0 * u) / 6.0) / (dx * dx)

def disk_mask(nx, ny, cx, cy, radius, device):
    x = torch.arange(nx, device=device).view(-1, 1)
    y = torch.arange(ny, device=device).view(1, -1)
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2

def gaussian2d(nx, ny, cx, cy, sigma_x, sigma_y, device):
    x = torch.arange(nx, device=device).view(-1, 1)
    y = torch.arange(ny, device=device).view(1, -1)
    return torch.exp(-0.5 * (((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2))

def seed_cache_path(eps, h, radius):
    key = f"eps={eps:.6f}|h={h:.6f}|r={radius}|nx={NX}|ny={NY}|dt={DT}|tw={T_WARM}|ns={N_CYCLE_SNAPSHOTS}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"seed_{digest}.pt")

def build_single_seed(device, h, eps, radius=8):
    phi = torch.full((NX, NY), PHI_LO, dtype=torch.float32, device=device)
    psi = torch.zeros((NX, NY), dtype=torch.float32, device=device)
    v = torch.zeros((NX, NY), dtype=torch.float32, device=device)

    cx, cy = NX // 2, NY // 2
    mask = disk_mask(NX, NY, cx, cy, radius, device)
    phi = phi.clone()
    phi[mask] = PHI_HI

    steps = int(T_WARM / DT)
    sample_stride = max(1, steps // N_CYCLE_SNAPSHOTS)
    snapshots = []

    for n in range(steps + 1):
        if n % sample_stride == 0:
            snapshots.append((
                phi.detach().cpu().clone(),
                psi.detach().cpu().clone(),
                v.detach().cpu().clone()
            ))
        lap_phi = laplacian9(phi, DX)
        lap_psi = laplacian9(psi, DX)
        phi = phi + DT * (M(phi) + h + psi)
        v = v + DT * (lap_phi + DW * lap_psi - eps * v)
        psi = psi + DT * v

    return snapshots[:N_CYCLE_SNAPSHOTS]

def load_or_build_seed(device, h, eps, radius=8):
    path = seed_cache_path(eps, h, radius)
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu")
        snapshots = []
        for item in data["snapshots"]:
            snapshots.append((item["phi"].to(device), item["psi"].to(device), item["v"].to(device)))
        return snapshots, True, path

    raw = build_single_seed(device, h, eps, radius)
    serializable = {"eps": eps, "h": h, "radius": radius, "snapshots": []}
    snapshots = []
    for phi, psi, v in raw:
        serializable["snapshots"].append({"phi": phi, "psi": psi, "v": v})
        snapshots.append((phi.to(device), psi.to(device), v.to(device)))
    torch.save(serializable, path)
    return snapshots, False, path

def place_shifted(base, shift_x, shift_y):
    return torch.roll(base, shifts=(shift_x, shift_y), dims=(0, 1))

def build_pair_from_cycle(device, params, seed_bank):
    left_key = (params.eps, params.h, 8)
    right_key = (params.eps, params.h, 8 + params.rad_asym)

    cycle_left = seed_bank[left_key]
    cycle_right = seed_bank[right_key]

    nL = 0
    nR = int(round(params.phase_shift * (len(cycle_right) - 1))) % len(cycle_right)

    phiL, psiL, vL = cycle_left[nL]
    phiR, psiR, vR = cycle_right[nR]

    phiR = PHI_LO + (phiR - PHI_LO) * (1.0 - params.amp_asym / max(PHI_HI, 1e-6))

    phi = torch.full((NX, NY), PHI_LO, dtype=torch.float32, device=device)
    psi = torch.zeros((NX, NY), dtype=torch.float32, device=device)
    v = torch.zeros((NX, NY), dtype=torch.float32, device=device)

    sx = params.dist // 2
    dy = params.dy

    phi += place_shifted(phiL - PHI_LO, -sx, -dy // 2) + place_shifted(phiR - PHI_LO, sx, dy // 2)
    psi += place_shifted(psiL, -sx, -dy // 2) + place_shifted(psiR, sx, dy // 2)
    v += place_shifted(vL, -sx, -dy // 2) + place_shifted(vR, sx, dy // 2)

    cx, cy = NX // 2, NY // 2
    kick_profile = gaussian2d(NX, NY, cx, cy, 10.0, 4.0, device)
    kick_profile = kick_profile * torch.sign(torch.arange(NY, device=device).view(1, -1) - cy)

    if params.kick_type == "v":
        v = v + params.kick * kick_profile
    elif params.kick_type == "psi":
        psi = psi + params.kick * kick_profile

    phi = torch.clamp(phi, min=PHI_LO - 0.2, max=PHI_HI + 0.5)
    return phi, psi, v

def detect_center(phi_np, thresh=CENTER_THRESHOLD):
    mask = phi_np > thresh
    mass = int(mask.sum())
    if mass < 20:
        return None, mass
    coords = np.argwhere(mask)
    vals = phi_np[mask]
    w = vals - thresh
    if np.all(w <= 0):
        w = np.ones_like(vals)
    cx = float(np.sum(coords[:, 0] * w) / np.sum(w))
    cy = float(np.sum(coords[:, 1] * w) / np.sum(w))
    return (cx, cy), mass

def classify(times, xs, ys, masses, breathing_signal):
    x = np.array(xs)
    y = np.array(ys)
    t = np.array(times)
    m = np.array(masses)
    b = np.array(breathing_signal)

    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 8:
        return {"label": "decay", "speed": 0.0, "r2": 0.0, "vx": 0.0, "vy": 0.0, "breath_amp": 0.0}

    x = x[good]
    y = y[good]
    t = t[good]

    px = np.polyfit(t, x, 1)
    py = np.polyfit(t, y, 1)
    xfit = np.polyval(px, t)
    yfit = np.polyval(py, t)

    r2x = 1.0 - np.sum((x - xfit) ** 2) / max(np.sum((x - x.mean()) ** 2), 1e-12)
    r2y = 1.0 - np.sum((y - yfit) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    speed = float(np.sqrt(px[0] ** 2 + py[0] ** 2))
    drift = float(max(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y)))
    breath_amp = float(np.nanpercentile(b, 95) - np.nanpercentile(b, 5)) if len(b) else 0.0
    mean_mass = float(np.nanmean(m)) if len(m) else 0.0
    r2 = float(max(r2x, r2y))

    if mean_mass < 40:
        label = "decay"
    elif speed > 0.015 and drift > 8 and r2 > 0.90:
        label = "glider"
    elif breath_amp > 0.15:
        label = "pinned_breather"
    elif drift > 5:
        label = "complex"
    else:
        label = "pinned"

    return {
        "label": label,
        "speed": speed,
        "r2": r2,
        "vx": float(px[0]),
        "vy": float(py[0]),
        "breath_amp": breath_amp
    }

def save_space_time(lines, times, png_path, csv_path):
    arr = np.array(lines)

    plt.figure(figsize=(8, 5))
    plt.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, arr.shape[1] - 1, times[0], times[-1]]
    )
    plt.colorbar(label="phi(center row, y, t)")
    plt.xlabel("y index")
    plt.ylabel("t")
    plt.title("Space-time plot")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"y{j}" for j in range(arr.shape[1])])
        for tt, row in zip(times, arr):
            w.writerow([tt] + row.tolist())

def tag_from_params(params):
    return (
        f"eps{params.eps:.3f}_h{params.h:.3f}_d{params.dist}"
        f"_ph{params.phase_shift:.2f}_dy{params.dy}"
        f"_k{params.kick:.2f}_{params.kick_type}"
        f"_aa{params.amp_asym:.2f}_ra{params.rad_asym}"
    ).replace(".", "p")

def case_output_paths(params):
    tag = tag_from_params(params)
    return {
        "tag": tag,
        "track_csv": os.path.join(OUT, f"track_{tag}.csv"),
        "space_time_png": os.path.join(OUT, f"space_time_{tag}.png"),
        "space_time_csv": os.path.join(OUT, f"space_time_{tag}.csv"),
        "preview_gif": os.path.join(OUT, f"preview_{tag}.gif"),
    }

def find_existing_by_prefix(prefix_name):
    matches = []
    for fn in os.listdir(OUT):
        if fn.startswith(prefix_name):
            matches.append(os.path.join(OUT, fn))
    return sorted(matches)

def case_already_done(params):
    p = case_output_paths(params)
    tag = p["tag"]

    track_exact = os.path.exists(p["track_csv"])
    st_exact = os.path.exists(p["space_time_csv"])

    if track_exact and st_exact:
        return True

    track_pref = find_existing_by_prefix(f"track_{tag}")
    st_pref = find_existing_by_prefix(f"space_time_{tag}")

    has_track = any(x.endswith(".csv") for x in track_pref)
    has_st_csv = any(x.endswith(".csv") for x in st_pref)

    return has_track and has_st_csv

def append_progress(rec):
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_existing_results():
    results = []
    seen_tags = set()

    if not os.path.exists(SCAN_CSV):
        return results, seen_tags

    with open(SCAN_CSV, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        r = {}
        for k, v in row.items():
            if k in {"kick_type", "label", "backend"}:
                r[k] = v
            elif k in {"dist", "dy", "rad_asym"}:
                r[k] = int(float(v))
            else:
                try:
                    r[k] = float(v)
                except Exception:
                    r[k] = v

        params = CaseParams(
            eps=float(r["eps"]),
            h=float(r["h"]),
            dist=int(r["dist"]),
            phase_shift=float(r["phase_shift"]),
            dy=int(r["dy"]),
            kick=float(r["kick"]),
            kick_type=str(r["kick_type"]),
            amp_asym=float(r["amp_asym"]),
            rad_asym=int(r["rad_asym"]),
        )
        tag = tag_from_params(params)
        seen_tags.add(tag)
        results.append(r)

    return results, seen_tags

def append_result_to_scan_csv(result):
    file_exists = os.path.exists(SCAN_CSV)
    fieldnames = list(result.keys())
    with open(SCAN_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(result)

def run_case(device, backend_name, params, seed_bank, save_gif=False):
    phi, psi, v = build_pair_from_cycle(device, params, seed_bank)
    steps = int(T_RUN / DT)

    times, xs, ys, masses, breathing_signal = [], [], [], [], []
    row_lines, gif_frames = [], []

    for n in range(steps + 1):
        t = n * DT

        if n % MONITOR_EVERY == 0:
            phi_np = phi.detach().cpu().numpy()
            center, mass = detect_center(phi_np)

            times.append(t)
            masses.append(mass)
            breathing_signal.append(float(phi_np[NX // 2, NY // 2]))

            if center is None:
                xs.append(np.nan)
                ys.append(np.nan)
            else:
                xs.append(center[0])
                ys.append(center[1])

            row_lines.append(phi_np[NX // 2, :].copy())

            if save_gif and n % FRAME_EVERY == 0:
                img = phi_np
                lo, hi = PHI_LO - 0.1, PHI_HI + 0.3
                img = np.clip((img - lo) / (hi - lo), 0, 1)
                rgba = plt.cm.plasma(img)
                gif_frames.append((255 * rgba[:, :, :3]).astype(np.uint8))

        lap_phi = laplacian9(phi, DX)
        lap_psi = laplacian9(psi, DX)
        phi = phi + DT * (M(phi) + params.h + psi)
        v = v + DT * (lap_phi + DW * lap_psi - params.eps * v)
        psi = psi + DT * v

    cls = classify(times, xs, ys, masses, breathing_signal)
    paths = case_output_paths(params)

    with open(paths["track_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "mass", "phi_center"])
        for row in zip(times, xs, ys, masses, breathing_signal):
            w.writerow(row)

    save_space_time(row_lines, times, paths["space_time_png"], paths["space_time_csv"])

    if save_gif and len(gif_frames) > 1:
        imageio.mimsave(paths["preview_gif"], gif_frames, duration=0.07)

    result = asdict(params)
    result.update(cls)
    result["backend"] = backend_name
    result["final_mass"] = float(masses[-1]) if masses else 0.0
    return result

def build_param_grid():
    cases = []
    for eps in EPS_LIST:
        for h in H_LIST:
            for dist in DIST_LIST:
                for phase_shift in PHASE_LIST:
                    for dy in DY_LIST:
                        for kick in KICK_LIST:
                            for kick_type in KICK_TYPE_LIST:
                                for amp_asym in AMP_ASYM_LIST:
                                    for rad_asym in RAD_ASYM_LIST:
                                        if kick == 0.0 and kick_type == "psi":
                                            continue
                                        cases.append(CaseParams(
                                            eps, h, dist, phase_shift, dy,
                                            kick, kick_type, amp_asym, rad_asym
                                        ))
    return cases

def save_velocity_map(results):
    with open(VELOCITY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eps", "h", "dist", "phase_shift", "dy", "kick", "kick_type", "mean_speed"])
        grouped = {}
        for r in results:
            key = (r["eps"], r["h"], r["dist"], r["phase_shift"], r["dy"], r["kick"], r["kick_type"])
            grouped.setdefault(key, []).append(r["speed"])
        for key, vals in grouped.items():
            w.writerow(list(key) + [float(np.mean(vals))])

def warm_seed_bank(device):
    bank = {}
    stats = []
    for eps in EPS_LIST:
        for h in H_LIST:
            for radius in sorted(set([8] + [8 + r for r in RAD_ASYM_LIST])):
                snapshots, from_cache, path = load_or_build_seed(device, h, eps, radius)
                bank[(eps, h, radius)] = snapshots
                stats.append({"eps": eps, "h": h, "radius": radius, "cached": from_cache, "path": path})
                print({"seed_eps": eps, "seed_h": h, "radius": radius, "cached": from_cache, "path": path})

    with open(os.path.join(OUT, "seed_cache_index_v22.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return bank, stats

def main():
    device, backend_name = select_device()
    print(f"Selected backend: {backend_name}")
    print(f"Selected device : {device}")

    t0 = time.time()
    seed_bank, seed_stats = warm_seed_bank(device)
    t1 = time.time()

    all_cases = build_param_grid()
    print(f"Total cases in grid: {len(all_cases)}")

    results, seen_tags = load_existing_results()
    print(f"Loaded existing results from scan CSV: {len(results)}")

    skipped_existing = 0
    newly_done = 0
    errors = 0

    for i, params in enumerate(all_cases, 1):
        tag = tag_from_params(params)
        print(f"[{i}/{len(all_cases)}] {tag}")

        if tag in seen_tags:
            print(f" -> skip: already in glider_scan_v22.csv")
            skipped_existing += 1
            continue

        if case_already_done(params):
            print(f"  -> skip: output files already exist")
            skipped_existing += 1
            append_progress({
                "tag": tag,
                "status": "skipped_existing_files",
                "time": time.time(),
            })
            continue

        case_t0 = time.time()
        try:
            res = run_case(device, backend_name, params, seed_bank, save_gif=False)
            results.append(res)
            seen_tags.add(tag)
            append_result_to_scan_csv(res)
            append_progress({
                "tag": tag,
                "status": "done",
                "time": time.time(),
                "runtime_sec": time.time() - case_t0,
                "label": res["label"],
                "speed": res["speed"],
            })
            newly_done += 1
            print(f"  -> done: {res}")

        except KeyboardInterrupt:
            append_progress({
                "tag": tag,
                "status": "interrupted",
                "time": time.time(),
            })
            raise

        except Exception as e:
            errors += 1
            append_progress({
                "tag": tag,
                "status": "error",
                "time": time.time(),
                "error": str(e),
            })
            print(f"  -> error: {e}")

    results_sorted = sorted(results, key=lambda r: (r["label"] != "glider", -r["speed"]))
    best = [r for r in results_sorted if r["label"] in ("glider", "complex")][:SAVE_TOP_GIFS]

    for r in best:
        params = CaseParams(
            r["eps"], r["h"], int(r["dist"]), r["phase_shift"], int(r["dy"]),
            r["kick"], r["kick_type"], r["amp_asym"], int(r["rad_asym"])
        )
        paths = case_output_paths(params)
        if not os.path.exists(paths["preview_gif"]):
            rerun = run_case(device, backend_name, params, seed_bank, save_gif=True)
            print("Saved preview:", rerun)

    candidates = [r for r in results if r["label"] in ("glider", "complex")]
    with open(CANDIDATES_JSON, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    summary = {
        "backend": backend_name,
        "device": str(device),
        "total_cases_in_grid": len(all_cases),
        "total_results_available": len(results),
        "newly_done_this_run": newly_done,
        "skipped_existing": skipped_existing,
        "errors_this_run": errors,
        "n_glider": sum(r["label"] == "glider" for r in results),
        "n_pinned": sum(r["label"] == "pinned" for r in results),
        "n_pinned_breather": sum(r["label"] == "pinned_breather" for r in results),
        "n_decay": sum(r["label"] == "decay" for r in results),
        "n_complex": sum(r["label"] == "complex" for r in results),
        "seed_build_sec": t1 - t0,
        "scan_sec": time.time() - t1,
        "elapsed_sec": time.time() - t0,
        "n_seed_entries": len(seed_stats),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_velocity_map(results)

    print("Done.")
    print(summary)

if __name__ == "__main__":
    main()