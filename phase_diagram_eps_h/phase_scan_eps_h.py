#!/usr/bin/env python3
"""
Minimal numerical scan to build a strictly data-based phase diagram
in the (eps, h) plane for the 2-field autosoliton PDE system:

    d_t phi   = M(phi) + h + psi
    d_tt psi + eps d_t psi = D_psi * Lap(psi) + Lap(phi)

with M(phi) = phi*sigma1(phi^2)*(1 - sigma2(phi^2)) - phi,
sigma_i(u) = 1/(1+exp(-k*(u-theta_i))), k=10, theta1=4, theta2=16, D_psi=0.

This reuses EXACTLY the spatial discretization (9-point isotropic
Laplacian, dx=1), the explicit time stepping scheme, and the radial-spot
initial condition from glider_serach_v22_windows.py, reimplemented in
NumPy (no torch available in this environment), so results stay
consistent with the paper.

No glider search, no extra scans -- only what is needed for the diagram.
"""

import csv
import numpy as np

# ---------------- model parameters (from the script / paper) ----------------
KAPPA = 10.0
THETA1 = 4.0
THETA2 = 16.0
PHI_LO = 0.533                    # low-amplitude background (stationary analysis)
PHI_HI = 3.9766096853487105      # spot amplitude
DW = 0.0                         # D_psi

# ---------------- discretization (reuse code conventions) -------------------
DX = 1.0
DT = 0.005                       # requested time step
T_FINAL = 250.0                  # final integration time (~200-300)
N_GRID = 64                      # moderate grid N=56-64
SEED_RADIUS = 8                  # radial-spot radius (same as code)
MONITOR_EVERY = 10               # sample center value every 10 steps (as in code)
DISCARD_FRAC = 0.55              # discard first ~55% of time series (transient)

# ---------------- classification thresholds (from task spec) ----------------
DELTA_PHI = 0.3                  # collapse cutoff: phi_max <= PHI_LO + DELTA_PHI
A_THR = 0.15                     # breathing amplitude threshold
UNSTABLE_ABS = 100.0             # |phi| above this => explicit-scheme blow-up

# ---------------- parameter grid -------------------------------------------
EPS_LIST = [1.5, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 6.0]
H_LIST = [0.515, 0.530, 0.533, 0.540, 0.545, 0.555]


def sigma1(u):
    return 1.0 / (1.0 + np.exp(-KAPPA * (u - THETA1)))


def sigma2(u):
    return 1.0 / (1.0 + np.exp(-KAPPA * (u - THETA2)))


def M(phi):
    return phi * sigma1(phi * phi) * (1.0 - sigma2(phi * phi)) - phi


def laplacian9(u, dx):
    """9-point isotropic Laplacian with periodic BCs (np.roll), exactly as in the code."""
    up = np.roll(u, 1, axis=0)
    dn = np.roll(u, -1, axis=0)
    lf = np.roll(u, 1, axis=1)
    rt = np.roll(u, -1, axis=1)
    ul = np.roll(up, 1, axis=1)
    ur = np.roll(up, -1, axis=1)
    dl = np.roll(dn, 1, axis=1)
    dr = np.roll(dn, -1, axis=1)
    return ((4.0 * (up + dn + lf + rt) + (ul + ur + dl + dr) - 20.0 * u) / 6.0) / (dx * dx)


def disk_mask(nx, ny, cx, cy, radius):
    x = np.arange(nx).reshape(-1, 1)
    y = np.arange(ny).reshape(1, -1)
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2


def build_seed(nx, ny):
    """Localized radial spot of amplitude PHI_HI on background PHI_LO; psi=v=0."""
    phi = np.full((nx, ny), PHI_LO, dtype=np.float64)
    psi = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)
    cx, cy = nx // 2, ny // 2
    mask = disk_mask(nx, ny, cx, cy, SEED_RADIUS)
    phi[mask] = PHI_HI
    return phi, psi, v


def run_case(eps, h):
    nx = ny = N_GRID
    cx, cy = nx // 2, ny // 2
    phi, psi, v = build_seed(nx, ny)

    steps = int(T_FINAL / DT)
    times = []
    center_series = []

    for n in range(steps + 1):
        if n % MONITOR_EVERY == 0:
            times.append(n * DT)
            center_series.append(float(phi[cx, cy]))

        lap_phi = laplacian9(phi, DX)
        lap_psi = laplacian9(psi, DX)
        # identical update order to the original script
        phi = phi + DT * (M(phi) + h + psi)
        v = v + DT * (lap_phi + DW * lap_psi - eps * v)
        psi = psi + DT * v

        # guard against numerical blow-up
        if not np.isfinite(phi[cx, cy]):
            break

    times = np.array(times)
    series = np.array(center_series)

    # discard transient: keep last (1 - DISCARD_FRAC) of the window
    keep_from = int(len(series) * DISCARD_FRAC)
    rec = series[keep_from:]
    rec = rec[np.isfinite(rec)]
    if rec.size == 0:
        rec = np.array([PHI_LO])

    phi_max = float(np.max(rec))
    phi_min = float(np.min(rec))
    phi_mean = float(np.mean(rec))
    A = phi_max - phi_min

    # ---------------- numerical-stability guard ----------------
    # The explicit scheme becomes unstable at very low eps for this dt;
    # such runs grow without bound (|phi| >> physical scale). Flag them
    # so they are not mislabeled as physical breathing.
    unstable = (not np.all(np.isfinite(series))) or (np.max(np.abs(rec)) > UNSTABLE_ABS)

    # ---------------- classification ----------------
    if unstable:
        regime = -1  # numerically unstable; excluded from physical diagram
    # Collapse (0): no localized soliton
    elif phi_max <= PHI_LO + DELTA_PHI:
        regime = 0
    # Breathing (2): persistent breathing
    elif A > A_THR:
        regime = 2
    # Stationary (1): localized but non-breathing
    else:
        regime = 1

    return {
        "eps": eps,
        "h": h,
        "A": A,
        "phi_max": phi_max,
        "phi_min": phi_min,
        "phi_mean": phi_mean,
        "regime_code": regime,
    }


def main():
    rows = []
    total = len(EPS_LIST) * len(H_LIST)
    i = 0
    for eps in EPS_LIST:
        for h in H_LIST:
            i += 1
            res = run_case(eps, h)
            rows.append(res)
            print(f"[{i}/{total}] eps={eps:.3f} h={h:.3f} -> "
                  f"A={res['A']:.4f} phi_max={res['phi_max']:.4f} "
                  f"regime={res['regime_code']}", flush=True)

    out_csv = "/home/user/workspace/phase_map_eps_h_data.csv"
    fieldnames = ["eps", "h", "A", "phi_max", "phi_min", "phi_mean", "regime_code"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
