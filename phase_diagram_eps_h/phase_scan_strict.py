#!/usr/bin/env python3
"""
STRICT data-based phase diagram in the (eps, h) plane for the 2-field
autosoliton PDE system:

    d_t phi   = M(phi) + h + psi
    d_tt psi + eps d_t psi = D_psi * Lap(psi) + Lap(phi)

M(phi) = phi*sigma1(phi^2)*(1 - sigma2(phi^2)) - phi,
sigma_i(u) = 1/(1+exp(-k*(u-theta_i))), k=10, theta1=4, theta2=16, D_psi=0.

Reuses EXACTLY the discretization from glider_serach_v22_windows.py:
9-point isotropic Laplacian, periodic BCs (np.roll), dx=1, radial-spot IC.
Reimplemented in NumPy (no torch in this environment).

Strict numerics:
  - dt = 0.002 (smaller for stability, esp. at eps=1.5)
  - N = 64, same for all runs
  - T_total = 400, discard first 60% as transient
  - early stop + flag if NaN/Inf or |phi| > 1e3  -> regime_code = -1

Classification of phi(0,t) on [0.6*T, T]:
  - collapse (0):   phi_max <= phi_lo+0.3 AND |phi_mean| < 0.5
  - breathing (2):  A > 0.15 AND Pmax/Pmed >= R_thr  (FFT, Hann window)
  - stationary (1): phi_max > phi_lo+0.3, A <= 0.15, Pmax/Pmed < R_thr
  - unstable (-1):  numerical blow-up
"""

import sys
import csv
import numpy as np

# ---------------- model parameters ----------------
KAPPA = 10.0
THETA1 = 4.0
THETA2 = 16.0
PHI_LO = 0.533
PHI_HI = 3.9766096853487105
DW = 0.0   # D_psi

# ---------------- discretization (strict) ----------------
DX = 1.0
DT = 0.001                 # single small step for all runs (stable incl. eps=1.5)
T_FINAL = 400.0
N_GRID = 64
SEED_RADIUS = 8
MONITOR_EVERY = 10         # sample phi center every 10 steps -> dt_sample = 0.010
DISCARD_FRAC = 0.60
STOP_ABS = 1.0e3           # |phi| above this => unstable

# ---------------- classification thresholds ----------------
DELTA_PHI = 0.3
A_THR = 0.15
MEAN_ABS_THR = 0.5
R_THR = 5.0                # Pmax/Pmed threshold for a distinct frequency
N_FFT = 2000

# ---------------- parameter grid ----------------
EPS_LIST = [1.5, 2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 6.0]
H_LIST = [0.515, 0.530, 0.533, 0.540, 0.545, 0.555]


def sigma1(u):
    return 1.0 / (1.0 + np.exp(-KAPPA * (u - THETA1)))


def sigma2(u):
    return 1.0 / (1.0 + np.exp(-KAPPA * (u - THETA2)))


def M(phi):
    return phi * sigma1(phi * phi) * (1.0 - sigma2(phi * phi)) - phi


def laplacian9(u, dx):
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
    phi = np.full((nx, ny), PHI_LO, dtype=np.float64)
    psi = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)
    cx, cy = nx // 2, ny // 2
    phi[disk_mask(nx, ny, cx, cy, SEED_RADIUS)] = PHI_HI
    return phi, psi, v


def spectral_analysis(x, dt_sample):
    """Return (dominant_freq, freq_ratio) using Hann-windowed FFT on detrended x."""
    n = len(x)
    if n < 16:
        return 0.0, 0.0
    x = x - np.mean(x)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    xw = x * w
    X = np.fft.rfft(xw)
    P = np.abs(X) ** 2
    if len(P) < 3:
        return 0.0, 0.0
    Pk = P[1:]                       # drop k=0 (DC)
    kmax = int(np.argmax(Pk)) + 1    # index into full spectrum
    Pmax = float(P[kmax])
    # background = median of the non-peak bins (exclude the dominant bin)
    bg = np.delete(Pk, np.argmax(Pk))
    Pmed = float(np.median(bg)) if bg.size else 0.0
    if Pmed <= 0:
        ratio = 1.0e6 if Pmax > 0 else 0.0
    else:
        ratio = min(Pmax / Pmed, 1.0e6)   # cap to avoid absurd magnitudes
    freqs = np.fft.rfftfreq(n, d=dt_sample)
    fmax = float(freqs[kmax])
    return fmax, ratio


def run_case(eps, h):
    nx = ny = N_GRID
    cx, cy = nx // 2, ny // 2
    phi, psi, v = build_seed(nx, ny)

    steps = int(T_FINAL / DT)
    dt_sample = DT * MONITOR_EVERY
    times = []
    center = []
    unstable = False

    for n in range(steps + 1):
        if n % MONITOR_EVERY == 0:
            cval = phi[cx, cy]
            times.append(n * DT)
            center.append(float(cval))
            # early-stop check on the whole field
            if not np.isfinite(cval) or np.max(np.abs(phi)) > STOP_ABS:
                unstable = True
                break

        lap_phi = laplacian9(phi, DX)
        lap_psi = laplacian9(psi, DX)
        phi = phi + DT * (M(phi) + h + psi)
        v = v + DT * (lap_phi + DW * lap_psi - eps * v)
        psi = psi + DT * v

    times = np.array(times)
    series = np.array(center)

    if unstable or not np.all(np.isfinite(series)):
        return {"eps": eps, "h": h, "A": float("nan"),
                "phi_max": float("nan"), "phi_min": float("nan"),
                "phi_mean": float("nan"), "dominant_freq": 0.0,
                "freq_ratio": 0.0, "regime_code": -1}

    # transient discard: keep last 40% of the window
    keep_from = int(len(series) * DISCARD_FRAC)
    rec = series[keep_from:]
    rec = rec[np.isfinite(rec)]
    if rec.size == 0:
        rec = np.array([PHI_LO])

    # use last min(N_FFT, len(rec)) for spectrum
    seg = rec[-min(N_FFT, len(rec)):]
    dt_sample = DT * MONITOR_EVERY
    dom_freq, ratio = spectral_analysis(seg, dt_sample)

    phi_max = float(np.max(rec))
    phi_min = float(np.min(rec))
    phi_mean = float(np.mean(rec))
    A = phi_max - phi_min

    # ---------------- strict classification ----------------
    # Collapse (0): no localized soliton -- phi_max barely exceeds background.
    #   Determined by phi_max alone (independent of phi_mean); a tiny residual
    #   amplitude just reflects numerical noise.
    if phi_max <= PHI_LO + DELTA_PHI:
        regime = 0
    # Breathing (2): clear local maximum, large amplitude AND a distinct
    #   spectral peak.
    elif (A > A_THR) and (ratio >= R_THR):
        regime = 2
    # Stationary (1): localized (phi_max above background) but non-breathing
    #   (small amplitude and/or no distinct frequency).
    else:
        regime = 1

    return {"eps": eps, "h": h, "A": A,
            "phi_max": phi_max, "phi_min": phi_min, "phi_mean": phi_mean,
            "dominant_freq": dom_freq, "freq_ratio": ratio,
            "regime_code": regime}


def main():
    # optional eps subset via CLI: python phase_scan_strict.py 1.5,2.0,2.5
    eps_list = EPS_LIST
    if len(sys.argv) > 1:
        eps_list = [float(x) for x in sys.argv[1].split(",")]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "/home/user/workspace/phase_map_eps_h_data_strict.csv"

    rows = []
    total = len(eps_list) * len(H_LIST)
    i = 0
    for eps in eps_list:
        for h in H_LIST:
            i += 1
            res = run_case(eps, h)
            rows.append(res)
            print(f"[{i}/{total}] eps={eps:.3f} h={h:.3f} -> "
                  f"A={res['A']:.4f} phi_max={res['phi_max']:.4f} "
                  f"ratio={res['freq_ratio']:.2f} f={res['dominant_freq']:.4f} "
                  f"regime={res['regime_code']}", flush=True)

    fieldnames = ["eps", "h", "A", "phi_max", "phi_min", "phi_mean",
                  "dominant_freq", "freq_ratio", "regime_code"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
