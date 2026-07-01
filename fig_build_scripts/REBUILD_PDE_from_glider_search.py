#!/usr/bin/env python3
"""
rebuild_figures_from_glider_search.py
================================================================================
Main COMPUTE + FIGURE script for the breathing bifurcation study A(eps) of the
2-field autosoliton model (fixed h = 0.533). Requires PyTorch.

This is the only script of the three that actually integrates the PDE; the two
companion scripts (patch_bifurcation_rms_amplitude.py,
reprocess_bifurcation_with_collapse_band-3.py) only post-process its output.

MODEL / NUMERICS
----------------
Same model as glider_serach_v22 (kappa=10, theta1=4, theta2=16, D_psi=0):
    M(phi) = phi*sigma1(phi^2)*(1 - sigma2(phi^2)) - phi,
    d_t  phi = M(phi) + h + psi,
    d_tt psi + eps d_t psi = D_psi*Lap(psi) + Lap(phi).
Explicit stepping with v = d_t psi; 9-point isotropic Laplacian, periodic BCs.
Grid NX=NY=64, dx=1, dt=0.001, T_TOTAL=400. Backend: CUDA if available else CPU.

WHAT IT DOES
------------
- Sweeps eps in EPS_LIST = [2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 6.0]
  at fixed h = 0.533.
- Initial condition: a PAIR of spots (build_pair_initial_state) with a Gaussian
  velocity kick (default dist=18, kick=0.06 on the v-field).
- Integrates each case, monitoring every 10 steps: center-of-mass (x, y, mass)
  of the region phi > PEAK_THRESHOLD and the center value phi_center = phi(cx,cy).
  Stops early on NaN/Inf.
- On the tail t >= TAIL_T0 (=240) computes amplitude A = 0.5*(max-min) of
  phi_center and the breathing period via scipy.signal.find_peaks
  (fallback: local-maxima detection). Classifies each eps as
  'breathing' (A >= BREATHING_MIN_AMP=0.15 and finite period) or
  'collapse_or_nonbreathing'.
- For eps = 3.0 also saves a space-time slice, a phi_c(t) time series figure,
  and a (phi_c, d/dt phi_c) phase-portrait figure.

OUTPUTS (in output/)
--------------------
- track_<tag>.csv                : per-eps time series t, x, y, mass, phi_center.
- space_time_<tag>.csv/.png      : space-time diagram (eps = 3.0 only).
- bifurcation_raw.csv            : eps, A, status for every eps.
- bifurcation_clean.csv          : same, filtered to eps >= 2.0.
- period_vs_eps.csv              : eps, T for breathing cases.
- fig_bifurcation_A_eps_fixed.*  : bifurcation diagram A(eps) (red=breathing,
                                   blue=collapse).
- fig_period_T_eps_fixed.*       : breathing period T(eps).
- phi_center_timeseries_fixed.*, phase_portrait_eps3_fixed.* : eps = 3.0 detail.
- rebuild_summary.json           : run parameters and list of generated tracks.
================================================================================
"""
import os
import csv
import math
import json
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception as e:
    raise RuntimeError('This script requires PyTorch.') from e

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

KAPPA = 10.0
THETA1 = 4.0
THETA2 = 16.0
PHI_LO = 0.533
PHI_HI = 3.9766096853487105
DW = 0.0

NX = 64
NY = 64
DX = 1.0
DT = 0.001
T_TOTAL = 400.0
RADIUS = 8
TAIL_T0 = 240.0
BREATHING_MIN_AMP = 0.15
PEAK_THRESHOLD = 1.5

EPS_LIST = [2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 6.0]
H_FIXED = 0.533

OUT = 'output'
os.makedirs(OUT, exist_ok=True)

@dataclass
class CaseParams:
    eps: float
    h: float = H_FIXED
    dist: int = 18
    phase_shift: float = 0.0
    dy: int = 0
    kick: float = 0.06
    kick_type: str = 'v'
    amp_asym: float = 0.0
    rad_asym: int = 0


def fmt_num(x, digits=3):
    s = f"{x:.{digits}f}".rstrip('0').rstrip('.')
    return s.replace('-', 'm').replace('.', 'p')


def case_tag(params: CaseParams):
    return (
        f"eps{fmt_num(params.eps,3)}"
        f"_h{fmt_num(params.h,3)}"
        f"_d{params.dist}"
        f"_ph{fmt_num(params.phase_shift,2)}"
        f"_dy{params.dy}"
        f"_k{fmt_num(params.kick,2)}"
        f"_{params.kick_type}"
        f"_aa{fmt_num(params.amp_asym,2)}"
        f"_ra{params.rad_asym}"
    )


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


def build_pair_initial_state(device, params: CaseParams):
    phi = torch.full((NX, NY), PHI_LO, dtype=torch.float32, device=device)
    psi = torch.zeros((NX, NY), dtype=torch.float32, device=device)
    v = torch.zeros((NX, NY), dtype=torch.float32, device=device)

    cx, cy = NX // 2, NY // 2
    sx = params.dist // 2
    dy = params.dy

    rL = RADIUS
    rR = RADIUS + params.rad_asym

    maskL = disk_mask(NX, NY, cx - sx, cy - dy // 2, rL, device)
    maskR = disk_mask(NX, NY, cx + sx, cy + dy // 2, rR, device)

    phi[maskL] = PHI_HI
    ampR = PHI_HI * (1.0 - params.amp_asym / max(PHI_HI, 1e-6))
    phi[maskR] = max(PHI_LO, ampR)

    if abs(params.kick) > 0:
        kick_profile = gaussian2d(NX, NY, cx, cy, 10.0, 4.0, device)
        if params.kick_type == 'v':
            v = v + params.kick * kick_profile
        else:
            psi = psi + params.kick * kick_profile

    return phi, psi, v


def center_of_mass(phi_np, threshold=PEAK_THRESHOLD):
    mask = phi_np > threshold
    if not np.any(mask):
        return np.nan, np.nan, 0.0
    w = np.where(mask, phi_np - threshold, 0.0)
    m = w.sum()
    if m <= 0:
        return np.nan, np.nan, 0.0
    ii, jj = np.indices(phi_np.shape)
    x = float((ii * w).sum() / m)
    y = float((jj * w).sum() / m)
    return x, y, float(m)


def save_track(path, times, xs, ys, masses, phi_center):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 'x', 'y', 'mass', 'phi_center'])
        for row in zip(times, xs, ys, masses, phi_center):
            w.writerow(row)


def estimate_period(t, y):
    t = np.asarray(t)
    y = np.asarray(y)
    if len(t) < 10:
        return np.nan, np.array([]), np.array([])
    y0 = y - np.mean(y)
    amp = float(np.max(y) - np.min(y))
    if amp < BREATHING_MIN_AMP:
        return np.nan, np.array([]), np.array([])
    if find_peaks is not None:
        dt = np.median(np.diff(t))
        peaks, _ = find_peaks(y0, prominence=max(0.03, 0.15 * np.std(y0)), distance=max(1, int(3.0 / dt)))
    else:
        peaks = np.where((y0[1:-1] > y0[:-2]) & (y0[1:-1] > y0[2:]))[0] + 1
    if len(peaks) < 2:
        return np.nan, t[peaks], y[peaks]
    periods = np.diff(t[peaks])
    return float(np.median(periods)), t[peaks], y[peaks]


def run_case(params: CaseParams, save_space_time=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phi, psi, v = build_pair_initial_state(device, params)

    steps = int(T_TOTAL / DT)
    monitor_every = 10

    times, xs, ys, masses, phi_center = [], [], [], [], []
    row_lines, row_times = [], []

    center_x, center_y = NX // 2, NY // 2

    for n in range(steps + 1):
        if n % monitor_every == 0:
            t = n * DT
            phi_np = phi.detach().cpu().numpy()
            x, y, mass = center_of_mass(phi_np)
            times.append(t)
            xs.append(x)
            ys.append(y)
            masses.append(mass)
            phi_center.append(float(phi_np[center_x, center_y]))
            if save_space_time:
                row_lines.append(phi_np[center_x, :].copy())
                row_times.append(t)

        lap_phi = laplacian9(phi, DX)
        lap_psi = laplacian9(psi, DX)
        phi = phi + DT * (M(phi) + params.h + psi)
        v = v + DT * (lap_phi + DW * lap_psi - params.eps * v)
        psi = psi + DT * v

        if torch.isnan(phi).any() or torch.isinf(phi).any():
            break

    tag = case_tag(params)
    track_path = os.path.join(OUT, f'track_{tag}.csv')
    save_track(track_path, times, xs, ys, masses, phi_center)

    if save_space_time and row_lines:
        arr = np.array(row_lines)
        csv_path = os.path.join(OUT, f'space_time_{tag}.csv')
        png_path = os.path.join(OUT, f'space_time_{tag}.png')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t'] + [f'y{j}' for j in range(arr.shape[1])])
            for tt, row in zip(row_times, arr):
                w.writerow([tt] + row.tolist())
        plt.figure(figsize=(8, 5))
        plt.imshow(arr, aspect='auto', origin='lower', cmap='magma', extent=[0, arr.shape[1]-1, row_times[0], row_times[-1]])
        plt.colorbar(label='phi(center row, y, t)')
        plt.xlabel('y index')
        plt.ylabel('t')
        plt.title('Space-time plot')
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()

    return {
        'tag': tag,
        'track_path': track_path,
        'times': np.array(times),
        'xs': np.array(xs),
        'ys': np.array(ys),
        'masses': np.array(masses),
        'phi_center': np.array(phi_center),
    }


def make_timeseries_figure(times, signal):
    plt.figure(figsize=(6.6, 3.8))
    plt.plot(times, signal, color='black', lw=1.2)
    plt.axhline(PHI_LO, color='gray', ls='--', lw=0.9)
    plt.axhline(PHI_HI, color='gray', ls='--', lw=0.9)
    plt.xlabel('t')
    plt.ylabel(r'$\phi_c(t)$')
    plt.title('Central amplitude time series')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'phi_center_timeseries_fixed.png'), dpi=220)
    plt.savefig(os.path.join(OUT, 'phi_center_timeseries_fixed.pdf'))
    plt.close()


def make_phase_figure(times, signal):
    dt = np.median(np.diff(times))
    dphi = np.gradient(signal, dt)
    mask = times >= 0.55
    plt.figure(figsize=(5.2, 5.0))
    plt.plot(signal[mask], dphi[mask], color='black', lw=1.0)
    idx0 = np.argmax(mask)
    plt.scatter([signal[idx0]], [dphi[idx0]], s=28, color='black', zorder=3)
    plt.xlabel(r'$\phi_c$')
    plt.ylabel(r'$\dot{\phi}_c$')
    plt.title('Phase portrait')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'phase_portrait_eps3_fixed.png'), dpi=220)
    plt.savefig(os.path.join(OUT, 'phase_portrait_eps3_fixed.pdf'))
    plt.close()


def main():
    params_list = [CaseParams(eps=e) for e in EPS_LIST]
    results = []
    for p in params_list:
        results.append(run_case(p, save_space_time=(abs(p.eps - 3.0) < 1e-12)))

    bif_rows = []
    period_rows = []

    for res, p in zip(results, params_list):
        times = res['times']
        sig = res['phi_center']
        tail = times >= TAIL_T0
        if not np.any(tail):
            continue
        t_tail = times[tail]
        s_tail = sig[tail]
        amp = 0.5 * (float(np.max(s_tail)) - float(np.min(s_tail)))
        period, peak_t, peak_y = estimate_period(t_tail, s_tail)
        breathing = (amp >= BREATHING_MIN_AMP) and np.isfinite(period)
        status = 'breathing' if breathing else 'collapse_or_nonbreathing'
        bif_rows.append({'eps': p.eps, 'A': amp, 'status': status})
        if breathing:
            period_rows.append({'eps': p.eps, 'T': period})
        if abs(p.eps - 3.0) < 1e-12:
            make_timeseries_figure(times, sig)
            make_phase_figure(times, sig)

    raw_path = os.path.join(OUT, 'bifurcation_raw.csv')
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['eps', 'A', 'status'])
        w.writeheader()
        w.writerows(bif_rows)

    clean_bif = [r for r in bif_rows if r['eps'] >= 2.0]
    clean_bif_path = os.path.join(OUT, 'bifurcation_clean.csv')
    with open(clean_bif_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['eps', 'A', 'status'])
        w.writeheader()
        w.writerows(clean_bif)

    period_path = os.path.join(OUT, 'period_vs_eps.csv')
    with open(period_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['eps', 'T'])
        w.writeheader()
        w.writerows(period_rows)

    x = np.array([r['eps'] for r in clean_bif], dtype=float)
    y = np.array([r['A'] for r in clean_bif], dtype=float)
    status = [r['status'] for r in clean_bif]
    colors = ['tab:red' if s == 'breathing' else 'tab:blue' for s in status]

    plt.figure(figsize=(6.0, 4.0))
    plt.scatter(x, y, c=colors, s=42, zorder=3)
    breath = np.array([r['status'] == 'breathing' for r in clean_bif])
    if breath.sum() >= 2:
        xb = x[breath]
        yb = y[breath]
        order = np.argsort(xb)
        plt.plot(xb[order], yb[order], color='tab:red', lw=1.2, alpha=0.8)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('A')
    plt.title('Bifurcation diagram')
    plt.ylim(0, 4.5)
    plt.xlim(2.0, max(x.max(), 6.0))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_bifurcation_A_eps_fixed.png'), dpi=220)
    plt.savefig(os.path.join(OUT, 'fig_bifurcation_A_eps_fixed.pdf'))
    plt.close()

    if period_rows:
        xp = np.array([r['eps'] for r in period_rows], dtype=float)
        yp = np.array([r['T'] for r in period_rows], dtype=float)
        order = np.argsort(xp)
        xp = xp[order]
        yp = yp[order]
        plt.figure(figsize=(6.0, 4.0))
        plt.plot(xp, yp, color='black', lw=1.2)
        plt.scatter(xp, yp, color='black', s=38, zorder=3)
        plt.xlabel(r'$\varepsilon$')
        plt.ylabel('T')
        plt.title('Breathing period')
        plt.xlim(2.0, max(xp))
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'fig_period_T_eps_fixed.png'), dpi=220)
        plt.savefig(os.path.join(OUT, 'fig_period_T_eps_fixed.pdf'))
        plt.close()

    summary = {
        'eps_list': EPS_LIST,
        'h_fixed': H_FIXED,
        'dt': DT,
        'T_total': T_TOTAL,
        'tail_interval': [TAIL_T0, T_TOTAL],
        'nx': NX,
        'ny': NY,
        'radius': RADIUS,
        'generated_tracks': [r['track_path'] for r in results],
    }
    with open(os.path.join(OUT, 'rebuild_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Done.')

if __name__ == '__main__':
    main()
