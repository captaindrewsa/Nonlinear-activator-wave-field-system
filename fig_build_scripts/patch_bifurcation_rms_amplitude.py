#!/usr/bin/env python3
"""
patch_bifurcation_rms_amplitude.py
================================================================================
POST-PROCESSING patch: recompute the breathing bifurcation diagram using an
RMS amplitude instead of the (max - min) amplitude. Does NOT integrate the PDE.

It reads the track_*.csv time series already produced by
rebuild_figures_from_glider-2.py (or the original glider search).

WHAT IT DOES
------------
- Globs track_eps*_h0p533_*.csv (H_TARGET = 'h0p533') in output/ and cwd;
  parses eps from each filename (parse_eps, 'm'->'-', 'p'->'.').
- Loads the phi_center(t) series and keeps the tail t >= TAIL_T0 (=240).
- Amplitude measure: A_rms = sqrt( < (phi_c - <phi_c>)^2 > ) on the tail
  (rms_amplitude), which is more robust to outliers than max-min.
- Classifies each case (classify_tail): tail span < 0.05 ->
  'collapse_or_nonbreathing', else 'breathing' (A_rms reported only if breathing).
- Estimates the breathing/collapse boundary eps_c (infer_boundary): midpoint
  between the largest breathing eps and the smallest collapse eps.

OUTPUTS (in output/)
--------------------
- bifurcation_rms.csv            : eps, A_rms, status, tail_mean/min/max, file.
- fig_bifurcation_A_rms_eps.png/.pdf : A_rms(eps) with red breathing branch,
  gray 'collapse' band and dashed eps_c line to the right of the boundary.
- bifurcation_rms_meta.json      : tail interval, amplitude definition, counts,
  inferred boundary_eps.
================================================================================
"""
import os
import re
import csv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

OUT = 'output'
TAIL_T0 = 240.0
H_TARGET = 'h0p533'


def parse_eps(path):
    m = re.search(r'track_eps([0-9mp]+)_h', os.path.basename(path))
    if not m:
        return None
    return float(m.group(1).replace('m', '-').replace('p', '.'))


def load_track(path):
    t, phi = [], []
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row['t']))
            phi.append(float(row['phi_center']))
    return np.array(t), np.array(phi)


def rms_amplitude(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    return float(np.sqrt(np.mean(x * x)))


def classify_tail(phi_tail):
    span = float(np.max(phi_tail) - np.min(phi_tail))
    if span < 0.05:
        return 'collapse_or_nonbreathing'
    return 'breathing'


def infer_boundary(rows):
    breathing = [r for r in rows if r['status'] == 'breathing']
    collapse = [r for r in rows if r['status'] != 'breathing']
    if breathing and collapse:
        return 0.5 * (max(r['eps'] for r in breathing) + min(r['eps'] for r in collapse))
    if breathing:
        return max(r['eps'] for r in breathing)
    return None


def main():
    files = sorted(glob.glob(os.path.join(OUT, f'track_eps*_{H_TARGET}_*.csv')) + glob.glob(f'track_eps*_{H_TARGET}_*.csv'))
    rows = []
    for path in files:
        eps = parse_eps(path)
        if eps is None:
            continue
        t, phi = load_track(path)
        mask = t >= TAIL_T0
        if not np.any(mask):
            continue
        tt = t[mask]
        yy = phi[mask]
        status = classify_tail(yy)
        rec = {
            'eps': eps,
            'A_rms': rms_amplitude(yy) if status == 'breathing' else 0.0,
            'status': status,
            'tail_mean': float(np.mean(yy)),
            'tail_min': float(np.min(yy)),
            'tail_max': float(np.max(yy)),
            'file': os.path.basename(path),
        }
        rows.append(rec)

    rows.sort(key=lambda r: r['eps'])

    csv_out = os.path.join(OUT, 'bifurcation_rms.csv')
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['eps', 'A_rms', 'status', 'tail_mean', 'tail_min', 'tail_max', 'file'])
        w.writeheader()
        w.writerows(rows)

    breathing = [r for r in rows if r['status'] == 'breathing']
    collapse = [r for r in rows if r['status'] != 'breathing']
    boundary = infer_boundary(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if boundary is not None:
        ax.axvspan(boundary, 6.0, color='0.92', zorder=0)
        ax.axvline(boundary, color='0.45', lw=1.0, ls='--', zorder=1)
        ymax_hint = max([r['A_rms'] for r in breathing], default=1.0)
        ax.text(boundary + 0.10, 0.88 * ymax_hint, 'collapse', color='0.35', fontsize=10, ha='left', va='center')
        ax.text(boundary - 0.04, 0.06 * ymax_hint, r'$\varepsilon_c$', color='0.35', fontsize=10, ha='right', va='bottom')

    if breathing:
        xb = np.array([r['eps'] for r in breathing], dtype=float)
        yb = np.array([r['A_rms'] for r in breathing], dtype=float)
        order = np.argsort(xb)
        xb, yb = xb[order], yb[order]
        ax.plot(xb, yb, color='tab:red', lw=1.5, zorder=3)
        ax.scatter(xb, yb, color='tab:red', s=46, zorder=4)

    if collapse:
        xc = np.array([r['eps'] for r in collapse], dtype=float)
        ax.scatter(xc, np.zeros_like(xc), marker='x', color='0.35', s=34, linewidths=1.0, zorder=5)

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$A_{\mathrm{rms}}$')
    ax.set_title('Bifurcation diagram')
    ax.set_xlim(2.0, 6.0)
    ymax = max([r['A_rms'] for r in breathing], default=1.0)
    ax.set_ylim(0.0, 1.10 * ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png = os.path.join(OUT, 'fig_bifurcation_A_rms_eps.png')
    pdf = os.path.join(OUT, 'fig_bifurcation_A_rms_eps.pdf')
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    meta = {
        'tail_t0': TAIL_T0,
        'amplitude_definition': 'A_rms = sqrt(<(phi_c - <phi_c>)^2>) on tail t in [240,400]',
        'tracks_found': len(files),
        'rows_written': len(rows),
        'breathing_points': len(breathing),
        'collapse_points': len(collapse),
        'boundary_eps': boundary,
    }
    with open(os.path.join(OUT, 'bifurcation_rms_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
