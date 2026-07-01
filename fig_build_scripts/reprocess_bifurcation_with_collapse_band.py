#!/usr/bin/env python3
"""
reprocess_bifurcation_with_collapse_band-3.py
================================================================================
FINAL FIGURE-ONLY step: redraw the breathing bifurcation diagram A(eps) with a
gray 'collapse' band, from an already-aggregated CSV. Does NOT integrate the PDE
and does NOT read any track_*.csv time series.

INPUT
-----
Reads bifurcation_reprocessed.csv (looked up in output/ then cwd), expected to
contain per-eps columns:
    eps, status, amp_cycle_median, period_cycle_median, n_cycles, peak_ratio
Here the breathing amplitude is the per-cycle MEDIAN amplitude
(amp_cycle_median), i.e. a cleaner estimate than raw max-min or RMS.

WHAT IT DOES
------------
- Loads and sorts the rows by eps (load_rows), coercing numeric fields and
  handling empty/NaN entries.
- Splits into 'breathing' (finite amp_cycle_median) and collapse/nonbreathing.
- Infers the boundary eps_c (infer_boundary): midpoint between the largest
  breathing eps and the smallest collapse eps.
- Plots the red breathing branch amp_cycle_median(eps), marks collapse points
  as gray crosses at A = 0, and shades the region eps > eps_c as a gray
  'collapse' band with a dashed eps_c line.

OUTPUTS (in output/)
--------------------
- fig_bifurcation_A_eps_collapse_band.png/.pdf : the final bifurcation figure.
- fig_bifurcation_A_eps_collapse_band.json     : source CSV, boundary_eps,
  breathing/collapse point counts, and a descriptive note.
================================================================================
"""
import os
import re
import csv
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt

OUT = 'output'
INPUT_CSV_CANDIDATES = [
    os.path.join(OUT, 'bifurcation_reprocessed.csv'),
    'bifurcation_reprocessed.csv',
]


def load_rows():
    path = None
    for p in INPUT_CSV_CANDIDATES:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError('bifurcation_reprocessed.csv not found in output/ or current directory')

    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            out = dict(row)
            out['eps'] = float(row['eps'])
            out['amp_cycle_median'] = float(row['amp_cycle_median']) if row['amp_cycle_median'] not in ('', 'nan', 'NaN') else math.nan
            out['period_cycle_median'] = float(row['period_cycle_median']) if row['period_cycle_median'] not in ('', 'nan', 'NaN') else math.nan
            out['n_cycles'] = int(row['n_cycles'])
            out['peak_ratio'] = float(row['peak_ratio']) if row['peak_ratio'] not in ('', 'nan', 'NaN') else math.nan
            rows.append(out)
    rows.sort(key=lambda z: z['eps'])
    return rows, path


def infer_boundary(rows):
    breathing = [r for r in rows if r['status'] == 'breathing' and np.isfinite(r['amp_cycle_median'])]
    collapse = [r for r in rows if r['status'] != 'breathing']
    if breathing and collapse:
        return 0.5 * (max(r['eps'] for r in breathing) + min(r['eps'] for r in collapse))
    if breathing:
        return max(r['eps'] for r in breathing)
    return None


def main():
    rows, src = load_rows()
    breathing = [r for r in rows if r['status'] == 'breathing' and np.isfinite(r['amp_cycle_median'])]
    collapse = [r for r in rows if r['status'] != 'breathing']
    boundary = infer_boundary(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    if boundary is not None:
        ax.axvspan(boundary, 6.0, color='0.92', zorder=0)
        ax.axvline(boundary, color='0.45', lw=1.0, ls='--', zorder=1)
        ymax_hint = max([r['amp_cycle_median'] for r in breathing], default=1.0)
        ax.text(boundary + 0.12, 0.88 * ymax_hint, 'collapse', color='0.35', fontsize=10,
                ha='left', va='center')
        ax.text(boundary - 0.04, 0.12 * ymax_hint, r'$\varepsilon_c$', color='0.35', fontsize=10,
                ha='right', va='bottom')

    if breathing:
        xb = np.array([r['eps'] for r in breathing], dtype=float)
        yb = np.array([r['amp_cycle_median'] for r in breathing], dtype=float)
        order = np.argsort(xb)
        xb, yb = xb[order], yb[order]
        ax.plot(xb, yb, color='tab:red', lw=1.4, zorder=3)
        ax.scatter(xb, yb, color='tab:red', s=44, zorder=4)

    if collapse:
        xc = np.array([r['eps'] for r in collapse], dtype=float)
        ax.scatter(xc, np.zeros_like(xc), marker='x', color='0.35', s=34, linewidths=1.0, zorder=5)

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('A')
    ax.set_title('Bifurcation diagram')
    ax.set_xlim(2.0, 6.0)
    ymax = max([r['amp_cycle_median'] for r in breathing], default=1.0)
    ax.set_ylim(0, 1.08 * ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png = os.path.join(OUT, 'fig_bifurcation_A_eps_collapse_band.png')
    pdf = os.path.join(OUT, 'fig_bifurcation_A_eps_collapse_band.pdf')
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    meta = {
        'source_csv': src,
        'boundary_eps': boundary,
        'breathing_points': len(breathing),
        'collapse_points': len(collapse),
        'note': 'Breathing amplitudes plotted in red; collapse/nonbreathing region shown as gray band to the right of inferred boundary.'
    }
    with open(os.path.join(OUT, 'fig_bifurcation_A_eps_collapse_band.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
