#!/usr/bin/env python3
"""
plot_bifurcation_reprocessed.py
================================================================================
FIGURE-ONLY step: draw the "reprocessed" and "clipped" bifurcation figures
A(eps) directly from the aggregated CSVs produced by
reprocess_bifurcation_cycle_median.py. Does NOT integrate the PDE and does NOT
read any track_*.csv time series.

This complements reprocess_bifurcation_with_collapse_band.py (which renders the
single 'collapse band' figure) by producing the two remaining figure variants
that share the same aggregated data:
    * fig_bifurcation_A_eps_reprocessed.pdf/.png  (from bifurcation_reprocessed.csv)
    * fig_bifurcation_A_eps_clipped.pdf/.png      (from bifurcation_clipped.csv)

INPUT
-----
Reads (looked up in output/ then cwd):
    bifurcation_reprocessed.csv
    bifurcation_clipped.csv          (optional; skipped if absent)
Each is expected to contain per-eps columns:
    eps, status, amp_cycle_median, period_cycle_median, n_cycles, peak_ratio

WHAT IT DOES
------------
- Loads and sorts rows by eps, coercing numeric fields and handling empty/NaN.
- Splits into 'breathing' (finite amp_cycle_median) and collapse/nonbreathing.
- Infers the boundary eps_c: midpoint between the largest breathing eps and the
  smallest collapse eps.
- For each input CSV, plots the red breathing branch amp_cycle_median(eps),
  marks collapse points as gray crosses at A = 0, and shades eps > eps_c as a
  gray 'collapse' band with a dashed eps_c line. Matches the visual style of
  reprocess_bifurcation_with_collapse_band.py.

OUTPUTS (in output/)
--------------------
- fig_bifurcation_A_eps_reprocessed.png/.pdf
- fig_bifurcation_A_eps_clipped.png/.pdf   (only if bifurcation_clipped.csv exists)
- plot_bifurcation_reprocessed.json        : per-figure source CSV, boundary_eps,
  breathing/collapse point counts.
================================================================================
"""
import os
import csv
import json
import math
import numpy as np
import matplotlib.pyplot as plt

OUT = 'output'

# (input_csv_basename, output_figure_basename, title)
FIGURE_SPECS = [
    ('bifurcation_reprocessed.csv', 'fig_bifurcation_A_eps_reprocessed',
     'Bifurcation diagram (per-cycle median)'),
    ('bifurcation_clipped.csv', 'fig_bifurcation_A_eps_clipped',
     'Bifurcation diagram (95th-percentile clipped)'),
]


def _resolve(basename):
    """Look up a CSV in output/ then the current directory; None if missing."""
    for p in (os.path.join(OUT, basename), basename):
        if os.path.exists(p):
            return p
    return None


def _num(row, key):
    v = row.get(key, '')
    if v in ('', 'nan', 'NaN', None):
        return math.nan
    return float(v)


def load_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            out = dict(row)
            out['eps'] = float(row['eps'])
            out['amp_cycle_median'] = _num(row, 'amp_cycle_median')
            out['period_cycle_median'] = _num(row, 'period_cycle_median')
            try:
                out['n_cycles'] = int(float(row.get('n_cycles', 0) or 0))
            except (TypeError, ValueError):
                out['n_cycles'] = 0
            out['peak_ratio'] = _num(row, 'peak_ratio')
            rows.append(out)
    rows.sort(key=lambda z: z['eps'])
    return rows


def infer_boundary(rows):
    breathing = [r for r in rows if r['status'] == 'breathing' and np.isfinite(r['amp_cycle_median'])]
    collapse = [r for r in rows if r['status'] != 'breathing']
    if breathing and collapse:
        return 0.5 * (max(r['eps'] for r in breathing) + min(r['eps'] for r in collapse))
    if breathing:
        return max(r['eps'] for r in breathing)
    return None


def render(rows, out_base, title):
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
    ax.set_title(title)
    ax.set_xlim(2.0, 6.0)
    ymax = max([r['amp_cycle_median'] for r in breathing], default=1.0)
    ax.set_ylim(0, 1.08 * ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png = os.path.join(OUT, out_base + '.png')
    pdf = os.path.join(OUT, out_base + '.pdf')
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    return {
        'boundary_eps': boundary,
        'breathing_points': len(breathing),
        'collapse_points': len(collapse),
        'png': png,
        'pdf': pdf,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    made_any = False
    for csv_name, out_base, title in FIGURE_SPECS:
        path = _resolve(csv_name)
        if path is None:
            report[csv_name] = {'skipped': True, 'reason': 'input CSV not found'}
            print('SKIP %s: not found in output/ or cwd' % csv_name)
            continue
        rows = load_rows(path)
        info = render(rows, out_base, title)
        info['source_csv'] = path
        report[csv_name] = info
        made_any = True
        print('OK   %s -> %s(.png/.pdf)' % (csv_name, out_base))

    if not made_any:
        raise FileNotFoundError(
            'No input CSV found. Run reprocess_bifurcation_cycle_median.py first '
            'to produce bifurcation_reprocessed.csv / bifurcation_clipped.csv.'
        )

    with open(os.path.join(OUT, 'plot_bifurcation_reprocessed.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
