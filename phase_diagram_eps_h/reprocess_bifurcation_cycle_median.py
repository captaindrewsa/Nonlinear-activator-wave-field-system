#!/usr/bin/env python3
"""
reprocess_bifurcation_cycle_median.py
================================================================================
INTERMEDIATE AGGREGATION step of the A(eps) bifurcation pipeline. Does NOT
integrate the PDE -- it reads the track_*.csv time series produced by
REBUILD_PDE_from_glider_search.py and reduces each one to a single, robust
"per-cycle median" breathing amplitude and period.

It produces the aggregated CSV that reprocess_bifurcation_with_collapse_band.py
consumes, plus a "clipped" variant with outlier trimming.

PIPELINE POSITION
-----------------
    REBUILD_PDE_from_glider_search.py      (compute; writes track_*.csv)
        -> reprocess_bifurcation_cycle_median.py   (THIS script: aggregate)
            -> reprocess_bifurcation_with_collapse_band.py  (final figure)

WHAT IT DOES
------------
- Globs track_eps*_h0p533_*.csv (H_TARGET = 'h0p533') in output/ and cwd and
  parses eps from each filename (parse_eps, 'm'->'-', 'p'->'.'), matching the
  conventions of patch_bifurcation_rms_amplitude.py.
- Loads phi_center(t) and keeps the tail t >= TAIL_T0 (=240) to discard the
  transient.
- Detects oscillation peaks and troughs on the tail with scipy.signal.find_peaks
  (fallback: a simple 3-point local-extremum scan if SciPy is unavailable).
  - Per-cycle amplitude: for each successive peak/trough pair, A_i = peak_i - trough_i;
    amp_cycle_median = median_i(A_i)   (robust to a few irregular cycles).
  - Per-cycle period: median spacing between successive peaks, in time units
    (period_cycle_median = median(diff(peak_times))).
  - n_cycles: number of detected peaks on the tail.
- Spectral sharpness peak_ratio = P_max / P_med from a Hann-windowed FFT of the
  detrended tail (median power computed excluding the peak bin), used as a
  quality flag for a clean oscillation.
- Classifies each eps:
    'breathing'                if tail span >= SPAN_MIN (=0.05),
                                  amp_cycle_median >= AMP_MIN (=0.15),
                                  and at least MIN_CYCLES (=2) cycles are found;
    'collapse_or_nonbreathing' otherwise (amp/period reported as NaN).
- Writes two aggregated tables:
    * bifurcation_reprocessed.csv : the full per-cycle-median result.
    * bifurcation_clipped.csv     : same, but breathing amplitudes above the
      CLIP_PERCENTILE (=95th) are clipped to that percentile to tame outliers.

OUTPUTS (in output/)
--------------------
- bifurcation_reprocessed.csv          : eps, status, amp_cycle_median,
                                         period_cycle_median, n_cycles, peak_ratio,
                                         tail_mean, tail_min, tail_max, file.
- bifurcation_reprocessed_summary.json : run parameters, per-eps status counts,
                                         inferred boundary_eps.
- bifurcation_clipped.csv              : outlier-clipped copy of the table.
- bifurcation_clipped_meta.json        : clip percentile, clip value, counts.

The columns eps, status, amp_cycle_median, period_cycle_median, n_cycles and
peak_ratio exactly match what reprocess_bifurcation_with_collapse_band.py reads.
================================================================================
"""
import os
import re
import csv
import glob
import json
import math
import numpy as np

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

OUT = 'output'
TAIL_T0 = 240.0          # discard transient: keep tail t >= TAIL_T0
H_TARGET = 'h0p533'      # only h = 0.533 tracks
SPAN_MIN = 0.05          # min tail span (max-min) to be non-collapsed
AMP_MIN = 0.15           # min per-cycle median amplitude for "breathing"
MIN_CYCLES = 2           # need at least this many detected cycles
CLIP_PERCENTILE = 95.0   # percentile for the "clipped" amplitude variant


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


def _local_extrema(y):
    """Fallback peak/trough finder: indices of strict 3-point local max/min."""
    peaks, troughs = [], []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
        elif y[i] < y[i - 1] and y[i] < y[i + 1]:
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def detect_peaks_troughs(y):
    if HAS_SCIPY:
        span = float(np.max(y) - np.min(y))
        prom = max(0.02, 0.05 * span)   # ignore tiny numerical ripples
        pk, _ = find_peaks(y, prominence=prom)
        tr, _ = find_peaks(-y, prominence=prom)
        return pk, tr
    return _local_extrema(y)


def cycle_amplitude_and_period(t, y):
    """Return (amp_cycle_median, period_cycle_median, n_cycles)."""
    pk, tr = detect_peaks_troughs(y)
    n_cycles = int(len(pk))
    if len(pk) < 1 or len(tr) < 1:
        return math.nan, math.nan, n_cycles

    # per-cycle amplitude: pair each peak with the nearest following trough
    amps = []
    for ip in pk:
        later = tr[tr > ip]
        if later.size:
            amps.append(float(y[ip] - y[later[0]]))
    if not amps:
        # fall back to global peak/trough amplitude
        amps = [float(np.max(y[pk]) - np.min(y[tr]))]
    amp_med = float(np.median(amps))

    # per-cycle period: median spacing between successive peaks
    if len(pk) >= 2:
        period_med = float(np.median(np.diff(t[pk])))
    else:
        period_med = math.nan

    return amp_med, period_med, n_cycles


def spectral_peak_ratio(t, y):
    n = len(y)
    if n < 16:
        return math.nan
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    x = y - np.mean(y)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    X = np.fft.rfft(x * w)
    P = np.abs(X) ** 2
    if len(P) < 3:
        return math.nan
    Pk = P[1:]
    Pmax = float(np.max(Pk))
    bg = np.delete(Pk, int(np.argmax(Pk)))
    Pmed = float(np.median(bg)) if bg.size else 0.0
    if Pmed <= 0:
        return 1.0e6 if Pmax > 0 else 0.0
    return float(min(Pmax / Pmed, 1.0e6))


def classify(span, amp_med, n_cycles):
    if span < SPAN_MIN:
        return 'collapse_or_nonbreathing'
    if (not math.isfinite(amp_med)) or amp_med < AMP_MIN or n_cycles < MIN_CYCLES:
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


FIELDNAMES = ['eps', 'status', 'amp_cycle_median', 'period_cycle_median',
              'n_cycles', 'peak_ratio', 'tail_mean', 'tail_min', 'tail_max', 'file']


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDNAMES})


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(OUT, f'track_eps*_{H_TARGET}_*.csv'))
                   + glob.glob(f'track_eps*_{H_TARGET}_*.csv'))
    rows = []
    for path in files:
        eps = parse_eps(path)
        if eps is None:
            continue
        t, phi = load_track(path)
        mask = t >= TAIL_T0
        if not np.any(mask):
            continue
        tt, yy = t[mask], phi[mask]
        span = float(np.max(yy) - np.min(yy))
        amp_med, period_med, n_cycles = cycle_amplitude_and_period(tt, yy)
        ratio = spectral_peak_ratio(tt, yy)
        status = classify(span, amp_med, n_cycles)

        rows.append({
            'eps': eps,
            'status': status,
            'amp_cycle_median': amp_med if status == 'breathing' else math.nan,
            'period_cycle_median': period_med if status == 'breathing' else math.nan,
            'n_cycles': n_cycles,
            'peak_ratio': ratio,
            'tail_mean': float(np.mean(yy)),
            'tail_min': float(np.min(yy)),
            'tail_max': float(np.max(yy)),
            'file': os.path.basename(path),
        })

    rows.sort(key=lambda z: z['eps'])
    if not rows:
        raise SystemExit(f"No track_eps*_{H_TARGET}_*.csv files found in {OUT}/ or cwd.")

    # ---- main reprocessed table ----
    reproc_csv = os.path.join(OUT, 'bifurcation_reprocessed.csv')
    write_csv(reproc_csv, rows)

    from collections import Counter
    status_counts = dict(Counter(r['status'] for r in rows))
    boundary = infer_boundary(rows)
    summary = {
        'source_glob': f'track_eps*_{H_TARGET}_*.csv',
        'tail_t0': TAIL_T0,
        'amplitude_definition': 'per-cycle median of (peak - following trough)',
        'period_definition': 'median spacing between successive peaks',
        'thresholds': {'span_min': SPAN_MIN, 'amp_min': AMP_MIN, 'min_cycles': MIN_CYCLES},
        'n_points': len(rows),
        'status_counts': status_counts,
        'boundary_eps': boundary,
        'scipy_used': HAS_SCIPY,
    }
    with open(os.path.join(OUT, 'bifurcation_reprocessed_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # ---- clipped variant: tame breathing-amplitude outliers ----
    bre_amps = [r['amp_cycle_median'] for r in rows
                if r['status'] == 'breathing' and math.isfinite(r['amp_cycle_median'])]
    clip_val = float(np.percentile(bre_amps, CLIP_PERCENTILE)) if bre_amps else math.nan
    clipped_rows = []
    for r in rows:
        rc = dict(r)
        a = r['amp_cycle_median']
        if r['status'] == 'breathing' and math.isfinite(a) and math.isfinite(clip_val):
            rc['amp_cycle_median'] = min(a, clip_val)
        clipped_rows.append(rc)
    clipped_csv = os.path.join(OUT, 'bifurcation_clipped.csv')
    write_csv(clipped_csv, clipped_rows)
    with open(os.path.join(OUT, 'bifurcation_clipped_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'clip_percentile': CLIP_PERCENTILE,
            'clip_value': clip_val,
            'n_breathing': len(bre_amps),
            'note': 'Breathing amp_cycle_median clipped at the given percentile to tame outliers.',
        }, f, indent=2)

    print(json.dumps(summary, indent=2))
    print('Wrote', reproc_csv, 'and', clipped_csv)


if __name__ == '__main__':
    main()
