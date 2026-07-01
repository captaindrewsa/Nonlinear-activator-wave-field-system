# Breathing Autosolitons in an Activator–Wave-Field System

Simulation and analysis code for the paper

> **Wave inhibition and breathing autosolitons in a nonlinear activator–wave-field system**
<<<<<<< HEAD
> A. A. Vlasov, Immanuel Kant Baltic Federal University, Kaliningrad, Russia.

This repository contains the numerical integrators and post-processing scripts used to produce the figures, phase diagram, and bifurcation data in the paper (`autosoliton_paper_full_v3.pdf`).
=======
> A. A. Vlasov, Kaliningrad, Russia.

This repository contains the numerical integrators and post-processing scripts used to produce the figures, phase diagram, and bifurcation data in the paper (`Wave inhibition and breathing autosolitons in a nonlinear activator–wave-field system`).
>>>>>>> fecee9d4b8e20b95b3510958ba705896adce9f0d

> **Note on formulas.** This README uses MathJax/LaTeX math. It renders on any MathJax-enabled viewer (e.g. GitHub Pages with MathJax, JupyterLab, VS Code Markdown+Math). GitHub's default Markdown renderer also supports `$...$` and `$$...$$` math.

---

## The model

We study a two-field activator–inhibitor system on a periodic square domain in which the inhibitor obeys a **damped wave equation** (second order in time) instead of the usual first-order relaxation equation:

$$
\begin{aligned}
\partial_t \phi &= M(\phi) + h + \psi, \\
\partial_{tt}\psi + \varepsilon\,\partial_t \psi &= D_\psi\,\Delta\psi + \Delta\phi .
\end{aligned}
$$

- $\phi(x,t)$ — activator field
- $\psi(x,t)$ — inhibitor wave field
- $\varepsilon$ — wave damping coefficient (the main control parameter)
- $h$ — uniform external drive
- $D_\psi$ — inhibitor diffusivity (set to $0$ throughout)

The two-threshold sigmoidal nonlinearity is

$$
M(\phi) = \phi\,\sigma_1(\phi^2)\,\bigl(1 - \sigma_2(\phi^2)\bigr) - \phi,
\qquad
\sigma_i(u) = \frac{1}{1 + \exp\!\bigl(-k\,(u - \theta_i)\bigr)},
$$

with fixed parameters $k = 10$, $\theta_1 = 4$, $\theta_2 = 16$. At $h = 0.533$ the system is bistable, with homogeneous states $\phi_{\mathrm{lo}} \approx 0.533$ and $\phi_{\mathrm{hi}} \approx 3.977$.

Making the inhibitor second order in time turns the linearised characteristic equation from quadratic into **cubic**, so unstable modes oscillate rather than grow monotonically. The physical consequence is a **breathing autosoliton**: a spatially localised spot whose amplitude oscillates periodically.

### Main numerical results reproduced here

- For $\varepsilon \in [2.0,\,3.5]$ the system sustains breathing autosolitons (period $T_{\mathrm{breath}} \approx 5\text{–}6$, dominant frequency $f \approx 0.15\text{–}0.2$).
- For $\varepsilon \geq 4.0$ the spot dissolves back into the homogeneous background (**collapse**).
- The breathing→collapse boundary near $\varepsilon_c \approx 3.5\text{–}4.0$ is sharp and direct, with **no stationary autosoliton** observed — consistent with a supercritical Hopf bifurcation.
- At $\varepsilon = 1.5$ the solution diverges at $t \approx 74.5$ independently of the time step — a genuine property of the model at very low damping, not a numerical artefact.

---

## Common numerical scheme

All scripts share the same spatial discretization, so results are mutually consistent:

- **9-point isotropic Laplacian**, periodic boundaries, $dx = 1$.
- Explicit Euler stepping, rewritten with $v = \partial_t \psi$:

$$
\begin{aligned}
\phi &\leftarrow \phi + dt\,\bigl(M(\phi) + h + \psi\bigr), \\
v    &\leftarrow v + dt\,\bigl(\Delta\phi + D_\psi\,\Delta\psi - \varepsilon\,v\bigr), \\
\psi &\leftarrow \psi + dt\,v .
\end{aligned}
$$

- Grid $N = 64$, $dt = 0.001$, total time $T = 400$ (unless noted otherwise).
- Initial condition: a radial spot of radius $\approx 8$ at amplitude $\phi_{\mathrm{hi}}$ on the $\phi_{\mathrm{lo}}$ background ($\psi = v = 0$); some scripts use a **pair** of such spots.
- A run is flagged **numerically unstable** if $|\phi| > 10^3$ or `NaN/Inf` appears, and is halted.

### Regime classification

The centre-field time series $\phi_c(t) = \phi(0,t)$ is analysed on the tail $t \in [0.6\,T,\,T]$ (i.e. $[240,\,400]$) to discard transients. With
$\phi_{\max}$, $\phi_{\min}$ and amplitude $A = \phi_{\max} - \phi_{\min}$, and a Hann-windowed real FFT giving the spectral sharpness ratio $r = P_{\max}/P_{\mathrm{med}}$ (median power computed without the peak bin):

| Code | Regime | Criterion |
|------|--------|-----------|
| $-1$ | Numerically unstable | blow-up ($\lvert\phi\rvert > 10^3$ or NaN/Inf) |
| $0$  | Collapse | $\phi_{\max} \leq \phi_{\mathrm{lo}} + 0.3$ (spot dissolved) |
| $1$  | Stationary autosoliton | localised ($\phi_{\max} > \phi_{\mathrm{lo}} + 0.3$), $A \leq 0.15$, no sharp peak — *not observed in the surveyed range* |
| $2$  | Breathing autosoliton | $A > 0.15$ **and** a distinct spectral peak ($r \geq 5$) |

---

## Repository layout

```
.
├── README.md
│
├── phase_diagram_eps_h/           # Phase diagram in the (ε, h) plane — paper Fig. 9
│   ├── phase_scan_strict.py
│   ├── merge_and_classify.py
│   ├── make_phase_diagram_strict.py
│   ├── phase_scan_eps_h.py        # earlier minimal (amplitude-only) version
│   └── make_phase_diagram.py      # earlier minimal (amplitude-only) version
│
├── fig_build_scripts/             # Bifurcation diagram A(ε) at fixed h = 0.533
│   ├── REBUILD_PDE_from_glider_search.py
│   ├── reprocess_bifurcation_cycle_median.py
│   ├── reprocess_bifurcation_with_collapse_band.py
│   ├── plot_bifurcation_reprocessed.py
│   ├── patch_bifurcation_rms_amplitude.py
│   └── output/                    # generated CSVs, JSON summaries and figures
│
└── protoGliders and search_engine/
    ├── glider_search_v22_windows.py
    └── Examples/                  # sample preview GIFs, space-time and track CSVs
```

> **Backend note.** The two integrators (`glider_search_v22_windows.py`, `REBUILD_PDE_from_glider_search.py`) use **PyTorch** and auto-select CUDA → CPU. The phase-diagram scan (`phase_scan_strict.py`) is a **pure-NumPy** reimplementation of the same scheme. The remaining scripts are NumPy/Matplotlib post-processing only.

---

## Scripts

### `phase_diagram_eps_h/` — phase diagram in the $(\varepsilon, h)$ plane (Fig. 9)

- **`phase_scan_strict.py`** — Strict $(\varepsilon, h)$ scan (pure NumPy). Integrates each of the $11 \times 6$ parameter points, records $\phi_c(t)$, performs a Hann-windowed FFT for the frequency criterion, and classifies each point ($-1/0/1/2$). Writes `phase_map_eps_h_data_strict.csv` (columns `eps, h, A, phi_max, phi_min, phi_mean, dominant_freq, freq_ratio, regime_code`). Accepts an $\varepsilon$-subset argument so the grid can be split across parallel processes.
- **`merge_and_classify.py`** — Merges partial scan CSVs (`strict_partA.csv`, `strict_partB.csv`) and re-applies the final classification (collapse determined by $\phi_{\max}$ alone). Produces the definitive `phase_map_eps_h_data_strict.csv`.
- **`make_phase_diagram_strict.py`** — Renders the strict phase diagram `fig_parameter_map_eps_h_strict.pdf`: dark blue = collapse, orange = stationary, red = breathing, gray + cross = numerically unstable.
- **`phase_scan_eps_h.py`** / **`make_phase_diagram.py`** — Earlier *minimal* amplitude-only version of the scan and diagram (no FFT), kept for reference.

### `fig_build_scripts/` — bifurcation diagram $A(\varepsilon)$ at fixed $h = 0.533$

This is a four-stage pipeline: **compute → aggregate → redraw** (with an optional parallel RMS-amplitude variant).

- **`REBUILD_PDE_from_glider_search.py`** — *(compute; PyTorch)* The only bifurcation script that integrates the PDE. Sweeps $\varepsilon$ at $h = 0.533$ from a kicked spot pair, and produces `track_*.csv`, the raw bifurcation diagram `fig_bifurcation_A_eps_fixed.*`, the period `fig_period_T_eps_fixed.*`, and (for $\varepsilon = 3.0$) the $\phi_c(t)$ time series, phase portrait, and space-time diagram. Also writes `bifurcation_raw.csv`, `bifurcation_clean.csv`, `period_vs_eps.csv`, `rebuild_summary.json`.
- **`reprocess_bifurcation_cycle_median.py`** — *(aggregate)* Re-reads the `track_eps*_h0p533_*.csv` tails ($t \geq 240$), detects individual oscillation cycles (via `scipy.signal.find_peaks`, with a local-extrema fallback), and computes a robust **per-cycle median amplitude** (median of peak$-$following-trough) and **median period** (median peak spacing), plus a Hann-windowed FFT peak ratio and a `breathing/other` status. This is the intermediate step that feeds the final redraw. Writes the aggregated `bifurcation_reprocessed.csv` (columns `eps, status, amp_cycle_median, period_cycle_median, n_cycles, peak_ratio, tail_mean, tail_min, tail_max, file`), a 95th-percentile-clipped `bifurcation_clipped.csv`, and the summaries `bifurcation_reprocessed_summary.json`, `bifurcation_clipped_meta.json`.
- **`patch_bifurcation_rms_amplitude.py`** — *(post-process; optional parallel variant)* Re-reads the `track_*.csv` tails and replaces the max$-$min amplitude with a more robust **RMS amplitude** $A_{\mathrm{rms}} = \sqrt{\langle (\phi_c - \langle\phi_c\rangle)^2\rangle}$. Estimates the boundary $\varepsilon_c$ and draws `fig_bifurcation_A_rms_eps.*` with a gray collapse band. Writes `bifurcation_rms.csv`, `bifurcation_rms_meta.json`.
- **`reprocess_bifurcation_with_collapse_band.py`** — *(figure only)* Redraws the final bifurcation figure `fig_bifurcation_A_eps_collapse_band.*` from the aggregated `bifurcation_reprocessed.csv` (per-cycle **median** amplitude) produced by the previous step, shading the collapse region. Writes `fig_bifurcation_A_eps_collapse_band.json`.
- **`plot_bifurcation_reprocessed.py`** — *(figure only)* Draws the two remaining figure variants that share the same aggregated data, in the same visual style as the collapse-band figure: `fig_bifurcation_A_eps_reprocessed.*` from `bifurcation_reprocessed.csv` and `fig_bifurcation_A_eps_clipped.*` from `bifurcation_clipped.csv` (skipped if the clipped CSV is absent). Writes `plot_bifurcation_reprocessed.json`.

### `protoGliders and search_engine/` — glider search

- **`glider_search_v22_windows.py`** — Original exploratory **two-spot glider search** (PyTorch, $N = 224$). Warms single-spot seeds, builds interacting spot pairs with configurable separation, phase, kick and asymmetry, integrates for $T = 1400$, tracks the centre-of-mass trajectory, and classifies each run as `glider / pinned / pinned_breather / complex / decay`. Outputs per-case `track_*.csv`, space-time diagrams, preview GIFs, and JSON summaries. Resumable. The phase-diagram and bifurcation scripts reuse only its model, Laplacian, stepping scheme and (single-)spot initial condition.

---

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
torch          # only for glider_search_v22_windows.py and REBUILD_PDE_from_glider_search.py
```

Install with:

```bash
pip install numpy scipy matplotlib torch
```

---

## Reproducing the paper results

**Strict $(\varepsilon, h)$ phase diagram (Fig. 9):**

```bash
cd phase_diagram_eps_h
# Split the 11×6 grid across two processes, then merge and plot
python phase_scan_strict.py "1.5,2.0,2.5,2.8,3.0,3.2" strict_partA.csv
python phase_scan_strict.py "3.5,4.0,4.5,5.0,6.0"     strict_partB.csv
python merge_and_classify.py
python make_phase_diagram_strict.py
# -> phase_map_eps_h_data_strict.csv, fig_parameter_map_eps_h_strict.pdf
```

**Bifurcation diagram $A(\varepsilon)$ at $h = 0.533$ (Figs. 2–5):**

```bash
cd fig_build_scripts
python REBUILD_PDE_from_glider_search.py            # 1. integrate PDE + first figures + track_*.csv
python reprocess_bifurcation_cycle_median.py        # 2. aggregate tails -> bifurcation_reprocessed.csv, bifurcation_clipped.csv
python reprocess_bifurcation_with_collapse_band.py  # 3. final collapse-band figure
python plot_bifurcation_reprocessed.py              # 4. reprocessed + clipped figure variants
python patch_bifurcation_rms_amplitude.py           # (optional) parallel RMS-amplitude version
```

**Glider search (Sec. VI, preliminary):**

```bash
cd "protoGliders and search_engine"
python glider_search_v22_windows.py
```

---

## Notes and limitations

- The explicit Euler scheme is constrained by the wave-equation stability limit; large $D_\psi$ would require a semi-implicit or operator-splitting method.
<<<<<<< HEAD
- No stationary autosoliton ($\text{regime\_code} = 1$) is found for the radial-spot initial condition; whether one exists under a different initial condition remains open.
=======
- No stationary autosoliton ($\text{regime\code} = 1$) is found for the radial-spot initial condition; whether one exists under a different initial condition remains open.
>>>>>>> fecee9d4b8e20b95b3510958ba705896adce9f0d
- Glider-like drift is currently observed on $48 \times 48$ grids and awaits confirmation on larger grids.
- `reprocess_bifurcation_with_collapse_band.py` consumes the aggregated `bifurcation_reprocessed.csv`; this file is produced by `reprocess_bifurcation_cycle_median.py`, which must be run first.
- `reprocess_bifurcation_cycle_median.py` writes only the aggregated CSVs/JSON summaries; the `fig_bifurcation_A_eps_reprocessed.*` and `fig_bifurcation_A_eps_clipped.*` figure variants are rendered from those CSVs by `plot_bifurcation_reprocessed.py`.

---

## Citation

If you use this code, please cite the accompanying paper:

<<<<<<< HEAD
> A. A. Vlasov, *Wave inhibition and breathing autosolitons in a nonlinear activator–wave-field system*, Immanuel Kant Baltic Federal University.
=======
> A. A. Vlasov, *Wave inhibition and breathing autosolitons in a nonlinear activator–wave-field system*.
>>>>>>> fecee9d4b8e20b95b3510958ba705896adce9f0d
