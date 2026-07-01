# Breathing Autosolitons in an Activator–Wave-Field System

Simulation and analysis code for the paper

> \*\*Wave inhibition and breathing autosolitons in a nonlinear activator–wave-field system\*\*
> A. A. Vlasov, Kaliningrad, Russia.

This repository contains the numerical integrators and post-processing scripts used to produce the figures, phase diagram, and bifurcation data in the paper (`autosoliton\_paper\_full\_v3.pdf`).

\---

## The model

We study a two-field activator–inhibitor system on a periodic square domain in which the inhibitor obeys a **damped wave equation** (second order in time) instead of the usual first-order relaxation equation:

```
  ∂ₜ φ            = M(φ) + h + ψ
  ∂ₜₜ ψ + ε ∂ₜ ψ  = D\_ψ Δψ + Δφ
```

* `φ(x,t)` — activator field
* `ψ(x,t)` — inhibitor wave field
* `ε` — wave damping coefficient (the main control parameter)
* `h` — uniform external drive
* `D\_ψ` — inhibitor diffusivity (set to `0` throughout)

The two-threshold sigmoidal nonlinearity is

```
  M(φ) = φ·σ₁(φ²)·(1 − σ₂(φ²)) − φ,      σ\_i(u) = 1/(1 + exp(−k(u − θ\_i)))
```

with fixed parameters **k = 10, θ₁ = 4, θ₂ = 16**. At `h = 0.533` the system is bistable, with homogeneous states `φ\_lo ≈ 0.533` and `φ\_hi ≈ 3.977`.

Making the inhibitor second-order in time turns the linearised characteristic equation from quadratic into **cubic**, so unstable modes oscillate rather than grow monotonically. The physical consequence is a **breathing autosoliton**: a spatially localised spot whose amplitude oscillates periodically.

### Main numerical results reproduced here

* For **ε ∈ \[2.0, 3.5]** the system sustains breathing autosolitons (period `T ≈ 5–6`, dominant frequency `f ≈ 0.15–0.2`).
* For **ε ≥ 4.0** the spot dissolves back into the homogeneous background (**collapse**).
* The breathing→collapse boundary near **ε\_c ≈ 3.5–4.0** is sharp and direct, with **no stationary autosoliton** observed — consistent with a supercritical Hopf bifurcation.
* At **ε = 1.5** the solution diverges at `t ≈ 74.5` independently of the time step — a genuine property of the model at very low damping, not a numerical artefact.

\---

## Common numerical scheme

All scripts share the same spatial discretization, so results are mutually consistent:

* **9-point isotropic Laplacian**, periodic boundaries, `dx = 1`.
* Explicit Euler stepping, rewritten with `v = ∂ₜ ψ`:

```
  φ ← φ + dt·(M(φ) + h + ψ)
  v ← v + dt·(Δφ + D\_ψ·Δψ − ε·v)
  ψ ← ψ + dt·v
  ```

* Grid `N = 64`, `dt = 0.001`, total time `T = 400` (unless noted otherwise).
* Initial condition: a radial spot of radius ≈ 8 at amplitude `φ\_hi` on the `φ\_lo` background (`ψ = v = 0`); some scripts use a **pair** of such spots.
* A run is flagged **numerically unstable** if `|φ| > 10³` or `NaN/Inf` appears, and is halted.

### Regime classification

The centre-field time series `φ\_c(t) = φ(0,t)` is analysed on the tail `t ∈ \[0.6·T, T]` (i.e. `\[240, 400]`) to discard transients:

|Code|Regime|Criterion|
|-|-|-|
|`-1`|Numerically unstable|blow-up (`|
|`0`|Collapse|`φ\_max ≤ φ\_lo + 0.3` (spot dissolved)|
|`1`|Stationary autosoliton|localised (`φ\_max > φ\_lo + 0.3`), amplitude `A ≤ 0.15`, no sharp spectral peak — *not observed in the surveyed range*|
|`2`|Breathing autosoliton|`A > 0.15` **and** a distinct spectral peak (`P\_max/P\_med ≥ 5` from a Hann-windowed FFT)|

\---

## Repository contents

> \*\*Backend note.\*\* The two integrators (`glider\_serach\_v22\_windows.py`, `rebuild\_figures\_from\_glider-2.py`) use \*\*PyTorch\*\* and auto-select CUDA → CPU. The phase-diagram scan (`phase\_scan\_strict.py`) is a \*\*pure-NumPy\*\* reimplementation of the same scheme. The remaining scripts are NumPy/Matplotlib post-processing only.

### Core simulation

* **`glider\_serach\_v22\_windows.py`** — Original exploratory **two-spot glider search** (PyTorch, `N = 224`). Warms single-spot seeds, builds interacting spot pairs with configurable separation, phase, kick and asymmetry, integrates for `T = 1400`, tracks the centre-of-mass trajectory, and classifies each run as `glider / pinned / pinned\_breather / complex / decay`. Outputs per-case `track\_\*.csv`, space-time diagrams, and JSON summaries. Resumable. The phase-diagram scripts reuse only its model, Laplacian, stepping scheme and single-spot initial condition.

### Phase diagram in the (ε, h) plane

* **`phase\_scan\_strict.py`** — Strict `(ε, h)` scan (pure NumPy). Integrates each of the `11 × 6` parameter points, records `φ\_c(t)`, performs a Hann-windowed FFT for the frequency criterion, and classifies each point (`-1/0/1/2`). Writes `phase\_map\_eps\_h\_data\_strict.csv` (columns `eps, h, A, phi\_max, phi\_min, phi\_mean, dominant\_freq, freq\_ratio, regime\_code`). Accepts an `ε`-subset argument so the grid can be split across parallel processes.
* **`merge\_and\_classify.py`** — Merges partial scan CSVs (`strict\_partA.csv`, `strict\_partB.csv`) and re-applies the final classification (collapse determined by `φ\_max` alone). Produces the definitive `phase\_map\_eps\_h\_data\_strict.csv`.
* **`make\_phase\_diagram\_strict.py`** — Renders the strict phase diagram `fig\_parameter\_map\_eps\_h\_strict.pdf`: dark blue = collapse, orange = stationary, red = breathing, gray + cross = numerically unstable.
* **`phase\_scan\_eps\_h.py`** / **`make\_phase\_diagram.py`** — Earlier *minimal* amplitude-only version of the scan and diagram (no FFT), kept for reference.

### Bifurcation diagram A(ε) at fixed h = 0.533

These form a pipeline: **compute → recompute amplitude → redraw**.

* **`rebuild\_figures\_from\_glider-2.py`** — *(compute; PyTorch)* The only bifurcation script that integrates the PDE. Sweeps `ε` at `h = 0.533` from a kicked spot pair, and produces `track\_\*.csv`, the raw bifurcation diagram `A(ε)`, the period `T(ε)`, and (for `ε = 3.0`) the `φ\_c(t)` time series, phase portrait, and space-time diagram.
* **`patch\_bifurcation\_rms\_amplitude.py`** — *(post-process)* Re-reads the `track\_\*.csv` tails and replaces the max−min amplitude with a more robust **RMS amplitude** `A\_rms = √⟨(φ\_c − ⟨φ\_c⟩)²⟩`. Estimates the boundary `ε\_c` and draws `fig\_bifurcation\_A\_rms\_eps.pdf` with a gray collapse band.
* **`reprocess\_bifurcation\_with\_collapse\_band-3.py`** — *(figure only)* Redraws the final bifurcation figure `fig\_bifurcation\_A\_eps\_collapse\_band.pdf` from the aggregated `bifurcation\_reprocessed.csv` (per-cycle **median** amplitude), shading the collapse region.

\---

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
torch          # only for glider\_serach\_v22\_windows.py and rebuild\_figures\_from\_glider-2.py
```

Install with:

```bash
pip install numpy scipy matplotlib torch
```

\---

## Reproducing the paper results

**Strict (ε, h) phase diagram (Fig. 9):**

```bash
# Split the 11×6 grid across two processes, then merge and plot
python phase\_scan\_strict.py "1.5,2.0,2.5,2.8,3.0,3.2" strict\_partA.csv
python phase\_scan\_strict.py "3.5,4.0,4.5,5.0,6.0"     strict\_partB.csv
python merge\_and\_classify.py
python make\_phase\_diagram\_strict.py
# -> phase\_map\_eps\_h\_data\_strict.csv, fig\_parameter\_map\_eps\_h\_strict.pdf
```

**Bifurcation diagram A(ε) at h = 0.533 (Figs. 2–5):**

```bash
python rebuild\_figures\_from\_glider-2.py           # integrate + first figures
python patch\_bifurcation\_rms\_amplitude.py         # RMS-amplitude version
python reprocess\_bifurcation\_with\_collapse\_band-3.py   # final figure
```

**Glider search (Sec. VI, preliminary):**

```bash
python glider\_serach\_v22\_windows.py
```

\---

## Notes and limitations

* The explicit Euler scheme is constrained by the wave-equation stability limit; large `D\_ψ` would require a semi-implicit or operator-splitting method.
* No stationary autosoliton (`regime\_code = 1`) is found for the radial-spot initial condition; whether one exists under a different initial condition remains open.
* Glider-like drift is currently observed on `48×48` grids and awaits confirmation on larger grids.

\---

## Citation

If you use this code, please cite the accompanying paper:

> A. A. Vlasov, \*Wave inhibition and breathing autosolitons in a nonlinear activator–wave-field system\*.

