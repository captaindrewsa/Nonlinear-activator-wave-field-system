#!/usr/bin/env python3
"""Fourier IMEX v3.2: backward-compatible M0/M1 framework.
M1: phi_t=d Lap(phi)+M(phi)+h; psi_t=v;
tau v_t=Lap(phi)+D_psi Lap(psi)-mu psi-eps v+eta Lap(v).
Default new coefficients preserve legacy physics. Existing API and CN/AB2 remain.
Geometry defaults to grid units; opt in with coordinate_units='physical'.
Reaction is evaluated on the base grid, then truncated (not exact dealiasing).
NPZ initialization is a field restart, not a history-preserving AB2 restart.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import torch_directml

    has_directml = True
except Exception:
    torch_directml = None
    has_directml = False


logger = logging.getLogger("sim_framework_fourier_imex_v3")

SOLVER_VERSION = "fourier_imex_v3.2_m1"

kappa_default = 10.0
theta_1_default = 4.0
theta_2_default = 16.0
phi_lo_default = 0.533
phi_hi_default = 3.9766096853487105


# ─────────────────────────────────────────────────────────────
# Dataclasses (identical field layout to v1/v2 for config compat)
# ─────────────────────────────────────────────────────────────


@dataclass
class AttractorSpec:
    kind: str = "pump"
    cx: float = 80.0
    cy: float = 80.0
    strength: float = 0.05
    sigma: float = 8.0
    profile: str = "gaussian"
    ring_r: float = 20.0
    ring_w: float = 3.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AttractorSpec":
        return AttractorSpec(**data)


@dataclass
class InitSpot:
    cx: float = 80.0
    cy: float = 80.0
    radius: float = 8.0
    amp: Optional[float] = None
    shape: str = "disk"
    phase_v: float = 0.0
    interface_width: float = 1.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InitSpot":
        return InitSpot(**data)


@dataclass
class InitConfig:
    npz_path: Optional[str] = None
    spots: List[InitSpot] = field(default_factory=list)
    noise_amplitude: float = 0.0
    noise_seed: int = 42
    noise_lowpass: float = 0.0
    phi_background: Optional[float] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "InitConfig":
        payload = copy.deepcopy(data)
        payload["spots"] = [InitSpot.from_dict(i) for i in payload.get("spots", [])]
        return InitConfig(**payload)


@dataclass
class SnapshotPolicy:
    every_steps: int = 2000
    t_start: float = 0.0
    t_stop: float = -1.0
    save_phi: bool = True
    save_psi: bool = True
    save_v: bool = False
    max_snaps: int = -1
    labels_save: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SnapshotPolicy":
        return SnapshotPolicy(**data)


@dataclass
class SimConfig:
    eps: float = 3.0
    h_bg: float = 0.533
    D_psi: float = 0.0
    phi_lo: float = phi_lo_default
    phi_hi: float = phi_hi_default
    kappa: float = kappa_default
    theta1: float = theta_1_default
    theta2: float = theta_2_default
    nx: int = 160
    ny: int = 160
    dx: float = 1.0
    boundary: str = "periodic"
    dt: float = 0.0015
    t_total: float = 800.0
    attractors: List[AttractorSpec] = field(default_factory=list)
    gamma_bg: float = 0.0
    h_custom: Optional[str] = None
    gamma_custom: Optional[str] = None
    init: InitConfig = field(default_factory=InitConfig)
    snap: SnapshotPolicy = field(default_factory=SnapshotPolicy)
    out_dir: str = "./output_fourier_imex_v3"
    monitor_every: int = 10
    t_warm: float = 0.0
    seed_n_snaps: int = 16
    check_nonfinite: bool = True
    abort_on_instability: bool = True
    max_abs_phi: float = 1.0e6
    max_abs_psi: float = 1.0e6
    max_abs_v: float = 1.0e6
    run_id: str = ""
    tags: List[str] = field(default_factory=list)
    # ── v2 fields, accepted for config compatibility ──
    phi_heun: bool = True          # ignored by v3 (fully coupled step is always used)
    dealias: bool = True           # 2/3 spectral dealiasing
    dealias_fraction: float = 2.0 / 3.0
    # ── v3-only optional field ──
    ab2_bootstrap: str = "imex_euler"  # how the first step (no history) is handled

    # Appended fields preserve positional constructor compatibility.
    d: float = 0.0
    mu: float = 0.0
    eta: float = 0.0
    tau: float = 1.0
    model_id: str = "auto"
    coordinate_units: str = "grid"
    strict_config: bool = False

    def __post_init__(self):
        if self.tau <= 0 or self.dt <= 0 or self.dx <= 0:
            raise ValueError("tau, dt and dx must be positive")
        if min(self.d, self.mu, self.eta, self.D_psi, self.eps) < 0:
            raise ValueError("M1 requires nonnegative d, mu, eta, D_psi, eps")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must be >= 2")
        if not 0 < self.dealias_fraction <= 1:
            raise ValueError("dealias_fraction must be in (0,1]")
        if self.coordinate_units not in ("grid", "physical"):
            raise ValueError("coordinate_units must be grid or physical")
        if self.ab2_bootstrap != "imex_euler":
            raise ValueError("Only the legacy CN/source bootstrap 'imex_euler' is implemented")
        effective = "M0" if (self.d == self.mu == self.eta == 0 and self.tau == 1) else "M1"
        if self.model_id not in ("auto", "M0", "M1"):
            raise ValueError("model_id must be auto, M0 or M1")
        if self.model_id == "M0" and effective != "M0":
            raise ValueError("M0 is incompatible with nondefault M1 parameters")

    def resolved_model_id(self):
        return ("M0" if self.d == self.mu == self.eta == 0 and self.tau == 1 else "M1") if self.model_id == "auto" else self.model_id

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SimConfig":
        payload = copy.deepcopy(data)
        payload["attractors"] = [
            AttractorSpec.from_dict(i) for i in payload.get("attractors", [])
        ]
        payload["init"] = InitConfig.from_dict(payload.get("init", {}))
        payload["snap"] = SnapshotPolicy.from_dict(payload.get("snap", {}))
        # strip unknown keys so old/newer configs with extra fields don't crash
        known = {f.name for f in SimConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - known)
        if unknown:
            message = "Unknown configuration keys: " + ", ".join(unknown)
            if payload.get("strict_config", False):
                raise ValueError(message)
            logger.warning(message + " (ignored for backward compatibility)")
        payload = {k: v for k, v in payload.items() if k in known}
        return SimConfig(**payload)

    @staticmethod
    def from_json(path: str) -> "SimConfig":
        with open(path, "r", encoding="utf-8") as f:
            return SimConfig.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def resolve_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        return "fourier_imex_v3_" + hashlib.md5(encoded).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────
# Physics helpers (unchanged)
# ─────────────────────────────────────────────────────────────


def _sigma(u: torch.Tensor, kappa: float, theta: float) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-kappa * (u - theta)))


def nonlinear_m(phi: torch.Tensor, kappa: float, theta1: float, theta2: float) -> torch.Tensor:
    phi2 = phi * phi
    return phi * _sigma(phi2, kappa, theta1) * (1.0 - _sigma(phi2, kappa, theta2)) - phi


# ─────────────────────────────────────────────────────────────
# Geometry helpers (unchanged from v1/v2)
# ─────────────────────────────────────────────────────────────


def gaussian_2d(nx, ny, cx, cy, sigma_value, device, dtype):
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    return torch.exp(-0.5 * (((x - cx) / sigma_value) ** 2 + ((y - cy) / sigma_value) ** 2))


def disk_2d(nx, ny, cx, cy, radius, device, dtype):
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    return (((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2).to(dtype)


def tanh_disk_2d(nx, ny, cx, cy, radius, interface_width, device, dtype):
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    w = max(float(interface_width), 1.0e-8)
    return 0.5 * (1.0 - torch.tanh((r - radius) / w))


def ring_2d(nx, ny, cx, cy, ring_r, ring_w, device, dtype):
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return torch.exp(-0.5 * ((r - ring_r) / ring_w) ** 2)


def build_spatial_fields(cfg: SimConfig, device, dtype):
    if cfg.coordinate_units == "physical":
        cfg = copy.deepcopy(cfg)
        cfg.coordinate_units = "grid"
        for obj in list(cfg.attractors) + list(cfg.init.spots):
            for key in ("cx", "cy", "sigma", "ring_r", "ring_w", "radius", "interface_width"):
                if hasattr(obj, key):
                    setattr(obj, key, getattr(obj, key) / cfg.dx)
    h_field = torch.full((cfg.nx, cfg.ny), cfg.h_bg, dtype=dtype, device=device)
    gamma_field = torch.full((cfg.nx, cfg.ny), cfg.gamma_bg, dtype=dtype, device=device)

    if cfg.h_custom:
        p = Path(cfg.h_custom)
        if p.exists():
            h_field = torch.from_numpy(np.load(p).astype(np.float64)).to(device=device, dtype=dtype)
    if cfg.gamma_custom:
        p = Path(cfg.gamma_custom)
        if p.exists():
            gamma_field = torch.from_numpy(np.load(p).astype(np.float64)).to(device=device, dtype=dtype)

    for item in cfg.attractors:
        if item.profile == "gaussian":
            mask = gaussian_2d(cfg.nx, cfg.ny, item.cx, item.cy, item.sigma, device, dtype)
        elif item.profile == "disk":
            mask = disk_2d(cfg.nx, cfg.ny, item.cx, item.cy, item.sigma, device, dtype)
        elif item.profile == "ring":
            mask = ring_2d(cfg.nx, cfg.ny, item.cx, item.cy, item.ring_r, item.ring_w, device, dtype)
        else:
            raise ValueError(f"Unknown attractor profile: {item.profile}")

        if item.kind == "pump":
            h_field = h_field + item.strength * mask
        elif item.kind == "sink":
            gamma_field = gamma_field + item.strength * mask
        elif item.kind == "hsink":
            h_field = h_field - item.strength * mask
        else:
            raise ValueError(f"Unknown attractor kind: {item.kind}")

    return h_field, gamma_field


def build_initial_state(cfg: SimConfig, device, dtype):
    if cfg.coordinate_units == "physical":
        cfg = copy.deepcopy(cfg)
        cfg.coordinate_units = "grid"
        for obj in list(cfg.attractors) + list(cfg.init.spots):
            for key in ("cx", "cy", "sigma", "ring_r", "ring_w", "radius", "interface_width"):
                if hasattr(obj, key):
                    setattr(obj, key, getattr(obj, key) / cfg.dx)
    if cfg.init.npz_path:
        p = Path(cfg.init.npz_path)
        if p.exists():
            with np.load(p) as data:
                phi = torch.from_numpy(np.asarray(data["phi"], dtype=np.float64)).to(device=device, dtype=dtype)
                psi = torch.from_numpy(
                    np.asarray(data.get("psi", np.zeros_like(data["phi"])), dtype=np.float64)
                ).to(device=device, dtype=dtype)
                v = torch.from_numpy(
                    np.asarray(data.get("v", np.zeros_like(data["phi"])), dtype=np.float64)
                ).to(device=device, dtype=dtype)
            return phi, psi, v

    bg = cfg.phi_lo if cfg.init.phi_background is None else float(cfg.init.phi_background)
    phi = torch.full((cfg.nx, cfg.ny), bg, dtype=dtype, device=device)
    psi = torch.zeros_like(phi)
    v = torch.zeros_like(phi)

    for spot in cfg.init.spots:
        amp = cfg.phi_hi if spot.amp is None else float(spot.amp)
        if spot.shape == "disk":
            mask = disk_2d(cfg.nx, cfg.ny, spot.cx, spot.cy, spot.radius, device, dtype)
        elif spot.shape == "tanh_disk":
            mask = tanh_disk_2d(cfg.nx, cfg.ny, spot.cx, spot.cy, spot.radius, spot.interface_width, device, dtype)
        elif spot.shape == "gaussian":
            mask = gaussian_2d(cfg.nx, cfg.ny, spot.cx, spot.cy, spot.radius, device, dtype)
        else:
            raise ValueError(f"Unknown spot shape: {spot.shape}")
        phi = phi + (amp - bg) * mask
        if spot.phase_v != 0.0:
            v = v + float(spot.phase_v) * gaussian_2d(cfg.nx, cfg.ny, spot.cx, spot.cy, spot.radius, device, dtype)

    if cfg.init.noise_amplitude > 0.0:
        rng = np.random.default_rng(cfg.init.noise_seed)
        noise = rng.standard_normal((cfg.nx, cfg.ny)).astype(np.float64) * float(cfg.init.noise_amplitude)
        if cfg.init.noise_lowpass > 0.0:
            f = np.fft.fft2(noise)
            kx = np.fft.fftfreq(cfg.nx, d=cfg.dx) * cfg.nx
            ky = np.fft.fftfreq(cfg.ny, d=cfg.dx) * cfg.ny
            K2 = kx[:, None] ** 2 + ky[None, :] ** 2
            cutoff = (cfg.init.noise_lowpass * min(cfg.nx, cfg.ny) / 2) ** 2
            f[K2 > cutoff] = 0.0
            noise = np.fft.ifft2(f).real
        phi = phi + torch.from_numpy(noise).to(device=device, dtype=dtype)

    return phi, psi, v


# ─────────────────────────────────────────────────────────────
# Core stepper (v3: fully coupled Fourier-IMEX for phi, psi, v)
# ─────────────────────────────────────────────────────────────


class FourierImexStepperV3:
    """
    Spatial: Fourier pseudospectral with optional 2/3-rule dealiasing.

    Linear part (per Fourier mode, matrix A(k)):

        d/dt [phi_hat]   [ 0      1    0   ] [phi_hat]
             [psi_hat] = [ 0      0    1   ] [psi_hat]
             [v_hat  ]   [-k^2    0  -eps  ] [v_hat  ]

    treated with Crank-Nicolson (trapezoidal rule):

        (I - dt/2 A) u^{n+1} = (I + dt/2 A) u^n + dt * N*

    where N* = [AB2 extrapolation of (M(phi)+h)_hat, 0, 0] and the
    dealiasing mask is applied to phi_hat/psi_hat/v_hat and to the
    nonlinear term before every linear solve.

    Spatially uniform eps is required for the closed-form per-mode 3x3
    solve (same restriction v1/v2 already imposed on gamma_field: the
    Fourier-CN family cannot handle spatially varying wave damping).
    Spatially varying h_field (e.g. pump/sink attractors) is fully
    supported because it only enters through the explicit nonlinear/
    forcing term N* and never needs to be diagonalised in Fourier space.
    """

    def __init__(self, cfg: SimConfig, h_field: torch.Tensor, gamma_field: torch.Tensor) -> None:
        if cfg.boundary.lower() != "periodic":
            raise ValueError("FourierImexStepperV3 requires boundary='periodic'.")

        if torch.any(gamma_field != 0):
            raise ValueError(
                "The reference model (cnsns_autosoliton_wave, Eq. 2) has a "
                "single spatially uniform wave-damping coefficient eps; it "
                "does not define a spatially varying damping/sink term. "
                "Set gamma_bg=0 and use no 'sink' attractors so the model "
                "matches Eqs. (1)-(2). A spatially varying damping term "
                "would be a distinct model extension, not the one derived "
                "in the reference paper."
            )

        self.cfg = cfg
        self.h_field = h_field
        self.gamma_field = gamma_field
        self.dtype = h_field.dtype
        self.device = h_field.device
        self.dt = float(cfg.dt)
        self.half_dt = 0.5 * self.dt
        self.dealias_flag = bool(cfg.dealias)

        # ── wavenumbers ──
        kx = (2.0 * np.pi * torch.fft.fftfreq(cfg.nx, d=cfg.dx, device=self.device, dtype=self.dtype)).view(-1, 1)
        ky = (2.0 * np.pi * torch.fft.fftfreq(cfg.ny, d=cfg.dx, device=self.device, dtype=self.dtype)).view(1, -1)
        self.k2 = kx * kx + ky * ky  # (nx, ny), real

        # ── dealiasing mask (2/3 rule), same convention as v2 ──
        frac = float(cfg.dealias_fraction)
        kx_idx = torch.fft.fftfreq(cfg.nx, device=self.device, dtype=self.dtype).view(-1, 1).abs()
        ky_idx = torch.fft.fftfreq(cfg.ny, device=self.device, dtype=self.dtype).view(1, -1).abs()
        self.dealias_mask = ((kx_idx <= frac / 2) & (ky_idx <= frac / 2)).to(self.dtype)

        # ── CFL reference (diagnostic only) ──
        self.dt_cfl_ratio = self.dt / cfg.dx

        eps = float(cfg.eps)
        d_psi = float(cfg.D_psi)
        k2 = self.k2  # (nx, ny) real tensor

        phi0_ref = cfg.init.phi_background if cfg.init.phi_background is not None else cfg.phi_lo
        self.a_lin = self._linearised_reaction_slope(phi0_ref)
        # Legacy scalar threshold is valid only for a=-1, tau=1, d=mu=eta=0.
        applicable = (abs(self.a_lin + 1) < 1e-6 and cfg.tau == 1 and cfg.d == cfg.mu == cfg.eta == 0)
        self.dpsi_stability_bound = 1.0 / eps if applicable and eps > 0 else float("nan")
        self.dpsi_within_stability_bound = True  # retained compatibility field; see full spectrum below
        denom = 1.0 - eps * d_psi
        self.k_c2 = (eps * (1 + eps) / denom if denom > 0 else float("inf")) if applicable else float("nan")
        self.k_c = float(np.sqrt(self.k_c2))
        q_np = self.k2.detach().cpu().numpy()
        A = np.zeros(q_np.shape + (3, 3))
        A[..., 0, 0] = self.a_lin - cfg.d * q_np
        A[..., 0, 1] = 1
        A[..., 1, 2] = 1
        A[..., 2, 0] = -q_np / cfg.tau
        A[..., 2, 1] = -(cfg.D_psi * q_np + cfg.mu) / cfg.tau
        A[..., 2, 2] = -(cfg.eps + cfg.eta * q_np) / cfg.tau
        growth = np.linalg.eigvals(A).real.max(axis=-1)
        retained = self.dealias_mask.detach().cpu().numpy().astype(bool) if cfg.dealias else np.ones_like(q_np, dtype=bool)
        self.linear_diagnostics = {
            "reference_phi": float(phi0_ref), "a": self.a_lin,
            "interpretation": "Frozen homogeneous coefficients at reference phi; not nonlinear stability proof",
            "max_growth_retained": float(growth[retained].max()),
            "max_growth_nonzero_retained": float(growth[retained & (q_np > 0)].max()) if np.any(retained & (q_np > 0)) else None,
            "max_growth_zero": float(growth[0, 0]),
            "legacy_threshold_applicable": applicable,
        }

        # Complex dtype matching self.dtype precision
        cdtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64

        dt2 = self.half_dt
        one = torch.ones_like(k2)
        zero = torch.zeros_like(k2)

        # Model (cnsns_autosoliton_wave, Eqs. 1-2 / 6-8):
        #   d phi/dt = M(phi) + h + psi
        #   d psi/dt = v
        #   d v/dt   = D_psi * Laplacian(psi) + Laplacian(phi) - eps * v
        # In Fourier space Laplacian -> -k^2, giving the linear operator
        #   A(k) = [[ 0,          1,         0   ],
        #           [ 0,          0,         1   ],
        #           [-k^2,  -D_psi*k^2,     -eps ]]
        # (the (3,2) entry -D_psi*k^2 was missing previously, so D_psi
        #  had no effect on the coupled linear step.)
        # M_minus = I - dt/2 * A   (to invert)
        # M_plus  = I + dt/2 * A   (applied to u^n)
        m_minus = torch.stack(
            [
                torch.stack([one + dt2 * cfg.d * k2, -dt2 * one, zero], dim=-1),
                torch.stack([zero, one, -dt2 * one], dim=-1),
                torch.stack([dt2 * k2 / cfg.tau, dt2 * (d_psi * k2 + cfg.mu) / cfg.tau, one + dt2 * (eps + cfg.eta * k2) / cfg.tau], dim=-1),
            ],
            dim=-2,
        ).to(cdtype)  # (nx, ny, 3, 3)

        m_plus = torch.stack(
            [
                torch.stack([one - dt2 * cfg.d * k2, dt2 * one, zero], dim=-1),
                torch.stack([zero, one, dt2 * one], dim=-1),
                torch.stack([-dt2 * k2 / cfg.tau, -dt2 * (d_psi * k2 + cfg.mu) / cfg.tau, one - dt2 * (eps + cfg.eta * k2) / cfg.tau], dim=-1),
            ],
            dim=-2,
        ).to(cdtype)  # (nx, ny, 3, 3)

        # Precompute inverse of m_minus per mode via closed-form 3x3 inverse
        # (batched, avoids torch.linalg.inv dtype/device quirks on DirectML).
        self.m_minus_inv = self._batched_inverse_3x3(m_minus)  # (nx, ny, 3, 3) complex
        self.m_plus = m_plus  # (nx, ny, 3, 3) complex

        # previous-step nonlinear term (for AB2); None until first step done
        self._n_prev_hat: Optional[torch.Tensor] = None
        self.bootstrap_mode = str(cfg.ab2_bootstrap)

    def _linearised_reaction_slope(self, phi0: float) -> float:
        """Numerical a = M'(phi0) via central difference (cnsns_autosoliton_wave, Eq. 18)."""
        h = 1.0e-5
        phi0_t = torch.tensor([phi0 + h, phi0 - h], dtype=self.dtype, device=self.device)
        m_val = nonlinear_m(phi0_t, self.cfg.kappa, self.cfg.theta1, self.cfg.theta2)
        return float((m_val[0] - m_val[1]) / (2.0 * h))

    @staticmethod
    def _batched_inverse_3x3(m: torch.Tensor) -> torch.Tensor:
        """Closed-form inverse of a batch of 3x3 matrices (last two dims)."""
        a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
        d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
        g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        det = torch.where(det.abs() < 1e-30, torch.full_like(det, 1e-30), det)

        inv = torch.empty_like(m)
        inv[..., 0, 0] = (e * i - f * h) / det
        inv[..., 0, 1] = (c * h - b * i) / det
        inv[..., 0, 2] = (b * f - c * e) / det
        inv[..., 1, 0] = (f * g - d * i) / det
        inv[..., 1, 1] = (a * i - c * g) / det
        inv[..., 1, 2] = (c * d - a * f) / det
        inv[..., 2, 0] = (d * h - e * g) / det
        inv[..., 2, 1] = (b * g - a * h) / det
        inv[..., 2, 2] = (a * e - b * d) / det
        return inv

    # ── spectral helpers ──────────────────────────────────────

    def _fft(self, f: torch.Tensor) -> torch.Tensor:
        fh = torch.fft.fft2(f)
        if self.dealias_flag:
            fh = fh * self.dealias_mask
        return fh

    def _ifft(self, fh: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(fh).real

    def _nonlinear_hat(self, phi: torch.Tensor) -> torch.Tensor:
        """FFT of the explicit source term acting on phi's equation: M(phi) + h."""
        src = nonlinear_m(phi, self.cfg.kappa, self.cfg.theta1, self.cfg.theta2) + self.h_field
        return self._fft(src)

    def reset_history(self) -> None:
        self._n_prev_hat = None

    def step(
        self, phi: torch.Tensor, psi: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_hat = self._fft(phi)
        psi_hat = self._fft(psi)
        v_hat = self._fft(v)

        n_hat = self._nonlinear_hat(phi)

        if self._n_prev_hat is None:
            # Bootstrap: IMEX-Euler-consistent extrapolation (no history yet)
            n_star = n_hat
        else:
            # 2nd-order Adams-Bashforth extrapolation to the trapezoidal midpoint
            n_star = 1.5 * n_hat - 0.5 * self._n_prev_hat

        self._n_prev_hat = n_hat

        # u^n stacked as (nx, ny, 3) complex
        u_n = torch.stack([phi_hat, psi_hat, v_hat], dim=-1)  # (nx, ny, 3)

        # rhs = M_plus @ u_n + dt * [n_star, 0, 0]
        rhs = torch.einsum("xyij,xyj->xyi", self.m_plus, u_n)
        rhs[..., 0] = rhs[..., 0] + self.dt * n_star

        u_new = torch.einsum("xyij,xyj->xyi", self.m_minus_inv, rhs)

        phi_hat_new = u_new[..., 0]
        psi_hat_new = u_new[..., 1]
        v_hat_new = u_new[..., 2]

        if self.dealias_flag:
            phi_hat_new = phi_hat_new * self.dealias_mask
            psi_hat_new = psi_hat_new * self.dealias_mask
            v_hat_new = v_hat_new * self.dealias_mask

        phi_new = torch.fft.ifft2(phi_hat_new).real
        psi_new = torch.fft.ifft2(psi_hat_new).real
        v_new = torch.fft.ifft2(v_hat_new).real

        return phi_new, psi_new, v_new


# ─────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────


class Simulator:
    def __init__(self, cfg: SimConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float64
        self.run_id = cfg.resolve_run_id()
        self.out_dir = Path(cfg.out_dir) / self.run_id
        self.snap_dir = self.out_dir / "snapshots_2d"

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        cfg.to_json(str(self.out_dir / "config.json"))

        self.h_field, self.gamma_field = build_spatial_fields(cfg, device, self.dtype)
        np.save(self.out_dir / "h_field.npy", self.h_field.detach().cpu().numpy())
        np.save(self.out_dir / "gamma_field.npy", self.gamma_field.detach().cpu().numpy())

        self.phi, self.psi, self.v = build_initial_state(cfg, device, self.dtype)

        self.stepper = FourierImexStepperV3(cfg, self.h_field, self.gamma_field)

        self.hooks: List[Tuple[int, Callable]] = []

        self.track: Dict[str, List[float]] = {
            "step": [],
            "t": [],
            "mass": [],
            "phi_mean": [],
            "phi_rms": [],
            "phi_max": [],
            "phi_center": [],
            "psi_mean": [],
            "psi_rms": [],
            "psi_maxabs": [],
            "psi_center": [],
            "v_rms": [],
            "v_maxabs": [],
            "v_center": [],
            "cx": [],
            "cy": [],
            "total_energy": [],
            "spectral_peak_k": [],
            "spectral_energy_high": [],
        }

        self.status = "running"
        self.failure_reason: Optional[str] = None
        self.failure_step: Optional[int] = None
        self.failure_time: Optional[float] = None
        self.failure_meta: Dict[str, Any] = {}
        self.n_snaps = 0

        self.max_abs_phi = 0.0
        self.max_abs_psi = 0.0
        self.max_abs_v = 0.0

    def add_hook(self, every_n: int, callback: Callable) -> None:
        self.hooks.append((max(1, int(every_n)), callback))

    def set_failure(self, meta: Dict[str, Any]) -> None:
        self.status = "numerically_unstable"
        self.failure_reason = str(meta.get("reason", "unknown"))
        self.failure_step = int(meta["step"]) if meta.get("step") is not None else None
        self.failure_time = float(meta["time"]) if meta.get("time") is not None else None
        self.failure_meta = dict(meta)

    def check_instability(
        self, step: int, t_value: float,
        phi_np: np.ndarray, psi_np: np.ndarray, v_np: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        if cfg.check_nonfinite:
            if not (np.isfinite(phi_np).all() and np.isfinite(psi_np).all() and np.isfinite(v_np).all()):
                return {
                    "reason": "nonfinite", "step": step, "time": t_value,
                    "max_abs_phi": float(np.nanmax(np.abs(phi_np))),
                    "max_abs_psi": float(np.nanmax(np.abs(psi_np))),
                    "max_abs_v": float(np.nanmax(np.abs(v_np))),
                }
        mp = float(np.max(np.abs(phi_np)))
        ms = float(np.max(np.abs(psi_np)))
        mv = float(np.max(np.abs(v_np)))
        if mp > cfg.max_abs_phi:
            return {"reason": "phi_bound", "step": step, "time": t_value,
                    "max_abs_phi": mp, "max_abs_psi": ms, "max_abs_v": mv}
        if ms > cfg.max_abs_psi:
            return {"reason": "psi_bound", "step": step, "time": t_value,
                    "max_abs_phi": mp, "max_abs_psi": ms, "max_abs_v": mv}
        if mv > cfg.max_abs_v:
            return {"reason": "v_bound", "step": step, "time": t_value,
                    "max_abs_phi": mp, "max_abs_psi": ms, "max_abs_v": mv}
        return None

    def record_track(
        self, step: int, t_value: float,
        phi_np: np.ndarray, psi_np: np.ndarray, v_np: np.ndarray,
    ) -> None:
        threshold = 1.5
        mask = phi_np > threshold
        mass = int(np.sum(mask))

        phi_mean = float(np.mean(phi_np))
        phi_rms = float(np.sqrt(np.mean(phi_np ** 2)))
        phi_max = float(np.max(phi_np))

        psi_mean = float(np.mean(psi_np))
        psi_rms = float(np.sqrt(np.mean(psi_np ** 2)))
        psi_maxabs = float(np.max(np.abs(psi_np)))

        v_rms = float(np.sqrt(np.mean(v_np ** 2)))
        v_maxabs = float(np.max(np.abs(v_np)))

        total_energy = float(0.5 * np.mean(psi_np ** 2 + v_np ** 2))

        phi_hat = np.fft.fft2(phi_np)
        amp = np.abs(phi_hat)
        kx_arr = np.fft.fftfreq(phi_np.shape[0]) * phi_np.shape[0]
        ky_arr = np.fft.fftfreq(phi_np.shape[1]) * phi_np.shape[1]
        K2 = kx_arr[:, None] ** 2 + ky_arr[None, :] ** 2
        amp_no_dc = amp.copy()
        amp_no_dc[0, 0] = 0.0
        idx = np.unravel_index(np.argmax(amp_no_dc), amp_no_dc.shape)
        spectral_peak_k = float(np.sqrt(K2[idx]))

        frac = float(self.cfg.dealias_fraction)
        cutoff_x = frac / 2 * phi_np.shape[0]
        cutoff_y = frac / 2 * phi_np.shape[1]
        hi_mask = (np.abs(kx_arr[:, None]) > cutoff_x) | (np.abs(ky_arr[None, :]) > cutoff_y)
        energy_total = float(np.sum(amp ** 2)) + 1e-30
        spectral_energy_high = float(np.sum(amp[hi_mask] ** 2) / energy_total)

        if mass >= 20:
            coords = np.argwhere(mask)
            weights = phi_np[mask] - threshold
            ws = float(np.sum(weights))
            if np.isfinite(ws) and ws > 0.0:
                cx = float(np.sum(coords[:, 0] * weights) / ws)
                cy = float(np.sum(coords[:, 1] * weights) / ws)
            else:
                cx = cy = float("nan")
        else:
            cx = cy = float("nan")

        ni, nj = self.cfg.nx // 2, self.cfg.ny // 2
        self.track["step"].append(float(step))
        self.track["t"].append(float(t_value))
        self.track["mass"].append(float(mass))
        self.track["phi_mean"].append(phi_mean)
        self.track["phi_rms"].append(phi_rms)
        self.track["phi_max"].append(phi_max)
        self.track["phi_center"].append(float(phi_np[ni, nj]))
        self.track["psi_mean"].append(psi_mean)
        self.track["psi_rms"].append(psi_rms)
        self.track["psi_maxabs"].append(psi_maxabs)
        self.track["psi_center"].append(float(psi_np[ni, nj]))
        self.track["v_rms"].append(v_rms)
        self.track["v_maxabs"].append(v_maxabs)
        self.track["v_center"].append(float(v_np[ni, nj]))
        self.track["cx"].append(cx)
        self.track["cy"].append(cy)
        self.track["total_energy"].append(total_energy)
        self.track["spectral_peak_k"].append(spectral_peak_k)
        self.track["spectral_energy_high"].append(spectral_energy_high)
        extra = {
            "phi_min": float(phi_np.min()), "v_mean": float(v_np.mean()),
            "area_phys_legacy_threshold": float(mass * self.cfg.dx**2),
            "mean_mode_I": float(self.cfg.tau * v_np.mean() + self.cfg.eps * psi_np.mean()),
            "mean_mode_I_derivative_expected": float(-self.cfg.mu * psi_np.mean()),
            "excess_integral_ref": float((phi_np - self.cfg.phi_lo).sum() * self.cfg.dx**2),
        }
        for key, value in extra.items():
            self.track.setdefault(key, []).append(value)


    def save_snapshot(
        self, phi: torch.Tensor, psi: torch.Tensor, v: torch.Tensor,
        t_value: float, suffix: str = "",
    ) -> None:
        label = f"_{suffix}" if suffix else ""
        tc = int(round(t_value * 10.0))
        path = self.snap_dir / f"snapshot2d_{self.run_id}_t{tc:08d}{label}.npz"
        payload: Dict[str, np.ndarray] = {"t": np.asarray(t_value, dtype=np.float64)}
        if self.cfg.snap.save_phi:
            payload["phi"] = phi.detach().cpu().numpy().astype(np.float64)
        if self.cfg.snap.save_psi:
            payload["psi"] = psi.detach().cpu().numpy().astype(np.float64)
        if self.cfg.snap.save_v:
            payload["v"] = v.detach().cpu().numpy().astype(np.float64)
        np.savez_compressed(path, **payload)

    def save_track_csv(self) -> None:
        path = self.out_dir / "track.csv"
        keys = list(self.track.keys())
        if len({len(self.track[k]) for k in keys}) > 1:
            raise ValueError("Track length mismatch; refusing silent CSV truncation")
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for row in zip(*(self.track[k] for k in keys)):
                w.writerow(row)

    def save_result(self, last_step: int, last_t: float, wall_seconds: float) -> Dict[str, Any]:
        tr = self.track
        result = {
            "run_id": self.run_id,
            "solver": SOLVER_VERSION,
            "model_id": self.cfg.resolved_model_id(),
            "effective_config": self.cfg.to_dict(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "linear_diagnostics": self.stepper.linear_diagnostics,
            "nonlinear_rule": "Pointwise reaction on base grid followed by spectral truncation; not exact dealiasing",
            "legacy_metric_warning": "mass is cell count; total_energy is a diagnostic, not a proven conserved energy; spectral_peak_k is legacy index",
            "restart_note": "init.npz_path loads fields only; AB2 history is bootstrapped, not an exact restart",

            "spatial_discretisation": "Fourier pseudospectral",
            "time_integrator": "Fully coupled IMEX (Crank-Nicolson linear + AB2 nonlinear)",
            "wave_time_integrator": "Crank-Nicolson (coupled with phi and psi)",
            "activator_time_integrator": "Crank-Nicolson (coupled with psi, v); AB2 nonlinear reaction",
            "dealiasing": self.cfg.dealias,
            "dealias_fraction": self.cfg.dealias_fraction,
            "dt_cfl_ratio": self.stepper.dt_cfl_ratio,
            # Linear stability diagnostics (cnsns_autosoliton_wave, Sec. 3)
            "linear_a": self.stepper.a_lin,
            "linear_k_c": self.stepper.k_c,
            "linear_k_c2": self.stepper.k_c2,
            "linear_dpsi_stability_bound": self.stepper.dpsi_stability_bound,
            "linear_dpsi_within_stability_bound": self.stepper.dpsi_within_stability_bound,
            "status": self.status,
            "t_total_requested": float(self.cfg.t_total),
            "t_total_reached": float(last_t),
            "last_step": int(last_step),
            "dt": float(self.cfg.dt),
            "dx": float(self.cfg.dx),
            "monitor_every": int(self.cfg.monitor_every),
            "_n_snaps": int(self.n_snaps),
            "wallsec": round(float(wall_seconds), 3),
            "finalmass": int(tr["mass"][-1]) if tr["mass"] else 0,
            "terminated_early": self.status != "completed",
            "failure_reason": self.failure_reason,
            "failure_step": self.failure_step,
            "failure_time": self.failure_time,
            "failure_meta": self.failure_meta,
            "max_abs_phi": float(self.max_abs_phi),
            "max_abs_psi": float(self.max_abs_psi),
            "max_abs_v": float(self.max_abs_v),
            "peak_abs_phi": float(np.nanmax(tr["phi_max"]) if tr["phi_max"] else float("nan")),
            "peak_abs_psi": float(np.nanmax(tr["psi_maxabs"]) if tr["psi_maxabs"] else float("nan")),
            "peak_abs_v": float(np.nanmax(tr["v_maxabs"]) if tr["v_maxabs"] else float("nan")),
            "mean_total_energy": float(np.nanmean(tr["total_energy"]) if tr["total_energy"] else float("nan")),
            "final_total_energy": float(tr["total_energy"][-1] if tr["total_energy"] else float("nan")),
            "mean_spectral_energy_high": float(np.nanmean(tr["spectral_energy_high"]) if tr["spectral_energy_high"] else float("nan")),
            "final_spectral_peak_k": float(tr["spectral_peak_k"][-1] if tr["spectral_peak_k"] else float("nan")),
        }
        with (self.out_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        total_steps = int(round(cfg.t_total / cfg.dt))
        snap_start = int(np.ceil(cfg.snap.t_start / cfg.dt))
        snap_stop = int(np.floor(cfg.snap.t_stop / cfg.dt)) if cfg.snap.t_stop >= 0.0 else total_steps

        start_wall = time.time()
        phi, psi, v = self.phi, self.psi, self.v
        last_step = last_t = 0
        aborted = False

        logger.info(
            "%s v3 start | steps=%d nx=%d ny=%d dt=%g dx=%g dealias=%s",
            self.run_id, total_steps, cfg.nx, cfg.ny, cfg.dt, cfg.dx, cfg.dealias,
        )
        logger.info(
            "%s v3 linear-stability | a=%.4f k_c=%.4f (k_c^2=%.4f) D_psi_bound=%.4f within_bound=%s",
            self.run_id, self.stepper.a_lin, self.stepper.k_c, self.stepper.k_c2,
            self.stepper.dpsi_stability_bound, self.stepper.dpsi_within_stability_bound,
        )
        if not self.stepper.dpsi_within_stability_bound:
            logger.warning(
                "%s: D_psi=%.4g exceeds the Routh-Hurwitz bound (1+eps)/eps=%.4g "
                "for the homogeneous background (cnsns_autosoliton_wave, Eq. 28); "
                "the background state used for k_c is not linearly stable at any k.",
                self.run_id, cfg.D_psi, self.stepper.dpsi_stability_bound,
            )

        for step in range(total_steps + 1):
            t_value = step * cfg.dt
            last_step = step
            last_t = t_value

            should_monitor = (step % max(1, cfg.monitor_every) == 0 or step == total_steps)

            if should_monitor:
                phi_np = phi.detach().cpu().numpy()
                psi_np = psi.detach().cpu().numpy()
                v_np = v.detach().cpu().numpy()

                self.record_track(step, t_value, phi_np, psi_np, v_np)

                self.max_abs_phi = max(self.max_abs_phi, float(np.max(np.abs(phi_np))))
                self.max_abs_psi = max(self.max_abs_psi, float(np.max(np.abs(psi_np))))
                self.max_abs_v = max(self.max_abs_v, float(np.max(np.abs(v_np))))

                failure = self.check_instability(step, t_value, phi_np, psi_np, v_np)
                if failure is not None:
                    self.set_failure(failure)
                    logger.warning("%s: failure=%s", self.run_id, failure)
                    try:
                        self.save_snapshot(phi, psi, v, t_value, suffix="unstable_last")
                    except Exception as exc:
                        logger.warning("%s: could not save failure snapshot: %r", self.run_id, exc)
                    if cfg.abort_on_instability:
                        aborted = True
                        break

            in_window = snap_start <= step <= snap_stop
            on_stride = step % max(1, cfg.snap.every_steps) == 0
            below_lim = cfg.snap.max_snaps < 0 or self.n_snaps < cfg.snap.max_snaps
            if in_window and on_stride and below_lim:
                self.save_snapshot(phi, psi, v, t_value)
                self.n_snaps += 1

            if step == total_steps:
                break

            for every_n, cb in self.hooks:
                if step % every_n == 0:
                    cb(step, t_value, phi, psi, v, self)

            phi, psi, v = self.stepper.step(phi, psi, v)

            if step % 5000 == 0 and step > 0:
                elapsed = time.time() - start_wall
                fraction = step / max(total_steps, 1)
                eta = elapsed * (1.0 - fraction) / max(fraction, 1.0e-12)
                logger.info(
                    "%s: %.1f%% t=%.4f/%.4f elapsed=%.1fs eta=%.1fs",
                    self.run_id, 100.0 * fraction, t_value, cfg.t_total, elapsed, eta,
                )

        self.phi, self.psi, self.v = phi, psi, v

        if not aborted and self.status == "running":
            self.status = "completed"
            self.save_snapshot(phi, psi, v, last_t, suffix="final")

        self.save_track_csv()
        wall_seconds = time.time() - start_wall
        result = self.save_result(last_step, last_t, wall_seconds)

        logger.info(
            "%s: done status=%s t_reached=%g wall=%.2fs",
            self.run_id, self.status, last_t, wall_seconds,
        )

        return result


# ─────────────────────────────────────────────────────────────
# Device selection & CLI (unchanged API)
# ─────────────────────────────────────────────────────────────


def select_device(device_name: Optional[str]) -> Tuple[torch.device, str]:
    if device_name:
        return torch.device(device_name), device_name
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if has_directml:
        return torch_directml.device(), "directml"
    return torch.device("cpu"), "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fourier-IMEX v3 solver: fully coupled (phi, psi, v) Crank-Nicolson + AB2 nonlinear."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config (v1/v2-compatible).")
    parser.add_argument("--device", default=None, help="Torch device (cpu/cuda).")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = build_parser().parse_args()
    cfg = SimConfig.from_json(args.config)

    if cfg.boundary.lower() != "periodic":
        raise ValueError("The Fourier-IMEX v3 solver supports only boundary='periodic'.")

    device, backend = select_device(args.device)
    logger.info("backend=%s device=%s", backend, device)

    sim = Simulator(cfg, device)
    result = sim.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
