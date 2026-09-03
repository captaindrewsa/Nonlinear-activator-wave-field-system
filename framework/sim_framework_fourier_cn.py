#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import sys
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


logger = logging.getLogger("sim_framework_fourier_cn")

kappa_default = 10.0
theta_1_default = 4.0
theta_2_default = 16.0
phi_lo_default = 0.533
phi_hi_default = 3.9766096853487105


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
        payload["spots"] = [
            InitSpot.from_dict(item) for item in payload.get("spots", [])
        ]
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
    out_dir: str = "./output_fourier_cn"
    monitor_every: int = 10
    t_warm: float = 0.0
    seed_n_snaps: int = 16
    check_nonfinite: bool = True
    abort_on_instability: bool = True
    max_abs_phi: float = 1.0e3
    max_abs_psi: float = 1.0e3
    max_abs_v: float = 1.0e3
    run_id: str = ""
    tags: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SimConfig":
        payload = copy.deepcopy(data)
        payload["attractors"] = [
            AttractorSpec.from_dict(item)
            for item in payload.get("attractors", [])
        ]
        payload["init"] = InitConfig.from_dict(payload.get("init", {}))
        payload["snap"] = SnapshotPolicy.from_dict(payload.get("snap", {}))
        return SimConfig(**payload)

    @staticmethod
    def from_json(path: str) -> "SimConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return SimConfig.from_dict(json.load(handle))

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
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return "fourier_cn_" + hashlib.md5(encoded).hexdigest()[:12]


def sigma(u: torch.Tensor, kappa: float, theta: float) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-kappa * (u - theta)))


def nonlinear_m(
    phi: torch.Tensor,
    kappa: float,
    theta1: float,
    theta2: float,
) -> torch.Tensor:
    phi2 = phi * phi
    return phi * sigma(phi2, kappa, theta1) * (1.0 - sigma(phi2, kappa, theta2)) - phi


def gaussian_2d(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    sigma_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    return torch.exp(
        -0.5
        * (
            ((x - cx) / sigma_value) ** 2
            + ((y - cy) / sigma_value) ** 2
        )
    )


def disk_2d(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    return (((x - cx) ** 2 + (y - cy) ** 2) <= radius**2).to(dtype)


def tanh_disk_2d(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    interface_width: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    radius_field = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    width = max(float(interface_width), 1.0e-8)
    return 0.5 * (1.0 - torch.tanh((radius_field - radius) / width))


def ring_2d(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    ring_r: float,
    ring_w: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    radius_field = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return torch.exp(-0.5 * ((radius_field - ring_r) / ring_w) ** 2)


def build_spatial_fields(
    cfg: SimConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_field = torch.full(
        (cfg.nx, cfg.ny),
        cfg.h_bg,
        dtype=dtype,
        device=device,
    )
    gamma_field = torch.full(
        (cfg.nx, cfg.ny),
        cfg.gamma_bg,
        dtype=dtype,
        device=device,
    )

    if cfg.h_custom:
        h_path = Path(cfg.h_custom)
        if h_path.exists():
            h_field = torch.from_numpy(
                np.load(h_path).astype(np.float64)
            ).to(device=device, dtype=dtype)

    if cfg.gamma_custom:
        gamma_path = Path(cfg.gamma_custom)
        if gamma_path.exists():
            gamma_field = torch.from_numpy(
                np.load(gamma_path).astype(np.float64)
            ).to(device=device, dtype=dtype)

    for item in cfg.attractors:
        if item.profile == "gaussian":
            mask = gaussian_2d(
                cfg.nx,
                cfg.ny,
                item.cx,
                item.cy,
                item.sigma,
                device,
                dtype,
            )
        elif item.profile == "disk":
            mask = disk_2d(
                cfg.nx,
                cfg.ny,
                item.cx,
                item.cy,
                item.sigma,
                device,
                dtype,
            )
        elif item.profile == "ring":
            mask = ring_2d(
                cfg.nx,
                cfg.ny,
                item.cx,
                item.cy,
                item.ring_r,
                item.ring_w,
                device,
                dtype,
            )
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


def build_initial_state(
    cfg: SimConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.init.npz_path:
        path = Path(cfg.init.npz_path)
        if path.exists():
            with np.load(path) as data:
                phi = torch.from_numpy(
                    np.asarray(data["phi"], dtype=np.float64)
                ).to(device=device, dtype=dtype)
                psi = torch.from_numpy(
                    np.asarray(
                        data["psi"] if "psi" in data else np.zeros_like(data["phi"]),
                        dtype=np.float64,
                    )
                ).to(device=device, dtype=dtype)
                v = torch.from_numpy(
                    np.asarray(
                        data["v"] if "v" in data else np.zeros_like(data["phi"]),
                        dtype=np.float64,
                    )
                ).to(device=device, dtype=dtype)
                return phi, psi, v

    background = (
        cfg.phi_lo
        if cfg.init.phi_background is None
        else float(cfg.init.phi_background)
    )

    phi = torch.full(
        (cfg.nx, cfg.ny),
        background,
        dtype=dtype,
        device=device,
    )
    psi = torch.zeros_like(phi)
    v = torch.zeros_like(phi)

    for spot in cfg.init.spots:
        amplitude = cfg.phi_hi if spot.amp is None else float(spot.amp)

        if spot.shape == "disk":
            mask = disk_2d(
                cfg.nx,
                cfg.ny,
                spot.cx,
                spot.cy,
                spot.radius,
                device,
                dtype,
            )
        elif spot.shape == "tanh_disk":
            mask = tanh_disk_2d(
                cfg.nx,
                cfg.ny,
                spot.cx,
                spot.cy,
                spot.radius,
                spot.interface_width,
                device,
                dtype,
            )
        elif spot.shape == "gaussian":
            mask = gaussian_2d(
                cfg.nx,
                cfg.ny,
                spot.cx,
                spot.cy,
                spot.radius,
                device,
                dtype,
            )
        else:
            raise ValueError(f"Unknown initial spot shape: {spot.shape}")

        phi = phi + (amplitude - background) * mask

        if spot.phase_v != 0.0:
            v = v + float(spot.phase_v) * gaussian_2d(
                cfg.nx,
                cfg.ny,
                spot.cx,
                spot.cy,
                spot.radius,
                device,
                dtype,
            )

    if cfg.init.noise_amplitude > 0.0:
        rng = np.random.default_rng(cfg.init.noise_seed)
        noise = rng.standard_normal((cfg.nx, cfg.ny)).astype(np.float64)
        noise *= float(cfg.init.noise_amplitude)
        phi = phi + torch.from_numpy(noise).to(device=device, dtype=dtype)

    return phi, psi, v


class FourierCnStepper:
    """
    Fourier spatial discretisation with a Crank--Nicolson wave-field step.

    The activator reaction equation is advanced explicitly:
        phi^{n+1} = phi^n + dt [M(phi^n) + h + psi^n].

    The wave-field pair uses Crank--Nicolson:
        (psi^{n+1} - psi^n)/dt = (v^{n+1} + v^n)/2,
        (v^{n+1} - v^n)/dt =
            D_psi Delta[(psi^{n+1} + psi^n)/2]
            + Delta phi^n
            - eps (v^{n+1} + v^n)/2.

    With periodic boundaries, each Fourier mode is a 2x2 system that
    is solved directly without assembling a real-space sparse matrix.
    """

    def __init__(
        self,
        cfg: SimConfig,
        h_field: torch.Tensor,
        gamma_field: torch.Tensor,
    ) -> None:
        if cfg.boundary.lower() != "periodic":
            raise ValueError(
                "FourierCnStepper requires boundary='periodic'."
            )

        self.cfg = cfg
        self.h_field = h_field
        self.gamma_field = gamma_field
        self.dtype = h_field.dtype
        self.device = h_field.device

        kx = 2.0 * np.pi * torch.fft.fftfreq(
            cfg.nx,
            d=cfg.dx,
            device=self.device,
            dtype=self.dtype,
        ).view(-1, 1)
        ky = 2.0 * np.pi * torch.fft.fftfreq(
            cfg.ny,
            d=cfg.dx,
            device=self.device,
            dtype=self.dtype,
        ).view(1, -1)

        self.k2 = kx * kx + ky * ky
        self.dt = float(cfg.dt)
        self.half_dt = 0.5 * self.dt

        dpsi = float(cfg.D_psi)
        eps = float(cfg.eps)

        # CN system:
        #
        # psi_new - (dt/2) v_new = psi_old + (dt/2) v_old
        #
        # (1 + eps dt/2) v_new + (dt/2) D_psi k^2 psi_new
        #   = (1 - eps dt/2) v_old
        #     - (dt/2) D_psi k^2 psi_old
        #     - dt k^2 phi_hat_old
        #
        # A11=1, A12=-dt/2
        # A21=(dt/2) D k^2, A22=1+eps dt/2
        # det=A22+(dt^2/4)D k^2
        self.a22 = torch.full_like(
            self.k2,
            1.0 + self.half_dt * eps,
        )
        self.a21 = self.half_dt * dpsi * self.k2
        self.det = self.a22 + self.half_dt * self.a21

        self.b22 = 1.0 - self.half_dt * eps
        self.b21 = -self.half_dt * dpsi * self.k2

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        field_hat = torch.fft.fft2(field)
        lap_hat = -self.k2 * field_hat
        return torch.fft.ifft2(lap_hat).real

    def step(
        self,
        phi: torch.Tensor,
        psi: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        effective_eps = cfg.eps + self.gamma_field
        if torch.any(self.gamma_field != 0):
            raise ValueError(
                "Spatially varying gamma_field is not supported by "
                "the Fourier-CN linear wave step. Set gamma_bg=0 and "
                "use no sink attractors for this solver."
            )
        if abs(float(effective_eps.flatten()[0]) - cfg.eps) > 1e-14:
            raise ValueError(
                "Only spatially uniform damping eps is supported."
            )

        phi_new = phi + self.dt * (
            nonlinear_m(phi, cfg.kappa, cfg.theta1, cfg.theta2)
            + self.h_field
            + psi
        )

        phi_hat = torch.fft.fft2(phi)
        psi_hat = torch.fft.fft2(psi)
        v_hat = torch.fft.fft2(v)

        rhs_1 = psi_hat + self.half_dt * v_hat
        rhs_2 = (
            self.b22 * v_hat
            + self.b21 * psi_hat
            - self.dt * self.k2 * phi_hat
        )

        psi_hat_new = (
            self.a22 * rhs_1 + self.half_dt * rhs_2
        ) / self.det

        v_hat_new = (
            -self.a21 * rhs_1 + rhs_2
        ) / self.det

        psi_new = torch.fft.ifft2(psi_hat_new).real
        v_new = torch.fft.ifft2(v_hat_new).real

        return phi_new, psi_new, v_new


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

        self.h_field, self.gamma_field = build_spatial_fields(
            cfg,
            device,
            self.dtype,
        )
        np.save(
            self.out_dir / "h_field.npy",
            self.h_field.detach().cpu().numpy(),
        )
        np.save(
            self.out_dir / "gamma_field.npy",
            self.gamma_field.detach().cpu().numpy(),
        )

        self.phi, self.psi, self.v = build_initial_state(
            cfg,
            device,
            self.dtype,
        )

        self.stepper = FourierCnStepper(
            cfg,
            self.h_field,
            self.gamma_field,
        )

        self.hooks: List[
            Tuple[int, Callable[[int, float, torch.Tensor, torch.Tensor, torch.Tensor, "Simulator"], None]]
        ] = []

        self.track: Dict[str, List[float]] = {
            "step": [],
            "t": [],
            "mass": [],
            "phi_center": [],
            "psi_center": [],
            "v_center": [],
            "cx": [],
            "cy": [],
            "phi_max": [],
            "psi_maxabs": [],
            "v_maxabs": [],
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

    def add_hook(
        self,
        every_n: int,
        callback: Callable[
            [int, float, torch.Tensor, torch.Tensor, torch.Tensor, "Simulator"],
            None,
        ],
    ) -> None:
        self.hooks.append((max(1, int(every_n)), callback))

    def set_failure(self, meta: Dict[str, Any]) -> None:
        self.status = "numerically_unstable"
        self.failure_reason = str(meta.get("reason", "unknown"))
        self.failure_step = int(meta["step"]) if meta.get("step") is not None else None
        self.failure_time = float(meta["time"]) if meta.get("time") is not None else None
        self.failure_meta = dict(meta)

    def check_instability(
        self,
        step: int,
        t_value: float,
        phi_np: np.ndarray,
        psi_np: np.ndarray,
        v_np: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg

        if cfg.check_nonfinite:
            if not (
                np.isfinite(phi_np).all()
                and np.isfinite(psi_np).all()
                and np.isfinite(v_np).all()
            ):
                return {
                    "reason": "nonfinite",
                    "step": step,
                    "time": t_value,
                    "max_abs_phi": float(np.nanmax(np.abs(phi_np))),
                    "max_abs_psi": float(np.nanmax(np.abs(psi_np))),
                    "max_abs_v": float(np.nanmax(np.abs(v_np))),
                }

        max_phi = float(np.max(np.abs(phi_np)))
        max_psi = float(np.max(np.abs(psi_np)))
        max_v = float(np.max(np.abs(v_np)))

        if max_phi > cfg.max_abs_phi:
            return {
                "reason": "phi_bound",
                "step": step,
                "time": t_value,
                "max_abs_phi": max_phi,
                "max_abs_psi": max_psi,
                "max_abs_v": max_v,
            }

        if max_psi > cfg.max_abs_psi:
            return {
                "reason": "psi_bound",
                "step": step,
                "time": t_value,
                "max_abs_phi": max_phi,
                "max_abs_psi": max_psi,
                "max_abs_v": max_v,
            }

        if max_v > cfg.max_abs_v:
            return {
                "reason": "v_bound",
                "step": step,
                "time": t_value,
                "max_abs_phi": max_phi,
                "max_abs_psi": max_psi,
                "max_abs_v": max_v,
            }

        return None

    def record_track(
        self,
        step: int,
        t_value: float,
        phi_np: np.ndarray,
        psi_np: np.ndarray,
        v_np: np.ndarray,
    ) -> None:
        threshold = 1.5
        mask = phi_np > threshold
        mass = int(np.sum(mask))

        self.track["step"].append(float(step))
        self.track["t"].append(float(t_value))
        self.track["mass"].append(float(mass))
        self.track["phi_center"].append(
            float(phi_np[self.cfg.nx // 2, self.cfg.ny // 2])
        )
        self.track["psi_center"].append(
            float(psi_np[self.cfg.nx // 2, self.cfg.ny // 2])
        )
        self.track["v_center"].append(
            float(v_np[self.cfg.nx // 2, self.cfg.ny // 2])
        )
        self.track["phi_max"].append(float(np.max(phi_np)))
        self.track["psi_maxabs"].append(float(np.max(np.abs(psi_np))))
        self.track["v_maxabs"].append(float(np.max(np.abs(v_np))))

        if mass >= 20:
            coords = np.argwhere(mask)
            weights = phi_np[mask] - threshold
            weight_sum = float(np.sum(weights))
            if np.isfinite(weight_sum) and weight_sum > 0.0:
                cx = float(np.sum(coords[:, 0] * weights) / weight_sum)
                cy = float(np.sum(coords[:, 1] * weights) / weight_sum)
            else:
                cx = float("nan")
                cy = float("nan")
        else:
            cx = float("nan")
            cy = float("nan")

        self.track["cx"].append(cx)
        self.track["cy"].append(cy)

    def save_snapshot(
        self,
        phi: torch.Tensor,
        psi: torch.Tensor,
        v: torch.Tensor,
        t_value: float,
        suffix: str = "",
    ) -> None:
        label = f"_{suffix}" if suffix else ""
        time_code = int(round(t_value * 10.0))
        path = self.snap_dir / (
            f"snapshot2d_{self.run_id}_t{time_code:08d}{label}.npz"
        )

        payload: Dict[str, np.ndarray] = {
            "t": np.asarray(t_value, dtype=np.float64),
        }

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

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(keys)
            for row in zip(*(self.track[key] for key in keys)):
                writer.writerow(row)

    def save_result(
        self,
        last_step: int,
        last_t: float,
        wall_seconds: float,
    ) -> Dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "solver": "fourier_cn",
            "spatial_discretisation": "Fourier pseudospectral",
            "wave_time_integrator": "Crank-Nicolson",
            "activator_time_integrator": "explicit Euler",
            "status": self.status,
            "t_total_requested": float(self.cfg.t_total),
            "t_total_reached": float(last_t),
            "last_step": int(last_step),
            "dt": float(self.cfg.dt),
            "dx": float(self.cfg.dx),
            "monitor_every": int(self.cfg.monitor_every),
            "_n_snaps": int(self.n_snaps),
            "wallsec": round(float(wall_seconds), 3),
            "finalmass": (
                int(self.track["mass"][-1])
                if self.track["mass"]
                else 0
            ),
            "terminated_early": bool(self.status != "completed"),
            "failure_reason": self.failure_reason,
            "failure_step": self.failure_step,
            "failure_time": self.failure_time,
            "failure_meta": self.failure_meta,
            "max_abs_phi": float(self.max_abs_phi),
            "max_abs_psi": float(self.max_abs_psi),
            "max_abs_v": float(self.max_abs_v),
            "peak_abs_phi": float(
                np.nanmax(self.track["phi_max"])
                if self.track["phi_max"]
                else float("nan")
            ),
            "peak_abs_psi": float(
                np.nanmax(self.track["psi_maxabs"])
                if self.track["psi_maxabs"]
                else float("nan")
            ),
            "peak_abs_v": float(
                np.nanmax(self.track["v_maxabs"])
                if self.track["v_maxabs"]
                else float("nan")
            ),
        }

        with (self.out_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

        return result

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        total_steps = int(round(cfg.t_total / cfg.dt))
        snap_start_step = int(np.ceil(cfg.snap.t_start / cfg.dt))
        snap_stop_step = (
            int(np.floor(cfg.snap.t_stop / cfg.dt))
            if cfg.snap.t_stop >= 0.0
            else total_steps
        )

        start_wall = time.time()
        phi = self.phi
        psi = self.psi
        v = self.v

        last_step = 0
        last_t = 0.0
        aborted = False

        logger.info(
            "%s: Fourier-CN start, steps=%d, nx=%d, ny=%d, dt=%g",
            self.run_id,
            total_steps,
            cfg.nx,
            cfg.ny,
            cfg.dt,
        )

        for step in range(total_steps + 1):
            t_value = step * cfg.dt
            last_step = step
            last_t = t_value

            should_monitor = (
                step % max(1, cfg.monitor_every) == 0
                or step == total_steps
            )

            if should_monitor:
                phi_np = phi.detach().cpu().numpy()
                psi_np = psi.detach().cpu().numpy()
                v_np = v.detach().cpu().numpy()

                self.record_track(step, t_value, phi_np, psi_np, v_np)

                self.max_abs_phi = max(
                    self.max_abs_phi,
                    float(np.max(np.abs(phi_np))),
                )
                self.max_abs_psi = max(
                    self.max_abs_psi,
                    float(np.max(np.abs(psi_np))),
                )
                self.max_abs_v = max(
                    self.max_abs_v,
                    float(np.max(np.abs(v_np))),
                )

                failure = self.check_instability(
                    step,
                    t_value,
                    phi_np,
                    psi_np,
                    v_np,
                )

                if failure is not None:
                    self.set_failure(failure)
                    logger.warning("%s: failure=%s", self.run_id, failure)

                    try:
                        self.save_snapshot(
                            phi,
                            psi,
                            v,
                            t_value,
                            suffix="unstable_last",
                        )
                    except Exception as exc:
                        logger.warning(
                            "%s: could not save failure snapshot: %r",
                            self.run_id,
                            exc,
                        )

                    if cfg.abort_on_instability:
                        aborted = True
                        break

            save_window = snap_start_step <= step <= snap_stop_step
            save_stride = step % max(1, cfg.snap.every_steps) == 0
            below_limit = (
                cfg.snap.max_snaps < 0
                or self.n_snaps < cfg.snap.max_snaps
            )

            if save_window and save_stride and below_limit:
                self.save_snapshot(phi, psi, v, t_value)
                self.n_snaps += 1

            if step == total_steps:
                break

            for every_n, callback in self.hooks:
                if step % every_n == 0:
                    callback(step, t_value, phi, psi, v, self)

            phi, psi, v = self.stepper.step(phi, psi, v)

            if step % 5000 == 0 and step > 0:
                elapsed = time.time() - start_wall
                fraction = step / max(total_steps, 1)
                eta = elapsed * (1.0 - fraction) / max(fraction, 1.0e-12)
                logger.info(
                    "%s: %.1f%%, t=%.4f/%0.4f, elapsed=%.1fs, eta=%.1fs",
                    self.run_id,
                    100.0 * fraction,
                    t_value,
                    cfg.t_total,
                    elapsed,
                    eta,
                )

        self.phi = phi
        self.psi = psi
        self.v = v

        if not aborted and self.status == "running":
            self.status = "completed"
            self.save_snapshot(phi, psi, v, last_t, suffix="final")

        self.save_track_csv()
        wall_seconds = time.time() - start_wall
        result = self.save_result(last_step, last_t, wall_seconds)

        logger.info(
            "%s: done, status=%s, reached t=%g in %.2fs",
            self.run_id,
            self.status,
            last_t,
            wall_seconds,
        )

        return result


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
        description=(
            "Periodic Fourier pseudospectral solver with a "
            "Crank--Nicolson wave-field update."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON simulation configuration.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu or cuda.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = build_parser().parse_args()
    cfg = SimConfig.from_json(args.config)

    if cfg.boundary.lower() != "periodic":
        raise ValueError(
            "The Fourier-CN solver supports only boundary='periodic'."
        )

    device, backend = select_device(args.device)
    logger.info("backend=%s, device=%s", backend, device)

    simulator = Simulator(cfg, device)
    result = simulator.run()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()