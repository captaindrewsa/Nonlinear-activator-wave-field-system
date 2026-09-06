#!/usr/bin/env python3
"""
sim_framework_v1.py
================================================================================
Simulation Framework v1 — универсальная основа для симуляций активаторно-
волнового поля.

УРАВНЕНИЯ (неизменны):
    M(phi) = phi * sigma1(phi^2) * (1 - sigma2(phi^2)) - phi
    d_t phi  = M(phi) + h_field(x,y) + psi
    d_tt psi + eps * d_t psi = D_psi * Lap(psi) + Lap(phi)

    Вспомогательная переменная v = d_t psi:
        phi  <- phi + dt * (M(phi) + h_field + psi)
        v    <- v   + dt * (Lap(phi) + DW*Lap(psi) - (eps + gamma_field)*v)
        psi  <- psi + dt * v

НОВЫЕ ВОЗМОЖНОСТИ vs glider_search_v24_rich.py:
    1. SimConfig  — единый датакласс для ВСЕХ параметров симуляции.
    2. Пространственные поля h_field(x,y) и gamma_field(x,y):
         - фоновые константы
         - суперпозиция гауссовых «источников» (pump) и «стоков» (sink)
         - произвольный пользовательский .npy-массив
    3. AttractorSpec — декларативное описание аттракторов:
         pump   : локально повышает h_field  → аттрактор CORE
         sink   : локально повышает gamma    → диссипативная ловушка
         h_sink : понижает h_field           → репеллер
    4. InitConfig — гибкое задание начальных условий:
         - spots (disk / gaussian) с координатами, радиусами, амплитудой, kick
         - случайный шум (белый или low-pass)
         - загрузка из .npz snapshot
    5. SnapshotPolicy — гибкая политика сохранения 2D-снимков.
    6. Simulator  — главный цикл с хуками (add_hook).
    7. OnlineAnalyzer — готовые хуки: space-time, PNG-превью.
    8. ParallelRunner — последовательный и multiprocess (CPU) режимы.
    9. SeedBank  — кеш прогретых пятен (совместим с v24).
   10. CLI: --example | --config | --list | --export_config

СТРУКТУРА:
    I.   Импорты и константы модели
    II.  SimConfig, AttractorSpec, InitConfig, SnapshotPolicy
    III. PDE helpers
    IV.  SpatialFields
    V.   InitialConditions
    VI.  SeedBank
    VII. Stepper
    VIII.Simulator
    IX.  OnlineAnalyzer
    X.   ParallelRunner
    XI.  Примеры конфигураций
    XII. Device selection & main()
================================================================================
"""
from __future__ import annotations
import os, sys, json, csv, time, hashlib, copy, logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from pathlib import Path

import sys
import time

import numpy as np
import torch

try:
    import torch_directml
    HAS_DML = True
except Exception:
    HAS_DML = False

logger = logging.getLogger("sim_framework")

# =============================================================================
# I.  КОНСТАНТЫ МОДЕЛИ
# =============================================================================
KAPPA   = 10.0
THETA1  = 4.0
THETA2  = 16.0
PHI_LO_DEFAULT = 0.533
PHI_HI_DEFAULT = 3.9766096853487105

# =============================================================================
# II.  DATACLASSES КОНФИГУРАЦИИ
# =============================================================================

@dataclass
class AttractorSpec:
    """Один аттрактор (источник или сток) в пространстве.

    kind      : "pump"   — повышает h_field  (аттрактор CORE)
                "sink"   — повышает gamma    (диссипативная ловушка)
                "h_sink" — понижает h_field  (репеллер)
    cx, cy    : центр (решёточные единицы, float)
    strength  : амплитуда (>0)
    sigma     : ширина (для gaussian/disk)
    profile   : "gaussian" | "disk" | "ring"
    ring_r    : радиус кольца  (только profile="ring")
    ring_w    : полуширина кольца
    """
    kind:     str   = "pump"
    cx:       float = 80.0
    cy:       float = 80.0
    strength: float = 0.05
    sigma:    float = 8.0
    profile:  str   = "gaussian"
    ring_r:   float = 20.0
    ring_w:   float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AttractorSpec":
        return AttractorSpec(**d)


@dataclass
class InitSpot:
    """Одно начальное пятно.

    cx, cy  : центр
    radius  : радиус диска / sigma для gaussian
    amp     : значение phi (None → phi_hi)
    shape   : "disk" | "gaussian"
    phase_v : начальный kick в v-поле
    """
    cx:      float = 80.0
    cy:      float = 80.0
    radius:  float = 8.0
    amp:     Optional[float] = None
    shape:   str   = "disk"
    phase_v: float = 0.0
    interface_width: float = 1.0



@dataclass
class InitConfig:
    """Полное описание начальных условий.

    Приоритет: npz_path > spots + noise.
    """
    npz_path:        Optional[str]    = None
    spots:           List[InitSpot]   = field(default_factory=list)
    noise_amplitude: float            = 0.0
    noise_seed:      int              = 42
    noise_lowpass:   float            = 0.0
    phi_background:  Optional[float]  = None   # None -> phi_lo


@dataclass
class SnapshotPolicy:
    """Политика сохранения 2D-снимков.

    every_steps : каждые N шагов
    t_start     : не сохранять до этого t
    t_stop      : не сохранять после этого t (-1 -> до конца)
    save_phi / save_psi / save_v : какие поля пишем
    max_snaps   : максимум снимков (-1 -> без ограничения)
    labels_save : для каких label классификатора сохранять (пусто -> для всех)
    """
    every_steps: int       = 2000
    t_start:     float     = 0.0
    t_stop:      float     = -1.0
    save_phi:    bool      = True
    save_psi:    bool      = True
    save_v:      bool      = False
    max_snaps:   int       = -1
    labels_save: List[str] = field(default_factory=list)


@dataclass
class SimConfig:
    """Главный конфиг симуляции.

    Физика:
        eps       : диссипация волны
        h_bg      : фоновый порог активатора
        D_psi     : пространственная дисперсия psi
        phi_lo, phi_hi : стационарные уровни phi

    Геометрия:
        nx, ny    : размер сетки
        dx        : шаг сетки
        boundary  : "periodic" | "neumann"

    Численный метод:
        dt        : шаг времени
        t_total   : полное время прогона

    Пространственные поля:
        attractors   : список AttractorSpec
        gamma_bg     : фоновое однородное демпфирование
        h_custom     : путь к .npy с пользовательским h_field
        gamma_custom : путь к .npy с пользовательским gamma_field

    Начальные условия:
        init         : InitConfig

    Вывод:
        snap         : SnapshotPolicy
        out_dir      : директория вывода

    Мониторинг:
        monitor_every : каждые N шагов снимать метрики
        t_warm        : время прогрева seed-пятна
        seed_n_snaps  : число срезов цикла в seed bank

    Метаданные:
        run_id        : идентификатор (генерируется из конфига если пусто)
        tags          : произвольные теги
    """
    # Физика
    eps:    float = 2.8
    h_bg:   float = 0.533
    D_psi:  float = 0.0
    phi_lo: float = PHI_LO_DEFAULT
    phi_hi: float = PHI_HI_DEFAULT

    # Параметры сигмоиды
    kappa: float = 10.0
    theta1: float = 4.0
    theta2: float = 16.0

    # Геометрия
    nx:       int   = 160
    ny:       int   = 160
    dx:       float = 1.0
    boundary: str   = "periodic"

    # Численный метод
    dt:      float = 0.003
    t_total: float = 1400.0

    # Пространственные поля
    attractors:   List[AttractorSpec] = field(default_factory=list)
    gamma_bg:     float               = 0.0
    h_custom:     Optional[str]       = None
    gamma_custom: Optional[str]       = None

    # Начальные условия
    init: InitConfig = field(default_factory=InitConfig)

    # Вывод
    snap:    SnapshotPolicy = field(default_factory=SnapshotPolicy)
    out_dir: str            = "output_sim_framework"

    # Мониторинг
    monitor_every: int   = 10
    t_warm:        float = 120.0
    seed_n_snaps:  int   = 16

    # Numerical safety / early abort
    check_nonfinite: bool = True
    abort_on_instability: bool = True
    max_abs_phi: float = 1.0e3
    max_abs_psi: float = 1.0e3
    max_abs_v: float = 1.0e3

    # Метаданные
    run_id: str       = ""
    tags:   List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def resolve_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        key = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return "run_" + hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    @staticmethod
    def from_dict(d: dict) -> "SimConfig":
        d = copy.deepcopy(d)
        d["attractors"] = [AttractorSpec(**a) for a in d.get("attractors", [])]
        ic = d.get("init", {})
        ic["spots"] = [InitSpot(**s) for s in ic.get("spots", [])]
        d["init"] = InitConfig(**ic)
        d["snap"] = SnapshotPolicy(**d.get("snap", {}))
        return SimConfig(**d)

    @staticmethod
    def from_json(path: str) -> "SimConfig":
        return SimConfig.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    @staticmethod
    def suggest_dt(dx: float, dx_ref: float = 1.0, dt_ref: float = 0.0015) -> float:
        """dt ~ dx^2 масштабирование для устойчивости диффузионного/лапласиан-члена."""
        return dt_ref * (dx / dx_ref) ** 2


# =============================================================================
# III.  PDE HELPERS
# =============================================================================

def sigma1(u: torch.Tensor, kappa: float, theta1: float) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-kappa * (u - theta1)))

def sigma2(u: torch.Tensor, kappa: float, theta2: float) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-kappa * (u - theta2)))

def M_phi(phi: torch.Tensor, kappa: float, theta1: float, theta2: float) -> torch.Tensor:
    p2 = phi * phi
    return phi * sigma1(p2, kappa, theta1) * (1.0 - sigma2(p2, kappa, theta2)) - phi

def laplacian9(u: torch.Tensor, dx: float) -> torch.Tensor:
    """9-точечный изотропный лапласиан, периодические ГУ."""
    up = torch.roll(u,  1, 0); dn = torch.roll(u, -1, 0)
    lf = torch.roll(u,  1, 1); rt = torch.roll(u, -1, 1)
    ul = torch.roll(up, 1, 1); ur = torch.roll(up,-1, 1)
    dl = torch.roll(dn, 1, 1); dr = torch.roll(dn,-1, 1)
    return ((4.0*(up+dn+lf+rt) + (ul+ur+dl+dr) - 20.0*u) / 6.0) / (dx*dx)

def laplacian5(u: torch.Tensor, dx: float) -> torch.Tensor:
    """5-точечный лапласиан, периодические ГУ."""
    up = torch.roll(u,  1, 0); dn = torch.roll(u, -1, 0)
    lf = torch.roll(u,  1, 1); rt = torch.roll(u, -1, 1)
    return (up + dn + lf + rt - 4.0*u) / (dx*dx)

def laplacian_neumann(u: torch.Tensor, dx: float) -> torch.Tensor:
    """5-точечный лапласиан, ГУ Неймана (нулевые потоки)."""
    u_pad = torch.nn.functional.pad(
        u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate'
    ).squeeze(0).squeeze(0)
    up = u_pad[:-2, 1:-1]; dn = u_pad[2:, 1:-1]
    lf = u_pad[1:-1, :-2]; rt = u_pad[1:-1, 2:]
    return (up + dn + lf + rt - 4.0*u) / (dx*dx)

def get_laplacian(boundary: str) -> Callable:
    if boundary == "neumann":
        return laplacian_neumann
    return laplacian9


# =============================================================================
# IV.  SPATIAL FIELDS
# =============================================================================

def _gaussian_2d(nx, ny, cx, cy, sigma, device) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.arange(ny, dtype=torch.float32, device=device).view(1, -1)
    return torch.exp(-0.5 * (((x-cx)/sigma)**2 + ((y-cy)/sigma)**2))

def _disk_2d(nx, ny, cx, cy, radius, device) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.arange(ny, dtype=torch.float32, device=device).view(1, -1)
    return ((x-cx)**2 + (y-cy)**2 <= radius**2).float()

def _ring_2d(nx, ny, cx, cy, ring_r, ring_w, device) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.arange(ny, dtype=torch.float32, device=device).view(1, -1)
    r = torch.sqrt((x-cx)**2 + (y-cy)**2)
    return torch.exp(-0.5 * ((r - ring_r) / ring_w)**2)

def build_spatial_fields(cfg: SimConfig, device: torch.device
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Строит (h_field, gamma_field) формы (nx, ny)."""
    nx, ny = cfg.nx, cfg.ny
    h_field     = torch.full((nx, ny), cfg.h_bg,     dtype=torch.float32, device=device)
    gamma_field = torch.full((nx, ny), cfg.gamma_bg,  dtype=torch.float32, device=device)

    if cfg.h_custom and os.path.exists(cfg.h_custom):
        h_field = h_field + torch.from_numpy(
            np.load(cfg.h_custom).astype(np.float32)).to(device)
    if cfg.gamma_custom and os.path.exists(cfg.gamma_custom):
        gamma_field = gamma_field + torch.from_numpy(
            np.load(cfg.gamma_custom).astype(np.float32)).to(device)

    for a in cfg.attractors:
        if a.profile == "gaussian":
            mask = _gaussian_2d(nx, ny, a.cx, a.cy, a.sigma, device)
        elif a.profile == "disk":
            mask = _disk_2d(nx, ny, a.cx, a.cy, a.sigma, device)
        elif a.profile == "ring":
            mask = _ring_2d(nx, ny, a.cx, a.cy, a.ring_r, a.ring_w, device)
        else:
            raise ValueError(f"Неизвестный профиль: {a.profile!r}")

        if a.kind == "pump":
            h_field = h_field + a.strength * mask
        elif a.kind == "sink":
            gamma_field = gamma_field + a.strength * mask
        elif a.kind == "h_sink":
            h_field = h_field - a.strength * mask
        else:
            raise ValueError(f"Неизвестный тип аттрактора: {a.kind!r}")

    return h_field, gamma_field


# =============================================================================
# V.  INITIAL CONDITIONS
# =============================================================================

def build_initial_state(cfg: SimConfig, device: torch.device
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Строит начальное состояние (phi, psi, v)."""
    nx, ny = cfg.nx, cfg.ny
    ic = cfg.init

    if ic.npz_path and os.path.exists(ic.npz_path):
        data = np.load(ic.npz_path)
        phi = torch.from_numpy(data["phi"].astype(np.float32)).to(device)
        psi = torch.from_numpy(data["psi"].astype(np.float32)).to(device)
        v   = torch.from_numpy(
            data.get("v", np.zeros_like(data["phi"])).astype(np.float32)
        ).to(device)
        return phi, psi, v

    bg  = cfg.phi_lo if ic.phi_background is None else ic.phi_background
    phi = torch.full((nx, ny), bg,   dtype=torch.float32, device=device)
    psi = torch.zeros((nx, ny),       dtype=torch.float32, device=device)
    v   = torch.zeros((nx, ny),       dtype=torch.float32, device=device)

    for sp in ic.spots:
        amp = cfg.phi_hi if sp.amp is None else sp.amp
        x   = torch.arange(nx, dtype=torch.float32, device=device).view(-1, 1)
        y   = torch.arange(ny, dtype=torch.float32, device=device).view(1, -1)
        if sp.shape == "disk":
            mask = ((x - sp.cx)**2 + (y - sp.cy)**2 <= sp.radius**2).float()
            phi  = phi + (amp - bg) * mask
        elif sp.shape == "tanh_disk":
            # sp.radius and sp.interface_width are specified in grid cells.
            # At dx = 0.5: R0 = 8 physical units -> radius = 16 cells;
            # w = 0.5 physical units -> interface_width = 1 cell.
            r = torch.sqrt((x - sp.cx)**2 + (y - sp.cy)**2)
            width = max(float(sp.interface_width), 1.0e-6)
            profile = 0.5 * (1.0 - torch.tanh((r - sp.radius) / width))
            phi = phi + (amp - bg) * profile
        elif sp.shape == "gaussian":
            g   = torch.exp(-0.5*(((x-sp.cx)/sp.radius)**2+((y-sp.cy)/sp.radius)**2))
            phi = phi + (amp - bg) * g
        if sp.phase_v != 0.0:
            v = v + sp.phase_v * _gaussian_2d(nx, ny, sp.cx, sp.cy, sp.radius, device)

    if ic.noise_amplitude > 0.0:
        rng   = np.random.default_rng(ic.noise_seed)
        noise = rng.standard_normal((nx, ny)).astype(np.float32) * ic.noise_amplitude
        if ic.noise_lowpass > 0.0:
            try:
                from scipy.ndimage import gaussian_filter
                sigma_px = 1.0 / (2.0 * np.pi * max(ic.noise_lowpass, 1e-9))
                noise    = gaussian_filter(noise, sigma=sigma_px)
            except ImportError:
                logger.warning("scipy не установлен — noise_lowpass игнорируется")
        phi = phi + torch.from_numpy(noise).to(device)

    phi = torch.clamp(phi, min=cfg.phi_lo - 0.5, max=cfg.phi_hi + 1.0)
    return phi, psi, v


# =============================================================================
# VI.  SEED BANK  (совместим с v24)
# =============================================================================

class SeedBank:
    """Кеш прогретых одиночных пятен.  Ключ = (eps, h, radius)."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._bank: Dict[Tuple, List] = {}

    def _key_path(self, eps: float, h: float, radius: int, cfg: SimConfig) -> str:
        key    = (f"eps={eps:.6f}|h={h:.6f}|r={radius}"
                  f"|nx={cfg.nx}|ny={cfg.ny}|dt={cfg.dt}"
                  f"|tw={cfg.t_warm}|ns={cfg.seed_n_snaps}")
        digest = hashlib.md5(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"seed_{digest}.pt")

    def load_or_build(self, device: torch.device, eps: float,
                      h: float, radius: int, cfg: SimConfig) -> List:
        bk   = (eps, h, radius)
        if bk in self._bank:
            return self._bank[bk]
        path = self._key_path(eps, h, radius, cfg)
        if os.path.exists(path):
            data  = torch.load(path, map_location="cpu")
            snaps = [(it["phi"].to(device), it["psi"].to(device), it["v"].to(device))
                     for it in data["snapshots"]]
            self._bank[bk] = snaps
            return snaps
        snaps = self._build(device, eps, h, radius, cfg)
        torch.save({"eps": eps, "h": h, "radius": radius,
                    "snapshots": [{"phi": p, "psi": ps, "v": vv}
                                  for p, ps, vv in snaps]}, path)
        self._bank[bk] = snaps
        return snaps

    def _build(self, device: torch.device, eps: float,
               h: float, radius: int, cfg: SimConfig) -> List:
        lap  = get_laplacian(cfg.boundary)
        phi  = torch.full((cfg.nx, cfg.ny), cfg.phi_lo, dtype=torch.float32, device=device)
        psi  = torch.zeros_like(phi)
        v    = torch.zeros_like(phi)
        x    = torch.arange(cfg.nx, device=device).view(-1, 1)
        y    = torch.arange(cfg.ny, device=device).view(1, -1)
        mask = (x - cfg.nx//2)**2 + (y - cfg.ny//2)**2 <= radius**2
        phi[mask] = cfg.phi_hi
        steps  = int(cfg.t_warm / cfg.dt)
        stride = max(1, steps // cfg.seed_n_snaps)
        snaps  = []
        for n in range(steps + 1):
            if n % stride == 0 and len(snaps) < cfg.seed_n_snaps:
                snaps.append((phi.detach().cpu().clone(),
                              psi.detach().cpu().clone(),
                              v.detach().cpu().clone()))
            lp  = lap(phi, cfg.dx)
            ls  = lap(psi, cfg.dx)
            phi = phi + cfg.dt * (M_phi(phi, cfg.kappa, cfg.theta1, cfg.theta2) + h + psi)
            v   = v   + cfg.dt * (lp + cfg.D_psi*ls - eps*v)
            psi = psi + cfg.dt * v
        return snaps


# =============================================================================
# VII.  STEPPER
# =============================================================================

class Stepper:
    """Выполняет один шаг PDE."""

    def __init__(self, cfg: SimConfig,
                 h_field: torch.Tensor, gamma_field: torch.Tensor):
        self.cfg         = cfg
        self.h_field     = h_field
        self.gamma_field = gamma_field
        self.lap         = get_laplacian(cfg.boundary)

    def step(self, phi: torch.Tensor, psi: torch.Tensor,
             v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lp      = self.lap(phi, self.cfg.dx)
        ls      = self.lap(psi, self.cfg.dx)
        eff_eps = self.cfg.eps + self.gamma_field
        phi_new = phi + self.cfg.dt * (M_phi(phi, self.cfg.kappa, self.cfg.theta1, self.cfg.theta2) + self.h_field + psi)
        v_new   = v   + self.cfg.dt * (lp + self.cfg.D_psi*ls - eff_eps*v)
        psi_new = psi + self.cfg.dt * v_new
        return phi_new, psi_new, v_new


# =============================================================================
# VIII.  SIMULATOR
# =============================================================================

class Simulator:
    """Главный цикл симуляции.

    Использование:
        sim = Simulator(cfg, device)
        sim.add_hook(every_n, fn)   # fn(step, t, phi_cpu, psi_cpu, v_cpu, sim)
        result = sim.run()
        # результаты: sim.result  и  sim.out_dir/
    """

    def __init__(self, cfg: SimConfig, device: torch.device):
        self.cfg      = cfg
        self.device   = device
        self.run_id   = cfg.resolve_run_id()
        self.out_dir  = os.path.join(cfg.out_dir, self.run_id)
        self.snap_dir = os.path.join(self.out_dir, "snapshots_2d")
        self._max_abs_phi = 0.0
        self._max_abs_psi = 0.0
        self._max_abs_v = 0.0
        os.makedirs(self.out_dir,  exist_ok=True)
        os.makedirs(self.snap_dir, exist_ok=True)

        cfg.to_json(os.path.join(self.out_dir, "config.json"))

        self.h_field, self.gamma_field = build_spatial_fields(cfg, device)
        np.save(os.path.join(self.out_dir, "h_field.npy"),
                self.h_field.cpu().numpy())
        np.save(os.path.join(self.out_dir, "gamma_field.npy"),
                self.gamma_field.cpu().numpy())

        self.phi, self.psi, self.v = build_initial_state(cfg, device)
        self.stepper = Stepper(cfg, self.h_field, self.gamma_field)

        self._hooks: List[Tuple[int, Callable]] = []
        self._track: Dict[str, List] = {"step": [], "t": [], "mass": [], "phi_center": [],
                                        "cx": [], "cy": []}
        self._n_snaps = 0
        self.result: Optional[Dict] = None

        self.status: str = "running"
        self.failure_reason: Optional[str] = None
        self.failure_step: Optional[int] = None
        self.failure_time: Optional[float] = None
        self.failure_meta: Dict[str, Any] = {}

    def add_hook(self, every_n: int, fn: Callable) -> None:
        """Зарегистрировать хук.
        fn(step, t, phi_cpu, psi_cpu, v_cpu, sim) -> None
        """
        self._hooks.append((every_n, fn))

    def _set_failure(self, meta: Dict[str, Any]) -> None:
        self.status = "numerically_unstable"
        self.failure_reason = str(meta.get("reason", "unknown"))
        self.failure_step = (
            int(meta["step"]) if meta.get("step") is not None else None
        )
        self.failure_time = (
            float(meta["time"]) if meta.get("time") is not None else None
        )
        self.failure_meta = dict(meta)

    def _check_instability(
        self,
        step: int,
        t: float,
        phi_np: np.ndarray,
        psi_np: np.ndarray,
        v_np: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg

        if cfg.check_nonfinite:
            phi_finite = np.isfinite(phi_np).all()
            psi_finite = np.isfinite(psi_np).all()
            v_finite = np.isfinite(v_np).all()
            if not (phi_finite and psi_finite and v_finite):
                return {
                    "reason": "nonfinite",
                    "step": step,
                    "time": t,
                    "max_abs_phi": float(np.nanmax(np.abs(phi_np))),
                    "max_abs_psi": float(np.nanmax(np.abs(psi_np))),
                    "max_abs_v": float(np.nanmax(np.abs(v_np))),
                }

        max_phi = float(np.nanmax(np.abs(phi_np)))
        max_psi = float(np.nanmax(np.abs(psi_np)))
        max_v = float(np.nanmax(np.abs(v_np)))

        if max_phi > cfg.max_abs_phi:
            return {
                "reason": "phi_bound",
                "step": step,
                "time": t,
                "max_abs_phi": max_phi,
                "max_abs_psi": max_psi,
                "max_abs_v": max_v,
            }

        if max_psi > cfg.max_abs_psi:
            return {
                "reason": "psi_bound",
                "step": step,
                "time": t,
                "max_abs_phi": max_phi,
                "max_abs_psi": max_psi,
                "max_abs_v": max_v,
            }

        if max_v > cfg.max_abs_v:
            return {
                "reason": "v_bound",
                "step": step,
                "time": t,
                "max_abs_phi": max_phi,
                "max_abs_psi": max_psi,
                "max_abs_v": max_v,
            }

        return None

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        t0wall = time.time()
        steps = int(cfg.t_total / cfg.dt)
        sp = cfg.snap
        stepss = int(sp.t_start / cfg.dt)
        stepse = int(sp.t_stop / cfg.dt) if sp.t_stop > 0 else steps + 1

        logger.info(f"[{self.run_id}] Start {steps} steps on {self.device}")
        phi, psi, v = self.phi, self.psi, self.v
        progressevery = 1000

        last_step = 0
        last_t = 0.0
        aborted = False

        for n in range(steps + 1):
            tnow = n * cfg.dt
            last_step = n
            last_t = tnow

            if n % cfg.monitor_every == 0:
                phi_np = phi.detach().cpu().numpy()
                psi_np = psi.detach().cpu().numpy()
                v_np = v.detach().cpu().numpy()

                self.recordtrack(n, tnow, phi_np, psi_np)
                self._track.setdefault("vmaxabs", []).append(float(np.nanmax(np.abs(v_np))))

                fail = self._check_instability(n, tnow, phi_np, psi_np, v_np)
                if fail is not None:
                    self._set_failure(fail)
                    logger.warning(f"[{self.run_id}] instability detected: {fail}")

                    try:
                        self._save_snapshot(phi, psi, v, tnow, suffix="_unstable_last")
                    except Exception as e:
                        logger.warning(f"[{self.run_id}] failed to save unstable snapshot: {e}")

                    if cfg.abort_on_instability:
                        aborted = True
                        break
            phi_abs = float(phi.detach().abs().max().item())
            psi_abs = float(psi.detach().abs().max().item())
            v_abs = float(v.detach().abs().max().item())

            self._max_abs_phi = max(self._max_abs_phi, phi_abs)
            self._max_abs_psi = max(self._max_abs_psi, psi_abs)
            self._max_abs_v = max(self._max_abs_v, v_abs)

            if stepss <= n <= stepse and n % sp.every_steps == 0:
                if sp.max_snaps < 0 or self._n_snaps < sp.max_snaps:
                    self._save_snapshot(phi, psi, v, tnow)
                    self._n_snaps += 1

            for everyn, fn in self._hooks:
                if n % everyn == 0:
                    fn(n, tnow, phi.detach().cpu(), psi.detach().cpu(), v.detach().cpu(), self)

            phi, psi, v = self.stepper.step(phi, psi, v)

            if n % progressevery == 0 or n == steps:
                render_progress(n, steps, cfg.dt, t0wall)

        print()
        self.phi, self.psi, self.v = phi, psi, v

        if not aborted:
            self._save_snapshot(phi, psi, v, last_t, suffix="_final")

        self._save_track_csv()

        # Finalize the run status before constructing result.json.
        # A recorded failure always takes precedence over nominal loop completion.
        if self.failure_reason is not None:
            self.status = "numerically_unstable"
        elif self.status == "running":
            self.status = "completed"

        peak_abs_phi = max(self._track.get("phimax", [float("nan")]))
        peak_abs_psi = max(self._track.get("psimaxabs", [float("nan")]))
        peak_abs_v = max(self._track.get("vmaxabs", [float("nan")]))

        wall = time.time() - t0wall
        self.result = {
            "run_id": self.run_id,
            "status": self.status,
            "t_total_requested": float(cfg.t_total),
            "t_total_reached": float(last_t),
            "last_step": int(last_step),
            "dt": float(cfg.dt),
            "dx": float(cfg.dx),
            "monitor_every": int(cfg.monitor_every),
            "_n_snaps": int(self._n_snaps),
            "wallsec": round(wall, 2),
            "finalmass": self._track["mass"][-1] if self._track["mass"] else 0,
            "status": self.status,
            "terminated_early": bool(self.failure_reason is not None),
            "failure_reason": self.failure_reason,
            "failure_step": self.failure_step,
            "failure_time": self.failure_time,
            "max_abs_phi": self._max_abs_phi,
            "max_abs_psi": self._max_abs_psi,
            "max_abs_v": self._max_abs_v,
            "peak_abs_phi": float(peak_abs_phi),
            "peak_abs_psi": float(peak_abs_psi),
            "peak_abs_v": float(peak_abs_v),
        }

        Path(os.path.join(self.out_dir, "result.json")).write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        logger.info(
            f"[{self.run_id}] Done status={self.status} in {wall:.1f}s, snaps={self._n_snaps}"
        )


        return self.result

    def recordtrack(
        self,
        step: int,
        t: float,
        phi_np: np.ndarray,
        psi_np: Optional[np.ndarray] = None
    ) -> None:
        thresh = 1.5
        mask = phi_np > thresh
        mass = int(mask.sum())

        self._track["step"].append(step)
        self._track["t"].append(t)
        self._track["mass"].append(mass)
        self._track["phi_center"].append(float(phi_np[self.cfg.nx // 2, self.cfg.ny // 2]))
        self._track.setdefault("phimax", []).append(float(np.nanmax(np.abs(phi_np))))

        if psi_np is not None:
            self._track.setdefault("psimaxabs", []).append(float(np.nanmax(np.abs(psi_np))))

        if mass >= 20:
            coords = np.argwhere(mask)
            w = phi_np[mask] - thresh
            wsum = w.sum()
            if np.isfinite(wsum) and wsum > 0:
                cx = float((coords[:, 0] * w).sum() / wsum)
                cy = float((coords[:, 1] * w).sum() / wsum)
            else:
                cx = cy = float("nan")
        else:
            cx = cy = float("nan")

        self._track["cx"].append(cx)
        self._track["cy"].append(cy)

    def _save_snapshot(self, phi, psi, v, t_val: float, suffix: str = "") -> None:
        sp    = self.cfg.snap
        t_int = int(round(t_val * 10))
        fname = os.path.join(
            self.snap_dir,
            f"snapshot2d_{self.run_id}_t{t_int:08d}{suffix}.npz"
        )
        kw: Dict[str, Any] = {"t": np.array([t_val], dtype=np.float32)}
        if sp.save_phi:
            kw["phi"] = phi.detach().cpu().numpy().astype(np.float32)
        if sp.save_psi:
            kw["psi"] = psi.detach().cpu().numpy().astype(np.float32)
        if sp.save_v:
            kw["v"]   = v.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(fname, **kw)

    def _save_track_csv(self) -> None:
        path = os.path.join(self.out_dir, "track.csv")
        keys = list(self._track.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for row in zip(*[self._track[k] for k in keys]):
                w.writerow(row)


# =============================================================================
# IX.  ONLINE ANALYZER
# =============================================================================

class OnlineAnalyzer:
    """Готовые хуки для подключения к Simulator.

    Пример:
        ana = OnlineAnalyzer(sim)
        sim.add_hook(cfg.monitor_every, ana.spacetime_row())
        sim.add_hook(2000, ana.snapshot_png())
        sim.run()
        ana.save_spacetime_png()
    """

    def __init__(self, sim: Simulator):
        self.sim      = sim
        self._rows:   List[np.ndarray] = []
        self._row_t:  List[float]      = []

    def spacetime_row(self, row_index: Optional[int] = None) -> Callable:
        nx = self.sim.cfg.nx
        ri = row_index if row_index is not None else nx // 2

        def _hook(step, t, phi_cpu, psi_cpu, v_cpu, sim):
            self._rows.append(phi_cpu.numpy()[ri, :].copy())
            self._row_t.append(t)

        return _hook

    def save_spacetime_png(self, path: Optional[str] = None) -> None:
        if not self._rows:
            return
        import matplotlib.pyplot as plt
        arr = np.array(self._rows)
        p   = path or os.path.join(self.sim.out_dir, "spacetime.png")
        plt.figure(figsize=(10, 6))
        plt.imshow(arr, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, arr.shape[1]-1, self._row_t[0], self._row_t[-1]])
        plt.colorbar(label="phi(row, y)")
        plt.xlabel("y"); plt.ylabel("t")
        plt.title(f"Space-time  run={self.sim.run_id}")
        plt.tight_layout()
        plt.savefig(p, dpi=180)
        plt.close()
        logger.info(f"Space-time saved: {p}")

    def snapshot_png(self) -> Callable:
        """Сохраняет PNG-превью phi при каждом вызове."""
        import matplotlib.pyplot as plt

        def _hook(step, t, phi_cpu, psi_cpu, v_cpu, sim):
            p   = os.path.join(sim.out_dir, f"preview_t{int(t*10):08d}.png")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, arr, title in zip(axes,
                                      [phi_cpu.numpy(), psi_cpu.numpy()],
                                      ["phi", "psi"]):
                im = ax.imshow(arr.T, origin="lower", cmap="inferno")
                fig.colorbar(im, ax=ax, label=title)
                ax.set_title(f"{title}  t={t:.2f}")
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)

        return _hook

    def field_stats(self, every_n: int = 100) -> Callable:
        """Записывает глобальные статистики поля в JSONL."""
        path = os.path.join(self.sim.out_dir, "field_stats.jsonl")

        def _hook(step, t, phi_cpu, psi_cpu, v_cpu, sim):
            if step % every_n != 0:
                return
            p   = phi_cpu.numpy()
            rec = {
                "t": t, "step": step,
                "phi_mean": float(p.mean()),
                "phi_max":  float(p.max()),
                "phi_std":  float(p.std()),
                "mass_15":  int((p > 1.5).sum()),
                "mass_30":  int((p > 3.0).sum()),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        return _hook


# =============================================================================
# X.  PARALLEL RUNNER
# =============================================================================

def _worker(args: Tuple[dict, str]) -> dict:
    cfg_dict, device_str = args
    cfg    = SimConfig.from_dict(cfg_dict)
    device = torch.device(device_str)
    sim    = Simulator(cfg, device)
    return sim.run()


class ParallelRunner:
    """Запуск батча сценариев.

    Режимы:
        "sequential"   — последовательно (один GPU или CPU)
        "multiprocess" — несколько CPU-процессов (device должен быть "cpu")

    Пример:
        runner = ParallelRunner(configs, device="cpu",
                                mode="multiprocess", workers=4)
        results = runner.run_all()
    """

    def __init__(self, configs: List[SimConfig],
                 device:  str = "cpu",
                 mode:    str = "sequential",
                 workers: int = 4):
        self.configs = configs
        self.device  = device
        self.mode    = mode
        self.workers = workers

    def run_all(self) -> List[Dict]:
        if self.mode == "sequential":
            return self._sequential()
        elif self.mode == "multiprocess":
            return self._multiprocess()
        raise ValueError(f"Неизвестный режим: {self.mode!r}")

    def _sequential(self) -> List[Dict]:
        dev = torch.device(self.device)
        return [Simulator(cfg, dev).run() for cfg in self.configs]

    def _multiprocess(self) -> List[Dict]:
        import multiprocessing as mp
        args = [(cfg.to_dict(), self.device) for cfg in self.configs]
        with mp.Pool(processes=self.workers) as pool:
            return pool.map(_worker, args)


# =============================================================================
# XI.  ПРИМЕРЫ КОНФИГУРАЦИЙ
# =============================================================================

def example_single_spot() -> SimConfig:
    """Одиночное пятно, нет аттракторов, нет доп. диссипации."""
    return SimConfig(
        eps=2.8, h_bg=0.533, nx=160, ny=160, dt=0.003, t_total=600.0,
        init=InitConfig(spots=[InitSpot(cx=80, cy=80, radius=8, shape="disk")]),
        snap=SnapshotPolicy(every_steps=2000),
        out_dir="output_sim_framework", run_id="ex_single_spot",
        tags=["example", "single_spot"],
    )


def example_glider_pair() -> SimConfig:
    """Пара пятен с асимметричным kick (аналог v24 glider)."""
    return SimConfig(
        eps=2.8, h_bg=0.545, nx=160, ny=160, dt=0.003, t_total=1400.0,
        init=InitConfig(spots=[
            InitSpot(cx=73, cy=78, radius=8,  shape="disk"),
            InitSpot(cx=87, cy=82, radius=8,  shape="disk", phase_v=0.06),
        ]),
        snap=SnapshotPolicy(every_steps=2000),
        out_dir="output_sim_framework", run_id="ex_glider_pair",
        tags=["example", "glider"],
    )


def example_triad_attractors() -> SimConfig:
    """Три pump-аттрактора в вершинах треугольника + фоновая диссипация."""
    R  = 35.0
    cx, cy = 80.0, 80.0
    pumps = [
        AttractorSpec(kind="pump",
                      cx=cx + R*np.cos(np.radians(a)),
                      cy=cy + R*np.sin(np.radians(a)),
                      strength=0.04, sigma=10.0)
        for a in [90.0, 210.0, 330.0]
    ]
    spots = [
        InitSpot(cx=cx + R*np.cos(np.radians(a)),
                 cy=cy + R*np.sin(np.radians(a)), radius=8)
        for a in [90.0, 210.0, 330.0]
    ]
    return SimConfig(
        eps=2.8, h_bg=0.50, gamma_bg=0.15, nx=160, ny=160,
        dt=0.003, t_total=800.0,
        attractors=pumps,
        init=InitConfig(spots=spots),
        snap=SnapshotPolicy(every_steps=1000),
        out_dir="output_sim_framework", run_id="ex_triad_attractors",
        tags=["example", "triad", "attractors"],
    )


def example_sink_barrier() -> SimConfig:
    """Два pump-аттрактора, разделённых диссипативным барьером."""
    return SimConfig(
        eps=2.8, h_bg=0.533, gamma_bg=0.05, nx=160, ny=160,
        dt=0.003, t_total=800.0,
        attractors=[
            AttractorSpec(kind="pump", cx=55, cy=80, strength=0.04, sigma=10),
            AttractorSpec(kind="pump", cx=105, cy=80, strength=0.04, sigma=10),
            AttractorSpec(kind="sink", cx=80, cy=80, strength=0.8,  sigma=4.0),
        ],
        init=InitConfig(spots=[
            InitSpot(cx=55,  cy=80, radius=8),
            InitSpot(cx=105, cy=80, radius=8),
        ]),
        snap=SnapshotPolicy(every_steps=1000),
        out_dir="output_sim_framework", run_id="ex_sink_barrier",
        tags=["example", "barrier", "pump_sink"],
    )


def example_channel_ring() -> SimConfig:
    """Кольцевой pump-аттрактор — принудительное возбуждение канала."""
    return SimConfig(
        eps=2.8, h_bg=0.50, gamma_bg=0.10, nx=160, ny=160,
        dt=0.003, t_total=800.0,
        attractors=[
            AttractorSpec(kind="pump", cx=80, cy=80, strength=0.06,
                          sigma=4.0, profile="ring", ring_r=30.0, ring_w=4.0),
            AttractorSpec(kind="sink", cx=80, cy=80, strength=0.5, sigma=15.0),
        ],
        init=InitConfig(spots=[
            InitSpot(cx=80+30, cy=80, radius=6),
        ]),
        snap=SnapshotPolicy(every_steps=1000),
        out_dir="output_sim_framework", run_id="ex_channel_ring",
        tags=["example", "ring_attractor", "channel"],
    )


def example_full_field_noise() -> SimConfig:
    """Полноразмерное поле с низкочастотным шумом (без пятен)."""
    return SimConfig(
        eps=2.8, h_bg=0.533, nx=256, ny=256, dt=0.003, t_total=1200.0,
        init=InitConfig(phi_background=0.54,
                        noise_amplitude=0.03, noise_seed=7, noise_lowpass=0.05),
        snap=SnapshotPolicy(every_steps=3000),
        out_dir="output_sim_framework", run_id="ex_full_field_noise",
        tags=["example", "full_field", "noise"],
    )


EXAMPLE_CONFIGS: Dict[str, Callable[[], SimConfig]] = {
    "single_spot":      example_single_spot,
    "glider_pair":      example_glider_pair,
    "triad_attractors": example_triad_attractors,
    "sink_barrier":     example_sink_barrier,
    "channel_ring":     example_channel_ring,
    "full_field_noise": example_full_field_noise,
}

# =============================================================================
# XII.  DEVICE SELECTION & MAIN
# =============================================================================

def select_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if HAS_DML:
        return torch_directml.device(), "directml"
    return torch.device("cpu"), "cpu"

def render_progress(step, total_steps, dt, t0_wall, width=32):
    t_now = step * dt
    t_total = total_steps * dt
    frac = min(max(step / total_steps, 0.0), 1.0)

    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)

    elapsed = time.time() - t0_wall
    if frac > 0:
        eta = elapsed * (1.0 - frac) / frac
    else:
        eta = 0.0

    msg = (
        f"\r[{bar}] {100*frac:6.2f}% "
        f"| t={t_now:8.2f}/{t_total:.2f} "
        f"| step={step}/{total_steps} "
        f"| wall={elapsed:7.1f}s "
        f"| ETA={eta:7.1f}s"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="Sim Framework v1 — активаторно-волновое поле",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Примеры:",
            "  python sim_framework_v1.py --example triad_attractors",
            "  python sim_framework_v1.py --config my_run.json",
            "  python sim_framework_v1.py --example glider_pair --device cpu",
            "  python sim_framework_v1.py --list",
            "  python sim_framework_v1.py --example sink_barrier --export_config sink.json",
        ])
    )
    ap.add_argument("--config",        type=str,  default=None,
                    help="Путь к JSON-конфигу SimConfig")
    ap.add_argument("--example",       type=str,  default=None,
                    help=f"Имя готового примера: {list(EXAMPLE_CONFIGS)}")
    ap.add_argument("--list",          action="store_true",
                    help="Показать доступные примеры и выйти")
    ap.add_argument("--device",        type=str,  default=None,
                    help="Форсировать устройство: cpu | cuda | directml")
    ap.add_argument("--out_dir",       type=str,  default=None,
                    help="Переопределить out_dir")
    ap.add_argument("--t_total",       type=float,default=None,
                    help="Переопределить t_total")
    ap.add_argument("--export_config", type=str,  default=None,
                    help="Сохранить конфиг в JSON и выйти (без запуска)")
    ap.add_argument("--parallel",      type=int,  default=1,
                    help="Число CPU-воркеров для multiprocess (>1 -> parallel mode)")
    args = ap.parse_args()

    if args.list:
        print("Доступные примеры конфигурации:")
        for name, fn in EXAMPLE_CONFIGS.items():
            c = fn()
            print(f"  {name:25s}  nx={c.nx}  t={c.t_total}  tags={c.tags}")
        return

    if args.config:
        cfg = SimConfig.from_json(args.config)
    elif args.example:
        if args.example not in EXAMPLE_CONFIGS:
            ap.error(f"Неизвестный пример: {args.example!r}")
        cfg = EXAMPLE_CONFIGS[args.example]()
    else:
        ap.print_help()
        return

    if args.out_dir:  cfg.out_dir = args.out_dir
    if args.t_total:  cfg.t_total = args.t_total

    if args.export_config:
        cfg.to_json(args.export_config)
        print(f"Конфиг сохранён: {args.export_config}")
        return

    device_str = args.device
    if device_str:
        device  = torch.device(device_str)
        backend = device_str
    else:
        device, backend = select_device()

    print(f"Backend: {backend} | Device: {device}")
    print(f"Run ID : {cfg.resolve_run_id()}")

    # Параллельный запуск нескольких копий (если --parallel > 1)
    if args.parallel > 1:
        cfgs   = []
        for i in range(args.parallel):
            c = copy.deepcopy(cfg)
            c.run_id = cfg.resolve_run_id() + f"_w{i}"
            c.init.noise_seed = cfg.init.noise_seed + i
            cfgs.append(c)
        runner  = ParallelRunner(cfgs, device="cpu",
                                 mode="multiprocess", workers=args.parallel)
        results = runner.run_all()
    else:
        sim = Simulator(cfg, device)
        ana = OnlineAnalyzer(sim)
        sim.add_hook(cfg.monitor_every, ana.spacetime_row())
        sim.add_hook(cfg.monitor_every, ana.field_stats(every_n=cfg.monitor_every))
        results = [sim.run()]
        ana.save_spacetime_png()

    print("\n=== DONE ===")
    for r in results:
        print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
