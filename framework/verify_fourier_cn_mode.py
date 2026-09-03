#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
import torch

from sim_framework_fourier_cn import (
    FourierCnStepper,
    InitConfig,
    SimConfig,
    SnapshotPolicy,
)


def main() -> None:
    nx = 128
    ny = 128
    dx = 1.0
    dt = 5.0e-4
    t_total = 10.0

    eps = 3.0
    d_psi = 0.0
    mode_x = 2
    amplitude = 1.0

    device = torch.device("cpu")
    dtype = torch.float64

    cfg = SimConfig(
        eps=eps,
        h_bg=0.0,
        D_psi=d_psi,
        nx=nx,
        ny=ny,
        dx=dx,
        boundary="periodic",
        dt=dt,
        t_total=t_total,
        gamma_bg=0.0,
        init=InitConfig(),
        snap=SnapshotPolicy(),
    )

    h_field = torch.zeros((nx, ny), dtype=dtype, device=device)
    gamma_field = torch.zeros((nx, ny), dtype=dtype, device=device)
    stepper = FourierCnStepper(cfg, h_field, gamma_field)

    x = torch.arange(nx, dtype=dtype, device=device).view(-1, 1)
    phi_fixed = amplitude * torch.cos(2.0 * math.pi * mode_x * x / nx)
    phi_fixed = phi_fixed.repeat(1, ny)

    psi = torch.zeros((nx, ny), dtype=dtype, device=device)
    v = torch.zeros_like(psi)

    k = 2.0 * math.pi * mode_x / (nx * dx)
    forcing = -(k * k) * amplitude

    times = []
    numerical = []
    analytic = []

    steps = int(round(t_total / dt))

    for step in range(steps + 1):
        t = step * dt

        if step % 100 == 0:
            coefficient = (
                2.0
                / (nx * ny)
                * torch.sum(psi * torch.cos(2.0 * math.pi * mode_x * x / nx))
            ).item()

            exact = (
                forcing
                / eps
                * (
                    t
                    - (1.0 - math.exp(-eps * t)) / eps
                )
            )

            times.append(t)
            numerical.append(coefficient)
            analytic.append(exact)

        if step == steps:
            break

        phi_hat = torch.fft.fft2(phi_fixed)
        psi_hat = torch.fft.fft2(psi)
        v_hat = torch.fft.fft2(v)

        rhs_1 = psi_hat + stepper.half_dt * v_hat
        rhs_2 = (
            stepper.b22 * v_hat
            + stepper.b21 * psi_hat
            - stepper.dt * stepper.k2 * phi_hat
        )

        psi_hat_new = (
            stepper.a22 * rhs_1
            + stepper.half_dt * rhs_2
        ) / stepper.det

        v_hat_new = (
            -stepper.a21 * rhs_1
            + rhs_2
        ) / stepper.det

        psi = torch.fft.ifft2(psi_hat_new).real
        v = torch.fft.ifft2(v_hat_new).real

    numerical = np.asarray(numerical)
    analytic = np.asarray(analytic)
    error = np.max(np.abs(numerical - analytic))

    print("mode verification")
    print(f"nx={nx}, mode_x={mode_x}, k={k:.12f}")
    print(f"dt={dt:.6e}, t_total={t_total:.3f}")
    print(f"max_abs_error={error:.12e}")
    print(f"numerical_final={numerical[-1]:.12e}")
    print(f"analytic_final={analytic[-1]:.12e}")


if __name__ == "__main__":
    main()