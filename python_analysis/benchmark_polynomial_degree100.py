from __future__ import annotations

import time
from decimal import Decimal, getcontext

import numpy as np
import torch


def eval_poly_naive(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    degree = coeffs.numel() - 1
    y = torch.zeros_like(x)
    for i in range(degree + 1):
        y = y + coeffs[i] * torch.pow(x, i)
    return y


def eval_poly_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x) + coeffs[-1]
    for i in range(coeffs.numel() - 2, -1, -1):
        y = y * x + coeffs[i]
    return y


def eval_poly_phi_fma(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x) + coeffs[-1]
    for i in range(coeffs.numel() - 2, -1, -1):
        y = torch.addcmul(coeffs[i], y, x)
    return y


def eval_poly_decimal(coeffs: np.ndarray, x: float) -> float:
    xd = Decimal(str(float(x)))
    y = Decimal(str(float(coeffs[-1])))
    for i in range(len(coeffs) - 2, -1, -1):
        y = y * xd + Decimal(str(float(coeffs[i])))
    return float(y)


def timed(fn, coeffs: torch.Tensor, x: torch.Tensor, runs: int) -> tuple[torch.Tensor, float]:
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = None
    for _ in range(runs):
        y = fn(coeffs, x)
    if x.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / runs
    return y, dt


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    getcontext().prec = 80

    degree = 100
    n_points = 4096
    ref_points = 128
    runs = 20

    coeffs_np = np.random.uniform(-1.0, 1.0, size=(degree + 1,)).astype(np.float64)
    coeffs_cpu = torch.tensor(coeffs_np, dtype=torch.float64)
    x_cpu = torch.linspace(-1.0, 1.0, n_points, dtype=torch.float64)

    y_naive_cpu, t_naive = timed(eval_poly_naive, coeffs_cpu, x_cpu, runs)
    y_horner_cpu, t_horner = timed(eval_poly_horner, coeffs_cpu, x_cpu, runs)
    y_phi_cpu, t_phi = timed(eval_poly_phi_fma, coeffs_cpu, x_cpu, runs)

    idx = torch.linspace(0, n_points - 1, ref_points, dtype=torch.long)
    x_ref = x_cpu[idx].numpy()
    ref_vals = np.array([eval_poly_decimal(coeffs_np, float(v)) for v in x_ref], dtype=np.float64)

    e_naive = float(np.max(np.abs(y_naive_cpu[idx].numpy() - ref_vals)))
    e_horner = float(np.max(np.abs(y_horner_cpu[idx].numpy() - ref_vals)))
    e_phi = float(np.max(np.abs(y_phi_cpu[idx].numpy() - ref_vals)))

    print("=== Reto Polinomio Grado 100 (CPU float64) ===")
    print(f"max_error_naive_vs_decimal:  {e_naive:.3e}")
    print(f"max_error_horner_vs_decimal: {e_horner:.3e}")
    print(f"max_error_phi_vs_decimal:    {e_phi:.3e}")
    print(f"time_naive_s:  {t_naive:.6f}")
    print(f"time_horner_s: {t_horner:.6f}")
    print(f"time_phi_s:    {t_phi:.6f}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        coeffs_gpu = coeffs_cpu.to(device=device, dtype=torch.float32)
        x_gpu = x_cpu.to(device=device, dtype=torch.float32)

        _, tg_horner = timed(eval_poly_horner, coeffs_gpu, x_gpu, runs)
        _, tg_phi = timed(eval_poly_phi_fma, coeffs_gpu, x_gpu, runs)

        print("=== Reto Polinomio Grado 100 (GPU float32) ===")
        print(f"time_horner_gpu_s: {tg_horner:.6f}")
        print(f"time_phi_gpu_s:    {tg_phi:.6f}")


if __name__ == "__main__":
    main()
