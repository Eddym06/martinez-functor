"""Benchmark and validation helpers for Affine Collapse Functor (ACF)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import time

import numpy as np
import torch

from .core import (
    ChebyshevReducer,
    HornerReducer,
    KoopmanReducer,
    ACFFunctor,
    ACFInvariant,
)


class ConservationValidator:
    @staticmethod
    def validate_polynomial(
        coefficients: list,
        test_domain: Tuple[float, float] = (-5.0, 5.0),
        n_points: int = 10000,
        dtype: torch.dtype = torch.float64,
    ) -> Dict:
        degree = len(coefficients) - 1
        phi = ACFFunctor(default_dtype=dtype)
        reduction = phi.reduce_polynomial(coefficients)

        x = torch.linspace(*test_domain, n_points, dtype=dtype)
        direct = sum(coefficients[i] * x**i for i in range(len(coefficients)))
        horner = HornerReducer.execute_horner(coefficients, x)
        max_err = float(torch.max(torch.abs(direct - horner)).item())

        return {
            "degree": degree,
            "E_f": degree,
            "E_phi_f": reduction.computational_energy,
            "structural_conserved": degree == reduction.computational_energy,
            "max_numerical_error": max_err,
            "machine_epsilon": torch.finfo(dtype).eps,
        }

    @staticmethod
    def validate_transcendental(
        func_name: str,
        degree: int = 20,
        domain: Optional[Tuple[float, float]] = None,
        n_points: int = 10000,
        dtype: torch.dtype = torch.float64,
    ) -> Dict:
        phi = ACFFunctor(default_dtype=dtype)
        reduction = phi.reduce_transcendental(func_name, degree=degree, domain=domain)
        dom = reduction.domain
        x = torch.linspace(dom[0], dom[1], n_points, dtype=dtype)
        fn = ChebyshevReducer.CANONICAL_FUNCTIONS[func_name]["generator"]
        exact = fn(x)
        approx = HornerReducer.execute_horner(reduction.metadata["monomial_coefficients"], x)
        max_err = float(torch.max(torch.abs(exact - approx)).item())

        return {
            "function": func_name,
            "degree": reduction.metadata["degree"],
            "certified_epsilon": reduction.epsilon_bound,
            "measured_max_error": max_err,
            "error_within_certificate": max_err <= reduction.epsilon_bound * 1.01,
            "domain": dom,
            "computational_energy": reduction.computational_energy,
        }


class PerformanceBenchmark:
    @staticmethod
    def benchmark_polynomial(
        degree: int = 100,
        n_points: int = 100000,
        domain: Tuple[float, float] = (-1.0, 1.0),
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        n_warmup: int = 3,
        n_runs: int = 20,
    ) -> Dict:
        torch.manual_seed(42)
        coeffs = torch.randn(degree + 1, dtype=dtype, device=device)
        x = torch.linspace(*domain, n_points, dtype=dtype, device=device)

        def eval_naive():
            out = torch.zeros_like(x)
            for i, c in enumerate(coeffs):
                out = out + c * x.pow(i)
            return out

        def eval_horner():
            out = torch.full_like(x, coeffs[-1].item())
            for i in range(coeffs.numel() - 2, -1, -1):
                out = x * out + coeffs[i]
            return out

        def eval_phi_fma():
            return HornerReducer.execute_horner(coeffs, x)

        methods = {"naive": eval_naive, "horner": eval_horner, "phi_fma": eval_phi_fma}
        res = {}

        for name, fn in methods.items():
            for _ in range(n_warmup):
                _ = fn()
            if device != "cpu":
                torch.cuda.synchronize()
            times = []
            out = None
            for _ in range(n_runs):
                if device != "cpu":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn()
                if device != "cpu":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            res[name] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "output_sample": out[:5].detach().cpu().tolist(),
            }

        ref = eval_phi_fma().to(torch.float64)
        for name, fn in methods.items():
            out = fn().to(torch.float64)
            res[name]["max_error_vs_fp64_ref"] = float(torch.max(torch.abs(out - ref)).item())

        res["metadata"] = {
            "degree": degree,
            "n_points": n_points,
            "domain": domain,
            "dtype": str(dtype),
            "device": device,
        }
        return res


class KoopmanValidator:
    @staticmethod
    def validate_linear_system(
        n_dim: int = 3,
        n_steps: int = 500,
        dtype: torch.dtype = torch.float64,
    ) -> Dict:
        torch.manual_seed(42)
        A = torch.randn(n_dim, n_dim, dtype=dtype) * 0.5

        x = torch.zeros(n_dim, n_steps, dtype=dtype)
        x[:, 0] = torch.randn(n_dim, dtype=dtype)
        for t in range(n_steps - 1):
            x[:, t + 1] = A @ x[:, t]

        phi = ACFFunctor(default_dtype=dtype)
        red = phi.reduce_dynamical_system(x, observable_fn=lambda z: z, rank=n_dim)
        K = red.fma_sequence[0].weight
        err = float(torch.norm(K - A).item())
        pred = K @ x[:, -2:-1]
        pred_err = float((torch.norm(pred - x[:, -1:]) / (torch.norm(x[:, -1:]) + 1e-15)).item())
        alpha, delta = ACFInvariant.compute_alpha(torch.abs(torch.linalg.eigvals(K)))

        return {
            "matrix_recovery_error": err,
            "prediction_relative_error": pred_err,
            "reconstruction_error": red.metadata["reconstruction_error"],
            "acf_alpha": alpha,
            "truncation_delta": delta,
            "spectral_radius": red.metadata["spectral_radius"],
        }


def run_full_validation(device: str = "cpu") -> Dict:
    out = {}
    out["polynomial_conservation"] = ConservationValidator.validate_polynomial(
        [1.0, -0.5, 0.3, -0.1, 0.05, -0.02, 0.01, 0.005]
    )

    out["transcendental"] = {
        f: ConservationValidator.validate_transcendental(f, degree=24)
        for f in ["sin", "cos", "exp", "tanh"]
    }

    out["benchmark_poly"] = PerformanceBenchmark.benchmark_polynomial(degree=100, device=device)
    out["koopman_linear"] = KoopmanValidator.validate_linear_system()
    return out


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _ = run_full_validation(dev)
    print("Validation complete")
