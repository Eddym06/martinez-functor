"""Relative Kolmogorov entropy over the FMA alphabet (Evolution 12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
import math

import numpy as np
import torch

from .core import ChebyshevReducer, HornerReducer, ReductionResult


@dataclass
class EntropyProfile:
    epsilon_values: List[float]
    energy_values: List[int]
    alpha: float
    entropy_rate: float
    complexity_class: str
    theoretical_lower_bound: Callable[[float], float]

    def evaluate(self, epsilon: float) -> float:
        return float(self.theoretical_lower_bound(epsilon))

    def efficiency(self, actual_fma_count: int, epsilon: float) -> float:
        theoretical = self.evaluate(epsilon)
        if theoretical <= 0:
            return 1.0
        return min(theoretical / max(actual_fma_count, 1), 1.0)


@dataclass
class ConservationTest:
    e_f: int
    e_phi_f: int
    conserved: bool
    relative_gap: float
    details: Dict[str, Any] = field(default_factory=dict)


class KolmogorovEntropy:
    def __init__(
        self,
        min_degree: int = 2,
        max_degree: int = 120,
        degree_step: int = 2,
        n_probe: int = 5000,
        dtype: torch.dtype = torch.float64,
    ):
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.degree_step = degree_step
        self.n_probe = n_probe
        self.dtype = dtype

    def compute_entropy_profile(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> EntropyProfile:
        x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
        f_exact = f(x)

        epsilons: List[float] = []
        energies: List[int] = []

        for degree in range(self.min_degree, self.max_degree + 1, self.degree_step):
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y_approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                eps = float(torch.max(torch.abs(f_exact - y_approx)).item())
                if eps > 0:
                    epsilons.append(eps)
                    energies.append(degree)
            except Exception:
                continue

        if not epsilons:
            return EntropyProfile(
                epsilon_values=[],
                energy_values=[],
                alpha=1.0,
                entropy_rate=0.0,
                complexity_class="unknown",
                theoretical_lower_bound=lambda _eps: 1,
            )

        alpha, complexity_class = self._fit_asymptotics(epsilons, energies)

        if complexity_class == "polynomial":
            lower_bound = lambda eps, _e=energies: (_e[-1] if eps <= 1e-12 else _e[0])
        elif complexity_class == "analytic":
            log_inv = [math.log(1.0 / (e + 1e-30)) for e in epsilons]
            if len(log_inv) > 0 and log_inv[-1] > 0:
                C = energies[-1] / max(log_inv[-1] ** max(alpha, 0.1), 1e-12)
            else:
                C = 1.0
            lower_bound = lambda eps, _C=C, _a=max(alpha, 0.1): _C * math.log(1.0 / (eps + 1e-30)) ** _a
        else:
            k = max(1.0 / max(alpha, 1e-6), 1.0)
            C = energies[-1] * (epsilons[-1] + 1e-30) ** (1.0 / k)
            lower_bound = lambda eps, _C=C, _k=k: _C * (eps + 1e-30) ** (-1.0 / _k)

        if len(epsilons) >= 2 and epsilons[0] > epsilons[-1] > 0:
            total_info = math.log2(epsilons[0] / epsilons[-1])
            total_ops = energies[-1] - energies[0]
            entropy_rate = total_info / max(total_ops, 1)
        else:
            entropy_rate = 0.0

        return EntropyProfile(
            epsilon_values=epsilons,
            energy_values=energies,
            alpha=alpha,
            entropy_rate=entropy_rate,
            complexity_class=complexity_class,
            theoretical_lower_bound=lower_bound,
        )

    def test_conservation(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: int = 20,
    ) -> ConservationTest:
        phi_f = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
        e_f = degree

        coeffs = phi_f.metadata.get(
            "monomial_coefficients",
            phi_f.metadata.get("coefficients", []),
        )

        if coeffs:
            actual_degree = len(coeffs) - 1
            while actual_degree > 0 and abs(coeffs[actual_degree]) < 1e-30:
                actual_degree -= 1
            e_phi_f = actual_degree
        else:
            e_phi_f = e_f

        relative_gap = abs(e_f - e_phi_f) / max(e_f, 1)

        return ConservationTest(
            e_f=e_f,
            e_phi_f=e_phi_f,
            conserved=(e_f == e_phi_f) or (relative_gap < 0.1),
            relative_gap=relative_gap,
            details={
                "original_degree": degree,
                "effective_degree": e_phi_f,
                "phi_f_epsilon": float(phi_f.epsilon_bound),
                "n_coefficients": len(coeffs) if coeffs else 0,
            },
        )

    def compute_minimum_energy(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        target_epsilon: float,
    ) -> Tuple[int, ReductionResult]:
        x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
        f_exact = f(x)

        for degree in range(self.min_degree, self.max_degree + 1):
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                eps = float(torch.max(torch.abs(f_exact - y)).item())
                if eps <= target_epsilon:
                    return degree, reduction
            except Exception:
                continue

        return self.max_degree, ChebyshevReducer.reduce(f, degree=self.max_degree, domain=domain, dtype=self.dtype)

    def _fit_asymptotics(self, epsilons: List[float], energies: List[int]) -> Tuple[float, str]:
        if len(epsilons) < 3:
            return 1.0, "unknown"

        if epsilons[-1] < 1e-10:
            return 0.0, "polynomial"

        valid = [(e, d) for e, d in zip(epsilons, energies) if e > 1e-14]
        if len(valid) < 3:
            return 1.0, "analytic"

        log_energy = np.array([math.log(d + 1) for _, d in valid], dtype=np.float64)
        log_log_inv_eps = np.array(
            [math.log(max(math.log(1.0 / (e + 1e-30)), 0.01)) for e, _ in valid],
            dtype=np.float64,
        )

        try:
            coeffs = np.polyfit(log_log_inv_eps, log_energy, 1)
            alpha = max(float(coeffs[0]), 0.1)
            if alpha < 1.5:
                return alpha, "analytic"
            if alpha < 5.0:
                return alpha, "smooth"
            return alpha, "singular"
        except Exception:
            return 1.0, "analytic"
