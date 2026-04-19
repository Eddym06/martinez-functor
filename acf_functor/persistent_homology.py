"""Persistent homology over reduction filtrations (Evolution 10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import math

import numpy as np
import torch

from .core import ChebyshevReducer, HornerReducer


@dataclass
class PersistenceBar:
    birth_scale: int
    death_scale: int
    persistence: float
    feature_type: str
    amplitude: float

    @property
    def is_significant(self) -> bool:
        return self.persistence >= 3


@dataclass
class PersistenceDiagram:
    bars: List[PersistenceBar]
    optimal_scale: int
    noise_floor: float
    total_persistence: float
    n_significant_features: int

    def get_significant_bars(self) -> List[PersistenceBar]:
        return [bar for bar in self.bars if bar.is_significant]


@dataclass
class SpectralPersistenceMap:
    diagram: PersistenceDiagram
    optimal_degree: int
    optimal_epsilon: float
    acf_alpha: float
    complexity_class: str

    def summary(self) -> str:
        return (
            f"SPM(f): optimal_degree={self.optimal_degree}, "
            f"epsilon={self.optimal_epsilon:.2e}, alpha={self.acf_alpha:.4f}, "
            f"class={self.complexity_class}, features={self.diagram.n_significant_features}"
        )


class PersistentHomologyEngine:
    def __init__(
        self,
        min_scale: int = 2,
        max_scale: int = 100,
        scale_step: int = 2,
        n_probe: int = 5000,
        noise_percentile: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step = scale_step
        self.n_probe = n_probe
        self.noise_percentile = noise_percentile
        self.dtype = dtype

    def compute_spm(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> SpectralPersistenceMap:
        filtration = self._build_filtration(f, domain)
        if not filtration:
            diagram = PersistenceDiagram(
                bars=[],
                optimal_scale=self.min_scale,
                noise_floor=0.0,
                total_persistence=0.0,
                n_significant_features=0,
            )
            return SpectralPersistenceMap(diagram, self.min_scale, float("inf"), 0.0, "unknown")

        diagram = self._compute_persistence(filtration)
        optimal_degree = diagram.optimal_scale

        optimal_entry = None
        for entry in filtration:
            if entry["degree"] == optimal_degree:
                optimal_entry = entry
                break
        if optimal_entry is None:
            optimal_entry = min(filtration, key=lambda row: row["epsilon"])
            optimal_degree = int(optimal_entry["degree"])

        alpha = self._estimate_alpha(filtration)
        complexity = self._classify_complexity(filtration, diagram, alpha)
        return SpectralPersistenceMap(
            diagram=diagram,
            optimal_degree=optimal_degree,
            optimal_epsilon=float(optimal_entry["epsilon"]),
            acf_alpha=alpha,
            complexity_class=complexity,
        )

    def find_optimal_degree(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> Tuple[int, float]:
        spm = self.compute_spm(f, domain)
        return spm.optimal_degree, spm.optimal_epsilon

    def _build_filtration(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        filtration: List[Dict[str, Any]] = []
        x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
        f_exact = f(x)

        for degree in range(self.min_scale, self.max_scale + 1, self.scale_step):
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y_approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                abs_err = torch.abs(f_exact - y_approx)
                eps = float(torch.max(abs_err).item())
                mean_eps = float(torch.mean(abs_err).item())
                filtration.append(
                    {
                        "degree": degree,
                        "epsilon": eps,
                        "mean_epsilon": mean_eps,
                        "log_epsilon": math.log(eps + 1e-30),
                        "energy": degree,
                        "reduction": reduction,
                    }
                )
            except Exception:
                continue
        return filtration

    def _compute_persistence(self, filtration: List[Dict[str, Any]]) -> PersistenceDiagram:
        if len(filtration) < 2:
            only = filtration[0]["degree"] if filtration else self.min_scale
            return PersistenceDiagram([], only, 0.0, 0.0, 0)

        degrees = [int(row["degree"]) for row in filtration]
        eps = [float(row["epsilon"]) for row in filtration]
        log_eps = [float(row["log_epsilon"]) for row in filtration]

        rates: List[float] = []
        for i in range(1, len(log_eps)):
            rates.append((log_eps[i] - log_eps[i - 1]) / max(degrees[i] - degrees[i - 1], 1))

        bars: List[PersistenceBar] = []

        in_plateau = False
        plateau_start = 0
        for i, rate in enumerate(rates):
            if abs(rate) < 0.01 and not in_plateau:
                in_plateau = True
                plateau_start = i
            elif abs(rate) >= 0.01 and in_plateau:
                bars.append(
                    PersistenceBar(
                        birth_scale=degrees[plateau_start],
                        death_scale=degrees[i + 1],
                        persistence=float(degrees[i + 1] - degrees[plateau_start]),
                        feature_type="error_plateau",
                        amplitude=eps[plateau_start],
                    )
                )
                in_plateau = False

        if in_plateau:
            bars.append(
                PersistenceBar(
                    birth_scale=degrees[plateau_start],
                    death_scale=degrees[-1],
                    persistence=float(degrees[-1] - degrees[plateau_start]),
                    feature_type="error_plateau",
                    amplitude=eps[plateau_start],
                )
            )

        for i, rate in enumerate(rates):
            if rate < -0.5:
                bars.append(
                    PersistenceBar(
                        birth_scale=degrees[i],
                        death_scale=degrees[i + 1],
                        persistence=float(degrees[i + 1] - degrees[i]),
                        feature_type="convergence_event",
                        amplitude=abs(eps[i] - eps[i + 1]),
                    )
                )

        optimal = degrees[-1]
        for i in range(len(eps) - 1, 0, -1):
            improvement = eps[i - 1] - eps[i]
            if improvement > eps[i] * 0.1:
                optimal = degrees[i]
                break

        if eps[-1] < 1e-12:
            for i, value in enumerate(eps):
                if value < 1e-12:
                    optimal = degrees[i]
                    break

        persists = [bar.persistence for bar in bars] if bars else [0.0]
        noise_floor = float(np.percentile(persists, self.noise_percentile * 100.0)) if persists else 0.0
        total = float(sum(bar.persistence for bar in bars))
        significant = [bar for bar in bars if bar.is_significant]
        return PersistenceDiagram(bars, optimal, noise_floor, total, len(significant))

    def _estimate_alpha(self, filtration: List[Dict[str, Any]]) -> float:
        if len(filtration) < 3:
            return 1.0
        valid = [row for row in filtration if row["epsilon"] > 1e-30]
        if len(valid) < 3:
            return 1.0

        degrees = np.array([row["degree"] for row in valid], dtype=np.float64)
        log_eps = np.array([row["log_epsilon"] for row in valid], dtype=np.float64)

        try:
            coeffs = np.polyfit(degrees, log_eps, 1)
            slope = abs(float(coeffs[0]))
            if slope > 0.1:
                alpha = 1.0
            elif slope > 0.01:
                alpha = 1.0 / slope
            else:
                alpha = float("inf")
        except Exception:
            alpha = 1.0
        return min(alpha, 10.0)

    def _classify_complexity(
        self,
        filtration: List[Dict[str, Any]],
        diagram: PersistenceDiagram,
        alpha: float,
    ) -> str:
        if not filtration:
            return "unknown"
        min_eps = min(float(row["epsilon"]) for row in filtration)
        if min_eps < 1e-14:
            return "polynomial"
        if alpha <= 1.5:
            return "analytic"
        if alpha <= 5.0:
            return "smooth"
        return "singular"
