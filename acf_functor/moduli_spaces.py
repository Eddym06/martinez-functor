"""Moduli space exploration for FMA reductions (Evolution 9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .core import ChebyshevReducer, HornerReducer, ReductionResult


@dataclass
class ModuliPoint:
    degree: int
    domain: Tuple[float, float]
    reduction: ReductionResult
    epsilon: float
    computational_energy: int
    coordinates: Dict[str, float] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return float(self.epsilon) + 1e-6 * float(self.computational_energy)


@dataclass
class GeodesicPath:
    start: ModuliPoint
    end: ModuliPoint
    intermediate_points: List[ModuliPoint]
    total_length: float
    is_monotone: bool


@dataclass
class ModuliAnalysis:
    optimal_point: ModuliPoint
    explored_points: List[ModuliPoint]
    geodesic_to_optimal: Optional[GeodesicPath]
    curvature_at_optimal: float
    dimension_of_moduli: int
    local_minima_detected: int


class ModuliSpace:
    def __init__(
        self,
        degree_range: Tuple[int, int] = (4, 100),
        degree_step: int = 4,
        n_domain_splits: int = 5,
        n_probe: int = 5000,
        dtype: torch.dtype = torch.float64,
    ):
        self.degree_range = degree_range
        self.degree_step = degree_step
        self.n_domain_splits = n_domain_splits
        self.n_probe = n_probe
        self.dtype = dtype

    def explore(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        target_epsilon: Optional[float] = None,
    ) -> ModuliAnalysis:
        points = self._sample_moduli(f, domain)
        if not points:
            raise RuntimeError("Could not generate any valid reductions.")

        points = sorted(points, key=lambda p: p.cost)
        current_best = points[0]

        refined, geodesic = self._geodesic_descent(f, domain, current_best, target_epsilon)
        if refined.cost < current_best.cost:
            current_best = refined

        curvature = self._estimate_curvature(f, domain, current_best)
        local_minima = self._count_local_minima(points)

        return ModuliAnalysis(
            optimal_point=current_best,
            explored_points=points,
            geodesic_to_optimal=geodesic,
            curvature_at_optimal=curvature,
            dimension_of_moduli=2,
            local_minima_detected=local_minima,
        )

    def find_optimal_reduction(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        target_epsilon: float = 1e-8,
    ) -> ReductionResult:
        return self.explore(f, domain, target_epsilon).optimal_point.reduction

    def _sample_moduli(self, f: Callable[[torch.Tensor], torch.Tensor], domain: Tuple[float, float]) -> List[ModuliPoint]:
        points: List[ModuliPoint] = []
        for degree in range(self.degree_range[0], self.degree_range[1] + 1, self.degree_step):
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y_approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                y_exact = f(x)
                eps = float(torch.max(torch.abs(y_exact - y_approx)).item())
                points.append(
                    ModuliPoint(
                        degree=degree,
                        domain=domain,
                        reduction=reduction,
                        epsilon=eps,
                        computational_energy=degree,
                        coordinates={"degree": float(degree), "domain_width": float(domain[1] - domain[0])},
                    )
                )
            except Exception:
                continue
        return points

    def _geodesic_descent(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        start: ModuliPoint,
        target_epsilon: Optional[float],
    ) -> Tuple[ModuliPoint, GeodesicPath]:
        path_points = [start]
        best = start
        degree = start.degree

        for _ in range(20):
            degree += 2
            if degree > self.degree_range[1]:
                break
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y_approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                eps = float(torch.max(torch.abs(f(x) - y_approx)).item())
                point = ModuliPoint(
                    degree=degree,
                    domain=domain,
                    reduction=reduction,
                    epsilon=eps,
                    computational_energy=degree,
                    coordinates={"degree": float(degree)},
                )
                path_points.append(point)
                if point.cost < best.cost:
                    best = point
                if target_epsilon is not None and eps < target_epsilon:
                    break
                if len(path_points) >= 3:
                    improvement = path_points[-3].epsilon - point.epsilon
                    if improvement < point.epsilon * 0.01:
                        break
            except Exception:
                break

        eps = [p.epsilon for p in path_points]
        is_monotone = all(eps[i] >= eps[i + 1] - 1e-15 for i in range(len(eps) - 1))
        total_length = float(
            sum(abs(path_points[i + 1].degree - path_points[i].degree) for i in range(len(path_points) - 1))
        )
        return (
            best,
            GeodesicPath(
                start=start,
                end=best,
                intermediate_points=path_points,
                total_length=total_length,
                is_monotone=is_monotone,
            ),
        )

    def _estimate_curvature(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        point: ModuliPoint,
    ) -> float:
        degree = point.degree
        eps_by_degree: Dict[int, float] = {}

        for delta in [-2, -1, 0, 1, 2]:
            d = degree + delta
            if d < 2:
                continue
            try:
                reduction = ChebyshevReducer.reduce(f, degree=d, domain=domain, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
                y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                eps_by_degree[d] = float(torch.max(torch.abs(f(x) - y)).item())
            except Exception:
                continue

        if len(eps_by_degree) < 3:
            return 0.0

        ds = sorted(eps_by_degree.keys())
        mid = len(ds) // 2
        d0, d1, d2 = ds[mid - 1], ds[mid], ds[mid + 1]
        h = d2 - d1
        if h <= 0:
            return 0.0
        return abs(eps_by_degree[d2] - 2.0 * eps_by_degree[d1] + eps_by_degree[d0]) / float(h**2)

    def _count_local_minima(self, points: List[ModuliPoint]) -> int:
        if len(points) < 3:
            return 1 if points else 0
        pts = sorted(points, key=lambda p: p.degree)
        minima = 0
        for i in range(1, len(pts) - 1):
            if pts[i].epsilon < pts[i - 1].epsilon and pts[i].epsilon < pts[i + 1].epsilon:
                minima += 1
        return max(minima, 1)
