"""
Cohomological obstruction analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from .core import ChebyshevReducer, HornerReducer


@dataclass
class CohomologyGroup:
    degree: int
    rank: int
    generators: List[Tuple[float, float]]
    obstruction_type: str
    is_trivial: bool


@dataclass
class CohomologicalAnalysis:
    h0: CohomologyGroup
    h1: CohomologyGroup
    euler_characteristic: int
    is_fully_reducible: bool
    obstruction_summary: str


class CohomologyAnalyzer:
    def __init__(
        self,
        n_probe: int = 5000,
        epsilon_threshold: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ):
        self.n_probe = n_probe
        self.epsilon_threshold = epsilon_threshold
        self.dtype = dtype

    def analyze(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: int = 20,
    ) -> CohomologicalAnalysis:
        a, b = domain
        x = torch.linspace(a, b, self.n_probe, dtype=self.dtype)

        try:
            reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
        except Exception:
            return CohomologicalAnalysis(
                h0=CohomologyGroup(0, 0, [], "failed", True),
                h1=CohomologyGroup(1, 1, [(a, b)], "total_failure", False),
                euler_characteristic=-1,
                is_fully_reducible=False,
                obstruction_summary="Reduction failed completely.",
            )

        coeffs = reduction.metadata.get(
            "monomial_coefficients",
            reduction.metadata.get("coefficients", []),
        )
        exact = f(x)
        approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)

        residual = exact - approx
        good = torch.abs(residual) < self.epsilon_threshold
        bad = ~good

        h0_gen = self._connected_intervals(good, x)
        h1_gen = self._connected_intervals(bad, x)

        kinds = [self._classify_obstruction(residual, x, itv) for itv in h1_gen]
        primary = max(set(kinds), key=kinds.count) if kinds else "none"

        h0 = CohomologyGroup(0, len(h0_gen), h0_gen, "components", len(h0_gen) <= 1)
        h1 = CohomologyGroup(1, len(h1_gen), h1_gen, primary, len(h1_gen) == 0)

        euler = h0.rank - h1.rank
        if h1.rank == 0:
            summary = f"Fully reducible. H^0 has {h0.rank} component(s)."
        else:
            summary = f"{h1.rank} obstruction(s) detected ({primary}). H^0={h0.rank}, H^1={h1.rank}, chi={euler}."

        return CohomologicalAnalysis(
            h0=h0,
            h1=h1,
            euler_characteristic=euler,
            is_fully_reducible=h1.is_trivial,
            obstruction_summary=summary,
        )

    def _connected_intervals(self, mask: torch.Tensor, x: torch.Tensor) -> List[Tuple[float, float]]:
        intervals: List[Tuple[float, float]] = []
        in_itv = False
        start = 0.0

        for i in range(mask.numel()):
            if bool(mask[i]) and not in_itv:
                start = float(x[i].item())
                in_itv = True
            elif (not bool(mask[i])) and in_itv:
                intervals.append((start, float(x[i - 1].item())))
                in_itv = False

        if in_itv:
            intervals.append((start, float(x[-1].item())))

        return intervals

    def _classify_obstruction(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        interval: Tuple[float, float],
    ) -> str:
        a, b = interval
        local = residual[(x >= a) & (x <= b)]
        if local.numel() < 3:
            return "point"

        sign_changes = int(((local[1:] * local[:-1]) < 0).sum().item())
        density = sign_changes / max(local.numel() - 1, 1)
        if density > 0.3:
            return "oscillatory"

        dx = x[1] - x[0]
        grad = torch.abs(local[1:] - local[:-1]) / dx
        if grad.numel() > 1:
            max_grad = float(grad.max().item())
            mean_grad = float(grad.mean().item())
            if mean_grad > 1e-15 and max_grad / mean_grad > 10.0:
                return "discontinuity"

        return "curvature"
