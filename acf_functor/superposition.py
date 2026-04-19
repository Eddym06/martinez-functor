"""Superposition of reduction forms (Evolution 13)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import math

import torch

from .core import ChebyshevReducer, HornerReducer, ReductionResult
from .moduli_spaces import ModuliPoint


@dataclass
class SuperpositionState:
    candidates: List[ModuliPoint]
    amplitudes: torch.Tensor
    phases: torch.Tensor
    collapsed: bool = False
    optimal_index: Optional[int] = None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def entropy(self) -> float:
        probs = self.amplitudes**2
        probs = probs / (probs.sum() + 1e-30)
        return float((-(probs * torch.log2(probs + 1e-30))).sum().item())


@dataclass
class CollapseResult:
    optimal: ModuliPoint
    interference_score: float
    entropy_before: float
    entropy_after: float
    n_candidates_evaluated: int
    selection_confidence: float


class SuperpositionEngine:
    def __init__(
        self,
        degree_range: Tuple[int, int] = (4, 80),
        degree_step: int = 2,
        n_probe: int = 5000,
        temperature: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        self.degree_range = degree_range
        self.degree_step = degree_step
        self.n_probe = n_probe
        self.temperature = temperature
        self.dtype = dtype

    def superpose(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> SuperpositionState:
        candidates = self._generate_candidates(f, domain)
        if not candidates:
            raise RuntimeError("No valid candidates generated.")

        amplitudes = self._compute_amplitudes(candidates)
        phases = self._compute_phases(candidates, f, domain)
        return SuperpositionState(candidates=candidates, amplitudes=amplitudes, phases=phases)

    def collapse(
        self,
        state: SuperpositionState,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> CollapseResult:
        entropy_before = state.entropy
        interference = self._compute_interference(state, f, domain)

        scores = torch.zeros(state.n_candidates, dtype=self.dtype)
        for i in range(state.n_candidates):
            weighted = (state.amplitudes * interference[i]).sum()
            scores[i] = state.amplitudes[i] * weighted

        optimal_idx = int(scores.argmax().item())
        optimal = state.candidates[optimal_idx]

        sorted_scores = scores.sort(descending=True).values
        if len(sorted_scores) > 1 and sorted_scores[1] > 0:
            confidence = 1.0 - float((sorted_scores[1] / sorted_scores[0]).item())
        else:
            confidence = 1.0

        return CollapseResult(
            optimal=optimal,
            interference_score=float(scores[optimal_idx].item()),
            entropy_before=entropy_before,
            entropy_after=0.0,
            n_candidates_evaluated=state.n_candidates,
            selection_confidence=confidence,
        )

    def find_optimal(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> Tuple[ReductionResult, CollapseResult]:
        state = self.superpose(f, domain)
        result = self.collapse(state, f, domain)
        return result.optimal.reduction, result

    def _generate_candidates(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> List[ModuliPoint]:
        candidates: List[ModuliPoint] = []
        x = torch.linspace(domain[0], domain[1], self.n_probe, dtype=self.dtype)
        f_exact = f(x)

        for degree in range(self.degree_range[0], self.degree_range[1] + 1, self.degree_step):
            try:
                reduction = ChebyshevReducer.reduce(f, degree=degree, domain=domain, dtype=self.dtype)
                coeffs = reduction.metadata.get(
                    "monomial_coefficients",
                    reduction.metadata.get("coefficients", []),
                )
                y_approx = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                eps = float(torch.max(torch.abs(f_exact - y_approx)).item())
                candidates.append(
                    ModuliPoint(
                        degree=degree,
                        domain=domain,
                        reduction=reduction,
                        epsilon=eps,
                        computational_energy=degree,
                    )
                )
            except Exception:
                continue

        return candidates

    def _compute_amplitudes(self, candidates: List[ModuliPoint]) -> torch.Tensor:
        n = len(candidates)
        amplitudes = torch.zeros(n, dtype=self.dtype)
        for i, candidate in enumerate(candidates):
            cost = math.log(candidate.epsilon + 1e-30) + 0.01 * candidate.computational_energy
            amplitudes[i] = math.exp(-cost / self.temperature)
        total = amplitudes.sum()
        if total > 0:
            amplitudes /= total
        return amplitudes

    def _compute_phases(
        self,
        candidates: List[ModuliPoint],
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> torch.Tensor:
        n = len(candidates)
        phases = torch.zeros(n, dtype=self.dtype)
        x = torch.linspace(domain[0], domain[1], min(self.n_probe, 500), dtype=self.dtype)
        f_exact = f(x)

        for i, candidate in enumerate(candidates):
            coeffs = candidate.reduction.metadata.get(
                "monomial_coefficients",
                candidate.reduction.metadata.get("coefficients", []),
            )
            if coeffs:
                y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                residual = f_exact - y
                abs_r = torch.abs(residual)
                total = abs_r.sum()
                if total > 0:
                    phases[i] = float(((x * abs_r).sum() / total).item())

        return phases

    def _compute_interference(
        self,
        state: SuperpositionState,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
    ) -> torch.Tensor:
        n = state.n_candidates
        x = torch.linspace(domain[0], domain[1], min(self.n_probe, 500), dtype=self.dtype)
        f_exact = f(x)

        residuals = torch.zeros(n, len(x), dtype=self.dtype)
        for i, candidate in enumerate(state.candidates):
            coeffs = candidate.reduction.metadata.get(
                "monomial_coefficients",
                candidate.reduction.metadata.get("coefficients", []),
            )
            if coeffs:
                y = HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=self.dtype), x)
                residuals[i] = f_exact - y

        norms = torch.norm(residuals, dim=1, keepdim=True).clamp(min=1e-30)
        normalized = residuals / norms
        interference = normalized @ normalized.T
        interference = 1.0 - torch.abs(interference)
        for i in range(n):
            interference[i, i] = 1.0
        return interference
