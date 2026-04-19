"""
Functorial Composition Engine
============================
Rigorous composition for ACF reductions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import torch

from .core import (
    ReductionResult,
    ReductionPath,
    HornerReducer,
    ChebyshevReducer,
    ACFInvariant,
)


def _get_monomial_coeffs(reduction: ReductionResult, dtype: torch.dtype) -> Optional[torch.Tensor]:
    coeffs = reduction.metadata.get(
        "monomial_coefficients",
        reduction.metadata.get("coefficients", None),
    )
    if coeffs is None:
        return None
    return torch.as_tensor(coeffs, dtype=dtype)


def _eval_reduction(reduction: ReductionResult, x: torch.Tensor) -> torch.Tensor:
    coeffs = _get_monomial_coeffs(reduction, x.dtype)
    if coeffs is not None:
        return HornerReducer.execute_horner(coeffs.to(device=x.device, dtype=x.dtype), x)

    if reduction.path == ReductionPath.CHEBYSHEV_APPROX and reduction.domain is not None:
        cheb = reduction.metadata.get("chebyshev_coefficients", None)
        if cheb is not None:
            return ChebyshevReducer.evaluate_chebyshev_series(
                torch.as_tensor(cheb, dtype=x.dtype, device=x.device),
                x,
                reduction.domain,
            )

    return reduction.to(device=x.device, dtype=x.dtype).execute(x)


class LipschitzEstimator:
    @staticmethod
    def estimate_from_reduction(
        reduction: ReductionResult,
        domain: Tuple[float, float],
        n_samples: int = 5000,
        dtype: torch.dtype = torch.float64,
    ) -> float:
        coeffs = _get_monomial_coeffs(reduction, dtype)
        if coeffs is not None:
            return LipschitzEstimator._lipschitz_polynomial(coeffs, domain[0], domain[1], n_samples)

        x = torch.linspace(domain[0], domain[1], n_samples, dtype=dtype)
        y = _eval_reduction(reduction, x)
        dx = (domain[1] - domain[0]) / max(n_samples - 1, 1)
        if dx <= 0:
            return 0.0
        dy = torch.abs(y[1:] - y[:-1]) / dx
        return float((dy.max() * 1.1).item()) if dy.numel() else 0.0

    @staticmethod
    def _lipschitz_polynomial(
        coeffs: torch.Tensor,
        a: float,
        b: float,
        n_samples: int,
    ) -> float:
        if coeffs.numel() <= 1:
            return 0.0

        deriv = torch.zeros(coeffs.numel() - 1, dtype=coeffs.dtype)
        for i in range(1, coeffs.numel()):
            deriv[i - 1] = i * coeffs[i]

        x = torch.linspace(a, b, n_samples, dtype=coeffs.dtype)
        dy = HornerReducer.execute_horner(deriv, x)
        return float(torch.abs(dy).max().item())


class DomainMapper:
    @staticmethod
    def estimate_output_range(
        reduction: ReductionResult,
        input_domain: Tuple[float, float],
        n_samples: int = 5000,
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[float, float]:
        x = torch.linspace(input_domain[0], input_domain[1], n_samples, dtype=dtype)
        y = _eval_reduction(reduction, x)
        return float(y.min().item()), float(y.max().item())

    @staticmethod
    def compute_affine_rescale(
        source_range: Tuple[float, float],
        target_domain: Tuple[float, float],
    ) -> Tuple[float, float]:
        s0, s1 = source_range
        t0, t1 = target_domain
        width = s1 - s0
        if abs(width) < 1e-30:
            return 0.0, 0.5 * (t0 + t1)
        scale = (t1 - t0) / width
        shift = t0 - scale * s0
        return scale, shift

    @staticmethod
    def needs_remapping(
        inner_output_range: Tuple[float, float],
        outer_domain: Optional[Tuple[float, float]],
        tolerance: float = 0.1,
    ) -> bool:
        if outer_domain is None:
            return False
        o0, o1 = inner_output_range
        d0, d1 = outer_domain
        margin = (d1 - d0) * tolerance
        return (o0 < d0 - margin) or (o1 > d1 + margin)


class PolynomialComposer:
    @staticmethod
    def compose_polynomials(
        outer_coeffs: torch.Tensor,
        inner_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        dtype = outer_coeffs.dtype
        if outer_coeffs.numel() == 0 or inner_coeffs.numel() == 0:
            return torch.zeros(1, dtype=dtype)

        deg = (outer_coeffs.numel() - 1) * (inner_coeffs.numel() - 1)
        result = torch.zeros(deg + 1, dtype=dtype)

        g_power = torch.zeros(deg + 1, dtype=dtype)
        g_power[0] = 1.0
        result[0] = outer_coeffs[0]

        for i in range(1, outer_coeffs.numel()):
            g_power = PolynomialComposer._poly_mul(g_power, inner_coeffs, deg + 1)
            result[: g_power.numel()] += outer_coeffs[i] * g_power

        while result.numel() > 1 and torch.abs(result[-1]).item() < 1e-30:
            result = result[:-1]
        return result

    @staticmethod
    def _poly_mul(p: torch.Tensor, q: torch.Tensor, max_len: int) -> torch.Tensor:
        out = torch.zeros(min(p.numel() + q.numel() - 1, max_len), dtype=p.dtype)
        for i in range(p.numel()):
            for j in range(q.numel()):
                k = i + j
                if k < out.numel():
                    out[k] += p[i] * q[j]
        return out


@dataclass
class CompositionCertificate:
    epsilon_outer: float
    epsilon_inner: float
    lipschitz_outer: float
    epsilon_cross: float
    epsilon_total: float
    verification_error: float
    is_valid: bool


class CompositionCertifier:
    @staticmethod
    def certify(
        f_exact: Callable[[torch.Tensor], torch.Tensor],
        g_exact: Callable[[torch.Tensor], torch.Tensor],
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        phi_fg: ReductionResult,
        input_domain: Tuple[float, float],
        n_test: int = 10000,
        dtype: torch.dtype = torch.float64,
    ) -> CompositionCertificate:
        x = torch.linspace(input_domain[0], input_domain[1], n_test, dtype=dtype)
        exact = f_exact(g_exact(x))
        approx = _eval_reduction(phi_fg, x)
        measured = float(torch.max(torch.abs(exact - approx)).item())

        eps_f = float(phi_f.epsilon_bound)
        eps_g = float(phi_g.epsilon_bound)
        g_range = DomainMapper.estimate_output_range(phi_g, input_domain, dtype=dtype)
        Lf = LipschitzEstimator.estimate_from_reduction(phi_f, g_range, dtype=dtype)
        eps_cross = eps_f * eps_g
        eps_total = eps_f + Lf * eps_g + eps_cross

        return CompositionCertificate(
            epsilon_outer=eps_f,
            epsilon_inner=eps_g,
            lipschitz_outer=Lf,
            epsilon_cross=eps_cross,
            epsilon_total=eps_total,
            verification_error=measured,
            is_valid=(measured <= eps_total * 1.05 + 1e-12),
        )


class FunctorialComposer:
    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
        default_degree: int = 24,
        n_verification_points: int = 10000,
    ):
        self.dtype = dtype
        self.default_degree = default_degree
        self.n_verify = n_verification_points

    def compose(
        self,
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        f_exact: Optional[Callable] = None,
        g_exact: Optional[Callable] = None,
        input_domain: Optional[Tuple[float, float]] = None,
    ) -> ReductionResult:
        if input_domain is None:
            input_domain = phi_g.domain or (-5.0, 5.0)

        f_coeffs = _get_monomial_coeffs(phi_f, self.dtype)
        g_coeffs = _get_monomial_coeffs(phi_g, self.dtype)

        if f_coeffs is not None and g_coeffs is not None:
            return self._compose_poly_poly(phi_f, phi_g, input_domain)

        if phi_f.path == ReductionPath.STRATIFIED or phi_g.path == ReductionPath.STRATIFIED:
            return self._compose_with_stratified(phi_f, phi_g, f_exact, g_exact, input_domain)

        if f_exact is not None and g_exact is not None:
            return self._direct_reduce_composition(phi_f, phi_g, f_exact, g_exact, input_domain)

        return self._compose_general(phi_f, phi_g, input_domain)

    def compose_and_certify(
        self,
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        f_exact: Callable,
        g_exact: Callable,
        input_domain: Tuple[float, float],
    ) -> Tuple[ReductionResult, CompositionCertificate]:
        result = self.compose(phi_f, phi_g, f_exact, g_exact, input_domain)
        cert = CompositionCertifier.certify(
            f_exact,
            g_exact,
            phi_f,
            phi_g,
            result,
            input_domain,
            self.n_verify,
            self.dtype,
        )
        return result, cert

    def _compose_poly_poly(
        self,
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        input_domain: Tuple[float, float],
    ) -> ReductionResult:
        f_coeffs = _get_monomial_coeffs(phi_f, self.dtype)
        g_coeffs = _get_monomial_coeffs(phi_g, self.dtype)
        assert f_coeffs is not None and g_coeffs is not None

        composed = PolynomialComposer.compose_polynomials(f_coeffs, g_coeffs)
        result = HornerReducer.reduce(composed.tolist(), self.dtype)
        result.domain = input_domain
        result.metadata["composition"] = {
            "method": "exact_polynomial_composition",
            "f_degree": int(f_coeffs.numel() - 1),
            "g_degree": int(g_coeffs.numel() - 1),
            "result_degree": int(composed.numel() - 1),
            "epsilon": 0.0,
        }
        return result

    def _direct_reduce_composition(
        self,
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        f_exact: Callable,
        g_exact: Callable,
        input_domain: Tuple[float, float],
    ) -> ReductionResult:
        def fg(x: torch.Tensor) -> torch.Tensor:
            return f_exact(g_exact(x))

        degree = max(
            int(phi_f.metadata.get("degree", self.default_degree)),
            int(phi_g.metadata.get("degree", self.default_degree)),
        )
        result = ChebyshevReducer.reduce(fg, degree=degree, domain=input_domain, dtype=self.dtype)

        # Sequential bound for metadata traceability.
        g_range = DomainMapper.estimate_output_range(phi_g, input_domain, dtype=self.dtype)
        Lf = LipschitzEstimator.estimate_from_reduction(phi_f, g_range, dtype=self.dtype)
        seq_eps = float(phi_f.epsilon_bound) + Lf * float(phi_g.epsilon_bound) + float(phi_f.epsilon_bound) * float(phi_g.epsilon_bound)

        result.metadata["composition"] = {
            "method": "direct_chebyshev_of_composition",
            "f_path": phi_f.path.name,
            "g_path": phi_g.path.name,
            "sequential_epsilon": seq_eps,
            "direct_epsilon": float(result.epsilon_bound),
        }
        result.domain = input_domain
        return result

    def _compose_general(
        self,
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        input_domain: Tuple[float, float],
    ) -> ReductionResult:
        g_range = DomainMapper.estimate_output_range(phi_g, input_domain, dtype=self.dtype)
        Lf = LipschitzEstimator.estimate_from_reduction(phi_f, g_range, dtype=self.dtype)

        eps_f = float(phi_f.epsilon_bound)
        eps_g = float(phi_g.epsilon_bound)
        eps_cross = eps_f * eps_g
        eps_total = eps_f + Lf * eps_g + eps_cross

        return ReductionResult(
            path=ReductionPath.COMPOSITE,
            fma_sequence=phi_g.fma_sequence + phi_f.fma_sequence,
            computational_energy=phi_f.computational_energy + phi_g.computational_energy,
            epsilon_bound=eps_total,
            domain=input_domain,
            metadata={
                "method": "general_sequential_composition",
                "f_path": phi_f.path.name,
                "g_path": phi_g.path.name,
                "epsilon_f": eps_f,
                "epsilon_g": eps_g,
                "epsilon_cross": eps_cross,
                "lipschitz_f": Lf,
                "g_output_range": g_range,
            },
        )

    def _compose_with_stratified(
        self,
        phi_f: ReductionResult,
        phi_g: ReductionResult,
        f_exact: Optional[Callable],
        g_exact: Optional[Callable],
        input_domain: Tuple[float, float],
    ) -> ReductionResult:
        if f_exact is not None and g_exact is not None:
            return self._direct_reduce_composition(phi_f, phi_g, f_exact, g_exact, input_domain)

        warnings.warn("stratified composition without exact callables: using conservative epsilon propagation")
        eps_total = float(phi_f.epsilon_bound) + float(phi_g.epsilon_bound) + float(phi_f.epsilon_bound) * float(phi_g.epsilon_bound)
        return ReductionResult(
            path=ReductionPath.COMPOSITE,
            fma_sequence=phi_g.fma_sequence + phi_f.fma_sequence,
            computational_energy=phi_f.computational_energy + phi_g.computational_energy,
            epsilon_bound=eps_total,
            domain=input_domain,
            metadata={
                "method": "stratified_fallback",
                "f_path": phi_f.path.name,
                "g_path": phi_g.path.name,
            },
        )


def compose_with_invariant(
    phi_f: ReductionResult,
    phi_g: ReductionResult,
    eigenvalues: torch.Tensor,
) -> Tuple[ReductionResult, Tuple[float, float]]:
    composer = FunctorialComposer(dtype=eigenvalues.dtype)
    composed = composer.compose(phi_f, phi_g)
    alpha, delta = ACFInvariant.compute_alpha(eigenvalues)
    return composed, (alpha, delta)
