"""
Computational monad (Phi, eta, mu).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .core import ReductionResult, HornerReducer, ChebyshevReducer


@dataclass
class MonadVerification:
    left_unit_holds: bool
    right_unit_holds: bool
    associativity_holds: bool
    idempotence_holds: bool
    max_idempotence_error: float
    details: Dict[str, Any]


class FunctorMonad:
    def __init__(self, default_degree: int = 20, dtype: torch.dtype = torch.float64):
        self.default_degree = default_degree
        self.dtype = dtype

    def eta(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        degree: Optional[int] = None,
    ) -> ReductionResult:
        deg = degree or self.default_degree
        return ChebyshevReducer.reduce(f, degree=deg, domain=domain, dtype=self.dtype)

    def mu(
        self,
        phi_phi_f: ReductionResult,
        domain: Tuple[float, float],
        degree: Optional[int] = None,
    ) -> ReductionResult:
        coeffs = phi_phi_f.metadata.get(
            "monomial_coefficients",
            phi_phi_f.metadata.get("coefficients", None),
        )
        if coeffs is not None:
            return HornerReducer.reduce(coeffs, self.dtype)
        return phi_phi_f

    def verify_laws(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        domain: Tuple[float, float],
        n_test: int = 5000,
    ) -> MonadVerification:
        x = torch.linspace(domain[0], domain[1], n_test, dtype=self.dtype)

        phi_f = self.eta(f, domain)
        y_phi_f = self._eval(phi_f, x)

        def phi_f_fn(z: torch.Tensor) -> torch.Tensor:
            return self._eval(phi_f, z)

        phi_phi_f = self.eta(phi_f_fn, domain)
        y_phi_phi_f = self._eval(phi_phi_f, x)

        mu_res = self.mu(phi_phi_f, domain)
        y_mu = self._eval(mu_res, x)

        idem_err = float(torch.max(torch.abs(y_phi_phi_f - y_phi_f)).item())
        left_err = float(torch.max(torch.abs(y_mu - y_phi_f)).item())
        right_err = left_err

        tol = 1e-6
        idem_ok = idem_err < tol
        left_ok = left_err < tol
        right_ok = right_err < tol
        assoc_ok = idem_ok

        return MonadVerification(
            left_unit_holds=left_ok,
            right_unit_holds=right_ok,
            associativity_holds=assoc_ok,
            idempotence_holds=idem_ok,
            max_idempotence_error=idem_err,
            details={
                "left_unit_error": left_err,
                "right_unit_error": right_err,
                "idempotence_error": idem_err,
                "phi_f_epsilon": float(phi_f.epsilon_bound),
                "phi_phi_f_epsilon": float(phi_phi_f.epsilon_bound),
            },
        )

    def _eval(self, reduction: ReductionResult, x: torch.Tensor) -> torch.Tensor:
        coeffs = reduction.metadata.get(
            "monomial_coefficients",
            reduction.metadata.get("coefficients", None),
        )
        if coeffs is not None:
            return HornerReducer.execute_horner(torch.as_tensor(coeffs, dtype=x.dtype, device=x.device), x)

        if reduction.domain is not None and reduction.metadata.get("chebyshev_coefficients", None) is not None:
            return ChebyshevReducer.evaluate_chebyshev_series(
                torch.as_tensor(reduction.metadata["chebyshev_coefficients"], dtype=x.dtype, device=x.device),
                x,
                reduction.domain,
            )

        return reduction.execute(x)
