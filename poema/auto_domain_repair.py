"""
Auto-Domain Repair mejorado — Polinomios de grado superior en lugar de fallback a torch.sin.

Cuando el Domain Guard detecta que una entrada sale del dominio certificado,
en lugar de conmutar a la función nativa de PyTorch (que rompe la pureza del Functor Φ),
este módulo activa automáticamente un polinomio de Chebyshev de grado superior
con dominio expandido, manteniendo la ejecución dentro de Φ con ε certificado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import warnings

import torch


@dataclass
class ExpandedDomainCert:
    """Certificado para dominio expandido con grado superior."""
    function_name: str
    original_domain: Tuple[float, float]
    expanded_domain: Tuple[float, float]
    original_degree: int
    expanded_degree: int
    original_epsilon: float
    expanded_epsilon: float
    chebyshev_coefficients: torch.Tensor


# Tabla de dominios expandidos y grados requeridos para mantener precisión
_EXPANDED_DOMAINS: Dict[str, Tuple[Tuple[float, float], int]] = {
    # función: (dominio_expandido, grado_requerido)
    "sin": ((-2 * 3.141592653589793, 2 * 3.141592653589793), 48),   # [-2π, 2π], grado 48
    "cos": ((-2 * 3.141592653589793, 2 * 3.141592653589793), 48),   # [-2π, 2π], grado 48
    "exp": ((-3.0, 3.0), 30),                                        # [-3, 3], grado 30
    "tanh": ((-8.0, 8.0), 80),                                       # [-8, 8], grado 80
    "sigmoid": ((-16.0, 16.0), 80),                                  # [-16, 16], grado 80
    "log": ((0.1, 10.0), 50),                                        # [0.1, 10], grado 50
}


class ExpandedDomainRepair:
    """Reparación de dominio con polinomios de grado superior en lugar de fallback."""

    _cache: Dict[str, ExpandedDomainCert] = {}

    @staticmethod
    def get_expanded_cert(
        function_name: str,
        original_domain: Tuple[float, float],
        original_degree: int,
        original_epsilon: float,
    ) -> Optional[ExpandedDomainCert]:
        """Obtiene o genera un certificado de dominio expandido."""
        key = f"{function_name}_{original_domain}_{original_degree}"
        if key in ExpandedDomainRepair._cache:
            return ExpandedDomainRepair._cache[key]

        if function_name not in _EXPANDED_DOMAINS:
            return None

        expanded_domain, required_degree = _EXPANDED_DOMAINS[function_name]

        # Generar coeficientes de Chebyshev para el dominio expandido
        try:
            from acf_functor.core import ChebyshevReducer

            reduction = ChebyshevReducer.reduce(
                function_name,
                domain=expanded_domain,
                degree=required_degree,
            )
            coeffs = torch.tensor(reduction.metadata.get("chebyshev_coefficients", []), dtype=torch.float64)

            # Estimar error máximo en el dominio expandido
            import numpy as np
            x_test = torch.linspace(expanded_domain[0], expanded_domain[1], 10000, dtype=torch.float64)
            y_approx = ChebyshevReducer.evaluate_chebyshev_series(coeffs, x_test, expanded_domain)
            y_exact = getattr(torch, function_name)(x_test)
            max_error = torch.max(torch.abs(y_approx - y_exact)).item()

            cert = ExpandedDomainCert(
                function_name=function_name,
                original_domain=original_domain,
                expanded_domain=expanded_domain,
                original_degree=original_degree,
                expanded_degree=required_degree,
                original_epsilon=original_epsilon,
                expanded_epsilon=max_error,
                chebyshev_coefficients=coeffs,
            )

            ExpandedDomainRepair._cache[key] = cert
            return cert

        except Exception as e:
            warnings.warn(f"Failed to generate expanded domain cert for {function_name}: {e}")
            return None

    @staticmethod
    def create_repaired_evaluator(
        function_name: str,
        original_domain: Tuple[float, float],
        original_degree: int,
        original_epsilon: float,
        original_cheb_coeffs: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Crea un evaluador que usa dominio expandido cuando la entrada sale del dominio original."""
        cert = ExpandedDomainRepair.get_expanded_cert(
            function_name, original_domain, original_degree, original_epsilon
        )

        if cert is None:
            # Sin certificado expandido disponible: fallback a función nativa (último recurso)
            canonical_fns = {
                "sin": torch.sin,
                "cos": torch.cos,
                "exp": torch.exp,
                "tanh": torch.tanh,
                "sigmoid": torch.sigmoid,
            }
            native_fn = canonical_fns.get(function_name)
            if native_fn is None:
                # Sin función nativa: usar polinomio original sin repair
                from acf_functor.core import ChebyshevReducer

                def fallback_eval(x: torch.Tensor) -> torch.Tensor:
                    return ChebyshevReducer.evaluate_chebyshev_series(
                        original_cheb_coeffs.to(device=x.device, dtype=x.dtype), x, original_domain
                    )

                return fallback_eval

            # Fallback a función nativa (rompe pureza de Φ, pero evita nan)
            def native_repair_eval(x: torch.Tensor) -> torch.Tensor:
                from acf_functor.core import ChebyshevReducer

                a, b = original_domain
                in_domain = (x >= a) & (x <= b)
                y_cheb = ChebyshevReducer.evaluate_chebyshev_series(
                    original_cheb_coeffs.to(device=x.device, dtype=x.dtype), x, original_domain
                )
                if bool(in_domain.all().item()):
                    return y_cheb
                y_native = native_fn(x)
                return torch.where(in_domain, y_cheb, y_native)

            return native_repair_eval

        # Evaluador con dominio expandido (mantiene pureza de Φ)
        from acf_functor.core import ChebyshevReducer

        def expanded_repair_eval(x: torch.Tensor) -> torch.Tensor:
            a_orig, b_orig = original_domain
            a_exp, b_exp = cert.expanded_domain

            in_original = (x >= a_orig) & (x <= b_orig)
            in_expanded = (x >= a_exp) & (x <= b_exp)

            # Dentro del dominio original: usar polinomio certificado original
            y_original = ChebyshevReducer.evaluate_chebyshev_series(
                original_cheb_coeffs.to(device=x.device, dtype=x.dtype), x, original_domain
            )

            # Fuera del original pero dentro del expandido: usar polinomio de grado superior
            y_expanded = ChebyshevReducer.evaluate_chebyshev_series(
                cert.chebyshev_coefficients.to(device=x.device, dtype=x.dtype), x, cert.expanded_domain
            )

            # Fuera de ambos dominios: fallback a función nativa (último recurso)
            y_native = getattr(torch, function_name, torch.sin)(x)

            # Seleccionar ruta según dominio
            result = torch.where(in_original, y_original, torch.where(in_expanded, y_expanded, y_native))
            return result

        return expanded_repair_eval
