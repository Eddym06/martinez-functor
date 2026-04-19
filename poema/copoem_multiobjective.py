"""
CoPoem Multiobjetivo Mejorado — Método de Anderson y detección de incompatibilidad.

Este módulo reemplaza las proyecciones alternadas ingenuas con:
1. Método de punto fijo de Anderson para acelerar convergencia en conjuntos no convexos
2. Detección de estancamiento con relajación automática de restricciones conflictivas
3. Certificado de incompatibilidad analítica antes de iterar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch


@dataclass
class AndersonState:
    """Estado del método de Anderson para aceleración de punto fijo."""
    history_size: int = 5
    residuals: List[torch.Tensor] = field(default_factory=list)
    iterates: List[torch.Tensor] = field(default_factory=list)
    
    def add(self, W: torch.Tensor, residual: torch.Tensor):
        self.iterates.append(W.clone())
        self.residuals.append(residual.clone())
        # Mantener historial limitado
        if len(self.iterates) > self.history_size:
            self.iterates.pop(0)
            self.residuals.pop(0)
    
    def extrapolate(self, W_new: torch.Tensor) -> torch.Tensor:
        """Extrapolación de Anderson: W_anderson = W_new - sum(alpha_i * residual_i)"""
        if len(self.residuals) < 2:
            return W_new
        
        # Construir matriz de residuos R = [r_1, r_2, ..., r_k]
        R = torch.stack([r.flatten() for r in self.residuals], dim=1)  # (n, k)
        
        # Resolver mínimo cuadrados: min ||R @ alpha - r_k||^2
        try:
            alpha, _, _, _ = torch.linalg.lstsq(R[:, :-1], self.residuals[-1].flatten())
            # W_anderson = W_new - R @ alpha
            correction = R @ alpha
            W_anderson = W_new - correction.view_as(W_new)
            return W_anderson
        except Exception:
            # Fallback a iteración simple si lstsq falla
            return W_new


@dataclass
class IncompatibilityReport:
    """Reporte de incompatibilidad entre restricciones."""
    is_incompatible: bool = False
    conflicting_constraints: List[str] = field(default_factory=list)
    reason: str = ""
    suggestion: str = ""
    relaxation_needed: Optional[str] = None


class CoPoemMultiObjective:
    """Motor de síntesis multiobjetivo mejorado con Anderson y detección de incompatibilidad."""

    @staticmethod
    def check_compatibility(spec: Dict[str, Any]) -> IncompatibilityReport:
        """Verifica analíticamente si las restricciones son compatibles antes de iterar."""
        constraints = []
        report = IncompatibilityReport()
        
        spectral_radius = spec.get("spectral_radius")
        symmetry = spec.get("symmetry")
        lyapunov = spec.get("lyapunov_exponent")
        minimize_obj = spec.get("minimize_objective")
        minimize_budget = spec.get("minimize_budget")
        
        # Verificar compatibilidad radio espectral + Lyapunov
        if spectral_radius is not None and lyapunov is not None:
            target_max_real = torch.exp(torch.tensor(lyapunov, dtype=torch.float64)).item()
            if spectral_radius < target_max_real:
                report.is_incompatible = True
                report.conflicting_constraints = ["spectral_radius", "lyapunov_exponent"]
                report.reason = (
                    f"Radio espectral ({spectral_radius}) < exp(Lyapunov) ({target_max_real:.4f}). "
                    f"Las restricciones son contradictorias: el radio espectral impone |λ| < {spectral_radius}, "
                    f"pero Lyapunov requiere Re(λ) < {target_max_real}."
                )
                report.suggestion = f"Relajar radio espectral a >= {target_max_real:.4f} o Lyapunov a <= {torch.log(torch.tensor(spectral_radius)).item():.4f}"
                report.relaxation_needed = "lyapunov_exponent"
                return report
        
        # Verificar ortogonalidad + radio espectral arbitrario
        if symmetry == "orthogonal" and spectral_radius is not None:
            if abs(spectral_radius - 1.0) > 1e-6:
                report.is_incompatible = True
                report.conflicting_constraints = ["symmetry:orthogonal", "spectral_radius"]
                report.reason = (
                    f"Matrices ortogonales tienen radio espectral exactamente 1.0, "
                    f"pero se solicitó {spectral_radius}."
                )
                report.suggestion = "Establecer spectral_radius=1.0 para matrices ortogonales"
                report.relaxation_needed = "spectral_radius"
                return report
        
        # Verificar minimización de norma + radio espectral alto
        if minimize_obj == "frobenius_norm" and minimize_budget is not None and spectral_radius is not None:
            dim = spec.get("dimension", 64)
            # Norma mínima para un radio espectral dado: ||W||_F >= sqrt(dim) * rho
            min_frobenius = torch.sqrt(torch.tensor(dim, dtype=torch.float64)).item() * spectral_radius
            if minimize_budget < min_frobenius:
                report.is_incompatible = True
                report.conflicting_constraints = ["minimize:frobenius_norm", "spectral_radius"]
                report.reason = (
                    f"Presupuesto de norma Frobenius ({minimize_budget}) < mínimo teórico "
                    f"({min_frobenius:.4f}) para dim={dim} y rho={spectral_radius}."
                )
                report.suggestion = f"Aumentar minimize_budget a >= {min_frobenius:.4f}"
                report.relaxation_needed = "minimize_budget"
                return report
        
        return report

    @staticmethod
    def synthesize_with_anderson(
        spec: Dict[str, Any],
        dtype: torch.dtype = torch.float64,
        max_iter: int = 100,
        tol: float = 1e-8,
        anderson_history: int = 5,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Síntesis multiobjetivo con aceleración de Anderson y detección de estancamiento."""
        # Verificar compatibilidad primero
        compat_report = CoPoemMultiObjective.check_compatibility(spec)
        if compat_report.is_incompatible:
            warnings.warn(
                f"Restricciones incompatibles detectadas: {compat_report.reason}\n"
                f"Sugerencia: {compat_report.suggestion}"
            )
            # Continuar con restricciones relajadas si es posible
            if compat_report.relaxation_needed:
                spec = spec.copy()
                if compat_report.relaxation_needed == "spectral_radius":
                    spec["spectral_radius"] = 1.0
                elif compat_report.relaxation_needed == "lyapunov_exponent":
                    spec["lyapunov_exponent"] = torch.log(torch.tensor(spec.get("spectral_radius", 0.95))).item()
                elif compat_report.relaxation_needed == "minimize_budget":
                    dim = spec.get("dimension", 64)
                    spec["minimize_budget"] = torch.sqrt(torch.tensor(dim, dtype=torch.float64)).item() * spec.get("spectral_radius", 0.95)

        dim = spec.get("dimension", 64)
        W = torch.eye(dim, dtype=dtype) * 0.9  # Inicialización
        
        anderson = AndersonState(history_size=anderson_history)
        stagnation_counter = 0
        best_gap = float('inf')
        best_W = W.clone()
        gap_history = []
        
        for iteration in range(max_iter):
            W_prev = W.clone()
            
            # Aplicar proyecciones secuenciales
            W = CoPoemMultiObjective._apply_projections(W, spec, dtype)
            
            # Calcular residual
            residual = W - W_prev
            gap = torch.norm(residual, 'fro').item()
            
            # Anderson extrapolation
            anderson.add(W_prev, residual)
            W_anderson = anderson.extrapolate(W)
            
            # Verificar si Anderson mejora
            gap_anderson = torch.norm(W_anderson - W_prev, 'fro').item()
            if gap_anderson < gap:
                W = W_anderson
                gap = gap_anderson
            
            # Tracking de mejor resultado
            if gap < best_gap:
                best_gap = gap
                best_W = W.clone()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            gap_history.append(gap)
            
            # Detección de estancamiento: si no mejora en N iteraciones, relajar
            if stagnation_counter >= 10:
                warnings.warn(
                    f"Estancamiento detectado en iteración {iteration}. "
                    f"Mejor gap: {best_gap:.4e}. Relajando restricciones..."
                )
                # Relajar la restricción más conflictiva
                W = CoPoemMultiObjective._relax_constraints(W, spec, dtype)
                stagnation_counter = 0
                best_gap = gap
            
            if gap < tol:
                break
        
        # Calcular métricas finales
        report = {
            "converged": gap < tol,
            "iterations": iteration + 1,
            "final_gap": gap,
            "best_gap": best_gap,
            "stagnation_detected": stagnation_counter > 0,
            "gap_history": gap_history,
            "incompatibility_report": compat_report,
        }
        
        # Añadir métricas de adjunción
        spectral_radius_actual = torch.max(torch.abs(torch.linalg.eigvals(best_W))).item()
        report["spectral_radius_actual"] = spectral_radius_actual
        report["adjunction_gap"] = abs(spec.get("spectral_radius", 0.95) - spectral_radius_actual)
        report["frobenius_norm"] = torch.norm(best_W, 'fro').item()
        
        # Verificar simetría
        if spec.get("symmetry") == "symmetric":
            report["symmetry_verified"] = torch.norm(best_W - best_W.T, 'fro').item() < 1e-6
        elif spec.get("symmetry") == "orthogonal":
            WtW = best_W.T @ best_W
            report["symmetry_verified"] = torch.norm(WtW - torch.eye(dim, dtype=dtype), 'fro').item() < 1e-6
        else:
            report["symmetry_verified"] = True
        
        return best_W, report

    @staticmethod
    def _apply_projections(W: torch.Tensor, spec: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        """Aplica todas las proyecciones de restricciones."""
        # Proyección espectral
        if spec.get("spectral_radius") is not None:
            W = CoPoemMultiObjective._project_spectral_radius(W, spec["spectral_radius"], dtype)
        
        # Proyección de simetría
        symmetry = spec.get("symmetry")
        if symmetry == "symmetric":
            W = 0.5 * (W + W.T)
        elif symmetry == "orthogonal":
            U, _, Vt = torch.linalg.svd(W)
            W = U @ Vt * spec.get("spectral_radius", 1.0)
        
        # Proyección de Lyapunov
        if spec.get("lyapunov_exponent") is not None:
            W = CoPoemMultiObjective._project_lyapunov(W, spec["lyapunov_exponent"], dtype)
        
        # Minimización de norma
        if spec.get("minimize_objective") == "frobenius_norm" and spec.get("minimize_budget") is not None:
            fnorm = torch.norm(W, 'fro').item()
            if fnorm > spec["minimize_budget"]:
                W = W * (spec["minimize_budget"] / fnorm)
        
        return W

    @staticmethod
    def _project_spectral_radius(W: torch.Tensor, radius: float, dtype: torch.dtype) -> torch.Tensor:
        """Proyecta matriz sobre restricción de radio espectral."""
        eigvals, eigvecs = torch.linalg.eig(W)
        max_abs = torch.max(torch.abs(eigvals)).item()
        if max_abs > radius:
            scale = radius / max_abs
            eigvals_scaled = eigvals * scale
            W_real = torch.real(eigvecs @ torch.diag(eigvals_scaled) @ torch.linalg.inv(eigvecs))
            return W_real.to(dtype)
        return W

    @staticmethod
    def _project_lyapunov(W: torch.Tensor, lyap_exp: float, dtype: torch.dtype) -> torch.Tensor:
        """Proyecta matriz sobre restricción de estabilidad de Lyapunov."""
        eigvals = torch.linalg.eigvals(W)
        max_real = torch.max(eigvals.real).item()
        target_max = torch.exp(torch.tensor(lyap_exp, dtype=dtype)).item()
        if max_real > target_max:
            scale = target_max / max_real
            return W * scale
        return W

    @staticmethod
    def _relax_constraints(W: torch.Tensor, spec: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
        """Relaja las restricciones más conflictivas cuando hay estancamiento."""
        # Relajar radio espectral en 10%
        if spec.get("spectral_radius") is not None:
            spec = spec.copy()
            spec["spectral_radius"] *= 1.1
            W = CoPoemMultiObjective._project_spectral_radius(W, spec["spectral_radius"], dtype)
        
        # Relajar presupuesto de norma en 20%
        if spec.get("minimize_budget") is not None:
            spec = spec.copy()
            spec["minimize_budget"] *= 1.2
        
        return W
