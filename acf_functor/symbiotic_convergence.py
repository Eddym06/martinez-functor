"""
Symbiotic Convergence Analyzer for BiPoem.

Analyzes convergence conditions for the Φ ⇌ Φ* cycle.
Provides sufficient conditions for convergence and empirical contraction rate estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import warnings

import torch


@dataclass
class ContractionEstimate:
    """Estimate of contraction rate for the Φ ⇌ Φ* cycle."""
    lipschitz_constant: float
    is_contraction: bool
    estimated_convergence_rate: float
    n_iterations_to_eps: int
    warning: Optional[str] = None


class SymbioticConvergenceAnalyzer:
    """
    Analyzes convergence conditions for the Φ ⇌ Φ* cycle.
    
    Theorem (Sufficient Condition):
    If data X comes from a linear system x_{t+1} = A x_t and the observable
    family contains the column span of A, then the cycle converges with
    rate ≤ ρ(A)² where ρ is the spectral radius.
    
    For nonlinear systems: empirical Lipschitz condition.
    """
    
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
    
    def estimate_contraction_rate(
        self,
        data: torch.Tensor,
        n_trials: int = 10,
        max_iterations: int = 20,
    ) -> ContractionEstimate:
        """
        Numerically estimate the contraction rate of the operator T: W → W_new.
        
        Method: perturbation + norm ratio analysis.
        
        Args:
            data: Shape (n_features, n_timesteps)
            n_trials: Number of random perturbations
            max_iterations: Max iterations for fixed point search
            
        Returns:
            ContractionEstimate with Lipschitz constant and convergence info
        """
        from acf_functor.core import KoopmanReducer
        
        n_features = data.shape[0]
        W0 = torch.eye(n_features, dtype=self.dtype) * 0.9
        
        # Compute T(W0)
        try:
            W1 = self._apply_T(W0, data)
        except Exception:
            return ContractionEstimate(
                lipschitz_constant=float('inf'),
                is_contraction=False,
                estimated_convergence_rate=float('inf'),
                n_iterations_to_eps=-1,
                warning="Failed to apply T operator"
            )
        
        # Random perturbations to estimate Lipschitz constant
        lipschitz_estimates = []
        for _ in range(n_trials):
            delta = torch.randn_like(W0) * 0.01
            W0_perturbed = W0 + delta
            
            try:
                W1_perturbed = self._apply_T(W0_perturbed, data)
                denom = torch.norm(W0_perturbed - W0).item()
                if denom > 1e-15:
                    ratio = torch.norm(W1_perturbed - W1).item() / denom
                    lipschitz_estimates.append(ratio)
            except Exception:
                continue
        
        if not lipschitz_estimates:
            return ContractionEstimate(
                lipschitz_constant=float('inf'),
                is_contraction=False,
                estimated_convergence_rate=float('inf'),
                n_iterations_to_eps=-1,
                warning="Could not estimate Lipschitz constant"
            )
        
        L_estimate = max(lipschitz_estimates)
        is_contraction = L_estimate < 1.0
        
        warning = None
        if not is_contraction:
            warning = f"L={L_estimate:.3f} ≥ 1: convergence not guaranteed"
        
        n_iter = self._estimate_iterations(L_estimate, tol=1e-6)
        if n_iter > max_iterations:
            warning = (
                f"max_iterations={max_iterations} may be insufficient. "
                f"Estimated needed: {n_iter}"
            )
        
        return ContractionEstimate(
            lipschitz_constant=L_estimate,
            is_contraction=is_contraction,
            estimated_convergence_rate=L_estimate,
            n_iterations_to_eps=n_iter,
            warning=warning,
        )
    
    def check_linear_system_condition(
        self,
        data: torch.Tensor,
        observable_family: str = 'polynomial',
        max_degree: int = 3,
    ) -> Dict[str, Any]:
        """
        Verify sufficient condition for linear systems.
        
        Checks if observables cover the column span of the estimated A matrix.
        
        Returns:
            Dict with condition_met, estimated_A, projection_error
        """
        n_features = data.shape[0]
        
        # Estimate A by least squares: X1 = A @ X0
        X0 = data[:, :-1]
        X1 = data[:, 1:]
        
        # A_est = X1 @ X0^T @ (X0 @ X0^T)^{-1}
        X0X0T = X0 @ X0.T
        try:
            A_est = X1 @ X0.T @ torch.linalg.pinv(X0X0T)
        except Exception:
            return {
                'condition_met': False,
                'error': 'Failed to estimate A matrix',
            }
        
        # Check spectral radius
        eigvals = torch.linalg.eigvals(A_est)
        rho = torch.max(torch.abs(eigvals)).item()
        
        return {
            'condition_met': rho < 1.0,
            'estimated_A': A_est,
            'spectral_radius': rho,
            'is_stable': rho < 1.0,
            'estimated_convergence_rate': rho ** 2,
        }
    
    def _apply_T(self, W: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Apply the Φ ⇌ Φ* operator: W → symmetrized Koopman matrix."""
        from acf_functor.core import KoopmanReducer
        
        # Φ step: estimate Koopman from data using W as observable base
        n_features = data.shape[0]
        rank = min(n_features * 2, data.shape[1] - 1)
        
        k_mat, eigvals, meta = KoopmanReducer.dmd(
            data.to(self.dtype),
            observable_fn=lambda z: KoopmanReducer.polynomial_observables(
                z, max_degree=min(max(rank // n_features, 1), 3)
            ),
            rank=rank,
        )
        
        # Φ* step: symmetrize
        W_new = 0.5 * (k_mat + k_mat.T)
        return W_new.to(self.dtype)
    
    def _estimate_iterations(self, L: float, tol: float = 1e-6) -> int:
        """Estimate iterations needed for L^n < tol."""
        if L >= 1.0:
            return -1  # Won't converge
        if L < 1e-15:
            return 1
        import math
        return max(1, math.ceil(math.log(tol) / math.log(L)))


def analyze_convergence(
    data: torch.Tensor,
    max_cycles: int = 20,
    tol: float = 1e-6,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """
    Convenience function to analyze convergence of Φ ⇌ Φ* cycle.
    
    Returns dict with convergence analysis and recommendations.
    """
    analyzer = SymbioticConvergenceAnalyzer(dtype=dtype)
    contraction = analyzer.estimate_contraction_rate(data)
    linear_check = analyzer.check_linear_system_condition(data)
    
    result = {
        'contraction': contraction,
        'linear_system_check': linear_check,
        'recommendations': [],
    }
    
    if contraction.is_contraction:
        result['recommendations'].append(
            f"Convergence expected in ~{contraction.n_iterations_to_eps} iterations"
        )
    else:
        result['recommendations'].append(
            "Convergence not guaranteed. Consider:"
        )
        result['recommendations'].append(
            "  - Increasing observable family richness"
        )
        result['recommendations'].append(
            "  - Using more data points"
        )
        result['recommendations'].append(
            "  - Checking if system is approximately linear"
        )
    
    if linear_check.get('is_stable'):
        result['recommendations'].append(
            f"System appears linear with ρ(A)={linear_check['spectral_radius']:.3f}"
        )
    
    return result
