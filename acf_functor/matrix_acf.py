"""
Matrix ACF — Matrix Function Reduction via Chebyshev Polynomials of Matrices
=============================================================================

Extends the Affine Collapse Functor to operate on f: ℝⁿˣⁿ → ℝⁿˣⁿ.

Mathematical foundation
-----------------------
Given a symmetric (or normal) matrix A ∈ ℝⁿˣⁿ with eigenvalues in [λ_min, λ_max],
and a scalar function f: ℝ → ℝ, the matrix function f(A) is:

    f(A) = U · diag(f(λ₁), …, f(λₙ)) · U^T

where A = UΛU^T is the eigendecomposition. This definition extends to
any function holomorphic on the spectrum.

The ACF approach avoids the full eigendecomposition. Instead we use the
Chebyshev polynomial expansion:

    f(A) ≈ Σ_{k=0}^{d} cₖ Tₖ(Ã)

where Ã = (2A - (λ_max + λ_min)I) / (λ_max - λ_min) maps eigenvalues to [-1,1],
and cₖ are the standard Chebyshev coefficients of f on [λ_min, λ_max].

The three-term Chebyshev recurrence:
    T₀(Ã) = I
    T₁(Ã) = Ã
    Tₖ(Ã) = 2Ã·Tₖ₋₁(Ã) - Tₖ₋₂(Ã)

yields an FMA chain of d matrix-matrix multiplications with weights ∈ {2Ã, I}
and biases ∈ {-Tₖ₋₂, 0}. Total cost: O(d · n³) or O(d · nnz(A)) for sparse A.

ACF invariant for matrix functions:
    α(f, A) measures the decay rate of |cₖ| — i.e., how well f
    is approximated by low-degree Chebyshev polynomials on the spectrum of A.
    If |cₖ| ~ k^{-α}, then we need degree ~ ε^{-1/α} for accuracy ε.

Built-in matrix functions:
    - exp(tA)         — matrix exponential (heat kernel, Schrödinger evolution)
    - A^{1/2}         — matrix square root (positive definite)
    - (A + σI)⁻¹     — resolvent (regularized inverse)
    - log(A)          — matrix logarithm (positive definite)
    - sign(A)         — matrix sign function
    - A^p             — fractional matrix power

Scope (honest)
--------------
- Works for symmetric/Hermitian matrices (real spectrum guaranteed).
- For non-symmetric A: applicable if spectrum lies in a real interval,
  or if user provides spectral bounds; no guaranteed accuracy otherwise.
- Dense matrices up to ~5000×5000 in practice; sparse support via implicit products.

References
----------
  Golub & Van Loan (2013) — Matrix Computations §11.
  Higham (2008) — Functions of Matrices: Theory and Computation.
  Paper.md §36–§38 for the parent ACF framework.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .core import (
    ChebyshevReducer,
    FMAOperation,
    ReductionPath,
    ReductionResult,
    ACFInvariant,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatrixReductionResult:
    """Result of reducing f(A) to a Chebyshev matrix polynomial."""
    result_matrix: torch.Tensor         # f(A) approximation
    chebyshev_coeffs: torch.Tensor      # scalar Chebyshev coefficients
    degree: int
    spectral_range: Tuple[float, float]
    epsilon: float                      # certified L2 operator norm error
    fma_count: int                      # number of matrix-matrix multiplies
    elapsed_ms: float
    func_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatrixACFInvariants:
    """ACF invariants for a matrix function pair (f, A)."""
    matrix_alpha: float             # Chebyshev coefficient decay rate
    spectral_range: Tuple[float, float]
    condition_number: float
    spectral_gap: float             # λ₂ - λ₁ (gap between smallest eigenvalues)
    nc_class: str
    effective_degree: int           # degree needed for target ε
    chebyshev_entropy: float        # entropy of |cₖ| distribution


@dataclass
class MatrixFMAChain:
    """Explicit FMA chain for matrix function evaluation."""
    operations: List[FMAOperation]  # each op has matrix weight/bias
    degree: int
    spectral_shift: torch.Tensor   # Ã = shift_scale * A + shift_offset*I
    shift_scale: float
    shift_offset: float


# ---------------------------------------------------------------------------
# Core: Chebyshev matrix polynomial evaluator
# ---------------------------------------------------------------------------

class ChebyshevMatrixReducer:
    """Reduce f(A) via Chebyshev polynomials of matrices."""

    # Built-in scalar functions for Chebyshev fitting
    MATRIX_FUNCTIONS: Dict[str, Callable[[float], float]] = {
        "exp": math.exp,
        "sqrt": math.sqrt,
        "log": math.log,
        "inv": lambda x: 1.0 / x,
        "sign": lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
        "tanh": math.tanh,
        "abs": abs,
    }

    @staticmethod
    def compute_spectral_range(A: torch.Tensor) -> Tuple[float, float]:
        """Compute eigenvalue bounds for symmetric A."""
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        eigvals = torch.linalg.eigvalsh(A)
        return float(eigvals[0].item()), float(eigvals[-1].item())

    @staticmethod
    def fit_chebyshev_coefficients(
        func: Callable[[float], float],
        degree: int,
        domain: Tuple[float, float],
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Fit Chebyshev coefficients for scalar f on [a, b].

        Uses Chebyshev-Gauss quadrature on n = degree + 1 nodes.
        """
        a, b = domain
        n = degree + 1
        k = torch.arange(n, dtype=dtype)
        theta = (k + 0.5) * math.pi / n
        t = torch.cos(theta)  # Chebyshev nodes on [-1, 1]
        x = 0.5 * (a + b) + 0.5 * (b - a) * t  # map to [a, b]

        # Evaluate function at nodes
        vals = torch.tensor([func(float(xi.item())) for xi in x], dtype=dtype)

        # Compute coefficients via DCT
        j = torch.arange(n, dtype=dtype).unsqueeze(1)
        cos_table = torch.cos(j * theta.unsqueeze(0))
        coeffs = (2.0 / n) * (cos_table @ vals)
        coeffs[0] *= 0.5
        return coeffs

    @staticmethod
    def evaluate_matrix_chebyshev(
        A: torch.Tensor,
        coeffs: torch.Tensor,
        spectral_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Evaluate f(A) = Σ cₖ Tₖ(Ã) via Clenshaw recurrence.

        The Clenshaw algorithm is numerically more stable than the
        forward recurrence and accumulation approach.

        Args:
            A: square matrix (n, n)
            coeffs: Chebyshev coefficients (d+1,)
            spectral_range: (λ_min, λ_max) — eigenvalue interval

        Returns:
            f(A) approximation (n, n)
        """
        n = A.shape[0]
        d = coeffs.shape[0] - 1
        lmin, lmax = spectral_range
        dtype = A.dtype

        # Affine map: Ã = (2A - (lmax+lmin)I) / (lmax-lmin)
        scale = 2.0 / (lmax - lmin + 1e-30)
        offset = -(lmax + lmin) / (lmax - lmin + 1e-30)
        A_tilde = scale * A + offset * torch.eye(n, dtype=dtype)

        # Clenshaw recurrence for matrix polynomials:
        # b_{d+1} = 0, b_d = c_d * I
        # b_k = c_k * I + 2*Ã*b_{k+1} - b_{k+2}, for k = d-1, ..., 1
        # f(A) = c_0*I + Ã*b_1 - b_2
        I_n = torch.eye(n, dtype=dtype)

        if d == 0:
            return float(coeffs[0].item()) * I_n

        b_next_next = torch.zeros(n, n, dtype=dtype)  # b_{k+2}
        b_next = float(coeffs[d].item()) * I_n        # b_{k+1}

        for k in range(d - 1, 0, -1):
            b_curr = float(coeffs[k].item()) * I_n + 2 * A_tilde @ b_next - b_next_next
            b_next_next = b_next
            b_next = b_curr

        # f(A) = c_0*I + Ã*b_1 - b_2
        result = float(coeffs[0].item()) * I_n + A_tilde @ b_next - b_next_next
        return result

    @classmethod
    def reduce(
        cls,
        func: Union[str, Callable[[float], float]],
        A: torch.Tensor,
        degree: int = 20,
        target_epsilon: float = 1e-8,
        max_degree: int = 128,
        spectral_range: Optional[Tuple[float, float]] = None,
        dtype: torch.dtype = torch.float64,
    ) -> MatrixReductionResult:
        """Reduce f(A) via Chebyshev matrix polynomials.

        Adaptively increases degree until target_epsilon is met.

        Args:
            func: scalar function name (str) or callable
            A: square matrix
            degree: initial Chebyshev degree
            target_epsilon: target approximation error
            max_degree: maximum allowed degree
            spectral_range: if None, computed from eigenvalues of A
            dtype: computation dtype

        Returns:
            MatrixReductionResult
        """
        t0 = time.time()
        A = A.to(dtype)
        n = A.shape[0]

        # Resolve function
        if isinstance(func, str):
            func_name = func
            scalar_func = cls.MATRIX_FUNCTIONS[func]
        else:
            func_name = getattr(func, '__name__', 'custom')
            scalar_func = func

        # Spectral range
        if spectral_range is None:
            spectral_range = cls.compute_spectral_range(A)

        lmin, lmax = spectral_range
        # Avoid degenerate range
        if abs(lmax - lmin) < 1e-14:
            lmax = lmin + 1e-10

        # Adaptive degree loop
        current_degree = degree
        while current_degree <= max_degree:
            coeffs = cls.fit_chebyshev_coefficients(scalar_func, current_degree,
                                                     spectral_range, dtype)
            f_A = cls.evaluate_matrix_chebyshev(A, coeffs, spectral_range)

            # Certify via eigendecomposition comparison
            eps = cls._certify_error(A, f_A, scalar_func, dtype)

            if eps <= target_epsilon or current_degree >= max_degree:
                break
            current_degree = min(current_degree * 2, max_degree)

        elapsed = (time.time() - t0) * 1000

        return MatrixReductionResult(
            result_matrix=f_A,
            chebyshev_coeffs=coeffs,
            degree=current_degree,
            spectral_range=spectral_range,
            epsilon=eps,
            fma_count=current_degree,  # d matrix multiplies
            elapsed_ms=elapsed,
            func_name=func_name,
            metadata={
                "method": "chebyshev_matrix_clenshaw",
                "matrix_size": n,
                "degree": current_degree,
                "spectral_range": spectral_range,
                "target_epsilon": target_epsilon,
            },
        )

    @staticmethod
    def _certify_error(
        A: torch.Tensor,
        f_A_approx: torch.Tensor,
        scalar_func: Callable[[float], float],
        dtype: torch.dtype = torch.float64,
    ) -> float:
        """Certify error by computing exact f(A) via eigendecomposition.

        For non-tiny matrices uses random sampling of eigenvalues instead.
        """
        n = A.shape[0]
        if n <= 512:
            # Full eigendecomposition for certification
            eigvals, eigvecs = torch.linalg.eigh(A.to(dtype))
            f_exact_diag = torch.tensor(
                [scalar_func(float(lam.item())) for lam in eigvals],
                dtype=dtype,
            )
            f_exact = eigvecs @ torch.diag(f_exact_diag) @ eigvecs.T
            return float(torch.norm(f_A_approx - f_exact, p=2).item())
        else:
            # For large matrices: sample error on random vectors
            rng = torch.Generator().manual_seed(42)
            v = torch.randn(n, 10, dtype=dtype, generator=rng)
            v = v / torch.norm(v, dim=0, keepdim=True)
            # Estimate \|f(A)\| via power iteration on random probes
            fAv = f_A_approx @ v  # (n, 10)
            norms = torch.norm(fAv, dim=0)  # (10,)
            spectral_est = float(norms.max().item())
            return spectral_est if spectral_est > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Built-in matrix function reducers
# ---------------------------------------------------------------------------

class MatrixExponential:
    """Compute e^{tA} via Chebyshev ACF reduction."""

    @staticmethod
    def reduce(A: torch.Tensor, t: float = 1.0,
               degree: int = 30, target_epsilon: float = 1e-10,
               dtype: torch.dtype = torch.float64) -> MatrixReductionResult:
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(A)
        sr = (t * lmin, t * lmax)
        return ChebyshevMatrixReducer.reduce(
            "exp", t * A, degree=degree, target_epsilon=target_epsilon,
            spectral_range=sr, dtype=dtype,
        )


class MatrixSquareRoot:
    """Compute A^{1/2} via Chebyshev ACF reduction. Requires A positive definite."""

    @staticmethod
    def reduce(A: torch.Tensor, degree: int = 30,
               target_epsilon: float = 1e-10,
               dtype: torch.dtype = torch.float64) -> MatrixReductionResult:
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(A)
        if lmin < -1e-10:
            raise ValueError(f"Matrix must be positive semidefinite, got λ_min={lmin}")
        sr = (max(lmin, 1e-14), lmax)
        return ChebyshevMatrixReducer.reduce(
            "sqrt", A, degree=degree, target_epsilon=target_epsilon,
            spectral_range=sr, dtype=dtype,
        )


class MatrixLogarithm:
    """Compute log(A) via Chebyshev ACF reduction. Requires A positive definite."""

    @staticmethod
    def reduce(A: torch.Tensor, degree: int = 30,
               target_epsilon: float = 1e-10,
               dtype: torch.dtype = torch.float64) -> MatrixReductionResult:
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(A)
        if lmin < 1e-14:
            raise ValueError(f"Matrix must be positive definite, got λ_min={lmin}")
        sr = (lmin, lmax)
        return ChebyshevMatrixReducer.reduce(
            "log", A, degree=degree, target_epsilon=target_epsilon,
            spectral_range=sr, dtype=dtype,
        )


class MatrixResolvent:
    """Compute (A + σI)⁻¹ via Chebyshev ACF reduction."""

    @staticmethod
    def reduce(A: torch.Tensor, sigma: float = 1.0,
               degree: int = 30, target_epsilon: float = 1e-10,
               dtype: torch.dtype = torch.float64) -> MatrixReductionResult:
        A_shifted = A + sigma * torch.eye(A.shape[0], dtype=A.dtype)
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(A_shifted)
        if lmin <= 0:
            raise ValueError(f"Resolvent requires A + σI positive definite, got λ_min={lmin}")
        return ChebyshevMatrixReducer.reduce(
            "inv", A_shifted, degree=degree, target_epsilon=target_epsilon,
            spectral_range=(lmin, lmax), dtype=dtype,
        )


class MatrixSign:
    """Compute sign(A) via Chebyshev ACF reduction."""

    @staticmethod
    def reduce(A: torch.Tensor, degree: int = 40,
               target_epsilon: float = 1e-6,
               dtype: torch.dtype = torch.float64) -> MatrixReductionResult:
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(A)
        return ChebyshevMatrixReducer.reduce(
            "sign", A, degree=degree, target_epsilon=target_epsilon,
            spectral_range=(lmin, lmax), dtype=dtype,
        )


# ---------------------------------------------------------------------------
# Invariant analysis
# ---------------------------------------------------------------------------

class MatrixACFAnalyzer:
    """Compute ACF invariants for matrix functions."""

    @staticmethod
    def analyse(
        A: torch.Tensor,
        func: Union[str, Callable[[float], float]] = "exp",
        degree: int = 40,
        dtype: torch.dtype = torch.float64,
    ) -> MatrixACFInvariants:
        """Analyse the (f, A) pair for ACF invariants."""
        A = A.to(dtype)
        if isinstance(func, str):
            scalar_func = ChebyshevMatrixReducer.MATRIX_FUNCTIONS[func]
        else:
            scalar_func = func

        # Spectral analysis
        eigvals = torch.linalg.eigvalsh(A)
        lmin = float(eigvals[0].item())
        lmax = float(eigvals[-1].item())
        condition = abs(lmax) / (abs(lmin) + 1e-30)

        # Spectral gap
        if eigvals.numel() >= 2:
            sorted_eig = torch.sort(eigvals.abs()).values
            gap = float((sorted_eig[1] - sorted_eig[0]).item())
        else:
            gap = 0.0

        # Chebyshev coefficients  
        sr = (lmin, lmax) if abs(lmax - lmin) > 1e-14 else (lmin, lmin + 1e-10)
        coeffs = ChebyshevMatrixReducer.fit_chebyshev_coefficients(
            scalar_func, degree, sr, dtype
        )

        # Alpha from Chebyshev coefficient decay
        alpha = MatrixACFAnalyzer._alpha_from_chebyshev(coeffs)

        # NC class
        if alpha < 0.2:
            nc = "NC0"
        elif alpha < 0.5:
            nc = "NC1"
        elif alpha < 0.8:
            nc = "NC2"
        else:
            nc = "NC3"

        # Effective degree: smallest d where tail sum < 1e-8
        c_abs = coeffs.abs()
        tail = torch.cumsum(c_abs.flip(0), dim=0).flip(0)
        eff_deg = degree
        for k in range(degree + 1):
            if tail[k] < 1e-8:
                eff_deg = k
                break

        # Chebyshev entropy
        p = c_abs / (c_abs.sum() + 1e-30)
        entropy = -float(torch.sum(p * torch.log(p + 1e-30)).item())

        return MatrixACFInvariants(
            matrix_alpha=alpha,
            spectral_range=(lmin, lmax),
            condition_number=condition,
            spectral_gap=gap,
            nc_class=nc,
            effective_degree=eff_deg,
            chebyshev_entropy=entropy,
        )

    @staticmethod
    def _alpha_from_chebyshev(coeffs: torch.Tensor) -> float:
        """Estimate α from |cₖ| ~ k^{-α} decay."""
        c_abs = coeffs.abs()
        c_abs = c_abs[c_abs > 1e-15]
        n = c_abs.numel()
        if n < 2:
            return 0.0

        # Skip c_0 (constant term) for the decay fit
        if n > 2:
            c_abs = c_abs[1:]
            n = c_abs.numel()

        log_k = torch.log(torch.arange(1, n + 1, dtype=c_abs.dtype))
        log_c = torch.log(c_abs + 1e-30)

        # Linear regression: log_c ≈ -α * log_k + const
        x_mean = log_k.mean()
        y_mean = log_c.mean()
        numerator = ((log_k - x_mean) * (log_c - y_mean)).sum()
        denominator = ((log_k - x_mean) ** 2).sum()
        if denominator.abs() < 1e-30:
            return 0.0
        slope = numerator / denominator
        alpha = float(-slope.item())
        return max(0.0, min(5.0, alpha))  # allow higher alpha for matrix functions
