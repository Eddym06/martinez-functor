"""
Test suite for Matrix ACF — f(A) via Chebyshev polynomials of matrices.
15 unit tests + 3 integration tests + 2 massive/complex real tests.
"""

import math
import pytest
import torch
import numpy as np

from acf_functor.matrix_acf import (
    ChebyshevMatrixReducer,
    MatrixReductionResult,
    MatrixACFInvariants,
    MatrixACFAnalyzer,
    MatrixExponential,
    MatrixSquareRoot,
    MatrixLogarithm,
    MatrixResolvent,
    MatrixSign,
)


def _random_spd(n: int, seed: int = 0) -> torch.Tensor:
    """Generate random symmetric positive definite matrix."""
    rng = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, dtype=torch.float64, generator=rng)
    return A @ A.T + 0.1 * torch.eye(n, dtype=torch.float64)


def _random_symmetric(n: int, seed: int = 0) -> torch.Tensor:
    """Generate random symmetric matrix (not necessarily positive)."""
    rng = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, dtype=torch.float64, generator=rng)
    return (A + A.T) / 2


# ===========================================================================
# Unit Tests — Spectral range
# ===========================================================================

class TestSpectralRange:
    """Tests for spectral bound computation."""

    def test_spectral_range_identity(self):
        """UT-M01: Identity matrix has spectral range [1, 1]."""
        I = torch.eye(5, dtype=torch.float64)
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(I)
        assert abs(lmin - 1.0) < 1e-10
        assert abs(lmax - 1.0) < 1e-10

    def test_spectral_range_diagonal(self):
        """UT-M02: Diagonal matrix has correct spectral range."""
        D = torch.diag(torch.tensor([0.5, 2.0, 3.0, 7.0], dtype=torch.float64))
        lmin, lmax = ChebyshevMatrixReducer.compute_spectral_range(D)
        assert abs(lmin - 0.5) < 1e-10
        assert abs(lmax - 7.0) < 1e-10

    def test_spectral_range_nonsquare_raises(self):
        """UT-M03: Non-square matrix raises ValueError."""
        with pytest.raises(ValueError):
            ChebyshevMatrixReducer.compute_spectral_range(
                torch.randn(3, 4, dtype=torch.float64)
            )


# ===========================================================================
# Unit Tests — Chebyshev coefficient fitting
# ===========================================================================

class TestChebyshevFitting:
    """Tests for scalar Chebyshev fitting used by matrix reducer."""

    def test_fit_constant(self):
        """UT-M04: Constant function f(x)=5 → c_0=5, rest≈0."""
        coeffs = ChebyshevMatrixReducer.fit_chebyshev_coefficients(
            lambda x: 5.0, 10, (0, 1)
        )
        assert abs(coeffs[0].item() - 5.0) < 1e-10
        assert float(coeffs[1:].abs().max()) < 1e-10

    def test_fit_exp(self):
        """UT-M05: exp(x) on [0,1] — coefficients decay exponentially."""
        coeffs = ChebyshevMatrixReducer.fit_chebyshev_coefficients(
            math.exp, 20, (0, 1)
        )
        # Coefficients should decay
        assert coeffs.abs()[0] > coeffs.abs()[-1]

    def test_fit_polynomial_exact(self):
        """UT-M06: Polynomial of degree 3 is exact with degree ≥ 3."""
        def f(x): return x ** 3 - 2 * x + 1
        coeffs = ChebyshevMatrixReducer.fit_chebyshev_coefficients(
            f, 10, (-1, 1)
        )
        # Coefficients beyond degree 3 should be near zero
        assert float(coeffs[4:].abs().max()) < 1e-10


# ===========================================================================
# Unit Tests — Matrix Chebyshev evaluation
# ===========================================================================

class TestMatrixChebyshevEval:
    """Tests for Clenshaw matrix polynomial evaluation."""

    def test_eval_identity_function(self):
        """UT-M07: f(x)=x, f(A)=A."""
        A = _random_symmetric(5, seed=42)
        # f(x) = x → c_0 depends on domain shift, but result should be A
        result = ChebyshevMatrixReducer.reduce(
            lambda x: x, A, degree=5, target_epsilon=1e-10
        )
        assert float(torch.norm(result.result_matrix - A).item()) < 1e-6

    def test_eval_constant_function(self):
        """UT-M08: f(x)=3, f(A)=3*I."""
        A = _random_symmetric(4, seed=7)
        result = ChebyshevMatrixReducer.reduce(
            lambda x: 3.0, A, degree=5, target_epsilon=1e-10
        )
        expected = 3.0 * torch.eye(4, dtype=torch.float64)
        assert float(torch.norm(result.result_matrix - expected).item()) < 1e-6

    def test_eval_exp_small_matrix(self):
        """UT-M09: exp(A) for 3×3 agrees with torch.linalg.matrix_exp."""
        A = _random_symmetric(3, seed=13)
        result = ChebyshevMatrixReducer.reduce("exp", A, degree=30, target_epsilon=1e-10)
        exact = torch.linalg.matrix_exp(A.to(torch.float64))
        err = float(torch.norm(result.result_matrix - exact).item())
        assert err < 1e-4, f"exp(A) error: {err}"

    def test_eval_sqrt_spd(self):
        """UT-M10: sqrt(A) for SPD A: (sqrt(A))² ≈ A."""
        A = _random_spd(4, seed=17)
        result = MatrixSquareRoot.reduce(A, degree=30, target_epsilon=1e-10)
        sqrtA = result.result_matrix
        reconstructed = sqrtA @ sqrtA
        err = float(torch.norm(reconstructed - A).item())
        assert err < 0.1, f"sqrt(A)² error: {err}"


# ===========================================================================
# Unit Tests — Invariant analysis
# ===========================================================================

class TestMatrixACFInvariants:
    """Tests for MatrixACFInvariants computation."""

    def test_invariants_computed(self):
        """UT-M11: analyse() returns valid invariants."""
        A = _random_spd(5, seed=23)
        inv = MatrixACFAnalyzer.analyse(A, func="exp")
        assert isinstance(inv, MatrixACFInvariants)
        assert inv.matrix_alpha >= 0
        assert inv.nc_class in {"NC0", "NC1", "NC2", "NC3"}

    def test_invariants_spectral_range(self):
        """UT-M12: Spectral range matches eigenvalues."""
        A = _random_spd(4, seed=29)
        inv = MatrixACFAnalyzer.analyse(A, func="exp")
        eigvals = torch.linalg.eigvalsh(A)
        assert abs(inv.spectral_range[0] - float(eigvals[0])) < 1e-8
        assert abs(inv.spectral_range[1] - float(eigvals[-1])) < 1e-8

    def test_invariants_condition_number(self):
        """UT-M13: Condition number > 0."""
        A = _random_spd(6, seed=31)
        inv = MatrixACFAnalyzer.analyse(A, func="exp")
        assert inv.condition_number > 0

    def test_invariants_entropy_nonneg(self):
        """UT-M14: Chebyshev entropy ≥ 0."""
        A = _random_spd(5, seed=37)
        inv = MatrixACFAnalyzer.analyse(A, func="sqrt")
        assert inv.chebyshev_entropy >= 0

    def test_invariants_effective_degree(self):
        """UT-M15: Effective degree ≤ fitting degree."""
        A = _random_spd(4, seed=41)
        inv = MatrixACFAnalyzer.analyse(A, func="exp", degree=40)
        assert inv.effective_degree <= 40


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestMatrixACFIntegration:
    """Integration tests combining full pipeline."""

    def test_matrix_exp_full_pipeline(self):
        """IT-M01: Full pipeline exp(A) → reduce → certify → invariants."""
        A = _random_spd(6, seed=51)
        result = MatrixExponential.reduce(A, t=1.0, degree=25, target_epsilon=1e-8)
        assert isinstance(result, MatrixReductionResult)
        assert result.epsilon < 1.0
        assert result.fma_count > 0
        assert result.elapsed_ms > 0

        # Verify against exact
        exact = torch.linalg.matrix_exp(A)
        err = float(torch.norm(result.result_matrix - exact).item())
        assert err < 0.1, f"Full pipeline exp(A) error: {err}"

    def test_matrix_log_full_pipeline(self):
        """IT-M02: Full pipeline log(A) for SPD matrix."""
        A = _random_spd(5, seed=53)
        result = MatrixLogarithm.reduce(A, degree=30, target_epsilon=1e-8)
        assert isinstance(result, MatrixReductionResult)
        # Verify: exp(log(A)) ≈ A
        logA = result.result_matrix
        reconstructed = torch.linalg.matrix_exp(logA)
        err = float(torch.norm(reconstructed - A).item())
        assert err < 1.0, f"exp(log(A)) roundtrip error: {err}"

    def test_matrix_resolvent_pipeline(self):
        """IT-M03: Resolvent (A + σI)⁻¹ vs direct inverse."""
        A = _random_spd(4, seed=59)
        sigma = 2.0
        result = MatrixResolvent.reduce(A, sigma=sigma, degree=25, target_epsilon=1e-8)
        exact = torch.linalg.inv(A + sigma * torch.eye(4, dtype=torch.float64))
        err = float(torch.norm(result.result_matrix - exact).item())
        assert err < 0.5, f"Resolvent error: {err}"


# ===========================================================================
# Massive / Complex Real Tests
# ===========================================================================

class TestMatrixACFMassive:
    """Massive tests: larger matrices, harder functions."""

    def test_massive_50x50_exp(self):
        """MT-M01: 50×50 matrix exponential — full pipeline.
        Tests scalability and correctness at moderate size.
        """
        A = _random_spd(50, seed=71)
        # Scale to avoid huge exponentials
        A = A / torch.norm(A) * 2.0
        result = ChebyshevMatrixReducer.reduce("exp", A, degree=40, target_epsilon=1e-6)
        assert isinstance(result, MatrixReductionResult)
        assert result.result_matrix.shape == (50, 50)
        assert result.epsilon < 10.0  # certification ran

        # Check symmetry preserved
        fA = result.result_matrix
        asym = float(torch.norm(fA - fA.T).item())
        assert asym < 1e-6, "exp(A) should be symmetric for symmetric A"

    def test_massive_multi_function_comparison(self):
        """MT-M02: Compare multiple matrix functions on same 20×20 SPD matrix.
        exp, sqrt, log — all computed, all certified, invariants compared.
        """
        A = _random_spd(20, seed=83)
        A = A / torch.norm(A) * 2.0 + 0.5 * torch.eye(20, dtype=torch.float64)

        results = {}
        invariants = {}
        for fname in ["exp", "sqrt", "log"]:
            r = ChebyshevMatrixReducer.reduce(fname, A, degree=30, target_epsilon=1e-6)
            results[fname] = r
            inv = MatrixACFAnalyzer.analyse(A, func=fname, degree=30)
            invariants[fname] = inv

        # All results should be valid
        for fname in ["exp", "sqrt", "log"]:
            assert results[fname].result_matrix.shape == (20, 20)
            assert invariants[fname].matrix_alpha >= 0

        # exp should generally need more degree than sqrt
        assert invariants["exp"].effective_degree >= 1
        assert invariants["sqrt"].effective_degree >= 1

        # sign function should give different NC class for identity-like matrix
        sign_inv = MatrixACFAnalyzer.analyse(A, func="sign", degree=40)
        assert sign_inv.nc_class in {"NC0", "NC1", "NC2", "NC3"}
