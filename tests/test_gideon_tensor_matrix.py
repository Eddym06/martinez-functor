"""
Integration tests for GideonEngine Tensor ACF and Matrix ACF methods.

Validates that the engine correctly delegates to tensor_acf / matrix_acf
subsystems and returns the expected result types.
"""

import math
import pytest
import torch
import numpy as np

# ── Attempt Gideon import; skip all if rust bridge unavailable ──────────────
try:
    from poema.backends.gideon.engine import GideonEngine
    GIDEON_AVAILABLE = True
except Exception:
    GIDEON_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GIDEON_AVAILABLE, reason="GideonEngine not importable (rust bridge issue)"
)


@pytest.fixture(scope="module")
def engine():
    return GideonEngine()


# ===========================================================================
# Tensor ACF engine methods
# ===========================================================================


class TestGideonReduceTensor:
    """Tests for engine.reduce_tensor()."""

    def test_reduce_tensor_2d_separable(self, engine):
        """TT decomposition of a 2D separable product f(x,y)=x*y."""
        func = lambda x, y: x * y
        result = engine.reduce_tensor(func, [(-1, 1), (-1, 1)], default_degree=8, max_rank=5)
        from acf_functor.tensor_acf import TensorReductionResult
        assert isinstance(result, TensorReductionResult)
        assert result.epsilon < 1e-4
        assert result.tt is not None
        assert result.invariants is not None

    def test_reduce_tensor_tucker(self, engine):
        """Tucker decomposition path."""
        func = lambda x, y: x**2 + y**2
        from acf_functor.tensor_acf import TuckerReductionResult
        result = engine.reduce_tensor(func, [(-1, 1), (-1, 1)], method="tucker", default_degree=6)
        assert isinstance(result, TuckerReductionResult)
        assert result.tucker is not None

    def test_reduce_tensor_3d_rosenbrock(self, engine):
        """3D function → TT with ≥3 cores."""
        func = lambda x, y, z: (1 - x)**2 + 100*(y - x**2)**2 + z**2
        result = engine.reduce_tensor(
            func, [(-2, 2), (-2, 2), (-2, 2)],
            default_degree=8, max_rank=15,
        )
        assert len(result.tt.cores) == 3


class TestGideonAnalyseTensor:
    """Tests for engine.analyse_tensor()."""

    def test_analyse_tensor_invariants(self, engine):
        """Invariants returned with alpha and NC class."""
        func = lambda x, y: math.sin(x) * math.cos(y)
        from acf_functor.tensor_acf import TensorACFInvariants
        inv = engine.analyse_tensor(func, [(-3.14, 3.14), (-3.14, 3.14)], default_degree=10)
        assert isinstance(inv, TensorACFInvariants)
        assert inv.alpha_global >= 0.0
        assert inv.nc_class in ("NC0", "NC₁", "NC₂", "NC₃")
        assert inv.effective_dimension == 2

    def test_analyse_tensor_alpha_per_mode(self, engine):
        """Alpha values per mode are non-negative."""
        func = lambda x, y, z: x + y + z
        inv = engine.analyse_tensor(func, [(0, 1), (0, 1), (0, 1)], default_degree=6)
        assert len(inv.alpha_per_mode) == 3
        assert all(a >= 0.0 for a in inv.alpha_per_mode)


# ===========================================================================
# Matrix ACF engine methods
# ===========================================================================


class TestGideonReduceMatrixFunction:
    """Tests for engine.reduce_matrix_function()."""

    def test_reduce_matrix_exp(self, engine):
        """Matrix exponential via Chebyshev."""
        A = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        from acf_functor.matrix_acf import MatrixReductionResult
        result = engine.reduce_matrix_function("exp", A, degree=20)
        assert isinstance(result, MatrixReductionResult)
        expected_diag = torch.exp(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        err = torch.norm(torch.diag(result.result_matrix) - expected_diag).item()
        assert err < 1e-6

    def test_reduce_matrix_sqrt(self, engine):
        """Matrix sqrt of SPD matrix."""
        A = torch.diag(torch.tensor([4.0, 9.0, 16.0], dtype=torch.float64))
        result = engine.reduce_matrix_function("sqrt", A, degree=20)
        expected = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
        assert torch.allclose(result.result_matrix, expected, atol=1e-6)

    def test_reduce_matrix_custom_func(self, engine):
        """Custom scalar function applied to matrix."""
        A = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        result = engine.reduce_matrix_function(lambda x: x**2 + 1, A, degree=15)
        expected = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.float64))
        assert torch.allclose(result.result_matrix, expected, atol=1e-4)


class TestGideonAnalyseMatrix:
    """Tests for engine.analyse_matrix()."""

    def test_analyse_matrix_invariants(self, engine):
        """Matrix ACF invariants returned correctly."""
        A = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        from acf_functor.matrix_acf import MatrixACFInvariants
        inv = engine.analyse_matrix(A, func="exp", degree=20)
        assert isinstance(inv, MatrixACFInvariants)
        assert inv.matrix_alpha >= 0.0
        assert inv.nc_class in ("NC0", "NC1", "NC2", "NC3")
        assert inv.spectral_range[0] <= inv.spectral_range[1]


class TestGideonMatrixConvenience:
    """Tests for convenience wrappers: matrix_exp, matrix_sqrt, etc."""

    def test_matrix_exp(self, engine):
        A = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
        result = engine.matrix_exp(A)
        expected = torch.diag(torch.exp(torch.tensor([0.0, 1.0], dtype=torch.float64)))
        assert torch.allclose(result.result_matrix, expected, atol=1e-8)

    def test_matrix_sqrt(self, engine):
        A = torch.diag(torch.tensor([4.0, 25.0], dtype=torch.float64))
        result = engine.matrix_sqrt(A)
        expected = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.float64))
        assert torch.allclose(result.result_matrix, expected, atol=1e-6)

    def test_matrix_log(self, engine):
        A = torch.diag(torch.tensor([1.0, math.e], dtype=torch.float64))
        result = engine.matrix_log(A)
        expected = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
        assert torch.allclose(result.result_matrix, expected, atol=1e-6)

    def test_matrix_resolvent(self, engine):
        A = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        result = engine.matrix_resolvent(A, sigma=1.0)
        # (A + I)^{-1} = diag(1/2, 1/3)
        expected = torch.diag(torch.tensor([0.5, 1.0 / 3.0], dtype=torch.float64))
        assert torch.allclose(result.result_matrix, expected, atol=1e-6)
