"""
Test suite for Tensor ACF — TT and Tucker decomposition of multivariable functions.
15 unit tests + 3 integration tests + 2 massive/complex real tests.
"""

import math
import pytest
import torch
import numpy as np

from acf_functor.tensor_acf import (
    ChebyshevTensorSampler,
    TensorTrainBuilder,
    TensorTrainEvaluator,
    TensorACFReducer,
    TensorACFAnalyzer,
    TensorTrainDecomposition,
    TensorReductionResult,
    TuckerBuilder,
    TuckerEvaluator,
    TuckerReductionResult,
    StandardTensorFunctions,
)


# ===========================================================================
# Unit Tests — ChebyshevTensorSampler
# ===========================================================================

class TestChebyshevTensorSampler:
    """Tests for Chebyshev grid sampling and coefficient computation."""

    def test_chebyshev_nodes_in_domain(self):
        """UT-T01: Chebyshev nodes lie within [a, b]."""
        nodes = ChebyshevTensorSampler.chebyshev_nodes(16, (-2.0, 3.0))
        assert nodes.min() >= -2.0
        assert nodes.max() <= 3.0
        assert nodes.shape[0] == 16

    def test_chebyshev_nodes_count(self):
        """UT-T02: Correct number of nodes generated."""
        for n in [1, 5, 32, 64]:
            nodes = ChebyshevTensorSampler.chebyshev_nodes(n, (0.0, 1.0))
            assert nodes.shape[0] == n

    def test_1d_coefficients_constant(self):
        """UT-T03: Constant function → only c_0 nonzero."""
        values = torch.full((8,), 3.14, dtype=torch.float64)
        coeffs = ChebyshevTensorSampler.chebyshev_coefficients_1d(values)
        assert abs(coeffs[0].item() - 3.14) < 1e-12
        assert float(coeffs[1:].abs().max()) < 1e-12

    def test_1d_coefficients_linear(self):
        """UT-T04: Linear function T_1(x) = x → c_0=0, c_1=1."""
        n = 16
        k = torch.arange(n, dtype=torch.float64)
        nodes = torch.cos((k + 0.5) * math.pi / n)  # Chebyshev nodes on [-1,1]
        coeffs = ChebyshevTensorSampler.chebyshev_coefficients_1d(nodes)
        assert abs(coeffs[0].item()) < 1e-12
        assert abs(coeffs[1].item() - 1.0) < 1e-12

    def test_tensor_grid_shape(self):
        """UT-T05: Tensor grid produces correct shape."""
        def f(x, y): return x + y
        values = ChebyshevTensorSampler.build_tensor_grid(
            f, [4, 6], [(-1, 1), (-1, 1)]
        )
        assert values.shape == (4, 6)

    def test_tensor_to_chebyshev_roundtrip(self):
        """UT-T06: Constant function roundtrip: coeffs → eval ≈ original."""
        def f(x, y): return 5.0
        values = ChebyshevTensorSampler.build_tensor_grid(
            f, [4, 4], [(-1, 1), (-1, 1)]
        )
        C = ChebyshevTensorSampler.tensor_to_chebyshev_coeffs(values, [4, 4])
        # c_{0,0} should be 5.0, rest ≈ 0
        assert abs(C[0, 0].item() - 5.0) < 1e-10
        assert float(C[1:, :].abs().max()) < 1e-10


# ===========================================================================
# Unit Tests — TT-SVD
# ===========================================================================

class TestTTSVD:
    """Tests for Tensor Train SVD decomposition."""

    def test_rank1_tensor(self):
        """UT-T07: Rank-1 tensor → all TT-ranks = 1."""
        # f(x,y) = x*y is pure rank-1 in product basis
        T = torch.outer(torch.randn(5, dtype=torch.float64),
                        torch.randn(4, dtype=torch.float64))
        tt = TensorTrainBuilder.tt_svd(T)
        assert len(tt.cores) == 2
        # TT-ranks should be [1, 1, 1] for rank-1
        assert tt.ranks[1] == 1

    def test_full_rank_preserved(self):
        """UT-T08: Full-rank random matrix → TT captures all info."""
        T = torch.randn(3, 4, dtype=torch.float64)
        tt = TensorTrainBuilder.tt_svd(T, max_rank=50, epsilon=1e-14)
        # Reconstruct
        tt.domains = [(-1, 1), (-1, 1)]
        recon = torch.zeros(3, 4, dtype=torch.float64)
        for i in range(3):
            for j in range(4):
                v = tt.cores[0][:, i, :]  # (1, r1)
                v = v @ tt.cores[1][:, j, :]  # (1, 1)
                recon[i, j] = v.item()
        assert float(torch.norm(T - recon).item()) < 1e-10

    def test_3d_tensor_compression(self):
        """UT-T09: 3D tensor TT-SVD produces 3 cores."""
        T = torch.randn(5, 6, 4, dtype=torch.float64)
        tt = TensorTrainBuilder.tt_svd(T,  max_rank=50, epsilon=1e-14)
        assert len(tt.cores) == 3
        assert tt.ranks[0] == 1
        assert tt.ranks[-1] == 1

    def test_compression_ratio_positive(self):
        """UT-T10: Compression ratio > 0."""
        T = torch.randn(8, 8, 8, dtype=torch.float64)
        tt = TensorTrainBuilder.tt_svd(T, max_rank=3, epsilon=0.1)
        assert tt.compression_ratio > 0


# ===========================================================================
# Unit Tests — TT Evaluation
# ===========================================================================

class TestTTEvaluator:
    """Tests for TT evaluation (FMA zipper contraction)."""

    def test_evaluate_constant(self):
        """UT-T11: TT of constant function evaluates correctly."""
        reducer = TensorACFReducer(default_degree=4, max_rank=5,
                                    target_epsilon=1e-6, method="tt")
        result = reducer.reduce(
            lambda x, y: 7.0,
            [(-1, 1), (-1, 1)],
        )
        x = torch.tensor([0.5, 0.3], dtype=torch.float64)
        val = result.evaluate(x)
        assert abs(float(val.item()) - 7.0) < 0.1

    def test_evaluate_separable(self):
        """UT-T12: Separable function sin(x)*cos(y) well-approximated."""
        reducer = TensorACFReducer(default_degree=12, max_rank=10,
                                    target_epsilon=1e-6, method="tt")
        result = reducer.reduce(
            lambda x, y: math.sin(x) * math.cos(y),
            [(-3.0, 3.0), (-3.0, 3.0)],
        )
        # Test at random points
        rng = np.random.default_rng(123)
        for _ in range(10):
            x, y = rng.uniform(-3, 3, size=2)
            exact = math.sin(x) * math.cos(y)
            pt = torch.tensor([x, y], dtype=torch.float64)
            approx = float(result.evaluate(pt).item())
            assert abs(approx - exact) < 0.01, f"At ({x},{y}): {approx} vs {exact}"

    def test_batch_evaluation(self):
        """UT-T13: Batch evaluation produces correct shape."""
        reducer = TensorACFReducer(default_degree=6, max_rank=5,
                                    target_epsilon=1e-4, method="tt")
        result = reducer.reduce(
            lambda x, y: x + y,
            [(-1, 1), (-1, 1)],
        )
        batch = torch.rand(20, 2, dtype=torch.float64) * 2 - 1
        vals = result.evaluate(batch)
        assert vals.shape == (20,)


# ===========================================================================
# Unit Tests — Invariants
# ===========================================================================

class TestTensorACFAnalyzer:
    """Tests for TensorACFInvariants computation."""

    def test_invariants_alpha_range(self):
        """UT-T14: Alpha values in [0, 1]."""
        reducer = TensorACFReducer(default_degree=8, max_rank=10,
                                    target_epsilon=1e-6, method="tt")
        result = reducer.reduce(
            lambda x, y, z: math.sin(x) * math.cos(y) * math.exp(z),
            [(-2, 2), (-2, 2), (-1, 1)],
        )
        inv = result.invariants
        for a in inv.alpha_per_mode:
            assert 0.0 <= a <= 1.0, f"Alpha {a} out of range"

    def test_invariants_nc_class(self):
        """UT-T15: NC class is one of NC0-NC3."""
        reducer = TensorACFReducer(default_degree=8, max_rank=10,
                                    target_epsilon=1e-6, method="tt")
        result = reducer.reduce(
            lambda x, y: x ** 2 + y ** 2,
            [(-1, 1), (-1, 1)],
        )
        assert result.invariants.nc_class in {"NC0", "NC1", "NC2", "NC3"}


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestTensorACFIntegration:
    """Integration tests combining reducer → evaluation → certification."""

    def test_full_pipeline_2d_polynomial(self):
        """IT-T01: Full pipeline for 2D polynomial x²y + xy² + 1."""
        reducer = TensorACFReducer(default_degree=10, max_rank=15,
                                    target_epsilon=1e-8, method="tt")
        result = reducer.reduce(
            lambda x, y: x ** 2 * y + x * y ** 2 + 1,
            [(-2, 2), (-2, 2)],
        )
        assert isinstance(result, TensorReductionResult)
        assert result.epsilon < 0.1  # should converge well for polynomial

        # Spot check
        for x_val, y_val in [(0.5, 0.7), (-1.0, 1.5), (0.0, 0.0)]:
            exact = x_val ** 2 * y_val + x_val * y_val ** 2 + 1
            pt = torch.tensor([x_val, y_val], dtype=torch.float64)
            approx = float(result.evaluate(pt).item())
            assert abs(approx - exact) < 0.5, f"At ({x_val},{y_val}): {approx} vs {exact}"

    def test_full_pipeline_3d_wave(self):
        """IT-T02: 3D wave function sin(x+y)*cos(z)."""
        reducer = TensorACFReducer(default_degree=10, max_rank=15,
                                    target_epsilon=1e-6, method="tt")
        result = reducer.reduce(
            StandardTensorFunctions.wave_3d,
            [(-2, 2), (-2, 2), (-2, 2)],
        )
        assert result.tt.ndim == 3
        assert result.invariants.total_fma_count > 0
        assert result.elapsed_ms > 0

    def test_tucker_pipeline_2d(self):
        """IT-T03: Tucker decomposition on 2D Gaussian."""
        reducer = TensorACFReducer(default_degree=12, max_rank=10,
                                    target_epsilon=1e-6, method="tucker")
        result = reducer.reduce(
            StandardTensorFunctions.gaussian_2d,
            [(-3, 3), (-3, 3)],
        )
        assert isinstance(result, TuckerReductionResult)
        assert result.epsilon < 1.0  # certification worked


# ===========================================================================
# Massive / Complex Real Tests
# ===========================================================================

class TestTensorACFMassive:
    """Massive tests: real-world complexity, high dimensions."""

    def test_massive_5d_friedman(self):
        """MT-T01: 5D Friedman-1 function with degree 6 per dim.
        Total grid: 6^5 = 7776 points. TT should compress heavily.
        """
        reducer = TensorACFReducer(default_degree=6, max_rank=8,
                                    target_epsilon=1e-4, method="tt")
        result = reducer.reduce(
            StandardTensorFunctions.friedman1,
            [(0, 1)] * 5,
        )
        assert result.tt.ndim == 5
        assert result.tt.compression_ratio > 1.0  # must compress

        # Random spot checks
        rng = np.random.default_rng(999)
        errors = []
        for _ in range(50):
            pt_np = rng.uniform(0, 1, size=5)
            exact = StandardTensorFunctions.friedman1(*pt_np.tolist())
            pt = torch.tensor(pt_np, dtype=torch.float64)
            approx = float(result.evaluate(pt).item())
            errors.append(abs(approx - exact))
        mean_err = np.mean(errors)
        assert mean_err < 5.0, f"Mean error {mean_err} too high for 5D Friedman"

    def test_massive_rosenbrock_high_degree(self):
        """MT-T02: Rosenbrock (strongly non-separable) with high degree.
        Tests TT ability to handle coupled nonlinear interactions.
        """
        reducer = TensorACFReducer(default_degree=16, max_rank=20,
                                    target_epsilon=1e-6, method="tt")
        result = reducer.reduce(
            StandardTensorFunctions.rosenbrock,
            [(-2, 2), (-2, 2)],
        )
        assert result.tt.ndim == 2

        # Test at the minimum (1, 1) where f = 0
        pt = torch.tensor([1.0, 1.0], dtype=torch.float64)
        val = float(result.evaluate(pt).item())
        assert abs(val) < 5.0, f"Rosenbrock at minimum: {val} (expected ~0)"

        # Test at origin (1 + 100 = 101)
        pt0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        val0 = float(result.evaluate(pt0).item())
        assert abs(val0 - 1.0) < 50.0, f"Rosenbrock at origin: {val0} (expected 1)"

        # Invariants should show non-trivial structure
        assert result.invariants.alpha_global >= 0.0
        assert result.invariants.total_fma_count > 0
