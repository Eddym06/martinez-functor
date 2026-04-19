"""
Tests for Operator/Green Function ACF (acf_functor/operator_acf.py)
===================================================================
Running: pytest tests/test_operator_acf.py -v
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def green_heat(x, y, t=0.1):
    """Heat equation Green function G(x,y) = exp(-(x-y)²/4t) / sqrt(4πt)."""
    return np.exp(-((x - y) ** 2) / (4 * t)) / np.sqrt(4 * np.pi * t)


def simple_kernel(x, y):
    """G(x,y) = cos(π(x-y)/2)."""
    return float(np.cos(np.pi * (x - y) / 2))


def identity_kernel(x, y):
    """G(x,y) = δ(x-y) approximation: 1 if |x-y| < 0.1 else 0."""
    return 1.0 if abs(x - y) < 0.1 else 0.0


# ---------------------------------------------------------------------------
# GreenFunctionReducer
# ---------------------------------------------------------------------------

class TestGreenFunctionReducer:

    def test_import(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        assert GreenFunctionReducer is not None

    def test_construct(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer(n_points=32, order=8, domain=(-1.0, 1.0))
        assert r.n_points == 32
        assert not r._fitted

    def test_fit_heat_kernel(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer(n_points=32, order=8)
        r.fit(simple_kernel)
        assert r._fitted

    def test_call_symmetric(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer(n_points=32, order=8)
        r.fit(simple_kernel)
        # cos(π(x-y)/2) is symmetric
        val_xy = r(0.3, 0.5)
        val_yx = r(0.5, 0.3)
        assert abs(val_xy - val_yx) < 0.5  # reduced symmetry is approx.

    def test_call_shape_float(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer(n_points=16, order=6)
        r.fit(simple_kernel)
        val = r(0.0, 0.0)
        assert isinstance(val, float)

    def test_apply_returns_array(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer(n_points=16, order=6)
        r.fit(simple_kernel)
        u = np.ones(16)
        Lu = r.apply(u)
        assert Lu.shape == (16,)

    def test_invariants(self):
        from acf_functor.operator_acf import GreenFunctionReducer, KernelACFInvariants
        r = GreenFunctionReducer(n_points=16, order=6)
        r.fit(simple_kernel)
        inv = r.invariants()
        assert isinstance(inv, KernelACFInvariants)
        assert inv.alpha_kernel >= 0
        assert inv.compression_ratio > 0

    def test_invariants_with_original(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer(n_points=16, order=8)
        r.fit(simple_kernel)
        inv = r.invariants(G=simple_kernel)
        assert inv.operator_norm_error >= 0

    def test_reduce_green_function_factory(self):
        from acf_functor.operator_acf import reduce_green_function
        r = reduce_green_function(simple_kernel, n_points=16, order=6)
        assert r._fitted

    def test_before_fit_raises(self):
        from acf_functor.operator_acf import GreenFunctionReducer
        r = GreenFunctionReducer()
        with pytest.raises(RuntimeError):
            r(0.0, 0.0)


# ---------------------------------------------------------------------------
# IntegralOperatorACF
# ---------------------------------------------------------------------------

class TestIntegralOperatorACF:

    def test_import(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        assert IntegralOperatorACF is not None

    def test_fit(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF(n_points=32, rank=8)
        op.fit(simple_kernel)
        assert op._fitted

    def test_apply_zero_gives_zero(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF(n_points=32, rank=8)
        op.fit(simple_kernel)
        u_zero = np.zeros(32)
        Lu = op.apply(u_zero)
        np.testing.assert_allclose(Lu, np.zeros(32), atol=1e-10)

    def test_apply_shape(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF(n_points=32, rank=8)
        op.fit(simple_kernel)
        u = np.random.default_rng(0).normal(size=32)
        Lu = op.apply(u)
        assert Lu.shape == (32,)

    def test_alpha_positive(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF(n_points=32, rank=8)
        op.fit(simple_kernel)
        assert op.alpha() >= 0

    def test_invariants(self):
        from acf_functor.operator_acf import IntegralOperatorACF, KernelACFInvariants
        op = IntegralOperatorACF(n_points=32, rank=8)
        op.fit(simple_kernel)
        inv = op.invariants()
        assert isinstance(inv, KernelACFInvariants)
        assert inv.n_separable_terms <= 8

    def test_invariants_with_error(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF(n_points=32, rank=8)
        op.fit(simple_kernel)
        u_test = np.ones(32)
        inv = op.invariants(G=simple_kernel, test_u=u_test)
        assert inv.operator_norm_error >= 0

    def test_before_fit_raises(self):
        from acf_functor.operator_acf import IntegralOperatorACF
        op = IntegralOperatorACF()
        with pytest.raises(RuntimeError):
            op.apply(np.ones(64))


# ---------------------------------------------------------------------------
# AttentionKernelReducer
# ---------------------------------------------------------------------------

class TestAttentionKernelReducer:

    @pytest.fixture
    def qkv(self):
        rng = np.random.default_rng(7)
        n, d = 16, 8
        Q = rng.normal(size=(n, d)) * 0.5
        K = rng.normal(size=(n, d)) * 0.5
        V = rng.normal(size=(n, d)) * 0.5
        return Q, K, V

    def test_import(self):
        from acf_functor.operator_acf import AttentionKernelReducer
        assert AttentionKernelReducer is not None

    def test_fit_random_fourier(self, qkv):
        from acf_functor.operator_acf import AttentionKernelReducer
        Q, K, V = qkv
        reducer = AttentionKernelReducer(embed_dim=8, n_features=16, feature_type="random_fourier")
        reducer.fit()
        assert reducer._fitted

    def test_phi_shape(self, qkv):
        from acf_functor.operator_acf import AttentionKernelReducer
        Q, K, V = qkv
        reducer = AttentionKernelReducer(embed_dim=8, n_features=16)
        reducer.fit()
        phi = reducer.phi(Q)
        assert phi.shape == (16, 16)

    def test_fast_attention_shape(self, qkv):
        from acf_functor.operator_acf import AttentionKernelReducer
        Q, K, V = qkv
        reducer = AttentionKernelReducer(embed_dim=8, n_features=16)
        reducer.fit()
        out = reducer.fast_attention(Q, K, V)
        assert out.shape == (16, 8)

    def test_attention_relu(self, qkv):
        from acf_functor.operator_acf import AttentionKernelReducer
        Q, K, V = qkv
        reducer = AttentionKernelReducer(embed_dim=8, n_features=16, feature_type="relu")
        reducer.fit()
        out = reducer.fast_attention(Q, K, V)
        assert out.shape == (16, 8)

    def test_invariants(self, qkv):
        from acf_functor.operator_acf import AttentionKernelReducer, AttentionKernelInvariants
        Q, K, V = qkv
        reducer = AttentionKernelReducer(embed_dim=8, n_features=16)
        reducer.fit()
        inv = reducer.invariants(Q, K)
        assert isinstance(inv, AttentionKernelInvariants)
        assert inv.alpha_attention >= 0
        assert inv.speedup_factor >= 1.0
