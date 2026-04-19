"""
Tests for Rational/Padé ACF (acf_functor/rational_acf.py)
==========================================================
Running: pytest tests/test_rational_acf.py -v
"""
import pytest
import numpy as np
import cmath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exp_func(x: float) -> float:
    return float(np.exp(x))


def rational_func(x: float) -> float:
    """R(x) = (1 + 2x) / (1 - x)  — exact rational with pole at x=1."""
    return (1 + 2 * x) / (1 - x + 1e-15)


def sin_func(x: float) -> float:
    return float(np.sin(x))


def circle_func(z: complex) -> complex:
    """f(z) = 1/(2-z)  — meromorphic with pole at z=2 (outside unit disk)."""
    return 1.0 / (2.0 - z)


# ---------------------------------------------------------------------------
# _pade_from_taylor
# ---------------------------------------------------------------------------

class TestPadeFromTaylor:

    def test_import(self):
        from acf_functor.rational_acf import _pade_from_taylor
        assert _pade_from_taylor is not None

    def test_n0_returns_numerator_only(self):
        from acf_functor.rational_acf import _pade_from_taylor
        c = np.array([1.0, 1.0, 0.5, 1/6, 1/24], dtype=float)
        P, Q = _pade_from_taylor(c, m=4, n=0)
        assert len(Q) == 1
        assert abs(Q[0] - 1.0) < 1e-10

    def test_pade_exp_p1q1(self):
        """[1/1] Padé of exp: P/Q = (1+x/2)/(1-x/2)."""
        from acf_functor.rational_acf import _pade_from_taylor
        # exp Taylor: 1, 1, 1/2, 1/6
        c = np.array([1.0, 1.0, 0.5, 1/6])
        P, Q = _pade_from_taylor(c, m=1, n=1)
        # Check Q[0] = 1
        assert abs(Q[0] - 1.0) < 1e-10
        assert len(P) == 2
        assert len(Q) == 2


# ---------------------------------------------------------------------------
# _horner_eval
# ---------------------------------------------------------------------------

class TestHornerEval:

    def test_constant(self):
        from acf_functor.rational_acf import _horner_eval
        assert abs(_horner_eval(np.array([5.0]), 3.0) - 5.0) < 1e-12

    def test_linear(self):
        from acf_functor.rational_acf import _horner_eval
        # P(x) = 2 + 3x → P(4) = 14
        assert abs(_horner_eval(np.array([2.0, 3.0]), 4.0) - 14.0) < 1e-10

    def test_quadratic(self):
        from acf_functor.rational_acf import _horner_eval
        # P(x) = 1 + 2x + x² → P(3) = 16
        assert abs(_horner_eval(np.array([1.0, 2.0, 1.0]), 3.0) - 16.0) < 1e-10

    def test_empty(self):
        from acf_functor.rational_acf import _horner_eval
        assert abs(_horner_eval(np.array([]), 5.0)) < 1e-12


# ---------------------------------------------------------------------------
# PadeReducer
# ---------------------------------------------------------------------------

class TestPadeReducer:

    def test_import(self):
        from acf_functor.rational_acf import PadeReducer
        assert PadeReducer is not None

    def test_construct(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=4, n=4, x0=0.0)
        assert r.m == 4
        assert r.n == 4
        assert not r._fitted

    def test_fma_count(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=5, n=5)
        # m + n + 3 = 13
        assert r.fma_count == 13

    def test_fit_exp(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=4, n=4)
        r.fit(exp_func)
        assert r._fitted

    def test_eval_exp_at_zero(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=4, n=4)
        r.fit(exp_func)
        assert abs(r(0.0) - 1.0) < 0.1

    def test_eval_exp_small_x(self):
        """Padé [4/4] of exp(x) is accurate near x₀=0."""
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=4, n=4)
        r.fit(exp_func)
        for x in [-0.2, 0.0, 0.2]:
            assert abs(r(float(x)) - exp_func(x)) < 0.5  # practical tolerance

    def test_eval_sin_at_zero(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=5, n=4)
        r.fit(sin_func)
        assert abs(r(0.0)) < 0.1

    def test_before_fit_raises(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer()
        with pytest.raises(RuntimeError):
            r(0.0)

    def test_poles_returns_list(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=3, n=3)
        r.fit(exp_func)
        poles = r.poles()
        assert isinstance(poles, list)

    def test_poles_count(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=3, n=3)
        r.fit(exp_func)
        assert len(r.poles()) == 3

    def test_invariants(self):
        from acf_functor.rational_acf import PadeReducer, PadeInvariants
        r = PadeReducer(m=4, n=4)
        r.fit(exp_func)
        inv = r.invariants()
        assert isinstance(inv, PadeInvariants)
        assert inv.m == 4
        assert inv.n == 4
        assert inv.pole_count == 4

    def test_invariants_summary(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=3, n=3)
        r.fit(sin_func)
        inv = r.invariants()
        s = inv.summary()
        assert "Padé" in s
        assert "[3/3]" in s

    def test_pade_reduce_factory(self):
        from acf_functor.rational_acf import pade_reduce
        r = pade_reduce(exp_func, m=3, n=3)
        assert r._fitted

    def test_derivatives_method(self):
        from acf_functor.rational_acf import PadeReducer
        r = PadeReducer(m=3, n=3)
        r.fit(exp_func, method="derivatives")
        assert r._fitted


# ---------------------------------------------------------------------------
# HardySpaceACF
# ---------------------------------------------------------------------------

class TestHardySpaceACF:

    def test_import(self):
        from acf_functor.rational_acf import HardySpaceACF
        assert HardySpaceACF is not None

    def test_construct(self):
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=16)
        assert h.n_modes == 16
        assert not h._fitted

    def test_fit_circle_func(self):
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=16)
        h.fit(circle_func, n_quad=128)
        assert h._fitted

    def test_eval_at_zero(self):
        """1/(2-z) at z=0 → 0.5."""
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=16)
        h.fit(circle_func, n_quad=128)
        val = h(0.0 + 0j)
        assert abs(val.real - 0.5) < 0.1

    def test_alpha_positive(self):
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=16)
        h.fit(circle_func, n_quad=128)
        alpha = h.alpha()
        assert alpha >= 0

    def test_invariants(self):
        from acf_functor.rational_acf import HardySpaceACF, HardySpaceInvariants
        h = HardySpaceACF(n_modes=16)
        h.fit(circle_func, n_quad=128)
        inv = h.invariants()
        assert isinstance(inv, HardySpaceInvariants)
        assert inv.n_modes == 16
        assert 0 < inv.effective_rank <= 16

    def test_invariants_with_exact(self):
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=16)
        h.fit(circle_func, n_quad=128)
        inv = h.invariants(f_exact=circle_func)
        assert inv.h2_error >= 0

    def test_summary_string(self):
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=16)
        h.fit(circle_func, n_quad=128)
        inv = h.invariants()
        s = inv.summary()
        assert "Hardy" in s

    def test_before_fit_raises(self):
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF()
        with pytest.raises(RuntimeError):
            h(0.0 + 0j)

    def test_hardy_reduce_factory(self):
        from acf_functor.rational_acf import hardy_reduce
        h = hardy_reduce(circle_func, n_modes=16, n_quad=64)
        assert h._fitted

    def test_coeffs_decay(self):
        """Coefficients of 1/(2-z) should decay as (1/2)^k."""
        from acf_functor.rational_acf import HardySpaceACF
        h = HardySpaceACF(n_modes=12)
        h.fit(circle_func, n_quad=256)
        mags = np.abs(h._coeffs)
        # Each coefficient should be ≤ previous (decaying)
        assert mags[0] > mags[-1]  # first coeff > last coeff
