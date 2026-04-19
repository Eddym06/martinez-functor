"""
Tests for ODE/Control ACF (acf_functor/ode_acf.py)
===================================================
Running: pytest tests/test_ode_acf.py -v
"""
import pytest
import numpy as np
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def harmonic_field(x: np.ndarray) -> np.ndarray:
    """ẋ₁ = x₂, ẋ₂ = -x₁  (harmonic oscillator, stable)."""
    return np.array([x[1], -x[0]])


def damped_field(x: np.ndarray) -> np.ndarray:
    """ẋ₁ = -x₁, ẋ₂ = -x₂  (exponential decay, asymptotically stable)."""
    return np.array([-x[0], -x[1]])


def quadratic_lyapunov(x: np.ndarray) -> float:
    """V(x) = x₁² + x₂²  (quadratic Lyapunov candidate)."""
    return float(x[0] ** 2 + x[1] ** 2)


# ---------------------------------------------------------------------------
# VectorFieldReducer
# ---------------------------------------------------------------------------

class TestVectorFieldReducer:

    def test_import(self):
        from acf_functor.ode_acf import VectorFieldReducer
        assert VectorFieldReducer is not None

    def test_construct(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=6)
        assert r.n == 2
        assert r.order == 6
        assert not r._fitted

    def test_fit_damped(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=6)
        r.fit(damped_field, n_samples=100)
        assert r._fitted

    def test_call_returns_vector(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        x = np.array([0.3, -0.2])
        out = r(x)
        assert out.shape == (2,)

    def test_call_before_fit_raises(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2)
        with pytest.raises(RuntimeError):
            r(np.array([0.0, 0.0]))

    def test_domain_custom(self):
        from acf_functor.ode_acf import VectorFieldReducer
        domain = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        r = VectorFieldReducer(dimension=2, domain=domain)
        r.fit(damped_field, n_samples=60)
        assert r._fitted

    def test_invariants_returns_ode_invariants(self):
        from acf_functor.ode_acf import VectorFieldReducer, ODEACFInvariants
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        inv = r.invariants(damped_field, T=1.0)
        assert isinstance(inv, ODEACFInvariants)

    def test_invariants_n_components(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        inv = r.invariants(damped_field, T=1.0)
        assert inv.n_components == 2
        assert len(inv.alpha_per_component) == 2

    def test_invariants_alphas_positive(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        inv = r.invariants(damped_field, T=1.0)
        for a in inv.alpha_per_component:
            assert a >= 0

    def test_invariants_gronwall_nonneg(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        inv = r.invariants(damped_field, T=2.0, eps_target=1e-4)
        assert inv.gronwall_bound >= 0

    def test_stability_damped(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        inv = r.invariants(damped_field, T=1.0)
        assert inv.stability_certificate == "stable"

    def test_summary_string(self):
        from acf_functor.ode_acf import VectorFieldReducer
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        inv = r.invariants(damped_field, T=1.0)
        s = inv.summary()
        assert "ODE-ACF" in s
        assert "α_min" in s

    def test_reduce_vector_field_factory(self):
        from acf_functor.ode_acf import reduce_vector_field
        r = reduce_vector_field(damped_field, dimension=2, order=4, n_samples=60)
        assert r._fitted


# ---------------------------------------------------------------------------
# HJBReducer
# ---------------------------------------------------------------------------

class TestHJBReducer:

    def test_import(self):
        from acf_functor.ode_acf import HJBReducer
        assert HJBReducer is not None

    def test_fit_quadratic(self):
        from acf_functor.ode_acf import HJBReducer
        hjb = HJBReducer(dimension=2, order=6)
        hjb.fit(quadratic_lyapunov)
        assert hjb._fitted

    def test_value_at_origin_near_zero(self):
        from acf_functor.ode_acf import HJBReducer
        hjb = HJBReducer(dimension=2, order=6)
        hjb.fit(quadratic_lyapunov)
        val = hjb.value(np.zeros(2))
        assert abs(val) < 0.1  # V(0) ≈ 0

    def test_gradient_shape(self):
        from acf_functor.ode_acf import HJBReducer
        hjb = HJBReducer(dimension=2, order=6)
        hjb.fit(quadratic_lyapunov)
        grad = hjb.gradient(np.array([0.5, 0.3]))
        assert grad.shape == (2,)

    def test_gradient_quadratic_approx(self):
        """∇V(x) ≈ 2x for V(x) = x₁² + x₂²."""
        from acf_functor.ode_acf import HJBReducer
        hjb = HJBReducer(dimension=2, order=8)
        hjb.fit(quadratic_lyapunov)
        x = np.array([0.4, 0.3])
        grad = hjb.gradient(x)
        expected = 2 * x
        np.testing.assert_allclose(grad, expected, atol=0.2)

    def test_invariants(self):
        from acf_functor.ode_acf import HJBReducer, HJBInvariants
        hjb = HJBReducer(dimension=2, order=6)
        hjb.fit(quadratic_lyapunov)
        inv = hjb.invariants()
        assert isinstance(inv, HJBInvariants)
        assert inv.dimension == 2
        assert inv.alpha_value >= 0

    def test_before_fit_raises(self):
        from acf_functor.ode_acf import HJBReducer
        hjb = HJBReducer(dimension=2)
        with pytest.raises(RuntimeError):
            hjb.value(np.zeros(2))


# ---------------------------------------------------------------------------
# LyapunovACF
# ---------------------------------------------------------------------------

class TestLyapunovACF:

    def test_import(self):
        from acf_functor.ode_acf import LyapunovACF
        assert LyapunovACF is not None

    def test_certify_damped_stable(self):
        from acf_functor.ode_acf import LyapunovACF, LyapunovCertificate
        cert_obj = LyapunovACF(dimension=2, radius=1.0, grid_res=8)
        cert = cert_obj.certify(quadratic_lyapunov, damped_field)
        assert isinstance(cert, LyapunovCertificate)
        assert cert.is_lyapunov

    def test_v_positive(self):
        from acf_functor.ode_acf import LyapunovACF
        cert_obj = LyapunovACF(dimension=2, radius=1.0, grid_res=8)
        cert = cert_obj.certify(quadratic_lyapunov, damped_field)
        assert cert.v_positive_on_grid

    def test_vdot_negative(self):
        from acf_functor.ode_acf import LyapunovACF
        cert_obj = LyapunovACF(dimension=2, radius=1.0, grid_res=8)
        cert = cert_obj.certify(quadratic_lyapunov, damped_field)
        # V̇ = 2x·(-x) = -2(x₁² + x₂²) < 0 on grid excl. origin
        assert cert.vdot_negative_on_grid
        assert cert.max_vdot < 0

    def test_summary_string(self):
        from acf_functor.ode_acf import LyapunovACF
        cert_obj = LyapunovACF(dimension=2, radius=1.0, grid_res=8)
        cert = cert_obj.certify(quadratic_lyapunov, damped_field)
        s = cert.summary()
        assert "Lyapunov" in s

    def test_alpha_lyapunov_positive(self):
        from acf_functor.ode_acf import LyapunovACF
        cert_obj = LyapunovACF(dimension=2, radius=1.0, grid_res=8)
        cert = cert_obj.certify(quadratic_lyapunov, damped_field)
        assert cert.alpha_lyapunov >= 0


# ---------------------------------------------------------------------------
# TrajectoryACF
# ---------------------------------------------------------------------------

class TestTrajectoryACF:

    def test_import(self):
        from acf_functor.ode_acf import TrajectoryACF
        assert TrajectoryACF is not None

    def test_integrate_origin(self):
        from acf_functor.ode_acf import VectorFieldReducer, TrajectoryACF
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        traj = TrajectoryACF(r, dt=0.01)
        x_T, _ = traj.integrate(np.zeros(2), T=1.0)
        np.testing.assert_allclose(x_T, np.zeros(2), atol=0.5)

    def test_integrate_returns_trajectory(self):
        from acf_functor.ode_acf import VectorFieldReducer, TrajectoryACF
        r = VectorFieldReducer(dimension=2, order=4)
        r.fit(damped_field, n_samples=80)
        traj = TrajectoryACF(r, dt=0.05)
        traj_arr, _ = traj.integrate(np.array([1.0, 0.0]), T=0.5, return_trajectory=True)
        assert traj_arr.shape[1] == 2
        assert traj_arr.shape[0] > 1
