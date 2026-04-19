import math

import pytest
import torch

from acf_functor.core import ReductionPath
from acf_functor.self_modulation import (
    SelfModulatingFunctor,
    SheafReduction,
    TearType,
    TopologicalResidue,
)


@pytest.fixture
def phi():
    return SelfModulatingFunctor(
        default_dtype=torch.float64,
        base_degree=20,
        target_epsilon=1e-8,
        max_iterations=8,
        n_probe_points=2000,
    )


class TestSmoothFunctions:
    def test_sin_no_tears(self, phi):
        result = phi.reduce_transcendental("sin")
        x = torch.linspace(-math.pi, math.pi, 5000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        err = torch.max(torch.abs(y - torch.sin(x))).item()
        assert err < 1e-8

        if isinstance(result, SheafReduction):
            assert len(result.tear_history) == 0

    def test_exp_no_tears(self, phi):
        result = phi.reduce_transcendental("exp")
        a, b = result.domain
        x = torch.linspace(a, b, 5000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        err = torch.max(torch.abs(y - torch.exp(x))).item()
        assert err < 1e-8

    def test_polynomial_exact(self, phi):
        coeffs = [1.0, -2.0, 3.0, -0.5, 0.1]
        result = phi.reduce_polynomial(coeffs)
        assert result.epsilon_bound == 0.0
        assert result.path == ReductionPath.HORNER_EXACT


class TestSharpTransitions:
    def test_narrow_gaussian(self, phi):
        def narrow_gaussian(x):
            return torch.exp(-500 * x**2)

        result = phi.reduce_arbitrary(narrow_gaussian, domain=(-2.0, 2.0), target_epsilon=1e-6)
        x = torch.linspace(-2, 2, 10000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        y_true = narrow_gaussian(x)
        err = torch.max(torch.abs(y - y_true)).item()

        assert err < 1e-3
        assert isinstance(result, SheafReduction)

    def test_tanh_sharp(self, phi):
        def sharp_tanh(x):
            return torch.tanh(20 * x)

        result = phi.reduce_arbitrary(sharp_tanh, domain=(-2.0, 2.0), target_epsilon=1e-6)
        x = torch.linspace(-2, 2, 10000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        y_true = sharp_tanh(x)
        err = torch.max(torch.abs(y - y_true)).item()

        assert err < 1e-3


class TestDiscontinuousFunctions:
    def test_step_function(self, phi):
        def heaviside(x):
            return (x >= 0).to(x.dtype)

        result = phi.reduce_arbitrary(heaviside, domain=(-2.0, 2.0), target_epsilon=1e-2)

        assert isinstance(result, SheafReduction)
        assert len(result.strata) > 1
        assert any(
            t.tear_type in (TearType.DISCONTINUITY, TearType.OSCILLATORY, TearType.HIGH_CURVATURE)
            for t in result.tear_history
        )

        x_left = torch.linspace(-2, -0.1, 1000, dtype=torch.float64)
        x_right = torch.linspace(0.1, 2, 1000, dtype=torch.float64)
        y_left = phi.evaluate(result, x_left)
        y_right = phi.evaluate(result, x_right)

        assert torch.max(torch.abs(y_left - 0.0)).item() < 0.2
        assert torch.max(torch.abs(y_right - 1.0)).item() < 0.2

    def test_piecewise_linear(self, phi):
        def abs_val(x):
            return torch.abs(x)

        result = phi.reduce_arbitrary(abs_val, domain=(-3.0, 3.0), target_epsilon=1e-6)
        x = torch.linspace(-3, 3, 10000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        y_true = abs_val(x)
        err = torch.max(torch.abs(y - y_true)).item()
        assert err < 1e-3

    def test_relu(self, phi):
        def relu(x):
            return torch.relu(x)

        result = phi.reduce_arbitrary(relu, domain=(-3.0, 3.0), target_epsilon=1e-6)
        x = torch.linspace(-3, 3, 10000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        y_true = relu(x)
        err = torch.max(torch.abs(y - y_true)).item()
        assert err < 1e-3


class TestResidueAnalyzer:
    def test_detects_discontinuity(self):
        analyzer = TopologicalResidue(sensitivity_threshold=1e-3, gradient_ratio_threshold=5.0)

        x = torch.linspace(-2, 2, 1000, dtype=torch.float64)
        residual = torch.zeros_like(x)
        center = 500
        residual[center - 2 : center + 2] = 1.0

        tears = analyzer.detect_tears(x, residual, 1e-4)
        assert len(tears) > 0
        assert tears[0].tear_type in (TearType.DISCONTINUITY, TearType.HIGH_CURVATURE)

    def test_no_tears_for_smooth(self):
        analyzer = TopologicalResidue(sensitivity_threshold=1e-6)
        x = torch.linspace(-1, 1, 1000, dtype=torch.float64)
        residual = 1e-8 * torch.ones_like(x)
        tears = analyzer.detect_tears(x, residual, 1e-6)
        assert len(tears) == 0


class TestMonadicFixpoint:
    def test_idempotence_polynomial(self, phi):
        coeffs = [1.0, 2.0, 3.0]
        r1 = phi.reduce_polynomial(coeffs)
        x = torch.linspace(-5, 5, 1000, dtype=torch.float64)
        y1 = phi.evaluate(r1, x)

        def phi_f(t):
            return phi.evaluate(r1, t)

        r2 = phi.reduce_arbitrary(phi_f, domain=(-5.0, 5.0), target_epsilon=1e-10)
        y2 = phi.evaluate(r2, x)

        err = torch.max(torch.abs(y1 - y2)).item()
        assert err < 1e-8

    def test_idempotence_transcendental(self, phi):
        r1 = phi.reduce_transcendental("sin")
        x = torch.linspace(-math.pi, math.pi, 1000, dtype=torch.float64)
        y1 = phi.evaluate(r1, x)

        def phi_sin(t):
            return phi.evaluate(r1, t)

        r2 = phi.reduce_arbitrary(phi_sin, domain=(-math.pi, math.pi), target_epsilon=1e-10)
        y2 = phi.evaluate(r2, x)

        err = torch.max(torch.abs(y1 - y2)).item()
        assert err < 1e-8


class TestSelfModulationIntrinsicality:
    def test_unknown_discontinuity_location(self, phi):
        breakpoint = math.sqrt(2) - 1

        def mystery_func(x):
            return torch.where(x < breakpoint, torch.sin(x), torch.cos(x) + 0.5)

        result = phi.reduce_arbitrary(mystery_func, domain=(-2.0, 2.0), target_epsilon=1e-2)
        assert isinstance(result, SheafReduction)

        if result.tear_history:
            closest = min(result.tear_history, key=lambda t: abs(t.location - breakpoint))
            assert abs(closest.location - breakpoint) < 0.3

    def test_multiple_discontinuities(self, phi):
        def multi_step(x):
            result = torch.zeros_like(x)
            result = torch.where(x < -1, -torch.ones_like(x), result)
            result = torch.where((x >= -1) & (x < 0), torch.sin(x), result)
            result = torch.where((x >= 0) & (x < 1), torch.exp(x), result)
            result = torch.where(x >= 1, 2 * torch.ones_like(x), result)
            return result

        result = phi.reduce_arbitrary(multi_step, domain=(-3.0, 3.0), target_epsilon=1e-2)
        assert isinstance(result, SheafReduction)
        assert len(result.strata) >= 2

    def test_diagnose_reveals_structure(self, phi):
        def difficult_func(x):
            return torch.where(x < 0, torch.sin(10 * x), torch.exp(-x))

        diag = phi.diagnose(difficult_func, domain=(-3.0, 3.0))

        assert "n_tears_detected" in diag
        assert "tear_details" in diag
        assert diag["recommended_strategy"] in ("EXACT_CHEBYSHEV", "SHEAF_INJECTION")


class TestCompositionThroughTears:
    def test_smooth_composed_with_sharp(self, phi):
        def sharp_tanh(x):
            return torch.tanh(20 * x)

        r_inner = phi.reduce_arbitrary(sharp_tanh, domain=(-2.0, 2.0), target_epsilon=1e-4)
        r_outer = phi.reduce_transcendental("sin", domain=(-1.0, 1.0), target_epsilon=1e-8)

        composed = phi.compose(r_outer, r_inner)
        assert composed.epsilon_bound < 1.0


class TestConvergenceMonad:
    def test_convergence_decreases_error(self, phi):
        def oscillatory(x):
            return torch.sin(5 * x) * torch.exp(-x**2)

        result = phi.reduce_arbitrary(oscillatory, domain=(-3.0, 3.0), target_epsilon=1e-8)
        if isinstance(result, SheafReduction):
            hist = result.metadata.get("iteration_history", [])
            if len(hist) > 1:
                assert hist[-1]["max_error"] <= hist[0]["max_error"] * 1.1

    def test_achieves_target_for_smooth(self, phi):
        result = phi.reduce_arbitrary(torch.sin, domain=(-3.0, 3.0), target_epsilon=1e-8)
        x = torch.linspace(-3, 3, 10000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        err = torch.max(torch.abs(y - torch.sin(x))).item()
        assert err < 1e-7
