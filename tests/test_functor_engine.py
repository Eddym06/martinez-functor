import pytest
import torch

from acf_functor import (
    ChebyshevReducer,
    EnrichedFunctor,
    ACFFunctor,
    ReductionPath,
)


@pytest.fixture
def phi():
    return ACFFunctor(default_dtype=torch.float64)


class TestHornerExact:
    def test_constant(self, phi):
        result = phi.reduce_polynomial([42.0])
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        assert torch.allclose(y, torch.full_like(x, 42.0))
        assert result.epsilon_bound == 0.0

    def test_linear(self, phi):
        result = phi.reduce_polynomial([3.0, 2.0])
        x = torch.linspace(-10, 10, 1000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        expected = 3.0 + 2.0 * x
        assert torch.allclose(y, expected, atol=1e-12)
        assert result.computational_energy == 1

    def test_high_degree(self, phi):
        torch.manual_seed(42)
        coeffs = torch.randn(51, dtype=torch.float64).tolist()
        result = phi.reduce_polynomial(coeffs)
        assert result.path == ReductionPath.HORNER_EXACT
        assert result.computational_energy == 50

        x = torch.linspace(-1, 1, 5000, dtype=torch.float64)
        y = phi.evaluate(result, x)
        expected = sum(coeffs[i] * x**i for i in range(len(coeffs)))
        assert torch.allclose(y, expected, atol=1e-9)

    def test_conservation_law(self, phi):
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = phi.reduce_polynomial(coeffs)
        assert phi.verify_conservation(4, result)


class TestChebyshevApprox:
    @pytest.mark.parametrize("func_name", ["sin", "cos", "exp", "tanh"])
    def test_canonical_functions(self, phi, func_name):
        result = phi.reduce_transcendental(func_name, degree=24)
        assert result.path == ReductionPath.CHEBYSHEV_APPROX

        canon = ChebyshevReducer.CANONICAL_FUNCTIONS[func_name]
        a, b = canon["domain"]
        x = torch.linspace(a, b, 5000, dtype=torch.float64)

        y_approx = phi.evaluate(result, x)
        y_exact = canon["generator"](x)

        max_err = torch.max(torch.abs(y_approx - y_exact)).item()
        assert max_err <= result.epsilon_bound * 1.01 + 1e-9

    def test_adaptive_degree(self, phi):
        result = phi.reduce_transcendental("sin", target_epsilon=1e-8)
        assert result.epsilon_bound < 1e-8


class TestKoopman:
    def test_linear_system_recovery(self, phi):
        torch.manual_seed(42)
        A = torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float64)
        x = torch.zeros(2, 200, dtype=torch.float64)
        x[:, 0] = torch.tensor([1.0, 0.5])
        for t in range(199):
            x[:, t + 1] = A @ x[:, t]

        result = phi.reduce_dynamical_system(x, observable_fn=lambda z: z, rank=2)
        K = result.fma_sequence[0].weight
        err = torch.norm(K - A).item()
        assert err < 1e-6

    def test_spectral_analysis(self, phi):
        eigenvalues = torch.tensor([0.95, 0.8, 0.5, 0.1, 0.01], dtype=torch.float64)
        alpha, delta = phi.compute_invariant(eigenvalues)
        assert alpha > 0
        assert delta == pytest.approx(0.01, abs=1e-15)


class TestComposition:
    def test_polynomial_composition(self, phi):
        phi_f = phi.reduce_polynomial([1.0, 1.0])
        phi_g = phi.reduce_polynomial([0.0, 0.0, 1.0])
        composed = phi.compose(phi_f, phi_g)
        assert composed.path == ReductionPath.COMPOSITE
        assert composed.epsilon_bound == 0.0

    def test_mixed_composition(self, phi):
        phi_sin = phi.reduce_transcendental("sin", degree=20)
        phi_poly = phi.reduce_polynomial([0.0, 0.0, 1.0])
        composed = phi.compose(phi_sin, phi_poly)
        assert composed.path == ReductionPath.COMPOSITE
        assert composed.epsilon_bound <= (
            phi_sin.epsilon_bound + phi_poly.epsilon_bound + phi_sin.epsilon_bound * phi_poly.epsilon_bound
        )


class TestEnriched:
    def test_error_subadditivity(self):
        from acf_functor.core import ReductionResult

        r1 = ReductionResult(
            path=ReductionPath.CHEBYSHEV_APPROX,
            fma_sequence=[],
            computational_energy=10,
            epsilon_bound=1e-6,
        )
        r2 = ReductionResult(
            path=ReductionPath.CHEBYSHEV_APPROX,
            fma_sequence=[],
            computational_energy=15,
            epsilon_bound=2e-6,
        )

        composed = EnrichedFunctor.compose(r1, r2)
        assert composed.epsilon_bound <= r1.epsilon_bound + r2.epsilon_bound + r1.epsilon_bound * r2.epsilon_bound


class TestEndToEnd:
    def test_full_pipeline_cpu(self, phi):
        x = torch.linspace(-3, 3, 10000, dtype=torch.float64)
        result_sin = phi.reduce_transcendental("sin")
        y_sin = phi.evaluate(result_sin, x)
        err_sin = torch.max(torch.abs(y_sin - torch.sin(x))).item()
        assert err_sin < 2e-5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_pipeline_gpu(self, phi):
        x = torch.linspace(-3, 3, 10000, dtype=torch.float64, device="cuda")
        result = phi.reduce_transcendental("sin")
        y = phi.evaluate(result, x, device="cuda")
        err = torch.max(torch.abs(y - torch.sin(x))).item()
        assert err < 2e-5
