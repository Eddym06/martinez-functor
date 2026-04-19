import math
import random

import pytest
import torch

from poema import (
    AffineNode,
    BiPoem,
    CoPoem,
    ComposeNode,
    ConstantNode,
    IdentityNode,
    Poem,
    PoemCompiler,
    ScaleNode,
    Scalar,
    ShiftNode,
    TopologicalObstructionError,
    Vector,
)


@pytest.fixture
def poem():
    return Poem()


@pytest.fixture
def compiler():
    return PoemCompiler(target="pytorch", precision="fp64", verbose=False)


class TestFreeAlgebra:
    def test_scale_identity(self, poem):
        node = poem.scale(1.0)
        out = node.simplify()
        assert isinstance(out, IdentityNode)

    def test_shift_identity(self, poem):
        node = poem.shift(0.0)
        out = node.simplify()
        assert isinstance(out, IdentityNode)

    def test_scale_composition(self, poem):
        inner = poem.scale(3.0)
        outer = poem.scale(2.0, inner)
        out = outer.simplify()
        assert isinstance(out, ScaleNode)
        assert abs(out.factor.item() - 6.0) < 1e-12

    def test_shift_composition(self, poem):
        inner = poem.shift(3.0)
        outer = poem.shift(2.0, inner)
        out = outer.simplify()
        assert isinstance(out, ShiftNode)
        assert abs(out.value.item() - 5.0) < 1e-12

    def test_affine_from_scale_shift(self, poem):
        out = poem.compose(poem.scale(2.0), poem.shift(3.0)).simplify()
        assert isinstance(out, AffineNode)
        assert abs(out.scale_factor.item() - 2.0) < 1e-12
        assert abs(out.shift_value.item() - 6.0) < 1e-12


class TestTypeSystem:
    def test_composable_scalar(self):
        assert Scalar().is_composable_with(Scalar())

    def test_dimension_mismatch_raises(self):
        with pytest.raises(TopologicalObstructionError):
            ComposeNode(
                outer=ScaleNode(torch.tensor(1.0), geometric_type=Vector(2)),
                inner=ScaleNode(torch.tensor(1.0), geometric_type=Vector(3)),
            )


class TestCompilation:
    def test_polynomial_exact(self, poem, compiler):
        poly = poem.polynomial([1.0, 2.0, 3.0])
        exe, report = compiler.compile(poly)
        x = torch.linspace(-3, 3, 2000, dtype=torch.float64)
        y = exe(x)
        expected = 1.0 + 2.0 * x + 3.0 * x**2
        assert torch.allclose(y, expected, atol=1e-12)
        assert report.total_epsilon == 0.0

    @pytest.mark.parametrize(
        "name,domain,fn",
        [
            ("sin", (-math.pi, math.pi), torch.sin),
            ("cos", (-math.pi, math.pi), torch.cos),
            ("exp", (-2.0, 2.0), torch.exp),
            ("tanh", (-3.0, 3.0), torch.tanh),
        ],
    )
    def test_transcendentals(self, poem, compiler, name, domain, fn):
        node = getattr(poem, name)(domain=domain, degree=24)
        exe, report = compiler.compile(node, domain=domain)
        x = torch.linspace(domain[0], domain[1], 4000, dtype=torch.float64)
        y = exe(x)
        err = torch.max(torch.abs(y - fn(x))).item()
        assert err < 1e-4
        assert report.total_fma_ops > 0

    def test_expression_parser(self, poem, compiler):
        ast = poem.continuous_flow("sin(x)", domain=(-math.pi, math.pi), degree=24)
        exe, _ = compiler.compile(ast, domain=(-math.pi, math.pi))
        x = torch.linspace(-math.pi, math.pi, 2000, dtype=torch.float64)
        y = exe(x)
        assert torch.max(torch.abs(y - torch.sin(x))).item() < 1e-4

    def test_high_degree_compose_semantics_regression(self, poem, compiler):
        torch.manual_seed(0)

        f1 = poem.sin(domain=(-1.5, 1.5), degree=100)

        g = poem.polynomial([0.2, 0.6, -0.15, 0.06])
        h = poem.compose(f1, g)

        exe_h, _ = compiler.compile(h)

        x = torch.linspace(-1.5, 1.5, 10001, dtype=torch.float64)
        y = exe_h(x)
        ref = torch.sin(0.2 + 0.6 * x - 0.15 * x**2 + 0.06 * x**3)

        max_err = torch.max(torch.abs(y - ref)).item()
        mean_err = torch.mean(torch.abs(y - ref)).item()
        assert torch.isfinite(y).all().item()
        assert max_err < 1e-6
        assert mean_err < 1e-7


class TestModes:
    def test_copoem_spectrum(self):
        co = CoPoem()
        spec = co.spectrum(spectral_radius=0.95, dimension=16, symmetry="orthogonal")
        w = co.synthesize(spec)
        eig = torch.linalg.eigvals(w)
        assert w.shape == (16, 16)
        assert torch.max(torch.abs(eig)).item() < 1.0

    def test_bipoem_symbiosis(self):
        bi = BiPoem()
        torch.manual_seed(0)
        a = torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float64)
        x = torch.zeros(2, 80, dtype=torch.float64)
        x[:, 0] = torch.randn(2, dtype=torch.float64)
        for t in range(79):
            x[:, t + 1] = a @ x[:, t]

        out = bi.symbiosis(data=x, max_dimension=32, max_iterations=8)
        assert "koopman_matrix" in out
        assert "optimal_dimension" in out
        assert out["reconstruction_error"] < 0.2


class TestReporting:
    def test_report_summary(self, poem, compiler):
        poly = poem.polynomial([1.0, 0.0, 2.0])
        _, report = compiler.compile(poly)
        text = report.summary()
        assert "POEMA COMPILATION REPORT" in text
        assert "FMA ops" in text

    def test_domain_guard_summary_metrics_present(self, poem, compiler):
        inner = poem.polynomial([0.0, 3.0])
        outer = poem.sin(domain=(-1.0, 1.0), degree=48)
        ast = poem.compose(outer, inner)
        _, report = compiler.compile(ast, domain=(-1.0, 1.0))
        text = report.summary()
        assert "Domain guard checks" in text
        assert "Domain guard violations" in text
        assert report.domain_guard_checks >= 1
        assert report.domain_guard_violations >= 1
        assert report.domain_guard_max_overshoot > 0.0


class TestDeepRandomCompose:
    @staticmethod
    def _poly_eval(coeffs, x):
        y = torch.zeros_like(x)
        for c in reversed(coeffs):
            y = y * x + c
        return y

    @staticmethod
    def _poly_interval(coeffs, domain):
        x = torch.linspace(domain[0], domain[1], 2049, dtype=torch.float64)
        y = TestDeepRandomCompose._poly_eval(coeffs, x)
        return float(torch.min(y).item()), float(torch.max(y).item())

    def test_random_deep_compose_domain_safe_levels_2_and_3(self, poem, compiler):
        random.seed(0)
        torch.manual_seed(0)

        base_domain = (-1.0, 1.0)
        for depth in [2, 3]:
            for _ in range(8):
                coeffs = [
                    random.uniform(-0.1, 0.1),
                    random.uniform(0.25, 0.5),
                    random.uniform(-0.06, 0.06),
                    random.uniform(-0.02, 0.02),
                ]
                current = poem.polynomial(coeffs)
                current_domain = self._poly_interval(coeffs, base_domain)

                for level in range(depth):
                    margin = 0.2 + 0.05 * level
                    trans_domain = (current_domain[0] - margin, current_domain[1] + margin)
                    outer = poem.sin(domain=trans_domain, degree=48)
                    current = poem.compose(outer, current)
                    current_domain = (
                        float(torch.sin(torch.tensor(current_domain[0], dtype=torch.float64)).item()),
                        float(torch.sin(torch.tensor(current_domain[1], dtype=torch.float64)).item()),
                    )
                    current_domain = (min(current_domain), max(current_domain))

                exe, report = compiler.compile(current, domain=base_domain)
                x = torch.linspace(base_domain[0], base_domain[1], 4001, dtype=torch.float64)
                y = exe(x)
                assert torch.isfinite(y).all().item()
                assert report.domain_guard_checks >= depth
                assert report.domain_guard_violations == 0

    def test_random_deep_compose_domain_violation_detected(self, poem, compiler):
        random.seed(7)
        torch.manual_seed(7)

        base_domain = (-1.0, 1.0)
        coeffs = [
            random.uniform(-0.2, 0.2),
            random.uniform(1.6, 2.2),
            random.uniform(-0.25, 0.25),
            random.uniform(-0.1, 0.1),
        ]
        inner = poem.polynomial(coeffs)

        # Intentionally narrow certified domain to trigger guard on compose.
        outer = poem.sin(domain=(-0.9, 0.9), degree=40)
        deep = poem.compose(outer, inner)
        deep = poem.compose(poem.sin(domain=(-1.0, 1.0), degree=40), deep)

        _, report = compiler.compile(deep, domain=base_domain)
        assert report.domain_guard_checks >= 2
        assert report.domain_guard_violations >= 1
        assert report.domain_guard_max_overshoot > 0.0
        assert any("domain guard" in w for w in report.warnings)
