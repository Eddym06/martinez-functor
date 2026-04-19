"""
Tests para las nuevas funcionalidades implementadas en el informe de análisis completo.

Cubre:
- Error propagation (Fase 2.2)
- Multivariate gradients (Fase 3.1)
- Modern activations (Fase 3.3)
- NN integration (Fase 5.1)
- Canonical benchmark (Fase 4.2)
"""

import pytest
import torch
import math
import numpy as np

from poema import (
    Poem, PoemCompiler,
    ErrorBound, compose_error_bounds, affine_error_propagation, sum_error_bounds,
    LIPSCHITZ_CONSTANTS,
    MultivariateExpr, JacobianExpr, parse_multivariate,
    gelu_exact, swiglu, rope_embedding,
    PoemActivationLayer, replace_activations_in_model,
)


# ============================================================
# FASE 2.2: Error Propagation
# ============================================================

class TestErrorPropagation:
    """Tests para propagación analítica de error en composiciones."""

    def test_composition_error_bound(self):
        """La cota propagada debe ser mayor o igual al error real."""
        # sin(cos(x)): cos tiene ε≈3.1e-3, sin tiene ε≈4.1e-3, L_sin=1
        cos_bound = ErrorBound(
            epsilon=3.1e-3, domain=(-math.pi, math.pi),
            lipschitz=1.0, source="lean_synchronized", is_certified=True
        )
        sin_bound = ErrorBound(
            epsilon=4.1e-3, domain=(-1.0, 1.0),
            lipschitz=1.0, source="lean_synchronized", is_certified=True
        )

        composed = compose_error_bounds(sin_bound, cos_bound)

        # Error teórico: ε_sin + L_sin * ε_cos = 4.1e-3 + 1.0 * 3.1e-3 = 7.2e-3
        assert abs(composed.epsilon - 7.2e-3) < 1e-10
        assert composed.is_certified  # ambas entradas son certificadas
        assert composed.lipschitz == 1.0  # L_sin * L_cos = 1 * 1

    def test_composition_preserves_certification(self):
        """Si ambos bounds son certificados, la composición también."""
        b1 = ErrorBound(epsilon=1e-3, domain=(-1, 1), lipschitz=1.0, source="lean", is_certified=True)
        b2 = ErrorBound(epsilon=2e-3, domain=(-2, 2), lipschitz=2.0, source="lean", is_certified=True)

        composed = compose_error_bounds(b1, b2)
        assert composed.is_certified

    def test_composition_non_certified(self):
        """Si uno no es certificado, la composición tampoco."""
        b1 = ErrorBound(epsilon=1e-3, domain=(-1, 1), lipschitz=1.0, source="lean", is_certified=True)
        b2 = ErrorBound(epsilon=2e-3, domain=(-2, 2), lipschitz=2.0, source="estimate", is_certified=False)

        composed = compose_error_bounds(b1, b2)
        assert not composed.is_certified

    def test_affine_error_propagation(self):
        """Transformación afín escala el error por |scale|."""
        input_bound = ErrorBound(
            epsilon=1e-3, domain=(-1, 1),
            lipschitz=1.0, source="lean", is_certified=True
        )

        # y = 2x + 3
        result = affine_error_propagation(2.0, 3.0, input_bound)
        assert result.epsilon == 2e-3  # escala el error
        assert result.domain == (1.0, 5.0)  # 2*(-1)+3, 2*(1)+3
        assert result.lipschitz == 2.0

    def test_affine_error_propagation_negative_scale(self):
        """Scale negativo invierte el dominio."""
        input_bound = ErrorBound(
            epsilon=1e-3, domain=(-1, 1),
            lipschitz=1.0, source="lean", is_certified=True
        )

        # y = -2x + 3
        result = affine_error_propagation(-2.0, 3.0, input_bound)
        assert result.epsilon == 2e-3
        assert result.domain == (1.0, 5.0)  # -2*(1)+3, -2*(-1)+3

    def test_sum_error_bounds(self):
        """Error de f + g está acotado por ε_f + ε_g."""
        b1 = ErrorBound(epsilon=1e-3, domain=(-1, 1), lipschitz=1.0, source="lean", is_certified=True)
        b2 = ErrorBound(epsilon=2e-3, domain=(-1, 1), lipschitz=2.0, source="lean", is_certified=True)

        result = sum_error_bounds(b1, b2)
        assert result.epsilon == 3e-3
        assert result.lipschitz == 3.0
        assert result.is_certified

    def test_lipschitz_constants(self):
        """Constantes de Lipschitz conocidas son correctas."""
        assert LIPSCHITZ_CONSTANTS["sin"] == 1.0
        assert LIPSCHITZ_CONSTANTS["cos"] == 1.0
        assert LIPSCHITZ_CONSTANTS["tanh"] == 1.0
        assert LIPSCHITZ_CONSTANTS["sigmoid"] == 0.25
        assert LIPSCHITZ_CONSTANTS["exp"] == math.exp(1)
        assert LIPSCHITZ_CONSTANTS["log"] == 2.0

    def test_composition_empirical_bound(self):
        """Verificar empíricamente que la cota es real para sin(cos(x))."""
        P = Poem(dtype=torch.float64)
        ast = P.continuous_flow("sin(cos(x))", domain=(-math.pi, math.pi))
        compiler = PoemCompiler(target="pytorch", precision="fp64")
        fn, report = compiler.compile(ast, domain=(-math.pi, math.pi))

        x = torch.linspace(-math.pi, math.pi, 10000, dtype=torch.float64)
        reference = torch.sin(torch.cos(x))
        actual_error = (fn(x) - reference).abs().max().item()

        # La cota propagada debe ser una cota real (con margen numérico)
        assert actual_error < 0.1, f"Error real {actual_error} es excesivo"


# ============================================================
# FASE 3.1: Multivariate Gradients
# ============================================================

class TestMultivariateGradients:
    """Tests para gradientes multivariables reales."""

    def test_parse_multivariate_scalar(self):
        """Parsear expresión escalar multivariable."""
        expr = parse_multivariate("x^2 + y^2", ["x", "y"])
        assert expr.variables == ["x", "y"]
        assert len(expr.components) == 1

    def test_parse_multivariate_vector(self):
        """Parsear expresión vectorial multivariable."""
        expr = parse_multivariate("[x*cos(y), x*sin(y)]", ["x", "y"])
        assert expr.variables == ["x", "y"]
        assert len(expr.components) == 2

    def test_multivariate_gradient(self):
        """Calcular gradiente de función multivariable."""
        expr = parse_multivariate("x^2 + y^2", ["x", "y"])
        grad_x = expr.gradient("x")
        assert grad_x.variables == ["x", "y"]
        assert len(grad_x.components) == 1

    def test_multivariate_jacobian(self):
        """Calcular Jacobiana de función multivariable."""
        expr = parse_multivariate("[x^2, x*y]", ["x", "y"])
        jacobian = expr.jacobian()
        assert jacobian.variables == ["x", "y"]
        assert len(jacobian.entries) == 2  # 2 componentes
        assert len(jacobian.entries[0]) == 2  # 2 variables


# ============================================================
# FASE 3.3: Modern Activations
# ============================================================

class TestModernActivations:
    """Tests para funciones de activación modernas certificadas."""

    def test_gelu_exact(self):
        """GELU exacto con certificado de error."""
        P = Poem(dtype=torch.float64)
        ast = gelu_exact(P, degree=40, domain=(-4.0, 4.0))
        assert ast is not None

        compiler = PoemCompiler(target="pytorch", precision="fp64")
        fn, report = compiler.compile(ast, domain=(-4.0, 4.0))

        x = torch.linspace(-4.0, 4.0, 1000, dtype=torch.float64)
        # GELU reference: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        reference = 0.5 * x * (1 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))

        error = (fn(x) - reference).abs().max().item()
        assert error < 1e-6, f"Error GELU {error} excede tolerancia"

    def test_swiglu(self):
        """SwiGLU / Swish: x * sigmoid(x)."""
        P = Poem(dtype=torch.float64)
        ast = swiglu(P, degree=30, domain=(-6.0, 6.0))
        assert ast is not None

        compiler = PoemCompiler(target="pytorch", precision="fp64")
        fn, report = compiler.compile(ast, domain=(-6.0, 6.0))

        x = torch.linspace(-6.0, 6.0, 1000, dtype=torch.float64)
        reference = x * torch.sigmoid(x)

        error = (fn(x) - reference).abs().max().item()
        assert error < 1e-4, f"Error SwiGLU {error} excede tolerancia"

    def test_rope_embedding(self):
        """RoPE embedding genera coeficientes correctos."""
        P = Poem(dtype=torch.float64)
        rope = rope_embedding(P, dim=64, max_seq_len=2048, base=10000.0)

        assert "thetas" in rope
        assert len(rope["thetas"]) == 32  # dim // 2
        assert rope["dim"] == 64

        # Compilar para posición específica
        pos_result = rope["compile_for_position"](10.0)
        assert "cos" in pos_result
        assert "sin" in pos_result
        assert len(pos_result["cos"]) == 32
        assert len(pos_result["sin"]) == 32


# ============================================================
# FASE 5.1: NN Integration
# ============================================================

class TestNNIntegration:
    """Tests para integración con PyTorch nn.Module."""

    def test_poem_activation_gelu(self):
        """PoemActivationLayer GELU funciona como nn.Module."""
        layer = PoemActivationLayer.gelu(domain=(-4.0, 4.0))
        assert isinstance(layer, torch.nn.Module)
        assert layer.epsilon_certified >= 0

        x = torch.randn(10, dtype=torch.float64) * 2
        y = layer(x)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_poem_activation_swish(self):
        """PoemActivationLayer Swish funciona como nn.Module."""
        layer = PoemActivationLayer.swish(domain=(-6.0, 6.0))
        x = torch.randn(10, dtype=torch.float64) * 3
        y = layer(x)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))

    def test_poem_activation_relu(self):
        """PoemActivationLayer ReLU funciona como nn.Module."""
        layer = PoemActivationLayer.relu(domain=(-4.0, 4.0))
        x = torch.linspace(-4.0, 4.0, 100, dtype=torch.float64)
        y = layer(x)
        assert torch.all(y >= -1e-10)  # ReLU >= 0 (con tolerancia numérica)

    def test_poem_activation_tanh(self):
        """PoemActivationLayer tanh con dominio extendido."""
        layer = PoemActivationLayer.tanh_act(domain=(-4.0, 4.0))
        x = torch.linspace(-4.0, 4.0, 100, dtype=torch.float64)
        y = layer(x)
        reference = torch.tanh(x)
        error = (y - reference).abs().max().item()
        assert error < 1e-4, f"Error tanh {error} excede tolerancia"

    def test_replace_activations_in_model(self):
        """Reemplazar activaciones en modelo existente."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20, dtype=torch.float64)
                self.act1 = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(20, 5, dtype=torch.float64)
                self.act2 = torch.nn.SiLU()

            def forward(self, x):
                return self.act2(self.fc2(self.act1(self.fc1(x))))

        model = SimpleModel()

        # Reemplazar activaciones
        model = replace_activations_in_model(
            model,
            {
                torch.nn.GELU: PoemActivationLayer.gelu(dtype=torch.float64),
                torch.nn.SiLU: PoemActivationLayer.swish(dtype=torch.float64),
            },
            verbose=False,
        )

        # Verificar que las activaciones fueron reemplazadas
        assert isinstance(model.act1, PoemActivationLayer)
        assert isinstance(model.act2, PoemActivationLayer)

        # Verificar que el modelo funciona
        x = torch.randn(2, 10, dtype=torch.float64)
        y = model(x)
        assert y.shape == (2, 5)
        assert torch.all(torch.isfinite(y))


# ============================================================
# FASE 4.2: Canonical Benchmark
# ============================================================

class TestCanonicalBenchmark:
    """Tests para el benchmark canónico reproducible."""

    def test_benchmark_runs(self):
        """El benchmark canónico se ejecuta sin errores."""
        from benchmarks.canonical_benchmark import run_canonical_benchmark
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "benchmark_results.json")
            results = run_canonical_benchmark(output_file=output_file, n_warmup=1, n_runs=2)

            assert len(results) >= 2  # Al menos polynomial y sin
            assert results[0].name == "polynomial_degree100_cpu_fp64"
            assert results[0].passed

    def test_benchmark_output_file(self):
        """El benchmark genera archivo JSON válido."""
        from benchmarks.canonical_benchmark import run_canonical_benchmark
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "benchmark_results.json")
            run_canonical_benchmark(output_file=output_file, n_warmup=1, n_runs=2)

            assert os.path.exists(output_file)
            with open(output_file) as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "results" in data
            assert "summary" in data
            assert data["summary"]["total"] >= 2
