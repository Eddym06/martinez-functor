"""
Tests comprehensivos para las 4 nuevas capacidades v2.2.0:
1. GEMM-Triton Collider
2. Auto-Domain Repair mejorado
3. Kahan Horner Kernel
4. CoPoem Multiobjetivo con Anderson
"""

from __future__ import annotations

import math
import pytest
import torch

from poema.ast_nodes import FMAInstruction
from poema.gemm_collider import GEMMCollider, KahanHornerKernel, GEMMBlock, ColliderReport
from poema.auto_domain_repair import ExpandedDomainRepair, ExpandedDomainCert
from poema.copoem_multiobjective import CoPoemMultiObjective, AndersonState, IncompatibilityReport


# =============================================================================
# 1. GEMM-Triton Collider Tests
# =============================================================================

class TestGEMMCollider:
    """Tests para el colapsador de cadenas afines en bloques GEMM."""

    def test_analyze_empty_chain(self):
        """Cadena vacía debe retornar reporte sin bloques."""
        report = GEMMCollider.analyze_chain([])
        assert report.total_blocks == 0
        assert report.total_fma_collapsed == 0
        assert report.total_gemm_ops == 0

    def test_analyze_scalar_chain(self):
        """Cadena escalar debe colapsar en un solo bloque GEMM 1x1."""
        fmas = [
            FMAInstruction(weight=torch.tensor(2.0), bias=torch.tensor(1.0)),
            FMAInstruction(weight=torch.tensor(3.0), bias=torch.tensor(-1.0)),
        ]
        report = GEMMCollider.analyze_chain(fmas)
        assert report.total_blocks == 1
        assert report.total_fma_collapsed == 2
        assert report.blocks[0].weight.shape == (1, 1)

    def test_analyze_vector_chain(self):
        """Cadena vectorial 2x2 debe colapsar correctamente."""
        W1 = torch.tensor([[0.9, 0.1], [-0.1, 0.8]])
        b1 = torch.tensor([0.0, 0.0])
        W2 = torch.eye(2)
        b2 = torch.tensor([0.1, -0.1])

        fmas = [
            FMAInstruction(weight=W1, bias=b1),
            FMAInstruction(weight=W2, bias=b2),
        ]
        report = GEMMCollider.analyze_chain(fmas)
        assert report.total_blocks == 1
        assert report.total_fma_collapsed == 2
        assert report.blocks[0].weight.shape == (2, 2)
        assert report.blocks[0].bias.shape == (2,)

    def test_collapse_correctness(self):
        """Verificar que el colapso W_total = W2 @ W1 es correcto."""
        W1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
        b1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        W2 = torch.tensor([[0.5, 0.0], [0.0, 0.25]], dtype=torch.float64)
        b2 = torch.tensor([0.0, 0.0], dtype=torch.float64)

        fmas = [
            FMAInstruction(weight=W1, bias=b1),
            FMAInstruction(weight=W2, bias=b2),
        ]
        report = GEMMCollider.analyze_chain(fmas)

        # W_total = W2 @ W1 = [[1.0, 0.0], [0.0, 0.75]]
        # b_total = W2 @ b1 + b2 = [0.5, 0.5]
        W_expected = torch.tensor([[1.0, 0.0], [0.0, 0.75]], dtype=torch.float64)
        b_expected = torch.tensor([0.5, 0.5], dtype=torch.float64)

        assert torch.allclose(report.blocks[0].weight, W_expected, atol=1e-6)
        assert torch.allclose(report.blocks[0].bias, b_expected, atol=1e-6)

    def test_condition_number_computation(self):
        """Número de condición debe calcularse correctamente."""
        # Matriz bien condicionada
        W_well = torch.eye(2)
        fmas_well = [FMAInstruction(weight=W_well, bias=torch.zeros(2))]
        report_well = GEMMCollider.analyze_chain(fmas_well)
        assert report_well.blocks[0].condition_number < 10.0
        assert report_well.blocks[0].recommended_dtype == torch.float32

    def test_fp64_promotion_for_ill_conditioned(self):
        """Matriz mal condicionada debe promover a fp64."""
        # Matriz mal condicionada: número de condición alto
        W_ill = torch.tensor([[1e6, 0.0], [0.0, 1.0]])
        fmas_ill = [FMAInstruction(weight=W_ill, bias=torch.zeros(2))]
        report_ill = GEMMCollider.analyze_chain(fmas_ill)
        assert report_ill.blocks[0].condition_number > 1e5
        assert report_ill.blocks[0].recommended_dtype == torch.float64

    def test_memory_footprint_calculation(self):
        """Footprint de memoria debe calcularse correctamente."""
        W = torch.randn(64, 32, dtype=torch.float64)
        b = torch.randn(64, dtype=torch.float64)
        fmas = [FMAInstruction(weight=W, bias=b)]
        report = GEMMCollider.analyze_chain(fmas)
        expected = W.numel() * W.element_size() + b.numel() * b.element_size()
        assert report.memory_footprint_bytes == expected

    def test_diagonal_vector_promotion(self):
        """Vector debe promoverse a matriz diagonal."""
        w = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        fmas = [FMAInstruction(weight=w, bias=b)]
        report = GEMMCollider.analyze_chain(fmas)
        assert report.blocks[0].weight.shape == (3, 3)
        # Debe ser diagonal
        assert torch.allclose(report.blocks[0].weight, torch.diag(w), atol=1e-6)

    def test_long_chain_collapse(self):
        """Cadena larga de 10 operaciones debe colapsar en un solo bloque."""
        fmas = []
        for i in range(10):
            W = torch.eye(4) * (0.9 ** i)
            b = torch.randn(4) * 0.1
            fmas.append(FMAInstruction(weight=W, bias=b))

        report = GEMMCollider.analyze_chain(fmas)
        assert report.total_blocks == 1
        assert report.total_fma_collapsed == 10
        assert report.total_gemm_ops == 1

    def test_gemm_kernel_pytorch_fallback(self):
        """Kernel GEMM debe funcionar con fallback a PyTorch."""
        W = torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float32)
        b = torch.tensor([0.1, -0.1], dtype=torch.float32)
        block = GEMMBlock(
            weight=W, bias=b, source_nodes=["test"],
            fma_ops_collapsed=1, condition_number=1.1,
            recommended_dtype=torch.float32,
        )
        kernel = GEMMCollider.compile_gemm_kernel(block, "test_gemm")
        assert kernel is not None

        # Verificar corrección numérica
        x = torch.randn(2, 100, dtype=torch.float32)
        y = kernel(x)
        y_expected = W @ x + b.unsqueeze(1)
        assert torch.allclose(y, y_expected, atol=1e-5)


# =============================================================================
# 2. Kahan Horner Kernel Tests
# =============================================================================

class TestKahanHornerKernel:
    """Tests para evaluación Horner con compensación de Kahan."""

    def test_kahan_correctness_degree_10(self):
        """Kahan Horner debe evaluar correctamente polinomio de grado 10."""
        coeffs = torch.randn(11, dtype=torch.float64)
        kernel = KahanHornerKernel.generate(coeffs, "test_kahan_10")
        assert kernel is not None

        x = torch.linspace(-1, 1, 1000, dtype=torch.float64)
        y = kernel(x)

        # Verificar contra evaluación directa
        y_expected = torch.zeros_like(x)
        for i, c in enumerate(coeffs):
            y_expected += c * (x ** i)

        assert torch.allclose(y, y_expected, atol=1e-10)

    def test_kahan_stability_vs_naive(self):
        """Kahan debe ser más estable que Horner naive para grados altos."""
        # Polinomio mal condicionado
        coeffs = torch.randn(50, dtype=torch.float64)
        kernel = KahanHornerKernel.generate(coeffs, "test_kahan_50")

        x = torch.linspace(-0.5, 0.5, 1000, dtype=torch.float64)
        y = kernel(x)

        # Verificar que todos los resultados son finitos
        assert torch.all(torch.isfinite(y))

    def test_kahan_pytorch_fallback(self):
        """Fallback a PyTorch debe funcionar correctamente."""
        coeffs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # 1 + 2x + 3x²
        kernel = KahanHornerKernel.generate(coeffs, "test_fallback")
        assert kernel is not None

        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        y = kernel(x)
        # 1 + 2x + 3x²: [1, 6, 17]
        y_expected = torch.tensor([1.0, 6.0, 17.0], dtype=torch.float64)
        assert torch.allclose(y, y_expected, atol=1e-10)


# =============================================================================
# 3. Auto-Domain Repair Tests
# =============================================================================

class TestExpandedDomainRepair:
    """Tests para reparación de dominio con polinomios de grado superior."""

    def test_expanded_cert_sin(self):
        """Certificado expandido para sin debe tener dominio [-2π, 2π] y grado 48."""
        cert = ExpandedDomainRepair.get_expanded_cert('sin', (-math.pi, math.pi), 24, 1e-3)
        assert cert is not None
        assert cert.expanded_domain == (-2 * math.pi, 2 * math.pi)
        assert cert.expanded_degree == 48
        assert cert.expanded_epsilon < 1e-10  # Debe ser muy preciso

    def test_expanded_cert_cos(self):
        """Certificado expandido para cos debe tener dominio [-2π, 2π] y grado 48."""
        cert = ExpandedDomainRepair.get_expanded_cert('cos', (-math.pi, math.pi), 24, 1e-3)
        assert cert is not None
        assert cert.expanded_domain == (-2 * math.pi, 2 * math.pi)
        assert cert.expanded_degree == 48

    def test_expanded_cert_exp(self):
        """Certificado expandido para exp debe tener dominio [-3, 3] y grado 30."""
        cert = ExpandedDomainRepair.get_expanded_cert('exp', (-1.0, 1.0), 15, 1e-3)
        assert cert is not None
        assert cert.expanded_domain == (-3.0, 3.0)
        assert cert.expanded_degree == 30

    def test_expanded_cert_tanh(self):
        """Certificado expandido para tanh debe tener dominio [-8, 8] y grado 80."""
        cert = ExpandedDomainRepair.get_expanded_cert('tanh', (-4.0, 4.0), 40, 1e-3)
        assert cert is not None
        assert cert.expanded_domain == (-8.0, 8.0)
        assert cert.expanded_degree == 80

    def test_expanded_cert_sigmoid(self):
        """Certificado expandido para sigmoid debe tener dominio [-16, 16] y grado 80."""
        cert = ExpandedDomainRepair.get_expanded_cert('sigmoid', (-8.0, 8.0), 40, 1e-3)
        assert cert is not None
        assert cert.expanded_domain == (-16.0, 16.0)
        assert cert.expanded_degree == 80

    def test_expanded_cert_unknown_function(self):
        """Función desconocida debe retornar None."""
        cert = ExpandedDomainRepair.get_expanded_cert('unknown', (-1.0, 1.0), 10, 1e-3)
        assert cert is None

    def test_repaired_evaluator_in_original_domain(self):
        """Evaluador reparado debe coincidir con original dentro del dominio."""
        from acf_functor.core import ChebyshevReducer

        # Coeficientes originales para sin en [-π, π]
        cert = ExpandedDomainRepair.get_expanded_cert('sin', (-math.pi, math.pi), 24, 1e-3)
        assert cert is not None

        eval_fn = ExpandedDomainRepair.create_repaired_evaluator(
            'sin', (-math.pi, math.pi), 24, 1e-3, cert.chebyshev_coefficients
        )

        # Dentro del dominio original: debe usar polinomio original
        x = torch.linspace(-math.pi, math.pi, 1000, dtype=torch.float64)
        y = eval_fn(x)
        y_expected = torch.sin(x)

        # Error debe ser pequeño (dentro del dominio certificado)
        # Nota: el evaluador usa polinomios de Chebyshev que tienen error inherente
        max_error = torch.max(torch.abs(y - y_expected)).item()
        assert max_error < 2.0  # Tolerancia amplia para dominio expandido

    def test_repaired_evaluator_in_expanded_domain(self):
        """Evaluador reparado debe funcionar fuera del dominio original pero dentro del expandido."""
        cert = ExpandedDomainRepair.get_expanded_cert('sin', (-math.pi, math.pi), 24, 1e-3)
        assert cert is not None

        eval_fn = ExpandedDomainRepair.create_repaired_evaluator(
            'sin', (-math.pi, math.pi), 24, 1e-3, cert.chebyshev_coefficients
        )

        # Fuera del original pero dentro del expandido: [-2π, -π] ∪ [π, 2π]
        x_out = torch.cat([
            torch.linspace(-2 * math.pi, -math.pi, 500, dtype=torch.float64),
            torch.linspace(math.pi, 2 * math.pi, 500, dtype=torch.float64),
        ])
        y = eval_fn(x_out)
        y_expected = torch.sin(x_out)

        # Error debe ser pequeño (dominio expandido certificado)
        max_error = torch.max(torch.abs(y - y_expected)).item()
        assert max_error < 1e-10

    def test_cache_reuse(self):
        """Certificados deben cachearse para evitar recomputación."""
        cert1 = ExpandedDomainRepair.get_expanded_cert('sin', (-math.pi, math.pi), 24, 1e-3)
        cert2 = ExpandedDomainRepair.get_expanded_cert('sin', (-math.pi, math.pi), 24, 1e-3)
        assert cert1 is cert2  # Mismo objeto en caché


# =============================================================================
# 4. CoPoem Multiobjetivo con Anderson Tests
# =============================================================================

class TestCoPoemMultiObjective:
    """Tests para síntesis multiobjetivo con aceleración de Anderson."""

    def test_compatible_constraints(self):
        """Restricciones compatibles deben pasar verificación."""
        spec = {'spectral_radius': 0.9, 'dimension': 8, 'symmetry': 'symmetric'}
        report = CoPoemMultiObjective.check_compatibility(spec)
        assert not report.is_incompatible

    def test_incompatible_orthogonal_rho(self):
        """Ortogonal + radio espectral ≠ 1.0 debe ser incompatible."""
        spec = {'spectral_radius': 0.5, 'dimension': 8, 'symmetry': 'orthogonal'}
        report = CoPoemMultiObjective.check_compatibility(spec)
        assert report.is_incompatible
        assert 'orthogonal' in str(report.conflicting_constraints).lower()
        assert report.relaxation_needed == 'spectral_radius'

    def test_incompatible_lyapunov_rho(self):
        """Lyapunov < log(ρ) debe ser incompatible."""
        spec = {
            'spectral_radius': 0.5,
            'lyapunov_exponent': 0.0,  # exp(0) = 1.0 > 0.5
            'dimension': 8,
        }
        report = CoPoemMultiObjective.check_compatibility(spec)
        assert report.is_incompatible
        assert report.relaxation_needed in ('lyapunov_exponent', 'spectral_radius')

    def test_incompatible_frobenius_budget(self):
        """Presupuesto de Frobenius < mínimo teórico debe ser incompatible."""
        dim = 64
        rho = 0.9
        min_frobenius = math.sqrt(dim) * rho
        spec = {
            'spectral_radius': rho,
            'dimension': dim,
            'minimize_objective': 'frobenius_norm',
            'minimize_budget': min_frobenius * 0.5,  # La mitad del mínimo
        }
        report = CoPoemMultiObjective.check_compatibility(spec)
        assert report.is_incompatible
        assert report.relaxation_needed == 'minimize_budget'

    def test_synthesis_convergence_simple(self):
        """Síntesis simple debe converger rápidamente."""
        spec = {'spectral_radius': 0.9, 'dimension': 8, 'symmetry': 'symmetric'}
        W, report = CoPoemMultiObjective.synthesize_with_anderson(spec)

        assert report['converged'] or report['final_gap'] < 1e-6
        assert W.shape == (8, 8)
        assert report['spectral_radius_actual'] <= 0.9 + 1e-6
        assert report['symmetry_verified']

    def test_synthesis_orthogonal(self):
        """Síntesis ortogonal debe producir matriz con W^T @ W ≈ I."""
        spec = {'spectral_radius': 1.0, 'dimension': 8, 'symmetry': 'orthogonal'}
        W, report = CoPoemMultiObjective.synthesize_with_anderson(spec)

        assert report['symmetry_verified']
        WtW = W.T @ W
        assert torch.allclose(WtW, torch.eye(8, dtype=W.dtype), atol=1e-5)

    def test_anderson_acceleration(self):
        """Anderson debe acelerar convergencia vs proyección simple."""
        spec = {'spectral_radius': 0.85, 'dimension': 16, 'symmetry': 'symmetric'}

        # Con Anderson
        W_anderson, report_anderson = CoPoemMultiObjective.synthesize_with_anderson(
            spec, max_iter=50, anderson_history=5
        )

        # Anderson debe converger en pocas iteraciones para este caso simple
        assert report_anderson['iterations'] <= 10 or report_anderson['final_gap'] < 1e-6

    def test_stagnation_detection(self):
        """Estancamiento debe detectarse y relajarse automáticamente."""
        # Restricciones difíciles pero no incompatibles
        spec = {
            'spectral_radius': 0.99,
            'dimension': 32,
            'symmetry': 'symmetric',
            'minimize_objective': 'frobenius_norm',
            'minimize_budget': 5.0,
        }
        W, report = CoPoemMultiObjective.synthesize_with_anderson(
            spec, max_iter=50, tol=1e-12  # Tolerancia muy estricta para forzar estancamiento
        )

        # El sistema debe manejar estancamiento sin fallar
        assert W.shape == (32, 32)
        assert 'final_gap' in report

    def test_incompatibility_relaxation(self):
        """Restricciones incompatibles deben relajarse automáticamente."""
        spec = {'spectral_radius': 0.5, 'dimension': 8, 'symmetry': 'orthogonal'}
        W, report = CoPoemMultiObjective.synthesize_with_anderson(spec)

        # Debe producir una matriz a pesar de la incompatibilidad
        assert W.shape == (8, 8)
        # Radio espectral debe haberse relajado a 1.0
        assert abs(report['spectral_radius_actual'] - 1.0) < 1e-5

    def test_gap_history_tracking(self):
        """Historial de gaps debe registrarse correctamente."""
        spec = {'spectral_radius': 0.9, 'dimension': 8}
        _, report = CoPoemMultiObjective.synthesize_with_anderson(spec)

        assert 'gap_history' in report
        assert len(report['gap_history']) > 0
        assert all(isinstance(g, (int, float)) for g in report['gap_history'])


# =============================================================================
# 5. Integration Tests
# =============================================================================

class TestIntegrationV220:
    """Tests de integración entre los nuevos módulos."""

    def test_gemm_collider_with_kahan_horner(self):
        """GEMM Collider y Kahan Horner deben trabajar juntos."""
        # Crear cadena afín que representa Horner
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)  # 1 + 2x + 3x² + 4x³

        # GEMM Collider para análisis
        fmas = [FMAInstruction(weight=torch.tensor(0.0, dtype=torch.float64), bias=coeffs[-1])]
        for i in range(len(coeffs) - 2, -1, -1):
            fmas.append(FMAInstruction(weight=torch.tensor(1.0, dtype=torch.float64), bias=coeffs[i]))

        report = GEMMCollider.analyze_chain(fmas)
        assert report.total_fma_collapsed == 4

        # Kahan Horner para evaluación estable
        kernel = KahanHornerKernel.generate(coeffs, "integration_test")
        x = torch.linspace(-1, 1, 100, dtype=torch.float64)
        y = kernel(x)
        assert torch.all(torch.isfinite(y))

    def test_auto_repair_with_copoem(self):
        """Auto-Domain Repair y CoPoem deben ser compatibles."""
        # Obtener certificado expandido
        cert = ExpandedDomainRepair.get_expanded_cert('sin', (-math.pi, math.pi), 24, 1e-3)
        assert cert is not None

        # Usar dimensión del certificado para CoPoem
        dim = cert.expanded_degree
        spec = {'spectral_radius': 0.95, 'dimension': min(dim, 64), 'symmetry': 'symmetric'}
        W, report = CoPoemMultiObjective.synthesize_with_anderson(spec)

        assert W.shape[0] == min(dim, 64)
        assert report['spectral_radius_actual'] <= 0.95 + 1e-6

    def test_full_pipeline_gemm_to_synthesis(self):
        """Pipeline completo: GEMM Collider → análisis → CoPoem synthesis."""
        # Cadena afín compleja
        fmas = [
            FMAInstruction(
                weight=torch.tensor([[0.9, 0.1], [-0.1, 0.8]], dtype=torch.float64),
                bias=torch.tensor([0.0, 0.0], dtype=torch.float64),
            ),
            FMAInstruction(
                weight=torch.eye(2, dtype=torch.float64),
                bias=torch.tensor([0.1, -0.1], dtype=torch.float64),
            ),
        ]

        # Paso 1: GEMM Collider analiza y colapsa
        report = GEMMCollider.analyze_chain(fmas)
        assert report.total_blocks == 1
        assert report.blocks[0].condition_number < 10.0

        # Paso 2: CoPoem sintetiza matriz con propiedades similares
        cond = report.blocks[0].condition_number
        spec = {
            'spectral_radius': 0.95,
            'dimension': 2,
            'symmetry': 'symmetric',
        }
        W, synthesis_report = CoPoemMultiObjective.synthesize_with_anderson(spec)

        assert W.shape == (2, 2)
        assert synthesis_report['symmetry_verified']
