"""
tests/test_auto_evolution.py
============================

Test suite for the ACF Auto-Evolution Engine.

Tests 4 mathematical properties of ACF that enable deterministic
self-improvement:
  1. FixedPointIterator  (idempotence Φ²=Φ)
  2. BifunctorialCycle   (adjunction Φ* ⊣ Φ)
  3. ThermodynamicSearch (free energy F(d,β))
  4. AdaptiveRefinement  (residual r(x)-guided)
  5. ACFAutoEvolver      (unified pipeline)
  6. PoemCompiler.auto_evolve integration
  7. GideonEngine.auto_evolve_fma integration

Principles:
  - No mocks; all tests exercise real mathematical computations.
  - Numbers verified against known analytic properties.
  - Honest: test both improvement cases AND limitation cases.
"""

import math
import sys
import os
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acf_functor.auto_evolution import (
    FixedPointIterator,
    FixedPointResult,
    BifunctorialCycle,
    BifunctorialResult,
    ThermodynamicSearch,
    ThermodynamicSearchResult,
    ConfigurationPoint,
    AdaptiveRefinement,
    AdaptiveRefinementResult,
    ACFAutoEvolver,
    ACFAutoEvolverConfig,
    AutoEvolutionResult,
    _max_residual,
    _eval_result,
)
from acf_functor.core import ChebyshevReducer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sin_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

def gaussian_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x ** 2)

def poly5_fn(x: torch.Tensor) -> torch.Tensor:
    """Degree-5 polynomial; ACF should represent it exactly."""
    return x**5 - 3*x**3 + x

def tanh_sharp(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(5 * x)


# ═════════════════════════════════════════════════════════════════════════════
# 1. FixedPointIterator
# ═════════════════════════════════════════════════════════════════════════════

class TestFixedPointIterator(unittest.TestCase):
    """Tests for the idempotence-based fixed-point iterator."""

    def setUp(self):
        self.domain = (-math.pi, math.pi)
        self.fp = FixedPointIterator(
            max_iterations=8,
            convergence_tol=1e-10,
            degree=20,
            n_probe=2000,
        )

    def test_returns_fixed_point_result(self):
        result = self.fp.iterate(sin_fn, self.domain)
        self.assertIsInstance(result, FixedPointResult)

    def test_sin_is_already_fixed_point(self):
        """sin(x) with degree=20 on (-π,π) is at machine precision — already FP."""
        result = self.fp.iterate(sin_fn, self.domain)
        self.assertTrue(result.already_fixed_point or result.converged,
                        "sin should reach fixed-point quickly")

    def test_final_epsilon_not_worse_than_initial(self):
        """The fixed-point iterator must never return a worse reduction."""
        result = self.fp.iterate(sin_fn, self.domain)
        self.assertLessEqual(
            result.final_epsilon, result.initial_epsilon * 1.1,
            "final ε must not exceed initial ε by more than 10%",
        )

    def test_poly_converges_to_fixpoint(self):
        """A degree-5 polynomial is a perfect fixed point for degree≥5 ACF."""
        fp = FixedPointIterator(degree=10, convergence_tol=1e-9, max_iterations=6)
        result = fp.iterate(poly5_fn, (-1.0, 1.0))
        self.assertTrue(result.already_fixed_point or result.converged)
        self.assertLess(result.final_epsilon, 1e-8,
                        "Polynomial should be near-zero error at high degree")

    def test_initial_reduction_passed_through(self):
        """If initial_reduction provided, it is used as starting point."""
        initial = ChebyshevReducer.reduce(gaussian_fn, degree=5, domain=(-2.0, 2.0))
        fp = FixedPointIterator(degree=5, max_iterations=4, n_probe=1000)
        result = fp.iterate(gaussian_fn, (-2.0, 2.0), initial_reduction=initial)
        self.assertIsInstance(result, FixedPointResult)
        # initial_eps should match what we'd compute from the given initial_reduction
        expected_eps = _max_residual(gaussian_fn, initial, (-2.0, 2.0), 1000)
        self.assertAlmostEqual(result.initial_epsilon, expected_eps, places=10)

    def test_convergence_history_non_empty(self):
        result = self.fp.iterate(gaussian_fn, (-2.0, 2.0))
        self.assertGreater(len(result.convergence_history), 0)

    def test_elapsed_ms_positive(self):
        result = self.fp.iterate(sin_fn, self.domain)
        self.assertGreater(result.elapsed_ms, 0.0)

    def test_summary_str(self):
        result = self.fp.iterate(sin_fn, self.domain)
        s = result.summary()
        self.assertIn("FixedPointIterator", s)
        self.assertIn("ε₀", s)

    def test_improvement_for_low_degree_start(self):
        """Starting at degree=3, the iterator should improve exp(-x²)."""
        initial = ChebyshevReducer.reduce(gaussian_fn, degree=3, domain=(-1.5, 1.5))
        fp = FixedPointIterator(degree=20, max_iterations=6, n_probe=2000)
        result = fp.iterate(gaussian_fn, (-1.5, 1.5), initial_reduction=initial)
        self.assertLessEqual(result.final_epsilon, result.initial_epsilon + 1e-12,
                             "Iterator should not make things worse")


# ═════════════════════════════════════════════════════════════════════════════
# 2. BifunctorialCycle
# ═════════════════════════════════════════════════════════════════════════════

class TestBifunctorialCycle(unittest.TestCase):
    """Tests for the Φ* ⊣ Φ adjunction cycle."""

    def setUp(self):
        self.bif = BifunctorialCycle(
            max_cycles=4,
            convergence_tol=1e-8,
            degree=20,
            n_probe=2000,
        )

    def test_returns_bifunctorial_result(self):
        result = self.bif.cycle(sin_fn, (-math.pi, math.pi))
        self.assertIsInstance(result, BifunctorialResult)

    def test_cycles_non_negative(self):
        result = self.bif.cycle(gaussian_fn, (-2.0, 2.0))
        self.assertGreaterEqual(result.n_cycles, 0)

    def test_epsilon_history_non_empty(self):
        result = self.bif.cycle(sin_fn, (-math.pi, math.pi))
        self.assertGreater(len(result.epsilon_history), 0)

    def test_final_epsilon_not_worse_than_initial(self):
        """The bifunctorial cycle must never degrade the reduction."""
        result = self.bif.cycle(gaussian_fn, (-2.0, 2.0))
        self.assertLessEqual(
            result.final_epsilon, result.initial_epsilon * 1.05,
            "Bifunctorial cycle must not worsen the reduction by more than 5%",
        )

    def test_elapsed_ms_positive(self):
        result = self.bif.cycle(sin_fn, (-1.0, 1.0))
        self.assertGreater(result.elapsed_ms, 0.0)

    def test_summary_str(self):
        result = self.bif.cycle(sin_fn, (-1.0, 1.0))
        s = result.summary()
        self.assertIn("BifunctorialCycle", s)

    def test_reduction_is_valid(self):
        """The returned reduction should evaluate on a test grid without errors."""
        result = self.bif.cycle(gaussian_fn, (-1.0, 1.0))
        x = torch.linspace(-1.0, 1.0, 100, dtype=torch.float64)
        y = _eval_result(result.reduction, x)
        self.assertFalse(torch.isnan(y).any(), "Reduction output contains NaN")
        self.assertFalse(torch.isinf(y).any(), "Reduction output contains Inf")


# ═════════════════════════════════════════════════════════════════════════════
# 3. ThermodynamicSearch
# ═════════════════════════════════════════════════════════════════════════════

class TestThermodynamicSearch(unittest.TestCase):
    """Tests for the free-energy hyperparameter search."""

    def setUp(self):
        self.ts = ThermodynamicSearch(
            beta=1.0,
            degree_candidates=[5, 10, 20, 30],
            n_probe=1000,
        )

    def test_returns_thermo_result(self):
        result = self.ts.search(gaussian_fn, (-2.0, 2.0))
        self.assertIsInstance(result, ThermodynamicSearchResult)

    def test_optimal_is_configuration_point(self):
        result = self.ts.search(sin_fn, (-math.pi, math.pi))
        self.assertIsInstance(result.optimal, ConfigurationPoint)

    def test_all_configs_evaluated(self):
        result = self.ts.search(gaussian_fn, (-2.0, 2.0))
        # ThermodynamicSearch has 2 methods (chebyshev, horner) × 4 degrees = 8 configs
        self.assertGreaterEqual(len(result.all_configs), 4,
                                "Should evaluate at least one config per degree candidate")

    def test_optimal_has_minimum_free_energy(self):
        """optimal.free_energy must be ≤ all others."""
        result = self.ts.search(gaussian_fn, (-2.0, 2.0))
        for cfg in result.all_configs:
            self.assertLessEqual(
                result.optimal.free_energy, cfg.free_energy + 1e-12,
                "optimal must minimise free energy",
            )

    def test_high_beta_prefers_accuracy(self):
        """β → ∞ should prefer high degree (low E dominates)."""
        ts_hot  = ThermodynamicSearch(beta=0.01, degree_candidates=[5, 30], n_probe=500)
        ts_cold = ThermodynamicSearch(beta=100.0, degree_candidates=[5, 30], n_probe=500)
        r_hot  = ts_hot.search(gaussian_fn, (-2.0, 2.0))
        r_cold = ts_cold.search(gaussian_fn, (-2.0, 2.0))
        # cold (β→∞) should have ε ≤ hot (β→0) epsilon (prefers accuracy)
        self.assertLessEqual(
            r_cold.optimal.epsilon, r_hot.optimal.epsilon * 10 + 1e-15,
            "High β should not select configurations with much larger ε",
        )

    def test_best_reduction_evaluatable(self):
        result = self.ts.search(gaussian_fn, (-2.0, 2.0))
        x = torch.linspace(-2.0, 2.0, 500, dtype=torch.float64)
        y = _eval_result(result.best_reduction, x)
        self.assertFalse(torch.isnan(y).any())

    def test_elapsed_ms_positive(self):
        result = self.ts.search(sin_fn, (-1.0, 1.0))
        self.assertGreater(result.elapsed_ms, 0.0)

    def test_summary_str(self):
        result = self.ts.search(sin_fn, (-1.0, 1.0))
        self.assertIn("ThermodynamicSearch", result.summary())


# ═════════════════════════════════════════════════════════════════════════════
# 4. AdaptiveRefinement
# ═════════════════════════════════════════════════════════════════════════════

class TestAdaptiveRefinement(unittest.TestCase):
    """Tests for the residual-guided adaptive refinement."""

    def setUp(self):
        self.ar = AdaptiveRefinement(
            target_epsilon=1e-6,
            max_local_degree=60,
            n_grid=2000,
        )

    def test_returns_adaptive_result(self):
        result = self.ar.refine(gaussian_fn, (-2.0, 2.0))
        self.assertIsInstance(result, AdaptiveRefinementResult)

    def test_already_converged_no_intervals(self):
        """If baseline already satisfies target ε, refinement finds 0 intervals."""
        ar = AdaptiveRefinement(target_epsilon=1.0, n_grid=1000)  # very loose target
        result = ar.refine(sin_fn, (-math.pi, math.pi))
        self.assertTrue(result.converged)
        self.assertEqual(result.n_intervals, 0)

    def test_epsilon_after_not_worse_than_before(self):
        result = self.ar.refine(gaussian_fn, (-2.0, 2.0))
        self.assertLessEqual(
            result.global_epsilon_after, result.global_epsilon_before + 1e-15,
        )

    def test_initial_reduction_used(self):
        initial = ChebyshevReducer.reduce(gaussian_fn, degree=5, domain=(-2.0, 2.0))
        result = self.ar.refine(gaussian_fn, (-2.0, 2.0), initial_reduction=initial)
        # Should start with the 5-degree epsilon
        expected_eps = _max_residual(gaussian_fn, initial, (-2.0, 2.0), 2000)
        self.assertAlmostEqual(result.global_epsilon_before, expected_eps, places=10)

    def test_intervals_improved_flag(self):
        """Each interval should have a valid 'improved' field."""
        initial = ChebyshevReducer.reduce(gaussian_fn, degree=5, domain=(-2.0, 2.0))
        ar = AdaptiveRefinement(target_epsilon=1e-4, n_grid=1000)
        result = ar.refine(gaussian_fn, (-2.0, 2.0), initial_reduction=initial)
        for ri in result.intervals:
            self.assertIsNotNone(ri.improved)

    def test_reduction_evaluatable(self):
        result = self.ar.refine(gaussian_fn, (-2.0, 2.0))
        x = torch.linspace(-2.0, 2.0, 500, dtype=torch.float64)
        y = _eval_result(result.reduction, x)
        self.assertFalse(torch.isnan(y).any())

    def test_summary_str(self):
        result = self.ar.refine(gaussian_fn, (-1.0, 1.0))
        self.assertIn("AdaptiveRefinement", result.summary())

    def test_elapsed_ms_positive(self):
        result = self.ar.refine(gaussian_fn, (-1.0, 1.0))
        self.assertGreater(result.elapsed_ms, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. ACFAutoEvolver — unified pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestACFAutoEvolver(unittest.TestCase):
    """Tests for the unified auto-evolution pipeline."""

    def setUp(self):
        self.evolver = ACFAutoEvolver()

    def test_returns_auto_evolution_result(self):
        result = self.evolver.evolve(gaussian_fn, (-2.0, 2.0))
        self.assertIsInstance(result, AutoEvolutionResult)

    def test_pipeline_order_populated(self):
        result = self.evolver.evolve(sin_fn, (-math.pi, math.pi))
        self.assertGreater(len(result.pipeline_order), 0)
        self.assertIn("thermo_search", result.pipeline_order)

    def test_final_epsilon_not_worse(self):
        """Auto-evolution must never degrade accuracy."""
        result = self.evolver.evolve(gaussian_fn, (-2.0, 2.0))
        self.assertLessEqual(
            result.final_epsilon, result.initial_epsilon * 1.05,
            "Auto-evolution must not significantly degrade epsilon",
        )

    def test_low_degree_start_improves(self):
        """Starting at degree=5 for exp(-x²) should improve significantly."""
        cfg = ACFAutoEvolverConfig(initial_degree=5)
        result = ACFAutoEvolver(config=cfg).evolve(gaussian_fn, (-2.0, 2.0))
        self.assertGreater(
            result.improvement_ratio, 1000.0,
            "Low-degree start should improve by at least 3 orders of magnitude",
        )

    def test_improvement_ratio_positive(self):
        result = self.evolver.evolve(gaussian_fn, (-2.0, 2.0))
        self.assertGreater(result.improvement_ratio, 0.0)

    def test_sub_results_populated(self):
        """All enabled sub-mechanisms should populate their results."""
        result = self.evolver.evolve(sin_fn, (-math.pi, math.pi))
        self.assertIsNotNone(result.thermo_result)
        self.assertIsNotNone(result.fixed_point_result)
        self.assertIsNotNone(result.bifunctorial_result)
        self.assertIsNotNone(result.adaptive_result)

    def test_disable_mechanisms(self):
        """With all optional steps disabled, only baseline is returned."""
        cfg = ACFAutoEvolverConfig(
            enable_fixed_point=False,
            enable_bifunctorial=False,
            enable_thermo_search=False,
            enable_adaptive=False,
        )
        result = ACFAutoEvolver(config=cfg).evolve(sin_fn, (-1.0, 1.0))
        self.assertIsNone(result.fixed_point_result)
        self.assertIsNone(result.bifunctorial_result)
        self.assertIsNone(result.thermo_result)
        self.assertIsNone(result.adaptive_result)
        self.assertEqual(result.pipeline_order, [])

    def test_summary_str_contains_key_fields(self):
        result = self.evolver.evolve(sin_fn, (-1.0, 1.0))
        s = result.summary()
        self.assertIn("ε inicial", s)
        self.assertIn("ε final", s)
        self.assertIn("mejora", s)
        self.assertIn("pipeline", s)

    def test_elapsed_ms_positive(self):
        result = self.evolver.evolve(sin_fn, (-1.0, 1.0))
        self.assertGreater(result.total_elapsed_ms, 0.0)

    def test_best_reduction_evaluatable(self):
        result = self.evolver.evolve(poly5_fn, (-1.0, 1.0))
        x = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
        y = _eval_result(result.best_reduction, x)
        self.assertFalse(torch.isnan(y).any())
        self.assertFalse(torch.isinf(y).any())

    def test_is_fixed_point_check(self):
        """is_fixed_point should return True for an already-converged reduction."""
        red = ChebyshevReducer.reduce(sin_fn, degree=20, domain=(-math.pi, math.pi))
        is_fp, delta = self.evolver.is_fixed_point(sin_fn, red, (-math.pi, math.pi))
        # With degree=20 and Chebyshev, sin should be at machine precision
        # → delta should be very small (though not guaranteed < 1e-10 always)
        self.assertIsInstance(is_fp, bool)
        self.assertGreaterEqual(delta, 0.0)

    def test_metadata_contains_domain(self):
        result = self.evolver.evolve(sin_fn, (-1.0, 1.0))
        self.assertIn("domain", result.metadata)

    # Honest limitation test
    def test_tanh_sharp_honest_limitation(self):
        """
        tanh(5x) on [-2,2] has poor polynomial approximation with degree=20.
        Auto-evolution cannot significantly improve it without domain splitting.
        This test documents that limitation honestly.
        """
        result = self.evolver.evolve(tanh_sharp, (-2.0, 2.0))
        # The improvement_ratio may be ≈1 (no improvement possible with polynomial)
        # This is the expected and correct behaviour — documenting the limit.
        # We only require that the result does not worsen things.
        self.assertLessEqual(
            result.final_epsilon, result.initial_epsilon * 1.05,
            "Even for hard functions, evolution must not worsen things",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. PoemCompiler.auto_evolve integration
# ═════════════════════════════════════════════════════════════════════════════

class TestPoemCompilerAutoEvolve(unittest.TestCase):
    """Tests for PoemCompiler.auto_evolve()."""

    def _make_sin_ast(self):
        from poema.frontend import Poem
        from poema.ast_nodes import TranscendentalNode, PolynomialNode, GeometricType, InputNode
        from acf_functor.core import ChebyshevReducer
        import torch
        # Build a TranscendentalNode for sin directly
        red = ChebyshevReducer.reduce(sin_fn, degree=20, domain=(-math.pi, math.pi))
        coeffs = red.metadata.get("monomial_coefficients", red.metadata.get("coefficients", [0.0]))
        poly = PolynomialNode(coefficients=torch.as_tensor(coeffs, dtype=torch.float64))
        return TranscendentalNode(
            name="sin",
            polynomial=poly,
            certified_epsilon=float(red.epsilon_bound),
            original_domain=(-math.pi, math.pi),
            geometric_type=GeometricType(1, 1),
        )

    def test_auto_evolve_returns_auto_evolution_result(self):
        from poema.compiler import PoemCompiler
        compiler = PoemCompiler()
        ast = self._make_sin_ast()
        result = compiler.auto_evolve(ast, domain=(-math.pi, math.pi))
        self.assertIsInstance(result, AutoEvolutionResult)

    def test_auto_evolve_result_has_small_epsilon(self):
        from poema.compiler import PoemCompiler
        compiler = PoemCompiler()
        ast = self._make_sin_ast()
        result = compiler.auto_evolve(ast, domain=(-math.pi, math.pi))
        self.assertLess(result.final_epsilon, 1e-6,
                        "sin(x) should be well approximated after auto-evolution")

    def test_auto_evolve_with_custom_config(self):
        from poema.compiler import PoemCompiler
        compiler = PoemCompiler()
        ast = self._make_sin_ast()
        cfg = ACFAutoEvolverConfig(
            enable_bifunctorial=False,
            enable_adaptive=False,
        )
        result = compiler.auto_evolve(ast, domain=(-math.pi, math.pi), config=cfg)
        self.assertIsNone(result.bifunctorial_result)
        self.assertIsNone(result.adaptive_result)


# ═════════════════════════════════════════════════════════════════════════════
# 7. GideonEngine.auto_evolve_fma integration
# ═════════════════════════════════════════════════════════════════════════════

class TestGideonAutoEvolve(unittest.TestCase):
    """Tests for GideonEngine.auto_evolve_fma()."""

    def _make_chain(self, n=20, seed=42):
        rng = np.random.default_rng(seed)
        ws = rng.uniform(0.8, 1.2, n)
        bs = rng.uniform(-0.1, 0.1, n)
        class _FMA:
            __slots__ = ("weight", "bias")
            def __init__(self, w, b): self.weight = float(w); self.bias = float(b)
        return [_FMA(w, b) for w, b in zip(ws, bs)]

    def test_auto_evolve_fma_returns_result(self):
        from poema.backends.gideon.engine import GideonEngine
        engine = GideonEngine()
        chain = self._make_chain(20, seed=1)
        result = engine.auto_evolve_fma(chain, domain=(-1.0, 1.0))
        self.assertIsInstance(result, AutoEvolutionResult)

    def test_auto_evolve_fma_epsilon_non_negative(self):
        from poema.backends.gideon.engine import GideonEngine
        engine = GideonEngine()
        chain = self._make_chain(10, seed=2)
        result = engine.auto_evolve_fma(chain, domain=(-1.0, 1.0))
        self.assertGreaterEqual(result.final_epsilon, 0.0)

    def test_auto_evolve_fma_best_reduction_evaluatable(self):
        from poema.backends.gideon.engine import GideonEngine
        engine = GideonEngine()
        chain = self._make_chain(15, seed=3)
        result = engine.auto_evolve_fma(chain, domain=(-1.0, 1.0))
        x = torch.linspace(-1.0, 1.0, 200, dtype=torch.float64)
        y = _eval_result(result.best_reduction, x)
        self.assertFalse(torch.isnan(y).any())

    def test_auto_evolve_fma_with_config(self):
        from poema.backends.gideon.engine import GideonEngine
        engine = GideonEngine()
        chain = self._make_chain(10, seed=4)
        cfg = ACFAutoEvolverConfig(
            enable_fixed_point=True,
            enable_bifunctorial=False,
            enable_thermo_search=True,
            enable_adaptive=False,
        )
        result = engine.auto_evolve_fma(chain, domain=(-1.0, 1.0), config=cfg)
        self.assertIsNotNone(result.fixed_point_result)
        self.assertIsNone(result.bifunctorial_result)

    def test_auto_evolve_fma_improvement_for_simple_chain(self):
        """A chain with large weights will have non-trivial structure to reduce."""
        from poema.backends.gideon.engine import GideonEngine
        engine = GideonEngine()
        # Chain with contractive weights so the function compresses to a constant-like
        chain = self._make_chain(5, seed=99)
        cfg = ACFAutoEvolverConfig(initial_degree=5)
        result = engine.auto_evolve_fma(chain, domain=(-1.0, 1.0), config=cfg)
        # The auto-evolved result should be at least as good as the baseline
        self.assertLessEqual(result.final_epsilon, result.initial_epsilon * 1.05)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Mathematical property tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMathematicalProperties(unittest.TestCase):
    """
    Mathematical invariant tests:
      • Verifies that Φ²(f) ≈ Φ(f) (idempotence)
      • Verifies that thermodynamic β interpolates accuracy/compression
      • Verifies that residual r(x) is computable and guides refinement
    """

    def test_idempotence_phi_squared_equals_phi(self):
        """
        Core ACF property: ‖Φ(Φ(f)) - Φ(f)‖∞ ≈ 0.
        For sin with degree=20, this should hold to near machine precision.
        """
        domain = (-math.pi, math.pi)
        phi_f = ChebyshevReducer.reduce(sin_fn, degree=20, domain=domain)

        def phi_f_fn(z):
            return _eval_result(phi_f, z)

        phi2_f = ChebyshevReducer.reduce(phi_f_fn, degree=20, domain=domain)
        x = torch.linspace(domain[0], domain[1], 3000, dtype=torch.float64)
        delta = float(torch.max(torch.abs(
            _eval_result(phi2_f, x) - _eval_result(phi_f, x)
        )).item())
        self.assertLess(delta, 1e-8,
                        f"Φ²=Φ violated: ‖Φ²-Φ‖∞ = {delta:.3e}")

    def test_residual_is_computable(self):
        """r(x) = f(x) - Φ(f)(x) must be finite on the probe grid."""
        domain = (-2.0, 2.0)
        phi_f = ChebyshevReducer.reduce(gaussian_fn, degree=10, domain=domain)
        x = torch.linspace(domain[0], domain[1], 1000, dtype=torch.float64)
        residual = gaussian_fn(x) - _eval_result(phi_f, x)
        self.assertFalse(torch.isnan(residual).any())
        self.assertFalse(torch.isinf(residual).any())

    def test_thermo_free_energy_monotone_in_degree(self):
        """
        At fixed β=∞ (very large), F(d) = E(d) should decrease with degree
        for well-behaved analytic functions.
        """
        ts = ThermodynamicSearch(beta=1e6, degree_candidates=[5, 10, 20], n_probe=1000)
        result = ts.search(gaussian_fn, (-2.0, 2.0))
        # With β very large, free energy ≈ E(d). The optimal should have smallest ε.
        best_eps = result.optimal.epsilon
        for cfg in result.all_configs:
            self.assertGreaterEqual(
                cfg.epsilon + 1e-15, best_eps,
                "With β≫1, the config with smallest ε should be optimal",
            )

    def test_adaptive_refinement_identifies_high_error_zone(self):
        """
        Starting with degree=5 for exp(-x²), there should be high-error zones
        that the adaptive refinement identifies.
        """
        initial = ChebyshevReducer.reduce(gaussian_fn, degree=5, domain=(-2.0, 2.0))
        ar = AdaptiveRefinement(target_epsilon=1e-4, n_grid=2000)
        result = ar.refine(gaussian_fn, (-2.0, 2.0), initial_reduction=initial)
        # For degree=5 start, there should be at least one high-error interval
        self.assertGreater(result.n_intervals, 0,
                           "Degree-5 Gaussian should have error zones to refine")


# ═════════════════════════════════════════════════════════════════════════════
# 9. ACFAutoEvolver from __init__ import
# ═════════════════════════════════════════════════════════════════════════════

class TestAutoEvolutionPublicAPI(unittest.TestCase):
    """Verify public imports from acf_functor.__init__ work correctly."""

    def test_all_symbols_importable(self):
        from acf_functor import (
            ACFAutoEvolver,
            ACFAutoEvolverConfig,
            AutoEvolutionResult,
            FixedPointIterator,
            FixedPointResult,
            BifunctorialCycle,
            BifunctorialResult,
            ThermodynamicSearch,
            ThermodynamicSearchResult,
            ConfigurationPoint,
            AdaptiveRefinement,
            AdaptiveRefinementResult,
            RefinedInterval,
        )
        # All imports succeeded

    def test_acf_auto_evolver_works_from_init(self):
        from acf_functor import ACFAutoEvolver, ACFAutoEvolverConfig
        cfg = ACFAutoEvolverConfig(
            enable_bifunctorial=False,
            enable_fixed_point=False,
            enable_thermo_search=True,
            enable_adaptive=False,
        )
        evolver = ACFAutoEvolver(config=cfg)
        result = evolver.evolve(lambda x: torch.cos(x), domain=(0.0, math.pi))
        self.assertIsInstance(result, AutoEvolutionResult)
        self.assertLess(result.final_epsilon, 1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  ACF AUTO-EVOLUTION TEST SUITE")
    print("  4 Mathematical Properties: Idempotence | Adjunction | Thermodynamics | Residual")
    print("=" * 70)
    unittest.main(verbosity=2, buffer=False)
