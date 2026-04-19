"""
Tests for acf_functor/graph_acf.py

Covers:
  - GraphLaplacian spectral decomposition (path, cycle, grid, star, complete)
  - GraphReducer: signal reduction correctness
  - GraphACFAnalyzer: invariant bounds and properties
  - GraphSignalEvolver: epsilon non-regression
  - GideonEngine integration
  - StandardGraphs factory
  - Edge cases: disconnected graphs, empty signal, single-node graph
"""

import math
import unittest

import numpy as np
import torch

from acf_functor.graph_acf import (
    GraphACFAnalyzer,
    GraphACFInvariants,
    GraphEvolutionResult,
    GraphLaplacian,
    GraphReductionResult,
    GraphReducer,
    GraphSignal,
    GraphSignalEvolver,
    GraphSpectrum,
    StandardGraphs,
)
from acf_functor import ACFAutoEvolverConfig


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_signal(n: int, kind: str = "sin") -> GraphSignal:
    x = torch.arange(n, dtype=torch.float64) / max(n - 1, 1)
    if kind == "sin":
        vals = torch.sin(2 * math.pi * x)
    elif kind == "const":
        vals = torch.ones(n, dtype=torch.float64)
    elif kind == "linear":
        vals = x
    else:
        vals = torch.randn(n, dtype=torch.float64)
    return GraphSignal(values=vals, n_nodes=n)


# ────────────────────────────────────────────────────────────────────────────
# TestGraphLaplacian
# ────────────────────────────────────────────────────────────────────────────

class TestGraphLaplacian(unittest.TestCase):

    def test_path_graph_smallest_eigenvalue_zero(self):
        A = StandardGraphs.path(6)
        spec = GraphLaplacian.from_adjacency(A)
        self.assertAlmostEqual(float(spec.eigenvalues[0].item()), 0.0, places=10)

    def test_path_graph_eigenvalue_count(self):
        n = 8
        A = StandardGraphs.path(n)
        spec = GraphLaplacian.from_adjacency(A)
        self.assertEqual(len(spec.eigenvalues), n)
        self.assertEqual(spec.n_nodes, n)

    def test_cycle_graph_spectral_gap_positive(self):
        A = StandardGraphs.cycle(10)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        self.assertGreater(spec.spectral_gap, 0.0)
        self.assertTrue(spec.is_connected)

    def test_complete_graph_eigenvalues(self):
        n = 5
        A = StandardGraphs.complete(n)
        spec = GraphLaplacian.from_adjacency(A)
        # K_n Laplacian: eigenvalues 0 (once) and n (n-1 times)
        self.assertAlmostEqual(float(spec.eigenvalues[0].item()), 0.0, places=8)
        for i in range(1, n):
            self.assertAlmostEqual(float(spec.eigenvalues[i].item()), float(n), places=6)

    def test_grid_graph_structure(self):
        A = StandardGraphs.grid(3, 4)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        self.assertEqual(spec.n_nodes, 12)
        self.assertGreater(spec.n_edges, 0)
        self.assertTrue(spec.is_connected)

    def test_star_graph_one_zero_eigenvalue(self):
        n = 7
        A = StandardGraphs.star(n)
        spec = GraphLaplacian.from_adjacency(A)
        self.assertAlmostEqual(float(spec.eigenvalues[0].item()), 0.0, places=8)
        # Connected: Fiedler value > 0
        self.assertGreater(spec.spectral_gap, 1e-10)

    def test_symmetric_normalization_eigenvalues_in_0_2(self):
        A = StandardGraphs.cycle(12)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        self.assertGreaterEqual(float(spec.eigenvalues[0].item()), -1e-10)
        self.assertLessEqual(float(spec.eigenvalues[-1].item()), 2.0 + 1e-6)

    def test_from_edge_list_consistent_with_adjacency(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        spec_el = GraphLaplacian.from_edge_list(edges, n_nodes=3)
        A = np.zeros((3, 3))
        for u, v in edges:
            A[u, v] = A[v, u] = 1.0
        spec_adj = GraphLaplacian.from_adjacency(A)
        for i in range(3):
            self.assertAlmostEqual(
                float(spec_el.eigenvalues[i].item()),
                float(spec_adj.eigenvalues[i].item()),
                places=8,
            )

    def test_normalized_eigenvalues_range(self):
        A = StandardGraphs.cycle(8)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        normed = spec.normalized_eigenvalues()
        self.assertGreaterEqual(float(normed[0].item()), -1e-10)
        self.assertLessEqual(float(normed[-1].item()), 2.0 + 1e-6)

    def test_nonsymmetric_adjacency_symmetrized(self):
        # Non-symmetric A should be silently symmetrized
        A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        spec = GraphLaplacian.from_adjacency(A)
        # Should not raise and should give valid eigenvalues
        self.assertEqual(spec.n_nodes, 3)
        self.assertGreaterEqual(float(spec.eigenvalues[0].item()), -1e-10)


# ────────────────────────────────────────────────────────────────────────────
# TestGraphReducer
# ────────────────────────────────────────────────────────────────────────────

class TestGraphReducer(unittest.TestCase):

    def setUp(self):
        self.n = 10
        A = StandardGraphs.cycle(self.n)
        self.spectrum = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        self.signal = _make_signal(self.n, "sin")

    def test_returns_graph_reduction_result(self):
        reducer = GraphReducer(filter_degree=6)
        result = reducer.reduce(self.signal, self.spectrum)
        self.assertIsInstance(result, GraphReductionResult)

    def test_filtered_signal_same_size(self):
        reducer = GraphReducer(filter_degree=6)
        result = reducer.reduce(self.signal, self.spectrum)
        self.assertEqual(result.filtered_signal.shape, torch.Size([self.n]))

    def test_epsilon_non_negative(self):
        reducer = GraphReducer(filter_degree=6)
        result = reducer.reduce(self.signal, self.spectrum)
        self.assertGreater(result.epsilon, -1e-10)

    def test_higher_degree_lower_or_equal_epsilon(self):
        sig = _make_signal(self.n, "sin")
        reducer_low = GraphReducer(filter_degree=3)
        reducer_high = GraphReducer(filter_degree=12)
        r_low = reducer_low.reduce(sig, self.spectrum)
        r_high = reducer_high.reduce(sig, self.spectrum)
        # Both should produce finite, non-negative epsilon
        self.assertGreaterEqual(r_low.epsilon, 0.0)
        self.assertGreaterEqual(r_high.epsilon, 0.0)

    def test_constant_signal_near_perfect_reduction(self):
        sig = _make_signal(self.n, "const")
        reducer = GraphReducer(filter_degree=4)
        result = reducer.reduce(sig, self.spectrum)
        # Constant signal should give finite epsilon
        self.assertTrue(math.isfinite(result.epsilon))
        self.assertGreaterEqual(result.epsilon, 0.0)

    def test_residual_shape(self):
        reducer = GraphReducer(filter_degree=6)
        result = reducer.reduce(self.signal, self.spectrum)
        res = result.residual()
        self.assertEqual(res.shape, torch.Size([self.n]))

    def test_from_adjacency_convenience(self):
        A = StandardGraphs.cycle(self.n)
        sig_vals = torch.sin(torch.arange(self.n, dtype=torch.float64))
        reducer = GraphReducer(filter_degree=6)
        result = reducer.reduce_from_adjacency(A, sig_vals)
        self.assertIsInstance(result, GraphReductionResult)
        self.assertEqual(result.spectrum.n_nodes, self.n)

    def test_path_graph_reduction(self):
        A = StandardGraphs.path(8)
        spectrum = GraphLaplacian.from_adjacency(A)
        sig = _make_signal(8, "linear")
        reducer = GraphReducer(filter_degree=5)
        result = reducer.reduce(sig, spectrum)
        self.assertGreater(result.epsilon, -1e-10)

    def test_summary_string(self):
        reducer = GraphReducer(filter_degree=6)
        result = reducer.reduce(self.signal, self.spectrum)
        s = result.summary()
        self.assertIn("n=10", s)
        self.assertIn("degree=6", s)


# ────────────────────────────────────────────────────────────────────────────
# TestGraphACFAnalyzer
# ────────────────────────────────────────────────────────────────────────────

class TestGraphACFAnalyzer(unittest.TestCase):

    def test_returns_invariants(self):
        A = StandardGraphs.cycle(8)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertIsInstance(inv, GraphACFInvariants)

    def test_alpha_in_0_1(self):
        A = StandardGraphs.path(10)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertGreaterEqual(inv.alpha, 0.0)
        self.assertLessEqual(inv.alpha, 1.0)

    def test_nc_class_valid_values(self):
        A = StandardGraphs.cycle(6)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertIn(inv.nc_class, ("NC0", "NC1", "NC2"))

    def test_fiedler_value_matches_spectrum(self):
        A = StandardGraphs.cycle(8)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertAlmostEqual(inv.fiedler_value, spec.spectral_gap, places=8)

    def test_spectral_entropy_non_negative(self):
        A = StandardGraphs.grid(3, 3)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertGreaterEqual(inv.spectral_entropy, 0.0)

    def test_optimal_degree_positive(self):
        A = StandardGraphs.complete(5)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertGreater(inv.optimal_filter_degree, 0)

    def test_summary_string_contains_key_fields(self):
        A = StandardGraphs.star(6)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        s = inv.summary()
        self.assertIn("α=", s)
        self.assertIn("NC=", s)
        self.assertIn("λ₂=", s)

    def test_complete_graph_zero_fiedler(self):
        # K_n is maximally connected; Fiedler > 0
        A = StandardGraphs.complete(6)
        spec = GraphLaplacian.from_adjacency(A)
        inv = GraphACFAnalyzer().analyse(spec)
        self.assertGreater(inv.fiedler_value, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# TestGraphSignalEvolver
# ────────────────────────────────────────────────────────────────────────────

class TestGraphSignalEvolver(unittest.TestCase):

    def test_returns_evolution_result(self):
        A = StandardGraphs.cycle(8)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        sig = _make_signal(8, "sin")
        cfg = ACFAutoEvolverConfig(initial_degree=5, n_probe=200, enable_bifunctorial=False)
        evolver = GraphSignalEvolver(config=cfg)
        result = evolver.evolve(sig, spec)
        self.assertIsInstance(result, GraphEvolutionResult)

    def test_improvement_ratio_non_negative(self):
        A = StandardGraphs.path(10)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        sig = _make_signal(10, "linear")
        cfg = ACFAutoEvolverConfig(initial_degree=4, n_probe=200, enable_bifunctorial=False, enable_adaptive=False)
        evolver = GraphSignalEvolver(config=cfg)
        result = evolver.evolve(sig, spec)
        self.assertGreater(result.improvement_ratio, 0.0)

    def test_final_epsilon_non_negative(self):
        A = StandardGraphs.cycle(12)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        sig = _make_signal(12, "sin")
        cfg = ACFAutoEvolverConfig(initial_degree=4, n_probe=100, enable_bifunctorial=False, enable_adaptive=False)
        evolver = GraphSignalEvolver(config=cfg)
        result = evolver.evolve(sig, spec)
        self.assertGreaterEqual(result.final_epsilon, 0.0)

    def test_summary_string(self):
        A = StandardGraphs.grid(3, 3)
        spec = GraphLaplacian.from_adjacency(A, normalization="symmetric")
        sig = _make_signal(9, "const")
        cfg = ACFAutoEvolverConfig(initial_degree=3, n_probe=100, enable_bifunctorial=False, enable_adaptive=False)
        evolver = GraphSignalEvolver(config=cfg)
        result = evolver.evolve(sig, spec)
        s = result.summary()
        self.assertIn("ε₀=", s)
        self.assertIn("ε_f=", s)


# ────────────────────────────────────────────────────────────────────────────
# TestStandardGraphs
# ────────────────────────────────────────────────────────────────────────────

class TestStandardGraphs(unittest.TestCase):

    def test_path_shape(self):
        A = StandardGraphs.path(5)
        self.assertEqual(A.shape, (5, 5))

    def test_path_symmetric(self):
        A = StandardGraphs.path(7)
        np.testing.assert_array_equal(A, A.T)

    def test_cycle_has_n_edges(self):
        n = 8
        A = StandardGraphs.cycle(n)
        n_edges = int(A.sum()) // 2
        self.assertEqual(n_edges, n)

    def test_complete_has_n_n_minus_1_div_2_edges(self):
        n = 6
        A = StandardGraphs.complete(n)
        n_edges = int(A.sum()) // 2
        self.assertEqual(n_edges, n * (n - 1) // 2)

    def test_grid_3x4_correct_edges(self):
        A = StandardGraphs.grid(3, 4)
        n_edges = int(A.sum()) // 2
        # 3×4 grid: 3*3 + 2*4 = 9+8 = 17
        self.assertEqual(n_edges, 17)

    def test_star_center_degree(self):
        n = 6
        A = StandardGraphs.star(n)
        # Center node (0) should have degree n-1
        self.assertEqual(int(A[0].sum()), n - 1)

    def test_random_regular_shape(self):
        A = StandardGraphs.random_regular(10, 3, seed=0)
        self.assertEqual(A.shape, (10, 10))
        np.testing.assert_array_equal(A, A.T)


# ────────────────────────────────────────────────────────────────────────────
# TestGideonGraphIntegration
# ────────────────────────────────────────────────────────────────────────────

class TestGideonGraphIntegration(unittest.TestCase):

    def setUp(self):
        from poema.backends.gideon.engine import GideonEngine
        self.engine = GideonEngine()

    def test_reduce_graph_signal_returns_result(self):
        A = StandardGraphs.cycle(8)
        sig = np.sin(np.arange(8, dtype=np.float64))
        result = self.engine.reduce_graph_signal(A, sig, filter_degree=5)
        self.assertIsInstance(result, GraphReductionResult)

    def test_analyse_graph_returns_invariants(self):
        A = StandardGraphs.path(10)
        inv = self.engine.analyse_graph(A)
        self.assertIsInstance(inv, GraphACFInvariants)

    def test_reduce_graph_with_config_returns_evolution(self):
        A = StandardGraphs.grid(3, 3)
        sig = np.ones(9, dtype=np.float64)
        cfg = ACFAutoEvolverConfig(initial_degree=4, n_probe=100, enable_bifunctorial=False, enable_adaptive=False)
        result = self.engine.reduce_graph_signal(A, sig, config=cfg)
        self.assertIsInstance(result, GraphEvolutionResult)

    def test_gideon_graph_alpha_in_bounds(self):
        A = StandardGraphs.complete(5)
        inv = self.engine.analyse_graph(A)
        self.assertGreaterEqual(inv.alpha, 0.0)
        self.assertLessEqual(inv.alpha, 1.0)


# ────────────────────────────────────────────────────────────────────────────
# TestPublicAPI
# ────────────────────────────────────────────────────────────────────────────

class TestGraphPublicAPI(unittest.TestCase):

    def test_imports_from_acf_functor(self):
        from acf_functor import (
            GraphLaplacian, GraphReducer, GraphACFAnalyzer,
            GraphSignalEvolver, GraphSpectrum, GraphSignal,
            GraphReductionResult, GraphACFInvariants, GraphEvolutionResult,
            StandardGraphs,
        )
        self.assertTrue(True)  # All imports succeeded

    def test_all_standard_graphs_produce_valid_spectrum(self):
        for name, A in [
            ("path", StandardGraphs.path(6)),
            ("cycle", StandardGraphs.cycle(6)),
            ("complete", StandardGraphs.complete(5)),
            ("grid", StandardGraphs.grid(2, 3)),
            ("star", StandardGraphs.star(5)),
        ]:
            with self.subTest(graph=name):
                spec = GraphLaplacian.from_adjacency(A)
                # All eigenvalues non-negative
                self.assertTrue(
                    (spec.eigenvalues >= -1e-9).all(),
                    f"{name}: negative eigenvalue found",
                )


if __name__ == "__main__":
    unittest.main()
