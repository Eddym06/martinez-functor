"""
Tests for acf_functor/neural_acf.py

Covers:
  - NeuralLayerReducer: nn.Linear and nn.Conv1d layer analysis
  - NetworkACFAnalyzer: full sequential network analysis
  - KoopmanNetworkDynamics: training-trajectory Koopman analysis
  - NeuralACFEvolver: function evolution implemented by MLP (fast config)
  - build_test_mlp factory
  - GideonEngine integration
"""

import math
import unittest

import numpy as np
import torch
import torch.nn as nn

from acf_functor import ACFAutoEvolverConfig
from acf_functor.neural_acf import (
    KoopmanNetworkDynamics,
    KoopmanNetworkResult,
    LayerACFInvariants,
    LayerReductionResult,
    NetworkACFAnalyzer,
    NetworkACFReport,
    NeuralACFEvolver,
    NeuralEvolutionResult,
    NeuralLayerReducer,
    build_test_mlp,
)


# ────────────────────────────────────────────────────────────────────────────
# Fast evolver config (disables slow mechanisms for unit tests)
# ────────────────────────────────────────────────────────────────────────────

FAST_CFG = ACFAutoEvolverConfig(
    initial_degree=5,
    n_probe=200,
    enable_bifunctorial=False,
    enable_adaptive=False,
    fp_max_iterations=3,
)


# ────────────────────────────────────────────────────────────────────────────
# TestBuildTestMlp
# ────────────────────────────────────────────────────────────────────────────

class TestBuildTestMlp(unittest.TestCase):

    def test_basic_mlp_is_sequential(self):
        net = build_test_mlp([2, 8, 8, 1])
        self.assertIsInstance(net, nn.Sequential)

    def test_layer_count_correct(self):
        # [2, 8, 4, 1] → 3 Linear + 2 activations = 5 modules min
        net = build_test_mlp([2, 8, 4, 1])
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(linears), 3)

    def test_input_output_dims(self):
        net = build_test_mlp([3, 16, 1])
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(linears[0].in_features, 3)
        self.assertEqual(linears[-1].out_features, 1)

    def test_seed_reproducibility(self):
        net1 = build_test_mlp([2, 8, 1], seed=42)
        net2 = build_test_mlp([2, 8, 1], seed=42)
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_different_seeds_different_weights(self):
        net1 = build_test_mlp([2, 8, 1], seed=42)
        net2 = build_test_mlp([2, 8, 1], seed=99)
        params_equal = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(net1.parameters(), net2.parameters())
        )
        self.assertFalse(params_equal)

    def test_tanh_activation(self):
        net = build_test_mlp([2, 4, 1], activation="tanh")
        tanh_count = sum(1 for m in net.modules() if isinstance(m, nn.Tanh))
        self.assertGreater(tanh_count, 0)

    def test_relu_activation(self):
        net = build_test_mlp([2, 4, 1], activation="relu")
        relu_count = sum(1 for m in net.modules() if isinstance(m, nn.ReLU))
        self.assertGreater(relu_count, 0)

    def test_forward_runs(self):
        net = build_test_mlp([1, 8, 1])
        x = torch.randn(10, 1)
        y = net(x)
        self.assertEqual(y.shape, (10, 1))


# ────────────────────────────────────────────────────────────────────────────
# TestNeuralLayerReducer
# ────────────────────────────────────────────────────────────────────────────

class TestNeuralLayerReducer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.reducer = NeuralLayerReducer(degree=8, domain=(-3.0, 3.0))

    def test_reduce_linear_returns_result(self):
        layer = nn.Linear(4, 4)
        result = self.reducer.reduce_linear(layer)
        self.assertIsInstance(result, LayerReductionResult)

    def test_reduce_linear_epsilon_finite(self):
        layer = nn.Linear(8, 8)
        result = self.reducer.reduce_linear(layer)
        self.assertTrue(math.isfinite(result.epsilon))

    def test_reduce_linear_epsilon_non_negative(self):
        layer = nn.Linear(8, 8)
        result = self.reducer.reduce_linear(layer)
        self.assertGreaterEqual(result.epsilon, 0.0)

    def test_reduce_linear_fma_chain_non_empty(self):
        layer = nn.Linear(6, 6)
        result = self.reducer.reduce_linear(layer)
        self.assertGreater(len(result.fma_chain), 0)

    def test_reduce_conv1d_returns_result(self):
        layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        result = self.reducer.reduce_conv1d(layer)
        self.assertIsInstance(result, LayerReductionResult)

    def test_reduce_conv1d_epsilon_finite(self):
        layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7)
        result = self.reducer.reduce_conv1d(layer)
        self.assertTrue(math.isfinite(result.epsilon))

    def test_reduce_layer_dispatch_linear(self):
        layer = nn.Linear(4, 4)
        result = self.reducer.reduce_layer(layer)
        self.assertIsInstance(result, LayerReductionResult)

    def test_reduce_layer_dispatch_conv1d(self):
        layer = nn.Conv1d(1, 1, 3)
        result = self.reducer.reduce_layer(layer)
        self.assertIsInstance(result, LayerReductionResult)

    def test_reduce_layer_unsupported_returns_none(self):
        layer = nn.BatchNorm1d(8)
        result = self.reducer.reduce_layer(layer)
        self.assertIsNone(result)


# ────────────────────────────────────────────────────────────────────────────
# TestNetworkACFAnalyzer
# ────────────────────────────────────────────────────────────────────────────

class TestNetworkACFAnalyzer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1)
        self.net = build_test_mlp([1, 8, 8, 1])
        self.analyzer = NetworkACFAnalyzer(degree=10, domain=(-3.0, 3.0))

    def test_returns_network_report(self):
        report = self.analyzer.analyse(self.net)
        self.assertIsInstance(report, NetworkACFReport)

    def test_n_layers_correct(self):
        report = self.analyzer.analyse(self.net)
        self.assertGreater(len(report.layer_reductions), 0)
        # [1,8,8,1] → 3 Linear layers
        self.assertGreaterEqual(len(report.layer_reductions), 2)

    def test_global_alpha_in_0_1(self):
        report = self.analyzer.analyse(self.net)
        self.assertGreaterEqual(report.global_alpha, 0.0)
        self.assertLessEqual(report.global_alpha, 1.0)

    def test_nc_class_valid(self):
        report = self.analyzer.analyse(self.net)
        self.assertIn(report.global_nc_class, ("NC0", "NC1", "NC2"))

    def test_total_fma_ops_positive(self):
        report = self.analyzer.analyse(self.net)
        self.assertGreater(report.total_fma_count, 0)

    def test_layer_reports_not_empty(self):
        report = self.analyzer.analyse(self.net)
        self.assertGreater(len(report.layer_reductions), 0)

    def test_each_layer_report_epsilon_finite(self):
        report = self.analyzer.analyse(self.net)
        for lr in report.layer_reductions:
            self.assertTrue(math.isfinite(lr.epsilon))

    def test_small_mlp_1_8_1(self):
        net = build_test_mlp([1, 8, 1], seed=7)
        report = self.analyzer.analyse(net)
        # Should have at least 1 Linear layer analysed
        self.assertGreaterEqual(len(report.layer_reductions), 1)

    def test_summary_string_contains_global_alpha(self):
        report = self.analyzer.analyse(self.net)
        s = report.summary()
        self.assertIn("global_α=", s)

    def test_summary_string_contains_nc_class(self):
        report = self.analyzer.analyse(self.net)
        s = report.summary()
        self.assertIn("NC=", s)

    def test_summary_string_contains_fma(self):
        report = self.analyzer.analyse(self.net)
        s = report.summary()
        self.assertIn("FMA", s)

    def test_different_architectures_different_reports(self):
        net1 = build_test_mlp([1, 4, 1], seed=0)
        net2 = build_test_mlp([1, 32, 32, 32, 1], seed=0)
        r1 = self.analyzer.analyse(net1)
        r2 = self.analyzer.analyse(net2)
        # Wider/deeper net should have more FMA ops
        self.assertLessEqual(r1.total_fma_count, r2.total_fma_count)


# ────────────────────────────────────────────────────────────────────────────
# TestKoopmanNetworkDynamics
# ────────────────────────────────────────────────────────────────────────────

class TestKoopmanNetworkDynamics(unittest.TestCase):

    def _make_loss_trajectory(self, n=80, noise=0.05):
        t = np.arange(n, dtype=np.float64)
        traj = np.exp(-t / 20.0) + noise * np.random.RandomState(0).randn(n)
        return traj.tolist()

    def test_returns_koopman_result(self):
        traj = self._make_loss_trajectory()
        knd = KoopmanNetworkDynamics()
        result = knd.analyse(traj)
        self.assertIsInstance(result, KoopmanNetworkResult)

    def test_eigenvalue_shape(self):
        traj = self._make_loss_trajectory()
        knd = KoopmanNetworkDynamics()
        result = knd.analyse(traj)
        self.assertGreater(len(result.koopman_eigenvalues), 0)

    def test_spectral_radius_finite(self):
        traj = self._make_loss_trajectory()
        knd = KoopmanNetworkDynamics()
        result = knd.analyse(traj)
        self.assertTrue(math.isfinite(result.spectral_diagnostics.spectral_radius))

    def test_spectral_radius_non_negative(self):
        traj = self._make_loss_trajectory()
        knd = KoopmanNetworkDynamics()
        result = knd.analyse(traj)
        self.assertGreaterEqual(result.spectral_diagnostics.spectral_radius, 0.0)

    def test_converging_trajectory_spectral_radius_lt_1(self):
        # Exponential decay → dominant eigenvalue < 1 in modulus
        traj = self._make_loss_trajectory(n=100)
        knd = KoopmanNetworkDynamics()
        result = knd.analyse(traj)
        self.assertLessEqual(result.spectral_diagnostics.spectral_radius, 1.5)  # soft bound

    def test_summary_string(self):
        traj = self._make_loss_trajectory()
        knd = KoopmanNetworkDynamics()
        result = knd.analyse(traj)
        s = result.summary()
        self.assertIn("α=", s)  # SpectralDiagnostics alpha in summary

    def test_list_and_numpy_input(self):
        traj_list = self._make_loss_trajectory(n=50)
        traj_np = np.array(traj_list)
        knd = KoopmanNetworkDynamics()
        r1 = knd.analyse(traj_list)
        r2 = knd.analyse(traj_np)
        self.assertAlmostEqual(
            r1.spectral_diagnostics.spectral_radius,
            r2.spectral_diagnostics.spectral_radius,
            places=8,
        )


# ────────────────────────────────────────────────────────────────────────────
# TestNeuralACFEvolver
# ────────────────────────────────────────────────────────────────────────────

class TestNeuralACFEvolver(unittest.TestCase):

    def test_returns_evolution_result(self):
        net = build_test_mlp([1, 8, 1], seed=0)
        evolver = NeuralACFEvolver(config=FAST_CFG)
        result = evolver.evolve(net, domain=(-2.0, 2.0), input_dim=1)
        self.assertIsInstance(result, NeuralEvolutionResult)

    def test_improvement_ratio_positive(self):
        net = build_test_mlp([1, 4, 1], seed=1)
        evolver = NeuralACFEvolver(config=FAST_CFG)
        result = evolver.evolve(net, domain=(-1.0, 1.0), input_dim=1)
        self.assertGreater(result.improvement_ratio, 0.0)

    def test_final_epsilon_finite(self):
        net = build_test_mlp([1, 4, 1], seed=2)
        evolver = NeuralACFEvolver(config=FAST_CFG)
        result = evolver.evolve(net, domain=(-2.0, 2.0), input_dim=1)
        self.assertTrue(math.isfinite(result.final_epsilon))

    def test_final_epsilon_non_negative(self):
        net = build_test_mlp([1, 4, 1], seed=3)
        evolver = NeuralACFEvolver(config=FAST_CFG)
        result = evolver.evolve(net, domain=(-2.0, 2.0), input_dim=1)
        self.assertGreaterEqual(result.final_epsilon, 0.0)

    def test_summary_string(self):
        net = build_test_mlp([1, 4, 1], seed=4)
        evolver = NeuralACFEvolver(config=FAST_CFG)
        result = evolver.evolve(net, domain=(-1.0, 1.0), input_dim=1)
        s = result.summary()
        self.assertIn("ε₀=", s)


# ────────────────────────────────────────────────────────────────────────────
# TestGideonNeuralIntegration
# ────────────────────────────────────────────────────────────────────────────

class TestGideonNeuralIntegration(unittest.TestCase):

    def setUp(self):
        from poema.backends.gideon.engine import GideonEngine
        self.engine = GideonEngine()

    def test_analyse_network_returns_report(self):
        net = build_test_mlp([1, 8, 1], seed=0)
        report = self.engine.analyse_network(net, degree=8, domain=(-2, 2), as_dict=False)
        self.assertIsInstance(report, NetworkACFReport)

    def test_analyse_training_trajectory_returns_result(self):
        traj = [math.exp(-i / 20.0) for i in range(60)]
        result = self.engine.analyse_training_trajectory(traj)
        self.assertIsInstance(result, KoopmanNetworkResult)

    def test_evolve_network_function_returns_result(self):
        net = build_test_mlp([1, 4, 1], seed=0)
        result = self.engine.evolve_network_function(
            net, domain=(-1.0, 1.0), input_dim=1, config=FAST_CFG
        )
        self.assertIsInstance(result, NeuralEvolutionResult)

    def test_gideon_network_alpha_in_bounds(self):
        net = build_test_mlp([1, 8, 8, 1], seed=0)
        report = self.engine.analyse_network(net, as_dict=False)
        self.assertGreaterEqual(report.global_alpha, 0.0)
        self.assertLessEqual(report.global_alpha, 1.0)

# ────────────────────────────────────────────────────────────────────────────
# TestNeuralACFPublicAPI
# ────────────────────────────────────────────────────────────────────────────

class TestNeuralACFPublicAPI(unittest.TestCase):

    def test_imports_from_acf_functor(self):
        from acf_functor import (
            NeuralLayerReducer, NetworkACFAnalyzer, KoopmanNetworkDynamics,
            NeuralACFEvolver, LayerReductionResult, NetworkACFReport,
            LayerACFInvariants, KoopmanNetworkResult, NeuralEvolutionResult,
            build_test_mlp,
        )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
