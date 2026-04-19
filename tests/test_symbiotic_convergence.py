"""Tests for acf_functor/symbiotic_convergence.py."""

import pytest
import torch

from acf_functor.symbiotic_convergence import (
    ContractionEstimate,
    SymbioticConvergenceAnalyzer,
    analyze_convergence,
)


class TestSymbioticConvergenceAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SymbioticConvergenceAnalyzer(dtype=torch.float64)

    def _generate_stable_linear_data(self, n_features=3, n_timesteps=200, seed=42):
        torch.manual_seed(seed)
        A = torch.randn(n_features, n_features, dtype=torch.float64) * 0.2
        x = torch.randn(n_features, 1, dtype=torch.float64)
        data = [x.squeeze(-1)]
        for _ in range(n_timesteps):
            x = A @ x
            data.append(x.squeeze(-1))
        return torch.stack(data, dim=1), A

    def _generate_unstable_linear_data(self, n_features=3, n_timesteps=200, seed=42):
        torch.manual_seed(seed)
        A = torch.randn(n_features, n_features, dtype=torch.float64) * 2.0
        x = torch.randn(n_features, 1, dtype=torch.float64)
        data = [x.squeeze(-1)]
        for _ in range(n_timesteps):
            x = A @ x
            data.append(x.squeeze(-1))
        return torch.stack(data, dim=1), A

    def test_estimate_contraction_rate_stable_system(self, analyzer):
        data, A = self._generate_stable_linear_data()
        result = analyzer.estimate_contraction_rate(data, n_trials=5)
        assert isinstance(result, ContractionEstimate)
        assert isinstance(result.lipschitz_constant, float)
        assert isinstance(result.is_contraction, bool)
        assert isinstance(result.estimated_convergence_rate, float)
        assert isinstance(result.n_iterations_to_eps, int)

    def test_estimate_contraction_rate_returns_finite_for_stable(self, analyzer):
        data, _ = self._generate_stable_linear_data()
        result = analyzer.estimate_contraction_rate(data, n_trials=5)
        assert result.lipschitz_constant != float("inf")
        assert result.estimated_convergence_rate != float("inf")

    def test_estimate_contraction_rate_unstable_system(self, analyzer):
        data, _ = self._generate_unstable_linear_data()
        result = analyzer.estimate_contraction_rate(data, n_trials=5)
        assert isinstance(result, ContractionEstimate)
        linear_check = analyzer.check_linear_system_condition(data)
        assert linear_check["is_stable"] is False
        assert linear_check["condition_met"] is False

    def test_check_linear_system_condition_stable(self, analyzer):
        data, A = self._generate_stable_linear_data()
        result = analyzer.check_linear_system_condition(data)
        assert "condition_met" in result
        assert "estimated_A" in result
        assert "spectral_radius" in result
        assert "is_stable" in result
        assert isinstance(result["estimated_A"], torch.Tensor)
        assert result["spectral_radius"] >= 0.0

    def test_check_linear_system_condition_detects_stability(self, analyzer):
        data, A = self._generate_stable_linear_data()
        result = analyzer.check_linear_system_condition(data)
        assert result["is_stable"] is True
        assert result["condition_met"] is True
        assert result["spectral_radius"] < 1.0

    def test_check_linear_system_condition_detects_instability(self, analyzer):
        data, _ = self._generate_unstable_linear_data()
        result = analyzer.check_linear_system_condition(data)
        assert result["is_stable"] is False
        assert result["condition_met"] is False
        assert result["spectral_radius"] >= 1.0

    def test_check_linear_system_condition_estimated_convergence_rate(self, analyzer):
        data, _ = self._generate_stable_linear_data()
        result = analyzer.check_linear_system_condition(data)
        assert "estimated_convergence_rate" in result
        assert result["estimated_convergence_rate"] >= 0.0

    def test_analyze_convergence_stable_system(self):
        torch.manual_seed(42)
        A = torch.eye(3, dtype=torch.float64) * 0.3
        x = torch.randn(3, 1, dtype=torch.float64)
        data_list = [x.squeeze(-1)]
        for _ in range(200):
            x = A @ x
            data_list.append(x.squeeze(-1))
        data = torch.stack(data_list, dim=1)

        result = analyze_convergence(data)
        assert "contraction" in result
        assert "linear_system_check" in result
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_analyze_convergence_unstable_system(self):
        torch.manual_seed(42)
        A = torch.eye(3, dtype=torch.float64) * 2.0
        x = torch.randn(3, 1, dtype=torch.float64)
        data_list = [x.squeeze(-1)]
        for _ in range(100):
            x = A @ x
            data_list.append(x.squeeze(-1))
        data = torch.stack(data_list, dim=1)

        result = analyze_convergence(data)
        assert "contraction" in result
        assert "linear_system_check" in result
        assert "recommendations" in result

    def test_contraction_estimate_dataclass_fields(self):
        est = ContractionEstimate(
            lipschitz_constant=0.5,
            is_contraction=True,
            estimated_convergence_rate=0.5,
            n_iterations_to_eps=20,
        )
        assert est.lipschitz_constant == 0.5
        assert est.is_contraction is True
        assert est.estimated_convergence_rate == 0.5
        assert est.n_iterations_to_eps == 20
        assert est.warning is None

    def test_contraction_estimate_with_warning(self):
        est = ContractionEstimate(
            lipschitz_constant=1.5,
            is_contraction=False,
            estimated_convergence_rate=1.5,
            n_iterations_to_eps=-1,
            warning="L=1.500 >= 1: convergence not guaranteed",
        )
        assert est.is_contraction is False
        assert est.warning is not None

    def test_analyze_convergence_custom_dtype(self):
        torch.manual_seed(42)
        data = torch.randn(2, 100, dtype=torch.float64) * 0.5
        result = analyze_convergence(data, dtype=torch.float64)
        assert isinstance(result, dict)
        assert "contraction" in result
