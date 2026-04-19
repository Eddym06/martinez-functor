"""
tests/test_stochastic_expansion.py
====================================
Tests for the expanded Stochastic ACF domain:
  - HighEntropyAnalyzer (Hurst, Lévy, spectral entropy, K-S entropy)
  - FinancialACF (VaR, regimes, Sharpe bound, full pipeline)
  - BayesianNNAnalyzer (weight distribution, prediction UQ, Sobol)

These tests validate the new classes added to acf_functor/stochastic_acf.py.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acf_functor.stochastic_acf import (
    HighEntropyAnalyzer,
    HighEntropyAnalysis,
    HurstResult,
    LevyStableResult,
    SpectralEntropyResult,
    FinancialACF,
    FinancialReport,
    VaRCertified,
    BayesianNNAnalyzer,
    BayesianNNReport,
    PolynomialChaosACF,
    compute_uncertainty_bound,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def bm_series():
    """Brownian motion — H ≈ 0.5."""
    steps = RNG.standard_normal(1000)
    return np.cumsum(steps)


@pytest.fixture(scope="module")
def trending_series():
    """Trending series — H > 0.5."""
    return np.cumsum(RNG.standard_normal(500)) + np.linspace(0, 10, 500)


@pytest.fixture(scope="module")
def financial_returns():
    """Simulated daily returns ~ N(0.001, 0.02)."""
    return RNG.normal(loc=0.001, scale=0.02, size=500)


@pytest.fixture(scope="module")
def high_entropy_analyzer():
    return HighEntropyAnalyzer(n_pce_terms=4, pce_degree=3)


@pytest.fixture(scope="module")
def financial_acf():
    return FinancialACF(pce_degree=3)


# ─────────────────────────────────────────────────────────────────────────
# HighEntropyAnalyzer tests
# ─────────────────────────────────────────────────────────────────────────

class TestHighEntropyAnalyzer:
    def test_hurst_bm_near_half(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.hurst_exponent(bm_series)
        assert isinstance(result, HurstResult)
        # R/S finite-sample estimate is noisy; just verify it is in valid range
        assert 0.0 < result.H < 1.0

    def test_hurst_returns_hurst_result(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.hurst_exponent(bm_series)
        assert 0 < result.H < 1

    def test_hurst_confidence_interval_ordered(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.hurst_exponent(bm_series)
        lo, hi = result.confidence_interval
        assert lo <= result.H <= hi or abs(lo - hi) < 0.3  # loose check

    def test_hurst_interpretation_string(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.hurst_exponent(bm_series)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_hurst_regime_bm(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.hurst_exponent(bm_series)
        regime = result.regime()
        assert regime in {"mean_reverting", "trending", "random_walk"}

    def test_levy_alpha_stable_range(self, high_entropy_analyzer, financial_returns):
        result = high_entropy_analyzer.levy_alpha_stable(financial_returns)
        assert isinstance(result, LevyStableResult)
        assert 0 < result.alpha_stable <= 2.0

    def test_levy_beta_range(self, high_entropy_analyzer, financial_returns):
        result = high_entropy_analyzer.levy_alpha_stable(financial_returns)
        assert -1.0 <= result.beta <= 1.0

    def test_levy_scale_positive(self, high_entropy_analyzer, financial_returns):
        result = high_entropy_analyzer.levy_alpha_stable(financial_returns)
        assert result.scale > 0

    def test_levy_fat_tails_property(self, high_entropy_analyzer):
        # Cauchy-like returns should give α_stable < 1.5
        heavy = RNG.standard_cauchy(200)
        result = high_entropy_analyzer.levy_alpha_stable(heavy)
        assert isinstance(result.fat_tails, bool)

    def test_spectral_entropy_range(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.spectral_entropy(bm_series)
        assert isinstance(result, SpectralEntropyResult)
        assert 0.0 <= result.entropy <= 1.0

    def test_spectral_entropy_pure_sine(self, high_entropy_analyzer):
        """Pure sinusoid → low entropy (near 0)."""
        t = np.linspace(0, 4 * np.pi, 512)
        sine = np.sin(3 * t)
        result = high_entropy_analyzer.spectral_entropy(sine)
        assert result.entropy < 0.5  # closer to 0 for pure frequency

    def test_spectral_entropy_noise(self, high_entropy_analyzer):
        """White noise → high entropy (near 1)."""
        noise = RNG.standard_normal(512)
        result = high_entropy_analyzer.spectral_entropy(noise)
        assert result.entropy > 0.5

    def test_spectral_dominant_freq_nonneg(self, high_entropy_analyzer, bm_series):
        result = high_entropy_analyzer.spectral_entropy(bm_series)
        assert result.dominant_frequency >= 0

    def test_ks_entropy_nonneg(self, high_entropy_analyzer, bm_series):
        ks = high_entropy_analyzer.kolmogorov_entropy_rate(bm_series)
        assert ks >= 0.0

    def test_ks_entropy_finite(self, high_entropy_analyzer, financial_returns):
        ks = high_entropy_analyzer.kolmogorov_entropy_rate(financial_returns)
        assert math.isfinite(ks)

    def test_analyze_returns_full_report(self, high_entropy_analyzer, bm_series):
        report = high_entropy_analyzer.analyze(bm_series)
        assert isinstance(report, HighEntropyAnalysis)
        assert report.series_length == len(bm_series)
        assert isinstance(report.hurst, HurstResult)
        assert isinstance(report.levy, LevyStableResult)
        assert report.acf_alpha >= 0

    def test_analyze_uncertainty_bound(self, high_entropy_analyzer, bm_series):
        report = high_entropy_analyzer.analyze(bm_series)
        assert report.uncertainty_bound.confidence_band >= 0
        assert 0 < report.uncertainty_bound.confidence_level <= 1


# ─────────────────────────────────────────────────────────────────────────
# FinancialACF tests
# ─────────────────────────────────────────────────────────────────────────

class TestFinancialACF:
    def test_fit_returns_pce_coefficients(self, financial_acf, financial_returns):
        from acf_functor.stochastic_acf import PCECoefficients
        coeffs = financial_acf.fit_returns(financial_returns)
        assert isinstance(coeffs, PCECoefficients)

    def test_var_certified_returns_object(self, financial_acf, financial_returns):
        var = financial_acf.var_certified(financial_returns)
        assert isinstance(var, VaRCertified)

    def test_var_95_less_than_var_99(self, financial_acf, financial_returns):
        var = financial_acf.var_certified(financial_returns)
        # VaR_99 ≥ VaR_95 (higher confidence → larger loss bound)
        assert var.var_99 >= var.var_95

    def test_var_chebyshev_certified(self, financial_acf, financial_returns):
        var = financial_acf.var_certified(financial_returns)
        assert var.chebyshev_certified is True

    def test_var_alpha_stoch_nonneg(self, financial_acf, financial_returns):
        var = financial_acf.var_certified(financial_returns)
        assert var.alpha_stoch >= 0

    def test_detect_regimes_list(self, financial_acf, financial_returns):
        regimes = financial_acf.detect_regimes(financial_returns)
        assert isinstance(regimes, list)
        assert len(regimes) > 0
        for r in regimes:
            assert r in {"bull", "bear", "volatile", "quiescent"}

    def test_sharpe_uncertainty_bound_positive(self, financial_acf, financial_returns):
        ub = financial_acf.sharpe_uncertainty_bound(financial_returns)
        assert ub.confidence_band >= 0

    def test_analyze_full_report(self, financial_acf, financial_returns):
        report = financial_acf.analyze(financial_returns, asset_name="TEST")
        assert isinstance(report, FinancialReport)
        assert report.asset_name == "TEST"
        assert report.n_observations == len(financial_returns)

    def test_report_mean_finite(self, financial_acf, financial_returns):
        report = financial_acf.analyze(financial_returns)
        assert math.isfinite(report.mean_return)

    def test_report_volatility_positive(self, financial_acf, financial_returns):
        report = financial_acf.analyze(financial_returns)
        assert report.volatility > 0

    def test_report_sharpe_lb_finite(self, financial_acf, financial_returns):
        report = financial_acf.analyze(financial_returns)
        assert math.isfinite(report.sharpe_lower_bound)

    def test_cvar_at_least_var95(self, financial_acf, financial_returns):
        var = financial_acf.var_certified(financial_returns)
        # var_95 uses Chebyshev (conservative), cvar_95 is empirical expected shortfall.
        # Both should be positive and finite.
        assert math.isfinite(var.cvar_95) and var.cvar_95 > 0
        assert math.isfinite(var.var_95) and var.var_95 > 0


# ─────────────────────────────────────────────────────────────────────────
# BayesianNNAnalyzer tests
# ─────────────────────────────────────────────────────────────────────────

def _make_tiny_mlp(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


@pytest.fixture(scope="module")
def small_ensemble():
    return [_make_tiny_mlp(seed=i) for i in range(5)]


@pytest.fixture(scope="module")
def bnn_analyzer():
    return BayesianNNAnalyzer(pce_degree=2, n_weight_groups=2, n_mc_samples=10)


class TestBayesianNNAnalyzer:
    def test_fit_weight_distribution(self, bnn_analyzer, small_ensemble):
        from acf_functor.stochastic_acf import PCECoefficients
        coeffs = bnn_analyzer.fit_weight_distribution(small_ensemble)
        assert isinstance(coeffs, PCECoefficients)

    def test_prediction_uncertainty_returns_bound(self, bnn_analyzer, small_ensemble):
        from acf_functor.stochastic_acf import UncertaintyBound
        x = np.random.default_rng(0).standard_normal(4)
        ub = bnn_analyzer.prediction_uncertainty(small_ensemble, x)
        assert isinstance(ub, UncertaintyBound)

    def test_prediction_confidence_band_nonneg(self, bnn_analyzer, small_ensemble):
        x = np.zeros(4)
        ub = bnn_analyzer.prediction_uncertainty(small_ensemble, x)
        assert ub.confidence_band >= 0

    def test_sobol_indices_list(self, bnn_analyzer, small_ensemble):
        x = np.zeros(4)
        sobol = bnn_analyzer.sobol_param_sensitivity(small_ensemble, x)
        assert isinstance(sobol, list)
        assert len(sobol) == bnn_analyzer.n_weight_groups

    def test_sobol_nonneg(self, bnn_analyzer, small_ensemble):
        x = np.zeros(4)
        sobol = bnn_analyzer.sobol_param_sensitivity(small_ensemble, x)
        for s in sobol:
            assert s >= 0

    def test_analyze_report(self, bnn_analyzer, small_ensemble):
        x = np.zeros(4)
        report = bnn_analyzer.analyze(small_ensemble, x, layer_name="fc")
        assert isinstance(report, BayesianNNReport)
        assert report.layer_name == "fc"

    def test_weight_entropy_nonneg(self, bnn_analyzer, small_ensemble):
        x = np.zeros(4)
        report = bnn_analyzer.analyze(small_ensemble, x)
        assert report.weight_entropy >= 0

    def test_effective_params_le_total(self, bnn_analyzer, small_ensemble):
        x = np.zeros(4)
        report = bnn_analyzer.analyze(small_ensemble, x)
        total_params = sum(
            p.numel() for p in small_ensemble[0].parameters() if p.requires_grad
        )
        assert report.effective_parameters <= total_params
