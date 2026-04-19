import math

import numpy as np

from acf_functor.invariant_unified import (
    AlphaCombinatorial,
    AlphaGeometric,
    AlphaSpectral,
    ACFInvariantUnified,
)


class TestAlphaCombinatorial:
    def test_polynomial_is_near_zero(self):
        calc = AlphaCombinatorial(epsilon_range=[10 ** (-k) for k in range(1, 7)])
        alpha, _ci, _n = calc.compute(lambda x: x**3 + 2 * x + 1, domain=(-1.0, 1.0))
        assert alpha < 0.2

    def test_sin_increases_fma(self):
        calc = AlphaCombinatorial(epsilon_range=[1e-1, 1e-2, 1e-3])
        domain = (-math.pi / 2, math.pi / 2)
        e1 = calc.minimum_fma_count(np.sin, 1e-1, domain=domain)
        e2 = calc.minimum_fma_count(np.sin, 1e-2, domain=domain)
        e3 = calc.minimum_fma_count(np.sin, 1e-3, domain=domain)
        assert e1 <= e2 <= e3


class TestAlphaSpectral:
    def test_linear_system_small_alpha(self):
        calc = AlphaSpectral(n_observables=30, n_trajectory=3000)
        alpha, _ci = calc.compute(lambda x: 0.6 * x, domain=(-1.0, 1.0))
        assert alpha < 0.6


class TestAlphaGeometric:
    def test_fixed_point_alpha_geo_zero(self):
        calc = AlphaGeometric(n_trajectory=8000, n_scales=10)
        alpha, _ci = calc.compute(lambda _x: 0.2, domain=(-1.0, 1.0))
        assert alpha == 0.0


class TestUnifiedInvariant:
    def test_unified_compute_returns_consistency_fields(self):
        inv = ACFInvariantUnified(consistency_threshold=0.5)
        result = inv.compute(np.sin, domain=(-1.0, 1.0), function_name="sin", skip_geometric=True)
        assert result.function_name == "sin"
        assert result.n_fma_evaluations > 0
        assert result.max_discrepancy >= 0.0
