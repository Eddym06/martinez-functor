from __future__ import annotations

import math

import pytest
import torch

from python_analysis.phi_functor import PhiFunctor
from python_analysis.transcendental_generated import (
    chebyshev_exp,
    chebyshev_log,
    chebyshev_sin,
    clenshaw_eval_tensor,
    eval_exp_complete,
    eval_sin_complete,
)


class TestTranscendentalConservation:
    def test_sin_conservation_structural(self) -> None:
        approx = chebyshev_sin(n=20, epsilon=1e-6)
        e_theory = approx.degree
        fma_seq, _seed = PhiFunctor.polynomial_exact_from_chebyshev(approx)
        e_measured = len(fma_seq)
        assert e_theory == e_measured

    def test_sin_conservation_numerical(self) -> None:
        approx = chebyshev_sin(n=20, epsilon=1e-6)
        x = torch.linspace(-math.pi, math.pi, 10_000, dtype=torch.float64)
        y_phi = clenshaw_eval_tensor(approx.coeffs, x, (approx.domain_a, approx.domain_b))
        y_true = torch.sin(x)
        max_error = torch.max(torch.abs(y_true - y_phi)).item()
        assert max_error < approx.epsilon * 2.0

    def test_sin_domain_reduction(self) -> None:
        approx = chebyshev_sin(n=20, epsilon=1e-6)
        test_values = [10.0, 100.0, -50.0, 1000.0 * math.pi]
        for xv in test_values:
            y_phi = eval_sin_complete(xv, approx)
            y_true = math.sin(xv)
            err = abs(y_phi - y_true)
            assert err < approx.epsilon * 10.0

    def test_exp_conservation(self) -> None:
        approx = chebyshev_exp(n=15, epsilon=1e-6)
        x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64)
        y_phi = clenshaw_eval_tensor(approx.coeffs, x, (approx.domain_a, approx.domain_b))
        y_true = torch.exp(x)
        max_error = torch.max(torch.abs(y_true - y_phi)).item()
        assert max_error < approx.epsilon * 2.0

    def test_log_conservation(self) -> None:
        approx = chebyshev_log(n=25, epsilon=1e-6)
        x = torch.linspace(0.5, 2.0, 2000, dtype=torch.float64)
        y_phi = clenshaw_eval_tensor(approx.coeffs, x, (approx.domain_a, approx.domain_b))
        y_true = torch.log(x)
        max_error = torch.max(torch.abs(y_true - y_phi)).item()
        assert max_error < approx.epsilon * 2.0

    def test_exp_domain_extension(self) -> None:
        approx = chebyshev_exp(n=15, epsilon=1e-6)
        for xv in [-6.0, -2.5, 0.0, 3.0, 6.0]:
            y_phi = eval_exp_complete(xv, approx)
            y_true = math.exp(xv)
            rel = abs(y_phi - y_true) / max(1e-12, abs(y_true))
            assert rel < 5e-4

    def test_cross_function_composition(self) -> None:
        approx_sin = chebyshev_sin(n=22, epsilon=1e-7)
        approx_exp = chebyshev_exp(n=18, epsilon=1e-7)
        x = torch.linspace(-1.0, 1.0, 1000, dtype=torch.float64)

        y_sin = clenshaw_eval_tensor(approx_sin.coeffs, x, (approx_sin.domain_a, approx_sin.domain_b))
        y_composed = clenshaw_eval_tensor(
            approx_exp.coeffs,
            y_sin,
            (approx_exp.domain_a, approx_exp.domain_b),
        )
        y_true = torch.exp(torch.sin(x))

        max_error = torch.max(torch.abs(y_true - y_composed)).item()
        bound = approx_sin.epsilon + approx_exp.epsilon
        assert max_error < bound * 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
