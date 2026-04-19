"""Exhaustive composition tests for Affine Collapse Functor (ACF)."""

import math

import pytest
import torch

from acf_functor.core import HornerReducer, ChebyshevReducer
from acf_functor.composition import (
    DomainMapper,
    FunctorialComposer,
    LipschitzEstimator,
    PolynomialComposer,
)


def _coeffs_of(result):
    return result.metadata.get("monomial_coefficients", result.metadata.get("coefficients", []))


@pytest.fixture
def composer():
    return FunctorialComposer(dtype=torch.float64)


class TestPolyPoly:
    def test_linear_linear(self, composer):
        phi_f = HornerReducer.reduce([1.0, 2.0])
        phi_g = HornerReducer.reduce([-1.0, 3.0])
        result = composer.compose(phi_f, phi_g, input_domain=(-5.0, 5.0))

        x = torch.linspace(-5.0, 5.0, 5000, dtype=torch.float64)
        y = HornerReducer.execute_horner(torch.tensor(_coeffs_of(result), dtype=torch.float64), x)
        expected = 2.0 * (3.0 * x - 1.0) + 1.0

        assert torch.allclose(y, expected, atol=1e-12)
        assert result.epsilon_bound == 0.0

    def test_degree_multiplication(self, composer):
        phi_f = HornerReducer.reduce([1.0, 2.0, 3.0, 4.0])
        phi_g = HornerReducer.reduce([1.0, 1.0, 1.0])
        result = composer.compose(phi_f, phi_g, input_domain=(-1.0, 1.0))
        assert len(_coeffs_of(result)) <= 7


class TestTransPoly:
    def test_sin_of_linear(self, composer):
        phi_f = ChebyshevReducer.reduce("sin", degree=24, domain=(-7.0, 7.0))
        phi_g = HornerReducer.reduce([1.0, 2.0])

        result, cert = composer.compose_and_certify(
            phi_f,
            phi_g,
            f_exact=torch.sin,
            g_exact=lambda x: 2.0 * x + 1.0,
            input_domain=(-3.0, 3.0),
        )

        x = torch.linspace(-3.0, 3.0, 5000, dtype=torch.float64)
        y = HornerReducer.execute_horner(torch.tensor(_coeffs_of(result), dtype=torch.float64), x)
        expected = torch.sin(2.0 * x + 1.0)

        err = torch.max(torch.abs(y - expected)).item()
        assert err < 1e-4
        assert cert.verification_error <= cert.epsilon_total * 1.1 + 1e-10


class TestPolyTrans:
    def test_quadratic_of_sin(self, composer):
        phi_f = HornerReducer.reduce([0.0, 0.0, 1.0])
        phi_g = ChebyshevReducer.reduce("sin", degree=24, domain=(-math.pi, math.pi))

        result, _ = composer.compose_and_certify(
            phi_f,
            phi_g,
            f_exact=lambda x: x**2,
            g_exact=torch.sin,
            input_domain=(-math.pi, math.pi),
        )

        x = torch.linspace(-math.pi, math.pi, 5000, dtype=torch.float64)
        y = HornerReducer.execute_horner(torch.tensor(_coeffs_of(result), dtype=torch.float64), x)
        expected = torch.sin(x) ** 2

        err = torch.max(torch.abs(y - expected)).item()
        assert err < 2e-3


class TestTransTrans:
    def test_sin_of_exp(self, composer):
        phi_f = ChebyshevReducer.reduce("sin", degree=24, domain=(-3.0, 3.0))
        phi_g = ChebyshevReducer.reduce("exp", degree=24, domain=(-1.0, 1.0))

        _, cert = composer.compose_and_certify(
            phi_f,
            phi_g,
            f_exact=torch.sin,
            g_exact=torch.exp,
            input_domain=(-1.0, 1.0),
        )

        assert cert.epsilon_total < 1.0


class TestCertificates:
    def test_lipschitz_bound_polynomial(self):
        coeffs = [1.0, -2.0, 3.0, -0.5]
        reduction = HornerReducer.reduce(coeffs)
        L = LipschitzEstimator.estimate_from_reduction(reduction, (-5.0, 5.0))

        x = torch.linspace(-5.0, 5.0, 20000, dtype=torch.float64)
        deriv = -2.0 + 6.0 * x - 1.5 * x**2
        actual = torch.abs(deriv).max().item()
        assert L >= actual * 0.99

    def test_domain_mapping_detection(self):
        assert DomainMapper.needs_remapping((-5.0, 5.0), (-1.0, 1.0))
        assert not DomainMapper.needs_remapping((-0.5, 0.5), (-1.0, 1.0))


class TestPolynomialAlgebra:
    def test_identity_composition(self):
        p = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        identity = torch.tensor([0.0, 1.0], dtype=torch.float64)
        result = PolynomialComposer.compose_polynomials(p, identity)
        assert torch.allclose(result[: p.numel()], p, atol=1e-12)

    def test_square_substitution(self):
        f = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        g = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = PolynomialComposer.compose_polynomials(f, g)
        expected = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        assert torch.allclose(result[:3], expected, atol=1e-12)
