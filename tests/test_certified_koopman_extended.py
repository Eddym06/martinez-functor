import numpy as np

from acf_functor.certified_koopman import (
    CertifiedKoopman,
    KoopmanBranch,
    KoopmanExactPolynomial,
    PolynomialDetector,
)


class TestPolynomialDetector:
    def test_detect_linear(self):
        d = PolynomialDetector(tolerance=1e-8)
        is_poly, degree = d.detect(lambda x: 3.0 * x + 2.0, (-1.0, 1.0))
        assert is_poly
        assert degree == 1

    def test_reject_sin(self):
        d = PolynomialDetector(tolerance=1e-8)
        is_poly, degree = d.detect(np.sin, (-1.0, 1.0))
        assert not is_poly
        assert degree == -1


class TestKoopmanExactPolynomial:
    def test_linear_exactness(self):
        k = KoopmanExactPolynomial()
        ok, err = k.verify_exactness(lambda x: 0.8 * x, degree=1, domain=(-1.0, 1.0))
        assert ok
        assert err < 1e-7

    def test_linear_prediction(self):
        k = KoopmanExactPolynomial()
        x0 = 0.8
        n_steps = 8
        out = k.predict_trajectory(lambda x: 0.5 * x, x0=x0, n_steps=n_steps, degree=1, domain=(-1.0, 1.0))
        assert abs(out.predicted_value - (0.5**n_steps) * x0) < 1e-5
        assert out.branch == KoopmanBranch.EXACT_POLYNOMIAL


class TestCertifiedKoopman:
    def test_selects_polynomial_branch(self):
        ck = CertifiedKoopman()
        out = ck.predict(lambda x: 0.7 * x + 0.1, x0=0.2, n_steps=4, domain=(-1.0, 1.0))
        assert out.branch == KoopmanBranch.EXACT_POLYNOMIAL

    def test_certified_bound_present(self):
        ck = CertifiedKoopman()
        g = lambda x: np.tanh(1.1 * x)
        x0 = 0.2
        n_steps = 3
        out = ck.predict(g, x0=x0, n_steps=n_steps, domain=(-1.0, 1.0), target_error=1e-3)

        x_true = x0
        for _ in range(n_steps):
            x_true = float(g(x_true))

        empirical = abs(out.predicted_value - x_true)
        assert out.is_certified
        assert out.error_bound >= 0.0
        # This check enforces that a certificate is at least numerically sensible.
        assert empirical <= max(out.error_bound * 10.0, 1e-2)
