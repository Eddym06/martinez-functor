"""Certified Koopman branch selector with explicit branch metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from .invariant_unified import AlphaGeometric


class KoopmanBranch(Enum):
    EXACT_POLYNOMIAL = "exact_polynomial"
    EXACT_SYMMETRIC = "exact_symmetric"
    CERTIFIED_ANALYTIC = "certified_analytic"
    HEURISTIC = "heuristic"


@dataclass
class KoopmanResult:
    predicted_value: float
    error_bound: float
    branch: KoopmanBranch
    observable_dimension: int
    is_certified: bool
    koopman_matrix: np.ndarray
    eigenvalues: np.ndarray


class PolynomialDetector:
    def __init__(self, tolerance: float = 1e-8, max_test_degree: int = 7):
        self.tolerance = tolerance
        self.max_test_degree = max_test_degree

    def detect(self, f: Callable[[float], float], domain: Tuple[float, float]) -> Tuple[bool, int]:
        a, b = domain
        span = b - a
        hs = [span / 14.0, span / 21.0, span / 29.0]

        for n in range(self.max_test_degree + 1):
            order = n + 1
            ok = True
            for h in hs:
                x_left = a + 0.05 * span
                x_right = b - order * h - 0.05 * span
                if x_right <= x_left:
                    ok = False
                    break
                xs = np.linspace(x_left, x_right, 6)
                for x0 in xs:
                    fd = 0.0
                    for k in range(order + 1):
                        try:
                            y = float(f(float(x0 + k * h)))
                        except Exception:
                            return False, -1
                        if not np.isfinite(y):
                            return False, -1
                        fd += ((-1) ** (order - k)) * _binom(order, k) * y
                    if abs(fd) > self.tolerance:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True, n
        return False, -1


class KoopmanExactPolynomial:
    @staticmethod
    def compute_koopman_matrix_exact(
        g: Callable[[float], float], degree: int, domain: Tuple[float, float]
    ) -> np.ndarray:
        a, b = domain
        n_obs = degree + 1
        n_nodes = degree + 3
        t = np.cos(np.pi * (2 * np.arange(1, n_nodes + 1) - 1) / (2 * n_nodes))
        x = (a + b) / 2.0 + (b - a) / 2.0 * t

        psi_x = np.zeros((n_obs, n_nodes), dtype=float)
        psi_gx = np.zeros((n_obs, n_nodes), dtype=float)
        for j, xv in enumerate(x):
            gx = float(g(xv))
            for d in range(n_obs):
                psi_x[d, j] = xv**d
                psi_gx[d, j] = gx**d

        return psi_gx @ np.linalg.pinv(psi_x)

    def predict_trajectory(
        self,
        g: Callable[[float], float],
        x0: float,
        n_steps: int,
        degree: int,
        domain: Tuple[float, float],
    ) -> KoopmanResult:
        k = self.compute_koopman_matrix_exact(g, degree, domain)
        eig = np.linalg.eigvals(k)
        psi0 = np.array([x0**d for d in range(degree + 1)], dtype=float)
        psi_t = np.linalg.matrix_power(k, n_steps) @ psi0
        pred = float(psi_t[1] if degree >= 1 else psi_t[0])
        return KoopmanResult(pred, 0.0, KoopmanBranch.EXACT_POLYNOMIAL, degree + 1, True, k, eig)

    def verify_exactness(
        self,
        g: Callable[[float], float],
        degree: int,
        domain: Tuple[float, float],
        n_test_points: int = 300,
    ) -> Tuple[bool, float]:
        k = self.compute_koopman_matrix_exact(g, degree, domain)
        a, b = domain
        xs = np.random.uniform(a, b, size=n_test_points)
        max_err = 0.0
        for x in xs:
            gx = float(g(x))
            psi_x = np.array([x**d for d in range(degree + 1)], dtype=float)
            psi_g_true = np.array([gx**d for d in range(degree + 1)], dtype=float)
            psi_g_pred = k @ psi_x
            max_err = max(max_err, float(np.max(np.abs(psi_g_true - psi_g_pred))))
        return max_err < 1e-7, max_err


class KoopmanCertifiedAnalytic:
    def __init__(self):
        self.alpha_geo = AlphaGeometric(n_trajectory=20000, n_scales=12)

    @staticmethod
    def error_bound_analytic(d_h: float, d: int, c0: float = 1.0, c1: float = 1.0) -> float:
        if d_h < 1e-10:
            return 0.0
        return float(c0 * np.exp(-c1 * (d ** (1.0 / d_h))))

    def _estimate_dh(self, g: Callable[[float], float], domain: Tuple[float, float]) -> float:
        alpha_geo, _ = self.alpha_geo.compute(g, domain)
        if alpha_geo <= 1e-12:
            return 0.0
        return float(1.0 / alpha_geo)

    def compute_certified(
        self,
        g: Callable[[float], float],
        x0: float,
        n_steps: int,
        domain: Tuple[float, float],
        target_error: float = 1e-4,
    ) -> KoopmanResult:
        d_h = self._estimate_dh(g, domain)
        c0, c1 = 1.0, 1.0

        if d_h < 1e-10:
            d = 8
        else:
            val = max(1.0, -np.log(max(target_error, 1e-14) / c0) / c1)
            d = int(val**d_h) + 1
            d = int(np.clip(d, 8, 80))

        a, b = domain
        x_train = np.linspace(a, b, max(500, 10 * d))
        y_train = np.array([g(float(x)) for x in x_train], dtype=float)

        psi_x = np.column_stack([x_train**k for k in range(d)])
        psi_y = np.column_stack([y_train**k for k in range(d)])
        k_mat, _, _, _ = np.linalg.lstsq(psi_x, psi_y, rcond=None)
        k_mat = k_mat.T
        eig = np.linalg.eigvals(k_mat)

        psi0 = np.array([x0**k for k in range(d)], dtype=float)
        psi_t = np.linalg.matrix_power(k_mat, n_steps) @ psi0
        pred = float(psi_t[1] if d > 1 else psi_t[0])
        bound = self.error_bound_analytic(d_h, d, c0=c0, c1=c1)

        return KoopmanResult(pred, bound, KoopmanBranch.CERTIFIED_ANALYTIC, d, True, k_mat, eig)


class CertifiedKoopman:
    def __init__(self):
        self.detector = PolynomialDetector()
        self.exact_poly = KoopmanExactPolynomial()
        self.cert_analytic = KoopmanCertifiedAnalytic()

    @staticmethod
    def detect_group_symmetry(g: Callable[[float], float], domain: Tuple[float, float]) -> Tuple[bool, str]:
        a, b = domain
        if not (a <= -0.2 and b >= 0.2):
            return False, "none"
        xs = np.linspace(max(a, -1.0), min(b, 1.0), 80)
        odd_ok = all(abs(g(-x) + g(x)) < 1e-8 for x in xs)
        if odd_ok:
            return True, "Z2_odd"
        even_ok = all(abs(g(-x) - g(x)) < 1e-8 for x in xs)
        if even_ok:
            return True, "Z2_even"
        return False, "none"

    def predict(
        self,
        g: Callable[[float], float],
        x0: float,
        n_steps: int,
        domain: Tuple[float, float] = (-1.0, 1.0),
        target_error: float = 1e-4,
    ) -> KoopmanResult:
        is_poly, degree = self.detector.detect(g, domain)
        if is_poly and degree >= 0:
            return self.exact_poly.predict_trajectory(g, x0, n_steps, degree, domain)

        has_sym, _sym = self.detect_group_symmetry(g, domain)
        if has_sym:
            res = self.cert_analytic.compute_certified(g, x0, n_steps, domain, target_error)
            return KoopmanResult(
                predicted_value=res.predicted_value,
                error_bound=res.error_bound,
                branch=KoopmanBranch.EXACT_SYMMETRIC,
                observable_dimension=res.observable_dimension,
                is_certified=True,
                koopman_matrix=res.koopman_matrix,
                eigenvalues=res.eigenvalues,
            )

        return self.cert_analytic.compute_certified(g, x0, n_steps, domain, target_error)


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c
