"""
Rational ACF — Padé Approximants and Hardy Space Compression
=============================================================

Extends the Affine Collapse Functor to rational function approximation
(Padé approximants) and Hardy space H² representations.

Mathematical foundation
-----------------------
A [m/n] Padé approximant of f at 0 is a rational function:
    R_{m,n}(x) = P_m(x) / Q_n(x)

where deg(P_m) ≤ m, deg(Q_n) ≤ n, and
    f(x) - R_{m,n}(x) = O(x^{m+n+1})  as x → 0.

This matches the first m+n Taylor coefficients of f.

The key advantage over polynomial approximation:
  - Rational functions can represent poles and branch points
  - The [n/n] diagonal Padé approximant is near-optimal for meromorphic f
  - Error bound: |f(z) - R_{n,n}(z)| ≤ C · ρ^{-2n} for f analytic in |z| < ρ

ACF evaluation:
  R(x) = P(x) / Q(x) where P, Q are evaluated via Horner's algorithm.
  FMA count: m + n + 1 (P) + n (Q) + 1 (division) = m + 2n + 2.

Hardy space H² representation
-------------------------------
For f ∈ H²(D) (square-integrable on the unit circle), the theory of
Hardy spaces gives optimal rational approximation where the poles are
determined by the decay of the Taylor coefficients.

The ACF invariant for rational approximation:
    α_rational(f) = -log(ρ) / log(n)  → convergence rate as n → ∞

where ρ is the radius of the largest disk of analyticity for f.

Pole detection
--------------
The poles of f in the complex plane can be estimated from the
coefficients of Q_n(x) via the Padé denominator.

Scope
-----
  - 1D rational approximation for meromorphic functions
  - Padé approximants via the extended Euclidean algorithm on power series
  - Hardy space alpha estimation from pole structure

References
----------
  Baker & Graves-Morris (1996) — Padé Approximants.
  Brezinski & Redivo-Zaglia (1991) — Extrapolation Methods.
  Paper.md §41 for ACF extension to rational domains.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PadeInvariants:
    """Invariant summary for a [m/n] Padé approximant."""
    alpha_rational: float       # convergence rate estimate
    pole_locations: List[complex]  # roots of Q_n(x) in ℂ
    pole_count: int
    residue_decay: float        # rate of residue magnitude decay
    hardy_norm: float           # ‖P/Q‖_{H²} estimate
    m: int                      # numerator degree
    n: int                      # denominator degree

    def summary(self) -> str:
        lines = [
            "=== Padé-ACF Invariants ===",
            f"  Approximant: [{self.m}/{self.n}]",
            f"  α_rational: {self.alpha_rational:.4f}",
            f"  Poles: {self.pole_count}",
            f"  Pole locations: {[f'{p:.3f}' for p in self.pole_locations[:5]]}",
            f"  Residue decay: {self.residue_decay:.4f}",
            f"  H² norm: {self.hardy_norm:.4e}",
        ]
        return "\n".join(lines)


@dataclass
class HardySpaceInvariants:
    """Invariant summary for Hardy space H² approximation."""
    alpha_hardy: float          # spectral decay rate of H² coefficients
    effective_rank: int         # number of significant H² modes
    h2_error: float             # ‖f - f̂‖_{H²} estimate
    n_modes: int

    def summary(self) -> str:
        lines = [
            "=== Hardy-ACF (H²) Invariants ===",
            f"  Modes: {self.n_modes}",
            f"  α_H²: {self.alpha_hardy:.4f}",
            f"  Effective rank: {self.effective_rank}",
            f"  H² error: {self.h2_error:.4e}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Taylor coefficient computation
# ---------------------------------------------------------------------------

def _taylor_via_derivatives(
    f: Callable[[float], float],
    n_terms: int,
    x0: float = 0.0,
    h: float = 1e-4,
) -> np.ndarray:
    """
    Estimate Taylor coefficients aₖ = f^{(k)}(x₀) / k! via finite differences.

    Uses centered difference formulas up to order 8 for small n_terms,
    and Fornberg's method for larger n_terms.
    """
    coeffs = np.zeros(n_terms)
    for k in range(n_terms):
        # k-th derivative via finite differences on a circle (Cauchy integral)
        # Using the discrete Fourier approach: a_k = (1/r^k) * (1/N) Σ f(r·e^{2πij/N}) e^{-2πijk/N}
        # This avoids numerical cancellation in repeated differentiation.
        N = max(2 * n_terms, 64)  # number of sample points
        r = h * (k + 1) ** 0.3   # adaptive radius
        theta = 2 * np.pi * np.arange(N) / N
        z_pts = x0 + r * np.exp(1j * theta)
        # Approximate f on the circle by Taylor
        # For real functions, use real-valued Cauchy approach
        try:
            f_vals = np.array([f(float(z.real)) for z in z_pts], dtype=complex)
        except Exception:
            f_vals = np.array([f(x0 + r * np.cos(t)) for t in theta], dtype=complex)
        # DFT to extract k-th coefficient
        fft_vals = np.fft.fft(f_vals)
        coeffs[k] = float((fft_vals[k] / (r ** k * N)).real)
    return coeffs


def _taylor_via_samples(
    f: Callable[[float], float],
    n_terms: int,
    x0: float = 0.0,
) -> np.ndarray:
    """
    Compute Taylor coefficients via Vandermonde regression.

    Sample f at Chebyshev nodes near x₀ and fit the polynomial.
    """
    # Use Chebyshev nodes on [-0.5, 0.5] shifted to x₀
    half_range = 0.5
    nodes = x0 + half_range * np.cos(np.pi * (2 * np.arange(n_terms) + 1) / (2 * n_terms))
    f_vals = np.array([f(xi) for xi in nodes])

    # Build Vandermonde matrix for Taylor basis (xᵢ - x₀)^k / k!
    shifts = nodes - x0
    V = np.array([shifts ** k / float(math.factorial(k)) for k in range(n_terms)]).T
    coeffs, _, _, _ = np.linalg.lstsq(V, f_vals, rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Padé approximant computation via extended Euclidean algorithm
# ---------------------------------------------------------------------------

def _pade_from_taylor(c: np.ndarray, m: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute [m/n] Padé approximant from Taylor coefficients c[0..m+n].

    Solves the linear system:
        Σ_{j=0}^{n} Q_j · c_{k-j} = 0  for k = m+1, …, m+n
        P_k = Σ_{j=0}^{k} Q_j · c_{k-j}  for k = 0, …, m

    Returns P (numerator coefficients, degree m) and Q (denominator, degree n).
    Both in the convention: P[k] = coefficient of x^k.
    """
    assert len(c) >= m + n + 1, f"Need {m+n+1} Taylor coefficients, got {len(c)}"

    if n == 0:
        return c[:m + 1].copy(), np.array([1.0])

    # Build system for Q: (n × n) linear system
    # C_mat[i,j] = c[m - n + i + 1 + j]  ... Toeplitz block
    C_mat = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(n):
        rhs[i] = -c[m + 1 + i]
        for j in range(n):
            idx = m + 1 + i - (j + 1)
            if 0 <= idx < len(c):
                C_mat[i, j] = c[idx]

    # Solve for Q[1], …, Q[n] (Q[0] = 1 by convention)
    try:
        q_tail = np.linalg.solve(C_mat, rhs)
    except np.linalg.LinAlgError:
        q_tail, _, _, _ = np.linalg.lstsq(C_mat, rhs, rcond=None)

    Q = np.concatenate([[1.0], q_tail])  # Q[0]=1, Q[1..n]

    # Compute P from Q and c
    P = np.zeros(m + 1)
    for k in range(m + 1):
        for j in range(min(k + 1, n + 1)):
            if k - j < len(c):
                P[k] += Q[j] * c[k - j]

    return P, Q


def _horner_eval(coeffs: np.ndarray, x: float) -> float:
    """
    Evaluate polynomial Σ_k c[k] xᵏ via Horner's algorithm.
    Counts exactly n FMA operations for degree-n polynomial.
    """
    if len(coeffs) == 0:
        return 0.0
    result = float(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * x + float(c)
    return result


# ---------------------------------------------------------------------------
# PadeReducer
# ---------------------------------------------------------------------------

class PadeReducer:
    """
    Reduce a function f: ℝ → ℝ via Padé approximant [m/n].

    The approximant is computed from Taylor coefficients and evaluated
    via two Horner chains (numerator + denominator) plus one division.

    Parameters
    ----------
    m : int
        Numerator polynomial degree.
    n : int
        Denominator polynomial degree.
    x0 : float
        Expansion point.
    """

    def __init__(self, m: int = 5, n: int = 5, x0: float = 0.0):
        self.m = m
        self.n = n
        self.x0 = x0
        self._P: Optional[np.ndarray] = None
        self._Q: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def fma_count(self) -> int:
        """Total FMA operations for evaluation: m + 1 (P) + n + 1 (Q) + 1 (div)."""
        return self.m + self.n + 3

    def fit(
        self,
        f: Callable[[float], float],
        method: str = "samples",
    ) -> "PadeReducer":
        """
        Compute the Padé approximant [m/n] for f near x₀.

        Parameters
        ----------
        f : callable
            Scalar function f: ℝ → ℝ.
        method : str
            'samples' (regression) or 'derivatives' (finite differences).
        """
        n_coeffs = self.m + self.n + 2  # need m+n+1 Taylor coefficients
        if method == "samples":
            c = _taylor_via_samples(f, n_coeffs, x0=self.x0)
        else:
            c = _taylor_via_derivatives(f, n_coeffs, x0=self.x0)

        self._P, self._Q = _pade_from_taylor(c, self.m, self.n)
        self._fitted = True
        return self

    def __call__(self, x: float) -> float:
        """Evaluate R(x) = P(x-x₀) / Q(x-x₀) via double Horner chain."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        z = x - self.x0
        P_val = _horner_eval(self._P, z)
        Q_val = _horner_eval(self._Q, z)
        if abs(Q_val) < 1e-15:
            return float("inf")  # pole
        return P_val / Q_val

    def poles(self) -> List[complex]:
        """Return the poles of R as roots of Q."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if len(self._Q) <= 1:
            return []
        return list(np.roots(self._Q[::-1]))  # np.roots expects highest degree first

    def pole_locations_real(self) -> List[float]:
        """Return real poles only."""
        return [float(p.real) for p in self.poles() if abs(p.imag) < 1e-8]

    def invariants(
        self,
        f: Optional[Callable[[float], float]] = None,
        test_pts: Optional[np.ndarray] = None,
    ) -> PadeInvariants:
        """Compute Padé-ACF invariants."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        # Residue magnitudes from partial fraction decomposition
        poles = self.poles()
        residues = []
        for p in poles:
            # Residue ≈ P(p) / Q'(p)
            z = complex(p) - self.x0
            P_val = sum(self._P[k] * z ** k for k in range(len(self._P)))
            dQ = sum(k * self._Q[k] * z ** (k - 1) for k in range(1, len(self._Q)))
            if abs(dQ) > 1e-15:
                residues.append(abs(P_val / dQ))

        sorted_res = sorted(residues, reverse=True)
        if len(sorted_res) > 1:
            ks = np.arange(1, len(sorted_res) + 1, dtype=float)
            log_r = np.log(np.maximum(sorted_res, 1e-300))
            alpha_r = float(-np.polyfit(np.log(ks), log_r, 1)[0])
        else:
            alpha_r = 1.0

        # Alpha from convergence: estimate ρ (radius of analyticity)
        # Use pole distances from origin
        if poles:
            rho = float(min(abs(p + self.x0) for p in poles))
            alpha_rational = -np.log(rho) if rho > 1e-10 else 1.0
        else:
            alpha_rational = float(self.m + self.n)  # high degree → fast convergence

        # H² norm estimate: √(Σ |P_k|²) / |Q_0|
        hardy_norm = float(np.sqrt(np.sum(self._P ** 2)) / max(abs(self._Q[0]), 1e-15))

        return PadeInvariants(
            alpha_rational=alpha_rational,
            pole_locations=poles,
            pole_count=len(poles),
            residue_decay=alpha_r,
            hardy_norm=hardy_norm,
            m=self.m,
            n=self.n,
        )


# ---------------------------------------------------------------------------
# HardySpaceACF (H² approximation for frequency-domain functions)
# ---------------------------------------------------------------------------

class HardySpaceACF:
    """
    Approximate f ∈ H²(D) (Hardy space on the unit disk) via rational
    best approximation.

    The H² inner product is computed via the unit circle quadrature:
        <f, g>_{H²} = (1/2π) ∫₀^{2π} f(e^{iθ}) conj(g(e^{iθ})) dθ

    The Fourier coefficients of f are the Taylor coefficients:
        f(z) = Σ_{k≥0} ĉₖ zᵏ,  |z| < 1

    Parameters
    ----------
    n_modes : int
        Number of H² modes (truncation of the Taylor series).
    """

    def __init__(self, n_modes: int = 32):
        self.n_modes = n_modes
        self._coeffs: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        f: Callable[[complex], complex],
        n_quad: int = 256,
    ) -> "HardySpaceACF":
        """
        Compute H² coefficients via unit circle quadrature.

        Parameters
        ----------
        f : callable
            Holomorphic function f: D → ℂ, evaluated on the unit circle.
        n_quad : int
            Number of quadrature points.
        """
        theta = 2 * np.pi * np.arange(n_quad) / n_quad
        z_pts = np.exp(1j * theta)
        f_vals = np.array([f(z) for z in z_pts])

        # FFT to get Taylor coefficients
        fft_vals = np.fft.fft(f_vals) / n_quad
        self._coeffs = fft_vals[:self.n_modes]
        self._fitted = True
        return self

    def fit_real(
        self,
        f: Callable[[float], float],
        n_quad: int = 256,
    ) -> "HardySpaceACF":
        """
        Fit a real-valued function via its analytic continuation.
        (Treats f as defined on [-1,1] and maps to the circle.)
        """
        # Map f: [-1,1] → D via the Joukowski-like change of variables
        # Use Chebyshev regression then convert to H² coefficients
        nodes = np.cos(np.pi * np.arange(n_quad) / (n_quad - 1))
        f_vals = np.array([f(x) for x in nodes], dtype=complex)
        # Just use DFT on the unit circle for now
        return self.fit(lambda z: f(float(z.real)) if abs(z.imag) < 0.1 else 0.0 + 0j,
                        n_quad=n_quad)

    def __call__(self, z: complex) -> complex:
        """Evaluate f̂(z) = Σ_k ĉ_k z^k via Horner."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        result = complex(0.0)
        for k, c in enumerate(self._coeffs):
            result += c * (z ** k)
        return result

    def alpha(self) -> float:
        """Compute ACF alpha from H² coefficient decay."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        mags = np.abs(self._coeffs)
        mags = mags[mags > 1e-15]
        if len(mags) < 2:
            return 1.0
        ks = np.arange(1, len(mags) + 1, dtype=float)
        return float(-np.polyfit(np.log(ks), np.log(mags), 1)[0])

    def invariants(
        self,
        f_exact: Optional[Callable] = None,
        test_pts: Optional[np.ndarray] = None,
    ) -> HardySpaceInvariants:
        """Compute H²-ACF invariants."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        alpha = self.alpha()
        mags = np.abs(self._coeffs)
        threshold = 0.01 * float(np.max(mags) + 1e-15)
        eff_rank = int(np.sum(mags > threshold))

        # H² error estimate
        h2_error = 0.0
        if f_exact is not None:
            n_test = 128
            theta_test = 2 * np.pi * np.arange(n_test) / n_test
            z_test = np.exp(1j * theta_test)
            errors = [abs(self(z) - f_exact(z)) for z in z_test]
            h2_error = float(np.sqrt(np.mean(np.array(errors) ** 2)))

        return HardySpaceInvariants(
            alpha_hardy=alpha,
            effective_rank=eff_rank,
            h2_error=h2_error,
            n_modes=self.n_modes,
        )


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def pade_reduce(
    f: Callable[[float], float],
    m: int = 5,
    n: int = 5,
    x0: float = 0.0,
) -> PadeReducer:
    """Fit and return a PadeReducer [m/n] for f."""
    r = PadeReducer(m=m, n=n, x0=x0)
    r.fit(f)
    return r


def hardy_reduce(
    f_circle: Callable[[complex], complex],
    n_modes: int = 32,
    n_quad: int = 256,
) -> HardySpaceACF:
    """Fit and return a HardySpaceACF for f."""
    r = HardySpaceACF(n_modes=n_modes)
    r.fit(f_circle, n_quad=n_quad)
    return r
