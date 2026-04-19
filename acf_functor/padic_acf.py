"""
p-adic ACF — Analytic Functions over p-adic Numbers
====================================================

The p-adic numbers ℚ_p (Hensel's construction) are the completion of ℚ
with respect to the p-adic absolute value |x|_p = p^{-v_p(x)}.

A function f: ℚ_p → ℚ_p is *p-adically analytic* if it can be written
as a Mahler expansion or a power series converging in the p-adic metric:

    f(x) = Σ_{n=0}^∞ aₙ · C(x, n)     (Mahler series)
    f(x) = Σ_{n=0}^∞ cₙ · xⁿ          (power series, |cₙ|→0 p-adically)

where C(x,n) = x(x-1)…(x-n+1)/n! are the binomial coefficients.

ACF Extension
-------------
The ACF Horner algorithm works over any ring (see algebraic_acf.py).
For ℤ_p (p-adic integers, the "unit disk" |x|_p ≤ 1), the
power series f(x) = Σ cₙxⁿ converges whenever |cₙ|_p → 0.

Key difference from ℝ: convergence is AUTOMATIC for integer-coefficient
polynomials via Krasner's lemma — no Chebyshev tricks needed.

ACF Invariant over ℤ_p
-----------------------
  α_p(f) = -log(sup_n |aₙ|_p) / log(n)  → rate of p-adic coefficient decay
  d*(ε, α_p) = smallest d such that |Σ_{n>d} aₙ C(x,n)|_p ≤ ε

Applications
-----------
  - Number theory: L-functions, zeta functions (Bernoulli numbers → p-adic)
  - Cryptography: p-adic lifting of elliptic curve operations (LIFT-and-REDUCE)
  - Modular arithmetic: GF(p) → ℤ/p^k ℤ via Hensel lifting

Key Theorems
------------
  PADIC-1 (Hensel's Lemma):
    If f(a) ≡ 0 (mod p) and f'(a) ≢ 0 (mod p), then ∃ unique x ∈ ℤ_p
    with f(x) = 0 and x ≡ a (mod p). This is the p-adic Newton's method.

  PADIC-2 (Mahler's Theorem):
    Every continuous function f: ℤ_p → ℚ_p has a unique Mahler expansion
    f(x) = Σ aₙ C(x,n) with aₙ → 0.
    FMA count = d (Horner over ℤ_p, same as real case).

  PADIC-3 (ACF Lift):
    The lift of a GF(p) FMA sequence to ℤ_p via Hensel gives a p-adic
    FMA sequence with the same FMA count. E(f_lift) = E(f_GFp).

References
----------
  Hensel (1908) — p-adic numbers.
  Mahler (1958) — An interpolation series for continuous functions of a p-adic variable.
  Robert (2000) — A Course in p-adic Analysis.
  Paper.md §55 (p-adic ACF extension).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# § 1  p-adic Valuation and Absolute Value
# ===========================================================================

def p_adic_valuation(n: int, p: int) -> int:
    """
    Compute v_p(n) = largest k such that p^k | n.
    v_p(0) = +∞ (conventionally, we return a large number).
    """
    if n == 0:
        return 10**9
    v = 0
    n = abs(n)
    while n % p == 0:
        n //= p
        v += 1
    return v


def p_adic_abs(n: int, p: int) -> float:
    """
    p-adic absolute value: |n|_p = p^{-v_p(n)}.
    |0|_p = 0.
    """
    if n == 0:
        return 0.0
    return float(p) ** (-p_adic_valuation(n, p))


def p_adic_abs_rational(num: int, den: int, p: int) -> float:
    """p-adic absolute value of rational num/den."""
    if num == 0:
        return 0.0
    return p_adic_abs(num, p) / p_adic_abs(den, p)


# ===========================================================================
# § 2  p-adic Series Representation
# ===========================================================================

@dataclass
class PAdicExpansion:
    """
    A truncated p-adic expansion of an element x ∈ ℤ_p.

    x = a₀ + a₁p + a₂p² + … + a_{k-1}p^{k-1}  (mod p^k)

    Each digit aᵢ ∈ {0, 1, …, p-1}.
    """
    digits: List[int]   # digits[i] = coefficient of p^i
    p: int
    precision: int      # number of terms = k

    @classmethod
    def from_integer(cls, n: int, p: int, precision: int = 10) -> "PAdicExpansion":
        """Expand integer n ≥ 0 in base p to given precision (k digits)."""
        digits = []
        m = n % (p ** precision)
        for _ in range(precision):
            digits.append(m % p)
            m //= p
        return cls(digits=digits, p=p, precision=precision)

    def to_integer_mod_pk(self) -> int:
        """Recover the integer n = Σ digits[i] * p^i   (mod p^k)."""
        result = 0
        pk = 1
        for d in self.digits:
            result += d * pk
            pk *= self.p
        return result

    def p_adic_norm(self) -> float:
        """Norm |x|_p = p^{-v} where v = smallest non-zero digit index."""
        for i, d in enumerate(self.digits):
            if d != 0:
                return float(self.p) ** (-i)
        return 0.0  # x = 0

    def __repr__(self) -> str:
        return f"[{'+'.join(f'{d}·{self.p}^{i}' for i,d in enumerate(self.digits) if d!=0)}]_p={self.p}"


# ===========================================================================
# § 3  Mahler Series Expansion
# ===========================================================================

@dataclass
class MahlerSeries:
    """
    Mahler expansion of a p-adic analytic function.

    f(x) = Σ_{n=0}^d aₙ · C(x,n)   where C(x,n) = x(x-1)…(x-n+1)/n!

    The coefficients aₙ = Δⁿ f(0) (n-th finite difference at 0).
    """
    coefficients: List[int]   # aₙ (integer or rational)
    p: int
    degree: int               # truncation degree d

    @classmethod
    def from_function_values(
        cls,
        f_values: List[int],
        p: int,
    ) -> "MahlerSeries":
        """
        Compute Mahler coefficients from function values f(0), f(1), …, f(d).

        aₙ = Σ_{k=0}^n (-1)^{n-k} C(n,k) f(k)  (n-th finite difference)
        """
        d = len(f_values) - 1
        coeffs = []
        for n in range(d + 1):
            an = 0
            for k in range(n + 1):
                sign = (-1) ** (n - k)
                binom = math.comb(n, k)
                an += sign * binom * f_values[k]
            coeffs.append(an)
        return cls(coefficients=coeffs, p=p, degree=d)

    def evaluate(self, x: int) -> int:
        """Evaluate f(x) = Σ aₙ C(x,n) for non-negative integer x."""
        result = 0
        for n, an in enumerate(self.coefficients):
            if an == 0:
                continue
            # C(x, n) = x(x-1)…(x-n+1) / n!
            binom = math.comb(x, n)
            result += an * binom
        return result

    def fma_count(self) -> int:
        """Horner over the Mahler basis: d FMAs."""
        return self.degree

    def p_adic_coefficient_norms(self) -> List[float]:
        """p-adic absolute values |aₙ|_p for each coefficient."""
        return [p_adic_abs(an, self.p) for an in self.coefficients]

    def alpha(self) -> float:
        """
        p-adic alpha invariant: decay rate of |aₙ|_p as n → ∞.

        For continuous f: ℤ_p → ℤ_p, Mahler's theorem guarantees |aₙ|_p → 0.
        """
        norms = self.p_adic_coefficient_norms()
        if len(norms) < 3:
            return 1.0
        # Fit log|aₙ| ≈ log A - α·n
        norms_arr = np.array([n for n in norms if n > 0])
        if len(norms_arr) < 2:
            return 1.0
        k = np.arange(len(norms_arr), dtype=float)
        log_norms = np.log(norms_arr + 1e-300)
        slope = np.polyfit(k, log_norms, 1)[0]
        return max(0.0, -slope / math.log(self.p))


# ===========================================================================
# § 4  Hensel Lifting (PADIC-1)
# ===========================================================================

@dataclass
class HenselLiftResult:
    """Result of Hensel's lemma lifting f(a) ≡ 0 (mod p) → x ∈ ℤ_p."""
    root_mod_pk: int           # the lifted root mod p^precision
    p: int
    precision: int             # k = number of Hensel steps
    fma_count: int             # Newton iterations × FMAs per step
    converged: bool
    lift_sequence: List[int]   # root at each precision level
    certificates: List[str] = field(default_factory=lambda: ["PADIC-1"])


def hensel_lift(
    f_coeffs: List[int],
    a0: int,
    p: int,
    precision: int = 8,
) -> HenselLiftResult:
    """
    Lift a root a₀ of f mod p to a root mod p^precision via Hensel's lemma.

    f_coeffs = polynomial coefficients [c₀, c₁,…,cₙ] with f(x) = Σcᵢxⁱ.
    a₀ must satisfy f(a₀) ≡ 0 (mod p) and f'(a₀) ≢ 0 (mod p).

    Newton–Raphson p-adically:
        aₖ₊₁ = aₖ - f(aₖ)/f'(aₖ)    (exact division in ℤ/p^{k+1}ℤ)

    Each lift step requires one polynomial evaluation (FMA count = deg f).
    """
    def poly_eval(coeffs: List[int], x: int, mod: int) -> int:
        """Horner evaluation of coeffs at x mod pk."""
        acc = 0
        for c in reversed(coeffs):
            acc = (acc * x + c) % mod
        return acc

    def poly_deriv_eval(coeffs: List[int], x: int, mod: int) -> int:
        """Evaluate formal derivative at x mod mod."""
        d_coeffs = [i * coeffs[i] for i in range(1, len(coeffs))]
        return poly_eval(d_coeffs, x, mod)

    lift_seq = [a0 % p]
    fma_per_step = len(f_coeffs) - 1  # Horner step count
    total_fmas = 0
    a = a0 % p
    pk = p
    converged = True

    for k in range(1, precision):
        pk_next = pk * p
        fa = poly_eval(f_coeffs, a, pk_next)
        dfa_mod_p = poly_deriv_eval(f_coeffs, a, p)
        total_fmas += fma_per_step * 2  # eval f + eval f'

        if dfa_mod_p == 0:
            converged = False
            break

        # a_new = a - f(a)/f'(a) mod p^{k+1}
        dfa_inv = pow(dfa_mod_p, -1, p)          # inverse of f'(a) mod p
        correction = (fa * dfa_inv) % pk_next
        a = (a - correction) % pk_next
        pk = pk_next
        lift_seq.append(a % pk)

    return HenselLiftResult(
        root_mod_pk=a,
        p=p,
        precision=len(lift_seq),
        fma_count=total_fmas,
        converged=converged,
        lift_sequence=lift_seq,
    )


# ===========================================================================
# § 5  ACF Reduction for p-adic Functions
# ===========================================================================

@dataclass
class PAdicACFReport:
    """Full ACF analysis report for a p-adic analytic function."""
    p: int
    mahler_series: MahlerSeries
    alpha: float
    normalized_alpha: float
    fma_count: int
    d_star: int               # minimum degree for ε-approximation
    hensel_lift: Optional[HenselLiftResult]
    certificates: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PAdicACFReducer:
    """
    ACF reduction for p-adically analytic functions.

    Given a function f: {0,1,…,N} → ℤ (integer values), computes:
      1. Mahler series expansion aₙ = Δⁿ f(0)
      2. p-adic alpha invariant (decay of Mahler coefficients)
      3. Optimal truncation degree d*(ε, α_p)
      4. Horner FMA count over ℤ (= degree, same as real case)
      5. Optionally, Hensel lift of a root
    """

    def __init__(self, p: int):
        self.p = p

    def reduce(
        self,
        f_values: List[int],
        epsilon: float = 1e-4,
    ) -> PAdicACFReport:
        """
        Reduce a p-adically analytic function given by its values.

        f_values = [f(0), f(1), …, f(d)] — values at non-negative integers.
        """
        mahler = MahlerSeries.from_function_values(f_values, self.p)
        alpha = mahler.alpha()
        normalized_alpha = 1.0 / (1.0 + max(alpha, 0.0))

        # d*(ε, α_p): find smallest d where |aₙ|_p < ε for all n > d
        norms = mahler.p_adic_coefficient_norms()
        d_star = len(norms) - 1
        for i, norm in enumerate(norms):
            if all(norms[j] < epsilon for j in range(i, len(norms))):
                d_star = i
                break

        fma_count = d_star

        certs = ["PADIC-2 (Mahler)", "ALGACF-1 (Horner over ℤ)"]

        # Try Hensel lift if there's a root
        hensel = None
        coeffs_int = [int(round(c)) for c in f_values[:5]] if len(f_values) >= 1 else []
        for a0 in range(self.p):
            ev = sum(coeffs_int[i] * (a0 ** i) for i in range(len(coeffs_int))) % self.p
            if ev == 0 and len(coeffs_int) >= 2:
                try:
                    hensel = hensel_lift(coeffs_int, a0, self.p, precision=6)
                    certs.append("PADIC-1 (Hensel)")
                except Exception:
                    pass
                break

        return PAdicACFReport(
            p=self.p,
            mahler_series=mahler,
            alpha=alpha,
            normalized_alpha=normalized_alpha,
            fma_count=fma_count,
            d_star=d_star,
            hensel_lift=hensel,
            certificates=certs,
            metadata={"n_values": len(f_values), "epsilon": epsilon},
        )
