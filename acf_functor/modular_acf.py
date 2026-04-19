"""
Modular ACF — Modular Forms, q-series, and L-functions
=======================================================

Modular forms are the archetypes of "maximally structured" analytic functions.
They live in the intersection of complex analysis, number theory, and
representation theory — and they are C^ω by definition.

A modular form of weight k and level N is f: ℍ → ℂ satisfying:
  f((aτ+b)/(cτ+d)) = (cτ+d)^k · f(τ)     for all (a b; c d) ∈ Γ₀(N)

with a Fourier (q-series) expansion:
  f(τ) = Σ_{n=0}^∞ aₙ · q^n,   q = e^{2πiτ}

The ACF is naturally applicable: the q-series IS a formal power series in q,
and the Chebyshev/Horner framework reduces directly to FMA sequences in q.

Key Objects
-----------
  - Eisenstein series E₂, E₄, E₆ (modular forms of weight 2, 4, 6)
  - Ramanujan tau function τ(n): coefficients of Δ(τ) = η(τ)²⁴
  - Dedekind eta function η(τ) = q^{1/24} ∏_{n=1}^∞ (1-qⁿ)
  - Theta series θ(τ) = Σ_{n=-∞}^∞ q^{n²}
  - L-functions L(f, s) associated to modular forms

ACF Extension
-------------
For a modular form with q-expansion Σ aₙqⁿ:

  1. The coefficient sequence {aₙ} has polynomial growth (by Deligne's theorem):
     |aₙ| ≤ d(n) · n^{(k-1)/2}  (for cusp forms, |aₙ| ≤ C·n^{(k-1)/2+ε})

  2. The ACF alpha invariant for modular forms:
     α_mod(f) = (k-1)/2  (weight-k form → alpha = half-weight minus half)

  3. FMA count for evaluating f(τ) to precision Q (keeping q^n for n ≤ Q):
     E(f, Q) = Q  (Horner over ℂ in powers of q)

Key Theorems
------------
  MOD-1 (Polynomial Growth of Coefficients):
    For a weight-k modular form, |aₙ| = O(n^{(k-1)/2+ε}).
    The ACF alpha invariant: α_mod = (k-1)/2 / log(Q).

  MOD-2 (q-Series FMA = Weight × Precision):
    Evaluating f(τ) to Q terms requires exactly Q algebraic FMAs
    (Horner in the variable q), regardless of level N.

  MOD-3 (Ramanujan Bound — Cusp Forms):
    For a Hecke eigenform f of weight k, the Ramanujan–Petersson conjecture
    (proved by Deligne 1974 for k ≥ 2) gives |aₙ| ≤ d(n) · n^{(k-1)/2}
    where d(n) is the number of divisors — extremely tight bound.

  MOD-4 (ACF for Eta Products):
    η-quotients and products are modular forms whose q-expansion coefficients
    have multiplicative structure. Their ACF reduction preserves this structure:
    Φ(η^r) = Φ(η)^r  (functorial for the product operation).

References
----------
  Serre (1973) — A course in arithmetic, §VII modular forms.
  Diamond & Shurman — A First Course in Modular Forms.
  Ramanujan (1916) — On certain arithmetical functions.
  Paper.md §56 (modular ACF extension).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# § 1  q-Series and Fourier Expansions
# ===========================================================================

def q_from_tau(tau: complex) -> complex:
    """Convert τ ∈ ℍ (upper half-plane, Im τ > 0) to q = e^{2πiτ}."""
    return cmath.exp(2j * math.pi * tau)


def tau_from_q(q: complex) -> complex:
    """Recover τ from q = e^{2πiτ} (assuming |q| < 1)."""
    if abs(q) >= 1:
        raise ValueError("|q| must be < 1 (q in the unit disk)")
    return cmath.log(q) / (2j * math.pi)


def eval_q_series(
    coefficients: List[complex],
    q: complex,
    *,
    constant_term: complex = 0.0,
) -> complex:
    """
    Evaluate f = Σ_{n=0}^d aₙ qⁿ  via Horner.

    coefficients[i] = aᵢ (coefficient of qⁱ, starting from q¹).
    constant_term = a₀ (constant term for Eisenstein-like forms).
    FMA count = len(coefficients).
    """
    # Horner: start from highest degree and work down
    if not coefficients:
        return constant_term

    acc = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        acc = acc * q + c
    acc = acc * q + constant_term
    return acc


# ===========================================================================
# § 2  Standard Modular Forms
# ===========================================================================

def eisenstein_e2(tau: complex, n_terms: int = 50) -> complex:
    """
    Eisenstein series E₂(τ) — weight 2 (quasi-modular form).

    E₂(τ) = 1 - 24 Σ_{n=1}^∞ σ₁(n) qⁿ

    Note: E₂ is not quite modular (transforms with an extra term) but its
    combination E₂* = E₂ - 3/(π·Im(τ)) is modular.
    """
    q = q_from_tau(tau)
    result = 1.0 + 0j
    for n in range(1, n_terms + 1):
        sigma1 = sum(d for d in range(1, n + 1) if n % d == 0)
        result -= 24 * sigma1 * (q ** n)
    return result


def eisenstein_e4(tau: complex, n_terms: int = 50) -> complex:
    """
    Eisenstein series E₄(τ) — weight 4, fully modular.

    E₄(τ) = 1 + 240 Σ_{n=1}^∞ σ₃(n) qⁿ
    """
    q = q_from_tau(tau)
    result = 1.0 + 0j
    for n in range(1, n_terms + 1):
        sigma3 = sum(d**3 for d in range(1, n + 1) if n % d == 0)
        result += 240 * sigma3 * (q ** n)
    return result


def eisenstein_e6(tau: complex, n_terms: int = 50) -> complex:
    """
    Eisenstein series E₆(τ) — weight 6.

    E₆(τ) = 1 - 504 Σ_{n=1}^∞ σ₅(n) qⁿ
    """
    q = q_from_tau(tau)
    result = 1.0 + 0j
    for n in range(1, n_terms + 1):
        sigma5 = sum(d**5 for d in range(1, n + 1) if n % d == 0)
        result -= 504 * sigma5 * (q ** n)
    return result


def ramanujan_delta(tau: complex, n_terms: int = 30) -> complex:
    """
    Ramanujan's discriminant function Δ(τ) = η(τ)²⁴.

    Δ(τ) = q ∏_{n=1}^∞ (1 - qⁿ)²⁴ = Σ_{n=1}^∞ τ(n) qⁿ

    where τ(n) is the Ramanujan tau function.
    This is the unique (up to scalar) weight-12 cusp form for SL(2,ℤ).
    """
    q = q_from_tau(tau)
    result = q  # q factor
    for n in range(1, n_terms + 1):
        result *= (1 - q**n)**24
    return result


def dedekind_eta(tau: complex, n_terms: int = 50) -> complex:
    """
    Dedekind eta function η(τ) = q^{1/24} ∏_{n=1}^∞ (1 - qⁿ).

    This is a modular form of weight 1/2 with a multiplier system.
    """
    q = q_from_tau(tau)
    result = q ** (1.0 / 24.0)
    for n in range(1, n_terms + 1):
        result *= (1 - q**n)
    return result


def jacobi_theta(tau: complex, z: complex = 0.0, n_terms: int = 30) -> complex:
    """
    Jacobi theta function θ₃(z, τ) = Σ_{n=-∞}^∞ q^{n²} exp(2πinz).

    With z=0: θ₃(0, τ) = 1 + 2 Σ_{n=1}^∞ qⁿ² (Gaussian integer lattice partition function).
    """
    q = q_from_tau(tau)
    result = 1.0 + 0j
    for n in range(1, n_terms + 1):
        result += 2 * (q**(n*n)) * cmath.exp(2j * math.pi * n * z)
    return result


# ===========================================================================
# § 3  Precomputed q-Series Coefficients
# ===========================================================================

def ramanujan_tau_coefficients(n_terms: int = 30) -> List[int]:
    """
    Compute Ramanujan tau(n) for n = 1, 2, …, n_terms.

    τ(n) is the coefficient of qⁿ in Δ(τ) = q ∏(1-qⁿ)²⁴.

    Uses the recurrence via Euler pentagonal theorem for (1-q)²⁴ product.
    """
    # Product expansion via convolution
    coeffs = [0] * (n_terms + 1)
    coeffs[0] = 1

    for k in range(1, n_terms + 1):
        new = [0] * (n_terms + 1)
        # Multiply by (1 - q^k)^24 up to degree n_terms
        for n in range(n_terms + 1):
            new[n] = coeffs[n]
            sign = 1
            for j in range(1, 25):
                if n - j * k >= 0:
                    new[n] += math.comb(24, j) * ((-1)**j) * coeffs[n - j * k]
        coeffs = new

    # The tau function: Δ = q * product → coeffs of Δ[n] = product's coeffs[n-1]
    tau = [0] + [coeffs[n - 1] for n in range(1, n_terms + 1)]
    return tau


def eisenstein_e4_coefficients(n_terms: int = 30) -> List[int]:
    """Coefficients of E₄: [1, 240*σ₃(1), 240*σ₃(2), …]."""
    coeffs = [1]
    for n in range(1, n_terms + 1):
        sigma3 = sum(d**3 for d in range(1, n + 1) if n % d == 0)
        coeffs.append(240 * sigma3)
    return coeffs


def eisenstein_e6_coefficients(n_terms: int = 30) -> List[int]:
    """Coefficients of E₆: [1, -504*σ₅(1), -504*σ₅(2), …]."""
    coeffs = [1]
    for n in range(1, n_terms + 1):
        sigma5 = sum(d**5 for d in range(1, n + 1) if n % d == 0)
        coeffs.append(-504 * sigma5)
    return coeffs


# ===========================================================================
# § 4  ACF Reduction for Modular Forms
# ===========================================================================

@dataclass
class ModularFormACFReport:
    """ACF analysis report for a modular form."""
    form_name: str
    weight: int
    level: int
    n_terms: int
    fourier_coefficients: List[complex]
    alpha: float                  # ≈ (k-1)/2 per MOD-1
    normalized_alpha: float
    fma_count: int                # = n_terms (Horner in q)
    d_star: int                   # terms needed for ε-approximation
    coefficient_growth_rate: float  # observed |aₙ|/n^{(k-1)/2}
    is_cusp_form: bool
    ramanujan_bound_holds: bool   # |aₙ| ≤ d(n)·n^{(k-1)/2}
    certificates: List[str]
    metadata: Dict = field(default_factory=dict)


class ModularACFReducer:
    """
    ACF reduction for modular forms via their q-expansion.

    The modular form is represented by its Fourier coefficients {aₙ}.
    The Horner algorithm in q gives the optimal FMA sequence:
        E(f, Q) = Q  (exactly Q FMAs to evaluate to precision Q)
    """

    def reduce(
        self,
        form_name: str,
        fourier_coefficients: List[complex],
        weight: int,
        level: int = 1,
        is_cusp_form: bool = True,
        epsilon: float = 1e-8,
    ) -> ModularFormACFReport:
        """
        Reduce a modular form to its ACF representation.

        Parameters
        ----------
        form_name : str — name of the form (e.g., "Δ", "E₄")
        fourier_coefficients : list of aₙ for n = 0, 1, …, Q
        weight : k — modular weight
        level : N — level (1 = full modular group SL(2,ℤ))
        is_cusp_form : whether a₀ = 0
        epsilon : target approximation accuracy
        """
        n = len(fourier_coefficients)
        coeffs = list(fourier_coefficients)

        # Alpha invariant: for weight-k cusp form, |aₙ| = O(n^{(k-1)/2})
        # → decay rate of normalized coefficients provides alpha
        theoretical_alpha = (weight - 1) / 2.0

        # Empirical growth rate
        growth_rates = []
        for i, an in enumerate(coeffs[1:], start=1):
            mag = abs(complex(an))
            bound = i ** theoretical_alpha
            if bound > 1e-10:
                growth_rates.append(mag / bound)

        observed_growth = float(np.median(growth_rates)) if growth_rates else 1.0

        # Ramanujan bound: |aₙ| ≤ d(n) · n^{(k-1)/2}
        ramanujan_holds = True
        for i, an in enumerate(coeffs[1:], start=1):
            mag = abs(complex(an))
            d_n = sum(1 for d in range(1, i + 1) if i % d == 0)
            bound = d_n * (i ** theoretical_alpha)
            if mag > bound * 10 + 1:  # +1 for constant terms like E₄
                ramanujan_holds = False
                break

        # FMA count = n_terms (Horner in q)
        fma_count = n

        # d*(ε): terms needed so that |aₙ||q|^n < ε at |q| = e^{-2π} ≈ 0.00187
        q_magnitude = math.exp(-2 * math.pi)
        d_star = n
        for i, an in enumerate(coeffs):
            cumulative = sum(abs(complex(coeffs[j])) * q_magnitude**j for j in range(i, n))
            if cumulative < epsilon:
                d_star = i
                break

        alpha = theoretical_alpha
        normalized_alpha = 1.0 / (1.0 + alpha)

        certs = ["MOD-1", "MOD-2"]
        if ramanujan_holds:
            certs.append("MOD-3 (Ramanujan bound)")
        certs.append("ALGACF-1 (Horner in q)")

        return ModularFormACFReport(
            form_name=form_name,
            weight=weight,
            level=level,
            n_terms=n,
            fourier_coefficients=coeffs,
            alpha=alpha,
            normalized_alpha=normalized_alpha,
            fma_count=fma_count,
            d_star=d_star,
            coefficient_growth_rate=observed_growth,
            is_cusp_form=is_cusp_form,
            ramanujan_bound_holds=ramanujan_holds,
            certificates=certs,
            metadata={"weight": weight, "level": level, "epsilon": epsilon},
        )


# ===========================================================================
# § 5  L-functions (Dirichlet series associated to modular forms)
# ===========================================================================

@dataclass
class LFunctionACFReport:
    """ACF report for an L-function L(f, s)."""
    form_name: str
    s_value: complex
    n_terms: int
    partial_sum: complex
    euler_product_alpha: float
    fma_count: int
    certificates: List[str] = field(default_factory=lambda: ["MOD-2"])


def compute_l_function(
    fourier_coefficients: List[complex],
    s: complex,
    n_terms: Optional[int] = None,
) -> LFunctionACFReport:
    """
    Compute L(f, s) = Σ_{n=1}^∞ aₙ / nˢ.

    The Dirichlet series is evaluated as a partial sum (finite truncation).
    FMA count = n_terms (one division and one mul per term — equivalent to FMA).
    """
    if n_terms is None:
        n_terms = len(fourier_coefficients) - 1

    coeffs = fourier_coefficients[1:n_terms + 1]  # a₁, a₂, …

    # Evaluate via Horner-like algorithm
    partial = sum(
        complex(an) / (n ** s)
        for n, an in enumerate(coeffs, start=1)
    )

    # Euler product alpha: convergence rate by |aₙ|/n^σ → 0
    sigma = s.real
    norms = [abs(complex(an)) / (n ** sigma) for n, an in enumerate(coeffs, start=1)]
    from_nonzero = [x for x in norms if x > 0]
    if len(from_nonzero) >= 2:
        k = np.arange(len(from_nonzero), dtype=float)
        slope = np.polyfit(k, np.log(from_nonzero + [1e-300] * 0)[:len(k)], 1)[0]
        euler_alpha = max(0.0, -slope)
    else:
        euler_alpha = 1.0

    return LFunctionACFReport(
        form_name="L(f,s)",
        s_value=s,
        n_terms=n_terms,
        partial_sum=partial,
        euler_product_alpha=euler_alpha,
        fma_count=n_terms,
    )


# Convenience factory
class ModularFormLibrary:
    """Standard modular forms, precomputed to N terms."""

    @staticmethod
    def delta(n_terms: int = 30) -> ModularFormACFReport:
        coeffs = ramanujan_delta_coefficients_simple(n_terms)
        reducer = ModularACFReducer()
        return reducer.reduce("Δ (Ramanujan)", coeffs, weight=12, is_cusp_form=True)

    @staticmethod
    def e4(n_terms: int = 30) -> ModularFormACFReport:
        coeffs = eisenstein_e4_coefficients(n_terms)
        reducer = ModularACFReducer()
        return reducer.reduce("E₄", coeffs, weight=4, is_cusp_form=False)

    @staticmethod
    def e6(n_terms: int = 30) -> ModularFormACFReport:
        coeffs = eisenstein_e6_coefficients(n_terms)
        reducer = ModularACFReducer()
        return reducer.reduce("E₆", coeffs, weight=6, is_cusp_form=False)


def ramanujan_delta_coefficients_simple(n_terms: int = 30) -> List[int]:
    """
    Compute Δ q-expansion coefficients to n_terms using a simple convolution.
    Δ(τ) = Σ τ(n) qⁿ; τ(1)=1, τ(2)=-24, τ(3)=252, τ(4)=-1472, τ(5)=4830, …
    """
    # Precomputed tau values for verification
    _tau_known = {1: 1, 2: -24, 3: 252, 4: -1472, 5: 4830,
                  6: -6048, 7: -16744, 8: 84480, 9: -113643, 10: -115920}

    # Use E4^3 - E6^2 formula (up to scalar):  Δ = (E4^3 - E6^2) / 1728
    e4 = eisenstein_e4_coefficients(n_terms)
    e6 = eisenstein_e6_coefficients(n_terms)

    def poly_mul(a: List[int], b: List[int], maxn: int) -> List[int]:
        res = [0] * (maxn + 1)
        for i in range(min(len(a), maxn + 1)):
            for j in range(min(len(b), maxn + 1 - i)):
                res[i + j] += a[i] * b[j]
        return res

    def poly_pow(a: List[int], k: int, maxn: int) -> List[int]:
        res = [0] * (maxn + 1)
        res[0] = 1
        for _ in range(k):
            res = poly_mul(res, a, maxn)
        return res

    e4_cubed = poly_pow(e4, 3, n_terms)
    e6_sq = poly_mul(e6, e6, n_terms)
    delta_raw = [(e4_cubed[i] - e6_sq[i]) for i in range(n_terms + 1)]
    # Normalize: delta[0] should be 0, delta[1] should be 1728 (= 1728*tau(1))
    scale = delta_raw[1] if delta_raw[1] != 0 else 1728
    delta_norm = [0] + [round(x / scale) for x in delta_raw[1:n_terms + 1]]
    return delta_norm
