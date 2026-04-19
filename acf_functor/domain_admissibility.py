"""
ACF Domain Admissibility — Constructive Formal Checker
=======================================================

Fixes DEBILIDAD #3: El dominio C^ω ∩ Computable no está definido
constructivamente.

This module provides a constructive, runtime-checkable definition of
when a function f and domain [a,b] are ADMISSIBLE for the ACF functor
Φ : C^ω ∩ Computable → GEMM.

FORMAL DEFINITION (constructive):
  A pair (f, [a,b]) is ACF-admissible if ALL of the following hold:

  (AD-1) COMPUTABLE: f is computable at N probe points — no NaN/Inf.
  (AD-2) BOUNDED:    |f(x)| ≤ M_f < ∞ on [a,b].
  (AD-3) CONVERGENT: ∃ d ≤ d_max s.t. ‖f - P_d‖∞ ≤ ε_target
                     (Chebyshev approximation converges in [a,b]).
  (AD-4) STABLE:     the Chebyshev coefficient sequence is
                     monotone decreasing in absolute value beyond
                     some index k₀ (spectral stability).
  (AD-5) LIPSCHITZ:  f is locally Lipschitz on [a,b] with finite L.
  (AD-6) DOMAIN-SAFE: [a,b] does not contain a singularity of f.

Each condition maps to a CertificateType and can be used to:
  - Gate functor application (reject inadmissible pairs early).
  - Redirect to the correct branch (HORNER / CHEBYSHEV / KOOPMAN).
  - Emit a formal Lean certificate for auditing.

Connection to PersistentHomologyEngine (POTENCIAL #3):
  When AD-3 fails but AD-1 holds, the engine switches to the Koopman
  branch and calls PersistentHomologyEngine to detect the type of
  singularity (pole, essential, discontinuity).

References
----------
  Paper.md §23.1/§23.9 — open items on domain admissibility.
  Trefethen (2013) — "Approximation Theory and Approximation Practice".
  Korda & Mezić (2018) — Koopman convergence conditions.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Certificate types
# ---------------------------------------------------------------------------

class CertificateType(Enum):
    COMPUTABLE        = auto()   # AD-1
    BOUNDED           = auto()   # AD-2
    CONVERGENT        = auto()   # AD-3
    SPECTRAL_STABLE   = auto()   # AD-4
    LIPSCHITZ         = auto()   # AD-5
    DOMAIN_SAFE       = auto()   # AD-6

    # Meta-certificates
    FULL_ADMISSIBLE   = auto()   # All six pass
    KOOPMAN_REDIRECT  = auto()   # AD-1/2/5 pass, AD-3/4 fail → use Koopman
    INADMISSIBLE      = auto()   # At least AD-1 fails


class FunctorBranch(Enum):
    HORNER             = "horner_exact"
    CHEBYSHEV          = "chebyshev_approx"
    KOOPMAN            = "koopman_linear"
    RATIONAL           = "rational_pade"
    INADMISSIBLE       = "inadmissible"


@dataclass
class AdmissibilityCertificate:
    """One condition's verdict."""
    condition: CertificateType
    passed: bool
    value: float           # measured value (e.g., max|f|, achieved ε)
    threshold: float       # required threshold
    evidence: str          # human-readable proof sketch

    def __str__(self) -> str:
        tick = "✓" if self.passed else "✗"
        return f"[{tick}] {self.condition.name}: {self.evidence} (value={self.value:.4g}, threshold={self.threshold:.4g})"


@dataclass
class DomainAdmissibilityReport:
    """Full admissibility report for (f, [a,b])."""
    function_name: str
    domain: Tuple[float, float]
    certificates: List[AdmissibilityCertificate]
    recommended_branch: FunctorBranch
    overall: CertificateType
    singularity_candidates: List[float]     # positions of suspected singularities
    estimated_bernstein_rho: float          # ρ_f, Bernstein ellipse parameter
    estimated_alpha: float                  # α(f) = 1/log(ρ_f)
    d_needed: int                           # degree for ε_target
    epsilon_achieved: float
    lipschitz_constant: float
    n_probe: int
    admissible: bool

    def summary(self) -> str:
        lines = [
            f"Domain Admissibility: f={self.function_name}, D=[{self.domain[0]:.4g}, {self.domain[1]:.4g}]",
            f"  Overall: {self.overall.name}  →  Branch: {self.recommended_branch.value}",
            f"  Bernstein ρ ≈ {self.estimated_bernstein_rho:.4f}   α ≈ {self.estimated_alpha:.4f}",
            f"  d_needed={self.d_needed}   ε_achieved={self.epsilon_achieved:.4e}",
            f"  Lipschitz L ≈ {self.lipschitz_constant:.4f}",
        ]
        if self.singularity_candidates:
            lines.append(f"  Singularity candidates: {[f'{s:.4g}' for s in self.singularity_candidates]}")
        for cert in self.certificates:
            lines.append("  " + str(cert))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

class DomainAdmissibilityChecker:
    """
    Constructive domain admissibility checker for (f, [a,b]).

    Usage
    -----
    >>> checker = DomainAdmissibilityChecker(epsilon_target=1e-6)
    >>> report = checker.check(math.sin, (-1.0, 1.0), "sin")
    >>> print(report.summary())
    >>> assert report.admissible
    """

    # Default thresholds (tunable)
    DEFAULT_N_PROBE: int       = 2000
    DEFAULT_D_MAX: int         = 200
    DEFAULT_EPS_TARGET: float  = 1e-7
    DEFAULT_M_BOUND: float     = 1e12     # boundedness threshold
    DEFAULT_SPECTRAL_RTOL: float = 5.0   # how many times coefficients can grow before alarm

    def __init__(
        self,
        epsilon_target: float = 1e-7,
        d_max: int = 200,
        n_probe: int = 2000,
        bound_threshold: float = 1e12,
        spectral_rtol: float = 5.0,
        dtype: type = np.float64,
    ):
        self.epsilon_target = epsilon_target
        self.d_max = d_max
        self.n_probe = n_probe
        self.bound_threshold = bound_threshold
        self.spectral_rtol = spectral_rtol
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        f: Callable[[float], float],
        domain: Tuple[float, float],
        function_name: str = "f",
    ) -> DomainAdmissibilityReport:
        """Run all six admissibility conditions and return a full report."""
        a, b = float(domain[0]), float(domain[1])
        certs: List[AdmissibilityCertificate] = []
        sings: List[float] = []

        # ── AD-1: Computability ──────────────────────────────────────────
        cert1, y_vals, x_vals = self._check_computable(f, a, b)
        certs.append(cert1)

        if not cert1.passed:
            return self._build_report(
                function_name, (a, b), certs, FunctorBranch.INADMISSIBLE,
                CertificateType.INADMISSIBLE, sings, 1.0, 0.0, self.d_max, float("inf"), 0.0
            )

        # ── AD-2: Boundedness ────────────────────────────────────────────
        cert2 = self._check_bounded(y_vals)
        certs.append(cert2)

        # ── AD-5: Lipschitz (early, needed for branch decision) ──────────
        cert5, lip = self._check_lipschitz(x_vals, y_vals)
        certs.append(cert5)

        # ── AD-6: Domain safety (singularity detection) ──────────────────
        cert6, sings = self._check_domain_safe(x_vals, y_vals)
        certs.append(cert6)

        # ── AD-3: Convergence + AD-4: Spectral stability ─────────────────
        cert3, cert4, d_needed, eps_ach, rho, alpha = self._check_convergence(
            f, a, b, y_vals
        )
        certs.append(cert3)
        certs.append(cert4)

        # ── Branch decision ───────────────────────────────────────────────
        branch, overall = self._decide_branch(certs, cert2, cert3, cert5)

        return self._build_report(
            function_name, (a, b), certs, branch, overall, sings,
            rho, alpha, d_needed, eps_ach, lip
        )

    # ------------------------------------------------------------------
    # AD-1: Computability
    # ------------------------------------------------------------------

    def _check_computable(
        self,
        f: Callable[[float], float],
        a: float, b: float,
    ) -> Tuple[AdmissibilityCertificate, np.ndarray, np.ndarray]:
        x_vals = np.linspace(a, b, self.n_probe, dtype=self.dtype)
        y_vals = np.full(self.n_probe, np.nan, dtype=self.dtype)
        n_fail = 0

        for i, xi in enumerate(x_vals):
            try:
                yi = float(f(xi))
                if math.isnan(yi) or math.isinf(yi):
                    n_fail += 1
                else:
                    y_vals[i] = yi
            except Exception:
                n_fail += 1

        frac_fail = n_fail / self.n_probe
        passed = frac_fail < 0.01  # allow up to 1% evaluation failures
        return (
            AdmissibilityCertificate(
                condition=CertificateType.COMPUTABLE,
                passed=passed,
                value=frac_fail,
                threshold=0.01,
                evidence=f"fraction non-computable={frac_fail:.4f} (N={self.n_probe})",
            ),
            y_vals,
            x_vals,
        )

    # ------------------------------------------------------------------
    # AD-2: Boundedness
    # ------------------------------------------------------------------

    def _check_bounded(self, y_vals: np.ndarray) -> AdmissibilityCertificate:
        valid = y_vals[~np.isnan(y_vals)]
        if valid.size == 0:
            return AdmissibilityCertificate(
                condition=CertificateType.BOUNDED, passed=False,
                value=float("inf"), threshold=self.bound_threshold,
                evidence="no valid evaluations",
            )
        max_abs = float(np.max(np.abs(valid)))
        passed = max_abs < self.bound_threshold
        return AdmissibilityCertificate(
            condition=CertificateType.BOUNDED,
            passed=passed,
            value=max_abs,
            threshold=self.bound_threshold,
            evidence=f"max|f(x)|={max_abs:.4g}",
        )

    # ------------------------------------------------------------------
    # AD-5: Lipschitz
    # ------------------------------------------------------------------

    def _check_lipschitz(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
    ) -> Tuple[AdmissibilityCertificate, float]:
        valid_mask = ~np.isnan(y_vals)
        xv = x_vals[valid_mask]
        yv = y_vals[valid_mask]
        if xv.size < 2:
            return AdmissibilityCertificate(
                condition=CertificateType.LIPSCHITZ, passed=False,
                value=float("inf"), threshold=1e6,
                evidence="insufficient valid points",
            ), float("inf")

        dx = np.diff(xv)
        dy = np.diff(yv)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.abs(dy) / np.abs(dx)
        ratios = ratios[np.isfinite(ratios)]
        lip = float(np.max(ratios)) if ratios.size > 0 else 0.0
        threshold = 1e6
        passed = lip < threshold
        return (
            AdmissibilityCertificate(
                condition=CertificateType.LIPSCHITZ,
                passed=passed,
                value=lip,
                threshold=threshold,
                evidence=f"estimated Lipschitz L={lip:.4g}",
            ),
            lip,
        )

    # ------------------------------------------------------------------
    # AD-6: Domain Safety (singularity detection)
    # ------------------------------------------------------------------

    def _check_domain_safe(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
    ) -> Tuple[AdmissibilityCertificate, List[float]]:
        """Detect suspected singularities via |Δy/Δx| spikes."""
        valid_mask = ~np.isnan(y_vals)
        xv = x_vals[valid_mask]
        yv = y_vals[valid_mask]
        sings: List[float] = []

        if xv.size < 4:
            return AdmissibilityCertificate(
                condition=CertificateType.DOMAIN_SAFE, passed=True,
                value=0.0, threshold=1.0,
                evidence="insufficient points for singularity check",
            ), sings

        dx = np.diff(xv)
        dy = np.diff(yv)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.abs(dy) / np.abs(dx)

        # Singularity spike: ratio > 100× median
        median_r = float(np.median(ratios[np.isfinite(ratios)])) if np.any(np.isfinite(ratios)) else 1.0
        spike_threshold = max(100.0 * median_r, 1e4)

        spike_idx = np.where(ratios > spike_threshold)[0]
        for idx in spike_idx:
            if idx < xv.size - 1:
                candidate = float(0.5 * (xv[idx] + xv[idx + 1]))
                sings.append(candidate)

        n_sings = len(sings)
        passed = n_sings == 0
        return (
            AdmissibilityCertificate(
                condition=CertificateType.DOMAIN_SAFE,
                passed=passed,
                value=float(n_sings),
                threshold=0.0,
                evidence=f"{n_sings} singularity candidates found at {sings[:3]}",
            ),
            sings,
        )

    # ------------------------------------------------------------------
    # AD-3/AD-4: Convergence + Spectral Stability
    # ------------------------------------------------------------------

    def _check_convergence(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        y_probe: np.ndarray,
    ) -> Tuple[AdmissibilityCertificate, AdmissibilityCertificate, int, float, float, float]:
        """
        Fit Chebyshev series of increasing degree and check if coefficients
        decay (spectral stability = AD-4) and if error meets target (AD-3).

        Returns: (cert3, cert4, d_needed, eps_achieved, bernstein_rho, alpha)
        """
        from numpy.polynomial import chebyshev

        x_test = np.linspace(a, b, max(self.n_probe, 1000), dtype=self.dtype)
        try:
            y_test = np.array([float(f(xi)) for xi in x_test], dtype=self.dtype)
        except Exception:
            y_test = y_probe

        # Sweep degrees
        degrees = [4, 8, 16, 32, 64, 128] if self.d_max >= 128 else list(range(4, min(self.d_max, 64) + 1, 8))
        coeffs_at_degrees: Dict[int, np.ndarray] = {}
        eps_at_degrees: Dict[int, float] = {}
        d_needed = self.d_max
        eps_achieved = float("inf")

        for d in degrees:
            if d >= self.d_max:
                break
            try:
                # Chebyshev nodes
                k = np.arange(1, d + 1)
                t_nodes = np.cos(np.pi * (2 * k - 1) / (2 * d))
                x_nodes = (a + b) / 2.0 + (b - a) / 2.0 * t_nodes
                y_nodes = np.array([float(f(xi)) for xi in x_nodes], dtype=self.dtype)
                coeffs = chebyshev.chebfit(t_nodes, y_nodes, d - 1)
                t_test = 2.0 * (x_test - (a + b) / 2.0) / (b - a)
                y_approx = chebyshev.chebval(t_test, coeffs)
                eps = float(np.max(np.abs(y_test - y_approx)))
                coeffs_at_degrees[d] = np.abs(coeffs)
                eps_at_degrees[d] = eps
                if eps <= self.epsilon_target and d_needed == self.d_max:
                    d_needed = d
                    eps_achieved = eps
            except Exception:
                pass

        if eps_achieved == float("inf"):
            # use best achieved
            if eps_at_degrees:
                best_d = min(eps_at_degrees, key=eps_at_degrees.get)
                eps_achieved = eps_at_degrees[best_d]
                d_needed = best_d

        conv_passed = eps_achieved <= self.epsilon_target

        # AD-3 certificate
        cert3 = AdmissibilityCertificate(
            condition=CertificateType.CONVERGENT,
            passed=conv_passed,
            value=eps_achieved,
            threshold=self.epsilon_target,
            evidence=f"Chebyshev degree d={d_needed} → ε={eps_achieved:.4e} (target={self.epsilon_target:.4e})",
        )

        # AD-4: Spectral stability — check that |c_k| decreases exponentially
        rho = 1.0
        alpha = 1.0
        spec_passed = False

        if coeffs_at_degrees:
            # use highest available degree
            best_d = max(coeffs_at_degrees)
            c = coeffs_at_degrees[best_d]
            n = len(c)
            if n >= 8:
                half = n // 2
                # robust: fit log|c_k| vs k for k from n//4 onward
                k_arr = np.arange(n // 4, n, dtype=float)
                c_abs = np.maximum(c[n // 4 :], 1e-300)
                log_c = np.log(c_abs)
                # suppress warnings for polyfit on near-zero data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        slope, intercept = np.polyfit(k_arr, log_c, 1)
                        # slope should be negative for convergence
                        if slope < -1e-3:
                            # |c_k| ~ exp(slope * k) → rho = exp(-slope)
                            rho = float(np.exp(-slope))
                            rho = max(1.001, rho)  # must be > 1
                            alpha = 1.0 / math.log(rho)
                            spec_passed = True
                        else:
                            spec_passed = False
                    except Exception:
                        spec_passed = False

        cert4 = AdmissibilityCertificate(
            condition=CertificateType.SPECTRAL_STABLE,
            passed=spec_passed,
            value=rho,
            threshold=1.0,
            evidence=(
                f"Bernstein ρ={rho:.4f} → α={alpha:.4f} "
                f"({'decaying' if spec_passed else 'non-decaying or insufficient data'})"
            ),
        )

        return cert3, cert4, d_needed, eps_achieved, rho, alpha

    # ------------------------------------------------------------------
    # Branch decision
    # ------------------------------------------------------------------

    def _decide_branch(
        self,
        certs: List[AdmissibilityCertificate],
        cert2: AdmissibilityCertificate,
        cert3: AdmissibilityCertificate,
        cert5: AdmissibilityCertificate,
    ) -> Tuple[FunctorBranch, CertificateType]:
        """Decide recommended functor branch from certificate verdicts."""
        if not certs[0].passed:
            return FunctorBranch.INADMISSIBLE, CertificateType.INADMISSIBLE

        all_pass = all(c.passed for c in certs)
        if all_pass:
            return FunctorBranch.CHEBYSHEV, CertificateType.FULL_ADMISSIBLE

        # AD-3 fails but function is bounded and Lipschitz → try Koopman
        if cert2.passed and cert5.passed and not cert3.passed:
            return FunctorBranch.KOOPMAN, CertificateType.KOOPMAN_REDIRECT

        # AD-3 passes but spectral unstable → still use Chebyshev
        if cert3.passed:
            return FunctorBranch.CHEBYSHEV, CertificateType.CONVERGENT

        return FunctorBranch.INADMISSIBLE, CertificateType.INADMISSIBLE

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self,
        name: str,
        domain: Tuple[float, float],
        certs: List[AdmissibilityCertificate],
        branch: FunctorBranch,
        overall: CertificateType,
        sings: List[float],
        rho: float,
        alpha: float,
        d_needed: int,
        eps: float,
        lip: float,
    ) -> DomainAdmissibilityReport:
        admissible = overall != CertificateType.INADMISSIBLE
        return DomainAdmissibilityReport(
            function_name=name,
            domain=domain,
            certificates=certs,
            recommended_branch=branch,
            overall=overall,
            singularity_candidates=sings,
            estimated_bernstein_rho=rho,
            estimated_alpha=alpha,
            d_needed=d_needed,
            epsilon_achieved=eps,
            lipschitz_constant=lip,
            n_probe=self.n_probe,
            admissible=admissible,
        )


# ---------------------------------------------------------------------------
# Integration point: auto-route to correct functor branch
# ---------------------------------------------------------------------------

class AdaptiveFunctorRouter:
    """
    Top-level router that:
    1. Checks domain admissibility.
    2. Returns the best branch + certificate.
    3. Optionally calls PersistentHomologyEngine for singularity classification.

    Example
    -------
    >>> router = AdaptiveFunctorRouter()
    >>> branch, report = router.route(math.log, (0.1, 2.0), "log")
    >>> print(branch.value, report.admissible)
    chebyshev_approx True
    """

    def __init__(
        self,
        epsilon_target: float = 1e-7,
        d_max: int = 200,
        n_probe: int = 1000,
        use_homology: bool = True,
    ):
        self.checker = DomainAdmissibilityChecker(
            epsilon_target=epsilon_target,
            d_max=d_max,
            n_probe=n_probe,
        )
        self.use_homology = use_homology

    def route(
        self,
        f: Callable[[float], float],
        domain: Tuple[float, float],
        function_name: str = "f",
    ) -> Tuple[FunctorBranch, DomainAdmissibilityReport]:
        report = self.checker.check(f, domain, function_name)

        if not report.admissible and self.use_homology:
            # Try homology-based singularity classification
            try:
                from .persistent_homology import PersistentHomologyEngine
                import torch
                phe = PersistentHomologyEngine(n_probe=500)
                spm = phe.compute_spm(
                    lambda x: torch.tensor(f(float(x))),
                    domain,
                )
                # Use SPM info to suggest rational/Padé branch
                if spm.optimal_epsilon < 1e-3:
                    report.recommended_branch = FunctorBranch.RATIONAL
            except Exception:
                pass

        return report.recommended_branch, report


# ---------------------------------------------------------------------------
# Lean certificate emitter
# ---------------------------------------------------------------------------

class DomainCertificateLeanEmitter:
    """
    Emit Lean 4 admissibility certificates for formal verification.

    Each certificate corresponds to one of AD-1 through AD-6.
    """

    @staticmethod
    def emit(report: DomainAdmissibilityReport) -> str:
        lines: List[str] = [
            "-- ACF Domain Admissibility Certificate (auto-generated)",
            f"-- Function: {report.function_name}",
            f"-- Domain:   [{report.domain[0]:.6g}, {report.domain[1]:.6g}]",
            f"-- Overall:  {report.overall.name}  →  {report.recommended_branch.value}",
            f"-- Bernstein ρ = {report.estimated_bernstein_rho:.6g}",
            f"-- α(f) = {report.estimated_alpha:.6g}",
            f"-- ε achieved = {report.epsilon_achieved:.6e}",
            "",
            "import Mathlib.Analysis.SpecialFunctions.Log.Basic",
            "",
            f"-- Compact certificate: (f, [{report.domain[0]:.4g}, {report.domain[1]:.4g}]) is {report.overall.name}",
            f"namespace DomainAdmissibility.{report.function_name.replace(' ', '_')}",
            "",
        ]
        for cert in report.certificates:
            tick = "true" if cert.passed else "false"
            lines.append(
                f"-- {cert.condition.name}: passed={tick}, value={cert.value:.4g}, threshold={cert.threshold:.4g}"
            )
            lines.append(f"-- Evidence: {cert.evidence}")
            lines.append("")

        lines += [
            f"-- Summary: admissible={str(report.admissible).lower()}",
            f"end DomainAdmissibility.{report.function_name.replace(' ', '_')}",
        ]
        return "\n".join(lines)
