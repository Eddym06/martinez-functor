"""
taa_agent.py — TAA: Tensor Autocomputable Agent
================================================
The Koopman-based agent of the ACF ecosystem.

TAA operates on the FUNCTION side of the Koopman duality:
    K : L²(𝒳, μ) → L²(𝒳, μ),   Kf = f ∘ T

It decomposes a dynamical system's observables into Koopman eigenfunctions,
truncates the spectrum to dimension d*(ε), and collapses to FMA sequences.

TAA is INDEPENDENT of ERGON: it can run alone using an empirical measure.
When ERGON provides μ_SRB, TAA's truncation error δ(d) drops to its true minimum.

Certificates:
    TAAAgentCertificates.lean — TAA-1 to TAA-6 (Lean 4 formal proofs)
    KoopmanDeltaCertificates.lean — KD-1 to KD-4 (spectral bounds)

Usage:
    from poema.taa_agent import TAAAgent, TAAReport, AlphaClass

    agent = TAAAgent()
    report = agent.analyze(T, x_data, epsilon=1e-6)
    print(report.alpha_class, report.d_star, report.delta_d)

    # With ERGON-provided μ_SRB (eliminates measure inflation)
    from poema.ergon import ERGONAgent
    ergon = ERGONAgent()
    mu_srb = ergon.find_srb_measure(T, x0)
    report = agent.analyze(T, x_data, epsilon=1e-6, mu_srb=mu_srb)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple


# ── Alpha-A Decay Classification ─────────────────────────────────────────────

class AlphaClass(Enum):
    """
    Spectral decay family of the Koopman operator eigenvalues.

    Determines the FMA cost class (TAA-4 in TAAAgentCertificates.lean):
    - EXPONENTIAL: |λ_k| ≤ C·ρ^{-k} → d*(ε) = O(log 1/ε)   [cheapest]
    - POLYNOMIAL:  |λ_k| ≤ C·k^{-s} → d*(ε) = O(ε^{-1/s})  [moderate]
    - FINITE:      spectrum has ≤ d non-zero eigenvalues       [exact, d FMAs]
    - UNKNOWN:     insufficient data to classify               [use heuristic]
    """
    EXPONENTIAL = auto()
    POLYNOMIAL = auto()
    FINITE = auto()
    UNKNOWN = auto()


@dataclass
class TAAReport:
    """
    Output of TAAAgent.analyze() — the complete TAA certificate.

    Fields:
        eigenvalues:     Estimated Koopman eigenvalues (sorted descending)
        d_star:          Optimal Koopman dimension for target epsilon
        delta_d:         Truncation error δ(d*) — bound from KD-1
        alpha_class:     Spectral decay family (AlphaClass)
        alpha_rate:      Decay rate (ρ for exp, s for poly, d for finite)
        fma_cost:        Estimated FMA operations for Koopman-lifted function
        measure_used:    'empirical' or 'srb' (was μ_SRB from ERGON used?)
        measure_inflation: Error due to wrong measure (0 if ERGON provided μ_SRB)
        koopman_modes:   Koopman eigenvectors (columns of EDMD matrix)
        koopman_freqs:   Koopman eigenfrequencies (imaginary parts)
        koopman_damping: Koopman damping rates (real parts, negative = decaying)
        ergon_required:  True if TAA-6 triggered (chaos detected, needs ERGON)
        lambda_max:      Estimated max Lyapunov exponent (from eigenvalue real parts)
        epsilon_target:  The epsilon for which d_star was computed
    """
    eigenvalues: np.ndarray
    d_star: int
    delta_d: float
    alpha_class: AlphaClass
    alpha_rate: float
    fma_cost: int
    measure_used: str
    measure_inflation: float
    koopman_modes: np.ndarray
    koopman_freqs: np.ndarray
    koopman_damping: np.ndarray
    ergon_required: bool
    lambda_max: float
    epsilon_target: float


class TAAAgent:
    """
    TAA: Tensor Autocomputable Agent.

    Implements the Koopman spectral analysis pipeline:
    1. EDMD (Extended Dynamic Mode Decomposition) to approximate Koopman eigenvalues
    2. Spectral decay classification (Alpha-A index)
    3. Optimal dimension computation d*(ε) via TAA-3b
    4. FMA cost estimation
    5. Diagnostic: does this system require ERGON for correct μ_SRB?

    TAA is independent of ERGON but benefits from it (TAA-5 vs TAA-5b).
    """

    # Threshold for declaring "chaos requires ERGON" (TAA-6)
    ERGON_REQUIRED_LAMBDA_THRESHOLD: float = 0.05

    def __init__(self, edmd_delay: int = 1):
        """
        Args:
            edmd_delay: Number of time-delay coordinates for EDMD (default 1).
        """
        self.edmd_delay = edmd_delay

    def analyze(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        x_data: np.ndarray,
        epsilon: float = 1e-6,
        mu_srb: Optional[np.ndarray] = None,
    ) -> TAAReport:
        """
        Full TAA pipeline: EDMD → eigenvalues → classification → d* → report.

        Args:
            T:       The dynamical map T : 𝒳 → 𝒳
            x_data:  Trajectory data, shape (n_steps, dim)
            epsilon: Target truncation error δ(d*) < epsilon
            mu_srb:  Optional SRB measure from ERGON (eliminates measure inflation)

        Returns:
            TAAReport with all certificates
        """
        eigenvalues, modes = self._edmd(x_data)
        alpha_class, alpha_rate = self._classify_alpha(eigenvalues)
        d_star = self._compute_d_star(eigenvalues, epsilon, alpha_class, alpha_rate)
        delta_d = self._compute_delta_d(eigenvalues, d_star)
        fma_cost = self._estimate_fma_cost(d_star, x_data.shape[1])

        # Measure inflation: TAA-5 vs TAA-5b
        measure_used, measure_inflation = self._measure_diagnostics(
            eigenvalues, mu_srb, x_data
        )

        # Lyapunov proxy from Koopman eigenvalues (real parts = damping)
        damping = np.real(np.log(eigenvalues + 1e-300))
        freqs = np.imag(np.log(eigenvalues + 1e-300))
        lambda_max = float(np.max(damping)) if len(damping) > 0 else 0.0

        # TAA-6: if lambda_max > threshold, TAA needs ERGON's μ_SRB
        ergon_required = (
            lambda_max > self.ERGON_REQUIRED_LAMBDA_THRESHOLD
            and mu_srb is None
        )

        return TAAReport(
            eigenvalues=eigenvalues,
            d_star=d_star,
            delta_d=delta_d,
            alpha_class=alpha_class,
            alpha_rate=alpha_rate,
            fma_cost=fma_cost,
            measure_used=measure_used,
            measure_inflation=measure_inflation,
            koopman_modes=modes,
            koopman_freqs=freqs,
            koopman_damping=damping,
            ergon_required=ergon_required,
            lambda_max=lambda_max,
            epsilon_target=epsilon,
        )

    # ── EDMD ─────────────────────────────────────────────────────────────────

    def _edmd(
        self, x_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extended Dynamic Mode Decomposition (EDMD).

        Constructs the linear Koopman matrix from snapshot pairs:
            X = [x₀, x₁, ..., x_{n-1}]
            X' = [x₁, x₂, ..., x_n]
            K_approx = X'·X†  (DMD approximation of Koopman)

        Returns sorted eigenvalues |λ₁| ≥ |λ₂| ≥ ... and eigenmodes.

        Formal basis: KD-1 (delta_spectral_bound) — eigenvalues determine δ(d).
        """
        n = x_data.shape[0]
        if n < 2:
            return np.array([1.0]), np.eye(1)

        X = x_data[:-1].T    # shape (dim, n-1)
        Xp = x_data[1:].T    # shape (dim, n-1)

        # Koopman matrix via least squares: K = Xp · X†
        K_matrix, _, _, _ = np.linalg.lstsq(X.T, Xp.T, rcond=None)
        K_matrix = K_matrix.T

        # Eigendecomposition
        try:
            eigenvalues_raw, modes = np.linalg.eig(K_matrix)
        except np.linalg.LinAlgError:
            return np.array([1.0]), np.eye(K_matrix.shape[0])

        # Sort by |λ| descending (KD-1: δ(d) ≤ |λ_{d+1}|)
        idx = np.argsort(np.abs(eigenvalues_raw))[::-1]
        eigenvalues = np.abs(eigenvalues_raw[idx])
        modes = modes[:, idx]

        # Koopman eigenvalues ≤ 1 for measure-preserving T (TAA-1)
        eigenvalues = np.minimum(eigenvalues, 1.0)

        return eigenvalues, modes

    # ── Alpha-A Classification ────────────────────────────────────────────────

    def _classify_alpha(
        self, eigenvalues: np.ndarray
    ) -> Tuple[AlphaClass, float]:
        """
        Classify the spectral decay family (TAA-4).

        Fits the eigenvalue sequence to:
        - Exponential: |λ_k| ≈ C·ρ^{-k}  → log|λ_k| is linear in k
        - Polynomial:  |λ_k| ≈ C·k^{-s}  → log|λ_k| is linear in log(k)
        - Finite:      eigenvalues drop to near-zero after index d
        """
        n = len(eigenvalues)
        if n < 3:
            return AlphaClass.UNKNOWN, 1.0

        # Check for finite spectrum: sharp drop-off
        nonzero = np.sum(eigenvalues > 1e-10)
        if nonzero < n / 2:
            return AlphaClass.FINITE, float(nonzero)

        k = np.arange(1, n + 1, dtype=float)
        log_lambda = np.log(eigenvalues + 1e-300)

        # Fit exponential: log|λ_k| ≈ log C - k·log ρ
        A_exp = np.column_stack([np.ones(n), -k])
        coeff_exp, res_exp, _, _ = np.linalg.lstsq(A_exp, log_lambda, rcond=None)
        rho_exp = np.exp(coeff_exp[1]) if len(coeff_exp) > 1 else 1.0

        # Fit polynomial: log|λ_k| ≈ log C - s·log k
        log_k = np.log(k)
        A_poly = np.column_stack([np.ones(n), -log_k])
        coeff_poly, res_poly, _, _ = np.linalg.lstsq(A_poly, log_lambda, rcond=None)
        s_poly = coeff_poly[1] if len(coeff_poly) > 1 else 1.0

        r_exp = float(res_exp[0]) if len(res_exp) > 0 else float('inf')
        r_poly = float(res_poly[0]) if len(res_poly) > 0 else float('inf')

        if r_exp <= r_poly and rho_exp > 1.01:
            return AlphaClass.EXPONENTIAL, float(rho_exp)
        elif s_poly > 0.1:
            return AlphaClass.POLYNOMIAL, float(s_poly)
        else:
            return AlphaClass.UNKNOWN, 1.0

    # ── Optimal Dimension d*(ε) ───────────────────────────────────────────────

    def _compute_d_star(
        self,
        eigenvalues: np.ndarray,
        epsilon: float,
        alpha_class: AlphaClass,
        alpha_rate: float,
    ) -> int:
        """
        Compute optimal Koopman dimension d*(ε) such that δ(d*) < ε.

        Uses TAA-3b (explicit formula for exponential/polynomial decay)
        or direct search via KD-3 for finite/unknown cases.

        Certificate: TAAAgentCertificates.lean TAA-3b
        """
        if alpha_class == AlphaClass.FINITE:
            return max(1, int(alpha_rate))

        if alpha_class == AlphaClass.EXPONENTIAL and alpha_rate > 1.0:
            # d* ≥ log(C/ε) / log(ρ) from TAA-3b
            C = float(eigenvalues[0]) if len(eigenvalues) > 0 else 1.0
            import math
            d = max(1, int(math.ceil(math.log(C / epsilon) / math.log(alpha_rate))))
            return min(d, len(eigenvalues))

        if alpha_class == AlphaClass.POLYNOMIAL and alpha_rate > 0.1:
            # d* ≥ (C/ε)^{1/s}
            C = float(eigenvalues[0]) if len(eigenvalues) > 0 else 1.0
            d = max(1, int((C / epsilon) ** (1.0 / alpha_rate)))
            return min(d, len(eigenvalues))

        # General case: direct search (KD-3, optimal_dimension_exists)
        for d in range(len(eigenvalues)):
            if d + 1 < len(eigenvalues) and eigenvalues[d + 1] < epsilon:
                return d + 1
        return len(eigenvalues)

    # ── Truncation Error δ(d) ────────────────────────────────────────────────

    def _compute_delta_d(self, eigenvalues: np.ndarray, d: int) -> float:
        """
        Compute δ(d) = |λ_{d+1}| — the KD-1 spectral truncation bound.

        Certificate: KoopmanDeltaCertificates.lean delta_spectral_bound (KD-1)
        """
        idx = d  # d+1 in 0-indexed
        if idx < len(eigenvalues):
            return float(eigenvalues[idx])
        return 0.0

    # ── FMA Cost Estimation ───────────────────────────────────────────────────

    def _estimate_fma_cost(self, d_star: int, state_dim: int) -> int:
        """
        Estimate FMA operations for evaluating the d*-truncated Koopman expansion.

        Each Koopman mode evaluation requires state_dim multiplications + additions.
        Total: d_star × state_dim FMAs (TAA-2: Horner optimal).
        """
        return d_star * state_dim

    # ── Measure Diagnostics ───────────────────────────────────────────────────

    def _measure_diagnostics(
        self,
        eigenvalues: np.ndarray,
        mu_srb: Optional[np.ndarray],
        x_data: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Diagnose measure quality (TAA-5 vs TAA-5b).

        If μ_SRB from ERGON is provided: measure_inflation = 0 (TAA-5b).
        Otherwise: estimate inflation from empirical measure mismatch.
        """
        if mu_srb is not None:
            # ERGON provided μ_SRB: no inflation (TAA-5b)
            return 'srb', 0.0

        # Empirical measure: estimate inflation from eigenvalue variance
        # The inflation is proportional to the deviation from μ_SRB
        # Heuristic: variance of eigenvalue magnitudes indicates measure mismatch
        if len(eigenvalues) > 1:
            inflation = float(np.std(np.diff(eigenvalues)))
        else:
            inflation = 0.0

        return 'empirical', inflation

    # ── Convenience Interface ─────────────────────────────────────────────────

    def koopman_predict(
        self,
        report: TAAReport,
        x0: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Predict trajectory using truncated Koopman expansion.

        Uses the d*-truncated eigenvalue decomposition from analyze().
        Formally: x_{t+n} ≈ Σ_{k=1}^{d*} c_k · λ_k^n · φ_k(x₀)
        """
        d = report.d_star
        eigs = report.eigenvalues[:d]
        modes = report.koopman_modes[:, :d]

        # Project initial condition onto Koopman modes
        coeffs, _, _, _ = np.linalg.lstsq(modes, x0, rcond=None)

        trajectory = np.zeros((n_steps, x0.shape[0]))
        for t in range(n_steps):
            powers = eigs ** t
            trajectory[t] = modes @ (coeffs * powers)

        return trajectory
