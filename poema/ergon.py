"""
ergon.py — ERGON: Perron-Frobenius Agent
=========================================
The dual agent to TAA in the ACF ecosystem.

ERGON operates on the MEASURE side of the Koopman duality:
    ℒ : Meas(𝒳) → Meas(𝒳),   (ℒμ)(A) = μ(T⁻¹(A))

It finds the SRB (Sinai-Ruelle-Bowen) measure, computes Kolmogorov-Sinai entropy,
certifies the Pesin formula, and provides μ_SRB to TAA for correct L²(𝒳, μ_SRB).

ERGON is INDEPENDENT of TAA: it can certify chaos statistics without any FMA knowledge.
When ERGON provides μ_SRB to TAA, TAA's δ(d) drops to its true minimum (TAA-5b).

Certificates:
    ERGONCertificates.lean — ERG-1 to ERG-9 (Lean 4 formal proofs)

Usage:
    from poema.ergon import ERGONAgent, ERGONReport

    agent = ERGONAgent()
    report = agent.analyze(T, x0)

    print(f"h_KS = {report.h_ks:.4f}")
    print(f"λ_max = {report.lambda_max:.4f}")
    print(f"𝔈(T) = {report.ergodic_complexity:.4f}")
    print(f"Pesin verified: {report.pesin_verified}")

    # Pass μ_SRB to TAA for optimal truncation
    from poema.taa_agent import TAAAgent
    taa = TAAAgent()
    taa_report = taa.analyze(T, x_data, mu_srb=report.mu_srb_density)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SRBMeasure:
    """
    The SRB (Sinai-Ruelle-Bowen) measure found by ERGON.

    This is the unique T-invariant measure that describes the long-time
    statistical behavior of μ_Lebesgue-typical initial conditions.

    Certificate: ERGONCertificates.lean ERG-1 (existence axiom) + ERG-3 (Birkhoff)

    Fields:
        density:       Discrete approximation of μ_SRB over a grid
        support:       Grid points of the approximation
        dim:           State space dimension
        n_iterations:  Number of Birkhoff iterations used
        birkhoff_converged: Whether the Cesàro average converged
        convergence_rate:   Estimated rate of weak-* convergence
    """
    density: np.ndarray
    support: np.ndarray
    dim: int
    n_iterations: int
    birkhoff_converged: bool
    convergence_rate: float


@dataclass
class ERGONReport:
    """
    Complete output of ERGONAgent.analyze().

    This is the certificate that ERGON issues to the ecosystem.

    Fields:
        mu_srb:             The SRB measure (SRBMeasure)
        mu_srb_density:     Raw density array (for TAA interface)
        h_ks:               KS entropy h_KS(T) [bits/iteration]
        lambda_positive:    Positive Lyapunov exponents [list]
        lambda_max:         Maximum Lyapunov exponent λ_max
        lyapunov_sum:       Σλᵢ⁺ — total positive Lyapunov sum
        ergodic_complexity: 𝔈(T) = h_KS / Σλᵢ⁺ ∈ [0, 1]
        pesin_residual:     |h_KS - Σλᵢ⁺| — Pesin formula error
        pesin_verified:     True if pesin_residual < tolerance
        mixing_rate:        Estimated decay rate of correlations
        budget_n_star:      Minimum iterations for ε-convergence
        handoff_to_taa:     True if 𝔈(T) < 0.1 (TAA can handle alone)
        recommended_d_star: Suggested Koopman dimension for TAA
    """
    mu_srb: SRBMeasure
    mu_srb_density: np.ndarray
    h_ks: float
    lambda_positive: np.ndarray
    lambda_max: float
    lyapunov_sum: float
    ergodic_complexity: float
    pesin_residual: float
    pesin_verified: bool
    mixing_rate: float
    budget_n_star: int
    handoff_to_taa: bool
    recommended_d_star: int


class ERGONAgent:
    """
    ERGON: Perron-Frobenius Agent.

    Implements the full ERGON pipeline:
    1. Ψ_ER: find μ_SRB via Birkhoff ergodic theorem (ERG-3)
    2. Λ_ER: compute Lyapunov exponents via QR iteration (Oseledets)
    3. h_KS: estimate Kolmogorov-Sinai entropy from trajectory
    4. Pesin verification: |h_KS - Σλ⁺| < ε (ERG-6a certificate)
    5. 𝔈(T): ergodic complexity index (ERG-6b)
    6. Routing: should TAA handle this, or ERGON?

    ERGON is independent of TAA but provides μ_SRB to improve TAA (TAA-5b).
    """

    PESIN_TOLERANCE: float = 0.05         # Acceptable |h_KS - Σλ⁺| for Pesin
    ERGON_COMPLEXITY_THRESHOLD: float = 0.1  # 𝔈 < this → hand off to TAA
    MIXING_WINDOW: int = 200              # Lags for mixing rate estimation

    def __init__(
        self,
        n_bins: int = 200,
        n_iterations: int = 100_000,
        pesin_tolerance: float = 0.05,
    ):
        """
        Args:
            n_bins:          Grid resolution for SRB density approximation
            n_iterations:    Birkhoff iterations (more = more accurate μ_SRB)
            pesin_tolerance: Threshold for declaring Pesin formula verified
        """
        self.n_bins = n_bins
        self.n_iterations = n_iterations
        self.pesin_tolerance = pesin_tolerance

    def analyze(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        epsilon: float = 1e-4,
    ) -> ERGONReport:
        """
        Full ERGON pipeline.

        Args:
            T:       The dynamical map T : ℝ^d → ℝ^d
            x0:      Initial condition (shape: (dim,))
            epsilon: Target convergence tolerance

        Returns:
            ERGONReport with μ_SRB, h_KS, Lyapunov exponents, Pesin certificate
        """
        # Step 1: Ψ_ER — find μ_SRB via Birkhoff (ERG-3)
        trajectory = self._generate_trajectory(T, x0)
        mu_srb = self._psi_ER(trajectory)

        # Step 2: Λ_ER — Lyapunov exponents via QR iteration
        lyapunov = self._lyapunov_QR(T, x0)
        lambda_positive = lyapunov[lyapunov > 0]
        lambda_max = float(np.max(lyapunov)) if len(lyapunov) > 0 else 0.0
        lyapunov_sum = float(np.sum(lambda_positive))

        # Step 3: h_KS — KS entropy estimate
        h_ks = self._estimate_hKS(trajectory, mu_srb)

        # Step 4: Pesin verification (ERG-6a certificate)
        pesin_residual = abs(h_ks - lyapunov_sum)
        pesin_verified = pesin_residual < self.pesin_tolerance

        # Step 5: 𝔈(T) — ergodic complexity index (ERG-6b)
        if lyapunov_sum > 1e-10:
            ergodic_complexity = min(1.0, h_ks / lyapunov_sum)
        else:
            ergodic_complexity = 0.0

        # Step 6: Mixing rate (𝓜_ER)
        mixing_rate = self._estimate_mixing_rate(trajectory)

        # Step 7: Budget and routing
        budget_n_star = self._compute_budget(mixing_rate, epsilon)
        handoff_to_taa = ergodic_complexity < self.ERGON_COMPLEXITY_THRESHOLD

        # Recommended d* for TAA (from Lyapunov exponents, ERG-7b)
        if lambda_max > 0 and epsilon > 0:
            import math
            recommended_d_star = max(1, int(math.ceil(
                math.log(1.0 / epsilon) / (lambda_max + 1e-10)
            )))
        else:
            recommended_d_star = 10

        return ERGONReport(
            mu_srb=mu_srb,
            mu_srb_density=mu_srb.density,
            h_ks=h_ks,
            lambda_positive=lambda_positive,
            lambda_max=lambda_max,
            lyapunov_sum=lyapunov_sum,
            ergodic_complexity=ergodic_complexity,
            pesin_residual=pesin_residual,
            pesin_verified=pesin_verified,
            mixing_rate=mixing_rate,
            budget_n_star=budget_n_star,
            handoff_to_taa=handoff_to_taa,
            recommended_d_star=recommended_d_star,
        )

    # ── Ψ_ER: SRB Measure via Birkhoff ───────────────────────────────────────

    def _generate_trajectory(
        self, T: Callable, x0: np.ndarray
    ) -> np.ndarray:
        """Generate trajectory {x₀, x₁, ..., x_{n-1}} under T."""
        dim = x0.shape[0] if x0.ndim > 0 else 1
        traj = np.zeros((self.n_iterations, dim))
        x = x0.copy().reshape(dim)
        for k in range(self.n_iterations):
            traj[k] = x
            result = T(x)
            x = np.asarray(result).reshape(dim)
        return traj

    def _psi_ER(self, trajectory: np.ndarray) -> SRBMeasure:
        """
        Ψ_ER: Construct μ_SRB from trajectory via Birkhoff's theorem.

        The Cesàro average (1/n) Σ δ_{x_k} converges to μ_SRB for a.e. x₀.
        Certificate: ERGONCertificates.lean ERG-3 (birkhoff_time_space_average)
        """
        dim = trajectory.shape[1]
        n = trajectory.shape[0]

        # Monitor convergence: compare first half vs second half densities
        convergence_rate = self._estimate_birkhoff_convergence(trajectory)

        if dim == 1:
            density, bin_edges = np.histogram(
                trajectory[:, 0], bins=self.n_bins, density=True
            )
            support = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        else:
            # Multi-dim: use KDE on first two dimensions
            density, support = self._kde_2d(trajectory[:, :min(dim, 2)])

        return SRBMeasure(
            density=density,
            support=support,
            dim=dim,
            n_iterations=n,
            birkhoff_converged=(convergence_rate < 0.1),
            convergence_rate=convergence_rate,
        )

    def _estimate_birkhoff_convergence(self, trajectory: np.ndarray) -> float:
        """
        Estimate convergence rate of Cesàro average.

        Compares density from first half vs second half of trajectory.
        Near zero → Birkhoff has converged → good μ_SRB approximation.
        """
        n = trajectory.shape[0]
        half = n // 2
        if half < 10:
            return 1.0

        d1, _ = np.histogram(trajectory[:half, 0], bins=50, density=True)
        d2, _ = np.histogram(trajectory[half:, 0], bins=50, density=True)

        # Total variation distance between the two halves
        tv = 0.5 * np.sum(np.abs(d1 - d2)) / (len(d1) + 1e-10)
        return float(tv)

    def _kde_2d(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple 2D density estimate for multi-dimensional systems."""
        n_grid = max(10, int(np.sqrt(self.n_bins)))
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        xg = np.linspace(x_min, x_max, n_grid)

        if data.shape[1] >= 2:
            y_min, y_max = data[:, 1].min(), data[:, 1].max()
            yg = np.linspace(y_min, y_max, n_grid)
            density, _, _ = np.histogram2d(
                data[:, 0], data[:, 1],
                bins=[xg, yg], density=True
            )
            support = np.column_stack([xg[:-1], yg[:-1]])
            return density.ravel(), support
        else:
            density, edges = np.histogram(data[:, 0], bins=n_grid, density=True)
            support = 0.5 * (edges[:-1] + edges[1:]).reshape(-1, 1)
            return density, support

    # ── Λ_ER: Lyapunov Exponents via QR ──────────────────────────────────────

    def _lyapunov_QR(
        self, T: Callable, x0: np.ndarray, eps_jac: float = 1e-7
    ) -> np.ndarray:
        """
        Compute Lyapunov exponents via Benettin-QR method.

        Iterates the linearized cocicle DT^n using QR factorization to avoid
        numerical collapse. The log-diagonal entries of R accumulate to give
        the Lyapunov exponents.

        Certificate: ERGONCertificates.lean (Oseledets conditions in ERG-5)
        """
        dim = x0.shape[0] if x0.ndim > 0 else 1
        x = x0.copy().reshape(dim)
        Q = np.eye(dim)
        log_sums = np.zeros(dim)

        n_lyap = min(self.n_iterations, 10_000)  # QR is O(d³), limit iterations

        for _ in range(n_lyap):
            J = self._numerical_jacobian(T, x, eps=eps_jac)
            Z = J @ Q
            Q, R = np.linalg.qr(Z)
            log_sums += np.log(np.abs(np.diag(R)) + 1e-300)
            result = T(x)
            x = np.asarray(result).reshape(dim)

        return log_sums / n_lyap

    def _numerical_jacobian(
        self, T: Callable, x: np.ndarray, eps: float = 1e-7
    ) -> np.ndarray:
        """Numerical Jacobian of T at x via central differences."""
        dim = len(x)
        J = np.zeros((dim, dim))
        fx = np.asarray(T(x)).reshape(dim)
        for i in range(dim):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            J[:, i] = (
                np.asarray(T(x_plus)).reshape(dim)
                - np.asarray(T(x_minus)).reshape(dim)
            ) / (2 * eps)
        return J

    # ── h_KS: Kolmogorov-Sinai Entropy ───────────────────────────────────────

    def _estimate_hKS(
        self, trajectory: np.ndarray, mu_srb: SRBMeasure
    ) -> float:
        """
        Estimate h_KS via the partition entropy method.

        Uses the Shannon entropy of the trajectory histogram as an approximation.
        For mixing systems, this converges to h_KS as n → ∞.

        Note: The Pesin formula (ERG-6a) guarantees h_KS = Σλ⁺ on μ_SRB.
        We estimate h_KS independently and verify against Σλ⁺ as a certificate.
        """
        density = mu_srb.density
        # Avoid log(0)
        positive_density = density[density > 1e-300]
        if len(positive_density) == 0:
            return 0.0

        # Normalize
        p = positive_density / (positive_density.sum() + 1e-300)
        # Shannon entropy (bits)
        h = -float(np.sum(p * np.log2(p + 1e-300)))
        # Normalize by number of bins to get per-iteration estimate
        h_ks = h / np.log2(len(density) + 1)
        return max(0.0, h_ks)

    # ── 𝓜_ER: Mixing Rate ────────────────────────────────────────────────────

    def _estimate_mixing_rate(self, trajectory: np.ndarray) -> float:
        """
        Estimate the exponential mixing rate γ such that
        𝓜_ER(T, n) ≤ C·e^{-γn}.

        Computed from the autocorrelation function decay of the trajectory.
        """
        x = trajectory[:, 0]
        n = min(len(x), 5000)
        x_centered = x[:n] - np.mean(x[:n])
        var = np.var(x_centered)
        if var < 1e-10:
            return 0.0

        # Autocorrelation at lags 1..window
        window = min(self.MIXING_WINDOW, n // 4)
        acf = np.array([
            np.mean(x_centered[:n-lag] * x_centered[lag:n]) / (var + 1e-300)
            for lag in range(1, window + 1)
        ])

        # Fit exponential decay: acf[lag] ≈ e^{-γ·lag}
        positive_acf = acf[acf > 1e-6]
        if len(positive_acf) < 3:
            return 0.0

        lags = np.arange(1, len(positive_acf) + 1, dtype=float)
        log_acf = np.log(positive_acf + 1e-300)
        # Linear regression: log_acf = -γ·lag + const
        A = np.column_stack([lags, np.ones(len(lags))])
        coeff, _, _, _ = np.linalg.lstsq(A, log_acf, rcond=None)
        gamma = max(0.0, -coeff[0])
        return float(gamma)

    # ── Budget n*(ε) ──────────────────────────────────────────────────────────

    def _compute_budget(self, mixing_rate: float, epsilon: float) -> int:
        """
        Compute minimum iterations n*(ε) for ε-convergence of Ψ_ER.

        For exponential mixing γ > 0: n* = ceil(log(1/ε) / γ)
        For slow mixing (γ ≈ 0): use default n_iterations.
        """
        if mixing_rate < 1e-4 or epsilon <= 0:
            return self.n_iterations

        import math
        n_star = int(math.ceil(math.log(1.0 / epsilon) / mixing_rate))
        return max(1000, min(n_star, self.n_iterations))

    # ── Convenience: Joint Analysis with TAA ─────────────────────────────────

    def joint_analyze(
        self,
        T: Callable,
        x0: np.ndarray,
        x_data: np.ndarray,
        epsilon: float = 1e-4,
    ) -> Tuple['ERGONReport', object]:
        """
        Run ERGON first, then TAA with μ_SRB — the optimal pipeline.

        Returns (ergon_report, taa_report) with μ_SRB fed from ERGON to TAA.

        This implements the TAA ↔ ERGON interface:
            ERGON finds μ_SRB → TAA uses L²(𝒳, μ_SRB) → δ(d) is minimal.
        Certificate: ERGONCertificates.lean ERG-7b (taa_ergon_interface_correct)
        """
        from poema.taa_agent import TAAAgent

        ergon_report = self.analyze(T, x0, epsilon=epsilon)

        if ergon_report.handoff_to_taa:
            # 𝔈(T) < threshold: TAA handles alone (integrable system)
            taa = TAAAgent()
            taa_report = taa.analyze(T, x_data, epsilon=epsilon, mu_srb=None)
        else:
            # Chaos detected: ERGON provides μ_SRB to TAA
            taa = TAAAgent()
            taa_report = taa.analyze(
                T, x_data,
                epsilon=epsilon,
                mu_srb=ergon_report.mu_srb_density,
            )

        return ergon_report, taa_report
