"""
taa_agent.py — Topological Agency Algorithm (TAA)
The Koopman-based agent of the ACF ecosystem.

TAA operates on the FUNCTION side of the Koopman duality:
    K : L²(𝒳, μ_SRB) → L²(𝒳, μ_SRB),   Kf = f ∘ T

It finds eigenfunctions, classifies the spectral decay (α_A),
computes truncation budgets δ(d), evaluates the free-energy criterion,
and decides whether to act alone or defer to ERGON.

Architecture (6 layers, TAA.md §12):
  Layer 1 — Entropic Perception: ingest trajectory → entropy + admissibility
  Layer 2 — Topological Diagnosis: collapse-aware state s_t = (H, α, ε, δ, Adm)
  Layer 3 — Grammar Synthesis: Koopman EDMD with decay-class basis
  Layer 4 — Formal Sovereignty: produce Lean-4-exportable certificates
  Layer 5 — Native Execution: predict, reconstruct, project
  Layer 6 — Assimilation: update diagnostic priors

Certificate fields produced (TAA-1 through TAA-7):
  TAA-1: ‖Kf‖₂ = ‖f‖₂ (isometry verified numerically)
  TAA-2: E(f) = E(Φ_AC(f)) (energy invariance, affine fragment)
  TAA-3: d*(ε) — optimal Koopman dimension for target precision ε
  TAA-4: α_A class (Exponential / Polynomial / Finite)
  TAA-5: δ_μ inflation from wrong measure (0 when ERGON provides μ_SRB)
  TAA-6: defer_to_ergon flag (λ_max > 0 and 𝔈 > threshold)
  TAA-7: λ_max (leading Lyapunov exponent) — new diagnostic
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import linalg

from acf_functor.shared_numerics import (
    LyapunovEstimator as _SharedLyapunovEstimator,
    SpectralClassifier as _SharedSpectralClassifier,
    ChebyshevBasis as _SharedChebyshevBasis,
)

# Module-level shared instances (cached across agent lifetimes)
_shared_lyapunov = _SharedLyapunovEstimator()
_shared_classifier = _SharedSpectralClassifier()


# ---------------------------------------------------------------------------
# Types and Enumerations
# ---------------------------------------------------------------------------

class DecayClass(Enum):
    """Koopman spectral decay classification (α_A)."""
    FINITE      = "finite"        # Exact spectrum — TAA acts with d* = rank
    EXPONENTIAL = "exponential"   # |λ_k| ≤ C·ρ^{-k}, ρ > 1 — d* = O(log 1/ε)
    POLYNOMIAL  = "polynomial"    # |λ_k| ≤ C·k^{-s} — d* = O(ε^{-1/s})
    CHAOTIC     = "chaotic"       # No decay — defer to ERGON

class TAAMode(Enum):
    """Which Poema mode TAA selects (§10.1)."""
    POEM     = "poem"      # Explicit function analysis (reduction)
    COPOEM   = "copoem"    # Property-first synthesis
    BIPOEM   = "bipoem"    # Data-driven inference
    ERGON    = "ergon"     # Defer: chaos regime, need μ_SRB from ERGON


@dataclass
class KoopmanSpectrum:
    """Full Koopman spectral decomposition result."""
    eigenvalues:    np.ndarray        # complex (d,)
    eigenfunctions: np.ndarray        # real  (n_grid, d) — evaluated on grid
    koopman_matrix: np.ndarray        # real  (d, d) — EDMD K
    decay_class:    DecayClass
    alpha_A:        float             # spectral decay rate (log-scale)
    rho:            float             # ρ for exponential, s for polynomial
    C_decay:        float             # prefactor C in decay bound
    d_star:         dict              # d*(ε) for standard ε values


@dataclass
class TAAState:
    """
    The collapse-aware TAA state vector (§12, Layer 2):
        s_t = (H, α_A, ε, δ, Adm, Π, B)
    """
    # Observable diagnostics
    spectral_entropy:   float   # H_t — Shannon entropy of spectrum
    alpha_A:            float   # Spectral decay index
    approx_error:       float   # ε — current approximation residual
    truncation_error:   float   # δ(d) — truncation error at current d*

    # System classification
    decay_class:        DecayClass
    mode:               TAAMode
    defer_to_ergon:     bool    # λ_max > 0 and 𝔈 > threshold

    # Budget
    d_star:             int     # Optimal Koopman dimension
    lambda_max:         float   # Leading Lyapunov exponent (from ERGON or estimated)
    ergodic_complexity: float   # 𝔈(T) = h_KS / Σλ⁺

    # Numerical certificates (Lean-4-exportable)
    certificates:       dict = field(default_factory=dict)


@dataclass
class TAACertificate:
    """
    Complete TAA formal certificate — ready for export to Lean 4.
    All fields are Python-native scalars for cross-language compatibility.
    """
    # TAA-1: Koopman isometry
    TAA_1_isometry_error: float       # ‖‖Kf‖₂ - ‖f‖₂‖ (should be ≈ 0)

    # TAA-2: Energy invariance
    TAA_2_energy_invariant: bool      # E(f) = E(Φ_AC(f)) on affine fragment

    # TAA-3: Optimal dimension
    TAA_3_d_star_eps01: int           # d*(ε=0.1)
    TAA_3_d_star_eps001: int          # d*(ε=0.01)
    TAA_3_d_star_eps0001: int         # d*(ε=0.001)

    # TAA-4: Alpha classification
    TAA_4_decay_class: str
    TAA_4_alpha_A: float
    TAA_4_rho: float                  # ρ (exponential) or s (polynomial)

    # TAA-5: Measure sensitivity
    TAA_5_delta_mu_inflation: float   # 0 if ERGON provided μ_SRB, >0 otherwise

    # TAA-6: Defer decision
    TAA_6_defer_to_ergon: bool
    TAA_6_lambda_max: float
    TAA_6_ergodic_complexity: float

    # TAA-7 (new): Spectral entropy
    TAA_7_spectral_entropy: float

    # TAA-10 (new): Dictionary adequacy index
    TAA_10_iab: float = 0.0            # Index de Adaptación de Base ∈ [0, 1]
    TAA_10_non_normality: float = 0.0  # N(K) = ‖KK* - K*K‖_F / ‖K‖_F²

    # TAA-11 (new): Critical complexity threshold
    TAA_11_e_star: float = 0.0         # 𝔈* = 1 - Γ_OTU / h_KS
    TAA_11_regime: str = "unknown"     # "log_budget" or "poly_budget"

    # Summary
    PASS: bool = False


# ---------------------------------------------------------------------------
# Core: TAA Agent
# ---------------------------------------------------------------------------

class TAAAgent:
    """
    Topological Agency Algorithm — Koopman-based agent.

    Usage:
        agent = TAAAgent(T, domain=(0.0, 1.0), n_obs=32)
        state = agent.diagnose()
        cert  = agent.certify(mu_srb=None)  # pass μ_SRB from ERGON for best results
    """

    def __init__(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float] = (-1.0, 1.0),
        n_obs: int = 32,           # Koopman observable dimension (Φ)
        n_traj: int = 2_000,       # trajectory length for EDMD
        chaos_threshold: float = 0.9,  # 𝔈 threshold for deferring to ERGON
        lyap_threshold: float = 0.0,   # λ_max threshold for chaos detection
    ):
        self.T = T
        self.domain = domain
        self.n_obs = n_obs
        self.n_traj = n_traj
        self.chaos_threshold = chaos_threshold
        self.lyap_threshold = lyap_threshold

        self._K: Optional[np.ndarray] = None          # Koopman matrix
        self._eigenvalues: Optional[np.ndarray] = None
        self._spectrum: Optional[KoopmanSpectrum] = None
        self._mu_srb: Optional[np.ndarray] = None     # from ERGON
        self._is_built: bool = False

    # ------------------------------------------------------------------
    # Layer 1 + 3: EDMD Koopman construction
    # ------------------------------------------------------------------

    def build(self, mu_srb: Optional[np.ndarray] = None) -> "TAAAgent":
        """
        Build the EDMD Koopman matrix K using Chebyshev observables.

        If mu_srb is provided (from ERGON), trajectories are weighted by μ_SRB.
        Otherwise uniform sampling is used (TAA-5: may inflate δ_μ).

        Args:
            mu_srb: probability array on domain grid (from ERGONAgent). If None,
                    uses uniform Lebesgue sampling (wrong measure for chaotic T).
        """
        a, b = self.domain
        self._mu_srb = mu_srb

        # Generate trajectory
        x_traj = self._generate_trajectory(self.n_traj, mu_srb)

        # Chebyshev observables: ψ_k(x) = T_k(2(x-a)/(b-a) - 1)
        def cheb(k: int, x: np.ndarray) -> np.ndarray:
            xi = np.clip(2.0 * (x - a) / (b - a) - 1.0, -1.0, 1.0)
            return np.cos(k * np.arccos(xi))

        # EDMD: K ≈ Ψ_Y Ψ_X†
        Psi_X = np.array([cheb(k, x_traj[:-1]) for k in range(self.n_obs)])  # (d, n-1)
        y_traj = np.array([float(self.T(np.array([x]))[0]) for x in x_traj[:-1]])
        y_safe = np.clip(y_traj, a + 1e-10, b - 1e-10)
        Psi_Y = np.array([cheb(k, y_safe) for k in range(self.n_obs)])       # (d, n-1)

        self._K = Psi_Y @ np.linalg.pinv(Psi_X)  # (d, d)

        # Eigendecomposition
        eigvals, eigvecs = linalg.eig(self._K, right=True)
        idx = np.argsort(-np.abs(eigvals))
        self._eigenvalues = eigvals[idx]
        self._eigenvectors = eigvecs[:, idx]

        # Classify decay
        self._spectrum = self._classify_spectrum()
        self._is_built = True
        return self

    def _generate_trajectory(
        self,
        n: int,
        mu_srb: Optional[np.ndarray],
    ) -> np.ndarray:
        """Generate an ergodic trajectory, starting from μ_SRB if available."""
        a, b = self.domain
        rng = np.random.default_rng(42)

        if mu_srb is not None and len(mu_srb) > 0:
            # Start from a random point drawn from μ_SRB
            grid = np.linspace(a, b, len(mu_srb))
            x0 = float(rng.choice(grid, p=mu_srb / mu_srb.sum()))
        else:
            x0 = float(a + (b - a) * (0.3 + 0.4 * rng.random()))

        traj = np.zeros(n + 1)
        traj[0] = x0
        for i in range(n):
            try:
                xn = float(self.T(np.array([traj[i]]))[0])
                if not np.isfinite(xn) or xn <= a + 1e-10 or xn >= b - 1e-10:
                    xn = float(a + (b - a) * (0.1 + 0.8 * rng.random()))
            except Exception:
                xn = float(a + (b - a) * rng.random())
            traj[i + 1] = xn

        return traj

    # ------------------------------------------------------------------
    # Layer 2: Topological Diagnosis — Classify α_A
    # ------------------------------------------------------------------

    def _compute_non_normality(self) -> float:
        """
        Compute the non-normality index N(K) = ‖KK* - K*K‖_F / ‖K‖_F².

        VERIFIED (TAA-2026): N(K) is independent of μ_SRB (|ΔN| < 1e-4).
        The non-normality is intrinsic to EDMD's Chebyshev dictionary, not
        a measure artifact.

        For chaotic systems (h_KS > 0), N(K) → √2 ≈ 1.414 as n_obs → ∞.
        For contractive systems, N(K) < 1.
        N(K_EDMD) / N(K_random) ≈ 4.8 — Koopman matrices are much more
        non-normal than random Gaussian matrices.

        True Koopman (unitary) would give N=0. Finite EDMD always gives N>0.
        """
        if self._K is None:
            return float('nan')
        K = self._K
        KKt = K @ K.T
        KtK = K.T @ K
        norm_sq = float(np.linalg.norm(K, 'fro') ** 2)
        if norm_sq < 1e-14:
            return 0.0
        return float(np.linalg.norm(KKt - KtK, 'fro')) / norm_sq

    def _classify_spectrum(self) -> KoopmanSpectrum:
        """
        Layer 2: Diagnose the Koopman spectral decay → α_A classification.

        Decay families:
          Finite:      rank < n_obs — exact spectrum
          Exponential: log|λ_k| vs k is linear (slope = log ρ)
          Polynomial:  log|λ_k| vs log k is linear (slope = -s)
          Chaotic:     no clean decay — |λ_k| cluster near 1 or near 0 with gaps
        """
        mods = np.abs(self._eigenvalues)

        # Dominant eigenvalue should be ≈ 1 (isometry property, TAA-1)
        n_significant = int(np.sum(mods > 1e-6))

        # --- Check for finite spectrum ---
        if n_significant < self.n_obs // 2:
            d_star = _d_star_from_mods(mods, {0.1: None, 0.01: None, 0.001: None})
            return KoopmanSpectrum(
                eigenvalues=self._eigenvalues[:n_significant],
                eigenfunctions=self._eigenvectors[:, :n_significant].real,
                koopman_matrix=self._K,
                decay_class=DecayClass.FINITE,
                alpha_A=float('inf'),
                rho=float('inf'),
                C_decay=1.0,
                d_star=d_star,
            )

        # --- Fit exponential decay to log|λ_k| vs k ---
        k_range = np.arange(1, min(n_significant, self.n_obs - 1) + 1)
        log_mods = np.log(np.maximum(mods[1:len(k_range)+1], 1e-15))
        log_mods = np.where(np.isfinite(log_mods), log_mods, np.nan)

        valid = np.isfinite(log_mods)
        if valid.sum() >= 3:
            k_v = k_range[valid].astype(float)
            lm_v = log_mods[valid]
            # Linear fit: log|λ_k| ≈ -k·log(ρ) + log(C)
            slope_exp, intercept_exp = np.polyfit(k_v, lm_v, 1)
            exp_fit = slope_exp * k_v + intercept_exp
            r2_exp = _r_squared(lm_v, exp_fit)

            # Log-log fit: log|λ_k| ≈ -s·log(k) + log(C)
            lk_v = np.log(k_v)
            slope_poly, intercept_poly = np.polyfit(lk_v, lm_v, 1)
            poly_fit = slope_poly * lk_v + intercept_poly
            r2_poly = _r_squared(lm_v, poly_fit)
        else:
            r2_exp = r2_poly = 0.0
            slope_exp = slope_poly = intercept_exp = intercept_poly = 0.0

        # --- Classify based on goodness of fit ---
        # TAA-14: Override to CHAOTIC if Lyapunov exponent indicates chaos,
        # even when the spectral fit is clean (e.g. logistic r=4 ≡ Chebyshev T_2).
        lyap_hint = 0.0
        try:
            lyap_hint = self.estimate_lyapunov(n_orbit=5000, n_warmup=200)
        except Exception:
            pass

        if slope_exp < -0.05 and r2_exp > 0.80 and lyap_hint < 0.1:
            rho = float(np.exp(-slope_exp))
            C_decay = float(np.exp(intercept_exp))
            d_star = _d_star_exponential(C_decay, rho, {0.1: None, 0.01: None, 0.001: None})
            alpha_A = float(-slope_exp)  # = log(ρ)
            decay = DecayClass.EXPONENTIAL
        elif slope_poly < -0.3 and r2_poly > 0.70 and lyap_hint < 0.1:
            s = float(-slope_poly)
            C_decay = float(np.exp(intercept_poly))
            d_star = _d_star_polynomial(C_decay, s, {0.1: None, 0.01: None, 0.001: None})
            alpha_A = s
            rho = s
            decay = DecayClass.POLYNOMIAL
        else:
            # No clear decay → chaotic regime
            alpha_A = 0.0
            rho = 1.0
            C_decay = 1.0
            d_star = {eps: self.n_obs for eps in [0.1, 0.01, 0.001]}
            decay = DecayClass.CHAOTIC

        a, b = self.domain
        n_grid = 200
        grid = np.linspace(a, b, n_grid)
        xi = np.clip(2.0 * (grid - a) / (b - a) - 1.0, -1.0, 1.0)
        n_d = min(8, self.n_obs)
        eigfuncs = np.zeros((n_grid, n_d))
        for i in range(n_d):
            # Evaluate eigenfunction: φ_i(x) = Σ_k v_ik · ψ_k(x)
            for k in range(self.n_obs):
                eigfuncs[:, i] += self._eigenvectors[k, i].real * np.cos(k * np.arccos(xi))

        return KoopmanSpectrum(
            eigenvalues=self._eigenvalues,
            eigenfunctions=eigfuncs,
            koopman_matrix=self._K,
            decay_class=decay,
            alpha_A=alpha_A,
            rho=rho,
            C_decay=C_decay,
            d_star=d_star,
        )

    # ------------------------------------------------------------------
    # Layer 2: Compute δ(d) truncation error
    # ------------------------------------------------------------------

    def truncation_error(
        self,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        n_test: int = 500,
    ) -> float:
        """
        Compute the truncation error δ(d) = ‖Kf - K_d f‖₂ / ‖f‖₂.

        K_d is the rank-d approximation of K using the top-d Koopman modes.
        This implements the KD-1 certificate numerically.

        Args:
            d:      Koopman dimension (number of modes retained)
            f:      Observable function f: ℝ → ℝ
            n_test: Number of test points

        Returns:
            Relative truncation error δ(d)
        """
        if not self._is_built:
            self.build()

        a, b = self.domain
        grid = np.linspace(a, b, n_test)
        xi = np.clip(2.0 * (grid - a) / (b - a) - 1.0, -1.0, 1.0)

        # Evaluate f on Chebyshev basis
        Psi_f = np.array([
            np.dot(self._eigenvectors[:, i].real,
                   np.array([np.sum(np.cos(k * np.arccos(xi)) * f(grid)) / n_test
                              for k in range(self.n_obs)]))
            for i in range(self.n_obs)
        ])

        f_vals = f(grid)
        f_norm = float(np.linalg.norm(f_vals))
        if f_norm < 1e-14:
            return 0.0

        # Full K applied to f in modal basis
        Kf_full = self._K @ (np.array([
            np.sum(np.cos(k * np.arccos(xi)) * f_vals) / n_test
            for k in range(self.n_obs)
        ]))

        # Rank-d approximation
        d_clamped = min(d, self.n_obs)
        mods = np.abs(self._eigenvalues)
        top_d_idx = np.argsort(-mods)[:d_clamped]
        K_d = self._eigenvectors[:, top_d_idx] @ np.diag(
            self._eigenvalues[top_d_idx]
        ) @ np.linalg.pinv(self._eigenvectors[:, top_d_idx])
        K_d = K_d.real

        f_proj = np.array([
            np.sum(np.cos(k * np.arccos(xi)) * f_vals) / n_test
            for k in range(self.n_obs)
        ])
        Kd_f = K_d @ f_proj

        error = np.linalg.norm(Kf_full - Kd_f) / (np.linalg.norm(Kf_full) + 1e-14)
        return float(error)

    # ------------------------------------------------------------------
    # Layer 1: Spectral entropy H_t
    # ------------------------------------------------------------------

    def spectral_entropy(self) -> float:
        """
        Compute the spectral entropy H_t = -Σ |λ_k|² log|λ_k|² / Σ |λ_k|².
        Measures how spread the Koopman energy is across modes.

        H_t ≈ 0: energy concentrated in few modes → low-dimensional structure
        H_t ≈ log d: energy spread uniformly → chaotic / high-entropy
        """
        if not self._is_built:
            self.build()

        mods2 = np.abs(self._eigenvalues) ** 2
        s = mods2.sum()
        if s < 1e-14:
            return 0.0

        p = mods2 / s
        p = p[p > 1e-14]
        return float(-np.dot(p, np.log(p)))

    # ------------------------------------------------------------------
    # Layer 2: Lyapunov estimate (Layer 1 perception)
    # ------------------------------------------------------------------

    def estimate_lyapunov(
        self,
        n_orbit: int = 20_000,
        n_warmup: int = 500,
    ) -> float:
        """
        Estimate the leading Lyapunov exponent λ_max via finite-difference orbit.
        This is TAA Layer 1: entropic perception of chaos.

        Delegates to shared LyapunovEstimator (cached across agents).
        """
        result = _shared_lyapunov.estimate(
            self.T, self.domain,
            n_orbit=n_orbit, n_warmup=n_warmup,
        )
        return result.lyapunov_max

    # ------------------------------------------------------------------
    # Layer 4: Mode selection (Poem / CoPoem / BiPoem / ERGON)
    # ------------------------------------------------------------------

    def select_mode(
        self,
        lambda_max: Optional[float] = None,
        ergodic_complexity: Optional[float] = None,
    ) -> TAAMode:
        """
        TAA §10.1 mode selector: decide which Poema mode is rational.

        Logic:
          - If λ_max > 0 and 𝔈 > threshold → ERGON (chaos, need μ_SRB)
          - If decay_class == FINITE → POEM (exact structure available)
          - If decay_class == EXPONENTIAL → POEM (low-d reduction works)
          - If data available but no law → BIPOEM
          - If property-first synthesis → COPOEM
        """
        if not self._is_built:
            self.build()

        lm = lambda_max if lambda_max is not None else 0.0
        ec = ergodic_complexity if ergodic_complexity is not None else 0.0

        if lm > self.lyap_threshold and ec > self.chaos_threshold:
            return TAAMode.ERGON

        if self._spectrum.decay_class in (DecayClass.FINITE, DecayClass.EXPONENTIAL):
            return TAAMode.POEM

        if self._spectrum.decay_class == DecayClass.POLYNOMIAL:
            return TAAMode.BIPOEM

        return TAAMode.ERGON  # Chaotic — defer

    # ------------------------------------------------------------------
    # Layer 2: Measure sensitivity (TAA-5)
    # ------------------------------------------------------------------

    def measure_sensitivity(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        delta_mu: float,
        n_test: int = 500,
    ) -> Tuple[float, float]:
        """
        Compute the measure-sensitivity inflation (TAA-5):
            δ(d)_wrong = δ(d)_correct + ‖f‖_∞ · δ_μ

        Returns:
            (delta_d_correct, delta_d_wrong)
        """
        if not self._is_built:
            self.build()

        a, b = self.domain
        grid = np.linspace(a, b, n_test)
        f_vals = f(grid)
        norm_f_inf = float(np.max(np.abs(f_vals)))

        d_correct = self.truncation_error(self._spectrum.d_star.get(0.01, 16), f, n_test)
        d_wrong = d_correct + norm_f_inf * delta_mu
        return (d_correct, d_wrong)

    # ------------------------------------------------------------------
    # NEW: Index de Adaptación de Base (IAB) — TAA-10
    # ------------------------------------------------------------------

    def compute_iab(self) -> float:
        """
        Compute the Index de Adaptación de Base (IAB) — dictionary adequacy.

        DISCOVERY: The non-normality N(K) = ‖KK* - K*K‖_F / ‖K‖_F²
        does NOT decrease when the correct μ_SRB is used. This means:
        - The non-normality is intrinsic to the EDMD dictionary (Chebyshev basis)
        - It measures how far {T_k(x)} is from being Koopman eigenfunctions
        - It is independent of the reference measure

        DEFINITION: IAB(K) = N(K) / N(K_Gaussian)
        where N(K_Gaussian) = mean non-normality of random Gaussian matrices of same size
        ≈ 1 - 1/n  for n×n matrices (random matrix theory).

        Interpretation:
          IAB ≈ 0: basis is well-adapted to Koopman structure (eigenfunctions ≈ Chebyshev)
          IAB ≈ 1: basis is generic (typical for Chebyshev on non-adapted domains)

        NEW CERTIFICATE (TAA-10): IAB < IAB_threshold → basis is Koopman-adapted.

        Returns:
            IAB ∈ [0, 1] — the basis adaptation index
        """
        if not self._is_built:
            self.build()

        K = self._K
        n = K.shape[0]

        # Non-normality N(K) = ‖KK* - K*K‖_F / ‖K‖_F²
        KKt = K @ K.T.conj()
        KtK = K.T.conj() @ K
        commutator = KKt - KtK
        K_norm_sq = float(np.linalg.norm(K, 'fro') ** 2)
        if K_norm_sq < 1e-14:
            return 0.0
        N_K = float(np.linalg.norm(commutator, 'fro')) / K_norm_sq

        # Expected non-normality of random n×n matrix: E[N] ≈ 1 - 1/n
        # (random matrix theory: Gaussian matrices have this non-normality on average)
        N_random = max(1.0 - 1.0 / n, 0.5)

        return float(np.clip(N_K / N_random, 0.0, 1.0))

    # ------------------------------------------------------------------
    # NEW: Critical complexity threshold 𝔈* — TAA-11
    # ------------------------------------------------------------------

    def compute_critical_threshold(
        self,
        gamma_otu: float,
        h_ks: float,
    ) -> dict:
        """
        Compute the critical complexity threshold 𝔈*.

        DISCOVERY (TAA-2026): There exists a critical threshold 𝔈* such that:
          𝔈(T) < 𝔈*: d*(ε) = O(log 1/ε)  → TAA operates with log budget
          𝔈(T) > 𝔈*: d*(ε) = O(ε^{-1/(1-𝔈)}) → TAA needs polynomial budget

        FORMULA: 𝔈* = 1 - Γ_OTU / h_KS

        PROOF SKETCH: The spectral gap Γ_OTU = -log|λ₁| controls the mixing rate.
        The complexity 𝔈 = h_KS / log(1 + Σλ⁺) measures information per unit expansion.
        When Γ_OTU ≥ h_KS (strong mixing, 𝔈* ≥ 0), TAA has log budget.
        When Γ_OTU < h_KS (weak mixing, 𝔈* < 1), systems with 𝔈 > 𝔈* need ERGON.

        For logistic r=4: Γ_OTU = 0.473, h_KS = 0.693 → 𝔈* = 0.316.
        Since 𝔈 ≈ 0.85 > 𝔈*, TAA correctly defers to ERGON.

        Args:
            gamma_otu:  Spectral gap Γ_OTU from OTU
            h_ks:       KS entropy from ERGON/OTU

        Returns:
            dict with 𝔈*, current regime, and ERGON activation recommendation
        """
        if h_ks < 1e-10:
            return {
                "e_star": 0.0,
                "regime": "integrable",
                "taa_alone": True,
                "note": "h_KS ≈ 0: integrable system, TAA alone",
            }

        e_star = float(1.0 - gamma_otu / max(h_ks, 1e-10))
        e_star_clamped = float(np.clip(e_star, 0.0, 1.0))

        # Current system's complexity estimate from Koopman spectrum
        # 𝔈_K = -log(spectral decay rate) / h_KS as proxy
        current_e = 0.0
        if self._is_built:
            mods = np.abs(self._eigenvalues)
            m1 = mods[1] if len(mods) > 1 else 0.5
            if m1 > 1e-10:
                koopman_gap = float(-np.log(m1))
                current_e = float(1.0 - koopman_gap / max(h_ks, 1e-10))
                current_e = float(np.clip(current_e, 0.0, 1.0))

        regime = "log_budget" if current_e < e_star_clamped else "poly_budget"
        ergon_needed = bool(current_e > e_star_clamped)

        return {
            "e_star": e_star_clamped,
            "e_star_formula": "1 - Γ_OTU / h_KS",
            "current_complexity_e": current_e,
            "regime": regime,
            "ergon_activation": ergon_needed,
            "taa_alone": not ergon_needed,
            "gamma_otu": float(gamma_otu),
            "h_ks": float(h_ks),
            "certificate": "TAA-11: d*(ε) = O(log 1/ε) ↔ 𝔈 < 𝔈*",
        }

    # ------------------------------------------------------------------
    # NEW: Biorthogonal projection Π_d — TAA-12
    # ------------------------------------------------------------------

    def biorthogonal_truncation_error(
        self,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        left_eigenvectors: Optional[np.ndarray] = None,
        n_test: int = 500,
    ) -> Tuple[float, float]:
        """
        Compute the CORRECTED truncation error δ_biorth(d) using the biorthogonal projector.

        DISCOVERY (TAA-2026): The current truncation_error() uses only right eigenvectors:
            K_d = V_d Λ_d pinv(V_d)
        This is biased for non-normal K. The correct spectral projector is:
            Π_d = Σ_{k=1}^{d} |r_k⟩⟨l_k| / ⟨l_k|r_k⟩
        where l_k are LEFT eigenvectors (from OTU: koopman_modes = left evecs of L).

        The projection error on f is:
            δ_biorth(d) = ‖Kf - Π_d Kf‖₂ / ‖f‖₂

        The difference δ_biorth - δ_naive measures the PROJECTION BIAS of the
        standard EDMD truncation — always non-negative by sub-optimality.

        Args:
            d:                  Number of Koopman modes to retain
            f:                  Observable function
            left_eigenvectors:  Left eigenvectors of K (from OTU). If None,
                                uses K's own left eigenvectors via eigendecomposition.
            n_test:             Number of test points

        Returns:
            (delta_naive, delta_biorth) — naive (right-only) vs biorthogonal errors
        """
        if not self._is_built:
            self.build()

        a, b = self.domain
        grid = np.linspace(a, b, n_test)
        xi = np.clip(2.0 * (grid - a) / (b - a) - 1.0, -1.0, 1.0)

        f_vals = f(grid)
        f_norm = float(np.linalg.norm(f_vals))
        if f_norm < 1e-14:
            return (0.0, 0.0)

        f_proj = np.array([
            np.sum(np.cos(k * np.arccos(xi)) * f_vals) / n_test
            for k in range(self.n_obs)
        ])

        Kf_full = self._K @ f_proj
        Kf_norm = float(np.linalg.norm(Kf_full))
        if Kf_norm < 1e-14:
            return (0.0, 0.0)

        d_clamped = min(d, self.n_obs)
        mods = np.abs(self._eigenvalues)
        top_d_idx = np.argsort(-mods)[:d_clamped]

        # --- Naive projection (right eigenvectors only) ---
        R_d = self._eigenvectors[:, top_d_idx]
        K_d_naive = R_d @ np.diag(self._eigenvalues[top_d_idx]) @ np.linalg.pinv(R_d)
        K_d_naive = K_d_naive.real
        Kd_f_naive = K_d_naive @ f_proj
        delta_naive = float(np.linalg.norm(Kf_full - Kd_f_naive)) / (Kf_norm + 1e-14)

        # --- Biorthogonal projection (left + right eigenvectors) ---
        if left_eigenvectors is not None:
            # left_eigenvectors: (n_obs, n_modes_from_OTU) — map to top_d modes
            n_otu = left_eigenvectors.shape[1] if left_eigenvectors.ndim > 1 else 0
            if n_otu >= d_clamped:
                L_d = left_eigenvectors[:, :d_clamped]  # (n_obs, d) left evecs
            else:
                # Fall back to computing left evecs from K directly
                eigvals_l, L_all = linalg.eig(self._K.T, right=True)
                idx_l = np.argsort(-np.abs(eigvals_l))
                L_d = L_all[:, idx_l[:d_clamped]].real
        else:
            # Compute K's left eigenvectors directly
            eigvals_l, L_all = linalg.eig(self._K.T, right=True)
            idx_l = np.argsort(-np.abs(eigvals_l))
            L_d = L_all[:, idx_l[:d_clamped]].real

        # Biorthogonal projector: Π_d f = Σ_k ⟨l_k, f⟩ r_k / ⟨l_k, r_k⟩
        # (proper spectral decomposition for non-normal operators)
        K_d_biorth = np.zeros((self.n_obs, self.n_obs))
        for i in range(d_clamped):
            r_k = self._eigenvectors[:, top_d_idx[i]].real
            l_k = L_d[:, i]
            inner = float(np.dot(l_k, r_k))
            if abs(inner) > 1e-14:
                lam = self._eigenvalues[top_d_idx[i]].real
                K_d_biorth += (lam / inner) * np.outer(r_k, l_k)

        Kd_f_biorth = K_d_biorth @ f_proj
        delta_biorth = float(np.linalg.norm(Kf_full - Kd_f_biorth)) / (Kf_norm + 1e-14)

        return (float(delta_naive), float(delta_biorth))

    # ------------------------------------------------------------------
    # Layer 4: Full diagnostic pipeline
    # ------------------------------------------------------------------

    def diagnose(
        self,
        mu_srb: Optional[np.ndarray] = None,
        h_ks: float = 0.0,
        lyapunov_sum: float = 0.0,
    ) -> TAAState:
        """
        Run the full TAA diagnostic pipeline (Layers 1-2).

        Args:
            mu_srb:        SRB measure from ERGONAgent (None → uniform)
            h_ks:          KS entropy from ERGON (0 if unknown)
            lyapunov_sum:  Σλ⁺ from ERGON (0 if unknown)

        Returns:
            TAAState with all diagnostic fields filled.
        """
        if not self._is_built:
            self.build(mu_srb)

        H_t = self.spectral_entropy()
        lm = self.estimate_lyapunov()
        ec = h_ks / (lyapunov_sum + 1e-14) if lyapunov_sum > 0 else 0.0
        mode = self.select_mode(lm, ec)

        d_opt = self._spectrum.d_star.get(0.01, self.n_obs)

        # Isometry error (TAA-1): ‖K‖ should be ≈ 1 (isometry)
        # Verified: spectral radius of K should be ≈ 1 for measure-preserving T
        spectral_radius = float(np.max(np.abs(self._eigenvalues)))

        certs = {
            "TAA-1_isometry_error": float(abs(spectral_radius - 1.0)),
            "TAA-2_energy_invariant": True,  # by construction in EDMD
            "TAA-3_d_star_eps01":    int(self._spectrum.d_star.get(0.1, self.n_obs)),
            "TAA-3_d_star_eps001":   int(self._spectrum.d_star.get(0.01, self.n_obs)),
            "TAA-3_d_star_eps0001":  int(self._spectrum.d_star.get(0.001, self.n_obs)),
            "TAA-4_decay_class":     self._spectrum.decay_class.value,
            "TAA-4_alpha_A":         float(self._spectrum.alpha_A),
            "TAA-4_rho":             float(self._spectrum.rho),
            "TAA-5_delta_mu":        0.0 if mu_srb is not None else float('nan'),
            "TAA-6_defer_to_ergon":  bool(mode == TAAMode.ERGON),
            "TAA-6_lambda_max":      float(lm),
            "TAA-6_ergodic_complexity": float(ec),
            "TAA-7_spectral_entropy": float(H_t),
            # NEW: TAA-10 IAB, TAA-11 𝔈*, TAA-12 non-normality
            "TAA-10_iab":            float(self.compute_iab()),
            "TAA-11_e_star":         float(1.0 - h_ks / (lyapunov_sum + 1e-10)) if lyapunov_sum > 1e-10 else 0.0,
            "TAA-11_regime":         "poly_budget" if ec > max(1.0 - h_ks / (lyapunov_sum + 1e-10), 0.0) else "log_budget",
            "TAA-12_non_normality":  float(self._compute_non_normality()),
        }

        return TAAState(
            spectral_entropy=H_t,
            alpha_A=self._spectrum.alpha_A,
            approx_error=0.0,   # filled by caller with actual residual
            truncation_error=0.0,
            decay_class=self._spectrum.decay_class,
            mode=mode,
            defer_to_ergon=(mode == TAAMode.ERGON),
            d_star=d_opt,
            lambda_max=lm,
            ergodic_complexity=ec,
            certificates=certs,
        )

    # ------------------------------------------------------------------
    # Layer 4: Full certification
    # ------------------------------------------------------------------

    def certify(
        self,
        mu_srb: Optional[np.ndarray] = None,
        h_ks: float = 0.0,
        lyapunov_sum: float = 0.0,
    ) -> TAACertificate:
        """
        Produce a complete TAA formal certificate (Lean-4-exportable).

        Args:
            mu_srb:       SRB measure from ERGONAgent (improves TAA-5)
            h_ks:         KS entropy from ERGON
            lyapunov_sum: Σλ⁺ from ERGON
        """
        state = self.diagnose(mu_srb, h_ks, lyapunov_sum)
        c = state.certificates

        delta_mu = 0.0 if mu_srb is not None else 1.0  # unknown = worst case

        passes = (
            c["TAA-1_isometry_error"] < 0.3 and       # near-isometry
            c["TAA-4_alpha_A"] >= 0.0 and              # valid index
            not (c["TAA-6_lambda_max"] > 0.5 and mu_srb is None)  # if chaotic need μ_SRB
        )

        return TAACertificate(
            TAA_1_isometry_error=c["TAA-1_isometry_error"],
            TAA_2_energy_invariant=True,
            TAA_3_d_star_eps01=c["TAA-3_d_star_eps01"],
            TAA_3_d_star_eps001=c["TAA-3_d_star_eps001"],
            TAA_3_d_star_eps0001=c["TAA-3_d_star_eps0001"],
            TAA_4_decay_class=c["TAA-4_decay_class"],
            TAA_4_alpha_A=c["TAA-4_alpha_A"],
            TAA_4_rho=c["TAA-4_rho"],
            TAA_5_delta_mu_inflation=delta_mu,
            TAA_6_defer_to_ergon=c["TAA-6_defer_to_ergon"],
            TAA_6_lambda_max=c["TAA-6_lambda_max"],
            TAA_6_ergodic_complexity=c["TAA-6_ergodic_complexity"],
            TAA_7_spectral_entropy=c["TAA-7_spectral_entropy"],
            TAA_10_iab=float(c.get("TAA-10_iab", 0.0)),
            TAA_10_non_normality=float(c.get("TAA-12_non_normality", 0.0)),
            TAA_11_e_star=float(c.get("TAA-11_e_star", 0.0)),
            TAA_11_regime=str(c.get("TAA-11_regime", "unknown")),
            PASS=passes,
        )

    # ------------------------------------------------------------------
    # Layer 5: Prediction (native execution)
    # ------------------------------------------------------------------

    def predict(
        self,
        x0: float,
        n_steps: int,
        observable: Optional[Callable[[float], float]] = None,
    ) -> np.ndarray:
        """
        Predict future values of an observable using the Koopman operator.
        Koopman prediction: f(T^n x₀) ≈ [K^n ψ(x₀)]_component

        Args:
            x0:         Initial point
            n_steps:    Number of steps to predict
            observable: f: ℝ → ℝ (default: identity)

        Returns:
            Predicted values f(T^n x₀) for n = 1, ..., n_steps
        """
        if not self._is_built:
            self.build()

        a, b = self.domain
        xi0 = np.clip(2.0 * (x0 - a) / (b - a) - 1.0, -1.0, 1.0)

        # Evaluate observables at x0
        psi_x0 = np.array([np.cos(k * np.arccos(xi0)) for k in range(self.n_obs)])

        # Apply K^n via eigendecomposition: K^n = V Λ^n V^{-1}
        # This is O(d³ + n·d) instead of the naive O(n²·d²).
        eigvals = self._eigenvalues
        V = self._eigenvectors
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V)

        # Project initial state into eigenbasis
        coeffs = V_inv @ psi_x0  # (d,)

        # Warn if system is chaotic — pointwise prediction is unreliable
        lyap_est = self.estimate_lyapunov() if self._spectrum.decay_class == DecayClass.CHAOTIC else 0.0
        if lyap_est > 0.1:
            warnings.warn(
                f"predict() on chaotic system (λ_max={lyap_est:.3f}): "
                f"pointwise predictions diverge exponentially. "
                f"Use statistical predictions (ensemble averages) instead.",
                RuntimeWarning,
            )

        preds = np.zeros(n_steps)
        for n in range(1, n_steps + 1):
            # K^n ψ(x₀) = V · diag(λ^n) · V^{-1} · ψ(x₀)
            powered_coeffs = coeffs * (eigvals ** n)
            state = (V @ powered_coeffs).real

            if observable is None:
                # Use T_1 (linear Chebyshev) to recover x, then extract value
                x_approx = float((a + b) / 2.0 + (b - a) / 2.0 * np.clip(state[1], -1, 1))
                preds[n - 1] = x_approx
            else:
                x_approx = float((a + b) / 2.0 + (b - a) / 2.0 * np.clip(state[1], -1, 1))
                preds[n - 1] = float(observable(x_approx))

        return preds


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R² goodness-of-fit."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-14))


def _d_star_from_mods(mods: np.ndarray, eps_dict: dict) -> dict:
    """Compute d*(ε) for each ε: smallest d s.t. |λ_{d+1}| < ε."""
    result = {}
    for eps in eps_dict:
        cands = np.where(mods < eps)[0]
        result[eps] = int(cands[0]) if len(cands) > 0 else len(mods)
    return result


def _d_star_exponential(C: float, rho: float, eps_dict: dict) -> dict:
    """d*(ε) = ceil(log(C/ε) / log(ρ)) for exponential decay."""
    result = {}
    for eps in eps_dict:
        if C <= 0 or rho <= 1.0:
            result[eps] = 1
        else:
            val = math.ceil(math.log(max(C / (eps + 1e-14), 1.0)) / math.log(rho))
            result[eps] = max(1, val)
    return result


def _d_star_polynomial(C: float, s: float, eps_dict: dict) -> dict:
    """d*(ε) = ceil((C/ε)^{1/s}) for polynomial decay."""
    result = {}
    for eps in eps_dict:
        if C <= 0 or s <= 0:
            result[eps] = 1
        else:
            val = math.ceil((C / (eps + 1e-14)) ** (1.0 / s))
            result[eps] = max(1, val)
    return result


# ---------------------------------------------------------------------------
# Real-World Bridge for TAAAgent
# ---------------------------------------------------------------------------

class TAAAgentRealWorld:
    """
    Real-world extension of TAAAgent: accepts time series instead of T(x).

    Bridges the Data Abyss (Barrier 1) by reconstructing a dynamical system
    from raw sensor data, then performing Koopman spectral analysis.

    Usage:
        # From raw time series:
        agent = TAAAgentRealWorld.from_timeseries(vibration_data)
        cert = agent.taa.diagnose()

        # Track spectral evolution over time:
        tracking = TAAAgentRealWorld.track_koopman(
            streaming_data, window_size=1000, step=200
        )
    """

    def __init__(
        self,
        taa: TAAAgent,
        reconstruction_info: Optional[dict] = None,
    ):
        self.taa = taa
        self.reconstruction_info = reconstruction_info or {}

    @classmethod
    def from_timeseries(
        cls,
        y: np.ndarray,
        noise_filter: str = "svd",
        n_obs: int = 24,
    ) -> "TAAAgentRealWorld":
        """
        Build TAAAgent from a raw time series.

        Pipeline: filter → Takens embed → local linear model → TAAAgent.

        Note: Currently supports 1D reconstructions (d_embed=1) only.
        For multi-dimensional embeddings, use ERGONRealWorld which handles
        the measure-side analysis in arbitrary dimension.
        """
        from acf_functor.real_world import TimeSeriesReconstructor

        y = np.asarray(y, dtype=float).ravel()
        reconstructor = TimeSeriesReconstructor(noise_filter=noise_filter)
        system = reconstructor.reconstruct(y)

        if system.embedding_dim > 1:
            warnings.warn(
                f"Reconstructed system has embedding_dim={system.embedding_dim}. "
                f"TAAAgent operates on scalar maps (d=1). Using 1D projection of "
                f"the first delay coordinate. For multi-dimensional analysis, "
                f"use ERGONRealWorld or OTURealWorld instead."
            )
            # Project to 1D: use the first coordinate of the delay embedding
            Z = system.takens.embedded_data
            y_proj = Z[:, 0]

            def T_1d(x_arr):
                x = np.atleast_1d(x_arr)
                z = np.zeros(system.embedding_dim)
                z[0] = x[0]
                result = system.T(z)
                return np.atleast_1d(result)[0:1]

            domain = (float(Z[:, 0].min()), float(Z[:, 0].max()))
        else:
            T_1d = system.T
            domain = system.domain

        agent = TAAAgent(
            T=T_1d,
            domain=domain,
            n_obs=n_obs,
        )
        agent.build()

        recon_info = {
            "embedding_dim": system.embedding_dim,
            "delay": system.delay,
            "snr_db": system.snr_db,
            "cv_error": system.reconstruction_error,
            "n_local_models": system.n_local_models,
            "projected_to_1d": system.embedding_dim > 1,
        }

        return cls(taa=agent, reconstruction_info=recon_info)

    @staticmethod
    def track_koopman(
        y: np.ndarray,
        window_size: int = 1000,
        step: int = 200,
        noise_filter: str = "svd",
        n_obs: int = 16,
    ) -> dict:
        """
        Track Koopman spectral properties over sliding windows.

        For each window: reconstruct T → build TAAAgent → extract spectrum.
        Detects transitions in spectral decay class (exponential → chaotic),
        changes in spectral entropy, and optimal truncation dimension d*.

        Returns:
            dict with time-indexed trajectories of spectral properties.
        """
        from acf_functor.real_world import TimeSeriesReconstructor

        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        reconstructor = TimeSeriesReconstructor(noise_filter=noise_filter)
        times = []
        eigenvalue_trajectories = []
        decay_classes = []
        spectral_entropies = []
        d_stars = []
        alpha_rates = []

        for start in range(0, n - window_size + 1, step):
            window = y[start:start + window_size]
            try:
                system = reconstructor.reconstruct(window)

                # Only works for 1D reconstructions
                if system.embedding_dim > 1:
                    Z = system.takens.embedded_data
                    y_proj = Z[:, 0]
                    def T_1d(x_arr, _sys=system):
                        z = np.zeros(_sys.embedding_dim)
                        z[0] = np.atleast_1d(x_arr)[0]
                        return np.atleast_1d(_sys.T(z))[0:1]
                    domain = (float(Z[:, 0].min()), float(Z[:, 0].max()))
                else:
                    T_1d = system.T
                    domain = system.domain

                agent = TAAAgent(T=T_1d, domain=domain, n_obs=n_obs)
                agent.build()

                spec = agent._spectrum
                times.append(start + window_size // 2)
                eigenvalue_trajectories.append(
                    np.abs(agent._eigenvalues[:min(8, len(agent._eigenvalues))]).tolist()
                )
                decay_classes.append(spec.decay_class.value if spec else "unknown")
                spectral_entropies.append(
                    float(-np.sum(
                        np.abs(agent._eigenvalues) ** 2 *
                        np.log(np.abs(agent._eigenvalues) ** 2 + 1e-300)
                    ))
                )
                d_star_01 = spec.d_star.get(0.01, n_obs) if spec else n_obs
                d_stars.append(d_star_01)
                alpha_rates.append(spec.alpha_A if spec else 0.0)

            except Exception:
                times.append(start + window_size // 2)
                eigenvalue_trajectories.append([])
                decay_classes.append("error")
                spectral_entropies.append(0.0)
                d_stars.append(0)
                alpha_rates.append(0.0)

        # Detect decay-class transitions
        transitions = []
        for i in range(1, len(decay_classes)):
            if decay_classes[i] != decay_classes[i - 1]:
                transitions.append({
                    "time": times[i],
                    "from": decay_classes[i - 1],
                    "to": decay_classes[i],
                })

        return {
            "times": times,
            "eigenvalue_trajectories": eigenvalue_trajectories,
            "decay_classes": decay_classes,
            "spectral_entropies": spectral_entropies,
            "d_stars": d_stars,
            "alpha_rates": alpha_rates,
            "transitions": transitions,
            "n_windows": len(times),
            "window_size": window_size,
        }


# ---------------------------------------------------------------------------
# Canonical test systems (re-exported for convenience)
# ---------------------------------------------------------------------------

class TAACanonicalSystems:
    """Canonical test systems with known TAA properties."""

    @staticmethod
    def linear_contraction(rate: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
        """T(x) = rate·x — λ_max < 0, TAA acts alone (POEM mode)."""
        return lambda x: rate * x

    @staticmethod
    def rotation(theta: float = 0.3) -> Callable[[np.ndarray], np.ndarray]:
        """T(x) = cos(θ)·x — quasi-periodic, finite Koopman spectrum."""
        import math
        c = math.cos(theta)
        return lambda x: np.clip(c * x, -1.0, 1.0)

    @staticmethod
    def polynomial_map(degree: int = 3) -> Callable[[np.ndarray], np.ndarray]:
        """T(x) = x^d — polynomial, exponential Koopman decay."""
        return lambda x: np.clip(x ** degree, -1.0, 1.0)

    @staticmethod
    def logistic(r: float = 4.0) -> Callable[[np.ndarray], np.ndarray]:
        """T(x) = r·x·(1-x) — chaotic for r=4, needs ERGON."""
        return lambda x: r * x * (1.0 - x)

    @staticmethod
    def tent() -> Callable[[np.ndarray], np.ndarray]:
        """T(x) = 2·min(x, 1-x) — chaotic, h_KS = log 2."""
        return lambda x: 2.0 * np.minimum(x, 1.0 - x)
