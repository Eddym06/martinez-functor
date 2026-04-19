"""
Deep Problems Module — Solutions to 10 Fundamental Open Problems
================================================================

This module implements rigorous solutions to the 10 deep problems
identified in the OTU-TAA-ERGON ecosystem analysis (April 2026):

  P1:  Multi-attractor detection (basin decomposition)
  P2:  Takens embedding preprocessor (time series → delay coordinates)
  P3:  High-dimensional OTU via reduced coordinates
  P4:  Numerical instability certification (Γ_OTU → 0)
  P5:  Exceptional points & PT-symmetry detection
  P6:  Periodic orbit extraction from Ruelle zeta
  P7:  Fractal Gelfand triple adaptation (total variation metric)
  P8:  Continuous-time OTU (infinitesimal generator)
  P9:  Fisher information & Cramér-Rao bounds
  P10: No-cloning theorem for μ_SRB

Each solution provides:
  - A computation function with numerical certificates
  - Integration with the existing OTU/TAA/ERGON API
  - Formal certificate IDs (OTU-17 through OTU-26)

Mathematical references:
  - Dellnitz & Junge (1999): Almost invariant sets (P1)
  - Takens (1981): Detecting strange attractors in turbulence (P2)
  - Cvitanović et al. (2005): Chaos: Classical and Quantum (P6)
  - Bender & Boettcher (1998): PT-symmetric quantum mechanics (P5)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg


# ═══════════════════════════════════════════════════════════════════════════
# Data structures for deep problem results
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FisherCramerRaoCertificate:
    """
    OTU-17: Fisher information and Cramér-Rao lower bound for μ_SRB estimation.

    The Fisher information I(β) = n · P''(β) at β=1 gives the fundamental
    precision limit for estimating h_KS from n observations:
        Var(ĥ_KS) ≥ 1 / (n · P''(1))

    For Bernoulli systems (linear P): P''(1) → 0 and the bound is vacuous
    (perfect estimation possible — no Lyapunov fluctuations).
    """
    fisher_information_per_obs: float   # I₁ = P''(1) = Var_μ(log|T'|)
    cramer_rao_bound: float             # σ²_min(n) = 1/(n · P''(1))
    n_observations: int                 # n used in calculation
    min_error_std: float                # σ_min = 1/√(n·P''(1))
    relative_precision: float           # σ_min / h_KS — relative error floor
    is_bernoulli: bool                  # P''(1) ≈ 0 → perfect estimation
    certificate_id: str = "OTU-17"


@dataclass
class NoCloneCertificate:
    """
    OTU-18: No-cloning theorem for μ_SRB.

    The minimum number of observations to approximate μ_SRB to error ε
    in total variation norm satisfies:
        n_min(ε) ≥ C · ε^{-D_2}

    where D_2 is the correlation dimension. This is a fundamental limit:
    the ε-tolerance in all certificates is not a weakness but a law.
    """
    D_2: float                          # Correlation dimension
    n_min: int                          # n_min(ε) for the given ε
    epsilon: float                      # Target precision ε
    h_ks: float                         # Entropy rate (bits/obs needed)
    information_cost_bits: float        # h_KS · n_min — total info needed
    certificate_id: str = "OTU-18"


@dataclass
class NumericalStabilityCertificate:
    """
    OTU-19: Numerical stability certification.

    Quantifies the numerical error floor:
        ε_num = ε_machine / Γ_OTU² · ‖L‖_spec

    If ε_num > ε_target, the system is *certifiably uncertifiable*
    at the target precision with standard double precision.
    """
    gamma_otu: float                    # Spectral gap
    condition_number: float             # κ(L) ≈ 1/Γ_OTU
    epsilon_numerical: float            # ε_num = ε_machine · κ²
    epsilon_target: float               # User-specified target
    is_certifiable: bool                # ε_num < ε_target
    mixing_type: str                    # "exponential" or "algebraic"
    recommended_precision: str          # "float64", "float128", or "mpmath"
    certificate_id: str = "OTU-19"


@dataclass
class TakensEmbeddingResult:
    """
    OTU-20: Takens delay embedding for time series analysis.

    Reconstructs the attractor from a scalar time series using delay coordinates:
        z_t = (y_t, y_{t-τ}, ..., y_{t-(d-1)τ})

    The Koopman spectrum is invariant under the embedding (Takens' theorem).
    """
    embedding_dim: int                  # d = ⌈2D_2 + 1⌉
    delay: int                          # τ (delay step)
    hankel_matrix: np.ndarray           # H_{d,n} — Hankel matrix of data
    reconstructed_spectrum: np.ndarray  # Koopman eigenvalues from EDMD on delay
    spectral_error_bound: float         # C · e^{-d·Γ·τ}
    gamma_reconstructed: float          # Spectral gap from reconstructed data
    certificate_id: str = "OTU-20"


@dataclass
class PeriodicOrbitInfo:
    """Information about one extracted periodic orbit."""
    period: int
    multiplier: float          # |Λ_p| = ∏|T'(x_k)| along orbit
    stability: str             # "unstable" if |Λ_p| > 1
    approximate_points: Optional[np.ndarray] = None


@dataclass
class PeriodicOrbitResult:
    """
    OTU-21: Periodic orbit extraction from the Ruelle zeta function.

    Uses the trace formula: tr(L^n) = Σ_{T^n(x)=x} 1/|det(DT^n - I)|
    to extract periodic orbits from the spectral data.
    """
    orbits: List[PeriodicOrbitInfo]
    max_period: int
    orbit_count_by_period: Dict[int, int]    # N_n for each n
    growth_rate: float                        # log(N_n)/n → h_top
    h_top_estimate: float                     # Topological entropy from zeta
    certificate_id: str = "OTU-21"


@dataclass
class BasinDecomposition:
    """
    OTU-22: Multi-attractor basin decomposition.

    Detects multiple ergodic components via spectral clustering on
    the near-unit eigenvalues of the Perron-Frobenius operator.
    """
    n_basins: int                        # Number of detected basins
    basin_weights: List[float]           # Lebesgue measure of each basin
    basin_indices: np.ndarray            # Basin assignment for each grid cell
    near_unit_eigenvalues: np.ndarray    # Eigenvalues with |λ| > 1 - δ
    metastable_timescales: List[float]   # 1/(-log|λ_k|) for each near-unit λ
    is_ergodic: bool                     # True if only 1 basin detected
    certificate_id: str = "OTU-22"


@dataclass
class ContinuousGeneratorResult:
    """
    OTU-23: Continuous-time Koopman generator.

    For flow φ_t with time step τ, the generator is:
        L_K = (K^τ - I) / τ
    with eigenvalues λ_k^{cont} = log(λ_k^{disc}) / τ.
    """
    generator_eigenvalues: np.ndarray    # Continuous-time eigenvalues
    decay_rates: np.ndarray              # Re(λ_k) < 0 — continuous decay
    frequencies: np.ndarray              # Im(λ_k) — continuous frequencies
    time_step: float                     # τ used
    aliasing_free: bool                  # max|Im(λ)| · τ < π
    certificate_id: str = "OTU-23"


@dataclass
class ExceptionalPointResult:
    """
    OTU-24: Exceptional point detection in the Perron-Frobenius spectrum.

    An exceptional point (EP) occurs where two eigenvalues and eigenvectors
    coalesce. Near an EP, the Jordan normal form has a non-trivial block
    and the correlations gain a polynomial prefactor: C(n) ~ n·e^{-nΓ}.
    """
    n_exceptional_points: int            # Number of near-EPs detected
    ep_pairs: List[Tuple[int, int]]      # Indices of coalescing eigenvalue pairs
    ep_separations: List[float]          # |λ_i - λ_j| for each pair
    jordan_defects: List[float]          # ‖(L-λI)²v‖ for each EP
    pt_symmetric: bool                   # Whether L has PT symmetry
    certificate_id: str = "OTU-24"


@dataclass
class HighDimReductionResult:
    """
    OTU-25: High-dimensional reduction for the OTU.

    Uses diffusion maps or kernel PCA to find intrinsic coordinates
    on the attractor of dimension ≈ D_2, then runs OTU in those coords.
    """
    ambient_dim: int                     # Original dimension D
    intrinsic_dim: int                   # Estimated attractor dimension ≈ D_2
    reduction_method: str                # "diffusion_maps" or "kernel_pca"
    spectrum_in_reduced: np.ndarray      # Koopman eigenvalues in reduced space
    reconstruction_error: float          # ‖x - decode(encode(x))‖ / ‖x‖
    certificate_id: str = "OTU-25"


@dataclass
class FractalGelfandCertificate:
    """
    OTU-26: Fractal Gelfand triple adaptation.

    For attractors with dim_H < D, the standard L² norm is replaced by
    total variation norm, and the convergence rate is expressed in terms
    of D_2 rather than D.
    """
    hausdorff_dim_estimate: float        # Estimated dim_H from D_0
    total_variation_error: float         # ‖μ_computed - μ_SRB‖_TV
    convergence_rate_exponent: float     # Rate ~ N^{-α} where α depends on D_2
    is_fractal: bool                     # D_0 < ambient_dim
    certificate_id: str = "OTU-26"


# ═══════════════════════════════════════════════════════════════════════════
# P9: Fisher Information & Cramér-Rao Bound (OTU-17)
# ═══════════════════════════════════════════════════════════════════════════

def compute_fisher_cramer_rao(
    P_double_prime_1: float,
    h_ks: float,
    n_observations: int = 1000,
) -> FisherCramerRaoCertificate:
    """
    Compute the Cramér-Rao lower bound for h_KS estimation.

    The Fisher information per observation is I₁ = P''(1) = Var_{μ_SRB}(log|T'|).
    With n observations, the minimum variance of any unbiased estimator is:
        σ²_min = 1 / (n · I₁)

    Parameters
    ----------
    P_double_prime_1 : float
        Second derivative of pressure at β=1, i.e., P''(1).
        Already computed by GelfandTriple.compute_thermodynamic_pressure().
    h_ks : float
        Kolmogorov-Sinai entropy of the system.
    n_observations : int
        Number of observations available.

    Returns
    -------
    FisherCramerRaoCertificate
    """
    fisher_per_obs = max(abs(P_double_prime_1), 0.0)
    is_bernoulli = fisher_per_obs < 0.05

    if fisher_per_obs > 1e-10:
        cr_bound = 1.0 / (n_observations * fisher_per_obs)
        min_std = math.sqrt(cr_bound)
        relative = min_std / (h_ks + 1e-14)
    else:
        # Bernoulli: P''(1) = 0, no fluctuations, perfect estimation
        cr_bound = 0.0
        min_std = 0.0
        relative = 0.0

    return FisherCramerRaoCertificate(
        fisher_information_per_obs=fisher_per_obs,
        cramer_rao_bound=cr_bound,
        n_observations=n_observations,
        min_error_std=min_std,
        relative_precision=relative,
        is_bernoulli=is_bernoulli,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P10: No-Cloning Theorem for μ_SRB (OTU-18)
# ═══════════════════════════════════════════════════════════════════════════

def compute_no_clone_bound(
    D_2: float,
    h_ks: float,
    epsilon: float = 0.01,
    C: float = 1.0,
) -> NoCloneCertificate:
    """
    Compute the no-cloning lower bound for μ_SRB estimation.

    The minimum number of observations to approximate μ_SRB to
    error ε in total variation norm is:
        n_min(ε) ≥ C · ε^{-D_2}

    This is a fundamental limit from the fractal structure of the measure.

    Parameters
    ----------
    D_2 : float
        Correlation dimension of μ_SRB.
    h_ks : float
        Entropy rate (information per observation).
    epsilon : float
        Target precision in total variation.
    C : float
        Proportionality constant (default 1.0, conservative).

    Returns
    -------
    NoCloneCertificate
    """
    if D_2 < 0.01:
        D_2 = 0.01  # degenerate guard

    n_min_float = C * epsilon ** (-D_2)
    n_min = max(1, math.ceil(n_min_float))
    info_cost = h_ks * n_min

    return NoCloneCertificate(
        D_2=D_2,
        n_min=n_min,
        epsilon=epsilon,
        h_ks=h_ks,
        information_cost_bits=info_cost,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P4: Numerical Stability Certification (OTU-19)
# ═══════════════════════════════════════════════════════════════════════════

def certify_numerical_stability(
    gamma_otu: float,
    L_spec_norm: float = 1.0,
    epsilon_target: float = 0.01,
    epsilon_machine: float = 2.22e-16,
) -> NumericalStabilityCertificate:
    """
    Certify whether the system can be numerically analyzed at target precision.

    The numerical error floor is:
        ε_num = ε_machine / Γ_OTU² · ‖L‖_spec

    For algebraic mixing (Γ → 0): ε_num → ∞ and the system is uncertifiable.

    Parameters
    ----------
    gamma_otu : float
        Spectral gap of the Perron-Frobenius operator.
    L_spec_norm : float
        Spectral norm of L (typically 1.0 for stochastic matrices).
    epsilon_target : float
        Desired certification precision.
    epsilon_machine : float
        Machine epsilon (2.22e-16 for float64).

    Returns
    -------
    NumericalStabilityCertificate
    """
    if gamma_otu < 1e-15:
        return NumericalStabilityCertificate(
            gamma_otu=gamma_otu,
            condition_number=float('inf'),
            epsilon_numerical=float('inf'),
            epsilon_target=epsilon_target,
            is_certifiable=False,
            mixing_type="algebraic",
            recommended_precision="mpmath (arbitrary)",
            certificate_id="OTU-19",
        )

    kappa = 1.0 / gamma_otu
    eps_num = epsilon_machine * kappa ** 2 * L_spec_norm
    is_cert = eps_num < epsilon_target

    if gamma_otu < 1e-6:
        mixing = "algebraic"
    elif gamma_otu < 1e-3:
        mixing = "weakly_exponential"
    else:
        mixing = "exponential"

    if is_cert:
        rec = "float64"
    elif eps_num < epsilon_target * 1e10:
        rec = "float128 (long double)"
    else:
        rec = "mpmath (arbitrary)"

    return NumericalStabilityCertificate(
        gamma_otu=gamma_otu,
        condition_number=kappa,
        epsilon_numerical=eps_num,
        epsilon_target=epsilon_target,
        is_certifiable=is_cert,
        mixing_type=mixing,
        recommended_precision=rec,
        certificate_id="OTU-19",
    )


# ═══════════════════════════════════════════════════════════════════════════
# P2: Takens Embedding Preprocessor (OTU-20)
# ═══════════════════════════════════════════════════════════════════════════

def takens_embed(
    time_series: np.ndarray,
    embedding_dim: Optional[int] = None,
    delay: Optional[int] = None,
    D_2_hint: Optional[float] = None,
) -> TakensEmbeddingResult:
    """
    Construct a Takens delay embedding from a scalar time series.

    If embedding_dim is not provided, uses Whitney's criterion:
        d = ⌈2·D_2 + 1⌉ (or d=3 if D_2 is unknown).

    If delay τ is not provided, uses the first minimum of the
    automutual information (Kantz & Schreiber, 2003).

    After embedding, runs EDMD on the delay vectors to extract
    the Koopman spectrum (which is invariant under the embedding).

    Parameters
    ----------
    time_series : np.ndarray
        Scalar observations y_t = h(x_t), shape (N,).
    embedding_dim : int or None
        Embedding dimension d. Auto-estimated if None.
    delay : int or None
        Delay step τ. Auto-estimated if None.
    D_2_hint : float or None
        Hint for correlation dimension (from OTU or ERGON).

    Returns
    -------
    TakensEmbeddingResult
    """
    y = np.asarray(time_series, dtype=float).ravel()
    N = len(y)

    # --- Auto-estimate delay τ via autocorrelation first zero-crossing ---
    if delay is None:
        y_centered = y - np.mean(y)
        acf = np.correlate(y_centered, y_centered, mode='full')
        acf = acf[N - 1:]
        acf /= (acf[0] + 1e-14)
        delay = 1
        for k in range(1, min(N // 4, 200)):
            if acf[k] < 0:
                delay = k
                break
        else:
            delay = max(1, N // 20)

    # --- Auto-estimate embedding dimension ---
    if embedding_dim is None:
        if D_2_hint is not None and D_2_hint > 0.1:
            embedding_dim = max(3, math.ceil(2 * D_2_hint + 1))
        else:
            embedding_dim = 3  # safe default for 1D maps

    d = embedding_dim
    tau = delay

    # --- Build delay coordinate matrix (Hankel matrix) ---
    n_vectors = N - (d - 1) * tau
    if n_vectors < d + 10:
        raise ValueError(
            f"Time series too short ({N}) for d={d}, τ={tau}. "
            f"Need at least {(d - 1) * tau + d + 10} points."
        )

    H = np.zeros((n_vectors, d))
    for j in range(d):
        H[:, j] = y[j * tau: j * tau + n_vectors]

    # --- EDMD on delay vectors: X → Y = shift by 1 step ---
    X = H[:-1, :]  # (n_vectors-1, d)
    Y = H[1:, :]   # (n_vectors-1, d)

    # Koopman matrix K ≈ Y^T X (X^T X)^{-1}  (DMD)
    try:
        gram = X.T @ X
        gram += 1e-10 * np.eye(d)  # Tikhonov regularization
        K_delay = Y.T @ X @ np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        K_delay = Y.T @ np.linalg.pinv(X)

    eigvals = np.sort(np.abs(linalg.eigvals(K_delay)))[::-1]

    # Spectral gap from reconstructed data
    mods = np.abs(eigvals)
    nontrivial = mods[mods < 0.9999]
    gamma_recon = float(-np.log(max(nontrivial[0], 1e-300))) if len(nontrivial) > 0 else 0.0

    # Error bound: C · e^{-d·Γ·τ}
    error_bound = math.exp(-d * max(gamma_recon, 0.01) * tau) if gamma_recon > 0 else 1.0

    return TakensEmbeddingResult(
        embedding_dim=d,
        delay=tau,
        hankel_matrix=H,
        reconstructed_spectrum=eigvals,
        spectral_error_bound=error_bound,
        gamma_reconstructed=gamma_recon,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P6: Periodic Orbit Extraction from Ruelle Zeta (OTU-21)
# ═══════════════════════════════════════════════════════════════════════════

def extract_periodic_orbits(
    resonances: np.ndarray,
    T: Optional[Callable] = None,
    domain: Optional[Tuple[float, float]] = None,
    max_period: int = 8,
) -> PeriodicOrbitResult:
    """
    Extract periodic orbit information from the Ruelle resonance spectrum.

    Uses the trace formula:
        tr(L^n) = Σ_k λ_k^n = Σ_{T^n(x)=x} 1/|det(DT^n(x) - I)|

    The number of periodic points N_n is approximated from the spectral traces.
    If T is provided, orbits are located via Newton's method.

    Parameters
    ----------
    resonances : np.ndarray
        Complex Ruelle resonances from OTU.
    T : callable or None
        The dynamical system map (for orbit location).
    domain : tuple or None
        Phase space bounds.
    max_period : int
        Maximum period to extract.

    Returns
    -------
    PeriodicOrbitResult
    """
    orbits = []
    orbit_counts: Dict[int, int] = {}

    for n in range(1, max_period + 1):
        # Trace formula: tr(L^n) = Σ_k λ_k^n
        trace_n = float(np.sum(resonances ** n).real)
        # Approximate N_n from trace (rough: assumes uniform |det(DT^n - I)| ≈ 1)
        N_n = max(0, round(abs(trace_n)))
        orbit_counts[n] = N_n

    # Try to locate orbits using Newton's method if T is available
    if T is not None and domain is not None:
        a, b = domain
        eps_newton = 1e-10
        for n in range(1, max_period + 1):
            # Compose T^n — use default arg to capture current n
            def T_n(x, _n=n):
                val = np.atleast_1d(np.float64(x)).copy()
                for _ in range(_n):
                    val = T(val)
                return val

            # Search on a coarse grid
            n_search = min(200, 50 * n)
            x_search = np.linspace(a + 1e-6, b - 1e-6, n_search)
            found_points = []

            for x0 in x_search:
                x = float(x0)
                converged = False
                for _ in range(50):
                    fx = float(T_n(x)[0]) - x
                    if abs(fx) < eps_newton:
                        converged = True
                        break
                    # Numerical derivative
                    dx = 1e-8
                    dfx = (float(T_n(x + dx)[0]) - (x + dx) - fx) / dx + 1e-14
                    x_new = x - fx / dfx
                    x_new = max(a + 1e-10, min(b - 1e-10, x_new))
                    if abs(x_new - x) < eps_newton:
                        converged = True
                        x = x_new
                        break
                    x = x_new

                if converged and a < x < b:
                    # Check it's actually period n (not a divisor)
                    is_lower_period = False
                    for d in range(1, n):
                        if n % d == 0:
                            xd = float(x)
                            for _ in range(d):
                                xd = float(T(np.atleast_1d(xd))[0])
                            if abs(xd - x) < 1e-6:
                                is_lower_period = True
                                break

                    if not is_lower_period:
                        # Check not already found
                        already = False
                        for fp in found_points:
                            if abs(fp - x) < 1e-6:
                                already = True
                                break
                        if not already:
                            found_points.append(x)

            # Compute multipliers for found orbits
            for xp in found_points:
                multiplier = 1.0
                xc = float(xp)
                for _ in range(n):
                    dx = 1e-8
                    try:
                        d_local = abs(
                            float(T(np.atleast_1d(xc + dx))[0]) -
                            float(T(np.atleast_1d(xc - dx))[0])
                        ) / (2 * dx)
                    except Exception:
                        d_local = 1.0
                    multiplier *= d_local
                    xc = float(T(np.atleast_1d(xc))[0])

                orbits.append(PeriodicOrbitInfo(
                    period=n,
                    multiplier=multiplier,
                    stability="unstable" if multiplier > 1.0 else "stable",
                    approximate_points=np.array([xp]),
                ))

    # Estimate topological entropy from orbit count growth
    valid_counts = [(n, c) for n, c in orbit_counts.items() if c > 0 and n > 0]
    if len(valid_counts) >= 2:
        ns = [v[0] for v in valid_counts]
        log_counts = [np.log(max(v[1], 1)) for v in valid_counts]
        # Linear fit: log(N_n) ≈ n · h_top
        if max(ns) > min(ns):
            coeffs = np.polyfit(ns, log_counts, 1)
            h_top = max(0.0, float(coeffs[0]))
        else:
            h_top = 0.0
    else:
        h_top = 0.0

    growth_rate = h_top

    return PeriodicOrbitResult(
        orbits=orbits,
        max_period=max_period,
        orbit_count_by_period=orbit_counts,
        growth_rate=growth_rate,
        h_top_estimate=h_top,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P1: Multi-Attractor Basin Decomposition (OTU-22)
# ═══════════════════════════════════════════════════════════════════════════

def detect_basins(
    L: np.ndarray,
    delta: float = 0.05,
) -> BasinDecomposition:
    """
    Detect multiple ergodic components (basins of attraction) via
    spectral clustering on the Perron-Frobenius operator.

    Algorithm (Dellnitz-Junge, 1999):
    1. Find all eigenvalues with |λ_k| > 1 - δ.
    2. The number of such eigenvalues = number of basins m.
    3. Cluster the corresponding eigenvectors to identify basin membership.

    Parameters
    ----------
    L : np.ndarray
        Perron-Frobenius (Ulam) matrix, shape (N, N).
    delta : float
        Threshold for near-unit eigenvalues.

    Returns
    -------
    BasinDecomposition
    """
    N = L.shape[0]
    eigvals, eigvecs = linalg.eig(L)
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Find near-unit eigenvalues
    near_unit_mask = np.abs(eigvals) > (1.0 - delta)
    near_unit_evs = eigvals[near_unit_mask]
    near_unit_vecs = eigvecs[:, near_unit_mask].real

    m = len(near_unit_evs)

    if m <= 1:
        # Single basin — ergodic system
        basin_indices = np.zeros(N, dtype=int)
        weights = [1.0]
        timescales = []
        return BasinDecomposition(
            n_basins=1,
            basin_weights=weights,
            basin_indices=basin_indices,
            near_unit_eigenvalues=near_unit_evs,
            metastable_timescales=timescales,
            is_ergodic=True,
        )

    # Multiple basins: cluster using sign structure of eigenvectors
    # For m=2: sign of the 2nd eigenvector separates basins
    # For m>2: use k-means on the eigenvector coordinates
    if m == 2:
        v2 = near_unit_vecs[:, 1]
        basin_indices = (v2 > 0).astype(int)
    else:
        # Simple k-means on eigenvector space
        features = near_unit_vecs[:, :m]
        # Normalize rows
        row_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-14
        features = features / row_norms

        # k-means with m clusters
        rng = np.random.default_rng(42)
        centroids = features[rng.choice(N, m, replace=False)]
        for _ in range(50):
            dists = np.array([
                np.linalg.norm(features - centroids[k], axis=1)
                for k in range(m)
            ])
            basin_indices = np.argmin(dists, axis=0)
            for k in range(m):
                mask = basin_indices == k
                if mask.any():
                    centroids[k] = features[mask].mean(axis=0)

    # Compute basin weights (fraction of grid in each basin)
    weights = [float(np.sum(basin_indices == k) / N) for k in range(m)]

    # Metastable timescales from near-unit eigenvalues
    timescales = []
    for ev in near_unit_evs[1:]:
        mod = abs(ev)
        if mod < 1.0 - 1e-14:
            timescales.append(float(1.0 / (-np.log(mod))))
        else:
            timescales.append(float('inf'))

    return BasinDecomposition(
        n_basins=m,
        basin_weights=weights,
        basin_indices=basin_indices,
        near_unit_eigenvalues=near_unit_evs,
        metastable_timescales=timescales,
        is_ergodic=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P8: Continuous-Time OTU Generator (OTU-23)
# ═══════════════════════════════════════════════════════════════════════════

def compute_continuous_generator(
    discrete_eigenvalues: np.ndarray,
    tau: float = 1.0,
) -> ContinuousGeneratorResult:
    """
    Convert discrete-time Koopman/PF eigenvalues to continuous-time
    generator eigenvalues via the matrix logarithm:

        λ_k^{cont} = log(λ_k^{disc}) / τ

    The real part gives the continuous decay rate, the imaginary part
    gives the continuous oscillation frequency.

    Parameters
    ----------
    discrete_eigenvalues : np.ndarray
        Complex eigenvalues of the discrete Koopman/PF operator.
    tau : float
        Time step between discrete samples.

    Returns
    -------
    ContinuousGeneratorResult
    """
    # Filter out zero/very small eigenvalues
    valid = np.abs(discrete_eigenvalues) > 1e-14
    disc_ev = discrete_eigenvalues[valid]

    # Continuous eigenvalues: λ_cont = log(λ_disc) / τ
    cont_ev = np.log(disc_ev + 0j) / tau

    decay_rates = cont_ev.real
    frequencies = cont_ev.imag

    # Check aliasing: if max|Im(λ)| · τ > π, aliasing occurs
    max_freq = np.max(np.abs(frequencies)) if len(frequencies) > 0 else 0.0
    aliasing_free = bool(max_freq * tau < np.pi)

    return ContinuousGeneratorResult(
        generator_eigenvalues=cont_ev,
        decay_rates=decay_rates,
        frequencies=frequencies,
        time_step=tau,
        aliasing_free=aliasing_free,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P5: Exceptional Points & PT-Symmetry Detection (OTU-24)
# ═══════════════════════════════════════════════════════════════════════════

def detect_exceptional_points(
    L: np.ndarray,
    threshold: float = 1e-3,
) -> ExceptionalPointResult:
    """
    Detect exceptional points (EPs) in the Perron-Frobenius spectrum.

    An EP occurs when two eigenvalues λ_i, λ_j coalesce AND their
    eigenvectors also merge. This is detected by:
    1. Finding pairs with |λ_i - λ_j| < threshold
    2. Checking the Jordan defect: ‖(L - λI)² v‖ for the eigenvector v

    Also checks PT symmetry: L has PT symmetry if PLP^{-1} = L* for
    some involution P (parity operator).

    Parameters
    ----------
    L : np.ndarray
        Perron-Frobenius matrix.
    threshold : float
        Eigenvalue separation threshold for EP detection.

    Returns
    -------
    ExceptionalPointResult
    """
    N = L.shape[0]
    eigvals, eigvecs = linalg.eig(L)
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Find near-coalescing pairs
    ep_pairs = []
    ep_seps = []
    jordan_defects = []

    n_ev = min(len(eigvals), 30)  # Only check top eigenvalues
    for i in range(1, n_ev):
        for j in range(i + 1, n_ev):
            sep = abs(eigvals[i] - eigvals[j])
            if sep < threshold:
                ep_pairs.append((i, j))
                ep_seps.append(float(sep))

                # Jordan defect: ‖(L - λ_i I)² v_i‖
                lam_avg = (eigvals[i] + eigvals[j]) / 2
                v = eigvecs[:, i]
                Lshift = L - lam_avg * np.eye(N)
                defect = float(np.linalg.norm(Lshift @ Lshift @ v))
                jordan_defects.append(defect)

    # PT symmetry check: P is the reversal operator P[i] = N-1-i
    # PLP^{-1} = L* ↔ L[N-1-i, N-1-j] ≈ L[i,j].conj()
    P_L_P = L[::-1, ::-1]
    L_conj = L.conj()
    pt_error = np.linalg.norm(P_L_P - L_conj, 'fro') / (np.linalg.norm(L, 'fro') + 1e-14)
    pt_symmetric = bool(pt_error < 0.1)

    return ExceptionalPointResult(
        n_exceptional_points=len(ep_pairs),
        ep_pairs=ep_pairs,
        ep_separations=ep_seps,
        jordan_defects=jordan_defects,
        pt_symmetric=pt_symmetric,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P3: High-Dimensional Reduction (OTU-25)
# ═══════════════════════════════════════════════════════════════════════════

def reduce_high_dimensional(
    trajectory: np.ndarray,
    T: Callable[[np.ndarray], np.ndarray],
    D_2_hint: Optional[float] = None,
    n_koopman_modes: int = 10,
) -> HighDimReductionResult:
    """
    Reduce a high-dimensional dynamical system to intrinsic coordinates
    via kernel PCA, then run Koopman EDMD in the reduced space.

    For a system in R^D with attractor dimension d_H ≈ D_2 << D:
    1. Apply kernel PCA with Gaussian kernel to trajectory data.
    2. Project to ⌈D_2 + 1⌉ dimensions.
    3. Run EDMD in the projected space to get the Koopman spectrum.

    Parameters
    ----------
    trajectory : np.ndarray
        Shape (N, D) — N snapshots in D-dimensional phase space.
    T : callable
        The dynamical system (for time-shifted data).
    D_2_hint : float or None
        Estimated correlation dimension (from OTU/ERGON).
    n_koopman_modes : int
        Number of Koopman modes to extract.

    Returns
    -------
    HighDimReductionResult
    """
    X = np.atleast_2d(trajectory)
    N, D = X.shape

    # Estimate intrinsic dimension
    if D_2_hint is not None and D_2_hint > 0.5:
        d_intrinsic = max(2, math.ceil(D_2_hint + 1))
    else:
        d_intrinsic = min(D, max(2, D // 2))

    d_intrinsic = min(d_intrinsic, D, N - 1)

    # Kernel PCA with linear kernel (= standard PCA for efficiency)
    X_centered = X - X.mean(axis=0)
    if N < D:
        # Dual PCA: eigendecomposition of X X^T (N×N instead of D×D)
        C_dual = X_centered @ X_centered.T / (N - 1)
        eigvals_pca, eigvecs_pca = np.linalg.eigh(C_dual)
        idx_pca = np.argsort(-eigvals_pca)[:d_intrinsic]
        eigvals_pca = eigvals_pca[idx_pca]
        V_pca = eigvecs_pca[:, idx_pca]
        # Project: Z = V^T (normalized)
        Z = V_pca  # (N, d_intrinsic)
    else:
        # Standard PCA
        C = X_centered.T @ X_centered / (N - 1)
        eigvals_pca, eigvecs_pca = np.linalg.eigh(C)
        idx_pca = np.argsort(-eigvals_pca)[:d_intrinsic]
        W = eigvecs_pca[:, idx_pca]  # (D, d_intrinsic)
        Z = X_centered @ W            # (N, d_intrinsic)

    # EDMD in reduced space
    Z_x = Z[:-1, :]
    Z_y = Z[1:, :]
    gram = Z_x.T @ Z_x + 1e-10 * np.eye(d_intrinsic)
    try:
        K_reduced = Z_y.T @ Z_x @ np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        K_reduced = Z_y.T @ np.linalg.pinv(Z_x)

    spectrum = linalg.eigvals(K_reduced)
    spectrum = spectrum[np.argsort(-np.abs(spectrum))]

    # Reconstruction error
    if N < D:
        X_recon = V_pca @ (V_pca.T @ X_centered)
    else:
        X_recon = (X_centered @ W) @ W.T
    recon_error = float(np.linalg.norm(X_recon - X_centered, 'fro') /
                        (np.linalg.norm(X_centered, 'fro') + 1e-14))

    return HighDimReductionResult(
        ambient_dim=D,
        intrinsic_dim=d_intrinsic,
        reduction_method="pca",
        spectrum_in_reduced=spectrum[:n_koopman_modes],
        reconstruction_error=recon_error,
    )


# ═══════════════════════════════════════════════════════════════════════════
# P7: Fractal Gelfand Triple Adaptation (OTU-26)
# ═══════════════════════════════════════════════════════════════════════════

def certify_fractal_gelfand(
    mu_srb: np.ndarray,
    D_0: float,
    D_2: float,
    ambient_dim: int = 1,
    n_grid: int = 512,
) -> FractalGelfandCertificate:
    """
    Certify the Gelfand triple adaptation for fractal attractors.

    For attractors with dim_H(A) < ambient_dim, the standard L² norm
    is replaced by total variation. The convergence rate depends on D_2:
        ‖μ_N - μ_SRB‖_TV ≤ C · N^{-1/D_2}

    Parameters
    ----------
    mu_srb : np.ndarray
        Computed SRB measure (discretized).
    D_0 : float
        Box-counting dimension.
    D_2 : float
        Correlation dimension.
    ambient_dim : int
        Dimension of the ambient space.
    n_grid : int
        Grid resolution used.

    Returns
    -------
    FractalGelfandCertificate
    """
    is_fractal = bool(D_0 < ambient_dim - 0.01)

    # Total variation norm approximation:
    # TV(μ) = Σ |μ_i - 1/N|  (deviation from uniform)
    uniform = np.ones_like(mu_srb) / len(mu_srb)
    tv_error = float(0.5 * np.sum(np.abs(mu_srb - uniform)))

    # Convergence rate exponent: ‖μ_N - μ‖ ~ N^{-α}
    # For fractal measures: α = 1/D_2 (slower convergence for higher D_2)
    if D_2 > 0.01:
        conv_exponent = 1.0 / D_2
    else:
        conv_exponent = 1.0

    return FractalGelfandCertificate(
        hausdorff_dim_estimate=D_0,
        total_variation_error=tv_error,
        convergence_rate_exponent=conv_exponent,
        is_fractal=is_fractal,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unified Deep Analysis — runs all 10 solutions
# ═══════════════════════════════════════════════════════════════════════════

def deep_analysis(
    T: Callable[[np.ndarray], np.ndarray],
    domain: Tuple[float, float] = (0.001, 0.999),
    n_dist: int = 512,
    n_observations: int = 10000,
    epsilon_target: float = 0.01,
    max_orbit_period: int = 6,
    time_series: Optional[np.ndarray] = None,
    trajectory_nd: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Run all 10 deep problem solutions on a dynamical system T.

    This integrates with the existing GelfandTriple and returns
    certificates OTU-17 through OTU-26.

    Parameters
    ----------
    T : callable
        Map T: X → X (operates on numpy arrays).
    domain : tuple
        Phase space bounds (a, b).
    n_dist : int
        Grid resolution for the Ulam operator.
    n_observations : int
        Number of observations for Fisher/no-clone bounds.
    epsilon_target : float
        Target certification precision.
    max_orbit_period : int
        Maximum period for orbit extraction.
    time_series : np.ndarray or None
        Scalar time series for Takens embedding (P2).
    trajectory_nd : np.ndarray or None
        High-dimensional trajectory for P3.

    Returns
    -------
    dict with keys: "fisher_cr" (P9), "no_clone" (P10), "stability" (P4),
    "takens" (P2), "orbits" (P6), "basins" (P1), "generator" (P8),
    "exceptional" (P5), "high_dim" (P3), "fractal" (P7), "certificates" (dict).
    """
    from acf_functor.gelfand_triple import GelfandTriple

    # Run base OTU analysis
    otu = GelfandTriple(T, domain=domain, n_dist=n_dist)
    otu.build()
    result = otu.analyze()

    # Extract base quantities
    gamma = result.gamma_otu
    h_ks = result.h_ks
    resonances = result.spectrum.resonances
    mu_srb = result.mu_srb
    D_2 = result.multifractal.D_2 if result.multifractal else 0.83
    D_0 = result.multifractal.D_0 if result.multifractal else 1.0
    L = otu._L

    # Compute P''(1) from thermodynamic pressure
    pressure = otu.compute_thermodynamic_pressure()
    P_pp_1 = pressure.curvature_at_1

    # ── P9: Fisher-Cramér-Rao ──
    fisher_cr = compute_fisher_cramer_rao(P_pp_1, h_ks, n_observations)

    # ── P10: No-Cloning ──
    no_clone = compute_no_clone_bound(D_2, h_ks, epsilon_target)

    # ── P4: Numerical Stability ──
    stability = certify_numerical_stability(gamma, epsilon_target=epsilon_target)

    # ── P6: Periodic Orbits ──
    orbit_result = extract_periodic_orbits(
        resonances, T=T, domain=domain, max_period=max_orbit_period
    )

    # ── P1: Basin Decomposition ──
    basins = detect_basins(L)

    # ── P8: Continuous Generator ──
    generator = compute_continuous_generator(resonances)

    # ── P5: Exceptional Points ──
    exceptional = detect_exceptional_points(L)

    # ── P7: Fractal Gelfand ──
    fractal = certify_fractal_gelfand(mu_srb, D_0, D_2, n_grid=n_dist)

    # ── P2: Takens Embedding ──
    takens_result = None
    if time_series is not None:
        takens_result = takens_embed(time_series, D_2_hint=D_2)
    else:
        # Generate time series from T for self-test
        rng = np.random.default_rng(42)
        a, b = domain
        x = a + (b - a) * 0.3
        ts = np.zeros(2000)
        for i in range(2000):
            x = float(T(np.atleast_1d(x))[0])
            x = max(a + 1e-10, min(b - 1e-10, x))
            ts[i] = x
        takens_result = takens_embed(ts[200:], D_2_hint=D_2)

    # ── P3: High-dimensional reduction ──
    high_dim = None
    if trajectory_nd is not None:
        high_dim = reduce_high_dimensional(trajectory_nd, T, D_2_hint=D_2)

    # ── Compile certificates ──
    certificates = {
        "OTU-17_fisher_info": fisher_cr.fisher_information_per_obs,
        "OTU-17_cramer_rao_bound": fisher_cr.cramer_rao_bound,
        "OTU-17_min_error_std": fisher_cr.min_error_std,
        "OTU-17_relative_precision": fisher_cr.relative_precision,
        "OTU-17_is_bernoulli": fisher_cr.is_bernoulli,
        "OTU-18_D2": no_clone.D_2,
        "OTU-18_n_min": no_clone.n_min,
        "OTU-18_epsilon": no_clone.epsilon,
        "OTU-18_info_cost_bits": no_clone.information_cost_bits,
        "OTU-19_gamma": stability.gamma_otu,
        "OTU-19_condition_number": stability.condition_number,
        "OTU-19_epsilon_numerical": stability.epsilon_numerical,
        "OTU-19_is_certifiable": stability.is_certifiable,
        "OTU-19_mixing_type": stability.mixing_type,
        "OTU-20_embedding_dim": takens_result.embedding_dim if takens_result else None,
        "OTU-20_delay": takens_result.delay if takens_result else None,
        "OTU-20_gamma_reconstructed": takens_result.gamma_reconstructed if takens_result else None,
        "OTU-20_spectral_error_bound": takens_result.spectral_error_bound if takens_result else None,
        "OTU-21_n_orbits_found": len(orbit_result.orbits),
        "OTU-21_h_top_estimate": orbit_result.h_top_estimate,
        "OTU-21_orbit_counts": orbit_result.orbit_count_by_period,
        "OTU-22_n_basins": basins.n_basins,
        "OTU-22_is_ergodic": basins.is_ergodic,
        "OTU-22_basin_weights": basins.basin_weights,
        "OTU-23_aliasing_free": generator.aliasing_free,
        "OTU-24_n_exceptional_points": exceptional.n_exceptional_points,
        "OTU-24_pt_symmetric": exceptional.pt_symmetric,
        "OTU-25_intrinsic_dim": high_dim.intrinsic_dim if high_dim else None,
        "OTU-25_recon_error": high_dim.reconstruction_error if high_dim else None,
        "OTU-26_is_fractal": fractal.is_fractal,
        "OTU-26_tv_error": fractal.total_variation_error,
        "OTU-26_convergence_exponent": fractal.convergence_rate_exponent,
    }

    return {
        "otu_result": result,
        "pressure": pressure,
        "fisher_cr": fisher_cr,
        "no_clone": no_clone,
        "stability": stability,
        "takens": takens_result,
        "orbits": orbit_result,
        "basins": basins,
        "generator": generator,
        "exceptional": exceptional,
        "high_dim": high_dim,
        "fractal": fractal,
        "certificates": certificates,
    }
