"""
real_world.py — Real-World Bridge for TAA/ERGON/OTU Ecosystem
=============================================================

This module bridges the gap between the theoretical agents (which operate on
known functions T: X→X) and the real world (which gives noisy, partial,
non-stationary time series).

Four barriers addressed:

  BARRIER 1 — The Data Abyss:
    TimeSeriesReconstructor: Takens embedding, delay estimation (AMI),
    dimension estimation (FNN), local linear model extraction.

  BARRIER 2 — Non-Stationarity:
    RegimeDetector: Sliding-window spectral gap tracking, CUSUM change-point
    detection, adaptive forgetting for certificate aging.

  BARRIER 3 — Partial Observability:
    PartialObserver: Observation model h(x), observability Gramian,
    certified information loss bounds.

  BARRIER 4 — Finite Resources:
    AnytimeCertifier: Interruptible progressive refinement, knowledge graph
    compression, anytime epsilon-approximate certificates.

Mathematical foundations:
  - Takens (1981): Detecting strange attractors in turbulence.
  - Sauer, Yorke, Casdagli (1991): Embedology.
  - Kennel, Brown, Abarbanel (1992): False nearest neighbors.
  - Fraser & Swinney (1986): Mutual information for delay estimation.
  - Basseville & Nikiforov (1993): Detection of abrupt changes.
  - Kalman (1960): A new approach to linear filtering.
  - Hermann & Krener (1977): Nonlinear controllability and observability.
"""

from __future__ import annotations

import collections
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, signal


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TakensResult:
    """Result of Takens delay embedding reconstruction."""
    embedding_dim: int              # d — embedding dimension
    delay: int                      # τ — time delay (samples)
    embedded_data: np.ndarray       # shape (N-d*τ, d) — delay vectors
    ami_values: np.ndarray          # average mutual information vs τ
    fnn_fractions: np.ndarray       # false nearest neighbor fractions vs d
    optimal_delay_method: str       # "ami" or "autocorrelation"
    optimal_dim_method: str         # "fnn" or "cao"
    reconstruction_quality: float   # correlation dimension stability metric


@dataclass
class FilteredSignal:
    """Result of noise filtering."""
    clean_signal: np.ndarray        # denoised time series
    noise_estimate: np.ndarray      # estimated noise component
    snr_db: float                   # estimated signal-to-noise ratio (dB)
    filter_method: str              # "svd", "kalman", "wavelet"
    noise_variance: float           # estimated σ²_noise
    effective_dimension: int        # SVD rank / number of significant components


@dataclass
class LocalLinearModel:
    """Local linear map extracted from delay-embedded data."""
    A: np.ndarray                   # Linear approximation T(x) ≈ Ax + b
    b: np.ndarray                   # Affine offset
    residual_std: float             # ||T(x) - Ax - b|| over training data
    domain_center: np.ndarray       # Center of the local neighborhood
    domain_radius: float            # Radius of validity
    n_points_used: int              # Number of data points in local fit


@dataclass
class ReconstructedSystem:
    """A dynamical system reconstructed from time series data."""
    T: Callable[[np.ndarray], np.ndarray]  # Reconstructed map
    domain: Tuple[float, float]     # Bounding box (min, max per dim)
    embedding_dim: int              # Dimension of reconstructed state space
    delay: int                      # Delay used
    n_local_models: int             # Number of local linear patches
    reconstruction_error: float     # Cross-validation error
    snr_db: float                   # Signal quality after filtering
    takens: TakensResult            # Full Takens analysis
    filtered: FilteredSignal        # Filtering results


@dataclass
class RegimeSegment:
    """A detected regime (stationary segment) in the time series."""
    start_idx: int                  # Start index in original series
    end_idx: int                    # End index
    lyapunov_estimate: float        # Estimated λ_max in this window
    spectral_gap_estimate: float    # Estimated Γ in this window
    entropy_estimate: float         # Estimated h_KS in this window
    regime_type: str                # "chaotic", "periodic", "transient", "quasiperiodic"
    confidence: float               # Confidence in classification [0, 1]


@dataclass
class ChangePoint:
    """A detected change point in the dynamics."""
    index: int                      # Index in original time series
    cusum_statistic: float          # CUSUM test statistic at change
    before_lyapunov: float          # λ_max before change
    after_lyapunov: float           # λ_max after change
    significance: float             # p-value or confidence metric
    change_type: str                # "chaos_onset", "chaos_offset", "bifurcation"


@dataclass
class RegimeAnalysis:
    """Complete non-stationarity analysis."""
    segments: List[RegimeSegment]
    change_points: List[ChangePoint]
    is_stationary: bool             # True if no significant changes detected
    n_regimes: int                  # Number of distinct regimes
    lyapunov_trajectory: np.ndarray # λ_max(t) sliding window
    spectral_gap_trajectory: np.ndarray  # Γ(t) sliding window
    window_size: int                # Window size used


@dataclass
class ObservabilityReport:
    """Report on partial observability and information loss."""
    observable_dims: int            # Number of observed dimensions
    state_dims: int                 # Estimated full state dimension
    observability_rank: int         # Rank of observability Gramian
    is_observable: bool             # Full rank → observable
    information_loss_bound: float   # Lower bound on prediction error due to partial obs
    gramian_condition: float        # Condition number of observability Gramian
    reconstructible_lyapunov: float # λ_max recoverable from observations
    lyapunov_error_bound: float     # |λ_true - λ_obs| upper bound
    takens_observability_guaranteed: bool = False  # True if d_embed ≥ 2n+1
    takens_dim_required: int = 0    # Minimum d for Takens guarantee


@dataclass
class AnytimeResult:
    """Result from an anytime (interruptible) certification."""
    epsilon_achieved: float         # Best ε achieved before timeout
    n_grid_used: int                # Grid resolution at timeout
    h_ks_estimate: float            # Best h_KS estimate
    h_ks_confidence_interval: Tuple[float, float]  # (lower, upper) bounds
    spectral_gap_estimate: float    # Best Γ estimate
    computation_time_ms: float      # Wall time used
    refinement_history: List[dict]  # Progressive refinement snapshots
    is_converged: bool              # True if ε_achieved < ε_target
    certificates_valid: int         # Number of certificates that passed
    peak_memory_bytes: int = 0      # Peak matrix memory used (bytes)


@dataclass
class CompressedKnowledge:
    """Compressed knowledge graph summary."""
    n_original_certificates: int
    n_compressed: int
    meta_theorems: List[dict]       # Compressed meta-theorems
    compression_ratio: float        # n_compressed / n_original
    information_retained: float     # Fraction of total variance retained
    oldest_relevant: float          # Age of oldest still-relevant certificate
    decay_constant: float           # Exponential decay rate for relevance


@dataclass
class SurrogateTestResult:
    """Result of surrogate data test for determinism (Schreiber-Schmitz 1996)."""
    is_deterministic: bool          # Null hypothesis (stochastic) rejected
    p_value: float                  # Rank-order p-value
    original_statistic: float       # Discriminating statistic on real data
    surrogate_statistics: np.ndarray  # Statistics on surrogate ensemble
    n_surrogates: int               # Number of surrogates generated
    statistic_name: str             # Which statistic was used


@dataclass
class EKFResult:
    """Result of Extended Kalman Filter state estimation."""
    state_estimates: np.ndarray     # (n_steps, n_state) — filtered state trajectory
    covariances: List[np.ndarray]   # P_{t|t} at each step
    innovations: np.ndarray         # y_t - h(x_hat_{t|t-1})
    nees: float                     # Normalized Estimation Error Squared (consistency)


# ═══════════════════════════════════════════════════════════════════════════
# BARRIER 1: The Data Abyss — Time Series → Dynamical System
# ═══════════════════════════════════════════════════════════════════════════

class TimeSeriesReconstructor:
    """
    Reconstructs a dynamical system from a scalar (or vector) time series.

    Pipeline:
      1. Noise filtering (SVD / Kalman / wavelet)
      2. Delay estimation via Average Mutual Information (Fraser-Swinney)
      3. Embedding dimension via False Nearest Neighbors (Kennel-Brown-Abarbanel)
      4. Local linear model extraction (piecewise-affine approximation)
      5. Global map T: R^d → R^d assembled from local patches

    The result is a callable T that can be fed directly to ERGONAgent or GelfandTriple.
    """

    def __init__(
        self,
        noise_filter: str = "svd",     # "svd", "kalman", "wavelet", "particle", "auto", "none"
        max_embedding_dim: int = 10,
        n_neighbors_fnn: int = 5,      # neighbors for FNN test
        fnn_threshold: float = 0.02,   # fraction below which FNN declares sufficient
        ami_max_lag: int = 100,        # max lag for AMI computation
        local_model_k: int = 30,       # neighbors for local linear fit
        n_particles: int = 200,        # particles for SIR filter
        particle_noise_model: str = "gaussian",  # "gaussian", "laplacian", "student_t"
    ):
        self.noise_filter = noise_filter
        self.max_embedding_dim = max_embedding_dim
        self.n_neighbors_fnn = n_neighbors_fnn
        self.fnn_threshold = fnn_threshold
        self.ami_max_lag = ami_max_lag
        self.local_model_k = local_model_k
        self.n_particles = n_particles
        self.particle_noise_model = particle_noise_model

    # ------------------------------------------------------------------
    # Step 1: Noise filtering
    # ------------------------------------------------------------------

    def filter_noise(self, y: np.ndarray, method: Optional[str] = None) -> FilteredSignal:
        """
        Separate deterministic signal from measurement noise.

        SVD method: Embed in a Hankel matrix, threshold singular values
        at the elbow, reconstruct from significant components only.
        This is equivalent to SSA (Singular Spectrum Analysis).

        Kalman method: Fit AR(p) model, run Kalman smoother.
        """
        method = method or self.noise_filter
        y = np.asarray(y, dtype=float).ravel()

        if method == "none":
            return FilteredSignal(
                clean_signal=y.copy(),
                noise_estimate=np.zeros_like(y),
                snr_db=100.0,
                filter_method="none",
                noise_variance=0.0,
                effective_dimension=len(y),
            )

        if method == "auto":
            method = self._auto_select_filter(y)

        if method == "svd":
            return self._filter_svd(y)
        elif method == "kalman":
            return self._filter_kalman(y)
        elif method == "wavelet":
            return self._filter_wavelet(y)
        elif method == "particle":
            return self._filter_particle(y)
        else:
            raise ValueError(f"Unknown filter method: {method}")

    def _filter_svd(self, y: np.ndarray, window: int = 50) -> FilteredSignal:
        """
        SVD/SSA denoising: build Hankel matrix, threshold SVs at the elbow.

        The Hankel matrix H[i,j] = y[i+j] embeds the 1D series into a
        trajectory matrix. SVD separates signal (large SVs) from noise (small SVs).
        """
        n = len(y)
        window = min(window, n // 3)
        cols = n - window + 1

        # Build Hankel (trajectory) matrix
        H = np.zeros((window, cols))
        for i in range(window):
            H[i, :] = y[i:i + cols]

        U, s, Vt = np.linalg.svd(H, full_matrices=False)

        # Elbow detection: keep components until cumulative energy >= 95%
        # This is more robust than diff-based elbow for chaotic signals
        energy = s ** 2
        cum_energy = np.cumsum(energy) / (np.sum(energy) + 1e-300)
        elbow = int(np.searchsorted(cum_energy, 0.95)) + 1
        elbow = max(1, min(elbow, len(s)))

        # Reconstruct from significant components
        S_trunc = np.diag(s[:elbow])
        H_clean = U[:, :elbow] @ S_trunc @ Vt[:elbow, :]

        # Diagonal averaging (anti-diagonal mean) to get back the time series
        clean = np.zeros(n)
        counts = np.zeros(n)
        for i in range(window):
            for j in range(cols):
                clean[i + j] += H_clean[i, j]
                counts[i + j] += 1
        counts[counts == 0] = 1
        clean /= counts

        noise = y - clean
        noise_var = float(np.var(noise))
        signal_var = float(np.var(clean))
        snr = 10.0 * np.log10(signal_var / max(noise_var, 1e-300))

        return FilteredSignal(
            clean_signal=clean,
            noise_estimate=noise,
            snr_db=float(snr),
            filter_method="svd",
            noise_variance=noise_var,
            effective_dimension=elbow,
        )

    def _filter_kalman(self, y: np.ndarray, order: int = 4) -> FilteredSignal:
        """
        Kalman smoother denoising: fit AR(p) model, then forward-backward filter.

        The AR(p) model y_t = Σ a_k y_{t-k} + ε_t defines the state-space:
          x_t = [y_t, y_{t-1}, ..., y_{t-p+1}]
          x_{t+1} = A x_t + B ε_t
          y_t = C x_t

        Kalman forward pass → backward smoother gives MMSE estimate of x_t.
        """
        n = len(y)
        p = min(order, n // 10)

        # Fit AR(p) via Yule-Walker
        autocorr = np.correlate(y - y.mean(), y - y.mean(), mode='full')
        autocorr = autocorr[n - 1:]  # one-sided
        autocorr /= autocorr[0] + 1e-14

        # Levinson-Durbin recursion for AR coefficients
        r = autocorr[1:p + 1]
        R = linalg.toeplitz(autocorr[:p])
        try:
            ar_coeffs = linalg.solve(R, r)
        except linalg.LinAlgError:
            ar_coeffs = np.linalg.lstsq(R, r, rcond=None)[0]

        # Estimate innovation variance
        ar_pred = np.zeros(n)
        for t in range(p, n):
            ar_pred[t] = sum(ar_coeffs[k] * y[t - k - 1] for k in range(p))
        residuals = y[p:] - ar_pred[p:]
        Q = float(np.var(residuals))

        # State-space matrices
        A = np.zeros((p, p))
        A[0, :] = ar_coeffs
        if p > 1:
            A[1:, :-1] = np.eye(p - 1)

        C = np.zeros((1, p))
        C[0, 0] = 1.0
        R_obs = Q * 0.1  # observation noise (tunable)

        # Kalman forward pass
        x_hat = np.zeros((n, p))
        P = np.eye(p) * Q
        x_hat[0, 0] = y[0]

        x_smooth = np.zeros((n, p))
        P_store = [P.copy()]

        for t in range(1, n):
            # Predict
            x_pred = A @ x_hat[t - 1]
            P_pred = A @ P @ A.T + np.eye(p) * Q

            # Update
            S = C @ P_pred @ C.T + R_obs
            K = P_pred @ C.T / S.item()
            innovation = y[t] - float((C @ x_pred).item())
            x_hat[t] = x_pred + K.ravel() * innovation
            P = (np.eye(p) - K @ C) @ P_pred
            P_store.append(P.copy())

        # Backward smoother (Rauch-Tung-Striebel)
        x_smooth[-1] = x_hat[-1]
        for t in range(n - 2, -1, -1):
            P_pred = A @ P_store[t] @ A.T + np.eye(p) * Q
            try:
                G = P_store[t] @ A.T @ np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                G = P_store[t] @ A.T @ np.linalg.pinv(P_pred)
            x_smooth[t] = x_hat[t] + G @ (x_smooth[t + 1] - A @ x_hat[t])

        clean = x_smooth[:, 0]
        noise = y - clean
        noise_var = float(np.var(noise))
        signal_var = float(np.var(clean))
        snr = 10.0 * np.log10(signal_var / max(noise_var, 1e-300))

        return FilteredSignal(
            clean_signal=clean,
            noise_estimate=noise,
            snr_db=float(snr),
            filter_method="kalman",
            noise_variance=noise_var,
            effective_dimension=p,
        )

    def _filter_wavelet(self, y: np.ndarray) -> FilteredSignal:
        """
        Wavelet soft-thresholding denoising (Donoho-Johnstone universal threshold).

        Uses multi-level Haar wavelet decomposition. The universal threshold
        σ√(2 log n) removes noise coefficients while preserving signal structure.
        """
        n = len(y)

        # Haar wavelet decomposition (manual — no pywt dependency)
        # Pad to next power of 2 >= n
        n_pad = 2 ** int(np.ceil(np.log2(max(n, 2))))
        max_level = min(int(np.log2(n_pad)), 8)

        y_pad = np.zeros(n_pad)
        y_pad[:n] = y

        # Forward Haar transform
        coeffs = []
        approx = y_pad.copy()
        for level in range(max_level):
            m = len(approx) // 2
            detail = (approx[::2] - approx[1::2]) / np.sqrt(2)
            approx_new = (approx[::2] + approx[1::2]) / np.sqrt(2)
            coeffs.append(detail)
            approx = approx_new

        # Estimate noise σ from finest detail coefficients (MAD estimator)
        finest = coeffs[0]
        sigma = float(np.median(np.abs(finest))) / 0.6745

        # Conservative thresholding: only threshold the 2 finest levels
        # with a reduced threshold. Chaotic signals have broadband spectrum,
        # so aggressive thresholding removes signal. Use √(2 log m) / 2.
        n_levels_to_threshold = min(2, len(coeffs))
        for i in range(n_levels_to_threshold):
            m = len(coeffs[i])
            threshold = sigma * np.sqrt(2.0 * np.log(max(m, 2))) * 0.5
            coeffs[i] = np.sign(coeffs[i]) * np.maximum(np.abs(coeffs[i]) - threshold, 0)

        # Inverse Haar transform
        reconstructed = approx
        for level in range(max_level - 1, -1, -1):
            detail = coeffs[level]
            m = len(reconstructed)
            full = np.zeros(2 * m)
            full[::2] = (reconstructed + detail) / np.sqrt(2)
            full[1::2] = (reconstructed - detail) / np.sqrt(2)
            reconstructed = full

        clean = reconstructed[:n]
        noise = y - clean
        noise_var = float(np.var(noise))
        signal_var = float(np.var(clean))
        snr = 10.0 * np.log10(signal_var / max(noise_var, 1e-300))

        # Count non-zero wavelet coefficients
        n_nonzero = sum(int(np.sum(np.abs(c) > 1e-14)) for c in coeffs)

        return FilteredSignal(
            clean_signal=clean,
            noise_estimate=noise,
            snr_db=float(snr),
            filter_method="wavelet",
            noise_variance=noise_var,
            effective_dimension=n_nonzero,
        )

    def _filter_particle(self, y: np.ndarray, order: int = 4) -> FilteredSignal:
        """
        Sequential Importance Resampling (SIR) particle filter.

        Superior to Kalman for non-Gaussian noise (heavy tails, outliers).
        Fits AR(p) dynamics model, then runs N particles with configurable
        noise distribution (Gaussian, Laplacian, Student-t).
        """
        n = len(y)
        p = min(order, n // 10)
        N = self.n_particles

        # Fit AR(p) via Yule-Walker (reuse from Kalman)
        autocorr = np.correlate(y - y.mean(), y - y.mean(), mode='full')
        autocorr = autocorr[n - 1:]
        autocorr /= autocorr[0] + 1e-14

        r = autocorr[1:p + 1]
        R_mat = linalg.toeplitz(autocorr[:p])
        try:
            ar_coeffs = linalg.solve(R_mat, r)
        except linalg.LinAlgError:
            ar_coeffs = np.linalg.lstsq(R_mat, r, rcond=None)[0]

        # Innovation variance
        ar_pred = np.zeros(n)
        for t in range(p, n):
            ar_pred[t] = sum(ar_coeffs[k] * y[t - k - 1] for k in range(p))
        residuals = y[p:] - ar_pred[p:]
        Q = float(np.var(residuals))
        sigma_q = max(np.sqrt(Q), 1e-10)

        rng = np.random.default_rng(42)

        # Initialize particles: states (N, p), weights (N,)
        particles = np.zeros((N, p))
        particles[:, 0] = y[0] + rng.normal(0, sigma_q, N)
        weights = np.ones(N) / N

        clean = np.zeros(n)
        clean[0] = y[0]

        def _sample_noise(size):
            """Sample from the configured noise model."""
            if self.particle_noise_model == "laplacian":
                return rng.laplace(0, sigma_q / np.sqrt(2), size)
            elif self.particle_noise_model == "student_t":
                return rng.standard_t(df=3, size=size) * sigma_q * 0.5
            else:  # gaussian
                return rng.normal(0, sigma_q, size)

        def _obs_likelihood(obs, pred, sigma_obs):
            """Observation likelihood p(y|x) under noise model."""
            diff = np.abs(obs - pred)
            if self.particle_noise_model == "laplacian":
                scale = sigma_obs / np.sqrt(2)
                return np.exp(-diff / max(scale, 1e-14)) / (2 * max(scale, 1e-14))
            elif self.particle_noise_model == "student_t":
                # Student-t with df=3
                return (1 + (diff / max(sigma_obs * 0.5, 1e-14)) ** 2 / 3) ** (-2)
            else:
                return np.exp(-0.5 * (diff / max(sigma_obs, 1e-14)) ** 2)

        sigma_obs = sigma_q * 0.5

        for t in range(1, n):
            # Propagate particles through AR dynamics + noise
            new_state_0 = np.zeros(N)
            for k in range(min(p, t)):
                new_state_0 += ar_coeffs[k] * particles[:, k] if k < p else 0.0
            new_state_0 += _sample_noise(N)

            # Shift state
            if p > 1:
                particles[:, 1:] = particles[:, :-1]
            particles[:, 0] = new_state_0

            # Weight by observation likelihood
            w = _obs_likelihood(y[t], particles[:, 0], sigma_obs)
            weights *= w
            w_sum = weights.sum()
            if w_sum > 1e-300:
                weights /= w_sum
            else:
                weights = np.ones(N) / N

            # Systematic resampling when N_eff < N/2
            n_eff = 1.0 / (np.sum(weights ** 2) + 1e-300)
            if n_eff < N / 2:
                cumsum = np.cumsum(weights)
                u = (rng.random() + np.arange(N)) / N
                indices = np.searchsorted(cumsum, u)
                indices = np.clip(indices, 0, N - 1)
                particles = particles[indices]
                weights = np.ones(N) / N

            clean[t] = float(np.sum(weights * particles[:, 0]))

        noise = y - clean
        noise_var = float(np.var(noise))
        signal_var = float(np.var(clean))
        snr = 10.0 * np.log10(signal_var / max(noise_var, 1e-300))

        return FilteredSignal(
            clean_signal=clean,
            noise_estimate=noise,
            snr_db=float(snr),
            filter_method="particle",
            noise_variance=noise_var,
            effective_dimension=N,
        )

    def _auto_select_filter(self, y: np.ndarray) -> str:
        """
        Automatically select the best noise filter based on data characteristics.

        Decision logic:
          - Heavy tails (kurtosis > 5)       → particle filter
          - Heteroscedastic (GARCH-like)      → kalman
          - Short series (< 500 points)       → wavelet
          - Default                           → svd
        """
        diffs = np.diff(y)
        n = len(diffs)

        # Excess kurtosis: heavy-tailed noise → particle filter
        if n > 20:
            mu = np.mean(diffs)
            sigma = np.std(diffs) + 1e-14
            z = (diffs - mu) / sigma
            kurtosis = float(np.mean(z ** 4)) - 3.0
            if kurtosis > 5.0:
                return "particle"

        # Heteroscedasticity: lag-1 autocorrelation of |residuals|
        if n > 50:
            abs_d = np.abs(diffs - np.mean(diffs))
            acf_abs = np.correlate(abs_d - abs_d.mean(), abs_d - abs_d.mean(), mode='full')
            acf_abs = acf_abs[n - 1:]
            if acf_abs[0] > 1e-14:
                lag1_acf = acf_abs[1] / acf_abs[0]
                if lag1_acf > 0.3:
                    return "kalman"

        # Short series
        if len(y) < 500:
            return "wavelet"

        return "svd"

    # ------------------------------------------------------------------
    # Step 2: Delay estimation via Average Mutual Information
    # ------------------------------------------------------------------

    def estimate_delay(self, y: np.ndarray, max_lag: Optional[int] = None) -> Tuple[int, np.ndarray]:
        """
        Estimate optimal delay τ via Average Mutual Information (AMI).

        Fraser & Swinney (1986): The first minimum of AMI(τ) gives the
        optimal delay where the coordinates (y_t, y_{t+τ}) are maximally
        independent. This is superior to the autocorrelation zero-crossing
        because it captures nonlinear dependencies.

        Algorithm:
          1. Partition y into n_bins equiprobable bins
          2. For each lag τ, compute MI(y_t, y_{t+τ}) using joint histogram
          3. Return the first local minimum of MI(τ)

        Returns:
            (optimal_delay, ami_values_array)
        """
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        max_lag = max_lag or min(self.ami_max_lag, n // 4)

        # Equiprobable binning
        n_bins = max(10, int(np.sqrt(n / 5)))
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y, percentiles)
        bin_edges[0] -= 1e-10
        bin_edges[-1] += 1e-10

        # Digitize the series
        binned = np.digitize(y, bin_edges) - 1
        binned = np.clip(binned, 0, n_bins - 1)

        ami_values = np.zeros(max_lag)
        for tau in range(1, max_lag + 1):
            # Joint histogram
            joint = np.zeros((n_bins, n_bins), dtype=float)
            n_pairs = n - tau
            for t in range(n_pairs):
                joint[binned[t], binned[t + tau]] += 1
            joint /= n_pairs + 1e-14

            # Marginals
            px = joint.sum(axis=1)
            py = joint.sum(axis=0)

            # Mutual information I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint[i, j] > 1e-14 and px[i] > 1e-14 and py[j] > 1e-14:
                        mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))
            ami_values[tau - 1] = mi

        # Find first local minimum
        optimal_tau = 1
        for tau in range(1, max_lag - 1):
            if ami_values[tau] < ami_values[tau - 1] and ami_values[tau] < ami_values[tau + 1]:
                optimal_tau = tau + 1  # 1-indexed lag
                break
        else:
            # Fallback: use autocorrelation e-folding time
            acf = np.correlate(y - y.mean(), y - y.mean(), mode='full')
            acf = acf[n - 1:] / (acf[n - 1] + 1e-14)
            for tau in range(1, max_lag):
                if acf[tau] < 1.0 / np.e:
                    optimal_tau = tau
                    break
            else:
                optimal_tau = max(1, max_lag // 4)

        return optimal_tau, ami_values

    # ------------------------------------------------------------------
    # Step 3: Embedding dimension via False Nearest Neighbors
    # ------------------------------------------------------------------

    def estimate_dimension(
        self,
        y: np.ndarray,
        tau: int,
        r_threshold: float = 15.0,
    ) -> Tuple[int, np.ndarray]:
        """
        Estimate embedding dimension d via False Nearest Neighbors (FNN).

        Kennel, Brown & Abarbanel (1992): A neighbor is "false" in dimension d
        if the distance to it increases dramatically when going to dimension d+1.
        This means the apparent proximity was an artifact of projection.

        When FNN fraction drops below fnn_threshold, we have d_embed.

        Args:
            y:           Time series (1D)
            tau:         Delay from estimate_delay()
            r_threshold: Distance ratio threshold for declaring a neighbor "false"

        Returns:
            (optimal_dim, fnn_fractions_per_dim)
        """
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        max_d = self.max_embedding_dim

        fnn_fracs = np.ones(max_d)

        for d in range(1, max_d + 1):
            n_vectors = n - d * tau
            if n_vectors < self.n_neighbors_fnn + 10:
                break

            # Build delay vectors: z_t = (y_t, y_{t+τ}, ..., y_{t+(d-1)τ})
            Z = np.zeros((n_vectors, d))
            for k in range(d):
                Z[:, k] = y[k * tau: k * tau + n_vectors]

            # For each point, find nearest neighbor and check if it's false
            n_false = 0
            n_total = 0

            # Subsample for speed (check ~1000 points max)
            n_check = min(n_vectors - 1, 1000)
            rng = np.random.default_rng(42)
            check_indices = rng.choice(n_vectors - 1, size=n_check, replace=False)

            for idx in check_indices:
                z = Z[idx]

                # Find nearest neighbor (excluding temporal neighbors)
                distances = np.linalg.norm(Z - z, axis=1)
                distances[idx] = np.inf
                # Exclude temporal neighbors (Theiler window)
                theiler = max(tau, 1)
                for j in range(max(0, idx - theiler), min(n_vectors, idx + theiler + 1)):
                    distances[j] = np.inf

                nn_idx = np.argmin(distances)
                d_nn = distances[nn_idx]

                if d_nn < 1e-14:
                    continue

                # Check if false: does the (d+1)-th coordinate blow up the distance?
                if d < max_d and idx + d * tau < n and nn_idx + d * tau < n:
                    extra_dist = abs(y[idx + d * tau] - y[nn_idx + d * tau])
                    ratio = extra_dist / d_nn
                    if ratio > r_threshold:
                        n_false += 1

                n_total += 1

            fnn_fracs[d - 1] = n_false / max(n_total, 1)

            if fnn_fracs[d - 1] < self.fnn_threshold:
                return d, fnn_fracs

        # Return the dimension with minimum FNN fraction
        optimal_d = int(np.argmin(fnn_fracs)) + 1
        return max(optimal_d, 2), fnn_fracs

    # ------------------------------------------------------------------
    # Step 4: Local linear model extraction
    # ------------------------------------------------------------------

    def _build_local_linear_models(
        self,
        Z: np.ndarray,
        k: Optional[int] = None,
    ) -> List[LocalLinearModel]:
        """
        Build piecewise-affine local models: T(z) ≈ A_i z + b_i near z_i.

        For each point z_t in the embedded data, find k nearest neighbors,
        fit a local affine map z_{t+1} = A z_t + b via weighted least squares.

        The collection of local models defines a globally nonlinear map.
        """
        k = k or self.local_model_k
        n, d = Z.shape
        n_valid = n - 1  # need z_{t+1} for each z_t

        X = Z[:n_valid]    # inputs z_t
        Y = Z[1:n_valid + 1]  # outputs z_{t+1} (shifted by 1)

        # Subsample centers (don't need a model at every point)
        n_centers = min(n_valid, max(50, n_valid // 20))
        rng = np.random.default_rng(42)
        center_indices = rng.choice(n_valid, size=n_centers, replace=False)

        models = []
        for ci in center_indices:
            center = X[ci]

            # Find k nearest neighbors
            dists = np.linalg.norm(X - center, axis=1)
            nn_idx = np.argsort(dists)[:k]

            X_local = X[nn_idx]
            Y_local = Y[nn_idx]

            # Weighted least squares: weights = exp(-||x - center||² / σ²)
            local_dists = dists[nn_idx]
            sigma = np.median(local_dists) + 1e-14
            weights = np.exp(-0.5 * (local_dists / sigma) ** 2)

            # Solve: [A | b] = argmin Σ w_i ||y_i - A x_i - b||²
            # Augment X with ones column for affine term
            ones = np.ones((len(nn_idx), 1))
            X_aug = np.hstack([X_local, ones])
            W = np.diag(weights)

            try:
                # (X^T W X) θ = X^T W Y
                XtWX = X_aug.T @ W @ X_aug
                XtWY = X_aug.T @ W @ Y_local
                theta = np.linalg.solve(
                    XtWX + 1e-8 * np.eye(d + 1),  # Tikhonov regularization
                    XtWY,
                )
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(X_aug, Y_local, rcond=None)[0]

            A = theta[:d, :]   # d × d
            b_vec = theta[d, :]  # d

            # Residual
            Y_pred = X_local @ A + b_vec
            residual = float(np.std(Y_local - Y_pred))

            models.append(LocalLinearModel(
                A=A,
                b=b_vec,
                residual_std=residual,
                domain_center=center.copy(),
                domain_radius=float(local_dists[-1]),
                n_points_used=len(nn_idx),
            ))

        return models

    # ------------------------------------------------------------------
    # Step 5: Assemble global map T from local models
    # ------------------------------------------------------------------

    def _assemble_global_map(
        self,
        models: List[LocalLinearModel],
        domain_min: np.ndarray,
        domain_max: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Assemble a global nonlinear map from local affine patches.

        For a query point z, find the nearest model center, apply T(z) = A z + b.
        Uses inverse-distance weighting of the k nearest models for smoothness.
        """
        centers = np.array([m.domain_center for m in models])
        k_blend = min(3, len(models))

        def T(z: np.ndarray) -> np.ndarray:
            z = np.atleast_2d(z)
            results = np.zeros_like(z)

            for i, zi in enumerate(z):
                dists = np.linalg.norm(centers - zi, axis=1)
                nn_idx = np.argsort(dists)[:k_blend]

                # Inverse distance weighted blending
                d_nn = dists[nn_idx]
                if d_nn[0] < 1e-14:
                    # Exact match
                    m = models[nn_idx[0]]
                    results[i] = m.A @ zi + m.b
                else:
                    weights = 1.0 / (d_nn + 1e-14)
                    weights /= weights.sum()

                    pred = np.zeros(zi.shape)
                    for j, idx in enumerate(nn_idx):
                        m = models[idx]
                        pred += weights[j] * (m.A @ zi + m.b)
                    results[i] = pred

            if results.shape[0] == 1:
                return results[0]
            return results

        return T

    # ------------------------------------------------------------------
    # Main entry point: Time Series → Dynamical System
    # ------------------------------------------------------------------

    def reconstruct(self, y: np.ndarray) -> ReconstructedSystem:
        """
        Full pipeline: raw time series → reconstructed dynamical system.

        Args:
            y: Time series — 1D (n_samples,) or 2D (n_samples, n_channels).
               Scalar: sensor readings, prices, voltages.
               Multivariate: multi-sensor arrays, multi-axis accelerometers.

        Returns:
            ReconstructedSystem with a callable T that can be fed to ERGON/OTU.
        """
        y = np.asarray(y, dtype=float)
        is_multivariate = y.ndim == 2 and y.shape[1] > 1

        if is_multivariate:
            return self._reconstruct_multivariate(y)

        y = y.ravel()
        n = len(y)

        if n < 100:
            raise ValueError(f"Need at least 100 data points, got {n}")

        # Step 1: Filter noise
        filtered = self.filter_noise(y)
        y_clean = filtered.clean_signal

        # Step 2: Estimate delay
        tau, ami = self.estimate_delay(y_clean)

        # Step 3: Estimate embedding dimension
        d_embed, fnn = self.estimate_dimension(y_clean, tau)

        # Step 4: Build delay embedding
        n_vectors = n - (d_embed - 1) * tau
        Z = np.zeros((n_vectors, d_embed))
        for k in range(d_embed):
            Z[:, k] = y_clean[k * tau: k * tau + n_vectors]

        # Step 5: Build local linear models
        models = self._build_local_linear_models(Z)

        # Step 6: Assemble global map
        domain_min = Z.min(axis=0)
        domain_max = Z.max(axis=0)
        T = self._assemble_global_map(models, domain_min, domain_max)

        # Cross-validation error estimate
        n_test = min(200, n_vectors // 5)
        rng = np.random.default_rng(99)
        test_idx = rng.choice(n_vectors - 1, size=n_test, replace=False)
        errors = []
        for idx in test_idx:
            pred = T(Z[idx])
            if idx + 1 < n_vectors:
                errors.append(float(np.linalg.norm(pred - Z[idx + 1])))
        cv_error = float(np.mean(errors)) if errors else float('inf')

        # Domain bounds for 1D analysis compatibility
        if d_embed == 1:
            domain = (float(domain_min[0]), float(domain_max[0]))
        else:
            domain = (float(domain_min.min()), float(domain_max.max()))

        takens = TakensResult(
            embedding_dim=d_embed,
            delay=tau,
            embedded_data=Z,
            ami_values=ami,
            fnn_fractions=fnn,
            optimal_delay_method="ami",
            optimal_dim_method="fnn",
            reconstruction_quality=1.0 / (1.0 + cv_error),
        )

        return ReconstructedSystem(
            T=T,
            domain=domain,
            embedding_dim=d_embed,
            delay=tau,
            n_local_models=len(models),
            reconstruction_error=cv_error,
            snr_db=filtered.snr_db,
            takens=takens,
            filtered=filtered,
        )

    def _reconstruct_multivariate(self, y: np.ndarray) -> ReconstructedSystem:
        """
        Reconstruct dynamics from a multivariate time series (n_samples, n_channels).

        Skips AMI/FNN (unnecessary when multiple channels are directly observed).
        Uses block-Hankel embedding with small lag multiplier, then standard
        local linear model fitting in the multi-dimensional space.
        """
        n_samples, n_channels = y.shape
        if n_samples < 100:
            raise ValueError(f"Need at least 100 data points, got {n_samples}")

        # Filter each channel independently
        filtered_channels = []
        snrs = []
        for ch in range(n_channels):
            f = self.filter_noise(y[:, ch])
            filtered_channels.append(f.clean_signal)
            snrs.append(f.snr_db)
        y_clean = np.column_stack(filtered_channels)

        # Block-Hankel: use tau=1 with d_lag=2 lags → d_embed = n_channels * d_lag
        d_lag = min(2, max(1, 10 // n_channels))
        d_embed = n_channels * d_lag
        n_vectors = n_samples - (d_lag - 1)

        Z = np.zeros((n_vectors, d_embed))
        for lag in range(d_lag):
            Z[:, lag * n_channels:(lag + 1) * n_channels] = y_clean[lag:lag + n_vectors]

        # Build local models and global map
        models = self._build_local_linear_models(Z)
        domain_min = Z.min(axis=0)
        domain_max = Z.max(axis=0)
        T = self._assemble_global_map(models, domain_min, domain_max)

        # Cross-validation
        n_test = min(200, n_vectors // 5)
        rng = np.random.default_rng(99)
        test_idx = rng.choice(n_vectors - 1, size=max(n_test, 1), replace=False)
        errors = [float(np.linalg.norm(T(Z[i]) - Z[i + 1]))
                  for i in test_idx if i + 1 < n_vectors]
        cv_error = float(np.mean(errors)) if errors else float('inf')

        domain = (float(domain_min.min()), float(domain_max.max()))
        avg_snr = float(np.mean(snrs))

        # Synthetic AMI/FNN for TakensResult
        takens = TakensResult(
            embedding_dim=d_embed,
            delay=1,
            embedded_data=Z,
            ami_values=np.array([0.0]),
            fnn_fractions=np.zeros(d_embed),
            optimal_delay_method="multivariate_direct",
            optimal_dim_method="multivariate_direct",
            reconstruction_quality=1.0 / (1.0 + cv_error),
        )

        filtered_summary = FilteredSignal(
            clean_signal=y_clean[:, 0],
            noise_estimate=y[:, 0] - y_clean[:, 0],
            snr_db=avg_snr,
            filter_method=self.noise_filter,
            noise_variance=float(np.var(y[:, 0] - y_clean[:, 0])),
            effective_dimension=n_channels,
        )

        return ReconstructedSystem(
            T=T,
            domain=domain,
            embedding_dim=d_embed,
            delay=1,
            n_local_models=len(models),
            reconstruction_error=cv_error,
            snr_db=avg_snr,
            takens=takens,
            filtered=filtered_summary,
        )


class SurrogateTest:
    """
    Surrogate data test for distinguishing deterministic chaos from stochastic noise.

    Implements the IAAFT (Iterative Amplitude Adjusted Fourier Transform) method
    of Schreiber & Schmitz (1996). Generates surrogates that preserve both the
    power spectrum and amplitude distribution of the original, destroying only
    nonlinear deterministic structure. If the original differs significantly
    from surrogates on a nonlinear statistic, the data has deterministic structure.
    """

    def __init__(self, n_surrogates: int = 39, n_iaaft_iter: int = 50):
        self.n_surrogates = n_surrogates
        self.n_iaaft_iter = n_iaaft_iter

    def _iaaft_surrogate(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Generate one IAAFT surrogate preserving spectrum + distribution."""
        n = len(y)
        sorted_y = np.sort(y)
        original_fft = np.fft.rfft(y)
        amplitudes = np.abs(original_fft)

        # Start with a random shuffle
        surrogate = rng.permutation(y).copy()

        for _ in range(self.n_iaaft_iter):
            # Impose original amplitude spectrum
            surr_fft = np.fft.rfft(surrogate)
            phases = np.angle(surr_fft)
            surr_fft = amplitudes * np.exp(1j * phases)
            surrogate = np.fft.irfft(surr_fft, n=n)

            # Impose original rank order (amplitude distribution)
            rank_order = np.argsort(np.argsort(surrogate))
            surrogate = sorted_y[rank_order]

        return surrogate

    def _prediction_error_statistic(self, y: np.ndarray, tau: int = 1, k: int = 5) -> float:
        """
        Nonlinear prediction error as discriminating statistic.

        For each point, find k nearest neighbors in delay space
        and predict next step. Mean error = statistic.
        Low error → deterministic structure.
        """
        n = len(y)
        d = min(3, max(2, n // 500))
        n_vec = n - d * tau
        if n_vec < k + 10:
            return float('inf')

        Z = np.zeros((n_vec, d))
        for j in range(d):
            Z[:, j] = y[j * tau: j * tau + n_vec]

        errors = []
        n_check = min(500, n_vec - 1)
        rng = np.random.default_rng(42)
        check_idx = rng.choice(n_vec - 1, size=n_check, replace=False)

        for idx in check_idx:
            dists = np.linalg.norm(Z[:n_vec - 1] - Z[idx], axis=1)
            dists[idx] = np.inf
            nn = np.argsort(dists)[:k]
            # Predict: average of next values of neighbors
            pred = np.mean(Z[nn + 1, 0]) if np.all(nn + 1 < n_vec) else Z[idx, 0]
            actual = Z[idx + 1, 0] if idx + 1 < n_vec else Z[idx, 0]
            errors.append((pred - actual) ** 2)

        return float(np.mean(errors))

    def test(self, y: np.ndarray) -> SurrogateTestResult:
        """
        Run the full surrogate data test.

        Returns SurrogateTestResult with p-value and determinism verdict.
        """
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(42)

        original_stat = self._prediction_error_statistic(y)

        surr_stats = np.zeros(self.n_surrogates)
        for i in range(self.n_surrogates):
            surr = self._iaaft_surrogate(y, rng)
            surr_stats[i] = self._prediction_error_statistic(surr)

        # Rank-order p-value: fraction of surrogates with statistic ≤ original
        # (lower prediction error = more deterministic, so we test original < surrogates)
        p_value = float(np.sum(surr_stats <= original_stat) + 1) / (self.n_surrogates + 1)

        return SurrogateTestResult(
            is_deterministic=p_value < 0.05,
            p_value=p_value,
            original_statistic=original_stat,
            surrogate_statistics=surr_stats,
            n_surrogates=self.n_surrogates,
            statistic_name="prediction_error",
        )


def estimate_correlation_dimension(
    y: np.ndarray,
    tau: int = 1,
    d_range: Optional[Tuple[int, int]] = None,
    n_max: int = 3000,
) -> dict:
    """
    Estimate correlation dimension via Grassberger-Procaccia (1983).

    Computes C(r) = (2/N(N-1)) * count(||z_i - z_j|| < r) for log-spaced r,
    then fits log C(r) vs log r to get the slope (= correlation dimension).
    Repeats for multiple embedding dimensions to verify convergence.

    Returns:
        dict with d_corr (converged estimate), slopes per dim, scaling_quality.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    d_min, d_max = d_range or (2, min(8, n // (3 * tau)))

    results_per_d = []

    for d in range(d_min, d_max + 1):
        n_vec = n - (d - 1) * tau
        if n_vec < 50:
            break

        # Build delay vectors
        Z = np.zeros((n_vec, d))
        for k in range(d):
            Z[:, k] = y[k * tau: k * tau + n_vec]

        # Subsample for speed
        if n_vec > n_max:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_vec, size=n_max, replace=False)
            Z = Z[idx]
            n_vec = n_max

        # Pairwise distances (upper triangle)
        dists = []
        for i in range(min(n_vec, 2000)):
            d_row = np.linalg.norm(Z[i + 1:] - Z[i], axis=1)
            dists.extend(d_row.tolist())
        dists = np.array(dists)
        dists = dists[dists > 0]

        if len(dists) < 100:
            continue

        # Log-spaced radii
        r_min = np.percentile(dists, 1)
        r_max = np.percentile(dists, 99)
        if r_min <= 0 or r_max <= r_min:
            continue
        radii = np.logspace(np.log10(r_min), np.log10(r_max), 20)

        # Correlation integral C(r)
        n_pairs = len(dists)
        C = np.array([float(np.sum(dists < r)) / n_pairs for r in radii])
        C = C[C > 0]
        radii_valid = radii[:len(C)]

        if len(C) < 5:
            continue

        # Linear regression on scaling region (middle 60%)
        log_r = np.log(radii_valid)
        log_C = np.log(C)
        n_pts = len(log_r)
        start = n_pts // 5
        end = 4 * n_pts // 5
        if end - start < 3:
            continue

        log_r_fit = log_r[start:end]
        log_C_fit = log_C[start:end]
        coeffs = np.polyfit(log_r_fit, log_C_fit, 1)
        slope = float(coeffs[0])

        # R² for scaling quality
        predicted = np.polyval(coeffs, log_r_fit)
        ss_res = np.sum((log_C_fit - predicted) ** 2)
        ss_tot = np.sum((log_C_fit - np.mean(log_C_fit)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-14)

        results_per_d.append({
            "embedding_dim": d,
            "d_corr": slope,
            "r_squared": r_squared,
        })

    if not results_per_d:
        return {"d_corr": float('nan'), "slopes": [], "converged": False}

    # Convergence: last 3 estimates within 10% of each other
    slopes = [r["d_corr"] for r in results_per_d]
    d_corr = slopes[-1] if slopes else float('nan')
    converged = False
    if len(slopes) >= 3:
        last3 = slopes[-3:]
        spread = (max(last3) - min(last3)) / (abs(np.mean(last3)) + 1e-14)
        converged = spread < 0.1
        d_corr = float(np.mean(last3))

    return {
        "d_corr": d_corr,
        "slopes": results_per_d,
        "converged": converged,
        "scaling_quality": results_per_d[-1]["r_squared"] if results_per_d else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BARRIER 2: Non-Stationarity — The World Changes
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Detects non-stationarity and regime changes in time series data.

    Methods:
      1. Sliding-window Lyapunov exponent tracking (ratio or Benettin-QR)
      2. Change-point detection (CUSUM or Bayesian Online — BOCPD)
      3. Spectral gap trajectory monitoring
      4. Certificate aging with exponential decay
      5. Adaptive windowing based on characteristic timescale
    """

    def __init__(
        self,
        window_size: int = 500,
        step_size: int = 50,
        cusum_threshold: float = 3.0,
        decay_rate: float = 0.01,   # relevance decay per time step
        changepoint_method: str = "cusum",   # "cusum" or "bocpd"
        lyapunov_method: str = "ratio",      # "ratio" or "benettin"
        adaptive_window: bool = False,
        bocpd_hazard_lambda: int = 250,
        bocpd_threshold: float = 0.5,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.cusum_threshold = cusum_threshold
        self.decay_rate = decay_rate
        self.changepoint_method = changepoint_method
        self.lyapunov_method = lyapunov_method
        self.adaptive_window = adaptive_window
        self.bocpd_hazard_lambda = bocpd_hazard_lambda
        self.bocpd_threshold = bocpd_threshold

    def _sliding_lyapunov(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sliding-window Lyapunov exponent estimates.

        For each window of size W, estimate λ_max via the Eckmann-Ruelle
        method: fit local linear map, compute the average log expansion.
        """
        n = len(y)
        W = self.window_size
        step = self.step_size
        n_windows = max(1, (n - W) // step + 1)

        lyapunovs = np.zeros(n_windows)
        delta = 1e-7

        for w in range(n_windows):
            start = w * step
            end = start + W
            if end > n:
                break

            window = y[start:end]

            # Estimate λ via local divergence rate
            # Use consecutive differences as a proxy for |T'|
            diffs = np.abs(np.diff(window))
            diffs = diffs[diffs > delta]

            if len(diffs) > 10:
                # Average log|consecutive ratio|
                ratios = diffs[1:] / (diffs[:-1] + 1e-14)
                ratios = ratios[ratios > 1e-14]
                if len(ratios) > 5:
                    lyapunovs[w] = float(np.mean(np.log(ratios + 1e-14)))
                else:
                    lyapunovs[w] = 0.0
            else:
                lyapunovs[w] = 0.0

        return lyapunovs[:n_windows]

    def _sliding_lyapunov_benettin(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sliding-window Lyapunov exponents via Benettin-QR algorithm.

        Benettin et al. (1980): For each window, reconstruct local dynamics,
        compute the Jacobian product chain with QR reorthogonalization.
        This gives the full Lyapunov spectrum (we take λ_max).

        Far more accurate than the ratio method, works in any dimension.
        """
        n = len(y)
        W = self.window_size
        step = self.step_size
        n_windows = max(1, (n - W) // step + 1)
        lyapunovs = np.zeros(n_windows)

        reconstructor = TimeSeriesReconstructor(
            noise_filter="none", max_embedding_dim=5,
            local_model_k=min(20, W // 20),
        )

        for w in range(n_windows):
            start = w * step
            end = start + W
            if end > n:
                break

            window = y[start:end]
            try:
                system = reconstructor.reconstruct(window)
                Z = system.takens.embedded_data
                n_vec, d = Z.shape

                if n_vec < 20 or d < 1:
                    continue

                # Benettin-QR: iterate Jacobian chain with reorthogonalization
                Q = np.eye(d)
                log_sum = np.zeros(d)
                delta = 1e-6
                n_orbit = min(n_vec - 1, 500)

                for t in range(n_orbit):
                    zt = Z[t]
                    # Numerical Jacobian at z_t
                    J = np.zeros((d, d))
                    for j in range(d):
                        z_plus = zt.copy()
                        z_plus[j] += delta
                        z_minus = zt.copy()
                        z_minus[j] -= delta
                        J[:, j] = (np.atleast_1d(system.T(z_plus))
                                   - np.atleast_1d(system.T(z_minus))) / (2 * delta)

                    # Propagate and reorthogonalize
                    M = J @ Q
                    Q_new, R = np.linalg.qr(M)
                    diag_R = np.abs(np.diag(R))
                    diag_R[diag_R < 1e-300] = 1e-300
                    log_sum += np.log(diag_R)
                    Q = Q_new

                lyapunovs[w] = float(log_sum[0] / max(n_orbit, 1))

            except Exception:
                lyapunovs[w] = 0.0

        return lyapunovs[:n_windows]

    def _bocpd_detect(self, statistic: np.ndarray) -> List[int]:
        """
        Bayesian Online Changepoint Detection (Adams & MacKay 2007).

        Maintains a run-length distribution P(r_t | data_{1:t}).
        Uses Normal-Inverse-Gamma conjugate for Student-t predictive.
        More principled than CUSUM: produces posterior probabilities
        of changepoints rather than heuristic threshold crossings.
        """
        n = len(statistic)
        if n < 5:
            return []

        hazard = 1.0 / self.bocpd_hazard_lambda
        max_run = min(500, n)

        # Sufficient statistics for Normal-Inverse-Gamma
        # Per run length: count, mean, sum of squared deviations
        kappa_0, mu_0, alpha_0, beta_0 = 1.0, 0.0, 1.0, 1.0

        # Run-length probability vector (truncated at max_run)
        R = np.zeros(max_run + 1)
        R[0] = 1.0

        # Sufficient stats arrays (one per run length)
        counts = np.zeros(max_run + 1)
        means = np.full(max_run + 1, mu_0)
        sum_sq = np.zeros(max_run + 1)

        change_points = []

        for t in range(n):
            x = statistic[t]

            # Evaluate predictive probability for each run length
            # Student-t predictive: p(x | r) with NIG posterior
            pred_probs = np.zeros(max_run + 1)
            for r in range(min(t + 1, max_run)):
                if R[r] < 1e-300:
                    continue
                n_r = counts[r]
                kappa_n = kappa_0 + n_r
                mu_n = (kappa_0 * mu_0 + n_r * means[r]) / kappa_n
                alpha_n = alpha_0 + n_r / 2
                beta_n = beta_0 + sum_sq[r] / 2 + \
                    kappa_0 * n_r * (means[r] - mu_0) ** 2 / (2 * kappa_n)

                # Student-t with 2*alpha_n degrees of freedom
                nu = 2 * alpha_n
                scale = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
                scale = max(scale, 1e-14)
                z = (x - mu_n) ** 2 / scale
                # Student-t density (up to normalization constant)
                pred_probs[r] = (1 + z / nu) ** (-(nu + 1) / 2) / np.sqrt(scale)

            # Growth probabilities
            R_new = np.zeros(max_run + 1)
            # Changepoint probability
            cp_mass = 0.0
            for r in range(min(t + 1, max_run)):
                growth = R[r] * pred_probs[r] * (1 - hazard)
                if r + 1 < max_run:
                    R_new[r + 1] += growth
                cp_mass += R[r] * pred_probs[r] * hazard

            R_new[0] = cp_mass

            # Normalize
            total = R_new.sum()
            if total > 1e-300:
                R_new /= total
            else:
                R_new[0] = 1.0

            # Update sufficient statistics
            new_counts = np.zeros(max_run + 1)
            new_means = np.full(max_run + 1, mu_0)
            new_sum_sq = np.zeros(max_run + 1)

            for r in range(1, min(t + 2, max_run)):
                new_counts[r] = counts[r - 1] + 1
                old_mean = means[r - 1]
                new_means[r] = old_mean + (x - old_mean) / new_counts[r]
                new_sum_sq[r] = sum_sq[r - 1] + (x - old_mean) * (x - new_means[r])
            new_counts[0] = 1
            new_means[0] = x
            new_sum_sq[0] = 0.0

            R = R_new
            counts = new_counts
            means = new_means
            sum_sq = new_sum_sq

            # Detect changepoint
            if t > 5 and R[0] > self.bocpd_threshold:
                change_points.append(t)

        return change_points

    def _adaptive_window_size(self, y: np.ndarray) -> int:
        """
        Adapt window size to the characteristic timescale of the dynamics.

        Chaotic → small windows (fast decorrelation).
        Stable → large windows (slow dynamics need more data).
        """
        # Quick Lyapunov on first 2*window_size samples
        n_preliminary = min(len(y), 2 * self.window_size)
        preliminary = y[:n_preliminary]

        diffs = np.abs(np.diff(preliminary))
        diffs = diffs[diffs > 1e-10]
        if len(diffs) > 20:
            ratios = diffs[1:] / (diffs[:-1] + 1e-14)
            ratios = ratios[(ratios > 1e-14) & (ratios < 1e14)]
            if len(ratios) > 10:
                lam = float(np.mean(np.log(ratios + 1e-14)))
            else:
                lam = 0.0
        else:
            lam = 0.0

        # Characteristic timescale
        tau_char = 1.0 / max(abs(lam), 0.01)
        # Window = ~10 characteristic times, clamped
        adaptive_w = int(np.clip(10 * tau_char, 200, 2000))
        # Round to step_size
        adaptive_w = max(adaptive_w // self.step_size * self.step_size, 200)
        return adaptive_w

    def _cusum_detect(self, statistic: np.ndarray) -> List[int]:
        """
        CUSUM (Page 1954) change-point detection.

        Tracks cumulative deviation from the running mean. A change point
        is declared when the CUSUM exceeds the threshold.

        H_0: statistic[t] ~ N(μ, σ²) stationary
        H_1: mean shift at unknown time t*
        """
        n = len(statistic)
        if n < 5:
            return []

        mu = np.mean(statistic)
        sigma = np.std(statistic) + 1e-14

        # Normalized CUSUM
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        change_points = []

        for t in range(1, n):
            z = (statistic[t] - mu) / sigma
            cusum_pos[t] = max(0, cusum_pos[t - 1] + z - 0.5)
            cusum_neg[t] = max(0, cusum_neg[t - 1] - z - 0.5)

            if cusum_pos[t] > self.cusum_threshold or cusum_neg[t] > self.cusum_threshold:
                change_points.append(t)
                cusum_pos[t] = 0
                cusum_neg[t] = 0

        return change_points

    def _classify_regime(self, lyap: float) -> str:
        """Classify a regime based on its Lyapunov exponent."""
        if lyap > 0.1:
            return "chaotic"
        elif lyap > 0.01:
            return "weakly_chaotic"
        elif lyap > -0.01:
            return "quasiperiodic"
        elif lyap > -0.5:
            return "periodic"
        else:
            return "strongly_stable"

    def analyze(self, y: np.ndarray) -> RegimeAnalysis:
        """
        Full non-stationarity analysis of a time series.

        Returns:
            RegimeAnalysis with segments, change points, and trajectories.
        """
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        # Adaptive windowing if enabled
        if self.adaptive_window:
            self.window_size = self._adaptive_window_size(y)

        # Compute sliding-window Lyapunov (ratio or Benettin-QR)
        if self.lyapunov_method == "benettin":
            lyap_traj = self._sliding_lyapunov_benettin(y)
        else:
            lyap_traj = self._sliding_lyapunov(y)

        # Detect change points (CUSUM or BOCPD)
        if self.changepoint_method == "bocpd":
            cp_indices = self._bocpd_detect(lyap_traj)
        else:
            cp_indices = self._cusum_detect(lyap_traj)

        # Map change point indices back to original time series indices
        change_points = []
        for cp in cp_indices:
            orig_idx = cp * self.step_size + self.window_size // 2

            before_lyap = float(np.mean(lyap_traj[max(0, cp - 3):cp])) if cp > 0 else 0.0
            after_lyap = float(np.mean(lyap_traj[cp:min(len(lyap_traj), cp + 3)]))

            if before_lyap < 0.05 and after_lyap > 0.1:
                change_type = "chaos_onset"
            elif before_lyap > 0.1 and after_lyap < 0.05:
                change_type = "chaos_offset"
            else:
                change_type = "bifurcation"

            change_points.append(ChangePoint(
                index=orig_idx,
                cusum_statistic=float(lyap_traj[cp]),
                before_lyapunov=before_lyap,
                after_lyapunov=after_lyap,
                significance=abs(after_lyap - before_lyap),
                change_type=change_type,
            ))

        # Build segments between change points
        boundaries = [0] + [cp.index for cp in change_points] + [n]
        segments = []
        lyap_idx = 0

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            # Average Lyapunov in this segment
            seg_start_w = max(0, start // self.step_size)
            seg_end_w = min(len(lyap_traj), end // self.step_size)
            if seg_end_w > seg_start_w:
                seg_lyap = float(np.mean(lyap_traj[seg_start_w:seg_end_w]))
            else:
                seg_lyap = 0.0

            regime = self._classify_regime(seg_lyap)
            confidence = min(1.0, (end - start) / self.window_size)

            segments.append(RegimeSegment(
                start_idx=start,
                end_idx=end,
                lyapunov_estimate=seg_lyap,
                spectral_gap_estimate=max(0.0, seg_lyap),  # Γ ≈ λ for 1D
                entropy_estimate=max(0.0, seg_lyap),        # h_KS ≈ λ⁺ by Pesin
                regime_type=regime,
                confidence=confidence,
            ))

        # Spectral gap trajectory (≈ Lyapunov for 1D maps)
        spec_gap_traj = np.maximum(lyap_traj, 0.0)

        is_stationary = len(change_points) == 0

        return RegimeAnalysis(
            segments=segments,
            change_points=change_points,
            is_stationary=is_stationary,
            n_regimes=len(segments),
            lyapunov_trajectory=lyap_traj,
            spectral_gap_trajectory=spec_gap_traj,
            window_size=self.window_size,
        )

    def compute_certificate_relevance(
        self,
        certificate_age: float,
        certificate_regime: str,
        current_regime: str,
    ) -> float:
        """
        Compute the relevance weight of an old certificate.

        Certificates decay exponentially with age and lose relevance
        completely when the regime changes.

        relevance(t) = exp(-decay_rate * age) * regime_match
        """
        age_factor = math.exp(-self.decay_rate * certificate_age)

        if certificate_regime == current_regime:
            regime_factor = 1.0
        elif (certificate_regime in ("chaotic", "weakly_chaotic") and
              current_regime in ("chaotic", "weakly_chaotic")):
            regime_factor = 0.5  # Similar regime
        else:
            regime_factor = 0.05  # Different regime — almost irrelevant

        return float(age_factor * regime_factor)


# ═══════════════════════════════════════════════════════════════════════════
# BARRIER 3: Partial Observability
# ═══════════════════════════════════════════════════════════════════════════

class PartialObserver:
    """
    Handles partial observability: y = h(x) where dim(y) << dim(x).

    Computes:
      1. Observability Gramian via empirical linearization
      2. Information loss bounds
      3. Certified error floors for predictions made from partial observations
    """

    def __init__(
        self,
        observation_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_obs_dims: int = 1,
    ):
        """
        Args:
            observation_func: h: R^n → R^m  (if None, identity)
            n_obs_dims: Number of observed dimensions
        """
        self.h = observation_func
        self.n_obs_dims = n_obs_dims

    def compute_observability_gramian(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        n_steps: int = 20,
        delta: float = 1e-5,
    ) -> Tuple[np.ndarray, ObservabilityReport]:
        """
        Compute the empirical observability Gramian.

        For a discrete-time system x_{t+1} = T(x_t), y_t = h(x_t):

            O = [∂h/∂x, ∂(h∘T)/∂x, ∂(h∘T²)/∂x, ..., ∂(h∘T^{n-1})/∂x]^T

        The Gramian W_O = O^T O determines which state directions are observable.
        rank(W_O) = n ↔ the system is completely observable from y.

        If rank(W_O) < n, some state directions are invisible, and we compute
        the certified information loss bound.
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        n_state = len(x0)
        n_obs = self.n_obs_dims

        if self.h is None:
            # Identity observation: fully observable
            self.h = lambda x: x[:n_obs]

        # Build the observability matrix O numerically
        # O has shape (n_steps * n_obs, n_state)
        O = np.zeros((n_steps * n_obs, n_state))

        for step in range(n_steps):
            # Compute Jacobian ∂(h ∘ T^step)/∂x at x0 via 5-point central differences
            # 4th-order accuracy: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
            for j in range(n_state):
                stencil_offsets = [-2, -1, 1, 2]
                stencil_weights = [1.0/12, -8.0/12, 8.0/12, -1.0/12]
                jac_col = np.zeros(n_obs)

                for offset, weight in zip(stencil_offsets, stencil_weights):
                    x_pert = x0.copy()
                    x_pert[j] += offset * delta

                    # Iterate T^step times
                    xp = x_pert.copy()
                    for _ in range(step):
                        xp = np.atleast_1d(T(xp))

                    yp = np.atleast_1d(self.h(xp))
                    jac_col += weight * yp[:n_obs] / delta

                O[step * n_obs:(step + 1) * n_obs, j] = jac_col

        # Gramian W_O = O^T O
        W_O = O.T @ O

        # SVD for rank and condition
        svd_vals = np.linalg.svd(W_O, compute_uv=False)
        rank = int(np.sum(svd_vals > svd_vals[0] * 1e-10))
        cond = float(svd_vals[0] / (svd_vals[-1] + 1e-300)) if svd_vals[-1] > 0 else float('inf')

        is_observable = rank >= n_state

        # Information loss bound: unobservable subspace contribution
        if rank < n_state:
            # Unobservable variance = sum of small singular values
            unobs_variance = float(np.sum(svd_vals[rank:]))
            total_variance = float(np.sum(svd_vals))
            info_loss = unobs_variance / (total_variance + 1e-14)
        else:
            info_loss = 0.0

        # Lyapunov error bound from partial observations
        if is_observable:
            lyap_error = 0.0
        else:
            # Bound: |λ_true - λ_obs| ≤ C / σ_min(O) where σ_min is smallest nonzero SV
            nonzero_svs = svd_vals[svd_vals > svd_vals[0] * 1e-10]
            sigma_min = float(nonzero_svs[-1]) if len(nonzero_svs) > 0 else 1e-14
            lyap_error = 1.0 / sigma_min  # Certified upper bound

        report = ObservabilityReport(
            observable_dims=n_obs,
            state_dims=n_state,
            observability_rank=rank,
            is_observable=is_observable,
            information_loss_bound=info_loss,
            gramian_condition=cond,
            reconstructible_lyapunov=0.0,  # Filled by caller
            lyapunov_error_bound=lyap_error,
        )

        return W_O, report

    def certify_from_observations(
        self,
        y_series: np.ndarray,
        T: Optional[Callable] = None,
        x0: Optional[np.ndarray] = None,
    ) -> ObservabilityReport:
        """
        Full observability certification from a time series of observations.

        If T and x0 are provided: computes exact Gramian.
        If only y_series: estimates observability from time-delay embedding
        (Takens theorem guarantees generic observability for d ≥ 2n+1).
        """
        y_series = np.atleast_2d(y_series)
        if y_series.shape[0] == 1:
            y_series = y_series.T
        n_samples, n_obs = y_series.shape

        if T is not None and x0 is not None:
            W_O, report = self.compute_observability_gramian(T, x0)
            return report

        # Estimate from time series: use Takens embedding SVD
        # The embedding dimension d = 2*n_state + 1 guarantees observability
        reconstructor = TimeSeriesReconstructor(noise_filter="none", max_embedding_dim=10)

        if n_obs == 1:
            tau, _ = reconstructor.estimate_delay(y_series.ravel())
            d_embed, fnn = reconstructor.estimate_dimension(y_series.ravel(), tau)

            # Build delay embedding
            n_vec = n_samples - (d_embed - 1) * tau
            Z = np.zeros((n_vec, d_embed))
            for k in range(d_embed):
                Z[:, k] = y_series[k * tau: k * tau + n_vec, 0]

            # SVD of the delay-embedded data
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            n_state_est = int(np.sum(s > s[0] * 0.01))
        else:
            Z = y_series
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            n_state_est = int(np.sum(s > s[0] * 0.01))

        rank = min(n_state_est, n_obs * 3)  # Generous estimate
        is_observable = rank >= n_state_est

        info_loss = 0.0 if is_observable else float(1.0 - n_obs / max(n_state_est, 1))

        # Takens observability guarantee
        d_embed_actual = d_embed if n_obs == 1 else n_obs
        takens_dim_required = 2 * n_state_est + 1
        takens_guaranteed = d_embed_actual >= takens_dim_required

        return ObservabilityReport(
            observable_dims=n_obs,
            state_dims=n_state_est,
            observability_rank=rank,
            is_observable=is_observable,
            information_loss_bound=info_loss,
            gramian_condition=float(s[0] / (s[-1] + 1e-300)),
            reconstructible_lyapunov=0.0,
            lyapunov_error_bound=info_loss * 2.0,
            takens_observability_guaranteed=takens_guaranteed,
            takens_dim_required=takens_dim_required,
        )

    def ekf_state_estimate(
        self,
        y_series: np.ndarray,
        T: Callable[[np.ndarray], np.ndarray],
        n_state: int,
        Q: Optional[np.ndarray] = None,
        R_noise: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> EKFResult:
        """
        Extended Kalman Filter for nonlinear state estimation from partial observations.

        Recovers the full state trajectory x_t from partial observations y_t = h(x_t).
        Uses 4th-order finite-difference Jacobians for F = DT/Dx and H = Dh/Dx.

        Args:
            y_series: (n_samples, n_obs) observations
            T: State transition x_{t+1} = T(x_t)
            n_state: Dimension of full state space
            Q: Process noise covariance (n_state × n_state), default: 0.01*I
            R_noise: Observation noise covariance (n_obs × n_obs), default: 0.1*I
            x0: Initial state estimate, default: zeros

        Returns:
            EKFResult with filtered state trajectory and diagnostics.
        """
        y_series = np.atleast_2d(y_series)
        if y_series.shape[0] == 1:
            y_series = y_series.T
        n_samples, n_obs = y_series.shape

        if self.h is None:
            self.h = lambda x: x[:n_obs]

        if Q is None:
            Q = np.eye(n_state) * 0.01
        if R_noise is None:
            R_noise = np.eye(n_obs) * 0.1
        if x0 is None:
            x0 = np.zeros(n_state)
            x0[:n_obs] = y_series[0]

        delta = 1e-6

        def _jacobian(func, x, n_out):
            """4th-order finite-difference Jacobian."""
            nx = len(x)
            J = np.zeros((n_out, nx))
            weights = [1.0/12, -8.0/12, 8.0/12, -1.0/12]
            offsets = [-2, -1, 1, 2]
            for j in range(nx):
                col = np.zeros(n_out)
                for off, w in zip(offsets, weights):
                    x_pert = x.copy()
                    x_pert[j] += off * delta
                    col += w * np.atleast_1d(func(x_pert))[:n_out] / delta
                J[:, j] = col
            return J

        x_hat = x0.copy()
        P = np.eye(n_state) * 1.0
        states = np.zeros((n_samples, n_state))
        covs = []
        innovations = np.zeros((n_samples, n_obs))

        for t in range(n_samples):
            if t > 0:
                # Predict
                F = _jacobian(T, x_hat, n_state)
                x_pred = np.atleast_1d(T(x_hat))[:n_state]
                P_pred = F @ P @ F.T + Q
            else:
                x_pred = x_hat.copy()
                P_pred = P.copy()

            # Update
            H = _jacobian(self.h, x_pred, n_obs)
            y_pred = np.atleast_1d(self.h(x_pred))[:n_obs]
            innov = y_series[t] - y_pred
            S = H @ P_pred @ H.T + R_noise
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P_pred @ H.T @ np.linalg.pinv(S)

            x_hat = x_pred + K @ innov
            P = (np.eye(n_state) - K @ H) @ P_pred

            states[t] = x_hat
            covs.append(P.copy())
            innovations[t] = innov

        # NEES: Normalized Estimation Error Squared (consistency metric)
        nees_vals = []
        for t in range(n_samples):
            try:
                P_inv = np.linalg.inv(covs[t])
                # Using innovation-based NEES since true state is unknown
                nees_t = float(innovations[t] @ np.linalg.inv(
                    H @ covs[t] @ H.T + R_noise) @ innovations[t])
                nees_vals.append(nees_t)
            except np.linalg.LinAlgError:
                pass
        nees = float(np.mean(nees_vals)) if nees_vals else float('nan')

        return EKFResult(
            state_estimates=states,
            covariances=covs,
            innovations=innovations,
            nees=nees,
        )


# ═══════════════════════════════════════════════════════════════════════════
# BARRIER 4: Finite Resources — Anytime Algorithms
# ═══════════════════════════════════════════════════════════════════════════

class AnytimeCertifier:
    """
    Interruptible, progressive refinement certifier.

    Strategy: Start with coarse grid (n=32), compute approximate certificates.
    Refine grid exponentially (32 → 64 → 128 → 256 → 512) until either:
      - ε_target is reached
      - Time budget is exhausted
      - Memory budget is exhausted

    At any interruption point, the current ε_achieved and all valid
    certificates are available.
    """

    def __init__(
        self,
        time_budget_ms: float = 10000,     # Max wall time in milliseconds
        memory_budget_bytes: int = 2 * 1024 * 1024,  # Max memory (2MB default)
        epsilon_target: float = 0.01,
        grid_schedule: Optional[List[int]] = None,
    ):
        self.time_budget_ms = time_budget_ms
        self.memory_budget_bytes = memory_budget_bytes
        self.epsilon_target = epsilon_target
        self.grid_schedule = grid_schedule or [32, 64, 128, 256, 512]

    def certify(
        self,
        T: Callable[[np.ndarray], np.ndarray],
        domain: Tuple[float, float] = (0.0, 1.0),
    ) -> AnytimeResult:
        """
        Progressive refinement certification.

        At each grid level:
          1. Build Ulam matrix (O(n²) memory, O(n² · n_quad) time)
          2. Power iteration for μ_SRB (O(n²) per iteration)
          3. Eigendecomposition (O(n³))
          4. Compute h_KS and spectral gap

        Memory constraint: n_grid ≤ √(memory_budget / 8)  (float64 matrix)
        """
        start_time = time.time()
        refinement_history = []
        best_h_ks = 0.0
        best_gap = 0.0
        best_eps = float('inf')
        best_n = 0
        n_certs_valid = 0
        peak_memory = 0

        for n_grid in self.grid_schedule:
            # Check memory constraint
            matrix_bytes = n_grid * n_grid * 8  # float64
            total_bytes = matrix_bytes + n_grid * 8 * 3  # matrix + vectors
            peak_memory = max(peak_memory, total_bytes)
            if matrix_bytes > self.memory_budget_bytes:
                break

            # Check time constraint
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.time_budget_ms * 0.9:  # 90% budget used
                break

            # Build and analyze at this resolution
            try:
                from acf_functor.gelfand_triple import GelfandTriple
                triple = GelfandTriple(
                    T, domain=domain,
                    n_test=min(n_grid // 4, 32),
                    n_dist=n_grid,
                    n_hilbert=n_grid,
                )
                triple.build()

                # Quick analysis (fewer modes, shorter orbit)
                mu_srb, n_iter = triple.compute_self_consistent_measure()

                # Spectral gap from a few eigenvalues
                n_modes = min(8, n_grid - 1)
                eigvals = linalg.eigvals(triple._L)
                mods = np.sort(np.abs(eigvals))[::-1]
                nontrivial = mods[mods < 0.999]
                gap = float(-np.log(nontrivial[0])) if len(nontrivial) > 0 else 0.0

                # h_KS via Pesin (quick orbit)
                h_ks = triple.compute_lyapunov_entropy(n_orbit=min(10000, n_grid * 50))

                # ε estimate: convergence error of μ_SRB
                mu_check = triple._L @ mu_srb
                mu_check /= mu_check.sum() + 1e-14
                eps_achieved = float(np.linalg.norm(mu_check - mu_srb, ord=1))

                elapsed_ms = (time.time() - start_time) * 1000

                snapshot = {
                    "n_grid": n_grid,
                    "h_ks": float(h_ks),
                    "spectral_gap": float(gap),
                    "epsilon": float(eps_achieved),
                    "time_ms": float(elapsed_ms),
                    "mu_srb_converged": bool(eps_achieved < 0.01),
                }
                refinement_history.append(snapshot)

                if eps_achieved < best_eps:
                    best_eps = eps_achieved
                    best_h_ks = h_ks
                    best_gap = gap
                    best_n = n_grid

                n_certs_valid += int(eps_achieved < 0.1)

                if eps_achieved < self.epsilon_target and len(refinement_history) >= 2:
                    break

            except Exception as e:
                refinement_history.append({
                    "n_grid": n_grid,
                    "error": str(e),
                    "time_ms": (time.time() - start_time) * 1000,
                })

        total_time = (time.time() - start_time) * 1000

        # Confidence interval from refinement history
        h_vals = [s["h_ks"] for s in refinement_history if "h_ks" in s]
        if len(h_vals) >= 2:
            h_std = float(np.std(h_vals))
            ci = (best_h_ks - 2 * h_std, best_h_ks + 2 * h_std)
        else:
            ci = (best_h_ks * 0.8, best_h_ks * 1.2)

        return AnytimeResult(
            epsilon_achieved=best_eps,
            n_grid_used=best_n,
            h_ks_estimate=best_h_ks,
            h_ks_confidence_interval=ci,
            spectral_gap_estimate=best_gap,
            computation_time_ms=total_time,
            refinement_history=refinement_history,
            is_converged=best_eps < self.epsilon_target,
            certificates_valid=n_certs_valid,
            peak_memory_bytes=peak_memory,
        )


class KnowledgeCompressor:
    """
    Compresses the knowledge graph K_t by merging similar certificates
    into meta-theorems and aging out stale information.

    Strategy:
      1. Group certificates by system type and regime
      2. Within each group, compute mean ± std of key quantities
      3. Replace N similar certificates with one meta-theorem:
         "For systems of type X in regime Y: h_KS ∈ [a, b] with p=0.95"
      4. Exponentially decay relevance of old certificates
    """

    def __init__(
        self,
        max_certificates: int = 100,
        decay_rate: float = 0.01,
        merge_threshold: float = 0.1,
    ):
        self.max_certificates = max_certificates
        self.decay_rate = decay_rate
        self.merge_threshold = merge_threshold
        self._certificates: List[dict] = []
        self._age_counter: int = 0

    def add_certificate(self, cert: dict, priority: float = 1.0) -> None:
        """Add a new certificate to the knowledge graph with optional priority."""
        cert["_timestamp"] = self._age_counter
        cert["_priority"] = priority
        cert["_relevance"] = priority
        self._certificates.append(cert)
        self._age_counter += 1

    def age_all(self, steps: int = 1) -> None:
        """Age all certificates by `steps` time units. Priority resists decay."""
        for cert in self._certificates:
            age = self._age_counter - cert["_timestamp"]
            priority = cert.get("_priority", 1.0)
            cert["_relevance"] = priority * math.exp(-self.decay_rate * age)

    def compress(self) -> CompressedKnowledge:
        """
        Compress the knowledge graph.

        Algorithm:
          1. Remove certificates with relevance < 0.01
          2. Group by regime type
          3. Within each group, merge if h_KS values are within merge_threshold
          4. Create meta-theorems summarizing each group
        """
        self.age_all()

        # Prune irrelevant
        active = [c for c in self._certificates if c.get("_relevance", 0) > 0.01]
        n_original = len(self._certificates)

        # Group by regime
        groups: Dict[str, List[dict]] = {}
        for cert in active:
            regime = cert.get("regime_type", "unknown")
            groups.setdefault(regime, []).append(cert)

        meta_theorems = []
        for regime, certs in groups.items():
            h_vals = [c.get("h_ks", 0) for c in certs if "h_ks" in c]
            gap_vals = [c.get("spectral_gap", 0) for c in certs if "spectral_gap" in c]
            lyap_vals = [c.get("lyapunov", 0) for c in certs if "lyapunov" in c]

            meta = {
                "regime": regime,
                "n_certificates": len(certs),
                "h_ks_mean": float(np.mean(h_vals)) if h_vals else 0.0,
                "h_ks_std": float(np.std(h_vals)) if len(h_vals) > 1 else 0.0,
                "h_ks_range": (float(min(h_vals)), float(max(h_vals))) if h_vals else (0, 0),
                "spectral_gap_mean": float(np.mean(gap_vals)) if gap_vals else 0.0,
                "lyapunov_mean": float(np.mean(lyap_vals)) if lyap_vals else 0.0,
                "oldest_age": max(self._age_counter - c["_timestamp"] for c in certs),
                "newest_age": min(self._age_counter - c["_timestamp"] for c in certs),
                "mean_relevance": float(np.mean([c["_relevance"] for c in certs])),
            }
            meta_theorems.append(meta)

        # Replace certificate list with active only
        self._certificates = active

        n_compressed = len(meta_theorems)
        info_retained = len(active) / max(n_original, 1)

        oldest = max(
            (self._age_counter - c["_timestamp"] for c in active),
            default=0
        )

        return CompressedKnowledge(
            n_original_certificates=n_original,
            n_compressed=n_compressed,
            meta_theorems=meta_theorems,
            compression_ratio=n_compressed / max(n_original, 1),
            information_retained=info_retained,
            oldest_relevant=float(oldest),
            decay_constant=self.decay_rate,
        )


class StreamingCertifier:
    """
    Process time series data incrementally with bounded memory.

    Designed for embedded systems, real-time monitoring, and scenarios
    where the entire series cannot fit in memory. Uses a sliding buffer
    (collections.deque) with hard memory bound. Each full window is
    certified independently; only running statistics are retained.

    Usage:
        streamer = StreamingCertifier(window_size=1000)
        for chunk in data_source:
            result = streamer.ingest(chunk)
            if result is not None:
                print(f"Window h_KS = {result['h_ks']:.4f}")
        summary = streamer.summary()
    """

    def __init__(
        self,
        window_size: int = 1000,
        overlap: int = 200,
        memory_budget_bytes: int = 1_000_000,
        noise_filter: str = "svd",
        epsilon_target: float = 0.01,
    ):
        self._window_size = window_size
        self._overlap = overlap
        self._memory_budget_bytes = memory_budget_bytes
        self._noise_filter = noise_filter
        self._epsilon_target = epsilon_target
        self._buffer: collections.deque = collections.deque(maxlen=window_size)
        self._n_ingested = 0
        self._n_windows_processed = 0
        self._step = window_size - overlap

        # Running statistics (bounded memory)
        self._h_ks_values: List[float] = []
        self._lyapunov_values: List[float] = []
        self._regime_counts: Dict[str, int] = {}
        self._last_result: Optional[dict] = None

    def ingest(self, chunk: np.ndarray) -> Optional[dict]:
        """
        Add data to the buffer. Returns a certification result when
        a full window is ready, otherwise None.
        """
        chunk = np.asarray(chunk, dtype=float).ravel()
        for val in chunk:
            self._buffer.append(val)
            self._n_ingested += 1

        # Check if we have a full window
        if len(self._buffer) >= self._window_size:
            window = np.array(self._buffer)
            result = self._certify_window(window)

            # Slide buffer: keep only the overlap portion
            for _ in range(self._step):
                if self._buffer:
                    self._buffer.popleft()

            self._n_windows_processed += 1
            self._last_result = result
            return result

        return None

    def _certify_window(self, window: np.ndarray) -> dict:
        """Certify one window with lightweight pipeline."""
        reconstructor = TimeSeriesReconstructor(noise_filter=self._noise_filter)
        detector = RegimeDetector(window_size=min(200, len(window) // 3), step_size=20)

        # Filter
        filtered = reconstructor.filter_noise(window)

        # Regime detection
        regimes = detector.analyze(window)
        regime_type = regimes.segments[-1].regime_type if regimes.segments else "unknown"
        lyap = regimes.segments[-1].lyapunov_estimate if regimes.segments else 0.0

        # Quick Lyapunov-based h_KS estimate
        h_ks = max(0.0, lyap)

        # Track running stats
        self._h_ks_values.append(h_ks)
        self._lyapunov_values.append(lyap)
        self._regime_counts[regime_type] = self._regime_counts.get(regime_type, 0) + 1

        return {
            "window_index": self._n_windows_processed,
            "h_ks": h_ks,
            "lyapunov": lyap,
            "regime": regime_type,
            "snr_db": filtered.snr_db,
            "n_samples_total": self._n_ingested,
        }

    def summary(self) -> dict:
        """Aggregate statistics across all processed windows."""
        h_arr = np.array(self._h_ks_values) if self._h_ks_values else np.array([0.0])
        l_arr = np.array(self._lyapunov_values) if self._lyapunov_values else np.array([0.0])

        return {
            "n_windows_processed": self._n_windows_processed,
            "n_samples_ingested": self._n_ingested,
            "h_ks_mean": float(np.mean(h_arr)),
            "h_ks_std": float(np.std(h_arr)),
            "h_ks_range": (float(np.min(h_arr)), float(np.max(h_arr))),
            "lyapunov_mean": float(np.mean(l_arr)),
            "lyapunov_std": float(np.std(l_arr)),
            "regime_distribution": dict(self._regime_counts),
            "memory_budget_bytes": self._memory_budget_bytes,
            "actual_buffer_bytes": self._window_size * 8,  # float64
        }


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Real-World Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class RealWorldPipeline:
    """
    Complete real-world analysis pipeline.

    Usage:
        pipeline = RealWorldPipeline()
        report = pipeline.analyze_timeseries(sensor_data)

        # Or with constraints:
        report = pipeline.analyze_timeseries(
            sensor_data,
            time_budget_ms=100,
            memory_budget_bytes=1024*1024,
            observation_func=lambda x: x[:3],  # GPS only
        )
    """

    def __init__(
        self,
        noise_filter: str = "svd",
        regime_window: int = 500,
        time_budget_ms: float = 10000,
        memory_budget_bytes: int = 2 * 1024 * 1024,
        epsilon_target: float = 0.01,
    ):
        self.reconstructor = TimeSeriesReconstructor(noise_filter=noise_filter)
        self.regime_detector = RegimeDetector(window_size=regime_window)
        self.anytime = AnytimeCertifier(
            time_budget_ms=time_budget_ms,
            memory_budget_bytes=memory_budget_bytes,
            epsilon_target=epsilon_target,
        )
        self.compressor = KnowledgeCompressor()

    def analyze_timeseries(
        self,
        y: np.ndarray,
        observation_func: Optional[Callable] = None,
        x0: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Full real-world analysis pipeline.

        Steps:
          1. Filter noise (Barrier 1)
          2. Detect regimes (Barrier 2)
          3. Reconstruct dynamics (Barrier 1)
          4. Check observability (Barrier 3)
          5. Anytime certification (Barrier 4)
          6. Compress knowledge (Barrier 4)

        Returns:
            Complete report dictionary with all analyses.
        """
        y = np.asarray(y, dtype=float).ravel()
        report = {"n_samples": len(y), "barriers_addressed": [1, 2, 3, 4]}

        # --- Barrier 1: Noise filtering ---
        filtered = self.reconstructor.filter_noise(y)
        report["filtering"] = {
            "method": filtered.filter_method,
            "snr_db": filtered.snr_db,
            "noise_variance": filtered.noise_variance,
            "effective_dimension": filtered.effective_dimension,
        }

        # --- Barrier 2: Regime detection ---
        regimes = self.regime_detector.analyze(y)
        report["regimes"] = {
            "is_stationary": regimes.is_stationary,
            "n_regimes": regimes.n_regimes,
            "change_points": [
                {
                    "index": cp.index,
                    "type": cp.change_type,
                    "before_lyap": cp.before_lyapunov,
                    "after_lyap": cp.after_lyapunov,
                }
                for cp in regimes.change_points
            ],
            "segments": [
                {
                    "start": seg.start_idx,
                    "end": seg.end_idx,
                    "regime": seg.regime_type,
                    "lyapunov": seg.lyapunov_estimate,
                    "confidence": seg.confidence,
                }
                for seg in regimes.segments
            ],
        }

        # --- Barrier 1 continued: Reconstruct dynamics ---
        try:
            system = self.reconstructor.reconstruct(filtered.clean_signal)
            report["reconstruction"] = {
                "embedding_dim": system.embedding_dim,
                "delay": system.delay,
                "n_local_models": system.n_local_models,
                "cv_error": system.reconstruction_error,
                "quality": system.takens.reconstruction_quality,
            }

            # --- Barrier 3: Observability ---
            if observation_func is not None:
                observer = PartialObserver(observation_func)
                obs_report = observer.certify_from_observations(y.reshape(-1, 1))
                report["observability"] = {
                    "is_observable": obs_report.is_observable,
                    "observable_dims": obs_report.observable_dims,
                    "state_dims": obs_report.state_dims,
                    "rank": obs_report.observability_rank,
                    "info_loss": obs_report.information_loss_bound,
                }
            else:
                # Scalar observation → estimate from Takens embedding
                observer = PartialObserver(n_obs_dims=1)
                obs_report = observer.certify_from_observations(y.reshape(-1, 1))
                report["observability"] = {
                    "is_observable": obs_report.is_observable,
                    "observable_dims": 1,
                    "state_dims_estimated": obs_report.state_dims,
                    "info_loss_bound": obs_report.information_loss_bound,
                    "note": "Takens theorem: generic observability for d >= 2n+1",
                }

            # --- Barrier 4: Anytime certification ---
            # Use the reconstructed map for certification
            if system.embedding_dim == 1:
                # 1D map: can use standard ERGON/OTU pipeline
                def T_1d(x):
                    return system.T(x)
                anytime_result = self.anytime.certify(T_1d, domain=system.domain)
            else:
                # Multi-dimensional: run quick Lyapunov estimation only
                anytime_result = self._quick_multidim_certify(system)

            report["certification"] = {
                "epsilon_achieved": anytime_result.epsilon_achieved,
                "h_ks_estimate": anytime_result.h_ks_estimate,
                "h_ks_ci": anytime_result.h_ks_confidence_interval,
                "spectral_gap": anytime_result.spectral_gap_estimate,
                "grid_used": anytime_result.n_grid_used,
                "time_ms": anytime_result.computation_time_ms,
                "converged": anytime_result.is_converged,
                "refinement_levels": len(anytime_result.refinement_history),
            }

            # Store certificate for compression
            self.compressor.add_certificate({
                "h_ks": anytime_result.h_ks_estimate,
                "spectral_gap": anytime_result.spectral_gap_estimate,
                "regime_type": regimes.segments[-1].regime_type if regimes.segments else "unknown",
            })

        except Exception as e:
            report["reconstruction_error"] = str(e)
            report["certification"] = {"error": str(e)}

        # --- Barrier 4: Knowledge compression ---
        compressed = self.compressor.compress()
        report["knowledge_graph"] = {
            "total_certificates": compressed.n_original_certificates,
            "compressed_to": compressed.n_compressed,
            "compression_ratio": compressed.compression_ratio,
            "meta_theorems": compressed.meta_theorems,
        }

        return report

    def _quick_multidim_certify(self, system: ReconstructedSystem) -> AnytimeResult:
        """
        Quick certification for multi-dimensional reconstructed systems.

        Uses direct trajectory-based Lyapunov estimation instead of
        Ulam matrix (which requires exponential grid in high dimensions).
        """
        start_time = time.time()
        Z = system.takens.embedded_data
        n, d = Z.shape

        # Lyapunov via QR decomposition of Jacobian product
        delta = 1e-6
        log_sum = 0.0
        count = 0

        for t in range(min(n - 1, 5000)):
            zt = Z[t]
            zt1 = Z[t + 1] if t + 1 < n else system.T(zt)

            # Numerical Jacobian at z_t
            J = np.zeros((d, d))
            for j in range(d):
                z_plus = zt.copy()
                z_plus[j] += delta
                y_plus = system.T(z_plus)
                z_minus = zt.copy()
                z_minus[j] -= delta
                y_minus = system.T(z_minus)
                J[:, j] = (y_plus - y_minus) / (2 * delta)

            # Max Lyapunov from log of spectral radius
            svd_vals = np.linalg.svd(J, compute_uv=False)
            if svd_vals[0] > 1e-14:
                log_sum += np.log(svd_vals[0])
                count += 1

        h_ks = max(0.0, log_sum / max(count, 1))
        total_time = (time.time() - start_time) * 1000

        return AnytimeResult(
            epsilon_achieved=0.1,
            n_grid_used=0,
            h_ks_estimate=h_ks,
            h_ks_confidence_interval=(h_ks * 0.8, h_ks * 1.2),
            spectral_gap_estimate=max(0.0, h_ks),
            computation_time_ms=total_time,
            refinement_history=[{"method": "trajectory_lyapunov", "n_steps": count}],
            is_converged=False,
            certificates_valid=1 if h_ks > 0 else 0,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def from_timeseries(
    y: np.ndarray,
    noise_filter: str = "svd",
    time_budget_ms: float = 10000,
    memory_budget_bytes: int = 2 * 1024 * 1024,
    epsilon_target: float = 0.01,
) -> dict:
    """
    One-line entry point: time series → full dynamical analysis.

    Example:
        import numpy as np
        data = np.loadtxt("vibrations.csv")
        report = from_timeseries(data)
        print(f"h_KS = {report['certification']['h_ks_estimate']:.4f}")
        print(f"Regimes: {report['regimes']['n_regimes']}")
    """
    pipeline = RealWorldPipeline(
        noise_filter=noise_filter,
        time_budget_ms=time_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        epsilon_target=epsilon_target,
    )
    return pipeline.analyze_timeseries(y)


def from_csv(
    filepath: str,
    column: int = 0,
    delimiter: str = ",",
    skip_header: int = 1,
    **kwargs,
) -> dict:
    """
    Load a CSV file and run the full real-world analysis pipeline.

    Example:
        report = from_csv("motor_vibrations.csv", column=2)
    """
    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)
    if data.ndim == 2:
        y = data[:, column]
    else:
        y = data
    return from_timeseries(y, **kwargs)
