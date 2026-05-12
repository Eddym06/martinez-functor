"""
ProgramAnalyzer — Programs as Dynamical Systems
================================================

Treats program execution traces as trajectories of a dynamical system
and applies the full ACF diagnostic stack to discover structure.

THEORETICAL FOUNDATION
----------------------

A program P: X → Y maps input x to output y through a sequence of
intermediate states s₀, s₁, ..., s_T where:
  s₀ = encode(x)
  s_{t+1} = F_t(s_t)       (state transition at step t)
  y = decode(s_T)

This defines a DISCRETE DYNAMICAL SYSTEM with:
  - State space: ℝ^d (the flattened intermediate values)
  - Transition map: F_t (may be time-varying or autonomous)
  - Trajectory: {s_t}_{t=0}^T

ACF can then analyze this dynamical system:
  1. TAA (Koopman): Find the linear representation of F in lifted space
  2. ERGON: Compute Lyapunov exponents, entropy, ergodic properties
  3. OTU (Gelfand): Characterize the functional space structure
  4. SINDy: Discover sparse governing equations of the state dynamics

PROGRAM REGIONS (via OTU/ERGON diagnostics):
  - ANALYTIC: Smooth, well-approximated by Chebyshev → polynomial replacement
  - STRATIFIED: Piecewise structure → LUT or conditional FMA
  - CHAOTIC: Sensitive dependence → Koopman ROM (reduced order model)
  - DISSIPATIVE: Contracting dynamics → fixed-point iteration shortcut
  - PERIODIC: Repeating patterns → spectral (Fourier) shortcut

COMPUTATIONAL ENERGY:
  E(P) = Σ_{t=0}^{T-1} cost(F_t)
  where cost(F_t) counts FMA operations at step t.

  Goal: Find P' with E(P') < E(P) such that |P(x) - P'(x)| < ε on domain D.

CERTIFICATES:
  META-1: Trace captured with ≤ ε instrumentation overhead
  META-2: Region classification with confidence > 0.95
  META-3: Computational energy reduction > 0%
  META-4: Equivalence |P(x) - P'(x)| < ε on certified domain
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RegionKind(Enum):
    """Classification of a compute graph region."""
    ANALYTIC = "analytic"           # Smooth → Chebyshev/Horner
    STRATIFIED = "stratified"       # Piecewise → LUT/conditional
    CHAOTIC = "chaotic"             # Sensitive → Koopman ROM
    DISSIPATIVE = "dissipative"     # Contracting → fixed-point shortcut
    PERIODIC = "periodic"           # Repeating → Fourier shortcut
    LINEAR = "linear"               # Already linear → identity or GEMM
    UNKNOWN = "unknown"


class OptimizationStrategy(Enum):
    """What to do with a classified region."""
    CHEBYSHEV_REPLACE = "chebyshev_replace"
    HORNER_CHAIN = "horner_chain"
    LOOKUP_TABLE = "lookup_table"
    KOOPMAN_ROM = "koopman_rom"
    FIXED_POINT_SKIP = "fixed_point_skip"
    FOURIER_SHORTCUT = "fourier_shortcut"
    GEMM_FOLD = "gemm_fold"
    IDENTITY = "identity"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TracePoint:
    """Single state observation in a program trace."""
    step: int
    state: np.ndarray           # Flattened intermediate values
    operation: str              # Name/kind of operation
    fma_cost: int               # FMA count for this step
    wall_time_us: float = 0.0   # Microseconds


@dataclass
class ExecutionTrace:
    """Complete execution trace of a program on one input."""
    input_hash: str
    points: List[TracePoint]
    total_fma: int
    total_time_us: float
    input_value: Optional[np.ndarray] = None
    output_value: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return len(self.points)

    @property
    def state_dim(self) -> int:
        return self.points[0].state.shape[0] if self.points else 0

    def to_trajectory(self) -> np.ndarray:
        """Convert trace to (n_steps, state_dim) trajectory matrix."""
        return np.array([p.state for p in self.points])

    def fma_profile(self) -> np.ndarray:
        """FMA cost per step."""
        return np.array([p.fma_cost for p in self.points])


@dataclass
class RegionClassification:
    """Classification of a program region."""
    region_id: int
    kind: RegionKind
    confidence: float               # 0-1
    step_range: Tuple[int, int]     # (start, end) inclusive
    strategy: OptimizationStrategy
    estimated_speedup: float        # >1 means faster
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramProfile:
    """Complete analysis profile of a program."""
    n_traces: int
    state_dim: int
    total_steps: int
    total_fma: int
    regions: List[RegionClassification]
    lyapunov_exponents: np.ndarray
    spectral_entropy: float
    spectral_decay_rate: float      # α(P) — the ACF spectral index
    koopman_eigenvalues: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def estimated_total_speedup(self) -> float:
        """Harmonic mean of per-region speedups weighted by FMA cost."""
        if not self.regions:
            return 1.0
        total_orig = 0.0
        total_opt = 0.0
        for r in self.regions:
            span = r.step_range[1] - r.step_range[0] + 1
            total_orig += span
            total_opt += span / max(r.estimated_speedup, 0.01)
        return total_orig / max(total_opt, 1e-15)


# ---------------------------------------------------------------------------
# Program Tracer
# ---------------------------------------------------------------------------

class ProgramTracer:
    """
    Instrument and trace program execution.

    Captures the state trajectory of a program by hooking into
    intermediate computations and recording state snapshots.
    """

    def __init__(self, state_extractor: Optional[Callable] = None):
        """
        Parameters
        ----------
        state_extractor : callable, optional
            f(step, locals_dict) → np.ndarray that extracts state
            from the program's locals at each step.
        """
        self.state_extractor = state_extractor

    def trace_function(
        self,
        func: Callable,
        inputs: List[np.ndarray],
        dt: float = 1.0,
    ) -> List[ExecutionTrace]:
        """
        Trace a function by evaluating it on multiple inputs and
        recording input-output pairs as a "trajectory" in I/O space.

        For functions f: ℝ^n → ℝ^m, the trace captures the I/O manifold
        which ACF can then analyze for structure.
        """
        traces = []
        for idx, x in enumerate(inputs):
            t0 = time.perf_counter()
            x_flat = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()

            # Evaluate
            y = func(x)
            y_flat = np.atleast_1d(np.asarray(y, dtype=np.float64)).ravel()
            elapsed_us = (time.perf_counter() - t0) * 1e6

            # State = concatenation of input and output
            state = np.concatenate([x_flat, y_flat])
            n_fma = max(1, len(y_flat))  # Estimate: at least 1 FMA per output

            trace = ExecutionTrace(
                input_hash=f"input_{idx:06d}",
                points=[TracePoint(
                    step=0,
                    state=state,
                    operation="evaluate",
                    fma_cost=n_fma,
                    wall_time_us=elapsed_us,
                )],
                total_fma=n_fma,
                total_time_us=elapsed_us,
                input_value=x_flat,
                output_value=y_flat,
            )
            traces.append(trace)
        return traces

    def trace_iterative(
        self,
        step_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        n_steps: int,
        fma_per_step: int = 1,
    ) -> ExecutionTrace:
        """
        Trace an iterative program: x_{t+1} = step_func(x_t).

        This is the natural model for loops, solvers, optimizers, etc.
        """
        x = np.atleast_1d(np.asarray(x0, dtype=np.float64)).ravel()
        points = []
        total_time = 0.0

        for step in range(n_steps):
            t0 = time.perf_counter()
            x_new = np.atleast_1d(step_func(x)).ravel()
            elapsed = (time.perf_counter() - t0) * 1e6

            points.append(TracePoint(
                step=step,
                state=x.copy(),
                operation="step",
                fma_cost=fma_per_step,
                wall_time_us=elapsed,
            ))
            total_time += elapsed
            x = x_new

        # Final state
        points.append(TracePoint(
            step=n_steps,
            state=x.copy(),
            operation="final",
            fma_cost=0,
        ))

        return ExecutionTrace(
            input_hash=f"iter_{hash(x0.tobytes()) % 10**8:08d}",
            points=points,
            total_fma=fma_per_step * n_steps,
            total_time_us=total_time,
            input_value=x0.copy(),
            output_value=x.copy(),
        )


# ---------------------------------------------------------------------------
# Program Analyzer
# ---------------------------------------------------------------------------

class ProgramAnalyzer:
    """
    Analyze program traces as dynamical systems using the ACF diagnostic stack.

    Applies TAA (Koopman), ERGON (Lyapunov/entropy), OTU (spectral) diagnostics
    to classify program regions and recommend optimization strategies.
    """

    def __init__(
        self,
        lyapunov_window: int = 50,
        spectral_threshold: float = 0.1,
        periodicity_threshold: float = 0.95,
        min_region_length: int = 5,
    ):
        self.lyapunov_window = lyapunov_window
        self.spectral_threshold = spectral_threshold
        self.periodicity_threshold = periodicity_threshold
        self.min_region_length = min_region_length

    def analyze(
        self,
        traces: List[ExecutionTrace],
    ) -> ProgramProfile:
        """
        Full program analysis from execution traces.

        1. Build state trajectory from traces
        2. Compute Koopman decomposition (linearized dynamics)
        3. Compute Lyapunov exponents (chaos detection)
        4. Compute spectral properties (periodicity, decay)
        5. Classify regions (analytic, stratified, chaotic, etc.)
        """
        # Build trajectory from traces
        if len(traces) == 1 and traces[0].n_steps > 1:
            # Iterative program: use temporal trajectory
            traj = traces[0].to_trajectory()
            fma_profile = traces[0].fma_profile()
        else:
            # Function evaluation: use I/O manifold sampling
            traj = np.array([t.points[0].state for t in traces])
            fma_profile = np.array([t.total_fma for t in traces])

        n_steps, d = traj.shape

        # Koopman analysis (linearized dynamics)
        K, eigenvalues, eigenvectors = self._koopman_dmd(traj)

        # Lyapunov exponents
        lyapunov = self._compute_lyapunov(traj)

        # Spectral properties
        spectral_entropy = self._spectral_entropy(eigenvalues)
        decay_rate = self._spectral_decay_rate(traj)

        # Region classification
        regions = self._classify_regions(
            traj, lyapunov, eigenvalues, fma_profile,
        )

        return ProgramProfile(
            n_traces=len(traces),
            state_dim=d,
            total_steps=n_steps,
            total_fma=int(np.sum(fma_profile)),
            regions=regions,
            lyapunov_exponents=lyapunov,
            spectral_entropy=spectral_entropy,
            spectral_decay_rate=decay_rate,
            koopman_eigenvalues=eigenvalues,
            transition_matrix=K,
        )

    def _koopman_dmd(
        self,
        traj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dynamic Mode Decomposition (DMD) for Koopman approximation.

        Given trajectory X = [x₀, x₁, ..., x_{T-1}]:
          X' = [x₁, x₂, ..., x_T]
          K ≈ X' X⁺  (Koopman matrix via pseudoinverse)
        """
        if traj.shape[0] < 3:
            d = traj.shape[1]
            return np.eye(d), np.ones(d), np.eye(d)

        X = traj[:-1].T    # (d, T-1)
        Y = traj[1:].T     # (d, T-1)

        # SVD-based DMD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # Truncate small singular values
        r = max(1, np.sum(S > 1e-10 * S[0]))
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        # Koopman approximation in reduced space
        K_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / S_r)

        # Eigendecomposition
        eigenvalues, W = np.linalg.eig(K_tilde)
        eigenvectors = U_r @ W

        # Full-space Koopman
        K = U_r @ K_tilde @ U_r.T

        return K, eigenvalues, eigenvectors

    def _compute_lyapunov(self, traj: np.ndarray) -> np.ndarray:
        """
        Estimate Lyapunov exponents from trajectory via consecutive difference ratios.

        For a trajectory x₀, x₁, ..., x_T:
          λ ≈ (1/T) Σ log(|δx_{t+1}| / |δx_t|)
        where δx_t = x_{t+1} - x_t.

        This correctly estimates log|f'(x_t)| via the chain rule for contracting
        (dissipative) and expanding (chaotic) systems alike, avoiding the bias
        introduced by rank-1 Jacobian approximations.
        """
        n_steps, d = traj.shape
        if n_steps < 4:
            return np.zeros(d)

        diffs = np.diff(traj, axis=0)   # shape (n_steps-1, d): δx_t = x_{t+1} - x_t
        lyapunov = np.zeros(d)
        count = 0

        window = min(self.lyapunov_window, n_steps - 2)

        for t in range(window):
            dx_curr = diffs[t]        # δx_t
            dx_next = diffs[t + 1]    # δx_{t+1}

            norm_curr = np.linalg.norm(dx_curr)
            norm_next = np.linalg.norm(dx_next)

            if norm_curr < 1e-15:
                continue

            # log-ratio: estimates log|f'(x_t)| via finite-difference chain rule
            ratio = (norm_next + 1e-300) / (norm_curr + 1e-300)
            lyapunov += np.log(ratio)
            count += 1

        if count > 0:
            lyapunov /= count

        return lyapunov

    def _spectral_entropy(self, eigenvalues: np.ndarray) -> float:
        """
        Spectral entropy of Koopman eigenvalues.
        H = -Σ p_k log(p_k) where p_k = |λ_k|² / Σ|λ_j|²
        """
        magnitudes = np.abs(eigenvalues) ** 2
        total = np.sum(magnitudes)
        if total < 1e-15:
            return 0.0
        p = magnitudes / total
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)))

    def _spectral_decay_rate(self, traj: np.ndarray) -> float:
        """
        Estimate the ACF spectral decay index α(P).

        Uses SVD of the trajectory matrix:
          α = -slope of log(σ_k) vs log(k)
        Higher α → faster decay → more compressible → better ACF reduction.
        """
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
        S = S[S > 1e-15]
        if len(S) < 2:
            return float('inf')

        k = np.arange(1, len(S) + 1, dtype=float)
        log_k = np.log(k)
        log_S = np.log(S)

        # Linear regression: log(S) = -α * log(k) + b
        A = np.vstack([log_k, np.ones_like(log_k)]).T
        result = np.linalg.lstsq(A, log_S, rcond=None)
        slope = result[0][0]

        return float(-slope)  # α > 0 for decaying spectra

    def _classify_regions(
        self,
        traj: np.ndarray,
        lyapunov: np.ndarray,
        eigenvalues: np.ndarray,
        fma_profile: np.ndarray,
    ) -> List[RegionClassification]:
        """
        Classify program regions based on dynamical diagnostics.

        Decision tree:
          1. max(λ_Lyap) > 0 → CHAOTIC
          2. max(λ_Lyap) < -threshold → DISSIPATIVE
          3. |λ_Koopman| ≈ 1 and arg(λ) ≈ rational × 2π → PERIODIC
          4. α(P) > 2 → ANALYTIC (fast spectral decay)
          5. Large jumps in trajectory → STRATIFIED
          6. Otherwise → LINEAR
        """
        n_steps = traj.shape[0]
        regions = []

        # Global diagnostics
        max_lyap = float(np.max(lyapunov)) if len(lyapunov) > 0 else 0.0
        min_lyap = float(np.min(lyapunov)) if len(lyapunov) > 0 else 0.0

        # Eigenvalue analysis
        eig_magnitudes = np.abs(eigenvalues)
        eig_phases = np.angle(eigenvalues)

        # Check for periodicity
        unit_circle = np.abs(eig_magnitudes - 1.0) < 0.05
        n_periodic = np.sum(unit_circle)

        # Check for jumps (stratification)
        if n_steps > 2:
            diffs = np.diff(traj, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)
            median_diff = np.median(diff_norms)
            jumps = diff_norms > 5 * (median_diff + 1e-10)
            n_jumps = np.sum(jumps)
        else:
            n_jumps = 0
            jumps = np.zeros(0, dtype=bool)

        # Spectral decay
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
        S_norm = S / (S[0] + 1e-15)
        effective_rank = np.sum(S_norm > 0.01)

        # Classify the whole program as one region (or split at jumps)
        if n_jumps > 0 and n_steps > 2 * self.min_region_length:
            # Split at jumps
            jump_indices = np.where(jumps)[0]
            boundaries = [0] + list(jump_indices + 1) + [n_steps]

            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1] - 1
                if end - start < self.min_region_length:
                    continue

                sub_traj = traj[start:end+1]
                kind, conf, strat, speedup = self._classify_single_region(
                    sub_traj, lyapunov, eigenvalues,
                )
                regions.append(RegionClassification(
                    region_id=len(regions),
                    kind=kind,
                    confidence=conf,
                    step_range=(start, end),
                    strategy=strat,
                    estimated_speedup=speedup,
                ))
        else:
            # Single region
            kind, conf, strat, speedup = self._classify_single_region(
                traj, lyapunov, eigenvalues,
            )
            regions.append(RegionClassification(
                region_id=0,
                kind=kind,
                confidence=conf,
                step_range=(0, n_steps - 1),
                strategy=strat,
                estimated_speedup=speedup,
            ))

        return regions

    def _classify_single_region(
        self,
        traj: np.ndarray,
        lyapunov: np.ndarray,
        eigenvalues: np.ndarray,
    ) -> Tuple[RegionKind, float, OptimizationStrategy, float]:
        """Classify a single contiguous region."""
        max_lyap = float(np.max(lyapunov)) if len(lyapunov) > 0 else 0.0
        min_lyap = float(np.min(lyapunov)) if len(lyapunov) > 0 else 0.0

        eig_mags = np.abs(eigenvalues)
        n_periodic = np.sum(np.abs(eig_mags - 1.0) < 0.05) if len(eigenvalues) > 0 else 0

        # Spectral decay
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
        S_norm = S / (S[0] + 1e-15)
        effective_rank = int(np.sum(S_norm > 0.01))
        total_dim = traj.shape[1]

        if max_lyap > self.spectral_threshold:
            # Positive Lyapunov → chaotic
            return (
                RegionKind.CHAOTIC,
                min(0.99, 0.7 + 0.3 * max_lyap),
                OptimizationStrategy.KOOPMAN_ROM,
                1.0 + effective_rank / max(total_dim, 1),
            )

        if min_lyap < -1.0:
            # Strongly negative → dissipative (contracting)
            return (
                RegionKind.DISSIPATIVE,
                min(0.99, 0.7 + 0.1 * abs(min_lyap)),
                OptimizationStrategy.FIXED_POINT_SKIP,
                2.0 + abs(min_lyap),
            )

        if n_periodic > total_dim * 0.5:
            # Many eigenvalues near unit circle → periodic
            return (
                RegionKind.PERIODIC,
                min(0.99, 0.6 + 0.4 * n_periodic / max(total_dim, 1)),
                OptimizationStrategy.FOURIER_SHORTCUT,
                1.5,
            )

        # Check spectral decay (analytic vs linear)
        if len(S) > 1:
            k = np.arange(1, len(S) + 1, dtype=float)
            log_k = np.log(k)
            log_S = np.log(S + 1e-15)
            A = np.vstack([log_k, np.ones_like(log_k)]).T
            coeff = np.linalg.lstsq(A, log_S, rcond=None)[0]
            alpha = -coeff[0]
        else:
            alpha = 0.0

        if alpha > 2.0:
            # Fast decay → analytic, great for Chebyshev
            return (
                RegionKind.ANALYTIC,
                min(0.99, 0.5 + 0.1 * alpha),
                OptimizationStrategy.CHEBYSHEV_REPLACE,
                1.5 + alpha * 0.3,
            )

        if effective_rank <= max(2, total_dim // 3):
            # Low rank → linear structure
            return (
                RegionKind.LINEAR,
                0.8,
                OptimizationStrategy.GEMM_FOLD,
                1.0 + (total_dim - effective_rank) / max(total_dim, 1),
            )

        # Default: analytic with moderate speedup
        return (
            RegionKind.ANALYTIC,
            0.6,
            OptimizationStrategy.HORNER_CHAIN,
            1.2,
        )
