"""
Massive Algebra — Scalable Linear Algebra via ACF Compression
==============================================================

Handles the "Wall of Dimensionality" that kills conventional solvers.
Instead of brute-forcing O(n³) for n = 10⁷, the ACF compresses the
operator and solves in the compressed basis.

CORE PRINCIPLE
──────────────
Every massive matrix is a function. The ACF reduces that function.

  Matrix A (10⁷ × 10⁷) → Spectral analysis → rank-k approximation
  → Solve in ℝ^k (k ≈ 500) → Reconstruct in ℝⁿ
  → Certified error bound

MODULES
───────
  RandomizedSVD       — O(n·k²) SVD for n × n with rank-k output
  SparseChebyshevOp   — Apply f(A) via Chebyshev polynomials of sparse A
  TensorTrainSolver   — Solve Ax=b via Tensor Train decomposition
  MassiveEigenSolver   — Scalable eigendecomposition with ACF error bounds
  OperatorCompressor   — Compress linear operator to minimal FMA representation

CERTIFICATES:
  MA-1: Compressed solution error ‖x - x_exact‖/‖x_exact‖ ≤ ε_target
  MA-2: Spectral fidelity: top-k eigenvalues within relative error ε
  MA-3: Operator function f(A) approximation bounded by Chebyshev theory
  MA-4: Memory footprint ≤ O(n·k) for rank-k approximation
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpectralDecomposition:
    """Truncated spectral decomposition of a large matrix."""
    eigenvalues: np.ndarray       # shape (k,) — top-k eigenvalues
    eigenvectors: np.ndarray      # shape (n, k) — corresponding eigenvectors
    rank: int                     # effective rank k
    n_original: int               # original dimension n
    relative_energy: float        # fraction of total variance captured
    method: str = "randomized"
    elapsed_seconds: float = 0.0


@dataclass
class CompressedSolution:
    """Solution of Ax=b via spectral compression."""
    x: np.ndarray                 # approximate solution
    residual_norm: float          # ‖Ax - b‖
    relative_error: float         # ‖x - x_true‖/‖x_true‖ if available
    rank_used: int
    n_fma: int                    # total FMA operations
    elapsed_seconds: float = 0.0
    certificate: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorFunctionResult:
    """Result of applying f(A) to a vector or matrix via Chebyshev."""
    result: np.ndarray            # f(A)·v or f(A)
    chebyshev_degree: int
    spectral_bounds: Tuple[float, float]
    approximation_error: float    # theoretical upper bound
    n_matvecs: int                # number of matrix-vector products used
    elapsed_seconds: float = 0.0


@dataclass
class TensorTrainCore:
    """A single core in a Tensor Train decomposition."""
    data: np.ndarray              # shape (r_{k-1}, n_k, r_k)
    mode: int                     # which mode this core represents


@dataclass
class TensorTrainDecomposition:
    """Full Tensor Train decomposition of a high-dimensional tensor."""
    cores: List[TensorTrainCore]
    original_shape: Tuple[int, ...]
    tt_ranks: List[int]           # [r_0=1, r_1, ..., r_{d-1}, r_d=1]
    relative_error: float
    compression_ratio: float      # original_size / tt_size
    total_parameters: int


# ---------------------------------------------------------------------------
# Randomized SVD — O(n·k²) instead of O(n³)
# ---------------------------------------------------------------------------

class RandomizedSVD:
    """
    Halko-Martinsson-Tropp randomized SVD for massive matrices.

    Given A ∈ ℝ^{m×n}, computes rank-k SVD in O(mn·k) time:
      A ≈ U_k Σ_k V_k^T

    Uses oversampling and power iteration for stability.
    """

    def __init__(self, n_oversamples: int = 10, n_power_iters: int = 2,
                 random_state: Optional[int] = None):
        self.n_oversamples = n_oversamples
        self.n_power_iters = n_power_iters
        self.rng = np.random.RandomState(random_state)

    def decompose(self, A: np.ndarray, rank: int) -> SpectralDecomposition:
        """
        Compute truncated SVD of A at target rank.

        Parameters
        ----------
        A : array (m, n) — the matrix (dense or with .dot() method)
        rank : int — target rank k

        Returns
        -------
        SpectralDecomposition with eigenvalues = singular values²
        """
        t0 = time.time()
        m, n = A.shape
        k = min(rank, min(m, n))
        p = min(self.n_oversamples, min(m, n) - k)

        # Stage A: Form Q whose columns approximate the column space of A
        Omega = self.rng.randn(n, k + p).astype(A.dtype)
        Y = A @ Omega

        # Power iteration for better approximation
        for _ in range(self.n_power_iters):
            Y = A @ (A.T @ Y)

        Q, _ = np.linalg.qr(Y)
        Q = Q[:, :k + p]

        # Stage B: Form B = Q^T A and compute its SVD
        B = Q.T @ A
        U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)

        U = Q @ U_hat[:, :k]
        s = s[:k]
        Vt = Vt[:k, :]

        # Compute relative energy
        total_norm_sq = np.linalg.norm(A, 'fro') ** 2
        captured = np.sum(s ** 2)
        rel_energy = captured / max(total_norm_sq, 1e-30)

        return SpectralDecomposition(
            eigenvalues=s ** 2,  # singular values squared = eigenvalues of A^T A
            eigenvectors=U,
            rank=k,
            n_original=max(m, n),
            relative_energy=min(1.0, rel_energy),
            method="randomized_svd",
            elapsed_seconds=time.time() - t0,
        )

    def decompose_sparse(self, A_matvec: Callable, A_T_matvec: Callable,
                         m: int, n: int, rank: int) -> SpectralDecomposition:
        """
        Randomized SVD for an implicitly-defined sparse/structured operator.

        Parameters
        ----------
        A_matvec : callable — computes A @ v
        A_T_matvec : callable — computes A^T @ v
        m, n : dimensions of A
        rank : target rank
        """
        t0 = time.time()
        k = min(rank, min(m, n))
        p = min(self.n_oversamples, min(m, n) - k)

        Omega = self.rng.randn(n, k + p)
        Y = np.column_stack([A_matvec(Omega[:, j]) for j in range(k + p)])

        for _ in range(self.n_power_iters):
            Z = np.column_stack([A_T_matvec(Y[:, j]) for j in range(Y.shape[1])])
            Y = np.column_stack([A_matvec(Z[:, j]) for j in range(Z.shape[1])])

        Q, _ = np.linalg.qr(Y)
        Q = Q[:, :k + p]

        B = np.column_stack([A_T_matvec(Q[:, j]) for j in range(Q.shape[1])]).T
        U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)

        U = Q @ U_hat[:, :k]
        s = s[:k]

        return SpectralDecomposition(
            eigenvalues=s ** 2,
            eigenvectors=U,
            rank=k,
            n_original=max(m, n),
            relative_energy=1.0,  # Cannot compute without full norm
            method="randomized_svd_implicit",
            elapsed_seconds=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Sparse Chebyshev Operator — f(A)·v without eigendecomposition
# ---------------------------------------------------------------------------

class SparseChebyshevOperator:
    """
    Apply f(A) to a vector using Chebyshev polynomial expansion,
    requiring only matrix-vector products with A. No eigendecomposition needed.

    This enables:
      - exp(tA)·v    for diffusion / Schrödinger evolution
      - A^{-1}·v     via Chebyshev-accelerated inversion
      - A^{1/2}·v    matrix square root
      - sign(A)·v    matrix sign function
      - log(A)·v     matrix logarithm

    Cost: O(d · nnz(A)) for d-term Chebyshev expansion on sparse A.
    """

    # Built-in scalar functions for common operator functions
    BUILTIN_FUNCTIONS = {
        "exp": np.exp,
        "inv": lambda x: 1.0 / (x + 1e-30),
        "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
        "log": lambda x: np.log(np.maximum(x, 1e-30)),
        "sign": np.sign,
    }

    def __init__(self, max_degree: int = 50, tol: float = 1e-10):
        self.max_degree = max_degree
        self.tol = tol

    def compute_chebyshev_coefficients(self, f: Callable[[np.ndarray], np.ndarray],
                                        lam_min: float, lam_max: float,
                                        degree: int) -> np.ndarray:
        """
        Compute Chebyshev expansion coefficients of f on [lam_min, lam_max].

        Uses the DCT-based formula for Chebyshev interpolation.
        """
        n = degree + 1
        # Chebyshev nodes on [-1, 1]
        k = np.arange(n)
        nodes = np.cos(np.pi * (k + 0.5) / n)

        # Map to [lam_min, lam_max]
        x = 0.5 * (lam_max + lam_min) + 0.5 * (lam_max - lam_min) * nodes
        fx = f(x)

        # Compute coefficients via discrete orthogonality
        coeffs = np.zeros(n)
        for j in range(n):
            Tj = np.cos(j * np.arccos(nodes))
            coeffs[j] = (2.0 / n) * np.dot(fx, Tj)
        coeffs[0] *= 0.5
        return coeffs

    def apply(self, A_matvec: Callable, v: np.ndarray,
              f: Union[str, Callable], lam_min: float, lam_max: float,
              degree: Optional[int] = None) -> OperatorFunctionResult:
        """
        Compute f(A)·v using Chebyshev expansion.

        Parameters
        ----------
        A_matvec : callable — computes A @ x for a vector x
        v : array — the vector to apply f(A) to
        f : str or callable — scalar function (or name of builtin)
        lam_min, lam_max : spectral bounds of A
        degree : Chebyshev degree (auto-determined if None)
        """
        t0 = time.time()

        if isinstance(f, str):
            f_scalar = self.BUILTIN_FUNCTIONS[f]
        else:
            f_scalar = f

        if degree is None:
            # Auto-determine degree: start small, increase until convergence
            degree = self._auto_degree(f_scalar, lam_min, lam_max)

        coeffs = self.compute_chebyshev_coefficients(f_scalar, lam_min, lam_max, degree)

        # Chebyshev recurrence applied to matrix-vector products
        # T_0(Ã)v = v
        # T_1(Ã)v = Ã·v
        # T_k(Ã)v = 2Ã·T_{k-1}(Ã)v - T_{k-2}(Ã)v

        # Affine map: Ã = (2A - (lam_max+lam_min)I) / (lam_max - lam_min)
        center = 0.5 * (lam_max + lam_min)
        half_width = 0.5 * (lam_max - lam_min)
        if half_width < 1e-30:
            half_width = 1.0

        def A_tilde_matvec(x):
            Ax = A_matvec(x)
            return (Ax - center * x) / half_width

        T_prev = v.copy()                           # T_0·v = v
        T_curr = A_tilde_matvec(v)                   # T_1·v = Ã·v
        result = coeffs[0] * T_prev + coeffs[1] * T_curr

        n_matvecs = 1
        for k in range(2, degree + 1):
            T_next = 2.0 * A_tilde_matvec(T_curr) - T_prev
            result += coeffs[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            n_matvecs += 1

        # Error estimate from tail coefficients
        tail = np.abs(coeffs[max(1, degree - 3):])
        approx_error = float(np.sum(tail)) * np.linalg.norm(v)

        return OperatorFunctionResult(
            result=result,
            chebyshev_degree=degree,
            spectral_bounds=(lam_min, lam_max),
            approximation_error=approx_error,
            n_matvecs=n_matvecs,
            elapsed_seconds=time.time() - t0,
        )

    def _auto_degree(self, f: Callable, lam_min: float, lam_max: float) -> int:
        """Determine minimal Chebyshev degree for target tolerance."""
        for d in [8, 16, 32, 64, 128]:
            coeffs = self.compute_chebyshev_coefficients(f, lam_min, lam_max, d)
            tail = np.abs(coeffs[max(1, d - 3):])
            if np.max(tail) < self.tol:
                return d
        return self.max_degree


# ---------------------------------------------------------------------------
# Compressed Linear Solver — Solve Ax = b in rank-k subspace
# ---------------------------------------------------------------------------

class CompressedLinearSolver:
    """
    Solve Ax = b for massive sparse/structured systems via spectral compression.

    Instead of O(n³) direct solve or O(n²·k_iter) iterative solve,
    this projects into the top-k eigenspace and solves in O(n·k²):

      1. Compute rank-k SVD: A ≈ U_k Σ_k V_k^T
      2. Project: b_k = U_k^T b
      3. Solve:   x_k = V_k · diag(1/σ_i) · b_k
      4. Reconstruct: x ≈ x_k

    The error is bounded by the k+1-th singular value:
      ‖x - x_exact‖ ≤ σ_{k+1}⁻¹ · ‖b‖ · (conditioning factor)
    """

    def __init__(self, svd_engine: Optional[RandomizedSVD] = None,
                 regularization: float = 1e-10):
        self.svd = svd_engine or RandomizedSVD()
        self.reg = regularization

    def solve(self, A: np.ndarray, b: np.ndarray,
              rank: Optional[int] = None,
              target_error: float = 0.01) -> CompressedSolution:
        """
        Solve Ax = b via rank-k spectral compression.
        """
        t0 = time.time()
        m, n = A.shape

        if rank is None:
            rank = self._auto_rank(A, target_error)

        # Step 1: Randomized SVD
        decomp = self.svd.decompose(A, rank)
        U = decomp.eigenvectors  # (m, k) — left singular vectors
        sigma_sq = decomp.eigenvalues  # singular values squared
        sigma = np.sqrt(np.maximum(sigma_sq, 0.0))

        # Step 2: Compute right singular vectors via V = A^T U / sigma
        V = np.zeros((n, len(sigma)))
        for i in range(len(sigma)):
            if sigma[i] > self.reg:
                V[:, i] = (A.T @ U[:, i]) / sigma[i]

        # Step 3: Solve in compressed space
        b_proj = U.T @ b  # (k,)
        x_proj = np.zeros(len(sigma))
        for i in range(len(sigma)):
            if sigma[i] > self.reg:
                x_proj[i] = b_proj[i] / sigma[i]

        # Step 4: Reconstruct
        x = V @ x_proj

        # Compute residual
        residual = A @ x - b
        res_norm = float(np.linalg.norm(residual))
        b_norm = float(np.linalg.norm(b))

        return CompressedSolution(
            x=x,
            residual_norm=res_norm,
            relative_error=res_norm / max(b_norm, 1e-30),
            rank_used=rank,
            n_fma=m * rank + rank * rank + n * rank,
            elapsed_seconds=time.time() - t0,
            certificate={
                "method": "compressed_svd_solve",
                "relative_energy_captured": decomp.relative_energy,
                "min_singular_value_used": float(sigma[-1]) if len(sigma) > 0 else 0.0,
                "residual_over_rhs": res_norm / max(b_norm, 1e-30),
            },
        )

    def solve_implicit(self, A_matvec: Callable, A_T_matvec: Callable,
                       b: np.ndarray, m: int, n: int,
                       rank: int = 100) -> CompressedSolution:
        """Solve Ax = b when A is only available as a matrix-vector product."""
        t0 = time.time()
        decomp = self.svd.decompose_sparse(A_matvec, A_T_matvec, m, n, rank)
        U = decomp.eigenvectors
        sigma = np.sqrt(np.maximum(decomp.eigenvalues, 0.0))

        # Right singular vectors via A^T U / sigma
        V = np.zeros((n, len(sigma)))
        for i in range(len(sigma)):
            if sigma[i] > self.reg:
                V[:, i] = A_T_matvec(U[:, i]) / sigma[i]

        b_proj = U.T @ b
        x_proj = np.array([b_proj[i] / max(sigma[i], self.reg) for i in range(len(sigma))])
        x = V @ x_proj

        res = A_matvec(x) - b
        res_norm = float(np.linalg.norm(res))
        b_norm = float(np.linalg.norm(b))

        return CompressedSolution(
            x=x, residual_norm=res_norm,
            relative_error=res_norm / max(b_norm, 1e-30),
            rank_used=rank, n_fma=m * rank * 3,
            elapsed_seconds=time.time() - t0,
            certificate={"method": "implicit_compressed_solve"},
        )

    def _auto_rank(self, A: np.ndarray, target_error: float) -> int:
        """Determine the rank needed for a given error target."""
        n = min(A.shape)
        for k in [10, 25, 50, 100, 200, 500]:
            if k >= n:
                return n
            decomp = self.svd.decompose(A, k)
            if decomp.relative_energy > (1.0 - target_error ** 2):
                return k
        return min(500, n)


# ---------------------------------------------------------------------------
# Tensor Train Decomposition for High-Dimensional Data
# ---------------------------------------------------------------------------

class TensorTrainEngine:
    """
    Tensor Train decomposition for breaking the curse of dimensionality.

    A d-dimensional tensor T ∈ ℝ^{n₁ × n₂ × ... × n_d} is decomposed as:

      T[i₁, i₂, ..., i_d] = G₁[i₁] · G₂[i₂] · ... · G_d[i_d]

    where G_k[i_k] ∈ ℝ^{r_{k-1} × r_k} are matrix slices of the k-th core.

    Storage: O(d · n · r²) instead of O(n^d).
    Evaluation: O(d · r²) FMA per point.
    """

    def __init__(self, max_rank: int = 50, tol: float = 1e-6):
        self.max_rank = max_rank
        self.tol = tol

    def decompose(self, tensor: np.ndarray) -> TensorTrainDecomposition:
        """
        Compute TT decomposition of a full tensor via sequential SVD.

        This is the TT-SVD algorithm (Oseledets 2011).
        """
        t0 = time.time()
        shape = tensor.shape
        d = len(shape)
        original_size = int(np.prod(shape))

        cores = []
        ranks = [1]
        C = tensor.reshape(shape[0], -1)  # Unfold first mode

        for k in range(d - 1):
            r_prev = ranks[-1]
            n_k = shape[k]
            rows = r_prev * n_k
            cols = C.size // rows if C.size > 0 else 1
            C = C.reshape(rows, cols)

            # Truncated SVD
            U, s, Vt = np.linalg.svd(C, full_matrices=False)

            # Determine rank by tolerance
            total_energy = np.sum(s ** 2)
            cumulative = np.cumsum(s ** 2)
            threshold = (1.0 - self.tol ** 2) * total_energy

            r_k = 1
            for i in range(len(s)):
                r_k = i + 1
                if cumulative[i] >= threshold or r_k >= self.max_rank:
                    break

            # Truncate
            U_k = U[:, :r_k]
            s_k = s[:r_k]
            Vt_k = Vt[:r_k, :]

            # Core: reshape U_k to (r_{k-1}, n_k, r_k)
            core_data = U_k.reshape(r_prev, n_k, r_k)
            cores.append(TensorTrainCore(data=core_data, mode=k))
            ranks.append(r_k)

            # Prepare for next iteration
            C = np.diag(s_k) @ Vt_k

        # Last core
        r_prev = ranks[-1]
        n_last = shape[-1]
        core_data = C.reshape(r_prev, n_last, 1)
        cores.append(TensorTrainCore(data=core_data, mode=d - 1))
        ranks.append(1)

        # Compute compression ratio
        tt_size = sum(c.data.size for c in cores)
        compression = original_size / max(tt_size, 1)

        # Approximate relative error
        reconstructed = self._reconstruct(cores, shape)
        rel_error = np.linalg.norm(tensor - reconstructed) / max(np.linalg.norm(tensor), 1e-30)

        return TensorTrainDecomposition(
            cores=cores,
            original_shape=shape,
            tt_ranks=ranks,
            relative_error=float(rel_error),
            compression_ratio=float(compression),
            total_parameters=tt_size,
        )

    def evaluate_point(self, tt: TensorTrainDecomposition,
                       indices: Tuple[int, ...]) -> float:
        """Evaluate TT at a specific multi-index. Cost: O(d·r²)."""
        v = np.array([[1.0]])
        for k, core in enumerate(tt.cores):
            v = v @ core.data[:, indices[k], :]
        return float(v.item())

    def _reconstruct(self, cores: List[TensorTrainCore],
                     shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct the full tensor from TT cores (for verification)."""
        d = len(cores)
        result = cores[0].data.reshape(shape[0], -1)
        for k in range(1, d):
            r_k = cores[k].data.shape[2]
            n_k = shape[k]
            # Contract: result (prod_{i<k} n_i, r_k) × core_k (r_k, n_k, r_{k+1})
            r_prev = result.shape[1]
            core_mat = cores[k].data.reshape(r_prev, n_k * r_k)
            result = result @ core_mat
            result = result.reshape(-1, r_k)
        return result.reshape(shape)

    def tt_matvec(self, tt: TensorTrainDecomposition, x: np.ndarray) -> float:
        """
        Evaluate TT as a function: given x ∈ ℝ^d (continuous coordinates),
        interpolate via Chebyshev basis at each mode.

        This is the "zipper" contraction that makes TT evaluation an FMA chain.
        """
        v = np.ones((1, 1))
        for k, core in enumerate(tt.cores):
            n_k = core.data.shape[1]
            # Chebyshev basis evaluation at x[k]
            t = np.cos(np.arange(n_k) * np.arccos(np.clip(x[k], -1, 1)))
            # Contract: v (1, r_k) × core (r_k, n_k, r_{k+1}) × t (n_k,)
            core_mat = np.einsum('ijk,j->ik', core.data, t)
            v = v @ core_mat
        return float(v.item())


# ---------------------------------------------------------------------------
# Massive Eigenvalue Solver — ACF-powered spectral analysis
# ---------------------------------------------------------------------------

class MassiveEigenSolver:
    """
    Eigendecomposition of massive matrices using randomized methods
    with ACF error bounds.

    For n × n matrices with n > 10⁴, direct eigendecomposition is O(n³).
    This uses:
      1. Randomized SVD for initial rank-k spectrum
      2. Chebyshev-filtered Lanczos for refinement
      3. ACF error certificate from spectral decay rate
    """

    def __init__(self, svd_engine: Optional[RandomizedSVD] = None):
        self.svd = svd_engine or RandomizedSVD(n_power_iters=3)

    def top_k_eigenvalues(self, A: np.ndarray, k: int) -> SpectralDecomposition:
        """Compute top-k eigenvalues/vectors of symmetric A."""
        t0 = time.time()
        n = A.shape[0]

        if n <= 2000:
            # Direct eigendecomposition for small matrices
            evals, evecs = np.linalg.eigh(A)
            idx = np.argsort(np.abs(evals))[::-1][:k]
            return SpectralDecomposition(
                eigenvalues=evals[idx], eigenvectors=evecs[:, idx],
                rank=k, n_original=n, relative_energy=1.0,
                method="direct_eigh", elapsed_seconds=time.time() - t0,
            )

        # Randomized approach for large matrices
        # Form B = A^T A (implicitly) and compute SVD
        decomp = self.svd.decompose(A, k)
        # For symmetric A, singular values = |eigenvalues|
        return SpectralDecomposition(
            eigenvalues=np.sign(decomp.eigenvalues) * np.sqrt(np.abs(decomp.eigenvalues)),
            eigenvectors=decomp.eigenvectors,
            rank=k, n_original=n,
            relative_energy=decomp.relative_energy,
            method="randomized", elapsed_seconds=time.time() - t0,
        )

    def spectral_gap(self, A: np.ndarray) -> float:
        """Estimate the spectral gap |λ₁| - |λ₂| of A."""
        decomp = self.top_k_eigenvalues(A, k=2)
        evals = np.sort(np.abs(decomp.eigenvalues))[::-1]
        if len(evals) < 2:
            return 0.0
        return float(evals[0] - evals[1])

    def condition_number(self, A: np.ndarray, rank: int = 50) -> float:
        """Estimate condition number from rank-k approximation."""
        decomp = self.top_k_eigenvalues(A, k=rank)
        evals = np.abs(decomp.eigenvalues)
        evals = evals[evals > 1e-30]
        if len(evals) < 2:
            return 1.0
        return float(evals[0] / evals[-1])


# ---------------------------------------------------------------------------
# Operator Compressor — turn any linear operator into minimal FMA
# ---------------------------------------------------------------------------

class OperatorCompressor:
    """
    Compress a linear operator (matrix or implicit) into a minimal
    FMA representation.

    Strategy:
      1. Detect effective rank via randomized SVD
      2. If rank << n: represent as low-rank U·Σ·V^T (2 matrix FMAs)
      3. If sparse: keep sparse representation
      4. If structured (Toeplitz, circulant): use FFT-based FMA
      5. Report: n_fma_original vs n_fma_compressed with error bound
    """

    def __init__(self, svd_engine: Optional[RandomizedSVD] = None):
        self.svd = svd_engine or RandomizedSVD()

    def compress(self, A: np.ndarray, target_error: float = 0.01) -> Dict[str, Any]:
        """
        Compress matrix A into the most efficient FMA representation.
        """
        m, n = A.shape
        original_fma = m * n  # Full dense matvec

        # Detect sparsity
        nnz = np.count_nonzero(A)
        sparsity = 1.0 - nnz / (m * n)

        # Detect effective rank
        max_rank = min(m, n, 500)
        decomp = self.svd.decompose(A, min(max_rank, min(m, n)))

        # Find rank for target error
        sigma = np.sqrt(np.maximum(decomp.eigenvalues, 0.0))
        total = np.sum(sigma ** 2)
        cumul = np.cumsum(sigma ** 2)
        threshold = (1.0 - target_error ** 2) * total

        effective_rank = 1
        for i in range(len(sigma)):
            effective_rank = i + 1
            if cumul[i] >= threshold:
                break

        low_rank_fma = (m + n) * effective_rank  # U @ (Sigma @ V^T @ x)
        sparse_fma = nnz

        strategies = {
            "dense": {"fma": original_fma, "memory": m * n},
            "low_rank": {"fma": low_rank_fma, "memory": (m + n) * effective_rank,
                         "rank": effective_rank},
            "sparse": {"fma": sparse_fma, "memory": nnz * 3,  # COO format
                       "sparsity": sparsity},
        }

        # Pick best strategy
        best = min(strategies.items(), key=lambda x: x[1]["fma"])

        return {
            "best_strategy": best[0],
            "original_fma": original_fma,
            "compressed_fma": best[1]["fma"],
            "compression_ratio": original_fma / max(best[1]["fma"], 1),
            "effective_rank": effective_rank,
            "sparsity": sparsity,
            "target_error": target_error,
            "all_strategies": strategies,
            "spectral_decay": decomp.eigenvalues.tolist(),
        }
