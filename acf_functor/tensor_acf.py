"""
Tensor ACF — Multivariable Function Reduction via Tensor Decomposition
=======================================================================

Extends the Affine Collapse Functor to operate on functions f: ℝᵈ → ℝ.

Mathematical foundation
-----------------------
A function f: [-1,1]^d → ℝ can be approximated by a tensor product of
univariate Chebyshev expansions:

    f(x₁, …, x_d) ≈ Σ_{k₁,…,k_d} C_{k₁…k_d} T_{k₁}(x₁) ⋯ T_{k_d}(x_d)

The full tensor has O(n^d) entries (curse of dimensionality). Three
decomposition strategies break this curse:

1. TENSOR TRAIN (TT / MPS):
   C_{k₁…k_d} = A₁[k₁] A₂[k₂] ⋯ A_d[k_d]
   where Aₘ[kₘ] ∈ ℝ^{r_{m-1} × r_m}, total storage O(d n r²).

2. TUCKER:
   C = G ×₁ U₁ ×₂ U₂ ⋯ ×_d U_d
   where G ∈ ℝ^{r₁×⋯×r_d} is the core tensor and Uₘ ∈ ℝ^{n×r_m}.

3. CP (Canonical Polyadic):
   C_{k₁…k_d} = Σ_{j=1}^R λ_j u₁ⱼ[k₁] ⋯ u_dⱼ[k_d]
   Rank-R outer product; simplest but least flexible.

The ACF invariant α for multivariable functions is:
    α(f) = max_m α(σ_m)
where σ_m are the singular values of the mode-m unfolding of C.

FMA chain for TT evaluation (the key algorithm):
  For each x = (x₁, …, x_d):
    v₀ = 1 (scalar)
    For m = 1, …, d:
      t_m = Cheb(x_m)  ∈ ℝⁿ  — evaluate Chebyshev basis at x_m
      v_m = (v_{m-1} ⊗ I) · A_m · t_m  ∈ ℝ^{r_m}
    result = v_d ∈ ℝ (since r_d = 1)

  Each step is a matrix-vector FMA. Total cost: O(d · n · r²) FMAs.
  This is the tensor train "zipper" contraction — fully composable with
  the standard ACF FMA pipeline.

Scope (honest)
--------------
- Dimensions d ≤ 20 for TT (practical); d ≤ 6 for Tucker (core tensor explodes).
- TT-ranks auto-determined by singular value truncation at ε_target.
- Does NOT perform adaptive cross-approximation (TT-cross) — uses
  full grid sampling for d ≤ 6, random sampling for d > 6.

References
----------
  Oseledets (2011) — Tensor-Train Decomposition, SIAM J. Sci. Comput.
  De Lathauwer et al. (2000) — Tucker decomposition.
  Paper.md §33–§35 for the parent ACF framework.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .core import (
    ChebyshevReducer,
    HornerReducer,
    ReductionPath,
    ReductionResult,
    ACFInvariant,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TensorTrainCore:
    """Single TT-core A_m[k_m] of shape (r_{m-1}, r_m) for each k_m."""
    cores: torch.Tensor  # shape (n_m, r_{m-1}, r_m)
    mode: int
    n_basis: int
    rank_left: int
    rank_right: int


@dataclass
class TensorTrainDecomposition:
    """Full TT decomposition of a d-dimensional Chebyshev tensor."""
    cores: List[torch.Tensor]   # cores[m] shape (r_{m-1}, n_m, r_m)
    ranks: List[int]            # TT-ranks [r_0=1, r_1, ..., r_{d-1}, r_d=1]
    degrees: List[int]          # Chebyshev degree per dimension
    domains: List[Tuple[float, float]]
    compression_ratio: float
    total_params: int

    @property
    def ndim(self) -> int:
        return len(self.cores)


@dataclass
class TensorACFInvariants:
    """ACF invariants for a multivariable function."""
    alpha_per_mode: List[float]       # α for each tensor mode
    alpha_global: float               # max(alpha_per_mode)
    tt_ranks: List[int]               # TT-ranks
    nc_class: str                     # NC complexity class
    effective_dimension: float        # sum of (1 - α_m) — measures true dimensionality
    spectral_entropy: float           # entropy of singular value distribution
    total_fma_count: int              # FMAs to evaluate the TT


@dataclass
class TensorReductionResult:
    """Result of reducing f: ℝᵈ → ℝ to a TT-FMA chain."""
    tt: TensorTrainDecomposition
    invariants: TensorACFInvariants
    epsilon: float                    # certification error L∞
    elapsed_ms: float
    cheb_coeffs_per_mode: List[torch.Tensor]  # Chebyshev coefficients per dim
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate TT at points x of shape (batch, d) or (d,)."""
        return TensorTrainEvaluator.evaluate(self.tt, x)


@dataclass
class TuckerDecomposition:
    """Tucker decomposition: core G + factor matrices U_m."""
    core: torch.Tensor                # shape (r_1, ..., r_d)
    factors: List[torch.Tensor]       # factors[m] shape (n_m, r_m)
    ranks: List[int]
    degrees: List[int]
    domains: List[Tuple[float, float]]
    compression_ratio: float
    total_params: int


@dataclass
class TuckerReductionResult:
    """Result of Tucker-based reduction."""
    tucker: TuckerDecomposition
    epsilon: float
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return TuckerEvaluator.evaluate(self.tucker, x)


# ---------------------------------------------------------------------------
# Chebyshev tensor sampling
# ---------------------------------------------------------------------------

class ChebyshevTensorSampler:
    """Sample f on a Chebyshev tensor grid and build coefficient tensor."""

    @staticmethod
    def chebyshev_nodes(n: int, domain: Tuple[float, float],
                        dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Chebyshev nodes of the second kind on [a, b]."""
        a, b = domain
        k = torch.arange(n, dtype=dtype)
        t = torch.cos((k + 0.5) * math.pi / n)
        return 0.5 * (a + b) + 0.5 * (b - a) * t

    @staticmethod
    def chebyshev_coefficients_1d(
        values: torch.Tensor,  # shape (n,) — function values at Chebyshev nodes
    ) -> torch.Tensor:
        """Recover Chebyshev coefficients from values at Chebyshev nodes via DCT."""
        n = values.shape[0]
        dtype = values.dtype
        k = torch.arange(n, dtype=dtype)
        theta = (k + 0.5) * math.pi / n
        j = torch.arange(n, dtype=dtype).unsqueeze(1)
        cos_table = torch.cos(j * theta.unsqueeze(0))  # (n, n)
        coeffs = (2.0 / n) * (cos_table @ values)
        coeffs[0] *= 0.5
        return coeffs

    @staticmethod
    def build_tensor_grid(
        func: Callable,
        degrees: List[int],
        domains: List[Tuple[float, float]],
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Sample f on full Chebyshev tensor grid. Returns values tensor."""
        d = len(degrees)
        nodes_per_dim = []
        for m in range(d):
            nodes = ChebyshevTensorSampler.chebyshev_nodes(degrees[m], domains[m], dtype)
            nodes_per_dim.append(nodes)

        # Build meshgrid
        grids = torch.meshgrid(*nodes_per_dim, indexing='ij')
        # Stack into (n1*n2*...*nd, d)
        points = torch.stack([g.reshape(-1) for g in grids], dim=1)

        # Evaluate function
        values = torch.zeros(points.shape[0], dtype=dtype)
        for i in range(points.shape[0]):
            values[i] = float(func(*points[i].tolist()))

        return values.reshape([degrees[m] for m in range(d)])

    @staticmethod
    def tensor_to_chebyshev_coeffs(
        values_tensor: torch.Tensor,
        degrees: List[int],
    ) -> torch.Tensor:
        """Convert sampled values tensor to Chebyshev coefficient tensor
        by applying 1D DCT along each mode sequentially."""
        C = values_tensor.clone()
        d = len(degrees)
        for m in range(d):
            n = degrees[m]
            # Move mode m to last axis, apply DCT, move back
            C = C.transpose(m, -1)
            shape = C.shape
            C_flat = C.reshape(-1, n)
            for row in range(C_flat.shape[0]):
                C_flat[row] = ChebyshevTensorSampler.chebyshev_coefficients_1d(C_flat[row])
            C = C_flat.reshape(shape)
            C = C.transpose(m, -1)
        return C


# ---------------------------------------------------------------------------
# TT Decomposition (TT-SVD algorithm)
# ---------------------------------------------------------------------------

class TensorTrainBuilder:
    """Build TT decomposition from a full Chebyshev coefficient tensor."""

    @staticmethod
    def tt_svd(
        tensor: torch.Tensor,
        max_rank: int = 50,
        epsilon: float = 1e-10,
    ) -> TensorTrainDecomposition:
        """TT-SVD (Oseledets 2011): decompose full tensor into TT format.

        Truncation uses relative Frobenius norm: keep singular values
        until cumulative truncation < epsilon * ||tensor||_F / sqrt(d-1).
        """
        d = tensor.dim()
        shape = list(tensor.shape)
        delta = epsilon * torch.norm(tensor).item() / math.sqrt(max(d - 1, 1))

        cores = []
        ranks = [1]
        C = tensor.clone()

        for k in range(d - 1):
            nk = shape[k]
            rk = ranks[-1]
            # Reshape: (rk * nk) x (remaining)
            C = C.reshape(rk * nk, -1)
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)

            # Truncate
            cumsum = torch.cumsum(S ** 2, dim=0)
            total = cumsum[-1].item()
            # Keep r singular values such that dropped energy < delta^2
            r = 1
            for j in range(S.shape[0]):
                if total - cumsum[j].item() < delta ** 2:
                    r = j + 1
                    break
            else:
                r = S.shape[0]
            r = min(r, max_rank)

            core = U[:, :r].reshape(rk, nk, r)
            cores.append(core)
            ranks.append(r)
            C = torch.diag(S[:r]) @ Vh[:r]

        # Last core
        cores.append(C.reshape(ranks[-1], shape[-1], 1))
        ranks.append(1)

        total_params = sum(c.numel() for c in cores)
        full_params = 1
        for s in shape:
            full_params *= s
        compression = full_params / max(total_params, 1)

        return TensorTrainDecomposition(
            cores=cores,
            ranks=ranks,
            degrees=shape,
            domains=[],  # filled by caller
            compression_ratio=compression,
            total_params=total_params,
        )


# ---------------------------------------------------------------------------
# TT Evaluation (the FMA zipper)
# ---------------------------------------------------------------------------

class TensorTrainEvaluator:
    """Evaluate a TT at given points using the FMA-zipper contraction."""

    @staticmethod
    def evaluate(tt: TensorTrainDecomposition, x: torch.Tensor) -> torch.Tensor:
        """Evaluate TT at points x.

        Args:
            tt: TensorTrainDecomposition
            x: shape (batch, d) or (d,). Each column is one spatial dimension.

        Returns:
            shape (batch,) or scalar.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        d = tt.ndim
        dtype = tt.cores[0].dtype

        # Build Chebyshev basis values at each point for each mode
        cheb_basis = []
        for m in range(d):
            n_m = tt.degrees[m]
            a, b = tt.domains[m]
            xm = x[:, m]
            t_m = (2.0 * xm - (a + b)) / (b - a)  # map to [-1, 1]
            # Evaluate T_0(t), T_1(t), ..., T_{n-1}(t) via recurrence
            T = torch.zeros(batch, n_m, dtype=dtype)
            T[:, 0] = 1.0
            if n_m > 1:
                T[:, 1] = t_m
            for k in range(2, n_m):
                T[:, k] = 2.0 * t_m * T[:, k - 1] - T[:, k - 2]
            cheb_basis.append(T)  # (batch, n_m)

        # TT zipper contraction: v_0 = [1], v_m = sum_k T_k(x_m) * core_m[:, k, :]^T v_{m-1}
        v = torch.ones(batch, 1, dtype=dtype)  # (batch, r_0=1)
        for m in range(d):
            core = tt.cores[m]  # (r_{m-1}, n_m, r_m)
            T_m = cheb_basis[m]  # (batch, n_m)
            r_left, n_m, r_right = core.shape

            # Contract: for each batch element, v_new = v @ (sum_k T_k * core[:, k, :])
            # core[:, k, :] is (r_left, r_right)
            # Efficient: core reshaped to (r_left, n_m * r_right), then
            # T_m (batch, n_m) used to weight and sum
            # W = sum_k T_m[b, k] * core[:, k, :] — shape (batch, r_left, r_right)
            W = torch.einsum('bk, lkr -> blr', T_m, core)  # (batch, r_left, r_right)
            # v_new = v @ W — v is (batch, r_left) -> (batch, 1, r_left) @ (batch, r_left, r_right)
            v = torch.bmm(v.unsqueeze(1), W).squeeze(1)  # (batch, r_right)

        result = v.squeeze(-1)
        if result.shape[0] == 1:
            return result.squeeze(0)
        return result


# ---------------------------------------------------------------------------
# Tucker decomposition
# ---------------------------------------------------------------------------

class TuckerBuilder:
    """Build Tucker decomposition via HOSVD."""

    @staticmethod
    def hosvd(
        tensor: torch.Tensor,
        max_ranks: Optional[List[int]] = None,
        epsilon: float = 1e-10,
    ) -> TuckerDecomposition:
        """Higher-Order SVD: compute Tucker factors + core."""
        d = tensor.dim()
        shape = list(tensor.shape)
        if max_ranks is None:
            max_ranks = shape  # no truncation

        factors = []
        ranks = []
        core = tensor.clone()

        for m in range(d):
            # Mode-m unfolding
            unf = tensor.transpose(m, 0).reshape(shape[m], -1)
            U, S, _ = torch.linalg.svd(unf, full_matrices=False)
            # Truncate
            r = min(max_ranks[m], U.shape[1])
            if epsilon > 0:
                fro = torch.norm(tensor).item()
                delta = epsilon * fro / math.sqrt(max(d, 1))
                cumvar = torch.cumsum(S ** 2, dim=0)
                total = cumvar[-1].item()
                r_trunc = 1
                for j in range(S.shape[0]):
                    if total - cumvar[j].item() < delta ** 2:
                        r_trunc = j + 1
                        break
                else:
                    r_trunc = S.shape[0]
                r = min(r, r_trunc)

            U_trunc = U[:, :r]
            factors.append(U_trunc)
            ranks.append(r)

        # Compute core: project tensor onto truncated basis
        core = tensor.clone()
        for m in range(d):
            # n-mode product with U^T
            core = torch.tensordot(core, factors[m].T, dims=([m], [1]))
            # Move the new axis back to position m
            perm = list(range(core.dim()))
            perm.remove(core.dim() - 1)
            perm.insert(m, core.dim() - 1)
            core = core.permute(perm)

        total_params = core.numel() + sum(f.numel() for f in factors)
        full_params = 1
        for s in shape:
            full_params *= s
        compression = full_params / max(total_params, 1)

        return TuckerDecomposition(
            core=core,
            factors=factors,
            ranks=ranks,
            degrees=shape,
            domains=[],
            compression_ratio=compression,
            total_params=total_params,
        )


class TuckerEvaluator:
    """Evaluate Tucker decomposition at given points."""

    @staticmethod
    def evaluate(tucker: TuckerDecomposition, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Tucker at points x of shape (batch, d) or (d,)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        d = len(tucker.factors)
        dtype = tucker.core.dtype

        # Build Chebyshev basis at each point
        projections = []
        for m in range(d):
            n_m = tucker.degrees[m]
            a, b = tucker.domains[m]
            xm = x[:, m]
            t_m = (2.0 * xm - (a + b)) / (b - a)
            T = torch.zeros(batch, n_m, dtype=dtype)
            T[:, 0] = 1.0
            if n_m > 1:
                T[:, 1] = t_m
            for k in range(2, n_m):
                T[:, k] = 2.0 * t_m * T[:, k - 1] - T[:, k - 2]

            # Project onto Tucker factor: T @ U_m → (batch, r_m)
            projections.append(T @ tucker.factors[m])  # (batch, r_m)

        # Contract with core
        # core is (r_1, r_2, ..., r_d)
        # result[b] = sum_{i1,...,id} core[i1,...,id] * proj1[b,i1] * ... * projd[b,id]
        result = torch.zeros(batch, dtype=dtype)
        core = tucker.core
        # Iterative contraction
        val = core
        for m in range(d):
            # Contract mode 0 of val with projections[m]
            # val shape: (r_m, r_{m+1}, ..., r_{d-1}, batch) after first step
            # or (r_m, ..., r_{d-1}) initially
            val = torch.tensordot(val, projections[m].T, dims=([0], [0]))
            # The result has the batch dimension at the end

        # val should be shape (batch,)
        result = val.squeeze()
        if result.dim() == 0:
            return result
        return result


# ---------------------------------------------------------------------------
# Main Reducer
# ---------------------------------------------------------------------------

class TensorACFReducer:
    """Reduce f: ℝᵈ → ℝ to a Tensor Train or Tucker FMA chain."""

    def __init__(
        self,
        degrees: Optional[List[int]] = None,
        default_degree: int = 8,
        max_rank: int = 20,
        target_epsilon: float = 1e-8,
        method: str = "tt",  # "tt" or "tucker"
        dtype: torch.dtype = torch.float64,
    ):
        self.degrees = degrees
        self.default_degree = default_degree
        self.max_rank = max_rank
        self.target_epsilon = target_epsilon
        self.method = method
        self.dtype = dtype

    def reduce(
        self,
        func: Callable,
        domains: List[Tuple[float, float]],
        degrees: Optional[List[int]] = None,
    ) -> TensorReductionResult:
        """Reduce f: ℝᵈ → ℝ to TT or Tucker format.

        Args:
            func: f(x1, x2, ..., xd) → float. Must accept d positional float args.
            domains: list of (a_m, b_m) domains per dimension.
            degrees: Chebyshev degrees per dimension; if None, use default.

        Returns:
            TensorReductionResult with TT decomposition and certified epsilon.
        """
        t0 = time.time()
        d = len(domains)
        if degrees is None:
            degrees = self.degrees or [self.default_degree] * d

        # Step 1: Sample on Chebyshev grid
        values = ChebyshevTensorSampler.build_tensor_grid(func, degrees, domains, self.dtype)

        # Step 2: Convert to Chebyshev coefficient tensor
        C = ChebyshevTensorSampler.tensor_to_chebyshev_coeffs(values, degrees)

        # Step 3: Decompose
        if self.method == "tt":
            tt = TensorTrainBuilder.tt_svd(C, max_rank=self.max_rank, epsilon=self.target_epsilon)
            tt.domains = domains
            decomp = tt
        elif self.method == "tucker":
            tucker = TuckerBuilder.hosvd(C, epsilon=self.target_epsilon)
            tucker.domains = domains
            # Wrap in TT for unified interface — return TuckerReductionResult instead
            elapsed = (time.time() - t0) * 1000
            eps = self._certify(func, tucker, domains)
            return TuckerReductionResult(
                tucker=tucker,
                epsilon=eps,
                elapsed_ms=elapsed,
                metadata={"method": "tucker_hosvd", "d": d, "degrees": degrees},
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Step 4: Certify error
        eps = self._certify_tt(func, tt, domains)

        # Step 5: Compute invariants
        invariants = TensorACFAnalyzer.compute_invariants(tt)

        # Store Chebyshev coefficients per mode for potential re-use
        cheb_per_mode = []
        for m in range(d):
            # Extract mode-m Chebyshev coefficients from the TT core
            core_m = tt.cores[m]  # (r_{m-1}, n_m, r_m)
            # Average over ranks to get representative coefficients
            avg = core_m.mean(dim=(0, 2))  # (n_m,)
            cheb_per_mode.append(avg)

        elapsed = (time.time() - t0) * 1000
        return TensorReductionResult(
            tt=tt,
            invariants=invariants,
            epsilon=eps,
            elapsed_ms=elapsed,
            cheb_coeffs_per_mode=cheb_per_mode,
            metadata={
                "method": "tt_svd",
                "d": d,
                "degrees": degrees,
                "max_rank": self.max_rank,
                "target_epsilon": self.target_epsilon,
                "compression_ratio": tt.compression_ratio,
            },
        )

    def _certify_tt(
        self,
        func: Callable,
        tt: TensorTrainDecomposition,
        domains: List[Tuple[float, float]],
        n_test: int = 2000,
    ) -> float:
        """Certify TT approximation error on random test points."""
        d = len(domains)
        rng = np.random.default_rng(42)
        test_points = torch.zeros(n_test, d, dtype=self.dtype)
        for m in range(d):
            a, b = domains[m]
            test_points[:, m] = torch.from_numpy(
                rng.uniform(a, b, size=n_test)
            ).to(self.dtype)

        exact = torch.zeros(n_test, dtype=self.dtype)
        for i in range(n_test):
            exact[i] = float(func(*test_points[i].tolist()))

        approx = TensorTrainEvaluator.evaluate(tt, test_points)
        return float(torch.max(torch.abs(exact - approx)).item())

    def _certify(
        self,
        func: Callable,
        tucker: TuckerDecomposition,
        domains: List[Tuple[float, float]],
        n_test: int = 2000,
    ) -> float:
        d = len(domains)
        rng = np.random.default_rng(42)
        test_points = torch.zeros(n_test, d, dtype=self.dtype)
        for m in range(d):
            a, b = domains[m]
            test_points[:, m] = torch.from_numpy(
                rng.uniform(a, b, size=n_test)
            ).to(self.dtype)

        exact = torch.zeros(n_test, dtype=self.dtype)
        for i in range(n_test):
            exact[i] = float(func(*test_points[i].tolist()))

        approx = TuckerEvaluator.evaluate(tucker, test_points)
        return float(torch.max(torch.abs(exact - approx)).item())


# ---------------------------------------------------------------------------
# Invariant analyzer
# ---------------------------------------------------------------------------

class TensorACFAnalyzer:
    """Compute ACF invariants for tensor decompositions."""

    @staticmethod
    def compute_invariants(tt: TensorTrainDecomposition) -> TensorACFInvariants:
        """Compute per-mode and global α, NC class, effective dimension."""
        d = tt.ndim
        alpha_per_mode = []
        entropies = []

        for m in range(d):
            core = tt.cores[m]  # (r_{m-1}, n_m, r_m)
            r_left, n_m, r_right = core.shape
            # Mode-m unfolding: reshape to (n_m, r_left * r_right)
            unf = core.permute(1, 0, 2).reshape(n_m, r_left * r_right)
            S = torch.linalg.svdvals(unf)
            S = S[S > 1e-15]
            if S.numel() == 0:
                alpha_per_mode.append(1.0)
                entropies.append(0.0)
                continue

            # Compute alpha from singular value decay
            alpha = TensorACFAnalyzer._alpha_from_sv(S)
            alpha_per_mode.append(alpha)

            # Spectral entropy
            p = S / S.sum()
            ent = -float(torch.sum(p * torch.log(p + 1e-30)).item())
            entropies.append(ent)

        alpha_global = max(alpha_per_mode) if alpha_per_mode else 1.0

        # NC class from global alpha
        if alpha_global < 0.2:
            nc = "NC0"
        elif alpha_global < 0.5:
            nc = "NC1"
        elif alpha_global < 0.8:
            nc = "NC2"
        else:
            nc = "NC3"

        # Effective dimension: sum of (1 - alpha_m)
        eff_dim = sum(1.0 - a for a in alpha_per_mode)

        # Total FMA count
        total_fma = 0
        for m in range(d):
            r_left, n_m, r_right = tt.cores[m].shape
            total_fma += r_left * n_m * r_right  # matrix-vector multiply cost

        spectral_entropy = sum(entropies) / max(len(entropies), 1)

        return TensorACFInvariants(
            alpha_per_mode=alpha_per_mode,
            alpha_global=alpha_global,
            tt_ranks=tt.ranks,
            nc_class=nc,
            effective_dimension=eff_dim,
            spectral_entropy=spectral_entropy,
            total_fma_count=total_fma,
        )

    @staticmethod
    def _alpha_from_sv(S: torch.Tensor) -> float:
        """Estimate α from singular value decay log|σ_j| vs log(j)."""
        S_sorted = torch.sort(S, descending=True).values
        n = S_sorted.numel()
        if n < 2:
            return 0.0

        log_j = torch.log(torch.arange(1, n + 1, dtype=S.dtype))
        log_s = torch.log(S_sorted + 1e-30)

        # Linear regression: log_s ≈ -alpha * log_j + c
        x = log_j
        y = log_s
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        alpha = float(-slope.item())
        return max(0.0, min(1.0, alpha))


# ---------------------------------------------------------------------------
# Standard test functions
# ---------------------------------------------------------------------------

class StandardTensorFunctions:
    """Collection of standard multivariate test functions."""

    @staticmethod
    def separable_product(x1: float, x2: float, x3: float = 0.0) -> float:
        """f(x,y,z) = sin(x) * cos(y) * exp(z). Exactly rank-1 in CP."""
        return math.sin(x1) * math.cos(x2) * math.exp(x3)

    @staticmethod
    def rosenbrock(x1: float, x2: float) -> float:
        """f(x,y) = (1-x)² + 100(y-x²)². Classic non-separable."""
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    @staticmethod
    def gaussian_2d(x1: float, x2: float) -> float:
        """f(x,y) = exp(-(x² + y²))."""
        return math.exp(-(x1 ** 2 + x2 ** 2))

    @staticmethod
    def multivariate_polynomial(x1: float, x2: float, x3: float) -> float:
        """f(x,y,z) = x²y + yz² + xz + 3."""
        return x1 ** 2 * x2 + x2 * x3 ** 2 + x1 * x3 + 3.0

    @staticmethod
    def friedman1(x1: float, x2: float, x3: float, x4: float, x5: float) -> float:
        """Friedman #1: f = 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5."""
        return (10 * math.sin(math.pi * x1 * x2)
                + 20 * (x3 - 0.5) ** 2
                + 10 * x4
                + 5 * x5)

    @staticmethod
    def wave_3d(x1: float, x2: float, x3: float) -> float:
        """f(x,y,z) = sin(x+y)*cos(z) + 0.5*sin(2x)*sin(3z)."""
        return (math.sin(x1 + x2) * math.cos(x3)
                + 0.5 * math.sin(2 * x1) * math.sin(3 * x3))
