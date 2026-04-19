"""
Operator/Green Function ACF — Integral Operators and Attention Kernels
======================================================================

Extends the Affine Collapse Functor to integral operators, Green functions,
and transformer attention kernels.

Mathematical foundation
-----------------------
An integral operator L acts on functions u: Ω → ℝ as:
    (Lu)(x) = ∫_Ω G(x, y) u(y) dy

where G: Ω×Ω → ℝ is the kernel (Green function).

ACF reduction strategy:
  1. Represent G as a bivariate function G: [-1,1]² → ℝ (d=2 TensorACF).
  2. Reduce G via TT-decomposition: G(x,y) ≈ Σ_k aₖ(x) bₖ(y)
     (as a finite sum of separable terms).
  3. The reduced operator then decomposes as:
     (L̂u)(x) = Σ_k aₖ(x) · ∫ bₖ(y) u(y) dy
     Each integral ∫bₖ(y)u(y)dy is a cheap inner product.

For the transformer attention kernel K(q,k) = exp(q·k/√d)/Z(q):
  We compress K as a function of the relative token embedding q-k via
  polynomial approximation on the sphere.

Alpha invariant for Operator ACF
---------------------------------
  α_kernel(G) = α(G_unfolded) — ACF alpha of the 2D kernel matrix
  α_operator(L) = α_kernel(G)

Scope
-----
  - 1D Green functions G: [-1,1]² → ℝ
  - 2D separable kernels (can be extended via Tucker)
  - Transformer attention compression (linear attention approximation)

References
----------
  Evans (2010) — Partial Differential Equations, Ch. 2 (Green functions).
  Katharopoulos et al. (2020) — Transformers are RNNs (linear attention).
  Paper.md §39 for ACF extension to operator domains.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .tensor_acf import TensorACFReducer


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class KernelACFInvariants:
    """Invariant summary for a 2D kernel reduced via Operator-ACF."""
    alpha_kernel: float             # ACF alpha of the kernel matrix
    n_separable_terms: int          # effective rank of separable decomposition
    operator_norm_error: float      # ‖L - L̂‖_op estimate (via sampling)
    compression_ratio: float        # original grid points / reduced params
    dimension: int                  # ambient space dimension

    def summary(self) -> str:
        lines = [
            "=== Kernel-ACF Invariants ===",
            f"  dim: {self.dimension}",
            f"  α_kernel: {self.alpha_kernel:.4f}",
            f"  Separable rank: {self.n_separable_terms}",
            f"  ‖L - L̂‖ estimate: {self.operator_norm_error:.4e}",
            f"  Compression ratio: {self.compression_ratio:.1f}x",
        ]
        return "\n".join(lines)


@dataclass
class AttentionKernelInvariants:
    """Invariant summary for a compressed attention kernel."""
    alpha_attention: float
    n_features: int                 # number of feature maps in linear approx
    approx_error: float             # ‖K - K̂‖_F on test set
    speedup_factor: float           # O(n²) → O(nk) speedup estimate

    def summary(self) -> str:
        lines = [
            "=== Attention-ACF Invariants ===",
            f"  α_attention: {self.alpha_attention:.4f}",
            f"  Feature maps: {self.n_features}",
            f"  Approx error: {self.approx_error:.4e}",
            f"  Speedup estimate: {self.speedup_factor:.1f}x",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GreenFunctionReducer
# ---------------------------------------------------------------------------

class GreenFunctionReducer:
    """
    Reduce a Green function (2D kernel) G: Ω×Ω → ℝ via TT-ACF.

    Parameters
    ----------
    n_points : int
        Number of quadrature points per dimension.
    order : int
        Chebyshev expansion order.
    domain : tuple (a, b)
        1D spatial domain.
    """

    def __init__(
        self,
        n_points: int = 64,
        order: int = 16,
        domain: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.n_points = n_points
        self.order = order
        self.a, self.b = domain
        self._reducer: Optional[TensorACFReducer] = None
        self._xs = np.linspace(self.a, self.b, n_points)
        self._fitted = False

        # Quadrature weights (Simpson or trapezoidal)
        h = (self.b - self.a) / (n_points - 1)
        self._weights = np.ones(n_points) * h
        self._weights[0] *= 0.5
        self._weights[-1] *= 0.5

    def fit(self, G: Callable[[float, float], float]) -> "GreenFunctionReducer":
        """
        Fit a 2D TensorACFReducer to the kernel G(x,y).

        Parameters
        ----------
        G : callable
            Kernel function G(x, y) → ℝ.
        """
        # TensorACFReducer expects func(*positional_floats) -> float
        def G_pos(x: float, y: float) -> float:
            return float(G(x, y))

        domain_2d = [(self.a, self.b), (self.a, self.b)]
        base_reducer = TensorACFReducer(
            default_degree=self.order,
            method="tt",
        )
        self._reducer = base_reducer.reduce(G_pos, domain_2d)
        self._fitted = True
        return self

    def __call__(self, x: float, y: float) -> float:
        """Evaluate the reduced kernel Ĝ(x, y)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        import torch
        tx = torch.tensor([x, y], dtype=torch.float64)
        return float(self._reducer.evaluate(tx))

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        Apply the reduced operator to a function u sampled on the grid.

        (L̂u)(xᵢ) = Σⱼ Ĝ(xᵢ, xⱼ) u(xⱼ) wⱼ

        Parameters
        ----------
        u : ndarray, shape (n_points,)
            Function values u(xⱼ) on the stored quadrature grid.

        Returns
        -------
        Lu : ndarray, shape (n_points,)
            Operator applied to u.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        Lu = np.zeros(self.n_points)
        for i, xi in enumerate(self._xs):
            for j, xj in enumerate(self._xs):
                Lu[i] += self(xi, xj) * u[j] * self._weights[j]
        return Lu

    def invariants(
        self,
        G: Optional[Callable[[float, float], float]] = None,
    ) -> KernelACFInvariants:
        """Compute Kernel-ACF invariants."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        inv = self._reducer.invariants
        alpha = float(inv.alpha_global)

        # Estimate operator norm error if G provided
        op_error = 0.0
        if G is not None:
            errors = []
            xs = self._xs[::4]
            for xi in xs:
                for xj in xs:
                    errors.append(abs(self(xi, xj) - G(xi, xj)))
            op_error = float(np.max(errors)) if errors else 0.0

        n_params = self.order * 2  # coarse estimate for TT rank-1
        compression = (self.n_points ** 2) / max(n_params, 1)

        return KernelACFInvariants(
            alpha_kernel=alpha,
            n_separable_terms=self.order,
            operator_norm_error=op_error,
            compression_ratio=compression,
            dimension=1,
        )


# ---------------------------------------------------------------------------
# IntegralOperatorACF
# ---------------------------------------------------------------------------

class IntegralOperatorACF:
    """
    Full integral operator L with compressed Green function + fast application.

    Separable decomposition:
        G(x,y) ≈ Σ_{k=1}^R aₖ(x) bₖ(y)
    yields:
        (Lu)(x) ≈ Σ_k aₖ(x) · <bₖ, u>_w
    where <bₖ, u>_w = Σⱼ bₖ(yⱼ) u(yⱼ) wⱼ.

    This reduces O(n²) → O(R·n) application cost.

    Parameters
    ----------
    n_points : int
        Quadrature resolution.
    rank : int
        Number of separable terms R (effective TT rank).
    domain : tuple (a, b)
        Spatial domain.
    """

    def __init__(
        self,
        n_points: int = 64,
        rank: int = 16,
        domain: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.n_points = n_points
        self.rank = rank
        self.a, self.b = domain
        self._xs = np.linspace(self.a, self.b, n_points)
        h = (self.b - self.a) / (n_points - 1)
        self._weights = np.ones(n_points) * h
        self._weights[0] *= 0.5
        self._weights[-1] *= 0.5

        self._a_funcs: List[np.ndarray] = []   # shape (n_points,) each
        self._b_funcs: List[np.ndarray] = []   # shape (n_points,) each
        self._fitted = False

    def fit(self, G: Callable[[float, float], float]) -> "IntegralOperatorACF":
        """
        Build SVD-based separable decomposition of G on the grid.

        Parameters
        ----------
        G : callable
            Kernel G(x, y) → ℝ.
        """
        # Build kernel matrix
        K_mat = np.array([[G(xi, xj) for xj in self._xs] for xi in self._xs])

        # Truncated SVD for low-rank approximation
        U, s, Vt = np.linalg.svd(K_mat, full_matrices=False)
        r = min(self.rank, len(s))

        self._a_funcs = [U[:, k] * s[k] for k in range(r)]
        self._b_funcs = [Vt[k, :] for k in range(r)]
        self._fitted = True
        return self

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        Apply the compressed operator to u in O(R·n) time.

        Parameters
        ----------
        u : ndarray, shape (n_points,)
            Input function values on grid.

        Returns
        -------
        Lu : ndarray, shape (n_points,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        Lu = np.zeros(self.n_points)
        for a_k, b_k in zip(self._a_funcs, self._b_funcs):
            inner = float(np.dot(b_k * self._weights, u))
            Lu += a_k * inner
        return Lu

    def alpha(self) -> float:
        """Compute ACF alpha for the singular value decay."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        # Reconstruct singular values
        svs = np.array([np.linalg.norm(a) for a in self._a_funcs])
        svs = svs[svs > 1e-15]
        if len(svs) < 2:
            return 1.0
        ks = np.arange(1, len(svs) + 1, dtype=float)
        log_k = np.log(ks)
        log_s = np.log(svs)
        return float(-np.polyfit(log_k, log_s, 1)[0])

    def invariants(
        self,
        G: Optional[Callable[[float, float], float]] = None,
        test_u: Optional[np.ndarray] = None,
    ) -> KernelACFInvariants:
        """Compute operator invariants."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        alpha = self.alpha()
        op_error = 0.0
        if G is not None and test_u is not None:
            # Compare L̂u vs exact Lu via dense apply
            Lu_exact = np.array([
                sum(G(self._xs[i], self._xs[j]) * test_u[j] * self._weights[j]
                    for j in range(self.n_points))
                for i in range(self.n_points)
            ])
            Lu_approx = self.apply(test_u)
            op_error = float(np.max(np.abs(Lu_exact - Lu_approx)))

        compression = (self.n_points ** 2) / (2 * self.rank * self.n_points)
        return KernelACFInvariants(
            alpha_kernel=alpha,
            n_separable_terms=len(self._a_funcs),
            operator_norm_error=op_error,
            compression_ratio=compression,
            dimension=1,
        )


# ---------------------------------------------------------------------------
# AttentionKernelReducer
# ---------------------------------------------------------------------------

class AttentionKernelReducer:
    """
    Compress a transformer attention kernel K(q,k) = softmax(qᵀk/√d)
    via feature map decomposition (linearized attention).

    Linear attention approximation:
        K(q,k) ≈ φ(q)ᵀ φ(k)
    where φ: ℝᵈ → ℝ^R is a random Fourier feature or polynomial feature map.

    This reduces O(n²d) attention to O(nd·R + nR²) via prefix summing.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension d.
    n_features : int
        Number of random features R.
    feature_type : str
        'random_fourier' | 'polynomial' | 'relu'
    """

    def __init__(
        self,
        embed_dim: int,
        n_features: int = 64,
        feature_type: str = "random_fourier",
    ):
        self.d = embed_dim
        self.R = n_features
        self.feature_type = feature_type
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, seed: int = 42) -> "AttentionKernelReducer":
        """Initialize the feature map parameters."""
        rng = np.random.default_rng(seed)
        if self.feature_type == "random_fourier":
            self._W = rng.normal(0, 1, (self.R, self.d)) / np.sqrt(self.d)
            self._b = rng.uniform(0, 2 * np.pi, self.R)
        elif self.feature_type == "relu":
            self._W = rng.normal(0, 1, (self.R, self.d))
            self._b = np.zeros(self.R)
        else:  # polynomial (degree 2)
            self._W = rng.normal(0, 1, (self.R, self.d))
            self._b = np.zeros(self.R)
        self._fitted = True
        return self

    def phi(self, x: np.ndarray) -> np.ndarray:
        """
        Compute feature map φ(x).

        Parameters
        ----------
        x : ndarray, shape (d,) or (n, d)
            Input token embeddings.

        Returns
        -------
        features : ndarray, shape (R,) or (n, R)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if x.ndim == 1:
            z = self._W @ x + self._b
        else:
            z = x @ self._W.T + self._b

        if self.feature_type == "random_fourier":
            return np.sqrt(2 / self.R) * np.cos(z)
        elif self.feature_type == "relu":
            return np.maximum(0, z)
        else:  # polynomial
            return z ** 2

    def attention_matrix_approx(
        self,
        Q: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Compute approximate attention matrix  A ≈ φ(Q) φ(K)ᵀ (unnormalized).

        Parameters
        ----------
        Q, K : ndarray, shape (n_tokens, d)
        """
        phi_Q = self.phi(Q)   # (n, R)
        phi_K = self.phi(K)   # (n, R)
        return phi_Q @ phi_K.T

    def fast_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """
        Compute linear attention output in O(n·R·d) time.

        Output[i] = (Σⱼ φ(Qᵢ)ᵀφ(Kⱼ) · Vⱼ) / (Σⱼ φ(Qᵢ)ᵀφ(Kⱼ))

        Parameters
        ----------
        Q, K, V : ndarray, shape (n_tokens, d)
        """
        phi_Q = self.phi(Q)   # (n, R)
        phi_K = self.phi(K)   # (n, R)

        # Numerator: Σⱼ φ(Qᵢ)ᵀφ(Kⱼ) Vⱼ  = φ(Q) · (φ(K)ᵀ V)
        KV = phi_K.T @ V           # (R, d)
        num = phi_Q @ KV            # (n, d)

        # Denominator: Σⱼ φ(Qᵢ)ᵀφ(Kⱼ) = φ(Q) · Σⱼ φ(Kⱼ)
        denom = phi_Q @ phi_K.sum(axis=0)  # (n,)
        denom = np.maximum(denom, 1e-8)    # numerical safety

        return num / denom[:, None]

    def alpha(self, Q: np.ndarray, K: np.ndarray) -> float:
        """Estimate ACF alpha from the attention weight decay."""
        phi_Q = self.phi(Q)
        phi_K = self.phi(K)
        svs = np.linalg.svd(phi_Q @ phi_K.T, compute_uv=False)
        svs = svs[svs > 1e-15]
        if len(svs) < 2:
            return 1.0
        ks = np.arange(1, len(svs) + 1, dtype=float)
        return float(-np.polyfit(np.log(ks), np.log(svs), 1)[0])

    def invariants(self, Q: np.ndarray, K: np.ndarray) -> AttentionKernelInvariants:
        """Compute attention-ACF invariants."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        alpha = self.alpha(Q, K)
        n = len(Q)
        speedup = (n ** 2 * self.d) / (n * self.d * self.R + n * self.R * self.d)

        # Estimate approximation error vs exact softmax
        exact = np.exp(Q @ K.T / np.sqrt(self.d))
        approx = self.phi(Q) @ self.phi(K).T
        # Normalize both row-wise
        exact_norm = exact / (exact.sum(axis=1, keepdims=True) + 1e-8)
        denom = approx.sum(axis=1) + 1e-8
        approx_norm = approx / denom[:, None]
        err = float(np.linalg.norm(exact_norm - approx_norm, "fro") / np.sqrt(n))

        return AttentionKernelInvariants(
            alpha_attention=alpha,
            n_features=self.R,
            approx_error=err,
            speedup_factor=max(speedup, 1.0),
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def reduce_green_function(
    G: Callable[[float, float], float],
    n_points: int = 64,
    order: int = 16,
    domain: Tuple[float, float] = (-1.0, 1.0),
) -> GreenFunctionReducer:
    """Fit and return a GreenFunctionReducer for G."""
    r = GreenFunctionReducer(n_points=n_points, order=order, domain=domain)
    r.fit(G)
    return r
