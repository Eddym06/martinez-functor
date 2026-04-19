"""
ACF for Graphs — Spectral Graph Reduction
==========================================

Extends the Affine Collapse Functor to operate on graphs and graph signals.

Mathematical foundation
-----------------------
A graph G = (V, E) with adjacency matrix A ∈ ℝⁿˣⁿ has a graph Laplacian

    L = D − A,   D = diag(∑ⱼ Aᵢⱼ)

whose eigendecomposition L = U Λ Uᵀ defines the graph Fourier transform.
A graph signal s: V → ℝ is a function over the vertices. Its graph Fourier
transform is ŝ = Uᵀs ∈ ℝⁿ (coefficients in the eigenbasis of L).

The ACF for graphs (GraphACF) applies the standard ACF functor Φ to the
*spectral representation* of the graph signal:

    Φ_G(s) = Φ(f_s)

where f_s: [λ_min, λ_max] → ℝ is the function that maps a Laplacian
eigenvalue λ to the corresponding Fourier coefficient ŝ(λ). The result is
a polynomial filter H(λ) = Σ cₖ φₖ(λ) that can be applied efficiently via

    filtered_signal = U · H(Λ) · Uᵀ · s

This is equivalent to a polynomial graph convolution — exactly the
operation in Chebyshev Graph CNNs (ChebConv), but derived from ACF
principles rather than neural network design.

Scope (honest)
--------------
- GraphReducer reduces graph signals to polynomial spectral filters.
- GraphACFAnalyzer computes ACF invariants (α, NC-class, Koopman) on
  the spectral representation of a graph.
- GraphSignalEvolver applies ACFAutoEvolver to graph signals.
- Multi-layer GNN forward-pass analysis: NOT a trainable neural network.
  This is a functional analysis tool, not a training framework.

References
----------
  Defferrard et al. (2016) — Convolutional Neural Networks on Graphs with
  Fast Localized Spectral Filtering (ChebConv baseline).
  Paper.md §32 — ACF Auto-Evolution (parent framework).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .core import (
    ChebyshevReducer,
    HornerReducer,
    KoopmanReducer,
    ReductionResult,
    ACFInvariant,
    FMAOperation,
)
from .auto_evolution import ACFAutoEvolver, ACFAutoEvolverConfig, AutoEvolutionResult


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphSpectrum:
    """Spectral decomposition of a graph Laplacian."""
    eigenvalues: torch.Tensor    # shape (n,) — sorted ascending
    eigenvectors: torch.Tensor   # shape (n, n) — columns are eigenvectors
    n_nodes: int
    n_edges: int
    is_connected: bool
    spectral_gap: float          # λ₂ (algebraic connectivity, Fiedler value)
    spectral_radius: float       # λ_max
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_eigenvalues(self) -> torch.Tensor:
        """Eigenvalues normalized to [0, 2] for symmetric normalization."""
        if self.spectral_radius > 0:
            return 2.0 * self.eigenvalues / self.spectral_radius
        return self.eigenvalues.clone()


@dataclass
class GraphSignal:
    """A signal defined over a graph's vertices."""
    values: torch.Tensor     # shape (n_nodes,) or (n_nodes, n_features)
    n_nodes: int
    feature_names: Optional[List[str]] = None

    def is_multidimensional(self) -> bool:
        return self.values.dim() > 1 and self.values.shape[-1] > 1


@dataclass
class GraphReductionResult:
    """Result of applying Φ to a graph signal."""
    polynomial_filter: ReductionResult   # H(λ) polynomial in spectral domain
    filtered_signal: torch.Tensor        # U · H(Λ) · Uᵀ · s  (n_nodes,)
    original_signal: torch.Tensor        # original s
    spectrum: GraphSpectrum
    epsilon: float                       # ‖s_filtered - s‖∞ (reconstruction error)
    filter_degree: int
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def residual(self) -> torch.Tensor:
        return self.original_signal - self.filtered_signal

    def relative_error(self) -> float:
        denom = float(torch.max(torch.abs(self.original_signal)).item())
        return self.epsilon / denom if denom > 1e-15 else 0.0

    def summary(self) -> str:
        return (
            f"GraphReductionResult: n={self.spectrum.n_nodes}, "
            f"edges={self.spectrum.n_edges}, "
            f"degree={self.filter_degree}, "
            f"ε={self.epsilon:.3e}, "
            f"rel_err={self.relative_error():.3e}, "
            f"t={self.elapsed_ms:.1f}ms"
        )


@dataclass
class GraphACFInvariants:
    """ACF invariants computed for a graph."""
    alpha: float                 # spectral complexity α ∈ [0,1]
    delta: float                 # approximation metric δ
    nc_class: str                # "NC0", "NC1", "NC2" (polynomial degree class)
    fiedler_value: float         # spectral gap λ₂
    spectral_entropy: float      # H(λ/λ_max) — uncertainty of spectral distribution
    optimal_filter_degree: int   # degree d* minimizing F(d, β=1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"GraphACFInvariants: α={self.alpha:.4f}, δ={self.delta:.4f}, "
            f"NC={self.nc_class}, λ₂={self.fiedler_value:.4f}, "
            f"H_spec={self.spectral_entropy:.4f}, d*={self.optimal_filter_degree}"
        )


@dataclass
class GraphEvolutionResult:
    """Result of applying ACFAutoEvolver to a graph signal."""
    final_filter: GraphReductionResult
    evolution_result: AutoEvolutionResult
    initial_epsilon: float
    final_epsilon: float
    improvement_ratio: float
    elapsed_ms: float

    def summary(self) -> str:
        return (
            f"GraphEvolution: ε₀={self.initial_epsilon:.3e} → "
            f"ε_f={self.final_epsilon:.3e} | "
            f"ratio=×{self.improvement_ratio:.2e} | "
            f"t={self.elapsed_ms:.1f}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GraphLaplacian builder
# ─────────────────────────────────────────────────────────────────────────────

class GraphLaplacian:
    """
    Compute graph Laplacian and its spectral decomposition.

    Supports:
      - Unnormalized Laplacian: L = D - A
      - Symmetric normalized: L_sym = D^{-1/2} L D^{-1/2}  (eigenvalues in [0,2])
      - Random-walk normalized: L_rw = D^{-1} L
    """

    @staticmethod
    def from_adjacency(
        adjacency: Union[np.ndarray, torch.Tensor],
        normalization: str = "unnormalized",
        dtype: torch.dtype = torch.float64,
    ) -> GraphSpectrum:
        """
        Compute spectral decomposition from an adjacency matrix.

        Parameters
        ----------
        adjacency : square matrix (n, n), non-negative, symmetric recommended
        normalization : "unnormalized" | "symmetric" | "random_walk"
        dtype : floating-point precision

        Returns
        -------
        GraphSpectrum
        """
        if isinstance(adjacency, torch.Tensor):
            A = adjacency.numpy().astype(np.float64)
        else:
            A = np.asarray(adjacency, dtype=np.float64)

        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"Adjacency matrix must be square; got {A.shape}")

        # Symmetrize if needed (silently — caller should provide symmetric A)
        A = (A + A.T) / 2.0

        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A

        if normalization == "symmetric":
            d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L = D_inv_sqrt @ L @ D_inv_sqrt
        elif normalization == "random_walk":
            d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
            D_inv = np.diag(d_inv)
            L = D_inv @ L
        elif normalization != "unnormalized":
            raise ValueError(f"Unknown normalization: '{normalization}'")

        vals, vecs = np.linalg.eigh(L)
        # Clean up near-zero eigenvalues (numerical noise)
        vals = np.clip(vals, 0.0, None)

        eigenvalues = torch.tensor(vals, dtype=dtype)
        eigenvectors = torch.tensor(vecs, dtype=dtype)

        n_edges = int(np.sum(A > 0)) // 2
        spectral_gap = float(vals[1]) if n > 1 else 0.0
        is_connected = spectral_gap > 1e-10

        return GraphSpectrum(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            n_nodes=n,
            n_edges=n_edges,
            is_connected=is_connected,
            spectral_gap=spectral_gap,
            spectral_radius=float(vals[-1]),
            metadata={
                "normalization": normalization,
                "degrees": degrees.tolist(),
            },
        )

    @staticmethod
    def from_edge_list(
        edges: List[Tuple[int, int]],
        n_nodes: int,
        weights: Optional[List[float]] = None,
        normalization: str = "unnormalized",
        dtype: torch.dtype = torch.float64,
    ) -> GraphSpectrum:
        """Build adjacency from edge list and compute spectrum."""
        A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            w = weights[i] if weights is not None else 1.0
            A[u, v] = w
            A[v, u] = w
        return GraphLaplacian.from_adjacency(A, normalization=normalization, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# GraphReducer — core Φ for graph signals
# ─────────────────────────────────────────────────────────────────────────────

class GraphReducer:
    """
    Apply the ACF functor Φ to a graph signal via spectral reduction.

    The reduction maps the graph signal s: V → ℝ to a polynomial filter
    H(λ) = Σₖ cₖ Tₖ(λ̃) where λ̃ are normalized eigenvalues and Tₖ are
    Chebyshev polynomials.

    Applying H back to the graph gives the filtered signal:
        s_filtered = U · H(Λ) · Uᵀ · s

    This is equivalent to a K-hop Chebyshev graph convolution with the
    graph signal s as input, derived from ACF first principles.

    Honest limitation: this is a single-channel, single-scale filter.
    For multi-channel, trainable GNN layers, see NeuralACF.
    """

    def __init__(
        self,
        filter_degree: int = 10,
        normalization: str = "symmetric",
        dtype: torch.dtype = torch.float64,
    ):
        self.filter_degree = filter_degree
        self.normalization = normalization
        self.dtype = dtype

    def reduce(
        self,
        signal: GraphSignal,
        spectrum: GraphSpectrum,
    ) -> GraphReductionResult:
        """
        Reduce a graph signal to its polynomial spectral representation.

        The spectral representation f_s: [λ_min, λ_max] → ℝ maps each
        eigenvalue to the corresponding Fourier coefficient, then Φ reduces
        f_s to a degree-K polynomial filter H(λ).
        """
        t0 = time.perf_counter()

        s = signal.values.to(self.dtype)
        if s.dim() == 2:
            # Multi-feature: process first feature for reduction analysis
            s = s[:, 0]

        U = spectrum.eigenvectors.to(self.dtype)
        lambdas = spectrum.eigenvalues.to(self.dtype)

        # Graph Fourier transform: ŝ = Uᵀ s
        s_hat = U.T @ s  # shape (n_nodes,)

        # Build spectral signal: f_s(λ) = ŝ(λ)
        # Represent as function from eigenvalue index to coefficient
        lam_min = float(lambdas[0].item())
        lam_max = float(lambdas[-1].item())
        if lam_max - lam_min < 1e-12:
            # Degenerate graph (all eigenvalues equal)
            lam_domain = (lam_min - 1.0, lam_min + 1.0)
        else:
            lam_domain = (lam_min, lam_max)

        # Interpolate the spectral signal as a function over [lam_min, lam_max]
        lam_np = lambdas.numpy()
        s_hat_np = s_hat.numpy()

        def spectral_fn(x: torch.Tensor) -> torch.Tensor:
            """Interpolate ŝ as a function of λ."""
            x_np = x.numpy()
            y = np.interp(x_np, lam_np, s_hat_np)
            return torch.tensor(y, dtype=self.dtype)

        # Apply ACF Φ to the spectral function
        reduction = ChebyshevReducer.reduce(
            spectral_fn,
            degree=self.filter_degree,
            domain=lam_domain,
            dtype=self.dtype,
        )

        # Build polynomial filter H: apply coefficients to all eigenvalues
        H_lambda = self._apply_polynomial_filter(reduction, lambdas, lam_domain)

        # Reconstruct filtered signal: s_filtered = U · H(Λ) · Uᵀ · s
        s_filtered = U @ (H_lambda * s_hat)

        epsilon = float(torch.max(torch.abs(s_filtered - s)).item())
        elapsed = (time.perf_counter() - t0) * 1e3

        return GraphReductionResult(
            polynomial_filter=reduction,
            filtered_signal=s_filtered,
            original_signal=s,
            spectrum=spectrum,
            epsilon=epsilon,
            filter_degree=self.filter_degree,
            elapsed_ms=elapsed,
            metadata={
                "s_hat_norm": float(torch.norm(s_hat).item()),
                "spectral_domain": lam_domain,
            },
        )

    def _apply_polynomial_filter(
        self,
        reduction: ReductionResult,
        lambdas: torch.Tensor,
        domain: Tuple[float, float],
    ) -> torch.Tensor:
        """Apply polynomial filter H to all eigenvalues λ."""
        coeffs = reduction.metadata.get("chebyshev_coefficients")
        if coeffs is not None:
            return ChebyshevReducer.evaluate_chebyshev_series(
                torch.as_tensor(coeffs, dtype=lambdas.dtype),
                lambdas,
                domain,
            )
        # Fallback: evaluate via execute
        return reduction.execute(lambdas)

    def reduce_from_adjacency(
        self,
        adjacency: Union[np.ndarray, torch.Tensor],
        signal_values: Union[np.ndarray, torch.Tensor],
        normalization: str = "symmetric",
    ) -> GraphReductionResult:
        """Convenience: compute spectrum and reduce in one call."""
        spectrum = GraphLaplacian.from_adjacency(
            adjacency, normalization=normalization, dtype=self.dtype
        )
        if isinstance(signal_values, np.ndarray):
            signal_values = torch.tensor(signal_values, dtype=self.dtype)
        signal = GraphSignal(values=signal_values, n_nodes=spectrum.n_nodes)
        return self.reduce(signal, spectrum)


# ─────────────────────────────────────────────────────────────────────────────
# GraphACFAnalyzer — ACF invariants for graphs
# ─────────────────────────────────────────────────────────────────────────────

class GraphACFAnalyzer:
    """
    Compute ACF invariants for a graph via spectral analysis.

    The alpha index, NC-class, and Koopman complexity are computed on
    the eigenvalue distribution of the Laplacian, treated as the
    "output spectrum" of the graph-as-dynamical-system.
    """

    def __init__(
        self,
        beta: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        self.beta = beta
        self.dtype = dtype

    def analyse(self, spectrum: GraphSpectrum) -> GraphACFInvariants:
        """Compute all ACF invariants for a graph."""
        lam = spectrum.eigenvalues

        # Alpha: spectral complexity index from ACFInvariant
        alpha, delta = ACFInvariant.compute_alpha(lam)
        alpha = max(0.0, min(1.0, alpha))  # clamp to [0, 1]

        # NC-class based on alpha value (mirrors core NC analysis)
        if alpha < 0.1:
            nc_class = "NC0"
        elif alpha < 0.5:
            nc_class = "NC1"
        else:
            nc_class = "NC2"

        # Spectral entropy H = -Σ pᵢ log pᵢ where pᵢ = λᵢ / Σλ
        lam_pos = lam[lam > 1e-12]
        if len(lam_pos) > 0:
            p = lam_pos / lam_pos.sum()
            spectral_entropy = float(-torch.sum(p * torch.log(p + 1e-15)).item())
        else:
            spectral_entropy = 0.0

        # Optimal filter degree via free energy F(d,β) = E(d) - S(d)/β
        # E(d): integrate residual of degree-d Chebyshev vs. spectral density
        # S(d): log(1+d)
        optimal_d = self._find_optimal_degree(spectrum)

        return GraphACFInvariants(
            alpha=alpha,
            delta=delta,
            nc_class=nc_class,
            fiedler_value=spectrum.spectral_gap,
            spectral_entropy=spectral_entropy,
            optimal_filter_degree=optimal_d,
            metadata={
                "n_nodes": spectrum.n_nodes,
                "n_edges": spectrum.n_edges,
                "spectral_radius": spectrum.spectral_radius,
                "is_connected": spectrum.is_connected,
            },
        )

    def _find_optimal_degree(self, spectrum: GraphSpectrum, max_degree: int = 20) -> int:
        """Find d* minimizing F(d, β) for the spectral density."""
        lam = spectrum.eigenvalues.to(self.dtype)
        lam_min = float(lam[0].item())
        lam_max = float(lam[-1].item())
        domain = (lam_min, lam_max + 1e-10)

        # Spectral density as a function
        lam_np = lam.numpy()
        density = np.ones(len(lam_np)) / len(lam_np)  # uniform

        def f_density(x: torch.Tensor) -> torch.Tensor:
            vals = np.interp(x.numpy(), lam_np, density)
            return torch.tensor(vals, dtype=self.dtype)

        best_d = 1
        best_F = float("inf")
        for d in range(1, max_degree + 1):
            try:
                red = ChebyshevReducer.reduce(f_density, degree=d, domain=domain, dtype=self.dtype)
                x_probe = torch.linspace(domain[0], domain[1], 500, dtype=self.dtype)
                y_true = f_density(x_probe)
                y_approx = red.execute(x_probe)
                E = float(torch.max(torch.abs(y_true - y_approx)).item())
                S = math.log(1 + d)
                F = E - S / self.beta
                if F < best_F:
                    best_F = F
                    best_d = d
            except Exception:
                continue
        return best_d


# ─────────────────────────────────────────────────────────────────────────────
# GraphSignalEvolver — ACFAutoEvolver for graph signals
# ─────────────────────────────────────────────────────────────────────────────

class GraphSignalEvolver:
    """
    Apply ACFAutoEvolver to a graph signal in spectral domain.

    Uses the four auto-evolution mechanisms (idempotence, adjunction,
    thermodynamics, adaptive refinement) on the spectral representation
    f_s: [λ_min, λ_max] → ℝ of the signal.
    """

    def __init__(self, config: Optional[ACFAutoEvolverConfig] = None):
        self.config = config or ACFAutoEvolverConfig(
            initial_degree=10,
            n_probe=500,
            beta=1.0,
        )

    def evolve(
        self,
        signal: GraphSignal,
        spectrum: GraphSpectrum,
    ) -> GraphEvolutionResult:
        """Auto-evolve the graph signal's spectral representation."""
        t0 = time.perf_counter()

        s = signal.values.to(torch.float64)
        if s.dim() == 2:
            s = s[:, 0]

        U = spectrum.eigenvectors.to(torch.float64)
        lambdas = spectrum.eigenvalues.to(torch.float64)
        s_hat = U.T @ s

        lam_np = lambdas.numpy()
        s_hat_np = s_hat.numpy()
        lam_min = float(lambdas[0].item())
        lam_max = float(lambdas[-1].item())
        domain = (lam_min, lam_max + 1e-10) if lam_max - lam_min < 1e-12 else (lam_min, lam_max)

        def spectral_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                np.interp(x.numpy(), lam_np, s_hat_np),
                dtype=torch.float64,
            )

        evolver = ACFAutoEvolver(self.config)
        evo_result = evolver.evolve(spectral_fn, domain)

        # Apply best filter back to graph
        H_lambda = evo_result.best_reduction.execute(lambdas)
        s_filtered = U @ (H_lambda * s_hat)
        epsilon = float(torch.max(torch.abs(s_filtered - s)).item())

        # Wrap in GraphReductionResult
        final_filter = GraphReductionResult(
            polynomial_filter=evo_result.best_reduction,
            filtered_signal=s_filtered,
            original_signal=s,
            spectrum=spectrum,
            epsilon=epsilon,
            filter_degree=self.config.initial_degree,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )

        initial_eps = evo_result.initial_epsilon
        final_eps = evo_result.final_epsilon
        ratio = initial_eps / final_eps if final_eps > 1e-15 else 1.0

        return GraphEvolutionResult(
            final_filter=final_filter,
            evolution_result=evo_result,
            initial_epsilon=initial_eps,
            final_epsilon=final_eps,
            improvement_ratio=ratio,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph generators for testing and benchmarking
# ─────────────────────────────────────────────────────────────────────────────

class StandardGraphs:
    """Factory for standard graphs used in tests and benchmarks."""

    @staticmethod
    def path(n: int) -> np.ndarray:
        """Path graph P_n."""
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        return A

    @staticmethod
    def cycle(n: int) -> np.ndarray:
        """Cycle graph C_n."""
        A = StandardGraphs.path(n)
        A[0, n - 1] = A[n - 1, 0] = 1.0
        return A

    @staticmethod
    def complete(n: int) -> np.ndarray:
        """Complete graph K_n."""
        return np.ones((n, n)) - np.eye(n)

    @staticmethod
    def grid(rows: int, cols: int) -> np.ndarray:
        """2D grid graph."""
        n = rows * cols
        A = np.zeros((n, n))
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    j = r * cols + (c + 1)
                    A[i, j] = A[j, i] = 1.0
                if r + 1 < rows:
                    j = (r + 1) * cols + c
                    A[i, j] = A[j, i] = 1.0
        return A

    @staticmethod
    def star(n: int) -> np.ndarray:
        """Star graph S_n (one center, n-1 leaves)."""
        A = np.zeros((n, n))
        for i in range(1, n):
            A[0, i] = A[i, 0] = 1.0
        return A

    @staticmethod
    def random_regular(n: int, d: int, seed: int = 42) -> np.ndarray:
        """
        Random d-regular graph on n nodes (approximate via symmetric random).
        n*d must be even. Returns approximate regular graph.
        """
        rng = np.random.default_rng(seed)
        A = np.zeros((n, n))
        for i in range(n):
            # Connect to d random other nodes
            others = [j for j in range(n) if j != i and A[i, j] == 0]
            if len(others) >= d:
                chosen = rng.choice(others, size=d, replace=False)
                for j in chosen:
                    A[i, j] = A[j, i] = 1.0
        return A
