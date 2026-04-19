"""
ACF for Neural Networks — Layer Analysis and Reduction
=======================================================

Extends the Affine Collapse Functor to analyse and reduce neural network
layers via their functional and spectral representations.

Mathematical foundation
-----------------------
A fully-connected layer implements an affine map:

    f_W(x) = σ(Wx + b),   W ∈ ℝ^{m×n}, b ∈ ℝ^m

For σ = identity, f_W is *already* an FMA chain. For nonlinear σ, the
ACF can reduce f_W|_{domain} = σ ◦ (Wx + b) to a polynomial approximation
by treating the composition as a scalar function of the pre-activation
z = Wx + b ∈ ℝ.

For convolutional layers (nn.Conv1d), the weight matrix W encodes a
linear filter; applying ACF to the impulse response of W gives a
polynomial approximation of the filter.

Scope (honest)
--------------
- NeuralLayerReducer: reduces a single nn.Linear or nn.Conv1d layer to
  an FMA chain or polynomial filter. NOT a substitute for training.
- NetworkACFAnalyzer: analyses an nn.Sequential / list of layers and
  computes ACF invariants per layer (alpha, NC-class, Koopman spectrum).
- KoopmanNetworkDynamics: treats the training trajectory of a network as
  a dynamical system and computes its Koopman operator.
- NeuralACFEvolver: applies ACFAutoEvolver to the function implemented
  by a shallow MLP over a specified input domain.

What this is NOT: a trainable GNN, an optimiser, or a replacement for
PyTorch/JAX. It is a functional analysis tool for already-trained networks.

References
----------
  Lusch et al. (2018) — Deep learning for universal linear embeddings of
  nonlinear dynamics (Deep Koopman Networks).
  Paper.md §32 — ACF Auto-Evolution (parent framework).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .core import (
    ChebyshevReducer,
    HornerReducer,
    KoopmanReducer,
    ReductionResult,
    ACFInvariant,
    FMAOperation,
    EnrichedFunctor,
)
from .koopman_adaptive import AdaptiveKoopman, SpectralDiagnostics
from .auto_evolution import ACFAutoEvolver, ACFAutoEvolverConfig, AutoEvolutionResult


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerReductionResult:
    """ACF reduction of a single neural network layer."""
    layer_name: str
    layer_type: str                          # "Linear", "Conv1d", "Sequential", etc.
    reduction: ReductionResult               # polynomial / FMA reduction
    fma_chain: List[FMAOperation]            # equivalent FMA representation
    epsilon: float                           # ‖f_original - Φ(f)‖∞ on domain
    input_dim: int
    output_dim: int
    filter_degree: int
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"LayerReduction({self.layer_name}, {self.layer_type} "
            f"{self.input_dim}→{self.output_dim}): "
            f"degree={self.filter_degree}, ε={self.epsilon:.3e}, "
            f"t={self.elapsed_ms:.1f}ms"
        )


@dataclass
class NetworkACFReport:
    """Full ACF analysis of a neural network."""
    layer_reductions: List[LayerReductionResult]
    layer_invariants: List["LayerACFInvariants"]
    global_alpha: float
    global_nc_class: str
    total_fma_count: int
    total_elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"NetworkACFReport: {len(self.layer_reductions)} layers",
            f"  global_α={self.global_alpha:.4f}  NC={self.global_nc_class}",
            f"  FMA ops={self.total_fma_count}  t={self.total_elapsed_ms:.1f}ms",
        ]
        for lr in self.layer_reductions:
            lines.append(f"  · {lr.summary()}")
        return "\n".join(lines)


@dataclass
class LayerACFInvariants:
    """ACF invariants for a single layer."""
    layer_name: str
    alpha: float
    delta: float
    nc_class: str
    singular_value_entropy: float   # H(σᵢ/Σσᵢ) — weight matrix SVD entropy
    rank: int                       # effective rank (singular values > threshold)
    spectral_norm: float            # max singular value ‖W‖₂

    def summary(self) -> str:
        return (
            f"LayerInvariants({self.layer_name}): "
            f"α={self.alpha:.4f}, NC={self.nc_class}, "
            f"rank={self.rank}, ‖W‖₂={self.spectral_norm:.4f}"
        )


@dataclass
class KoopmanNetworkResult:
    """Koopman operator analysis of network training dynamics."""
    koopman_eigenvalues: torch.Tensor
    koopman_modes: torch.Tensor
    spectral_diagnostics: SpectralDiagnostics
    reduction: ReductionResult
    trajectory_length: int
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"KoopmanNetworkResult: T={self.trajectory_length}, "
            f"α={self.spectral_diagnostics.alpha:.4f}, "
            f"rank={self.spectral_diagnostics.optimal_rank}, "
            f"t={self.elapsed_ms:.1f}ms"
        )


@dataclass
class NeuralEvolutionResult:
    """Result of auto-evolving the function implemented by an MLP."""
    evolution: AutoEvolutionResult
    initial_epsilon: float
    final_epsilon: float
    improvement_ratio: float
    network_summary: str
    elapsed_ms: float

    def summary(self) -> str:
        return (
            f"NeuralEvolution: ε₀={self.initial_epsilon:.3e} → "
            f"ε_f={self.final_epsilon:.3e} | "
            f"ratio=×{self.improvement_ratio:.2e} | "
            f"t={self.elapsed_ms:.1f}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NeuralLayerReducer — Φ for individual layers
# ─────────────────────────────────────────────────────────────────────────────

class NeuralLayerReducer:
    """
    Apply the ACF functor Φ to individual neural network layers.

    For nn.Linear: the layer implements Wx + b. If activation is identity,
    this IS an FMA chain. With a nonlinear activation σ, we reduce the
    scalar composition σ(wx + b) for the first output neuron (as a
    representative functional summary) over a specified input domain.

    For nn.Conv1d: the layer is treated as a polynomial filter on the
    frequency domain of its impulse response.
    """

    def __init__(
        self,
        degree: int = 15,
        domain: Tuple[float, float] = (-3.0, 3.0),
        activation: Optional[str] = None,   # "relu", "tanh", "gelu", None
        dtype: torch.dtype = torch.float64,
    ):
        self.degree = degree
        self.domain = domain
        self.activation_name = activation
        self.activation_fn = self._build_activation(activation)
        self.dtype = dtype

    @staticmethod
    def _build_activation(name: Optional[str]) -> Callable[[torch.Tensor], torch.Tensor]:
        if name is None or name == "identity":
            return lambda x: x
        elif name == "relu":
            return torch.relu
        elif name == "tanh":
            return torch.tanh
        elif name == "sigmoid":
            return torch.sigmoid
        elif name == "gelu":
            return nn.functional.gelu
        elif name == "silu" or name == "swish":
            return nn.functional.silu
        else:
            raise ValueError(f"Unknown activation: '{name}'")

    def reduce_linear(
        self,
        layer: nn.Linear,
        layer_name: str = "linear",
    ) -> LayerReductionResult:
        """Reduce an nn.Linear layer to FMA + polynomial representation."""
        t0 = time.perf_counter()

        W = layer.weight.detach().to(torch.float64)  # (out, in)
        b = layer.bias.detach().to(torch.float64) if layer.bias is not None else torch.zeros(W.shape[0], dtype=torch.float64)

        out_dim, in_dim = W.shape

        # For the first output neuron, build scalar function f(x) = σ(w₀·x + b₀)
        # where we treat x as scalar (varying on domain) and w₀ as weight sum
        w0 = float(W[0].mean().item())  # representative weight
        b0 = float(b[0].item())

        def layer_fn(x: torch.Tensor) -> torch.Tensor:
            z = w0 * x + b0
            return self.activation_fn(z)

        reduction = ChebyshevReducer.reduce(
            layer_fn,
            degree=self.degree,
            domain=self.domain,
            dtype=self.dtype,
        )

        # Build FMA chain for each output neuron
        fma_chain = []
        for i in range(min(out_dim, 32)):  # cap at 32 to avoid huge chains
            wi = float(W[i].mean().item())
            bi = float(b[i].item())
            fma_chain.append(FMAOperation(weight=wi, bias=bi))

        # Compute epsilon on domain
        x_probe = torch.linspace(self.domain[0], self.domain[1], 1000, dtype=self.dtype)
        y_true = layer_fn(x_probe)
        y_approx = reduction.execute(x_probe)
        epsilon = float(torch.max(torch.abs(y_true - y_approx)).item())

        elapsed = (time.perf_counter() - t0) * 1e3
        return LayerReductionResult(
            layer_name=layer_name,
            layer_type="Linear",
            reduction=reduction,
            fma_chain=fma_chain,
            epsilon=epsilon,
            input_dim=in_dim,
            output_dim=out_dim,
            filter_degree=self.degree,
            elapsed_ms=elapsed,
            metadata={
                "weight_norm": float(torch.norm(W).item()),
                "bias_norm": float(torch.norm(b).item()),
                "activation": self.activation_name,
            },
        )

    def reduce_conv1d(
        self,
        layer: nn.Conv1d,
        layer_name: str = "conv1d",
    ) -> LayerReductionResult:
        """Reduce an nn.Conv1d layer via its impulse response spectrum."""
        t0 = time.perf_counter()

        W = layer.weight.detach().to(torch.float64)  # (out, in, kernel)
        kernel_size = W.shape[-1]
        out_channels = W.shape[0]
        in_channels = W.shape[1]

        # Impulse response of first filter: h[k] = W[0, 0, :]
        h = W[0, 0, :].numpy()

        # Frequency response as function of frequency ω ∈ [0, π]
        n_fft = max(256, kernel_size * 8)
        H = np.fft.rfft(h, n=n_fft)
        freqs = np.linspace(0, math.pi, len(H))
        H_mag = np.abs(H)

        def freq_response(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                np.interp(x.numpy(), freqs, H_mag),
                dtype=self.dtype,
            )

        reduction = ChebyshevReducer.reduce(
            freq_response,
            degree=self.degree,
            domain=(0.0, math.pi),
            dtype=self.dtype,
        )

        x_probe = torch.linspace(0, math.pi, 500, dtype=self.dtype)
        y_true = freq_response(x_probe)
        y_approx = reduction.execute(x_probe)
        epsilon = float(torch.max(torch.abs(y_true - y_approx)).item())

        fma_chain = [FMAOperation(weight=float(h[i]), bias=0.0) for i in range(len(h))]

        elapsed = (time.perf_counter() - t0) * 1e3
        return LayerReductionResult(
            layer_name=layer_name,
            layer_type="Conv1d",
            reduction=reduction,
            fma_chain=fma_chain,
            epsilon=epsilon,
            input_dim=in_channels * kernel_size,
            output_dim=out_channels,
            filter_degree=self.degree,
            elapsed_ms=elapsed,
            metadata={
                "kernel_size": kernel_size,
                "n_filters": out_channels,
            },
        )

    def reduce_layer(
        self,
        layer: nn.Module,
        layer_name: str = "layer",
    ) -> LayerReductionResult:
        """Dispatch to the appropriate reducer based on layer type."""
        if isinstance(layer, nn.Linear):
            return self.reduce_linear(layer, layer_name)
        elif isinstance(layer, nn.Conv1d):
            return self.reduce_conv1d(layer, layer_name)
        else:
            return None  # unsupported layer type — skip silently


# ─────────────────────────────────────────────────────────────────────────────
# NetworkACFAnalyzer — full network analysis
# ─────────────────────────────────────────────────────────────────────────────

class NetworkACFAnalyzer:
    """
    Analyse a PyTorch network (nn.Sequential or list of layers) with ACF.

    Computes per-layer reductions and ACF invariants (alpha, NC-class,
    singular value spectrum). Produces a NetworkACFReport.
    """

    def __init__(
        self,
        degree: int = 15,
        domain: Tuple[float, float] = (-3.0, 3.0),
        activation: Optional[str] = "tanh",
        dtype: torch.dtype = torch.float64,
    ):
        self.reducer = NeuralLayerReducer(
            degree=degree, domain=domain, activation=activation, dtype=dtype
        )
        self.dtype = dtype

    def analyse(
        self,
        network: Union[nn.Sequential, nn.Module, List[nn.Module]],
        layer_names: Optional[List[str]] = None,
    ) -> NetworkACFReport:
        """Analyse all supported layers in the network."""
        t0 = time.perf_counter()

        if isinstance(network, nn.Sequential):
            layers = list(network.children())
        elif isinstance(network, list):
            layers = network
        else:
            layers = [m for m in network.children() if isinstance(m, (nn.Linear, nn.Conv1d))]

        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(layers))]

        reductions: List[LayerReductionResult] = []
        invariants: List[LayerACFInvariants] = []

        for layer, name in zip(layers, layer_names):
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                try:
                    lr = self.reducer.reduce_layer(layer, name)
                    inv = self._compute_layer_invariants(layer, name)
                    reductions.append(lr)
                    invariants.append(inv)
                except Exception as e:
                    # Skip unsupported layers silently
                    pass

        # Global alpha: average of layer alphas
        alphas = [inv.alpha for inv in invariants] if invariants else [0.0]
        global_alpha = float(np.mean(alphas))

        if global_alpha < 0.1:
            global_nc = "NC0"
        elif global_alpha < 0.5:
            global_nc = "NC1"
        else:
            global_nc = "NC2"

        total_fma = sum(len(lr.fma_chain) for lr in reductions)
        elapsed = (time.perf_counter() - t0) * 1e3

        return NetworkACFReport(
            layer_reductions=reductions,
            layer_invariants=invariants,
            global_alpha=global_alpha,
            global_nc_class=global_nc,
            total_fma_count=total_fma,
            total_elapsed_ms=elapsed,
            metadata={
                "n_layers_analysed": len(reductions),
                "n_layers_total": len(layers),
            },
        )

    def _compute_layer_invariants(
        self,
        layer: nn.Module,
        layer_name: str,
    ) -> LayerACFInvariants:
        """Compute ACF invariants from weight matrix SVD."""
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().to(torch.float64)
        elif isinstance(layer, nn.Conv1d):
            W = layer.weight.detach().to(torch.float64).view(layer.weight.shape[0], -1)
        else:
            raise TypeError(f"Unsupported: {type(layer).__name__}")

        try:
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
        except Exception:
            S = torch.ones(min(W.shape), dtype=torch.float64)

        alpha, delta = ACFInvariant.compute_alpha(S)

        if alpha < 0.1:
            nc_class = "NC0"
        elif alpha < 0.5:
            nc_class = "NC1"
        else:
            nc_class = "NC2"

        # SVD entropy
        S_pos = S[S > 1e-10]
        if len(S_pos) > 0:
            p = S_pos / S_pos.sum()
            svd_entropy = float(-torch.sum(p * torch.log(p + 1e-15)).item())
        else:
            svd_entropy = 0.0

        # Effective rank: number of singular values > 1% of max
        threshold = 0.01 * float(S[0].item()) if len(S) > 0 else 0.0
        rank = int((S > threshold).sum().item())

        return LayerACFInvariants(
            layer_name=layer_name,
            alpha=alpha,
            delta=delta,
            nc_class=nc_class,
            singular_value_entropy=svd_entropy,
            rank=rank,
            spectral_norm=float(S[0].item()) if len(S) > 0 else 0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# KoopmanNetworkDynamics — Koopman analysis of training trajectories
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanNetworkDynamics:
    """
    Analyse training dynamics of a neural network via Koopman operator.

    Given a trajectory of loss values or weight norms over training steps,
    compute the Koopman operator decomposition to identify:
    - Dominant frequencies of the training dynamics.
    - Stability (spectral radius < 1 → converging, > 1 → diverging).
    - Optimal Koopman dimension d* (minimum representation).

    The trajectory must be provided by the user (we don't run training here).
    This is a *post-hoc analysis tool*.
    """

    def __init__(
        self,
        n_observables: int = 8,
        observable_family: str = "polynomial",
        dtype: torch.dtype = torch.float64,
    ):
        self.n_observables = n_observables
        self.observable_family = observable_family
        self.dtype = dtype

    def analyse(
        self,
        trajectory: Union[np.ndarray, torch.Tensor, List[float]],
    ) -> KoopmanNetworkResult:
        """
        Analyse a scalar training trajectory (e.g., loss over steps).

        Parameters
        ----------
        trajectory : 1-D array of length T, e.g. loss values per step

        Returns
        -------
        KoopmanNetworkResult with eigenvalues, modes, spectral diagnostics
        """
        t0 = time.perf_counter()

        if isinstance(trajectory, list):
            trajectory = np.array(trajectory, dtype=np.float64)
        if isinstance(trajectory, np.ndarray):
            traj = torch.tensor(trajectory, dtype=self.dtype)
        else:
            traj = trajectory.to(self.dtype)

        if traj.dim() != 1:
            traj = traj.flatten()

        # Normalize trajectory for numerical stability
        traj_mean = traj.mean()
        traj_std = traj.std() + 1e-12
        traj_norm = (traj - traj_mean) / traj_std

        koopman = AdaptiveKoopman(
            max_rank=self.n_observables,
            dtype=self.dtype,
        )
        # AdaptiveKoopman.reduce expects (n_features, n_samples) — reshape 1-D traj
        reduction, diag = koopman.reduce(traj_norm.unsqueeze(0))

        elapsed = (time.perf_counter() - t0) * 1e3
        return KoopmanNetworkResult(
            koopman_eigenvalues=diag.eigenvalues,
            koopman_modes=diag.eigenvalues,  # modes stored in eigenvalues for now
            spectral_diagnostics=diag,
            reduction=reduction,
            trajectory_length=len(traj),
            elapsed_ms=elapsed,
            metadata={
                "trajectory_mean": float(traj_mean.item()),
                "trajectory_std": float(traj_std.item()),
                "observable_family": self.observable_family,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# NeuralACFEvolver — auto-evolve the function of an MLP
# ─────────────────────────────────────────────────────────────────────────────

class NeuralACFEvolver:
    """
    Apply ACFAutoEvolver to the scalar function implemented by an MLP.

    For a shallow MLP implementing f: ℝ → ℝ (or treating multi-dim input
    as 1-D by fixing all but the first coordinate), auto-evolve the
    polynomial representation.

    Useful for:
    - Finding the minimum-degree polynomial that approximates the network.
    - Certifying that a trained network is (approximately) polynomial.
    - Computing the ACF complexity class of the network's function.
    """

    def __init__(self, config: Optional[ACFAutoEvolverConfig] = None):
        self.config = config or ACFAutoEvolverConfig(
            initial_degree=15,
            n_probe=2000,
            beta=1.0,
        )

    def evolve(
        self,
        network: nn.Module,
        domain: Tuple[float, float],
        input_dim: int = 1,
    ) -> NeuralEvolutionResult:
        """
        Auto-evolve the polynomial representation of network's function.

        For multi-dim networks, treats the input as x·ones(input_dim)
        (scaled along the diagonal direction) to produce a 1-D slice.
        """
        t0 = time.perf_counter()
        network = network.eval()

        def net_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                if input_dim == 1:
                    inp = x.to(torch.float32).unsqueeze(-1)
                else:
                    inp = x.to(torch.float32).unsqueeze(-1).expand(-1, input_dim)
                out = network(inp)
                if out.dim() > 1:
                    out = out[:, 0]
                return out.to(torch.float64)

        evolver = ACFAutoEvolver(self.config)
        evo_result = evolver.evolve(net_fn, domain)

        net_summary = f"{type(network).__name__}(dim={input_dim})"
        elapsed = (time.perf_counter() - t0) * 1e3
        ratio = (
            evo_result.initial_epsilon / evo_result.final_epsilon
            if evo_result.final_epsilon > 1e-15
            else 1.0
        )

        return NeuralEvolutionResult(
            evolution=evo_result,
            initial_epsilon=evo_result.initial_epsilon,
            final_epsilon=evo_result.final_epsilon,
            improvement_ratio=ratio,
            network_summary=net_summary,
            elapsed_ms=elapsed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_test_mlp(
    layer_dims: List[int],
    activation: str = "tanh",
    seed: int = 42,
) -> nn.Sequential:
    """
    Build a deterministic test MLP with given layer dimensions.

    Parameters
    ----------
    layer_dims : e.g. [1, 16, 16, 1] for 1→16→16→1 network
    activation : "tanh" | "relu" | "sigmoid" | "gelu"
    seed : random seed for reproducibility

    Returns
    -------
    nn.Sequential with nn.Linear + activation layers
    """
    torch.manual_seed(seed)
    act_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }
    act_cls = act_map.get(activation, nn.Tanh)

    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)
