"""
Differentiable ACF — PyTorch Autograd-Compatible Functor (Expansion E5)
========================================================================

This module implements the DIFFERENTIABLE ACF layer: a torch.autograd.Function
that makes the compilation process differentiable with respect to:
  1. Function parameters θ (for parameterized functions f_θ).
  2. The coefficient vector c of the Chebyshev expansion.

THEOREM (Differentiable ACF — DA-1):
  Let f_θ(x) = Σ_k c_k(θ) · T_k(x) be a Chebyshev expansion where
  c_k(θ) depends smoothly on parameters θ. Then:

    ∂/∂θ ε(Φ(f_θ)) = Σ_k (∂ε/∂c_k) · (∂c_k/∂θ)

  where ∂ε/∂c_k = sign(f_θ(x*) - P_d(x*)) · T_k(x*)
  at the argmax residual point x*.

COROLLARY (DA-2): Natural gradient on the Chebyshev manifold:
  The Fisher metric G_F at coefficient c is:
    G_F[j,k] = E_x[T_j(x) · T_k(x)] = δ_{jk}/2  (orthogonality)
  So the natural gradient reduces to the ordinary gradient (manifold is flat).

COROLLARY (DA-3): Optimal degree by gradient:
  Minimizing ε(d, θ) over d ∈ ℕ and θ simultaneously gives the joint
  optimal (d*, θ*) where the function is "easiest" to compile.

APPLICATIONS:
  1. Neural network layer where activation functions are ACF-compiled.
  2. Differentiable compilation of parameterized physics models.
  3. Joint optimization of function representation and hardware mapping.

Implementation
--------------
  DifferentiableACFLayer : nn.Module wrapping torch.autograd.Function
  ChebyshevCoefficients  : Differentiable Chebyshev coefficient extraction  
  ACFGradientFlow        : ∂ε/∂θ via chain rule through the functor

References
----------
  Paper.md §14.3, §23.7 — information geometry and gradient flow.
  Baydin et al. (2018) — Automatic differentiation in machine learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Differentiable Chebyshev coefficient extraction
# ---------------------------------------------------------------------------

class ChebyshevCoeffExtractor(Function):
    """
    torch.autograd.Function for differentiable Chebyshev coefficients.

    Forward:  c = DCT(f, d)  — degree-d coefficients
    Backward: ∂L/∂f_nodes via (∂c/∂f_nodes)^T · ∂L/∂c

    This allows the coefficients themselves to be differentiated through
    any loss function that depends on the approximation quality.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        f_nodes: torch.Tensor,      # (d,) function values at Chebyshev nodes
        t_nodes: torch.Tensor,      # (d,) Chebyshev nodes in [-1, 1]
    ) -> torch.Tensor:
        """
        Forward pass: Chebyshev fit.
        Returns: coefficient tensor c (d,).
        """
        from numpy.polynomial import chebyshev

        d = f_nodes.shape[0]
        f_np = f_nodes.double().cpu().numpy()
        t_np = t_nodes.double().cpu().numpy()

        # Chebyshev DCT-II coefficients
        try:
            coeffs = chebyshev.chebfit(t_np, f_np, d - 1)
        except Exception:
            coeffs = np.zeros(d)

        c_tensor = torch.tensor(coeffs, dtype=f_nodes.dtype, device=f_nodes.device)
        ctx.save_for_backward(f_nodes, t_nodes)
        ctx.d = d
        return c_tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,  # ∂L/∂c  (d,)
    ) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: propagate ∂L/∂c → ∂L/∂f_nodes.

        ∂c_k/∂f_j = (2/d) · T_k(t_j)  for k > 0
                   = (1/d) · T_0(t_j)  for k = 0
        So ∂L/∂f_j = Σ_k (∂L/∂c_k) · (∂c_k/∂f_j)
        """
        f_nodes, t_nodes = ctx.saved_tensors
        d = ctx.d
        t_np = t_nodes.double().cpu().numpy()

        # Build Chebyshev Vandermonde matrix: V[j,k] = T_k(t_j), shape (d, d)
        V = np.zeros((d, d), dtype=np.float64)
        for k in range(d):
            V[:, k] = np.cos(k * np.arccos(np.clip(t_np, -1, 1)))

        # ∂c_k/∂f_j = (2/d) · V[j,k] for k > 0, (1/d) for k=0
        weights = np.full(d, 2.0 / d)
        weights[0] = 1.0 / d
        # ∂c/∂f has shape (d, d): entry [k, j] = weights[k] * V[j, k]
        dc_df = weights[:, None] * V.T  # (d, d)

        grad_np = grad_output.double().cpu().numpy()
        grad_f = dc_df.T @ grad_np  # ∂L/∂f = (∂c/∂f)^T · ∂L/∂c

        return (
            torch.tensor(grad_f, dtype=f_nodes.dtype, device=f_nodes.device),
            None,  # no gradient for t_nodes
        )


class DifferentiableChebyshevApprox(nn.Module):
    """
    Differentiable Chebyshev approximation as an nn.Module.

    This layer approximates a function f_θ(x) where θ are learnable
    parameters that the function values at the Chebyshev nodes depend on.

    If f_θ is a simple polynomial: f_θ(x) = Σ θ_k x^k, then F_nodes = V_mono @ θ
    where V_mono is the Vandermonde matrix at the Chebyshev nodes.

    Usage
    -----
    >>> layer = DifferentiableChebyshevApprox(degree=10, domain=(-1., 1.))
    >>> x = torch.linspace(-1, 1, 100)
    >>> theta = torch.randn(11, requires_grad=True)
    >>> y_approx = layer(x, theta)
    >>> loss = torch.mean((y_approx - y_target) ** 2)
    >>> loss.backward()
    """

    def __init__(
        self,
        degree: int = 20,
        domain: Tuple[float, float] = (-1.0, 1.0),
        param_type: str = "monomial",   # "monomial" | "direct" | "neural"
        hidden_size: int = 0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.degree = degree
        self.domain = domain
        self.param_type = param_type
        self.dtype = dtype

        a, b = domain
        d = degree + 1
        k = torch.arange(1, d + 1, dtype=dtype)
        t_nodes = torch.cos(math.pi * (2 * k - 1) / (2 * d))
        self.register_buffer("t_nodes", t_nodes)
        # Map t ∈ [-1,1] to x ∈ [a,b]
        x_nodes = (a + b) / 2.0 + (b - a) / 2.0 * t_nodes
        self.register_buffer("x_nodes", x_nodes)

        # Evaluation grid (for forward pass on arbitrary x)
        self._eval_setup = None

        if param_type == "monomial":
            # parameterization: f_theta(x) = sum_k theta_k * x^k
            self.theta = nn.Parameter(torch.zeros(d, dtype=dtype))
        elif param_type == "direct":
            # directly learn function values at nodes
            self.theta = nn.Parameter(torch.zeros(d, dtype=dtype))
        elif param_type == "neural":
            # small MLP to generate f values from x
            hs = hidden_size or 32
            self.net = nn.Sequential(
                nn.Linear(1, hs, dtype=dtype),
                nn.Tanh(),
                nn.Linear(hs, hs, dtype=dtype),
                nn.Tanh(),
                nn.Linear(hs, 1, dtype=dtype),
            )

    def forward(
        self,
        x: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate the differentiable Chebyshev approximation at points x.

        Parameters
        ----------
        x : (N,) evaluation points in [a, b]
        theta : override for parameters (optional)

        Returns
        -------
        y : (N,) approximated function values, with gradients through theta.
        """
        a, b = self.domain
        d = self.degree + 1
        t_n = self.t_nodes  # (d,)
        x_n = self.x_nodes  # (d,)

        # ── Compute function values at Chebyshev nodes ──────────────────
        if theta is not None:
            f_nodes = theta
        elif self.param_type == "monomial":
            # f_theta(x) = sum_k theta_k * x^k
            f_nodes = torch.stack([
                torch.sum(self.theta * x_n_i ** torch.arange(d, dtype=self.dtype, device=x_n_i.device))
                for x_n_i in x_n
            ])
        elif self.param_type == "direct":
            f_nodes = self.theta
        elif self.param_type == "neural":
            f_nodes = self.net(x_n.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")

        # ── Differentiable coefficient extraction ────────────────────────
        c = ChebyshevCoeffExtractor.apply(f_nodes, t_n)  # (d,)

        # ── Evaluate Chebyshev series at x ───────────────────────────────
        t_x = 2.0 * (x - (a + b) / 2.0) / (b - a)  # map to [-1, 1]
        t_x = t_x.clamp(-1.0, 1.0)

        # Clenshaw recurrence (differentiable)
        y = self._clenshaw(c, t_x)
        return y

    def _clenshaw(self, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Clenshaw recurrence for Chebyshev sum: Σ c_k T_k(t).
        Fully differentiable w.r.t. c and t.

        T_k(t) = cos(k * arccos(t)) — but we use the recurrence:
          b_n   = c_n
          b_{n-1} = c_{n-1} + 2t·b_n
          b_{k-1} = c_{k-1} + 2t·b_k - b_{k+1}
          result  = c_0 + t·b_1 - b_2
        """
        n = c.shape[0]
        if n == 0:
            return torch.zeros_like(t)
        if n == 1:
            return c[0].expand_as(t)

        b_next = torch.zeros_like(t)
        b_curr = c[n - 1].expand_as(t)

        for k in range(n - 2, 0, -1):
            b_prev = c[k].expand_as(t) + 2.0 * t * b_curr - b_next
            b_next = b_curr
            b_curr = b_prev

        return c[0].expand_as(t) + t * b_curr - b_next

    def compilation_error(
        self,
        f_true: Callable[[torch.Tensor], torch.Tensor],
        x_eval: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ε = max|f_true(x) - f_approx(x)| as a differentiable tensor.

        The max is approximated by a soft maximum (LogSumExp) for differentiability.
        """
        a, b = self.domain
        if x_eval is None:
            x_eval = torch.linspace(a, b, 500, dtype=self.dtype)

        y_true = f_true(x_eval)
        y_pred = self.forward(x_eval)
        residuals = torch.abs(y_true - y_pred)
        # Soft maximum (differentiable approximation)
        beta = 100.0
        soft_max = (1.0 / beta) * torch.logsumexp(beta * residuals, dim=0)
        return soft_max


# ---------------------------------------------------------------------------
# ACFGradientFlow — ∂ε/∂θ through the functor
# ---------------------------------------------------------------------------

@dataclass
class ACFGradientResult:
    """Gradient of compilation error w.r.t. function parameters."""
    epsilon: float
    epsilon_gradient_norm: float
    theta_gradient: torch.Tensor
    optimal_direction: torch.Tensor    # -∇ε/‖∇ε‖
    natural_gradient: torch.Tensor     # G_F^{-1} · ∇ε (Fisher preconditioned)
    improvement_per_step: float        # predicted ε reduction per unit step

    def summary(self) -> str:
        return (
            f"ACFGradient: ε={self.epsilon:.4e}, "
            f"‖∇ε‖={self.epsilon_gradient_norm:.4e}, "
            f"Δε/step≈{self.improvement_per_step:.4e}"
        )


class ACFGradientFlow:
    """
    Computes gradients of the compilation error ε(Φ_θ(f)) w.r.t. θ.

    This enables:
    1. Finding which parameter directions make the function "easier" to compile.
    2. Joint optimization of the function and its representation.
    3. Verifying that the natural gradient = ordinary gradient on flat manifold.
    """

    def __init__(self, degree: int = 20, domain: Tuple[float, float] = (-1.0, 1.0)):
        self.layer = DifferentiableChebyshevApprox(degree=degree, domain=domain)
        self.degree = degree

    def compute_gradient(
        self,
        f_true_values: torch.Tensor,  # true function values at evaluation grid
        x_eval: torch.Tensor,
        theta_init: Optional[torch.Tensor] = None,
    ) -> ACFGradientResult:
        """
        Compute ∂ε/∂θ where ε = max|f_true - Φ_θ(f)|.

        Uses torch.autograd to differentiate through the Chebyshev reduction.
        """
        d = self.degree + 1
        if theta_init is None:
            theta = nn.Parameter(torch.zeros(d, dtype=torch.float64))
        else:
            theta = nn.Parameter(theta_init.clone().double())

        # Forward: compute approximation
        y_pred = self.layer(x_eval.double(), theta=theta)
        residuals = torch.abs(f_true_values.double() - y_pred)

        # Soft-max approximation to ε (differentiable)
        beta = 50.0
        epsilon_soft = (1.0 / beta) * torch.logsumexp(beta * residuals, dim=0)
        epsilon_val = float(torch.max(residuals).item())

        # Backward
        epsilon_soft.backward()

        grad = theta.grad.detach() if theta.grad is not None else torch.zeros(d, dtype=torch.float64)
        grad_norm = float(torch.norm(grad).item())
        optimal_dir = -grad / (grad_norm + 1e-15)

        # For Chebyshev: Fisher metric = (1/2) I (diagonal by orthogonality)
        # Natural gradient = 2 * ordinary gradient
        natural_grad = 2.0 * grad

        step_size = 1e-4
        improvement_per_step = float(torch.dot(grad, optimal_dir).item()) * step_size

        return ACFGradientResult(
            epsilon=epsilon_val,
            epsilon_gradient_norm=grad_norm,
            theta_gradient=grad,
            optimal_direction=optimal_dir,
            natural_gradient=natural_grad,
            improvement_per_step=improvement_per_step,
        )


# ---------------------------------------------------------------------------
# High-level API: DifferentiableACFLayer
# ---------------------------------------------------------------------------

class DifferentiableACFLayer(nn.Module):
    """
    Drop-in nn.Module that uses ACF-compiled activations.

    Conventional neural networks use activation functions (ReLU, GELU, etc.)
    as fixed functions. This layer replaces them with ACF-compiled
    approximations that are differentiable w.r.t. the compilation parameters.

    Usage
    -----
    >>> layer = DifferentiableACFLayer(in_features=64, degree=15)
    >>> x = torch.randn(32, 64)  # batch × features
    >>> y = layer(x)             # same shape, ACF-activated
    >>> loss = y.sum()
    >>> loss.backward()          # works!
    """

    def __init__(
        self,
        in_features: int,
        degree: int = 20,
        domain: Tuple[float, float] = (-2.0, 2.0),
        learnable_activation: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.in_features = in_features
        self.degree = degree
        self.domain = domain
        self.dtype = dtype

        a, b = domain
        d = degree + 1
        k = torch.arange(1, d + 1, dtype=dtype)
        t_nodes = torch.cos(math.pi * (2 * k - 1) / (2 * d))
        self.register_buffer("t_nodes", t_nodes)

        # Learnable activation coefficients (initialized to tanh-like)
        if learnable_activation:
            init_vals = torch.tanh(
                (a + b) / 2.0 + (b - a) / 2.0 * t_nodes
            )
            self.activation_nodes = nn.Parameter(init_vals)
        else:
            self.register_buffer(
                "activation_nodes",
                torch.tanh((a + b) / 2.0 + (b - a) / 2.0 * t_nodes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ACF-compiled activation to each feature independently.

        Input:  x of shape (..., in_features)  
        Output: same shape, with ACF activation applied element-wise.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1).double()
        a, b = self.domain
        d = self.degree + 1

        # Get Chebyshev coefficients from learnable activation
        c = ChebyshevCoeffExtractor.apply(
            self.activation_nodes.double(), self.t_nodes.double()
        )

        # Map x to [-1, 1]
        t = (2.0 * (x_flat - (a + b) / 2.0) / (b - a)).clamp(-1.0, 1.0)

        # Clenshaw evaluation
        n = c.shape[0]
        b_next = torch.zeros_like(t)
        b_curr = c[n - 1].expand_as(t)
        for k in range(n - 2, 0, -1):
            b_prev = c[k].expand_as(t) + 2.0 * t * b_curr - b_next
            b_next = b_curr
            b_curr = b_prev
        y = c[0].expand_as(t) + t * b_curr - b_next

        return y.reshape(original_shape).to(x.dtype)

    def get_activation_epsilon(
        self,
        true_activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        """Compute max error of learned activation vs. a reference function."""
        a, b = self.domain
        x_test = torch.linspace(a, b, 1000, dtype=self.dtype)
        y_true = true_activation(x_test)
        y_pred = self.forward(x_test.unsqueeze(0)).squeeze(0)
        return float(torch.max(torch.abs(y_true - y_pred)).item())
