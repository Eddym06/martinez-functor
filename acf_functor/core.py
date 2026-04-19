"""
Affine Collapse Functor (ACF) Phi - Core Engine
==================================
Constructive implementations for:
- Exact polynomial reduction via Horner FMA sequences
- Transcendental approximation via Chebyshev polynomials
- Koopman-style linearization via EDMD
- Stratified execution for piecewise functions
- Compositional error tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import warnings

import numpy as np
import torch


class ReductionPath(Enum):
    HORNER_EXACT = auto()
    CHEBYSHEV_APPROX = auto()
    KOOPMAN_LINEAR = auto()
    STRATIFIED = auto()
    COMPOSITE = auto()


@dataclass
class FMAOperation:
    """Single y = W*x + b operator (scalar or matrix)."""

    weight: torch.Tensor
    bias: torch.Tensor

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dim() >= 2:
            if x.dim() == 1:
                return self.weight @ x + self.bias.reshape(-1)
            return x @ self.weight + self.bias
        return torch.addcmul(self.bias.expand_as(x), self.weight.expand_as(x), x)

    def to(self, device=None, dtype=None) -> "FMAOperation":
        return FMAOperation(
            weight=self.weight.to(device=device, dtype=dtype),
            bias=self.bias.to(device=device, dtype=dtype),
        )


@dataclass
class ReductionResult:
    path: ReductionPath
    fma_sequence: List[FMAOperation]
    computational_energy: int
    epsilon_bound: float
    domain: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for op in self.fma_sequence:
            out = op.execute(out)
        return out

    def to(self, device=None, dtype=None) -> "ReductionResult":
        return ReductionResult(
            path=self.path,
            fma_sequence=[op.to(device=device, dtype=dtype) for op in self.fma_sequence],
            computational_energy=self.computational_energy,
            epsilon_bound=self.epsilon_bound,
            domain=self.domain,
            metadata=self.metadata,
        )


class HornerReducer:
    """Exact polynomial reduction and evaluation."""

    @staticmethod
    def reduce(
        coefficients: Union[List[float], torch.Tensor, np.ndarray],
        dtype: torch.dtype = torch.float64,
    ) -> ReductionResult:
        coeffs = torch.as_tensor(coefficients, dtype=dtype)
        if coeffs.numel() == 0:
            raise ValueError("Empty coefficient list")

        degree = max(0, int(coeffs.numel() - 1))
        return ReductionResult(
            path=ReductionPath.HORNER_EXACT,
            fma_sequence=[],
            computational_energy=degree,
            epsilon_bound=0.0,
            metadata={
                "method": "horner_exact",
                "degree": degree,
                "coefficients": coeffs.tolist(),
            },
        )

    @staticmethod
    def execute_horner(
        coefficients: Union[List[float], torch.Tensor, np.ndarray],
        x: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = torch.as_tensor(coefficients, dtype=x.dtype, device=x.device)
        if coeffs.numel() == 0:
            return torch.zeros_like(x)

        out = torch.full_like(x, coeffs[-1].item())
        for i in range(coeffs.numel() - 2, -1, -1):
            out = torch.addcmul(torch.full_like(x, coeffs[i].item()), x, out)
        return out


class ChebyshevReducer:
    """Transcendental reduction with Chebyshev + Horner."""

    CANONICAL_FUNCTIONS: Dict[str, Dict[str, Any]] = {
        "sin": {"domain": (-math.pi, math.pi), "generator": torch.sin},
        "cos": {"domain": (-math.pi, math.pi), "generator": torch.cos},
        "exp": {"domain": (-1.0, 1.0), "generator": torch.exp},
        "log": {"domain": (0.5, 2.0), "generator": torch.log},
        "tanh": {"domain": (-3.0, 3.0), "generator": torch.tanh},
        "sigmoid": {"domain": (-6.0, 6.0), "generator": torch.sigmoid},
    }

    @staticmethod
    def _fit_chebyshev_orthogonal(
        func: Callable[[torch.Tensor], torch.Tensor],
        degree: int,
        domain: Tuple[float, float],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Numerically stable Chebyshev fit using orthogonal basis on Gauss nodes."""
        a, b = domain
        n_nodes = max(128, 4 * (degree + 1))
        k = torch.arange(n_nodes, dtype=dtype)
        theta = (k + 0.5) * math.pi / n_nodes
        t = torch.cos(theta)
        x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * t

        y = func(x_nodes)
        j = torch.arange(degree + 1, dtype=dtype).unsqueeze(1)
        cos_table = torch.cos(j * theta.unsqueeze(0))
        cheb_coeffs = (2.0 / n_nodes) * (cos_table @ y)
        cheb_coeffs[0] *= 0.5
        return cheb_coeffs

    @staticmethod
    def _chebyshev_to_monomial(
        cheb_coeffs: torch.Tensor,
        domain: Tuple[float, float],
    ) -> torch.Tensor:
        """Convert Chebyshev coefficients to monomial coefficients in x-domain."""
        n = int(cheb_coeffs.numel())
        dtype = cheb_coeffs.dtype
        a, b = domain

        # Build T_k(t) monomial coefficients in t using recurrence.
        # Rows: k, Cols: t^j coefficient.
        T = torch.zeros((n, n), dtype=dtype)
        T[0, 0] = 1.0
        if n > 1:
            T[1, 1] = 1.0
        for k in range(2, n):
            T[k, 1:] += 2.0 * T[k - 1, :-1]
            T[k, :] -= T[k - 2, :]

        mono_t = T.T @ cheb_coeffs

        # Affine map: t = alpha*x + beta
        alpha = 2.0 / (b - a)
        beta = -(a + b) / (b - a)
        mono_x = torch.zeros(n, dtype=dtype)

        # Expand (alpha*x + beta)^k recursively to avoid huge combinatorics.
        for k in range(n):
            ck = mono_t[k]
            if torch.abs(ck).item() < 1e-30:
                continue

            poly = torch.tensor([1.0], dtype=dtype)
            for _ in range(k):
                nxt = torch.zeros(poly.numel() + 1, dtype=dtype)
                nxt[1:] += alpha * poly
                nxt[:-1] += beta * poly
                poly = nxt

            mono_x[: k + 1] += ck * poly

        return mono_x

    @staticmethod
    def evaluate_chebyshev_series(
        cheb_coeffs: Union[List[float], torch.Tensor, np.ndarray],
        x: torch.Tensor,
        domain: Tuple[float, float],
    ) -> torch.Tensor:
        """Evaluate sum c_k T_k(t) with Clenshaw recurrence (stable for high degree)."""
        c = torch.as_tensor(cheb_coeffs, dtype=x.dtype, device=x.device)
        if c.numel() == 0:
            return torch.zeros_like(x)

        a, b = domain
        t = (2.0 * x - (a + b)) / (b - a)

        if c.numel() == 1:
            return torch.full_like(x, c[0].item())

        b_k1 = torch.zeros_like(t)
        b_k2 = torch.zeros_like(t)
        for idx in range(c.numel() - 1, 0, -1):
            b_k = 2.0 * t * b_k1 - b_k2 + c[idx]
            b_k2 = b_k1
            b_k1 = b_k
        return t * b_k1 - b_k2 + c[0]

    @staticmethod
    def certify_error(
        func: Callable[[torch.Tensor], torch.Tensor],
        coeffs: torch.Tensor,
        domain: Tuple[float, float],
        n_test_points: int = 10000,
        mode: str = "chebyshev",
    ) -> float:
        a, b = domain
        x = torch.linspace(a, b, n_test_points, dtype=coeffs.dtype)
        exact = func(x)
        if mode == "chebyshev":
            approx = ChebyshevReducer.evaluate_chebyshev_series(coeffs, x, domain)
        else:
            approx = HornerReducer.execute_horner(coeffs, x)
        return float(torch.max(torch.abs(exact - approx)).item())

    @classmethod
    def reduce(
        cls,
        func: Union[str, Callable[[torch.Tensor], torch.Tensor]],
        degree: int = 20,
        domain: Optional[Tuple[float, float]] = None,
        target_epsilon: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
    ) -> ReductionResult:
        if isinstance(func, str):
            key = func.lower().strip()
            if key not in cls.CANONICAL_FUNCTIONS:
                raise ValueError(f"Unknown canonical function '{func}'")
            info = cls.CANONICAL_FUNCTIONS[key]
            fn = info["generator"]
            if domain is None:
                domain = info["domain"]
        elif callable(func):
            if domain is None:
                raise ValueError("domain must be provided for callable functions")
            fn = func
            key = getattr(func, "__name__", "callable")
        else:
            raise TypeError("func must be a string or callable")

        max_degree = 256
        cur_degree = max(2, int(degree))
        coeffs = None
        eps = None

        while True:
            cheb_coeffs = cls._fit_chebyshev_orthogonal(fn, cur_degree, domain, dtype)
            eps = cls.certify_error(fn, cheb_coeffs, domain, mode="chebyshev")
            if target_epsilon is None or eps <= target_epsilon or cur_degree >= max_degree:
                break
            cur_degree += 4

        mono_coeffs = cls._chebyshev_to_monomial(cheb_coeffs, domain)
        if not torch.isfinite(mono_coeffs).all():
            warnings.warn(
                "Monomial conversion became ill-conditioned; keeping Chebyshev/Clenshaw as primary runtime path."
            )
            mono_coeffs = torch.zeros_like(mono_coeffs)

        if cur_degree >= 80:
            warnings.warn(
                "High Chebyshev degree detected. Prefer Clenshaw evaluation to avoid monomial conditioning issues."
            )

        return ReductionResult(
            path=ReductionPath.CHEBYSHEV_APPROX,
            fma_sequence=[],
            computational_energy=cur_degree,
            epsilon_bound=float(eps),
            domain=domain,
            metadata={
                "method": "chebyshev_orthogonal_clenshaw",
                "degree": cur_degree,
                "domain": domain,
                "chebyshev_coefficients": cheb_coeffs.tolist(),
                "monomial_coefficients": mono_coeffs.tolist(),
                "certified_epsilon": float(eps),
                "function": key,
                "evaluation_mode": "clenshaw",
            },
        )


class KoopmanReducer:
    """Finite-dimensional Koopman approximation via EDMD."""

    @staticmethod
    def polynomial_observables(x: torch.Tensor, max_degree: int = 2) -> torch.Tensor:
        n_features, n_steps = x.shape
        terms = [torch.ones(1, n_steps, dtype=x.dtype, device=x.device), x]
        if max_degree >= 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    terms.append(x[i : i + 1] * x[j : j + 1])
        if max_degree >= 3:
            for i in range(n_features):
                terms.append(x[i : i + 1] ** 3)
        return torch.cat(terms, dim=0)

    @staticmethod
    def fourier_observables(x: torch.Tensor, max_harmonic: int = 3) -> torch.Tensor:
        n_features, n_steps = x.shape
        terms = [torch.ones(1, n_steps, dtype=x.dtype, device=x.device), x]
        for h in range(1, max_harmonic + 1):
            for i in range(n_features):
                terms.append(torch.sin(h * x[i : i + 1]))
                terms.append(torch.cos(h * x[i : i + 1]))
        return torch.cat(terms, dim=0)

    @staticmethod
    def rbf_observables(x: torch.Tensor, n_centers: int = 8) -> torch.Tensor:
        n_features, n_steps = x.shape
        terms = [torch.ones(1, n_steps, dtype=x.dtype, device=x.device), x]
        var = torch.var(x, dim=1, keepdim=True) + 1e-8
        gamma = 1.0 / var
        for i in range(n_features):
            lo = x[i].min()
            hi = x[i].max()
            centers = torch.linspace(lo, hi, n_centers, dtype=x.dtype, device=x.device)
            xi = x[i : i + 1]
            for c in centers:
                terms.append(torch.exp(-gamma[i : i + 1] * (xi - c) ** 2))
        return torch.cat(terms, dim=0)

    @staticmethod
    def mixed_observables(
        x: torch.Tensor,
        poly_degree: int = 2,
        max_harmonic: int = 2,
        n_centers: int = 4,
    ) -> torch.Tensor:
        poly = KoopmanReducer.polynomial_observables(x, max_degree=poly_degree)
        fourier = KoopmanReducer.fourier_observables(x, max_harmonic=max_harmonic)
        rbf = KoopmanReducer.rbf_observables(x, n_centers=n_centers)
        return torch.cat([poly, fourier, rbf], dim=0)

    @staticmethod
    def _solve_koopman(
        psi_x: torch.Tensor,
        psi_y: torch.Tensor,
        rank: Optional[int],
        reg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        U, S, Vh = torch.linalg.svd(psi_x, full_matrices=False)
        if rank is not None:
            r = min(int(rank), S.shape[0])
            U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        S_inv = S / (S**2 + reg)
        K = psi_y @ Vh.T @ torch.diag(S_inv) @ U.T
        pred = K @ psi_x
        rec_error = float((torch.norm(psi_y - pred) / (torch.norm(psi_y) + 1e-15)).item())
        eigvals = torch.linalg.eigvals(K)
        return K, eigvals, rec_error

    @staticmethod
    def _library_fn(
        library: str,
        poly_degree: int,
        n_fourier: int,
        n_rbf: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if library == "polynomial":
            return lambda z: KoopmanReducer.polynomial_observables(z, max_degree=poly_degree)
        if library == "fourier":
            return lambda z: KoopmanReducer.fourier_observables(z, max_harmonic=n_fourier)
        if library == "rbf":
            return lambda z: KoopmanReducer.rbf_observables(z, n_centers=n_rbf)
        if library == "mixed":
            return lambda z: KoopmanReducer.mixed_observables(
                z,
                poly_degree=poly_degree,
                max_harmonic=n_fourier,
                n_centers=n_rbf,
            )
        raise ValueError(f"Unknown observable library '{library}'")

    @staticmethod
    def dmd(
        snapshots: torch.Tensor,
        observable_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        rank: Optional[int] = None,
        reg: float = 1e-10,
        observable_library: str = "auto",
        poly_degree: int = 2,
        n_fourier: int = 3,
        n_rbf: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        X = snapshots[:, :-1]
        Y = snapshots[:, 1:]

        if observable_fn is not None:
            selected_library = "custom"
            psi_x = observable_fn(X)
            psi_y = observable_fn(Y)
            K, eigvals, rec_error = KoopmanReducer._solve_koopman(psi_x, psi_y, rank, reg)
            candidate_errors = {}
        elif observable_library == "auto":
            candidates = ["polynomial", "fourier", "mixed", "rbf"]
            best = None
            best_score = float("inf")
            candidate_errors = {}
            selected_library = "polynomial"

            for lib in candidates:
                fn = KoopmanReducer._library_fn(lib, poly_degree, n_fourier, n_rbf)
                psi_x_try = fn(X)
                psi_y_try = fn(Y)
                K_try, eig_try, err_try = KoopmanReducer._solve_koopman(psi_x_try, psi_y_try, rank, reg)
                score = err_try + 1e-8 * psi_x_try.shape[0]
                candidate_errors[lib] = err_try
                if score < best_score:
                    best_score = score
                    best = (K_try, eig_try, err_try, psi_x_try)
                    selected_library = lib

            K, eigvals, rec_error, psi_x = best
        else:
            selected_library = observable_library
            fn = KoopmanReducer._library_fn(observable_library, poly_degree, n_fourier, n_rbf)
            psi_x = fn(X)
            psi_y = fn(Y)
            K, eigvals, rec_error = KoopmanReducer._solve_koopman(psi_x, psi_y, rank, reg)
            candidate_errors = {}

        if rec_error > 1e-2:
            warnings.warn(
                "High EDMD reconstruction error. Consider richer observables (mixed/rbf) or larger rank."
            )

        meta = {
            "method": "edmd",
            "observable_dim": int(psi_x.shape[0]),
            "state_dim": int(X.shape[0]),
            "n_snapshots": int(X.shape[1]),
            "reconstruction_error": rec_error,
            "spectral_radius": float(torch.max(torch.abs(eigvals)).item()),
            "observable_library": selected_library,
            "candidate_library_errors": candidate_errors,
        }
        return K, eigvals, meta

    @classmethod
    def reduce(
        cls,
        trajectory_data: torch.Tensor,
        observable_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        rank: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
        observable_library: str = "auto",
        poly_degree: int = 2,
        n_fourier: int = 3,
        n_rbf: int = 8,
    ) -> ReductionResult:
        data = trajectory_data.to(dtype)
        K, eigvals, meta = cls.dmd(
            data,
            observable_fn=observable_fn,
            rank=rank,
            observable_library=observable_library,
            poly_degree=poly_degree,
            n_fourier=n_fourier,
            n_rbf=n_rbf,
        )

        op = FMAOperation(weight=K, bias=torch.zeros((1, K.shape[1]), dtype=dtype))
        sorted_abs = torch.sort(torch.abs(eigvals), descending=True).values
        delta_d = float(sorted_abs[-1].item()) if sorted_abs.numel() else 0.0

        return ReductionResult(
            path=ReductionPath.KOOPMAN_LINEAR,
            fma_sequence=[op],
            computational_energy=int(K.numel()),
            epsilon_bound=meta["reconstruction_error"],
            metadata={**meta, "koopman_eigenvalues": eigvals.tolist(), "truncation_delta": delta_d},
        )


class StratifiedReducer:
    @staticmethod
    def reduce_relu(dtype: torch.dtype = torch.float64) -> ReductionResult:
        return ReductionResult(
            path=ReductionPath.STRATIFIED,
            fma_sequence=[],
            computational_energy=2,
            epsilon_bound=0.0,
            metadata={"method": "stratified_relu"},
        )


class ACFInvariant:
    @staticmethod
    def compute_alpha(eigenvalues: Union[torch.Tensor, np.ndarray]) -> Tuple[float, float]:
        eig = torch.as_tensor(eigenvalues, dtype=torch.float64)
        eig = torch.sort(torch.abs(eig), descending=True).values
        eig = eig[eig > 1e-15]
        if eig.numel() < 2:
            return 0.0, 0.0

        j = torch.arange(1, eig.numel() + 1, dtype=torch.float64)
        alpha = float(torch.max(-torch.log(eig) / torch.log(j + 1.0)).item())
        delta_d = float(eig[-1].item())
        return alpha, delta_d


class EnrichedFunctor:
    @staticmethod
    def compose(phi_f: ReductionResult, phi_g: ReductionResult) -> ReductionResult:
        eps_f, eps_g = phi_f.epsilon_bound, phi_g.epsilon_bound
        eps_cross = eps_f * eps_g
        return ReductionResult(
            path=ReductionPath.COMPOSITE,
            fma_sequence=phi_g.fma_sequence + phi_f.fma_sequence,
            computational_energy=phi_f.computational_energy + phi_g.computational_energy,
            epsilon_bound=eps_f + eps_g + eps_cross,
            metadata={
                "method": "functorial_composition",
                "inner_path": phi_g.path.name,
                "outer_path": phi_f.path.name,
                "epsilon_f": eps_f,
                "epsilon_g": eps_g,
                "epsilon_cross": eps_cross,
            },
        )

    @staticmethod
    def verify_conservation(original_energy: int, reduced_energy: int) -> bool:
        return int(original_energy) == int(reduced_energy)


class ACFFunctor:
    """Unified facade for all reduction paths."""

    def __init__(self, default_dtype: torch.dtype = torch.float64):
        self.dtype = default_dtype
        self._reductions: Dict[str, ReductionResult] = {}

    def reduce_polynomial(self, coefficients: Union[List[float], torch.Tensor, np.ndarray]) -> ReductionResult:
        out = HornerReducer.reduce(coefficients, dtype=self.dtype)
        self._reductions["last_polynomial"] = out
        return out

    def reduce_transcendental(
        self,
        func: Union[str, Callable[[torch.Tensor], torch.Tensor]],
        degree: int = 20,
        domain: Optional[Tuple[float, float]] = None,
        target_epsilon: Optional[float] = None,
    ) -> ReductionResult:
        out = ChebyshevReducer.reduce(
            func,
            degree=degree,
            domain=domain,
            target_epsilon=target_epsilon,
            dtype=self.dtype,
        )
        self._reductions["last_transcendental"] = out
        return out

    def reduce_dynamical_system(
        self,
        trajectory: torch.Tensor,
        observable_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        rank: Optional[int] = None,
        observable_library: str = "auto",
        poly_degree: int = 2,
        n_fourier: int = 3,
        n_rbf: int = 8,
    ) -> ReductionResult:
        out = KoopmanReducer.reduce(
            trajectory,
            observable_fn=observable_fn,
            rank=rank,
            dtype=self.dtype,
            observable_library=observable_library,
            poly_degree=poly_degree,
            n_fourier=n_fourier,
            n_rbf=n_rbf,
        )
        self._reductions["last_koopman"] = out
        return out

    def reduce_piecewise(self, name: str = "relu") -> ReductionResult:
        if name.lower() != "relu":
            raise ValueError(f"Unknown piecewise function: {name}")
        out = StratifiedReducer.reduce_relu(dtype=self.dtype)
        self._reductions["last_stratified"] = out
        return out

    def compose(self, phi_f: ReductionResult, phi_g: ReductionResult) -> ReductionResult:
        out = EnrichedFunctor.compose(phi_f, phi_g)
        self._reductions["last_composition"] = out
        return out

    def compute_invariant(self, eigenvalues: Union[torch.Tensor, np.ndarray]) -> Tuple[float, float]:
        return ACFInvariant.compute_alpha(eigenvalues)

    def verify_conservation(self, f_energy: int, reduction: ReductionResult) -> bool:
        return EnrichedFunctor.verify_conservation(f_energy, reduction.computational_energy)

    def evaluate(self, reduction: ReductionResult, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        x_dev = x.to(device)
        if reduction.path == ReductionPath.HORNER_EXACT:
            coeffs = reduction.metadata.get("coefficients")
            return HornerReducer.execute_horner(coeffs, x_dev)

        if reduction.path == ReductionPath.CHEBYSHEV_APPROX:
            cheb = reduction.metadata.get("chebyshev_coefficients")
            if cheb is not None and reduction.domain is not None:
                return ChebyshevReducer.evaluate_chebyshev_series(cheb, x_dev, reduction.domain)
            coeffs = reduction.metadata.get("monomial_coefficients", [])
            return HornerReducer.execute_horner(coeffs, x_dev)

        if reduction.path == ReductionPath.KOOPMAN_LINEAR:
            op = reduction.fma_sequence[0].to(device=device, dtype=x_dev.dtype)
            return op.execute(x_dev)

        if reduction.path == ReductionPath.STRATIFIED:
            return torch.relu(x_dev)

        if reduction.path == ReductionPath.COMPOSITE:
            # Conservative execution path: execute by sequence if available.
            if reduction.fma_sequence:
                out = x_dev
                for op in reduction.fma_sequence:
                    out = op.to(device=device, dtype=x_dev.dtype).execute(out)
                return out
            return x_dev

        return reduction.execute(x_dev)


# Canonical ACF naming (preferred) with backward-compatible legacy aliases.
