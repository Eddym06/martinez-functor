"""
Adaptive Koopman Reduction Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math
import warnings

import torch

from .core import (
    FMAOperation,
    ReductionPath,
    ReductionResult,
    KoopmanReducer,
    ACFInvariant,
)


@dataclass
class SpectralDiagnostics:
    eigenvalues: torch.Tensor
    alpha: float
    delta: float
    optimal_rank: int
    spectral_gap: float
    spectral_radius: float
    reconstruction_error: float
    effective_dimension: int
    condition_number: float
    regularization_used: float


class ObservableLibrary:
    @staticmethod
    def polynomial(x: torch.Tensor, max_degree: int = 3) -> torch.Tensor:
        n_features, n_samples = x.shape
        obs = [torch.ones(1, n_samples, dtype=x.dtype, device=x.device), x]

        if max_degree >= 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    obs.append(x[i : i + 1] * x[j : j + 1])

        if max_degree >= 3:
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        obs.append(x[i : i + 1] * x[j : j + 1] * x[k : k + 1])

        if max_degree >= 4:
            for i in range(n_features):
                obs.append(x[i : i + 1] ** 4)

        return torch.cat(obs, dim=0)

    @staticmethod
    def fourier(x: torch.Tensor, n_harmonics: int = 5, period: float = 2.0 * math.pi) -> torch.Tensor:
        n_features, n_samples = x.shape
        obs = [torch.ones(1, n_samples, dtype=x.dtype, device=x.device)]
        for i in range(n_features):
            for k in range(1, n_harmonics + 1):
                freq = 2.0 * math.pi * k / period
                obs.append(torch.cos(freq * x[i : i + 1]))
                obs.append(torch.sin(freq * x[i : i + 1]))
        return torch.cat(obs, dim=0)

    @staticmethod
    def radial_basis(x: torch.Tensor, n_centers: int = 10, width: float = 1.0) -> torch.Tensor:
        n_features, n_samples = x.shape
        obs = [torch.ones(1, n_samples, dtype=x.dtype, device=x.device), x]

        for i in range(n_features):
            lo = x[i].min().item()
            hi = x[i].max().item()
            centers = torch.linspace(lo, hi, n_centers, dtype=x.dtype, device=x.device)
            for c in centers:
                obs.append(torch.exp(-((x[i : i + 1] - c) ** 2) / (2.0 * width**2)))

        return torch.cat(obs, dim=0)

    @staticmethod
    def mixed(x: torch.Tensor, poly_degree: int = 2, n_harmonics: int = 3) -> torch.Tensor:
        poly = ObservableLibrary.polynomial(x, max_degree=poly_degree)
        four = ObservableLibrary.fourier(x, n_harmonics=n_harmonics)
        return torch.cat([poly, four[1:]], dim=0)


class AdaptiveKoopman:
    def __init__(
        self,
        observable_families: Optional[List[str]] = None,
        max_rank: Optional[int] = None,
        spectral_threshold: float = 1e-6,
        max_poly_degree: int = 4,
        max_fourier_harmonics: int = 5,
        dtype: torch.dtype = torch.float64,
    ):
        self.observable_families = observable_families or ["polynomial", "fourier", "mixed"]
        self.max_rank = max_rank
        self.spectral_threshold = spectral_threshold
        self.max_poly_degree = max_poly_degree
        self.max_fourier_harmonics = max_fourier_harmonics
        self.dtype = dtype

    def reduce(self, trajectory: torch.Tensor) -> Tuple[ReductionResult, SpectralDiagnostics]:
        data = trajectory.to(self.dtype)
        n_features, _ = data.shape

        best_result: Optional[ReductionResult] = None
        best_diag: Optional[SpectralDiagnostics] = None
        best_score = float("inf")

        for family in self.observable_families:
            try:
                obs_fn = self._get_observable_fn(family, n_features)
                result, diag = self._reduce_with_observables(data, obs_fn, family)
                score = diag.reconstruction_error + 1e-3 * diag.optimal_rank
                if score < best_score:
                    best_score = score
                    best_result = result
                    best_diag = diag
            except Exception as exc:
                warnings.warn(f"observable family '{family}' failed: {exc}")

        if best_result is None or best_diag is None:
            raise RuntimeError("all observable families failed")

        return best_result, best_diag

    def _get_observable_fn(self, family: str, n_features: int) -> Callable[[torch.Tensor], torch.Tensor]:
        if family == "polynomial":
            degree = min(self.max_poly_degree, max(2, 20 // max(1, n_features)))
            return lambda x: ObservableLibrary.polynomial(x, max_degree=degree)
        if family == "fourier":
            return lambda x: ObservableLibrary.fourier(x, n_harmonics=self.max_fourier_harmonics)
        if family == "radial":
            return lambda x: ObservableLibrary.radial_basis(x)
        if family == "mixed":
            degree = min(self.max_poly_degree, max(2, 15 // max(1, n_features)))
            harmonics = min(self.max_fourier_harmonics, 3)
            return lambda x: ObservableLibrary.mixed(x, poly_degree=degree, n_harmonics=harmonics)
        raise ValueError(f"unknown observable family: {family}")

    def _reduce_with_observables(
        self,
        data: torch.Tensor,
        obs_fn: Callable[[torch.Tensor], torch.Tensor],
        family_name: str,
    ) -> Tuple[ReductionResult, SpectralDiagnostics]:
        X = data[:, :-1]
        Y = data[:, 1:]

        Psi_X = obs_fn(X)
        Psi_Y = obs_fn(Y)

        U, S, Vh = torch.linalg.svd(Psi_X, full_matrices=False)
        rank, spectral_gap = self._find_optimal_rank(S)
        reg = self._compute_regularization(S, rank)

        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        S_inv = S_r / (S_r**2 + reg)
        K = Psi_Y @ Vh_r.T @ torch.diag(S_inv) @ U_r.T

        Psi_pred = K @ Psi_X
        recon_error = float((torch.norm(Psi_Y - Psi_pred) / (torch.norm(Psi_Y) + 1e-15)).item())

        eigvals = torch.linalg.eigvals(K)
        eig_abs = torch.sort(torch.abs(eigvals), descending=True).values
        alpha, delta = ACFInvariant.compute_alpha(eig_abs)
        spectral_radius = float(torch.max(torch.abs(eigvals)).item())

        effective_dim = int((eig_abs > self.spectral_threshold).sum().item())
        cond = float((S[0] / S[min(rank - 1, S.numel() - 1)]).item()) if rank > 0 else float("inf")

        result = ReductionResult(
            path=ReductionPath.KOOPMAN_LINEAR,
            fma_sequence=[FMAOperation(weight=K, bias=torch.zeros((1, K.shape[1]), dtype=self.dtype))],
            computational_energy=int(rank * rank),
            epsilon_bound=recon_error,
            metadata={
                "method": "adaptive_koopman",
                "observable_family": family_name,
                "observable_dim": int(Psi_X.shape[0]),
                "optimal_rank": rank,
                "regularization": reg,
                "reconstruction_error": recon_error,
                "acf_alpha": alpha,
                "truncation_delta": delta,
                "effective_dimension": effective_dim,
                "spectral_gap": spectral_gap,
                "condition_number": cond,
                "koopman_eigenvalues": eigvals.tolist(),
            },
        )

        diag = SpectralDiagnostics(
            eigenvalues=eigvals,
            alpha=alpha,
            delta=delta,
            optimal_rank=rank,
            spectral_gap=spectral_gap,
            spectral_radius=spectral_radius,
            reconstruction_error=recon_error,
            effective_dimension=effective_dim,
            condition_number=cond,
            regularization_used=reg,
        )
        return result, diag

    def _find_optimal_rank(self, singular_values: torch.Tensor) -> Tuple[int, float]:
        n = singular_values.numel()
        if n <= 1:
            return n, 0.0

        if self.max_rank is not None:
            n = min(n, self.max_rank)

        S = singular_values[:n]
        gaps = torch.zeros(max(n - 1, 1), dtype=S.dtype)
        if n > 1:
            for i in range(n - 1):
                if S[i] > 1e-15:
                    gaps[i] = (S[i] - S[i + 1]) / S[i]

        above_threshold = int((S > self.spectral_threshold * S[0]).sum().item())
        if n > 1:
            idx = int(torch.argmax(gaps).item())
            gval = float(gaps[idx].item())
            rank_by_gap = idx + 1 if gval > 0.1 else above_threshold
        else:
            rank_by_gap = above_threshold
            gval = 0.0

        rank = max(min(rank_by_gap, n), 1)
        return rank, gval

    def _compute_regularization(self, singular_values: torch.Tensor, rank: int) -> float:
        if rank >= singular_values.numel():
            return 1e-12
        noise = singular_values[rank:]
        noise_floor = float(noise.mean().item()) if noise.numel() else 1e-12
        return max(noise_floor**2, 1e-14)

    def predict_trajectory(
        self,
        result: ReductionResult,
        initial_state: torch.Tensor,
        observable_fn: Callable[[torch.Tensor], torch.Tensor],
        n_steps: int,
        n_features: int,
    ) -> Tuple[torch.Tensor, List[float]]:
        K = result.fma_sequence[0].weight
        preds = torch.zeros(n_features, n_steps, dtype=self.dtype)
        preds[:, 0] = initial_state.reshape(-1)[:n_features]

        residuals: List[float] = []
        state = initial_state.reshape(-1, 1).to(self.dtype)

        for t in range(1, n_steps):
            psi = observable_fn(state)
            psi_next = K @ psi
            state_next = psi_next[1 : n_features + 1]
            preds[:, t] = state_next.reshape(-1)

            psi_reproj = observable_fn(state_next)
            residual = float((torch.norm(psi_next - psi_reproj) / (torch.norm(psi_next) + 1e-15)).item())
            residuals.append(residual)
            state = state_next

        return preds, residuals


def adaptive_koopman_reduce(trajectory: torch.Tensor, **kwargs) -> Tuple[ReductionResult, SpectralDiagnostics]:
    return AdaptiveKoopman(**kwargs).reduce(trajectory)
