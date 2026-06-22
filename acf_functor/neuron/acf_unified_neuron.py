"""
acf_unified_neuron.py — Neurona ACF Unificada: μ y σ desde UNA sola base
=========================================================================

Fusión real (no router) de ACF tensorial y ACF distribucional.
Una sola neurona que evalúa la base de Chebyshev UNA vez y produce
simultáneamente media μ(x) e incertidumbre σ(x).

Ventajas sobre ACF-Dist (2 neuronas separadas):
  - 1 evaluación de ψ(x) en lugar de 2
  - 1 matriz P en lugar de 2
  - 1 paso RLS en lugar de 2
  - Modos compartidos: sinergia natural entre μ y σ

Arquitectura:
  [C_μ | C_σ] ∈ R^{2m × N}  donde N = |base Chebyshev|
  P ∈ R^{N × N} compartida
  y_pred = [μ(x) | σ(x)] = [C_μ | C_σ] · ψ(x)

Autor: AXIOM-1
"""

import math, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .acf_tensorial_neuron import (
    generate_multivariate_chebyshev_indices,
    evaluate_multivariate_chebyshev,
    TensorialMode, TensorialKnowledge,
)


@dataclass
class UnifiedPrediction:
    """Predicción unificada: media + incertidumbre."""
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray

    @property
    def is_reliable(self) -> bool:
        mag = np.linalg.norm(self.mean)
        if mag < 1e-10:
            return float(np.max(self.std)) < 1.0
        return float(np.max(self.std) / max(mag, 1e-10)) < 2.0

    @property
    def max_uncertainty_dim(self) -> int:
        return int(np.argmax(self.std))

    def confidence_interval(self, dim: int, k: float = 2.0) -> Tuple[float, float]:
        mu = float(self.mean[dim]); s = float(self.std[dim])
        return (mu - k * s, mu + k * s)

    def __repr__(self):
        r = "✓" if self.is_reliable else "⚠"
        return f"UniPred({r} μ_max={np.max(self.mean):.3f}, σ_max={np.max(self.std):.3f})"


class ACFUnifiedNeuron:
    """
    Neurona ACF Unificada — μ y σ desde una sola base.

    f(x) = [μ(x) | σ(x)] = [C_μ | C_σ] · ψ(x)

    donde ψ(x) es la base de Chebyshev multivariada (evaluada UNA vez)
    y C_μ, C_σ comparten la misma matriz de covarianza P.
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 max_degree: int = 2, l2_lambda: float = 0.5,
                 domain_lo: Optional[np.ndarray] = None,
                 domain_hi: Optional[np.ndarray] = None,
                 spectral_threshold: float = 0.01):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.max_degree = max_degree
        self.domain_lo = domain_lo if domain_lo is not None else np.full(n_input, -2.0)
        self.domain_hi = domain_hi if domain_hi is not None else np.full(n_input, 2.0)
        self.spectral_threshold = spectral_threshold
        self.l2_lambda = l2_lambda

        # Base compartida
        self._indices = generate_multivariate_chebyshev_indices(n_input, max_degree)
        self.n_basis = len(self._indices)

        # Matriz unificada: [C_μ; C_σ] ∈ R^{2m × N}
        self.C_mu = np.zeros((n_output, self.n_basis))
        self.C_sigma = np.zeros((n_output, self.n_basis))

        # Matriz P compartida
        self.P = np.eye(self.n_basis) / l2_lambda
        self.forgetting_factor = 0.995

        # Modos
        self.modes_mu: List[TensorialMode] = []
        self.modes_sigma: List[TensorialMode] = []

        # Métricas
        self.epsilon_mu = float("inf")
        self.epsilon_sigma = float("inf")
        self.alpha_A = 0.0
        self.total_updates = 0

        self._error_window: List[float] = []
        self._sigma_window: List[float] = []

    # ═══════════════════════════════════════════════════════════════
    # API PRINCIPAL
    # ═══════════════════════════════════════════════════════════════

    def observe_and_learn(self, x: np.ndarray, y_target: np.ndarray) -> UnifiedPrediction:
        """
        Una observación, UN paso de RLS, DOS salidas (μ y σ).

        1. Evaluar ψ(x) — UNA SOLA VEZ
        2. Predecir μ, calcular error
        3. Actualizar C_μ y C_σ simultáneamente con RLS compartido
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y_target, dtype=np.float64).ravel()

        # 1. Evaluar base (UNA VEZ)
        psi = evaluate_multivariate_chebyshev(x, self._indices,
                                              self.domain_lo, self.domain_hi)

        # 2. Predecir μ
        mu_pred = self.C_mu @ psi
        abs_error = np.abs(y - mu_pred)

        # 3. Calcular ganancia de Kalman (COMPARTIDA)
        P_psi = self.P @ psi
        denom = self.forgetting_factor + float(psi @ P_psi)

        if denom > 1e-16:
            kalman_gain = P_psi / denom

            # Actualizar C_μ y C_σ con la MISMA ganancia
            for j in range(self.n_output):
                self.C_mu[j] += kalman_gain * (y[j] - mu_pred[j])
                self.C_sigma[j] += kalman_gain * (abs_error[j] - self.C_sigma[j] @ psi)

            # Actualizar P (COMPARTIDA — UNA SOLA VEZ)
            self.P = (self.P - np.outer(kalman_gain, P_psi)) / self.forgetting_factor

        self.total_updates += 1
        self._error_window.append(float(np.mean(abs_error)))
        self._sigma_window.append(float(np.mean(np.maximum(self.C_sigma @ psi, 0))))

        # Poda periódica
        if self.total_updates % 100 == 0:
            self._prune_modes()

        # Actualizar métricas
        self._update_metrics()

        sigma_pred = np.maximum(self.C_sigma @ psi, 1e-8)
        return UnifiedPrediction(mean=mu_pred, std=sigma_pred, variance=sigma_pred**2)

    def predict(self, x: np.ndarray) -> UnifiedPrediction:
        """Predecir (μ, σ) sin actualizar."""
        x = np.asarray(x, dtype=np.float64).ravel()
        psi = evaluate_multivariate_chebyshev(x, self._indices,
                                              self.domain_lo, self.domain_hi)
        mu = self.C_mu @ psi
        sigma = np.maximum(self.C_sigma @ psi, 1e-8)
        return UnifiedPrediction(mean=mu, std=sigma, variance=sigma**2)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Predecir solo μ (compatibilidad con API estándar)."""
        return self.predict(x).mean

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 20) -> Dict:
        """Ajuste iterativo."""
        t0 = time.perf_counter()
        errors, sigmas = [], []
        for ep in range(epochs):
            indices = np.random.permutation(len(X))
            ep_err = 0.0; ep_sig = 0.0
            for idx in indices:
                pred = self.observe_and_learn(X[idx], Y[idx])
                ep_err += float(np.mean(np.abs(Y[idx] - pred.mean)))
                ep_sig += float(np.mean(pred.std))
            errors.append(ep_err / len(indices))
            sigmas.append(ep_sig / len(indices))
        elapsed = time.perf_counter() - t0
        self._prune_modes()
        return {
            "time": elapsed, "errors": errors, "sigmas": sigmas,
            "final_error": errors[-1], "final_sigma": sigmas[-1],
            "total_updates": self.total_updates,
            "mu_modes": len(self.modes_mu), "sigma_modes": len(self.modes_sigma),
            "epsilon_mu": self.epsilon_mu, "epsilon_sigma": self.epsilon_sigma,
        }

    # ═══════════════════════════════════════════════════════════════
    # PODA Y MÉTRICAS
    # ═══════════════════════════════════════════════════════════════

    def _prune_modes(self):
        """Podar modos de μ y σ."""
        self.modes_mu, self.modes_sigma = [], []
        for k, multi_idx in enumerate(self._indices):
            v_mu = self.C_mu[:, k]; sigma_mu = float(np.linalg.norm(v_mu))
            v_sig = self.C_sigma[:, k]; sigma_sig = float(np.linalg.norm(v_sig))
            if sigma_mu > self.spectral_threshold:
                self.modes_mu.append(TensorialMode(multi_index=multi_idx, coefficient=sigma_mu,
                                                    output_vector=v_mu / max(sigma_mu, 1e-16),
                                                    epsilon=float(np.sqrt(abs(self.P[k, k])))))
            if sigma_sig > self.spectral_threshold:
                self.modes_sigma.append(TensorialMode(multi_index=multi_idx, coefficient=sigma_sig,
                                                       output_vector=v_sig / max(sigma_sig, 1e-16),
                                                       epsilon=float(np.sqrt(abs(self.P[k, k])))))
        self.modes_mu.sort(key=lambda m: m.coefficient, reverse=True)
        self.modes_sigma.sort(key=lambda m: m.coefficient, reverse=True)

    def _update_metrics(self):
        if len(self._error_window) >= 10:
            self.epsilon_mu = float(np.mean(self._error_window[-20:]))
        if len(self._sigma_window) >= 10:
            self.epsilon_sigma = float(np.mean(self._sigma_window[-20:]))

    def detect_ood(self, x: np.ndarray) -> Tuple[bool, str]:
        x = np.asarray(x, dtype=np.float64).ravel()
        for i in range(self.n_input):
            if x[i] < self.domain_lo[i] or x[i] > self.domain_hi[i]:
                return True, f"Dim {i}: x[{i}]={x[i]:.3f} ∉ [{self.domain_lo[i]:.2f}, {self.domain_hi[i]:.2f}]"
        pred = self.predict(x)
        if not pred.is_reliable:
            return True, f"Alta incertidumbre: σ_max={np.max(pred.std):.3f}"
        return False, ""

    @property
    def self_knowledge(self) -> Dict:
        return {
            "n_input": self.n_input, "n_output": self.n_output,
            "n_basis": self.n_basis, "max_degree": self.max_degree,
            "mu_modes": len(self.modes_mu), "sigma_modes": len(self.modes_sigma),
            "epsilon_mu": self.epsilon_mu, "epsilon_sigma": self.epsilon_sigma,
            "total_updates": self.total_updates,
        }

    def summary(self) -> str:
        sk = self.self_knowledge
        return (
            f"ACFUnifiedNeuron('{self.name}')\n"
            f"  Base: {sk['n_basis']} términos (grado ≤ {sk['max_degree']})\n"
            f"  μ: {sk['mu_modes']} modos, ε={sk['epsilon_mu']:.2e}\n"
            f"  σ: {sk['sigma_modes']} modos, ε={sk['epsilon_sigma']:.2e}\n"
            f"  P compartida, {sk['total_updates']} actualizaciones"
        )
