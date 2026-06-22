"""
acf_distributional_neuron.py — Neurona ACF Distribucional: f(x) → (μ(x), σ(x))
================================================================================

La neurona NO solo predice un punto — predice una DISTRIBUCIÓN.
Para cada entrada x, produce:
  - μ(x): la mejor predicción (media)
  - σ(x): la incertidumbre (desviación estándar)

Esto permite:
  - Cuantificar ambigüedad: σ alto = "hay varias respuestas posibles"
  - Detectar overfitting: σ bajo en train, σ alto en test = sobreajuste
  - Rechazar predicciones inseguras: si σ > umbral, decir "no sé"

Experimento: comparar contra MLP en tareas con ambigüedad inherente.

Autor: AXIOM-1
"""

import math, time, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, ".")
from acf_functor.neuron.acf_tensorial_neuron import ACFTensorialNeuron, evaluate_multivariate_chebyshev, generate_multivariate_chebyshev_indices


@dataclass
class DistributionalPrediction:
    """Predicción distribucional completa."""
    mean: np.ndarray          # μ(x) ∈ R^m
    std: np.ndarray           # σ(x) ∈ R^m (desviación estándar)
    variance: np.ndarray      # σ²(x) ∈ R^m

    @property
    def is_reliable(self) -> bool:
        """¿Es confiable esta predicción?"""
        # Confiable si la incertidumbre relativa es baja
        mean_mag = np.linalg.norm(self.mean)
        if mean_mag < 1e-10:
            return float(np.max(self.std)) < 1.0
        return float(np.max(self.std) / max(mean_mag, 1e-10)) < 2.0

    @property
    def max_uncertainty_dim(self) -> int:
        """Dimensión con mayor incertidumbre."""
        return int(np.argmax(self.std))

    def confidence_interval(self, dim: int, k: float = 2.0) -> Tuple[float, float]:
        """Intervalo de confianza para dimensión dim."""
        mu = float(self.mean[dim])
        s = float(self.std[dim])
        return (mu - k * s, mu + k * s)

    def __repr__(self):
        reliable = "✓" if self.is_reliable else "⚠"
        return (f"DistPred({reliable} μ={self.mean[:3]}..., "
                f"σ_max={np.max(self.std):.3f})")


class ACFDistributionalNeuron:
    """
    Neurona ACF Distribucional.

    Aprende dos funciones simultáneamente:
      - f_μ: R^n → R^m  (media — predicción puntual)
      - f_σ: R^n → R^m  (desviación estándar — incertidumbre)

    Ambas usan RLS sobre base de Chebyshev multivariada.
    f_σ se entrena para predecir el error absoluto |y - μ(x)|.
    """

    def __init__(self, name: str, n_input: int, n_output: int,
                 max_degree: int = 2, l2_lambda: float = 0.5,
                 domain_lo: Optional[np.ndarray] = None,
                 domain_hi: Optional[np.ndarray] = None):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output

        # Neurona para la media
        self.mu_neuron = ACFTensorialNeuron(
            f"{name}_mu", n_input, n_output, max_degree=max_degree,
            domain_lo=domain_lo, domain_hi=domain_hi,
            spectral_threshold=0.01, l2_lambda=l2_lambda,
        )
        # Neurona para la desviación estándar
        self.sigma_neuron = ACFTensorialNeuron(
            f"{name}_sigma", n_input, n_output, max_degree=max_degree,
            domain_lo=domain_lo, domain_hi=domain_hi,
            spectral_threshold=0.01, l2_lambda=l2_lambda,
        )

        # Estadísticas
        self.total_updates: int = 0
        self._error_history: List[float] = []
        self._sigma_history: List[float] = []

    def observe_and_learn(self, x: np.ndarray, y_target: np.ndarray) -> DistributionalPrediction:
        """
        Una observación. Dos aprendizajes (μ y σ).

        1. Predecir μ(x)
        2. Calcular error e = |y - μ(x)|
        3. Actualizar μ con RLS (target = y)
        4. Actualizar σ con RLS (target = e)
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y_target, dtype=np.float64).ravel()

        # 1. Predecir μ
        mu_pred = self.mu_neuron.evaluate(x)
        abs_error = np.abs(y - mu_pred)

        # 2. Actualizar μ → target real
        self.mu_neuron.observe_and_learn(x, y)

        # 3. Actualizar σ → target es el error absoluto
        self.sigma_neuron.observe_and_learn(x, abs_error)

        self.total_updates += 1
        self._error_history.append(float(np.mean(abs_error)))
        self._sigma_history.append(float(np.mean(self.sigma_neuron.evaluate(x))))

        return DistributionalPrediction(
            mean=mu_pred,
            std=np.maximum(self.sigma_neuron.evaluate(x), 1e-8),
            variance=np.maximum(self.sigma_neuron.evaluate(x)**2, 1e-16),
        )

    def predict(self, x: np.ndarray) -> DistributionalPrediction:
        """Predecir con distribución completa."""
        x = np.asarray(x, dtype=np.float64).ravel()
        mu = self.mu_neuron.evaluate(x)
        sigma = np.maximum(self.sigma_neuron.evaluate(x), 1e-8)
        return DistributionalPrediction(mean=mu, std=sigma, variance=sigma**2)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Predecir solo la media (compatible con ACFTensorialNeuron)."""
        return self.mu_neuron.evaluate(x)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 20) -> Dict:
        """Ajuste completo."""
        t0 = time.perf_counter()
        errors = []
        sigmas = []
        for ep in range(epochs):
            indices = np.random.permutation(len(X))
            ep_err = 0.0
            ep_sig = 0.0
            for idx in indices:
                pred = self.observe_and_learn(X[idx], Y[idx])
                ep_err += float(np.mean(np.abs(Y[idx] - pred.mean)))
                ep_sig += float(np.mean(pred.std))
            errors.append(ep_err / len(indices))
            sigmas.append(ep_sig / len(indices))
        elapsed = time.perf_counter() - t0
        return {
            "time": elapsed,
            "errors": errors,
            "sigmas": sigmas,
            "final_error": errors[-1],
            "final_sigma": sigmas[-1],
            "total_updates": self.total_updates,
            "mu_modes": self.mu_neuron.self_knowledge.n_modes,
            "sigma_modes": self.sigma_neuron.self_knowledge.n_modes,
        }

    def detect_ood(self, x: np.ndarray) -> Tuple[bool, str]:
        """OOD: fuera de dominio o incertidumbre muy alta."""
        # Check de dominio
        ood_domain, reason = self.mu_neuron.detect_ood(x)
        if ood_domain:
            return True, reason
        # Check de incertidumbre
        pred = self.predict(x)
        if not pred.is_reliable:
            return True, f"Alta incertidumbre: σ_max={np.max(pred.std):.3f}"
        return False, ""

    @property
    def self_knowledge(self):
        return {
            "mu": self.mu_neuron.self_knowledge,
            "sigma": self.sigma_neuron.self_knowledge,
            "total_updates": self.total_updates,
            "mean_error": float(np.mean(self._error_history[-50:])) if self._error_history else -1,
            "mean_sigma": float(np.mean(self._sigma_history[-50:])) if self._sigma_history else -1,
        }

    def summary(self) -> str:
        sk = self.self_knowledge
        return (
            f"ACFDistributionalNeuron('{self.name}')\n"
            f"  μ: {sk['mu'].n_modes} modos, ε={sk['mu'].global_epsilon:.2e}\n"
            f"  σ: {sk['sigma'].n_modes} modos, ε={sk['sigma'].global_epsilon:.2e}\n"
            f"  Error medio: {sk['mean_error']:.3f}\n"
            f"  σ medio: {sk['mean_sigma']:.3f}\n"
        )
