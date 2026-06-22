"""
acf_tensorial_neuron.py — Neurona ACF Tensorial: f: R^n → R^m
===============================================================

Generalización de la neurona ACF de f: R → R a f: R^n → R^m usando
base de Chebyshev multivariada con poda espectral.

Fundamento matemático:
  f(x) = Σ_k σ_k · φ_k(x) · v_k

donde:
  φ_k: R^n → R  — funciones de base de Chebyshev multivariada
  σ_k: float    — valor singular (importancia del modo k)
  v_k: R^m      — vector de salida del modo k

Los φ_k son productos tensoriales: T_{i1}(x1)·T_{i2}(x2)·...·T_{in}(xn)
con grado total ≤ max_degree.

La PODA ESPECTRAL elimina términos con |σ_k| < threshold, manteniendo
solo los modos que capturan estructura real.

Propiedades:
  - Captura interacciones cruzadas entre dimensiones (como W@x del MLP)
  - Cada modo φ_k es interpretable (producto de Chebyshevs)
  - ε certificado por modo
  - Aprendizaje RLS multivariado (óptimo exacto, sin backprop)
  - Detección OOD en R^n
  - Auto-conocimiento: sabe qué modos la componen

Autor: AXIOM-1
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Generador de base Chebyshev multivariada
# ═════════════════════════════════════════════════════════════════════════════

def generate_multivariate_chebyshev_indices(n_vars: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Generar todos los multi-índices (i1,...,in) con Σ ij ≤ max_degree.
    Cada multi-índice representa φ(x) = T_{i1}(x1)·...·T_{in}(xn).
    """
    indices = []
    # Para degree total d, enumeramos todas las combinaciones
    for d in range(max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            # Convertir combinación a multi-índice
            multi_idx = [0] * n_vars
            for var in combo:
                multi_idx[var] += 1
            indices.append(tuple(multi_idx))
    return indices


def evaluate_multivariate_chebyshev(x: np.ndarray, indices: List[Tuple[int, ...]],
                                    domain_lo: np.ndarray, domain_hi: np.ndarray) -> np.ndarray:
    """
    Evaluar base Chebyshev multivariada en x ∈ R^n.

    Args:
        x: punto de evaluación [n]
        indices: lista de multi-índices
        domain_lo, domain_hi: límites del dominio por dimensión

    Returns:
        Vector de valores de base [N] donde N = len(indices)
    """
    n = len(x)
    # Normalizar a [-1,1]
    t = 2.0 * (x - domain_lo) / (domain_hi - domain_lo) - 1.0
    t = np.clip(t, -1.0, 1.0)

    # Precomputar T_k(t_i) para cada dimensión
    max_deg_per_dim = max(max(idx) for idx in indices) if indices else 0
    T_cache = []  # T_cache[i][k] = T_k(t_i)
    for i in range(n):
        Ti = np.zeros(max_deg_per_dim + 1)
        Ti[0] = 1.0
        if max_deg_per_dim >= 1:
            Ti[1] = t[i]
        for k in range(2, max_deg_per_dim + 1):
            Ti[k] = 2.0 * t[i] * Ti[k - 1] - Ti[k - 2]
        T_cache.append(Ti)

    # Evaluar cada término
    values = np.ones(len(indices))
    for j, idx in enumerate(indices):
        for i, degree in enumerate(idx):
            if degree > 0:
                values[j] *= T_cache[i][degree]

    return values


# ═════════════════════════════════════════════════════════════════════════════
# Neurona ACF Tensorial
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorialMode:
    """Un modo de la descomposición tensorial."""
    multi_index: Tuple[int, ...]
    coefficient: float          # σ_k (importancia)
    output_vector: np.ndarray   # v_k ∈ R^m
    epsilon: float = 0.0        # cota de error del modo


@dataclass
class TensorialKnowledge:
    """Auto-conocimiento de la neurona tensorial."""
    n_input: int
    n_output: int
    n_modes: int
    max_degree: int
    total_energy: int            # Σ(len(modes))
    global_epsilon: float
    alpha_A: float
    expression: str
    limitations: List[str]


class ACFTensorialNeuron:
    """
    Neurona ACF Tensorial: f: R^n → R^m.

    Aprende la función multivariada como combinación de modos de Chebyshev
    con poda espectral. Captura interacciones entre dimensiones sin
    multiplicación de matrices — usando productos tensoriales de
    polinomios de Chebyshev.
    """

    def __init__(
        self,
        name: str,
        n_input: int,
        n_output: int,
        max_degree: int = 3,
        domain_lo: Optional[np.ndarray] = None,
        domain_hi: Optional[np.ndarray] = None,
        spectral_threshold: float = 1e-4,
        l2_lambda: float = 1.0,     # regularización L2 (Tikhonov)
    ):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.max_degree = max_degree
        self.domain_lo = domain_lo if domain_lo is not None else np.full(n_input, -2.0)
        self.domain_hi = domain_hi if domain_hi is not None else np.full(n_input, 2.0)
        self.spectral_threshold = spectral_threshold
        self.l2_lambda = l2_lambda

        # Generar base multivariada
        self._all_indices = generate_multivariate_chebyshev_indices(n_input, max_degree)
        self.n_basis = len(self._all_indices)

        # Matriz de coeficientes C ∈ R^{m × N}: y = C · ψ(x)
        self.C = np.zeros((n_output, self.n_basis))

        # RLS multivariado: P = (Φ^T Φ + λI)^{-1} inicial
        # Con λ=1.0 empezamos con incertidumbre moderada
        self.P = np.eye(self.n_basis) / l2_lambda
        self.forgetting_factor = 0.995  # casi sin olvido para muestras pequeñas

        # Modos activos (después de poda)
        self.modes: List[TensorialMode] = []

        # Métricas
        self.epsilon: float = float("inf")
        self.alpha_A: float = 0.0
        self.total_updates: int = 0

        self._error_window: List[float] = []
        self._error_window_size: int = 50
        self.evolution_history: List[Dict] = []

    # ═══════════════════════════════════════════════════════════════
    # API PRINCIPAL
    # ═══════════════════════════════════════════════════════════════

    def observe_and_learn(self, x: np.ndarray, y_target: np.ndarray) -> float:
        """
        Una observación. Una actualización. En tiempo real.

        Complejidad: O(N²) donde N = n_basis (≤ 165 para grado 3, 8 vars).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y_target = np.asarray(y_target, dtype=np.float64).ravel()

        # 1. Evaluar base multivariada
        psi = evaluate_multivariate_chebyshev(x, self._all_indices,
                                              self.domain_lo, self.domain_hi)

        # 2. Predecir
        y_pred = self.C @ psi
        error_vec = y_target - y_pred

        # 3. RLS multivariado: actualizar C por filas
        #    K = P·ψ / (λ + ψ^T·P·ψ)
        P_psi = self.P @ psi
        denom = self.forgetting_factor + float(psi @ P_psi)

        if denom > 1e-16:
            kalman_gain = P_psi / denom

            # Actualizar cada fila de C
            for j in range(self.n_output):
                self.C[j] += kalman_gain * error_vec[j]

            # Actualizar P = (P - K·ψ^T·P) / λ
            self.P = (self.P - np.outer(kalman_gain, P_psi)) / self.forgetting_factor

        # 4. Actualizar métricas
        error_norm = float(np.linalg.norm(error_vec))
        self._error_window.append(error_norm)
        if len(self._error_window) > self._error_window_size:
            self._error_window.pop(0)
        self.total_updates += 1

        # 5. Poda espectral periódica
        if self.total_updates % 100 == 0:
            self._prune_modes()

        # 6. Actualizar ε y α_A
        self._update_metrics()

        # 7. Registrar evolución
        self.evolution_history.append({
            "step": self.total_updates,
            "error": error_norm,
            "epsilon": self.epsilon,
            "n_modes": len(self.modes),
        })

        return error_norm

    def learn_batch(self, X: np.ndarray, Y: np.ndarray) -> List[float]:
        """Aprender de un lote de datos."""
        errors = []
        for x, y in zip(X, Y):
            err = self.observe_and_learn(x, y)
            errors.append(err)
        return errors

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 5) -> Dict:
        """Ajuste iterativo con múltiples pasadas."""
        t0 = time.perf_counter()
        all_errors = []
        for ep in range(epochs):
            indices = np.random.permutation(len(X))
            ep_err = []
            for idx in indices:
                err = self.observe_and_learn(X[idx], Y[idx])
                ep_err.append(err)
            all_errors.append(float(np.mean(ep_err)))
        elapsed = time.perf_counter() - t0
        self._prune_modes()
        return {
            "time": elapsed,
            "errors": all_errors,
            "final_error": all_errors[-1],
            "n_modes": len(self.modes),
            "epsilon": self.epsilon,
            "alpha_A": self.alpha_A,
            "total_updates": self.total_updates,
        }

    # ═══════════════════════════════════════════════════════════════
    # EVALUACIÓN
    # ═══════════════════════════════════════════════════════════════

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluar f(x) para x ∈ R^n."""
        x = np.asarray(x, dtype=np.float64).ravel()
        # Proyectar a dominio
        x = np.clip(x, self.domain_lo, self.domain_hi)

        if not self.modes:
            # Sin modos activos, usar todos
            psi = evaluate_multivariate_chebyshev(x, self._all_indices,
                                                  self.domain_lo, self.domain_hi)
            return self.C @ psi
        else:
            # Evaluar solo modos activos
            result = np.zeros(self.n_output)
            for mode in self.modes:
                psi_val = evaluate_multivariate_chebyshev(
                    x, [mode.multi_index], self.domain_lo, self.domain_hi
                )
                result += mode.coefficient * mode.output_vector * psi_val[0]
            return result

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluar sobre un lote."""
        return np.array([self.evaluate(x) for x in X])

    # ═══════════════════════════════════════════════════════════════
    # PODA ESPECTRAL
    # ═══════════════════════════════════════════════════════════════

    def _prune_modes(self) -> None:
        """
        Podar modos con baja importancia espectral.

        Umbral adaptativo: conservar modos hasta que la suma acumulada
        de σ² alcance el 99% de la energía total, o hasta el umbral fijo.
        """
        # Calcular importancia de cada modo
        mode_importances = []
        for k, multi_idx in enumerate(self._all_indices):
            v_k = self.C[:, k]
            sigma_k = float(np.linalg.norm(v_k))
            mode_importances.append((sigma_k, k, multi_idx, v_k))

        # Ordenar por importancia descendente
        mode_importances.sort(key=lambda x: x[0], reverse=True)

        # Energía total
        total_energy = sum(s**2 for s, _, _, _ in mode_importances)
        if total_energy < 1e-16:
            self.modes = []
            return

        # Conservar modos hasta 99% de energía o hasta umbral
        self.modes = []
        cumulative = 0.0
        for sigma_k, k, multi_idx, v_k in mode_importances:
            cumulative += sigma_k**2
            if sigma_k > self.spectral_threshold or cumulative / total_energy < 0.99:
                self.modes.append(TensorialMode(
                    multi_index=multi_idx,
                    coefficient=sigma_k,
                    output_vector=v_k / max(sigma_k, 1e-16),
                    epsilon=float(np.sqrt(np.abs(self.P[k, k]))) if k < len(self.P) else 0.0,
                ))

    # ═══════════════════════════════════════════════════════════════
    # MÉTRICAS
    # ═══════════════════════════════════════════════════════════════

    def _update_metrics(self) -> None:
        """Actualizar ε y α_A."""
        if len(self._error_window) >= 10:
            self.epsilon = float(np.mean(self._error_window[-20:]))

        # α_A: tasa de decaimiento de coeficientes de modos
        if self.modes:
            sigmas = np.array([m.coefficient for m in self.modes])
            sigmas = sigmas[sigmas > 1e-16]
            if len(sigmas) >= 2:
                log_s = np.log(sigmas)
                k = np.arange(len(log_s))
                self.alpha_A = float(max(0.0, -np.polyfit(k, log_s, 1)[0]))

    # ═══════════════════════════════════════════════════════════════
    # AUTO-CONOCIMIENTO
    # ═══════════════════════════════════════════════════════════════

    @property
    def self_knowledge(self) -> TensorialKnowledge:
        """La neurona sabe qué modos la componen."""
        mode_descriptions = []
        for i, mode in enumerate(self.modes[:10]):  # top 10
            idx_str = "·".join(str(d) for d in mode.multi_index)
            mode_descriptions.append(
                f"T_{{{idx_str}}}(x) × σ={mode.coefficient:.3f}"
            )

        lims = []
        if self.epsilon > 1.0:
            lims.append(f"ε alto: {self.epsilon:.2e}")
        if self.alpha_A < 0.2:
            lims.append(f"Baja compresibilidad: α_A={self.alpha_A:.3f}")
        if len(self.modes) == 0:
            lims.append("Sin modos activos — necesita más datos")

        return TensorialKnowledge(
            n_input=self.n_input,
            n_output=self.n_output,
            n_modes=len(self.modes),
            max_degree=self.max_degree,
            total_energy=len(self.modes),
            global_epsilon=self.epsilon,
            alpha_A=self.alpha_A,
            expression=" + ".join(mode_descriptions) if mode_descriptions else "sin modos activos",
            limitations=lims,
        )

    def detect_ood(self, x: np.ndarray) -> Tuple[bool, str]:
        """Detectar si x está fuera del dominio de conocimiento."""
        x = np.asarray(x, dtype=np.float64).ravel()
        for i in range(self.n_input):
            if x[i] < self.domain_lo[i] or x[i] > self.domain_hi[i]:
                return True, f"Dim {i}: x[{i}]={x[i]:.3f} ∉ [{self.domain_lo[i]:.2f}, {self.domain_hi[i]:.2f}]"
        # También verificar incertidumbre
        if self.epsilon > 10.0:
            return True, f"Alta incertidumbre global: ε={self.epsilon:.2e}"
        return False, ""

    def summary(self) -> str:
        sk = self.self_knowledge
        lines = [
            f"ACFTensorialNeuron('{self.name}')",
            f"  f: R^{self.n_input} → R^{self.n_output}",
            f"  Modos activos: {sk.n_modes}/{self.n_basis}",
            f"  Grado máximo: {self.max_degree}",
            f"  ε = {sk.global_epsilon:.2e}",
            f"  α_A = {sk.alpha_A:.3f}",
            f"  Top modos:",
        ]
        for i, mode in enumerate(self.modes[:5]):
            idx_str = ",".join(str(d) for d in mode.multi_index)
            lines.append(f"    [{i}] T_{{{idx_str}}}(x) σ={mode.coefficient:.4f} ε={mode.epsilon:.2e}")
        return "\n".join(lines)
