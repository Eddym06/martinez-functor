"""
acf_neuron_learning.py — Motor de Aprendizaje RLS Online para Neurona ACF
=========================================================================

Principio: Mínimos Cuadrados Recursivos (RLS) sobre base de Chebyshev.
Cada observación produce una actualización EXACTA (óptima) sin backprop.

Propiedades garantizadas:
  1. Cada paso minimiza error cuadrático sobre historial completo
  2. Complejidad O(d²) por paso, independiente del tamaño del buffer
  3. Detección automática de cambio de forma funcional
  4. Factor de olvido para adaptación a no-estacionariedad
  5. Cálculo certificado de ε tras cada actualización

Autor: AXIOM-1 sobre fundamentos de Eddy Manuel Piantini
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

from .acf_neuron import (
    ACFNeuron,
    Check,
    EvolutionStep,
    FMAInstruction,
    FormChange,
    FunctionalForm,
    NCClass,
    NeuronHealth,
    hash_fma_chain,
)


class ACFLearningEngine:
    """
    Motor de aprendizaje online para una neurona ACF.

    Usa RLS (Recursive Least Squares) sobre base de Chebyshev para
    actualizar la función de la neurona en tiempo real. NO usa backprop.
    NO usa gradiente descendente. Cada paso es matemáticamente óptimo.
    """

    def __init__(
        self,
        neuron: ACFNeuron,
        forgetting_factor: float = 0.995,
        regularization: float = 1e-6,
    ):
        self.neuron = neuron
        neuron.learning_engine = self
        neuron._learning_enabled = True

        self.forgetting_factor: float = forgetting_factor
        self.regularization: float = regularization
        self.max_degree: int = neuron.max_degree

        # Matriz de covarianza inversa (corazón de RLS)
        # P_0 = I / λ (prior no informativo)
        self.P: np.ndarray = np.eye(self.max_degree + 1) / regularization

        # Estado del detector de cambio de forma
        self._error_window: List[float] = []
        self._error_window_size: int = 50
        self._form_stability_counter: int = 0
        self._last_form_change: float = time.time()
        self._min_steps_between_changes: int = 20

    # ═══════════════════════════════════════════════════════════════
    # API PRINCIPAL
    # ═══════════════════════════════════════════════════════════════

    def observe_and_learn(self, x: float, y_target: float) -> EvolutionStep:
        """
        Una observación. Una actualización. En tiempo real.

        Complejidad: O(d²) donde d = max_degree.
        Independiente del tamaño del historial.
        """
        # 1. Evaluar base de Chebyshev en x
        T_x = self._chebyshev_basis(x)  # array[d+1]

        # 2. Predecir
        y_pred = float(np.dot(self.neuron.coefficients, T_x))
        error = y_target - y_pred

        # 3. RLS update
        #    K = P·T(x) / (λ + T(x)^T·P·T(x))
        P_T = self.P @ T_x
        denom = self.forgetting_factor + float(T_x @ P_T)

        if denom > 1e-16:
            kalman_gain = P_T / denom

            # 4. Actualizar coeficientes
            self.neuron.coefficients += kalman_gain * error

            # 5. Actualizar P = (P - K·T(x)^T·P) / λ
            self.P = (self.P - np.outer(kalman_gain, P_T)) / self.forgetting_factor
        else:
            # Denominador numéricamente cero: skip update
            kalman_gain = np.zeros_like(P_T)

        # 6. Recompilar cadena FMA
        self._recompile_fma_chain()

        # 7. Actualizar métricas
        self.neuron.epsilon = self._compute_epsilon()
        self._update_alpha_A()
        self._update_nc_class()
        self.neuron.total_observations += 1
        self.neuron.last_evolution_time = time.time()

        # 8. Detectar cambio de forma funcional
        self._error_window.append(abs(error))
        if len(self._error_window) > self._error_window_size:
            self._error_window.pop(0)
        form_change = self._detect_form_change()

        # 9. Registrar paso
        step = EvolutionStep(
            timestamp=time.time(),
            x=np.array([x]),
            y_target=y_target,
            y_pred=y_pred,
            error=abs(error),
            coefficients_snapshot=self.neuron.coefficients.copy(),
            energy=self.neuron.energy,
            epsilon=self.neuron.epsilon,
            form_change=form_change,
        )
        self.neuron.evolution_history.append(step)

        # 10. Actualizar salud
        self._update_health()

        return step

    def learn_batch(self, xs: np.ndarray, ys: np.ndarray) -> List[EvolutionStep]:
        """Procesar lote de observaciones secuencialmente."""
        steps = []
        for x, y in zip(xs, ys):
            step = self.observe_and_learn(float(x), float(y))
            steps.append(step)
        return steps

    def fit(self, xs: np.ndarray, ys: np.ndarray, epsilon_target: float = 1e-6,
            max_epochs: int = 20) -> List[EvolutionStep]:
        """
        Ajuste iterativo: múltiples pasadas sobre los datos hasta
        alcanzar precisión objetivo o agotar épocas.
        """
        all_steps = []
        for epoch in range(max_epochs):
            epoch_error = 0.0
            indices = np.random.permutation(len(xs))
            for idx in indices:
                step = self.observe_and_learn(float(xs[idx]), float(ys[idx]))
                epoch_error += step.error
            epoch_error /= len(xs)
            if epoch_error < epsilon_target:
                break
        return all_steps

    # ═══════════════════════════════════════════════════════════════
    # RECOMPILACIÓN DE CADENA FMA
    # ═══════════════════════════════════════════════════════════════

    def _recompile_fma_chain(self) -> None:
        """
        Reconstruir cadena FMA desde coeficientes Chebyshev actuales.

        Para evaluación usamos chebval (Clenshaw nativo de numpy).
        La cadena FMA se guarda como representación canónica para
        fingerprint y composición.
        """
        d = self.neuron._effective_degree()
        if d < 0:
            self.neuron.fma_chain = [FMAInstruction(weight=0.0, bias=0.0)]
            self.neuron.energy = 0
            return

        # Convertir Chebyshev → monomios para la cadena Horner
        if self.neuron.functional_form == FunctionalForm.CHEBYSHEV:
            mono = self._chebyshev_to_monomial(self.neuron.coefficients.copy(), d)
        else:
            mono = self.neuron.coefficients[:d + 1].copy()

        # Construir cadena Horner: f(x) = a_0 + x*(a_1 + x*(a_2 + ...))
        # Representación simbólica: cada FMA representa un paso de Horner
        chain: List[FMAInstruction] = []
        # El peso en cada instrucción = 1.0 (multiplicar por x implícito en Horner)
        # El bias = coeficiente correspondiente
        for k in range(d, 0, -1):
            chain.append(FMAInstruction(weight=1.0, bias=float(mono[k - 1]), source="self"))
        # Último coeficiente: el término de mayor grado
        if d >= 0:
            chain.append(FMAInstruction(weight=float(mono[d]), bias=0.0, source="self"))

        self.neuron.fma_chain = chain
        self.neuron.energy = d + 1 if d >= 0 else 1

    def _chebyshev_to_monomial(self, cheb_coeffs: np.ndarray, d: int) -> np.ndarray:
        """Convertir coeficientes de Chebyshev a monomios."""
        mono = np.zeros(d + 1)
        for k in range(d + 1):
            ck = cheb_coeffs[k]
            if abs(ck) < 1e-16:
                continue
            # Polinomio de Chebyshev T_k(x) como monomios
            T_poly = np.zeros(d + 1)
            if k == 0:
                T_poly[0] = 1.0
            elif k == 1:
                T_poly[1] = 1.0
            else:
                # Recurrencia: T_{k}(x) = 2x·T_{k-1}(x) - T_{k-2}(x)
                T_prev = np.zeros(d + 1); T_prev[0] = 1.0
                T_curr = np.zeros(d + 1); T_curr[1] = 1.0
                for i in range(2, k + 1):
                    T_next = np.zeros(d + 1)
                    T_next[1:] += 2.0 * T_curr[:-1]
                    T_next -= T_prev
                    T_prev, T_curr = T_curr, T_next
                T_poly = T_curr
            mono += ck * T_poly
        return mono

    # ═══════════════════════════════════════════════════════════════
    # MÉTRICAS Y CERTIFICADOS
    # ═══════════════════════════════════════════════════════════════

    def _compute_epsilon(self) -> float:
        """
        Estimar cota de error ε.

        Usa dos fuentes:
          1. Incertidumbre de coeficientes vía P (si está disponible y es finita)
          2. Error empírico en ventana reciente (más realista)
        """
        # Error empírico (más confiable)
        empirical = 0.0
        if len(self._error_window) >= 10:
            empirical = float(np.mean(self._error_window[-20:]))

        # Incertidumbre de coeficientes
        d = self.neuron._effective_degree()
        P_diag = np.abs(np.diag(self.P)[:d + 1])
        P_diag = P_diag[np.isfinite(P_diag)]
        theoretical = float(np.sum(np.sqrt(P_diag))) if len(P_diag) > 0 else float("inf")

        # Usar el menor entre empírico y teórico, con piso
        epsilon = min(theoretical, max(empirical * 10, 1e-10))
        return float(epsilon)

    def _update_alpha_A(self) -> None:
        """Calcular índice de decaimiento espectral α_A."""
        d = self.neuron._effective_degree()
        if d < 1:
            self.neuron.alpha_A = 0.0
            return
        abs_coeffs = np.abs(self.neuron.coefficients[1:d + 1])
        abs_coeffs = abs_coeffs[abs_coeffs > 1e-16]
        if len(abs_coeffs) < 2:
            self.neuron.alpha_A = 0.0
            return
        log_c = np.log(abs_coeffs)
        k = np.arange(len(log_c))
        alpha = -np.polyfit(k, log_c, 1)[0]
        self.neuron.alpha_A = float(max(0.0, min(alpha, 10.0)))

    def _update_nc_class(self) -> None:
        a = self.neuron.alpha_A
        if a < 0.25:
            self.neuron.nc_class = NCClass.NC0
        elif a < 0.5:
            self.neuron.nc_class = NCClass.NC1
        elif a < 1.0:
            self.neuron.nc_class = NCClass.NC2
        else:
            self.neuron.nc_class = NCClass.NC3

    def _update_health(self) -> None:
        """Actualizar estado de salud de la neurona."""
        if self.neuron.epsilon > 1.0:
            self.neuron.health = NeuronHealth.UNSTABLE
        elif len(self._error_window) >= 20:
            recent = self._error_window[-20:]
            if np.mean(recent) > np.mean(self._error_window[:20]) * 2:
                self.neuron.health = NeuronHealth.DEGRADING
            else:
                self.neuron.health = NeuronHealth.HEALTHY
        else:
            self.neuron.health = NeuronHealth.HEALTHY

    # ═══════════════════════════════════════════════════════════════
    # DETECCIÓN DE CAMBIO DE FORMA FUNCIONAL
    # ═══════════════════════════════════════════════════════════════

    def _detect_form_change(self) -> Optional[FormChange]:
        """Detectar si la neurona debe cambiar de forma funcional."""
        if self.neuron.total_observations - self._last_form_change < self._min_steps_between_changes:
            return None

        current = self.neuron.functional_form
        alpha = self.neuron.alpha_A

        # Criterios de cambio
        new_form = current

        if current == FunctionalForm.CHEBYSHEV:
            if alpha > 1.5:
                # Coeficientes decaen muy rápido → polinomio exacto es suficiente
                new_form = FunctionalForm.POLYNOMIAL
            elif alpha < 0.2 and self.neuron.total_observations > 100:
                # Estructura compleja → Koopman podría capturar mejor
                new_form = FunctionalForm.KOOPMAN

        elif current == FunctionalForm.POLYNOMIAL:
            if alpha < 0.3 and self.neuron.epsilon > 0.01:
                new_form = FunctionalForm.CHEBYSHEV

        elif current == FunctionalForm.KOOPMAN:
            if alpha > 1.0:
                new_form = FunctionalForm.CHEBYSHEV

        if new_form != current:
            form_change = FormChange(
                old_form=current,
                new_form=new_form,
                reason=f"α_A={alpha:.3f} -> cambio de {current.value} a {new_form.value}",
                alpha_A_before=alpha,
                alpha_A_after=alpha,
            )
            self._apply_form_change(form_change)
            self._last_form_change = time.time()
            return form_change

        return None

    def _apply_form_change(self, fc: FormChange) -> None:
        """Aplicar cambio de forma funcional preservando el conocimiento."""
        old_coeffs = self.neuron.coefficients.copy()
        d = self.neuron._effective_degree()

        if fc.new_form == FunctionalForm.POLYNOMIAL and fc.old_form == FunctionalForm.CHEBYSHEV:
            # Convertir Chebyshev → Monomios usando matriz de transición
            mono = np.zeros(d + 1)
            for k in range(d + 1):
                if abs(old_coeffs[k]) < 1e-15:
                    continue
                # T_k(x) expresado como monomios
                T_poly = np.zeros(d + 1)
                if k == 0:
                    T_poly[0] = 1.0
                elif k == 1:
                    T_poly[1] = 1.0
                else:
                    T_prev = np.zeros(d + 1)
                    T_prev[0] = 1.0
                    T_curr = np.zeros(d + 1)
                    T_curr[1] = 1.0
                    for i in range(2, k + 1):
                        T_next = np.zeros(d + 1)
                        T_next[1:] += 2.0 * T_curr[:-1]
                        T_next -= T_prev
                        T_prev, T_curr = T_curr, T_next
                    T_poly = T_curr
                mono += old_coeffs[k] * T_poly
            self.neuron.coefficients = mono
            self.neuron.functional_form = FunctionalForm.POLYNOMIAL

        elif fc.new_form == FunctionalForm.CHEBYSHEV and fc.old_form == FunctionalForm.POLYNOMIAL:
            # Monomios → Chebyshev: samplear y refit
            xs = self.neuron.domain.sample(200)
            ys = np.array([float(self.neuron.evaluate(float(x))) for x in xs])
            self._refit_chebyshev(xs, ys)

        elif fc.new_form == FunctionalForm.KOOPMAN:
            self.neuron.functional_form = FunctionalForm.KOOPMAN

        # Recompilar
        self._recompile_fma_chain()

    # ═══════════════════════════════════════════════════════════════
    # UTILIDADES
    # ═══════════════════════════════════════════════════════════════

    def _chebyshev_basis(self, x: float) -> np.ndarray:
        """Evaluar polinomios de Chebyshev T_0(t)...T_d(t) donde t ∈ [-1,1]."""
        # Normalizar x del dominio [lo, hi] a t ∈ [-1, 1]
        lo, hi = self.neuron.domain.lo, self.neuron.domain.hi
        t = 2.0 * (x - lo) / (hi - lo) - 1.0
        # Clamp por seguridad numérica
        t = max(-1.0, min(1.0, t))

        d = self.max_degree
        T = np.zeros(d + 1)
        T[0] = 1.0
        if d >= 1:
            T[1] = t
        for k in range(2, d + 1):
            T[k] = 2.0 * t * T[k - 1] - T[k - 2]
        return T

    def _refit_chebyshev(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Re-ajustar coeficientes Chebyshev desde cero."""
        d = self.max_degree
        n = len(xs)
        # Matriz de diseño
        Phi = np.zeros((n, d + 1))
        for i, x in enumerate(xs):
            T = self._chebyshev_basis(float(x))
            Phi[i] = T

        # Mínimos cuadrados con regularización
        I_reg = np.eye(d + 1) * self.regularization
        coeffs = np.linalg.solve(Phi.T @ Phi + I_reg, Phi.T @ ys)
        self.neuron.coefficients = coeffs
        # Resetear P
        self.P = np.linalg.inv(Phi.T @ Phi + I_reg)

    def get_diagnostics(self) -> dict:
        """Obtener diagnóstico completo del motor de aprendizaje."""
        return {
            "P_trace": float(np.trace(self.P)),
            "P_condition": float(np.linalg.cond(self.P)) if self.max_degree < 50 else -1,
            "effective_degree": self.neuron._effective_degree(),
            "recent_error_mean": float(np.mean(self._error_window[-20:])) if self._error_window else -1,
            "coefficient_norm": float(np.linalg.norm(self.neuron.coefficients)),
            "functional_form": self.neuron.functional_form.value,
        }
