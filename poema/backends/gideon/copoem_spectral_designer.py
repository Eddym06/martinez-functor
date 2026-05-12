"""
copoem_spectral_designer.py — CoPoem Inverse Spectral Solver.

Resuelve el problema inverso supremo de la turbulencia:

  "Dado un espectro objetivo E_target(k) ~ k^{-3}, encontrar la función
   de forzamiento F(k) y el perfil de viscosidad ν(k) que lo producen
   de manera óptima."

Esto es la manifestación del Co-Functor Φ* en el contexto de Navier-Stokes:
  - Φ  : flujo ω(x,t) → d_95 modos Koopman (compresión)
  - Φ* : propiedades espectrales deseadas → F(k), ν(k) (síntesis)

El ciclo adjuntivo Φ ⇌ Φ* converge cuando E_actual(k) ≈ E_target(k).

Algoritmo (Model-Predictive Spectral Control):
  1. Medir E_actual(k) del estado actual de la simulación
  2. Calcular misfit espectral: J = Σ_k w(k)[log E_actual(k) - log E_target(k)]²
  3. Estimar gradiente per-modo usando balance energético espectral:
       ∂E(k)/∂t ≈ F(k) - 2νk²E(k) - 2ν₄k⁸E(k) + T(k)
  4. Actualizar A(k) ← A(k) - η · ∂J/∂A(k) con proyección sobre restricciones
  5. Proyectar vía CoPoem multi-objetivo:
       - Estabilidad (radio espectral Koopman < 1.2)
       - Presupuesto energético (Σ_k A(k)² ≤ P_max)
       - Suavidad (∇²_k A(k) penalizado)

Martínez's Invariant — Abril 2026
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralDesignState:
    """Estado completo del diseñador espectral en un instante t."""
    t: float
    misfit: float                     # J = error espectral total
    slope_actual: float
    slope_target: float
    amplitudes: np.ndarray            # A(k) per-mode forcing
    nu_profile: np.ndarray            # ν_eff(k) per-mode viscosity
    k_shells: np.ndarray              # shells de k
    E_actual: np.ndarray
    E_target: np.ndarray
    adjunction_gap: float             # ||E_actual - E_target|| / ||E_target||
    n_iterations: int                 # iterations of optimizer this cycle
    koopman_spectral_radius: float
    total_power: float                # Σ A(k)²
    phase: str                        # 'ramping', 'sculpting', 'converged'


@dataclass
class DesignerConfig:
    """Configuración del CoPoem Spectral Designer."""
    # Target spectrum
    target_slope: float = -3.0           # k^{-3} enstrophy cascade
    target_slope_range: Tuple[float, float] = (6, 50)  # k range for slope fit

    # Broadband forcing
    k_force_min: int = 4
    k_force_max: int = 10
    initial_amplitude: float = 15.0      # A_0 per-mode

    # Optimizer
    learning_rate: float = 0.4           # η for amplitude update
    nu_learning_rate: float = 0.1        # η for viscosity profile
    momentum: float = 0.6               # momentum for SGD
    smoothness_penalty: float = 0.05     # λ_smooth for ∇²A(k) penalty

    # Constraints (CoPoem projections)
    max_total_power: float = 5000.0      # Σ A(k)² ≤ P_max
    max_single_mode: float = 200.0       # max A(k) per mode
    min_single_mode: float = 0.5         # min A(k) per mode
    spectral_radius_limit: float = 1.3   # Koopman stability constraint

    # Viscosity design
    enable_nu_design: bool = True        # also optimize ν(k)
    nu_base_factor: float = 1.0          # multiplier on base ν₄
    nu_max_factor: float = 5.0           # max ν₄ at any scale
    nu_min_factor: float = 0.05          # min ν₄ at any scale

    # Convergence
    misfit_threshold: float = 0.3        # J < threshold → converged
    slope_tolerance: float = 0.5         # |slope - target| < tol → converged
    max_opt_iters: int = 5               # inner optimization iterations per cycle

    # Ramping
    ramp_cycles: int = 3                 # cycles in 'ramping' phase


# ═════════════════════════════════════════════════════════════════════════════
# CoPoem Spectral Designer
# ═════════════════════════════════════════════════════════════════════════════

class CoPoemSpectralDesigner:
    """
    Inverse spectral solver: dado E_target(k) ~ k^{-3}, sintetiza F(k), ν(k).

    Implementa el Co-Functor Φ* del ACF aplicado a Navier-Stokes:
      Φ* : {E_target(k), ρ_spec ≤ r_max, Σ A² ≤ P} → {A(k), ν(k)}

    Tres fases:
      1. RAMPING: incrementar gradualmente el forzamiento para evitar blowup
      2. SCULPTING: optimización activa de A(k) para converger a E_target(k)
      3. CONVERGED: ajustes finos, mantener el espectro k^{-3}
    """

    def __init__(self, cfg: DesignerConfig, N: int, nu: float, nu4: float):
        self.cfg = cfg
        self.N = N
        self.nu = nu               # molecular viscosity = 1/Re
        self.nu4_base = nu4         # base hyperviscosity

        # Shell structure for broadband forcing
        self.k_shells = np.arange(cfg.k_force_min, cfg.k_force_max + 1, dtype=float)
        self.n_shells = len(self.k_shells)

        # Per-mode amplitudes A(k) — the design variable
        self.amplitudes = np.full(self.n_shells, cfg.initial_amplitude)

        # Per-mode viscosity multiplier ν_mult(k)
        # ν_eff(k) = ν_mult(k) * ν₄_base * k^6
        self.nu_multipliers = np.ones(self.n_shells)

        # Momentum buffer for SGD
        self._amp_velocity = np.zeros(self.n_shells)
        self._nu_velocity = np.zeros(self.n_shells)

        # Target spectrum (computed lazily for each analysis)
        self._E_target_cache: Optional[np.ndarray] = None
        self._k_target_cache: Optional[np.ndarray] = None

        # State
        self.phase = "ramping"
        self.cycle_count = 0
        self.history: List[SpectralDesignState] = []

        # CoPoem adjunction tracking
        self._adjunction_gaps: List[float] = []

    # ─────────────────────────────────────────────────────────────────────
    # Target spectrum
    # ─────────────────────────────────────────────────────────────────────

    def compute_target_spectrum(
        self, k_bins: np.ndarray, E_actual: np.ndarray
    ) -> np.ndarray:
        """
        Compute E_target(k) = C * k^{target_slope} normalized to match
        the total energy in the inertial range of E_actual.

        The normalization ensures we're not asking for 10x more energy than
        the forcing can provide — we're asking for the SHAPE, not the magnitude.
        """
        cfg = self.cfg
        k_min, k_max = cfg.target_slope_range

        # Inertial range mask
        mask = (k_bins >= k_min) & (k_bins <= min(k_max, self.N // 3))
        if np.sum(mask) < 3:
            # Fallback: use wider range
            mask = (k_bins >= 3) & (k_bins <= self.N // 4)

        k_ir = k_bins[mask]
        E_ir = E_actual[mask]

        # Total energy in inertial range of actual spectrum
        E_total_actual = np.sum(E_ir[E_ir > 0])
        if E_total_actual < 1e-20:
            E_total_actual = 1.0

        # Target: E_target(k) = C * k^slope
        E_target_shape = k_ir ** cfg.target_slope
        E_total_target_shape = np.sum(E_target_shape)
        if E_total_target_shape < 1e-20:
            E_total_target_shape = 1.0

        C = E_total_actual / E_total_target_shape

        # Full target over all k_bins
        E_target = np.zeros_like(E_actual)
        E_target[mask] = C * k_bins[mask] ** cfg.target_slope
        # Outside inertial range: set to actual (don't try to control those)
        E_target[~mask] = E_actual[~mask]
        E_target = np.maximum(E_target, 1e-30)  # floor

        self._E_target_cache = E_target
        self._k_target_cache = k_bins
        return E_target

    # ─────────────────────────────────────────────────────────────────────
    # Spectral misfit
    # ─────────────────────────────────────────────────────────────────────

    def compute_misfit(
        self, k_bins: np.ndarray, E_actual: np.ndarray, E_target: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute weighted log-spectral misfit and per-shell residual.

        J = Σ_k w(k) [log E_actual(k) - log E_target(k)]²

        Returns (J_total, r(k)) where r(k) = log(E_actual/E_target) per shell.
        """
        cfg = self.cfg
        k_min, k_max = cfg.target_slope_range

        mask = (k_bins >= k_min) & (k_bins <= min(k_max, self.N // 3))
        mask &= (E_actual > 1e-25) & (E_target > 1e-25)

        if np.sum(mask) < 3:
            return 10.0, np.zeros(self.n_shells)

        log_ratio = np.log(E_actual[mask] + 1e-30) - np.log(E_target[mask] + 1e-30)

        # Weight: emphasize the forcing range and inertial range
        k_m = k_bins[mask]
        weights = np.ones_like(k_m)
        # Boost weight in forcing band
        in_force = (k_m >= cfg.k_force_min) & (k_m <= cfg.k_force_max)
        weights[in_force] *= 2.0

        J = float(np.sum(weights * log_ratio**2) / np.sum(weights))

        # Per-shell residual for the forcing band
        residual = np.zeros(self.n_shells)
        for i, k in enumerate(self.k_shells):
            k_mask = (k_bins >= k - 0.5) & (k_bins < k + 0.5) & (E_actual > 1e-25)
            if np.any(k_mask):
                residual[i] = np.mean(
                    np.log(E_actual[k_mask] + 1e-30) - np.log(E_target[k_mask] + 1e-30)
                )

        return J, residual

    # ─────────────────────────────────────────────────────────────────────
    # Gradient estimation (model-based)
    # ─────────────────────────────────────────────────────────────────────

    def estimate_gradient(
        self,
        residual: np.ndarray,
        E_actual_shells: np.ndarray,
        E_target_shells: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate gradient of misfit J w.r.t. A(k) and ν_mult(k).

        Uses the spectral energy balance:
          ∂E(k)/∂t ≈ A(k)² · S(k) - 2ν k² E(k) - 2ν₄ν_mult(k) k⁸ E(k) + T(k)

        where S(k) is the spectral density of forcing at shell k.

        The gradient w.r.t. A(k) is approximately:
          ∂J/∂A(k) ≈ -2 · r(k) · [2A(k) · S(k)] / E_actual(k)
                    ≈ -2 · r(k) · 2A(k) / E_actual(k)  (S ≈ 1)

        Simplified: if E_actual < E_target (r < 0), increase A(k).
                    if E_actual > E_target (r > 0), decrease A(k).
        """
        cfg = self.cfg

        # dJ/dA(k) ∝ -r(k) * A(k) / max(E_actual(k), ε)
        # This is the "proportional control" based on spectral energy balance
        E_safe = np.maximum(E_actual_shells, 1e-20)
        grad_A = -residual * self.amplitudes / E_safe

        # Normalize gradient to avoid huge steps
        grad_norm = np.linalg.norm(grad_A)
        if grad_norm > 1.0:
            grad_A = grad_A / grad_norm

        # dJ/dν_mult(k) ∝ r(k) * k^8 * E_actual(k)
        # If E_actual > E_target (r > 0), increase dissipation at that scale
        grad_nu = np.zeros(self.n_shells)
        if cfg.enable_nu_design:
            for i, k in enumerate(self.k_shells):
                grad_nu[i] = residual[i] * (k ** 2) * E_safe[i]
            gn_norm = np.linalg.norm(grad_nu)
            if gn_norm > 1.0:
                grad_nu = grad_nu / gn_norm

        return grad_A, grad_nu

    # ─────────────────────────────────────────────────────────────────────
    # CoPoem Projection (constraint enforcement)
    # ─────────────────────────────────────────────────────────────────────

    def project_constraints(self):
        """
        Project A(k) and ν_mult(k) onto the CoPoem constraint set:
          1. Per-mode bounds: A_min ≤ A(k) ≤ A_max
          2. Total power: Σ A(k)² ≤ P_max
          3. Smoothness: penalize ∇²A(k)
          4. Viscosity bounds: ν_min ≤ ν_mult(k) ≤ ν_max
        """
        cfg = self.cfg

        # 1. Per-mode clipping
        self.amplitudes = np.clip(self.amplitudes, cfg.min_single_mode, cfg.max_single_mode)

        # 2. Total power projection (rescale if over budget)
        total_power = np.sum(self.amplitudes ** 2)
        if total_power > cfg.max_total_power:
            scale = np.sqrt(cfg.max_total_power / total_power)
            self.amplitudes *= scale

        # 3. Smoothness: Laplacian regularization
        if self.n_shells > 2 and cfg.smoothness_penalty > 0:
            # Compute discrete Laplacian of A(k)
            laplacian = np.zeros(self.n_shells)
            for i in range(1, self.n_shells - 1):
                laplacian[i] = (
                    self.amplitudes[i - 1]
                    - 2 * self.amplitudes[i]
                    + self.amplitudes[i + 1]
                )
            # Smooth: A(k) += λ * ∇²A(k)
            self.amplitudes += cfg.smoothness_penalty * laplacian

        # Re-clip after smoothing
        self.amplitudes = np.clip(self.amplitudes, cfg.min_single_mode, cfg.max_single_mode)

        # 4. Viscosity bounds
        if cfg.enable_nu_design:
            self.nu_multipliers = np.clip(
                self.nu_multipliers, cfg.nu_min_factor, cfg.nu_max_factor
            )

    # ─────────────────────────────────────────────────────────────────────
    # Main control cycle
    # ─────────────────────────────────────────────────────────────────────

    def control_cycle(
        self,
        k_bins: np.ndarray,
        E_k: np.ndarray,
        koopman_spectral_radius: float,
        energy: float,
        enstrophy: float,
        t: float,
    ) -> Dict[str, Any]:
        """
        Execute one CoPoem control cycle.

        1. Compute target spectrum
        2. Compute misfit
        3. Estimate gradient
        4. Update A(k) with momentum SGD
        5. Project constraints
        6. Determine phase transitions

        Returns control action dict for the NS solver.
        """
        cfg = self.cfg
        self.cycle_count += 1

        # ── Phase management ──
        if self.cycle_count <= cfg.ramp_cycles:
            self.phase = "ramping"
        else:
            self.phase = "sculpting"

        # ── Compute target spectrum ──
        E_target = self.compute_target_spectrum(k_bins, E_k)

        # ── Compute misfit ──
        J, residual = self.compute_misfit(k_bins, E_k, E_target)

        # ── Compute actual slope ──
        slope_actual = self._compute_slope(k_bins, E_k)

        # ── Check convergence ──
        if (J < cfg.misfit_threshold and
                abs(slope_actual - cfg.target_slope) < cfg.slope_tolerance):
            self.phase = "converged"

        # ── Get E_actual and E_target at forcing shells ──
        E_actual_shells = np.zeros(self.n_shells)
        E_target_shells = np.zeros(self.n_shells)
        for i, k in enumerate(self.k_shells):
            k_mask = (k_bins >= k - 0.5) & (k_bins < k + 0.5)
            if np.any(k_mask):
                E_actual_shells[i] = np.mean(E_k[k_mask])
                E_target_shells[i] = np.mean(E_target[k_mask])
        E_actual_shells = np.maximum(E_actual_shells, 1e-25)
        E_target_shells = np.maximum(E_target_shells, 1e-25)

        # ── Optimization iterations ──
        n_opt = 1 if self.phase == "ramping" else cfg.max_opt_iters
        if self.phase == "converged":
            n_opt = 1

        for _iter in range(n_opt):
            grad_A, grad_nu = self.estimate_gradient(
                residual, E_actual_shells, E_target_shells
            )

            # ── Stability guard from Koopman ──
            stability_scale = 1.0
            if koopman_spectral_radius > cfg.spectral_radius_limit:
                # Reduce aggressiveness if system is unstable
                excess = (koopman_spectral_radius - 1.0) / 0.5
                stability_scale = max(0.1, 1.0 / (1.0 + excess))

            # ── Phase-dependent learning rate ──
            lr = cfg.learning_rate * stability_scale
            if self.phase == "ramping":
                # Ramp up gradually: cycle 1 → 33%, cycle 2 → 67%, cycle 3 → 100%
                ramp_frac = self.cycle_count / cfg.ramp_cycles
                lr *= ramp_frac
                # Also ramp the amplitudes themselves
                target_amp = cfg.initial_amplitude * ramp_frac
                self.amplitudes = np.maximum(self.amplitudes, target_amp * 0.5)
            elif self.phase == "converged":
                lr *= 0.1  # very gentle corrections

            # ── Momentum SGD update for A(k) ──
            self._amp_velocity = cfg.momentum * self._amp_velocity + lr * grad_A
            self.amplitudes += self._amp_velocity

            # ── Viscosity profile update ──
            if cfg.enable_nu_design:
                lr_nu = cfg.nu_learning_rate * stability_scale
                self._nu_velocity = cfg.momentum * self._nu_velocity + lr_nu * grad_nu
                self.nu_multipliers += self._nu_velocity

            # ── Project onto constraint set ──
            self.project_constraints()

        # ── Adjunction gap ──
        adj_gap = float(np.sqrt(np.mean(residual ** 2)))
        self._adjunction_gaps.append(adj_gap)

        # ── Compute effective ν₄ (weighted average of nu_multipliers) ──
        nu4_effective = self.nu4_base * float(np.mean(self.nu_multipliers))

        # ── Record state ──
        state = SpectralDesignState(
            t=t,
            misfit=J,
            slope_actual=slope_actual,
            slope_target=cfg.target_slope,
            amplitudes=self.amplitudes.copy(),
            nu_profile=self.nu_multipliers.copy(),
            k_shells=self.k_shells.copy(),
            E_actual=E_k.copy() if len(E_k) < 200 else E_k[:200].copy(),
            E_target=E_target.copy() if len(E_target) < 200 else E_target[:200].copy(),
            adjunction_gap=adj_gap,
            n_iterations=n_opt,
            koopman_spectral_radius=koopman_spectral_radius,
            total_power=float(np.sum(self.amplitudes ** 2)),
            phase=self.phase,
        )
        self.history.append(state)

        # ── Build control action ──
        return {
            "amplitudes": self.amplitudes.copy(),
            "k_shells": self.k_shells.copy(),
            "nu4_effective": nu4_effective,
            "nu_multipliers": self.nu_multipliers.copy(),
            "misfit": J,
            "adjunction_gap": adj_gap,
            "slope_actual": slope_actual,
            "phase": self.phase,
            "total_power": float(np.sum(self.amplitudes ** 2)),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _compute_slope(self, k_bins: np.ndarray, E_k: np.ndarray) -> float:
        """Compute spectral slope in inertial range."""
        cfg = self.cfg
        k_min, k_max = cfg.target_slope_range
        mask = (
            (k_bins >= k_min) &
            (k_bins <= min(k_max, self.N // 3)) &
            (E_k > 1e-20)
        )
        if np.sum(mask) < 5:
            return -99.0
        log_k = np.log(k_bins[mask])
        log_E = np.log(E_k[mask])
        slope, _ = np.polyfit(log_k, log_E, 1)
        return float(slope)

    def get_broadband_forcing(
        self, kx: np.ndarray, ky: np.ndarray, phases: np.ndarray
    ) -> np.ndarray:
        """
        Build the full 2D forcing field in Fourier space with per-mode amplitudes.

        F_hat(kx, ky) = Σ_{k ∈ shells} A(k) * δ(|k| ≈ k) * exp(iφ)

        Returns complex array of shape (N, N).
        """
        k_mag = np.sqrt(kx**2 + ky**2)
        F_hat = np.zeros_like(kx, dtype=complex)

        for i, k in enumerate(self.k_shells):
            band = (np.abs(k_mag - k) < 0.8) & (k_mag > 0)
            if np.any(band):
                F_hat[band] = self.amplitudes[i] * np.exp(1j * phases[band])

        return F_hat

    def get_nu4_profile(
        self, kx: np.ndarray, ky: np.ndarray
    ) -> np.ndarray:
        """
        Build scale-dependent hyperviscosity profile in Fourier space.

        ν₄_eff(k) = ν₄_base * ν_mult(|k|)

        For k in forcing shells, use the designed multiplier.
        For other k, use ν₄_base (multiplier = 1).

        Returns the modified linear operator contribution: -ν₄_eff(k) * k^8
        """
        k_mag = np.sqrt(kx**2 + ky**2)
        k2_full = kx**2 + ky**2

        # Start with base hyperviscosity
        nu4_field = np.full_like(kx, self.nu4_base, dtype=float)

        # Apply per-shell multipliers
        for i, k in enumerate(self.k_shells):
            band = (np.abs(k_mag - k) < 0.8)
            if np.any(band):
                nu4_field[band] = self.nu4_base * self.nu_multipliers[i]

        # Return the dissipation contribution: -ν₄(k) * k^8
        return -nu4_field * k2_full**4


# ═════════════════════════════════════════════════════════════════════════════
# CoPoem Oracle — Integrates into Multi-Oracle AMR
# ═════════════════════════════════════════════════════════════════════════════

class CoPoemOracle:
    """
    Oráculo 6: CoPoem — Control inverso espectral.

    Este oráculo es DIFERENTE a los otros 5: en lugar de votar sobre
    factores escalares (ν₄_factor, force_factor), produce directamente
    un perfil de forzamiento A(k) per-modo.

    Reglas:
      - Mide el misfit J entre E_actual y E_target
      - Si J > threshold → fase "sculpting", optimización activa
      - Si J < threshold → fase "converged", ajustes finos
      - Produce amplitudes per-modo + perfil de viscosidad

    El confidence de este oráculo escala con 1/(1+J): menor misfit → más confianza.
    """

    def __init__(self, designer: CoPoemSpectralDesigner):
        self.designer = designer

    def vote(
        self,
        k_bins: np.ndarray,
        E_k: np.ndarray,
        koopman_spectral_radius: float,
        energy: float,
        enstrophy: float,
        t: float,
    ) -> Dict[str, Any]:
        """Execute CoPoem control cycle and return full action."""
        return self.designer.control_cycle(
            k_bins, E_k, koopman_spectral_radius, energy, enstrophy, t
        )
