"""
turbulence_thermostat.py — Multi-Oracle AMR + Cascade Acceleration.

Sistema de Refinamiento Adaptativo de Malla (AMR) guiado por el ecosistema
ACF completo, no solo Koopman. Cinco oráculos independientes votan sobre
las acciones de control (ν₄, forcing, friction) y un arbitrador bayesiano
fusiona sus recomendaciones.

Oráculos:
  O1: Koopman (TAA)         — gap espectral, d_95, modos coherentes
  O2: Ruelle  (OTU)         — entropía h_KS, resonancias, Γ_OTU
  O3: ERGON                 — detección de regímenes, Lyapunov
  O4: Termodinámica (ACF)   — transiciones de fase, Fisher information
  O5: Espectral (Kolmogorov) — pendiente E(k), balance energético

Termostato de Turbulencia:
  El "Cascade Accelerator" inyecta energía inteligentemente cuando la
  turbulencia no se ha desarrollado (d_95 < target), y estabiliza cuando
  sí lo ha hecho (h_KS > threshold). El objetivo: conducir el flujo a
  turbulencia completamente desarrollada y MANTENERLO ahí usando solo
  d_95 modos de Koopman.

Arquitectura:
  ┌─────────────────────────────────────────────────────────┐
  │                    NS Solver (RK4-IF)                   │
  │                  ω(t) → ω(t+dt)                         │
  └────────────────────┬────────────────────────────────────┘
                       │ snapshots
           ┌───────────┼───────────┐
           ▼           ▼           ▼
  ┌─────────────┐ ┌─────────┐ ┌──────────┐
  │  O1:Koopman │ │ O2:Ruelle│ │ O3:ERGON │
  └──────┬──────┘ └────┬────┘ └─────┬────┘
         │             │            │
         ▼             ▼            ▼
  ┌─────────────────────────────────────────┐
  │     O4: Thermodynamic     O5: Spectral  │
  └───────────────┬─────────────────────────┘
                  │ votes
                  ▼
  ┌─────────────────────────────────────────┐
  │       Bayesian Arbiter (fusión)         │
  │   → ν₄_new, A_new, α_new, dt_adapt     │
  └───────────────┬─────────────────────────┘
                  │ control actions
                  ▼
  ┌─────────────────────────────────────────┐
  │     Cascade Accelerator (termostato)    │
  │   → override si d_95 < target           │
  └─────────────────────────────────────────┘

Martínez's Invariant — Abril 2026
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# Oracle Result — Estructura unificada para todas las recomendaciones
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleVote:
    """Recomendación de un oráculo individual."""
    oracle_name: str
    nu4_factor: float = 1.0        # multiplicador sobre ν₄ base
    force_factor: float = 1.0      # multiplicador sobre A base
    friction_factor: float = 1.0   # multiplicador sobre α base
    confidence: float = 0.5        # confianza [0, 1]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warning: Optional[str] = None


@dataclass
class ThermostatState:
    """Estado completo del termostato de turbulencia."""
    t: float
    d_95: int
    h_ks: float
    spectral_slope: float
    nu4_effective: float
    force_effective: float
    friction_effective: float
    energy: float
    enstrophy: float
    n_coherent: int
    phase: str            # 'acceleration', 'stabilization', 'equilibrium'
    oracles_active: int
    reduction_factor: float


# ═════════════════════════════════════════════════════════════════════════════
# ORÁCULO 1: Koopman (TAA) — Gap espectral, dimensión intrínseca
# ═════════════════════════════════════════════════════════════════════════════

class KoopmanOracle:
    """
    Oráculo basado en análisis Koopman EDMD.

    Reglas:
      - Si d_95 < d_target → la turbulencia no se ha desarrollado:
        reducir ν₄ (menos disipación) y aumentar A (más energía).
      - Si |λ_max| > 1.1 → inestabilidad inminente:
        aumentar ν₄ para estabilizar.
      - Si n_coherent es alto → el flujo está bien organizado:
        los modos capturan la dinámica, mantener estable.
    """

    def __init__(self, d_target: int = 30):
        self.d_target = d_target

    def vote(self, koopman_result: Any, energy: float, enstrophy: float) -> OracleVote:
        if koopman_result is None:
            return OracleVote("koopman", confidence=0.0)

        eigs = koopman_result.eigenvalues
        d_95 = koopman_result.d_95
        n_coherent = koopman_result.n_coherent
        max_mag = float(np.max(np.abs(eigs)))

        nu4_f = 1.0
        force_f = 1.0
        friction_f = 1.0
        confidence = 0.8

        # Cascade acceleration: if d_95 is too low, turbulence hasn't developed
        if d_95 < self.d_target:
            deficit = (self.d_target - d_95) / self.d_target
            force_f = 1.0 + 0.3 * deficit     # boost forcing up to 30%
            nu4_f = max(0.3, 1.0 - 0.7 * deficit)  # cut hyperviscosity up to 70%
            friction_f = max(0.5, 1.0 - 0.3 * deficit)

        # Stability guard: if eigenvalues escaping unit circle
        if max_mag > 1.15:
            instab = min((max_mag - 1.0) / 0.5, 5.0)
            nu4_f = max(nu4_f, 1.0 + instab * 2)  # override: increase ν₄
            force_f = min(force_f, 1.0 / (1.0 + instab * 0.5))
            confidence = 0.95  # high confidence in stability action

        # Well-resolved regime: many coherent modes captured
        if n_coherent > d_95 * 0.5 and d_95 >= self.d_target:
            nu4_f = 1.0  # maintain
            force_f = 0.95  # slightly reduce
            friction_f = 1.0

        return OracleVote(
            oracle_name="koopman",
            nu4_factor=nu4_f,
            force_factor=force_f,
            friction_factor=friction_f,
            confidence=confidence,
            diagnostics={
                "d_95": d_95,
                "max_mag": max_mag,
                "n_coherent": n_coherent,
                "d_target": self.d_target,
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# ORÁCULO 2: Ruelle (OTU) — Entropía, resonancias, gap espectral
# ═════════════════════════════════════════════════════════════════════════════

class RuelleOracle:
    """
    Oráculo basado en análisis Ruelle/OTU.

    Reglas:
      - Si h_KS < h_target → muy poca mezcla caótica:
        inyectar más energía (amplificar forcing).
      - Si h_KS > h_target_max → demasiado caos:
        estabilizar aumentando disipación.
      - P''(1) mide intermitencia: si es alta, la cascada está activa.
    """

    def __init__(self, h_ks_target: float = 0.1, h_ks_max: float = 2.0):
        self.h_ks_target = h_ks_target
        self.h_ks_max = h_ks_max

    def vote(self, enstrophy_series: np.ndarray) -> OracleVote:
        if len(enstrophy_series) < 50:
            return OracleVote("ruelle", confidence=0.1)

        nu4_f = 1.0
        force_f = 1.0
        friction_f = 1.0
        confidence = 0.6
        diag = {}

        try:
            from acf_functor.gelfand_triple import GelfandTriple

            Z = np.asarray(enstrophy_series, dtype=float).ravel()
            Z_n = Z[:-1]
            Z_np1 = Z[1:]
            z_min, z_max = Z_n.min(), Z_n.max()
            z_range = z_max - z_min
            if z_range < 1e-12:
                return OracleVote("ruelle", confidence=0.1, warning="constant_enstrophy")

            Z_n_norm = (Z_n - z_min) / z_range
            Z_np1_norm = (Z_np1 - z_min) / z_range
            sort_idx = np.argsort(Z_n_norm)

            def T_return(x):
                return np.clip(np.interp(x, Z_n_norm[sort_idx], Z_np1_norm[sort_idx]), 0, 1)

            gt = GelfandTriple(T_return, (0.0, 1.0), n_test=16, n_dist=64)
            gt.build()

            # Thermodynamic pressure → h_KS
            betas = np.linspace(0.1, 3.0, 30)
            pressure = gt.compute_thermodynamic_pressure(betas)
            h_ks = pressure.slope_at_1
            P_pp_1 = pressure.curvature_at_1

            diag["h_ks"] = float(h_ks)
            diag["P_pp_1"] = float(P_pp_1)

            # Resonances for gap
            mu_srb, _ = gt.compute_self_consistent_measure()
            resonances = gt.compute_ruelle_spectrum(mu_srb, n_modes=8)[0]
            if len(resonances) > 1:
                gamma_otu = -np.log(np.abs(resonances[1]) + 1e-30)
                diag["gamma_otu"] = float(gamma_otu)

            # Control rules
            if h_ks < self.h_ks_target:
                deficit = (self.h_ks_target - h_ks) / self.h_ks_target
                nu4_f = max(0.3, 1.0 - deficit * 0.7)  # reduce dissipation
                force_f = 1.0 + deficit * 0.4            # boost forcing
                confidence = 0.7

            if h_ks > self.h_ks_max:
                excess = (h_ks - self.h_ks_max) / self.h_ks_max
                nu4_f = 1.0 + excess * 2.0   # increase dissipation
                force_f = max(0.5, 1.0 - excess * 0.3)
                confidence = 0.8

            # Intermittency: P''(1) > 0.5 means active cascade
            if P_pp_1 > 0.5:
                diag["cascade_active"] = True
                confidence = min(confidence + 0.1, 1.0)

        except Exception as e:
            diag["error"] = str(e)
            confidence = 0.1

        return OracleVote("ruelle", nu4_f, force_f, friction_f, confidence, diag)


# ═════════════════════════════════════════════════════════════════════════════
# ORÁCULO 3: ERGON — Regímenes turbulentos, Lyapunov
# ═════════════════════════════════════════════════════════════════════════════

class ErgonOracle:
    """
    Oráculo basado en detección de regímenes ERGON.

    Reglas:
      - Si n_regimes == 1 y Lyapunov ≈ 0 → flujo laminar:
        necesita más energía para transicionar.
      - Si n_regimes > 2 → intermitencia entre regímenes:
        la cascada está activa, mantener.
      - Si Lyapunov > 0 → expansivo → reducir ligeramente.
    """

    def vote(self, enstrophy_series: np.ndarray) -> OracleVote:
        if len(enstrophy_series) < 100:
            return OracleVote("ergon", confidence=0.1)

        nu4_f = 1.0
        force_f = 1.0
        friction_f = 1.0
        confidence = 0.5
        diag = {}

        try:
            from acf_functor.ergon_agent import ERGONRealWorld

            monitor = ERGONRealWorld.monitor(
                enstrophy_series,
                window_size=min(300, len(enstrophy_series) // 3),
                step_size=max(20, len(enstrophy_series) // 20),
            )

            if isinstance(monitor, dict):
                n_regimes = monitor.get("n_regimes", 1)
                diag["n_regimes"] = n_regimes

                lyap_series = monitor.get("lyapunov_series", [])
                if len(lyap_series) > 0:
                    lyap_mean = float(np.mean(lyap_series))
                    diag["lyapunov_mean"] = lyap_mean

                    if n_regimes <= 1 and abs(lyap_mean) < 0.05:
                        # Laminar: push toward turbulence
                        force_f = 1.2
                        nu4_f = 0.5
                        confidence = 0.6

                    elif lyap_mean > 0.5:
                        # Strongly expanding: risk of blowup
                        nu4_f = 1.5
                        force_f = 0.9
                        confidence = 0.7

                    elif n_regimes >= 2:
                        # Multiple regimes: cascade active
                        diag["turbulent"] = True
                        confidence = 0.7

        except Exception as e:
            diag["error"] = str(e)
            confidence = 0.1

        return OracleVote("ergon", nu4_f, force_f, friction_f, confidence, diag)


# ═════════════════════════════════════════════════════════════════════════════
# ORÁCULO 4: Termodinámica — Transiciones de fase, Fisher
# ═════════════════════════════════════════════════════════════════════════════

class ThermodynamicOracle:
    """
    Oráculo termodinámico basado en ThermodynamicACF.

    Reglas:
      - Transición de fase detectada → la cascada está en un punto crítico:
        mantener parámetros, la física está activa.
      - Fisher info alta → buena estimabilidad, sistema rico:
        mantener.
      - Fisher info ≈ 0 → sistema trivial (Bernoulli):
        inyectar energía.
    """

    def vote(self, koopman_spectrum: np.ndarray) -> OracleVote:
        if len(koopman_spectrum) < 3:
            return OracleVote("thermodynamic", confidence=0.1)

        nu4_f = 1.0
        force_f = 1.0
        friction_f = 1.0
        confidence = 0.5
        diag = {}

        try:
            import torch
            from acf_functor.thermodynamic_acf import ThermodynamicACF

            eigs = torch.tensor(np.abs(koopman_spectrum), dtype=torch.float64)
            thermo = ThermodynamicACF(eigs)
            report = thermo.analyze()
            diag["analyzed"] = True

            if hasattr(report, 'phase_transition') and report.phase_transition is not None:
                diag["phase_transition"] = True
                # At criticality: maintain parameters, physics is working
                confidence = 0.8

            if hasattr(report, 'mdl_dimension'):
                d_mdl = report.mdl_dimension
                diag["d_mdl"] = d_mdl
                if d_mdl < 5:
                    # Too compressed: need more complexity
                    force_f = 1.15
                    nu4_f = 0.7
                    confidence = 0.6

        except Exception as e:
            diag["error"] = str(e)
            confidence = 0.1

        return OracleVote("thermodynamic", nu4_f, force_f, friction_f, confidence, diag)


# ═════════════════════════════════════════════════════════════════════════════
# ORÁCULO 5: Espectral/Kolmogorov — Pendiente E(k), balance energético
# ═════════════════════════════════════════════════════════════════════════════

class SpectralOracle:
    """
    Oráculo basado en el espectro de energía E(k).

    Reglas:
      - Si slope > -2 → energía concentrada en k bajas, no hay cascada:
        reducir ν₄ y subir A.
      - Si |slope + 3| < 0.5 → cascada de enstrofía k⁻³:
        EQUILIBRIO. Mantener.
      - Si slope < -5 → disipación excesiva:
        reducir ν₄ masivamente.
    """

    def __init__(self, target_slope: float = -3.0, slope_tolerance: float = 0.8):
        self.target_slope = target_slope
        self.slope_tolerance = slope_tolerance

    def vote(self, k_bins: np.ndarray, E_k: np.ndarray, N: int) -> OracleVote:
        nu4_f = 1.0
        force_f = 1.0
        friction_f = 1.0
        confidence = 0.5
        diag = {}

        # Compute spectral slope in inertial range
        mask = (k_bins >= 6) & (k_bins <= min(40, N // 4)) & (E_k > 1e-20)
        if np.sum(mask) < 5:
            return OracleVote("spectral", confidence=0.2, diagnostics={"too_few_modes": True})

        log_k = np.log(k_bins[mask])
        log_E = np.log(E_k[mask])
        slope, _ = np.polyfit(log_k, log_E, 1)
        diag["slope"] = float(slope)

        # Near k^-3: cascade established
        if abs(slope - self.target_slope) < self.slope_tolerance:
            diag["cascade_type"] = "enstrophy_k-3"
            confidence = 0.85
            # Maintain: do nothing drastic

        elif slope > -2.0:
            # Too steep at large scales: energy piled up, no cascade
            diag["cascade_type"] = "pre_cascade"
            deficit = abs(slope - self.target_slope) / abs(self.target_slope)
            nu4_f = max(0.2, 1.0 - deficit * 0.5)
            force_f = 1.0 + min(deficit * 0.3, 0.5)
            confidence = 0.7

        elif slope < -5.0:
            # Excessive dissipation cutting the inertial range
            diag["cascade_type"] = "over_dissipated"
            nu4_f = 0.3
            force_f = 1.1
            confidence = 0.75

        else:
            # Intermediate: cascade developing
            diag["cascade_type"] = "developing"
            gap = abs(slope - self.target_slope)
            if slope > self.target_slope:
                nu4_f = max(0.5, 1.0 - gap * 0.2)
                force_f = 1.0 + gap * 0.1
            confidence = 0.6

        return OracleVote("spectral", nu4_f, force_f, friction_f, confidence, diag)


# ═════════════════════════════════════════════════════════════════════════════
# ÁRBITRO BAYESIANO — Fusión de votos de los 5 oráculos
# ═════════════════════════════════════════════════════════════════════════════

class BayesianArbiter:
    """
    Fusiona las recomendaciones de los 5 oráculos usando media ponderada
    por confianza con clipping de seguridad.

    Control actions:
      ν₄_new = ν₄_base × Σ(wᵢ × nu4_factorᵢ) / Σ(wᵢ)
      A_new  = A_base  × Σ(wᵢ × force_factorᵢ) / Σ(wᵢ)
      α_new  = α_base  × Σ(wᵢ × friction_factorᵢ) / Σ(wᵢ)

    Safety limits:
      ν₄ ∈ [ν₄_base × 0.01, ν₄_base × 100]
      A  ∈ [A_base × 0.3, A_base × 5.0]
      α  ∈ [α_base × 0.2, α_base × 3.0]
    """

    def __init__(
        self,
        nu4_base: float,
        force_base: float,
        friction_base: float,
        nu4_bounds: Tuple[float, float] = (0.01, 100.0),
        force_bounds: Tuple[float, float] = (0.3, 5.0),
        friction_bounds: Tuple[float, float] = (0.2, 3.0),
    ):
        self.nu4_base = nu4_base
        self.force_base = force_base
        self.friction_base = friction_base
        self.nu4_bounds = nu4_bounds
        self.force_bounds = force_bounds
        self.friction_bounds = friction_bounds

    def fuse(self, votes: List[OracleVote]) -> Dict[str, float]:
        """Fuse oracle votes into a single control action."""
        if not votes:
            return {
                "nu4": self.nu4_base,
                "force_amplitude": self.force_base,
                "friction": self.friction_base,
            }

        total_w = sum(v.confidence for v in votes)
        if total_w < 1e-12:
            total_w = 1.0

        nu4_f = sum(v.confidence * v.nu4_factor for v in votes) / total_w
        force_f = sum(v.confidence * v.force_factor for v in votes) / total_w
        friction_f = sum(v.confidence * v.friction_factor for v in votes) / total_w

        # Clip to safety bounds
        nu4_f = np.clip(nu4_f, self.nu4_bounds[0], self.nu4_bounds[1])
        force_f = np.clip(force_f, self.force_bounds[0], self.force_bounds[1])
        friction_f = np.clip(friction_f, self.friction_bounds[0], self.friction_bounds[1])

        return {
            "nu4": self.nu4_base * nu4_f,
            "force_amplitude": self.force_base * force_f,
            "friction": self.friction_base * friction_f,
            "nu4_factor": float(nu4_f),
            "force_factor": float(force_f),
            "friction_factor": float(friction_f),
            "n_oracles": len([v for v in votes if v.confidence > 0.1]),
        }


# ═════════════════════════════════════════════════════════════════════════════
# CASCADE ACCELERATOR — Termostato de Turbulencia
# ═════════════════════════════════════════════════════════════════════════════

class CascadeAccelerator:
    """
    Termostato de turbulencia que CONDUCE el flujo hacia turbulencia
    completamente desarrollada y lo MANTIENE ahí.

    Tres fases:
      1. ACCELERATION: d_95 < d_target → inyectar energía agresivamente
      2. STABILIZATION: d_95 ≈ d_target pero slope != -3 → afinar
      3. EQUILIBRIUM: d_95 ≈ d_target y slope ≈ -3 → mantener

    La clave es que el ACF sabe CUÁNDO parar de empujar: cuando los
    modos Koopman capturan la cascada en ~d_target dimensiones.
    """

    def __init__(
        self,
        d_target: int = 30,
        h_ks_target: float = 0.1,
        slope_target: float = -3.0,
        slope_tol: float = 1.0,
        max_force_boost: float = 5.0,
        min_nu4_ratio: float = 0.01,
    ):
        self.d_target = d_target
        self.h_ks_target = h_ks_target
        self.slope_target = slope_target
        self.slope_tol = slope_tol
        self.max_force_boost = max_force_boost
        self.min_nu4_ratio = min_nu4_ratio
        self.phase = "acceleration"
        self.history: List[ThermostatState] = []

    def update(
        self,
        arbiter_action: Dict[str, float],
        d_95: int,
        h_ks: float,
        spectral_slope: float,
        energy: float,
        enstrophy: float,
        n_coherent: int,
        t: float,
        N: int,
    ) -> Dict[str, float]:
        """
        Apply thermostat override on top of arbiter actions.

        Returns the FINAL control action (ν₄, A, α).
        """
        nu4 = arbiter_action["nu4"]
        A = arbiter_action["force_amplitude"]
        alpha = arbiter_action["friction"]
        nu4_base = arbiter_action.get("nu4_factor", 1.0)

        # Determine phase
        at_target_d = d_95 >= self.d_target * 0.7
        at_target_slope = abs(spectral_slope - self.slope_target) < self.slope_tol
        h_active = h_ks > self.h_ks_target * 0.5

        if not at_target_d:
            self.phase = "acceleration"
        elif at_target_d and not at_target_slope:
            self.phase = "stabilization"
        elif at_target_d and at_target_slope:
            self.phase = "equilibrium"

        # Phase-specific overrides
        if self.phase == "acceleration":
            # AGGRESSIVE: push hard toward turbulence
            deficit = max(0, (self.d_target - d_95) / self.d_target)
            boost = 1.0 + deficit * (self.max_force_boost - 1.0)
            A = A * min(boost, self.max_force_boost)
            nu4 = nu4 * max(self.min_nu4_ratio, 1.0 - deficit * 0.9)
            # Also reduce friction to let energy accumulate
            alpha = alpha * max(0.3, 1.0 - deficit * 0.5)

        elif self.phase == "stabilization":
            # MODERATE: fine-tune toward k^-3
            if spectral_slope > self.slope_target:
                # Not steep enough: slightly boost
                A = A * 1.05
                nu4 = nu4 * 0.9
            else:
                # Too steep: slightly damp
                A = A * 0.95
                nu4 = nu4 * 1.1

        elif self.phase == "equilibrium":
            # MAINTAIN: only correct small deviations
            pass  # Use arbiter action as-is

        reduction = N**2 / max(d_95, 1)

        state = ThermostatState(
            t=t, d_95=d_95, h_ks=h_ks, spectral_slope=spectral_slope,
            nu4_effective=nu4, force_effective=A, friction_effective=alpha,
            energy=energy, enstrophy=enstrophy, n_coherent=n_coherent,
            phase=self.phase, oracles_active=arbiter_action.get("n_oracles", 0),
            reduction_factor=reduction,
        )
        self.history.append(state)

        return {"nu4": nu4, "force_amplitude": A, "friction": alpha}


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-ORACLE AMR CONTROLLER — Integra todos los oráculos
# ═════════════════════════════════════════════════════════════════════════════

class MultiOracleAMR:
    """
    Controlador AMR multi-oráculo para el solver acoplado NS-ACF.

    Ejecuta los 5 oráculos, fusiona con el árbitro bayesiano,
    y aplica el termostato de cascada.
    """

    def __init__(
        self,
        N: int,
        nu4_base: float,
        force_base: float,
        friction_base: float,
        d_target: int = 30,
        h_ks_target: float = 0.1,
    ):
        self.N = N
        self.koopman_oracle = KoopmanOracle(d_target=d_target)
        self.ruelle_oracle = RuelleOracle(h_ks_target=h_ks_target)
        self.ergon_oracle = ErgonOracle()
        self.thermo_oracle = ThermodynamicOracle()
        self.spectral_oracle = SpectralOracle(target_slope=-3.0)

        self.arbiter = BayesianArbiter(
            nu4_base=nu4_base,
            force_base=force_base,
            friction_base=friction_base,
        )

        self.cascade = CascadeAccelerator(
            d_target=d_target,
            h_ks_target=h_ks_target,
        )

        self.vote_log: List[Dict[str, Any]] = []

    def control(
        self,
        koopman_result: Any,
        enstrophy_series: np.ndarray,
        k_bins: np.ndarray,
        E_k: np.ndarray,
        energy: float,
        enstrophy: float,
        t: float,
    ) -> Dict[str, float]:
        """
        Execute all oracles, fuse, and apply thermostat.

        Returns final control action dict.
        """
        votes = []

        # O1: Koopman
        v1 = self.koopman_oracle.vote(koopman_result, energy, enstrophy)
        votes.append(v1)

        # O2: Ruelle (only if enough data)
        if len(enstrophy_series) >= 50:
            v2 = self.ruelle_oracle.vote(enstrophy_series)
            votes.append(v2)

        # O3: ERGON (only if enough data)
        if len(enstrophy_series) >= 100:
            v3 = self.ergon_oracle.vote(enstrophy_series)
            votes.append(v3)

        # O4: Thermodynamic (needs Koopman spectrum)
        if koopman_result is not None:
            v4 = self.thermo_oracle.vote(koopman_result.eigenvalues)
            votes.append(v4)

        # O5: Spectral
        v5 = self.spectral_oracle.vote(k_bins, E_k, self.N)
        votes.append(v5)

        # Fuse votes
        action = self.arbiter.fuse(votes)

        # Extract diagnostics for thermostat
        d_95 = v1.diagnostics.get("d_95", 2) if v1.confidence > 0.1 else 2
        n_coherent = v1.diagnostics.get("n_coherent", 0) if v1.confidence > 0.1 else 0
        h_ks = 0.0
        for v in votes:
            if v.oracle_name == "ruelle" and "h_ks" in v.diagnostics:
                h_ks = v.diagnostics["h_ks"]
                break
        slope = 0.0
        for v in votes:
            if v.oracle_name == "spectral" and "slope" in v.diagnostics:
                slope = v.diagnostics["slope"]
                break

        # Apply thermostat
        final = self.cascade.update(
            action, d_95, h_ks, slope, energy, enstrophy, n_coherent, t, self.N
        )

        # Log
        self.vote_log.append({
            "t": t,
            "votes": [(v.oracle_name, v.confidence, v.nu4_factor, v.force_factor)
                      for v in votes],
            "arbiter": action,
            "thermostat": final,
            "phase": self.cascade.phase,
            "d_95": d_95,
            "h_ks": h_ks,
            "slope": slope,
        })

        return final
